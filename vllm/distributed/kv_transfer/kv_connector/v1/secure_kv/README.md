# SecureKV: Encrypted External KV Cache Connector

KV cache leaving the GPU is a privacy liability: KV derives directly from
user input and can be reconstructed into the prompt (arXiv 2508.09442).
SecureKV is a `KVConnectorBase_V1` implementation that seals every KV
payload with **AES-256-GCM at the GPU-egress boundary**: GPU memory stays
plaintext (native attention kernels, zero changes), while everything that
reaches a storage backend — disk, remote store — is ciphertext under
per-tenant keys. Backends therefore do not need to be trusted.

| File | Role |
|---|---|
| `connector.py` | `SecureKVConnector` (scheduler + worker sides) |
| `crypto.py` | SealedBlob format, AES-256-GCM engine, tensor (de)serialization |
| `keys.py` | Key hierarchy (master → tenant, HKDF-SHA256), nonce discipline, HMAC index keys |
| `backend.py` | `StorageBackend` ABC + `LocalDiskBackend` + `RemoteBackend` (TCP) |
| `remote_server.py` | Minimal remote ciphertext store for `RemoteBackend` |
| `fs_tier.py` | Encrypted secondary tier for `TieringOffloadingSpec` |
| `fs_capacity.py` | Cross-process disk capacity accounting and LRU eviction |
| `selftest.py` | M1 unit suite (crypto/keys/backend; no GPU) |
| `m3m5_test.py` | Tenant isolation, degrade, throughput, remote, attack-neutralization units |
| `e2e_test.py`, `e2e_m3m4.py` | End-to-end phases against a real engine (see below) |

## Enabling

```bash
export SKV_MASTER_KEY=<128/256-bit hex>   # KMS-injected in production

vllm serve <model> ... \
  --kv-transfer-config '{
    "kv_connector": "SecureKVConnector",
    "kv_role": "kv_both",
    "kv_load_failure_policy": "recompute",
    "kv_connector_extra_config": {
      "backend": "local_disk",
      "backend_config": {"root_dir": "/data/kv_cache"},
      "key_source": "env:SKV_MASTER_KEY",
      "tenant_id": "default",
      "tenant_arg": "skv_tenant"
    }
  }'
```

Per-request tenants: a gateway injects `"vllm_xargs": {"skv_tenant": "acme"}`;
the connector keys every blob and index entry with that tenant's derived key.
The same prefix under two tenants maps to different HMAC index keys, so
cross-tenant sharing and prefix probing are both impossible by construction.

## Bounded encrypted filesystem tier

`encrypted_fs` can also be used as a secondary tier in the official
`OffloadingConnector + TieringOffloadingSpec` stack:

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_load_failure_policy": "recompute",
  "kv_connector_extra_config": {
    "spec_name": "TieringOffloadingSpec",
    "cpu_bytes_to_use": 8589934592,
    "secondary_tiers": [
      {
        "type": "encrypted_fs",
        "root_dir": "/data/kv_cache",
        "max_bytes": 274877906944,
        "min_free_bytes": 137438953472,
        "eviction_low_watermark": 0.9
      }
    ]
  }
}
```

Capacity fields:

| Field | Meaning | Default |
|---|---|---|
| `max_bytes` | Hard limit for encrypted `.bin` files; omit to disable this limit | unset |
| `min_free_bytes` | Filesystem space that must remain free after a write | `0` |
| `eviction_low_watermark` | Target fraction after max-capacity eviction | `0.9` |

Existing encrypted blocks are scanned when the tier starts. Before admitting a
new block, the manager evicts the oldest files by `mtime` until both limits can
be met. A successful decrypt updates `mtime`, making the policy an access-based
LRU approximation. Admission and eviction use `fcntl.flock`, so multiple vLLM
processes sharing one root cannot independently exceed the hard limit. Reads
hold a shared lock while copying ciphertext from the file, and writes remain
atomic through a temporary file plus `os.replace`.

Eviction runs during tier startup and new block admission; it is not a periodic
background task. If no files can be removed while preserving
`min_free_bytes`, the store job fails and the configured
`kv_load_failure_policy` remains responsible for normal recomputation behavior.

If a block cannot be read, authenticated, or decoded to the expected size,
the tier removes it from disk and shared capacity accounting. The failed load
job also invalidates the async lookup manager's cached HIT, so the same request
observes a MISS on its next lookup instead of repeatedly promoting the bad
blob. With `kv_load_failure_policy: "recompute"`, vLLM then rebuilds the
affected prefix and may store a fresh encrypted block.

The tier exports these Prometheus metrics:

```text
vllm:kv_offload_encrypted_fs_cache_bytes
vllm:kv_offload_encrypted_fs_cache_files
vllm:kv_offload_encrypted_fs_cache_usage_perc
vllm:kv_offload_encrypted_fs_disk_free_bytes
vllm:kv_offload_encrypted_fs_evicted_bytes
vllm:kv_offload_encrypted_fs_evicted_files
vllm:kv_offload_encrypted_fs_admission_rejections
vllm:kv_offload_encrypted_fs_stale_temp_files_removed
vllm:kv_offload_encrypted_fs_encrypt_seconds
vllm:kv_offload_encrypted_fs_decrypt_seconds
vllm:kv_offload_encrypted_fs_read_seconds
vllm:kv_offload_encrypted_fs_write_seconds
vllm:kv_offload_encrypted_fs_plaintext_copy_seconds{operation="load|store"}
vllm:kv_offload_encrypted_fs_job_queue_seconds{operation="load|store"}
vllm:kv_offload_encrypted_fs_job_total_seconds{operation="load|store"}
vllm:kv_offload_encrypted_fs_encrypted_bytes
vllm:kv_offload_encrypted_fs_decrypted_bytes
vllm:kv_offload_encrypted_fs_read_bytes
vllm:kv_offload_encrypted_fs_written_bytes
vllm:kv_offload_encrypted_fs_queue_depth{operation="load|store"}
vllm:kv_offload_encrypted_fs_inflight_jobs{operation="load|store"}
vllm:kv_offload_encrypted_fs_decrypt_failures
vllm:kv_offload_encrypted_fs_load_failures
vllm:kv_offload_encrypted_fs_store_failures
vllm:kv_offload_encrypted_fs_invalidated_blocks
```

Stage histograms are observed once per block. Queue and total histograms are
observed once per submitted secondary-tier job, which may contain multiple
blocks. `job_total_seconds` starts at enqueue and ends when the encrypted FS
task finishes; GPU-to-CPU and CPU-to-GPU transfer time remains in vLLM's
standard `kv_offload_{store,load}_time` metrics.

## Security properties (tested)

- Sealed payloads only: on-disk/remote bytes are `SKV1` blobs; ciphertext is
  statistically uniform (entropy > 7.999 bits/byte, chi-square clean).
- AAD binds {index key, layer, model fingerprint, tenant tag}: relocation,
  replay, cross-tenant and cross-model reuse all fail GCM authentication.
- Decrypt failure never crashes the engine: affected blocks are reported via
  `get_block_ids_with_load_errors()` and recomputed
  (set `kv_load_failure_policy: "recompute"`), with a security warning logged.
- Nonce: 64-bit per-process random salt + 32-bit counter per key; no reuse.

## KV-injection numerics (read before writing tests)

The encryption round-trip is byte-exact (unit-tested). However, a cache-hit
forward pass (inject prefix KV + recompute the unaligned tail) uses different
batch shapes than a full prefill, so bf16 logits differ by ~1e-3 and greedy
decoding may fork at low-confidence tail tokens. This is inherent to vLLM KV
injection — the in-tree plaintext ExampleConnector forks at the same
positions, and SecureKV's load output is token-identical to the plaintext
connector's. E2E acceptance therefore checks a long exact prefix (>=85%)
rather than full-sequence equality, plus exact equality on fresh-compute
paths.
