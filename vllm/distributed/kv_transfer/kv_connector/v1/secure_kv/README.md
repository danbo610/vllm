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
