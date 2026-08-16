# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SecureKVConnector: encrypted external KV cache (SecureKV M2).

Structure follows the in-tree ExampleConnector (disk KV save/load), with
three changes from the design doc (kv-encryption-connector-design.md):

1. Payloads leave the process only as AES-256-GCM SealedBlobs — the storage
   backend never sees plaintext KV (§4: encrypt-at-the-boundary).
2. Index keys are HMAC(tenant_key, prompt_bytes) instead of a plain hash
   (§7.3: no prefix probing, per-tenant namespaces for free).
3. The AAD binds {layer, index key, model fingerprint, tenant} into the GCM
   authentication (§7.2: relocation / replay / cross-tenant blobs fail).

M2 scope notes: synchronous save/load (async pipelining is M4); decrypt
failure is fail-fast with a clear error (graceful degradation to recompute
via get_block_ids_with_load_errors is M3); single default tenant unless
`tenant_id` is set in kv_connector_extra_config.
"""

import hashlib
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.secure_kv.backend import (
    make_backend,
)
from vllm.distributed.kv_transfer.kv_connector.v1.secure_kv.crypto import (
    CryptoEngine,
    DecryptError,
    pack_tensor,
    unpack_tensor,
)
from vllm.distributed.kv_transfer.kv_connector.v1.secure_kv.keys import (
    KeyManager,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class ReqMeta:
    token_ids: torch.Tensor
    slot_mapping: torch.Tensor
    is_store: bool
    mm_hashes: list[str]
    # Tenant resolved on the scheduler side, carried to the worker so both
    # sides derive the same key (they are different processes; §5.2 note).
    tenant_id: str
    # Block ids for this request, so a decrypt failure can name the exact
    # blocks to recompute (§3 point 7 / M3 graceful degradation).
    block_ids: list[int]

    @staticmethod
    def make_meta(
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
        mm_hashes: list[str],
        tenant_id: str,
    ) -> "ReqMeta":
        valid_num_tokens = align_to_block_size(len(token_ids), block_size)
        token_ids_tensor = torch.tensor(token_ids)[:valid_num_tokens]
        block_ids_tensor = torch.tensor(block_ids)
        num_blocks = block_ids_tensor.shape[0]
        block_offsets = torch.arange(0, block_size)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids_tensor.reshape((num_blocks, 1)) * block_size
        )
        slot_mapping = slot_mapping.flatten()[:valid_num_tokens]
        return ReqMeta(
            token_ids=token_ids_tensor,
            slot_mapping=slot_mapping,
            is_store=is_store,
            mm_hashes=mm_hashes,
            tenant_id=tenant_id,
            block_ids=list(block_ids),
        )


@dataclass
class SecureKVConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)

    def add_request(
        self,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
        mm_hashes: list[str],
        tenant_id: str,
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(token_ids, block_ids, block_size, is_store,
                              mm_hashes, tenant_id)
        )


class SecureKVConnector(KVConnectorBase_V1):
    """Disk/remote KV connector whose external payloads are always
    AES-256-GCM encrypted; see module docstring."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: dict[str, Request] = {}

        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        key_source = extra.get("key_source", "env:SKV_MASTER_KEY")
        master_hex = None
        if key_source.startswith("env:"):
            master_hex = os.environ.get(key_source[4:], "")
        # Default tenant when a request carries no tenant (single-tenant mode).
        self._default_tenant = extra.get("tenant_id", "default")
        # Request extra_args key that carries the tenant (gateway injects it,
        # same channel as SynthID watermarking; §8).
        self._tenant_arg = extra.get("tenant_arg", "skv_tenant")
        self._km = KeyManager(master_hex)
        self._crypto = CryptoEngine(self._km,
                                    workers=int(extra.get("crypto_workers", 4)))
        self._backend = make_backend(extra.get("backend", "local_disk"),
                                     extra.get("backend_config", {}))
        # Model fingerprint for AAD: cross-model blob reuse must fail auth.
        self._model_fp = hashlib.sha256(
            vllm_config.model_config.model.encode()).digest()[:8]
        # Scheduler-side: request_id -> tenant_id, resolved in on_new_request.
        self._req_tenant: dict[str, str] = {}
        # Worker-side: block ids whose decrypt failed this pass (M3 recompute).
        self._load_error_blocks: set[int] = set()
        logger.info("SecureKV: backend=%s default_tenant=%s (payloads "
                    "AES-256-GCM, per-request tenant keying)",
                    extra.get("backend", "local_disk"), self._default_tenant)

    # ==============================
    # Worker side
    # ==============================

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        def inject_kv_into_layer(
            dst_kv_cache_layer: torch.Tensor,
            src_kv_cache: torch.Tensor,
            slot_mapping: torch.Tensor,
            attn_metadata: AttentionMetadata,
        ) -> None:
            if isinstance(attn_metadata, MLACommonMetadata):
                dst_kv_cache_layer_shape = dst_kv_cache_layer.shape
                num_pages = dst_kv_cache_layer_shape[0]
                page_size = dst_kv_cache_layer_shape[1]
                dst_kv_cache_layer = dst_kv_cache_layer.reshape(
                    num_pages * page_size, -1
                )
                dst_kv_cache_layer[slot_mapping, ...] = src_kv_cache
            else:
                block_idxs = slot_mapping // self._block_size
                offsets = slot_mapping % self._block_size
                dst_kv_cache_layer[block_idxs, :, offsets] = src_kv_cache

        metadata: KVConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, SecureKVConnectorMetadata)

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.warning("In connector.start_load_kv, but the attn_metadata is None")
            return

        for request in metadata.requests:
            if request.is_store:
                continue
            index_key = self._index_key(request.token_ids, request.mm_hashes,
                                        request.tenant_id)
            # Gather every attention layer's blob, then decrypt them all in
            # parallel (M4: the per-request critical path is L sequential
            # decrypts otherwise; L=24..64 layers). Decrypt failure on ANY
            # layer taints the whole request — a partially-injected prefix is
            # unusable, so recompute it all.
            layers, blobs = [], []
            missing = False
            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]
                kv_cache_layer = getattr(layer, "kv_cache", None)
                if kv_cache_layer is None:
                    continue
                blob = self._backend.get(self._layer_key(index_key, layer_name))
                if blob is None:
                    missing = True
                    break
                layers.append((layer_name, kv_cache_layer))
                blobs.append(blob)

            if missing:
                self._load_error_blocks.update(request.block_ids)
                continue

            results = self._crypto.unseal_many([
                (blob, request.tenant_id,
                 self._aad(index_key, ln, request.tenant_id))
                for (ln, _), blob in zip(layers, blobs)
            ])
            bad = next((r for r in results if isinstance(r, DecryptError)), None)
            if bad is not None:
                # Never crash the engine: mark blocks for recompute and raise a
                # security alert (§3 point 7). Tag failure => tampering /
                # cross-tenant / corruption.
                logger.warning(
                    "SecureKV SECURITY: decrypt failed (index %s…, tenant %s): "
                    "%s — recomputing %d blocks",
                    index_key[:12], request.tenant_id, bad,
                    len(request.block_ids))
                self._load_error_blocks.update(request.block_ids)
                continue

            for (layer_name, kv_cache_layer), plaintext in zip(layers, results):
                kv_cache = unpack_tensor(plaintext).to(device=kv_cache_layer.device)
                if isinstance(attn_metadata, dict):
                    inject_kv_into_layer(
                        kv_cache_layer, kv_cache, request.slot_mapping,
                        attn_metadata[layer_name])
            logger.info(
                "SecureKV: decrypt-and-inject KV of %d tokens into paged "
                "memory (tenant %s, %d layers)",
                len(request.slot_mapping), request.tenant_id, len(layers))

    def get_block_ids_with_load_errors(self) -> set[int]:
        # Reported once, then cleared: the engine recomputes these blocks
        # (requires kv_load_failure_policy="recompute").
        errs = self._load_error_blocks
        self._load_error_blocks = set()
        return errs

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        def extract_kv_from_layer(
            layer: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> torch.Tensor:
            if isinstance(attn_metadata, MLACommonMetadata):
                num_pages, page_size = layer.shape[0], layer.shape[1]
                return layer.reshape(num_pages * page_size, -1)[slot_mapping, ...]
            block_idxs = slot_mapping // self._block_size
            offsets = slot_mapping % self._block_size
            return layer[block_idxs, :, offsets]

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, SecureKVConnectorMetadata)
        for request in connector_metadata.requests:
            if request.is_store:
                index_key = self._index_key(request.token_ids, request.mm_hashes,
                                            request.tenant_id)
                kv_cache = extract_kv_from_layer(kv_layer, request.slot_mapping)
                blob = self._crypto.seal(
                    pack_tensor(kv_cache),
                    aad=self._aad(index_key, layer_name, request.tenant_id),
                    tenant_id=request.tenant_id)
                self._backend.put(self._layer_key(index_key, layer_name), blob)

    def wait_for_save(self):
        return

    # ==============================
    # Scheduler side
    # ==============================

    def on_new_request(self, request: "Request") -> None:
        """Resolve the request's tenant (§8): extra_args[tenant_arg] if the
        gateway injected one, else the configured default. Cached for the
        scheduler-side match/meta calls."""
        tenant = self._default_tenant
        sp = getattr(request, "sampling_params", None)
        extra = getattr(sp, "extra_args", None) if sp else None
        if extra and extra.get(self._tenant_arg):
            tenant = str(extra[self._tenant_arg])
        self._req_tenant[request.request_id] = tenant

    def _tenant_of(self, request: "Request") -> str:
        return self._req_tenant.get(request.request_id, self._default_tenant)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        if not self._found_match_for_request(request):
            return 0, False

        logger.info("SecureKV: external (encrypted) cache hit! (tenant %s)",
                    self._tenant_of(request))
        token_ids = request.prompt_token_ids or []
        num_tokens_to_check = align_to_block_size(len(token_ids) - 1, self._block_size)
        return num_tokens_to_check - num_computed_tokens, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = SecureKVConnectorMetadata()

        total_need_load = 0
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            mm_hashes = [f.identifier for f in new_req.mm_features]
            tenant = self._req_tenant.get(new_req.req_id, self._default_tenant)
            if new_req.req_id in self._requests_need_load:
                meta.add_request(
                    token_ids=token_ids,
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                    is_store=False,
                    mm_hashes=mm_hashes,
                    tenant_id=tenant,
                )
                total_need_load += 1
            else:
                if not self._found_match_for_prompt(token_ids, mm_hashes, tenant):
                    meta.add_request(
                        token_ids=token_ids,
                        block_ids=new_req.block_ids[0],
                        block_size=self._block_size,
                        is_store=True,
                        mm_hashes=mm_hashes,
                        tenant_id=tenant,
                    )

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            resumed_from_preemption = req_id in cached_reqs.resumed_req_ids
            if not resumed_from_preemption or req_id not in self._requests_need_load:
                continue

            num_computed_tokens = cached_reqs.num_computed_tokens[i]
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            new_block_ids = cached_reqs.new_block_ids[i]

            request = self._requests_need_load[req_id]
            total_tokens = num_computed_tokens + num_new_tokens
            token_ids = request.all_token_ids[:total_tokens]

            assert new_block_ids is not None
            block_ids = new_block_ids[0]

            meta.add_request(
                token_ids=token_ids,
                block_ids=block_ids,
                block_size=self._block_size,
                is_store=False,
                mm_hashes=[f.identifier for f in request.mm_features],
                tenant_id=self._req_tenant.get(req_id, self._default_tenant),
            )
            total_need_load += 1

        assert total_need_load == len(self._requests_need_load)
        self._requests_need_load.clear()
        return meta

    def request_finished(self, request, block_ids):
        # Drop cached tenant mapping when the request leaves the scheduler.
        self._req_tenant.pop(request.request_id, None)
        return False, None

    def shutdown(self):
        self._km.wipe()

    # ==============================
    # Helper functions
    # ==============================

    def _index_key(self, token_ids: torch.Tensor, mm_hashes: list[str],
                   tenant_id: str) -> str:
        """HMAC index key over the (aligned) prompt tokens (§7.3). Keyed by
        the tenant, so the same prefix maps to different keys per tenant —
        cross-tenant sharing and prefix probing are both impossible."""
        token_bytes = token_ids.numpy().tobytes()
        if mm_hashes:
            token_bytes += "-".join(mm_hashes).encode("utf-8")
        return self._km.index_hmac(tenant_id, token_bytes)

    def _layer_key(self, index_key: str, layer_name: str) -> str:
        layer_digest = hashlib.sha256(layer_name.encode()).hexdigest()[:16]
        return f"{index_key}/{layer_digest}"

    def _aad(self, index_key: str, layer_name: str, tenant_id: str) -> bytes:
        """Blob identity bound into GCM authentication (§7.2)."""
        return b"|".join([
            b"skv1",
            index_key.encode(),
            layer_name.encode(),
            self._model_fp,
            self._km.tenant_tag(tenant_id),
        ])

    def _found_match_for_request(self, request: "Request") -> bool:
        return self._found_match_for_prompt(
            list(request.prompt_token_ids or []),
            [f.identifier for f in request.mm_features],
            self._tenant_of(request),
        )

    def _found_match_for_prompt(
        self,
        prompt_token_ids: list[int],
        mm_hashes: list[str],
        tenant_id: str,
    ) -> bool:
        num_tokens_to_check = align_to_block_size(
            len(prompt_token_ids) - 1, self._block_size
        )
        index_key = self._index_key(
            torch.tensor(prompt_token_ids)[:num_tokens_to_check], mm_hashes,
            tenant_id)
        return self._backend.contains(index_key)


def align_to_block_size(num_tokens: int, block_size) -> int:
    """Align the number of tokens to the block size."""
    return (num_tokens - 1) // block_size * block_size
