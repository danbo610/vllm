# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encrypted filesystem secondary tier for the official offloading stack.

Subclasses the in-tree ``FileSystemTierManager`` and swaps ONLY the byte-IO
tasks: blocks are sealed with AES-256-GCM before they reach disk and
unsealed (tag-verified) on load. Everything else — cache semantics, hybrid
(GDN/Mamba) group handling, lookup, thread pool, eviction — is inherited
from the official stack, which is the whole point (see
secure-kv-implementation-report.md §9.5–9.7).

Key model: one global tier key derived from ``SKV_MASTER_KEY`` (the tier
layer has no request context; per-tenant isolation is the job of vLLM's
``cache_salt`` at the hit layer, §9.7). The AAD binds each blob to its
block content hash and KV group index, so ciphertext relocation or
cross-group replay fails authentication.

Enable via kv_connector_extra_config:

    "secondary_tiers": [{"type": "encrypted_fs", "root_dir": "/data/kv"}]

O_DIRECT note: sealing changes the payload length (header + tag), so this
tier always uses buffered IO with atomic temp-file renames.
"""

from __future__ import annotations

import functools
import os
import struct
import threading

from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.secure_kv.crypto import (
    CryptoEngine,
)
from vllm.distributed.kv_transfer.kv_connector.v1.secure_kv.keys import (
    KeyManager,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    get_offload_block_hash,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager

logger = init_logger(__name__)

# Global-key namespace for the tier (no request context at this layer;
# isolation is cache_salt's job at the hit layer).
_TIER_KEY_NS = "fs-tier"


def _aad(key) -> bytes:
    """Blob identity for GCM authentication: content hash + KV group."""
    return b"|".join([
        b"skvfs1",
        get_offload_block_hash(key),
        struct.pack("<H", get_offload_group_idx(key)),
    ])


def _tmp_suffix() -> str:
    return f".tmp{os.getpid()}_{threading.get_ident()}"


def _encrypt_store(crypto: CryptoEngine, paths, keys, view: memoryview,
                   offsets, block_size: int) -> None:
    """Seal each block and write it atomically. Raises on first error
    (the thread pool converts that into job failure)."""
    flat = view.cast("B")
    for path, key, off in zip(paths, keys, offsets):
        if os.path.exists(path):
            continue
        blob = crypto.seal(bytes(flat[off:off + block_size]), _aad(key),
                           _TIER_KEY_NS)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + _tmp_suffix()
        try:
            with open(tmp, "wb") as f:
                f.write(blob)
            os.replace(tmp, path)
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise


def _decrypt_load(crypto: CryptoEngine, paths, keys, view: memoryview,
                  offsets, block_size: int) -> None:
    """Read, tag-verify and unseal each block into the shared CPU buffer.
    A failed tag raises DecryptError -> job failure -> the engine treats the
    load as failed (never injects garbage KV)."""
    flat = view.cast("B")
    for path, key, off in zip(paths, keys, offsets):
        with open(path, "rb") as f:
            blob = f.read()
        plaintext = crypto.unseal(blob, _TIER_KEY_NS, expected_aad=_aad(key))
        if len(plaintext) != block_size:
            raise ValueError(
                f"sealed block {os.path.basename(path)} decrypted to "
                f"{len(plaintext)} bytes, expected {block_size}")
        flat[off:off + block_size] = plaintext


class EncryptedFileSystemTierManager(FileSystemTierManager):
    """FileSystemTierManager whose on-disk bytes are always AES-256-GCM
    SealedBlobs. Only the store/load byte tasks differ from the parent."""

    def __init__(self, *args, key_source: str = "env:SKV_MASTER_KEY",
                 crypto_workers: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        # Ciphertext length is header+payload+tag: not O_DIRECT-alignable.
        self._use_o_direct = False
        master_hex = None
        if key_source.startswith("env:"):
            master_hex = os.environ.get(key_source[4:], "")
        self._crypto = CryptoEngine(KeyManager(master_hex),
                                    workers=crypto_workers)
        logger.info(
            "EncryptedFileSystemTier '%s': AES-256-GCM at rest under %s "
            "(global tier key; per-tenant isolation via cache_salt)",
            self.tier_type, self.file_mapper.get_config_file_path())

    @override
    def submit_store(self, job_metadata) -> None:
        if self.events is not None:
            self._store_job_keys[job_metadata.job_id] = list(job_metadata.keys)
        task = functools.partial(
            _encrypt_store,
            self._crypto,
            [self.file_mapper.get_file_name(k) for k in job_metadata.keys],
            list(job_metadata.keys),
            self._primary_kv_view,
            [int(bid) * self._block_size for bid in job_metadata.block_ids],
            self._block_size,
        )
        self._pool.enqueue_store(job_metadata.job_id, 1, [task])

    @override
    def submit_load(self, job_metadata) -> None:
        task = functools.partial(
            _decrypt_load,
            self._crypto,
            [self.file_mapper.get_file_name(k) for k in job_metadata.keys],
            list(job_metadata.keys),
            self._primary_kv_view,
            [int(bid) * self._block_size for bid in job_metadata.block_ids],
            self._block_size,
        )
        self._pool.enqueue_load(job_metadata.job_id, 1, [task])
