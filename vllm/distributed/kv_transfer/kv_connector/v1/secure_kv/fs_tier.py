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
import time
from contextlib import nullcontext, suppress
from typing import Any

from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.secure_kv.crypto import (
    CryptoEngine,
    DecryptError,
)
from vllm.distributed.kv_transfer.kv_connector.v1.secure_kv.fs_capacity import (
    EncryptedFsCapacityManager,
    read_blob,
    write_blob_parts,
)
from vllm.distributed.kv_transfer.kv_connector.v1.secure_kv.fs_concurrency import (
    EncryptedFsConcurrency,
)
from vllm.distributed.kv_transfer.kv_connector.v1.secure_kv.fs_metrics import (
    EncryptedFsMetrics,
    EncryptedFsTelemetry,
    run_timed_job,
)
from vllm.distributed.kv_transfer.kv_connector.v1.secure_kv.keys import (
    KeyManager,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    OffloadingMetricMetadata,
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
    return b"|".join(
        [
            b"skvfs1",
            get_offload_block_hash(key),
            struct.pack("<H", get_offload_group_idx(key)),
        ]
    )


def _tmp_suffix() -> str:
    return f".tmp{os.getpid()}_{threading.get_ident()}"


def _remove_failed_blob(
    path: str,
    capacity: EncryptedFsCapacityManager | None,
    concurrency: EncryptedFsConcurrency | None = None,
) -> bool:
    io_slot = (
        concurrency.write_slot("load") if concurrency is not None else nullcontext()
    )
    try:
        with io_slot:
            if capacity is None:
                os.remove(path)
                removed = True
            else:
                removed = capacity.remove_blob(path)
    except FileNotFoundError:
        return False
    except Exception:
        logger.exception("Failed to remove unreadable encrypted KV block %s", path)
        return False
    if removed:
        logger.warning("Removed unreadable encrypted KV block %s", path)
    return removed


def _encrypt_store(
    crypto: CryptoEngine,
    paths,
    keys,
    view: memoryview,
    offsets,
    block_size: int,
    capacity: EncryptedFsCapacityManager | None = None,
    telemetry: EncryptedFsTelemetry | None = None,
    concurrency: EncryptedFsConcurrency | None = None,
) -> None:
    """Seal each block and write it atomically. Raises on first error
    (the thread pool converts that into job failure)."""
    flat = view.cast("B")
    for path, key, off in zip(paths, keys, offsets):
        block_slot = (
            concurrency.block_slot("store")
            if concurrency is not None
            else nullcontext()
        )
        with block_slot:
            if capacity is None and os.path.exists(path):
                continue

            plaintext = flat[off : off + block_size]

            crypto_slot = (
                concurrency.crypto_slot("store")
                if concurrency is not None
                else nullcontext()
            )
            with crypto_slot:
                started_at = time.monotonic()
                try:
                    blob_parts = crypto.seal_parts(plaintext, _aad(key), _TIER_KEY_NS)
                finally:
                    if telemetry is not None:
                        telemetry.observe(
                            EncryptedFsMetrics.ENCRYPT_SECONDS,
                            time.monotonic() - started_at,
                        )
            if telemetry is not None:
                telemetry.increase(EncryptedFsMetrics.ENCRYPTED_BYTES, len(plaintext))
            blob_size = sum(len(part) for part in blob_parts)

            write_slot = (
                concurrency.write_slot("store")
                if concurrency is not None
                else nullcontext()
            )
            with write_slot:
                started_at = time.monotonic()
                if capacity is not None:
                    try:
                        stored = capacity.store_blob_parts(path, blob_parts)
                    finally:
                        if telemetry is not None:
                            telemetry.observe(
                                EncryptedFsMetrics.FS_WRITE_SECONDS,
                                time.monotonic() - started_at,
                            )
                    if telemetry is not None and stored:
                        telemetry.increase(EncryptedFsMetrics.FS_WRITE_BYTES, blob_size)
                    continue
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = path + _tmp_suffix()
                try:
                    with open(tmp, "xb", buffering=0) as f:
                        write_blob_parts(f.fileno(), blob_parts)
                    os.replace(tmp, path)
                except Exception:
                    with suppress(OSError):
                        os.remove(tmp)
                    raise
                finally:
                    if telemetry is not None:
                        telemetry.observe(
                            EncryptedFsMetrics.FS_WRITE_SECONDS,
                            time.monotonic() - started_at,
                        )
            if telemetry is not None:
                telemetry.increase(EncryptedFsMetrics.FS_WRITE_BYTES, blob_size)


def _decrypt_load(
    crypto: CryptoEngine,
    paths,
    keys,
    view: memoryview,
    offsets,
    block_size: int,
    capacity: EncryptedFsCapacityManager | None = None,
    telemetry: EncryptedFsTelemetry | None = None,
    concurrency: EncryptedFsConcurrency | None = None,
) -> None:
    """Read, tag-verify and unseal each block into the shared CPU buffer.
    A failed tag raises DecryptError -> job failure -> the engine treats the
    load as failed (never injects garbage KV)."""
    flat = view.cast("B")
    for path, key, off in zip(paths, keys, offsets):
        block_slot = (
            concurrency.block_slot("load") if concurrency is not None else nullcontext()
        )
        with block_slot:
            try:
                read_slot = (
                    concurrency.read_slot("load")
                    if concurrency is not None
                    else nullcontext()
                )
                with read_slot:
                    started_at = time.monotonic()
                    try:
                        if capacity is None:
                            fd = os.open(path, os.O_RDONLY)
                            try:
                                blob = read_blob(fd)
                            finally:
                                os.close(fd)
                        else:
                            blob = capacity.load_blob(path)
                    finally:
                        if telemetry is not None:
                            telemetry.observe(
                                EncryptedFsMetrics.FS_READ_SECONDS,
                                time.monotonic() - started_at,
                            )
                if telemetry is not None:
                    telemetry.increase(EncryptedFsMetrics.FS_READ_BYTES, len(blob))

                crypto_slot = (
                    concurrency.crypto_slot("load")
                    if concurrency is not None
                    else nullcontext()
                )
                with crypto_slot:
                    started_at = time.monotonic()
                    try:
                        plaintext = crypto.unseal(
                            blob, _TIER_KEY_NS, expected_aad=_aad(key)
                        )
                    except DecryptError:
                        if telemetry is not None:
                            telemetry.increase(EncryptedFsMetrics.DECRYPT_FAILURES)
                        raise
                    finally:
                        if telemetry is not None:
                            telemetry.observe(
                                EncryptedFsMetrics.DECRYPT_SECONDS,
                                time.monotonic() - started_at,
                            )
                if len(plaintext) != block_size:
                    raise ValueError(
                        f"sealed block {os.path.basename(path)} decrypted to "
                        f"{len(plaintext)} bytes, expected {block_size}"
                    )
            except Exception:
                removed = _remove_failed_blob(path, capacity, concurrency)
                if removed and telemetry is not None:
                    telemetry.increase(EncryptedFsMetrics.INVALIDATED_BLOCKS)
                raise

            started_at = time.monotonic()
            try:
                flat[off : off + block_size] = plaintext
            finally:
                if telemetry is not None:
                    telemetry.observe(
                        EncryptedFsMetrics.COPY_SECONDS,
                        time.monotonic() - started_at,
                        ("load",),
                    )
            if telemetry is not None:
                telemetry.increase(EncryptedFsMetrics.DECRYPTED_BYTES, len(plaintext))
            if capacity is not None:
                capacity.touch(path)


class EncryptedFileSystemTierManager(FileSystemTierManager):
    """FileSystemTierManager whose on-disk bytes are always AES-256-GCM
    SealedBlobs. Only the store/load byte tasks differ from the parent."""

    def __init__(
        self,
        *args,
        key_source: str = "env:SKV_MASTER_KEY",
        crypto_workers: int = 2,
        read_io_workers: int = 2,
        write_io_workers: int = 2,
        max_inflight_blocks: int = 4,
        n_read_threads: int = 4,
        n_write_threads: int = 4,
        max_bytes: int | None = None,
        min_free_bytes: int = 0,
        eviction_low_watermark: float = 0.9,
        **kwargs,
    ):
        root_dir = kwargs.get("root_dir")
        if not isinstance(root_dir, str):
            raise ValueError("encrypted_fs requires a string root_dir")
        for name, value in (
            ("n_read_threads", n_read_threads),
            ("n_write_threads", n_write_threads),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._telemetry = EncryptedFsTelemetry()
        self._concurrency = EncryptedFsConcurrency(
            self._telemetry,
            crypto_workers=crypto_workers,
            read_io_workers=read_io_workers,
            write_io_workers=write_io_workers,
            max_inflight_blocks=max_inflight_blocks,
        )
        super().__init__(
            *args,
            n_read_threads=n_read_threads,
            n_write_threads=n_write_threads,
            **kwargs,
        )
        # Ciphertext length is header+payload+tag: not O_DIRECT-alignable.
        self._use_o_direct = False
        master_hex = None
        if key_source.startswith("env:"):
            master_hex = os.environ.get(key_source[4:], "")
        self._crypto = CryptoEngine(KeyManager(master_hex), workers=crypto_workers)
        self._load_job_keys = {}
        self._capacity = (
            EncryptedFsCapacityManager(
                root_dir=root_dir,
                max_bytes=max_bytes,
                min_free_bytes=min_free_bytes,
                low_watermark=eviction_low_watermark,
            )
            if max_bytes is not None or min_free_bytes > 0
            else None
        )
        logger.info(
            "EncryptedFileSystemTier '%s': AES-256-GCM at rest under %s "
            "(global tier key; per-tenant isolation via cache_salt; "
            "max_bytes=%s; min_free_bytes=%d; low_watermark=%.3f; "
            "threads=%d+%d; pipeline=%d; crypto=%d; read_io=%d; write_io=%d)",
            self.tier_type,
            self.file_mapper.get_config_file_path(),
            str(max_bytes),
            min_free_bytes,
            eviction_low_watermark,
            n_read_threads,
            n_write_threads,
            max_inflight_blocks,
            crypto_workers,
            read_io_workers,
            write_io_workers,
        )

    @classmethod
    @override
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        return EncryptedFsMetrics.definitions()

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
            self._capacity,
            self._telemetry,
            self._concurrency,
        )
        enqueued_at = self._telemetry.job_enqueued("store")
        timed_task = functools.partial(
            run_timed_job,
            self._telemetry,
            "store",
            enqueued_at,
            task,
        )
        self._pool.enqueue_store(job_metadata.job_id, 1, [timed_task])

    @override
    def submit_load(self, job_metadata) -> None:
        self._load_job_keys[job_metadata.job_id] = list(job_metadata.keys)
        task = functools.partial(
            _decrypt_load,
            self._crypto,
            [self.file_mapper.get_file_name(k) for k in job_metadata.keys],
            list(job_metadata.keys),
            self._primary_kv_view,
            [int(bid) * self._block_size for bid in job_metadata.block_ids],
            self._block_size,
            self._capacity,
            self._telemetry,
            self._concurrency,
        )
        enqueued_at = self._telemetry.job_enqueued("load")
        timed_task = functools.partial(
            run_timed_job,
            self._telemetry,
            "load",
            enqueued_at,
            task,
        )
        self._pool.enqueue_load(job_metadata.job_id, 1, [timed_task])

    @override
    def get_finished_jobs(self):
        results = list(super().get_finished_jobs())
        for result in results:
            keys = self._load_job_keys.pop(result.job_id, None)
            if keys is not None and not result.success:
                self._lookup_manager.invalidate(keys)
        return results

    @override
    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = self._telemetry.take_stats()
        if self._capacity is None:
            return stats
        snapshot = self._capacity.snapshot()
        stats.set_gauge(EncryptedFsMetrics.CACHE_BYTES, snapshot.current_bytes)
        stats.set_gauge(EncryptedFsMetrics.CACHE_FILES, snapshot.current_files)
        usage = (
            snapshot.current_bytes / snapshot.max_bytes
            if snapshot.max_bytes is not None
            else 0.0
        )
        stats.set_gauge(EncryptedFsMetrics.CACHE_USAGE_PERC, usage)
        stats.set_gauge(EncryptedFsMetrics.DISK_FREE_BYTES, snapshot.disk_free_bytes)
        stats.increase_counter(EncryptedFsMetrics.EVICTED_BYTES, snapshot.evicted_bytes)
        stats.increase_counter(EncryptedFsMetrics.EVICTED_FILES, snapshot.evicted_files)
        stats.increase_counter(
            EncryptedFsMetrics.ADMISSION_REJECTIONS,
            snapshot.admission_rejections,
        )
        stats.increase_counter(
            EncryptedFsMetrics.STALE_TEMP_FILES_REMOVED,
            snapshot.stale_temp_files_removed,
        )
        return stats
