# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thread-safe metrics for the encrypted filesystem KV tier."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Literal

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.v1.kv_offload.base import (
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadingMetricMetadata,
)

Operation = Literal["load", "store"]

_STAGE_TIME_BUCKETS = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
)


class EncryptedFsMetrics:
    CACHE_BYTES = "vllm:kv_offload_encrypted_fs_cache_bytes"
    CACHE_FILES = "vllm:kv_offload_encrypted_fs_cache_files"
    CACHE_USAGE_PERC = "vllm:kv_offload_encrypted_fs_cache_usage_perc"
    DISK_FREE_BYTES = "vllm:kv_offload_encrypted_fs_disk_free_bytes"
    EVICTED_BYTES = "vllm:kv_offload_encrypted_fs_evicted_bytes"
    EVICTED_FILES = "vllm:kv_offload_encrypted_fs_evicted_files"
    ADMISSION_REJECTIONS = "vllm:kv_offload_encrypted_fs_admission_rejections"
    STALE_TEMP_FILES_REMOVED = "vllm:kv_offload_encrypted_fs_stale_temp_files_removed"

    ENCRYPT_SECONDS = "vllm:kv_offload_encrypted_fs_encrypt_seconds"
    DECRYPT_SECONDS = "vllm:kv_offload_encrypted_fs_decrypt_seconds"
    FS_READ_SECONDS = "vllm:kv_offload_encrypted_fs_read_seconds"
    FS_WRITE_SECONDS = "vllm:kv_offload_encrypted_fs_write_seconds"
    COPY_SECONDS = "vllm:kv_offload_encrypted_fs_plaintext_copy_seconds"
    JOB_QUEUE_SECONDS = "vllm:kv_offload_encrypted_fs_job_queue_seconds"
    JOB_TOTAL_SECONDS = "vllm:kv_offload_encrypted_fs_job_total_seconds"

    ENCRYPTED_BYTES = "vllm:kv_offload_encrypted_fs_encrypted_bytes"
    DECRYPTED_BYTES = "vllm:kv_offload_encrypted_fs_decrypted_bytes"
    FS_READ_BYTES = "vllm:kv_offload_encrypted_fs_read_bytes"
    FS_WRITE_BYTES = "vllm:kv_offload_encrypted_fs_written_bytes"
    INVALIDATED_BLOCKS = "vllm:kv_offload_encrypted_fs_invalidated_blocks"
    STORE_FAILURES = "vllm:kv_offload_encrypted_fs_store_failures"
    LOAD_FAILURES = "vllm:kv_offload_encrypted_fs_load_failures"
    DECRYPT_FAILURES = "vllm:kv_offload_encrypted_fs_decrypt_failures"
    QUEUE_DEPTH = "vllm:kv_offload_encrypted_fs_queue_depth"
    INFLIGHT_JOBS = "vllm:kv_offload_encrypted_fs_inflight_jobs"

    @classmethod
    def definitions(cls) -> dict[str, OffloadingMetricMetadata]:
        operation_label = ("operation",)
        return {
            cls.CACHE_BYTES: OffloadingGaugeMetadata(
                documentation="Encrypted filesystem cache size in bytes."
            ),
            cls.CACHE_FILES: OffloadingGaugeMetadata(
                documentation="Number of encrypted filesystem cache blocks."
            ),
            cls.CACHE_USAGE_PERC: OffloadingGaugeMetadata(
                documentation="Encrypted filesystem max capacity usage."
            ),
            cls.DISK_FREE_BYTES: OffloadingGaugeMetadata(
                documentation="Free bytes on the encrypted cache filesystem."
            ),
            cls.EVICTED_BYTES: OffloadingCounterMetadata(
                documentation="Bytes evicted from the encrypted filesystem cache."
            ),
            cls.EVICTED_FILES: OffloadingCounterMetadata(
                documentation="Blocks evicted from the encrypted filesystem cache."
            ),
            cls.ADMISSION_REJECTIONS: OffloadingCounterMetadata(
                documentation="Encrypted blocks rejected by capacity limits."
            ),
            cls.STALE_TEMP_FILES_REMOVED: OffloadingCounterMetadata(
                documentation="Stale encrypted cache temp files removed."
            ),
            cls.ENCRYPT_SECONDS: OffloadingHistogramMetadata(
                documentation="Time spent encrypting one KV block.",
                buckets=_STAGE_TIME_BUCKETS,
            ),
            cls.DECRYPT_SECONDS: OffloadingHistogramMetadata(
                documentation="Time spent decrypting one KV block.",
                buckets=_STAGE_TIME_BUCKETS,
            ),
            cls.FS_READ_SECONDS: OffloadingHistogramMetadata(
                documentation="Time spent reading one encrypted KV block.",
                buckets=_STAGE_TIME_BUCKETS,
            ),
            cls.FS_WRITE_SECONDS: OffloadingHistogramMetadata(
                documentation="Time spent admitting and writing one encrypted block.",
                buckets=_STAGE_TIME_BUCKETS,
            ),
            cls.COPY_SECONDS: OffloadingHistogramMetadata(
                documentation="Time spent copying plaintext KV bytes.",
                labelnames=operation_label,
                buckets=_STAGE_TIME_BUCKETS,
            ),
            cls.JOB_QUEUE_SECONDS: OffloadingHistogramMetadata(
                documentation="Time encrypted FS jobs wait before execution.",
                labelnames=operation_label,
                buckets=_STAGE_TIME_BUCKETS,
            ),
            cls.JOB_TOTAL_SECONDS: OffloadingHistogramMetadata(
                documentation="End-to-end encrypted FS job latency.",
                labelnames=operation_label,
                buckets=_STAGE_TIME_BUCKETS,
            ),
            cls.ENCRYPTED_BYTES: OffloadingCounterMetadata(
                documentation="Plaintext KV bytes successfully encrypted."
            ),
            cls.DECRYPTED_BYTES: OffloadingCounterMetadata(
                documentation="Plaintext KV bytes successfully decrypted."
            ),
            cls.FS_READ_BYTES: OffloadingCounterMetadata(
                documentation="Ciphertext bytes read from the filesystem."
            ),
            cls.FS_WRITE_BYTES: OffloadingCounterMetadata(
                documentation="Ciphertext bytes written to the filesystem."
            ),
            cls.INVALIDATED_BLOCKS: OffloadingCounterMetadata(
                documentation="Unreadable encrypted blocks removed from the cache."
            ),
            cls.STORE_FAILURES: OffloadingCounterMetadata(
                documentation="Encrypted filesystem store jobs that failed."
            ),
            cls.LOAD_FAILURES: OffloadingCounterMetadata(
                documentation="Encrypted filesystem load jobs that failed."
            ),
            cls.DECRYPT_FAILURES: OffloadingCounterMetadata(
                documentation="Encrypted blocks that failed authentication or parsing."
            ),
            cls.QUEUE_DEPTH: OffloadingGaugeMetadata(
                documentation="Encrypted filesystem jobs waiting for a worker.",
                labelnames=operation_label,
            ),
            cls.INFLIGHT_JOBS: OffloadingGaugeMetadata(
                documentation="Encrypted filesystem jobs queued or executing.",
                labelnames=operation_label,
            ),
        }


class EncryptedFsTelemetry:
    """Collect worker-thread observations for scheduler-thread export."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats = OffloadingConnectorStats()
        self._queue_depth: dict[Operation, int] = {"load": 0, "store": 0}
        self._inflight_jobs: dict[Operation, int] = {"load": 0, "store": 0}

    def job_enqueued(self, operation: Operation) -> float:
        enqueued_at = time.monotonic()
        with self._lock:
            self._queue_depth[operation] += 1
            self._inflight_jobs[operation] += 1
        return enqueued_at

    def job_started(self, operation: Operation, enqueued_at: float) -> None:
        with self._lock:
            self._queue_depth[operation] -= 1
            self._stats.observe_histogram(
                EncryptedFsMetrics.JOB_QUEUE_SECONDS,
                time.monotonic() - enqueued_at,
                (operation,),
            )

    def job_finished(
        self, operation: Operation, enqueued_at: float, *, success: bool
    ) -> None:
        failure_metric = (
            EncryptedFsMetrics.LOAD_FAILURES
            if operation == "load"
            else EncryptedFsMetrics.STORE_FAILURES
        )
        with self._lock:
            self._inflight_jobs[operation] -= 1
            self._stats.observe_histogram(
                EncryptedFsMetrics.JOB_TOTAL_SECONDS,
                time.monotonic() - enqueued_at,
                (operation,),
            )
            if not success:
                self._stats.increase_counter(failure_metric)

    def observe(
        self,
        metric: str,
        value: int | float,
        labelvalues: tuple[str, ...] = (),
    ) -> None:
        with self._lock:
            self._stats.observe_histogram(metric, value, labelvalues)

    def increase(
        self,
        metric: str,
        value: int | float = 1,
        labelvalues: tuple[str, ...] = (),
    ) -> None:
        with self._lock:
            self._stats.increase_counter(metric, value, labelvalues)

    def take_stats(self) -> OffloadingConnectorStats:
        with self._lock:
            stats = self._stats
            self._stats = OffloadingConnectorStats()
            for operation in ("load", "store"):
                labelvalues = (operation,)
                stats.set_gauge(
                    EncryptedFsMetrics.QUEUE_DEPTH,
                    self._queue_depth[operation],
                    labelvalues,
                )
                stats.set_gauge(
                    EncryptedFsMetrics.INFLIGHT_JOBS,
                    self._inflight_jobs[operation],
                    labelvalues,
                )
            for metric in (
                EncryptedFsMetrics.ENCRYPTED_BYTES,
                EncryptedFsMetrics.DECRYPTED_BYTES,
                EncryptedFsMetrics.FS_READ_BYTES,
                EncryptedFsMetrics.FS_WRITE_BYTES,
                EncryptedFsMetrics.INVALIDATED_BLOCKS,
                EncryptedFsMetrics.STORE_FAILURES,
                EncryptedFsMetrics.LOAD_FAILURES,
                EncryptedFsMetrics.DECRYPT_FAILURES,
            ):
                stats.increase_counter(metric, 0)
            return stats


def run_timed_job(
    telemetry: EncryptedFsTelemetry,
    operation: Operation,
    enqueued_at: float,
    task: Callable[[], None],
) -> None:
    telemetry.job_started(operation, enqueued_at)
    try:
        task()
    except Exception:
        telemetry.job_finished(operation, enqueued_at, success=False)
        raise
    telemetry.job_finished(operation, enqueued_at, success=True)
