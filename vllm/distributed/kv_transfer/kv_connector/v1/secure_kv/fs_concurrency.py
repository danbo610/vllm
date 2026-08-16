# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded stage concurrency for encrypted filesystem jobs."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

from vllm.distributed.kv_transfer.kv_connector.v1.secure_kv.fs_metrics import (
    ConcurrencyResource,
    EncryptedFsTelemetry,
    Operation,
)


class EncryptedFsConcurrency:
    def __init__(
        self,
        telemetry: EncryptedFsTelemetry,
        *,
        crypto_workers: int,
        read_io_workers: int,
        write_io_workers: int,
        max_inflight_blocks: int,
    ) -> None:
        limits: dict[ConcurrencyResource, int] = {
            "crypto": crypto_workers,
            "read_io": read_io_workers,
            "write_io": write_io_workers,
            "pipeline": max_inflight_blocks,
        }
        for name, value in limits.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} concurrency must be a positive integer")

        self.limits = limits
        self._telemetry = telemetry
        self._pipeline = threading.BoundedSemaphore(max_inflight_blocks)
        self._crypto = threading.BoundedSemaphore(crypto_workers)
        self._read_io = threading.BoundedSemaphore(read_io_workers)
        self._write_io = threading.BoundedSemaphore(write_io_workers)
        telemetry.set_concurrency_limits(limits)

    @contextmanager
    def block_slot(self, operation: Operation) -> Iterator[None]:
        with self._slot(self._pipeline, "pipeline", operation):
            yield

    @contextmanager
    def crypto_slot(self, operation: Operation) -> Iterator[None]:
        with self._slot(self._crypto, "crypto", operation):
            yield

    @contextmanager
    def read_slot(self, operation: Operation) -> Iterator[None]:
        with self._slot(self._read_io, "read_io", operation):
            yield

    @contextmanager
    def write_slot(self, operation: Operation) -> Iterator[None]:
        with self._slot(self._write_io, "write_io", operation):
            yield

    @contextmanager
    def _slot(
        self,
        semaphore: threading.BoundedSemaphore,
        resource: ConcurrencyResource,
        operation: Operation,
    ) -> Iterator[None]:
        started_at = time.monotonic()
        semaphore.acquire()
        self._telemetry.slot_acquired(
            resource, operation, time.monotonic() - started_at
        )
        try:
            yield
        finally:
            self._telemetry.slot_released(resource, operation)
            semaphore.release()
