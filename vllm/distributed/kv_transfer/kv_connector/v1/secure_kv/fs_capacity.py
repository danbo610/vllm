# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capacity enforcement for the encrypted filesystem KV tier."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import stat
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import BinaryIO

from vllm.logger import init_logger

logger = init_logger(__name__)

_LOCK_FILE = ".encrypted_fs_capacity.lock"
_STATE_FILE = ".encrypted_fs_capacity.state"


class CacheCapacityError(RuntimeError):
    """The cache cannot admit a block without violating its limits."""


@dataclass(frozen=True)
class CapacitySnapshot:
    current_bytes: int
    current_files: int
    max_bytes: int | None
    min_free_bytes: int
    disk_free_bytes: int
    evicted_bytes: int
    evicted_files: int
    admission_rejections: int
    stale_temp_files_removed: int


@dataclass(frozen=True)
class _CacheState:
    total_bytes: int
    total_files: int


@dataclass(frozen=True)
class _Candidate:
    mtime_ns: int
    path: str
    size: int


class EncryptedFsCapacityManager:
    """Cross-thread and cross-process LRU capacity manager.

    AES work happens before ``store_blob``. The directory lock only covers
    admission, eviction, and the final file write, so crypto can still run in
    parallel while the hard capacity limit remains serialized.
    """

    def __init__(
        self,
        root_dir: str,
        max_bytes: int | None,
        min_free_bytes: int = 0,
        low_watermark: float = 0.9,
    ) -> None:
        if max_bytes is not None and (
            not isinstance(max_bytes, int)
            or isinstance(max_bytes, bool)
            or max_bytes <= 0
        ):
            raise ValueError("max_bytes must be a positive integer or None")
        if (
            not isinstance(min_free_bytes, int)
            or isinstance(min_free_bytes, bool)
            or min_free_bytes < 0
        ):
            raise ValueError("min_free_bytes must be a non-negative integer")
        if (
            not isinstance(low_watermark, int | float)
            or isinstance(low_watermark, bool)
            or not 0 < low_watermark <= 1
        ):
            raise ValueError("eviction_low_watermark must be in (0, 1]")

        self.root_dir = os.path.abspath(root_dir)
        self.max_bytes = max_bytes
        self.min_free_bytes = min_free_bytes
        self.low_watermark = low_watermark
        self._lock_path = os.path.join(self.root_dir, _LOCK_FILE)
        self._state_path = os.path.join(self.root_dir, _STATE_FILE)
        self._writer_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._current_bytes = 0
        self._current_files = 0
        self._evicted_bytes = 0
        self._evicted_files = 0
        self._admission_rejections = 0
        self._stale_temp_files_removed = 0

        os.makedirs(self.root_dir, exist_ok=True)
        with self._writer_lock, self._directory_lock(exclusive=True):
            stale_removed = self._remove_stale_temp_files_locked()
            state, _ = self._scan_locked()
            if self._needs_eviction(state, incoming_bytes=0):
                state = self._make_room_locked(incoming_bytes=0)
            self._write_state_locked(state)
            self._set_current(state)
            with self._stats_lock:
                self._stale_temp_files_removed += stale_removed

        logger.info(
            "Encrypted FS capacity initialized: root=%s bytes=%d files=%d "
            "max_bytes=%s min_free_bytes=%d low_watermark=%.3f",
            self.root_dir,
            state.total_bytes,
            state.total_files,
            str(self.max_bytes),
            self.min_free_bytes,
            self.low_watermark,
        )

    @contextmanager
    def _directory_lock(self, *, exclusive: bool) -> Iterator[BinaryIO]:
        with open(self._lock_path, "a+b") as lock_file:
            operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(lock_file.fileno(), operation)
            try:
                yield lock_file
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _iter_cache_files(self) -> Iterator[str]:
        for current_root, _, names in os.walk(self.root_dir):
            for name in names:
                if name.endswith(".bin"):
                    yield os.path.join(current_root, name)

    def _scan_locked(self) -> tuple[_CacheState, list[_Candidate]]:
        total_bytes = 0
        candidates: list[_Candidate] = []
        for path in self._iter_cache_files():
            try:
                file_stat = os.stat(path, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(file_stat.st_mode):
                continue
            total_bytes += file_stat.st_size
            candidates.append(
                _Candidate(file_stat.st_mtime_ns, path, file_stat.st_size)
            )
        return _CacheState(total_bytes, len(candidates)), candidates

    def _read_state_locked(self) -> _CacheState:
        try:
            with open(self._state_path, encoding="utf-8") as state_file:
                payload = json.load(state_file)
            total_bytes = int(payload["total_bytes"])
            total_files = int(payload["total_files"])
            if total_bytes < 0 or total_files < 0:
                raise ValueError("negative cache state")
            return _CacheState(total_bytes, total_files)
        except (
            FileNotFoundError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            state, _ = self._scan_locked()
            return state

    def _write_state_locked(self, state: _CacheState) -> None:
        temp_path = f"{self._state_path}.tmp{os.getpid()}_{threading.get_ident()}"
        try:
            with open(temp_path, "w", encoding="utf-8") as state_file:
                json.dump(
                    {
                        "total_bytes": state.total_bytes,
                        "total_files": state.total_files,
                    },
                    state_file,
                    separators=(",", ":"),
                )
            os.replace(temp_path, self._state_path)
        finally:
            with suppress(FileNotFoundError):
                os.remove(temp_path)

    def _needs_eviction(self, state: _CacheState, incoming_bytes: int) -> bool:
        over_cache_limit = (
            self.max_bytes is not None
            and state.total_bytes + incoming_bytes > self.max_bytes
        )
        disk_free = shutil.disk_usage(self.root_dir).free
        below_free_reserve = disk_free - incoming_bytes < self.min_free_bytes
        return over_cache_limit or below_free_reserve

    def _required_eviction_bytes(
        self,
        state: _CacheState,
        incoming_bytes: int,
        disk_free: int,
    ) -> int:
        cache_bytes = 0
        if (
            self.max_bytes is not None
            and state.total_bytes + incoming_bytes > self.max_bytes
        ):
            low_target = int(self.max_bytes * self.low_watermark)
            retained_before_write = max(low_target - incoming_bytes, 0)
            cache_bytes = max(state.total_bytes - retained_before_write, 0)

        free_bytes = max(
            self.min_free_bytes + incoming_bytes - disk_free,
            0,
        )
        return max(cache_bytes, free_bytes)

    def _make_room_locked(self, incoming_bytes: int) -> _CacheState:
        state, candidates = self._scan_locked()
        disk_free = shutil.disk_usage(self.root_dir).free
        required = self._required_eviction_bytes(state, incoming_bytes, disk_free)
        if required == 0:
            return state

        evicted_bytes = 0
        evicted_files = 0
        for candidate in sorted(
            candidates, key=lambda item: (item.mtime_ns, item.path)
        ):
            try:
                os.remove(candidate.path)
            except FileNotFoundError:
                continue
            evicted_bytes += candidate.size
            evicted_files += 1
            self._remove_empty_shard_dirs(candidate.path)
            if evicted_bytes >= required:
                break

        state = _CacheState(
            max(state.total_bytes - evicted_bytes, 0),
            max(state.total_files - evicted_files, 0),
        )
        effective_free = disk_free + evicted_bytes
        over_cache_limit = (
            self.max_bytes is not None
            and state.total_bytes + incoming_bytes > self.max_bytes
        )
        below_free_reserve = effective_free - incoming_bytes < self.min_free_bytes
        if evicted_files:
            self._record_eviction(state, evicted_bytes, evicted_files, incoming_bytes)
        if over_cache_limit or below_free_reserve:
            with self._stats_lock:
                self._admission_rejections += 1
            self._write_state_locked(state)
            self._set_current(state)
            raise CacheCapacityError(
                "encrypted_fs cannot admit block without violating capacity: "
                f"current={state.total_bytes}, incoming={incoming_bytes}, "
                f"max={self.max_bytes}, disk_free={effective_free}, "
                f"min_free={self.min_free_bytes}"
            )

        return state

    def _record_eviction(
        self,
        state: _CacheState,
        evicted_bytes: int,
        evicted_files: int,
        incoming_bytes: int,
    ) -> None:
        with self._stats_lock:
            self._evicted_bytes += evicted_bytes
            self._evicted_files += evicted_files
        logger.info(
            "Encrypted FS LRU eviction: files=%d bytes=%d "
            "remaining_bytes=%d incoming_bytes=%d",
            evicted_files,
            evicted_bytes,
            state.total_bytes,
            incoming_bytes,
        )

    @staticmethod
    def _remove_empty_shard_dirs(path: str) -> None:
        parent = os.path.dirname(path)
        for _ in range(2):
            try:
                os.rmdir(parent)
            except OSError:
                return
            parent = os.path.dirname(parent)

    def _remove_stale_temp_files_locked(self) -> int:
        removed = 0
        for current_root, _, names in os.walk(self.root_dir):
            for name in names:
                if ".tmp" not in name:
                    continue
                path = os.path.join(current_root, name)
                try:
                    os.remove(path)
                except FileNotFoundError:
                    continue
                removed += 1
        if removed:
            logger.info("Removed %d stale encrypted FS temp files", removed)
        return removed

    def _set_current(self, state: _CacheState) -> None:
        with self._stats_lock:
            self._current_bytes = state.total_bytes
            self._current_files = state.total_files

    def store_blob(self, path: str, blob: bytes) -> bool:
        """Atomically admit and store one encrypted block.

        Returns ``False`` when another writer has already stored the path.
        """
        blob_size = len(blob)
        if self.max_bytes is not None and blob_size > self.max_bytes:
            with self._stats_lock:
                self._admission_rejections += 1
            raise CacheCapacityError(
                f"encrypted block size {blob_size} exceeds max_bytes {self.max_bytes}"
            )

        with self._writer_lock, self._directory_lock(exclusive=True):
            if os.path.exists(path):
                return False
            state = self._read_state_locked()
            if self._needs_eviction(state, blob_size):
                state = self._make_room_locked(blob_size)

            admitted_state = _CacheState(
                state.total_bytes + blob_size,
                state.total_files + 1,
            )
            # Account before the data write. A process crash can over-count,
            # which causes early eviction rather than exceeding the hard cap.
            self._write_state_locked(admitted_state)

            os.makedirs(os.path.dirname(path), exist_ok=True)
            temp_path = f"{path}.tmp{os.getpid()}_{threading.get_ident()}"
            try:
                with open(temp_path, "xb") as cache_file:
                    cache_file.write(blob)
                os.replace(temp_path, path)
            except Exception:
                self._write_state_locked(state)
                with suppress(FileNotFoundError):
                    os.remove(temp_path)
                raise

            self._set_current(admitted_state)
            return True

    def load_blob(self, path: str) -> bytes:
        """Read one block while preventing concurrent eviction."""
        with (
            self._directory_lock(exclusive=False),
            open(path, "rb") as cache_file,
        ):
            return cache_file.read()

    def remove_blob(self, path: str) -> bool:
        """Remove one invalid block and update shared capacity state."""
        with self._writer_lock, self._directory_lock(exclusive=True):
            try:
                file_stat = os.stat(path, follow_symlinks=False)
            except FileNotFoundError:
                return False

            state = self._read_state_locked()
            try:
                os.remove(path)
            except FileNotFoundError:
                return False

            updated_state = _CacheState(
                max(state.total_bytes - file_stat.st_size, 0),
                max(state.total_files - 1, 0),
            )
            self._write_state_locked(updated_state)
            self._set_current(updated_state)
            self._remove_empty_shard_dirs(path)
            return True

    def touch(self, path: str) -> None:
        """Mark a successfully decrypted block as recently used."""
        with self._directory_lock(exclusive=False), suppress(FileNotFoundError):
            os.utime(path, None, follow_symlinks=False)

    def snapshot(self, *, reset_counters: bool = True) -> CapacitySnapshot:
        """Return current gauges and process-local counter deltas."""
        with self._directory_lock(exclusive=False):
            state = self._read_state_locked()
            disk_free_bytes = shutil.disk_usage(self.root_dir).free
        with self._stats_lock:
            self._current_bytes = state.total_bytes
            self._current_files = state.total_files
            snapshot = CapacitySnapshot(
                current_bytes=self._current_bytes,
                current_files=self._current_files,
                max_bytes=self.max_bytes,
                min_free_bytes=self.min_free_bytes,
                disk_free_bytes=disk_free_bytes,
                evicted_bytes=self._evicted_bytes,
                evicted_files=self._evicted_files,
                admission_rejections=self._admission_rejections,
                stale_temp_files_removed=self._stale_temp_files_removed,
            )
            if reset_counters:
                self._evicted_bytes = 0
                self._evicted_files = 0
                self._admission_rejections = 0
                self._stale_temp_files_removed = 0
            return snapshot
