# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for FileSystemTierManager.

These tests use real disk I/O to verify the filesystem tier implementation.
The tier manager writes KV cache blocks to disk and reads them back, verifying
data integrity throughout the process.
"""

import errno
import mmap
import os
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.secure_kv.fs_capacity import (
    EncryptedFsCapacityManager,
)
from vllm.distributed.kv_transfer.kv_connector.v1.secure_kv.fs_metrics import (
    EncryptedFsMetrics,
)
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadingKVEventsConfig,
    OffloadKey,
    ReqContext,
    ScheduleEndContext,
    make_offload_key,
)
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.tiering.base import JobMetadata
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm.v1.kv_offload.tiering.fs.manager import (
    FileSystemTierManager,
)
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NUM_BLOCKS = 8
_BLOCK_ELEMENTS = 128 * mmap.PAGESIZE  # 2MB per block for pagesize 4096.
_DTYPE: torch.dtype = torch.float32
_CTX = ReqContext(req_id="test")


def _make_offloading_spec(
    enable_kv_cache_events: bool = False,
    *,
    tp_size: int = 1,
    rank: int = 0,
    world_size: int | None = None,
    replicated_layout: bool = False,
    is_parallelism_agnostic: bool = False,
) -> MagicMock:
    """Mock spec with an explicit global KV events flag."""
    if world_size is None:
        world_size = tp_size
    spec = MagicMock()
    spec.config = OffloadingConfig(
        groups=(),
        worker_kv_bytes_per_block=0,
        enable_kv_cache_events=enable_kv_cache_events,
        extra_config={},
        engine_id="test-engine",
        model=OffloadingModelConfig(name="test-model", dtype="float32"),
        cache=OffloadingCacheConfig(tokens_per_hash=16, blocks_per_chunk=1),
        parallel=OffloadingParallelConfig(
            rank=rank,
            world_size=world_size,
            tp_size=tp_size,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            is_parallelism_agnostic=is_parallelism_agnostic,
        ),
        replicated_layout=replicated_layout,
    )
    spec.blocks_per_chunk = 1
    spec.kv_events_config = OffloadingKVEventsConfig(
        enable_kv_cache_events=enable_kv_cache_events,
        self_describing_kv_events=False,
    )
    return spec


_MOCK_OFFLOADING_SPEC = _make_offloading_spec(enable_kv_cache_events=False)


def key(n: int) -> OffloadKey:
    return make_offload_key(n.to_bytes(8, "big"), 0)


def make_job(
    job_id: int,
    keys: list[OffloadKey],
    block_ids: list[int] | None = None,
    is_promotion: bool = False,
) -> JobMetadata:
    if block_ids is None:
        block_ids = list(range(len(keys)))
    return JobMetadata(
        job_id=job_id,
        keys=keys,
        block_ids=np.array(block_ids, dtype=np.int64),
        is_promotion=is_promotion,
        req_context=_CTX,
    )


def drain(tier: FileSystemTierManager) -> list:
    """Block until all in-flight jobs finish, then collect results."""
    tier.drain_jobs()
    return list(tier.get_finished_jobs())


def lookup_and_wait(
    tier: FileSystemTierManager,
    keys: list[OffloadKey],
    ctx: ReqContext = _CTX,
    timeout: float = 1.0,
) -> list[LookupResult]:
    """Perform a full async lookup cycle and return resolved results."""
    for k in keys:
        tier.lookup(k, ctx)
    tier.on_schedule_end(ScheduleEndContext(new_req_ids=[], preempted_req_ids=()))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not tier._lookup_manager._pending_results.empty():
            break
        time.sleep(0.01)
    return [tier.lookup(k, ctx) for k in keys]


def _page_aligned_zero_tensor(
    num_blocks: int, block_elements: int, dtype: torch.dtype = _DTYPE
) -> torch.Tensor:
    page_size = mmap.PAGESIZE
    dtype_num_bytes = torch.tensor([], dtype=dtype).element_size()

    num_bytes = num_blocks * block_elements * dtype_num_bytes
    num_bytes_aligned = num_bytes + page_size
    t = torch.zeros(num_bytes_aligned, dtype=torch.uint8)

    ptr = t.data_ptr()
    alignment_offset = ptr % page_size
    # Move tensor to next page regardless.
    shift = page_size - alignment_offset
    t = t[shift : shift + num_bytes]
    return t.view(dtype).view(num_blocks, block_elements)


def _page_aligned_rand_tensor(
    num_blocks: int, block_elements: int, dtype: torch.dtype = _DTYPE
) -> torch.Tensor:
    rand_tensor = _page_aligned_zero_tensor(num_blocks, block_elements)
    rand_tensor[:] = torch.rand(num_blocks, block_elements, dtype=dtype)
    return rand_tensor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fs_tier(tmp_path):
    tensor = _page_aligned_zero_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)
    mock_view = memoryview(tensor.numpy())
    tier = FileSystemTierManager(
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        primary_kv_view=mock_view,
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=4,
        n_write_threads=4,
    )
    yield tier, tensor
    tier.shutdown()


@pytest.fixture
def fs_tier_with_events(tmp_path):
    tensor = _page_aligned_zero_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)
    mock_view = memoryview(tensor.numpy())
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(enable_kv_cache_events=True),
        primary_kv_view=mock_view,
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=4,
        n_write_threads=4,
        enable_kv_events=True,
        locality="LOCAL",
    )
    yield tier
    tier.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_lookup_empty_tier(fs_tier):
    tier, _ = fs_tier
    results = lookup_and_wait(tier, [key(1), key(2)])
    assert results == [LookupResult.MISS, LookupResult.MISS]


def test_store_creates_file_and_lookup_succeeds(fs_tier):
    tier, _ = fs_tier
    job = make_job(1, [key(1)], [0])
    tier.submit_store(job)
    results = drain(tier)
    assert len(results) == 1
    assert results[0].success
    assert lookup_and_wait(tier, [key(1)]) == [LookupResult.HIT]
    dest = tier.file_mapper.get_file_name(key(1))
    assert os.path.exists(dest), f"Expected file at {dest}"


def test_store_then_load_roundtrip(fs_tier):
    tier, _ = fs_tier
    job_s = make_job(1, [key(1), key(2)], [0, 1])
    tier.submit_store(job_s)
    store_results = drain(tier)
    assert all(r.success for r in store_results)

    assert lookup_and_wait(tier, [key(1), key(2)]) == [
        LookupResult.HIT,
        LookupResult.HIT,
    ]

    job_l = make_job(2, [key(1), key(2)], [2, 3], is_promotion=True)
    tier.submit_load(job_l)
    load_results = drain(tier)
    assert all(r.success for r in load_results)
    # Blocks stay on disk after load
    assert lookup_and_wait(tier, [key(1), key(2)]) == [
        LookupResult.HIT,
        LookupResult.HIT,
    ]


def test_invalid_path_raises_at_construction():
    """Construction must fail immediately when the config file cannot be written."""
    tensor = _page_aligned_zero_tensor(32, _BLOCK_ELEMENTS)
    mock_view = memoryview(tensor.numpy())

    with pytest.raises(OSError):
        FileSystemTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=mock_view,
            tier_type="fs",
            root_dir="/dev/null/invalid_path",
        )


@pytest.mark.parametrize("locality", ["local", ""])
def test_invalid_locality_raises_at_construction(tmp_path, locality):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)

    with pytest.raises(ValueError, match="Locality"):
        FileSystemTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=memoryview(tensor.numpy()),
            tier_type="fs",
            root_dir=str(tmp_path),
            locality=locality,
        )


def test_factory_forwards_locality_to_fs_tier(tmp_path):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    tier = SecondaryTierFactory.create_secondary_tier(
        {
            "type": "fs",
            "root_dir": str(tmp_path),
            "n_read_threads": 1,
            "n_write_threads": 1,
            "locality": "LOCAL",
        },
        memoryview(tensor.numpy()),
        _MOCK_OFFLOADING_SPEC,
    )
    try:
        assert isinstance(tier, FileSystemTierManager)
        assert tier.locality is Locality.LOCAL
    finally:
        tier.shutdown()


def test_factory_configures_encrypted_fs_capacity_and_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("SKV_MASTER_KEY", "00" * 16)
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    max_bytes = 64 * 1024 * 1024
    tier = SecondaryTierFactory.create_secondary_tier(
        {
            "type": "encrypted_fs",
            "root_dir": str(tmp_path),
            "max_bytes": max_bytes,
            "min_free_bytes": 0,
            "eviction_low_watermark": 0.8,
            "crypto_workers": 2,
            "read_io_workers": 3,
            "write_io_workers": 1,
            "max_inflight_blocks": 4,
            "n_read_threads": 1,
            "n_write_threads": 1,
        },
        memoryview(tensor.numpy()),
        _MOCK_OFFLOADING_SPEC,
    )
    try:
        assert tier._capacity is not None
        assert tier._capacity.max_bytes == max_bytes
        assert tier._capacity.low_watermark == 0.8

        definitions = tier.build_metric_definitions({})
        assert set(definitions) == set(EncryptedFsMetrics.definitions())
        assert definitions[EncryptedFsMetrics.QUEUE_DEPTH].labelnames == ("operation",)
        assert definitions[EncryptedFsMetrics.JOB_TOTAL_SECONDS].labelnames == (
            "operation",
        )
        assert definitions[EncryptedFsMetrics.CONCURRENCY_WAIT_SECONDS].labelnames == (
            "resource",
            "operation",
        )

        stats = tier.get_stats()
        assert stats is not None
        reduced = stats.reduce()
        assert reduced["vllm:kv_offload_encrypted_fs_cache_bytes"] == 0
        assert reduced["vllm:kv_offload_encrypted_fs_cache_files"] == 0
        assert reduced["vllm:kv_offload_encrypted_fs_cache_usage_perc"] == 0
        assert reduced["vllm:kv_offload_encrypted_fs_disk_free_bytes"] > 0
        assert reduced["vllm:kv_offload_encrypted_fs_evicted_bytes"] == 0
        assert reduced["vllm:kv_offload_encrypted_fs_evicted_files"] == 0
        assert reduced["vllm:kv_offload_encrypted_fs_admission_rejections"] == 0
        assert reduced["vllm:kv_offload_encrypted_fs_stale_temp_files_removed"] == 0
        assert reduced[EncryptedFsMetrics.STORE_FAILURES] == 0
        assert reduced[EncryptedFsMetrics.LOAD_FAILURES] == 0
        assert reduced[EncryptedFsMetrics.DECRYPT_FAILURES] == 0
        assert reduced[f"{EncryptedFsMetrics.QUEUE_DEPTH}:('load',)"] == 0
        assert reduced[f"{EncryptedFsMetrics.QUEUE_DEPTH}:('store',)"] == 0
        assert reduced[f"{EncryptedFsMetrics.INFLIGHT_JOBS}:('load',)"] == 0
        assert reduced[f"{EncryptedFsMetrics.INFLIGHT_JOBS}:('store',)"] == 0
        assert reduced[f"{EncryptedFsMetrics.CONCURRENCY_LIMIT}:('pipeline',)"] == 4
        assert reduced[f"{EncryptedFsMetrics.CONCURRENCY_LIMIT}:('crypto',)"] == 2
        assert reduced[f"{EncryptedFsMetrics.CONCURRENCY_LIMIT}:('read_io',)"] == 3
        assert reduced[f"{EncryptedFsMetrics.CONCURRENCY_LIMIT}:('write_io',)"] == 1
        for resource in ("pipeline", "crypto", "read_io", "write_io"):
            for operation in ("load", "store"):
                assert (
                    reduced[
                        f"{EncryptedFsMetrics.ACTIVE_SLOTS}:"
                        f"('{resource}', '{operation}')"
                    ]
                    == 0
                )
    finally:
        tier.shutdown()


def test_encrypted_fs_limits_crypto_and_write_concurrency(tmp_path, monkeypatch):
    monkeypatch.setenv("SKV_MASTER_KEY", "00" * 16)
    tensor = _page_aligned_rand_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)
    tier = SecondaryTierFactory.create_secondary_tier(
        {
            "type": "encrypted_fs",
            "root_dir": str(tmp_path),
            "max_bytes": 64 * 1024 * 1024,
            "min_free_bytes": 0,
            "crypto_workers": 2,
            "read_io_workers": 2,
            "write_io_workers": 1,
            "max_inflight_blocks": 3,
            "n_read_threads": 4,
            "n_write_threads": 4,
        },
        memoryview(tensor.numpy()),
        _MOCK_OFFLOADING_SPEC,
    )
    lock = threading.Lock()
    active = {"crypto": 0, "write_io": 0}
    peak = {"crypto": 0, "write_io": 0}

    def track_stage(stage, task, *args, **kwargs):
        with lock:
            active[stage] += 1
            peak[stage] = max(peak[stage], active[stage])
        try:
            time.sleep(0.03)
            return task(*args, **kwargs)
        finally:
            with lock:
                active[stage] -= 1

    original_seal = tier._crypto.seal_parts
    original_store = tier._capacity.store_blob_parts
    monkeypatch.setattr(
        tier._crypto,
        "seal_parts",
        lambda *args, **kwargs: track_stage("crypto", original_seal, *args, **kwargs),
    )
    monkeypatch.setattr(
        tier._capacity,
        "store_blob_parts",
        lambda *args, **kwargs: track_stage(
            "write_io", original_store, *args, **kwargs
        ),
    )

    try:
        for job_id in range(6):
            tier.submit_store(make_job(job_id, [key(job_id)], [job_id]))
        results = drain(tier)

        assert len(results) == 6
        assert all(result.success for result in results)
        assert peak == {"crypto": 2, "write_io": 1}

        stats = tier.get_stats()
        assert stats is not None
        values = stats.data["data"]
        wait_metric = values[EncryptedFsMetrics.CONCURRENCY_WAIT_SECONDS]
        assert len(wait_metric[("pipeline", "store")]) == 6
        assert len(wait_metric[("crypto", "store")]) == 6
        assert len(wait_metric[("write_io", "store")]) == 6
        for resource in ("pipeline", "crypto", "read_io", "write_io"):
            for operation in ("load", "store"):
                assert (
                    values[EncryptedFsMetrics.ACTIVE_SLOTS][(resource, operation)] == 0
                )
    finally:
        tier.shutdown()


def test_encrypted_fs_buffered_io_advises_and_drops_cache(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        os,
        "posix_fallocate",
        lambda fd, offset, length: calls.append(("fallocate", offset, length)),
    )
    monkeypatch.setattr(
        os,
        "posix_fadvise",
        lambda fd, offset, length, advice: calls.append(("fadvise", advice)),
    )
    monkeypatch.setattr(os, "fdatasync", lambda fd: calls.append(("fdatasync",)))
    capacity = EncryptedFsCapacityManager(
        str(tmp_path), max_bytes=1024 * 1024, min_free_bytes=0
    )
    path = str(tmp_path / "block.bin")
    blob = b"header" + b"ciphertext" * 1024

    assert capacity.store_blob_parts(path, (blob[:6], blob[6:]))
    assert capacity.load_blob(path) == blob
    assert ("fallocate", 0, len(blob)) in calls
    assert calls.count(("fdatasync",)) == 1
    assert calls.count(("fadvise", os.POSIX_FADV_SEQUENTIAL)) == 2
    assert calls.count(("fadvise", os.POSIX_FADV_DONTNEED)) == 2


def test_encrypted_fs_unsupported_preallocation_falls_back(tmp_path, monkeypatch):
    def unsupported(fd, offset, length):
        raise OSError(errno.EOPNOTSUPP, "not supported")

    monkeypatch.setattr(os, "posix_fallocate", unsupported)
    capacity = EncryptedFsCapacityManager(
        str(tmp_path), max_bytes=1024 * 1024, min_free_bytes=0
    )
    path = str(tmp_path / "block.bin")

    assert capacity.store_blob(path, b"ciphertext")
    assert capacity.load_blob(path) == b"ciphertext"


def test_encrypted_fs_sync_failure_rolls_back_admission(tmp_path, monkeypatch):
    def fail_sync(fd):
        raise OSError(errno.EIO, "sync failed")

    monkeypatch.setattr(os, "fdatasync", fail_sync)
    capacity = EncryptedFsCapacityManager(
        str(tmp_path), max_bytes=1024 * 1024, min_free_bytes=0
    )
    path = str(tmp_path / "block.bin")

    with pytest.raises(OSError, match="sync failed"):
        capacity.store_blob(path, b"ciphertext")

    snapshot = capacity.snapshot(reset_counters=False)
    assert not os.path.exists(path)
    assert not list(tmp_path.glob("*.tmp*"))
    assert snapshot.current_bytes == 0
    assert snapshot.current_files == 0


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("crypto_workers", 0),
        ("read_io_workers", False),
        ("write_io_workers", -1),
        ("max_inflight_blocks", 0),
        ("n_read_threads", 0),
        ("n_write_threads", False),
    ],
)
def test_encrypted_fs_rejects_invalid_concurrency(tmp_path, monkeypatch, option, value):
    monkeypatch.setenv("SKV_MASTER_KEY", "00" * 16)
    tensor = _page_aligned_zero_tensor(1, _BLOCK_ELEMENTS)

    with pytest.raises(ValueError, match="positive integer"):
        SecondaryTierFactory.create_secondary_tier(
            {
                "type": "encrypted_fs",
                "root_dir": str(tmp_path),
                option: value,
            },
            memoryview(tensor.numpy()),
            _MOCK_OFFLOADING_SPEC,
        )


def test_encrypted_fs_failed_decrypt_removes_blob_and_invalidates_hit(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SKV_MASTER_KEY", "00" * 16)
    tensor = _page_aligned_rand_tensor(4, _BLOCK_ELEMENTS)
    tier = SecondaryTierFactory.create_secondary_tier(
        {
            "type": "encrypted_fs",
            "root_dir": str(tmp_path),
            "max_bytes": 64 * 1024 * 1024,
            "min_free_bytes": 0,
            "n_read_threads": 1,
            "n_write_threads": 1,
        },
        memoryview(tensor.numpy()),
        _MOCK_OFFLOADING_SPEC,
    )
    try:
        block_key = key(123)
        tier.submit_store(make_job(1, [block_key], [0]))
        assert drain(tier)[0].success
        assert lookup_and_wait(tier, [block_key]) == [LookupResult.HIT]

        path = tier.file_mapper.get_file_name(block_key)
        with open(path, "r+b") as cache_file:
            cache_file.seek(-1, os.SEEK_END)
            final_byte = cache_file.read(1)
            cache_file.seek(-1, os.SEEK_END)
            cache_file.write(bytes([final_byte[0] ^ 1]))

        tier.submit_load(make_job(2, [block_key], [1], is_promotion=True))
        result = drain(tier)[0]
        assert not result.success
        assert not os.path.exists(path)
        assert tier.lookup(block_key, _CTX) is LookupResult.MISS
        assert tier._capacity.snapshot().current_files == 0
        assert tier._capacity.snapshot().current_bytes == 0

        stats = tier.get_stats()
        assert stats is not None
        values = stats.data["data"]
        assert values[EncryptedFsMetrics.ENCRYPTED_BYTES][()] == tier._block_size
        assert values[EncryptedFsMetrics.FS_WRITE_BYTES][()] > tier._block_size
        assert values[EncryptedFsMetrics.DECRYPT_FAILURES][()] == 1
        assert values[EncryptedFsMetrics.LOAD_FAILURES][()] == 1
        assert values[EncryptedFsMetrics.STORE_FAILURES][()] == 0
        assert values[EncryptedFsMetrics.INVALIDATED_BLOCKS][()] == 1
        assert len(values[EncryptedFsMetrics.JOB_QUEUE_SECONDS][("store",)]) == 1
        assert len(values[EncryptedFsMetrics.JOB_QUEUE_SECONDS][("load",)]) == 1
        assert len(values[EncryptedFsMetrics.JOB_TOTAL_SECONDS][("store",)]) == 1
        assert len(values[EncryptedFsMetrics.JOB_TOTAL_SECONDS][("load",)]) == 1
        assert len(values[EncryptedFsMetrics.ENCRYPT_SECONDS][()]) == 1
        assert len(values[EncryptedFsMetrics.DECRYPT_SECONDS][()]) == 1
        assert len(values[EncryptedFsMetrics.FS_WRITE_SECONDS][()]) == 1
        assert len(values[EncryptedFsMetrics.FS_READ_SECONDS][()]) == 1
        assert EncryptedFsMetrics.COPY_SECONDS not in values
    finally:
        tier.shutdown()


def test_failed_load_missing_file(fs_tier):
    """Test that loading a block whose file does not exist results in a failed job."""
    tier, _ = fs_tier
    job = make_job(1, [key(99)], [0], is_promotion=True)
    tier.submit_load(job)
    results = drain(tier)
    assert len(results) == 1
    assert not results[0].success


def test_multiple_jobs_tracked_independently(fs_tier):
    tier, _ = fs_tier
    job1 = make_job(1, [key(1)], [0])
    job2 = make_job(2, [key(2)], [1])
    tier.submit_store(job1)
    tier.submit_store(job2)
    results = drain(tier)
    job_ids = {r.job_id for r in results}
    assert job_ids == {1, 2}
    assert lookup_and_wait(tier, [key(1), key(2)]) == [
        LookupResult.HIT,
        LookupResult.HIT,
    ]


def test_multi_block_job_partial_failure(fs_tier):
    """A load job where one block file is missing yields a single failed JobResult."""
    tier, _ = fs_tier
    # Store two of three keys
    tier.submit_store(make_job(1, [key(10), key(11)], [0, 1]))
    assert all(r.success for r in drain(tier))

    # Load all three — key(99) was never stored
    tier.submit_load(
        make_job(2, [key(10), key(11), key(99)], [0, 1, 2], is_promotion=True)
    )
    results = drain(tier)

    assert len(results) == 1
    assert results[0].job_id == 2
    assert not results[0].success


def test_shutdown_discards_pending_tasks(fs_tier):
    """Shutdown clears both queues and stops all worker threads without draining."""
    tier, _ = fs_tier
    # Submit many tasks to ensure some remain pending
    for i in range(10):
        tier.submit_store(make_job(i, [key(i)], [i % 4]))

    # Shutdown immediately without draining
    tier.shutdown()

    # Verify queues are cleared and threads stopped
    assert len(tier._pool._load_q) == 0
    assert len(tier._pool._store_q) == 0
    assert all(not t.is_alive() for t in tier._pool._threads)


@pytest.mark.parametrize("batch_size", [0, 1, 2, 5])
@pytest.mark.parametrize("use_c_ext", [True, False])
def test_store_load_data_integrity(fs_tier, monkeypatch, use_c_ext, batch_size):
    """Data written by store must be exactly recovered by load, for batches
    of any size -- including the empty batch."""
    import vllm.v1.kv_offload.tiering.fs.io as io_mod

    if use_c_ext and not io_mod._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", use_c_ext)

    tier, tensor = fs_tier
    # Populate tensor with random data
    tensor[:] = _page_aligned_rand_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)

    keys = [key(i) for i in range(batch_size)]
    store_block_ids = list(range(batch_size))
    load_block_ids = list(range(_NUM_BLOCKS - batch_size, _NUM_BLOCKS))
    expected = tensor[:batch_size].clone()

    tier.submit_store(make_job(1, keys, store_block_ids))
    store_results = drain(tier)
    assert len(store_results) == 1
    assert store_results[0].success
    assert all(os.path.exists(tier.file_mapper.get_file_name(k)) for k in keys)

    # reset tensor to prove data is read from disk
    tensor[:] = 0.0

    # Load into a range disjoint by index from the store ids, to also
    # exercise loading a block into a different id than it was stored from.
    tier.submit_load(make_job(2, keys, load_block_ids, is_promotion=True))
    load_results = drain(tier)
    assert len(load_results) == 1
    assert load_results[0].success

    for i, bid in enumerate(load_block_ids):
        assert torch.allclose(tensor[bid], expected[i]), (
            f"Block {bid} data mismatch after store+load"
        )


def test_store_load_roundtrip_without_o_direct(tmp_path, monkeypatch):
    """Buffered fallback must round-trip data when O_DIRECT is unsupported.

    Simulates filesystems (e.g. overlayfs, some NFS) that reject O_DIRECT by
    forcing the capability probe to report it unavailable.
    """
    monkeypatch.setattr(
        "vllm.v1.kv_offload.tiering.fs.manager.probe_o_direct",
        lambda _dir: False,
    )
    tensor = _page_aligned_rand_tensor(4, _BLOCK_ELEMENTS)
    tier = FileSystemTierManager(
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=4,
        n_write_threads=4,
    )
    try:
        assert tier._use_o_direct is False

        keys = [key(0), key(1)]
        expected = tensor[:2].clone()
        tier.submit_store(make_job(1, keys, [0, 1]))
        assert all(r.success for r in drain(tier))

        tensor[:2] = 0.0
        tier.submit_load(make_job(2, keys, [2, 3], is_promotion=True))
        assert all(r.success for r in drain(tier))

        for i, bid in enumerate([2, 3]):
            assert torch.allclose(tensor[bid], expected[i])
    finally:
        tier.shutdown()


def test_wait_idle_blocks_until_tasks_complete():
    """wait_idle must not return while a task is still in flight."""
    pool = DualQueueThreadPool(n_read_threads=1, n_write_threads=1)
    gate = threading.Event()
    pool.enqueue_store(job_id=1, n_tasks=1, tasks=[lambda: gate.wait(timeout=5.0)])

    waiter = threading.Thread(target=pool.wait_idle)
    waiter.start()
    try:
        waiter.join(timeout=0.2)
        assert waiter.is_alive(), "wait_idle returned before task completed"
        gate.set()
        waiter.join(timeout=5.0)
        assert not waiter.is_alive(), "wait_idle did not unblock"
    finally:
        gate.set()
        pool.shutdown(wait=True)
        waiter.join(timeout=5.0)


def test_batch_lookup_c_extension(tmp_path):
    """Validates batch_lookup_C: empty, single, all-existing, all-missing,
    mixed ordering, and input type validation."""
    try:
        from vllm.fs_io_C import batch_lookup as batch_lookup_C
    except ImportError:
        pytest.skip("fs_io_C extension not built")

    # Setup
    all_exist = [str(tmp_path / f"e{i}.bin") for i in range(3)]
    for p in all_exist:
        open(p, "w").close()
    all_missing = [str(tmp_path / f"m{i}.bin") for i in range(3)]

    # Empty list
    assert batch_lookup_C([]) == []

    # Single existing / missing
    assert batch_lookup_C([all_exist[0]]) == [True]
    assert batch_lookup_C([all_missing[0]]) == [False]

    # All existing / all missing
    assert batch_lookup_C(all_exist) == [True, True, True]
    assert batch_lookup_C(all_missing) == [False, False, False]

    # Mixed — verifies index ordering is preserved
    paths = [val for pair in zip(all_exist, all_missing) for val in pair]
    assert batch_lookup_C(paths) == [True, False, True, False, True, False]

    # Input validation: non-list argument
    with pytest.raises(TypeError):
        batch_lookup_C(("/tmp/foo",))
    with pytest.raises(TypeError):
        batch_lookup_C(None)

    # Input validation: non-str elements in list
    with pytest.raises(TypeError):
        batch_lookup_C([None])
    with pytest.raises(TypeError):
        batch_lookup_C([b"/tmp/foo"])
    with pytest.raises(TypeError):
        batch_lookup_C([42])
    with pytest.raises(TypeError):
        batch_lookup_C([all_exist[0], None])  # valid first, invalid mid-list


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_batch_lookup_dispatch(fs_tier, monkeypatch, use_c_ext):
    import vllm.v1.kv_offload.tiering.fs.manager as mgr_mod

    if use_c_ext and not mgr_mod._HAS_BATCH_LOOKUP_C:
        pytest.skip("fs_io_C extension not built")

    monkeypatch.setattr(mgr_mod, "_HAS_BATCH_LOOKUP_C", use_c_ext)

    tier, _ = fs_tier
    tier.submit_store(make_job(1, [key(1)], [0]))
    assert all(r.success for r in drain(tier))

    results = lookup_and_wait(tier, [key(1), key(2)])
    assert results == [LookupResult.HIT, LookupResult.MISS]


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_out_of_bounds_block_id_smoke(fs_tier, monkeypatch, use_c_ext):
    """Smoke test: a block id beyond the primary tensor's block count must
    fail the job, for both the C extension and the Python fallback."""
    import vllm.v1.kv_offload.tiering.fs.io as io_mod

    if use_c_ext and not io_mod._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", use_c_ext)

    tier, tensor = fs_tier
    out_of_bounds_bid = tensor.shape[0]  # one past the last valid block

    tier.submit_store(make_job(1, [key(1)], [out_of_bounds_bid]))
    store_results = drain(tier)
    assert len(store_results) == 1
    assert not store_results[0].success

    tier.submit_load(make_job(2, [key(1)], [out_of_bounds_bid], is_promotion=True))
    load_results = drain(tier)
    assert len(load_results) == 1
    assert not load_results[0].success


# ---------------------------------------------------------------------------
# KV events
# ---------------------------------------------------------------------------


def test_successful_store_emits_stored_event(fs_tier_with_events):
    """A completed store job emits one stored event with the job's keys."""
    tier = fs_tier_with_events
    keys = [key(1), key(2)]
    tier.submit_store(make_job(1, keys, [0, 1]))
    assert all(r.success for r in drain(tier))

    events = list(tier.take_events())
    assert len(events) == 1
    assert events[0].keys == keys
    assert events[0].medium == Medium.STORAGE
    assert events[0].locality is Locality.LOCAL
    assert not events[0].removed
    # take_events drains the buffer.
    assert list(tier.take_events()) == []


@pytest.mark.parametrize(
    ("locality", "expected"),
    [(None, None), ("REMOTE", Locality.REMOTE)],
)
def test_store_event_uses_configured_locality(tmp_path, locality, expected):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    locality_config = {} if locality is None else {"locality": locality}
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(enable_kv_cache_events=True),
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        enable_kv_events=True,
        **locality_config,
    )
    try:
        tier.submit_store(make_job(1, [key(1)], [0]))
        assert all(r.success for r in drain(tier))

        events = list(tier.take_events())
        assert len(events) == 1
        assert events[0].locality is expected
    finally:
        tier.shutdown()


def test_load_job_emits_no_event(fs_tier_with_events):
    tier = fs_tier_with_events
    tier.submit_store(make_job(1, [key(1)], [0]))
    results = drain(tier)
    assert len(results) == 1
    assert results[0].success
    list(tier.take_events())

    tier.submit_load(make_job(2, [key(1)], [1], is_promotion=True))
    results = drain(tier)
    assert len(results) == 1
    assert results[0].success
    assert list(tier.take_events()) == []


def test_mixed_job_results_emit_event_only_for_successful_job(
    fs_tier_with_events, monkeypatch
):
    """With a failed and a successful store job in flight, exactly one event
    is emitted and its keys belong to the successful job."""
    import vllm.v1.kv_offload.tiering.fs.manager as mgr_mod

    tier = fs_tier_with_events
    failing_path = tier.file_mapper.get_file_name(key(1))
    original_batch_store_block = mgr_mod.batch_store_block

    def flaky_batch_store_block(paths, *args, **kwargs):
        if failing_path in paths:
            raise OSError("injected store failure")
        return original_batch_store_block(paths, *args, **kwargs)

    monkeypatch.setattr(mgr_mod, "batch_store_block", flaky_batch_store_block)

    tier.submit_store(make_job(1, [key(1)], [0]))
    tier.submit_store(make_job(2, [key(2)], [1]))
    results = drain(tier)
    assert len(results) == 2
    by_id = {r.job_id: r for r in results}
    assert not by_id[1].success
    assert by_id[2].success

    events = list(tier.take_events())
    assert len(events) == 1
    assert events[0].keys == [key(2)]


def test_partially_failed_store_emits_no_event(fs_tier_with_events, monkeypatch):
    """A store job with any failed block emits no event for the whole job."""
    import vllm.v1.kv_offload.tiering.fs.manager as mgr_mod

    tier = fs_tier_with_events
    failing_path = tier.file_mapper.get_file_name(key(2))
    original_batch_store_block = mgr_mod.batch_store_block

    def flaky_batch_store_block(paths, *args, **kwargs):
        if failing_path in paths:
            raise OSError("injected store failure")
        return original_batch_store_block(paths, *args, **kwargs)

    monkeypatch.setattr(mgr_mod, "batch_store_block", flaky_batch_store_block)

    tier.submit_store(make_job(1, [key(1), key(2)], [0, 1]))
    results = drain(tier)
    assert len(results) == 1
    assert not results[0].success
    assert list(tier.take_events()) == []
    assert tier._store_job_keys == {}


def test_events_disabled_by_default(fs_tier):
    tier, _ = fs_tier
    tier.submit_store(make_job(1, [key(1)], [0]))
    results = drain(tier)
    assert len(results) == 1
    assert results[0].success
    assert tier.events is None
    assert tier._store_job_keys == {}
    assert list(tier.take_events()) == []


def test_events_require_global_kv_events_flag(tmp_path):
    """Tier-level opt-in alone is not enough; the global flag gates events."""
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(enable_kv_cache_events=False),
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        enable_kv_events=True,
    )
    try:
        assert tier.events is None
        tier.submit_store(make_job(1, [key(1)], [0]))
        results = drain(tier)
        assert len(results) == 1
        assert results[0].success
        assert list(tier.take_events()) == []
        assert tier._store_job_keys == {}
    finally:
        tier.shutdown()


def test_cascade_store_emits_fs_event_through_tiering_manager(tmp_path):
    """A GPU->CPU->fs cascade surfaces the tier-owned FS stored event via the
    TieringOffloadingManager's aggregated take_events()."""
    from vllm.v1.kv_offload.tiering.manager import (
        CPUPrimaryTierOffloadingManager,
        TieringOffloadingManager,
    )

    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    view = memoryview(tensor.numpy())
    mock_region = MagicMock()
    mock_region.create_kv_memoryview.return_value = view
    primary = CPUPrimaryTierOffloadingManager(num_blocks=4, mmap_region=mock_region)
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(enable_kv_cache_events=True),
        primary_kv_view=primary.get_kv_memoryview(),
        tier_type="fs",
        root_dir=str(tmp_path),
        enable_kv_events=True,
    )
    manager = TieringOffloadingManager(primary_tier=primary, secondary_tiers=[tier])
    try:
        keys = [key(1), key(2)]
        manager.on_new_request(_CTX)
        assert manager.prepare_store(keys, _CTX) is not None
        manager.complete_store(keys, _CTX)  # cascades to the fs tier

        events: list[OffloadingEvent] = []
        ctx = ScheduleEndContext(new_req_ids=[], preempted_req_ids=())
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not events:
            manager.on_schedule_end(ctx)
            events.extend(manager.take_events())
            time.sleep(0.01)

        fs_events = [e for e in events if e.medium == Medium.STORAGE]
        assert len(fs_events) == 1
        assert set(fs_events[0].keys) == set(keys)
        assert not fs_events[0].removed
    finally:
        tier.shutdown()


def test_fs_tier_cross_tp_round_trip(tmp_path):
    """TP=2 replicated writer and TP=4 reader share namespace and bytes."""
    root = str(tmp_path)
    writer_tensor = _page_aligned_rand_tensor(4, _BLOCK_ELEMENTS)
    expected = writer_tensor[0].clone()
    writer = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(
            tp_size=2, world_size=2, rank=0, replicated_layout=True
        ),
        primary_kv_view=memoryview(writer_tensor.numpy()),
        tier_type="fs",
        root_dir=root,
        n_read_threads=2,
        n_write_threads=2,
    )
    try:
        writer.submit_store(make_job(1, [key(7)], [0]))
        assert all(r.success for r in drain(writer))
        writer_base = writer.file_mapper.base_path
        writer_path = writer.file_mapper.get_file_name(key(7))
    finally:
        writer.shutdown()

    reader_tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    reader = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(
            tp_size=4, world_size=4, rank=3, replicated_layout=True
        ),
        primary_kv_view=memoryview(reader_tensor.numpy()),
        tier_type="fs",
        root_dir=root,
        n_read_threads=2,
        n_write_threads=2,
    )
    try:
        assert reader.file_mapper.base_path == writer_base
        assert reader.file_mapper.get_file_name(key(7)) == writer_path
        assert lookup_and_wait(reader, [key(7)]) == [LookupResult.HIT]
        reader.submit_load(make_job(2, [key(7)], [1], is_promotion=True))
        assert all(r.success for r in drain(reader))
        assert torch.allclose(reader_tensor[1], expected)
    finally:
        reader.shutdown()
