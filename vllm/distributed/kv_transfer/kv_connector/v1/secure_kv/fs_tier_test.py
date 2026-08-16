# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test for the encrypted fs tier byte tasks (no engine, no GPU).

Run: SKV_MASTER_KEY=<hex> python fs_tier_test.py
"""

import math
import os
import secrets
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("SKV_MASTER_KEY", secrets.token_hex(16))

from vllm.distributed.kv_transfer.kv_connector.v1.secure_kv.crypto import (  # noqa: E402
    CryptoEngine,
    DecryptError,
)
from vllm.distributed.kv_transfer.kv_connector.v1.secure_kv.fs_capacity import (  # noqa: E402
    CacheCapacityError,
    EncryptedFsCapacityManager,
)
from vllm.distributed.kv_transfer.kv_connector.v1.secure_kv.fs_tier import (  # noqa: E402
    _decrypt_load,
    _encrypt_store,
)
from vllm.distributed.kv_transfer.kv_connector.v1.secure_kv.keys import (  # noqa: E402
    KeyManager,
)
from vllm.v1.kv_offload.base import make_offload_key  # noqa: E402

PASS = True


def check(name, cond, note=""):
    global PASS
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {note}" if note else ""))
    PASS = PASS and cond


def entropy(b):
    c = [0] * 256
    for x in b:
        c[x] += 1
    n = len(b)
    return -sum(v / n * math.log2(v / n) for v in c if v)


def check_capacity_eviction():
    blob_size = 4096
    blob = b"x" * blob_size
    with tempfile.TemporaryDirectory() as d:
        capacity = EncryptedFsCapacityManager(
            d, max_bytes=4 * blob_size, low_watermark=0.75
        )
        paths = [os.path.join(d, f"block{i}.bin") for i in range(7)]
        old_time = time.time_ns() - 20_000_000_000
        for i, path in enumerate(paths[:4]):
            capacity.store_blob(path, blob)
            os.utime(path, ns=(old_time + i, old_time + i))

        capacity.store_blob(paths[4], blob)
        snapshot = capacity.snapshot(reset_counters=False)
        check(
            "capacity returns to low watermark", snapshot.current_bytes == 3 * blob_size
        )
        check(
            "oldest blocks evicted",
            not os.path.exists(paths[0]) and not os.path.exists(paths[1]),
        )
        check("newer blocks retained", all(os.path.exists(path) for path in paths[2:5]))
        check(
            "eviction counters",
            snapshot.evicted_files == 2 and snapshot.evicted_bytes == 2 * blob_size,
        )

        check("capacity load returns bytes", capacity.load_blob(paths[2]) == blob)
        time.sleep(0.01)
        capacity.touch(paths[2])
        capacity.store_blob(paths[5], blob)
        capacity.store_blob(paths[6], blob)
        check("recently touched block survives LRU", os.path.exists(paths[2]))
        check(
            "hard max is never exceeded",
            capacity.snapshot().current_bytes <= 4 * blob_size,
        )


def check_capacity_restart_and_concurrency():
    blob_size = 1024
    blob = b"z" * blob_size
    with tempfile.TemporaryDirectory() as d:
        old_time = time.time_ns() - 20_000_000_000
        for i in range(5):
            path = os.path.join(d, f"restart{i}.bin")
            with open(path, "wb") as cache_file:
                cache_file.write(blob)
            os.utime(path, ns=(old_time + i, old_time + i))

        capacity = EncryptedFsCapacityManager(
            d, max_bytes=4 * blob_size, low_watermark=0.75
        )
        snapshot = capacity.snapshot(reset_counters=False)
        check(
            "startup scan enforces capacity",
            snapshot.current_bytes == 3 * blob_size and snapshot.current_files == 3,
        )

        paths = [os.path.join(d, f"concurrent{i}.bin") for i in range(16)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda path: capacity.store_blob(path, blob), paths))
        snapshot = capacity.snapshot()
        check(
            "concurrent writers respect hard max",
            snapshot.current_bytes <= 4 * blob_size,
        )
        actual_bytes = sum(
            os.path.getsize(os.path.join(root, name))
            for root, _, names in os.walk(d)
            for name in names
            if name.endswith(".bin")
        )
        check("capacity state matches files", snapshot.current_bytes == actual_bytes)

        peer = EncryptedFsCapacityManager(
            d, max_bytes=4 * blob_size, low_watermark=0.75
        )
        peer_path = os.path.join(d, "peer.bin")
        capacity.store_blob(peer_path, blob)
        check(
            "capacity gauge observes peer writers",
            peer.snapshot().current_bytes == capacity.snapshot().current_bytes,
        )

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "only.bin")
        with open(path, "wb") as cache_file:
            cache_file.write(blob)
        impossible_reserve = shutil.disk_usage(d).free + 1024 * 1024
        try:
            EncryptedFsCapacityManager(
                d,
                max_bytes=4 * blob_size,
                min_free_bytes=impossible_reserve,
            )
            check("unreachable free-space reserve rejects startup", False)
        except CacheCapacityError:
            check("unreachable free-space reserve rejects startup", True)
        check(
            "failed reserve still accounts completed eviction", not os.path.exists(path)
        )


def main():
    crypto = CryptoEngine(KeyManager())
    block_size = 1 << 20  # 1 MiB blocks
    n = 3
    src = bytearray(os.urandom(block_size * n))
    view = memoryview(src)
    keys = [make_offload_key(os.urandom(32), g) for g in range(n)]
    offsets = [i * block_size for i in range(n)]

    with tempfile.TemporaryDirectory() as d:
        paths = [os.path.join(d, f"blk{i}.bin") for i in range(n)]

        print("== store: seal to disk ==")
        _encrypt_store(crypto, paths, keys, view, offsets, block_size)
        check("files written", all(os.path.exists(p) for p in paths))
        with open(paths[0], "rb") as cache_file:
            blob = cache_file.read()
        check("on-disk magic SKV1", blob[:4] == b"SKV1")
        check(
            "on-disk entropy > 7.99",
            entropy(blob[200:]) > 7.99,
            f"{entropy(blob[200:]):.4f}",
        )
        check("no plaintext window on disk", bytes(src[:64]) not in blob)

        print("== load: verify + unseal into fresh buffer ==")
        dst = bytearray(block_size * n)
        _decrypt_load(crypto, paths, keys, memoryview(dst), offsets, block_size)
        check("byte-exact roundtrip", dst == src)

        print("== idempotent store (existing file skipped) ==")
        before = os.path.getmtime(paths[0])
        _encrypt_store(crypto, paths, keys, view, offsets, block_size)
        check("existing blobs not rewritten", os.path.getmtime(paths[0]) == before)

        print("== failure semantics ==")
        # relocation: right file, wrong key identity -> AAD mismatch
        try:
            _decrypt_load(
                crypto, [paths[0]], [keys[1]], memoryview(dst), [0], block_size
            )
            check("relocated blob rejected", False, "unexpectedly loaded")
        except DecryptError:
            check("relocated blob rejected", True)
        check("relocated blob removed", not os.path.exists(paths[0]))
        # tamper: flip one ciphertext byte
        with open(paths[2], "rb") as cache_file:
            b2 = bytearray(cache_file.read())
        b2[len(b2) // 2] ^= 1
        with open(paths[2], "wb") as cache_file:
            cache_file.write(bytes(b2))
        try:
            _decrypt_load(
                crypto, [paths[2]], [keys[2]], memoryview(dst), [0], block_size
            )
            check("tampered blob rejected", False, "unexpectedly loaded")
        except DecryptError:
            check("tampered blob rejected", True)
        check("tampered blob removed", not os.path.exists(paths[2]))
        # wrong master key -> nothing recoverable
        crypto2 = CryptoEngine(KeyManager(secrets.token_hex(16)))
        try:
            _decrypt_load(
                crypto2, [paths[1]], [keys[1]], memoryview(dst), [0], block_size
            )
            check("wrong master key rejected", False)
        except DecryptError:
            check("wrong master key rejected", True)
        check("wrong-key blob removed", not os.path.exists(paths[1]))

    print("== capacity: LRU, restart, concurrency, free reserve ==")
    check_capacity_eviction()
    check_capacity_restart_and_concurrency()

    print()
    print("ENCRYPTED FS TIER UNIT:", "PASS" if PASS else "FAIL")
    raise SystemExit(0 if PASS else 1)


if __name__ == "__main__":
    main()
