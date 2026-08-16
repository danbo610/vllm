# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test for the encrypted fs tier byte tasks (no engine, no GPU).

Run: SKV_MASTER_KEY=<hex> python fs_tier_test.py
"""

import math
import os
import secrets
import tempfile

os.environ.setdefault("SKV_MASTER_KEY", secrets.token_hex(16))

from vllm.distributed.kv_transfer.kv_connector.v1.secure_kv.crypto import (  # noqa: E402
    CryptoEngine, DecryptError)
from vllm.distributed.kv_transfer.kv_connector.v1.secure_kv.fs_tier import (  # noqa: E402
    _aad, _decrypt_load, _encrypt_store, _TIER_KEY_NS)
from vllm.distributed.kv_transfer.kv_connector.v1.secure_kv.keys import (  # noqa: E402
    KeyManager)
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
        blob = open(paths[0], "rb").read()
        check("on-disk magic SKV1", blob[:4] == b"SKV1")
        check("on-disk entropy > 7.99", entropy(blob[200:]) > 7.99,
              f"{entropy(blob[200:]):.4f}")
        check("no plaintext window on disk",
              bytes(src[:64]) not in blob)

        print("== load: verify + unseal into fresh buffer ==")
        dst = bytearray(block_size * n)
        _decrypt_load(crypto, paths, keys, memoryview(dst), offsets, block_size)
        check("byte-exact roundtrip", dst == src)

        print("== failure semantics ==")
        # relocation: right file, wrong key identity -> AAD mismatch
        try:
            _decrypt_load(crypto, [paths[0]], [keys[1]], memoryview(dst),
                          [0], block_size)
            check("relocated blob rejected", False, "unexpectedly loaded")
        except DecryptError:
            check("relocated blob rejected", True)
        # tamper: flip one ciphertext byte
        b2 = bytearray(open(paths[2], "rb").read())
        b2[len(b2) // 2] ^= 1
        open(paths[2], "wb").write(bytes(b2))
        try:
            _decrypt_load(crypto, [paths[2]], [keys[2]], memoryview(dst),
                          [0], block_size)
            check("tampered blob rejected", False, "unexpectedly loaded")
        except DecryptError:
            check("tampered blob rejected", True)
        # wrong master key -> nothing recoverable
        crypto2 = CryptoEngine(KeyManager(secrets.token_hex(16)))
        try:
            _decrypt_load(crypto2, [paths[0]], [keys[0]], memoryview(dst),
                          [0], block_size)
            check("wrong master key rejected", False)
        except DecryptError:
            check("wrong master key rejected", True)

        print("== idempotent store (existing file skipped) ==")
        before = os.path.getmtime(paths[0])
        _encrypt_store(crypto, paths, keys, view, offsets, block_size)
        check("existing blobs not rewritten",
              os.path.getmtime(paths[0]) == before)

    print()
    print("ENCRYPTED FS TIER UNIT:", "PASS" if PASS else "FAIL")
    raise SystemExit(0 if PASS else 1)


if __name__ == "__main__":
    main()
