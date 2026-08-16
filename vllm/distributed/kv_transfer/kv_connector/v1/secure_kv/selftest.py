# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SecureKV M1 self-test: crypto/key/backend units, no GPU, no vLLM engine.

Run:  SKV_MASTER_KEY=<hex> python selftest.py
"""

import math
import os
import secrets
import sys
import tempfile

import torch

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from backend import LocalDiskBackend
    from crypto import CryptoEngine, DecryptError, pack_tensor, unpack_tensor
    from keys import KeyManager
else:
    from .backend import LocalDiskBackend
    from .crypto import CryptoEngine, DecryptError, pack_tensor, unpack_tensor
    from .keys import KeyManager

PASS = True


def check(name: str, ok: bool, note: str = ""):
    global PASS
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {note}" if note else ""))
    PASS = PASS and ok


def entropy_bits_per_byte(data: bytes) -> float:
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    n = len(data)
    return -sum(c / n * math.log2(c / n) for c in counts if c)


def main():
    os.environ.setdefault("SKV_MASTER_KEY", secrets.token_hex(16))
    km = KeyManager()
    eng = CryptoEngine(km)

    print("== 1. tensor pack/unpack (byte-exact, incl. bfloat16) ==")
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        t = torch.randn(16, 2, 64, dtype=dtype)
        r = unpack_tensor(pack_tensor(t))
        check(f"roundtrip {dtype}", bool(torch.equal(t, r)))

    print("== 2. seal/unseal roundtrip + ciphertext quality ==")
    payload = pack_tensor(torch.randn(16, 2, 128, dtype=torch.bfloat16))
    aad = b"skv1|idx|layer|fp|tag"
    blob = eng.seal(payload, aad, "tenantA")
    check("roundtrip", eng.unseal(blob, "tenantA", expected_aad=aad) == payload)
    ct = blob[36 + len(aad):]
    check("ciphertext entropy > 7.9 bits/byte",
          entropy_bits_per_byte(ct) > 7.9,
          f"{entropy_bits_per_byte(ct):.3f}")
    check("magic header", blob[:4] == b"SKV1")

    print("== 3. tamper / relocation / cross-tenant must all fail ==")
    bad = bytearray(blob)
    bad[len(bad) // 2] ^= 0x01
    for name, fn in [
        ("bit-flip detected", lambda: eng.unseal(bytes(bad), "tenantA",
                                                 expected_aad=aad)),
        ("AAD relocation detected", lambda: eng.unseal(blob, "tenantA",
                                                       expected_aad=b"other|aad")),
        ("cross-tenant detected", lambda: eng.unseal(blob, "tenantB",
                                                     expected_aad=aad)),
    ]:
        try:
            fn()
            check(name, False, "unexpectedly succeeded")
        except DecryptError:
            check(name, True)

    print("== 4. key derivation: deterministic, tenant-isolated ==")
    km2 = KeyManager(os.environ["SKV_MASTER_KEY"])
    check("HKDF deterministic across instances",
          km.tenant_key("t1") == km2.tenant_key("t1"))
    check("tenants isolated",
          km.tenant_key("t1")[0] != km.tenant_key("t2")[0])
    check("index_hmac deterministic + tenant-scoped",
          km.index_hmac("t1", b"x") == km2.index_hmac("t1", b"x")
          and km.index_hmac("t1", b"x") != km.index_hmac("t2", b"x"))

    print("== 5. nonce uniqueness (10k draws) ==")
    nonces = {km.next_nonce("t1") for _ in range(10000)}
    check("10k unique nonces", len(nonces) == 10000)

    print("== 6. disk backend: put/get/contains/delete + dir probe ==")
    with tempfile.TemporaryDirectory() as d:
        be = LocalDiskBackend(d)
        idx = km.index_hmac("t1", b"prompt-tokens")
        key = f"{idx}/deadbeefdeadbeef"
        check("miss before put", not be.contains(idx))
        be.put(key, blob)
        check("object hit", be.contains(key))
        check("index dir probe hit", be.contains(idx))
        check("get roundtrip", be.get(key) == blob)
        check("on-disk bytes are sealed (magic)",
              be.get(key)[:4] == b"SKV1")
        be.delete(key)
        check("miss after delete", be.get(key) is None)

    print()
    print("SECUREKV M1 SELFTEST:", "PASS" if PASS else "FAIL")
    raise SystemExit(0 if PASS else 1)


if __name__ == "__main__":
    main()
