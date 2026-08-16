# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AES-256-GCM sealing for KV blobs (SecureKV M1).

SealedBlob wire format (little-endian), self-describing:

    magic "SKV1" (4B) | flags u16 | key_version u16 | tenant_tag 8B |
    nonce 12B | aad_len u32 | ct_len u32 | AAD (plaintext, authenticated) |
    ciphertext||gcm_tag

The AAD binds block identity (layer, index key, model fingerprint, tenant)
into the authentication: any ciphertext relocation / cross-tenant replay /
cross-model reuse fails the tag check. See design doc §7.1/§7.2.
"""

from __future__ import annotations

import json
import struct
from concurrent.futures import ThreadPoolExecutor

import torch
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"SKV1"
FLAG_COMPRESSED = 0x1
_HEADER = struct.Struct("<4sHH8s12sII")


class DecryptError(Exception):
    """Tag verification failed / malformed blob. Callers must treat the
    affected block as a cache miss (recompute), never crash the engine."""


def pack_tensor(t: torch.Tensor) -> bytes:
    """Serialize tensor to bytes: json head (dtype/shape) + raw bytes.
    Byte-exact for any dtype incl. bfloat16 (no numpy round-trip)."""
    t = t.detach().contiguous().cpu()
    head = json.dumps(
        {"dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape)}
    ).encode()
    raw = t.view(torch.uint8).numpy().tobytes() if t.numel() else b""
    return struct.pack("<I", len(head)) + head + raw


def unpack_tensor(data: bytes) -> torch.Tensor:
    (head_len,) = struct.unpack_from("<I", data, 0)
    head = json.loads(data[4 : 4 + head_len])
    raw = data[4 + head_len :]
    dtype = getattr(torch, head["dtype"])
    t = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    return t.view(dtype).reshape(head["shape"])


class CryptoEngine:
    """AES-256-GCM seal/unseal with a thread pool for batches."""

    def __init__(self, key_manager, workers: int = 4):
        self.km = key_manager
        self.pool = ThreadPoolExecutor(max_workers=workers,
                                       thread_name_prefix="skv-crypto")

    def seal(self, plaintext: bytes, aad: bytes, tenant_id: str) -> bytes:
        key, key_version = self.km.tenant_key(tenant_id)
        tenant_tag = self.km.tenant_tag(tenant_id)
        nonce = self.km.next_nonce(tenant_id)
        ct = AESGCM(key).encrypt(nonce, plaintext, aad)  # ciphertext||tag
        header = _HEADER.pack(MAGIC, 0, key_version, tenant_tag, nonce,
                              len(aad), len(ct))
        return header + aad + ct

    def unseal(self, blob: bytes, tenant_id: str,
               expected_aad: bytes | None = None) -> bytes:
        try:
            magic, flags, key_version, tenant_tag, nonce, aad_len, ct_len = (
                _HEADER.unpack_from(blob, 0))
        except struct.error as e:
            raise DecryptError(f"malformed header: {e}") from e
        if magic != MAGIC:
            raise DecryptError("bad magic")
        off = _HEADER.size
        aad = blob[off : off + aad_len]
        ct = blob[off + aad_len : off + aad_len + ct_len]
        if expected_aad is not None and aad != expected_aad:
            raise DecryptError("AAD mismatch (relocated/replayed blob)")
        if tenant_tag != self.km.tenant_tag(tenant_id):
            raise DecryptError("tenant tag mismatch")
        key, _ = self.km.tenant_key(tenant_id, version=key_version)
        try:
            return AESGCM(key).decrypt(nonce, ct, aad)
        except InvalidTag as e:
            raise DecryptError("GCM tag verification failed") from e

    # -- batch helpers (order-preserving) ------------------------------
    def seal_many(self, items):
        """items: list of (plaintext, aad, tenant_id) -> list[bytes]"""
        return list(self.pool.map(lambda x: self.seal(*x), items))

    def unseal_many(self, items):
        """items: list of (blob, tenant_id, expected_aad)
        -> list[bytes | DecryptError] (positional, never raises)."""
        def one(x):
            try:
                return self.unseal(*x)
            except DecryptError as e:
                return e
        return list(self.pool.map(one, items))
