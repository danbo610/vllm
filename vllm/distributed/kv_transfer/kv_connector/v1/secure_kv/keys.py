# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Key hierarchy for SecureKV (design doc §6.2 / §7.3 / §7.4).

    master (env SKV_MASTER_KEY, 128/256-bit hex; KMS-injected in prod)
      └─ tenant_key  = HKDF(master, info=f"skv/v{ver}/tenant/{tenant_id}")
           └─ (session granularity reserved for M3)

Nonce discipline (GCM invariant — a single reuse under one key is fatal):
96-bit nonce = 64-bit per-process random salt || 32-bit monotonic counter,
guarded by a lock. Counter state is memory-only; a restart re-randomizes
the salt, so no persistence is needed.
"""

from __future__ import annotations

import hashlib
import hmac as hmac_mod
import os
import secrets
import struct
import threading

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

CURRENT_KEY_VERSION = 1


class KeyManager:
    def __init__(self, master_hex: str | None = None):
        master_hex = master_hex or os.environ.get("SKV_MASTER_KEY", "")
        if not master_hex:
            raise ValueError(
                "SecureKV requires a master key: set SKV_MASTER_KEY "
                "(128/256-bit hex) or pass key_source in "
                "kv_connector_extra_config.")
        self._master = bytes.fromhex(master_hex)
        self._tenant_keys: dict[tuple[str, int], bytes] = {}
        self._nonce_state: dict[str, tuple[bytes, int]] = {}
        self._lock = threading.Lock()

    def tenant_key(self, tenant_id: str,
                   version: int = CURRENT_KEY_VERSION) -> tuple[bytes, int]:
        k = (tenant_id, version)
        if k not in self._tenant_keys:
            hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                        info=f"skv/v{version}/tenant/{tenant_id}".encode())
            self._tenant_keys[k] = hkdf.derive(self._master)
        return self._tenant_keys[k], version

    def tenant_tag(self, tenant_id: str) -> bytes:
        """8-byte keyed tag identifying the tenant without leaking its name."""
        key, _ = self.tenant_key(tenant_id)
        return hmac_mod.new(key, b"skv-tenant-tag", hashlib.sha256).digest()[:8]

    def index_hmac(self, tenant_id: str, data: bytes) -> str:
        """Keyed index key (design §7.3): prevents prefix-probing and gives
        per-tenant namespace isolation for free."""
        key, _ = self.tenant_key(tenant_id)
        return hmac_mod.new(key, data, hashlib.sha256).hexdigest()

    def next_nonce(self, tenant_id: str) -> bytes:
        with self._lock:
            salt, counter = self._nonce_state.get(
                tenant_id, (secrets.token_bytes(8), 0))
            if counter >= 0xFFFFFFFF:
                # Rotate salt long before the 32-bit counter could wrap.
                salt, counter = secrets.token_bytes(8), 0
            self._nonce_state[tenant_id] = (salt, counter + 1)
        return salt + struct.pack("<I", counter)

    def wipe(self) -> None:
        """Best-effort key material cleanup on shutdown."""
        self._tenant_keys.clear()
        self._nonce_state.clear()
        self._master = b"\x00" * len(self._master)
