# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pluggable storage backends for SecureKV (design doc §6.3).

Backends only ever see opaque bytes under opaque keys — all cryptography
happens in the connector layer, so the backend does not need to be trusted.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod


class StorageBackend(ABC):
    @abstractmethod
    def put(self, key: str, blob: bytes) -> None: ...

    @abstractmethod
    def get(self, key: str) -> bytes | None: ...

    @abstractmethod
    def contains(self, key: str) -> bool: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...


class LocalDiskBackend(StorageBackend):
    """PoC disk backend with atomic writes.

    Keys are HMAC hex strings from KeyManager.index_hmac, optionally
    hierarchical ("index/leaf"): "abc…"        -> <root>/ab/abc…/
                                 "abc…/deadbe" -> <root>/ab/abc…/deadbe.skv
    `contains` on a bare index key probes the directory (any object under
    it). Capacity eviction is left to ops for the PoC.
    """

    def __init__(self, root_dir: str):
        self.root = root_dir
        os.makedirs(self.root, exist_ok=True)

    def _dir(self, index_key: str) -> str:
        assert all(c in "0123456789abcdef" for c in index_key), "unsafe key"
        return os.path.join(self.root, index_key[:2], index_key)

    def _path(self, key: str) -> str:
        index_key, _, leaf = key.partition("/")
        if not leaf:
            raise ValueError("object keys must be 'index/leaf'")
        assert all(c in "0123456789abcdef" for c in leaf), "unsafe key"
        return os.path.join(self._dir(index_key), f"{leaf}.skv")

    def put(self, key: str, blob: bytes) -> None:
        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(blob)
        os.replace(tmp, path)  # atomic: readers never see partial blobs

    def get(self, key: str) -> bytes | None:
        try:
            with open(self._path(key), "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None

    def contains(self, key: str) -> bool:
        if "/" in key:
            return os.path.exists(self._path(key))
        d = self._dir(key)
        return os.path.isdir(d) and bool(os.listdir(d))

    def delete(self, key: str) -> None:
        try:
            os.remove(self._path(key))
        except FileNotFoundError:
            pass


def make_backend(kind: str, config: dict) -> StorageBackend:
    if kind == "local_disk":
        return LocalDiskBackend(config.get("root_dir", "/tmp/skv_cache"))
    raise ValueError(f"unknown SecureKV backend: {kind!r} "
                     "(supported: local_disk; remote/lmcache arrive with M4)")
