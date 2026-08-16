# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal remote SecureKV store for the RemoteBackend (M4 test aid).

An in-memory, length-prefixed TCP KV store. It only ever holds ciphertext
SealedBlobs — it has no keys and cannot read the KV. `contains` on a bare
index key (no '/') answers a directory-style membership probe.

Run:  python remote_server.py --port 14580
"""

import argparse
import socketserver
import struct
import threading

_STORE: dict[str, bytes] = {}
_INDEX: dict[str, set[str]] = {}  # index_key -> set of full "index/leaf" keys
_LOCK = threading.Lock()


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("client closed early")
        buf += chunk
    return bytes(buf)


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        op = _recv_exact(self.request, 1)
        (klen,) = struct.unpack("<I", _recv_exact(self.request, 4))
        key = _recv_exact(self.request, klen).decode()
        (vlen,) = struct.unpack("<I", _recv_exact(self.request, 4))
        val = _recv_exact(self.request, vlen)

        status, data = 0, b""
        with _LOCK:
            if op == b"P":
                _STORE[key] = val
                idx = key.split("/", 1)[0]
                _INDEX.setdefault(idx, set()).add(key)
                status = 1
            elif op == b"G":
                if key in _STORE:
                    status, data = 1, _STORE[key]
            elif op == b"C":
                hit = key in _STORE or bool(_INDEX.get(key))
                status = 1 if hit else 0
            elif op == b"D":
                _STORE.pop(key, None)
                idx = key.split("/", 1)[0]
                if idx in _INDEX:
                    _INDEX[idx].discard(key)
                status = 1
        self.request.sendall(bytes([status]) + struct.pack("<I", len(data)) + data)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=14580)
    args = ap.parse_args()
    with Server((args.host, args.port), Handler) as srv:
        print(f"SecureKV remote store on {args.host}:{args.port} (ciphertext only)",
              flush=True)
        srv.serve_forever()


if __name__ == "__main__":
    main()
