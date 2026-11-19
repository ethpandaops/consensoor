"""Client for the leanvm-sidecar aggregation prover (protocol version 1)."""

import os
import socket
import struct
import threading
from typing import Optional, Sequence

MAGIC = b"leanvmsc"
VERSION = 1
VERIFIER_TAG = b"leanvm-v1"
OP_AGGREGATE = 1
OP_VERIFY = 2
MAX_CHILDREN = 16
DEFAULT_LOG_INV_RATE = 2
PROVE_TIMEOUT = 300.0
VERIFY_TIMEOUT = 30.0
RESPONSE_ROOM = 16 << 20

Claim = tuple[bytes, bytes]
Child = tuple[Sequence[Claim], bytes]

_socket_path: Optional[str] = os.environ.get("CONSENSOOR_LEANVM_SOCKET") or None


class SidecarError(RuntimeError):
    pass


class SidecarRejected(SidecarError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"sidecar status {status}: {message}")
        self.status = status


def configure(socket_path: Optional[str]) -> None:
    global _socket_path
    _socket_path = socket_path or None
    for connection in (_prove_connection, _verify_connection):
        connection.close()


def enabled() -> bool:
    return _socket_path is not None


def normalize_claims(claims: Sequence[Claim]) -> list[Claim]:
    return sorted({(bytes(pubkey), bytes(message)) for pubkey, message in claims})


def encode_claims(claims: Sequence[Claim]) -> bytes:
    out = bytearray(struct.pack("<Q", len(claims)))
    for pubkey, message in claims:
        out += pubkey
        out += message
    out += struct.pack("<Q", 0)
    out.append(0)
    return bytes(out)


def decode_claims(data: bytes) -> list[Claim]:
    (count,) = struct.unpack_from("<Q", data, 0)
    if len(data) != 8 + count * 64 + 9:
        raise SidecarError("claims do not decode")
    return [(data[8 + i * 64:40 + i * 64], data[40 + i * 64:72 + i * 64]) for i in range(count)]


def _field(data: bytes) -> bytes:
    return struct.pack("<Q", len(data)) + data


class _Connection:
    def __init__(self, timeout: float) -> None:
        self._timeout = timeout
        self._lock = threading.Lock()
        self._sock: Optional[socket.socket] = None
        self._max_request = 0

    def close(self) -> None:
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def _read(self, size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            chunk = self._sock.recv(min(size - len(chunks), 1 << 20))
            if not chunk:
                raise SidecarError("sidecar closed the connection")
            chunks += chunk
        return bytes(chunks)

    def _read_frame(self, limit: int) -> bytes:
        (length,) = struct.unpack("<Q", self._read(8))
        if length > limit:
            raise SidecarError(f"frame of {length} bytes exceeds {limit}")
        return self._read(length)

    def _open(self) -> None:
        if _socket_path is None:
            raise SidecarError("no leanvm sidecar socket configured")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self._timeout)
        sock.connect(_socket_path)
        self._sock = sock
        hello = self._read_frame(RESPONSE_ROOM)
        if hello[:8] != MAGIC:
            raise SidecarError("not a leanVM sidecar")
        (version,) = struct.unpack_from("<I", hello, 8)
        if version != VERSION:
            raise SidecarError(f"sidecar protocol version {version}")
        (self._max_request,) = struct.unpack_from("<Q", hello, 12)
        (tag_len,) = struct.unpack_from("<Q", hello, 20)
        tag = hello[28:28 + tag_len]
        if tag != VERIFIER_TAG:
            raise SidecarError(f"sidecar verifier {tag!r}")

    def call(self, body: bytes) -> bytes:
        with self._lock:
            reused = self._sock is not None
            while True:
                try:
                    if self._sock is None:
                        self._open()
                    if len(body) > self._max_request:
                        raise SidecarRejected(5, f"request of {len(body)} bytes is too large")
                    self._sock.sendall(_field(body))
                    return self._read_frame(len(body) + 8 + RESPONSE_ROOM)
                except SidecarRejected:
                    raise
                except (OSError, SidecarError) as error:
                    self.close()
                    if reused and not isinstance(error, socket.timeout):
                        reused = False
                        continue
                    if isinstance(error, SidecarError):
                        raise
                    raise SidecarError(f"sidecar connection: {error}") from error


_prove_connection = _Connection(PROVE_TIMEOUT)
_verify_connection = _Connection(VERIFY_TIMEOUT)


def _rejection(response: bytes) -> SidecarRejected:
    (length,) = struct.unpack_from("<Q", response, 1)
    return SidecarRejected(response[0], response[9:9 + length].decode("utf-8", "replace"))


def aggregate(
    children: Sequence[Child],
    signatures: Sequence[tuple[bytes, bytes, bytes]],
    log_inv_rate: int = DEFAULT_LOG_INV_RATE,
) -> tuple[list[Claim], bytes]:
    body = bytearray([OP_AGGREGATE, log_inv_rate])
    body += struct.pack("<Q", len(children))
    for claims, proof in children:
        body += _field(encode_claims(claims))
        body += _field(bytes(proof))
    body += struct.pack("<Q", len(signatures))
    for pubkey, message, signature in signatures:
        body += pubkey
        body += message
        body += signature
    body += struct.pack("<Q", 0)
    body += struct.pack("<Q", 0)
    body.append(0)
    response = _prove_connection.call(bytes(body))
    if response[0] != 0:
        raise _rejection(response)
    (claims_len,) = struct.unpack_from("<Q", response, 1)
    claims = decode_claims(response[9:9 + claims_len])
    offset = 9 + claims_len
    (proof_len,) = struct.unpack_from("<Q", response, offset)
    proof = response[offset + 8:offset + 8 + proof_len]
    if len(proof) != proof_len or offset + 8 + proof_len != len(response):
        raise SidecarError("aggregate response does not decode")
    return claims, proof


def verify(claims: Sequence[Claim], proof: bytes) -> bool:
    body = bytes([OP_VERIFY]) + _field(encode_claims(claims)) + _field(bytes(proof))
    response = _verify_connection.call(body)
    if response[0] != 0:
        raise _rejection(response)
    if len(response) != 2 or response[1] > 1:
        raise SidecarError("verify response does not decode")
    return response[1] == 1


def prove(claims: Sequence[Claim], signatures: Sequence[tuple[bytes, bytes, bytes]]) -> bytes:
    expected = normalize_claims(claims)
    if not expected:
        return b""
    proven, proof = aggregate([], signatures)
    if proven != expected:
        raise SidecarError(f"sidecar proved {len(proven)} claims, expected {len(expected)}")
    return proof


def merge(children: Sequence[Child]) -> bytes:
    level: list[tuple[list[Claim], bytes]] = []
    for claims, proof in children:
        claims, proof = normalize_claims(claims), bytes(proof)
        if not claims and not proof:
            continue
        if not claims or not proof:
            raise SidecarError("child has claims without a proof or a proof without claims")
        level.append((claims, proof))
    while len(level) > 1:
        merged: list[tuple[list[Claim], bytes]] = []
        for start in range(0, len(level), MAX_CHILDREN):
            chunk = level[start:start + MAX_CHILDREN]
            if len(chunk) == 1:
                merged.append(chunk[0])
                continue
            expected = normalize_claims([claim for claims, _ in chunk for claim in claims])
            proven, proof = aggregate(chunk, [])
            if proven != expected:
                raise SidecarError(f"sidecar merged {len(proven)} claims, expected {len(expected)}")
            merged.append((expected, proof))
        level = merged
    return level[0][1] if level else b""


def check(claims: Sequence[Claim], proof: bytes) -> bool:
    claims, proof = normalize_claims(claims), bytes(proof)
    if not claims or not proof:
        return not claims and not proof
    return verify(claims, proof)
