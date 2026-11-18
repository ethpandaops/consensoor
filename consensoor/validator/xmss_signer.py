"""leanXMSS signing for validator keys, with a durable record of the leaves used."""

import asyncio
import logging
import os
import threading
from pathlib import Path
from typing import Optional

from ..crypto import xmss
from .types import ValidatorKey

logger = logging.getLogger(__name__)


class LeafReuseError(Exception):
    pass


class LeafRecord:
    """Signing root signed at each (public key, leaf), persisted before every signature."""

    def __init__(self, path: Optional[Path] = None):
        self._path = path
        self._lock = threading.Lock()
        self._roots: dict[tuple[bytes, int], bytes] = {}
        self._file = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                for line in path.read_text().splitlines():
                    parts = line.split()
                    if len(parts) != 3:
                        continue
                    self._roots[(bytes.fromhex(parts[0]), int(parts[1]))] = bytes.fromhex(parts[2])
            self._file = open(path, "a")

    def record(self, pubkey: bytes, leaf_index: int, signing_root: bytes) -> None:
        key = (bytes(pubkey), int(leaf_index))
        signing_root = bytes(signing_root)
        with self._lock:
            known = self._roots.get(key)
            if known is not None:
                if known != signing_root:
                    raise LeafReuseError(
                        f"leaf {leaf_index} of key {key[0].hex()[:16]} already signed a different message"
                    )
                return
            if self._file is not None:
                self._file.write(f"{key[0].hex()} {key[1]} {signing_root.hex()}\n")
                self._file.flush()
                os.fsync(self._file.fileno())
            self._roots[key] = signing_root


_record = LeafRecord()


def configure(data_dir: str) -> None:
    global _record
    _record = LeafRecord(Path(data_dir) / "xmss_leaves.txt")


def derive(key: ValidatorKey) -> None:
    key.xmss_seed = xmss.seed_from_privkey(key.privkey)
    key.vrf_seed = xmss.vrf_seed_from_privkey(key.privkey)
    key.pubkey = xmss.public_key(key.xmss_seed)


def prepare(key: ValidatorKey, leaf_index: int) -> None:
    xmss.prepare(key.xmss_seed, leaf_index)


def sign(key: ValidatorKey, leaf_index: int, signing_root: bytes) -> bytes:
    _record.record(key.pubkey, leaf_index, signing_root)
    return xmss.sign(key.xmss_seed, leaf_index, signing_root)


async def sign_async(key: ValidatorKey, leaf_index: int, signing_root: bytes) -> bytes:
    return await asyncio.to_thread(sign, key, leaf_index, signing_root)


def vrf_proof(key: ValidatorKey, slot: int) -> list[bytes]:
    return xmss.vrf_proof(key.vrf_seed, slot)
