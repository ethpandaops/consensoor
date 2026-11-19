"""leanXMSS signing for validator keys, with a durable record of the leaves used."""

import asyncio
import logging
import os
import threading
import urllib.request
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


VRF_RECORD_SIZE = 32 + xmss.VRF_STATE_SIZE


def _fetch(url: str, first: Optional[int] = None, last: Optional[int] = None) -> bytes:
    headers = {}
    if first is not None:
        headers["Range"] = f"bytes={first * VRF_RECORD_SIZE}-{(last + 1) * VRF_RECORD_SIZE - 1}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
        if first is not None and response.status != 206:
            raise ValueError(f"range request answered with status {response.status}")
        return response.read()


def load_vrf_states(keys: list[ValidatorKey], state, url: str) -> int:
    """Load the precomputed VRF tree state of each key whose VRF root the file at url holds."""
    wanted = {}
    for key in keys:
        if key.validator_index is not None and key.vrf_seed is not None and not key.vrf_state_loaded:
            wanted[bytes(state.validators[key.validator_index].vrf_root)] = key
    if not wanted:
        return 0
    records = {}
    indices = [key.validator_index for key in wanted.values()]
    first = min(indices)
    try:
        data = _fetch(url, first, max(indices))
        for root, key in wanted.items():
            offset = (key.validator_index - first) * VRF_RECORD_SIZE
            if data[offset:offset + 32] == root and len(data) >= offset + VRF_RECORD_SIZE:
                records[root] = data[offset:offset + VRF_RECORD_SIZE]
    except Exception as e:
        logger.warning(f"Cannot fetch VRF states by range, fetching the whole file: {e}")
    if len(records) < len(wanted):
        try:
            data = _fetch(url)
            for offset in range(0, len(data) - VRF_RECORD_SIZE + 1, VRF_RECORD_SIZE):
                root = data[offset:offset + 32]
                if root in wanted and root not in records:
                    records[root] = data[offset:offset + VRF_RECORD_SIZE]
        except Exception as e:
            logger.warning(f"Cannot fetch VRF states from {url}: {e}")
    loaded = 0
    for root, record in records.items():
        key = wanted[root]
        if xmss.vrf_load_state(key.vrf_seed, record[32:]) and xmss.vrf_root(key.vrf_seed) == root:
            key.vrf_state_loaded = True
            loaded += 1
        else:
            logger.warning(f"VRF state of validator {key.validator_index} does not match its seed")
    return loaded


def vrf_proof(key: ValidatorKey, slot: int) -> list[bytes]:
    return xmss.vrf_proof(key.vrf_seed, slot)
