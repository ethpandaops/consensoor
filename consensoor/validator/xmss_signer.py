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
MAX_XMSS_CACHE_RECORDS = 10_000


def _fetch(
    url: str,
    first: Optional[int] = None,
    last: Optional[int] = None,
    record_size: int = VRF_RECORD_SIZE,
) -> bytes:
    if "://" not in url or url.startswith("file://"):
        path = url[len("file://"):] if url.startswith("file://") else url
        with open(path, "rb") as f:
            if first is None:
                return f.read()
            f.seek(first * record_size)
            return f.read((last + 1 - first) * record_size)
    headers = {"Accept-Encoding": "identity"}
    if first is not None:
        headers["Range"] = f"bytes={first * record_size}-{(last + 1) * record_size - 1}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
        if first is not None and response.status != 206:
            raise ValueError(f"range request answered with status {response.status}")
        return response.read()


def load_xmss_caches(keys: list[ValidatorKey], state, url: str) -> int:
    """Load the XMSS trees of each key from its record in the published tree cache at url.

    The records follow the validator indices of the genesis, 14,696 bytes each, and hold no public key:
    a key's record is the one of the validator whose XMSS public key ends with the key's public parameter.
    """
    pubkeys = [bytes(v.pubkey) for v in list(state.validators)[:MAX_XMSS_CACHE_RECORDS]]
    by_param: dict[bytes, int] = {}
    ambiguous = set()
    for index, pubkey in enumerate(pubkeys):
        param = pubkey[-xmss.PUBLIC_PARAM_SIZE:]
        if param in by_param:
            ambiguous.add(param)
        by_param[param] = index
    wanted: dict[int, ValidatorKey] = {}
    for key in keys:
        if key.xmss_seed is None:
            key.xmss_seed = xmss.seed_from_privkey(key.privkey)
        param = xmss.seed_public_param(key.xmss_seed)
        if param in by_param and param not in ambiguous:
            wanted[by_param[param]] = key
    if not wanted:
        return 0
    first, last = min(wanted), max(wanted)
    size = xmss.TREE_CACHE_RECORD_SIZE
    try:
        data = _fetch(url, first, last, size)
    except Exception as e:
        logger.warning(f"Cannot fetch the XMSS tree cache from {url}: {e}")
        return 0
    loaded = 0
    for index, key in wanted.items():
        record = data[(index - first) * size:(index - first + 1) * size]
        if len(record) == size and xmss.load_cache(key.xmss_seed, pubkeys[index], record):
            loaded += 1
        else:
            logger.warning(f"XMSS tree cache record {index} does not match its key, building the key")
    return loaded


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
