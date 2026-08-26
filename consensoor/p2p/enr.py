"""Minimal ENR (EIP-778) helpers.

The Rust p2p binding hands us our signed local ENR as a base64 string; the
discv5 ``NodeID`` (needed for PeerDAS custody-group assignment) is
``keccak256(uncompressed secp256k1 pubkey)``, so we decode the ``secp256k1``
key/value pair and decompress the 33-byte point in pure Python.
"""

import base64
import logging
from typing import Optional

import rlp
from Crypto.Hash import keccak

logger = logging.getLogger(__name__)

# secp256k1 field prime
_P = 2**256 - 2**32 - 977


def _decompress_secp256k1(compressed: bytes) -> bytes:
    """Return the 64-byte uncompressed (x || y) point for a 33-byte SEC1 key."""
    if len(compressed) != 33 or compressed[0] not in (2, 3):
        raise ValueError("not a compressed secp256k1 point")
    x = int.from_bytes(compressed[1:], "big")
    y_sq = (pow(x, 3, _P) + 7) % _P
    y = pow(y_sq, (_P + 1) // 4, _P)
    if (y * y) % _P != y_sq:
        raise ValueError("point not on curve")
    if (y & 1) != (compressed[0] & 1):
        y = _P - y
    return x.to_bytes(32, "big") + y.to_bytes(32, "big")


def decode_enr(enr_str: str) -> dict[bytes, bytes]:
    """Decode an ``enr:`` base64 record into its key/value map."""
    if enr_str.startswith("enr:"):
        enr_str = enr_str[4:]
    padding_needed = (4 - len(enr_str) % 4) % 4
    enr_bytes = base64.urlsafe_b64decode(enr_str + "=" * padding_needed)
    decoded = rlp.decode(enr_bytes)
    # [signature, seq, key1, val1, key2, val2, ...]
    return {decoded[i]: decoded[i + 1] for i in range(2, len(decoded) - 1, 2)}


def node_id_from_enr(enr_str: str) -> Optional[int]:
    """discv5 NodeID (uint256) of the record's ``secp256k1`` identity key."""
    try:
        pubkey = decode_enr(enr_str).get(b"secp256k1")
        if pubkey is None:
            return None
        digest = keccak.new(digest_bits=256, data=_decompress_secp256k1(pubkey)).digest()
        return int.from_bytes(digest, "big")
    except Exception as e:
        logger.debug(f"Failed to derive node_id from ENR: {e}")
        return None
