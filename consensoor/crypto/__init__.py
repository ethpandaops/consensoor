"""Cryptographic utilities."""

import hashlib
from typing import Sequence

from py_ecc.bls import G2ProofOfPossession as bls
from py_ecc.bls.g2_primitives import pubkey_to_G1, G1_to_pubkey
from remerkleable.core import View


def sha256(data: bytes) -> bytes:
    """Compute SHA256 hash."""
    return hashlib.sha256(data).digest()


def hash_tree_root(obj) -> bytes:
    """Compute the hash tree root of an SSZ object or return bytes directly.

    Args:
        obj: SSZ object with hash_tree_root() method, or 32-byte root

    Returns:
        32-byte hash tree root
    """
    # If it's already bytes (e.g., a Root/block root), return directly
    if isinstance(obj, bytes):
        if len(obj) == 32:
            return obj
        raise ValueError(f"Expected 32-byte root, got {len(obj)} bytes")

    # SSZ object - call its hash_tree_root method
    if hasattr(obj, 'hash_tree_root'):
        root = obj.hash_tree_root()
        if isinstance(root, bytes):
            return root
        return bytes(root)

    raise TypeError(f"Cannot compute hash_tree_root of {type(obj)}")


def compute_signing_root(obj, domain: bytes) -> bytes:
    """Compute the signing root for a message and domain.

    Args:
        obj: SSZ object or 32-byte root (for block roots)
        domain: 32-byte domain

    Returns:
        32-byte signing root
    """
    from consensoor.spec.types import SigningData, Root

    # Get the object root - either hash it or use directly if already bytes
    if isinstance(obj, bytes):
        if len(obj) == 32:
            object_root = obj
        else:
            raise ValueError(f"Expected 32-byte root, got {len(obj)} bytes")
    elif hasattr(obj, 'hash_tree_root'):
        root = obj.hash_tree_root()
        object_root = bytes(root) if not isinstance(root, bytes) else root
    else:
        raise TypeError(f"Cannot compute signing root of {type(obj)}")

    signing_data = SigningData(
        object_root=Root(object_root),
        domain=domain,
    )
    return hash_tree_root(signing_data)


def sign(privkey: int, message: bytes) -> bytes:
    """Sign a message with a BLS private key."""
    return bls.Sign(privkey, message)


def verify(pubkey: bytes, message: bytes, signature: bytes) -> bool:
    """Verify a BLS signature."""
    try:
        return bls.Verify(pubkey, message, signature)
    except Exception:
        return False


def aggregate_signatures(signatures: Sequence[bytes]) -> bytes:
    """Aggregate multiple BLS signatures."""
    return bls.Aggregate(list(signatures))


def verify_aggregate(pubkeys: Sequence[bytes], messages: Sequence[bytes], signature: bytes) -> bool:
    """Verify an aggregate BLS signature."""
    try:
        return bls.AggregateVerify(list(pubkeys), list(messages), signature)
    except Exception:
        return False


def fast_aggregate_verify(pubkeys: Sequence[bytes], message: bytes, signature: bytes) -> bool:
    """Verify an aggregate signature where all signers signed the same message.

    Implements eth_fast_aggregate_verify from the consensus spec.
    When pubkeys is empty, checks if signature is the point at infinity.
    """
    try:
        if len(pubkeys) == 0:
            # G2 point at infinity: 0xc0 followed by 95 zero bytes
            g2_point_at_infinity = b'\xc0' + b'\x00' * 95
            return signature == g2_point_at_infinity
        return bls.FastAggregateVerify(list(pubkeys), message, signature)
    except Exception:
        return False


def pubkey_from_privkey(privkey: int) -> bytes:
    """Derive public key from private key."""
    return bls.SkToPk(privkey)


def aggregate_pubkeys(pubkeys: Sequence[bytes]) -> bytes:
    """Aggregate multiple public keys."""
    if not pubkeys:
        raise ValueError("Cannot aggregate empty list of pubkeys")
    return bls._AggregatePKs(list(pubkeys))


# Aliases for consensus spec compatibility
bls_verify = fast_aggregate_verify
bls_aggregate_pubkeys = aggregate_pubkeys


__all__ = [
    "sha256",
    "hash_tree_root",
    "compute_signing_root",
    "sign",
    "verify",
    "aggregate_signatures",
    "verify_aggregate",
    "fast_aggregate_verify",
    "bls_verify",
    "bls_aggregate_pubkeys",
    "pubkey_from_privkey",
    "aggregate_pubkeys",
]
