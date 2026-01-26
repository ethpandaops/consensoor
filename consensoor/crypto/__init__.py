"""Cryptographic utilities.

Uses blspy (fast C/assembly) when available, falls back to py_ecc (pure Python).
"""

import hashlib
import logging
from typing import Sequence

logger = logging.getLogger(__name__)

# Try to use blspy (fast) first, fall back to py_ecc (slow)
_USE_BLSPY = False
try:
    from blspy import (
        PrivateKey as BlsPrivateKey,
        G1Element,
        G2Element,
        AugSchemeMPL,
        PopSchemeMPL,
    )
    _USE_BLSPY = True
    logger.info("Using blspy for BLS cryptography (fast)")
except ImportError:
    from py_ecc.bls import G2ProofOfPossession as _py_ecc_bls
    logger.warning("blspy not available, using py_ecc (slow) - install blspy for better performance")


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
    if isinstance(obj, bytes):
        if len(obj) == 32:
            return obj
        raise ValueError(f"Expected 32-byte root, got {len(obj)} bytes")

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
    if _USE_BLSPY:
        # Convert int to 32-byte big-endian, then to blspy PrivateKey
        privkey_bytes = privkey.to_bytes(32, 'big')
        sk = BlsPrivateKey.from_bytes(privkey_bytes)
        sig = PopSchemeMPL.sign(sk, message)
        return bytes(sig)
    else:
        return _py_ecc_bls.Sign(privkey, message)


def verify(pubkey: bytes, message: bytes, signature: bytes) -> bool:
    """Verify a BLS signature."""
    try:
        if _USE_BLSPY:
            pk = G1Element.from_bytes(pubkey)
            sig = G2Element.from_bytes(signature)
            return PopSchemeMPL.verify(pk, message, sig)
        else:
            return _py_ecc_bls.Verify(pubkey, message, signature)
    except Exception:
        return False


def aggregate_signatures(signatures: Sequence[bytes]) -> bytes:
    """Aggregate multiple BLS signatures."""
    if _USE_BLSPY:
        sigs = [G2Element.from_bytes(s) for s in signatures]
        agg = AugSchemeMPL.aggregate(sigs)
        return bytes(agg)
    else:
        return _py_ecc_bls.Aggregate(list(signatures))


def verify_aggregate(pubkeys: Sequence[bytes], messages: Sequence[bytes], signature: bytes) -> bool:
    """Verify an aggregate BLS signature."""
    try:
        if _USE_BLSPY:
            pks = [G1Element.from_bytes(pk) for pk in pubkeys]
            sig = G2Element.from_bytes(signature)
            return AugSchemeMPL.aggregate_verify(pks, list(messages), sig)
        else:
            return _py_ecc_bls.AggregateVerify(list(pubkeys), list(messages), signature)
    except Exception:
        return False


def fast_aggregate_verify(pubkeys: Sequence[bytes], message: bytes, signature: bytes) -> bool:
    """Verify an aggregate signature where all signers signed the same message.

    Implements eth_fast_aggregate_verify from the consensus spec.
    When pubkeys is empty, checks if signature is the point at infinity.
    """
    try:
        if len(pubkeys) == 0:
            g2_point_at_infinity = b'\xc0' + b'\x00' * 95
            result = signature == g2_point_at_infinity
            if not result:
                logger.error(
                    f"fast_aggregate_verify: empty pubkeys but sig != infinity. "
                    f"sig_len={len(signature)}, sig={signature.hex()[:32]}..., "
                    f"expected={g2_point_at_infinity.hex()[:32]}..."
                )
            return result

        if _USE_BLSPY:
            pks = [G1Element.from_bytes(pk) for pk in pubkeys]
            sig = G2Element.from_bytes(signature)
            return PopSchemeMPL.fast_aggregate_verify(pks, message, sig)
        else:
            return _py_ecc_bls.FastAggregateVerify(list(pubkeys), message, signature)
    except Exception:
        return False


def pubkey_from_privkey(privkey: int) -> bytes:
    """Derive public key from private key."""
    if _USE_BLSPY:
        privkey_bytes = privkey.to_bytes(32, 'big')
        sk = BlsPrivateKey.from_bytes(privkey_bytes)
        return bytes(sk.get_g1())
    else:
        return _py_ecc_bls.SkToPk(privkey)


def aggregate_pubkeys(pubkeys: Sequence[bytes]) -> bytes:
    """Aggregate multiple public keys."""
    if not pubkeys:
        raise ValueError("Cannot aggregate empty list of pubkeys")

    if _USE_BLSPY:
        pks = [G1Element.from_bytes(pk) for pk in pubkeys]
        agg = pks[0]
        for pk in pks[1:]:
            agg = agg + pk
        return bytes(agg)
    else:
        return _py_ecc_bls._AggregatePKs(list(pubkeys))


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
