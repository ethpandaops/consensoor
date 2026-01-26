"""Cryptographic utilities.

Uses blspy (fast C/assembly) when available, falls back to py_ecc (pure Python).
"""

from .crypto import (
    sha256,
    hash_tree_root,
    compute_signing_root,
    sign,
    verify,
    aggregate_signatures,
    verify_aggregate,
    fast_aggregate_verify,
    bls_verify,
    bls_aggregate_pubkeys,
    pubkey_from_privkey,
    aggregate_pubkeys,
)

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
