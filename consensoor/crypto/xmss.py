"""leanXMSS signatures and VRF trees, backed by the consensoor-xmss shared library."""

import asyncio
import ctypes
import hashlib
import os
from pathlib import Path
from typing import Optional, Sequence

from . import leanvm

PUBKEY_SIZE = 32
PUBLIC_PARAM_SIZE = 16
TREE_CACHE_RECORD_SIZE = 4096 + 2 * 5300
SIGNATURE_SIZE = 2404
MESSAGE_SIZE = 32
LEAF_INDEX_SIZE = 4
VRF_PROOF_LENGTH = 32
VRF_STATE_SIZE = 4 + 31 * 32

SEED_DOMAIN = b"leanxmss-seed"
VRF_SEED_DOMAIN = b"vrf-seed"

Statement = tuple[Sequence[bytes], int, bytes]

_lib: Optional[ctypes.CDLL] = None


def _candidates() -> list[Path]:
    paths = []
    env = os.environ.get("CONSENSOOR_XMSS_LIB")
    if env:
        paths.append(Path(env))
    source_dir = Path(__file__).resolve().parents[2] / "consensoor-xmss"
    paths += [
        source_dir / "libleanxmss.so",
        source_dir / "libleanxmss.dylib",
        Path("/usr/local/lib/libleanxmss.so"),
    ]
    return paths


def available() -> bool:
    return any(path.exists() for path in _candidates())


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    for path in _candidates():
        if path.exists():
            lib = ctypes.CDLL(str(path))
            break
    else:
        raise RuntimeError(
            "leanXMSS library not found, set CONSENSOOR_XMSS_LIB or build consensoor-xmss"
        )
    lib.xmss_public_key.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.xmss_public_key.restype = None
    lib.xmss_seed_public_param.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.xmss_seed_public_param.restype = None
    lib.xmss_load_cache.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.xmss_load_cache.restype = ctypes.c_int
    lib.xmss_prepare.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
    lib.xmss_prepare.restype = ctypes.c_int
    lib.xmss_sign.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_char_p]
    lib.xmss_sign.restype = ctypes.c_int
    lib.xmss_verify.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_char_p]
    lib.xmss_verify.restype = ctypes.c_int
    lib.vrf_root.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.vrf_root.restype = None
    lib.vrf_proof.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.c_char_p]
    lib.vrf_proof.restype = None
    lib.vrf_set_filled_leaves.argtypes = [ctypes.c_uint64]
    lib.vrf_set_filled_leaves.restype = ctypes.c_int
    lib.vrf_load_state.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.vrf_load_state.restype = ctypes.c_int
    _lib = lib
    return lib


def seed_from_privkey(privkey: int) -> bytes:
    return hashlib.sha256(SEED_DOMAIN + privkey.to_bytes(32, "big")).digest()


def vrf_seed_from_privkey(privkey: int) -> bytes:
    return hashlib.sha256(VRF_SEED_DOMAIN + privkey.to_bytes(32, "big")).digest()


def public_key(seed: bytes) -> bytes:
    out = ctypes.create_string_buffer(PUBKEY_SIZE)
    _load().xmss_public_key(seed, out)
    return out.raw


def seed_public_param(seed: bytes) -> bytes:
    out = ctypes.create_string_buffer(PUBLIC_PARAM_SIZE)
    _load().xmss_seed_public_param(seed, out)
    return out.raw


def load_cache(seed: bytes, pubkey: bytes, cache: bytes) -> bool:
    assert len(pubkey) == PUBKEY_SIZE
    return _load().xmss_load_cache(seed, pubkey, cache, len(cache)) == 0


def prepare(seed: bytes, leaf_index: int) -> None:
    if _load().xmss_prepare(seed, leaf_index) != 0:
        raise RuntimeError(f"cannot prepare leaf {leaf_index}")


def sign(seed: bytes, leaf_index: int, message: bytes) -> bytes:
    assert len(message) == MESSAGE_SIZE
    out = ctypes.create_string_buffer(SIGNATURE_SIZE)
    if _load().xmss_sign(seed, leaf_index, message, out) != 0:
        raise RuntimeError(f"cannot sign at leaf {leaf_index}")
    return out.raw


def get_leaf_index(signature: bytes) -> int:
    return int.from_bytes(bytes(signature)[:LEAF_INDEX_SIZE], "little")


def verify(pubkey: bytes, leaf_index: int, message: bytes, signature: bytes) -> bool:
    from .crypto import bls_verification_enabled

    if not bls_verification_enabled():
        return True
    pubkey, message, signature = bytes(pubkey), bytes(message), bytes(signature)
    if (
        len(pubkey) != PUBKEY_SIZE
        or len(message) != MESSAGE_SIZE
        or len(signature) != SIGNATURE_SIZE
        or not 0 <= leaf_index < 2**32
    ):
        return False
    return _load().xmss_verify(pubkey, leaf_index, message, signature) == 1


def vrf_root(vrf_seed: bytes) -> bytes:
    out = ctypes.create_string_buffer(32)
    _load().vrf_root(vrf_seed, out)
    return out.raw


def vrf_set_filled_leaves(leaves: int) -> None:
    """Zero the VRF leaves from ``leaves`` on, which only tests do: such a tree is not the specification's."""
    if _load().vrf_set_filled_leaves(leaves) != 0:
        raise ValueError(f"cannot fill {leaves} VRF leaves")


def vrf_load_state(vrf_seed: bytes, state: bytes) -> bool:
    assert len(state) == VRF_STATE_SIZE
    return _load().vrf_load_state(vrf_seed, state) == 0


def vrf_proof(vrf_seed: bytes, slot: int) -> list[bytes]:
    out = ctypes.create_string_buffer(32 * VRF_PROOF_LENGTH)
    _load().vrf_proof(vrf_seed, slot, out)
    return [out.raw[i * 32:(i + 1) * 32] for i in range(VRF_PROOF_LENGTH)]


def _claims(statements: Sequence[Statement]) -> list[tuple[bytes, bytes]]:
    return [(bytes(pubkey), bytes(message)) for pubkeys, _, message in statements for pubkey in pubkeys]


def prove_aggregates(statements: Sequence[Statement], signatures: Sequence[Sequence[bytes]]) -> bytes:
    if len(statements) != len(signatures):
        raise ValueError("one signature group per statement is required")
    for (pubkeys, leaf_index, _), group in zip(statements, signatures):
        if len(pubkeys) != len(group):
            raise ValueError("one signature per public key is required")
        if any(get_leaf_index(signature) != leaf_index for signature in group):
            raise ValueError("signature made at another leaf than its statement")
    if leanvm.enabled():
        return leanvm.prove(
            _claims(statements),
            [
                (bytes(pubkey), bytes(message), bytes(signature))
                for (pubkeys, _, message), group in zip(statements, signatures)
                for pubkey, signature in zip(pubkeys, group)
            ],
        )
    return b"".join(bytes(s) for group in signatures for s in group)


def merge_proofs(children: Sequence[tuple[Sequence[Statement], bytes]]) -> bytes:
    if leanvm.enabled():
        return leanvm.merge([(_claims(statements), proof) for statements, proof in children])
    return b"".join(bytes(proof) for _, proof in children)


def verify_aggregates(statements: Sequence[Statement], proof: bytes) -> bool:
    from .crypto import bls_verification_enabled

    if not bls_verification_enabled():
        return True
    proof = bytes(proof)
    if leanvm.enabled():
        return leanvm.check(_claims(statements), proof)
    offset = 0
    for pubkeys, leaf_index, message in statements:
        for pubkey in pubkeys:
            signature = proof[offset:offset + SIGNATURE_SIZE]
            if len(signature) != SIGNATURE_SIZE:
                return False
            if not verify(pubkey, leaf_index, message, signature):
                return False
            offset += SIGNATURE_SIZE
    return offset == len(proof)


async def sign_async(seed: bytes, leaf_index: int, message: bytes) -> bytes:
    return await asyncio.to_thread(sign, seed, leaf_index, message)


async def verify_async(pubkey: bytes, leaf_index: int, message: bytes, signature: bytes) -> bool:
    return await asyncio.to_thread(verify, pubkey, leaf_index, message, signature)
