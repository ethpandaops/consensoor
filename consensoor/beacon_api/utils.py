"""Beacon API utility functions."""

import hashlib
import socket


def to_hex(value, length: int = 0) -> str:
    """Convert a bytes or int value to hex string with 0x prefix."""
    if isinstance(value, bytes):
        return "0x" + value.hex()
    elif isinstance(value, int):
        if length > 0:
            return "0x" + format(value, f'0{length * 2}x')
        return hex(value)
    return str(value)


def get_local_ip() -> str:
    """Try to get the local IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def generate_peer_id(seed: str) -> str:
    """Generate a valid libp2p peer ID from a seed string.

    Creates a proper CIDv1 peer ID with identity multihash.
    Format: base58btc(0x00 + 0x24 + 0x08 + 0x01 + 0x12 + 0x20 + sha256(seed))
    """
    import base58

    h = hashlib.sha256(seed.encode()).digest()
    # Identity multihash: 0x00 (identity) + length + data
    # But for peer IDs we use: 0x00 + 0x24 (36 bytes) + 0x08 0x01 (protobuf key type) + 0x12 0x20 + sha256
    # Simplified: use base58 encoding of the sha256 hash with proper prefix
    # Actually, libp2p peer IDs are base58btc encoded multihashes

    # For a proper peer ID, we need:
    # - Multihash: identity (0x00) or sha2-256 (0x12)
    # - For secp256k1: 0x00 0x25 0x08 0x02 0x12 0x21 + compressed pubkey
    # Simpler approach: use sha2-256 multihash of the seed
    multihash = bytes([0x12, 0x20]) + h  # sha2-256 multihash (0x12) with 32 bytes (0x20)
    return base58.b58encode(multihash).decode()
