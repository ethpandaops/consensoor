"""SSZ serialization utilities."""

from remerkleable.core import View


def encode(obj: View) -> bytes:
    """Encode an SSZ object to bytes."""
    return obj.encode_bytes()


def decode(cls: type[View], data: bytes) -> View:
    """Decode bytes into an SSZ object."""
    return cls.decode_bytes(data)


def hash_tree_root(obj: View) -> bytes:
    """Compute the hash tree root of an SSZ object."""
    return obj.hash_tree_root().encode_bytes()


__all__ = ["encode", "decode", "hash_tree_root"]
