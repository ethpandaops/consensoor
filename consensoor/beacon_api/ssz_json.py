"""Generic SSZ <-> beacon-API JSON codec.

Walks remerkleable types: integers are decimal strings, booleans are JSON
booleans, byte vectors/lists and bitfields are 0x-prefixed hex of their SSZ
encoding, containers are objects and other lists/vectors are arrays.
"""

from __future__ import annotations

from remerkleable.basic import boolean, uint
from remerkleable.bitfields import Bitlist, Bitvector
from remerkleable.byte_arrays import ByteList, ByteVector
from remerkleable.progressive import ProgressiveBitlist, ProgressiveByteList

_BYTES_TYPES = (ByteVector, ByteList, ProgressiveByteList, Bitvector, Bitlist, ProgressiveBitlist)


def _is_hex_type(typ) -> bool:
    if issubclass(typ, _BYTES_TYPES):
        return True
    elem = getattr(typ, "element_cls", None)
    if elem is None:
        return False
    elem = elem()
    return issubclass(elem, uint) and elem.type_byte_length() == 1 and not issubclass(elem, boolean)


def _hex_to_bytes(value: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError(f"expected hex string, got {type(value).__name__}")
    return bytes.fromhex(value[2:] if value.startswith("0x") else value)


def to_json(value):
    typ = type(value)
    if isinstance(value, boolean):
        return bool(value)
    if isinstance(value, uint):
        return str(int(value))
    if _is_hex_type(typ):
        return "0x" + bytes(value.encode_bytes()).hex()
    if hasattr(typ, "fields"):
        return {name: to_json(getattr(value, name)) for name in typ.fields().keys()}
    if hasattr(typ, "element_cls"):
        return [to_json(v) for v in value]
    raise TypeError(f"unsupported SSZ type {typ}")


def from_json(typ, obj):
    if issubclass(typ, boolean):
        if not isinstance(obj, bool):
            raise ValueError(f"expected boolean, got {obj!r}")
        return typ(obj)
    if issubclass(typ, uint):
        return typ(int(obj))
    if _is_hex_type(typ):
        return typ.decode_bytes(_hex_to_bytes(obj))
    if hasattr(typ, "fields"):
        if not isinstance(obj, dict):
            raise ValueError(f"expected object for {typ.__name__}")
        kwargs = {}
        for name, ftyp in typ.fields().items():
            if name not in obj:
                raise ValueError(f"missing field {name!r} in {typ.__name__}")
            kwargs[name] = from_json(ftyp, obj[name])
        return typ(**kwargs)
    if hasattr(typ, "element_cls"):
        if not isinstance(obj, list):
            raise ValueError(f"expected array for {typ.__name__}")
        elem = typ.element_cls()
        return typ(*[from_json(elem, v) for v in obj])
    raise TypeError(f"unsupported SSZ type {typ}")
