"""Compare SSZ type definitions between platforms."""
import sys
import platform

print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()} {platform.machine()}")

from consensoor.spec.constants import set_preset, SLOTS_PER_HISTORICAL_ROOT, EPOCHS_PER_HISTORICAL_VECTOR
set_preset("minimal")

print(f"\n=== Constants ===")
print(f"SLOTS_PER_HISTORICAL_ROOT: {SLOTS_PER_HISTORICAL_ROOT()}")
print(f"EPOCHS_PER_HISTORICAL_VECTOR: {EPOCHS_PER_HISTORICAL_VECTOR()}")

from consensoor.spec.types.bellatrix import BellatrixBeaconState
from consensoor.spec.types.phase0 import BeaconBlockHeader

print(f"\n=== BellatrixBeaconState fields ===")
for name, typ in BellatrixBeaconState.__annotations__.items():
    print(f"  {name}: {typ}")

print(f"\n=== BeaconBlockHeader fields ===")
for name, typ in BeaconBlockHeader.__annotations__.items():
    print(f"  {name}: {typ}")

print(f"\n=== BeaconBlockHeader __annotations__ order ===")
print(list(BeaconBlockHeader.__annotations__.keys()))

print(f"\n=== Test SSZ encode/decode consistency ===")
from consensoor.spec.types.base import uint64, Root

hdr = BeaconBlockHeader(
    slot=575,
    proposer_index=90,
    parent_root=Root(bytes.fromhex("1ab9e921bde2dfe7445e536c09639abf515d8618a32975c2d6ae6878f342325f")),
    state_root=Root(bytes.fromhex("7f18ff5219d3c46730de6e7edf36d38dc66c2733d401e6b10c5990dd633ee91e")),
    body_root=Root(bytes.fromhex("10206332475085973633134e259a2287586c3b9da7024774831dca2fc67ce7a1")),
)

serialized = bytes(hdr.encode_bytes())
print(f"Serialized length: {len(serialized)}")
print(f"Serialized hex: {serialized.hex()}")
print(f"hash_tree_root: {hdr.hash_tree_root().hex()}")

decoded = BeaconBlockHeader.decode_bytes(serialized)
print(f"Decoded hash_tree_root: {decoded.hash_tree_root().hex()}")
print(f"Round-trip match: {hdr.hash_tree_root() == decoded.hash_tree_root()}")
