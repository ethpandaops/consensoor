"""Debug script to identify architecture-dependent hash differences."""
import sys
import platform
import copy
import snappy
from pathlib import Path

print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()} {platform.machine()}")

from consensoor.spec.constants import set_preset
set_preset("minimal")

from consensoor.spec.types.bellatrix import BellatrixBeaconState, SignedBellatrixBeaconBlock
from consensoor.spec.types.phase0 import BeaconBlockHeader
from consensoor.crypto import hash_tree_root

test_dir = Path("tests/spec-tests/tests/minimal/bellatrix/random/random/pyspec_tests/randomized_10")

pre_file = test_dir / "pre.ssz_snappy"
block0_file = test_dir / "blocks_0.ssz_snappy"

with open(pre_file, "rb") as f:
    pre_state = BellatrixBeaconState.decode_bytes(snappy.decompress(f.read()))

with open(block0_file, "rb") as f:
    block0 = SignedBellatrixBeaconBlock.decode_bytes(snappy.decompress(f.read()))

print(f"\n=== Pre-state latest_block_header ===")
hdr = pre_state.latest_block_header
print(f"slot: {hdr.slot}")
print(f"proposer_index: {hdr.proposer_index}")
print(f"parent_root: {bytes(hdr.parent_root).hex()}")
print(f"state_root: {bytes(hdr.state_root).hex()}")
print(f"body_root: {bytes(hdr.body_root).hex()}")
print(f"hash_tree_root: {hash_tree_root(hdr).hex()}")

print(f"\n=== Block 0 ===")
blk = block0.message
print(f"slot: {blk.slot}")
print(f"proposer_index: {blk.proposer_index}")
print(f"parent_root: {bytes(blk.parent_root).hex()}")
print(f"body hash_tree_root: {hash_tree_root(blk.body).hex()}")

print(f"\n=== State hash_tree_root (pre-state) ===")
print(f"hash: {hash_tree_root(pre_state).hex()}")

print(f"\n=== Testing copy.deepcopy vs SSZ round-trip ===")
state_deepcopy = copy.deepcopy(pre_state)
state_ssz_copy = BellatrixBeaconState.decode_bytes(bytes(pre_state.encode_bytes()))

print(f"Original state hash:  {hash_tree_root(pre_state).hex()}")
print(f"deepcopy state hash:  {hash_tree_root(state_deepcopy).hex()}")
print(f"SSZ copy state hash:  {hash_tree_root(state_ssz_copy).hex()}")
print(f"deepcopy matches: {hash_tree_root(pre_state) == hash_tree_root(state_deepcopy)}")
print(f"SSZ copy matches: {hash_tree_root(pre_state) == hash_tree_root(state_ssz_copy)}")

print(f"\n=== BeaconBlockHeader field order (from __annotations__) ===")
print(list(BeaconBlockHeader.__annotations__.keys()))

print(f"\n=== Serialized BeaconBlockHeader (pre-state) ===")
serialized = bytes(hdr.encode_bytes())
print(f"length: {len(serialized)}")
print(f"hex: {serialized.hex()}")

print(f"\n=== Testing process_slots and process_block_header ===")
from consensoor.spec.state_transition import process_slots
from consensoor.spec.state_transition.block import process_block_header
from consensoor.spec.types.phase0 import BeaconBlockHeader

state_copy = BellatrixBeaconState.decode_bytes(bytes(pre_state.encode_bytes()))
print(f"State before process_slots: slot={state_copy.slot}")

process_slots(state_copy, int(block0.message.slot))
print(f"State after process_slots: slot={state_copy.slot}")

print(f"\n=== State latest_block_header after process_slots ===")
hdr_after = state_copy.latest_block_header
print(f"slot: {hdr_after.slot}")
print(f"proposer_index: {hdr_after.proposer_index}")
print(f"parent_root: {bytes(hdr_after.parent_root).hex()}")
print(f"state_root: {bytes(hdr_after.state_root).hex()}")
print(f"body_root: {bytes(hdr_after.body_root).hex()}")
print(f"hash_tree_root: {hash_tree_root(hdr_after).hex()}")
print(f"Serialized: {bytes(hdr_after.encode_bytes()).hex()}")

print(f"\n=== Block 0 parent_root ===")
print(f"Block parent_root: {bytes(block0.message.parent_root).hex()}")
print(f"Expected (state.latest_block_header hash): {hash_tree_root(hdr_after).hex()}")
print(f"Match: {bytes(block0.message.parent_root) == hash_tree_root(hdr_after)}")

print(f"\n=== Processing FULL block 0 (process_block_header only) ===")
state_copy2 = BellatrixBeaconState.decode_bytes(bytes(pre_state.encode_bytes()))
process_slots(state_copy2, int(block0.message.slot))
process_block_header(state_copy2, block0.message)
print(f"State after process_block_header:")
print(f"  latest_block_header.slot: {state_copy2.latest_block_header.slot}")
print(f"  latest_block_header hash: {hash_tree_root(state_copy2.latest_block_header).hex()}")
print(f"  latest_block_header.body_root: {bytes(state_copy2.latest_block_header.body_root).hex()}")
print(f"  Block body hash: {hash_tree_root(block0.message.body).hex()}")

print(f"\n=== Loading block 1 ===")
block1_file = test_dir / "blocks_1.ssz_snappy"
with open(block1_file, "rb") as f:
    block1 = SignedBellatrixBeaconBlock.decode_bytes(snappy.decompress(f.read()))
print(f"Block 1 slot: {block1.message.slot}")
print(f"Block 1 parent_root: {bytes(block1.message.parent_root).hex()}")

print(f"\n=== Processing slots from 576 to {block1.message.slot} ===")
process_slots(state_copy2, int(block1.message.slot))
print(f"State slot after process_slots: {state_copy2.slot}")
print(f"latest_block_header hash: {hash_tree_root(state_copy2.latest_block_header).hex()}")

print(f"\n=== Details of latest_block_header (for block 1 parent check) ===")
hdr3 = state_copy2.latest_block_header
print(f"slot: {hdr3.slot}")
print(f"proposer_index: {hdr3.proposer_index}")
print(f"parent_root: {bytes(hdr3.parent_root).hex()}")
print(f"state_root: {bytes(hdr3.state_root).hex()}")
print(f"body_root: {bytes(hdr3.body_root).hex()}")
print(f"Serialized: {bytes(hdr3.encode_bytes()).hex()}")

print(f"\n=== Block 1 parent check ===")
print(f"Block 1 parent_root: {bytes(block1.message.parent_root).hex()}")
print(f"Expected parent_root: {hash_tree_root(hdr3).hex()}")
print(f"Match: {bytes(block1.message.parent_root) == hash_tree_root(hdr3)}")
