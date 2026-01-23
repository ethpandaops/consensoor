"""Trace state hashes through block processing to find divergence point."""
import sys
import platform
import snappy
from pathlib import Path

print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()} {platform.machine()}")

from consensoor.spec.constants import set_preset
set_preset("minimal")

from consensoor.spec.network_config import NetworkConfig, set_config
net_config = NetworkConfig.minimal()
set_config(net_config)
print(f"Network config seconds_per_slot: {net_config.seconds_per_slot}")

from consensoor.spec.types.bellatrix import BellatrixBeaconState, SignedBellatrixBeaconBlock
from consensoor.crypto import hash_tree_root
from consensoor.spec.state_transition import process_slots
from consensoor.spec.state_transition.block import (
    process_block_header,
    process_randao,
    process_eth1_data,
    process_sync_aggregate,
)
from consensoor.spec.state_transition.block.operations import (
    process_proposer_slashing,
    process_attester_slashing,
    process_attestation,
    process_deposit,
    process_voluntary_exit,
)

test_dir = Path("tests/spec-tests/tests/minimal/bellatrix/random/random/pyspec_tests/randomized_10")

pre_file = test_dir / "pre.ssz_snappy"
block0_file = test_dir / "blocks_0.ssz_snappy"

with open(pre_file, "rb") as f:
    pre_state = BellatrixBeaconState.decode_bytes(snappy.decompress(f.read()))

with open(block0_file, "rb") as f:
    block0 = SignedBellatrixBeaconBlock.decode_bytes(snappy.decompress(f.read()))

state = BellatrixBeaconState.decode_bytes(bytes(pre_state.encode_bytes()))
block = block0.message

print(f"\n=== Initial state ===")
print(f"State slot: {state.slot}")
print(f"State hash: {hash_tree_root(state).hex()[:32]}")
print(f"latest_block_header hash: {hash_tree_root(state.latest_block_header).hex()[:32]}")

print(f"\n=== Processing slots to {block.slot} ===")
process_slots(state, int(block.slot))
print(f"State slot after process_slots: {state.slot}")
print(f"State hash: {hash_tree_root(state).hex()[:32]}")
print(f"latest_block_header hash: {hash_tree_root(state.latest_block_header).hex()[:32]}")

print(f"\n=== process_block_header ===")
process_block_header(state, block)
print(f"State hash: {hash_tree_root(state).hex()[:32]}")
print(f"latest_block_header.body_root: {bytes(state.latest_block_header.body_root).hex()[:32]}")

print(f"\n=== process_randao ===")
process_randao(state, block.body)
print(f"State hash: {hash_tree_root(state).hex()[:32]}")

print(f"\n=== process_eth1_data ===")
process_eth1_data(state, block.body)
print(f"State hash: {hash_tree_root(state).hex()[:32]}")

print(f"\n=== process_proposer_slashings ({len(block.body.proposer_slashings)}) ===")
for op in block.body.proposer_slashings:
    process_proposer_slashing(state, op)
print(f"State hash: {hash_tree_root(state).hex()[:32]}")

print(f"\n=== process_attester_slashings ({len(block.body.attester_slashings)}) ===")
for op in block.body.attester_slashings:
    process_attester_slashing(state, op)
print(f"State hash: {hash_tree_root(state).hex()[:32]}")

print(f"\n=== process_attestations ({len(block.body.attestations)}) ===")
for i, att in enumerate(block.body.attestations):
    print(f"  Attestation {i}: slot={att.data.slot}, index={att.data.index}")
    process_attestation(state, att)
    print(f"  State hash after: {hash_tree_root(state).hex()[:32]}")
print(f"State hash: {hash_tree_root(state).hex()[:32]}")

print(f"\n=== process_deposits ({len(block.body.deposits)}) ===")
for op in block.body.deposits:
    process_deposit(state, op)
print(f"State hash: {hash_tree_root(state).hex()[:32]}")

print(f"\n=== process_voluntary_exits ({len(block.body.voluntary_exits)}) ===")
for op in block.body.voluntary_exits:
    process_voluntary_exit(state, op)
print(f"State hash: {hash_tree_root(state).hex()[:32]}")

print(f"\n=== process_sync_aggregate ===")
process_sync_aggregate(state, block.body.sync_aggregate)
print(f"State hash: {hash_tree_root(state).hex()[:32]}")

print(f"\n=== Check execution payload ===")
from consensoor.spec.state_transition.helpers.predicates import is_execution_enabled
print(f"Execution enabled: {is_execution_enabled(state, block.body)}")
print(f"Has execution_payload: {hasattr(block.body, 'execution_payload')}")
if hasattr(block.body, 'execution_payload'):
    print(f"Execution payload block_hash: {bytes(block.body.execution_payload.block_hash).hex()[:32]}")

print(f"\n=== Running FULL state_transition ===")
from consensoor.spec.state_transition import state_transition

full_state = BellatrixBeaconState.decode_bytes(bytes(pre_state.encode_bytes()))
print(f"State type: {type(full_state)}")
print(f"State class __module__: {type(full_state).__module__}")
print(f"Before state_transition:")
print(f"  State hash: {hash_tree_root(full_state).hex()[:32]}")

result_state = state_transition(full_state, block0, validate_result=False)

print(f"After state_transition:")
print(f"  State type: {type(result_state)}")
print(f"  State hash: {hash_tree_root(result_state).hex()[:32]}")
print(f"  latest_block_header hash: {hash_tree_root(result_state.latest_block_header).hex()}")

print(f"\n=== Processing block 1 ===")
block1_file = test_dir / "blocks_1.ssz_snappy"
with open(block1_file, "rb") as f:
    block1 = SignedBellatrixBeaconBlock.decode_bytes(snappy.decompress(f.read()))
print(f"Block 1 slot: {block1.message.slot}")
print(f"Block 1 parent_root: {bytes(block1.message.parent_root).hex()}")

print(f"\n=== Advancing state to block 1 slot ===")
process_slots(result_state, int(block1.message.slot))
print(f"State slot: {result_state.slot}")
print(f"latest_block_header hash: {hash_tree_root(result_state.latest_block_header).hex()}")

print(f"\n=== Parent root check for block 1 ===")
computed_parent = hash_tree_root(result_state.latest_block_header)
expected_parent = bytes(block1.message.parent_root)
print(f"Block 1 expects:    {expected_parent.hex()}")
print(f"State computed:     {computed_parent.hex()}")
print(f"Match: {expected_parent == computed_parent}")

print(f"\n=== latest_block_header details ===")
hdr = result_state.latest_block_header
print(f"slot: {hdr.slot}")
print(f"proposer_index: {hdr.proposer_index}")
print(f"parent_root: {bytes(hdr.parent_root).hex()}")
print(f"state_root: {bytes(hdr.state_root).hex()}")
print(f"body_root: {bytes(hdr.body_root).hex()}")
print(f"Serialized: {bytes(hdr.encode_bytes()).hex()}")
