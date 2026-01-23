"""Block header processing.

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/phase0/beacon-chain.md
"""

from typing import TYPE_CHECKING

from ..helpers.beacon_committee import get_beacon_proposer_index
from ....crypto import hash_tree_root

if TYPE_CHECKING:
    from ...types import BeaconState, BeaconBlock


def process_block_header(state: "BeaconState", block: "BeaconBlock") -> None:
    """Process and validate the block header.

    Verifies:
    - Slot matches state slot
    - Block slot is greater than latest block header slot
    - Proposer index is correct
    - Parent root matches latest block header root
    - Proposer is not slashed

    Args:
        state: Beacon state (modified in place)
        block: Beacon block

    Raises:
        AssertionError: If any validation fails
    """
    from ...types.phase0 import BeaconBlockHeader

    # Verify that the slot matches
    assert int(block.slot) == int(state.slot), (
        f"Block slot {block.slot} doesn't match state slot {state.slot}"
    )

    # Verify that the block is newer than the latest block header
    assert int(block.slot) > int(state.latest_block_header.slot), (
        f"Block slot {block.slot} not greater than latest header slot "
        f"{state.latest_block_header.slot}"
    )

    # Verify that proposer index is correct
    expected_proposer = get_beacon_proposer_index(state)
    assert int(block.proposer_index) == expected_proposer, (
        f"Block proposer {block.proposer_index} doesn't match expected {expected_proposer}"
    )

    # Verify that parent root matches
    parent_root = hash_tree_root(state.latest_block_header)
    assert bytes(block.parent_root) == parent_root, (
        f"Block parent root {bytes(block.parent_root).hex()[:16]} doesn't match "
        f"expected {parent_root.hex()[:16]}"
    )

    # Verify proposer is not slashed
    proposer = state.validators[int(block.proposer_index)]
    assert not proposer.slashed, "Proposer is slashed"

    # Cache current block as the new latest block header
    # (state_root is left empty, to be filled in later)
    state.latest_block_header = BeaconBlockHeader(
        slot=block.slot,
        proposer_index=block.proposer_index,
        parent_root=block.parent_root,
        state_root=b"\x00" * 32,  # Overwritten in process_slot
        body_root=hash_tree_root(block.body),
    )
