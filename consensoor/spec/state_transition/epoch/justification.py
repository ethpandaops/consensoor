"""Justification and finalization processing.

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/phase0/beacon-chain.md
"""

from typing import TYPE_CHECKING

from ...constants import (
    GENESIS_EPOCH,
    TIMELY_TARGET_FLAG_INDEX,
)
from ..helpers.accessors import (
    get_current_epoch,
    get_previous_epoch,
    get_total_active_balance,
    get_block_root,
)
from ..helpers.attestation import get_unslashed_participating_indices
from ..helpers.accessors import get_total_balance

if TYPE_CHECKING:
    from ...types import BeaconState, Checkpoint


def process_justification_and_finalization(state: "BeaconState") -> None:
    """Process justification and finalization.

    This function updates the justification bits and checkpoints based on
    attestation participation, implementing the Casper FFG finality gadget.

    Args:
        state: Beacon state (modified in place)
    """
    # Skip first two epochs (need previous epoch data)
    current_epoch = get_current_epoch(state)
    if current_epoch <= GENESIS_EPOCH + 1:
        return

    previous_epoch = get_previous_epoch(state)

    # Get participation balances
    previous_indices = get_unslashed_participating_indices(
        state, TIMELY_TARGET_FLAG_INDEX, previous_epoch
    )
    current_indices = get_unslashed_participating_indices(
        state, TIMELY_TARGET_FLAG_INDEX, current_epoch
    )

    total_active_balance = get_total_active_balance(state)
    previous_target_balance = get_total_balance(state, previous_indices)
    current_target_balance = get_total_balance(state, current_indices)

    weigh_justification_and_finalization(
        state,
        total_active_balance,
        previous_target_balance,
        current_target_balance,
    )


def weigh_justification_and_finalization(
    state: "BeaconState",
    total_active_balance: int,
    previous_epoch_target_balance: int,
    current_epoch_target_balance: int,
) -> None:
    """Weigh justification and finalization based on attestation balances.

    Args:
        state: Beacon state (modified in place)
        total_active_balance: Total effective balance of active validators
        previous_epoch_target_balance: Balance of validators attesting to previous epoch target
        current_epoch_target_balance: Balance of validators attesting to current epoch target
    """
    from ...types import Checkpoint, Root

    previous_epoch = get_previous_epoch(state)
    current_epoch = get_current_epoch(state)

    old_previous_justified_checkpoint = state.previous_justified_checkpoint
    old_current_justified_checkpoint = state.current_justified_checkpoint

    # Process justifications
    state.previous_justified_checkpoint = state.current_justified_checkpoint

    # Shift justification bits (new bit at position 0)
    # justification_bits is a Bitvector[4]
    bits = [state.justification_bits[i] for i in range(4)]
    bits = [False] + bits[:3]  # Shift right

    # Justify previous epoch if supermajority attested
    if previous_epoch_target_balance * 3 >= total_active_balance * 2:
        state.current_justified_checkpoint = Checkpoint(
            epoch=previous_epoch,
            root=Root(get_block_root(state, previous_epoch)),
        )
        bits[1] = True  # 2nd position (previous epoch)

    # Justify current epoch if supermajority attested
    if current_epoch_target_balance * 3 >= total_active_balance * 2:
        state.current_justified_checkpoint = Checkpoint(
            epoch=current_epoch,
            root=Root(get_block_root(state, current_epoch)),
        )
        bits[0] = True  # 1st position (current epoch)

    # Update justification bits
    for i in range(4):
        state.justification_bits[i] = bits[i]

    # Process finalizations using Casper FFG rules
    # The 2/3/4 rules determine when we can finalize

    # Check for finalization
    # Rule 1: epochs k-3, k-2, k-1, k - all justified, finalize k-3
    if all(bits[1:4]) and int(old_previous_justified_checkpoint.epoch) + 3 == current_epoch:
        state.finalized_checkpoint = old_previous_justified_checkpoint

    # Rule 2: epochs k-2, k-1, k justified, finalize k-2
    if all(bits[1:3]) and int(old_previous_justified_checkpoint.epoch) + 2 == current_epoch:
        state.finalized_checkpoint = old_previous_justified_checkpoint

    # Rule 3: epochs k-1, k justified, finalize k-1 (from previous epoch perspective)
    if all(bits[0:2]) and int(old_current_justified_checkpoint.epoch) + 1 == current_epoch:
        state.finalized_checkpoint = old_current_justified_checkpoint

    # Rule 4: epochs k-2, k-1 justified, finalize k-2 (from current epoch perspective)
    if all(bits[0:3]) and int(old_current_justified_checkpoint.epoch) + 2 == current_epoch:
        state.finalized_checkpoint = old_current_justified_checkpoint
