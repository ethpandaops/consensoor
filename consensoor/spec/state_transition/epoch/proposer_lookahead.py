"""Proposer lookahead processing (Fulu).

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/fulu/beacon-chain.md
"""

from typing import TYPE_CHECKING

from ...constants import (
    SLOTS_PER_EPOCH,
    MIN_SEED_LOOKAHEAD,
    DOMAIN_BEACON_PROPOSER,
)
from ..helpers.accessors import get_current_epoch, get_active_validator_indices, get_seed
from ..helpers.beacon_committee import get_beacon_proposer_indices

if TYPE_CHECKING:
    from ...types import BeaconState


def process_proposer_lookahead(state: "BeaconState") -> None:
    """Update the proposer lookahead vector for the next epoch (Fulu).

    At the end of each epoch, rotates the lookahead by one epoch and
    computes proposer indices for the furthest lookahead epoch.

    Args:
        state: Beacon state (modified in place)
    """
    if not hasattr(state, "proposer_lookahead"):
        return

    slots_per_epoch = SLOTS_PER_EPOCH()
    lookahead_length = (MIN_SEED_LOOKAHEAD + 1) * slots_per_epoch

    # Rotate the lookahead: shift by one epoch
    # Remove the first epoch's worth of slots
    new_lookahead = list(state.proposer_lookahead[slots_per_epoch:])

    # Compute proposer indices for the furthest lookahead epoch
    # This is epoch = current_epoch + MIN_SEED_LOOKAHEAD + 1
    current_epoch = get_current_epoch(state)
    furthest_epoch = current_epoch + MIN_SEED_LOOKAHEAD + 1

    # Get proposer indices for the furthest epoch
    new_proposers = get_beacon_proposer_indices(state, furthest_epoch)

    # Append new proposers to the lookahead
    new_lookahead.extend(new_proposers)

    # Update state
    for i, proposer in enumerate(new_lookahead):
        if i < len(state.proposer_lookahead):
            state.proposer_lookahead[i] = proposer
