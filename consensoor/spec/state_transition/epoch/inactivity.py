"""Inactivity score processing (Altair+).

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/altair/beacon-chain.md
"""

from typing import TYPE_CHECKING

from ...constants import (
    GENESIS_EPOCH,
    TIMELY_TARGET_FLAG_INDEX,
    INACTIVITY_SCORE_BIAS,
    INACTIVITY_SCORE_RECOVERY_RATE,
)
from ..helpers.accessors import (
    get_current_epoch,
    get_previous_epoch,
    get_eligible_validator_indices,
    is_in_inactivity_leak,
)
from ..helpers.attestation import get_unslashed_participating_indices

if TYPE_CHECKING:
    from ...types import BeaconState


def process_inactivity_updates(state: "BeaconState") -> None:
    """Update inactivity scores for validators (Altair+).

    Validators that don't attest to the correct target have their
    inactivity score increased. During normal operation, scores decrease.

    Args:
        state: Beacon state (modified in place)
    """
    # Skip the genesis epoch (based on previous epoch participation)
    if get_current_epoch(state) == GENESIS_EPOCH:
        return

    previous_epoch = get_previous_epoch(state)

    # Get validators that participated in previous epoch
    unslashed_participating = get_unslashed_participating_indices(
        state, TIMELY_TARGET_FLAG_INDEX, previous_epoch
    )

    in_inactivity_leak = is_in_inactivity_leak(state)

    for index in get_eligible_validator_indices(state):
        # Increase score if validator didn't participate
        if index not in unslashed_participating:
            state.inactivity_scores[index] = (
                int(state.inactivity_scores[index]) + INACTIVITY_SCORE_BIAS
            )
        else:
            # Decrease score for participating validators
            state.inactivity_scores[index] = max(
                0,
                int(state.inactivity_scores[index]) - 1,
            )

        # During inactivity leak, scores increase faster for non-participants
        # and decrease slower for participants
        if not in_inactivity_leak:
            state.inactivity_scores[index] = max(
                0,
                int(state.inactivity_scores[index]) - INACTIVITY_SCORE_RECOVERY_RATE,
            )
