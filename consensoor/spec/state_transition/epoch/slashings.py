"""Slashings processing.

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/phase0/beacon-chain.md
"""

from typing import TYPE_CHECKING

from ...constants import (
    EPOCHS_PER_SLASHINGS_VECTOR,
    EFFECTIVE_BALANCE_INCREMENT,
    PROPORTIONAL_SLASHING_MULTIPLIER,
    PROPORTIONAL_SLASHING_MULTIPLIER_ALTAIR,
    PROPORTIONAL_SLASHING_MULTIPLIER_BELLATRIX,
)
from ..helpers.accessors import (
    get_current_epoch,
    get_total_active_balance,
)
from ..helpers.mutators import decrease_balance
from ..helpers.math import saturating_sub

if TYPE_CHECKING:
    from ...types import BeaconState


def get_proportional_slashing_multiplier(state: "BeaconState") -> int:
    """Get the proportional slashing multiplier for the current fork.

    Returns:
        - Phase0: PROPORTIONAL_SLASHING_MULTIPLIER (1 mainnet, 2 minimal)
        - Altair: PROPORTIONAL_SLASHING_MULTIPLIER_ALTAIR (2)
        - Bellatrix+: PROPORTIONAL_SLASHING_MULTIPLIER_BELLATRIX (3)
    """
    # Check for Bellatrix+ (has latest_execution_payload_header) or Gloas (execution_payload_availability)
    if hasattr(state, "latest_execution_payload_header") or hasattr(state, "execution_payload_availability"):
        return PROPORTIONAL_SLASHING_MULTIPLIER_BELLATRIX
    # Check for Altair+ (has previous_epoch_participation)
    if hasattr(state, "previous_epoch_participation"):
        return PROPORTIONAL_SLASHING_MULTIPLIER_ALTAIR
    # Phase0
    return PROPORTIONAL_SLASHING_MULTIPLIER()


def process_slashings(state: "BeaconState") -> None:
    """Process slashings for validators at their withdrawable epoch.

    Applies a penalty proportional to the total amount slashed in the
    slashing window.

    Args:
        state: Beacon state (modified in place)
    """
    epoch = get_current_epoch(state)
    total_balance = get_total_active_balance(state)

    # Get fork-appropriate multiplier
    multiplier = get_proportional_slashing_multiplier(state)

    # Sum all slashings in the window
    adjusted_total_slashing_balance = min(
        sum(int(s) for s in state.slashings) * multiplier,
        total_balance,
    )

    increment = EFFECTIVE_BALANCE_INCREMENT
    # Check if this is Electra+ (has pending_deposits field)
    is_electra = hasattr(state, "pending_deposits")

    if is_electra:
        # Electra: new formula to avoid uint64 overflow
        penalty_per_effective_balance_increment = adjusted_total_slashing_balance // (
            total_balance // increment
        )
        for index, validator in enumerate(state.validators):
            if (
                validator.slashed
                and epoch + EPOCHS_PER_SLASHINGS_VECTOR() // 2
                == int(validator.withdrawable_epoch)
            ):
                effective_balance_increments = int(validator.effective_balance) // increment
                penalty = penalty_per_effective_balance_increment * effective_balance_increments
                decrease_balance(state, index, penalty)
    else:
        # Pre-Electra formula
        for index, validator in enumerate(state.validators):
            if (
                validator.slashed
                and epoch + EPOCHS_PER_SLASHINGS_VECTOR() // 2
                == int(validator.withdrawable_epoch)
            ):
                penalty_numerator = (
                    int(validator.effective_balance) // increment * adjusted_total_slashing_balance
                )
                penalty = penalty_numerator // total_balance * increment
                decrease_balance(state, index, penalty)
