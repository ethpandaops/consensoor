"""Effective balance updates processing.

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/phase0/beacon-chain.md
"""

from typing import TYPE_CHECKING

from ...constants import (
    EFFECTIVE_BALANCE_INCREMENT,
    HYSTERESIS_QUOTIENT,
    HYSTERESIS_DOWNWARD_MULTIPLIER,
    HYSTERESIS_UPWARD_MULTIPLIER,
    MAX_EFFECTIVE_BALANCE,
)
from ..helpers.accessors import get_max_effective_balance

if TYPE_CHECKING:
    from ...types import BeaconState


def process_effective_balance_updates(state: "BeaconState") -> None:
    """Update validator effective balances.

    Uses hysteresis to prevent thrashing from small balance fluctuations.
    The effective balance only changes when the actual balance is sufficiently
    different from the current effective balance.

    Args:
        state: Beacon state (modified in place)
    """
    # Hysteresis thresholds
    hysteresis_increment = EFFECTIVE_BALANCE_INCREMENT // HYSTERESIS_QUOTIENT
    downward_threshold = hysteresis_increment * HYSTERESIS_DOWNWARD_MULTIPLIER
    upward_threshold = hysteresis_increment * HYSTERESIS_UPWARD_MULTIPLIER

    for index, validator in enumerate(state.validators):
        balance = int(state.balances[index])
        effective_balance = int(validator.effective_balance)

        # Get max effective balance for this validator (Electra: depends on credentials)
        max_effective = get_max_effective_balance(validator)

        # Check if effective balance should decrease
        if (
            balance + downward_threshold < effective_balance
            or effective_balance + upward_threshold < balance
        ):
            # Update effective balance (capped at max, rounded down to increment)
            new_effective = min(
                balance - balance % EFFECTIVE_BALANCE_INCREMENT,
                max_effective,
            )
            validator.effective_balance = new_effective
