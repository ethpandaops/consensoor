"""Pending consolidations processing (Electra).

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/electra/beacon-chain.md
"""

from typing import TYPE_CHECKING

from ..helpers.accessors import get_current_epoch
from ..helpers.mutators import increase_balance, decrease_balance
from ..helpers.predicates import is_active_validator

if TYPE_CHECKING:
    from ...types import BeaconState


def process_pending_consolidations(state: "BeaconState") -> None:
    """Process pending consolidations from the queue (Electra).

    Transfers balance from source validators to target validators
    once the source validator is withdrawable.

    Args:
        state: Beacon state (modified in place)
    """
    if not hasattr(state, "pending_consolidations"):
        return

    next_epoch = get_current_epoch(state) + 1
    next_pending_consolidation = 0

    for consolidation in state.pending_consolidations:
        source_index = int(consolidation.source_index)
        target_index = int(consolidation.target_index)
        source_validator = state.validators[source_index]

        # Skip slashed source validators
        if source_validator.slashed:
            next_pending_consolidation += 1
            continue

        # Stop processing if source validator is not yet withdrawable
        if int(source_validator.withdrawable_epoch) > next_epoch:
            break

        # Calculate the consolidated balance (min of balance and effective balance)
        source_effective_balance = min(
            int(state.balances[source_index]),
            int(source_validator.effective_balance),
        )

        # Move active balance to target. Excess balance is withdrawable.
        decrease_balance(state, source_index, source_effective_balance)
        increase_balance(state, target_index, source_effective_balance)
        next_pending_consolidation += 1

    # Remove processed consolidations
    state.pending_consolidations = list(state.pending_consolidations)[next_pending_consolidation:]
