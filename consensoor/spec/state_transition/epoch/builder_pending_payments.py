"""Builder pending payments processing (Gloas/ePBS).

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/gloas/beacon-chain.md
"""

from typing import TYPE_CHECKING

from ...constants import (
    SLOTS_PER_EPOCH,
    BUILDER_PAYMENT_THRESHOLD_NUMERATOR,
    BUILDER_PAYMENT_THRESHOLD_DENOMINATOR,
)
from ..helpers.accessors import get_total_active_balance

if TYPE_CHECKING:
    from ...types.gloas import BeaconState


def get_builder_payment_quorum_threshold(state: "BeaconState") -> int:
    """Calculate the quorum threshold for builder payments.

    Args:
        state: Beacon state

    Returns:
        Quorum threshold in Gwei
    """
    per_slot_balance = get_total_active_balance(state) // SLOTS_PER_EPOCH()
    quorum = per_slot_balance * BUILDER_PAYMENT_THRESHOLD_NUMERATOR
    return quorum // BUILDER_PAYMENT_THRESHOLD_DENOMINATOR


def process_builder_pending_payments(state: "BeaconState") -> None:
    """Process the builder pending payments from the previous epoch.

    Payments that have accumulated enough weight (meeting quorum) are moved
    to the builder_pending_withdrawals queue for withdrawal processing.

    Args:
        state: Beacon state (modified in place)
    """
    from ...types.gloas import BuilderPendingPayment

    if not hasattr(state, "builder_pending_payments"):
        return

    quorum = get_builder_payment_quorum_threshold(state)
    slots_per_epoch = SLOTS_PER_EPOCH()

    # Process payments from the previous epoch (first SLOTS_PER_EPOCH entries)
    for payment in state.builder_pending_payments[:slots_per_epoch]:
        if int(payment.weight) >= quorum:
            state.builder_pending_withdrawals.append(payment.withdrawal)

    # Rotate the queue: old payments from current epoch become previous epoch
    # New empty payments are added for the current epoch
    old_payments = list(state.builder_pending_payments[slots_per_epoch:])
    new_payments = [BuilderPendingPayment() for _ in range(slots_per_epoch)]
    state.builder_pending_payments = old_payments + new_payments
