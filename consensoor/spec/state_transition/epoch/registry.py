"""Validator registry processing.

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/phase0/beacon-chain.md
Reference (Electra): https://github.com/ethereum/consensus-specs/blob/master/specs/electra/beacon-chain.md
"""

from typing import TYPE_CHECKING

from ...constants import (
    FAR_FUTURE_EPOCH,
    EJECTION_BALANCE,
    MAX_EFFECTIVE_BALANCE,
    MAX_PER_EPOCH_ACTIVATION_CHURN_LIMIT,
)
from ..helpers.accessors import (
    get_current_epoch,
    get_validator_churn_limit,
)
from ..helpers.predicates import (
    is_active_validator,
    is_eligible_for_activation_queue,
    is_eligible_for_activation,
)
from ..helpers.mutators import initiate_validator_exit
from ..helpers.misc import compute_activation_exit_epoch

if TYPE_CHECKING:
    from ...types import BeaconState


def process_registry_updates(state: "BeaconState") -> None:
    """Process validator registry updates.

    This function:
    1. Marks validators eligible for activation queue
    2. Ejects validators with balance below ejection threshold
    3. Activates queued validators (Electra: all eligible; pre-Electra: up to churn limit)

    Args:
        state: Beacon state (modified in place)
    """
    current_epoch = get_current_epoch(state)
    activation_epoch = compute_activation_exit_epoch(current_epoch)

    # Check if this is Electra+ (has pending_deposits field)
    is_electra = hasattr(state, "pending_deposits")

    if is_electra:
        # Electra: simplified single-pass processing
        for index, validator in enumerate(state.validators):
            if is_eligible_for_activation_queue(validator):
                validator.activation_eligibility_epoch = current_epoch + 1
            elif (
                is_active_validator(validator, current_epoch)
                and int(validator.effective_balance) <= EJECTION_BALANCE
            ):
                initiate_validator_exit(state, index)
            elif is_eligible_for_activation(state, validator):
                validator.activation_epoch = activation_epoch
    else:
        # Pre-Electra: separate passes with churn limit
        for index, validator in enumerate(state.validators):
            if is_eligible_for_activation_queue(validator):
                validator.activation_eligibility_epoch = current_epoch + 1

            if (
                is_active_validator(validator, current_epoch)
                and int(validator.effective_balance) <= EJECTION_BALANCE
            ):
                initiate_validator_exit(state, index)

        # Queue validators for activation
        activation_queue = sorted(
            [
                index
                for index, v in enumerate(state.validators)
                if is_eligible_for_activation(state, v)
            ],
            key=lambda i: (
                int(state.validators[i].activation_eligibility_epoch),
                i,
            ),
        )

        # Activate validators up to the churn limit
        churn_limit = get_validator_activation_churn_limit(state)
        for index in activation_queue[:churn_limit]:
            validator = state.validators[index]
            validator.activation_epoch = activation_epoch


def get_validator_activation_churn_limit(state: "BeaconState") -> int:
    """Return the validator activation churn limit (Deneb+).

    This is the minimum of MAX_PER_EPOCH_ACTIVATION_CHURN_LIMIT and
    the normal churn limit.

    Args:
        state: Beacon state

    Returns:
        Maximum number of validators that can be activated per epoch
    """
    return min(MAX_PER_EPOCH_ACTIVATION_CHURN_LIMIT(), get_validator_churn_limit(state))
