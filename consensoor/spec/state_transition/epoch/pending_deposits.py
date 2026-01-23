"""Pending deposits processing (Electra).

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/electra/beacon-chain.md
"""

from typing import TYPE_CHECKING

from ...constants import (
    MAX_PENDING_DEPOSITS_PER_EPOCH,
    EFFECTIVE_BALANCE_INCREMENT,
    MAX_EFFECTIVE_BALANCE,
    MAX_EFFECTIVE_BALANCE_ELECTRA,
    FAR_FUTURE_EPOCH,
    GENESIS_SLOT,
    SLOTS_PER_EPOCH,
)
from ..helpers.accessors import (
    get_current_epoch,
    get_activation_exit_churn_limit,
)
from ..helpers.mutators import increase_balance
from ..helpers.predicates import (
    has_compounding_withdrawal_credential,
    has_execution_withdrawal_credential,
    is_compounding_withdrawal_credential,
)

if TYPE_CHECKING:
    from ...types import BeaconState
    from ...types.electra import PendingDeposit


def process_pending_deposits(state: "BeaconState") -> None:
    """Process pending deposits from the queue (Electra).

    Applies deposits up to the per-epoch limit and churn budget.

    Args:
        state: Beacon state (modified in place)
    """
    if not hasattr(state, "pending_deposits"):
        return

    next_epoch = get_current_epoch(state) + 1
    available_for_processing = (
        int(state.deposit_balance_to_consume) + get_activation_exit_churn_limit(state)
    )
    processed_amount = 0
    next_deposit_index = 0
    deposits_to_postpone = []
    is_churn_limit_reached = False

    # Compute finalized slot
    finalized_slot = int(state.finalized_checkpoint.epoch) * SLOTS_PER_EPOCH()

    for deposit in state.pending_deposits:
        # Do not process deposit requests if Eth1 bridge deposits are not yet applied
        if (
            int(deposit.slot) > GENESIS_SLOT
            and int(state.eth1_deposit_index) < int(state.deposit_requests_start_index)
        ):
            break

        # Check if deposit has been finalized
        if int(deposit.slot) > finalized_slot:
            break

        # Check if per-epoch limit reached
        if next_deposit_index >= MAX_PENDING_DEPOSITS_PER_EPOCH:
            break

        # Read validator state
        is_validator_exited = False
        is_validator_withdrawn = False
        validator_pubkeys = [bytes(v.pubkey) for v in state.validators]
        pubkey = bytes(deposit.pubkey)

        if pubkey in validator_pubkeys:
            validator_index = validator_pubkeys.index(pubkey)
            validator = state.validators[validator_index]
            is_validator_exited = int(validator.exit_epoch) < FAR_FUTURE_EPOCH
            is_validator_withdrawn = int(validator.withdrawable_epoch) < next_epoch

        if is_validator_withdrawn:
            # Deposited balance will never become active. Increase balance but do not consume churn
            apply_pending_deposit(state, deposit)
        elif is_validator_exited:
            # Validator is exiting, postpone the deposit until after withdrawable epoch
            deposits_to_postpone.append(deposit)
        else:
            # Check if deposit fits in the churn, otherwise, do no more deposit processing
            is_churn_limit_reached = processed_amount + int(deposit.amount) > available_for_processing
            if is_churn_limit_reached:
                break

            # Consume churn and apply deposit
            processed_amount += int(deposit.amount)
            apply_pending_deposit(state, deposit)

        # Regardless of how the deposit was handled, we move on in the queue
        next_deposit_index += 1

    # Unprocessed deposits first, then postponed deposits
    state.pending_deposits = list(state.pending_deposits)[next_deposit_index:] + deposits_to_postpone

    # Accumulate churn only if the churn limit has been hit
    if is_churn_limit_reached:
        state.deposit_balance_to_consume = available_for_processing - processed_amount
    else:
        state.deposit_balance_to_consume = 0


def apply_pending_deposit(state: "BeaconState", deposit: "PendingDeposit") -> bool:
    """Apply a single pending deposit.

    Args:
        state: Beacon state (modified in place)
        deposit: Pending deposit to apply

    Returns:
        True if the deposit was valid and applied (for new validators)
    """
    validator_pubkeys = [bytes(v.pubkey) for v in state.validators]
    pubkey = bytes(deposit.pubkey)

    if pubkey not in validator_pubkeys:
        # New validator - validate signature first
        from ..block.operations.deposit import is_valid_deposit_signature

        if is_valid_deposit_signature(
            pubkey,
            bytes(deposit.withdrawal_credentials),
            int(deposit.amount),
            bytes(deposit.signature),
        ):
            _add_validator_from_deposit(state, deposit)
            return True
        return False
    else:
        # Existing validator - increase balance
        validator_index = validator_pubkeys.index(pubkey)
        increase_balance(state, validator_index, int(deposit.amount))

        # Check if this triggers compounding switch
        if (
            is_compounding_withdrawal_credential(deposit.withdrawal_credentials)
            and has_execution_withdrawal_credential(state.validators[validator_index])
            and not has_compounding_withdrawal_credential(state.validators[validator_index])
        ):
            from ..helpers.mutators import switch_to_compounding_validator

            switch_to_compounding_validator(state, validator_index)

        return False


def _add_validator_from_deposit(state: "BeaconState", deposit: "PendingDeposit") -> None:
    """Add a new validator from a pending deposit.

    Args:
        state: Beacon state (modified in place)
        deposit: Pending deposit
    """
    from ...types.phase0 import Validator

    # Determine max effective balance based on withdrawal credentials
    if is_compounding_withdrawal_credential(deposit.withdrawal_credentials):
        max_effective = MAX_EFFECTIVE_BALANCE_ELECTRA
    else:
        max_effective = MAX_EFFECTIVE_BALANCE

    # Calculate effective balance
    effective_balance = min(
        int(deposit.amount) - int(deposit.amount) % EFFECTIVE_BALANCE_INCREMENT,
        max_effective,
    )

    validator = Validator(
        pubkey=deposit.pubkey,
        withdrawal_credentials=deposit.withdrawal_credentials,
        effective_balance=effective_balance,
        slashed=False,
        activation_eligibility_epoch=FAR_FUTURE_EPOCH,
        activation_epoch=FAR_FUTURE_EPOCH,
        exit_epoch=FAR_FUTURE_EPOCH,
        withdrawable_epoch=FAR_FUTURE_EPOCH,
    )

    state.validators.append(validator)
    state.balances.append(int(deposit.amount))
    state.previous_epoch_participation.append(0)
    state.current_epoch_participation.append(0)
    state.inactivity_scores.append(0)
