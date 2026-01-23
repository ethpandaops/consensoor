"""Withdrawal request processing (Electra+).

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/electra/beacon-chain.md
"""

from typing import TYPE_CHECKING

from ....constants import (
    FAR_FUTURE_EPOCH,
    MIN_VALIDATOR_WITHDRAWABILITY_DELAY,
    MIN_ACTIVATION_BALANCE,
    FULL_EXIT_REQUEST_AMOUNT,
    PENDING_PARTIAL_WITHDRAWALS_LIMIT,
    SHARD_COMMITTEE_PERIOD,
)
from ...helpers.predicates import (
    has_execution_withdrawal_credential,
    has_compounding_withdrawal_credential,
    is_active_validator,
)
from ...helpers.accessors import (
    get_current_epoch,
    get_pending_balance_to_withdraw,
)
from ...helpers.mutators import (
    initiate_validator_exit,
    compute_exit_epoch_and_update_churn,
)

if TYPE_CHECKING:
    from ....types import BeaconState
    from ....types.electra import WithdrawalRequest


def process_withdrawal_request(
    state: "BeaconState", withdrawal_request: "WithdrawalRequest"
) -> None:
    """Process a withdrawal request from the execution layer (Electra+).

    Handles full exits or partial withdrawals based on the requested amount.

    Args:
        state: Beacon state (modified in place)
        withdrawal_request: Withdrawal request from execution layer
    """
    from ....types.electra import PendingPartialWithdrawal

    amount = int(withdrawal_request.amount)
    is_full_exit_request = amount == FULL_EXIT_REQUEST_AMOUNT

    # If partial withdrawal queue is full, only full exits are processed
    if (
        len(state.pending_partial_withdrawals) == PENDING_PARTIAL_WITHDRAWALS_LIMIT()
        and not is_full_exit_request
    ):
        return

    # Find validator by pubkey
    validator_pubkeys = [bytes(v.pubkey) for v in state.validators]
    request_pubkey = bytes(withdrawal_request.validator_pubkey)

    if request_pubkey not in validator_pubkeys:
        return

    validator_index = validator_pubkeys.index(request_pubkey)
    validator = state.validators[validator_index]

    # Verify withdrawal credentials
    has_correct_credential = has_execution_withdrawal_credential(validator)
    is_correct_source_address = (
        bytes(validator.withdrawal_credentials)[12:] == bytes(withdrawal_request.source_address)
    )
    if not (has_correct_credential and is_correct_source_address):
        return

    # Verify the validator is active
    current_epoch = get_current_epoch(state)
    if not is_active_validator(validator, current_epoch):
        return

    # Verify exit has not been initiated
    if int(validator.exit_epoch) != FAR_FUTURE_EPOCH:
        return

    # Verify the validator has been active long enough
    if current_epoch < int(validator.activation_epoch) + SHARD_COMMITTEE_PERIOD():
        return

    pending_balance_to_withdraw = get_pending_balance_to_withdraw(state, validator_index)

    if is_full_exit_request:
        # Only exit validator if it has no pending withdrawals in the queue
        if pending_balance_to_withdraw == 0:
            initiate_validator_exit(state, validator_index)
        return

    has_sufficient_effective_balance = int(validator.effective_balance) >= MIN_ACTIVATION_BALANCE
    has_excess_balance = (
        int(state.balances[validator_index]) > MIN_ACTIVATION_BALANCE + pending_balance_to_withdraw
    )

    # Only allow partial withdrawals with compounding withdrawal credentials
    if (
        has_compounding_withdrawal_credential(validator)
        and has_sufficient_effective_balance
        and has_excess_balance
    ):
        to_withdraw = min(
            int(state.balances[validator_index]) - MIN_ACTIVATION_BALANCE - pending_balance_to_withdraw,
            amount,
        )
        exit_queue_epoch = compute_exit_epoch_and_update_churn(state, to_withdraw)
        withdrawable_epoch = exit_queue_epoch + MIN_VALIDATOR_WITHDRAWABILITY_DELAY

        state.pending_partial_withdrawals.append(
            PendingPartialWithdrawal(
                validator_index=validator_index,
                amount=to_withdraw,
                withdrawable_epoch=withdrawable_epoch,
            )
        )
