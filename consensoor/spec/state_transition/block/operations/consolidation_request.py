"""Consolidation request processing (Electra+).

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/electra/beacon-chain.md
"""

from typing import TYPE_CHECKING

from ....constants import (
    FAR_FUTURE_EPOCH,
    MIN_VALIDATOR_WITHDRAWABILITY_DELAY,
    MIN_ACTIVATION_BALANCE,
    SHARD_COMMITTEE_PERIOD,
    PENDING_CONSOLIDATIONS_LIMIT,
)
from ...helpers.predicates import (
    has_execution_withdrawal_credential,
    has_compounding_withdrawal_credential,
    is_active_validator,
    is_valid_switch_to_compounding_request,
)
from ...helpers.accessors import (
    get_current_epoch,
    get_consolidation_churn_limit,
    get_pending_balance_to_withdraw,
)
from ...helpers.mutators import (
    switch_to_compounding_validator,
    compute_consolidation_epoch_and_update_churn,
)

if TYPE_CHECKING:
    from ....types import BeaconState
    from ....types.electra import ConsolidationRequest


def process_consolidation_request(
    state: "BeaconState", consolidation_request: "ConsolidationRequest"
) -> None:
    """Process a consolidation request from the execution layer (Electra+).

    Consolidates two validators into one (source into target).

    Args:
        state: Beacon state (modified in place)
        consolidation_request: Consolidation request from execution layer
    """
    from ....types.electra import PendingConsolidation

    # Check if this is a valid switch-to-compounding request
    if is_valid_switch_to_compounding_request(state, consolidation_request):
        validator_pubkeys = [bytes(v.pubkey) for v in state.validators]
        request_source_pubkey = bytes(consolidation_request.source_pubkey)
        source_index = validator_pubkeys.index(request_source_pubkey)
        switch_to_compounding_validator(state, source_index)
        return

    # Verify that source != target, so a consolidation cannot be used as an exit
    if bytes(consolidation_request.source_pubkey) == bytes(consolidation_request.target_pubkey):
        return

    # If the pending consolidations queue is full, consolidation requests are ignored
    if len(state.pending_consolidations) == PENDING_CONSOLIDATIONS_LIMIT():
        return

    # If there is too little available consolidation churn limit, consolidation requests are ignored
    if get_consolidation_churn_limit(state) <= MIN_ACTIVATION_BALANCE:
        return

    validator_pubkeys = [bytes(v.pubkey) for v in state.validators]
    request_source_pubkey = bytes(consolidation_request.source_pubkey)
    request_target_pubkey = bytes(consolidation_request.target_pubkey)

    # Verify pubkeys exist
    if request_source_pubkey not in validator_pubkeys:
        return
    if request_target_pubkey not in validator_pubkeys:
        return

    source_index = validator_pubkeys.index(request_source_pubkey)
    target_index = validator_pubkeys.index(request_target_pubkey)
    source_validator = state.validators[source_index]
    target_validator = state.validators[target_index]

    # Verify source withdrawal credentials
    has_correct_credential = has_execution_withdrawal_credential(source_validator)
    is_correct_source_address = (
        bytes(source_validator.withdrawal_credentials)[12:] == bytes(consolidation_request.source_address)
    )
    if not (has_correct_credential and is_correct_source_address):
        return

    # Verify that target has compounding withdrawal credentials
    if not has_compounding_withdrawal_credential(target_validator):
        return

    # Verify the source and the target are active
    current_epoch = get_current_epoch(state)
    if not is_active_validator(source_validator, current_epoch):
        return
    if not is_active_validator(target_validator, current_epoch):
        return

    # Verify exits for source and target have not been initiated
    if int(source_validator.exit_epoch) != FAR_FUTURE_EPOCH:
        return
    if int(target_validator.exit_epoch) != FAR_FUTURE_EPOCH:
        return

    # Verify the source has been active long enough
    if current_epoch < int(source_validator.activation_epoch) + SHARD_COMMITTEE_PERIOD():
        return

    # Verify the source has no pending withdrawals in the queue
    if get_pending_balance_to_withdraw(state, source_index) > 0:
        return

    # Initiate source validator exit and append pending consolidation
    source_validator.exit_epoch = compute_consolidation_epoch_and_update_churn(
        state, int(source_validator.effective_balance)
    )
    source_validator.withdrawable_epoch = (
        int(source_validator.exit_epoch) + MIN_VALIDATOR_WITHDRAWABILITY_DELAY
    )
    state.pending_consolidations.append(
        PendingConsolidation(source_index=source_index, target_index=target_index)
    )
