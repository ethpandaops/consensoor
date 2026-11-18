"""PQ execution-layer requests: preregistrations and voluntary exits."""

from typing import TYPE_CHECKING

from ....constants import (
    FAR_FUTURE_EPOCH,
    PREREGISTRATION_EXPIRY_SLOTS,
    PREREGISTRATIONS_LIMIT,
    SHARD_COMMITTEE_PERIOD,
)
from ...helpers.accessors import get_current_epoch, get_pending_balance_to_withdraw
from ...helpers.mutators import initiate_validator_exit
from ...helpers.predicates import is_active_validator
from ...helpers.pq import (
    get_active_builder_preregistration,
    get_active_validator_preregistration,
    get_builder_preregistration_index,
    get_validator_preregistration_index,
    is_active_builder_preregistration,
    is_active_validator_preregistration,
    is_pending_validator,
)

if TYPE_CHECKING:
    from ....types.pq import (
        BeaconState,
        BuilderPreregistrationRequest,
        ValidatorPreregistrationRequest,
        VoluntaryExitRequest,
    )


def process_validator_preregistration_request(
    state: "BeaconState", request: "ValidatorPreregistrationRequest"
) -> None:
    from ....types.pq import ValidatorPreregistration

    pubkey = bytes(request.pubkey)

    if get_active_validator_preregistration(state, pubkey) is not None:
        return
    if pubkey in [bytes(v.pubkey) for v in state.validators]:
        return
    if is_pending_validator(state.pending_deposits, pubkey):
        return

    active_count = sum(
        1
        for preregistration in state.validator_preregistrations
        if is_active_validator_preregistration(state, preregistration)
    )
    if active_count >= PREREGISTRATIONS_LIMIT:
        return

    preregistration = ValidatorPreregistration(
        pubkey=pubkey,
        withdrawal_credentials=request.withdrawal_credentials,
        vrf_root=request.vrf_root,
        expiry_slot=int(state.slot) + PREREGISTRATION_EXPIRY_SLOTS,
    )
    index = get_validator_preregistration_index(state, pubkey)
    if index is not None:
        state.validator_preregistrations[index] = preregistration
    else:
        state.validator_preregistrations.append(preregistration)


def process_builder_preregistration_request(
    state: "BeaconState", request: "BuilderPreregistrationRequest"
) -> None:
    from ....types.pq import BuilderPreregistration

    pubkey = bytes(request.pubkey)

    if get_active_builder_preregistration(state, pubkey) is not None:
        return
    if pubkey in [bytes(b.pubkey) for b in state.builders]:
        return

    active_count = sum(
        1
        for preregistration in state.builder_preregistrations
        if is_active_builder_preregistration(state, preregistration)
    )
    if active_count >= PREREGISTRATIONS_LIMIT:
        return

    preregistration = BuilderPreregistration(
        pubkey=pubkey,
        withdrawal_credentials=request.withdrawal_credentials,
        expiry_slot=int(state.slot) + PREREGISTRATION_EXPIRY_SLOTS,
    )
    index = get_builder_preregistration_index(state, pubkey)
    if index is not None:
        state.builder_preregistrations[index] = preregistration
    else:
        state.builder_preregistrations.append(preregistration)


def process_voluntary_exit_request(state: "BeaconState", request: "VoluntaryExitRequest") -> None:
    pubkey = bytes(request.pubkey)
    validator_pubkeys = [bytes(v.pubkey) for v in state.validators]
    if pubkey not in validator_pubkeys:
        return
    index = validator_pubkeys.index(pubkey)
    validator = state.validators[index]
    current_epoch = get_current_epoch(state)

    if not is_active_validator(validator, current_epoch):
        return
    if int(validator.exit_epoch) != FAR_FUTURE_EPOCH:
        return
    if current_epoch < int(validator.activation_epoch) + SHARD_COMMITTEE_PERIOD():
        return
    if get_pending_balance_to_withdraw(state, index) != 0:
        return

    initiate_validator_exit(state, index)


def process_preregistration_expiry(state: "BeaconState") -> None:
    state.validator_preregistrations = [
        preregistration
        for preregistration in state.validator_preregistrations
        if is_active_validator_preregistration(state, preregistration)
    ]
    state.builder_preregistrations = [
        preregistration
        for preregistration in state.builder_preregistrations
        if is_active_builder_preregistration(state, preregistration)
    ]
