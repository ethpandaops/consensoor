"""Deposit request processing (Electra+).

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/electra/beacon-chain.md
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ....types import BeaconState
    from ....types.electra import DepositRequest


def process_deposit_request(
    state: "BeaconState", deposit_request: "DepositRequest"
) -> None:
    """Process a deposit request from the execution layer (Electra+).

    Queues the deposit for processing via pending_deposits.

    Args:
        state: Beacon state (modified in place)
        deposit_request: Deposit request from execution layer
    """
    from ....types.electra import PendingDeposit
    from ....types.gloas import Builder
    from ....constants import UNSET_DEPOSIT_REQUESTS_START_INDEX, FAR_FUTURE_EPOCH
    from ...helpers.predicates import is_builder_withdrawal_credential
    from ...helpers.accessors import get_current_epoch
    from .deposit import is_valid_deposit_signature

    def get_index_for_new_builder() -> int:
        current_epoch = get_current_epoch(state)
        for index, builder in enumerate(state.builders):
            if int(builder.withdrawable_epoch) <= current_epoch and int(builder.balance) == 0:
                return index
        return len(state.builders)

    def get_builder_from_deposit(pubkey, withdrawal_credentials, amount) -> Builder:
        return Builder(
            pubkey=pubkey,
            version=withdrawal_credentials[0],
            execution_address=withdrawal_credentials[12:],
            balance=amount,
            deposit_epoch=get_current_epoch(state),
            withdrawable_epoch=FAR_FUTURE_EPOCH,
        )

    def add_builder_to_registry(pubkey, withdrawal_credentials, amount) -> None:
        index = get_index_for_new_builder()
        builder = get_builder_from_deposit(pubkey, withdrawal_credentials, amount)
        if index < len(state.builders):
            state.builders[index] = builder
        else:
            state.builders.append(builder)

    def apply_deposit_for_builder(pubkey, withdrawal_credentials, amount, signature) -> None:
        builder_pubkeys = [b.pubkey for b in state.builders]
        if pubkey not in builder_pubkeys:
            if is_valid_deposit_signature(
                bytes(pubkey),
                bytes(withdrawal_credentials),
                int(amount),
                bytes(signature),
            ):
                add_builder_to_registry(pubkey, withdrawal_credentials, amount)
        else:
            builder_index = builder_pubkeys.index(pubkey)
            state.builders[builder_index].balance = (
                int(state.builders[builder_index].balance) + int(amount)
            )

    # Set deposit request start index on first deposit request (pre-Gloas)
    if (
        not hasattr(state, "builders")
        and int(state.deposit_requests_start_index) == UNSET_DEPOSIT_REQUESTS_START_INDEX
    ):
        state.deposit_requests_start_index = int(deposit_request.index)

    builder_pubkeys = [b.pubkey for b in state.builders]
    validator_pubkeys = [v.pubkey for v in state.validators]
    is_builder = deposit_request.pubkey in builder_pubkeys
    is_validator = deposit_request.pubkey in validator_pubkeys
    is_builder_prefix = is_builder_withdrawal_credential(bytes(deposit_request.withdrawal_credentials))

    if is_builder or (is_builder_prefix and not is_validator):
        apply_deposit_for_builder(
            deposit_request.pubkey,
            deposit_request.withdrawal_credentials,
            deposit_request.amount,
            deposit_request.signature,
        )
        return

    state.pending_deposits.append(
        PendingDeposit(
            pubkey=deposit_request.pubkey,
            withdrawal_credentials=deposit_request.withdrawal_credentials,
            amount=int(deposit_request.amount),
            signature=deposit_request.signature,
            slot=int(state.slot),
        )
    )
