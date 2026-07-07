"""Builder deposit/exit request processing (EIP-8282, Gloas).

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/gloas/beacon-chain.md
"""

from typing import TYPE_CHECKING

from ....constants import (
    FAR_FUTURE_EPOCH,
    DOMAIN_BUILDER_DEPOSIT,
    MIN_BUILDER_WITHDRAWABILITY_DELAY,
)
from ...helpers.accessors import (
    get_current_epoch,
    get_pending_balance_to_withdraw_for_builder,
)
from ...helpers.predicates import is_active_builder
from ...helpers.mutators import initiate_builder_exit
from ...helpers.domain import compute_domain, compute_signing_root
from ...helpers.misc import compute_epoch_at_slot
from .....crypto import bls_verify

if TYPE_CHECKING:
    from ....types import BeaconState
    from ....types.gloas import BuilderDepositRequest, BuilderExitRequest


def is_valid_builder_deposit_signature(request: "BuilderDepositRequest") -> bool:
    """Verify a builder deposit signature (proof of possession).

    Builder deposits are signed over the ``DepositMessage`` under
    ``DOMAIN_BUILDER_DEPOSIT`` so validator and builder deposit signatures
    cannot be replayed against the other deposit contract.
    """
    from ....types.phase0 import DepositMessage
    from ....network_config import get_config

    deposit_message = DepositMessage(
        pubkey=request.pubkey,
        withdrawal_credentials=request.withdrawal_credentials,
        amount=request.amount,
    )
    # compute_domain defaults to GENESIS_FORK_VERSION in the spec
    domain = compute_domain(
        DOMAIN_BUILDER_DEPOSIT, get_config().genesis_fork_version, b"\x00" * 32
    )
    signing_root = compute_signing_root(deposit_message, domain)
    return bls_verify([bytes(request.pubkey)], signing_root, bytes(request.signature))


def get_index_for_new_builder(state: "BeaconState") -> int:
    current_epoch = get_current_epoch(state)
    for index, builder in enumerate(state.builders):
        if int(builder.withdrawable_epoch) <= current_epoch and int(builder.balance) == 0:
            return index
    return len(state.builders)


def add_builder_to_registry(
    state: "BeaconState",
    pubkey: bytes,
    version: int,
    execution_address: bytes,
    amount: int,
    slot: int,
) -> None:
    from ....types.gloas import Builder

    builder = Builder(
        pubkey=pubkey,
        version=version,
        execution_address=execution_address,
        balance=amount,
        deposit_epoch=compute_epoch_at_slot(slot),
        withdrawable_epoch=FAR_FUTURE_EPOCH,
    )
    index = get_index_for_new_builder(state)
    if index < len(state.builders):
        state.builders[index] = builder
    else:
        state.builders.append(builder)


def process_builder_deposit_request(
    state: "BeaconState", request: "BuilderDepositRequest"
) -> None:
    """Process a builder deposit request (EIP-8282).

    Builder indices are reusable: an exited builder's index may later be
    reassigned to a different builder with a new public key. Deposits to an
    exited builder are withdrawn to the builder's execution address.
    """
    builder_pubkeys = [b.pubkey for b in state.builders]
    if request.pubkey not in builder_pubkeys:
        if is_valid_builder_deposit_signature(request):
            add_builder_to_registry(
                state,
                request.pubkey,
                int(bytes(request.withdrawal_credentials)[0]),
                bytes(request.withdrawal_credentials)[12:],
                int(request.amount),
                int(state.slot),
            )
    else:
        builder_index = builder_pubkeys.index(request.pubkey)
        builder = state.builders[builder_index]

        # If exited and swept, reset the withdrawable epoch
        if int(builder.withdrawable_epoch) != FAR_FUTURE_EPOCH and int(builder.balance) == 0:
            epoch = get_current_epoch(state)
            builder.withdrawable_epoch = epoch + MIN_BUILDER_WITHDRAWABILITY_DELAY()

        # Increase balance by deposit amount
        builder.balance = int(builder.balance) + int(request.amount)


def process_builder_exit_request(
    state: "BeaconState", request: "BuilderExitRequest"
) -> None:
    """Process a builder exit request (EIP-8282)."""
    builder_pubkeys = [b.pubkey for b in state.builders]
    if request.pubkey not in builder_pubkeys:
        return

    builder_index = builder_pubkeys.index(request.pubkey)
    builder = state.builders[builder_index]

    if not is_active_builder(state, builder_index):
        return
    if bytes(builder.execution_address) != bytes(request.source_address):
        return
    if get_pending_balance_to_withdraw_for_builder(state, builder_index) != 0:
        return

    initiate_builder_exit(state, builder_index)
