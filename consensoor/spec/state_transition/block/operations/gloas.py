"""Gloas (ePBS) specific operations.

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/gloas/beacon-chain.md
"""

from typing import TYPE_CHECKING

from ....constants import (
    BUILDER_INDEX_SELF_BUILD,
    DOMAIN_BEACON_BUILDER,
)
from ...helpers.accessors import (
    get_current_epoch,
    get_randao_mix,
    can_builder_cover_bid,
    get_blob_parameters,
)
from ...helpers.misc import compute_epoch_at_slot
from ...helpers.predicates import is_active_builder, is_valid_indexed_payload_attestation
from ...helpers.domain import compute_signing_root, compute_domain
from ...helpers.ptc import get_indexed_payload_attestation
from .....crypto import bls_verify

if TYPE_CHECKING:
    from ....types.gloas import BeaconState, SignedExecutionPayloadBid, PayloadAttestation


def process_execution_payload_bid(state: "BeaconState", bid_source) -> None:
    """Process an execution payload bid (ePBS).

    Args:
        state: Beacon state (modified in place)
        signed_bid: Signed execution payload bid

    Raises:
        AssertionError: If validation fails
    """
    if hasattr(bid_source, "body") and hasattr(bid_source.body, "signed_execution_payload_bid"):
        block = bid_source
        signed_bid = block.body.signed_execution_payload_bid
        expected_slot = int(block.slot)
        expected_parent_root = bytes(block.parent_root)
    else:
        block = None
        signed_bid = bid_source
        expected_slot = int(state.slot)
        from .....crypto import hash_tree_root
        expected_parent_root = hash_tree_root(state.latest_block_header)

    bid = signed_bid.message
    builder_index = int(bid.builder_index)
    amount = int(bid.value)

    if builder_index == BUILDER_INDEX_SELF_BUILD:
        assert amount == 0, "Self-build bid must have zero value"
        g2_point_at_infinity = b"\xc0" + b"\x00" * 95
        assert bytes(signed_bid.signature) == g2_point_at_infinity, (
            "Self-build bid must use point-at-infinity signature"
        )
    else:
        assert is_active_builder(state, builder_index), "Builder is not active"
        assert can_builder_cover_bid(state, builder_index, amount), "Builder cannot cover bid"
        domain = compute_domain(
            DOMAIN_BEACON_BUILDER, state.fork.current_version, state.genesis_validators_root
        )
        signing_root = compute_signing_root(bid, domain)
        builder = state.builders[builder_index]
        assert bls_verify(
            [bytes(builder.pubkey)],
            signing_root,
            bytes(signed_bid.signature),
        ), "Invalid execution payload bid signature"

    blob_params = get_blob_parameters(get_current_epoch(state))
    assert len(bid.blob_kzg_commitments) <= blob_params.max_blobs_per_block, (
        "blob_kzg_commitments exceeds limit"
    )

    assert int(bid.slot) == expected_slot, "Bid slot mismatch"
    assert bytes(bid.parent_block_hash) == bytes(state.latest_block_hash), "Parent block hash mismatch"
    assert bytes(bid.parent_block_root) == expected_parent_root, "Parent block root mismatch"
    assert bytes(bid.prev_randao) == bytes(get_randao_mix(state, get_current_epoch(state))), (
        "Prev randao mismatch"
    )

    if amount > 0:
        from ....types.gloas import BuilderPendingPayment, BuilderPendingWithdrawal
        from ....constants import SLOTS_PER_EPOCH

        pending_payment = BuilderPendingPayment(
            weight=0,
            withdrawal=BuilderPendingWithdrawal(
                fee_recipient=bid.fee_recipient,
                amount=amount,
                builder_index=bid.builder_index,
            ),
        )
        state.builder_pending_payments[
            SLOTS_PER_EPOCH() + int(bid.slot) % SLOTS_PER_EPOCH()
        ] = pending_payment

    state.latest_execution_payload_bid = bid


def settle_builder_payment(state: "BeaconState", payment_index: int) -> None:
    """Settle a builder pending payment.

    If amount > 0, append the withdrawal to builder_pending_withdrawals,
    then clear the slot.
    """
    from ....types.gloas import BuilderPendingPayment

    payment = state.builder_pending_payments[payment_index]
    if int(payment.withdrawal.amount) > 0:
        state.builder_pending_withdrawals.append(payment.withdrawal)
    state.builder_pending_payments[payment_index] = BuilderPendingPayment()


def apply_parent_execution_payload(state: "BeaconState", requests) -> None:
    """Apply the parent block's execution payload (Gloas EIP-7732)."""
    from ....constants import SLOTS_PER_EPOCH, SLOTS_PER_HISTORICAL_ROOT
    from ....types.gloas import BuilderPendingWithdrawal
    from .deposit_request import process_deposit_request
    from .withdrawal_request import process_withdrawal_request
    from .consolidation_request import process_consolidation_request

    parent_bid = state.latest_execution_payload_bid
    parent_slot = int(parent_bid.slot)
    parent_epoch = compute_epoch_at_slot(parent_slot)

    for op in requests.deposits:
        process_deposit_request(state, op)
    for op in requests.withdrawals:
        process_withdrawal_request(state, op)
    for op in requests.consolidations:
        process_consolidation_request(state, op)

    current_epoch = get_current_epoch(state)
    previous_epoch = current_epoch - 1 if current_epoch > 0 else 0
    if parent_epoch == current_epoch:
        payment_index = SLOTS_PER_EPOCH() + parent_slot % SLOTS_PER_EPOCH()
        settle_builder_payment(state, payment_index)
    elif parent_epoch == previous_epoch:
        payment_index = parent_slot % SLOTS_PER_EPOCH()
        settle_builder_payment(state, payment_index)
    elif int(parent_bid.value) > 0:
        state.builder_pending_withdrawals.append(
            BuilderPendingWithdrawal(
                fee_recipient=parent_bid.fee_recipient,
                amount=parent_bid.value,
                builder_index=parent_bid.builder_index,
            )
        )

    state.execution_payload_availability[parent_slot % SLOTS_PER_HISTORICAL_ROOT()] = True
    state.latest_block_hash = parent_bid.block_hash


def process_parent_execution_payload(state: "BeaconState", block) -> None:
    """Process the parent block's execution payload (Gloas EIP-7732).

    All envelope-derived state mutations land HERE (deferred from envelope
    receipt — see envelope handler comment). When parent's payload was
    revealed, we update latest_block_hash, set the availability bit, and
    apply the parent execution_requests.
    """
    from .....crypto import hash_tree_root
    from ....constants import SLOTS_PER_HISTORICAL_ROOT
    from ....types.electra import ExecutionRequests

    bid = block.body.signed_execution_payload_bid.message
    parent_bid = state.latest_execution_payload_bid
    requests = block.body.parent_execution_requests

    if bytes(bid.parent_block_hash) != bytes(parent_bid.block_hash):
        # Parent was EMPTY - no execution requests expected
        assert requests == ExecutionRequests(), "Parent execution requests must be empty"
        return

    # Parent was FULL - verify the bid commitment and apply the payload
    assert hash_tree_root(requests) == bytes(parent_bid.execution_requests_root), (
        "execution_requests_root mismatch"
    )
    apply_parent_execution_payload(state, requests)

    # Mark parent slot's payload as available + latch the parent's
    # block_hash as our `latest_block_hash`. These were what the envelope
    # handler used to do; we move them here so state matches lighthouse's
    # "envelope = pure verification" semantics.
    parent_slot = int(parent_bid.slot)
    state.execution_payload_availability[parent_slot % SLOTS_PER_HISTORICAL_ROOT()] = 0b1
    state.latest_block_hash = parent_bid.block_hash


def process_payload_attestation(state: "BeaconState", payload_attestation: "PayloadAttestation") -> None:
    """Process a payload attestation (ePBS).

    Args:
        state: Beacon state (modified in place)
        payload_attestation: Payload attestation from PTC

    Raises:
        AssertionError: If validation fails
    """
    data = payload_attestation.data

    assert data.beacon_block_root == state.latest_block_header.parent_root
    assert int(data.slot) + 1 == int(state.slot)

    indexed_payload_attestation = get_indexed_payload_attestation(state, payload_attestation)
    assert is_valid_indexed_payload_attestation(state, indexed_payload_attestation)
