"""Execution payload envelope processing (Gloas/ePBS).

Reference: EIP-7732 - Enshrined Proposer-Builder Separation

In ePBS, the full execution payload is delivered separately from the
beacon block via P2P as an ExecutionPayloadEnvelope. This module handles
processing the envelope and applying it to state.
"""

from typing import TYPE_CHECKING

from ...constants import (
    DOMAIN_BEACON_BUILDER,
    BUILDER_INDEX_SELF_BUILD,
    SLOTS_PER_EPOCH,
    SLOTS_PER_HISTORICAL_ROOT,
)
from ..helpers.domain import get_domain, compute_signing_root
from ..helpers.misc import compute_time_at_slot, compute_epoch_at_slot
from ..helpers.accessors import get_current_epoch, get_blob_parameters
from ....crypto import bls_verify, hash_tree_root
from ...network_config import get_config

if TYPE_CHECKING:
    from ...types import BeaconState
    from ...types.gloas import SignedExecutionPayloadEnvelope, ExecutionPayloadEnvelope


class NewPayloadRequest:
    """Lightweight execution engine request container."""

    def __init__(self, execution_payload, versioned_hashes, parent_beacon_block_root, execution_requests):
        self.execution_payload = execution_payload
        self.versioned_hashes = versioned_hashes
        self.parent_beacon_block_root = parent_beacon_block_root
        self.execution_requests = execution_requests


def verify_execution_payload_envelope_signature(
    state: "BeaconState", signed_envelope: "SignedExecutionPayloadEnvelope"
) -> bool:
    builder_index = int(signed_envelope.message.builder_index)
    if builder_index == BUILDER_INDEX_SELF_BUILD:
        validator_index = int(state.latest_block_header.proposer_index)
        pubkey = state.validators[validator_index].pubkey
    else:
        pubkey = state.builders[builder_index].pubkey

    signing_root = compute_signing_root(
        signed_envelope.message, get_domain(state, DOMAIN_BEACON_BUILDER)
    )
    return bls_verify([bytes(pubkey)], signing_root, bytes(signed_envelope.signature))


def process_execution_payload_envelope(
    state: "BeaconState",
    signed_envelope: "SignedExecutionPayloadEnvelope",
    execution_engine=None,
    verify: bool = True,
) -> None:
    """Process an execution payload envelope (Gloas/ePBS)."""
    envelope = signed_envelope.message
    payload = envelope.payload

    if verify:
        assert verify_execution_payload_envelope_signature(state, signed_envelope)

    previous_state_root = hash_tree_root(state)
    if bytes(state.latest_block_header.state_root) == b"\x00" * 32:
        state.latest_block_header.state_root = previous_state_root

    assert envelope.beacon_block_root == hash_tree_root(state.latest_block_header)
    assert int(envelope.slot) == int(state.slot)

    committed_bid = state.latest_execution_payload_bid
    assert int(envelope.builder_index) == int(committed_bid.builder_index)
    assert committed_bid.prev_randao == payload.prev_randao

    assert hash_tree_root(payload.withdrawals) == hash_tree_root(state.payload_expected_withdrawals)

    assert committed_bid.gas_limit == payload.gas_limit
    assert committed_bid.block_hash == payload.block_hash
    assert payload.parent_hash == state.latest_block_hash

    network_config = get_config()
    expected_timestamp = compute_time_at_slot(
        int(state.genesis_time), int(state.slot), network_config.slot_duration_ms
    )
    assert int(payload.timestamp) == expected_timestamp

    epoch = compute_epoch_at_slot(int(state.slot))
    blob_params = get_blob_parameters(epoch)
    assert len(committed_bid.blob_kzg_commitments) <= blob_params.max_blobs_per_block

    if execution_engine is not None:
        request = NewPayloadRequest(
            execution_payload=payload,
            versioned_hashes=list(committed_bid.blob_kzg_commitments),
            parent_beacon_block_root=state.latest_block_header.parent_root,
            execution_requests=envelope.execution_requests,
        )
        assert execution_engine.verify_and_notify_new_payload(request)

    from ..block.operations import (
        process_deposit_request,
        process_withdrawal_request,
        process_consolidation_request,
    )

    requests = envelope.execution_requests
    for deposit_request in requests.deposits:
        process_deposit_request(state, deposit_request)
    for withdrawal_request in requests.withdrawals:
        process_withdrawal_request(state, withdrawal_request)
    for consolidation_request in requests.consolidations:
        process_consolidation_request(state, consolidation_request)

    payment = state.builder_pending_payments[
        SLOTS_PER_EPOCH() + int(state.slot) % SLOTS_PER_EPOCH()
    ]
    amount = int(payment.withdrawal.amount)
    if amount > 0:
        state.builder_pending_withdrawals.append(payment.withdrawal)
    from ...types.gloas import BuilderPendingPayment
    state.builder_pending_payments[
        SLOTS_PER_EPOCH() + int(state.slot) % SLOTS_PER_EPOCH()
    ] = BuilderPendingPayment()

    state.execution_payload_availability[int(state.slot) % SLOTS_PER_HISTORICAL_ROOT()] = 0b1
    state.latest_block_hash = payload.block_hash

    if verify:
        assert envelope.state_root == hash_tree_root(state)


def process_payload_from_envelope(
    state: "BeaconState", envelope: "ExecutionPayloadEnvelope"
) -> None:
    """Process the execution payload from an envelope.

    Applies the payload to state, including:
    - Updating latest_execution_payload_header
    - Processing execution requests (deposits, withdrawals, consolidations)
    - Updating latest_block_hash and latest_withdrawals_root

    Args:
        state: Beacon state (modified in place)
        envelope: Execution payload envelope
    """
    # Legacy helper retained for compatibility; no-op in Gloas.
    return


def process_execution_requests(state: "BeaconState", requests) -> None:
    """Process execution requests from the payload envelope.

    In ePBS, execution requests (deposits, withdrawals, consolidations)
    are processed when the full payload is revealed, not during block
    processing.

    Args:
        state: Beacon state (modified in place)
        requests: Execution requests container
    """
    from ..block.operations import (
        process_deposit_request,
        process_withdrawal_request,
        process_consolidation_request,
    )

    # Process deposit requests
    if hasattr(requests, "deposits"):
        for deposit_request in requests.deposits:
            process_deposit_request(state, deposit_request)

    # Process withdrawal requests
    if hasattr(requests, "withdrawals"):
        for withdrawal_request in requests.withdrawals:
            process_withdrawal_request(state, withdrawal_request)

    # Process consolidation requests
    if hasattr(requests, "consolidations"):
        for consolidation_request in requests.consolidations:
            process_consolidation_request(state, consolidation_request)


def verify_execution_payload_envelope(
    state: "BeaconState", signed_envelope: "SignedExecutionPayloadEnvelope"
) -> bool:
    """Verify an execution payload envelope without applying it.

    Used for initial validation before full processing.

    Args:
        state: Beacon state
        signed_envelope: Signed envelope to verify

    Returns:
        True if envelope is valid
    """
    try:
        envelope = signed_envelope.message

        if int(envelope.slot) != int(state.slot):
            return False
        expected_block_root = hash_tree_root(state.latest_block_header)
        if bytes(envelope.beacon_block_root) != expected_block_root:
            return False
        return verify_execution_payload_envelope_signature(state, signed_envelope)

    except Exception:
        return False
