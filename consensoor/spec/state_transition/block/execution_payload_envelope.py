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

    # Per spec we'd mutate state.latest_block_header.state_root with the
    # current state's hash so the envelope's beacon_block_root assertion
    # works. But mutating it here means the next process_slot computes
    # state_roots[N] from a state with state_root already filled in, while
    # peers (prysm) cache the pre-fill hash. Result: state_roots[N]
    # diverges immediately after the first envelope is processed.
    # Workaround: do the assertion against a *local* filled-in header copy,
    # leave state.latest_block_header.state_root untouched.
    previous_state_root = hash_tree_root(state)
    header = state.latest_block_header
    if bytes(header.state_root) == b"\x00" * 32:
        from ...types import BeaconBlockHeader
        check_header = BeaconBlockHeader(
            slot=header.slot,
            proposer_index=header.proposer_index,
            parent_root=header.parent_root,
            state_root=previous_state_root,
            body_root=header.body_root,
        )
    else:
        check_header = header
    assert envelope.beacon_block_root == hash_tree_root(check_header)
    # alpha-7 envelope schema doesn't carry an explicit `slot` field; the
    # slot is implicit via beacon_block_root → state.latest_block_header.slot.
    assert int(state.latest_block_header.slot) == int(state.slot)

    committed_bid = state.latest_execution_payload_bid
    assert int(envelope.builder_index) == int(committed_bid.builder_index)
    assert committed_bid.prev_randao == payload.prev_randao

    assert len(payload.withdrawals) == len(state.payload_expected_withdrawals)
    for i, w in enumerate(payload.withdrawals):
        assert hash_tree_root(w) == hash_tree_root(state.payload_expected_withdrawals[i])

    assert committed_bid.gas_limit == payload.gas_limit
    assert committed_bid.block_hash == payload.block_hash
    assert payload.parent_hash == state.latest_block_hash

    network_config = get_config()
    expected_timestamp = compute_time_at_slot(
        int(state.genesis_time), int(state.slot), network_config.slot_duration_ms
    )
    assert int(payload.timestamp) == expected_timestamp

    if execution_engine is not None:
        request = NewPayloadRequest(
            execution_payload=payload,
            versioned_hashes=[],
            parent_beacon_block_root=state.latest_block_header.parent_root,
            execution_requests=envelope.execution_requests,
        )
        assert execution_engine.verify_and_notify_new_payload(request)

    # Per lighthouse + alpha-7: envelope verify performs PURE VERIFICATION.
    # All state mutations (execution_requests processing,
    # builder_pending_payments rotation, latest_block_hash,
    # execution_payload_availability) are deferred to the NEXT block's
    # process_parent_execution_payload. If we mutate state here, our state
    # at slot N has bit N=1 + latest_block_hash=payload.block_hash, while
    # lighthouse's has bit N=0 + latest_block_hash=parent's value (until
    # the next block triggers parent_execution_payload). Diverged states
    # → diverged block_roots → chain stuck.


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
