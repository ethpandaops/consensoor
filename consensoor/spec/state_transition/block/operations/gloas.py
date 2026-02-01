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

    assert int(bid.slot) == expected_slot, "Bid slot mismatch"
    assert bytes(bid.parent_block_hash) == bytes(state.latest_block_hash), "Parent block hash mismatch"
    assert bytes(bid.parent_block_root) == expected_parent_root, "Parent block root mismatch"
    assert bytes(bid.prev_randao) == bytes(get_randao_mix(state, get_current_epoch(state))), (
        "Prev randao mismatch"
    )

    epoch = compute_epoch_at_slot(int(bid.slot))
    blob_params = get_blob_parameters(epoch)
    assert len(bid.blob_kzg_commitments) <= blob_params.max_blobs_per_block, (
        f"Too many blob commitments: {len(bid.blob_kzg_commitments)} > {blob_params.max_blobs_per_block}"
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
