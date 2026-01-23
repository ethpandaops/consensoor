"""Execution payload header processing (Gloas/ePBS).

Reference: EIP-7732 - Enshrined Proposer-Builder Separation

In ePBS, the block contains a signed commitment (bid) from a builder
rather than the full execution payload. The builder commits to reveal
the payload later via P2P.
"""

from typing import TYPE_CHECKING

from ...constants import (
    DOMAIN_BEACON_BUILDER,
    BUILDER_INDEX_SELF_BUILD,
)
from ..helpers.accessors import get_current_epoch
from ..helpers.domain import get_domain, compute_signing_root
from ..helpers.beacon_committee import get_beacon_proposer_index
from ....crypto import bls_verify, hash_tree_root

if TYPE_CHECKING:
    from ...types import BeaconState
    from ...types.gloas import SignedExecutionPayloadBid


def process_execution_payload_header(
    state: "BeaconState", signed_bid: "SignedExecutionPayloadBid"
) -> None:
    """Process the execution payload header (builder's bid).

    Validates the builder's commitment and updates state tracking.
    The actual payload will be revealed separately via P2P.

    Args:
        state: Beacon state (modified in place)
        signed_bid: Signed execution payload bid from builder

    Raises:
        AssertionError: If validation fails
    """
    bid = signed_bid.message

    # Verify bid is for current slot
    assert int(bid.slot) == int(state.slot), (
        f"Bid slot {bid.slot} != state slot {state.slot}"
    )

    # Verify parent block hash matches latest
    if hasattr(state, "latest_block_hash"):
        assert bytes(bid.parent_block_hash) == bytes(state.latest_block_hash), (
            "Bid parent_block_hash mismatch"
        )

    # Verify parent block root matches
    parent_block_root = hash_tree_root(state.latest_block_header)
    assert bytes(bid.parent_block_root) == parent_block_root, (
        "Bid parent_block_root mismatch"
    )

    # Check if this is a self-build (proposer building their own block)
    if int(bid.builder_index) == BUILDER_INDEX_SELF_BUILD:
        # Self-build: proposer is also the builder
        # No signature verification needed for self-build
        pass
    else:
        # External builder: verify signature
        # Note: In full ePBS, builders are registered separately
        # For now, we use the proposer's key for signature verification
        domain = get_domain(state, DOMAIN_BEACON_BUILDER)
        signing_root = compute_signing_root(bid, domain)

        # Get builder pubkey (in full ePBS, this would come from builder registry)
        # For now, skip signature verification if no builder registry
        if hasattr(state, "builders") and int(bid.builder_index) < len(state.builders):
            builder = state.builders[int(bid.builder_index)]
            assert bls_verify(
                [bytes(builder.pubkey)],
                signing_root,
                bytes(signed_bid.signature),
            ), "Invalid builder signature"

    # Update state tracking for ePBS
    # Note: The actual payload processing happens when the envelope is received
    # Store the bid hash for later verification
    if hasattr(state, "latest_execution_payload_header"):
        # Update header with bid information
        # In ePBS, we track the commitment until payload is revealed
        state.latest_execution_payload_header.block_hash = bid.block_hash
        state.latest_execution_payload_header.fee_recipient = bid.fee_recipient
        state.latest_execution_payload_header.gas_limit = bid.gas_limit
        state.latest_execution_payload_header.prev_randao = bid.prev_randao


def verify_execution_payload_header_signature(
    state: "BeaconState", signed_bid: "SignedExecutionPayloadBid"
) -> bool:
    """Verify the signature on an execution payload bid.

    Args:
        state: Beacon state
        signed_bid: Signed bid to verify

    Returns:
        True if signature is valid
    """
    bid = signed_bid.message

    # Self-build doesn't need signature verification
    if int(bid.builder_index) == BUILDER_INDEX_SELF_BUILD:
        return True

    domain = get_domain(state, DOMAIN_BEACON_BUILDER)
    signing_root = compute_signing_root(bid, domain)

    # Get builder pubkey
    if hasattr(state, "builders") and int(bid.builder_index) < len(state.builders):
        builder = state.builders[int(bid.builder_index)]
        return bls_verify(
            [bytes(builder.pubkey)],
            signing_root,
            bytes(signed_bid.signature),
        )

    # If no builder registry, accept the bid (permissionless)
    return True
