"""Payload attestations processing (Gloas/ePBS).

Reference: EIP-7732 - Enshrined Proposer-Builder Separation

Payload attestations are produced by the Payload Timeliness Committee (PTC)
to attest to the timely revelation of execution payloads.
"""

from typing import TYPE_CHECKING

from ...constants import (
    MAX_PAYLOAD_ATTESTATIONS,
)
from ..helpers.ptc import get_indexed_payload_attestation
from ..helpers.predicates import is_valid_indexed_payload_attestation

if TYPE_CHECKING:
    from ...types import BeaconState
    from ...types.gloas import PayloadAttestation


def process_payload_attestations(
    state: "BeaconState", payload_attestations: list
) -> None:
    """Process payload attestations from the block.

    PTC members attest to whether the execution payload was revealed timely
    and whether blob data was available.

    Args:
        state: Beacon state (modified in place)
        payload_attestations: List of payload attestations

    Raises:
        AssertionError: If validation fails
    """
    # Verify we don't exceed the limit
    assert len(payload_attestations) <= MAX_PAYLOAD_ATTESTATIONS, (
        f"Too many payload attestations: {len(payload_attestations)}"
    )

    for attestation in payload_attestations:
        process_payload_attestation(state, attestation)


def process_payload_attestation(
    state: "BeaconState", attestation: "PayloadAttestation"
) -> None:
    """Process a single payload attestation.

    Validates the attestation and updates state tracking for payload
    timeliness.

    Args:
        state: Beacon state (modified in place)
        attestation: Payload attestation to process

    Raises:
        AssertionError: If validation fails
    """
    data = attestation.data

    # Check that the attestation is for the parent beacon block
    assert data.beacon_block_root == state.latest_block_header.parent_root
    # Check that the attestation is for the previous slot
    assert int(data.slot) + 1 == int(state.slot)
    # Verify signature
    indexed_payload_attestation = get_indexed_payload_attestation(state, attestation)
    assert is_valid_indexed_payload_attestation(state, indexed_payload_attestation)


def is_valid_payload_attestation(
    state: "BeaconState", attestation: "PayloadAttestation"
) -> bool:
    """Check if a payload attestation is valid.

    Args:
        state: Beacon state
        attestation: Payload attestation to validate

    Returns:
        True if attestation is valid
    """
    try:
        data = attestation.data

        if data.beacon_block_root != state.latest_block_header.parent_root:
            return False
        if int(data.slot) + 1 != int(state.slot):
            return False

        indexed_payload_attestation = get_indexed_payload_attestation(state, attestation)
        return is_valid_indexed_payload_attestation(state, indexed_payload_attestation)

    except Exception:
        return False
