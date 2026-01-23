"""Payload Timeliness Committee (PTC) helper functions for Gloas/ePBS.

Reference: EIP-7732 - Enshrined Proposer-Builder Separation
"""

from typing import TYPE_CHECKING, Sequence

from ...constants import (
    PTC_SIZE,
    SLOTS_PER_EPOCH,
    DOMAIN_PTC_ATTESTER,
)
from .accessors import get_current_epoch, get_seed
from .beacon_committee import (
    compute_balance_weighted_selection,
    get_committee_count_per_slot,
    get_beacon_committee,
)
from .misc import compute_epoch_at_slot
from ....crypto import sha256

if TYPE_CHECKING:
    from ...types import BeaconState


def get_ptc(state: "BeaconState", slot: int) -> Sequence[int]:
    """Get the Payload Timeliness Committee for a given slot.

    The PTC is responsible for attesting to timely payload revelation.
    Members are selected from active validators using shuffling.

    Args:
        state: Beacon state
        slot: Target slot

    Returns:
        Sequence of validator indices in the PTC
    """
    epoch = compute_epoch_at_slot(slot)
    seed = sha256(get_seed(state, epoch, DOMAIN_PTC_ATTESTER) + slot.to_bytes(8, "little"))

    indices = []
    committees_per_slot = get_committee_count_per_slot(state, epoch)
    for i in range(committees_per_slot):
        committee = get_beacon_committee(state, slot, i)
        indices.extend(committee)

    return compute_balance_weighted_selection(
        state, indices, seed, size=PTC_SIZE(), shuffle_indices=False
    )


def get_ptc_slot(state: "BeaconState", validator_index: int) -> int:
    """Get the slot at which a validator is in the PTC.

    Args:
        state: Beacon state
        validator_index: Validator index to check

    Returns:
        Slot number or -1 if not in PTC for current epoch
    """
    epoch = get_current_epoch(state)
    start_slot = epoch * SLOTS_PER_EPOCH()

    for slot in range(start_slot, start_slot + SLOTS_PER_EPOCH()):
        ptc = get_ptc(state, slot)
        if validator_index in ptc:
            return slot

    return -1


def is_ptc_member(state: "BeaconState", slot: int, validator_index: int) -> bool:
    """Check if a validator is a PTC member for the given slot.

    Args:
        state: Beacon state
        slot: Target slot
        validator_index: Validator index to check

    Returns:
        True if validator is in the PTC for this slot
    """
    ptc = get_ptc(state, slot)
    return validator_index in ptc


def get_indexed_payload_attestation(
    state: "BeaconState", payload_attestation
) -> "IndexedPayloadAttestation":
    """Return the indexed payload attestation for payload_attestation."""
    from ...types.gloas import IndexedPayloadAttestation

    slot = int(payload_attestation.data.slot)
    ptc = get_ptc(state, slot)
    bits = payload_attestation.aggregation_bits
    attesting_indices = [index for i, index in enumerate(ptc) if bits[i]]

    return IndexedPayloadAttestation(
        attesting_indices=sorted(attesting_indices),
        data=payload_attestation.data,
        signature=payload_attestation.signature,
    )
