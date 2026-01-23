"""Attestation helper functions for state transition.

Implements attestation validation and participation tracking.
Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/altair/beacon-chain.md
"""

from typing import TYPE_CHECKING, Sequence, Set

from ...constants import (
    SLOTS_PER_EPOCH,
    TIMELY_SOURCE_FLAG_INDEX,
    TIMELY_TARGET_FLAG_INDEX,
    TIMELY_HEAD_FLAG_INDEX,
    SLOTS_PER_HISTORICAL_ROOT,
    MIN_ATTESTATION_INCLUSION_DELAY,
)
from .misc import compute_epoch_at_slot, compute_start_slot_at_epoch
from .accessors import (
    get_current_epoch,
    get_previous_epoch,
    get_block_root,
    get_block_root_at_slot,
)
from .beacon_committee import get_beacon_committee, get_committee_count_per_slot
from .predicates import is_active_validator
from .math import integer_squareroot

if TYPE_CHECKING:
    from ...types import BeaconState, Attestation, IndexedAttestation, AttestationData


def is_attestation_same_slot(state: "BeaconState", data: "AttestationData") -> bool:
    """Check if the attestation is for the block proposed at the attestation slot."""
    if int(data.slot) == 0:
        return True

    block_root = bytes(data.beacon_block_root)
    slot_block_root = get_block_root_at_slot(state, int(data.slot))
    prev_block_root = get_block_root_at_slot(state, int(data.slot) - 1)

    return block_root == slot_block_root and block_root != prev_block_root


def add_flag(flags: int, flag_index: int) -> int:
    """Return a new ParticipationFlags value with the flag_index set.

    Args:
        flags: Current participation flags
        flag_index: Flag index to set (0-7)

    Returns:
        Updated participation flags
    """
    flag = 2**flag_index
    return flags | flag


def has_flag(flags: int, flag_index: int) -> bool:
    """Check if the flag_index is set in flags.

    Args:
        flags: Participation flags
        flag_index: Flag index to check (0-7)

    Returns:
        True if flag is set
    """
    flag = 2**flag_index
    return (flags & flag) == flag


def get_committee_indices(committee_bits) -> Sequence[int]:
    """Get the committee indices from a committee_bits bitvector (Electra).

    Args:
        committee_bits: Bitvector indicating which committees are attesting

    Returns:
        Sequence of committee indices
    """
    return [i for i in range(len(committee_bits)) if committee_bits[i]]


def get_attesting_indices(
    state: "BeaconState", attestation: "Attestation"
) -> Set[int]:
    """Return the set of attesting validator indices from an attestation.

    Args:
        state: Beacon state
        attestation: Attestation

    Returns:
        Set of attesting validator indices
    """
    data = attestation.data

    # Check if this is an Electra attestation (has committee_bits)
    if hasattr(attestation, "committee_bits"):
        committee_indices = get_committee_indices(attestation.committee_bits)
        # Electra: aggregate multiple committees
        all_indices = []
        for committee_index in committee_indices:
            committee = get_beacon_committee(state, int(data.slot), committee_index)
            all_indices.extend(committee)

        # Filter by aggregation bits
        aggregation_bits = attestation.aggregation_bits
        return set(
            index
            for i, index in enumerate(all_indices)
            if i < len(aggregation_bits) and aggregation_bits[i]
        )
    else:
        # Pre-Electra: single committee
        committee = get_beacon_committee(state, int(data.slot), int(data.index))
        aggregation_bits = attestation.aggregation_bits
        return set(
            index
            for i, index in enumerate(committee)
            if i < len(aggregation_bits) and aggregation_bits[i]
        )


def get_indexed_attestation(
    state: "BeaconState", attestation: "Attestation"
) -> "IndexedAttestation":
    """Convert an attestation to an indexed attestation.

    Args:
        state: Beacon state
        attestation: Attestation

    Returns:
        IndexedAttestation with sorted indices
    """
    attesting_indices = get_attesting_indices(state, attestation)

    # Use fork-appropriate IndexedAttestation type
    if hasattr(attestation, "committee_bits"):
        # Electra+ attestations have committee_bits, use larger IndexedAttestation
        from ...types.electra import IndexedAttestation
    else:
        # Phase0 through Deneb use the same IndexedAttestation
        from ...types.phase0 import Phase0IndexedAttestation as IndexedAttestation

    return IndexedAttestation(
        attesting_indices=sorted(attesting_indices),
        data=attestation.data,
        signature=attestation.signature,
    )


def get_attestation_participation_flag_indices(
    state: "BeaconState",
    data: "AttestationData",
    inclusion_delay: int,
) -> Sequence[int]:
    """Return the participation flag indices to set for an attestation.

    Args:
        state: Beacon state
        data: Attestation data
        inclusion_delay: Slots between attestation and inclusion

    Returns:
        Sequence of flag indices to set
    """
    from ...constants import (
        TIMELY_SOURCE_WEIGHT,
        TIMELY_TARGET_WEIGHT,
        TIMELY_HEAD_WEIGHT,
    )

    # Check if source/target/head are correct
    justified_checkpoint = state.current_justified_checkpoint
    if int(data.target.epoch) == get_previous_epoch(state):
        justified_checkpoint = state.previous_justified_checkpoint

    # Determine correctness
    is_matching_source = data.source == justified_checkpoint

    target_root = get_block_root(state, int(data.target.epoch))
    is_matching_target = is_matching_source and bytes(data.target.root) == target_root

    head_root = get_block_root_at_slot(state, int(data.slot))
    head_root_matches = bytes(data.beacon_block_root) == head_root

    # Gloas: add payload availability constraint for head matching
    if hasattr(state, "execution_payload_availability"):
        if is_attestation_same_slot(state, data):
            assert int(data.index) == 0
            payload_matches = True
        else:
            slot_index = int(data.slot) % SLOTS_PER_HISTORICAL_ROOT()
            payload_index = int(state.execution_payload_availability[slot_index])
            payload_matches = int(data.index) == payload_index
        is_matching_head = is_matching_target and head_root_matches and payload_matches
        assert is_matching_source
    else:
        is_matching_head = is_matching_target and head_root_matches

    participation_flags = []

    # Source flag - must be timely (within sqrt(SLOTS_PER_EPOCH) of target epoch start)
    if is_matching_source and inclusion_delay <= integer_squareroot(SLOTS_PER_EPOCH()):
        participation_flags.append(TIMELY_SOURCE_FLAG_INDEX)

    # Target flag - must be within target epoch
    target_epoch = int(data.target.epoch)
    current_epoch = get_current_epoch(state)
    if is_matching_target and int(data.target.epoch) == current_epoch:
        # Target must be included within the target epoch
        if inclusion_delay <= SLOTS_PER_EPOCH():
            participation_flags.append(TIMELY_TARGET_FLAG_INDEX)
    elif is_matching_target:
        # Previous epoch target
        participation_flags.append(TIMELY_TARGET_FLAG_INDEX)

    # Head flag - must be timely (inclusion delay of MIN_ATTESTATION_INCLUSION_DELAY)
    if is_matching_head and inclusion_delay == MIN_ATTESTATION_INCLUSION_DELAY:
        participation_flags.append(TIMELY_HEAD_FLAG_INDEX)

    return participation_flags


def get_unslashed_participating_indices(
    state: "BeaconState", flag_index: int, epoch: int
) -> Set[int]:
    """Return the set of unslashed validators with a specific participation flag.

    Args:
        state: Beacon state
        flag_index: Participation flag index
        epoch: Target epoch

    Returns:
        Set of unslashed validator indices with the flag set
    """
    previous_epoch = get_previous_epoch(state)
    current_epoch = get_current_epoch(state)

    assert epoch in (previous_epoch, current_epoch)

    # Check if this is a Phase0 state (uses PendingAttestation) or Altair+ (uses participation flags)
    if hasattr(state, "previous_epoch_participation"):
        # Altair+ path: use participation flags
        if epoch == current_epoch:
            epoch_participation = state.current_epoch_participation
        else:
            epoch_participation = state.previous_epoch_participation

        active_validator_indices = [
            i for i, v in enumerate(state.validators) if is_active_validator(v, epoch)
        ]

        return set(
            i
            for i in active_validator_indices
            if has_flag(int(epoch_participation[i]), flag_index)
            and not state.validators[i].slashed
        )
    else:
        # Phase0 path: use PendingAttestation objects
        return get_unslashed_participating_indices_phase0(state, flag_index, epoch)


def get_unslashed_participating_indices_phase0(
    state: "BeaconState", flag_index: int, epoch: int
) -> Set[int]:
    """Phase0 implementation using PendingAttestation objects.

    In Phase0, we track participation via PendingAttestation objects instead of flags.
    We need to find validators who attested with correct source/target/head.
    """
    # Get matching attestations based on flag type
    if flag_index == TIMELY_TARGET_FLAG_INDEX:
        attestations = get_matching_target_attestations_phase0(state, epoch)
    elif flag_index == TIMELY_SOURCE_FLAG_INDEX:
        attestations = get_matching_source_attestations_phase0(state, epoch)
    elif flag_index == TIMELY_HEAD_FLAG_INDEX:
        attestations = get_matching_head_attestations_phase0(state, epoch)
    else:
        attestations = []

    return get_unslashed_attesting_indices_phase0(state, attestations)


def get_matching_source_attestations_phase0(state: "BeaconState", epoch: int):
    """Return attestations for the given epoch (Phase0).

    In Phase0, all attestations are already validated for correct source
    when they're included in a block.
    """
    previous_epoch = get_previous_epoch(state)
    current_epoch = get_current_epoch(state)
    assert epoch in (previous_epoch, current_epoch)

    if epoch == current_epoch:
        return list(state.current_epoch_attestations)
    else:
        return list(state.previous_epoch_attestations)


def get_matching_target_attestations_phase0(state: "BeaconState", epoch: int):
    """Return attestations with matching target root (Phase0)."""
    target_root = get_block_root(state, epoch)
    return [
        a for a in get_matching_source_attestations_phase0(state, epoch)
        if bytes(a.data.target.root) == target_root
    ]


def get_matching_head_attestations_phase0(state: "BeaconState", epoch: int):
    """Return attestations with matching head root (Phase0)."""
    return [
        a for a in get_matching_target_attestations_phase0(state, epoch)
        if bytes(a.data.beacon_block_root) == get_block_root_at_slot(state, int(a.data.slot))
    ]


def get_unslashed_attesting_indices_phase0(state: "BeaconState", attestations) -> Set[int]:
    """Return unslashed attesting indices from attestations (Phase0)."""
    output: Set[int] = set()
    for a in attestations:
        output = output.union(get_attesting_indices_from_pending_attestation(state, a))
    return set(filter(lambda index: not state.validators[index].slashed, output))


def get_attesting_indices_from_pending_attestation(state: "BeaconState", attestation) -> Set[int]:
    """Get attesting indices from a PendingAttestation (Phase0)."""
    data = attestation.data
    committee = get_beacon_committee(state, int(data.slot), int(data.index))
    aggregation_bits = attestation.aggregation_bits
    return set(
        index
        for i, index in enumerate(committee)
        if i < len(aggregation_bits) and aggregation_bits[i]
    )


def get_attesting_balance_phase0(state: "BeaconState", attestations) -> int:
    """Return combined effective balance of unslashed attesting validators (Phase0)."""
    from .accessors import get_total_balance
    return get_total_balance(state, get_unslashed_attesting_indices_phase0(state, attestations))


# Sync committee functions


def get_sync_committee_indices(state: "BeaconState", epoch: int) -> Sequence[int]:
    """Return the sync committee indices for the given epoch (Altair+).

    Uses shuffling to select validators weighted by effective balance.
    Electra+ uses 16-bit random values and MAX_EFFECTIVE_BALANCE_ELECTRA.

    Args:
        state: Beacon state
        epoch: Target epoch

    Returns:
        Sequence of validator indices for the sync committee
    """
    from ...constants import (
        SYNC_COMMITTEE_SIZE,
        EPOCHS_PER_SYNC_COMMITTEE_PERIOD,
        DOMAIN_SYNC_COMMITTEE,
        MAX_EFFECTIVE_BALANCE,
        MAX_EFFECTIVE_BALANCE_ELECTRA,
    )
    from .accessors import get_active_validator_indices, get_seed
    from .beacon_committee import compute_shuffled_index, compute_balance_weighted_selection
    from ....crypto import sha256

    is_electra = hasattr(state, "pending_deposits")

    base_epoch = (
        epoch // EPOCHS_PER_SYNC_COMMITTEE_PERIOD()
    ) * EPOCHS_PER_SYNC_COMMITTEE_PERIOD()
    active_validator_indices = get_active_validator_indices(state, base_epoch)
    active_validator_count = len(active_validator_indices)
    seed = get_seed(state, base_epoch, DOMAIN_SYNC_COMMITTEE)

    if hasattr(state, "execution_payload_availability"):
        return compute_balance_weighted_selection(
            state, active_validator_indices, seed, size=SYNC_COMMITTEE_SIZE(), shuffle_indices=True
        )

    i = 0
    sync_committee_indices = []

    if is_electra:
        max_random_value = 2**16 - 1
        max_effective_balance = MAX_EFFECTIVE_BALANCE_ELECTRA
    else:
        max_random_value = 2**8 - 1
        max_effective_balance = MAX_EFFECTIVE_BALANCE

    while len(sync_committee_indices) < SYNC_COMMITTEE_SIZE():
        shuffled_index = compute_shuffled_index(
            i % active_validator_count, active_validator_count, seed
        )
        candidate_index = active_validator_indices[shuffled_index]

        if is_electra:
            random_bytes = sha256(seed + (i // 16).to_bytes(8, "little"))
            offset = (i % 16) * 2
            random_value = int.from_bytes(random_bytes[offset:offset + 2], "little")
        else:
            random_value = sha256(seed + (i // 32).to_bytes(8, "little"))[i % 32]

        effective_balance = int(state.validators[candidate_index].effective_balance)

        if effective_balance * max_random_value >= max_effective_balance * random_value:
            sync_committee_indices.append(candidate_index)

        i += 1

    return sync_committee_indices


def get_next_sync_committee(state: "BeaconState"):
    """Return the next sync committee for the state's epoch (Altair).

    Args:
        state: Beacon state

    Returns:
        SyncCommittee with pubkeys and aggregate pubkey
    """
    from ...types.altair import SyncCommittee
    from ...constants import EPOCHS_PER_SYNC_COMMITTEE_PERIOD
    from ....crypto import bls_aggregate_pubkeys

    indices = get_sync_committee_indices(
        state, get_current_epoch(state) + EPOCHS_PER_SYNC_COMMITTEE_PERIOD()
    )
    pubkeys = [state.validators[i].pubkey for i in indices]
    aggregate_pubkey = bls_aggregate_pubkeys(pubkeys)

    return SyncCommittee(
        pubkeys=pubkeys,
        aggregate_pubkey=aggregate_pubkey,
    )
