"""Beacon committee helper functions for state transition.

Implements shuffling, committee selection, and proposer selection.
Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/phase0/beacon-chain.md
"""

from typing import TYPE_CHECKING, Sequence

from ...constants import (
    SLOTS_PER_EPOCH,
    SHUFFLE_ROUND_COUNT,
    TARGET_COMMITTEE_SIZE,
    MAX_COMMITTEES_PER_SLOT,
    DOMAIN_BEACON_PROPOSER,
    DOMAIN_BEACON_ATTESTER,
    MAX_EFFECTIVE_BALANCE,
    EFFECTIVE_BALANCE_INCREMENT,
    MAX_EFFECTIVE_BALANCE_ELECTRA,
)
from .misc import compute_epoch_at_slot
from .accessors import get_current_epoch, get_active_validator_indices, get_seed
from ....crypto import sha256

if TYPE_CHECKING:
    from ...types import BeaconState


def compute_shuffled_index(index: int, index_count: int, seed: bytes) -> int:
    """Return the shuffled index corresponding to seed (swap-or-not shuffle).

    This implements the Fisher-Yates shuffle variant used in Ethereum.

    Args:
        index: Index to shuffle
        index_count: Total number of indices
        seed: 32-byte seed

    Returns:
        Shuffled index

    Raises:
        AssertionError: If index >= index_count
    """
    assert index < index_count

    for current_round in range(SHUFFLE_ROUND_COUNT()):
        pivot = (
            int.from_bytes(
                sha256(seed + current_round.to_bytes(1, "little"))[:8], "little"
            )
            % index_count
        )
        flip = (pivot + index_count - index) % index_count
        position = max(index, flip)
        source = sha256(
            seed
            + current_round.to_bytes(1, "little")
            + (position // 256).to_bytes(4, "little")
        )
        byte = source[(position % 256) // 8]
        bit = (byte >> (position % 8)) & 1
        index = flip if bit else index

    return index


def compute_proposer_index(
    state: "BeaconState", indices: Sequence[int], seed: bytes
) -> int:
    """Return the proposer index from among the given indices.

    Uses a weighted random selection based on effective balance.

    Note: Electra (EIP-7251) modifies this function to use 2-byte random values
    and MAX_EFFECTIVE_BALANCE_ELECTRA for improved precision with higher balances.

    Args:
        state: Beacon state
        indices: Sequence of validator indices to choose from
        seed: 32-byte seed

    Returns:
        Selected proposer index

    Raises:
        AssertionError: If indices is empty
    """
    assert len(indices) > 0

    # Detect Electra+ by checking for pending_deposits field
    is_electra = hasattr(state, "pending_deposits")

    i = 0
    total = len(indices)

    if is_electra:
        # Electra algorithm: 2-byte random values, MAX_EFFECTIVE_BALANCE_ELECTRA
        from ...constants import MAX_EFFECTIVE_BALANCE_ELECTRA
        max_random_value = 2**16 - 1
        while True:
            candidate_index = indices[compute_shuffled_index(i % total, total, seed)]
            random_bytes = sha256(seed + (i // 16).to_bytes(8, "little"))
            offset = (i % 16) * 2
            random_value = int.from_bytes(random_bytes[offset:offset + 2], "little")
            effective_balance = int(state.validators[candidate_index].effective_balance)
            if effective_balance * max_random_value >= MAX_EFFECTIVE_BALANCE_ELECTRA * random_value:
                return candidate_index
            i += 1
    else:
        # Pre-Electra algorithm: 1-byte random values, MAX_EFFECTIVE_BALANCE
        max_random_byte = 2**8 - 1
        while True:
            candidate_index = indices[compute_shuffled_index(i % total, total, seed)]
            random_byte = sha256(seed + (i // 32).to_bytes(8, "little"))[i % 32]
            effective_balance = int(state.validators[candidate_index].effective_balance)
            if effective_balance * max_random_byte >= MAX_EFFECTIVE_BALANCE * random_byte:
                return candidate_index
            i += 1


def compute_balance_weighted_acceptance(
    state: "BeaconState", index: int, seed: bytes, i: int
) -> bool:
    """Return whether to accept the selection of the validator index."""
    max_random_value = 2**16 - 1
    random_bytes = sha256(seed + (i // 16).to_bytes(8, "little"))
    offset = (i % 16) * 2
    random_value = int.from_bytes(random_bytes[offset:offset + 2], "little")
    effective_balance = int(state.validators[index].effective_balance)
    return effective_balance * max_random_value >= MAX_EFFECTIVE_BALANCE_ELECTRA * random_value


def compute_balance_weighted_selection(
    state: "BeaconState",
    indices: Sequence[int],
    seed: bytes,
    size: int,
    shuffle_indices: bool,
) -> Sequence[int]:
    """Return size indices sampled by effective balance."""
    total = len(indices)
    assert total > 0
    selected = []
    i = 0
    while len(selected) < size:
        next_index = i % total
        if shuffle_indices:
            next_index = compute_shuffled_index(next_index, total, seed)
        candidate_index = indices[next_index]
        if compute_balance_weighted_acceptance(state, candidate_index, seed, i):
            selected.append(candidate_index)
        i += 1
    return selected


def compute_committee(
    indices: Sequence[int], seed: bytes, index: int, count: int
) -> Sequence[int]:
    """Return the committee at index out of count committees using seed.

    Args:
        indices: Full list of validator indices
        seed: 32-byte seed
        index: Committee index
        count: Total number of committees

    Returns:
        Sequence of validator indices in the committee
    """
    start = len(indices) * index // count
    end = len(indices) * (index + 1) // count
    return [
        indices[compute_shuffled_index(i, len(indices), seed)]
        for i in range(start, end)
    ]


def get_committee_count_per_slot(state: "BeaconState", epoch: int) -> int:
    """Return the number of committees per slot for the given epoch.

    Args:
        state: Beacon state
        epoch: Target epoch

    Returns:
        Number of committees per slot (1 to MAX_COMMITTEES_PER_SLOT)
    """
    active_validators = get_active_validator_indices(state, epoch)
    return max(
        1,
        min(
            MAX_COMMITTEES_PER_SLOT(),
            len(active_validators) // SLOTS_PER_EPOCH() // TARGET_COMMITTEE_SIZE(),
        ),
    )


def get_beacon_committee(
    state: "BeaconState", slot: int, index: int
) -> Sequence[int]:
    """Return the beacon committee at slot for index.

    Args:
        state: Beacon state
        slot: Slot number
        index: Committee index

    Returns:
        Sequence of validator indices in the committee
    """
    epoch = compute_epoch_at_slot(slot)
    committees_per_slot = get_committee_count_per_slot(state, epoch)
    indices = get_active_validator_indices(state, epoch)
    seed = get_seed(state, epoch, DOMAIN_BEACON_ATTESTER)

    committee_index = (slot % SLOTS_PER_EPOCH()) * committees_per_slot + index
    committee_count = committees_per_slot * SLOTS_PER_EPOCH()

    return compute_committee(indices, seed, committee_index, committee_count)


def get_beacon_proposer_index(state: "BeaconState") -> int:
    """Return the current beacon proposer index.

    For Fulu+, uses the proposer_lookahead vector.
    For earlier forks, computes on demand.

    Args:
        state: Beacon state

    Returns:
        Proposer validator index
    """
    # Check if we have Fulu proposer_lookahead
    if hasattr(state, "proposer_lookahead") and len(state.proposer_lookahead) > 0:
        from ...constants import MIN_SEED_LOOKAHEAD

        slot = int(state.slot)
        epoch = compute_epoch_at_slot(slot)
        slot_in_epoch = slot % SLOTS_PER_EPOCH()

        # In Fulu, the lookahead is populated at the start of the epoch
        # Index 0 is the first slot of the current epoch
        lookahead_index = slot_in_epoch
        if 0 <= lookahead_index < len(state.proposer_lookahead):
            return int(state.proposer_lookahead[lookahead_index])

    # Fall back to computing on demand
    return _compute_proposer_index_on_demand(state)


def _compute_proposer_index_on_demand(state: "BeaconState") -> int:
    """Compute the proposer index on demand (pre-Fulu behavior).

    Args:
        state: Beacon state

    Returns:
        Proposer validator index
    """
    epoch = get_current_epoch(state)
    slot = int(state.slot)
    seed = sha256(
        get_seed(state, epoch, DOMAIN_BEACON_PROPOSER)
        + slot.to_bytes(8, "little")
    )
    indices = get_active_validator_indices(state, epoch)
    return compute_proposer_index(state, indices, seed)


# Fulu proposer lookahead functions


def compute_proposer_indices(
    state: "BeaconState", epoch: int, seed: bytes, indices: Sequence[int]
) -> Sequence[int]:
    """Compute proposer indices for all slots in an epoch (Fulu).

    Args:
        state: Beacon state
        epoch: Target epoch
        seed: Base seed for epoch
        indices: Active validator indices

    Returns:
        Sequence of proposer indices for each slot in the epoch
    """
    proposers = []
    for slot_offset in range(SLOTS_PER_EPOCH()):
        slot = epoch * SLOTS_PER_EPOCH() + slot_offset
        slot_seed = sha256(seed + slot.to_bytes(8, "little"))
        if hasattr(state, "execution_payload_availability"):
            proposer = compute_balance_weighted_selection(
                state, indices, slot_seed, size=1, shuffle_indices=True
            )[0]
        else:
            proposer = compute_proposer_index(state, indices, slot_seed)
        proposers.append(proposer)
    return proposers


def get_beacon_proposer_indices(state: "BeaconState", epoch: int) -> Sequence[int]:
    """Get proposer indices for all slots in an epoch (Fulu).

    Args:
        state: Beacon state
        epoch: Target epoch

    Returns:
        Sequence of proposer indices for each slot in the epoch
    """
    indices = get_active_validator_indices(state, epoch)
    seed = get_seed(state, epoch, DOMAIN_BEACON_PROPOSER)
    return compute_proposer_indices(state, epoch, seed, indices)
