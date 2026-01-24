"""Validator shuffling and proposer selection algorithms."""

from ..crypto import sha256
from ..spec.constants import (
    SLOTS_PER_EPOCH,
    SHUFFLE_ROUND_COUNT,
    EPOCHS_PER_HISTORICAL_VECTOR,
    MIN_SEED_LOOKAHEAD,
    DOMAIN_BEACON_PROPOSER,
)


def _uint_to_bytes(n: int, length: int = 8) -> bytes:
    """Convert uint to little-endian bytes."""
    return n.to_bytes(length, "little")


def _bytes_to_uint64(data: bytes) -> int:
    """Convert little-endian bytes to uint64."""
    return int.from_bytes(data[:8], "little")


def compute_shuffled_index(index: int, index_count: int, seed: bytes) -> int:
    """Return the shuffled index corresponding to seed (and target index_count).

    This is the swap-or-not shuffle algorithm from the Ethereum consensus spec.
    """
    assert index < index_count

    for current_round in range(SHUFFLE_ROUND_COUNT()):
        pivot_input = seed + _uint_to_bytes(current_round, 1)
        pivot = _bytes_to_uint64(sha256(pivot_input)) % index_count
        flip = (pivot + index_count - index) % index_count
        position = max(index, flip)
        source_input = seed + _uint_to_bytes(current_round, 1) + _uint_to_bytes(position // 256, 4)
        source = sha256(source_input)
        byte = source[(position % 256) // 8]
        bit = (byte >> (position % 8)) % 2
        index = flip if bit else index

    return index


def get_active_validator_indices(state, epoch: int) -> list[int]:
    """Return the sequence of active validator indices at epoch."""
    return [
        i for i, v in enumerate(state.validators)
        if int(v.activation_epoch) <= epoch < int(v.exit_epoch)
    ]


def get_randao_mix(state, epoch: int) -> bytes:
    """Return the randao mix at epoch."""
    epochs_per_historical = EPOCHS_PER_HISTORICAL_VECTOR()
    return bytes(state.randao_mixes[epoch % epochs_per_historical])


def get_seed(state, epoch: int, domain_type: bytes) -> bytes:
    """Return the seed at epoch."""
    epochs_per_historical = EPOCHS_PER_HISTORICAL_VECTOR()
    mix_epoch = epoch + epochs_per_historical - MIN_SEED_LOOKAHEAD - 1
    mix = get_randao_mix(state, mix_epoch)
    return sha256(domain_type + _uint_to_bytes(epoch, 8) + mix)


def compute_proposer_index(state, indices: list[int], seed: bytes) -> int:
    """Return from indices a random index sampled by effective balance.

    Note: Electra (EIP-7251) modifies this function to use 2-byte random values
    and MAX_EFFECTIVE_BALANCE_ELECTRA for improved precision with higher balances.
    """
    assert len(indices) > 0

    is_electra = hasattr(state, "pending_deposits")

    i = 0
    total = len(indices)

    if is_electra:
        MAX_RANDOM_VALUE = 2**16 - 1
        MAX_EFFECTIVE_BALANCE_ELECTRA = 2048 * 10**9
        while True:
            candidate_index = indices[compute_shuffled_index(i % total, total, seed)]
            random_bytes = sha256(seed + _uint_to_bytes(i // 16, 8))
            offset = (i % 16) * 2
            random_value = int.from_bytes(random_bytes[offset:offset + 2], "little")
            effective_balance = int(state.validators[candidate_index].effective_balance)
            if effective_balance * MAX_RANDOM_VALUE >= MAX_EFFECTIVE_BALANCE_ELECTRA * random_value:
                return candidate_index
            i += 1
    else:
        MAX_RANDOM_BYTE = 2**8 - 1
        MAX_EFFECTIVE_BALANCE = 32 * 10**9
        while True:
            candidate_index = indices[compute_shuffled_index(i % total, total, seed)]
            random_byte_input = seed + _uint_to_bytes(i // 32, 8)
            random_byte = sha256(random_byte_input)[i % 32]
            effective_balance = int(state.validators[candidate_index].effective_balance)
            if effective_balance * MAX_RANDOM_BYTE >= MAX_EFFECTIVE_BALANCE * random_byte:
                return candidate_index
            i += 1


def get_beacon_proposer_index(state, slot: int) -> int:
    """Return the beacon proposer index at slot.

    Note: This uses the state's randao_mixes which need to be current.
    For accurate results with stale state, use proposer_lookahead if available.
    """
    epoch = slot // SLOTS_PER_EPOCH()
    base_seed = get_seed(state, epoch, DOMAIN_BEACON_PROPOSER)
    seed = sha256(base_seed + _uint_to_bytes(slot, 8))
    indices = get_active_validator_indices(state, epoch)
    return compute_proposer_index(state, indices, seed)
