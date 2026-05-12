"""PTC window processing (Gloas EIP-7732).

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/gloas/beacon-chain.md
"""

from typing import TYPE_CHECKING

from ...constants import (
    DOMAIN_BEACON_ATTESTER,
    DOMAIN_PTC_ATTESTER,
    MAX_COMMITTEES_PER_SLOT,
    MAX_EFFECTIVE_BALANCE_ELECTRA,
    MIN_SEED_LOOKAHEAD,
    PTC_SIZE,
    SHUFFLE_ROUND_COUNT,
    SLOTS_PER_EPOCH,
    TARGET_COMMITTEE_SIZE,
)
from ..helpers.accessors import (
    get_active_validator_indices,
    get_current_epoch,
    get_seed,
)
from ..helpers.misc import compute_start_slot_at_epoch
from ....crypto import sha256

if TYPE_CHECKING:
    from ...types import BeaconState


def _shuffle_inplace(values: list[int], seed: bytes) -> None:
    """Swap-or-not shuffle the whole list in place, equivalent to building
    `[values[compute_shuffled_index(k, n, seed)] for k in range(n)]`.
    Amortises the per-round pivot and source-chunk hashes across all
    positions: for 768 indices x 90 rounds, ~138,240 sha256 calls collapse
    to ~270 (×~500 fewer hashes).

    Implementation: each round forms swap pairs (i, flip) with
    flip = (pivot - i) mod n; both pair members share the swap bit at
    `max(i, flip)`, so we walk i in [0, n) and swap only when i < flip
    (visiting every pair once, skipping self-pairs). The Ethereum spec's
    `compute_shuffled_index` runs rounds 0..R-1; whole-list shuffling in
    the SAME forward order produces the INVERSE permutation. To match
    `values[compute_shuffled_index(k)]` we therefore iterate rounds in
    REVERSE — this is the standard `shuffle_list` direction used by
    lighthouse and the Python pyspec helpers.
    """
    n = len(values)
    if n <= 1:
        return

    rounds = SHUFFLE_ROUND_COUNT()
    for current_round in range(rounds - 1, -1, -1):
        round_byte = current_round.to_bytes(1, "little")
        pivot = int.from_bytes(sha256(seed + round_byte)[:8], "little") % n

        last_chunk = -1
        chunk_bytes = b""
        for i in range(n):
            flip = (pivot + n - i) % n
            if i >= flip:
                continue
            position = flip  # max(i, flip) = flip since i < flip
            chunk = position >> 8
            if chunk != last_chunk:
                chunk_bytes = sha256(
                    seed + round_byte + chunk.to_bytes(4, "little")
                )
                last_chunk = chunk
            byte = chunk_bytes[(position & 0xFF) >> 3]
            bit = (byte >> (position & 0x7)) & 1
            if bit:
                values[i], values[flip] = values[flip], values[i]


def process_ptc_window(state: "BeaconState") -> None:
    """Update the cached PTC window.

    Shifts all epochs forward by one and computes the new last epoch.
    All 32 new entries share the same epoch, so the per-epoch inputs
    (active indices, beacon-attester / PTC seeds, committees-per-slot,
    effective-balance snapshot) are computed ONCE here and reused for
    every slot — versus the spec's per-slot `compute_ptc` which would
    re-walk `state.validators` ~1024 times per slot via remerkleable's
    tree-backed accessors. That redundant work was costing 300+ seconds
    per epoch transition in mainnet-preset devnets, wedging the chain.
    """
    spe = SLOTS_PER_EPOCH()
    window_len = len(state.ptc_window)

    new_window = [None] * window_len
    for i in range(window_len - spe):
        new_window[i] = state.ptc_window[i + spe]

    next_epoch = get_current_epoch(state) + MIN_SEED_LOOKAHEAD + 1
    start_slot = compute_start_slot_at_epoch(next_epoch)

    active_indices = get_active_validator_indices(state, next_epoch)
    total_active = len(active_indices)
    committees_per_slot = max(
        1,
        min(
            MAX_COMMITTEES_PER_SLOT(),
            total_active // spe // TARGET_COMMITTEE_SIZE(),
        ),
    )
    committee_count = committees_per_slot * spe
    seed_attester = get_seed(state, next_epoch, DOMAIN_BEACON_ATTESTER)
    seed_ptc_base = get_seed(state, next_epoch, DOMAIN_PTC_ATTESTER)

    n_validators = len(state.validators)
    effective_balances = [
        int(state.validators[i].effective_balance) for i in range(n_validators)
    ]

    # Shuffle the active set once for the whole epoch. compute_committee's
    # per-slot per-validator `compute_shuffled_index` produces the same
    # permutation as a whole-list shuffle of the active set; for 768
    # validators that's ~500x fewer sha256 calls.
    shuffled_active = list(active_indices)
    _shuffle_inplace(shuffled_active, seed_attester)

    ptc_size = PTC_SIZE()
    max_random_value = 2**16 - 1
    max_effective = MAX_EFFECTIVE_BALANCE_ELECTRA

    for j, slot in enumerate(range(start_slot, start_slot + spe)):
        slot_committee_base = (slot % spe) * committees_per_slot
        indices = []
        for c in range(committees_per_slot):
            committee_index = slot_committee_base + c
            start = total_active * committee_index // committee_count
            end = total_active * (committee_index + 1) // committee_count
            indices.extend(shuffled_active[start:end])

        slot_seed = sha256(seed_ptc_base + int(slot).to_bytes(8, "little"))

        total = len(indices)
        selected: list[int] = []
        # Hoist the sha256 out of the inner loop — random_bytes only
        # refreshes every 16 iterations (8 random 16-bit values per hash).
        i = 0
        random_bytes = sha256(slot_seed + b"\x00" * 8)
        while len(selected) < ptc_size:
            r = i & 15
            if r == 0 and i != 0:
                random_bytes = sha256(
                    slot_seed + (i >> 4).to_bytes(8, "little")
                )
            offset = r * 2
            random_value = int.from_bytes(random_bytes[offset:offset + 2], "little")
            candidate = indices[i % total]
            if effective_balances[candidate] * max_random_value >= max_effective * random_value:
                selected.append(candidate)
            i += 1

        new_window[window_len - spe + j] = selected

    for i in range(window_len):
        state.ptc_window[i] = new_window[i]
