"""State accessor helper functions for state transition.

Implements functions to read state values.
Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/phase0/beacon-chain.md
"""

from typing import TYPE_CHECKING, Sequence, Set

from dataclasses import dataclass

from ...constants import (
    SLOTS_PER_EPOCH,
    SLOTS_PER_HISTORICAL_ROOT,
    EPOCHS_PER_HISTORICAL_VECTOR,
    GENESIS_EPOCH,
    EFFECTIVE_BALANCE_INCREMENT,
    BASE_REWARD_FACTOR,
    PROPOSER_REWARD_QUOTIENT,
    MIN_EPOCHS_TO_INACTIVITY_PENALTY,
    CHURN_LIMIT_QUOTIENT,
    MIN_PER_EPOCH_CHURN_LIMIT,
    MIN_PER_EPOCH_CHURN_LIMIT_ELECTRA,
    MAX_EFFECTIVE_BALANCE,
    MAX_EFFECTIVE_BALANCE_ELECTRA,
    MAX_PER_EPOCH_ACTIVATION_EXIT_CHURN_LIMIT,
    CHURN_LIMIT_QUOTIENT_GLOAS,
    MAX_PER_EPOCH_ACTIVATION_CHURN_LIMIT_GLOAS,
    CONSOLIDATION_CHURN_LIMIT_QUOTIENT,
    TIMELY_SOURCE_FLAG_INDEX,
    TIMELY_TARGET_FLAG_INDEX,
    TIMELY_HEAD_FLAG_INDEX,
    WEIGHT_DENOMINATOR,
    BASE_REWARDS_PER_EPOCH,
    MIN_DEPOSIT_AMOUNT,
    MAX_BLOBS_PER_BLOCK_ELECTRA,
)
from .math import integer_squareroot
from .misc import compute_epoch_at_slot
from .predicates import (
    is_active_validator,
    has_compounding_withdrawal_credential,
)
from ....crypto import sha256

if TYPE_CHECKING:
    from ...types import BeaconState, Validator

# Cache for expensive epoch-level computations.
#
# Keys MUST include a state-content identity, not just the slot/epoch:
# two conflicting forks routinely reach the same slot with different
# validator sets / participation, and a slot-only key lets one fork's
# value leak into the other's state transition. That exact collision
# wedged the node: attester-duty clones advanced the node's OWN fork
# across an epoch boundary, populating slot-keyed entries, and the
# subsequent reorg replay of the CANONICAL chain consumed them —
# producing a wrong post-epoch state root and a permanent parent-root
# mismatch on the next canonical block.
#
# We use the remerkleable backing Node of the field(s) the value is
# derived from as the identity component. Copies share subtree nodes
# structurally, so two states whose validators are untouched share the
# SAME node object (cache hit, identical content); any mutation swaps
# in a new node (cache miss, recompute). Keeping the node in the key
# holds a strong reference, so ids can't be recycled into stale hits.
_total_active_balance_cache: dict[tuple, int] = {}
_base_reward_per_increment_cache: dict[tuple, int] = {}
_eligible_validator_indices_cache: dict[tuple, list[int]] = {}
_CACHE_MAX_SIZE = 64


def _trim_cache(cache: dict) -> None:
    """Trim cache to prevent unbounded growth."""
    if len(cache) > _CACHE_MAX_SIZE:
        # Remove oldest entries (arbitrary order in Python 3.7+)
        keys_to_remove = list(cache.keys())[: len(cache) - _CACHE_MAX_SIZE // 2]
        for key in keys_to_remove:
            del cache[key]


def clear_accessors_caches() -> None:
    """Clear all module-level caches in accessors.

    This should be called between tests to prevent cache pollution.
    """
    _total_active_balance_cache.clear()
    _base_reward_per_increment_cache.clear()
    _eligible_validator_indices_cache.clear()
    _ACTIVE_INDICES_CACHE.clear()
    _EFFECTIVE_BALANCES_CACHE.clear()


def get_current_epoch(state: "BeaconState") -> int:
    """Return the current epoch of the state.

    Args:
        state: Beacon state

    Returns:
        Current epoch
    """
    return compute_epoch_at_slot(int(state.slot))


def get_previous_epoch(state: "BeaconState") -> int:
    """Return the previous epoch (or genesis epoch if in genesis epoch).

    Args:
        state: Beacon state

    Returns:
        Previous epoch (minimum GENESIS_EPOCH)
    """
    current_epoch = get_current_epoch(state)
    return max(current_epoch - 1, GENESIS_EPOCH)


def get_block_root(state: "BeaconState", epoch: int) -> bytes:
    """Return the block root at the start of a recent epoch.

    Args:
        state: Beacon state
        epoch: Target epoch

    Returns:
        Block root at start of epoch
    """
    from .misc import compute_start_slot_at_epoch

    return get_block_root_at_slot(state, compute_start_slot_at_epoch(epoch))


def get_block_root_at_slot(state: "BeaconState", slot: int) -> bytes:
    """Return the block root at a given slot.

    Args:
        state: Beacon state
        slot: Target slot (must be within SLOTS_PER_HISTORICAL_ROOT of current)

    Returns:
        Block root at slot

    Raises:
        AssertionError: If slot is out of range
    """
    assert slot < int(state.slot) <= slot + SLOTS_PER_HISTORICAL_ROOT()
    return bytes(state.block_roots[slot % SLOTS_PER_HISTORICAL_ROOT()])


def get_randao_mix(state: "BeaconState", epoch: int) -> bytes:
    """Return the RANDAO mix at a given epoch.

    Args:
        state: Beacon state
        epoch: Target epoch

    Returns:
        32-byte RANDAO mix
    """
    return bytes(state.randao_mixes[epoch % EPOCHS_PER_HISTORICAL_VECTOR()])


# Module-level caches keyed by id(state). Cleared automatically when a
# state object goes out of scope and a new one takes the same id slot —
# we use a WeakValueDictionary-style miss-on-mismatch by also remembering
# the validator count: if the snapshot length differs we recompute. This
# avoids a 50s state-transition stall at the Fulu→Gloas fork boundary,
# where compute_ptc gets called (1 + MIN_SEED_LOOKAHEAD) * SLOTS_PER_EPOCH
# times against the same pre-fork state and the inner loops hammer
# state.validators[i].effective_balance through remerkleable on every
# acceptance check.
import weakref as _weakref
from collections import OrderedDict as _OrderedDict

_ACTIVE_INDICES_CACHE: "_OrderedDict[tuple[int, int, int], list[int]]" = _OrderedDict()
_ACTIVE_INDICES_CACHE_MAX = 16
_EFFECTIVE_BALANCES_CACHE: "_OrderedDict[tuple[int, int], list[int]]" = _OrderedDict()
_EFFECTIVE_BALANCES_CACHE_MAX = 4


def get_active_validator_indices(state: "BeaconState", epoch: int) -> Sequence[int]:
    """Return the indices of active validators at the given epoch.

    Cached per (id(state), validator_count, epoch). The validator count is
    part of the key so the same id(state) reused for a different state
    object after GC doesn't return a stale answer.
    """
    n = len(state.validators)
    key = (id(state), n, int(epoch))
    cached = _ACTIVE_INDICES_CACHE.get(key)
    if cached is not None:
        _ACTIVE_INDICES_CACHE.move_to_end(key)
        return cached

    result = [
        i
        for i, v in enumerate(state.validators)
        if is_active_validator(v, epoch)
    ]

    _ACTIVE_INDICES_CACHE[key] = result
    # Evict on GC of the state so a recycled id() can never serve stale data
    _weakref.finalize(state, _ACTIVE_INDICES_CACHE.pop, key, None)
    while len(_ACTIVE_INDICES_CACHE) > _ACTIVE_INDICES_CACHE_MAX:
        _ACTIVE_INDICES_CACHE.popitem(last=False)
    return result


def get_effective_balances(state: "BeaconState") -> list[int]:
    """Return state.validators[*].effective_balance as a plain Python list.

    Each access via `state.validators[i].effective_balance` walks
    remerkleable containers and is ~100x slower than a list index. PTC and
    balance-weighted selection do this in a hot loop — caching the snapshot
    once per state turns 32k validator-field reads at the gloas fork into 32k
    plain int reads.

    Keyed by (id(state), len(state.validators)). Mutations to validator
    balances within the same state make the cache stale; we deliberately
    don't try to invalidate on mutate. Use only in code paths that don't
    mutate effective_balance (PTC init / committee balance-weighted
    sampling). The validator-count guard catches the common
    fork-upgrade churn case where a new state object grows by deposits.
    """
    n = len(state.validators)
    key = (id(state), n)
    cached = _EFFECTIVE_BALANCES_CACHE.get(key)
    if cached is not None:
        _EFFECTIVE_BALANCES_CACHE.move_to_end(key)
        return cached

    balances = [int(v.effective_balance) for v in state.validators]
    _EFFECTIVE_BALANCES_CACHE[key] = balances
    # Evict on GC of the state so a recycled id() can never serve stale data
    _weakref.finalize(state, _EFFECTIVE_BALANCES_CACHE.pop, key, None)
    while len(_EFFECTIVE_BALANCES_CACHE) > _EFFECTIVE_BALANCES_CACHE_MAX:
        _EFFECTIVE_BALANCES_CACHE.popitem(last=False)
    return balances


def get_validator_churn_limit(state: "BeaconState") -> int:
    """Return the validator churn limit for the current epoch.

    Args:
        state: Beacon state

    Returns:
        Maximum number of validators that can enter/exit per epoch
    """
    active_validator_indices = get_active_validator_indices(
        state, get_current_epoch(state)
    )
    return max(
        MIN_PER_EPOCH_CHURN_LIMIT(),
        len(active_validator_indices) // CHURN_LIMIT_QUOTIENT(),
    )


def _is_gloas_state(state: "BeaconState") -> bool:
    """Check if state is Gloas (ePBS) by presence of the builders field."""
    return hasattr(state, "builders")


def get_balance_churn_limit(state: "BeaconState") -> int:
    """Return the balance churn limit for the current epoch (Electra/Gloas).

    Args:
        state: Beacon state

    Returns:
        Maximum balance that can churn per epoch in Gwei
    """
    quotient = CHURN_LIMIT_QUOTIENT_GLOAS() if _is_gloas_state(state) else CHURN_LIMIT_QUOTIENT()
    churn = max(
        MIN_PER_EPOCH_CHURN_LIMIT_ELECTRA(),
        get_total_active_balance(state) // quotient,
    )
    return churn - (churn % EFFECTIVE_BALANCE_INCREMENT)


def get_exit_churn_limit(state: "BeaconState") -> int:
    """Return the exit churn limit for the current epoch (Gloas EIP-8061)."""
    return get_balance_churn_limit(state)


def get_activation_churn_limit_gloas(state: "BeaconState") -> int:
    """Return the activation churn limit for the current epoch (Gloas EIP-8061)."""
    churn = get_balance_churn_limit(state)
    return min(MAX_PER_EPOCH_ACTIVATION_CHURN_LIMIT_GLOAS(), churn)


def get_pending_balance_to_withdraw_for_builder(
    state: "BeaconState", builder_index: int
) -> int:
    """Return the pending balance to withdraw for the builder."""
    pending_withdrawals = sum(
        int(w.amount)
        for w in state.builder_pending_withdrawals
        if int(w.builder_index) == int(builder_index)
    )
    pending_payments = sum(
        int(p.withdrawal.amount)
        for p in state.builder_pending_payments
        if int(p.withdrawal.builder_index) == int(builder_index)
    )
    return pending_withdrawals + pending_payments


def can_builder_cover_bid(
    state: "BeaconState", builder_index: int, bid_amount: int
) -> bool:
    """Check if builder has enough funds to cover a bid."""
    builder_balance = int(state.builders[builder_index].balance)
    pending_withdrawals_amount = get_pending_balance_to_withdraw_for_builder(
        state, builder_index
    )
    min_balance = int(MIN_DEPOSIT_AMOUNT) + pending_withdrawals_amount
    if builder_balance < min_balance:
        return False
    return builder_balance - min_balance >= int(bid_amount)


def get_activation_exit_churn_limit(state: "BeaconState") -> int:
    """Return the activation/exit churn limit for the current epoch.

    Pre-Gloas (Electra/Fulu): min(MAX_PER_EPOCH_ACTIVATION_EXIT_CHURN_LIMIT, balance_churn_limit)
    Gloas: split into get_exit_churn_limit and get_activation_churn_limit_gloas; this returns exit.
    """
    if _is_gloas_state(state):
        return get_exit_churn_limit(state)
    return min(MAX_PER_EPOCH_ACTIVATION_EXIT_CHURN_LIMIT(), get_balance_churn_limit(state))


def get_consolidation_churn_limit(state: "BeaconState") -> int:
    """Return the consolidation churn limit for the current epoch.

    Pre-Gloas: balance_churn_limit - activation_exit_churn_limit
    Gloas: total_active_balance // CONSOLIDATION_CHURN_LIMIT_QUOTIENT
    """
    if _is_gloas_state(state):
        churn = get_total_active_balance(state) // CONSOLIDATION_CHURN_LIMIT_QUOTIENT()
        return churn - (churn % EFFECTIVE_BALANCE_INCREMENT)
    return get_balance_churn_limit(state) - get_activation_exit_churn_limit(state)


def get_seed(state: "BeaconState", epoch: int, domain_type: bytes) -> bytes:
    """Return the seed for the given epoch and domain.

    Args:
        state: Beacon state
        epoch: Target epoch
        domain_type: 4-byte domain type

    Returns:
        32-byte seed
    """
    from ...constants import MIN_SEED_LOOKAHEAD

    mix = get_randao_mix(state, epoch + EPOCHS_PER_HISTORICAL_VECTOR() - MIN_SEED_LOOKAHEAD - 1)
    return sha256(domain_type + epoch.to_bytes(8, "little") + mix)


def get_total_balance(state: "BeaconState", indices: Set[int]) -> int:
    """Return the total effective balance of the given validator indices.

    Args:
        state: Beacon state
        indices: Set of validator indices

    Returns:
        Total effective balance (minimum EFFECTIVE_BALANCE_INCREMENT)
    """
    return max(
        EFFECTIVE_BALANCE_INCREMENT,
        sum(int(state.validators[i].effective_balance) for i in indices),
    )


def get_total_active_balance(state: "BeaconState") -> int:
    """Return the total effective balance of active validators.

    Args:
        state: Beacon state

    Returns:
        Total active balance
    """
    # Depends on the validator set and the current epoch only. Keyed by
    # the validators subtree node so conflicting forks at the same slot
    # can never serve each other's value (see cache comment above).
    key = (get_current_epoch(state), state.validators.get_backing())
    if key in _total_active_balance_cache:
        return _total_active_balance_cache[key]

    result = get_total_balance(
        state, set(get_active_validator_indices(state, get_current_epoch(state)))
    )

    _total_active_balance_cache[key] = result
    _trim_cache(_total_active_balance_cache)
    return result


def get_base_reward_per_increment(state: "BeaconState") -> int:
    """Return the base reward per increment (Altair+).

    Args:
        state: Beacon state

    Returns:
        Base reward per increment in Gwei
    """
    # Derived from total_active_balance — same fork-safe key shape.
    key = (get_current_epoch(state), state.validators.get_backing())
    if key in _base_reward_per_increment_cache:
        return _base_reward_per_increment_cache[key]

    result = (
        EFFECTIVE_BALANCE_INCREMENT
        * BASE_REWARD_FACTOR
        // integer_squareroot(get_total_active_balance(state))
    )

    _base_reward_per_increment_cache[key] = result
    _trim_cache(_base_reward_per_increment_cache)
    return result


def get_base_reward(state: "BeaconState", index: int) -> int:
    """Return the base reward for a validator.

    Handles both Phase0 and Altair+ formulas.

    Args:
        state: Beacon state
        index: Validator index

    Returns:
        Base reward in Gwei
    """
    # Check if Phase0 (uses previous_epoch_attestations) or Altair+ (uses participation flags)
    if hasattr(state, "previous_epoch_participation"):
        # Altair+ formula: increments * base_reward_per_increment
        increments = (
            int(state.validators[index].effective_balance) // EFFECTIVE_BALANCE_INCREMENT
        )
        return increments * get_base_reward_per_increment(state)
    else:
        # Phase0 formula: effective_balance * BASE_REWARD_FACTOR // sqrt(total) // BASE_REWARDS_PER_EPOCH
        total_balance = get_total_active_balance(state)
        effective_balance = int(state.validators[index].effective_balance)
        return (
            effective_balance
            * BASE_REWARD_FACTOR
            // integer_squareroot(total_balance)
            // BASE_REWARDS_PER_EPOCH
        )


def get_proposer_reward(state: "BeaconState", attester_index: int) -> int:
    """Return the proposer reward for including an attestation.

    Args:
        state: Beacon state
        attester_index: Attester's validator index

    Returns:
        Proposer reward in Gwei
    """
    from ...constants import PROPOSER_WEIGHT

    return get_base_reward(state, attester_index) * PROPOSER_WEIGHT // WEIGHT_DENOMINATOR // BASE_REWARDS_PER_EPOCH


def get_finality_delay(state: "BeaconState") -> int:
    """Return the finality delay (epochs since last finalization).

    Args:
        state: Beacon state

    Returns:
        Number of epochs since finalization
    """
    return get_previous_epoch(state) - int(state.finalized_checkpoint.epoch)


def is_in_inactivity_leak(state: "BeaconState") -> bool:
    """Check if the chain is in an inactivity leak.

    Args:
        state: Beacon state

    Returns:
        True if in inactivity leak (finality delay > MIN_EPOCHS_TO_INACTIVITY_PENALTY)
    """
    return get_finality_delay(state) > MIN_EPOCHS_TO_INACTIVITY_PENALTY


def get_eligible_validator_indices(state: "BeaconState") -> Sequence[int]:
    """Return indices of validators eligible for rewards/penalties.

    A validator is eligible if:
    - Active in previous epoch, OR
    - Slashed and not yet withdrawable

    Args:
        state: Beacon state

    Returns:
        Sequence of eligible validator indices
    """
    # Depends on the validator set and the previous epoch only. Fork-safe
    # key: validators subtree node (see cache comment above).
    previous_epoch = get_previous_epoch(state)
    key = (previous_epoch, state.validators.get_backing())
    if key in _eligible_validator_indices_cache:
        return _eligible_validator_indices_cache[key]

    result = [
        i
        for i, v in enumerate(state.validators)
        if is_active_validator(v, previous_epoch)
        or (v.slashed and previous_epoch + 1 < int(v.withdrawable_epoch))
    ]

    _eligible_validator_indices_cache[key] = result
    _trim_cache(_eligible_validator_indices_cache)
    return result


def get_max_effective_balance(validator: "Validator") -> int:
    """Return the maximum effective balance for a validator (Electra).

    Compounding validators can have higher effective balance.

    Args:
        validator: Validator

    Returns:
        Maximum effective balance in Gwei
    """
    if has_compounding_withdrawal_credential(validator):
        return MAX_EFFECTIVE_BALANCE_ELECTRA
    return MAX_EFFECTIVE_BALANCE


def get_pending_balance_to_withdraw(state: "BeaconState", validator_index: int) -> int:
    """Return the total pending balance to withdraw for a validator (Electra).

    Args:
        state: Beacon state
        validator_index: Validator index

    Returns:
        Total pending withdrawal balance in Gwei
    """
    if not hasattr(state, "pending_partial_withdrawals"):
        return 0
    return sum(
        int(w.amount)
        for w in state.pending_partial_withdrawals
        if int(w.validator_index) == validator_index
    )


@dataclass
class BlobParameters:
    """Blob parameters for a given epoch."""
    epoch: int
    max_blobs_per_block: int


def get_blob_parameters(epoch: int) -> BlobParameters:
    """Return blob parameters for a given epoch.

    Looks up the BLOB_SCHEDULE from network config and returns the
    appropriate max_blobs_per_block for the epoch.

    Args:
        epoch: Epoch to get parameters for

    Returns:
        BlobParameters with max_blobs_per_block for the epoch
    """
    from ...network_config import get_config

    config = get_config()
    blob_schedule = getattr(config, "blob_schedule", None) or []

    for entry in sorted(blob_schedule, key=lambda e: e["epoch"], reverse=True):
        entry_epoch = entry["epoch"]
        if epoch >= entry_epoch:
            return BlobParameters(
                epoch=entry_epoch,
                max_blobs_per_block=entry["max_blobs_per_block"],
            )

    return BlobParameters(epoch=0, max_blobs_per_block=MAX_BLOBS_PER_BLOCK_ELECTRA())
