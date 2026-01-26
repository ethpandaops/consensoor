"""State accessor helper functions for state transition.

Implements functions to read state values.
Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/phase0/beacon-chain.md
"""

from typing import TYPE_CHECKING, Sequence, Set

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
    TIMELY_SOURCE_FLAG_INDEX,
    TIMELY_TARGET_FLAG_INDEX,
    TIMELY_HEAD_FLAG_INDEX,
    WEIGHT_DENOMINATOR,
    BASE_REWARDS_PER_EPOCH,
    MIN_DEPOSIT_AMOUNT,
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

# Cache for expensive epoch-level computations (keyed by state slot)
_total_active_balance_cache: dict[int, int] = {}
_base_reward_per_increment_cache: dict[int, int] = {}
_eligible_validator_indices_cache: dict[int, list[int]] = {}
_CACHE_MAX_SIZE = 64


def _trim_cache(cache: dict) -> None:
    """Trim cache to prevent unbounded growth."""
    if len(cache) > _CACHE_MAX_SIZE:
        # Remove oldest entries (arbitrary order in Python 3.7+)
        keys_to_remove = list(cache.keys())[: len(cache) - _CACHE_MAX_SIZE // 2]
        for key in keys_to_remove:
            del cache[key]


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


def get_active_validator_indices(state: "BeaconState", epoch: int) -> Sequence[int]:
    """Return the indices of active validators at the given epoch.

    Args:
        state: Beacon state
        epoch: Target epoch

    Returns:
        Sequence of validator indices
    """
    return [
        i
        for i, v in enumerate(state.validators)
        if is_active_validator(v, epoch)
    ]


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


def get_balance_churn_limit(state: "BeaconState") -> int:
    """Return the balance churn limit for the current epoch (Electra).

    Args:
        state: Beacon state

    Returns:
        Maximum balance that can churn per epoch in Gwei
    """
    churn = max(
        MIN_PER_EPOCH_CHURN_LIMIT_ELECTRA(),
        get_total_active_balance(state) // CHURN_LIMIT_QUOTIENT(),
    )
    return churn - (churn % EFFECTIVE_BALANCE_INCREMENT)


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
    """Return the activation/exit churn limit for the current epoch (Electra).

    This is the minimum of the per-epoch activation exit churn limit and the balance churn limit.

    Args:
        state: Beacon state

    Returns:
        Churn limit in Gwei
    """
    return min(MAX_PER_EPOCH_ACTIVATION_EXIT_CHURN_LIMIT(), get_balance_churn_limit(state))


def get_consolidation_churn_limit(state: "BeaconState") -> int:
    """Return the consolidation churn limit for the current epoch (Electra).

    Args:
        state: Beacon state

    Returns:
        Churn limit in Gwei
    """
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
    slot = int(state.slot)
    if slot in _total_active_balance_cache:
        return _total_active_balance_cache[slot]

    result = get_total_balance(
        state, set(get_active_validator_indices(state, get_current_epoch(state)))
    )

    _total_active_balance_cache[slot] = result
    _trim_cache(_total_active_balance_cache)
    return result


def get_base_reward_per_increment(state: "BeaconState") -> int:
    """Return the base reward per increment (Altair+).

    Args:
        state: Beacon state

    Returns:
        Base reward per increment in Gwei
    """
    slot = int(state.slot)
    if slot in _base_reward_per_increment_cache:
        return _base_reward_per_increment_cache[slot]

    result = (
        EFFECTIVE_BALANCE_INCREMENT
        * BASE_REWARD_FACTOR
        // integer_squareroot(get_total_active_balance(state))
    )

    _base_reward_per_increment_cache[slot] = result
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
    slot = int(state.slot)
    if slot in _eligible_validator_indices_cache:
        return _eligible_validator_indices_cache[slot]

    previous_epoch = get_previous_epoch(state)
    result = [
        i
        for i, v in enumerate(state.validators)
        if is_active_validator(v, previous_epoch)
        or (v.slashed and previous_epoch + 1 < int(v.withdrawable_epoch))
    ]

    _eligible_validator_indices_cache[slot] = result
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
