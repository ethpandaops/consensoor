"""Rewards and penalties processing.

Supports both Phase0 (PendingAttestation-based) and Altair+ (participation flags).
Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/phase0/beacon-chain.md
Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/altair/beacon-chain.md
"""

from typing import TYPE_CHECKING, Tuple, Sequence

from ...constants import (
    GENESIS_EPOCH,
    TIMELY_SOURCE_FLAG_INDEX,
    TIMELY_TARGET_FLAG_INDEX,
    TIMELY_HEAD_FLAG_INDEX,
    PARTICIPATION_FLAG_WEIGHTS,
    SYNC_REWARD_WEIGHT,
    PROPOSER_WEIGHT,
    WEIGHT_DENOMINATOR,
    EFFECTIVE_BALANCE_INCREMENT,
    INACTIVITY_PENALTY_QUOTIENT_ALTAIR,
    INACTIVITY_PENALTY_QUOTIENT_BELLATRIX,
    INACTIVITY_PENALTY_QUOTIENT,
    INACTIVITY_SCORE_BIAS,
    PROPOSER_REWARD_QUOTIENT,
    MIN_EPOCHS_TO_INACTIVITY_PENALTY,
    BASE_REWARDS_PER_EPOCH,
)
from ..helpers.accessors import (
    get_current_epoch,
    get_previous_epoch,
    get_total_active_balance,
    get_base_reward,
    get_eligible_validator_indices,
    is_in_inactivity_leak,
    get_total_balance,
    get_finality_delay,
)
from ..helpers.attestation import (
    get_unslashed_participating_indices,
    get_matching_source_attestations_phase0,
    get_matching_target_attestations_phase0,
    get_matching_head_attestations_phase0,
    get_unslashed_attesting_indices_phase0,
    get_attesting_indices_from_pending_attestation,
)
from ..helpers.mutators import increase_balance, decrease_balance

if TYPE_CHECKING:
    from ...types import BeaconState




def process_rewards_and_penalties(state: "BeaconState") -> None:
    """Process rewards and penalties for the epoch.

    Handles both Phase0 (PendingAttestation) and Altair+ (participation flags).

    Args:
        state: Beacon state (modified in place)
    """
    # Skip genesis epoch (rewards based on previous epoch)
    if get_current_epoch(state) == GENESIS_EPOCH:
        return

    # Check if Phase0 or Altair+ based on state fields
    if hasattr(state, "previous_epoch_participation"):
        # Altair+ path
        process_rewards_and_penalties_altair(state)
    else:
        # Phase0 path
        process_rewards_and_penalties_phase0(state)


def process_rewards_and_penalties_phase0(state: "BeaconState") -> None:
    """Process rewards and penalties for Phase0."""
    rewards, penalties = get_attestation_deltas_phase0(state)
    for index in range(len(state.validators)):
        increase_balance(state, index, rewards[index])
        decrease_balance(state, index, penalties[index])


def process_rewards_and_penalties_altair(state: "BeaconState") -> None:
    """Process rewards and penalties for Altair+."""
    flag_deltas = [
        get_flag_index_deltas(state, flag_index)
        for flag_index in range(len(PARTICIPATION_FLAG_WEIGHTS))
    ]
    deltas = flag_deltas + [get_inactivity_penalty_deltas(state)]
    for rewards, penalties in deltas:
        for index in range(len(state.validators)):
            increase_balance(state, index, rewards[index])
            decrease_balance(state, index, penalties[index])


# =============================================================================
# Phase0 Rewards Functions
# =============================================================================


def get_attestation_deltas_phase0(state: "BeaconState") -> Tuple[Sequence[int], Sequence[int]]:
    """Return attestation reward/penalty deltas for each validator (Phase0)."""
    source_rewards, source_penalties = get_source_deltas_phase0(state)
    target_rewards, target_penalties = get_target_deltas_phase0(state)
    head_rewards, head_penalties = get_head_deltas_phase0(state)
    inclusion_delay_rewards, _ = get_inclusion_delay_deltas_phase0(state)
    _, inactivity_penalties = get_inactivity_penalty_deltas_phase0(state)

    rewards = [
        source_rewards[i] + target_rewards[i] + head_rewards[i] + inclusion_delay_rewards[i]
        for i in range(len(state.validators))
    ]

    penalties = [
        source_penalties[i] + target_penalties[i] + head_penalties[i] + inactivity_penalties[i]
        for i in range(len(state.validators))
    ]

    return rewards, penalties


def get_attestation_component_deltas_phase0(
    state: "BeaconState", attestations
) -> Tuple[Sequence[int], Sequence[int]]:
    """Helper with shared logic for source, target, and head deltas (Phase0)."""
    rewards = [0] * len(state.validators)
    penalties = [0] * len(state.validators)

    total_balance = get_total_active_balance(state)
    unslashed_attesting_indices = get_unslashed_attesting_indices_phase0(state, attestations)
    attesting_balance = get_total_balance(state, unslashed_attesting_indices)

    in_leak = is_in_inactivity_leak(state)

    for index in get_eligible_validator_indices(state):
        if index in unslashed_attesting_indices:
            increment = EFFECTIVE_BALANCE_INCREMENT
            if in_leak:
                # During inactivity leak: full base reward compensation
                rewards[index] += get_base_reward(state, index)
            else:
                # Normal operation: proportional reward
                reward_numerator = get_base_reward(state, index) * (attesting_balance // increment)
                rewards[index] += reward_numerator // (total_balance // increment)
        else:
            penalties[index] += get_base_reward(state, index)

    return rewards, penalties


def get_source_deltas_phase0(state: "BeaconState") -> Tuple[Sequence[int], Sequence[int]]:
    """Return attester micro-rewards/penalties for source-vote (Phase0)."""
    matching_source_attestations = get_matching_source_attestations_phase0(
        state, get_previous_epoch(state)
    )
    return get_attestation_component_deltas_phase0(state, matching_source_attestations)


def get_target_deltas_phase0(state: "BeaconState") -> Tuple[Sequence[int], Sequence[int]]:
    """Return attester micro-rewards/penalties for target-vote (Phase0)."""
    matching_target_attestations = get_matching_target_attestations_phase0(
        state, get_previous_epoch(state)
    )
    return get_attestation_component_deltas_phase0(state, matching_target_attestations)


def get_head_deltas_phase0(state: "BeaconState") -> Tuple[Sequence[int], Sequence[int]]:
    """Return attester micro-rewards/penalties for head-vote (Phase0)."""
    matching_head_attestations = get_matching_head_attestations_phase0(
        state, get_previous_epoch(state)
    )
    return get_attestation_component_deltas_phase0(state, matching_head_attestations)


def get_proposer_reward_phase0(state: "BeaconState", attesting_index: int) -> int:
    """Get proposer reward for including an attester's attestation (Phase0)."""
    return get_base_reward(state, attesting_index) // PROPOSER_REWARD_QUOTIENT


def get_inclusion_delay_deltas_phase0(state: "BeaconState") -> Tuple[Sequence[int], Sequence[int]]:
    """Return proposer and inclusion delay micro-rewards/penalties (Phase0)."""
    rewards = [0] * len(state.validators)
    penalties = [0] * len(state.validators)

    matching_source_attestations = get_matching_source_attestations_phase0(
        state, get_previous_epoch(state)
    )

    for index in get_unslashed_attesting_indices_phase0(state, matching_source_attestations):
        # Find the attestation with minimum inclusion delay for this validator
        relevant_attestations = [
            a for a in matching_source_attestations
            if index in get_attesting_indices_from_pending_attestation(state, a)
        ]
        if not relevant_attestations:
            continue

        attestation = min(relevant_attestations, key=lambda a: int(a.inclusion_delay))

        # Proposer reward
        rewards[int(attestation.proposer_index)] += get_proposer_reward_phase0(state, index)

        # Attester reward based on inclusion delay
        max_attester_reward = get_base_reward(state, index) - get_proposer_reward_phase0(state, index)
        rewards[index] += max_attester_reward // int(attestation.inclusion_delay)

    return rewards, penalties


def get_inactivity_penalty_deltas_phase0(state: "BeaconState") -> Tuple[Sequence[int], Sequence[int]]:
    """Return inactivity reward/penalty deltas (Phase0)."""
    rewards = [0] * len(state.validators)
    penalties = [0] * len(state.validators)

    if is_in_inactivity_leak(state):
        matching_target_attestations = get_matching_target_attestations_phase0(
            state, get_previous_epoch(state)
        )
        matching_target_attesting_indices = get_unslashed_attesting_indices_phase0(
            state, matching_target_attestations
        )

        for index in get_eligible_validator_indices(state):
            # If validator is performing optimally, this cancels all rewards for neutral balance
            base_reward = get_base_reward(state, index)
            penalties[index] += BASE_REWARDS_PER_EPOCH * base_reward - get_proposer_reward_phase0(state, index)

            if index not in matching_target_attesting_indices:
                effective_balance = int(state.validators[index].effective_balance)
                penalties[index] += effective_balance * get_finality_delay(state) // INACTIVITY_PENALTY_QUOTIENT

    return rewards, penalties


def get_flag_index_deltas(
    state: "BeaconState", flag_index: int
) -> Tuple[Sequence[int], Sequence[int]]:
    """Return the deltas for a given flag_index by scanning through the participation flags."""
    validator_count = len(state.validators)
    rewards = [0] * validator_count
    penalties = [0] * validator_count

    previous_epoch = get_previous_epoch(state)
    unslashed_participating_indices = get_unslashed_participating_indices(
        state, flag_index, previous_epoch
    )
    weight = PARTICIPATION_FLAG_WEIGHTS[flag_index]
    unslashed_participating_balance = get_total_balance(state, unslashed_participating_indices)
    unslashed_participating_increments = unslashed_participating_balance // EFFECTIVE_BALANCE_INCREMENT
    active_increments = get_total_active_balance(state) // EFFECTIVE_BALANCE_INCREMENT

    for index in get_eligible_validator_indices(state):
        base_reward = get_base_reward(state, index)
        if index in unslashed_participating_indices:
            if not is_in_inactivity_leak(state):
                reward_numerator = base_reward * weight * unslashed_participating_increments
                rewards[index] += reward_numerator // (active_increments * WEIGHT_DENOMINATOR)
        elif flag_index != TIMELY_HEAD_FLAG_INDEX:
            penalties[index] += base_reward * weight // WEIGHT_DENOMINATOR

    return rewards, penalties


def get_inactivity_penalty_deltas(
    state: "BeaconState",
) -> Tuple[Sequence[int], Sequence[int]]:
    """Calculate inactivity penalties based on inactivity scores.

    Args:
        state: Beacon state

    Returns:
        Tuple of (rewards, penalties) sequences indexed by validator
    """
    validator_count = len(state.validators)
    rewards = [0] * validator_count
    penalties = [0] * validator_count

    previous_epoch = get_previous_epoch(state)
    matching_target_indices = get_unslashed_participating_indices(
        state, TIMELY_TARGET_FLAG_INDEX, previous_epoch
    )

    # Use correct inactivity penalty quotient based on fork
    # Bellatrix+ has latest_execution_payload_header, Altair doesn't
    if hasattr(state, "latest_execution_payload_header"):
        inactivity_penalty_quotient = INACTIVITY_PENALTY_QUOTIENT_BELLATRIX
    else:
        inactivity_penalty_quotient = INACTIVITY_PENALTY_QUOTIENT_ALTAIR

    for index in get_eligible_validator_indices(state):
        if index not in matching_target_indices:
            # Penalty based on inactivity score and effective balance
            penalty_numerator = (
                int(state.validators[index].effective_balance)
                * int(state.inactivity_scores[index])
            )
            penalty_denominator = INACTIVITY_SCORE_BIAS * inactivity_penalty_quotient
            penalties[index] += penalty_numerator // penalty_denominator

    return rewards, penalties
