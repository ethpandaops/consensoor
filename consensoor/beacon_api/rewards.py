"""Beacon-API rewards: block, sync committee and attestation rewards."""

from __future__ import annotations

from ..spec.constants import (
    EFFECTIVE_BALANCE_INCREMENT,
    PARTICIPATION_FLAG_WEIGHTS,
    PROPOSER_WEIGHT,
    SLOTS_PER_EPOCH,
    SYNC_COMMITTEE_SIZE,
    SYNC_REWARD_WEIGHT,
    TIMELY_HEAD_FLAG_INDEX,
    TIMELY_SOURCE_FLAG_INDEX,
    TIMELY_TARGET_FLAG_INDEX,
    WEIGHT_DENOMINATOR,
)
from ..spec.state_transition.helpers.accessors import (
    get_base_reward_per_increment,
    get_eligible_validator_indices,
    get_previous_epoch,
    get_total_active_balance,
    get_total_balance,
    is_in_inactivity_leak,
)
from ..spec.state_transition.helpers.attestation import get_unslashed_participating_indices

_FLAG_NAMES = {
    TIMELY_SOURCE_FLAG_INDEX: "source",
    TIMELY_TARGET_FLAG_INDEX: "target",
    TIMELY_HEAD_FLAG_INDEX: "head",
}


def _state_at_slot(pre_state, slot: int):
    from ..spec.state_transition import process_slots

    state = pre_state.copy()
    if int(state.slot) < slot:
        state = process_slots(state, slot)
    return state


def _sync_rewards(state) -> tuple[int, int]:
    total_active_increments = get_total_active_balance(state) // EFFECTIVE_BALANCE_INCREMENT
    total_base_rewards = get_base_reward_per_increment(state) * total_active_increments
    max_participant_rewards = (
        total_base_rewards * SYNC_REWARD_WEIGHT // WEIGHT_DENOMINATOR // SLOTS_PER_EPOCH()
    )
    participant_reward = max_participant_rewards // SYNC_COMMITTEE_SIZE()
    proposer_reward = participant_reward * PROPOSER_WEIGHT // (WEIGHT_DENOMINATOR - PROPOSER_WEIGHT)
    return participant_reward, proposer_reward


def _sync_committee_indices(state) -> list[int]:
    index_by_pubkey = {bytes(v.pubkey): i for i, v in enumerate(state.validators)}
    return [index_by_pubkey[bytes(pk)] for pk in state.current_sync_committee.pubkeys]


def compute_block_rewards(pre_state, signed_block) -> dict:
    from ..spec.state_transition.block import (
        process_block_header,
        process_eth1_data,
        process_execution_payload,
        process_execution_payload_bid,
        process_randao,
        process_withdrawals,
    )
    from ..spec.state_transition.block.operations import (
        process_attestation,
        process_attester_slashing,
        process_parent_execution_payload,
        process_proposer_slashing,
    )
    from ..spec.state_transition.helpers.predicates import is_execution_enabled

    block = signed_block.message
    body = block.body
    proposer_index = int(block.proposer_index)
    state = _state_at_slot(pre_state, int(block.slot))

    parent_slot = None
    if hasattr(body, "signed_execution_payload_bid"):
        parent_slot = int(state.latest_block_header.slot)
        process_parent_execution_payload(state, block)
        process_block_header(state, block)
        process_withdrawals(state)
        process_execution_payload_bid(state, body.signed_execution_payload_bid)
    else:
        process_block_header(state, block)
        if hasattr(body, "execution_payload") and is_execution_enabled(state, body):
            if hasattr(body.execution_payload, "withdrawals"):
                process_withdrawals(state, body.execution_payload)
            process_execution_payload(state, body)
    process_randao(state, body)
    if hasattr(body, "eth1_data"):
        process_eth1_data(state, body)

    def proposer_delta(fn, items) -> int:
        before = int(state.balances[proposer_index])
        for item in items:
            fn(state, item)
        return max(0, int(state.balances[proposer_index]) - before)

    proposer_slashings = proposer_delta(process_proposer_slashing, body.proposer_slashings)
    attester_slashings = proposer_delta(process_attester_slashing, body.attester_slashings)
    attestations = proposer_delta(
        lambda s, a: process_attestation(s, a, parent_slot=parent_slot), body.attestations
    )

    sync_aggregate = 0
    if hasattr(body, "sync_aggregate"):
        _, proposer_reward = _sync_rewards(state)
        participants = sum(1 for bit in body.sync_aggregate.sync_committee_bits if bit)
        sync_aggregate = participants * proposer_reward

    return {
        "proposer_index": proposer_index,
        "total": attestations + sync_aggregate + proposer_slashings + attester_slashings,
        "attestations": attestations,
        "sync_aggregate": sync_aggregate,
        "proposer_slashings": proposer_slashings,
        "attester_slashings": attester_slashings,
    }


def compute_sync_committee_rewards(pre_state, signed_block, validator_ids: set[int] | None) -> list[dict]:
    block = signed_block.message
    if not hasattr(block.body, "sync_aggregate"):
        raise ValueError("block has no sync aggregate")
    state = _state_at_slot(pre_state, int(block.slot))
    participant_reward, _ = _sync_rewards(state)

    deltas: dict[int, int] = {}
    bits = block.body.sync_aggregate.sync_committee_bits
    for position, validator_index in enumerate(_sync_committee_indices(state)):
        delta = participant_reward if bits[position] else -participant_reward
        deltas[validator_index] = deltas.get(validator_index, 0) + delta

    return [
        {"validator_index": index, "reward": reward}
        for index, reward in deltas.items()
        if not validator_ids or index in validator_ids
    ]


def compute_attestation_rewards(state, validator_ids: set[int] | None) -> dict:
    from ..spec.state_transition.epoch import (
        process_inactivity_updates,
        process_justification_and_finalization,
    )
    from ..spec.state_transition.epoch.rewards import (
        get_flag_index_deltas,
        get_inactivity_penalty_deltas,
    )

    if not hasattr(state, "previous_epoch_participation"):
        raise ValueError("attestation rewards are not available for phase0")

    state = state.copy()
    process_justification_and_finalization(state)
    process_inactivity_updates(state)

    previous_epoch = get_previous_epoch(state)
    eligible = list(get_eligible_validator_indices(state))
    in_leak = is_in_inactivity_leak(state)
    base_reward_per_increment = get_base_reward_per_increment(state, previous_epoch)
    active_increments = get_total_active_balance(state) // EFFECTIVE_BALANCE_INCREMENT

    effective_balances = sorted({int(state.validators[i].effective_balance) for i in eligible})
    ideal = {
        balance: {"effective_balance": balance, "head": 0, "target": 0, "source": 0, "inactivity": 0}
        for balance in effective_balances
    }
    totals = {
        index: {"validator_index": index, "head": 0, "target": 0, "source": 0, "inactivity": 0}
        for index in eligible
        if not validator_ids or index in validator_ids
    }

    for flag_index, weight in enumerate(PARTICIPATION_FLAG_WEIGHTS):
        name = _FLAG_NAMES[flag_index]
        participating = get_unslashed_participating_indices(state, flag_index, previous_epoch)
        participating_increments = get_total_balance(state, participating) // EFFECTIVE_BALANCE_INCREMENT
        for balance, entry in ideal.items():
            base_reward = balance // EFFECTIVE_BALANCE_INCREMENT * base_reward_per_increment
            if not in_leak:
                entry[name] = (
                    base_reward * weight * participating_increments
                    // (active_increments * WEIGHT_DENOMINATOR)
                )
        rewards, penalties = get_flag_index_deltas(state, flag_index)
        for index, entry in totals.items():
            entry[name] = int(rewards[index]) - int(penalties[index])

    _, inactivity_penalties = get_inactivity_penalty_deltas(state)
    for index, entry in totals.items():
        entry["inactivity"] = -int(inactivity_penalties[index])

    return {"ideal_rewards": list(ideal.values()), "total_rewards": list(totals.values())}
