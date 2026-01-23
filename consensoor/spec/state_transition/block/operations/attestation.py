"""Attestation processing.

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/phase0/beacon-chain.md
Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/electra/beacon-chain.md
"""

from typing import TYPE_CHECKING

from ....constants import (
    SLOTS_PER_EPOCH,
    MIN_ATTESTATION_INCLUSION_DELAY,
    TIMELY_SOURCE_FLAG_INDEX,
    TIMELY_TARGET_FLAG_INDEX,
    TIMELY_HEAD_FLAG_INDEX,
    PROPOSER_WEIGHT,
    WEIGHT_DENOMINATOR,
    PARTICIPATION_FLAG_WEIGHTS,
)
from ...helpers.predicates import is_valid_indexed_attestation
from ...helpers.accessors import (
    get_current_epoch,
    get_previous_epoch,
    get_base_reward,
)
from ...helpers.mutators import increase_balance
from ...helpers.beacon_committee import get_beacon_proposer_index, get_beacon_committee
from ...helpers.attestation import (
    get_attesting_indices,
    get_indexed_attestation,
    get_attestation_participation_flag_indices,
    add_flag,
    has_flag,
    is_attestation_same_slot,
)
from ...helpers.misc import compute_epoch_at_slot


def _is_gloas_state(state) -> bool:
    """Check if state is a gloas (ePBS) state.

    Uses try/except because remerkleable containers may raise exceptions
    for unknown attributes instead of returning AttributeError.
    """
    try:
        _ = state.builders
        return True
    except Exception:
        return False


if TYPE_CHECKING:
    from ....types import BeaconState
    from ....types.electra import Attestation


def process_attestation(state: "BeaconState", attestation: "Attestation") -> None:
    """Process an attestation.

    Handles both Phase0 (PendingAttestation) and Altair+ (participation flags).

    Args:
        state: Beacon state (modified in place)
        attestation: Attestation to process

    Raises:
        AssertionError: If validation fails
    """
    # Check if Phase0 (uses previous_epoch_attestations) or Altair+ (uses participation flags)
    if hasattr(state, "previous_epoch_participation"):
        process_attestation_altair(state, attestation)
    else:
        process_attestation_phase0(state, attestation)


def process_attestation_phase0(state: "BeaconState", attestation: "Attestation") -> None:
    """Process an attestation in Phase0 style (PendingAttestation)."""
    from ....types.phase0 import PendingAttestation

    data = attestation.data
    current_epoch = get_current_epoch(state)
    previous_epoch = get_previous_epoch(state)

    # Verify attestation target epoch
    assert int(data.target.epoch) in (previous_epoch, current_epoch), (
        "Attestation target epoch not current or previous"
    )
    assert int(data.target.epoch) == compute_epoch_at_slot(int(data.slot)), (
        "Attestation slot not in target epoch"
    )

    # Verify inclusion timing
    assert (
        int(data.slot) + MIN_ATTESTATION_INCLUSION_DELAY
        <= int(state.slot)
        <= int(data.slot) + SLOTS_PER_EPOCH()
    ), "Attestation inclusion delay out of bounds"

    # Verify committee index
    from ...helpers.beacon_committee import get_committee_count_per_slot

    committee_count = get_committee_count_per_slot(state, int(data.target.epoch))
    assert int(data.index) < committee_count, f"Committee index {data.index} >= {committee_count}"

    # Verify aggregation bits length
    committee = get_beacon_committee(state, int(data.slot), int(data.index))
    assert len(attestation.aggregation_bits) == len(committee), (
        f"Aggregation bits length {len(attestation.aggregation_bits)} != "
        f"committee size {len(committee)}"
    )

    # Create pending attestation
    pending_attestation = PendingAttestation(
        aggregation_bits=attestation.aggregation_bits,
        data=data,
        inclusion_delay=int(state.slot) - int(data.slot),
        proposer_index=get_beacon_proposer_index(state),
    )

    # Verify source and add to appropriate list
    if int(data.target.epoch) == current_epoch:
        assert data.source == state.current_justified_checkpoint, (
            "Attestation source doesn't match current justified checkpoint"
        )
        state.current_epoch_attestations.append(pending_attestation)
    else:
        assert data.source == state.previous_justified_checkpoint, (
            "Attestation source doesn't match previous justified checkpoint"
        )
        state.previous_epoch_attestations.append(pending_attestation)

    # Verify signature
    indexed_attestation = get_indexed_attestation(state, attestation)
    assert is_valid_indexed_attestation(state, indexed_attestation), (
        "Invalid indexed attestation"
    )


def process_attestation_altair(state: "BeaconState", attestation: "Attestation") -> None:
    """Process an attestation in Altair+ style (participation flags)."""
    data = attestation.data
    current_epoch = get_current_epoch(state)
    previous_epoch = get_previous_epoch(state)

    # Verify attestation is for current or previous epoch
    assert int(data.target.epoch) in (previous_epoch, current_epoch), (
        f"Attestation target epoch {data.target.epoch} not in "
        f"[{previous_epoch}, {current_epoch}]"
    )

    # Verify attestation slot is within bounds
    attestation_epoch = compute_epoch_at_slot(int(data.slot))
    assert attestation_epoch == int(data.target.epoch), (
        "Attestation slot not in target epoch"
    )

    # Verify attestation is included timely
    # Deneb (EIP-7045) removes the upper bound on attestation inclusion delay
    is_deneb_or_later = (
        (hasattr(state, "latest_execution_payload_header")
         and hasattr(state.latest_execution_payload_header, "blob_gas_used"))
        or hasattr(state, "execution_payload_availability")
    )
    if is_deneb_or_later:
        assert int(data.slot) + MIN_ATTESTATION_INCLUSION_DELAY <= int(state.slot), (
            "Attestation inclusion delay out of bounds"
        )
    else:
        assert int(data.slot) + MIN_ATTESTATION_INCLUSION_DELAY <= int(state.slot) <= int(data.slot) + SLOTS_PER_EPOCH(), (
            "Attestation inclusion delay out of bounds"
        )

    # Verify committee index and aggregation bits length
    if hasattr(attestation, "committee_bits"):
        # Electra attestation with committee_bits
        from ...helpers.attestation import get_committee_indices
        from ...helpers.beacon_committee import get_committee_count_per_slot

        # In Electra, data.index must be 0 (committee selection uses committee_bits)
        # In Gloas (ePBS), data.index can be 0 or 1
        is_gloas = _is_gloas_state(state)
        if not is_gloas:
            assert int(data.index) == 0, (
                f"Attestation data.index must be 0 in Electra, got {data.index}"
            )
        else:
            assert int(data.index) < 2, (
                f"Attestation data.index {data.index} exceeds maximum of 1 in Gloas"
            )

        committee_indices = get_committee_indices(attestation.committee_bits)

        assert len(committee_indices) > 0, "No committee indices set"

        committee_count = get_committee_count_per_slot(state, attestation_epoch)
        for committee_index in committee_indices:
            assert committee_index < committee_count, (
                f"Committee index {committee_index} >= {committee_count}"
            )

        # Get committees and verify aggregation bits
        committees = [
            get_beacon_committee(state, int(data.slot), ci)
            for ci in committee_indices
        ]
        total_committee_size = sum(len(c) for c in committees)

        assert len(attestation.aggregation_bits) == total_committee_size, (
            f"Aggregation bits length {len(attestation.aggregation_bits)} != "
            f"committee size {total_committee_size}"
        )

        # Verify each committee has at least one participant
        agg_bits = list(attestation.aggregation_bits)
        bit_offset = 0
        for committee_index, committee in zip(committee_indices, committees):
            committee_bits = agg_bits[bit_offset:bit_offset + len(committee)]
            assert any(committee_bits), (
                f"Committee {committee_index} has no participating validators"
            )
            bit_offset += len(committee)
    else:
        # Pre-Electra attestation with index field
        from ...helpers.beacon_committee import get_committee_count_per_slot

        committee_count = get_committee_count_per_slot(state, attestation_epoch)
        assert int(data.index) < committee_count, (
            f"Committee index {data.index} >= {committee_count}"
        )

        # Verify aggregation bits length matches committee size
        committee = get_beacon_committee(state, int(data.slot), int(data.index))
        assert len(attestation.aggregation_bits) == len(committee), (
            f"Aggregation bits length {len(attestation.aggregation_bits)} != "
            f"committee size {len(committee)}"
        )

    # Verify source checkpoint
    if int(data.target.epoch) == current_epoch:
        assert data.source == state.current_justified_checkpoint, (
            "Attestation source doesn't match current justified checkpoint"
        )
    else:
        assert data.source == state.previous_justified_checkpoint, (
            "Attestation source doesn't match previous justified checkpoint"
        )

    # Verify indexed attestation
    indexed_attestation = get_indexed_attestation(state, attestation)
    assert is_valid_indexed_attestation(state, indexed_attestation), (
        "Invalid indexed attestation"
    )

    # Get attesting indices
    attesting_indices = get_attesting_indices(state, attestation)

    # Calculate inclusion delay
    inclusion_delay = int(state.slot) - int(data.slot)

    # Get participation flag indices
    participation_flags = get_attestation_participation_flag_indices(
        state, data, inclusion_delay
    )

    # Update epoch participation
    proposer_reward_numerator = 0
    is_current_epoch = int(data.target.epoch) == current_epoch
    is_gloas = _is_gloas_state(state)

    if is_gloas:
        if is_current_epoch:
            payment = state.builder_pending_payments[
                SLOTS_PER_EPOCH() + int(data.slot) % SLOTS_PER_EPOCH()
            ]
        else:
            payment = state.builder_pending_payments[int(data.slot) % SLOTS_PER_EPOCH()]

    for index in attesting_indices:
        if is_current_epoch:
            epoch_participation = state.current_epoch_participation
        else:
            epoch_participation = state.previous_epoch_participation

        if is_gloas:
            will_set_new_flag = False
            for flag_index, weight in enumerate(PARTICIPATION_FLAG_WEIGHTS):
                if flag_index in participation_flags and not has_flag(
                    int(epoch_participation[index]), flag_index
                ):
                    epoch_participation[index] = add_flag(
                        int(epoch_participation[index]), flag_index
                    )
                    base_reward = get_base_reward(state, index)
                    proposer_reward_numerator += base_reward * weight
                    will_set_new_flag = True

            if (
                will_set_new_flag
                and is_attestation_same_slot(state, data)
                and int(payment.withdrawal.amount) > 0
            ):
                payment.weight += state.validators[index].effective_balance
        else:
            for flag_index in participation_flags:
                if not has_flag(int(epoch_participation[index]), flag_index):
                    # Update participation flag
                    epoch_participation[index] = add_flag(
                        int(epoch_participation[index]), flag_index
                    )
                    # Add to proposer reward
                    base_reward = get_base_reward(state, index)
                    proposer_reward_numerator += base_reward * get_flag_weight(flag_index)

    # Reward proposer
    if proposer_reward_numerator > 0:
        proposer_reward = (
            proposer_reward_numerator // (WEIGHT_DENOMINATOR - PROPOSER_WEIGHT)
            * PROPOSER_WEIGHT // WEIGHT_DENOMINATOR
        )
        proposer_index = get_beacon_proposer_index(state)
        increase_balance(state, proposer_index, proposer_reward)

    if is_gloas:
        if is_current_epoch:
            state.builder_pending_payments[
                SLOTS_PER_EPOCH() + int(data.slot) % SLOTS_PER_EPOCH()
            ] = payment
        else:
            state.builder_pending_payments[int(data.slot) % SLOTS_PER_EPOCH()] = payment


def get_flag_weight(flag_index: int) -> int:
    """Get the weight for a participation flag.

    Args:
        flag_index: Flag index

    Returns:
        Weight for the flag
    """
    from ....constants import (
        TIMELY_SOURCE_WEIGHT,
        TIMELY_TARGET_WEIGHT,
        TIMELY_HEAD_WEIGHT,
    )

    if flag_index == TIMELY_SOURCE_FLAG_INDEX:
        return TIMELY_SOURCE_WEIGHT
    elif flag_index == TIMELY_TARGET_FLAG_INDEX:
        return TIMELY_TARGET_WEIGHT
    elif flag_index == TIMELY_HEAD_FLAG_INDEX:
        return TIMELY_HEAD_WEIGHT
    return 0
