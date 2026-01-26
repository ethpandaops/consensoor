"""Main state transition implementation.

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/phase0/beacon-chain.md
Reference (Fulu): https://github.com/ethereum/consensus-specs/blob/master/specs/fulu/beacon-chain.md

Implements the complete state transition function for Ethereum consensus layer.
"""

import logging
from typing import TYPE_CHECKING, Optional

from ..constants import SLOTS_PER_EPOCH, SLOTS_PER_HISTORICAL_ROOT
from .helpers.misc import compute_epoch_at_slot
from .helpers.accessors import get_current_epoch
from ...crypto import hash_tree_root

if TYPE_CHECKING:
    from ..types import BeaconState, BeaconBlock, SignedBeaconBlock

logger = logging.getLogger(__name__)


def state_transition(
    state: "BeaconState",
    signed_block: "SignedBeaconBlock",
    validate_result: bool = True,
) -> "BeaconState":
    """Execute the state transition for a block.

    This is the main entry point for state transition. It:
    1. Advances the state to the block's slot (process_slots)
    2. Verifies and processes the block (process_block)
    3. Optionally validates the state root

    Args:
        state: Pre-state (not modified)
        signed_block: Signed beacon block to process
        validate_result: If True, verify state root matches

    Returns:
        Post-state after applying the block

    Raises:
        AssertionError: If validation fails
    """
    # Work on a copy to preserve original state
    # Use SSZ round-trip instead of copy.deepcopy for deterministic behavior
    state = state.__class__.decode_bytes(bytes(state.encode_bytes()))

    block = signed_block.message

    # Process slots (including missed slots) - may return upgraded state type
    state = process_slots(state, int(block.slot))

    # Verify block signature (if validating)
    if validate_result:
        verify_block_signature(state, signed_block)

    # Process block
    process_block(state, block)

    # Verify state root (if validating)
    if validate_result:
        assert bytes(block.state_root) == hash_tree_root(state), (
            "State root mismatch"
        )

    return state


def verify_block_signature(
    state: "BeaconState", signed_block: "SignedBeaconBlock"
) -> None:
    """Verify the block's proposer signature.

    Args:
        state: Beacon state
        signed_block: Signed beacon block

    Raises:
        AssertionError: If signature is invalid
    """
    from .helpers.beacon_committee import get_beacon_proposer_index
    from .helpers.domain import get_domain, compute_signing_root
    from ..constants import DOMAIN_BEACON_PROPOSER
    from ...crypto import bls_verify

    proposer_index = get_beacon_proposer_index(state)
    proposer = state.validators[proposer_index]

    domain = get_domain(state, DOMAIN_BEACON_PROPOSER)
    signing_root = compute_signing_root(signed_block.message, domain)

    assert bls_verify(
        [bytes(proposer.pubkey)],
        signing_root,
        bytes(signed_block.signature),
    ), "Invalid block signature"


def process_slots(state: "BeaconState", slot: int) -> "BeaconState":
    """Process slots up to (but not including) the target slot.

    For each slot, this:
    1. Processes the slot (state caches)
    2. At epoch boundaries, processes epoch transition
    3. At fork boundaries, upgrades state to new fork type

    Args:
        state: Beacon state
        slot: Target slot

    Returns:
        The state after processing (may be a different type after fork upgrade)
    """
    import time
    assert slot > int(state.slot), (
        f"Target slot {slot} <= state slot {state.slot}"
    )

    start_slot = int(state.slot)
    slots_to_process = slot - start_slot
    if slots_to_process > 10:
        logger.warning(f"process_slots: processing {slots_to_process} slots ({start_slot} -> {slot})")

    slot_times = []
    epoch_times = []

    while int(state.slot) < slot:
        t0 = time.time()
        process_slot(state)
        slot_time = time.time() - t0
        slot_times.append(slot_time)

        # At epoch boundary, process epoch and check for fork upgrade
        if (int(state.slot) + 1) % SLOTS_PER_EPOCH() == 0:
            t0 = time.time()
            process_epoch(state)
            epoch_time = time.time() - t0
            epoch_times.append(epoch_time)
            next_epoch = (int(state.slot) + 1) // SLOTS_PER_EPOCH()
            state = upgrade_fork_if_needed(state, next_epoch)

        # Advance slot
        state.slot = int(state.slot) + 1

    if slots_to_process > 10:
        avg_slot = sum(slot_times) / len(slot_times) if slot_times else 0
        avg_epoch = sum(epoch_times) / len(epoch_times) if epoch_times else 0
        logger.warning(
            f"process_slots done: {slots_to_process} slots, "
            f"avg_slot={avg_slot*1000:.1f}ms, avg_epoch={avg_epoch*1000:.1f}ms, "
            f"total_slot={sum(slot_times)*1000:.1f}ms, total_epoch={sum(epoch_times)*1000:.1f}ms"
        )

    return state


def upgrade_fork_if_needed(state: "BeaconState", epoch: int) -> "BeaconState":
    """Upgrade state if a fork transition occurs at this epoch.

    Per the consensus spec, when crossing a fork boundary, the state must be
    upgraded to the new fork's state type with appropriate new fields initialized.
    This is necessary for correct domain computation and new fork features.

    Args:
        state: Beacon state
        epoch: The epoch we're transitioning into

    Returns:
        Upgraded state if a fork occurs, otherwise the original state
    """
    from .fork_upgrade import maybe_upgrade_state
    return maybe_upgrade_state(state, epoch)


def process_slot(state: "BeaconState") -> None:
    """Process a single slot (cache updates).

    Updates:
    - block_roots: previous slot's block root
    - state_roots: previous slot's state root

    Args:
        state: Beacon state (modified in place)
    """
    # Cache state root
    previous_state_root = hash_tree_root(state)
    state.state_roots[int(state.slot) % SLOTS_PER_HISTORICAL_ROOT()] = previous_state_root

    # Cache latest block header state root if empty
    if bytes(state.latest_block_header.state_root) == b"\x00" * 32:
        state.latest_block_header.state_root = previous_state_root

    # Cache block root
    previous_block_root = hash_tree_root(state.latest_block_header)
    state.block_roots[int(state.slot) % SLOTS_PER_HISTORICAL_ROOT()] = previous_block_root

    if hasattr(state, "execution_payload_availability"):
        next_slot_index = (int(state.slot) + 1) % SLOTS_PER_HISTORICAL_ROOT()
        state.execution_payload_availability[next_slot_index] = 0


def process_epoch(state: "BeaconState") -> None:
    """Process epoch transition.

    Executes all epoch processing in order:
    1. Justification and finalization
    2. Inactivity updates
    3. Rewards and penalties
    4. Registry updates
    5. Slashings
    6. Eth1 data reset
    7. Effective balance updates
    8. Slashings reset
    9. Randao mixes reset
    10. Historical summaries update
    11. Participation flag updates
    12. Sync committee updates (Altair+)
    13. Pending deposits (Electra+)
    14. Pending consolidations (Electra+)
    15. Proposer lookahead (Fulu+)

    Args:
        state: Beacon state (modified in place)
    """
    from .epoch import (
        process_justification_and_finalization,
        process_inactivity_updates,
        process_rewards_and_penalties,
        process_registry_updates,
        process_slashings,
        process_effective_balance_updates,
        process_participation_flag_updates,
        process_sync_committee_updates,
        process_eth1_data_reset,
        process_slashings_reset,
        process_randao_mixes_reset,
        process_historical_summaries_update,
        process_pending_deposits,
        process_pending_consolidations,
        process_builder_pending_payments,
        process_proposer_lookahead,
    )

    import time
    timings = {}

    t0 = time.time()
    process_justification_and_finalization(state)
    timings['justification'] = time.time() - t0

    # Inactivity updates only for Altair+ (uses inactivity_scores)
    if hasattr(state, "inactivity_scores"):
        t0 = time.time()
        process_inactivity_updates(state)
        timings['inactivity'] = time.time() - t0

    t0 = time.time()
    process_rewards_and_penalties(state)
    timings['rewards'] = time.time() - t0

    t0 = time.time()
    process_registry_updates(state)
    timings['registry'] = time.time() - t0

    t0 = time.time()
    process_slashings(state)
    timings['slashings'] = time.time() - t0

    t0 = time.time()
    process_eth1_data_reset(state)
    timings['eth1_reset'] = time.time() - t0

    # Electra+ epoch processing (before effective balance updates per spec)
    if hasattr(state, "pending_deposits"):
        t0 = time.time()
        process_pending_deposits(state)
        timings['pending_deposits'] = time.time() - t0

    if hasattr(state, "pending_consolidations"):
        t0 = time.time()
        process_pending_consolidations(state)
        timings['pending_consolidations'] = time.time() - t0

    if hasattr(state, "builder_pending_payments"):
        t0 = time.time()
        process_builder_pending_payments(state)
        timings['builder_payments'] = time.time() - t0

    t0 = time.time()
    process_effective_balance_updates(state)
    timings['effective_balance'] = time.time() - t0

    t0 = time.time()
    process_slashings_reset(state)
    timings['slashings_reset'] = time.time() - t0

    t0 = time.time()
    process_randao_mixes_reset(state)
    timings['randao_reset'] = time.time() - t0

    t0 = time.time()
    process_historical_summaries_update(state)
    timings['historical_summaries'] = time.time() - t0

    # Participation updates (handles both Phase0 and Altair+)
    t0 = time.time()
    process_participation_flag_updates(state)
    timings['participation'] = time.time() - t0

    # Altair+ epoch processing
    if hasattr(state, "current_sync_committee"):
        t0 = time.time()
        process_sync_committee_updates(state)
        timings['sync_committee'] = time.time() - t0

    # Fulu+ epoch processing
    if hasattr(state, "proposer_lookahead"):
        t0 = time.time()
        process_proposer_lookahead(state)
        timings['proposer_lookahead'] = time.time() - t0

    # Log slow operations (>10ms)
    slow_ops = {k: v*1000 for k, v in timings.items() if v > 0.01}
    if slow_ops:
        logger.debug(f"process_epoch timings (ms): {slow_ops}")


def process_block(state: "BeaconState", block: "BeaconBlock") -> None:
    """Process a beacon block.

    Executes all block processing in order:
    1. Block header
    2. Withdrawals (Capella+, not in Gloas - handled via envelope)
    3. Execution payload (Bellatrix+) OR Execution payload header (Gloas)
    4. Randao
    5. Eth1 data
    6. Operations
    7. Sync aggregate (Altair+)
    8. Payload attestations (Gloas)

    Args:
        state: Beacon state (modified in place)
        block: Beacon block to process

    Raises:
        AssertionError: If any validation fails
    """
    from .block import (
        process_block_header,
        process_randao,
        process_eth1_data,
        process_execution_payload,
        process_withdrawals,
        process_sync_aggregate,
        process_execution_payload_bid,
        process_payload_attestations,
    )
    from .helpers.predicates import is_execution_enabled

    # Process block header
    process_block_header(state, block)

    # Check if this is a Gloas block (ePBS)
    is_gloas_block = hasattr(block.body, "signed_execution_payload_bid")

    if is_gloas_block:
        # Gloas (ePBS): process withdrawals and execution payload bid
        process_withdrawals(state)
        process_execution_payload_bid(state, block)
    else:
        # Pre-Gloas: process execution payload (Bellatrix+)
        # Only process if execution is enabled (merge complete or merge transition block)
        if hasattr(block.body, "execution_payload") and is_execution_enabled(state, block.body):
            # Process withdrawals first (Capella+)
            if hasattr(block.body.execution_payload, "withdrawals"):
                process_withdrawals(state, block.body.execution_payload)

            process_execution_payload(state, block.body)

    # Process randao
    process_randao(state, block.body)

    # Process eth1 data
    process_eth1_data(state, block.body)

    # Process operations
    process_operations(state, block.body, is_gloas=is_gloas_block)

    # Process sync aggregate (Altair+)
    if hasattr(block.body, "sync_aggregate"):
        process_sync_aggregate(state, block.body.sync_aggregate)

    # Process payload attestations (Gloas)
    if is_gloas_block and hasattr(block.body, "payload_attestations"):
        process_payload_attestations(state, list(block.body.payload_attestations))


def process_operations(state: "BeaconState", body, is_gloas: bool = False) -> None:
    """Process all block operations.

    Processes in order:
    1. Proposer slashings
    2. Attester slashings
    3. Attestations
    4. Deposits
    5. Voluntary exits
    6. BLS to execution changes (Capella+)
    7. Execution requests (Electra+, not in Gloas - handled via envelope)

    Args:
        state: Beacon state (modified in place)
        body: Block body containing operations
        is_gloas: If True, skip execution requests (processed via payload envelope)

    Raises:
        AssertionError: If any operation validation fails
    """
    from .block.operations import (
        process_proposer_slashing,
        process_attester_slashing,
        process_attestation,
        process_deposit,
        process_voluntary_exit,
        process_bls_to_execution_change,
        process_deposit_request,
        process_withdrawal_request,
        process_consolidation_request,
    )
    from ..constants import (
        MAX_PROPOSER_SLASHINGS,
        MAX_ATTESTER_SLASHINGS,
        MAX_ATTESTER_SLASHINGS_PRE_ELECTRA,
        MAX_ATTESTATIONS,
        MAX_ATTESTATIONS_PRE_ELECTRA,
        MAX_DEPOSITS,
        MAX_VOLUNTARY_EXITS,
        MAX_BLS_TO_EXECUTION_CHANGES,
    )

    # Determine fork-appropriate limits
    is_electra = hasattr(state, "pending_deposits")
    max_attester_slashings = MAX_ATTESTER_SLASHINGS if is_electra else MAX_ATTESTER_SLASHINGS_PRE_ELECTRA
    max_attestations = MAX_ATTESTATIONS if is_electra else MAX_ATTESTATIONS_PRE_ELECTRA

    # Verify operation counts don't exceed limits
    assert len(body.proposer_slashings) <= MAX_PROPOSER_SLASHINGS, (
        f"Too many proposer slashings: {len(body.proposer_slashings)}"
    )
    assert len(body.attester_slashings) <= max_attester_slashings, (
        f"Too many attester slashings: {len(body.attester_slashings)}"
    )
    assert len(body.attestations) <= max_attestations, (
        f"Too many attestations: {len(body.attestations)}"
    )
    assert len(body.deposits) <= MAX_DEPOSITS, (
        f"Too many deposits: {len(body.deposits)}"
    )
    assert len(body.voluntary_exits) <= MAX_VOLUNTARY_EXITS, (
        f"Too many voluntary exits: {len(body.voluntary_exits)}"
    )

    # Process proposer slashings
    for proposer_slashing in body.proposer_slashings:
        process_proposer_slashing(state, proposer_slashing)

    # Process attester slashings
    for attester_slashing in body.attester_slashings:
        process_attester_slashing(state, attester_slashing)

    # Process attestations
    for attestation in body.attestations:
        process_attestation(state, attestation)

    # Process deposits
    if hasattr(state, "deposit_requests_start_index"):
        # Electra+: disable former deposit mechanism once all prior deposits are processed
        eth1_deposit_index_limit = min(
            int(state.eth1_data.deposit_count),
            int(state.deposit_requests_start_index),
        )
        if int(state.eth1_deposit_index) < eth1_deposit_index_limit:
            expected_deposits = min(
                MAX_DEPOSITS,
                eth1_deposit_index_limit - int(state.eth1_deposit_index),
            )
            assert len(body.deposits) == expected_deposits, (
                f"Deposit count mismatch: got {len(body.deposits)}, "
                f"expected {expected_deposits}"
            )
        else:
            assert len(body.deposits) == 0, (
                f"Expected 0 deposits after transition, got {len(body.deposits)}"
            )
    else:
        # Pre-Electra: verify deposit count matches expectations
        expected_deposits = min(
            MAX_DEPOSITS,
            int(state.eth1_data.deposit_count) - int(state.eth1_deposit_index),
        )
        assert len(body.deposits) == expected_deposits, (
            f"Deposit count mismatch: got {len(body.deposits)}, "
            f"expected {expected_deposits}"
        )

    for deposit in body.deposits:
        process_deposit(state, deposit)

    # Process voluntary exits
    for voluntary_exit in body.voluntary_exits:
        process_voluntary_exit(state, voluntary_exit)

    # Process BLS to execution changes (Capella+)
    if hasattr(body, "bls_to_execution_changes"):
        assert len(body.bls_to_execution_changes) <= MAX_BLS_TO_EXECUTION_CHANGES, (
            f"Too many BLS to execution changes: {len(body.bls_to_execution_changes)}"
        )
        for signed_change in body.bls_to_execution_changes:
            process_bls_to_execution_change(state, signed_change)

    # Process execution requests (Electra+)
    # In Gloas (ePBS), execution requests are processed via the payload envelope,
    # not during block processing
    if not is_gloas and hasattr(body, "execution_requests"):
        requests = body.execution_requests

        # Process deposit requests
        if hasattr(requests, "deposits"):
            for deposit_request in requests.deposits:
                process_deposit_request(state, deposit_request)

        # Process withdrawal requests
        if hasattr(requests, "withdrawals"):
            for withdrawal_request in requests.withdrawals:
                process_withdrawal_request(state, withdrawal_request)

        # Process consolidation requests
        if hasattr(requests, "consolidations"):
            for consolidation_request in requests.consolidations:
                process_consolidation_request(state, consolidation_request)
