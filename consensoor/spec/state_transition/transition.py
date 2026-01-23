"""Main state transition implementation.

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/phase0/beacon-chain.md
Reference (Fulu): https://github.com/ethereum/consensus-specs/blob/master/specs/fulu/beacon-chain.md

Implements the complete state transition function for Ethereum consensus layer.
"""

from typing import TYPE_CHECKING, Optional
import copy

from ..constants import SLOTS_PER_EPOCH, SLOTS_PER_HISTORICAL_ROOT
from .helpers.misc import compute_epoch_at_slot
from .helpers.accessors import get_current_epoch
from ...crypto import hash_tree_root

if TYPE_CHECKING:
    from ..types import BeaconState, BeaconBlock, SignedBeaconBlock


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
    state = copy.deepcopy(state)

    # DEBUG
    from ...crypto import hash_tree_root as _htr
    print(f"DEBUG: state_transition entry, state type={type(state).__name__}, state_hash={_htr(state).hex()[:16]}")
    block = signed_block.message

    # Process slots (including missed slots)
    process_slots(state, int(block.slot))
    print(f"DEBUG: after process_slots, state_hash={_htr(state).hex()[:16]}")

    # Verify block signature (if validating)
    if validate_result:
        verify_block_signature(state, signed_block)

    # Process block
    process_block(state, block)

    # DEBUG
    print(f"DEBUG: after process_block, state_hash={_htr(state).hex()[:16]}, latest_block_header_hash={_htr(state.latest_block_header).hex()[:16]}")

    # Verify state root (if validating)
    if validate_result:
        assert bytes(block.state_root) == hash_tree_root(state), (
            "State root mismatch"
        )

    print(f"DEBUG: state_transition returning, state_hash={_htr(state).hex()}")
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


def process_slots(state: "BeaconState", slot: int) -> None:
    """Process slots up to (but not including) the target slot.

    For each slot, this:
    1. Processes the slot (state caches)
    2. At epoch boundaries, processes epoch transition

    Args:
        state: Beacon state (modified in place)
        slot: Target slot
    """
    assert slot > int(state.slot), (
        f"Target slot {slot} <= state slot {state.slot}"
    )

    while int(state.slot) < slot:
        process_slot(state)

        # At epoch boundary, process epoch
        if (int(state.slot) + 1) % SLOTS_PER_EPOCH() == 0:
            process_epoch(state)

        # Advance slot
        state.slot = int(state.slot) + 1


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

    process_justification_and_finalization(state)
    # Inactivity updates only for Altair+ (uses inactivity_scores)
    if hasattr(state, "inactivity_scores"):
        process_inactivity_updates(state)
    process_rewards_and_penalties(state)
    process_registry_updates(state)
    process_slashings(state)
    process_eth1_data_reset(state)

    # Electra+ epoch processing (before effective balance updates per spec)
    if hasattr(state, "pending_deposits"):
        process_pending_deposits(state)
    if hasattr(state, "pending_consolidations"):
        process_pending_consolidations(state)
    if hasattr(state, "builder_pending_payments"):
        process_builder_pending_payments(state)

    process_effective_balance_updates(state)
    process_slashings_reset(state)
    process_randao_mixes_reset(state)
    process_historical_summaries_update(state)
    # Participation updates (handles both Phase0 and Altair+)
    process_participation_flag_updates(state)

    # Altair+ epoch processing
    if hasattr(state, "current_sync_committee"):
        process_sync_committee_updates(state)

    # Fulu+ epoch processing
    if hasattr(state, "proposer_lookahead"):
        process_proposer_lookahead(state)


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
    from ...crypto import hash_tree_root as _htr

    # Process block header
    process_block_header(state, block)
    print(f"  DEBUG process_block: after header, state={_htr(state).hex()[:16]}")

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
            print(f"  DEBUG process_block: after exec_payload, state={_htr(state).hex()[:16]}")

    # Process randao
    process_randao(state, block.body)
    print(f"  DEBUG process_block: after randao, state={_htr(state).hex()[:16]}")

    # Process eth1 data
    process_eth1_data(state, block.body)
    print(f"  DEBUG process_block: after eth1, state={_htr(state).hex()[:16]}")

    # Process operations
    process_operations(state, block.body, is_gloas=is_gloas_block)
    print(f"  DEBUG process_block: after operations, state={_htr(state).hex()[:16]}")

    # Process sync aggregate (Altair+)
    if hasattr(block.body, "sync_aggregate"):
        process_sync_aggregate(state, block.body.sync_aggregate)
        print(f"  DEBUG process_block: after sync_agg, state={_htr(state).hex()[:16]}")

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

    from ...crypto import hash_tree_root as _htr

    # Process proposer slashings
    for i, proposer_slashing in enumerate(body.proposer_slashings):
        process_proposer_slashing(state, proposer_slashing)
    print(f"    DEBUG ops: after {len(body.proposer_slashings)} prop_slash, state={_htr(state).hex()[:16]}")

    # Process attester slashings
    for i, attester_slashing in enumerate(body.attester_slashings):
        process_attester_slashing(state, attester_slashing)
    print(f"    DEBUG ops: after {len(body.attester_slashings)} att_slash, state={_htr(state).hex()[:16]}")

    # Process attestations
    for i, attestation in enumerate(body.attestations):
        process_attestation(state, attestation)
        if i < 3 or i == len(body.attestations) - 1:
            print(f"    DEBUG ops: after att[{i}], state={_htr(state).hex()[:16]}")

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
    print(f"    DEBUG ops: after {len(body.deposits)} deposits, state={_htr(state).hex()[:16]}")

    # Process voluntary exits
    for voluntary_exit in body.voluntary_exits:
        process_voluntary_exit(state, voluntary_exit)
    print(f"    DEBUG ops: after {len(body.voluntary_exits)} exits, state={_htr(state).hex()[:16]}")

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
