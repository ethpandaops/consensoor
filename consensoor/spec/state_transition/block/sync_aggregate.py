"""Sync aggregate processing (Altair+).

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/altair/beacon-chain.md
"""

from typing import TYPE_CHECKING

from ...constants import (
    DOMAIN_SYNC_COMMITTEE,
    SYNC_COMMITTEE_SIZE,
    SYNC_REWARD_WEIGHT,
    PROPOSER_WEIGHT,
    WEIGHT_DENOMINATOR,
    SLOTS_PER_EPOCH,
    EFFECTIVE_BALANCE_INCREMENT,
)
from ..helpers.accessors import get_current_epoch, get_total_active_balance, get_base_reward_per_increment
from ..helpers.misc import compute_epoch_at_slot
from ..helpers.domain import get_domain, compute_signing_root
from ..helpers.beacon_committee import get_beacon_proposer_index
from ..helpers.mutators import increase_balance, decrease_balance
from ....crypto import bls_verify, bls_aggregate_pubkeys

if TYPE_CHECKING:
    from ...types import BeaconState
    from ...types.altair import SyncAggregate


def process_sync_aggregate(
    state: "BeaconState", sync_aggregate: "SyncAggregate"
) -> None:
    """Process the sync aggregate from the block (Altair+).

    Validates the aggregate signature and distributes rewards to
    participating sync committee members and the proposer.

    Args:
        state: Beacon state (modified in place)
        sync_aggregate: Sync aggregate from block

    Raises:
        AssertionError: If signature verification fails
    """
    # Get sync committee participant pubkeys
    committee_pubkeys = list(state.current_sync_committee.pubkeys)
    participant_pubkeys = [
        committee_pubkeys[i]
        for i in range(SYNC_COMMITTEE_SIZE())
        if sync_aggregate.sync_committee_bits[i]
    ]

    # Verify signature
    previous_slot = max(int(state.slot), 1) - 1
    from ..helpers.accessors import get_block_root_at_slot

    # The signature is over the previous slot's block root
    # Domain must use the epoch of previous_slot, not current slot (per Altair spec)
    domain = get_domain(
        state,
        DOMAIN_SYNC_COMMITTEE,
        compute_epoch_at_slot(previous_slot),
    )

    # Get the block root being signed
    if previous_slot < int(state.slot):
        signing_root = compute_signing_root(
            get_block_root_at_slot(state, previous_slot),
            domain,
        )
    else:
        # At genesis, sign the genesis block root
        from ....crypto import hash_tree_root

        signing_root = compute_signing_root(
            hash_tree_root(state.latest_block_header),
            domain,
        )

    # Verify aggregate signature (even with zero participants)
    import logging
    _logger = logging.getLogger(__name__)
    sig_bytes = bytes(sync_aggregate.sync_committee_signature)
    g2_infinity = b'\xc0' + b'\x00' * 95
    _logger.debug(
        f"Sync aggregate verify: participants={len(participant_pubkeys)}, "
        f"state_slot={int(state.slot)}, previous_slot={previous_slot}, "
        f"sig_is_infinity={sig_bytes == g2_infinity}, "
        f"sig_len={len(sig_bytes)}, sig_prefix={sig_bytes[:4].hex()}"
    )
    verify_result = bls_verify(
        [bytes(pk) for pk in participant_pubkeys],
        signing_root,
        sig_bytes,
    )
    if not verify_result:
        _logger.error(
            f"Sync aggregate signature verification FAILED: "
            f"participants={len(participant_pubkeys)}, sig_bytes[:8]={sig_bytes[:8].hex()}"
        )
    assert verify_result, "Invalid sync aggregate signature"

    # Compute participant and proposer rewards
    total_active_balance = get_total_active_balance(state)
    total_active_increments = total_active_balance // EFFECTIVE_BALANCE_INCREMENT
    total_base_rewards = get_base_reward_per_increment(state) * total_active_increments
    max_participant_rewards = (
        total_base_rewards * SYNC_REWARD_WEIGHT // WEIGHT_DENOMINATOR // SLOTS_PER_EPOCH()
    )
    participant_reward = max_participant_rewards // SYNC_COMMITTEE_SIZE()
    proposer_reward = (
        participant_reward * PROPOSER_WEIGHT // (WEIGHT_DENOMINATOR - PROPOSER_WEIGHT)
    )

    # Apply participant and proposer rewards
    # Get committee indices by looking up pubkeys in validator list
    all_pubkeys = [bytes(v.pubkey) for v in state.validators]
    committee_indices = []
    for pubkey in state.current_sync_committee.pubkeys:
        pk_bytes = bytes(pubkey)
        if pk_bytes in all_pubkeys:
            committee_indices.append(all_pubkeys.index(pk_bytes))
        else:
            committee_indices.append(0)

    proposer_index = get_beacon_proposer_index(state)

    for participant_index, participation_bit in zip(
        committee_indices, sync_aggregate.sync_committee_bits
    ):
        if participation_bit:
            increase_balance(state, participant_index, participant_reward)
            increase_balance(state, proposer_index, proposer_reward)
        else:
            decrease_balance(state, participant_index, participant_reward)


def get_sync_committee_participant_indices(state: "BeaconState") -> list:
    """Get validator indices for the current sync committee.

    Args:
        state: Beacon state

    Returns:
        List of validator indices in the sync committee
    """
    committee_pubkeys = [bytes(pk) for pk in state.current_sync_committee.pubkeys]
    validator_pubkeys = {
        bytes(v.pubkey): i for i, v in enumerate(state.validators)
    }

    return [
        validator_pubkeys.get(pk, 0)
        for pk in committee_pubkeys
    ]
