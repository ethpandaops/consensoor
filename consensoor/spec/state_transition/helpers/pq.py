"""PQ helpers (leanXMSS leaves, VRF proofs, aggregation committees, preregistrations)."""

from typing import TYPE_CHECKING, Optional, Sequence

from ...constants import (
    AGGREGATORS_PER_COMMITTEE,
    DOMAIN_BEACON_ATTESTER,
    DOMAIN_SELECTION_PROOF,
    LEGACY_VRF_ROOT_DOMAIN,
    LEGACY_XMSS_PUBKEY_DOMAIN,
    MIN_SEED_LOOKAHEAD,
    SLOTS_PER_EPOCH,
    SLOTS_PER_HISTORICAL_ROOT,
    VRF_TREE_DEPTH,
    XMSS_ATTESTATION_LEAVES,
    XMSS_BUILDER_BID_LEAVES,
    XMSS_DEPOSIT_PREFIX,
    XMSS_LEAVES_PER_SLOT,
    XMSS_RESERVED_LEAVES,
)
from ....crypto import sha256, xmss

if TYPE_CHECKING:
    from ...types.pq import BeaconState

XMSSAggregateStatement = tuple[Sequence[bytes], int, bytes]


def is_pq_state(state) -> bool:
    return hasattr(state, "aggregation_committee_window")


def is_pq_body(body) -> bool:
    return hasattr(body, "proof")


def compute_legacy_xmss_pubkey(pubkey: bytes) -> bytes:
    return sha256(bytes(pubkey) + LEGACY_XMSS_PUBKEY_DOMAIN)


def compute_legacy_vrf_root(pubkey: bytes) -> bytes:
    return sha256(bytes(pubkey) + LEGACY_VRF_ROOT_DOMAIN)


def is_legacy_pubkey(pubkey: bytes) -> bool:
    return bytes(pubkey)[:16] != XMSS_DEPOSIT_PREFIX


def compute_deposit_pubkey(pubkey: bytes) -> bytes:
    if is_legacy_pubkey(pubkey):
        return compute_legacy_xmss_pubkey(pubkey)
    return bytes(pubkey)[16:]


def compute_deposit_vrf_root(deposit) -> bytes:
    if is_legacy_pubkey(deposit.pubkey):
        return compute_legacy_vrf_root(deposit.pubkey)
    return bytes(deposit.signature)[:32]


def compute_xmss_attestation_leaf_index(epoch: int) -> int:
    return XMSS_RESERVED_LEAVES + int(epoch)


def compute_xmss_leaf_index(slot: int, leaf_offset: int) -> int:
    return (
        XMSS_RESERVED_LEAVES
        + XMSS_ATTESTATION_LEAVES
        + XMSS_BUILDER_BID_LEAVES
        + int(slot) * XMSS_LEAVES_PER_SLOT
        + leaf_offset
    )


def is_builder_bid_leaf_index(leaf_index: int) -> bool:
    start = XMSS_RESERVED_LEAVES + XMSS_ATTESTATION_LEAVES
    return start <= leaf_index < start + XMSS_BUILDER_BID_LEAVES


def is_valid_vrf_proof(state: "BeaconState", validator_index: int, slot: int, proof) -> bool:
    from .predicates import is_valid_merkle_branch

    nodes = [bytes(node) for node in proof]
    return is_valid_merkle_branch(
        leaf=nodes[0],
        branch=nodes[1:],
        depth=VRF_TREE_DEPTH,
        index=int(slot),
        root=bytes(state.validators[validator_index].vrf_root),
    )


def compute_aggregation_committee(state: "BeaconState", slot: int, index: int) -> list[int]:
    from .accessors import get_seed
    from .beacon_committee import compute_balance_weighted_selection, get_beacon_committee
    from .misc import compute_epoch_at_slot

    epoch = compute_epoch_at_slot(slot)
    seed = sha256(
        get_seed(state, epoch, DOMAIN_SELECTION_PROOF)
        + int(slot).to_bytes(8, "little")
        + int(index).to_bytes(8, "little")
    )
    candidates = [int(v) for v in get_beacon_committee(state, slot, index)]
    aggregators: list[int] = []
    while len(aggregators) < AGGREGATORS_PER_COMMITTEE and len(candidates) > 0:
        aggregator = compute_balance_weighted_selection(
            state,
            candidates,
            sha256(seed + len(aggregators).to_bytes(8, "little")),
            size=1,
            shuffle_indices=False,
        )[0]
        aggregators.append(aggregator)
        candidates = [candidate for candidate in candidates if candidate != aggregator]
    return aggregators


def compute_aggregation_committees(state: "BeaconState", slot: int) -> list[list[int]]:
    from .beacon_committee import get_committee_count_per_slot
    from .misc import compute_epoch_at_slot

    committees_per_slot = get_committee_count_per_slot(state, compute_epoch_at_slot(slot))
    return [
        compute_aggregation_committee(state, slot, index)
        for index in range(committees_per_slot)
    ]


def get_aggregation_committee(state: "BeaconState", slot: int, index: int) -> list[int]:
    from .accessors import get_current_epoch
    from .misc import compute_epoch_at_slot

    epoch = compute_epoch_at_slot(slot)
    state_epoch = get_current_epoch(state)
    spe = SLOTS_PER_EPOCH()
    if epoch < state_epoch:
        assert epoch + 1 == state_epoch
        return [int(v) for v in state.aggregation_committee_window[int(slot) % spe][index]]
    assert epoch <= state_epoch + MIN_SEED_LOOKAHEAD
    offset = (epoch - state_epoch + 1) * spe
    return [int(v) for v in state.aggregation_committee_window[offset + int(slot) % spe][index]]


def is_aggregator(state: "BeaconState", slot: int, index: int, validator_index: int) -> bool:
    return int(validator_index) in get_aggregation_committee(state, slot, index)


def initialize_aggregation_committee_window(state: "BeaconState") -> list[list[list[int]]]:
    from .accessors import get_current_epoch
    from .misc import compute_start_slot_at_epoch

    spe = SLOTS_PER_EPOCH()
    current_epoch = get_current_epoch(state)
    start_slot = compute_start_slot_at_epoch(current_epoch)
    end_slot = compute_start_slot_at_epoch(current_epoch + MIN_SEED_LOOKAHEAD + 1)
    window: list[list[list[int]]] = [[] for _ in range(spe)]
    window += [compute_aggregation_committees(state, slot) for slot in range(start_slot, end_slot)]
    return window


def get_attestation_statements(state: "BeaconState", attestation) -> list[XMSSAggregateStatement]:
    from .attestation import get_committee_indices
    from .beacon_committee import get_beacon_committee
    from .domain import compute_signing_root, get_domain

    data = attestation.data
    target_epoch = int(data.target.epoch)
    domain = get_domain(state, DOMAIN_BEACON_ATTESTER, target_epoch)
    message = compute_signing_root(data, domain)
    leaf_index = compute_xmss_attestation_leaf_index(target_epoch)
    bits = attestation.aggregation_bits
    statements: list[XMSSAggregateStatement] = []
    committee_offset = 0
    for committee_index in get_committee_indices(attestation.committee_bits):
        committee = get_beacon_committee(state, int(data.slot), committee_index)
        pubkeys = [
            bytes(state.validators[validator_index].pubkey)
            for i, validator_index in enumerate(committee)
            if bits[committee_offset + i]
        ]
        statements.append((pubkeys, leaf_index, message))
        committee_offset += len(committee)
    return statements


def get_latest_block_slot(state: "BeaconState", slot: int, parent_slot: int) -> int:
    from .accessors import get_block_root_at_slot

    slot = int(slot)
    if parent_slot <= slot:
        return int(parent_slot)
    block_root = bytes(get_block_root_at_slot(state, slot))
    while (
        slot > 0
        and slot + SLOTS_PER_HISTORICAL_ROOT() > int(state.slot)
        and bytes(get_block_root_at_slot(state, slot - 1)) == block_root
    ):
        slot -= 1
    return slot


def _is_active_preregistration(state: "BeaconState", preregistration) -> bool:
    return int(state.latest_execution_payload_bid.slot) < int(preregistration.expiry_slot)


is_active_validator_preregistration = _is_active_preregistration
is_active_builder_preregistration = _is_active_preregistration


def _preregistration_index(preregistrations, pubkey: bytes) -> Optional[int]:
    pubkey = bytes(pubkey)
    for index, preregistration in enumerate(preregistrations):
        if bytes(preregistration.pubkey) == pubkey:
            return index
    return None


def get_validator_preregistration_index(state: "BeaconState", pubkey: bytes) -> Optional[int]:
    return _preregistration_index(state.validator_preregistrations, pubkey)


def get_builder_preregistration_index(state: "BeaconState", pubkey: bytes) -> Optional[int]:
    return _preregistration_index(state.builder_preregistrations, pubkey)


def get_active_validator_preregistration(state: "BeaconState", pubkey: bytes):
    index = get_validator_preregistration_index(state, pubkey)
    if index is None:
        return None
    preregistration = state.validator_preregistrations[index]
    if not is_active_validator_preregistration(state, preregistration):
        return None
    return preregistration


def get_active_builder_preregistration(state: "BeaconState", pubkey: bytes):
    index = get_builder_preregistration_index(state, pubkey)
    if index is None:
        return None
    preregistration = state.builder_preregistrations[index]
    if not is_active_builder_preregistration(state, preregistration):
        return None
    return preregistration


def remove_validator_preregistration(state: "BeaconState", pubkey: bytes) -> None:
    pubkey = bytes(pubkey)
    state.validator_preregistrations = [
        p for p in state.validator_preregistrations if bytes(p.pubkey) != pubkey
    ]


def remove_builder_preregistration(state: "BeaconState", pubkey: bytes) -> None:
    pubkey = bytes(pubkey)
    state.builder_preregistrations = [
        p for p in state.builder_preregistrations if bytes(p.pubkey) != pubkey
    ]


def is_pending_validator(pending_deposits, pubkey: bytes) -> bool:
    pubkey = bytes(pubkey)
    for pending_deposit in pending_deposits:
        if compute_deposit_pubkey(pending_deposit.pubkey) == pubkey:
            return True
    return False


def verify_xmss(pubkey: bytes, leaf_index: int, signing_root: bytes, signature: bytes) -> bool:
    return xmss.verify(bytes(pubkey), leaf_index, signing_root, bytes(signature))
