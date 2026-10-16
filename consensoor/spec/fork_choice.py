"""Spec fork choice (phase0 base + Gloas/ePBS modifications).

A faithful port of ``specs/phase0/fork-choice.md`` with the Gloas overrides
from ``specs/gloas/fork-choice.md`` (consensus-specs v1.7.0-beta.4):
``Store``, ``on_tick``/``on_block``/``on_attestation``/``on_attester_slashing``,
``on_execution_payload_envelope``/``on_payload_attestation_message`` and
``get_head`` over (root, payload_status) fork-choice nodes.

Times are seconds (``store.time``) like the spec; checkpoints are plain frozen
dataclasses so they can key dicts. Implementation-dependent hooks
(``is_data_available``, ``verify_new_payload``) are injectable callables that
default to "yes" — the node wires them to its DAS manager / execution engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from ..crypto import hash_tree_root
from .constants import EFFECTIVE_BALANCE_INCREMENT, GENESIS_EPOCH, GENESIS_SLOT, MIN_SEED_LOOKAHEAD, PTC_SIZE, SLOTS_PER_EPOCH
from .network_config import get_config
from .state_transition.epoch.justification import process_justification_and_finalization
from .state_transition.helpers.accessors import (
    get_active_validator_indices,
    get_current_epoch,
)
from .state_transition.helpers.attestation import get_indexed_attestation
from .state_transition.helpers.beacon_committee import get_beacon_committee, get_committee_count_per_slot
from .state_transition.helpers.misc import compute_epoch_at_slot, compute_start_slot_at_epoch, compute_time_at_slot
from .state_transition.helpers.predicates import (
    is_slashable_attestation_data,
    is_valid_indexed_attestation,
    is_valid_indexed_payload_attestation,
)
from .state_transition.helpers.ptc import get_indexed_payload_attestation, get_ptc
from .state_transition.transition import process_slots, state_transition

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- constants

PAYLOAD_STATUS_EMPTY = 0
PAYLOAD_STATUS_FULL = 1
PAYLOAD_STATUS_PENDING = 2

ATTESTATION_TIMELINESS_INDEX = 0
PTC_TIMELINESS_INDEX = 1

BASIS_POINTS = 10000
UINT64_MAX = 2**64 - 1
ZERO_ROOT = b"\x00" * 32


def SLOT_DURATION_MS() -> int:
    return int(get_config().slot_duration_ms)


def PAYLOAD_TIMELY_THRESHOLD() -> int:
    return PTC_SIZE() // 2


def DATA_AVAILABILITY_TIMELY_THRESHOLD() -> int:
    return PTC_SIZE() // 2


def PROPOSER_SCORE_BOOST() -> int:
    return int(getattr(get_config(), "proposer_score_boost", 40))


def REORG_HEAD_WEIGHT_THRESHOLD() -> int:
    return int(getattr(get_config(), "reorg_head_weight_threshold", 20))


def REORG_PARENT_WEIGHT_THRESHOLD() -> int:
    return int(getattr(get_config(), "reorg_parent_weight_threshold", 160))


def REORG_MAX_EPOCHS_SINCE_FINALIZATION() -> int:
    return int(getattr(get_config(), "reorg_max_epochs_since_finalization", 2))


# ---------------------------------------------------------------- data types


@dataclass(eq=True, frozen=True)
class Checkpoint:
    epoch: int
    root: bytes


def cp(checkpoint) -> Checkpoint:
    """Convert an SSZ Checkpoint (or Checkpoint) into the hashable dataclass."""
    if isinstance(checkpoint, Checkpoint):
        return checkpoint
    return Checkpoint(epoch=int(checkpoint.epoch), root=bytes(checkpoint.root))


@dataclass(eq=True, frozen=True)
class ForkChoiceNode:
    root: bytes
    payload_status: int = PAYLOAD_STATUS_PENDING


@dataclass(eq=True, frozen=True)
class LatestMessage:
    slot: int
    root: bytes
    payload_present: bool


@dataclass
class Store:
    time: int
    genesis_time: int
    justified_checkpoint: Checkpoint
    finalized_checkpoint: Checkpoint
    unrealized_justified_checkpoint: Checkpoint
    unrealized_finalized_checkpoint: Checkpoint
    proposer_boost_root: bytes
    equivocating_indices: Set[int]
    blocks: Dict[bytes, object]
    block_states: Dict[bytes, object]
    block_timeliness: Dict[bytes, List[bool]]
    checkpoint_states: Dict[Checkpoint, object]
    latest_messages: Dict[int, LatestMessage]
    unrealized_justifications: Dict[bytes, Checkpoint]
    payloads: Dict[bytes, object]
    payload_timeliness_vote: Dict[bytes, List[Optional[bool]]]
    payload_data_availability_vote: Dict[bytes, List[Optional[bool]]]
    # [New in Heze:EIP7805] beacon block root -> payload satisfied the ILs
    payload_inclusion_list_satisfaction: Dict[bytes, bool] = field(default_factory=dict)
    # implementation hooks
    is_data_available: Callable[[bytes], bool] = field(default=lambda root: True)
    verify_new_payload: Callable[..., bool] = field(default=lambda *a, **k: True)
    # [New in Heze:EIP7805] ExecutionEngine.is_inclusion_list_satisfied
    is_inclusion_list_satisfied: Callable[..., bool] = field(default=lambda *a, **k: True)
    # performance: plain-python index of the immutable per-block facts the
    # hot paths need (slot, parent_root, bid block_hash, parent payload
    # status) so get_ancestor/get_weight never touch remerkleable views;
    # plus memoised ancestor walks and the justified-state attester list.
    block_meta: Dict[bytes, tuple] = field(default_factory=dict)
    _ancestor_cache: Dict[tuple, "ForkChoiceNode"] = field(default_factory=dict)
    _score_cache: Any = None


def get_forkchoice_store(anchor_state, anchor_block, is_data_available=None, verify_new_payload=None) -> Store:
    assert bytes(anchor_block.state_root) == hash_tree_root(anchor_state)
    anchor_root = hash_tree_root(anchor_block)
    anchor_epoch = get_current_epoch(anchor_state)
    justified_checkpoint = Checkpoint(epoch=anchor_epoch, root=anchor_root)
    finalized_checkpoint = Checkpoint(epoch=anchor_epoch, root=anchor_root)
    store = Store(
        time=get_config().compute_time_at_slot(int(anchor_state.genesis_time), int(anchor_state.slot)),
        genesis_time=int(anchor_state.genesis_time),
        justified_checkpoint=justified_checkpoint,
        finalized_checkpoint=finalized_checkpoint,
        unrealized_justified_checkpoint=justified_checkpoint,
        unrealized_finalized_checkpoint=finalized_checkpoint,
        proposer_boost_root=ZERO_ROOT,
        equivocating_indices=set(),
        blocks={anchor_root: anchor_block.copy()},
        block_states={anchor_root: anchor_state.copy()},
        block_timeliness={anchor_root: [True, True]},
        checkpoint_states={justified_checkpoint: anchor_state.copy()},
        latest_messages={},
        unrealized_justifications={anchor_root: justified_checkpoint},
        payloads={},
        # The spec starts these empty (a genesis anchor never gets PTC votes);
        # a node anchoring mid-chain does receive votes for the anchor slot.
        payload_timeliness_vote={anchor_root: [None] * PTC_SIZE()},
        payload_data_availability_vote={anchor_root: [None] * PTC_SIZE()},
    )
    _index_block(store, anchor_root, anchor_block)
    if is_data_available is not None:
        store.is_data_available = is_data_available
    if verify_new_payload is not None:
        store.verify_new_payload = verify_new_payload
    return store


# ---------------------------------------------------------------- time helpers


def get_slots_since_genesis(store: Store) -> int:
    # [Modified in Heze:EIP8198] piecewise over get_slot_durations
    return get_config().compute_slot_at_time_ms(
        seconds_to_milliseconds(store.genesis_time), seconds_to_milliseconds(store.time)
    )


def get_time_into_slot_ms(store: Store) -> int:
    """Milliseconds elapsed since the start of the store's current slot."""
    config = get_config()
    genesis_ms = seconds_to_milliseconds(store.genesis_time)
    slot_start_ms = config.compute_time_at_slot_ms(genesis_ms, get_current_slot(store))
    return seconds_to_milliseconds(store.time) - slot_start_ms


def get_current_slot(store: Store) -> int:
    return GENESIS_SLOT + get_slots_since_genesis(store)


def get_current_store_epoch(store: Store) -> int:
    return compute_epoch_at_slot(get_current_slot(store))


def compute_slots_since_epoch_start(slot: int) -> int:
    return slot - compute_start_slot_at_epoch(compute_epoch_at_slot(slot))


def seconds_to_milliseconds(seconds: int) -> int:
    if seconds > UINT64_MAX // 1000:
        return UINT64_MAX
    return seconds * 1000


def get_slot_component_duration_ms(basis_points: int, slot: int | None = None) -> int:
    """Deadline offset in ms. [Modified in Heze:EIP8198] priced at the slot
    duration in effect at ``slot`` (genesis duration when omitted)."""
    config = get_config()
    duration = (
        config.genesis_slot_duration_ms if slot is None else config.get_slot_duration_ms_at_slot(slot)
    )
    return basis_points * duration // BASIS_POINTS


def get_attestation_due_ms(slot: int | None = None) -> int:
    return get_slot_component_duration_ms(int(get_config().attestation_due_bps_gloas), slot)


def get_aggregate_due_ms(slot: int | None = None) -> int:
    return get_slot_component_duration_ms(int(get_config().aggregate_due_bps_gloas), slot)


def get_proposer_reorg_cutoff_ms(slot: int | None = None) -> int:
    return get_slot_component_duration_ms(int(get_config().proposer_reorg_cutoff_bps), slot)


def get_payload_due_ms(slot: int | None = None) -> int:
    return get_slot_component_duration_ms(int(get_config().payload_due_bps), slot)


def get_payload_attestation_due_ms(slot: int | None = None) -> int:
    return get_slot_component_duration_ms(int(get_config().payload_attestation_due_bps), slot)


def get_inclusion_list_due_ms(slot: int | None = None) -> int:
    """[New in Heze:EIP7805]"""
    return get_slot_component_duration_ms(int(get_config().inclusion_list_due_bps), slot)


# ---------------------------------------------------------------- payload status helpers


def is_payload_verified(store: Store, root: bytes) -> bool:
    return root in store.payloads


def payload_timeliness(store: Store, root: bytes, timely: bool) -> bool:
    assert root in store.payload_timeliness_vote
    if not is_payload_verified(store, root):
        return not timely
    votes = store.payload_timeliness_vote[root]
    return sum(vote == timely for vote in votes) > PAYLOAD_TIMELY_THRESHOLD()


def payload_data_availability(store: Store, root: bytes, available: bool) -> bool:
    assert root in store.payload_data_availability_vote
    if not is_payload_verified(store, root):
        return not available
    votes = store.payload_data_availability_vote[root]
    return sum(vote == available for vote in votes) > DATA_AVAILABILITY_TIMELY_THRESHOLD()


def _bid(block):
    return block.body.signed_execution_payload_bid.message


def _index_block(store: Store, root: bytes, block) -> tuple:
    """Record (slot, parent_root, bid.block_hash, parent_payload_status) for ``root``."""
    bid = _bid(block)
    parent_root = bytes(block.parent_root)
    parent_meta = store.block_meta.get(parent_root)
    if parent_meta is None:
        parent_block = store.blocks.get(parent_root)
        if parent_block is not None:
            parent_meta = _index_block(store, parent_root, parent_block)
    if parent_meta is None:
        # ``block`` is the anchor: its parent is below the store. A node
        # anchors on a fully-imported block, so its parent is treated as FULL.
        parent_status = PAYLOAD_STATUS_FULL
    else:
        parent_status = (
            PAYLOAD_STATUS_FULL if bytes(bid.parent_block_hash) == parent_meta[2] else PAYLOAD_STATUS_EMPTY
        )
    meta = (int(block.slot), parent_root, bytes(bid.block_hash), parent_status)
    store.block_meta[root] = meta
    return meta


def _meta(store: Store, root: bytes) -> Optional[tuple]:
    meta = store.block_meta.get(root)
    if meta is None:
        block = store.blocks.get(root)
        if block is not None:
            meta = _index_block(store, root, block)
    return meta


def get_parent_payload_status(store: Store, block) -> int:
    parent_root = bytes(block.parent_root)
    parent_meta = _meta(store, parent_root)
    if parent_meta is None:
        return PAYLOAD_STATUS_FULL  # anchor block, see _index_block
    parent_block_hash = bytes(_bid(block).parent_block_hash)
    return PAYLOAD_STATUS_FULL if parent_block_hash == parent_meta[2] else PAYLOAD_STATUS_EMPTY


def is_parent_node_full(store: Store, block) -> bool:
    return get_parent_payload_status(store, block) == PAYLOAD_STATUS_FULL


# ---------------------------------------------------------------- tree helpers


def get_ancestor(store: Store, node: ForkChoiceNode, slot: int) -> ForkChoiceNode:
    # iterative + memoised version of the recursive spec function
    meta = _meta(store, node.root)
    if meta is None or meta[0] <= slot:
        return node
    key = (node.root, slot)
    cached = store._ancestor_cache.get(key)
    if cached is not None:
        return cached
    cur_root, cur_status = meta[1], meta[3]
    while True:
        meta = _meta(store, cur_root)
        if meta is None or meta[0] <= slot:
            # meta None: walked below the anchor (e.g. dependent root of the anchor epoch)
            result = ForkChoiceNode(root=cur_root, payload_status=cur_status)
            break
        cur_root, cur_status = meta[1], meta[3]
    store._ancestor_cache[key] = result
    return result


def is_ancestor(store: Store, node: ForkChoiceNode, ancestor: ForkChoiceNode) -> bool:
    node_ancestor = get_ancestor(store, node, _meta(store, ancestor.root)[0])
    if node_ancestor.root != ancestor.root:
        return False
    return (
        node_ancestor.payload_status == ancestor.payload_status
        or ancestor.payload_status == PAYLOAD_STATUS_PENDING
    )


def calculate_committee_fraction(state, committee_percent: int) -> int:
    # [Modified in beta.4] slashed validators excluded (specs #5679)
    validators = state.validators
    total_balance = sum(
        int(validators[index].effective_balance)
        for index in get_active_validator_indices(state, get_current_epoch(state))
        if not validators[index].slashed
    )
    total_balance = max(EFFECTIVE_BALANCE_INCREMENT, total_balance)
    committee_weight = total_balance // SLOTS_PER_EPOCH()
    return (committee_weight * committee_percent) // 100


def get_checkpoint_block(store: Store, root: bytes, epoch: int) -> bytes:
    epoch_first_slot = compute_start_slot_at_epoch(epoch)
    node = ForkChoiceNode(root=root, payload_status=PAYLOAD_STATUS_PENDING)
    return get_ancestor(store, node, epoch_first_slot).root


def get_supported_node(store: Store, message: LatestMessage) -> ForkChoiceNode:
    meta = _meta(store, message.root)
    if meta is not None and meta[0] < message.slot:
        payload_status = PAYLOAD_STATUS_FULL if message.payload_present else PAYLOAD_STATUS_EMPTY
    else:
        payload_status = PAYLOAD_STATUS_PENDING
    return ForkChoiceNode(root=message.root, payload_status=payload_status)


def get_attestation_score(store: Store, node: ForkChoiceNode, state) -> int:
    cache = store._score_cache
    if cache is None or cache[0] is not state:
        current_epoch = get_current_epoch(state)
        validators = state.validators
        attesters = []
        for i in get_active_validator_indices(state, current_epoch):
            v = validators[i]
            if not v.slashed:
                attesters.append((i, int(v.effective_balance)))
        cache = store._score_cache = (state, attesters)
    latest_messages = store.latest_messages
    equivocating = store.equivocating_indices
    total = 0
    for i, effective_balance in cache[1]:
        msg = latest_messages.get(i)
        if msg is None or i in equivocating:
            continue
        if is_ancestor(store, get_supported_node(store, msg), node):
            total += effective_balance
    return total


def compute_proposer_score(state) -> int:
    return calculate_committee_fraction(state, PROPOSER_SCORE_BOOST())


def get_proposer_score(store: Store) -> int:
    return compute_proposer_score(store.checkpoint_states[store.justified_checkpoint])


def is_previous_slot_payload_decision(store: Store, node: ForkChoiceNode) -> bool:
    is_previous_slot = _meta(store, node.root)[0] + 1 == get_current_slot(store)
    is_payload_decision = node.payload_status in (PAYLOAD_STATUS_EMPTY, PAYLOAD_STATUS_FULL)
    return is_previous_slot and is_payload_decision


def should_build_on_full(store: Store, head: ForkChoiceNode, slot: int) -> bool:
    assert head.payload_status != PAYLOAD_STATUS_PENDING
    if _meta(store, head.root)[0] + 1 != slot:
        return head.payload_status == PAYLOAD_STATUS_FULL
    if head.payload_status == PAYLOAD_STATUS_EMPTY:
        return False
    if payload_timeliness(store, head.root, timely=False):
        return False
    if payload_data_availability(store, head.root, available=False):
        return False
    return True


def is_payload_inclusion_list_satisfied(store: Store, root: bytes) -> bool:
    """[New in Heze:EIP7805] A payload not locally available never satisfies.

    Pre-Heze payloads have no entry and are treated as satisfied.
    """
    if not is_payload_verified(store, root):
        return False
    return store.payload_inclusion_list_satisfaction.get(root, True)


def should_extend_payload(store: Store, root: bytes) -> bool:
    assert _meta(store, root)[0] + 1 == get_current_slot(store)
    if not is_payload_verified(store, root):
        return False
    # [New in Heze:EIP7805]
    if not is_payload_inclusion_list_satisfied(store, root):
        return False
    proposer_root = store.proposer_boost_root
    payload_is_timely = payload_timeliness(store, root, timely=True)
    payload_data_is_available = payload_data_availability(store, root, available=True)
    return (
        (payload_is_timely and payload_data_is_available)
        or proposer_root == ZERO_ROOT
        or bytes(store.blocks[proposer_root].parent_root) != root
        or is_parent_node_full(store, store.blocks[proposer_root])
    )


def get_payload_status_tiebreaker(store: Store, node: ForkChoiceNode) -> int:
    if is_previous_slot_payload_decision(store, node):
        if node.payload_status == PAYLOAD_STATUS_EMPTY:
            return 1
        if should_extend_payload(store, node.root):
            return 2
        return 0
    return node.payload_status


def should_apply_proposer_boost(store: Store) -> bool:
    if store.proposer_boost_root == ZERO_ROOT:
        return False
    block = store.blocks[store.proposer_boost_root]
    parent_root = bytes(block.parent_root)
    parent = store.blocks[parent_root]
    slot = int(block.slot)
    if int(parent.slot) + 1 < slot:
        return True
    if not is_head_weak(store, parent_root):
        return True
    equivocations = [
        root
        for root, b in store.blocks.items()
        if (
            store.block_timeliness[root][PTC_TIMELINESS_INDEX]
            and int(b.proposer_index) == int(parent.proposer_index)
            and int(b.slot) + 1 == slot
            and root != parent_root
        )
    ]
    return len(equivocations) == 0


def get_weight(store: Store, node: ForkChoiceNode) -> int:
    if is_previous_slot_payload_decision(store, node):
        return 0
    state = store.checkpoint_states[store.justified_checkpoint]
    attestation_score = get_attestation_score(store, node, state)
    if not should_apply_proposer_boost(store):
        return attestation_score
    proposer_score = 0
    proposer_boost_node = ForkChoiceNode(root=store.proposer_boost_root, payload_status=PAYLOAD_STATUS_PENDING)
    if is_ancestor(store, proposer_boost_node, node):
        proposer_score = get_proposer_score(store)
    return attestation_score + proposer_score


def get_voting_source(store: Store, block_root: bytes) -> Checkpoint:
    block = store.blocks[block_root]
    current_epoch = get_current_store_epoch(store)
    block_epoch = compute_epoch_at_slot(int(block.slot))
    if current_epoch > block_epoch:
        return store.unrealized_justifications[block_root]
    return cp(store.block_states[block_root].current_justified_checkpoint)


def _children_index(store: Store) -> Dict[bytes, List[bytes]]:
    index: Dict[bytes, List[bytes]] = {}
    for root, block in store.blocks.items():
        index.setdefault(bytes(block.parent_root), []).append(root)
    return index


def _is_leaf_viable(store: Store, root: bytes) -> bool:
    current_epoch = get_current_store_epoch(store)
    voting_source = get_voting_source(store, root)
    correct_justified = (
        store.justified_checkpoint.epoch == GENESIS_EPOCH
        or voting_source.epoch == store.justified_checkpoint.epoch
        or voting_source.epoch + 2 >= current_epoch
    )
    finalized_checkpoint_block = get_checkpoint_block(store, root, store.finalized_checkpoint.epoch)
    correct_finalized = (
        store.finalized_checkpoint.epoch == GENESIS_EPOCH
        or store.finalized_checkpoint.root == finalized_checkpoint_block
    )
    return correct_justified and correct_finalized


def get_node_children(store: Store, node: ForkChoiceNode, children_index=None) -> Sequence[ForkChoiceNode]:
    if node.payload_status == PAYLOAD_STATUS_PENDING:
        children = [ForkChoiceNode(root=node.root, payload_status=PAYLOAD_STATUS_EMPTY)]
        if is_payload_verified(store, node.root):
            children.append(ForkChoiceNode(root=node.root, payload_status=PAYLOAD_STATUS_FULL))
        return children
    if children_index is None:
        children_index = _children_index(store)
    children = []
    for root in children_index.get(node.root, []):
        if node.payload_status == _meta(store, root)[3]:
            children.append(ForkChoiceNode(root=root, payload_status=PAYLOAD_STATUS_PENDING))
    return children


def filter_node_tree(store: Store, node: ForkChoiceNode, children_index=None) -> List[ForkChoiceNode]:
    """[Modified in Gloas] filter over (root, payload_status) nodes so a
    childless unviable EMPTY/FULL variant is pruned on its own (specs #5509)."""
    if children_index is None:
        children_index = _children_index(store)
    children = get_node_children(store, node, children_index)
    if any(children):
        viable_nodes: List[ForkChoiceNode] = []
        for child in children:
            viable_nodes.extend(filter_node_tree(store, child, children_index))
        if any(viable_nodes):
            return viable_nodes + [node]
        return []
    if _is_leaf_viable(store, node.root):
        return [node]
    return []


def get_filtered_node_tree(store: Store) -> Set[ForkChoiceNode]:
    base = ForkChoiceNode(root=store.justified_checkpoint.root, payload_status=PAYLOAD_STATUS_PENDING)
    return set(filter_node_tree(store, base))


def get_head(store: Store) -> ForkChoiceNode:
    children_index = _children_index(store)
    nodes = get_filtered_node_tree(store)
    # [New in Gloas:EIP7732] no viable nodes -> empty variant of the justified root
    if not nodes:
        return ForkChoiceNode(root=store.justified_checkpoint.root, payload_status=PAYLOAD_STATUS_EMPTY)
    head = ForkChoiceNode(root=store.justified_checkpoint.root, payload_status=PAYLOAD_STATUS_PENDING)
    while True:
        children = [c for c in get_node_children(store, head, children_index) if c in nodes]
        if len(children) == 0:
            return head
        head = max(
            children,
            key=lambda child: (get_weight(store, child), child.root, get_payload_status_tiebreaker(store, child)),
        )


def update_checkpoints(store: Store, justified_checkpoint: Checkpoint, finalized_checkpoint: Checkpoint) -> None:
    if justified_checkpoint.epoch > store.justified_checkpoint.epoch:
        store.justified_checkpoint = justified_checkpoint
    if finalized_checkpoint.epoch > store.finalized_checkpoint.epoch:
        store.finalized_checkpoint = finalized_checkpoint


def update_unrealized_checkpoints(store: Store, uj: Checkpoint, uf: Checkpoint) -> None:
    if uj.epoch > store.unrealized_justified_checkpoint.epoch:
        store.unrealized_justified_checkpoint = uj
    if uf.epoch > store.unrealized_finalized_checkpoint.epoch:
        store.unrealized_finalized_checkpoint = uf


def get_latest_message_epoch(latest_message: LatestMessage) -> int:
    return compute_epoch_at_slot(latest_message.slot)


# ---------------------------------------------------------------- proposer head / reorg helpers


def is_head_late(store: Store, head_root: bytes) -> bool:
    return not store.block_timeliness[head_root][ATTESTATION_TIMELINESS_INDEX]


def is_ffg_competitive(store: Store, head_root: bytes, parent_root: bytes) -> bool:
    return store.unrealized_justifications[head_root] == store.unrealized_justifications[parent_root]


def is_finalization_ok(store: Store, slot: int) -> bool:
    epochs_since_finalization = compute_epoch_at_slot(slot) - store.finalized_checkpoint.epoch
    return epochs_since_finalization <= REORG_MAX_EPOCHS_SINCE_FINALIZATION()


def is_proposing_on_time(store: Store) -> bool:
    time_into_slot_ms = get_time_into_slot_ms(store)
    return time_into_slot_ms <= get_proposer_reorg_cutoff_ms(get_current_slot(store))


def is_head_weak(store: Store, head_root: bytes) -> bool:
    justified_state = store.checkpoint_states[store.justified_checkpoint]
    reorg_threshold = calculate_committee_fraction(justified_state, REORG_HEAD_WEIGHT_THRESHOLD())
    head_state = store.block_states[head_root]
    head_block = store.blocks[head_root]
    epoch = compute_epoch_at_slot(int(head_block.slot))
    head_node = ForkChoiceNode(root=head_root, payload_status=PAYLOAD_STATUS_PENDING)
    head_weight = get_attestation_score(store, head_node, justified_state)
    for index in range(get_committee_count_per_slot(head_state, epoch)):
        committee = get_beacon_committee(head_state, int(head_block.slot), index)
        head_weight += sum(
            int(justified_state.validators[i].effective_balance) for i in committee if i in store.equivocating_indices
        )
    return head_weight < reorg_threshold


def is_parent_strong(store: Store, root: bytes) -> bool:
    justified_state = store.checkpoint_states[store.justified_checkpoint]
    parent_threshold = calculate_committee_fraction(justified_state, REORG_PARENT_WEIGHT_THRESHOLD())
    parent_root = bytes(store.blocks[root].parent_root)
    parent_node = ForkChoiceNode(root=parent_root, payload_status=PAYLOAD_STATUS_PENDING)
    return get_attestation_score(store, parent_node, justified_state) > parent_threshold


def is_proposer_equivocation(store: Store, root: bytes) -> bool:
    block = store.blocks[root]
    matching = [
        r for r, b in store.blocks.items()
        if int(b.proposer_index) == int(block.proposer_index) and int(b.slot) == int(block.slot)
    ]
    return len(matching) > 1


def get_proposer_head(store: Store, head_node: ForkChoiceNode, slot: int) -> ForkChoiceNode:
    head_block = store.blocks[head_node.root]
    parent_root = bytes(head_block.parent_root)
    parent_block = store.blocks[parent_root]
    parent_node = ForkChoiceNode(root=parent_root, payload_status=get_parent_payload_status(store, head_block))
    head_late = is_head_late(store, head_node.root)
    ffg_competitive = is_ffg_competitive(store, head_node.root, parent_root)
    finalization_ok = is_finalization_ok(store, slot)
    proposing_on_time = is_proposing_on_time(store)
    single_slot_reorg = int(parent_block.slot) + 1 == int(head_block.slot) and int(head_block.slot) + 1 == slot
    current_time_ok = int(head_block.slot) + 1 == slot
    assert store.proposer_boost_root != head_node.root
    head_weak = is_head_weak(store, head_node.root)
    parent_strong = is_parent_strong(store, head_node.root)
    proposer_equivocation = is_proposer_equivocation(store, head_node.root)
    if all([head_late, ffg_competitive, finalization_ok, proposing_on_time, single_slot_reorg, head_weak, parent_strong]):
        return parent_node
    if all([head_weak, current_time_ok, proposer_equivocation]):
        return parent_node
    return head_node


# ---------------------------------------------------------------- on_* internals


def compute_pulled_up_tip(store: Store, block_root: bytes) -> None:
    state = store.block_states[block_root].copy()
    process_justification_and_finalization(state)
    store.unrealized_justifications[block_root] = cp(state.current_justified_checkpoint)
    update_unrealized_checkpoints(store, cp(state.current_justified_checkpoint), cp(state.finalized_checkpoint))
    block_epoch = compute_epoch_at_slot(int(store.blocks[block_root].slot))
    if block_epoch < get_current_store_epoch(store):
        update_checkpoints(store, cp(state.current_justified_checkpoint), cp(state.finalized_checkpoint))


def on_tick_per_slot(store: Store, time: int) -> None:
    previous_slot = get_current_slot(store)
    store.time = int(time)
    current_slot = get_current_slot(store)
    if current_slot > previous_slot:
        store.proposer_boost_root = ZERO_ROOT
    if current_slot > previous_slot and compute_slots_since_epoch_start(current_slot) == 0:
        update_checkpoints(store, store.unrealized_justified_checkpoint, store.unrealized_finalized_checkpoint)


def on_tick(store: Store, time: int) -> None:
    time = int(time)
    config = get_config()
    tick_slot = config.compute_slot_at_time_ms(
        seconds_to_milliseconds(store.genesis_time), seconds_to_milliseconds(time)
    )
    while get_current_slot(store) < tick_slot:
        previous_time = config.compute_time_at_slot(store.genesis_time, get_current_slot(store) + 1)
        on_tick_per_slot(store, previous_time)
    on_tick_per_slot(store, time)


def validate_target_epoch_against_current_time(store: Store, attestation) -> None:
    target = attestation.data.target
    current_epoch = get_current_store_epoch(store)
    previous_epoch = current_epoch - 1 if current_epoch > GENESIS_EPOCH else GENESIS_EPOCH
    assert int(target.epoch) in (current_epoch, previous_epoch)


def validate_on_attestation(store: Store, attestation, is_from_block: bool) -> None:
    data = attestation.data
    target = data.target
    if not is_from_block:
        validate_target_epoch_against_current_time(store, attestation)
    assert int(target.epoch) == compute_epoch_at_slot(int(data.slot))
    assert bytes(target.root) in store.blocks
    beacon_block_root = bytes(data.beacon_block_root)
    assert beacon_block_root in store.blocks
    block_slot = int(store.blocks[beacon_block_root].slot)
    assert block_slot <= int(data.slot)
    assert int(data.index) in (0, 1)
    if block_slot == int(data.slot):
        assert int(data.index) == 0
    if int(data.index) == 1:
        assert is_payload_verified(store, beacon_block_root)
    assert bytes(target.root) == get_checkpoint_block(store, beacon_block_root, int(target.epoch))
    assert get_current_slot(store) >= int(data.slot) + 1


def store_target_checkpoint_state(store: Store, target: Checkpoint) -> None:
    if target not in store.checkpoint_states:
        base_state = store.block_states[target.root].copy()
        if int(base_state.slot) < compute_start_slot_at_epoch(target.epoch):
            base_state = process_slots(base_state, compute_start_slot_at_epoch(target.epoch)) or base_state
        store.checkpoint_states[target] = base_state


def update_latest_messages(store: Store, attesting_indices: Sequence[int], attestation) -> None:
    slot = int(attestation.data.slot)
    beacon_block_root = bytes(attestation.data.beacon_block_root)
    payload_present = int(attestation.data.index) == 1
    for i in attesting_indices:
        i = int(i)
        if i in store.equivocating_indices:
            continue
        if i not in store.latest_messages or slot > store.latest_messages[i].slot:
            store.latest_messages[i] = LatestMessage(slot=slot, root=beacon_block_root, payload_present=payload_present)


def record_block_timeliness(store: Store, root: bytes) -> None:
    block = store.blocks[root]
    time_into_slot_ms = get_time_into_slot_ms(store)
    is_current_slot = get_current_slot(store) == int(block.slot)
    slot = int(block.slot)
    store.block_timeliness[root] = [
        is_current_slot and time_into_slot_ms < threshold
        for threshold in (get_attestation_due_ms(slot), get_payload_attestation_due_ms(slot))
    ]


def compute_shuffling_dependent_slot(epoch: int) -> int:
    if epoch <= MIN_SEED_LOOKAHEAD:
        return GENESIS_SLOT
    return compute_start_slot_at_epoch(epoch - MIN_SEED_LOOKAHEAD) - 1


def get_shuffling_dependent_root(store: Store, root: bytes, epoch: int) -> bytes:
    node = ForkChoiceNode(root=root, payload_status=PAYLOAD_STATUS_PENDING)
    return get_ancestor(store, node, compute_shuffling_dependent_slot(epoch)).root


def update_proposer_boost_root(store: Store, head: bytes, root: bytes) -> None:
    is_first_block = store.proposer_boost_root == ZERO_ROOT
    is_timely = store.block_timeliness[root][ATTESTATION_TIMELINESS_INDEX]
    epoch = get_current_store_epoch(store)
    is_same_dependent_root = (
        get_shuffling_dependent_root(store, head, epoch) == get_shuffling_dependent_root(store, root, epoch)
    )
    if is_timely and is_first_block and is_same_dependent_root:
        store.proposer_boost_root = root


def notify_ptc_messages(store: Store, state, payload_attestations) -> None:
    if int(state.slot) == 0:
        return
    from .types.gloas import PayloadAttestationMessage
    for payload_attestation in payload_attestations:
        indexed = get_indexed_payload_attestation(state, payload_attestation)
        for idx in indexed.attesting_indices:
            on_payload_attestation_message(
                store,
                PayloadAttestationMessage(validator_index=int(idx), data=payload_attestation.data, signature=b"\x00" * 96),
                is_from_block=True,
            )


# ---------------------------------------------------------------- handlers


def on_block(store: Store, signed_block) -> None:
    block = signed_block.message
    block_root = hash_tree_root(block)
    if block_root in store.blocks:
        return
    parent_root = bytes(block.parent_root)
    assert parent_root in store.block_states
    if is_parent_node_full(store, block):
        assert is_payload_verified(store, parent_root)
    current_slot = get_current_slot(store)
    assert current_slot >= int(block.slot)
    finalized_slot = compute_start_slot_at_epoch(store.finalized_checkpoint.epoch)
    assert int(block.slot) > finalized_slot
    assert store.finalized_checkpoint.root == get_checkpoint_block(store, parent_root, store.finalized_checkpoint.epoch)

    state = store.block_states[parent_root].copy()
    new_state = state_transition(state, signed_block, validate_result=True)
    if new_state is not None:
        state = new_state

    head = get_head(store)
    store.blocks[block_root] = block
    store.block_states[block_root] = state
    _index_block(store, block_root, block)
    store.payload_timeliness_vote[block_root] = [None] * PTC_SIZE()
    store.payload_data_availability_vote[block_root] = [None] * PTC_SIZE()

    notify_ptc_messages(store, state, block.body.payload_attestations)
    record_block_timeliness(store, block_root)
    update_proposer_boost_root(store, head.root, block_root)
    update_checkpoints(store, cp(state.current_justified_checkpoint), cp(state.finalized_checkpoint))
    compute_pulled_up_tip(store, block_root)


def on_attestation(store: Store, attestation, is_from_block: bool = False) -> None:
    validate_on_attestation(store, attestation, is_from_block)
    target = cp(attestation.data.target)
    store_target_checkpoint_state(store, target)
    target_state = store.checkpoint_states[target]
    indexed_attestation = get_indexed_attestation(target_state, attestation)
    assert is_valid_indexed_attestation(target_state, indexed_attestation)
    update_latest_messages(store, [int(i) for i in indexed_attestation.attesting_indices], attestation)


def on_attester_slashing(store: Store, attester_slashing) -> None:
    a1 = attester_slashing.attestation_1
    a2 = attester_slashing.attestation_2
    assert is_slashable_attestation_data(a1.data, a2.data)
    state = store.block_states[store.justified_checkpoint.root]
    assert is_valid_indexed_attestation(state, a1)
    assert is_valid_indexed_attestation(state, a2)
    for index in set(int(i) for i in a1.attesting_indices).intersection(int(i) for i in a2.attesting_indices):
        store.equivocating_indices.add(index)


def verify_execution_payload_envelope(store: Store, state, signed_envelope) -> None:
    from .state_transition.block.execution_payload_envelope import verify_execution_payload_envelope_signature
    from .types import BeaconBlockHeader

    envelope = signed_envelope.message
    payload = envelope.payload
    assert verify_execution_payload_envelope_signature(state, signed_envelope)

    header = state.latest_block_header
    filled = BeaconBlockHeader(
        slot=header.slot,
        proposer_index=header.proposer_index,
        parent_root=header.parent_root,
        state_root=hash_tree_root(state),
        body_root=header.body_root,
    )
    assert bytes(envelope.beacon_block_root) == hash_tree_root(filled)
    assert bytes(envelope.parent_beacon_block_root) == bytes(state.latest_block_header.parent_root)

    bid = state.latest_execution_payload_bid
    assert int(envelope.builder_index) == int(bid.builder_index)
    assert bytes(payload.prev_randao) == bytes(bid.prev_randao)
    assert int(payload.gas_limit) == int(bid.gas_limit)
    assert bytes(payload.block_hash) == bytes(bid.block_hash)
    assert hash_tree_root(envelope.execution_requests) == bytes(bid.execution_requests_root)

    assert int(payload.slot_number) == int(state.slot)
    assert bytes(payload.parent_hash) == bytes(state.latest_block_hash)
    assert int(payload.timestamp) == compute_time_at_slot(int(state.genesis_time), int(state.slot))
    assert hash_tree_root(payload.withdrawals) == hash_tree_root(state.payload_expected_withdrawals)
    assert store.verify_new_payload(
        payload,
        [bytes(c) for c in bid.blob_kzg_commitments],
        bytes(envelope.parent_beacon_block_root),
        envelope.execution_requests,
    )


def on_execution_payload_envelope(store: Store, signed_envelope) -> None:
    envelope = signed_envelope.message
    root = bytes(envelope.beacon_block_root)
    assert root in store.block_states
    assert store.is_data_available(root)
    state = store.block_states[root]
    verify_execution_payload_envelope(store, state, signed_envelope)
    # [New in Heze:EIP7805]
    if hasattr(_bid(store.blocks[root]), "inclusion_list_bits"):
        record_payload_inclusion_list_satisfaction(
            store, root, store.is_inclusion_list_satisfied(root, envelope.payload)
        )
    store.payloads[root] = envelope


def on_payload_attestation_message(store: Store, ptc_message, is_from_block: bool = False) -> None:
    from .types.gloas import IndexedPayloadAttestation

    data = ptc_message.data
    root = bytes(data.beacon_block_root)
    assert root in store.block_states
    state = store.block_states[root]
    if int(data.slot) != int(state.slot):
        return
    ptc = get_ptc(state, int(data.slot))
    validator_index = int(ptc_message.validator_index)
    ptc_indices = [i for i, v in enumerate(ptc) if int(v) == validator_index]
    assert len(ptc_indices) > 0
    if not is_from_block:
        assert int(data.slot) == get_current_slot(store)
        assert is_valid_indexed_payload_attestation(
            state,
            IndexedPayloadAttestation(attesting_indices=[validator_index], data=data, signature=ptc_message.signature),
        )
    tv = store.payload_timeliness_vote[root]
    dv = store.payload_data_availability_vote[root]
    for i in ptc_indices:
        tv[i] = bool(data.payload_present)
        dv[i] = bool(data.blob_data_available)


# ---------------------------------------------------------------- node-integration helpers
# (not part of the spec: variants that trust work the node already did)


def on_block_with_state(store: Store, signed_block, post_state) -> bool:
    """``on_block`` for a block the node has already validated and whose
    post-state it already computed (skips the state transition). Returns
    False if the parent is unknown to the store (pre-anchor block)."""
    block = signed_block.message
    block_root = hash_tree_root(block)
    if block_root in store.blocks:
        return True
    parent_root = bytes(block.parent_root)
    if parent_root not in store.block_states:
        return False
    head = get_head(store)
    store.blocks[block_root] = block
    store.block_states[block_root] = post_state
    _index_block(store, block_root, block)
    store.payload_timeliness_vote[block_root] = [None] * PTC_SIZE()
    store.payload_data_availability_vote[block_root] = [None] * PTC_SIZE()
    try:
        notify_ptc_messages(store, post_state, block.body.payload_attestations)
    except Exception as e:  # PTC bookkeeping must never break import
        logger.debug(f"notify_ptc_messages failed for {block_root.hex()[:12]}: {e}")
    record_block_timeliness(store, block_root)
    update_proposer_boost_root(store, head.root, block_root)
    update_checkpoints(store, cp(post_state.current_justified_checkpoint), cp(post_state.finalized_checkpoint))
    compute_pulled_up_tip(store, block_root)
    return True


def on_execution_payload_envelope_trusted(
    store: Store, signed_envelope, inclusion_list_satisfied: Optional[bool] = None
) -> bool:
    """Record an envelope the node already verified (signature, bid
    consistency, EL newPayload). Returns False if the block is unknown.

    ``inclusion_list_satisfied`` is the EL's PayloadStatusV2 verdict (Heze);
    None means not a Heze payload or not yet validated (optimistically
    satisfied per heze/optimistic-sync.md).
    """
    root = bytes(signed_envelope.message.beacon_block_root)
    if root not in store.block_states:
        return False
    store.payloads[root] = signed_envelope.message
    if inclusion_list_satisfied is not None:
        record_payload_inclusion_list_satisfaction(store, root, bool(inclusion_list_satisfied))
    return True


def record_payload_inclusion_list_satisfaction(store: Store, root: bytes, satisfied: bool) -> None:
    """[New in Heze:EIP7805] A payload recorded as satisfied stays satisfied."""
    if store.payload_inclusion_list_satisfaction.get(root) is True:
        return
    store.payload_inclusion_list_satisfaction[root] = bool(satisfied)


def compute_shuffling_lookahead_start_slot(epoch: int) -> int:
    lookahead_epoch = max(0, int(epoch) - MIN_SEED_LOOKAHEAD)
    return compute_start_slot_at_epoch(lookahead_epoch)


def is_valid_dependent_root(store: Store, root: bytes, dependent_slot: int) -> bool:
    """Whether ``root`` is, or could become on some branch, the latest block
    at or before ``dependent_slot`` (gloas/fork-choice.md)."""
    if root == get_head(store).root:
        return True
    for block_root, meta in store.block_meta.items():
        # meta = (slot, parent_root, ...)
        if meta[1] == root and meta[0] > dependent_slot:
            return True
    return False


def on_inclusion_list(store: Store, signed_inclusion_list, timely: Optional[bool] = None) -> None:
    """[New in Heze:EIP7805] ``on_inclusion_list``; raises on invalid input.

    ``timely`` may be supplied by the caller (gossip arrival time); otherwise
    it is derived from the store clock as in the spec.
    """
    from .inclusion_list import (
        get_inclusion_list_committee,
        get_inclusion_list_store,
        is_valid_inclusion_list_signature,
        process_inclusion_list,
    )
    from .state_transition import process_slots

    inclusion_list = signed_inclusion_list.message
    current_slot = get_current_slot(store)
    il_slot = int(inclusion_list.slot)
    assert il_slot <= current_slot
    assert il_slot + int(get_config().min_slots_for_inclusion_lists_requests) >= current_slot

    txs = [bytes(tx) for tx in inclusion_list.transactions]
    size = sum(len(tx) for tx in txs)
    assert size > 0
    assert size <= int(get_config().max_transactions_bytes_per_inclusion_list)
    assert all(len(tx) > 0 for tx in txs)

    dependent_root = bytes(inclusion_list.dependent_root)
    assert dependent_root in store.blocks
    assert dependent_root in store.block_states

    epoch = compute_epoch_at_slot(il_slot)
    dependent_slot = compute_shuffling_dependent_slot(epoch)
    assert _meta(store, dependent_root)[0] <= dependent_slot
    assert is_valid_dependent_root(store, dependent_root, dependent_slot)

    state = store.block_states[dependent_root]
    lookahead_start_slot = compute_shuffling_lookahead_start_slot(epoch)
    if int(state.slot) < lookahead_start_slot:
        state = process_slots(state.copy(), lookahead_start_slot)
    committee = get_inclusion_list_committee(state, il_slot)
    assert int(inclusion_list.validator_index) in committee
    assert is_valid_inclusion_list_signature(state, signed_inclusion_list)

    if timely is None:
        timely = il_slot == current_slot and get_time_into_slot_ms(store) < get_inclusion_list_due_ms(il_slot)
    process_inclusion_list(get_inclusion_list_store(), signed_inclusion_list, bool(timely))


def prune_store(store: Store, keep_epochs: int = 2, extra_keep: Sequence[bytes] = ()) -> int:
    """Drop blocks/states older than ``keep_epochs`` before finalization
    (the finalized checkpoint block itself is kept, as are ``extra_keep``
    roots, e.g. the fast-confirmation roots). Returns count removed."""
    cutoff_epoch = max(0, store.finalized_checkpoint.epoch - keep_epochs)
    cutoff_slot = compute_start_slot_at_epoch(cutoff_epoch)
    keep = {store.finalized_checkpoint.root, store.justified_checkpoint.root,
            store.unrealized_justified_checkpoint.root, store.unrealized_finalized_checkpoint.root}
    keep.update(bytes(r) for r in extra_keep if r)
    doomed = [r for r, b in store.blocks.items() if int(b.slot) < cutoff_slot and r not in keep]
    store._ancestor_cache.clear()
    for r in doomed:
        store.blocks.pop(r, None)
        store.block_meta.pop(r, None)
        store.block_states.pop(r, None)
        store.block_timeliness.pop(r, None)
        store.unrealized_justifications.pop(r, None)
        store.payloads.pop(r, None)
        store.payload_timeliness_vote.pop(r, None)
        store.payload_data_availability_vote.pop(r, None)
        store.payload_inclusion_list_satisfaction.pop(r, None)
    # A vote for a pruned block cannot support any node left in the tree
    # (everything kept descends from the finalized checkpoint); keeping it
    # would make get_supported_node dereference an unknown root.
    if doomed:
        doomed_set = set(doomed)
        for vi in [vi for vi, m in store.latest_messages.items() if m.root in doomed_set]:
            del store.latest_messages[vi]
        store._score_cache = None
    for ckpt in [c for c in store.checkpoint_states if c.epoch < cutoff_epoch and c not in
                 (store.justified_checkpoint, store.finalized_checkpoint)]:
        store.checkpoint_states.pop(ckpt, None)
    return len(doomed)
