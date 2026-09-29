"""Fast Confirmation Rule (FCR) — port of ``specs/phase0/fast-confirmation.md``
(consensus-specs v1.7.0-alpha.14) on top of :mod:`consensoor.spec.fork_choice`.

``on_fast_confirmation`` is meant to run once per slot (after ``on_tick`` for
that slot); ``get_safe_execution_block_hash`` gives the execution block hash
to use as ``safe_block_hash`` in ``notify_forkchoice_updated``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Set

from .constants import SLOTS_PER_EPOCH
from .fork_choice import (
    Checkpoint,
    ForkChoiceNode,
    PAYLOAD_STATUS_PENDING,
    Store,
    compute_proposer_score,
    compute_slots_since_epoch_start,
    cp,
    get_ancestor,
    get_attestation_score,
    get_checkpoint_block,
    get_current_slot,
    get_current_store_epoch,
    get_head,
    get_latest_message_epoch,
    get_supported_node,
    get_voting_source,
    is_ancestor,
)
from .network_config import get_config
from .state_transition.helpers.accessors import (
    get_active_validator_indices,
    get_current_epoch,
    get_total_active_balance,
)
from .state_transition.helpers.beacon_committee import get_beacon_committee, get_committee_count_per_slot
from .state_transition.helpers.misc import compute_epoch_at_slot, compute_start_slot_at_epoch
from .state_transition.helpers.predicates import is_active_validator
from .state_transition.transition import process_slots

COMMITTEE_WEIGHT_ESTIMATION_ADJUSTMENT_FACTOR = 5


def CONFIRMATION_BYZANTINE_THRESHOLD() -> int:
    return int(getattr(get_config(), "confirmation_byzantine_threshold", 25))


@dataclass
class FastConfirmationStore:
    store: Store
    confirmed_root: bytes
    previous_epoch_observed_justified_checkpoint: Checkpoint
    current_epoch_observed_justified_checkpoint: Checkpoint
    previous_epoch_greatest_unrealized_checkpoint: Checkpoint
    previous_slot_head: bytes
    current_slot_head: bytes


def get_fast_confirmation_store(store: Store) -> FastConfirmationStore:
    return FastConfirmationStore(
        store=store,
        confirmed_root=store.finalized_checkpoint.root,
        previous_epoch_observed_justified_checkpoint=store.finalized_checkpoint,
        current_epoch_observed_justified_checkpoint=store.finalized_checkpoint,
        previous_epoch_greatest_unrealized_checkpoint=store.finalized_checkpoint,
        previous_slot_head=store.finalized_checkpoint.root,
        current_slot_head=store.finalized_checkpoint.root,
    )


# ---------------------------------------------------------------- misc helpers


def get_node_for_root(block_root: bytes) -> ForkChoiceNode:
    return ForkChoiceNode(root=block_root, payload_status=PAYLOAD_STATUS_PENDING)


def get_block_slot(store: Store, block_root: bytes) -> int:
    return int(store.blocks[block_root].slot)


def get_block_epoch(store: Store, block_root: bytes) -> int:
    return compute_epoch_at_slot(get_block_slot(store, block_root))


def get_checkpoint_for_block(store: Store, block_root: bytes, epoch: int) -> Checkpoint:
    return Checkpoint(epoch=epoch, root=get_checkpoint_block(store, block_root, epoch))


def get_current_target(store: Store) -> Checkpoint:
    head = get_head(store).root
    return get_checkpoint_for_block(store, head, get_current_store_epoch(store))


def is_start_slot_at_epoch(slot: int) -> bool:
    return compute_slots_since_epoch_start(slot) == 0


def get_ancestor_roots(store: Store, block_root: bytes, terminal_root: bytes) -> Sequence[bytes]:
    root = block_root
    ancestor_roots: list[bytes] = []
    while int(store.blocks[root].slot) > int(store.blocks[terminal_root].slot):
        ancestor_roots.insert(0, root)
        root = bytes(store.blocks[root].parent_root)
        if root == terminal_root:
            return ancestor_roots
    return []


# ---------------------------------------------------------------- state helpers


def get_slot_committee(store: Store, slot: int) -> Set[int]:
    head = get_head(store).root
    shuffling_source = store.block_states[head]
    committees_count = get_committee_count_per_slot(shuffling_source, compute_epoch_at_slot(slot))
    participants: Set[int] = set()
    for i in range(committees_count):
        participants.update(int(v) for v in get_beacon_committee(shuffling_source, slot, i))
    return participants


def get_pulled_up_head_state(store: Store):
    head = get_head(store).root
    head_state = store.block_states[head]
    if get_current_epoch(head_state) < get_current_store_epoch(store):
        pulled_up = head_state.copy()
        pulled_up = process_slots(pulled_up, compute_start_slot_at_epoch(get_current_store_epoch(store))) or pulled_up
        return pulled_up
    return head_state


def get_previous_balance_source(fcr_store: FastConfirmationStore):
    return fcr_store.store.checkpoint_states[fcr_store.previous_epoch_observed_justified_checkpoint]


def get_current_balance_source(fcr_store: FastConfirmationStore):
    return fcr_store.store.checkpoint_states[fcr_store.current_epoch_observed_justified_checkpoint]


# ---------------------------------------------------------------- LMD-GHOST helpers


def get_node_support_between_slots(store: Store, balance_source, node: ForkChoiceNode, start_slot: int, end_slot: int) -> int:
    """Support for ``node`` (root + payload status) by validators assigned to
    slots [start_slot, end_slot] (specs #5672: node, not block root)."""
    participants: Set[int] = set()
    for slot in range(start_slot, end_slot + 1):
        participants.update(get_slot_committee(store, slot))
    epoch = get_current_epoch(balance_source)
    total = 0
    for i in participants:
        v = balance_source.validators[i]
        if v.slashed or not is_active_validator(v, epoch):
            continue
        msg = store.latest_messages.get(i)
        if (
            msg is not None
            and i not in store.equivocating_indices
            and get_supported_node(store, msg) == node
        ):
            total += int(v.effective_balance)
    return total


def is_full_validator_set_covered(start_slot: int, end_slot: int) -> bool:
    start_full_epoch = compute_epoch_at_slot(start_slot + SLOTS_PER_EPOCH() - 1)
    end_full_epoch = compute_epoch_at_slot(end_slot + 1)
    return start_full_epoch < end_full_epoch


def adjust_committee_weight_estimate_to_ensure_safety(estimate: int) -> int:
    ceil = (estimate + 999) // 1000
    return ceil * (1000 + COMMITTEE_WEIGHT_ESTIMATION_ADJUSTMENT_FACTOR)


def estimate_committee_weight_between_slots(total_active_balance: int, start_slot: int, end_slot: int) -> int:
    if start_slot > end_slot:
        return 0
    if is_full_validator_set_covered(start_slot, end_slot):
        return total_active_balance
    spe = SLOTS_PER_EPOCH()
    start_epoch = compute_epoch_at_slot(start_slot)
    end_epoch = compute_epoch_at_slot(end_slot)
    committee_weight = total_active_balance // spe
    if start_epoch == end_epoch:
        return committee_weight * (end_slot - start_slot + 1)
    num_slots_in_end_epoch = compute_slots_since_epoch_start(end_slot) + 1
    remaining_slots_in_end_epoch = spe - num_slots_in_end_epoch
    num_slots_in_start_epoch = spe - compute_slots_since_epoch_start(start_slot)
    start_epoch_weight = committee_weight * num_slots_in_start_epoch
    end_epoch_weight = committee_weight * num_slots_in_end_epoch
    start_epoch_weight_pro_rated = start_epoch_weight // spe * remaining_slots_in_end_epoch
    return adjust_committee_weight_estimate_to_ensure_safety(start_epoch_weight_pro_rated + end_epoch_weight)


def get_equivocation_score(store: Store, balance_source, start_slot: int, end_slot: int) -> int:
    committee_indices: Set[int] = set()
    for slot in range(start_slot, end_slot + 1):
        committee_indices.update(get_slot_committee(store, slot))
    epoch = get_current_epoch(balance_source)
    return sum(
        int(balance_source.validators[i].effective_balance)
        for i in committee_indices.intersection(store.equivocating_indices)
        if is_active_validator(balance_source.validators[i], epoch)
    )


def compute_adversarial_weight(store: Store, balance_source, start_slot: int, end_slot: int) -> int:
    total_active_balance = get_total_active_balance(balance_source)
    maximum_weight = estimate_committee_weight_between_slots(total_active_balance, start_slot, end_slot)
    max_adversarial_weight = maximum_weight // 100 * CONFIRMATION_BYZANTINE_THRESHOLD()
    equivocation_score = get_equivocation_score(store, balance_source, start_slot, end_slot)
    if max_adversarial_weight > equivocation_score:
        return max_adversarial_weight - equivocation_score
    return 0


def get_adversarial_weight(store: Store, balance_source, block_root: bytes) -> int:
    current_slot = get_current_slot(store)
    block = store.blocks[block_root]
    if get_block_epoch(store, block_root) > get_block_epoch(store, bytes(block.parent_root)):
        start_slot = compute_start_slot_at_epoch(get_block_epoch(store, block_root))
        return compute_adversarial_weight(store, balance_source, start_slot, current_slot - 1)
    return compute_adversarial_weight(store, balance_source, int(block.slot), current_slot - 1)


def compute_empty_slot_support_discount(store: Store, balance_source, block_root: bytes) -> int:
    block = store.blocks[block_root]
    parent_block = store.blocks[bytes(block.parent_root)]
    if int(parent_block.slot) + 1 == int(block.slot):
        return 0
    # Discount votes for the parent *node* (with the payload status this
    # block builds on) from the committees of the empty slots (specs #5672)
    parent_node = get_ancestor(store, get_node_for_root(block_root), int(parent_block.slot))
    parent_support_in_empty_slots = get_node_support_between_slots(
        store, balance_source, parent_node, int(parent_block.slot) + 1, int(block.slot) - 1
    )
    adversarial_weight = compute_adversarial_weight(
        store, balance_source, int(parent_block.slot) + 1, int(block.slot) - 1
    )
    if parent_support_in_empty_slots > adversarial_weight:
        return parent_support_in_empty_slots - adversarial_weight
    return 0


def get_support_discount(store: Store, balance_source, block_root: bytes) -> int:
    return compute_empty_slot_support_discount(store, balance_source, block_root)


def compute_safety_threshold(store: Store, block_root: bytes, balance_source) -> int:
    current_slot = get_current_slot(store)
    block = store.blocks[block_root]
    parent_block = store.blocks[bytes(block.parent_root)]
    total_active_balance = get_total_active_balance(balance_source)
    proposer_score = compute_proposer_score(balance_source)
    maximum_support = estimate_committee_weight_between_slots(
        total_active_balance, int(parent_block.slot) + 1, current_slot - 1
    )
    support_discount = get_support_discount(store, balance_source, block_root)
    adversarial_weight = get_adversarial_weight(store, balance_source, block_root)
    if support_discount < maximum_support + proposer_score + 2 * adversarial_weight:
        return (maximum_support + proposer_score + 2 * adversarial_weight - support_discount) // 2
    return 0


def is_one_confirmed(store: Store, balance_source, block_root: bytes) -> bool:
    support = get_attestation_score(store, get_node_for_root(block_root), balance_source)
    safety_threshold = compute_safety_threshold(store, block_root, balance_source)
    return support > safety_threshold


def is_confirmed_chain_safe(fcr_store: FastConfirmationStore, confirmed_root: bytes) -> bool:
    store = fcr_store.store
    observed = fcr_store.current_epoch_observed_justified_checkpoint
    if observed != get_checkpoint_for_block(store, confirmed_root, observed.epoch):
        return False
    current_epoch = get_current_store_epoch(store)
    if observed.epoch + 1 >= current_epoch:
        start_root_exclusive = observed.root
    else:
        ancestor_at_previous_epoch_start = get_ancestor(
            store, get_node_for_root(confirmed_root), compute_start_slot_at_epoch(current_epoch - 1)
        ).root
        if get_block_epoch(store, ancestor_at_previous_epoch_start) + 1 == current_epoch:
            start_root_exclusive = bytes(store.blocks[ancestor_at_previous_epoch_start].parent_root)
        else:
            start_root_exclusive = ancestor_at_previous_epoch_start
    chain_roots = get_ancestor_roots(store, confirmed_root, start_root_exclusive)
    balance_source = get_previous_balance_source(fcr_store)
    return all(is_one_confirmed(store, balance_source, root) for root in chain_roots)


# ---------------------------------------------------------------- FFG helpers


def get_current_target_score(store: Store) -> int:
    target = get_current_target(store)
    state = get_pulled_up_head_state(store)
    epoch = get_current_epoch(state)
    total = 0
    for i in get_active_validator_indices(state, epoch):
        v = state.validators[i]
        if v.slashed:
            continue
        msg = store.latest_messages.get(i)
        if msg is None or i in store.equivocating_indices:
            continue
        if target == get_checkpoint_for_block(store, msg.root, get_latest_message_epoch(msg)):
            total += int(v.effective_balance)
    return total


def compute_honest_ffg_support_for_current_target(store: Store) -> int:
    current_slot = get_current_slot(store)
    current_epoch = compute_epoch_at_slot(current_slot)
    balance_source = get_pulled_up_head_state(store)
    total_active_balance = get_total_active_balance(balance_source)
    ffg_support_for_checkpoint = get_current_target_score(store)
    ffg_weight_till_now = estimate_committee_weight_between_slots(
        total_active_balance, compute_start_slot_at_epoch(current_epoch), current_slot - 1
    )
    remaining_ffg_weight = total_active_balance - ffg_weight_till_now
    remaining_honest_ffg_weight = remaining_ffg_weight // 100 * (100 - CONFIRMATION_BYZANTINE_THRESHOLD())
    adversarial_weight = compute_adversarial_weight(
        store, balance_source, compute_start_slot_at_epoch(current_epoch), current_slot - 1
    )
    min_honest_ffg_support = ffg_support_for_checkpoint - min(adversarial_weight, ffg_support_for_checkpoint)
    return min_honest_ffg_support + remaining_honest_ffg_weight


def will_no_conflicting_checkpoint_be_justified(store: Store) -> bool:
    if get_current_target(store) == store.unrealized_justified_checkpoint:
        return True
    state = get_pulled_up_head_state(store)
    total_active_balance = get_total_active_balance(state)
    honest_ffg_support = compute_honest_ffg_support_for_current_target(store)
    return 3 * honest_ffg_support > 1 * total_active_balance


def will_current_target_be_justified(store: Store) -> bool:
    state = get_pulled_up_head_state(store)
    total_active_balance = get_total_active_balance(state)
    honest_ffg_support = compute_honest_ffg_support_for_current_target(store)
    return 3 * honest_ffg_support >= 2 * total_active_balance


# ---------------------------------------------------------------- the rule


def update_fast_confirmation_variables(fcr_store: FastConfirmationStore) -> None:
    store = fcr_store.store
    fcr_store.previous_slot_head = fcr_store.current_slot_head
    fcr_store.current_slot_head = get_head(store).root
    if is_start_slot_at_epoch(get_current_slot(store) + 1):
        fcr_store.previous_epoch_greatest_unrealized_checkpoint = store.unrealized_justified_checkpoint
    if is_start_slot_at_epoch(get_current_slot(store)):
        fcr_store.previous_epoch_observed_justified_checkpoint = fcr_store.current_epoch_observed_justified_checkpoint
        fcr_store.current_epoch_observed_justified_checkpoint = fcr_store.previous_epoch_greatest_unrealized_checkpoint


def find_latest_confirmed_descendant(fcr_store: FastConfirmationStore, latest_confirmed_root: bytes) -> bytes:
    store = fcr_store.store
    head = get_head(store).root
    current_epoch = get_current_store_epoch(store)
    confirmed_root = latest_confirmed_root

    if (
        get_block_epoch(store, confirmed_root) + 1 == current_epoch
        and get_voting_source(store, fcr_store.previous_slot_head).epoch + 2 >= current_epoch
        and (
            is_start_slot_at_epoch(get_current_slot(store))
            or (
                will_no_conflicting_checkpoint_be_justified(store)
                and (
                    store.unrealized_justifications[fcr_store.previous_slot_head].epoch + 1 >= current_epoch
                    or store.unrealized_justifications[head].epoch + 1 >= current_epoch
                )
            )
        )
    ):
        canonical_roots = get_ancestor_roots(store, head, confirmed_root)
        for block_root in canonical_roots:
            if get_block_epoch(store, block_root) == current_epoch:
                break
            if not is_ancestor(store, get_node_for_root(fcr_store.previous_slot_head), get_node_for_root(block_root)):
                break
            if not is_one_confirmed(store, get_current_balance_source(fcr_store), block_root):
                break
            confirmed_root = block_root

    if is_start_slot_at_epoch(get_current_slot(store)) or store.unrealized_justifications[head].epoch + 1 >= current_epoch:
        canonical_roots = get_ancestor_roots(store, head, confirmed_root)
        tentative_confirmed_root = confirmed_root
        for block_root in canonical_roots:
            block_epoch = get_block_epoch(store, block_root)
            tentative_confirmed_epoch = get_block_epoch(store, tentative_confirmed_root)
            if block_epoch > tentative_confirmed_epoch:
                if not will_current_target_be_justified(store):
                    break
            if not is_one_confirmed(store, get_current_balance_source(fcr_store), block_root):
                break
            tentative_confirmed_root = block_root
        if get_block_epoch(store, tentative_confirmed_root) == current_epoch or (
            get_voting_source(store, tentative_confirmed_root).epoch + 2 >= current_epoch
            and (is_start_slot_at_epoch(get_current_slot(store)) or will_no_conflicting_checkpoint_be_justified(store))
        ):
            confirmed_root = tentative_confirmed_root

    return confirmed_root


def get_latest_confirmed(fcr_store: FastConfirmationStore) -> bytes:
    store = fcr_store.store
    confirmed_root = fcr_store.confirmed_root
    current_epoch = get_current_store_epoch(store)
    head = get_head(store).root
    if (
        get_block_epoch(store, confirmed_root) + 1 < current_epoch
        or not is_ancestor(store, get_node_for_root(head), get_node_for_root(confirmed_root))
        or (is_start_slot_at_epoch(get_current_slot(store)) and not is_confirmed_chain_safe(fcr_store, confirmed_root))
    ):
        confirmed_root = store.finalized_checkpoint.root

    is_epoch_start = is_start_slot_at_epoch(get_current_slot(store))
    observed = fcr_store.current_epoch_observed_justified_checkpoint
    observed_justified_block_slot = get_block_slot(store, observed.root)
    is_observed_justified_block_epoch_ok = compute_epoch_at_slot(observed_justified_block_slot) + 1 == current_epoch
    is_head_unrealized_justified_ok = observed == store.unrealized_justifications[head]
    is_confirmed_block_stale = get_block_slot(store, confirmed_root) < observed_justified_block_slot
    if is_epoch_start and is_observed_justified_block_epoch_ok and is_head_unrealized_justified_ok and is_confirmed_block_stale:
        confirmed_root = observed.root

    if get_block_epoch(store, confirmed_root) + 1 >= current_epoch:
        return find_latest_confirmed_descendant(fcr_store, confirmed_root)
    return confirmed_root


def on_fast_confirmation(fcr_store: FastConfirmationStore) -> None:
    update_fast_confirmation_variables(fcr_store)
    fcr_store.confirmed_root = get_latest_confirmed(fcr_store)


def get_safe_execution_block_hash(fcr_store: FastConfirmationStore) -> bytes:
    """``safe_block_hash`` for ``notify_forkchoice_updated`` (specs #5542):
    the execution head the confirmed block builds on."""
    safe_block = fcr_store.store.blocks[fcr_store.confirmed_root]
    return bytes(safe_block.body.signed_execution_payload_bid.message.parent_block_hash)
