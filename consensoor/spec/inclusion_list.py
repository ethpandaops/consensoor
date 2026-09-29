"""Heze (EIP-7805, FOCIL) inclusion list helpers.

Mirrors specs/heze/{beacon-chain,inclusion-list,validator}.md. The
``InclusionListStore`` is a process-wide singleton (spec
``get_inclusion_list_store``); the node prunes it by slot.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from ..crypto import sign, verify
from .constants import (
    DOMAIN_INCLUSION_LIST_COMMITTEE,
    INCLUSION_LIST_COMMITTEE_SIZE,
    SLOTS_PER_EPOCH,
)
from .state_transition.helpers.beacon_committee import (
    get_beacon_committee,
    get_committee_count_per_slot,
)
from .state_transition.helpers.domain import compute_domain, compute_signing_root
from .state_transition.helpers.misc import compute_epoch_at_slot


# ---------------------------------------------------------------------------
# beacon-chain.md
# ---------------------------------------------------------------------------

def get_inclusion_list_committee(state, slot: int) -> list[int]:
    """Concatenate all beacon committees of ``slot`` and take the first
    INCLUSION_LIST_COMMITTEE_SIZE members, wrapping around."""
    epoch = compute_epoch_at_slot(slot)
    indices: list[int] = []
    for i in range(get_committee_count_per_slot(state, epoch)):
        indices.extend(int(v) for v in get_beacon_committee(state, slot, i))
    return [indices[i % len(indices)] for i in range(INCLUSION_LIST_COMMITTEE_SIZE)]


def inclusion_list_domain(state, slot: int) -> bytes:
    """DOMAIN_INCLUSION_LIST_COMMITTEE at the fork version of ``slot``'s epoch.

    The spec takes ``get_domain(state, ...)`` from the dependent block's
    state advanced only to the lookahead start (the previous epoch). In the
    first epoch of a fork that state still carries the old fork version,
    while the signer's state is already upgraded, so every IL of that epoch
    would fail verification. Pinning the domain to the message epoch's fork
    version (as specs #5665 did for proposer preferences) makes signer and
    verifier agree regardless of which state each holds.
    """
    from .network_config import get_config
    fork_version = get_config().get_fork_version(compute_epoch_at_slot(int(slot)))
    return compute_domain(
        DOMAIN_INCLUSION_LIST_COMMITTEE, fork_version, bytes(state.genesis_validators_root)
    )


def is_valid_inclusion_list_signature(state, signed_inclusion_list) -> bool:
    message = signed_inclusion_list.message
    index = int(message.validator_index)
    if index >= len(state.validators):
        return False
    pubkey = bytes(state.validators[index].pubkey)
    domain = inclusion_list_domain(state, int(message.slot))
    signing_root = compute_signing_root(message, domain)
    return verify(pubkey, signing_root, bytes(signed_inclusion_list.signature))


# ---------------------------------------------------------------------------
# validator.md
# ---------------------------------------------------------------------------

def get_inclusion_list_committee_assignment(
    state, epoch: int, validator_index: int
) -> Optional[int]:
    start_slot = epoch * SLOTS_PER_EPOCH()
    for slot in range(start_slot, start_slot + SLOTS_PER_EPOCH()):
        if validator_index in get_inclusion_list_committee(state, slot):
            return slot
    return None


def get_inclusion_list_signature(state, inclusion_list, privkey: int) -> bytes:
    domain = inclusion_list_domain(state, int(inclusion_list.slot))
    return sign(privkey, compute_signing_root(inclusion_list, domain))


def build_signed_inclusion_list(
    state,
    slot: int,
    validator_index: int,
    dependent_root: bytes,
    transactions: Sequence[bytes],
    privkey: int,
):
    """spec ``get_signed_inclusion_list`` with the dependent root resolved by
    the caller (it needs the fork-choice store)."""
    from .types.heze import InclusionList, SignedInclusionList, Transactions

    inclusion_list = InclusionList(
        slot=slot,
        validator_index=validator_index,
        dependent_root=dependent_root,
        transactions=Transactions(*[bytes(tx) for tx in transactions]),
    )
    signature = get_inclusion_list_signature(state, inclusion_list, privkey)
    return SignedInclusionList(message=inclusion_list, signature=signature)


# ---------------------------------------------------------------------------
# inclusion-list.md
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InclusionListEntry:
    signed_inclusion_list: object
    timely: bool
    # hash_tree_root of the message, used for equivocation comparison
    message_root: bytes


@dataclass
class InclusionListStore:
    inclusion_lists: dict = field(default_factory=lambda: defaultdict(dict))
    equivocators: dict = field(default_factory=lambda: defaultdict(set))
    lock: threading.RLock = field(default_factory=threading.RLock)

    def prune(self, min_slot: int) -> None:
        """Drop inclusion lists for slots < ``min_slot``."""
        with self.lock:
            for key in [k for k in self.inclusion_lists if k[0] < min_slot]:
                self.inclusion_lists.pop(key, None)
            for key in [k for k in self.equivocators if k[0] < min_slot]:
                self.equivocators.pop(key, None)


_store: Optional[InclusionListStore] = None


def get_inclusion_list_store() -> InclusionListStore:
    global _store
    if _store is None:
        _store = InclusionListStore()
    return _store


def reset_inclusion_list_store() -> None:
    global _store
    _store = None


def process_inclusion_list(store: InclusionListStore, signed_inclusion_list, timely: bool) -> None:
    inclusion_list = signed_inclusion_list.message
    validator_index = int(inclusion_list.validator_index)
    key = (int(inclusion_list.slot), bytes(inclusion_list.dependent_root))
    message_root = bytes(inclusion_list.hash_tree_root())

    with store.lock:
        lists = store.inclusion_lists[key]
        if validator_index in lists:
            # A different inclusion list from the same validator is an equivocation
            if lists[validator_index].message_root != message_root:
                store.equivocators[key].add(validator_index)
            return
        lists[validator_index] = InclusionListEntry(
            signed_inclusion_list=signed_inclusion_list,
            timely=timely,
            message_root=message_root,
        )


def _valid_entries(store: InclusionListStore, slot: int, dependent_root: bytes, only_timely: bool):
    key = (int(slot), bytes(dependent_root))
    with store.lock:
        lists = dict(store.inclusion_lists.get(key, {}))
        equivocators = set(store.equivocators.get(key, set()))
    for validator_index, entry in lists.items():
        if validator_index in equivocators:
            continue
        if only_timely and not entry.timely:
            continue
        yield validator_index, entry


def get_inclusion_list_transactions(
    store: InclusionListStore, slot: int, dependent_root: bytes, only_timely: bool = True
) -> list[bytes]:
    """Unique transactions from all valid, non-equivocating inclusion lists.

    Order is not significant to the spec; we keep first-seen order so the
    engine sees a stable list.
    """
    seen: set[bytes] = set()
    out: list[bytes] = []
    for _, entry in _valid_entries(store, slot, dependent_root, only_timely):
        for tx in entry.signed_inclusion_list.message.transactions:
            b = bytes(tx)
            if b not in seen:
                seen.add(b)
                out.append(b)
    return out


def get_inclusion_list_bits(
    store: InclusionListStore,
    committee: Iterable[int],
    slot: int,
    dependent_root: bytes,
    only_timely: bool = True,
) -> list[bool]:
    indices = {vi for vi, _ in _valid_entries(store, slot, dependent_root, only_timely)}
    return [int(v) in indices for v in committee]


def is_inclusion_list_bits_inclusive(
    store: InclusionListStore,
    committee: Iterable[int],
    slot: int,
    dependent_root: bytes,
    inclusion_list_bits,
    only_timely: bool = True,
) -> bool:
    local = get_inclusion_list_bits(store, committee, slot, dependent_root, only_timely)
    for i in range(INCLUSION_LIST_COMMITTEE_SIZE):
        if local[i] and not bool(inclusion_list_bits[i]):
            return False
    return True


def get_inclusion_lists_by_bits(
    store: InclusionListStore, slot: int, dependent_root: bytes, committee, bits
) -> list:
    """Serve InclusionListsByIndices: valid, non-equivocating lists whose
    committee position is set in ``bits``."""
    wanted = {int(committee[i]) for i in range(len(committee)) if bool(bits[i])}
    return [
        entry.signed_inclusion_list
        for vi, entry in _valid_entries(store, slot, dependent_root, only_timely=False)
        if vi in wanted
    ]
