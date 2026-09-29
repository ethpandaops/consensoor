"""Heze (EIP-7805, FOCIL) runtime: inclusion list gossip, duties, req/resp.

Glue between the node and the spec helpers in ``spec/inclusion_list.py`` /
``spec/fork_choice.py``:

* gossip ``inclusion_list``: validate (heze/p2p-interface.md
  ``validate_inclusion_list_gossip``) and record via ``on_inclusion_list``;
* IL committee duty: ``engine_getInclusionListV1`` -> sign -> publish;
* proposer / self-builder view: IL transactions for the payload attributes
  and ``inclusion_list_bits`` for the self-build bid (only_timely=False);
* payload validation: IL transactions for ``engine_newPayloadV6``
  (only_timely=True) and the EL's ``inclusionListSatisfied`` verdict;
* req/resp ``InclusionListsByIndices`` v1 (serving).
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import TYPE_CHECKING, Optional

from .spec.constants import SLOTS_PER_EPOCH, MIN_SEED_LOOKAHEAD
from .spec.inclusion_list import (
    build_signed_inclusion_list,
    get_inclusion_list_bits,
    get_inclusion_list_committee,
    get_inclusion_list_store,
    get_inclusion_list_transactions,
    get_inclusion_lists_by_bits,
    is_valid_inclusion_list_signature,
    process_inclusion_list,
)
from .spec.network_config import get_config

if TYPE_CHECKING:
    from .node import BeaconNode

logger = logging.getLogger(__name__)

PROTO_INCLUSION_LISTS_BY_INDICES = "/eth2/beacon_chain/req/inclusion_lists_by_indices/1/ssz_snappy"


class GossipIgnore(Exception):
    pass


class GossipReject(Exception):
    pass


def compute_shuffling_lookahead_start_slot(epoch: int) -> int:
    return max(0, epoch - MIN_SEED_LOOKAHEAD) * SLOTS_PER_EPOCH()


def compute_shuffling_dependent_slot(epoch: int) -> int:
    return max(0, compute_shuffling_lookahead_start_slot(epoch) - 1)


class FocilService:
    def __init__(self, node: "BeaconNode"):
        self.node = node
        self.seen_counts: Counter = Counter()
        self._produced: set[tuple[int, int]] = set()
        self.stats = Counter()

    # ------------------------------------------------------------------
    # chain helpers (work with or without the spec fork-choice Store)
    # ------------------------------------------------------------------

    def _block(self, root: bytes):
        try:
            signed = self.node.store.get_block(root)
        except Exception:
            signed = None
        if signed is None:
            return None
        return signed.message if hasattr(signed, "message") else signed

    def dependent_root(self, root: bytes, epoch: int) -> Optional[bytes]:
        """``get_shuffling_dependent_root``: ancestor of ``root`` at the
        shuffling dependent slot of ``epoch``."""
        fc_store = getattr(self.node, "fc_store", None)
        if fc_store is not None and root in fc_store.blocks:
            try:
                from .spec import fork_choice as fc
                return fc.get_shuffling_dependent_root(fc_store, root, epoch)
            except Exception:
                pass
        target = compute_shuffling_dependent_slot(epoch)
        cur = root
        for _ in range(4 * SLOTS_PER_EPOCH() + 8):
            blk = self._block(cur)
            if blk is None:
                return None
            if int(blk.slot) <= target:
                return cur
            parent = bytes(blk.parent_root)
            if parent == b"\x00" * 32:
                return cur
            cur = parent
        return None

    def _state_for_committee(self, dependent_root: bytes, slot: int):
        """State able to compute the IL committee of ``slot``: the dependent
        block's post-state advanced to the lookahead start (spec), falling
        back to the head state. Cached per (dependent root, epoch): advancing
        a pre-fork dependent state across a fork boundary (e.g. the last
        Fulu block for the first Heze epochs) runs the fork upgrade, which
        is far too slow to repeat for every inclusion list."""
        epoch = slot // SLOTS_PER_EPOCH()
        cache = self.node.__dict__.setdefault("_il_state_cache", {})
        key = (bytes(dependent_root), epoch)
        if key in cache:
            return cache[key]
        state = self._load_state_for_committee(dependent_root, slot)
        cache[key] = state
        while len(cache) > 4:
            cache.pop(next(iter(cache)))
        return state

    def _load_state_for_committee(self, dependent_root: bytes, slot: int):
        epoch = slot // SLOTS_PER_EPOCH()
        state = None
        fc_store = getattr(self.node, "fc_store", None)
        if fc_store is not None:
            state = fc_store.block_states.get(dependent_root)
        if state is None:
            try:
                state = self.node.store.get_state(dependent_root)
            except Exception:
                state = None
        if state is None:
            state = self.node.state
        lookahead_start = compute_shuffling_lookahead_start_slot(epoch)
        if state is not None and int(state.slot) < lookahead_start:
            from .spec.state_transition import process_slots
            state = process_slots(state.copy(), lookahead_start)
        return state

    def committee(self, dependent_root: bytes, slot: int) -> list[int]:
        cache = self.node.__dict__.setdefault("_il_committee_cache", {})
        key = (slot, dependent_root)
        if key not in cache:
            state = None
            head_state = self.node.state
            epoch = slot // SLOTS_PER_EPOCH()
            # Fast path: the head state shares the shuffling whenever its own
            # dependent root for ``epoch`` is ``dependent_root`` and it is at
            # most one epoch behind (committees are stable current..next).
            if head_state is not None and getattr(self.node, "head_root", None) is not None:
                head_epoch = int(head_state.slot) // SLOTS_PER_EPOCH()
                if epoch - 1 <= head_epoch <= epoch and self.dependent_root(
                    bytes(self.node.head_root), epoch
                ) == dependent_root:
                    state = head_state
            if state is None:
                state = self._state_for_committee(dependent_root, slot)
            cache[key] = get_inclusion_list_committee(state, slot)
            if len(cache) > 256:
                for k in sorted(cache)[:128]:
                    cache.pop(k, None)
        return cache[key]

    def _time_into_slot_ms(self, slot: int, now: Optional[float] = None) -> int:
        now = time.time() if now is None else now
        return int((now - self.node._slot_start_time(slot)) * 1000)

    def _il_due_ms(self, slot: int) -> int:
        return get_config().get_slot_component_duration_ms(
            int(get_config().inclusion_list_due_bps), slot
        )

    # ------------------------------------------------------------------
    # gossip
    # ------------------------------------------------------------------

    async def on_gossip(self, data: bytes, from_peer: str) -> None:
        from .spec.types.heze import SignedInclusionList

        now = time.time()
        try:
            signed = SignedInclusionList.decode_bytes(data)
        except Exception as e:
            self.stats["decode_fail"] += 1
            logger.debug(f"inclusion_list decode failed from {from_peer[:12]}: {e}")
            return
        try:
            await self.node._on_state_thread(self.validate_and_process, signed, now)
            self.stats["accepted"] += 1
        except GossipIgnore as e:
            self.stats["ignored"] += 1
            logger.debug(f"inclusion_list IGNORE: {e}")
        except GossipReject as e:
            self.stats["rejected"] += 1
            logger.info(f"inclusion_list REJECT from {from_peer[:12]}: {e}")
        except Exception as e:
            self.stats["error"] += 1
            logger.warning(f"inclusion_list handling failed: {e!r}")

    def validate_and_process(self, signed, now: float, local: bool = False) -> None:
        """heze/p2p-interface.md ``validate_inclusion_list_gossip`` followed by
        ``on_inclusion_list`` (timeliness from local arrival time)."""
        config = get_config()
        il = signed.message
        slot = int(il.slot)
        vi = int(il.validator_index)
        dep = bytes(il.dependent_root)
        key = (slot, dep, vi)

        if self.seen_counts[key] >= 2:
            raise GossipIgnore("already seen two valid inclusion lists from this validator")
        current_slot = self.node._wall_slot(now)
        # is_current_slot (phase0 is_within_slot_range, slot_range=0) with
        # MAXIMUM_GOSSIP_CLOCK_DISPARITY on both ends, in integer ms
        now_ms = int(round(now * 1000))
        genesis_ms = int(self.node._genesis_time) * 1000
        disparity = int(config.maximum_gossip_clock_disparity)
        start_ms = config.compute_time_at_slot_ms(genesis_ms, slot)
        end_ms = config.compute_time_at_slot_ms(genesis_ms, slot + 1)
        if now_ms + disparity < start_ms or end_ms + disparity < now_ms:
            raise GossipIgnore(f"inclusion list slot {slot} is not the current slot {current_slot}")
        txs = [bytes(t) for t in il.transactions]
        size = sum(len(t) for t in txs)
        if size == 0:
            raise GossipIgnore("inclusion list contains no transactions")
        if size > int(config.max_transactions_bytes_per_inclusion_list):
            raise GossipReject("inclusion list transactions exceed the maximum size")
        if any(len(t) == 0 for t in txs):
            raise GossipReject("inclusion list contains an empty transaction")
        dep_block = self._block(dep)
        if dep_block is None:
            raise GossipIgnore("dependent block has not been seen")
        fc_store = getattr(self.node, "fc_store", None)
        if fc_store is not None and dep in fc_store.blocks and dep not in fc_store.block_states:
            raise GossipIgnore("dependent block failed validation")
        epoch = slot // SLOTS_PER_EPOCH()
        dependent_slot = compute_shuffling_dependent_slot(epoch)
        if int(dep_block.slot) > dependent_slot:
            raise GossipReject("dependent block is after the shuffling dependent slot")
        if fc_store is not None and dep in fc_store.blocks:
            from .spec import fork_choice as fc
            if not fc.is_valid_dependent_root(fc_store, dep, dependent_slot):
                raise GossipIgnore("dependent block is not a possible dependent block")
        committee = self.committee(dep, slot)
        if vi not in committee:
            raise GossipReject("includer is not a member of the committee")
        state = self._state_for_committee(dep, slot)
        if not local and not is_valid_inclusion_list_signature(state, signed):
            raise GossipReject("invalid inclusion list signature")
        self.seen_counts[key] += 1

        timely = slot == current_slot and self._time_into_slot_ms(slot, now) < self._il_due_ms(slot)
        process_inclusion_list(get_inclusion_list_store(), signed, timely)
        api = getattr(self.node, "beacon_api", None)
        if api is not None:
            try:
                api.emit_inclusion_list(signed)
            except Exception as e:
                logger.debug(f"inclusion_list SSE emit failed: {e}")
        logger.debug(
            f"inclusion_list recorded: slot={slot} validator={vi} txs={len(txs)} "
            f"bytes={size} timely={timely} dep={dep.hex()[:12]}"
        )

    # ------------------------------------------------------------------
    # IL committee duty
    # ------------------------------------------------------------------

    async def produce(self, slot: int) -> None:
        node = self.node
        vc = getattr(node, "validator_client", None)
        if vc is None or not vc.keys or node.state is None or node.head_root is None:
            return
        if not get_config().is_heze_active(slot // SLOTS_PER_EPOCH()):
            return
        if not node._is_synced():
            return
        head_root = bytes(node.head_root)
        dep = self.dependent_root(head_root, slot // SLOTS_PER_EPOCH())
        if dep is None:
            logger.debug(f"[IL] slot={slot}: dependent root unknown")
            return
        try:
            committee = self.committee(dep, slot)
        except Exception as e:
            logger.warning(f"[IL] slot={slot}: committee computation failed: {e!r}")
            return
        members = {int(v) for v in committee}
        ours = [
            k for k in vc.keys.values()
            if k.validator_index is not None and int(k.validator_index) in members
            and (slot, int(k.validator_index)) not in self._produced
        ]
        if not ours:
            return
        try:
            txs = await node.engine.get_inclusion_list_v1()
        except Exception as e:
            logger.warning(f"[IL] slot={slot}: engine_getInclusionListV1 failed: {e!r}")
            return
        txs = [t for t in txs if len(t) > 0]
        # Honour the byte cap even if the EL overshoots.
        cap = int(get_config().max_transactions_bytes_per_inclusion_list)
        kept, total = [], 0
        for t in txs:
            if total + len(t) > cap:
                break
            kept.append(t)
            total += len(t)
        if not kept:
            logger.info(f"[IL] slot={slot}: EL returned no transactions, nothing to publish ({len(ours)} duties)")
            for k in ours:
                self._produced.add((slot, int(k.validator_index)))
            return
        state = node.state
        for key in ours:
            vi = int(key.validator_index)
            self._produced.add((slot, vi))
            if node._wall_slot() != slot:
                self.stats["late_skipped"] += 1
                logger.debug(f"[IL] slot={slot} validator={vi}: slot already over, not publishing")
                continue
            try:
                signed = build_signed_inclusion_list(state, slot, vi, dep, kept, key.privkey)
                self.validate_and_process(signed, time.time(), local=True)
                if node.beacon_gossip is not None:
                    await node.beacon_gossip.publish_inclusion_list(bytes(signed.encode_bytes()))
                self.stats["produced"] += 1
                logger.info(
                    f"[IL] published inclusion list slot={slot} validator={vi} "
                    f"txs={len(kept)} bytes={total} dep={dep.hex()[:12]}"
                )
            except Exception as e:
                logger.warning(f"[IL] slot={slot} validator={vi}: produce failed: {e!r}")

    # ------------------------------------------------------------------
    # proposer / builder / payload validation views
    # ------------------------------------------------------------------

    def payload_attribute_transactions(self, proposal_slot: int, head_root: bytes) -> list[bytes]:
        """IL transactions for building the payload of ``proposal_slot``
        (validator.md prepare_execution_payload: slot - 1, only_timely=False)."""
        if proposal_slot == 0 or not get_config().is_heze_active(proposal_slot // SLOTS_PER_EPOCH()):
            return []
        il_slot = proposal_slot - 1
        dep = self.dependent_root(head_root, il_slot // SLOTS_PER_EPOCH())
        if dep is None:
            return []
        return get_inclusion_list_transactions(get_inclusion_list_store(), il_slot, dep, only_timely=False)

    def bits_for_bid(self, state, slot: int, parent_block_root: bytes) -> list[bool]:
        """builder.md: bid.inclusion_list_bits with only_timely=False."""
        il_slot = slot - 1
        dep = self.dependent_root(parent_block_root, il_slot // SLOTS_PER_EPOCH())
        committee = get_inclusion_list_committee(state, il_slot)
        if dep is None:
            return [False] * len(committee)
        return get_inclusion_list_bits(get_inclusion_list_store(), committee, il_slot, dep, only_timely=False)

    def new_payload_transactions(
        self, block_root: bytes, block_slot: Optional[int] = None, parent_root: Optional[bytes] = None
    ) -> list[bytes]:
        """IL transactions the payload of ``block_root`` must satisfy
        (record_payload_inclusion_list_satisfaction: slot - 1, only_timely=True).

        For a block not yet in the store (own proposal) pass its slot and
        parent root; the dependent block is an ancestor of the parent.
        """
        if block_slot is None:
            blk = self._block(block_root)
            if blk is None:
                return []
            block_slot = int(blk.slot)
            parent_root = bytes(blk.parent_root)
        slot = int(block_slot) - 1
        if slot < 0 or not get_config().is_heze_active(int(block_slot) // SLOTS_PER_EPOCH()):
            return []
        dep = self.dependent_root(parent_root, slot // SLOTS_PER_EPOCH())
        if dep is None:
            return []
        return get_inclusion_list_transactions(get_inclusion_list_store(), slot, dep, only_timely=True)

    def record_satisfaction(self, block_root: bytes, satisfied: Optional[bool]) -> None:
        """Remember the EL verdict (fork choice + builder parent choice)."""
        if satisfied is None:
            return
        self.node.__dict__.setdefault("_il_satisfaction", {})[bytes(block_root)] = bool(satisfied)
        if not satisfied:
            self.stats["payload_il_unsatisfied"] += 1
            logger.warning(f"[IL] payload of block {bytes(block_root).hex()[:16]} does NOT satisfy inclusion lists")
        fc_store = getattr(self.node, "fc_store", None)
        if fc_store is not None:
            from .spec import fork_choice as fc
            fc.record_payload_inclusion_list_satisfaction(fc_store, bytes(block_root), bool(satisfied))

    # ------------------------------------------------------------------
    # req/resp
    # ------------------------------------------------------------------

    def serve_by_indices(self, payload: bytes) -> list[tuple[bytes, bytes]]:
        from .spec.types.heze import InclusionListsByIndicesRequest

        try:
            req = InclusionListsByIndicesRequest.decode_bytes(payload)
        except Exception as e:
            logger.debug(f"bad InclusionListsByIndices request: {e}")
            return []
        slot = int(req.slot)
        current = self.node._wall_slot()
        heze_start = int(get_config().heze_fork_epoch) * SLOTS_PER_EPOCH()
        minimum = max(current - int(get_config().min_slots_for_inclusion_lists_requests), heze_start)
        if slot < minimum or slot > current:
            return []
        dep = bytes(req.dependent_root)
        try:
            committee = self.committee(dep, slot)
        except Exception:
            return []
        lists = get_inclusion_lists_by_bits(get_inclusion_list_store(), slot, dep, committee, req.indices)
        digest = self.node._digest_for_slot(slot)
        if digest is None:
            return []
        max_n = int(get_config().max_request_inclusion_list)
        return [(bytes(digest), bytes(sl.encode_bytes())) for sl in lists[:max_n]]

    # ------------------------------------------------------------------

    def prune(self, current_slot: int) -> None:
        keep = current_slot - int(get_config().min_slots_for_inclusion_lists_requests) - 2
        get_inclusion_list_store().prune(keep)
        for k in [k for k in self.seen_counts if k[0] < keep]:
            del self.seen_counts[k]
        self._produced = {k for k in self._produced if k[0] >= keep}
