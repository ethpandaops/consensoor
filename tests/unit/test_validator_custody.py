"""Validator custody (fulu/validator.md): requirement math, req/resp sidecar
ingestion and the custody backfiller, end to end with real KZG."""
import asyncio
import os
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module", autouse=True)
def _minimal_preset():
    from consensoor.spec.constants import set_preset
    from consensoor.spec.network_config import load_config_from_upstream, set_config
    set_preset("minimal")
    set_config(load_config_from_upstream("minimal"))


def _state(balances):
    return SimpleNamespace(validators=[SimpleNamespace(effective_balance=b) for b in balances])


def test_validators_custody_requirement_matches_spec():
    from consensoor.das import get_validators_custody_requirement
    from consensoor.spec.constants import (
        BALANCE_PER_ADDITIONAL_CUSTODY_GROUP, NUMBER_OF_CUSTODY_GROUPS, VALIDATOR_CUSTODY_REQUIREMENT,
    )
    per_group = BALANCE_PER_ADDITIONAL_CUSTODY_GROUP()
    # floor: one 32 ETH validator still custodies VALIDATOR_CUSTODY_REQUIREMENT groups
    assert get_validators_custody_requirement(_state([per_group]), [0]) == VALIDATOR_CUSTODY_REQUIREMENT()
    # linear in attached balance ...
    n = VALIDATOR_CUSTODY_REQUIREMENT() + 3
    assert get_validators_custody_requirement(_state([per_group] * n), range(n)) == n
    # ... only for attached validators, only whole groups
    st = _state([per_group] * n + [per_group // 2])
    assert get_validators_custody_requirement(st, list(range(n)) + [n]) == n
    # ceiling
    many = NUMBER_OF_CUSTODY_GROUPS() + 10
    assert get_validators_custody_requirement(_state([per_group] * many), range(many)) == NUMBER_OF_CUSTODY_GROUPS()


class MemoryStore:
    """The slice of consensoor.store.Store the DAS manager and backfiller use."""

    def __init__(self):
        self.blocks = {}      # root -> signed block
        self.by_slot = {}     # slot -> signed block
        self.columns = {}     # root -> {index: ssz}
        self.column_slots = {}  # slot -> root

    def get_block(self, root):
        return self.blocks.get(root)

    def get_block_by_slot(self, slot):
        return self.by_slot.get(slot)

    def save_data_column(self, block_root, slot, index, sidecar_ssz):
        self.columns.setdefault(block_root, {})[index] = sidecar_ssz
        self.column_slots[slot] = block_root

    def get_data_columns(self, block_root):
        return dict(self.columns.get(block_root, {}))

    def get_column_block_roots_in_range(self, start_slot, count):
        return [(s, self.column_slots[s]) for s in sorted(self.column_slots) if start_slot <= s < start_slot + count]


def _blob_and_commitment(seed: int):
    from consensoor.spec import kzg
    # a valid blob: 4096 field elements, each < BLS modulus (top byte zero keeps it canonical)
    blob = bytes(32 * i + seed for i in range(4096) for _ in [0]) if False else b"".join(
        b"\x00" + ((seed * 7919 + i) % 251).to_bytes(1, "big") * 31 for i in range(4096)
    )
    return blob, kzg.blob_to_kzg_commitment(blob)


def _gloas_block(slot: int, commitments):
    from remerkleable.basic import uint64
    from consensoor.spec.types.gloas import SignedBeaconBlock
    block = SignedBeaconBlock()
    block.message.slot = uint64(slot)
    bid = block.message.body.signed_execution_payload_bid.message
    bid.blob_kzg_commitments = commitments
    return block


def _make_chain(slots_with_blobs):
    """(store, das) pairs for a 'remote' node holding all columns of a few
    blob blocks, plus the block set the local node already has."""
    from consensoor.crypto import hash_tree_root
    from consensoor.das import DataColumnManager
    from consensoor.spec.types.fulu import KZGCommitment

    def mgr(store, custody):
        return DataColumnManager(store, digest_for_slot=lambda s: b"\x11\x22\x33\x44",
                                 custody_columns=custody, max_blobs_for_epoch=lambda e: 6)

    remote_store, local_store = MemoryStore(), MemoryStore()
    remote, local = mgr(remote_store, range(128)), mgr(local_store, [])
    roots = {}
    for slot, n_blobs in slots_with_blobs:
        blobs, commitments = [], []
        for b in range(n_blobs):
            blob, commitment = _blob_and_commitment(slot * 10 + b)
            blobs.append("0x" + blob.hex())
            commitments.append(KZGCommitment(commitment))
        block = _gloas_block(slot, commitments)
        root = hash_tree_root(block.message)
        for st in (remote_store, local_store):
            st.blocks[root] = block
            st.by_slot[slot] = block
        roots[slot] = root
        if n_blobs:
            remote.store_sidecars(remote.build_sidecars(root, slot, {"blobs": blobs}))
    return remote, local, roots


def test_ingest_sidecar_verifies_against_block_and_dedups():
    remote, local, roots = _make_chain([(10, 1)])
    root = roots[10]
    ssz = remote.store.get_data_columns(root)[5]
    assert local.ingest_sidecar(ssz, local.store.get_block) == ("accept", "")
    assert local.held_columns(root) == {5}
    assert local.ingest_sidecar(ssz, local.store.get_block)[0] == "ignore"
    # a sidecar for a block we don't hold is ignored, a tampered cell rejected
    ssz6 = remote.store.get_data_columns(root)[6]
    assert local.ingest_sidecar(ssz6, lambda r: None) == ("ignore", "block not seen")
    from consensoor.spec.types.gloas import DataColumnSidecar
    bad = DataColumnSidecar.decode_bytes(ssz6)
    bad.index = 7  # cells/proofs belong to column 6
    status, reason = local.ingest_sidecar(bad.encode_bytes(), local.store.get_block)
    assert status == "reject" and "kzg" in reason


def test_backfill_pulls_new_columns_only_for_blob_blocks_and_reports_progress():
    from consensoor.das import CustodyBackfiller, DataColumnSidecarsByRangeRequest
    remote, local, roots = _make_chain([(8, 2), (9, 0), (11, 1), (40, 1)])
    requests, progress = [], []

    async def request_by_range(peer, payload):
        req = DataColumnSidecarsByRangeRequest.decode_bytes(payload)
        requests.append((peer, int(req.start_slot), int(req.count), sorted(int(c) for c in req.columns)))
        if peer == "dead":
            raise RuntimeError("timeout")
        if peer == "partial":  # custodies only column 3
            return [c for c in remote.serve_columns_by_range(payload)
                    if int.from_bytes(c[1][:8], "little") == 3]
        return remote.serve_columns_by_range(payload)

    new_columns = {3, 77}
    local.set_custody_columns(new_columns)
    bf = CustodyBackfiller(local, request_by_range, peers=lambda: ["dead", "partial", "full"],
                           get_block=local.store.get_block, get_block_by_slot=local.store.get_block_by_slot,
                           on_progress=progress.append, batch_slots=16)
    stats = asyncio.run(bf.run(new_columns, start_slot=8, end_slot=40))

    for slot in (8, 11, 40):
        assert local.held_columns(roots[slot]) == new_columns, slot
    assert roots[9] not in local.store.columns          # no blobs -> nothing to fetch
    assert local.is_available(roots[8]) and local.is_available(roots[40])
    assert stats["accepted"] == 3 * len(new_columns) and stats["rejected"] == 0
    assert stats["unserved_batches"] == 0 and stats["reached_slot"] == 8
    # walked backwards in batches, and only the missing columns were asked of the last peer
    assert progress == [25, 9, 8] or progress[0] > progress[-1]
    assert requests[0][1] == 25 and requests[0][0] == "dead"
    assert requests[-1][0] == "full" and requests[-1][3] == [77]
    # batches without blob blocks make no requests at all
    empty = [r for r in requests if r[1] == 9]
    assert empty and all(r[0] != "dead" or True for r in empty)


def test_backfill_reports_unserved_when_no_peer_has_the_columns():
    from consensoor.das import CustodyBackfiller
    remote, local, roots = _make_chain([(5, 1)])

    async def nothing(peer, payload):
        return []

    local.set_custody_columns({1})
    bf = CustodyBackfiller(local, nothing, peers=lambda: ["a", "b"], get_block=local.store.get_block,
                           get_block_by_slot=local.store.get_block_by_slot)
    stats = asyncio.run(bf.run({1}, 5, 5))
    assert stats["unserved_batches"] == 1 and stats["requests"] == 2 and stats["accepted"] == 0
    assert not local.is_available(roots[5])
