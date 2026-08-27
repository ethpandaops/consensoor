"""PeerDAS data-column handling (Fulu/Gloas).

Consensoor advertises its custody group count from validator balance (a node
with >= 128 x 32 ETH custodies everything), so it must actually take part in
DAS: verify and keep the column sidecars it custodies, serve them over
``DataColumnSidecarsByRoot/Range``, publish the sidecars of its own payloads,
and only vote ``blob_data_available`` in the PTC when its custody columns are
present.

Reference: specs/fulu/das-core.md, specs/fulu/p2p-interface.md,
specs/gloas/p2p-interface.md (validate_data_column_sidecar_gossip),
specs/gloas/builder.md (get_data_column_sidecars).
"""


import asyncio
import logging
import time
from typing import Awaitable, Callable, Iterable, Optional

from remerkleable.basic import uint64
from remerkleable.complex import Container, List

from .crypto import hash_tree_root
from .spec import kzg
from .spec.constants import (
    NUMBER_OF_COLUMNS,
    DATA_COLUMN_SIDECAR_SUBNET_COUNT,
    SLOTS_PER_EPOCH,
    NUMBER_OF_CUSTODY_GROUPS,
    VALIDATOR_CUSTODY_REQUIREMENT,
    BALANCE_PER_ADDITIONAL_CUSTODY_GROUP,
)
from .spec.types.base import Root
from .spec.types.fulu import Cell, DataColumnsByRootIdentifier
from .spec.types.gloas import DataColumnSidecar, SignedExecutionPayloadEnvelope

logger = logging.getLogger(__name__)

MAX_REQUEST_BLOCKS_DENEB = 128
MAX_REQUEST_PAYLOADS = 128

# Req/resp protocol ids served through the generic raw transport in consensoor-p2p-rs.
PROTO_ENVELOPES_BY_ROOT = "/eth2/beacon_chain/req/execution_payload_envelopes_by_root/1/ssz_snappy"
PROTO_COLUMNS_BY_ROOT = "/eth2/beacon_chain/req/data_column_sidecars_by_root/1/ssz_snappy"
PROTO_COLUMNS_BY_RANGE = "/eth2/beacon_chain/req/data_column_sidecars_by_range/1/ssz_snappy"


class DataColumnsByRootIdentifiers(List[DataColumnsByRootIdentifier, MAX_REQUEST_BLOCKS_DENEB]):
    pass


class DataColumnIndices(List[uint64, NUMBER_OF_COLUMNS]):
    pass


class DataColumnSidecarsByRangeRequest(Container):
    start_slot: uint64
    count: uint64
    columns: DataColumnIndices


class ExecutionPayloadEnvelopeRoots(List[Root, MAX_REQUEST_PAYLOADS]):
    pass


def get_validators_custody_requirement(state, validator_indices: Iterable[int]) -> int:
    """``get_validators_custody_requirement`` from fulu/validator.md: the number
    of custody groups a node with these validators attached must custody —
    one per BALANCE_PER_ADDITIONAL_CUSTODY_GROUP of attached effective
    balance, clamped to [VALIDATOR_CUSTODY_REQUIREMENT, NUMBER_OF_CUSTODY_GROUPS]."""
    validators = state.validators
    total_node_balance = sum(int(validators[int(i)].effective_balance) for i in validator_indices)
    count = total_node_balance // BALANCE_PER_ADDITIONAL_CUSTODY_GROUP()
    return min(max(count, VALIDATOR_CUSTODY_REQUIREMENT()), NUMBER_OF_CUSTODY_GROUPS())


def compute_subnet_for_data_column_sidecar(column_index: int) -> int:
    return int(column_index) % DATA_COLUMN_SIDECAR_SUBNET_COUNT()


class DataColumnManager:
    """Tracks, verifies, stores and serves data column sidecars."""

    def __init__(
        self,
        store,
        digest_for_slot: Callable[[int], Optional[bytes]],
        custody_columns: Iterable[int],
        max_blobs_for_epoch: Callable[[int], int],
    ) -> None:
        self.store = store
        self._digest_for_slot = digest_for_slot
        self.custody_columns: set[int] = set(int(c) for c in custody_columns)
        self._max_blobs_for_epoch = max_blobs_for_epoch
        # (block_root, index) we've already accepted (gossip IGNORE rule)
        self._seen: set[tuple[bytes, int]] = set()
        # block_root -> set of column indices held (mirror of the store)
        self._held: dict[bytes, set[int]] = {}

    # ------------------------------------------------------------------ custody

    def set_custody_columns(self, columns: Iterable[int]) -> None:
        self.custody_columns = set(int(c) for c in columns)

    @property
    def custody_subnets(self) -> list[int]:
        return sorted({compute_subnet_for_data_column_sidecar(c) for c in self.custody_columns})

    # ------------------------------------------------------------------ own payloads

    def build_sidecars(self, beacon_block_root: bytes, slot: int, blobs_bundle: dict) -> list[DataColumnSidecar]:
        """``get_data_column_sidecars`` for our own payload: extend every blob
        into cells + proofs and assemble one sidecar per column."""
        blobs_hex = blobs_bundle.get("blobs") or []
        if not blobs_hex:
            return []
        t0 = time.monotonic()
        cells_and_proofs = []
        for blob_hex in blobs_hex:
            blob = bytes.fromhex(blob_hex[2:] if blob_hex.startswith("0x") else blob_hex)
            cells_and_proofs.append(kzg.compute_cells_and_kzg_proofs(blob))
        sidecars = []
        for column_index in range(NUMBER_OF_COLUMNS):
            column = [Cell(cells[column_index]) for cells, _ in cells_and_proofs]
            proofs = [proofs[column_index] for _, proofs in cells_and_proofs]
            sidecars.append(
                DataColumnSidecar(
                    index=uint64(column_index),
                    column=column,
                    kzg_proofs=proofs,
                    slot=uint64(slot),
                    beacon_block_root=Root(beacon_block_root),
                )
            )
        logger.info(
            f"Built {len(sidecars)} data column sidecars for slot {slot} "
            f"({len(blobs_hex)} blobs) in {(time.monotonic() - t0) * 1000:.0f}ms"
        )
        return sidecars

    def store_sidecars(self, sidecars: Iterable[DataColumnSidecar]) -> None:
        for sc in sidecars:
            self._remember(bytes(sc.beacon_block_root), int(sc.slot), int(sc.index), bytes(sc.encode_bytes()))

    def _remember(self, root: bytes, slot: int, index: int, ssz: bytes) -> None:
        self._seen.add((root, index))
        self._held.setdefault(root, set()).add(index)
        self.store.save_data_column(root, slot, index, ssz)

    # ------------------------------------------------------------------ gossip

    def on_gossip_sidecar(self, data: bytes, subnet_id: int, get_block) -> tuple[str, str]:
        """Validate a gossip DataColumnSidecar per ``validate_data_column_sidecar_gossip``.

        Returns ("accept"|"ignore"|"reject", reason). ``get_block(root)`` must
        return the SignedBeaconBlock we hold for ``root`` or None.
        """
        try:
            sidecar = DataColumnSidecar.decode_bytes(data)
        except Exception as e:
            return "reject", f"undecodable sidecar: {e}"
        root = bytes(sidecar.beacon_block_root)
        index = int(sidecar.index)
        if (root, index) in self._seen:
            return "ignore", "already seen"
        if index >= NUMBER_OF_COLUMNS:
            return "reject", "column index out of range"
        if compute_subnet_for_data_column_sidecar(index) != int(subnet_id):
            return "reject", "wrong subnet"
        status, reason = self._verify_sidecar(sidecar, get_block)
        if status != "accept":
            return status, reason
        self._remember(root, int(sidecar.slot), index, bytes(data))
        return "accept", ""

    def ingest_sidecar(self, data: bytes, get_block) -> tuple[str, str]:
        """Verify + store a sidecar obtained through req/resp (custody backfill,
        by-root fetches). Same checks as gossip minus the subnet rule; a
        column we already hold is ignored."""
        try:
            sidecar = DataColumnSidecar.decode_bytes(data)
        except Exception as e:
            return "reject", f"undecodable sidecar: {e}"
        root = bytes(sidecar.beacon_block_root)
        index = int(sidecar.index)
        if index >= NUMBER_OF_COLUMNS:
            return "reject", "column index out of range"
        if index in self.held_columns(root):
            return "ignore", "already held"
        status, reason = self._verify_sidecar(sidecar, get_block)
        if status != "accept":
            return status, reason
        self._remember(root, int(sidecar.slot), index, bytes(data))
        return "accept", ""

    def _verify_sidecar(self, sidecar: DataColumnSidecar, get_block) -> tuple[str, str]:
        """Block-dependent checks of ``validate_data_column_sidecar_gossip``:
        the sidecar must match a block we hold and its cells must verify
        against that block's KZG commitments."""
        root = bytes(sidecar.beacon_block_root)
        index = int(sidecar.index)
        signed_block = get_block(root)
        if signed_block is None:
            return "ignore", "block not seen"
        block = signed_block.message
        if int(block.slot) != int(sidecar.slot):
            return "reject", "slot mismatch"
        try:
            commitments = [bytes(c) for c in block.body.signed_execution_payload_bid.message.blob_kzg_commitments]
        except Exception:
            return "reject", "block has no bid"
        n = len(commitments)
        if n == 0:
            return "reject", "sidecar for a block without blobs"
        if n > self._max_blobs_for_epoch(int(block.slot) // SLOTS_PER_EPOCH()):
            return "reject", "too many commitments"
        if len(sidecar.column) != n or len(sidecar.kzg_proofs) != n:
            return "reject", "column/proof length mismatch"
        cells = [bytes(c) for c in sidecar.column]
        proofs = [bytes(p) for p in sidecar.kzg_proofs]
        if not kzg.verify_cell_kzg_proof_batch(commitments, [index] * n, cells, proofs):
            return "reject", "kzg verification failed"
        return "accept", ""

    # ------------------------------------------------------------------ availability

    def held_columns(self, root: bytes) -> set[int]:
        held = self._held.get(root)
        if held is None:
            held = set(self.store.get_data_columns(root).keys())
            if held:
                self._held[root] = held
        return held

    def is_available(self, root: bytes) -> bool:
        """True if every column we custody is present for ``root``."""
        if not self.custody_columns:
            return True
        return self.custody_columns.issubset(self.held_columns(root))

    # ------------------------------------------------------------------ req/resp serving

    def _chunk(self, slot: int, ssz: bytes) -> Optional[tuple[bytes, bytes]]:
        digest = self._digest_for_slot(int(slot))
        if digest is None:
            return None
        return bytes(digest), ssz

    def serve_columns_by_root(self, payload: bytes) -> list[tuple[bytes, bytes]]:
        out: list[tuple[bytes, bytes]] = []
        try:
            request = DataColumnsByRootIdentifiers.decode_bytes(payload)
        except Exception as e:
            logger.debug(f"bad DataColumnSidecarsByRoot request: {e}")
            return out
        for ident in request:
            root = bytes(ident.block_root)
            held = self.store.get_data_columns(root)
            if not held:
                continue
            wanted = [int(i) for i in ident.columns] or sorted(held.keys())
            for index in wanted:
                ssz = held.get(index)
                if ssz is None:
                    continue
                chunk = self._chunk(int(DataColumnSidecar.decode_bytes(ssz).slot), ssz)
                if chunk:
                    out.append(chunk)
        return out

    def serve_columns_by_range(self, payload: bytes) -> list[tuple[bytes, bytes]]:
        out: list[tuple[bytes, bytes]] = []
        try:
            request = DataColumnSidecarsByRangeRequest.decode_bytes(payload)
        except Exception as e:
            logger.debug(f"bad DataColumnSidecarsByRange request: {e}")
            return out
        start, count = int(request.start_slot), int(request.count)
        wanted = sorted({int(i) for i in request.columns})
        limit = MAX_REQUEST_BLOCKS_DENEB * NUMBER_OF_COLUMNS
        for slot, root in self.store.get_column_block_roots_in_range(start, min(count, 4096)):
            held = self.store.get_data_columns(root)
            for index in (wanted or sorted(held.keys())):
                ssz = held.get(index)
                if ssz is None:
                    continue
                chunk = self._chunk(slot, ssz)
                if chunk:
                    out.append(chunk)
                if len(out) >= limit:
                    return out
        return out

    def serve_envelopes_by_root(self, payload: bytes) -> list[tuple[bytes, bytes]]:
        out: list[tuple[bytes, bytes]] = []
        try:
            roots = ExecutionPayloadEnvelopeRoots.decode_bytes(payload)
        except Exception as e:
            logger.debug(f"bad ExecutionPayloadEnvelopesByRoot request: {e}")
            return out
        for root in roots:
            signed = self.store.get_payload(bytes(root))
            if signed is None:
                continue
            try:
                slot = int(signed.message.payload.slot_number)
            except Exception:
                blk = self.store.get_block(bytes(root))
                if blk is None:
                    continue
                slot = int(blk.message.slot)
            chunk = self._chunk(slot, bytes(signed.encode_bytes()))
            if chunk:
                out.append(chunk)
        return out


# ---------------------------------------------------------------------------
# Custody backfill (fulu/validator.md, "Validator custody")
# ---------------------------------------------------------------------------


def blob_commitment_count(signed_block) -> int:
    """Number of blob KZG commitments carried by a block (0 if none / pre-Fulu)."""
    try:
        body = signed_block.message.body
        if hasattr(body, "signed_execution_payload_bid"):
            return len(body.signed_execution_payload_bid.message.blob_kzg_commitments)
        return len(body.blob_kzg_commitments)
    except Exception:
        return 0


class CustodyBackfiller:
    """Fetch the columns of newly custodied groups for the retention window.

    When a node's custody requirement grows (more validators / balance) it
    widens its custody set and SHOULD advertise the new count immediately;
    it MAY backfill the new columns for blocks it already has, lowering
    ``earliest_available_slot`` as it goes. This walks the window backwards
    in batches, asks peers for ``data_column_sidecars_by_range`` restricted to
    the new columns, verifies every sidecar against the block we hold and
    stores it. Peers that don't custody a column simply don't return it,
    so each batch is retried on the next peer until nothing is missing.
    """

    def __init__(
        self,
        das: DataColumnManager,
        request_by_range: Callable[[str, bytes], "Awaitable[Optional[list[tuple[bytes, bytes]]]]"],
        peers: Callable[[], list[str]],
        get_block,
        get_block_by_slot,
        on_progress: Optional[Callable[[int], None]] = None,
        batch_slots: int = 32,
    ) -> None:
        self.das = das
        self._request = request_by_range
        self._peers = peers
        self._get_block = get_block
        self._get_block_by_slot = get_block_by_slot
        self._on_progress = on_progress
        self.batch_slots = max(1, int(batch_slots))

    def _wanted(self, start_slot: int, count: int) -> dict[bytes, int]:
        """block_root -> slot for the blob-carrying blocks we hold in the range."""
        wanted: dict[bytes, int] = {}
        for slot in range(start_slot, start_slot + count):
            signed_block = self._get_block_by_slot(slot)
            if signed_block is None or blob_commitment_count(signed_block) == 0:
                continue
            wanted[hash_tree_root(signed_block.message)] = slot
        return wanted

    def _missing(self, wanted: dict[bytes, int], columns: set[int]) -> dict[bytes, set[int]]:
        out: dict[bytes, set[int]] = {}
        for root in wanted:
            missing = columns - self.das.held_columns(root)
            if missing:
                out[root] = missing
        return out

    async def run(self, columns: Iterable[int], start_slot: int, end_slot: int) -> dict:
        columns = {int(c) for c in columns}
        stats = {"batches": 0, "accepted": 0, "rejected": 0, "ignored": 0, "requests": 0,
                 "failed_requests": 0, "unserved_batches": 0, "reached_slot": end_slot + 1}
        if not columns or end_slot < start_slot:
            return stats
        cursor = int(end_slot)
        while cursor >= start_slot:
            batch_start = max(int(start_slot), cursor - self.batch_slots + 1)
            count = cursor - batch_start + 1
            stats["batches"] += 1
            wanted = self._wanted(batch_start, count)
            missing = self._missing(wanted, columns)
            if missing:
                for peer in self._peers():
                    # Ask only for what is still missing after the previous peer.
                    request_columns = sorted(set().union(*missing.values()))
                    payload = DataColumnSidecarsByRangeRequest(
                        start_slot=uint64(batch_start),
                        count=uint64(count),
                        columns=[uint64(c) for c in request_columns],
                    ).encode_bytes()
                    stats["requests"] += 1
                    try:
                        chunks = await self._request(peer, payload)
                    except Exception as e:
                        stats["failed_requests"] += 1
                        logger.debug(f"custody backfill: {peer} failed: {e!r}")
                        continue
                    if not chunks:
                        continue
                    for _context, ssz in chunks:
                        status, reason = self.das.ingest_sidecar(ssz, self._get_block)
                        if status == "accept":
                            stats["accepted"] += 1
                        elif status == "ignore":
                            stats["ignored"] += 1
                        else:
                            stats["rejected"] += 1
                            logger.debug(f"custody backfill: rejected sidecar from {peer}: {reason}")
                    missing = self._missing(wanted, columns)
                    if not missing:
                        break
                if missing:
                    stats["unserved_batches"] += 1
                    logger.info(
                        f"custody backfill: slots {batch_start}-{cursor}: "
                        f"{sum(len(m) for m in missing.values())} columns still missing after all peers"
                    )
            stats["reached_slot"] = batch_start
            if self._on_progress is not None:
                try:
                    self._on_progress(batch_start)
                except Exception as e:
                    logger.debug(f"custody backfill progress callback failed: {e!r}")
            cursor = batch_start - 1
            await asyncio.sleep(0)
        return stats
