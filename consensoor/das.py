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


import logging
import time
from typing import Callable, Iterable, Optional

from remerkleable.basic import uint64
from remerkleable.complex import Container, List

from .crypto import hash_tree_root
from .spec import kzg
from .spec.constants import NUMBER_OF_COLUMNS, DATA_COLUMN_SIDECAR_SUBNET_COUNT, SLOTS_PER_EPOCH
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
        self._remember(root, int(sidecar.slot), index, bytes(data))
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
