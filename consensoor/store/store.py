"""LevelDB-backed store for beacon state and blocks."""

import logging
import struct
from typing import Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

PREFIX_STATE = b"s:"
PREFIX_STATE_SLOT = b"ss:"
PREFIX_BLOCK = b"b:"
PREFIX_BLOCK_SLOT = b"bs:"
PREFIX_BLOCK_PARENT = b"bp:"
PREFIX_PAYLOAD = b"p:"
PREFIX_BLOBS = b"bl:"
PREFIX_META = b"m:"


class Store:
    """LevelDB-backed store for beacon state and blocks.

    Stores SSZ-encoded states and blocks in LevelDB for persistence.
    Keeps recent items in memory cache for fast access.
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else Path(".")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._state_cache: dict[bytes, object] = {}
        self._block_cache: dict[bytes, object] = {}
        self._payload_cache: dict[bytes, object] = {}
        self._blob_cache: dict[bytes, list] = {}
        self._bid_cache: dict[int, object] = {}
        self._cache_limit = 128

        self.head_root: Optional[bytes] = None
        self.finalized_root: bytes = b"\x00" * 32
        self.finalized_epoch: int = 0
        self.justified_root: bytes = b"\x00" * 32
        self.justified_epoch: int = 0

        self._db_path = self.data_dir / "beacon.ldb"
        self._db = None
        self._init_db()
        self._load_metadata()

    def _init_db(self) -> None:
        """Initialize LevelDB database."""
        import plyvel
        self._db = plyvel.DB(
            str(self._db_path),
            create_if_missing=True,
            write_buffer_size=64 * 1024 * 1024,
            max_open_files=512,
            bloom_filter_bits=10,
        )
        logger.info(f"LevelDB store initialized at {self._db_path}")

    def _load_metadata(self) -> None:
        """Load metadata from database."""
        head = self._db.get(PREFIX_META + b"head_root")
        if head:
            self.head_root = head

        finalized = self._db.get(PREFIX_META + b"finalized_root")
        if finalized:
            self.finalized_root = finalized

        finalized_epoch = self._db.get(PREFIX_META + b"finalized_epoch")
        if finalized_epoch:
            self.finalized_epoch = int.from_bytes(finalized_epoch, "little")

        justified = self._db.get(PREFIX_META + b"justified_root")
        if justified:
            self.justified_root = justified

        justified_epoch = self._db.get(PREFIX_META + b"justified_epoch")
        if justified_epoch:
            self.justified_epoch = int.from_bytes(justified_epoch, "little")

        if self.head_root:
            logger.info(f"Loaded head_root from DB: {self.head_root.hex()[:16]}")

    def _save_metadata(self, key: str, value: bytes) -> None:
        """Save a metadata value to database."""
        self._db.put(PREFIX_META + key.encode(), value)

    def _detect_fork(self, obj: Any) -> str:
        """Detect the fork type of a state or block."""
        type_name = type(obj).__name__
        module_name = type(obj).__module__

        logger.info(f"Fork detection: type={type_name}, module={module_name}")

        # Check module name for fork-specific modules
        if "gloas" in module_name.lower():
            return "gloas"
        if "fulu" in module_name.lower():
            return "fulu"

        if "Fulu" in type_name:
            return "fulu"
        if "Gloas" in type_name:
            return "gloas"

        # For states/blocks where the class name doesn't include the fork name,
        # check for fork-specific attributes
        inner = obj.message if hasattr(obj, "message") else obj
        body = inner.body if hasattr(inner, "body") else inner

        # GLOAS has builders list and latest_block_hash in state, and
        # signed_execution_payload_bid in block body
        # Use try/except to handle SSZ containers which may not support hasattr properly
        try:
            if hasattr(body, "signed_execution_payload_bid") or hasattr(obj, "builders"):
                return "gloas"
        except Exception:
            pass

        # Fulu has proposer_lookahead in state
        try:
            if hasattr(obj, "proposer_lookahead") and not hasattr(obj, "builders"):
                return "fulu"
        except Exception:
            pass

        if "Electra" in type_name:
            msg = obj.message if hasattr(obj, "message") else obj
            if hasattr(msg, "slot"):
                from ..spec.network_config import get_config
                from ..spec.constants import SLOTS_PER_EPOCH
                try:
                    config = get_config()
                    slot = int(msg.slot)
                    epoch = slot // SLOTS_PER_EPOCH()
                    logger.info(f"Fork detection for Electra type: slot={slot}, epoch={epoch}, fulu_fork_epoch={config.fulu_fork_epoch}")
                    if hasattr(config, 'fulu_fork_epoch') and epoch >= config.fulu_fork_epoch:
                        logger.info(f"Detected as fulu (epoch {epoch} >= fulu_fork_epoch {config.fulu_fork_epoch})")
                        return "fulu"
                except Exception as e:
                    logger.warning(f"Fork detection exception: {e}")
            return "electra"

        if "Deneb" in type_name:
            return "deneb"
        if "Capella" in type_name:
            return "capella"
        if "Bellatrix" in type_name:
            return "bellatrix"
        if "Altair" in type_name:
            return "altair"
        return "phase0"

    def _get_state_types(self):
        """Get all beacon state types for deserialization."""
        from ..spec.types.fulu import FuluBeaconState
        from ..spec.types.gloas import BeaconState as GloasBeaconState
        from ..spec.types.electra import ElectraBeaconState
        from ..spec.types.deneb import DenebBeaconState
        from ..spec.types.capella import CapellaBeaconState
        from ..spec.types.bellatrix import BellatrixBeaconState
        from ..spec.types.altair import AltairBeaconState
        from ..spec.types.phase0 import Phase0BeaconState

        return {
            "fulu": FuluBeaconState,
            "gloas": GloasBeaconState,
            "electra": ElectraBeaconState,
            "deneb": DenebBeaconState,
            "capella": CapellaBeaconState,
            "bellatrix": BellatrixBeaconState,
            "altair": AltairBeaconState,
            "phase0": Phase0BeaconState,
        }

    def _get_block_types(self):
        """Get all signed beacon block types for deserialization."""
        from ..spec.types.fulu import FuluSignedBeaconBlock
        from ..spec.types.gloas import SignedBeaconBlock as GloasSignedBeaconBlock
        from ..spec.types.electra import ElectraSignedBeaconBlock
        from ..spec.types.deneb import DenebSignedBeaconBlock
        from ..spec.types.capella import CapellaSignedBeaconBlock
        from ..spec.types.bellatrix import BellatrixSignedBeaconBlock
        from ..spec.types.altair import AltairSignedBeaconBlock
        from ..spec.types.phase0 import Phase0SignedBeaconBlock

        return {
            "fulu": FuluSignedBeaconBlock,
            "gloas": GloasSignedBeaconBlock,
            "electra": ElectraSignedBeaconBlock,
            "deneb": DenebSignedBeaconBlock,
            "capella": CapellaSignedBeaconBlock,
            "bellatrix": BellatrixSignedBeaconBlock,
            "altair": AltairSignedBeaconBlock,
            "phase0": Phase0SignedBeaconBlock,
        }

    def _encode_value(self, fork: str, data: bytes) -> bytes:
        """Encode fork and data into a single value."""
        fork_bytes = fork.encode().ljust(16, b'\x00')
        return fork_bytes + data

    def _decode_value(self, value: bytes) -> tuple[str, bytes]:
        """Decode fork and data from a value."""
        fork = value[:16].rstrip(b'\x00').decode()
        data = value[16:]
        return fork, data

    def save_state(self, root: bytes, state: object) -> None:
        """Save a beacon state by root."""
        self._state_cache[root] = state
        self._trim_cache(self._state_cache)

        try:
            fork = self._detect_fork(state)
            slot = int(state.slot) if hasattr(state, "slot") else 0
            data = state.encode_bytes()

            batch = self._db.write_batch()
            batch.put(PREFIX_STATE + root, self._encode_value(fork, data))
            batch.put(PREFIX_STATE_SLOT + struct.pack(">Q", slot), root)
            batch.write()

            logger.debug(f"Saved state: slot={slot}, fork={fork}, root={root.hex()[:16]}")
        except Exception as e:
            logger.warning(f"Failed to persist state to LevelDB: {e}")

    def get_state(self, root: bytes) -> Optional[object]:
        """Get a beacon state by root."""
        logger.info(f"get_state called with root={root.hex()[:16]}, cache_size={len(self._state_cache)}")
        if root in self._state_cache:
            logger.info(f"get_state: found in cache")
            return self._state_cache[root]

        try:
            logger.info(f"get_state: checking LevelDB")
            value = self._db.get(PREFIX_STATE + root)
            if value:
                logger.info(f"get_state: found in LevelDB, decoding...")
                fork, data = self._decode_value(value)
                state_types = self._get_state_types()
                state_type = state_types.get(fork)
                if state_type:
                    state = state_type.decode_bytes(data)
                    self._state_cache[root] = state
                    logger.info(f"get_state: successfully decoded state for fork={fork}")
                    return state
                else:
                    logger.warning(f"get_state: no state type for fork={fork}")
            else:
                logger.info(f"get_state: not found in LevelDB")
        except Exception as e:
            logger.warning(f"Failed to load state from LevelDB: {e}")

        return None

    def get_state_by_slot(self, slot: int) -> Optional[object]:
        """Get a beacon state by slot (returns first match)."""
        try:
            root = self._db.get(PREFIX_STATE_SLOT + struct.pack(">Q", slot))
            if root:
                return self.get_state(root)
        except Exception as e:
            logger.warning(f"Failed to load state by slot from LevelDB: {e}")
        return None

    def save_block(self, root: bytes, block: object) -> None:
        """Save a signed beacon block by root."""
        self._block_cache[root] = block
        self._trim_cache(self._block_cache)

        try:
            fork = self._detect_fork(block)
            msg = block.message if hasattr(block, "message") else block
            slot = int(msg.slot) if hasattr(msg, "slot") else 0
            parent_root = bytes(msg.parent_root) if hasattr(msg, "parent_root") else b"\x00" * 32
            data = block.encode_bytes()

            batch = self._db.write_batch()
            batch.put(PREFIX_BLOCK + root, self._encode_value(fork, data))
            batch.put(PREFIX_BLOCK_SLOT + struct.pack(">Q", slot), root)
            batch.put(PREFIX_BLOCK_PARENT + parent_root + root, b"")
            batch.write()

            logger.debug(f"Saved block: slot={slot}, fork={fork}, root={root.hex()[:16]}")
        except Exception as e:
            logger.warning(f"Failed to persist block to LevelDB: {e}")

    def get_block(self, root: bytes) -> Optional[object]:
        """Get a signed beacon block by root."""
        if root in self._block_cache:
            return self._block_cache[root]

        try:
            value = self._db.get(PREFIX_BLOCK + root)
            if value:
                fork, data = self._decode_value(value)
                block_types = self._get_block_types()
                block_type = block_types.get(fork)
                if block_type:
                    block = block_type.decode_bytes(data)
                    self._block_cache[root] = block
                    return block
        except Exception as e:
            logger.warning(f"Failed to load block from LevelDB: {e}")

        return None

    def get_block_by_slot(self, slot: int) -> Optional[object]:
        """Get a signed beacon block by slot (returns first match)."""
        try:
            root = self._db.get(PREFIX_BLOCK_SLOT + struct.pack(">Q", slot))
            if root:
                return self.get_block(root)
        except Exception as e:
            logger.warning(f"Failed to load block by slot from LevelDB: {e}")
        return None

    def get_blocks_by_parent(self, parent_root: bytes) -> list:
        """Get all blocks with a given parent root."""
        blocks = []
        try:
            prefix = PREFIX_BLOCK_PARENT + parent_root
            for key, _ in self._db.iterator(prefix=prefix):
                root = key[len(prefix):]
                block = self.get_block(root)
                if block:
                    blocks.append(block)
        except Exception as e:
            logger.warning(f"Failed to load blocks by parent from LevelDB: {e}")
        return blocks

    def save_payload(self, root: bytes, payload: object) -> None:
        """Save an execution payload envelope by root."""
        self._payload_cache[root] = payload
        self._trim_cache(self._payload_cache)

        try:
            data = payload.encode_bytes() if hasattr(payload, "encode_bytes") else b""
            if data:
                self._db.put(PREFIX_PAYLOAD + root, data)
        except Exception as e:
            logger.debug(f"Failed to persist payload to LevelDB: {e}")

    def get_payload(self, root: bytes) -> Optional[object]:
        """Get an execution payload envelope by root."""
        return self._payload_cache.get(root)

    def save_bid(self, slot: int, bid: object) -> None:
        """Save an execution payload bid by slot."""
        self._bid_cache[slot] = bid

    def get_bid(self, slot: int) -> Optional[object]:
        """Get an execution payload bid by slot."""
        return self._bid_cache.get(slot)

    def save_blobs(self, block_root: bytes, slot: int, blobs_bundle: dict, kzg_commitments: list, signed_block=None) -> None:
        """Save blob sidecars for a block.

        Args:
            block_root: Root hash of the block
            slot: Slot number
            blobs_bundle: Dict with blobs, commitments, proofs from execution layer
            kzg_commitments: List of KZG commitments from block body
            signed_block: Optional signed beacon block for header info
        """
        import json

        blobs = blobs_bundle.get("blobs", [])
        commitments = blobs_bundle.get("commitments", [])
        proofs = blobs_bundle.get("proofs", [])

        # Extract header info from signed block if available
        header_info = {
            "slot": str(slot),
            "proposer_index": "0",
            "parent_root": "0x" + "00" * 32,
            "state_root": "0x" + "00" * 32,
            "body_root": "0x" + "00" * 32,
        }
        signature = "0x" + "00" * 96

        if signed_block is not None:
            block = signed_block.message if hasattr(signed_block, "message") else signed_block
            if hasattr(block, "proposer_index"):
                header_info["proposer_index"] = str(block.proposer_index)
            if hasattr(block, "parent_root"):
                header_info["parent_root"] = "0x" + bytes(block.parent_root).hex()
            if hasattr(block, "state_root"):
                header_info["state_root"] = "0x" + bytes(block.state_root).hex()
            if hasattr(block, "body"):
                from ..crypto import hash_tree_root
                header_info["body_root"] = "0x" + hash_tree_root(block.body).hex()
            if hasattr(signed_block, "signature"):
                signature = "0x" + bytes(signed_block.signature).hex()

        sidecars = []
        for i in range(len(blobs)):
            sidecar = {
                "index": str(i),
                "blob": blobs[i] if i < len(blobs) else "0x" + "00" * 131072,
                "kzg_commitment": commitments[i] if i < len(commitments) else "0x" + "00" * 48,
                "kzg_proof": proofs[i] if i < len(proofs) else "0x" + "00" * 48,
                "signed_block_header": {
                    "message": header_info,
                    "signature": signature,
                },
                "kzg_commitment_inclusion_proof": ["0x" + "00" * 32] * 17,
            }
            sidecars.append(sidecar)

        self._blob_cache[block_root] = sidecars
        self._trim_cache(self._blob_cache)

        try:
            data = json.dumps(sidecars).encode()
            self._db.put(PREFIX_BLOBS + block_root, data)
            logger.debug(f"Saved {len(sidecars)} blob sidecars for block {block_root.hex()[:16]}")
        except Exception as e:
            logger.warning(f"Failed to persist blobs to LevelDB: {e}")

    def get_blobs(self, block_root: bytes) -> list:
        """Get blob sidecars for a block."""
        import json

        if block_root in self._blob_cache:
            return self._blob_cache[block_root]

        try:
            value = self._db.get(PREFIX_BLOBS + block_root)
            if value:
                sidecars = json.loads(value.decode())
                self._blob_cache[block_root] = sidecars
                return sidecars
        except Exception as e:
            logger.warning(f"Failed to load blobs from LevelDB: {e}")

        return []

    def set_head(self, root: bytes) -> None:
        """Set the current head root."""
        self.head_root = root
        self._save_metadata("head_root", root)

    def set_finalized(self, root: bytes, epoch: int) -> None:
        """Set the finalized checkpoint."""
        self.finalized_root = root
        self.finalized_epoch = epoch
        self._save_metadata("finalized_root", root)
        self._save_metadata("finalized_epoch", epoch.to_bytes(8, "little"))

    def set_justified(self, root: bytes, epoch: int) -> None:
        """Set the justified checkpoint."""
        self.justified_root = root
        self.justified_epoch = epoch
        self._save_metadata("justified_root", root)
        self._save_metadata("justified_epoch", epoch.to_bytes(8, "little"))

    def _trim_cache(self, cache: dict, limit: Optional[int] = None) -> None:
        """Trim cache to limit size."""
        limit = limit or self._cache_limit
        while len(cache) > limit:
            oldest_key = next(iter(cache))
            del cache[oldest_key]

    def prune(self, keep_slots: int = 8192) -> int:
        """Prune old states and blocks. Returns number of items pruned."""
        pruned = 0
        try:
            max_slot = 0
            for key, _ in self._db.iterator(prefix=PREFIX_STATE_SLOT, reverse=True):
                max_slot = struct.unpack(">Q", key[len(PREFIX_STATE_SLOT):])[0]
                break

            if max_slot > 0:
                cutoff_slot = max_slot - keep_slots
                batch = self._db.write_batch()

                for key, value in self._db.iterator(prefix=PREFIX_STATE_SLOT):
                    slot = struct.unpack(">Q", key[len(PREFIX_STATE_SLOT):])[0]
                    if 0 < slot < cutoff_slot:
                        root = value
                        batch.delete(key)
                        batch.delete(PREFIX_STATE + root)
                        pruned += 1

                for key, value in self._db.iterator(prefix=PREFIX_BLOCK_SLOT):
                    slot = struct.unpack(">Q", key[len(PREFIX_BLOCK_SLOT):])[0]
                    if slot < cutoff_slot:
                        root = value
                        batch.delete(key)
                        batch.delete(PREFIX_BLOCK + root)
                        pruned += 1

                batch.write()

                if pruned > 0:
                    logger.info(f"Pruned {pruned} old states/blocks (cutoff slot: {cutoff_slot})")
        except Exception as e:
            logger.warning(f"Failed to prune database: {e}")

        return pruned

    def get_stats(self) -> dict:
        """Get database statistics."""
        stats = {
            "state_cache_size": len(self._state_cache),
            "block_cache_size": len(self._block_cache),
            "states_count": 0,
            "blocks_count": 0,
            "state_slot_range": (0, 0),
            "block_slot_range": (0, 0),
        }
        try:
            for _ in self._db.iterator(prefix=PREFIX_STATE):
                stats["states_count"] += 1

            for _ in self._db.iterator(prefix=PREFIX_BLOCK):
                stats["blocks_count"] += 1

            min_state_slot, max_state_slot = None, None
            for key, _ in self._db.iterator(prefix=PREFIX_STATE_SLOT):
                slot = struct.unpack(">Q", key[len(PREFIX_STATE_SLOT):])[0]
                if min_state_slot is None or slot < min_state_slot:
                    min_state_slot = slot
                if max_state_slot is None or slot > max_state_slot:
                    max_state_slot = slot
            if min_state_slot is not None:
                stats["state_slot_range"] = (min_state_slot, max_state_slot)

            min_block_slot, max_block_slot = None, None
            for key, _ in self._db.iterator(prefix=PREFIX_BLOCK_SLOT):
                slot = struct.unpack(">Q", key[len(PREFIX_BLOCK_SLOT):])[0]
                if min_block_slot is None or slot < min_block_slot:
                    min_block_slot = slot
                if max_block_slot is None or slot > max_block_slot:
                    max_block_slot = slot
            if min_block_slot is not None:
                stats["block_slot_range"] = (min_block_slot, max_block_slot)
        except Exception as e:
            logger.warning(f"Failed to get DB stats: {e}")
        return stats

    def close(self) -> None:
        """Close the database connection."""
        if self._db:
            self._db.close()
            self._db = None
            logger.info("LevelDB store closed")
