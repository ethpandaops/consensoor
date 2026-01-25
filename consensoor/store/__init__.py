"""State and block storage with SQLite persistence."""

import logging
import sqlite3
from typing import Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class Store:
    """SQLite-backed store for beacon state and blocks.

    Stores SSZ-encoded states and blocks in SQLite for persistence.
    Keeps recent items in memory cache for fast access.
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else Path(".")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache for hot data
        self._state_cache: dict[bytes, object] = {}
        self._block_cache: dict[bytes, object] = {}
        self._payload_cache: dict[bytes, object] = {}
        self._bid_cache: dict[int, object] = {}
        self._cache_limit = 128  # Keep last N items in memory

        # Metadata (always in memory, persisted to DB)
        self.head_root: Optional[bytes] = None
        self.finalized_root: bytes = b"\x00" * 32
        self.finalized_epoch: int = 0
        self.justified_root: bytes = b"\x00" * 32
        self.justified_epoch: int = 0

        # Initialize SQLite
        self._db_path = self.data_dir / "beacon.db"
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()
        self._load_metadata()

    def _init_db(self) -> None:
        """Initialize SQLite database and tables."""
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
        self._conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes, still safe

        # Create tables
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS states (
                root BLOB PRIMARY KEY,
                slot INTEGER,
                fork TEXT,
                data BLOB
            );
            CREATE TABLE IF NOT EXISTS blocks (
                root BLOB PRIMARY KEY,
                slot INTEGER,
                parent_root BLOB,
                fork TEXT,
                data BLOB
            );
            CREATE TABLE IF NOT EXISTS payloads (
                root BLOB PRIMARY KEY,
                data BLOB
            );
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value BLOB
            );
            CREATE INDEX IF NOT EXISTS idx_states_slot ON states(slot);
            CREATE INDEX IF NOT EXISTS idx_blocks_slot ON blocks(slot);
            CREATE INDEX IF NOT EXISTS idx_blocks_parent ON blocks(parent_root);
        """)
        self._conn.commit()
        logger.info(f"SQLite store initialized at {self._db_path}")

    def _load_metadata(self) -> None:
        """Load metadata from database."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT key, value FROM metadata")
        for key, value in cursor.fetchall():
            if key == "head_root" and value:
                self.head_root = value
            elif key == "finalized_root" and value:
                self.finalized_root = value
            elif key == "finalized_epoch" and value:
                self.finalized_epoch = int.from_bytes(value, "little")
            elif key == "justified_root" and value:
                self.justified_root = value
            elif key == "justified_epoch" and value:
                self.justified_epoch = int.from_bytes(value, "little")

        if self.head_root:
            logger.info(f"Loaded head_root from DB: {self.head_root.hex()[:16]}")

    def _save_metadata(self, key: str, value: bytes) -> None:
        """Save a metadata value to database."""
        self._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, value)
        )
        self._conn.commit()

    def _detect_fork(self, obj: Any) -> str:
        """Detect the fork type of a state or block."""
        type_name = type(obj).__name__

        # For Fulu state types
        if "Fulu" in type_name:
            return "fulu"
        if "Gloas" in type_name:
            return "gloas"

        # For blocks, check slot against fork epochs since Electra/Fulu share block types
        if "Electra" in type_name:
            msg = obj.message if hasattr(obj, "message") else obj
            if hasattr(msg, "slot"):
                from ..spec.network_config import get_config
                from ..spec.constants import SLOTS_PER_EPOCH
                try:
                    config = get_config()
                    slot = int(msg.slot)
                    epoch = slot // SLOTS_PER_EPOCH()
                    if hasattr(config, 'fulu_fork_epoch') and epoch >= config.fulu_fork_epoch:
                        return "fulu"
                except Exception:
                    pass
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
        from ..spec.types import BeaconState  # Fulu/latest
        from ..spec.types.electra import ElectraBeaconState
        from ..spec.types.deneb import DenebBeaconState
        from ..spec.types.capella import CapellaBeaconState
        from ..spec.types.bellatrix import BellatrixBeaconState
        from ..spec.types.altair import AltairBeaconState
        from ..spec.types.phase0 import Phase0BeaconState

        return {
            "fulu": BeaconState,
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
        from ..spec.types.electra import ElectraSignedBeaconBlock
        from ..spec.types.deneb import DenebSignedBeaconBlock
        from ..spec.types.capella import CapellaSignedBeaconBlock
        from ..spec.types.bellatrix import BellatrixSignedBeaconBlock
        from ..spec.types.altair import AltairSignedBeaconBlock
        from ..spec.types.phase0 import Phase0SignedBeaconBlock

        return {
            "fulu": FuluSignedBeaconBlock,
            "electra": ElectraSignedBeaconBlock,
            "deneb": DenebSignedBeaconBlock,
            "capella": CapellaSignedBeaconBlock,
            "bellatrix": BellatrixSignedBeaconBlock,
            "altair": AltairSignedBeaconBlock,
            "phase0": Phase0SignedBeaconBlock,
        }

    def save_state(self, root: bytes, state: object) -> None:
        """Save a beacon state by root."""
        # Cache in memory
        self._state_cache[root] = state
        self._trim_cache(self._state_cache)

        # Persist to SQLite
        try:
            fork = self._detect_fork(state)
            slot = int(state.slot) if hasattr(state, "slot") else 0
            data = state.encode_bytes()

            self._conn.execute(
                "INSERT OR REPLACE INTO states (root, slot, fork, data) VALUES (?, ?, ?, ?)",
                (root, slot, fork, data)
            )
            self._conn.commit()
            logger.debug(f"Saved state: slot={slot}, fork={fork}, root={root.hex()[:16]}")
        except Exception as e:
            logger.warning(f"Failed to persist state to SQLite: {e}")

    def get_state(self, root: bytes) -> Optional[object]:
        """Get a beacon state by root."""
        # Check cache first
        if root in self._state_cache:
            return self._state_cache[root]

        # Load from SQLite
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT fork, data FROM states WHERE root = ?", (root,))
            row = cursor.fetchone()
            if row:
                fork, data = row
                state_types = self._get_state_types()
                state_type = state_types.get(fork)
                if state_type:
                    state = state_type.decode_bytes(data)
                    self._state_cache[root] = state  # Cache it
                    return state
        except Exception as e:
            logger.warning(f"Failed to load state from SQLite: {e}")

        return None

    def get_state_by_slot(self, slot: int) -> Optional[object]:
        """Get a beacon state by slot (returns first match)."""
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT root, fork, data FROM states WHERE slot = ? LIMIT 1", (slot,))
            row = cursor.fetchone()
            if row:
                root, fork, data = row
                state_types = self._get_state_types()
                state_type = state_types.get(fork)
                if state_type:
                    state = state_type.decode_bytes(data)
                    self._state_cache[root] = state
                    return state
        except Exception as e:
            logger.warning(f"Failed to load state by slot from SQLite: {e}")
        return None

    def save_block(self, root: bytes, block: object) -> None:
        """Save a signed beacon block by root."""
        # Cache in memory
        self._block_cache[root] = block
        self._trim_cache(self._block_cache)

        # Persist to SQLite
        try:
            fork = self._detect_fork(block)
            msg = block.message if hasattr(block, "message") else block
            slot = int(msg.slot) if hasattr(msg, "slot") else 0
            parent_root = bytes(msg.parent_root) if hasattr(msg, "parent_root") else b"\x00" * 32
            data = block.encode_bytes()

            self._conn.execute(
                "INSERT OR REPLACE INTO blocks (root, slot, parent_root, fork, data) VALUES (?, ?, ?, ?, ?)",
                (root, slot, parent_root, fork, data)
            )
            self._conn.commit()
            logger.debug(f"Saved block: slot={slot}, fork={fork}, root={root.hex()[:16]}")
        except Exception as e:
            logger.warning(f"Failed to persist block to SQLite: {e}")

    def get_block(self, root: bytes) -> Optional[object]:
        """Get a signed beacon block by root."""
        # Check cache first
        if root in self._block_cache:
            return self._block_cache[root]

        # Load from SQLite
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT fork, data FROM blocks WHERE root = ?", (root,))
            row = cursor.fetchone()
            if row:
                fork, data = row
                block_types = self._get_block_types()
                block_type = block_types.get(fork)
                if block_type:
                    block = block_type.decode_bytes(data)
                    self._block_cache[root] = block
                    return block
        except Exception as e:
            logger.warning(f"Failed to load block from SQLite: {e}")

        return None

    def get_block_by_slot(self, slot: int) -> Optional[object]:
        """Get a signed beacon block by slot (returns first match)."""
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT root, fork, data FROM blocks WHERE slot = ? LIMIT 1", (slot,))
            row = cursor.fetchone()
            if row:
                root, fork, data = row
                block_types = self._get_block_types()
                block_type = block_types.get(fork)
                if block_type:
                    block = block_type.decode_bytes(data)
                    self._block_cache[root] = block
                    return block
        except Exception as e:
            logger.warning(f"Failed to load block by slot from SQLite: {e}")
        return None

    def get_blocks_by_parent(self, parent_root: bytes) -> list:
        """Get all blocks with a given parent root."""
        blocks = []
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT root, fork, data FROM blocks WHERE parent_root = ?", (parent_root,))
            block_types = self._get_block_types()
            for root, fork, data in cursor.fetchall():
                block_type = block_types.get(fork)
                if block_type:
                    block = block_type.decode_bytes(data)
                    self._block_cache[root] = block
                    blocks.append(block)
        except Exception as e:
            logger.warning(f"Failed to load blocks by parent from SQLite: {e}")
        return blocks

    def save_payload(self, root: bytes, payload: object) -> None:
        """Save an execution payload envelope by root."""
        self._payload_cache[root] = payload
        self._trim_cache(self._payload_cache)

        # Persist to SQLite
        try:
            data = payload.encode_bytes() if hasattr(payload, "encode_bytes") else b""
            self._conn.execute(
                "INSERT OR REPLACE INTO payloads (root, data) VALUES (?, ?)",
                (root, data)
            )
            self._conn.commit()
        except Exception as e:
            logger.debug(f"Failed to persist payload to SQLite: {e}")

    def get_payload(self, root: bytes) -> Optional[object]:
        """Get an execution payload envelope by root."""
        return self._payload_cache.get(root)

    def save_bid(self, slot: int, bid: object) -> None:
        """Save an execution payload bid by slot."""
        self._bid_cache[slot] = bid

    def get_bid(self, slot: int) -> Optional[object]:
        """Get an execution payload bid by slot."""
        return self._bid_cache.get(slot)

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
            # Remove oldest item (first key)
            oldest_key = next(iter(cache))
            del cache[oldest_key]

    def prune(self, keep_slots: int = 8192) -> int:
        """Prune old states and blocks. Returns number of items pruned."""
        pruned = 0
        try:
            # Get the latest slot
            cursor = self._conn.cursor()
            cursor.execute("SELECT MAX(slot) FROM states")
            result = cursor.fetchone()
            if result and result[0]:
                max_slot = result[0]
                cutoff_slot = max_slot - keep_slots

                # Prune old states (keep finalized states)
                cursor.execute("DELETE FROM states WHERE slot < ? AND slot > 0", (cutoff_slot,))
                pruned += cursor.rowcount

                # Prune old blocks
                cursor.execute("DELETE FROM blocks WHERE slot < ?", (cutoff_slot,))
                pruned += cursor.rowcount

                self._conn.commit()
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
        }
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM states")
            stats["states_count"] = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM blocks")
            stats["blocks_count"] = cursor.fetchone()[0]
            cursor.execute("SELECT MIN(slot), MAX(slot) FROM states")
            row = cursor.fetchone()
            stats["state_slot_range"] = (row[0], row[1]) if row[0] else (0, 0)
            cursor.execute("SELECT MIN(slot), MAX(slot) FROM blocks")
            row = cursor.fetchone()
            stats["block_slot_range"] = (row[0], row[1]) if row[0] else (0, 0)
        except Exception as e:
            logger.warning(f"Failed to get DB stats: {e}")
        return stats

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("SQLite store closed")


__all__ = ["Store"]
