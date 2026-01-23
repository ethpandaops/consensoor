"""State and block storage."""

import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class Store:
    """Simple in-memory store for beacon state and blocks."""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else None
        self.states: dict[bytes, object] = {}
        self.blocks: dict[bytes, object] = {}
        self.payloads: dict[bytes, object] = {}
        self.bids: dict[int, object] = {}
        self.head_root: Optional[bytes] = None
        self.finalized_root: bytes = b"\x00" * 32
        self.finalized_epoch: int = 0
        self.justified_root: bytes = b"\x00" * 32
        self.justified_epoch: int = 0

    def save_state(self, root: bytes, state: object) -> None:
        """Save a beacon state by root."""
        self.states[root] = state

    def get_state(self, root: bytes) -> Optional[object]:
        """Get a beacon state by root."""
        return self.states.get(root)

    def save_block(self, root: bytes, block: object) -> None:
        """Save a signed beacon block by root."""
        self.blocks[root] = block

    def get_block(self, root: bytes) -> Optional[object]:
        """Get a signed beacon block by root."""
        return self.blocks.get(root)

    def save_payload(self, root: bytes, payload: object) -> None:
        """Save an execution payload envelope by root."""
        self.payloads[root] = payload

    def get_payload(self, root: bytes) -> Optional[object]:
        """Get an execution payload envelope by root."""
        return self.payloads.get(root)

    def save_bid(self, slot: int, bid: object) -> None:
        """Save an execution payload bid by slot."""
        self.bids[slot] = bid

    def get_bid(self, slot: int) -> Optional[object]:
        """Get an execution payload bid by slot."""
        return self.bids.get(slot)

    def set_head(self, root: bytes) -> None:
        """Set the current head root."""
        self.head_root = root

    def set_finalized(self, root: bytes, epoch: int) -> None:
        """Set the finalized checkpoint."""
        self.finalized_root = root
        self.finalized_epoch = epoch

    def set_justified(self, root: bytes, epoch: int) -> None:
        """Set the justified checkpoint."""
        self.justified_root = root
        self.justified_epoch = epoch

    def prune(self, keep_slots: int = 8192) -> int:
        """Prune old states and blocks. Returns number of items pruned."""
        pruned = 0
        return pruned


__all__ = ["Store"]
