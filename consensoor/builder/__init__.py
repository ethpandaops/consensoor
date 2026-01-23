"""Builder functionality for ePBS."""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BuilderKey:
    """A builder's key pair."""

    pubkey: bytes
    privkey: int
    execution_address: bytes


class BuilderClient:
    """Builder client for producing execution payload bids."""

    def __init__(self, keys: list[BuilderKey]):
        self.keys = {k.pubkey: k for k in keys}
        self.pubkeys = set(k.pubkey for k in keys)

    def has_key(self, pubkey: bytes) -> bool:
        """Check if we have a key for this pubkey."""
        return pubkey in self.pubkeys

    async def produce_bid(
        self,
        state,
        slot: int,
        parent_hash: bytes,
        parent_block_root: bytes,
    ) -> object:
        """Produce an execution payload bid for the given slot."""
        pass

    async def produce_payload_envelope(
        self,
        state,
        bid,
        beacon_block_root: bytes,
    ) -> object:
        """Produce an execution payload envelope for the given bid."""
        pass


from .block_builder import BlockBuilder

__all__ = ["BuilderKey", "BuilderClient", "BlockBuilder"]
