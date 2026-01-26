"""Ethereum consensus gossip protocol implementation.

Handles subscription to beacon chain topics and message routing.
"""

import base64
import logging
from typing import Callable, Awaitable, Optional

import rlp

from .host import P2PHost, P2PConfig
from .encoding import (
    get_topic_name,
    get_blob_sidecar_topic,
    encode_message,
    decode_message,
    compute_fork_digest,
    BEACON_BLOCK_TOPIC,
    BEACON_AGGREGATE_AND_PROOF_TOPIC,
    VOLUNTARY_EXIT_TOPIC,
    PROPOSER_SLASHING_TOPIC,
    ATTESTER_SLASHING_TOPIC,
    BLS_TO_EXECUTION_CHANGE_TOPIC,
    SYNC_COMMITTEE_CONTRIBUTION_AND_PROOF_TOPIC,
    BLOB_SIDECAR_TOPIC_PREFIX,
)

logger = logging.getLogger(__name__)


def extract_fork_digest_from_enr(enr_str: str) -> bytes | None:
    """Extract fork_digest from an ENR string.

    Returns the fork_digest (4 bytes) from the eth2 field, or None if not present.
    """
    try:
        if enr_str.startswith("enr:"):
            enr_str = enr_str[4:]

        # Add proper base64 padding
        padding_needed = (4 - len(enr_str) % 4) % 4
        enr_bytes = base64.urlsafe_b64decode(enr_str + "=" * padding_needed)

        decoded = rlp.decode(enr_bytes)

        # RLP structure: [signature, seq, key1, val1, key2, val2, ...]
        i = 2
        while i < len(decoded) - 1:
            key = decoded[i]
            value = decoded[i + 1]
            if key == b"eth2" and len(value) >= 4:
                return value[:4]
            i += 2
    except Exception as e:
        logger.debug(f"Failed to extract fork_digest from ENR: {e}")

    return None

MessageHandler = Callable[[bytes, str], Awaitable[None]]


class BeaconGossip:
    """Manages gossip subscriptions for beacon chain topics."""

    def __init__(
        self,
        fork_version: bytes,
        genesis_validators_root: bytes,
        listen_port: int = 9000,
        static_peers: Optional[list[str]] = None,
        next_fork_version: Optional[bytes] = None,
        next_fork_epoch: int = 2**64 - 1,
        attnets: Optional[bytes] = None,
        syncnets: Optional[bytes] = None,
        fork_digest_override: Optional[bytes] = None,
        supernode: bool = False,
        all_fork_digests: Optional[list[bytes]] = None,
    ):
        # Use override if provided, otherwise compute from fork_version and genesis_validators_root
        computed_digest = compute_fork_digest(fork_version, genesis_validators_root)
        if fork_digest_override:
            self.fork_digest = fork_digest_override
            if fork_digest_override != computed_digest:
                logger.info(
                    f"Using fork_digest override: {fork_digest_override.hex()} "
                    f"(computed would be: {computed_digest.hex()})"
                )
        else:
            self.fork_digest = computed_digest

        # Subscribe to all provided fork digests (for multi-fork devnets)
        self._all_fork_digests = all_fork_digests or [self.fork_digest]
        if self.fork_digest not in self._all_fork_digests:
            self._all_fork_digests.append(self.fork_digest)

        self._handlers: dict[str, MessageHandler] = {}

        config = P2PConfig(
            listen_port=listen_port,
            static_peers=static_peers or [],
            fork_digest=self.fork_digest,
            next_fork_version=next_fork_version or fork_version,
            next_fork_epoch=next_fork_epoch,
            attnets=attnets or b"\xff\xff\xff\xff\xff\xff\xff\xff",
            syncnets=syncnets or b"\x0f",
            supernode=supernode,
        )
        self._host = P2PHost(config)
        logger.info(f"P2P custody_group_count: {config.custody_group_count} (supernode={supernode})")

    async def start(self) -> None:
        """Start the gossip network."""
        await self._host.start()
        logger.info(f"Beacon gossip started with fork_digest={self.fork_digest.hex()}")

    async def stop(self) -> None:
        """Stop the gossip network."""
        await self._host.stop()

    def subscribe_blocks(self, handler: MessageHandler) -> None:
        """Subscribe to beacon block messages."""
        self._handlers[BEACON_BLOCK_TOPIC] = handler

    def subscribe_aggregates(self, handler: MessageHandler) -> None:
        """Subscribe to aggregate attestation messages."""
        self._handlers[BEACON_AGGREGATE_AND_PROOF_TOPIC] = handler

    def subscribe_voluntary_exits(self, handler: MessageHandler) -> None:
        """Subscribe to voluntary exit messages."""
        self._handlers[VOLUNTARY_EXIT_TOPIC] = handler

    def subscribe_proposer_slashings(self, handler: MessageHandler) -> None:
        """Subscribe to proposer slashing messages."""
        self._handlers[PROPOSER_SLASHING_TOPIC] = handler

    def subscribe_attester_slashings(self, handler: MessageHandler) -> None:
        """Subscribe to attester slashing messages."""
        self._handlers[ATTESTER_SLASHING_TOPIC] = handler

    def subscribe_sync_committee_contributions(self, handler: MessageHandler) -> None:
        """Subscribe to sync committee contribution and proof messages."""
        self._handlers[SYNC_COMMITTEE_CONTRIBUTION_AND_PROOF_TOPIC] = handler

    def subscribe_blob_sidecars(self, handler: MessageHandler) -> None:
        """Subscribe to blob sidecar messages on all subnets."""
        self._blob_sidecar_handler = handler

    async def activate_subscriptions(self) -> None:
        """Activate all registered subscriptions on the P2P host.

        Subscribes to topics for ALL fork digests to handle fork transitions.
        """
        subscribed_topics = []
        for base_topic, handler in self._handlers.items():
            wrapped_handler = self._wrap_handler(handler)
            for fork_digest in self._all_fork_digests:
                topic = get_topic_name(base_topic, fork_digest)
                await self._host.subscribe(topic, wrapped_handler)
            subscribed_topics.append(base_topic)

        # Subscribe to blob sidecar topics (6 subnets: 0-5)
        blob_subnet_count = 0
        if hasattr(self, '_blob_sidecar_handler') and self._blob_sidecar_handler:
            wrapped_handler = self._wrap_handler(self._blob_sidecar_handler)
            for fork_digest in self._all_fork_digests:
                for subnet_id in range(6):
                    topic = get_blob_sidecar_topic(subnet_id, fork_digest)
                    await self._host.subscribe(topic, wrapped_handler)
                    blob_subnet_count += 1
            subscribed_topics.append(f"blob_sidecar (6 subnets)")

        fork_count = len(self._all_fork_digests)
        logger.info(f"Subscribed to gossip topics: {subscribed_topics} ({fork_count} fork digests)")

    def _wrap_handler(self, handler: MessageHandler) -> MessageHandler:
        """Wrap a handler to decode incoming messages."""
        async def wrapped(data: bytes, from_peer: str) -> None:
            try:
                decoded = decode_message(data)
                await handler(decoded, from_peer)
            except Exception as e:
                logger.error(f"Error decoding message: {e}")

        return wrapped

    def update_fork_digest(self, fork_digest: bytes) -> None:
        """Update the current fork digest for publishing messages.

        Call this when crossing fork boundaries to ensure messages are
        published to the correct topic.
        """
        if fork_digest != self.fork_digest:
            logger.info(f"Updating fork_digest for publishing: {self.fork_digest.hex()} -> {fork_digest.hex()}")
            self.fork_digest = fork_digest

    async def publish_block(self, block_ssz: bytes) -> None:
        """Publish a signed beacon block."""
        topic = get_topic_name(BEACON_BLOCK_TOPIC, self.fork_digest)
        encoded = encode_message(block_ssz)
        await self._host.publish(topic, encoded)
        logger.info(f"Published block: {len(block_ssz)} bytes (topic={topic})")

    async def publish_aggregate(self, aggregate_ssz: bytes) -> None:
        """Publish an aggregate attestation."""
        topic = get_topic_name(BEACON_AGGREGATE_AND_PROOF_TOPIC, self.fork_digest)
        encoded = encode_message(aggregate_ssz)
        await self._host.publish(topic, encoded)

    async def publish_sync_committee_contribution(self, contribution_ssz: bytes) -> None:
        """Publish a sync committee contribution and proof."""
        topic = get_topic_name(SYNC_COMMITTEE_CONTRIBUTION_AND_PROOF_TOPIC, self.fork_digest)
        encoded = encode_message(contribution_ssz)
        await self._host.publish(topic, encoded)
        logger.debug(f"Published sync committee contribution: {len(contribution_ssz)} bytes")

    def set_status_provider(self, provider: Callable[[], dict]) -> None:
        """Set the status provider callback for P2P status messages.

        The provider should return a dict with:
        - head_slot: int
        - head_root: bytes (32 bytes)
        - finalized_epoch: int
        - finalized_root: bytes (32 bytes)
        - earliest_available_slot: int (optional, defaults to 0)
        """
        self._host.set_status_provider(provider)

    @property
    def peer_id(self) -> Optional[str]:
        """Get the local peer ID."""
        return self._host.peer_id

    @property
    def peer_count(self) -> int:
        """Get the number of connected peers."""
        return self._host.peer_count
