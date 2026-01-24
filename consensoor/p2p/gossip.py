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
    encode_message,
    decode_message,
    compute_fork_digest,
    BEACON_BLOCK_TOPIC,
    BEACON_AGGREGATE_AND_PROOF_TOPIC,
    VOLUNTARY_EXIT_TOPIC,
    PROPOSER_SLASHING_TOPIC,
    ATTESTER_SLASHING_TOPIC,
    BLS_TO_EXECUTION_CHANGE_TOPIC,
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

    async def activate_subscriptions(self) -> None:
        """Activate all registered subscriptions on the P2P host."""
        for base_topic, handler in self._handlers.items():
            topic = get_topic_name(base_topic, self.fork_digest)
            await self._host.subscribe(topic, self._wrap_handler(handler))
            logger.info(f"Subscribed to {base_topic}")

    def _wrap_handler(self, handler: MessageHandler) -> MessageHandler:
        """Wrap a handler to decode incoming messages."""
        async def wrapped(data: bytes, from_peer: str) -> None:
            try:
                decoded = decode_message(data)
                await handler(decoded, from_peer)
            except Exception as e:
                logger.error(f"Error decoding message: {e}")

        return wrapped

    async def publish_block(self, block_ssz: bytes) -> None:
        """Publish a signed beacon block."""
        topic = get_topic_name(BEACON_BLOCK_TOPIC, self.fork_digest)
        encoded = encode_message(block_ssz)
        await self._host.publish(topic, encoded)
        logger.info(f"Published block: {len(block_ssz)} bytes")

    async def publish_aggregate(self, aggregate_ssz: bytes) -> None:
        """Publish an aggregate attestation."""
        topic = get_topic_name(BEACON_AGGREGATE_AND_PROOF_TOPIC, self.fork_digest)
        encoded = encode_message(aggregate_ssz)
        await self._host.publish(topic, encoded)

    @property
    def peer_id(self) -> Optional[str]:
        """Get the local peer ID."""
        return self._host.peer_id

    @property
    def peer_count(self) -> int:
        """Get the number of connected peers."""
        return self._host.peer_count
