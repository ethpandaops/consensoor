"""Network layer for consensoor."""

import asyncio
import logging
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Awaitable, Optional

logger = logging.getLogger(__name__)

MessageHandler = Callable[[bytes, tuple[str, int]], Awaitable[None]]


class MessageType(IntEnum):
    BEACON_BLOCK = 1
    ATTESTATION = 2
    AGGREGATE_AND_PROOF = 3
    EXECUTION_PAYLOAD_BID = 4
    EXECUTION_PAYLOAD = 5
    PAYLOAD_ATTESTATION = 6
    STATUS = 7
    GOODBYE = 8


@dataclass
class NetworkMessage:
    """A network message with type and payload."""

    msg_type: MessageType
    payload: bytes

    def encode(self) -> bytes:
        """Encode the message for transmission."""
        header = struct.pack("<BH", self.msg_type, len(self.payload))
        return header + self.payload

    @classmethod
    def decode(cls, data: bytes) -> Optional["NetworkMessage"]:
        """Decode a message from bytes."""
        if len(data) < 3:
            return None
        msg_type, length = struct.unpack("<BH", data[:3])
        if len(data) < 3 + length:
            return None
        try:
            return cls(msg_type=MessageType(msg_type), payload=data[3 : 3 + length])
        except ValueError:
            return None


@dataclass
class GossipConfig:
    """Configuration for gossip network."""

    listen_host: str = "0.0.0.0"
    listen_port: int = 9000
    peers: list[str] = field(default_factory=list)
    max_message_size: int = 65535


class UDPProtocol(asyncio.DatagramProtocol):
    """UDP protocol handler for gossip messages."""

    def __init__(self, gossip: "Gossip"):
        self.gossip = gossip
        self.transport: Optional[asyncio.DatagramTransport] = None

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:
        self.transport = transport

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        asyncio.create_task(self.gossip._handle_datagram(data, addr))

    def error_received(self, exc: Exception) -> None:
        logger.error(f"UDP error: {exc}")


class Gossip:
    """Simple UDP gossip layer for local network testing."""

    def __init__(self, config: GossipConfig):
        self.config = config
        self.handlers: dict[MessageType, MessageHandler] = {}
        self.transport: Optional[asyncio.DatagramTransport] = None
        self.protocol: Optional[UDPProtocol] = None
        self._running = False

    def subscribe(self, msg_type: MessageType, handler: MessageHandler) -> None:
        """Subscribe to a message type with a handler."""
        self.handlers[msg_type] = handler
        logger.debug(f"Subscribed to {msg_type.name}")

    def unsubscribe(self, msg_type: MessageType) -> None:
        """Unsubscribe from a message type."""
        self.handlers.pop(msg_type, None)

    async def start(self) -> None:
        """Start the gossip network (UDP only - TCP handled by py-libp2p)."""
        loop = asyncio.get_event_loop()
        self.transport, self.protocol = await loop.create_datagram_endpoint(
            lambda: UDPProtocol(self),
            local_addr=(self.config.listen_host, self.config.listen_port),
        )

        # NOTE: TCP server removed - py-libp2p handles TCP on port 9000
        # The previous TCP server was a stub that just closed connections

        self._running = True
        logger.info(
            f"Gossip (UDP) listening on {self.config.listen_host}:{self.config.listen_port}"
        )

    async def stop(self) -> None:
        """Stop the gossip network."""
        self._running = False
        if self.transport:
            self.transport.close()
            self.transport = None

    async def broadcast(self, msg_type: MessageType, payload: bytes) -> None:
        """Broadcast a message to all peers."""
        if not self.transport:
            logger.warning("Cannot broadcast: transport not initialized")
            return

        message = NetworkMessage(msg_type=msg_type, payload=payload)
        encoded = message.encode()

        for peer in self.config.peers:
            try:
                host, port = self._parse_peer(peer)
                self.transport.sendto(encoded, (host, port))
            except Exception as e:
                logger.error(f"Failed to send to {peer}: {e}")

    async def send_to(
        self, msg_type: MessageType, payload: bytes, addr: tuple[str, int]
    ) -> None:
        """Send a message to a specific peer."""
        if not self.transport:
            return

        message = NetworkMessage(msg_type=msg_type, payload=payload)
        self.transport.sendto(message.encode(), addr)

    async def _handle_datagram(self, data: bytes, addr: tuple[str, int]) -> None:
        """Handle an incoming datagram."""
        message = NetworkMessage.decode(data)
        if message is None:
            logger.debug(f"Invalid message from {addr}")
            return

        handler = self.handlers.get(message.msg_type)
        if handler:
            try:
                await handler(message.payload, addr)
            except Exception as e:
                logger.error(f"Handler error for {message.msg_type.name}: {e}")
        else:
            logger.debug(f"No handler for {message.msg_type.name}")

    def _parse_peer(self, peer: str) -> tuple[str, int]:
        """Parse a peer address string into host and port."""
        if ":" in peer:
            host, port_str = peer.rsplit(":", 1)
            return host, int(port_str)
        return peer, self.config.listen_port

    @property
    def peer_count(self) -> int:
        """Return the number of configured peers."""
        return len(self.config.peers)


__all__ = [
    "Gossip",
    "GossipConfig",
    "NetworkMessage",
    "MessageType",
    "MessageHandler",
]
