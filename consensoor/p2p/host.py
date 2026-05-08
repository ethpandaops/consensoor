"""libp2p host wrapped around the native Rust stack (consensoor_p2p).

This replaces the legacy py-libp2p host. Public API (start/stop/subscribe/
publish/peer_id/etc.) stays the same so BeaconGossip and node.py don't
have to change. The Rust binding handles:

- TCP + Noise + Yamux transport
- Gossipsub with the Eth2 SHA-256 message-id rule
- Raw-snappy compression on publish, decompression on receive (so callers
  pass + receive raw SSZ bytes)
- Status v2 / Ping v1 / Goodbye v1 / MetaData v3 RPC handshakes
"""

from __future__ import annotations

import asyncio
import logging
import socket
import threading
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from .. import metrics

logger = logging.getLogger(__name__)

MessageHandler = Callable[[bytes, str], Awaitable[None]]


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "0.0.0.0"


@dataclass
class P2PConfig:
    """Configuration for P2P networking. Public API is unchanged from the
    py-libp2p era so callers (BeaconGossip) compile without touching."""

    listen_host: str = "0.0.0.0"
    listen_port: int = 9000
    static_peers: list[str] = field(default_factory=list)
    fork_digest: bytes = field(default_factory=lambda: b"\x00\x00\x00\x00")
    next_fork_version: bytes = field(default_factory=lambda: b"\x00\x00\x00\x00")
    next_fork_epoch: int = 2**64 - 1
    attnets: bytes = field(default_factory=lambda: b"\xff\xff\xff\xff\xff\xff\xff\xff")
    syncnets: bytes = field(default_factory=lambda: b"\x0f")
    supernode: bool = False
    gossip_degree: int = 3
    gossip_degree_low: int = 1
    gossip_degree_high: int = 6

    @property
    def custody_group_count(self) -> int:
        return 128 if self.supernode else 4


class P2PHost:
    """Host wrapping consensoor_p2p.Network.

    Provides the same coroutine-friendly API consensoor's BeaconGossip and
    node.py expect. Internally:
      - Boots a Rust libp2p host via consensoor_p2p.Network.start
      - Spawns a background thread that polls Network.next_message and
        dispatches each (decompressed) gossipsub payload to the matching
        handler on the asyncio event loop.
      - Same for Status / Ping / Metadata events.
    """

    def __init__(self, config: P2PConfig):
        self.config = config
        self._network = None  # consensoor_p2p.Network
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._handlers: dict[str, MessageHandler] = {}
        self._stop = threading.Event()
        self._gossip_thread: Optional[threading.Thread] = None
        self._status_thread: Optional[threading.Thread] = None
        self._ping_thread: Optional[threading.Thread] = None
        self._meta_thread: Optional[threading.Thread] = None

        self._block_provider: Optional[Callable[[int], Optional[tuple[bytes, bytes]]]] = None
        self._status_provider: Optional[Callable[[], dict]] = None

        self._peer_id: Optional[str] = None
        self._listen_addrs: list[str] = []
        self._our_metadata_seq = 0

    # ------------------------------------------------------------------ providers

    def set_block_provider(
        self, provider: Callable[[int], Optional[tuple[bytes, bytes]]]
    ) -> None:
        self._block_provider = provider

    def set_status_provider(self, provider: Callable[[], dict]) -> None:
        self._status_provider = provider

    def update_fork_digest(self, fork_digest: bytes) -> None:
        self.config.fork_digest = fork_digest

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        try:
            import consensoor_p2p as cp
        except ImportError as e:
            raise RuntimeError(
                "consensoor_p2p Rust wheel not installed. Build with "
                "`cd consensoor-p2p-rs && python3 -m maturin develop --release`"
            ) from e

        self._loop = asyncio.get_running_loop()

        net_cfg = cp.NetworkConfig(
            listen_addr=f"/ip4/{self.config.listen_host}/tcp/{self.config.listen_port}",
            external_addr=None,
            bootnodes=list(self.config.static_peers),
            fork_digest=list(self.config.fork_digest),
            seed_phrase=None,
            max_peers=64,
            agent_version="consensoor/0.1.0",
        )
        self._network = cp.Network.start(net_cfg)
        self._peer_id = self._network.peer_id()
        # Give the listener a moment to bind, then snapshot addresses.
        await asyncio.sleep(0.1)
        self._listen_addrs = list(self._network.listen_addresses())

        # Dial each configured static peer (multiaddr including /p2p/<peer_id>).
        for addr in self.config.static_peers:
            try:
                self._network.dial(addr)
            except Exception as e:
                logger.warning(f"dial failed for {addr}: {e}")

        # Background dispatchers
        self._gossip_thread = threading.Thread(
            target=self._gossip_loop, name="p2p-gossip-dispatch", daemon=True
        )
        self._gossip_thread.start()

        self._status_thread = threading.Thread(
            target=self._status_loop, name="p2p-status-dispatch", daemon=True
        )
        self._status_thread.start()

        self._ping_thread = threading.Thread(
            target=self._ping_loop, name="p2p-ping-dispatch", daemon=True
        )
        self._ping_thread.start()

        self._meta_thread = threading.Thread(
            target=self._meta_loop, name="p2p-meta-dispatch", daemon=True
        )
        self._meta_thread.start()

        logger.info(
            f"P2P host started: peer_id={self._peer_id}, "
            f"listen={self._listen_addrs}, fork_digest={self.config.fork_digest.hex()}"
        )

    async def stop(self) -> None:
        self._stop.set()
        if self._network is not None:
            try:
                self._network.shutdown()
            except Exception:
                pass

    # ------------------------------------------------------------------ pubsub

    async def subscribe(self, topic: str, handler: MessageHandler) -> None:
        self._handlers[topic] = handler
        if self._network is not None:
            try:
                self._network.subscribe(topic)
            except Exception as e:
                logger.warning(f"subscribe({topic}) failed: {e}")

    async def publish(self, topic: str, data: bytes) -> None:
        """Publish raw uncompressed SSZ bytes; the Rust binding handles
        snappy compression and Eth2 message-id calculation."""
        if self._network is None:
            return
        try:
            self._network.publish(topic, list(data))
        except Exception as e:
            logger.warning(f"publish({topic}) failed: {e}")

    # ------------------------------------------------------------------ event pumps

    def _gossip_loop(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self._network.next_message(1000)
            except Exception as e:
                logger.warning(f"next_message error: {e}")
                continue
            if msg is None:
                continue
            handler = self._handlers.get(msg.topic)
            if handler is None:
                continue
            data = bytes(msg.data)  # already snappy-decompressed by rust side
            from_peer = msg.from_peer
            if self._loop is None:
                continue
            try:
                asyncio.run_coroutine_threadsafe(handler(data, from_peer), self._loop)
            except Exception as e:
                logger.warning(f"dispatch failed for topic {msg.topic}: {e}")

    def _status_loop(self) -> None:
        import consensoor_p2p as cp
        while not self._stop.is_set():
            try:
                ev = self._network.next_status(1000)
            except Exception as e:
                logger.warning(f"next_status error: {e}")
                continue
            if ev is None:
                continue
            if ev.kind.startswith("request:") and self._status_provider is not None:
                req_id = int(ev.kind.split(":")[1])
                try:
                    s = self._status_provider() or {}
                    fork_digest = bytes(s.get("fork_digest", self.config.fork_digest))
                    msg = cp.StatusMessage(
                        fork_digest=list(fork_digest),
                        finalized_root=list(bytes(s.get("finalized_root", b"\x00" * 32))),
                        finalized_epoch=int(s.get("finalized_epoch", 0)),
                        head_root=list(bytes(s.get("head_root", b"\x00" * 32))),
                        head_slot=int(s.get("head_slot", 0)),
                        earliest_available_slot=int(s.get("earliest_available_slot", 0)),
                    )
                    self._network.answer_status(req_id, msg)
                except Exception as e:
                    logger.warning(f"answer_status failed: {e}")
            elif ev.kind == "response":
                logger.debug(f"Status response from {ev.peer[:24]}...: {ev.message}")
            elif ev.kind == "failure":
                logger.debug(f"Status RPC failure with {ev.peer[:24]}...: {ev.error}")

    def _ping_loop(self) -> None:
        import consensoor_p2p as cp
        while not self._stop.is_set():
            try:
                ev = self._network.next_ping(1000)
            except Exception as e:
                logger.warning(f"next_ping error: {e}")
                continue
            if ev is None:
                continue
            if ev.kind.startswith("request:"):
                req_id = int(ev.kind.split(":")[1])
                try:
                    self._network.answer_ping(req_id, cp.PingMessage(self._our_metadata_seq))
                except Exception as e:
                    logger.warning(f"answer_ping failed: {e}")

    def _meta_loop(self) -> None:
        import consensoor_p2p as cp
        while not self._stop.is_set():
            try:
                ev = self._network.next_metadata(1000)
            except Exception as e:
                logger.warning(f"next_metadata error: {e}")
                continue
            if ev is None:
                continue
            if ev.kind.startswith("request:"):
                req_id = int(ev.kind.split(":")[1])
                try:
                    md = cp.MetaDataMessage(
                        seq_number=self._our_metadata_seq,
                        attnets=list(self.config.attnets),
                        syncnets=int(self.config.syncnets[0]) if self.config.syncnets else 0,
                        custody_group_count=self.config.custody_group_count,
                    )
                    self._network.answer_metadata(req_id, md)
                except Exception as e:
                    logger.warning(f"answer_metadata failed: {e}")

    # ------------------------------------------------------------------ introspection

    @property
    def peer_id(self) -> Optional[str]:
        return self._peer_id

    @property
    def peer_count(self) -> int:
        # The rust binding does not yet expose a connected-peer count.
        # Return 0 conservatively; gossipsub mesh-empty warnings will guide.
        return 0

    def connected_peers(self) -> list[dict]:
        return []

    @property
    def enr(self) -> Optional[str]:
        # discv5 / ENR not yet implemented in the rust binding.
        return None

    @property
    def multiaddr(self) -> Optional[str]:
        if not self._listen_addrs:
            return None
        addr = self._listen_addrs[0]
        return f"{addr}/p2p/{self._peer_id}" if self._peer_id else addr

    def get_mesh_info(self, topic: str) -> str:
        return f"topic={topic} peers=unknown (rust binding stub)"
