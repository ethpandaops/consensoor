"""libp2p host management for Ethereum consensus P2P.

This module wraps py-libp2p to provide Ethereum-specific P2P functionality.
py-libp2p uses Trio, so we run it in a separate thread.
"""

import asyncio
import logging
import threading
import queue
from typing import Optional, Callable, Awaitable, Any, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from libp2p.pubsub.pubsub import Pubsub
    from libp2p.pubsub.gossipsub import GossipSub

from .. import metrics

logger = logging.getLogger(__name__)

MessageHandler = Callable[[bytes, str], Awaitable[None]]


def get_local_ip() -> str:
    """Get the local IP address (non-loopback)."""
    import socket
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
    """Configuration for P2P networking."""

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
    heartbeat_interval: float = 0.7
    advertised_ip: str = ""

    @property
    def custody_group_count(self) -> int:
        """Calculate custody group count based on supernode flag.

        Per Fulu/PeerDAS spec:
        - Supernodes custody all 128 groups (NUMBER_OF_CUSTODY_GROUPS)
        - Regular nodes custody CUSTODY_REQUIREMENT groups (4)
        """
        from ..spec.constants import NUMBER_OF_CUSTODY_GROUPS, CUSTODY_REQUIREMENT
        return NUMBER_OF_CUSTODY_GROUPS if self.supernode else CUSTODY_REQUIREMENT


class P2PHost:
    """Ethereum consensus P2P host using libp2p.

    Provides gossipsub-based pub/sub for beacon chain messages.
    Runs py-libp2p in a separate Trio thread since it requires Trio.
    """

    def __init__(self, config: P2PConfig):
        self.config = config
        self._host = None
        self._pubsub: Optional["Pubsub"] = None
        self._gossipsub: Optional["GossipSub"] = None
        self._subscriptions: dict[str, Any] = {}
        self._handlers: dict[str, MessageHandler] = {}
        self._running = False
        self._trio_thread: Optional[threading.Thread] = None
        self._message_queue: queue.Queue = queue.Queue()
        self._publish_queue: queue.Queue = queue.Queue()
        self._subscribe_queue: queue.Queue = queue.Queue()
        self._peer_id: Optional[str] = None
        self._peer_count: int = 0
        self._connected_peers: dict[str, dict] = {}  # peer_id -> peer_info
        self._asyncio_loop: Optional[asyncio.AbstractEventLoop] = None
        self._started_event = threading.Event()
        self._stop_event = threading.Event()
        self._enr: Optional[str] = None
        self._listen_ip: Optional[str] = None
        self._private_key_bytes: Optional[bytes] = None
        self._status_provider: Optional[Callable[[], dict]] = None

    def set_status_provider(self, provider: Callable[[], dict]) -> None:
        """Set the status provider callback.

        The provider should return a dict with:
        - head_slot: int
        - head_root: bytes (32 bytes)
        - finalized_epoch: int
        - finalized_root: bytes (32 bytes)
        - earliest_available_slot: int (optional, defaults to 0)
        """
        self._status_provider = provider

    def _get_chain_status(self) -> dict:
        """Get current chain status from provider or return defaults."""
        if self._status_provider:
            try:
                return self._status_provider()
            except Exception as e:
                logger.warning(f"Status provider error: {e}")
        return {
            "head_slot": 0,
            "head_root": b"\x00" * 32,
            "finalized_epoch": 0,
            "finalized_root": b"\x00" * 32,
            "earliest_available_slot": 0,
        }

    async def start(self) -> None:
        """Start the P2P host in a separate Trio thread."""
        try:
            import trio
        except ImportError as e:
            logger.error(f"Trio not available: {e}")
            logger.warning("Running without P2P networking")
            return

        self._asyncio_loop = asyncio.get_event_loop()
        self._running = True
        self._stop_event.clear()

        self._trio_thread = threading.Thread(
            target=self._run_trio_host,
            daemon=True,
            name="p2p-trio-thread",
        )
        self._trio_thread.start()

        if not self._started_event.wait(timeout=10.0):
            logger.error("P2P host failed to start within timeout")
            self._running = False
            return

        asyncio.create_task(self._process_incoming_messages())
        logger.info(f"P2P host started: peer_id={self._peer_id}")

    def _run_trio_host(self) -> None:
        """Run the Trio-based libp2p host in this thread."""
        import trio
        trio.run(self._trio_main)

    async def _trio_main(self) -> None:
        """Main Trio coroutine for the P2P host."""
        import trio

        try:
            from libp2p import new_host
            from libp2p.crypto.secp256k1 import create_new_key_pair
            from libp2p.custom_types import TProtocol
            from libp2p.pubsub.gossipsub import GossipSub
            from libp2p.pubsub.pubsub import Pubsub
            from libp2p.tools.async_service.trio_service import background_trio_service
            from multiaddr import Multiaddr
            import secrets

            self._private_key_bytes = secrets.token_bytes(32)
            key_pair = create_new_key_pair(self._private_key_bytes)

            # Use default security (Noise + TLS + SECIO + Plaintext)
            # py-libp2p should negotiate the best common protocol
            logger.info("Using DEFAULT security transports (Noise, TLS, SECIO, Plaintext)")

            self._host = new_host(
                key_pair=key_pair,
                muxer_preference="MPLEX",
                sec_opt=None,  # Use defaults
            )

            advertised_ip = self.config.advertised_ip or get_local_ip()
            listen_host = self.config.listen_host  # Keep 0.0.0.0 for listening on all interfaces
            self._listen_ip = advertised_ip
            logger.info(f"P2P using IP: listen={listen_host}, advertised={advertised_ip}")

            listen_addrs = [
                Multiaddr(f"/ip4/{listen_host}/tcp/{self.config.listen_port}")
            ]

            # Create GossipSub router with meshsub protocol (Lighthouse requires this)
            gossipsub_protocol = TProtocol("/meshsub/1.1.0")
            self._gossipsub = GossipSub(
                protocols=[gossipsub_protocol],
                degree=self.config.gossip_degree,
                degree_low=self.config.gossip_degree_low,
                degree_high=self.config.gossip_degree_high,
                time_to_live=300,
                gossip_window=3,
                gossip_history=5,
                heartbeat_initial_delay=1.0,
                heartbeat_interval=self.config.heartbeat_interval,
            )

            # Create Pubsub with strict_signing=False for now (Ethereum uses custom signing)
            self._pubsub = Pubsub(
                self._host,
                self._gossipsub,
                strict_signing=False,
            )

            async with self._host.run(listen_addrs=listen_addrs):
                self._peer_id = self._host.get_id().to_base58()
                self._enr = self._generate_enr(advertised_ip, self.config.listen_port)
                logger.info(f"Trio P2P host running: peer_id={self._peer_id}")
                logger.info(f"Listening on: {listen_addrs}")
                logger.info(f"ENR: {self._enr}")

                self._register_protocol_handlers()

                # Start GossipSub and Pubsub services
                async with background_trio_service(self._pubsub):
                    async with background_trio_service(self._gossipsub):
                        logger.info("GossipSub and Pubsub services started")
                        await self._pubsub.wait_until_ready()
                        logger.info("Pubsub ready - gossipsub protocol will be advertised")

                        self._started_event.set()

                        # Set up connection notifier
                        network = self._host.get_network()
                        network.register_notifee(self._create_connection_notifier())

                        for peer in self.config.static_peers:
                            try:
                                await self._connect_to_peer_trio(peer)
                            except Exception as e:
                                logger.warning(f"Failed to connect to peer {peer}: {e}")

                        # Start subscription message reader as a background task
                        async with trio.open_nursery() as nursery:
                            last_reconnect_attempt = 0
                            last_mesh_log = 0
                            reconnect_interval = 10  # seconds
                            mesh_log_interval = 30  # seconds
                            while not self._stop_event.is_set():
                                await self._process_publish_queue_trio()
                                await self._process_subscribe_queue_trio(nursery)

                                import time
                                now = time.time()

                                # Periodically retry connecting to static peers
                                if now - last_reconnect_attempt > reconnect_interval:
                                    last_reconnect_attempt = now
                                    if len(self._connected_peers) < len(self.config.static_peers):
                                        for peer in self.config.static_peers:
                                            try:
                                                await self._connect_to_peer_trio(peer)
                                            except Exception as e:
                                                logger.debug(f"Reconnect to {peer[:30]}... failed: {e}")

                                # Periodically log mesh state and ensure peers are in mesh
                                if now - last_mesh_log > mesh_log_interval:
                                    last_mesh_log = now
                                    await self._log_and_maintain_mesh()

                                await trio.sleep(0.1)

        except ImportError as e:
            logger.error(f"libp2p not available: {e}")
            self._started_event.set()
        except Exception as e:
            logger.error(f"Trio P2P host error: {e}")
            import traceback
            traceback.print_exc()
            self._started_event.set()

    def _create_connection_notifier(self):
        """Create a notifier to track connection state changes."""
        from libp2p.abc import INotifee
        host = self

        class ConnectionNotifier(INotifee):
            async def opened_stream(self, network, stream):
                protocol = getattr(stream, 'protocol_id', 'unknown')
                peer_id = stream.muxed_conn.peer_id.to_base58()[:16]
                logger.info(f"Stream opened: peer={peer_id}, protocol={protocol}")

            async def closed_stream(self, network, stream):
                peer_id = stream.muxed_conn.peer_id.to_base58()[:16]
                logger.info(f"Stream closed: peer={peer_id}")

            async def connected(self, network, conn):
                full_peer_id = conn.muxed_conn.peer_id.to_base58()
                peer_id = full_peer_id[:16]
                logger.info(f"Notifier: peer connected: {peer_id}")
                addrs = [str(a) for a in conn.muxed_conn.remote_addrs] if hasattr(conn.muxed_conn, 'remote_addrs') else []
                host._connected_peers[full_peer_id] = {
                    "peer_id": full_peer_id,
                    "addrs": addrs,
                    "direction": "inbound" if hasattr(conn, 'is_initiator') and not conn.is_initiator else "outbound"
                }
                host._peer_count = len(host._connected_peers)
                metrics.update_peers(host._peer_count)

            async def disconnected(self, network, conn):
                full_peer_id = conn.muxed_conn.peer_id.to_base58()
                peer_id = full_peer_id[:16]
                logger.info(f"Notifier: peer disconnected: {peer_id}")
                host._connected_peers.pop(full_peer_id, None)
                host._peer_count = len(host._connected_peers)
                metrics.update_peers(host._peer_count)

            async def listen(self, network, multiaddr):
                logger.info(f"Listening on: {multiaddr}")

            async def listen_close(self, network, multiaddr):
                logger.info(f"Stopped listening on: {multiaddr}")

        return ConnectionNotifier()

    def _register_protocol_handlers(self) -> None:
        """Register Ethereum consensus protocol handlers."""
        from libp2p.custom_types import TProtocol

        # Status v1 (pre-Fulu)
        status_v1_protocol = TProtocol("/eth2/beacon_chain/req/status/1/ssz_snappy")
        self._host.set_stream_handler(status_v1_protocol, self._handle_status_v1_request)

        # Status v2 (Fulu+) - includes earliest_available_slot
        status_v2_protocol = TProtocol("/eth2/beacon_chain/req/status/2/ssz_snappy")
        self._host.set_stream_handler(status_v2_protocol, self._handle_status_v2_request)

        # Metadata v2 (Altair+)
        metadata_v2_protocol = TProtocol("/eth2/beacon_chain/req/metadata/2/ssz_snappy")
        self._host.set_stream_handler(metadata_v2_protocol, self._handle_metadata_v2_request)

        # Metadata v3 (Fulu+) - includes custody_group_count
        metadata_v3_protocol = TProtocol("/eth2/beacon_chain/req/metadata/3/ssz_snappy")
        self._host.set_stream_handler(metadata_v3_protocol, self._handle_metadata_v3_request)

        ping_protocol = TProtocol("/eth2/beacon_chain/req/ping/1/ssz_snappy")
        self._host.set_stream_handler(ping_protocol, self._handle_ping_request)

        logger.info(
            f"Registered Ethereum P2P protocol handlers "
            f"(custody_group_count={self.config.custody_group_count})"
        )

    def _snappy_frame_compress(self, data: bytes) -> bytes:
        """Compress data using snappy framing format (required by Eth2 req/resp)."""
        import snappy
        compressor = snappy.StreamCompressor()
        return compressor.add_chunk(data)

    @staticmethod
    def _decode_varint(buf: bytes) -> tuple[int, int]:
        """Decode a varint from bytes and return (value, bytes_consumed)."""
        value = 0
        shift = 0
        for i, byte in enumerate(buf):
            value |= (byte & 0x7f) << shift
            shift += 7
            if (byte & 0x80) == 0:
                return value, i + 1
        raise ValueError("Truncated varint")

    async def _handle_status_v1_request(self, stream) -> None:
        """Handle incoming status/1 request (pre-Fulu)."""
        import varint

        try:
            peer_id = stream.muxed_conn.peer_id.to_base58()
            logger.info(f"Received status/1 request from {peer_id[:16]}...")

            await stream.read(1024)

            status = self._get_chain_status()
            finalized_root = status["finalized_root"]
            finalized_epoch = status["finalized_epoch"]
            head_root = status["head_root"]
            head_slot = status["head_slot"]

            # Status v1: 84 bytes
            # fork_digest (4) + finalized_root (32) + finalized_epoch (8) +
            # head_root (32) + head_slot (8)
            status_ssz = (
                self.config.fork_digest +
                finalized_root +
                finalized_epoch.to_bytes(8, "little") +
                head_root +
                head_slot.to_bytes(8, "little")
            )

            compressed = self._snappy_frame_compress(status_ssz)

            response_code = b"\x00"
            length_prefix = varint.encode(len(status_ssz))
            response = response_code + length_prefix + compressed

            await stream.write(response)
            await stream.close()
            logger.debug(f"Sent status/1 response to {peer_id[:16]}: slot={head_slot}, finalized_epoch={finalized_epoch}")

        except Exception as e:
            logger.warning(f"Error handling status/1 request: {e}")
            try:
                await stream.close()
            except Exception:
                pass

    async def _handle_status_v2_request(self, stream) -> None:
        """Handle incoming status/2 request (Fulu+).

        Status v2 adds earliest_available_slot field (8 bytes).
        Note: custody_group_count is exchanged via metadata/3, not status.
        """
        import varint

        try:
            peer_id = stream.muxed_conn.peer_id.to_base58()
            logger.info(f"Received status/2 request from {peer_id[:16]}...")

            await stream.read(1024)

            status = self._get_chain_status()
            finalized_root = status["finalized_root"]
            finalized_epoch = status["finalized_epoch"]
            head_root = status["head_root"]
            head_slot = status["head_slot"]
            earliest_available_slot = status.get("earliest_available_slot", 0)

            # Status v2: 92 bytes
            # fork_digest (4) + finalized_root (32) + finalized_epoch (8) +
            # head_root (32) + head_slot (8) + earliest_available_slot (8)
            status_ssz = (
                self.config.fork_digest +
                finalized_root +
                finalized_epoch.to_bytes(8, "little") +
                head_root +
                head_slot.to_bytes(8, "little") +
                earliest_available_slot.to_bytes(8, "little")
            )

            compressed = self._snappy_frame_compress(status_ssz)

            response_code = b"\x00"
            length_prefix = varint.encode(len(status_ssz))
            response = response_code + length_prefix + compressed

            await stream.write(response)
            await stream.close()
            logger.debug(f"Sent status/2 response to {peer_id[:16]}: slot={head_slot}, earliest={earliest_available_slot}")

        except Exception as e:
            logger.warning(f"Error handling status/2 request: {e}")
            try:
                await stream.close()
            except Exception:
                pass

    async def _handle_metadata_v2_request(self, stream) -> None:
        """Handle incoming metadata/2 request (Altair+ format)."""
        import varint

        try:
            peer_id = stream.muxed_conn.peer_id.to_base58()
            logger.debug(f"Received metadata/2 request from {peer_id[:16]}...")

            await stream.read(1024)

            seq_number = 1
            attnets = self.config.attnets
            syncnets = self.config.syncnets

            # MetaData v2 (Altair+): seq_number (8) + attnets (8) + syncnets (1) = 17 bytes
            metadata_ssz = (
                seq_number.to_bytes(8, "little") +
                attnets +
                syncnets
            )

            compressed = self._snappy_frame_compress(metadata_ssz)
            response_code = b"\x00"
            # Eth2 RPC uses varint-encoded UNCOMPRESSED length
            length_prefix = varint.encode(len(metadata_ssz))
            response = response_code + length_prefix + compressed

            await stream.write(response)
            await stream.close()

        except Exception as e:
            logger.warning(f"Error handling metadata/2 request: {e}")
            try:
                await stream.close()
            except Exception:
                pass

    async def _handle_metadata_v3_request(self, stream) -> None:
        """Handle incoming metadata/3 request (Fulu+ format with custody_group_count)."""
        import varint

        try:
            peer_id = stream.muxed_conn.peer_id.to_base58()
            logger.debug(f"Received metadata/3 request from {peer_id[:16]}...")

            await stream.read(1024)

            seq_number = 1
            attnets = self.config.attnets
            syncnets = self.config.syncnets
            custody_group_count = self.config.custody_group_count

            # MetaData v3 (Fulu+): seq_number (8) + attnets (8) + syncnets (1) + custody_group_count (8) = 25 bytes
            metadata_ssz = (
                seq_number.to_bytes(8, "little") +
                attnets +
                syncnets +
                custody_group_count.to_bytes(8, "little")
            )

            compressed = self._snappy_frame_compress(metadata_ssz)
            response_code = b"\x00"
            length_prefix = varint.encode(len(metadata_ssz))
            response = response_code + length_prefix + compressed

            await stream.write(response)
            await stream.close()
            logger.debug(f"Sent metadata/3 response to {peer_id[:16]} (cgc={custody_group_count})...")

        except Exception as e:
            logger.warning(f"Error handling metadata/3 request: {e}")
            try:
                await stream.close()
            except Exception:
                pass

    async def _handle_ping_request(self, stream) -> None:
        """Handle incoming ping request."""
        import varint

        try:
            peer_id = stream.muxed_conn.peer_id.to_base58()
            logger.debug(f"Received ping from {peer_id[:16]}...")

            await stream.read(1024)

            seq_number = 1
            ping_ssz = seq_number.to_bytes(8, "little")
            compressed = self._snappy_frame_compress(ping_ssz)

            response_code = b"\x00"
            # Eth2 RPC uses varint-encoded UNCOMPRESSED length
            length_prefix = varint.encode(len(ping_ssz))
            response = response_code + length_prefix + compressed

            await stream.write(response)
            await stream.close()

        except Exception as e:
            logger.warning(f"Error handling ping request: {e}")
            try:
                await stream.close()
            except Exception:
                pass

    def _generate_enr(self, ip: str, tcp_port: int) -> str:
        """Generate an ENR for this node.

        ENR format: enr:-<base64url encoded RLP>
        Includes eth2 field for network identification.
        """
        import base64
        import rlp
        from coincurve import PrivateKey

        if not self._private_key_bytes:
            return ""

        privkey = PrivateKey(self._private_key_bytes)
        pubkey_bytes = privkey.public_key.format(compressed=True)

        ip_bytes = bytes(int(x) for x in ip.split("."))
        tcp_bytes = tcp_port.to_bytes(2, "big")
        udp_bytes = tcp_port.to_bytes(2, "big")

        eth2_value = (
            self.config.fork_digest +
            self.config.next_fork_version +
            self.config.next_fork_epoch.to_bytes(8, "little")
        )

        seq = 1

        content = [
            b"attnets", self.config.attnets,
            b"eth2", eth2_value,
            b"id", b"v4",
            b"ip", ip_bytes,
            b"secp256k1", pubkey_bytes,
            b"syncnets", self.config.syncnets,
            b"tcp", tcp_bytes,
            b"udp", udp_bytes,
        ]

        from Crypto.Hash import keccak

        signature_input = rlp.encode([seq] + content)
        keccak_hash = keccak.new(digest_bits=256)
        keccak_hash.update(signature_input)
        msg_hash = keccak_hash.digest()
        signature = privkey.sign_recoverable(msg_hash, hasher=None)
        sig_bytes = signature[:64]

        enr_content = [sig_bytes, seq] + content
        enr_rlp = rlp.encode(enr_content)

        enr_b64 = base64.urlsafe_b64encode(enr_rlp).rstrip(b"=").decode()
        return f"enr:{enr_b64}"

    def _parse_enr(self, enr_str: str) -> tuple[str, int, str, bytes | None]:
        """Parse an ENR string to extract IP, port, peer ID, and fork_digest.

        Returns (ip, port, peer_id, fork_digest) tuple.
        fork_digest is the first 4 bytes of the eth2 field, or None if not present.
        """
        import base64
        import rlp

        if enr_str.startswith("enr:"):
            enr_str = enr_str[4:]

        # Note: "-" is a valid base64url character (replaces "+"), not a separator
        # Do NOT strip it

        # Add proper base64 padding
        padding_needed = (4 - len(enr_str) % 4) % 4
        enr_bytes = base64.urlsafe_b64decode(enr_str + "=" * padding_needed)

        decoded = rlp.decode(enr_bytes)

        ip = None
        tcp_port = None
        secp256k1_key = None
        fork_digest = None

        # RLP structure: [signature, seq, key1, val1, key2, val2, ...]
        # Skip signature (index 0) and seq (index 1), start at index 2
        i = 2
        while i < len(decoded) - 1:
            key = decoded[i]
            value = decoded[i + 1]

            if key == b"eth2" and len(value) >= 4:
                # eth2 field format: fork_digest (4) + next_fork_version (4) + next_fork_epoch (8)
                fork_digest = value[:4]
            elif key == b"ip":
                ip = ".".join(str(b) for b in value)
            elif key == b"tcp":
                tcp_port = int.from_bytes(value, "big")
            elif key == b"secp256k1":
                secp256k1_key = value
            i += 2

        if not ip or not tcp_port:
            raise ValueError("ENR missing ip or tcp port")

        if secp256k1_key:
            from libp2p.peer.id import ID
            from libp2p.crypto.secp256k1 import Secp256k1PublicKey

            pubkey = Secp256k1PublicKey.from_bytes(secp256k1_key)
            peer_id = ID.from_pubkey(pubkey).to_base58()
        else:
            raise ValueError("ENR missing secp256k1 key")

        if fork_digest:
            logger.info(f"ENR contains fork_digest: {fork_digest.hex()}")

        return ip, tcp_port, peer_id, fork_digest

    async def _connect_to_peer_trio(self, peer_addr: str) -> None:
        """Connect to a peer (Trio context).

        Supports multiple formats:
        - Full multiaddr: /ip4/x.x.x.x/tcp/9000/p2p/16Uiu2HAm...
        - ENR: enr:-...
        - Simple: host:port (requires peer to connect to us instead)
        """
        import socket
        from multiaddr import Multiaddr
        from libp2p.peer.peerinfo import info_from_p2p_addr, PeerInfo
        from libp2p.peer.id import ID

        try:
            enr_fork_digest = None
            if peer_addr.startswith("enr:") or peer_addr.startswith("-"):
                ip, port, peer_id, enr_fork_digest = self._parse_enr(peer_addr)
                maddr = Multiaddr(f"/ip4/{ip}/tcp/{port}/p2p/{peer_id}")
                logger.info(f"Parsed ENR: ip={ip}, port={port}, peer_id={peer_id[:16]}...")
                if enr_fork_digest and enr_fork_digest != self.config.fork_digest:
                    logger.warning(
                        f"Fork digest mismatch: ENR has {enr_fork_digest.hex()}, "
                        f"we have {self.config.fork_digest.hex()}"
                    )
                peer_info = info_from_p2p_addr(maddr)

            elif peer_addr.startswith("/"):
                if "/p2p/" in peer_addr:
                    maddr = Multiaddr(peer_addr)
                    peer_info = info_from_p2p_addr(maddr)
                else:
                    logger.warning(f"Multiaddr missing /p2p/ peer ID: {peer_addr}")
                    return
            else:
                host, port = peer_addr.rsplit(":", 1) if ":" in peer_addr else (peer_addr, "9000")
                try:
                    ip = socket.gethostbyname(host)
                    logger.debug(f"Resolved {host} to {ip}")
                except socket.gaierror:
                    ip = host

                logger.warning(
                    f"Peer {peer_addr} specified without peer ID. "
                    f"Use full multiaddr format: /ip4/{ip}/tcp/{port}/p2p/<peer_id> "
                    f"or ENR format: enr:-..."
                )
                return

            full_peer_id = peer_info.peer_id.to_base58()

            # Skip if already connected
            if full_peer_id in self._connected_peers:
                logger.debug(f"Already connected to peer: {full_peer_id[:16]}...")
                return

            await self._host.connect(peer_info)
            logger.info(f"Connected to peer: {full_peer_id[:16]}...")

            # Track the connection (notifier should also do this, but be explicit)
            self._connected_peers[full_peer_id] = {
                "peer_id": full_peer_id,
                "addrs": [str(a) for a in peer_info.addrs],
                "direction": "outbound"
            }
            self._peer_count = len(self._connected_peers)
            metrics.update_peers(self._peer_count)

            # Explicitly add peer to gossipsub mesh for all subscribed topics
            if self._gossipsub:
                for topic in self._subscriptions.keys():
                    try:
                        # Add peer to mesh - this ensures messages will be forwarded
                        if hasattr(self._gossipsub, 'mesh') and topic in self._gossipsub.mesh:
                            if peer_info.peer_id not in self._gossipsub.mesh[topic]:
                                self._gossipsub.mesh[topic].add(peer_info.peer_id)
                                logger.info(f"Added peer {full_peer_id[:16]} to mesh for topic {topic[:50]}...")
                    except Exception as e:
                        logger.debug(f"Could not add peer to mesh for {topic}: {e}")

            # Try to open a status stream to keep connection alive and verify it works
            # Use status/2 for Fulu (includes earliest_available_slot)
            try:
                from libp2p.custom_types import TProtocol
                import varint
                import snappy

                # Try status/2 first (Fulu+), fallback to status/1 if not supported
                status_v2_protocol = TProtocol("/eth2/beacon_chain/req/status/2/ssz_snappy")
                status_v1_protocol = TProtocol("/eth2/beacon_chain/req/status/1/ssz_snappy")
                logger.info(f"Opening status/2 stream to {peer_info.peer_id.to_base58()[:16]}...")

                try:
                    stream = await self._host.new_stream(peer_info.peer_id, [status_v2_protocol])
                    use_v2 = True
                    logger.info(f"Status/2 stream opened successfully to {peer_info.peer_id.to_base58()[:16]}")
                except Exception as e:
                    logger.debug(f"Status/2 not supported, falling back to status/1: {e}")
                    stream = await self._host.new_stream(peer_info.peer_id, [status_v1_protocol])
                    use_v2 = False
                    logger.info(f"Status/1 stream opened successfully to {peer_info.peer_id.to_base58()[:16]}")

                # Build our status message from chain state
                status = self._get_chain_status()
                finalized_root = status["finalized_root"]
                finalized_epoch = status["finalized_epoch"]
                head_root = status["head_root"]
                head_slot = status["head_slot"]

                if use_v2:
                    earliest_available_slot = status.get("earliest_available_slot", 0)
                    # Status v2: 92 bytes (v1 + earliest_available_slot)
                    status_ssz = (
                        self.config.fork_digest +
                        finalized_root +
                        finalized_epoch.to_bytes(8, "little") +
                        head_root +
                        head_slot.to_bytes(8, "little") +
                        earliest_available_slot.to_bytes(8, "little")
                    )
                else:
                    status_ssz = (
                        self.config.fork_digest +
                        finalized_root +
                        finalized_epoch.to_bytes(8, "little") +
                        head_root +
                        head_slot.to_bytes(8, "little")
                    )

                # Eth2 RPC format: varint-encoded UNCOMPRESSED length + snappy framed data
                compressed = self._snappy_frame_compress(status_ssz)
                length_prefix = varint.encode(len(status_ssz))
                message = length_prefix + compressed
                version = "v2" if use_v2 else "v1"
                logger.info(f"Sending status/{version}: {len(status_ssz)} bytes SSZ, {len(compressed)} bytes compressed")

                await stream.write(message)
                logger.info(f"Sent status request to {peer_info.peer_id.to_base58()[:16]}")

                # Read response - first read response code
                import trio
                with trio.move_on_after(5.0) as cancel_scope:
                    response_code = await stream.read(1)
                    if response_code:
                        logger.info(f"Response code: {response_code[0]}")
                        if response_code[0] == 0:  # Success
                            # Read varint length (uncompressed size)
                            length_bytes = await stream.read(10)  # Max varint size
                            if length_bytes:
                                uncompressed_length, consumed = self._decode_varint(length_bytes)
                                logger.info(f"Response uncompressed length: {uncompressed_length} bytes")
                                # Read remaining data (framed snappy)
                                # The framed format has variable size, read generously
                                remaining = length_bytes[consumed:]
                                more_data = await stream.read(1024)
                                if more_data:
                                    remaining += more_data
                                # Decompress using framed format
                                decompressor = snappy.StreamDecompressor()
                                status_data = decompressor.decompress(remaining)
                                logger.info(f"Received peer status: {len(status_data)} bytes decompressed")
                                # Parse fork digest
                                if len(status_data) >= 4:
                                    peer_fork_digest = status_data[:4].hex()
                                    logger.info(f"Peer fork_digest: {peer_fork_digest}")
                    else:
                        logger.info("No response code received")
                if cancel_scope.cancelled_caught:
                    logger.warning("Timeout waiting for status response")

                await stream.close()
            except Exception as e:
                logger.warning(f"Failed to send status to peer: {e}")
                import traceback
                traceback.print_exc()

        except Exception as e:
            logger.warning(f"Failed to connect to peer {peer_addr[:50]}: {e}")

    async def _log_and_maintain_mesh(self) -> None:
        """Log current mesh state and ensure all peers are in the mesh."""
        if not self._gossipsub:
            return

        try:
            mesh = getattr(self._gossipsub, 'mesh', {})
            fanout = getattr(self._gossipsub, 'fanout', {})
            peers_in_mesh = {}

            # Aggregate mesh status by base topic name (ignore fork digest variations)
            topic_summary: dict[str, set] = {}
            for topic, peer_set in mesh.items():
                short_topic = topic.split("/")[-2] if "/" in topic else topic[:30]
                if short_topic not in topic_summary:
                    topic_summary[short_topic] = set()
                if peer_set:
                    topic_summary[short_topic].update(str(p)[:16] for p in peer_set)
                peers_in_mesh[short_topic] = len(topic_summary[short_topic])

            # Log only once per base topic type
            empty_topics = []
            for short_topic, peer_ids in topic_summary.items():
                if peer_ids:
                    logger.debug(f"Mesh[{short_topic}]: {len(peer_ids)} peers")
                else:
                    empty_topics.append(short_topic)

            if empty_topics:
                logger.warning(f"Empty mesh for topics: {empty_topics}")

            # Try to add connected peers to mesh if mesh is empty
            for topic in self._subscriptions.keys():
                if topic in mesh and len(mesh[topic]) == 0:
                    for peer_id_str, peer_info in self._connected_peers.items():
                        try:
                            from libp2p.peer.id import ID
                            peer_id = ID.from_base58(peer_id_str)
                            mesh[topic].add(peer_id)
                            logger.info(f"Added {peer_id_str[:16]} to empty mesh for {topic[:50]}")
                        except Exception as e:
                            logger.debug(f"Could not add peer to mesh: {e}")

            logger.info(f"Connected peers: {len(self._connected_peers)}, Subscriptions: {len(self._subscriptions)}")
        except Exception as e:
            logger.debug(f"Error logging mesh state: {e}")

    async def _process_publish_queue_trio(self) -> None:
        """Process outgoing messages from the publish queue (Trio context)."""
        try:
            while True:
                try:
                    topic, data = self._publish_queue.get_nowait()
                    if self._pubsub:
                        # Log mesh state before publishing
                        mesh = getattr(self._gossipsub, 'mesh', {})
                        mesh_peers = len(mesh.get(topic, set())) if mesh else 0
                        short_topic = topic.split("/")[-2] if "/" in topic else topic[:30]

                        await self._pubsub.publish(topic, data)
                        metrics.record_gossip_sent(topic)
                        logger.info(f"GOSSIP SEND to {short_topic}: {len(data)} bytes (mesh_peers={mesh_peers})")
                except queue.Empty:
                    break
        except Exception as e:
            logger.error(f"Error processing publish queue: {e}")

    async def _process_subscribe_queue_trio(self, nursery) -> None:
        """Process subscription requests from the subscribe queue (Trio context)."""
        try:
            while True:
                try:
                    topic = self._subscribe_queue.get_nowait()
                    if self._pubsub and topic not in self._subscriptions:
                        subscription = await self._pubsub.subscribe(topic)
                        self._subscriptions[topic] = subscription
                        logger.debug(f"Subscribed to gossipsub topic: {topic}")
                        # Start a task to read messages from this subscription
                        nursery.start_soon(self._read_subscription_trio, topic, subscription)
                except queue.Empty:
                    break
        except Exception as e:
            logger.error(f"Error processing subscribe queue: {e}")

    async def _read_subscription_trio(self, topic: str, subscription) -> None:
        """Read messages from a gossipsub subscription and queue them (Trio context)."""
        logger.info(f"Started reading messages from topic: {topic}")
        msg_count = 0
        try:
            while self._running:
                try:
                    msg = await subscription.get()
                    msg_count += 1
                    # Put the message in the queue for asyncio processing
                    from_peer = msg.from_id.hex() if hasattr(msg, 'from_id') else "unknown"
                    self._message_queue.put((topic, msg.data, from_peer))
                    metrics.record_gossip_received(topic)
                    # Log received messages at INFO level to track gossip activity
                    short_topic = topic.split("/")[-2] if "/" in topic else topic
                    logger.info(f"GOSSIP RECV #{msg_count} on {short_topic}: {len(msg.data)} bytes from {from_peer[:16]}")
                except Exception as e:
                    if self._running:
                        logger.warning(f"Error reading from subscription {topic}: {e}")
                    break
        except Exception as e:
            logger.error(f"Subscription reader for {topic} failed: {e}")

    async def _process_incoming_messages(self) -> None:
        """Process incoming messages and call handlers (asyncio context)."""
        while self._running:
            try:
                topic, data, from_peer = self._message_queue.get_nowait()
                handler = self._handlers.get(topic)
                if handler:
                    await handler(data, from_peer)
            except queue.Empty:
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error processing incoming message: {e}")

    async def stop(self) -> None:
        """Stop the P2P host."""
        self._running = False
        self._stop_event.set()

        if self._trio_thread and self._trio_thread.is_alive():
            self._trio_thread.join(timeout=5.0)

        logger.info("P2P host stopped")

    async def subscribe(self, topic: str, handler: MessageHandler) -> None:
        """Subscribe to a gossip topic with a message handler."""
        self._handlers[topic] = handler
        # Queue the subscription request for the Trio thread to process
        self._subscribe_queue.put(topic)
        logger.info(f"Queued subscription for topic: {topic}")

    async def publish(self, topic: str, data: bytes) -> None:
        """Publish a message to a gossip topic."""
        if not self._running:
            logger.warning(f"Cannot publish to {topic}: host not running")
            return

        self._publish_queue.put((topic, data))
        logger.debug(f"Queued message for {topic}: {len(data)} bytes")

    @property
    def peer_id(self) -> Optional[str]:
        """Get the local peer ID."""
        return self._peer_id

    @property
    def peer_count(self) -> int:
        """Get the number of connected peers."""
        return self._peer_count

    @property
    def connected_peers(self) -> list[dict]:
        """Get list of connected peers with their info."""
        return list(self._connected_peers.values())

    @property
    def enr(self) -> Optional[str]:
        """Get the local ENR."""
        return self._enr

    @property
    def multiaddr(self) -> Optional[str]:
        """Get the local multiaddr with peer ID."""
        if self._listen_ip and self._peer_id:
            return f"/ip4/{self._listen_ip}/tcp/{self.config.listen_port}/p2p/{self._peer_id}"
        return None
