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


# Per-PR #606 PeerScoreReason penalty applied to a peer's running score
# when an event is recorded. Numbers are deliberately small and roughly
# proportional to severity — they're a placeholder for a real scoring
# policy, not a production-grade peer manager. Range is consistent with
# lighthouse-style scoring (baseline 0, worse peers go more negative);
# the spec calls these implementation-defined and not cross-client
# comparable. Unknown reasons fall through to -1.0.
_PEER_SCORE_PENALTY: dict[str, float] = {
    "gossip_invalid_block": -10.0,
    "gossip_invalid_attestation": -5.0,
    "gossip_invalid_blob_sidecar": -5.0,
    "gossip_invalid_data_column_sidecar": -5.0,
    "rpc_invalid_request": -5.0,
    "rpc_invalid_response": -5.0,
    "rpc_rate_limited": -1.0,
    "rpc_timeout": -3.0,
    "rpc_io_error": -3.0,
    "rpc_bad_blocks_by_range": -7.0,
    "rpc_bad_blocks_by_root": -7.0,
    "sync_bad_batch": -10.0,
    # Fork mismatch → disconnect; flag aggressively even though we
    # don't validate Status fork_digest yet.
    "status_unviable_fork": -100.0,
    "behaviour_penalty": -5.0,
    "unknown": -1.0,
}


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

    # Mutable. Initial value is the conservative non-validator default
    # (CUSTODY_REQUIREMENT). BeaconNode overwrites this after validator
    # keys + state are available, applying
    # `get_validators_custody_requirement` from
    # `consensus-specs/specs/fulu/validator.md`. Supernodes are pinned to
    # NUMBER_OF_CUSTODY_GROUPS up-front. The spec forbids decreases, so
    # the setter on P2PHost takes max(old, new).
    custody_group_count: int = 4


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
        self._status_sender_thread: Optional[threading.Thread] = None
        # Peers we've already sent an outbound Status RPC to. Eth2 spec says
        # the dialer is expected to send Status first; if we don't, peers
        # like prysm disconnect us with "no chain status for peer".
        self._status_sent_to: set[str] = set()

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

    def push_status_snapshot(self, status: dict) -> None:
        """Install a StatusMessage snapshot in the Rust binding so inbound
        Status RPCs are answered without crossing into Python. Should be
        invoked whenever head_root / head_slot / finalized_checkpoint
        changes — see the rationale in network.rs (cached_status field).
        """
        if self._network is None:
            return
        try:
            import consensoor_p2p as cp
            fork_digest = bytes(status.get("fork_digest", self.config.fork_digest))
            msg = cp.StatusMessage(
                fork_digest=list(fork_digest),
                finalized_root=list(bytes(status.get("finalized_root", b"\x00" * 32))),
                finalized_epoch=int(status.get("finalized_epoch", 0)),
                head_root=list(bytes(status.get("head_root", b"\x00" * 32))),
                head_slot=int(status.get("head_slot", 0)),
                earliest_available_slot=int(status.get("earliest_available_slot", 0)),
            )
            self._network.set_cached_status(msg)
        except Exception as e:
            logger.warning(f"push_status_snapshot failed: {e}")

    def update_fork_digest(self, fork_digest: bytes) -> None:
        self.config.fork_digest = fork_digest
        if self._network is not None:
            try:
                self._network.update_enr(fork_digest=list(fork_digest))
            except Exception as e:
                logger.warning(f"ENR fork_digest update failed: {e}")

    def update_custody_group_count(self, new_count: int) -> None:
        """Bump our advertised custody_group_count.

        Per `consensus-specs/specs/fulu/validator.md`, the count MUST NOT
        decrease — once a node advertises a higher CGC it keeps custodying
        and serving those columns. We take max(current, new) and bump
        seq_number so peers re-pull MetaData v3.
        """
        old = self.config.custody_group_count
        target = max(old, int(new_count))
        if target == old:
            return
        self.config.custody_group_count = target
        self._our_metadata_seq += 1
        logger.info(
            f"custody_group_count updated: {old} -> {target} "
            f"(seq_number={self._our_metadata_seq})"
        )
        if self._network is not None:
            try:
                self._network.update_enr(cgc=target)
            except Exception as e:
                logger.warning(f"ENR cgc update failed: {e}")

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
            next_fork_version=list(self.config.next_fork_version),
            next_fork_epoch=int(self.config.next_fork_epoch),
            attnets=list(self.config.attnets),
            syncnets=list(self.config.syncnets),
            cgc=int(self.config.custody_group_count),
            external_ip=get_local_ip(),
        )
        self._network = cp.Network.start(net_cfg)
        self._peer_id = self._network.peer_id()

        # Spawn the rust→python dispatcher threads BEFORE anything that yields
        # to the asyncio loop. Inbound peers (prysm, lighthouse) hit us with a
        # Status RPC immediately on connect and disconnect us with "no chain
        # status for peer" if we don't reply within ~20s. If we delay these
        # threads until after `await asyncio.sleep` / dial / etc., the rust
        # side queues the Status request but no python consumer is reading.
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

        self._status_sender_thread = threading.Thread(
            target=self._status_sender_loop, name="p2p-status-send", daemon=True
        )
        self._status_sender_thread.start()

        # Give the listener a moment to bind, then snapshot addresses.
        await asyncio.sleep(0.1)
        self._listen_addrs = list(self._network.listen_addresses())

        # Dial each configured static peer (multiaddr including /p2p/<peer_id>).
        for addr in self.config.static_peers:
            try:
                self._network.dial(addr)
            except Exception as e:
                logger.warning(f"dial failed for {addr}: {e}")

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

    def _status_sender_loop(self) -> None:
        """Proactively send Status RPC to every newly-connected peer.

        Eth2 spec: the dialer is expected to send Status first after connect.
        Prysm/lighthouse/teku disconnect peers that don't initiate Status
        within ~20s with the error "no chain status for peer". The rust
        binding only forwards INBOUND Status requests via the status channel,
        so we have to poll connected_peers() and send our Status to anyone
        we haven't sent it to yet.
        """
        import consensoor_p2p as cp
        import time as _time
        while not self._stop.is_set():
            _time.sleep(0.5)
            if self._network is None:
                continue
            if self._status_provider is None:
                continue
            try:
                connected = set(self._network.connected_peers())
            except Exception as e:
                logger.warning(f"connected_peers poll failed: {e}")
                continue
            # Drop tracking entries for peers that have since disconnected so
            # we re-send Status if we ever reconnect.
            stale = self._status_sent_to - connected
            if stale:
                self._status_sent_to -= stale
            for peer_id in connected - self._status_sent_to:
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
                    self._network.send_status(peer_id, msg)
                    self._status_sent_to.add(peer_id)
                    logger.info(f"Sent outbound Status to {peer_id[:24]}...")
                except Exception as e:
                    logger.warning(f"send_status to {peer_id[:24]}... failed: {e}")

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
                # PR #606 PeerScoreReason classification from libp2p
                # request-response error Display strings. We see things
                # like "Timeout", "Io(...)", "ConnectionClosed",
                # "UnsupportedProtocols", "DialFailure". Only the first
                # two map cleanly to spec enum values — everything else
                # is bucketed as rpc_invalid_response.
                err_str = (ev.error or "").lower()
                if "timeout" in err_str:
                    reason = "rpc_timeout"
                elif "io" in err_str:
                    reason = "rpc_io_error"
                else:
                    reason = "rpc_invalid_response"
                self.record_peer_score_event(ev.peer, reason)

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
        if self._network is None:
            return 0
        try:
            return len(self._network.connected_peers())
        except Exception:
            return 0

    def connected_peers(self) -> list[dict]:
        """Return currently-connected peers as Beacon-API-shaped dicts.

        Includes per-peer fields needed by `/eth/v1/node/peers`
        (beacon-APIs PR #606): base64 `enr` (empty if discv5 never
        surfaced one), libp2p identify `agent_version` (empty until
        handshake), client-native `score` (None until something records
        one), and `downscore_reasons` (empty list until events recorded).
        """
        if self._network is None:
            return []
        # Prefer the richer accessor that ships all PR #606 fields; fall
        # back to the legacy direction-only one if the binding hasn't
        # been rebuilt yet (e.g. dev cycle where the .so is stale).
        try:
            # Default score to 0.0 (lighthouse-style baseline) for every
            # connected peer that hasn't been penalized yet — keeps the
            # field visible in `/eth/v1/node/peers` instead of being
            # omitted under the "field is None" gate.
            return [
                {
                    "peer_id": pid,
                    "addrs": [],
                    "direction": direction,
                    "enr": enr,
                    "agent_version": agent,
                    "score": 0.0 if score is None else score,
                    "downscore_reasons": list(downscore_reasons or []),
                }
                for pid, direction, enr, agent, score, downscore_reasons
                in self._network.connected_peers_with_meta()
            ]
        except AttributeError:
            try:
                return [
                    {
                        "peer_id": pid,
                        "addrs": [],
                        "direction": direction,
                        "enr": "",
                        "agent_version": "",
                        "score": 0.0,
                        "downscore_reasons": [],
                    }
                    for pid, direction in self._network.connected_peers_with_direction()
                ]
            except Exception:
                return []
        except Exception:
            return []

    def set_peer_score(self, peer_id: str, score: float) -> None:
        """Record the client-native score for `peer_id` (beacon-APIs
        PR #606 `score` field). Plumbing only — consensoor doesn't
        compute scores today; this exists so a future scoring policy
        can write through to the beacon API without further binding work."""
        if self._network is None:
            return
        try:
            self._network.set_peer_score(peer_id, float(score))
        except (AttributeError, Exception):
            pass

    def record_peer_score_event(self, peer_id: str, reason: str) -> None:
        """Push a PR #606 `PeerScoreReason` onto the peer's downscore-
        reasons ring buffer AND apply the corresponding penalty from
        `_PEER_SCORE_PENALTY` to the peer's running score (atomic on
        the Rust side). Unknown reasons get -1.0. The spec tolerates
        unknown values for `reason`, so no enum validation here."""
        if self._network is None:
            return
        delta = _PEER_SCORE_PENALTY.get(reason, -1.0)
        try:
            self._network.record_peer_score_event(peer_id, str(reason), float(delta))
        except (AttributeError, Exception):
            pass

    @property
    def enr(self) -> Optional[str]:
        """Latest signed local ENR (`enr:` base64 string).

        The Rust binding builds + signs this with the libp2p secp256k1
        keypair on construction and re-signs it whenever `fork_digest`,
        `cgc`, `attnets`, or `syncnets` change. discv5 isn't running yet,
        so the ENR isn't published over UDP — but `/eth/v1/node/identity`
        returns it and dora can decode every field.
        """
        if self._network is None:
            return None
        try:
            return self._network.enr()
        except Exception:
            return None

    @property
    def multiaddr(self) -> Optional[str]:
        if not self._listen_addrs:
            return None
        addr = self._listen_addrs[0]
        return f"{addr}/p2p/{self._peer_id}" if self._peer_id else addr

    def get_mesh_info(self, topic: str) -> str:
        return f"topic={topic} peers=unknown (rust binding stub)"

    # ------------------------------------------------------------------ ReqResp

    async def request_blocks_by_range(
        self, start_slot: int, count: int, timeout: float = 15.0
    ) -> list[bytes]:
        """Issue a BeaconBlocksByRange request via the rust binding.

        Returns a list of raw SSZ-encoded SignedBeaconBlock bytes (one per
        chunk). Empty list if every peer either errored, timed out, or
        responded with no chunks.

        Iterates through all currently-connected peers — a single peer
        returning empty is common (they don't have those blocks yet, or
        decline to serve sync ranges close to head). Without retry we miss
        blocks any time gossipsub also dropped them, which strands the
        whole node on a stale head (slot 15 incident: 2026-05-15).
        Per-peer budget is `timeout / max_peers` so the overall call
        still respects the caller's deadline.
        """
        if self._network is None:
            return []
        try:
            connected = list(self._network.connected_peers())
        except Exception:
            connected = []
        candidates: list[dict] = []
        if connected:
            candidates = list(connected)
        else:
            # No live peers — fall back to whatever was statically configured.
            for addr in self.config.static_peers:
                if "/p2p/" in addr:
                    candidates.append({"peer_id": addr.rsplit("/p2p/", 1)[1]})

        if not candidates:
            logger.debug("request_blocks_by_range: no known peer to dial")
            return []

        # Bound the per-peer wait so iterating across all peers still
        # fits in the caller-provided deadline. Floor at 1.5s so we
        # don't ask peers to respond in milliseconds.
        per_peer_timeout = max(1.5, timeout / max(1, len(candidates)))
        loop = asyncio.get_running_loop()

        for peer_info in candidates:
            peer_id = peer_info.get("peer_id") if isinstance(peer_info, dict) else peer_info
            if not peer_id:
                continue
            try:
                self._network.request_blocks_by_range(peer_id, int(start_slot), int(count))
            except Exception as e:
                logger.debug(f"BeaconBlocksByRange send to {peer_id[:24]}... failed: {e}")
                continue
            ev = await loop.run_in_executor(
                None, lambda: self._network.next_blocks_by_range(int(per_peer_timeout * 1000))
            )
            if ev is None:
                logger.debug(
                    f"BeaconBlocksByRange timeout from {peer_id[:24]}... "
                    f"(slots {start_slot}..{start_slot + count - 1})"
                )
                continue
            if ev.error:
                logger.debug(
                    f"BeaconBlocksByRange error from {peer_id[:24]}...: {ev.error}"
                )
                continue
            if ev.response is None or not ev.response.chunks:
                logger.debug(
                    f"BeaconBlocksByRange empty response from {peer_id[:24]}... "
                    f"(slots {start_slot}..{start_slot + count - 1})"
                )
                continue
            return [bytes(c.ssz_block) for c in ev.response.chunks]

        logger.warning(
            f"BeaconBlocksByRange exhausted {len(candidates)} peer(s) without chunks "
            f"(slots {start_slot}..{start_slot + count - 1})"
        )
        return []
