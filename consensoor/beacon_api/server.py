"""Beacon API HTTP server."""

import asyncio
import json
import logging
from typing import Optional
from aiohttp import web

from .utils import get_local_ip, generate_peer_id
from .spec import build_spec_response
from ..spec.network_config import get_config as get_network_config

logger = logging.getLogger(__name__)


class BeaconAPI:
    """Simple Beacon API server."""

    def __init__(self, node, host: str = "0.0.0.0", port: int = 5052):
        self.node = node
        self.host = host
        self.port = port
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self._event_subscribers: list[asyncio.Queue] = []
        self._last_head_slot: int = 0
        self._last_head_root: Optional[bytes] = None
        self._last_finalized_epoch: int = 0
        self._event_emitter_task: Optional[asyncio.Task] = None
        self._setup_routes()

    def _setup_routes(self):
        """Set up API routes."""
        self.app.router.add_get("/eth/v1/node/health", self.get_health)
        self.app.router.add_get("/eth/v1/node/version", self.get_version)
        self.app.router.add_get("/eth/v1/node/syncing", self.get_syncing)
        self.app.router.add_get("/eth/v1/node/identity", self.get_identity)
        self.app.router.add_get("/eth/v1/node/peers", self.get_peers)
        self.app.router.add_get("/eth/v1/beacon/genesis", self.get_genesis)
        self.app.router.add_get("/eth/v1/beacon/states/{state_id}/root", self.get_state_root)
        self.app.router.add_get("/eth/v1/beacon/states/{state_id}/finality_checkpoints", self.get_finality_checkpoints)
        self.app.router.add_get("/eth/v1/beacon/headers", self.get_headers)
        self.app.router.add_get("/eth/v1/beacon/headers/{block_id}", self.get_header)
        self.app.router.add_get("/eth/v2/beacon/blocks/{block_id}", self.get_block)
        self.app.router.add_get("/eth/v1/config/spec", self.get_spec)
        self.app.router.add_get("/eth/v1/events", self.get_events)

    async def start(self):
        """Start the API server."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()
        self._event_emitter_task = asyncio.create_task(self._emit_events_loop())
        logger.info(f"Beacon API listening on {self.host}:{self.port}")

    async def stop(self):
        """Stop the API server."""
        if self._event_emitter_task:
            self._event_emitter_task.cancel()
            try:
                await self._event_emitter_task
            except asyncio.CancelledError:
                pass
        if self.runner:
            await self.runner.cleanup()

    async def get_health(self, request: web.Request) -> web.Response:
        """GET /eth/v1/node/health"""
        return web.Response(status=200)

    async def get_version(self, request: web.Request) -> web.Response:
        """GET /eth/v1/node/version"""
        return web.json_response({
            "data": {
                "version": "consensoor/v0.1.0"
            }
        })

    async def get_syncing(self, request: web.Request) -> web.Response:
        """GET /eth/v1/node/syncing"""
        return web.json_response({
            "data": {
                "head_slot": str(self.node.head_slot),
                "sync_distance": "0",
                "is_syncing": False,
                "is_optimistic": False,
                "el_offline": False,
            }
        })

    async def get_identity(self, request: web.Request) -> web.Response:
        """GET /eth/v1/node/identity"""
        local_ip = get_local_ip()
        listen_port = self.node.config.listen_port

        if self.node.beacon_gossip and self.node.beacon_gossip.peer_id:
            peer_id = self.node.beacon_gossip.peer_id
            enr = self.node.beacon_gossip._host.enr or ""
            multiaddr = self.node.beacon_gossip._host.multiaddr
            p2p_addresses = [multiaddr] if multiaddr else []
        else:
            peer_id = generate_peer_id(f"consensoor-{local_ip}-{self.port}")
            enr = ""
            p2p_addresses = [f"/ip4/{local_ip}/tcp/{listen_port}/p2p/{peer_id}"]

        return web.json_response({
            "data": {
                "peer_id": peer_id,
                "enr": enr,
                "p2p_addresses": p2p_addresses,
                "discovery_addresses": [
                    f"/ip4/{local_ip}/udp/{listen_port}/p2p/{peer_id}"
                ],
                "metadata": {
                    "seq_number": "1",
                    "attnets": "0xffffffffffffffff",
                    "syncnets": "0x0f",
                }
            }
        })

    async def get_peers(self, request: web.Request) -> web.Response:
        """GET /eth/v1/node/peers"""
        peers = []
        for peer_addr in self.node.config.peers:
            if ":" in peer_addr:
                host, port = peer_addr.rsplit(":", 1)
            else:
                host = peer_addr
                port = "9000"
            peer_id = generate_peer_id(f"peer-{host}-{port}")
            peers.append({
                "peer_id": peer_id,
                "enr": "",
                "last_seen_p2p_address": f"/ip4/{host}/tcp/{port}/p2p/{peer_id}",
                "state": "connected",
                "direction": "outbound",
            })
        return web.json_response({
            "data": peers,
            "meta": {
                "count": len(peers)
            }
        })

    async def get_genesis(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/genesis"""
        if not self.node.state:
            return web.json_response({"message": "Genesis not loaded"}, status=404)
        return web.json_response({
            "data": {
                "genesis_time": str(self.node.state.genesis_time),
                "genesis_validators_root": "0x" + bytes(self.node.state.genesis_validators_root).hex(),
                "genesis_fork_version": "0x" + get_network_config().genesis_fork_version.hex(),
            }
        })

    async def get_state_root(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/states/{state_id}/root"""
        state_id = request.match_info["state_id"]
        if state_id == "head" and self.node.head_root:
            return web.json_response({
                "data": {"root": "0x" + self.node.head_root.hex()}
            })
        return web.json_response({"message": "State not found"}, status=404)

    async def get_finality_checkpoints(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/states/{state_id}/finality_checkpoints"""
        state_id = request.match_info["state_id"]
        if state_id == "head" and self.node.state:
            return web.json_response({
                "execution_optimistic": False,
                "finalized": True,
                "data": {
                    "previous_justified": {
                        "epoch": str(self.node.state.previous_justified_checkpoint.epoch),
                        "root": "0x" + bytes(self.node.state.previous_justified_checkpoint.root).hex(),
                    },
                    "current_justified": {
                        "epoch": str(self.node.state.current_justified_checkpoint.epoch),
                        "root": "0x" + bytes(self.node.state.current_justified_checkpoint.root).hex(),
                    },
                    "finalized": {
                        "epoch": str(self.node.state.finalized_checkpoint.epoch),
                        "root": "0x" + bytes(self.node.state.finalized_checkpoint.root).hex(),
                    },
                }
            })
        return web.json_response({"message": "State not found"}, status=404)

    async def get_headers(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/headers"""
        return web.json_response({"data": []})

    def _resolve_block_id(self, block_id: str) -> tuple[Optional[bytes], Optional[object]]:
        """Resolve a block_id to (root, signed_block).

        Supports: "head", "finalized", "0x..." (root), slot number.
        Returns (None, None) if not found.
        """
        if block_id == "head":
            if self.node.head_root:
                block = self.node.store.get_block(self.node.head_root)
                return self.node.head_root, block
            return None, None

        if block_id == "finalized":
            if self.node.state:
                root = bytes(self.node.state.finalized_checkpoint.root)
                block = self.node.store.get_block(root)
                return root, block
            return None, None

        if block_id.startswith("0x"):
            try:
                root = bytes.fromhex(block_id[2:])
                if len(root) == 32:
                    block = self.node.store.get_block(root)
                    if block:
                        return root, block
            except ValueError:
                pass
            return None, None

        try:
            slot = int(block_id)
            for root, block in self.node.store.blocks.items():
                if hasattr(block, "message") and int(block.message.slot) == slot:
                    return root, block
            return None, None
        except ValueError:
            return None, None

    def _get_block_version(self, signed_block) -> str:
        """Determine the fork version string for a block."""
        if not hasattr(signed_block, "message"):
            return "phase0"

        block = signed_block.message
        if not hasattr(block, "body"):
            return "phase0"

        body = block.body
        if hasattr(body, "signed_execution_payload_header"):
            return "fulu"
        if hasattr(body, "blob_kzg_commitments"):
            if hasattr(body, "execution_requests"):
                return "electra"
            return "deneb"
        if hasattr(body, "execution_payload"):
            if hasattr(body.execution_payload, "withdrawals"):
                return "capella"
            return "bellatrix"
        if hasattr(body, "sync_aggregate"):
            return "altair"
        return "phase0"

    async def get_header(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/headers/{block_id}"""
        block_id = request.match_info["block_id"]
        root, signed_block = self._resolve_block_id(block_id)

        if root is None:
            if block_id == "head" and self.node.state and self.node.head_root:
                header = self.node.state.latest_block_header
                return web.json_response({
                    "execution_optimistic": False,
                    "finalized": False,
                    "data": {
                        "root": "0x" + self.node.head_root.hex(),
                        "canonical": True,
                        "header": {
                            "message": {
                                "slot": str(self.node.head_slot),
                                "proposer_index": str(header.proposer_index),
                                "parent_root": "0x" + bytes(header.parent_root).hex(),
                                "state_root": "0x" + bytes(header.state_root).hex(),
                                "body_root": "0x" + bytes(header.body_root).hex(),
                            },
                            "signature": "0x" + "00" * 96,
                        }
                    }
                })
            return web.json_response({"message": "Block not found"}, status=404)

        block = signed_block.message
        from ..crypto import hash_tree_root
        body_root = hash_tree_root(block.body)

        signature = bytes(signed_block.signature) if hasattr(signed_block, "signature") else b"\x00" * 96

        return web.json_response({
            "execution_optimistic": False,
            "finalized": False,
            "data": {
                "root": "0x" + root.hex(),
                "canonical": True,
                "header": {
                    "message": {
                        "slot": str(block.slot),
                        "proposer_index": str(block.proposer_index),
                        "parent_root": "0x" + bytes(block.parent_root).hex(),
                        "state_root": "0x" + bytes(block.state_root).hex(),
                        "body_root": "0x" + body_root.hex(),
                    },
                    "signature": "0x" + signature.hex(),
                }
            }
        })

    async def get_block(self, request: web.Request) -> web.Response:
        """GET /eth/v2/beacon/blocks/{block_id}"""
        block_id = request.match_info["block_id"]
        root, signed_block = self._resolve_block_id(block_id)

        if root is None or signed_block is None:
            return web.json_response({"message": "Block not found"}, status=404)

        accept = request.headers.get("Accept", "application/json")

        if "application/octet-stream" in accept:
            ssz_bytes = signed_block.encode_bytes()
            version = self._get_block_version(signed_block)
            return web.Response(
                body=ssz_bytes,
                content_type="application/octet-stream",
                headers={"Eth-Consensus-Version": version},
            )

        version = self._get_block_version(signed_block)
        block_json = self._signed_block_to_json(signed_block)

        return web.json_response({
            "version": version,
            "execution_optimistic": False,
            "finalized": False,
            "data": block_json,
        })

    def _signed_block_to_json(self, signed_block) -> dict:
        """Convert a signed beacon block to JSON format."""
        block = signed_block.message
        signature = bytes(signed_block.signature) if hasattr(signed_block, "signature") else b"\x00" * 96

        result = {
            "message": self._block_to_json(block),
            "signature": "0x" + signature.hex(),
        }
        return result

    def _block_to_json(self, block) -> dict:
        """Convert a beacon block to JSON format."""
        result = {
            "slot": str(block.slot),
            "proposer_index": str(block.proposer_index),
            "parent_root": "0x" + bytes(block.parent_root).hex(),
            "state_root": "0x" + bytes(block.state_root).hex(),
            "body": self._block_body_to_json(block.body),
        }
        return result

    def _block_body_to_json(self, body) -> dict:
        """Convert a beacon block body to JSON format."""
        result = {
            "randao_reveal": "0x" + bytes(body.randao_reveal).hex(),
            "eth1_data": {
                "deposit_root": "0x" + bytes(body.eth1_data.deposit_root).hex(),
                "deposit_count": str(body.eth1_data.deposit_count),
                "block_hash": "0x" + bytes(body.eth1_data.block_hash).hex(),
            },
            "graffiti": "0x" + bytes(body.graffiti).hex(),
            "proposer_slashings": [],
            "attester_slashings": [],
            "attestations": [],
            "deposits": [],
            "voluntary_exits": [],
        }

        if hasattr(body, "sync_aggregate"):
            result["sync_aggregate"] = {
                "sync_committee_bits": "0x" + bytes(body.sync_aggregate.sync_committee_bits).hex(),
                "sync_committee_signature": "0x" + bytes(body.sync_aggregate.sync_committee_signature).hex(),
            }

        if hasattr(body, "execution_payload"):
            payload = body.execution_payload
            result["execution_payload"] = {
                "parent_hash": "0x" + bytes(payload.parent_hash).hex(),
                "fee_recipient": "0x" + bytes(payload.fee_recipient).hex(),
                "state_root": "0x" + bytes(payload.state_root).hex(),
                "receipts_root": "0x" + bytes(payload.receipts_root).hex(),
                "logs_bloom": "0x" + bytes(payload.logs_bloom).hex(),
                "prev_randao": "0x" + bytes(payload.prev_randao).hex(),
                "block_number": str(payload.block_number),
                "gas_limit": str(payload.gas_limit),
                "gas_used": str(payload.gas_used),
                "timestamp": str(payload.timestamp),
                "extra_data": "0x" + bytes(payload.extra_data).hex(),
                "base_fee_per_gas": str(payload.base_fee_per_gas),
                "block_hash": "0x" + bytes(payload.block_hash).hex(),
                "transactions": ["0x" + bytes(tx).hex() for tx in payload.transactions],
            }
            if hasattr(payload, "withdrawals"):
                result["execution_payload"]["withdrawals"] = [
                    {
                        "index": str(w.index),
                        "validator_index": str(w.validator_index),
                        "address": "0x" + bytes(w.address).hex(),
                        "amount": str(w.amount),
                    }
                    for w in payload.withdrawals
                ]
            if hasattr(payload, "blob_gas_used"):
                result["execution_payload"]["blob_gas_used"] = str(payload.blob_gas_used)
                result["execution_payload"]["excess_blob_gas"] = str(payload.excess_blob_gas)

        if hasattr(body, "bls_to_execution_changes"):
            result["bls_to_execution_changes"] = []

        if hasattr(body, "blob_kzg_commitments"):
            result["blob_kzg_commitments"] = [
                "0x" + bytes(c).hex() for c in body.blob_kzg_commitments
            ]

        return result

    async def get_spec(self, request: web.Request) -> web.Response:
        """GET /eth/v1/config/spec"""
        spec = build_spec_response()
        return web.json_response({"data": spec})

    async def get_events(self, request: web.Request) -> web.StreamResponse:
        """GET /eth/v1/events - SSE endpoint for beacon events."""
        topics_param = request.query.get("topics", "")
        topics = set(t.strip() for t in topics_param.split(",") if t.strip())

        valid_topics = {"head", "block", "finalized_checkpoint", "chain_reorg"}
        requested_topics = topics & valid_topics if topics else valid_topics

        if not requested_topics:
            return web.json_response(
                {"message": "No valid topics specified"}, status=400
            )

        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)

        event_queue: asyncio.Queue = asyncio.Queue()
        self._event_subscribers.append(event_queue)

        logger.info(f"SSE client connected, topics: {requested_topics}")

        try:
            while True:
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=15.0)
                    event_type = event.get("event")
                    if event_type in requested_topics:
                        sse_data = f"event: {event_type}\ndata: {json.dumps(event['data'])}\n\n"
                        await response.write(sse_data.encode())
                except asyncio.TimeoutError:
                    await response.write(b": keepalive\n\n")
                except asyncio.CancelledError:
                    break
        except ConnectionResetError:
            logger.debug("SSE client disconnected")
        finally:
            if event_queue in self._event_subscribers:
                self._event_subscribers.remove(event_queue)
            logger.info("SSE client disconnected")

        return response

    async def _emit_events_loop(self) -> None:
        """Background task that checks for state changes and emits events."""
        self._last_head_slot = self.node.head_slot
        self._last_head_root = self.node.head_root
        if self.node.state:
            self._last_finalized_epoch = int(self.node.state.finalized_checkpoint.epoch)

        while True:
            try:
                await asyncio.sleep(0.5)

                current_slot = self.node.head_slot
                current_root = self.node.head_root

                if current_slot != self._last_head_slot or current_root != self._last_head_root:
                    if current_root:
                        head_event = {
                            "event": "head",
                            "data": {
                                "slot": str(current_slot),
                                "block": "0x" + current_root.hex(),
                                "state": "0x" + current_root.hex(),
                                "epoch_transition": current_slot % 8 == 0,
                                "execution_optimistic": False,
                            },
                        }
                        await self._broadcast_event(head_event)

                        block_event = {
                            "event": "block",
                            "data": {
                                "slot": str(current_slot),
                                "block": "0x" + current_root.hex(),
                                "execution_optimistic": False,
                            },
                        }
                        await self._broadcast_event(block_event)

                    self._last_head_slot = current_slot
                    self._last_head_root = current_root

                if self.node.state:
                    current_finalized_epoch = int(self.node.state.finalized_checkpoint.epoch)
                    if current_finalized_epoch != self._last_finalized_epoch:
                        finalized_root = bytes(self.node.state.finalized_checkpoint.root)
                        finalized_event = {
                            "event": "finalized_checkpoint",
                            "data": {
                                "block": "0x" + finalized_root.hex(),
                                "state": "0x" + finalized_root.hex(),
                                "epoch": str(current_finalized_epoch),
                                "execution_optimistic": False,
                            },
                        }
                        await self._broadcast_event(finalized_event)
                        self._last_finalized_epoch = current_finalized_epoch

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in event emitter loop: {e}")
                await asyncio.sleep(1.0)

    async def _broadcast_event(self, event: dict) -> None:
        """Broadcast an event to all SSE subscribers."""
        for subscriber_queue in self._event_subscribers:
            try:
                subscriber_queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("Event queue full for subscriber, dropping event")
