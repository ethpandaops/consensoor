"""Beacon API HTTP server."""

import asyncio
import json
import logging
import queue
import threading
from typing import Optional
from aiohttp import web

from .utils import get_local_ip, generate_peer_id
from .spec import build_spec_response
from .ssz_json import to_json, from_json
from ..spec.network_config import get_config as get_network_config
from ..version import get_cl_version, get_cl_commit, get_cl_client_version_info, CL_CLIENT_NAME

logger = logging.getLogger(__name__)


class BeaconAPI:
    """Simple Beacon API server."""

    def __init__(self, node, host: str = "0.0.0.0", port: int = 5052):
        self.node = node
        self.host = host
        self.port = port
        self.app = web.Application(client_max_size=256 * 1024 * 1024)
        self.runner: Optional[web.AppRunner] = None
        self._event_subscribers: list[asyncio.Queue] = []
        self._last_head_slot: int = 0
        self._last_head_root: Optional[bytes] = None
        self._last_finalized_epoch: int = 0
        self._event_emitter_task: Optional[asyncio.Task] = None
        self._event_drainer_task: Optional[asyncio.Task] = None
        # The HTTP server runs on its own thread + asyncio loop so the kurtosis
        # healthcheck and other API consumers stay responsive even when the
        # main loop is busy with state-transition / signing / engine RPCs.
        self._api_loop: Optional[asyncio.AbstractEventLoop] = None
        self._api_thread: Optional[threading.Thread] = None
        # Cross-loop event hand-off: emit_* methods are called from the main
        # loop and push events here; a coroutine on the API loop drains them.
        self._cross_loop_events: "queue.Queue[dict]" = queue.Queue()
        self._setup_routes()

    def _setup_routes(self):
        """Set up API routes."""
        self.app.router.add_get("/eth/v1/node/health", self.get_health)
        self.app.router.add_get("/eth/v1/node/version", self.get_version)
        self.app.router.add_get("/eth/v2/node/version", self.get_version_v2)
        self.app.router.add_get("/eth/v1/node/syncing", self.get_syncing)
        self.app.router.add_get("/eth/v1/node/identity", self.get_identity)
        self.app.router.add_get("/consensoor/v1/fast_confirmation", self.get_fast_confirmation)
        self.app.router.add_get("/consensoor/v1/custody", self.get_custody)
        self.app.router.add_post("/consensoor/v1/custody/backfill", self.post_custody_backfill)
        self.app.router.add_get("/eth/v1/node/peers", self.get_peers)
        self.app.router.add_get("/eth/v1/beacon/genesis", self.get_genesis)
        self.app.router.add_get("/eth/v1/beacon/states/{state_id}/root", self.get_state_root)
        self.app.router.add_get("/eth/v1/beacon/states/{state_id}/fork", self.get_state_fork)
        self.app.router.add_get("/eth/v1/beacon/states/{state_id}/finality_checkpoints", self.get_finality_checkpoints)
        self.app.router.add_get("/eth/v1/beacon/states/{state_id}/validators", self.get_validators)
        self.app.router.add_get("/eth/v1/beacon/states/{state_id}/validators/{validator_id}", self.get_validator)
        self.app.router.add_get("/eth/v1/beacon/states/{state_id}/validator_balances", self.get_validator_balances)
        self.app.router.add_get("/eth/v1/beacon/states/{state_id}/committees", self.get_committees)
        self.app.router.add_get("/eth/v1/beacon/states/{state_id}/sync_committees", self.get_sync_committees)
        self.app.router.add_get("/eth/v1/beacon/states/{state_id}/randao", self.get_randao)
        self.app.router.add_get("/eth/v1/beacon/states/{state_id}/proposer_lookahead", self.get_proposer_lookahead)
        self.app.router.add_get("/eth/v1/beacon/states/{state_id}/ptc", self.get_state_ptc)
        self.app.router.add_post("/eth/v1/beacon/states/{state_id}/builders", self.post_state_builders)
        self.app.router.add_get("/eth/v1/beacon/states/{state_id}/builder_pending_payments", self.get_builder_pending_payments)
        self.app.router.add_get("/eth/v1/beacon/states/{state_id}/builder_pending_withdrawals", self.get_builder_pending_withdrawals)
        self.app.router.add_get("/eth/v1/beacon/headers", self.get_headers)
        self.app.router.add_get("/eth/v1/beacon/headers/{block_id}", self.get_header)
        self.app.router.add_post("/eth/v2/beacon/blocks", self.post_block_v2)
        self.app.router.add_get("/eth/v2/beacon/blocks/{block_id}", self.get_block)
        self.app.router.add_get("/eth/v1/beacon/blocks/{block_id}/root", self.get_block_root)
        self.app.router.add_get("/eth/v1/beacon/blobs/{block_id}", self.get_blobs)
        self.app.router.add_get("/eth/v1/beacon/pool/payload_attestations", self.get_pool_payload_attestations)
        self.app.router.add_post("/eth/v1/beacon/pool/payload_attestations", self.post_pool_payload_attestations)
        self.app.router.add_get("/eth/v1/beacon/proposer_preferences", self.get_proposer_preferences)
        self.app.router.add_get("/eth/v2/debug/beacon/states/{state_id}", self.get_debug_state)
        self.app.router.add_get("/eth/v1/debug/fork_choice", self.get_fork_choice_v1)
        self.app.router.add_get("/eth/v2/debug/fork_choice", self.get_fork_choice_v2)
        self.app.router.add_get("/eth/v1/config/spec", self.get_spec)
        self.app.router.add_get("/eth/v1/config/fork_schedule", self.get_fork_schedule)
        self.app.router.add_get("/eth/v1/config/deposit_contract", self.get_deposit_contract)
        self.app.router.add_get("/eth/v1/beacon/execution_payload_envelope/{block_id}", self.get_execution_payload_envelope)
        self.app.router.add_get("/eth/v1/beacon/execution_payload_envelopes/{block_id}", self.get_execution_payload_envelope)
        self.app.router.add_post("/eth/v1/beacon/execution_payload_envelopes", self.post_execution_payload_envelopes)
        self.app.router.add_post("/eth/v1/beacon/execution_payload_bids", self.post_execution_payload_bids)
        self.app.router.add_post("/eth/v1/validator/proposer_preferences", self.post_proposer_preferences)
        self.app.router.add_post("/eth/v1/validator/builder_preferences", self.post_builder_preferences)
        self.app.router.add_get("/eth/v1/validator/duties/proposer/{epoch}", self.get_proposer_duties_v1)
        self.app.router.add_get("/eth/v2/validator/duties/proposer/{epoch}", self.get_proposer_duties_v2)
        self.app.router.add_post("/eth/v1/validator/duties/ptc/{epoch}", self.post_ptc_duties)
        self.app.router.add_get("/eth/v1/validator/payload_attestation_data", self.get_payload_attestation_data)
        self.app.router.add_post("/eth/v4/validator/blocks/{slot}", self.post_produce_block_v4)
        self.app.router.add_get(
            "/eth/v1/validator/execution_payload_envelopes/{slot}/{beacon_block_root}",
            self.get_validator_execution_payload_envelope,
        )
        self.app.router.add_get("/eth/v1/events", self.get_events)

    async def start(self):
        """Start the API server on its own thread + event loop."""
        # Capture the node's loop so POST handlers can schedule gossip
        # publishes / node-state mutations back onto it.
        self._node_loop = asyncio.get_running_loop()
        ready = threading.Event()
        startup_error: list[BaseException] = []

        def _run_api_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._api_loop = loop
            try:
                loop.run_until_complete(self._setup_on_api_loop())
            except BaseException as e:
                startup_error.append(e)
                ready.set()
                return
            ready.set()
            try:
                loop.run_forever()
            finally:
                try:
                    loop.run_until_complete(self._teardown_on_api_loop())
                finally:
                    loop.close()

        self._api_thread = threading.Thread(
            target=_run_api_loop, name="beacon-api", daemon=True
        )
        self._api_thread.start()
        ready.wait()
        if startup_error:
            raise startup_error[0]
        logger.info(f"Beacon API listening on {self.host}:{self.port}")

    async def _setup_on_api_loop(self) -> None:
        """Set up the aiohttp app + background tasks on the API loop."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()
        self._event_emitter_task = asyncio.create_task(self._emit_events_loop())
        self._event_drainer_task = asyncio.create_task(self._drain_cross_loop_events())

    async def _teardown_on_api_loop(self) -> None:
        for task in (self._event_emitter_task, self._event_drainer_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if self.runner is not None:
            await self.runner.cleanup()

    async def stop(self):
        """Stop the API server and join its thread."""
        loop = self._api_loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if self._api_thread is not None:
            self._api_thread.join(timeout=5.0)

    async def _drain_cross_loop_events(self) -> None:
        """Pull events posted from other threads and broadcast them on this loop."""
        while True:
            try:
                event = await asyncio.to_thread(self._cross_loop_events.get)
            except asyncio.CancelledError:
                break
            try:
                await self._broadcast_event(event)
            except Exception as e:
                logger.error(f"Error broadcasting cross-loop event: {e}")

    async def get_health(self, request: web.Request) -> web.Response:
        """GET /eth/v1/node/health"""
        return web.Response(status=200)

    async def get_version(self, request: web.Request) -> web.Response:
        """GET /eth/v1/node/version"""
        version = get_cl_version()
        commit = get_cl_commit()
        if commit:
            version_str = f"{CL_CLIENT_NAME}/v{version}/{commit[:8]}"
        else:
            version_str = f"{CL_CLIENT_NAME}/v{version}"
        return web.json_response({
            "data": {
                "version": version_str
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

    async def get_custody(self, request):
        """GET /consensoor/v1/custody — PeerDAS custody state of this node."""
        node = self.node
        das = getattr(node, "das", None)
        gossip = getattr(node, "beacon_gossip", None)
        task = getattr(node, "_custody_backfill_task", None)
        return web.json_response({
            "data": {
                "custody_group_count": int(gossip.custody_group_count) if gossip else None,
                "supernode": bool(node.config.supernode),
                "validator_custody_requirement": (
                    None if das is None or not node.validator_client or node.state is None
                    else int(__import__("consensoor.das", fromlist=["x"]).get_validators_custody_requirement(
                        node.state, node._attached_validator_indices()))
                ),
                "attached_validators": len(node._attached_validator_indices()),
                "custody_columns": sorted(das.custody_columns) if das else [],
                "custody_subnets": das.custody_subnets if das else [],
                "earliest_available_slot": int(getattr(node, "_earliest_available_slot", 0)),
                "backfill": None if task is None else ("running" if not task.done() else "finished"),
            }
        })

    async def post_custody_backfill(self, request):
        """POST /consensoor/v1/custody/backfill {"start_slot", "end_slot", "columns"?}
        — fetch custody columns from peers for a slot range (operator tool;
        the node does this itself when its custody widens)."""
        node = self.node
        das = getattr(node, "das", None)
        if das is None:
            return web.json_response({"code": 503, "message": "DAS not active"}, status=503)
        try:
            body = await request.json()
            start_slot = int(body["start_slot"])
            end_slot = int(body["end_slot"])
            columns = set(int(c) for c in body.get("columns", [])) or set(das.custody_columns)
        except Exception as e:
            return web.json_response({"code": 400, "message": f"bad request: {e}"}, status=400)
        if not columns or end_slot < start_slot:
            return web.json_response({"code": 400, "message": "nothing to backfill"}, status=400)
        node._start_custody_backfill_range(columns, start_slot, end_slot)
        return web.json_response({"data": {"columns": sorted(columns), "start_slot": start_slot, "end_slot": end_slot}})

    async def get_fast_confirmation(self, request):
        """GET /consensoor/v1/fast_confirmation — latest FCR result."""
        node = self.node
        root = getattr(node, "confirmed_root", None)
        if not getattr(node.config, "fast_confirmation", True):
            return web.json_response({"code": 501, "message": "fast confirmation disabled (--no-fast-confirmation)"}, status=501)
        if root is None or node.fc_store is None or root not in node.fc_store.blocks:
            return web.json_response({"code": 503, "message": "fast confirmation not available yet"}, status=503)
        from ..spec import fast_confirmation as fcr
        block = node.fc_store.blocks[root]
        return web.json_response({
            "data": {
                "confirmed_root": "0x" + root.hex(),
                "confirmed_slot": str(int(block.slot)),
                "safe_execution_block_hash": "0x" + fcr.get_safe_execution_block_hash(node.fcr_store).hex(),
                "head_root": ("0x" + node.head_root.hex()) if node.head_root else None,
            }
        })

    async def get_identity(self, request: web.Request) -> web.Response:
        """GET /eth/v1/node/identity"""
        local_ip = get_local_ip()
        listen_port = self.node.config.listen_port

        host = self.node.beacon_gossip._host if self.node.beacon_gossip else None
        if self.node.beacon_gossip and self.node.beacon_gossip.peer_id:
            peer_id = self.node.beacon_gossip.peer_id
            enr = host.enr or "" if host else ""
            multiaddr = host.multiaddr if host else None
        else:
            peer_id = generate_peer_id(f"consensoor-{local_ip}-{self.port}")
            enr = ""
            multiaddr = None

        # Kurtosis extracts p2p_addresses[0] during service bring-up, so always
        # return a usable multiaddr — fall back to a synthesised one when the
        # rust libp2p host hasn't published its listen addrs yet.
        if not multiaddr:
            multiaddr = f"/ip4/{local_ip}/tcp/{listen_port}/p2p/{peer_id}"
        p2p_addresses = [multiaddr]

        # MetaData v3 mirrors what we advertise on the `/eth2/.../meta_data/3`
        # RPC. Pull the live values out of the P2P host so dora and other
        # monitors see the current seq_number and the post-bump
        # custody_group_count instead of the static "1" + hardcoded attnets.
        if host is not None:
            meta = {
                "seq_number": str(host._our_metadata_seq),
                "attnets": "0x" + host.config.attnets.hex(),
                "syncnets": "0x" + host.config.syncnets.hex(),
                "custody_group_count": str(host.config.custody_group_count),
            }
        else:
            meta = {
                "seq_number": "0",
                "attnets": "0xffffffffffffffff",
                "syncnets": "0x0f",
                "custody_group_count": "4",
            }

        return web.json_response({
            "data": {
                "peer_id": peer_id,
                "enr": enr,
                "p2p_addresses": p2p_addresses,
                "discovery_addresses": [
                    f"/ip4/{local_ip}/udp/{listen_port}/p2p/{peer_id}"
                ],
                "metadata": meta,
            }
        })

    async def get_peers(self, request: web.Request) -> web.Response:
        """GET /eth/v1/node/peers

        Emits the base `Peer` schema plus the additive PR #606 fields
        we track: `agent_version`, `score`, `downscore_reasons`. All
        three are spec-OPTIONAL and omitted when empty/unknown rather
        than emitted as nulls. `disconnect_reason` is also defined by
        PR #606 but is gated by the spec to `disconnected`/`disconnecting`
        peers only — this endpoint currently lists only `connected`
        peers, so it never applies here.
        """
        peers = []

        if self.node.beacon_gossip and hasattr(self.node.beacon_gossip, '_host'):
            p2p_host = self.node.beacon_gossip._host
            if p2p_host and hasattr(p2p_host, 'connected_peers'):
                for peer_info in p2p_host.connected_peers():
                    peer_id = peer_info.get("peer_id", "")
                    addrs = peer_info.get("addrs", [])
                    direction = peer_info.get("direction", "unknown")
                    enr = peer_info.get("enr", "") or ""
                    agent_version = peer_info.get("agent_version", "") or ""
                    score = peer_info.get("score")
                    downscore_reasons = peer_info.get("downscore_reasons") or []
                    entry = {
                        "peer_id": peer_id,
                        "enr": enr,
                        "last_seen_p2p_address": addrs[0] if addrs else "",
                        "state": "connected",
                        "direction": direction,
                    }
                    if agent_version:
                        entry["agent_version"] = agent_version
                    if score is not None:
                        entry["score"] = score
                    if downscore_reasons:
                        entry["downscore_reasons"] = downscore_reasons
                    peers.append(entry)

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
        state = self._resolve_state_id(state_id)
        if state is None:
            return web.json_response({"message": "State not found"}, status=404)

        from ..crypto import hash_tree_root
        state_root = hash_tree_root(state)
        return web.json_response({
            "execution_optimistic": False,
            "finalized": False,
            "data": {"root": "0x" + state_root.hex()}
        })

    async def get_state_fork(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/states/{state_id}/fork"""
        state_id = request.match_info["state_id"]
        state = self._resolve_state(state_id)
        if state is None:
            return web.json_response({"message": "State not found"}, status=404)

        return web.json_response({
            "execution_optimistic": False,
            "finalized": False,
            "data": {
                "previous_version": "0x" + bytes(state.fork.previous_version).hex(),
                "current_version": "0x" + bytes(state.fork.current_version).hex(),
                "epoch": str(state.fork.epoch),
            }
        })

    def _resolve_state(self, state_id: str):
        """Resolve state_id to a state object."""
        if state_id == "head":
            return self.node.state
        if state_id == "finalized":
            return self.node.state
        if state_id == "justified":
            return self.node.state
        if state_id == "genesis":
            return self.node.store.get_state_by_slot(0)
        if state_id.startswith("0x"):
            root = bytes.fromhex(state_id[2:])
            return self.node.store.get_state(root)
        try:
            slot = int(state_id)
            return self.node.store.get_state_by_slot(slot)
        except ValueError:
            pass
        return None

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

    async def get_validators(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/states/{state_id}/validators"""
        state_id = request.match_info["state_id"]
        if state_id not in ("head", "finalized", "justified", "genesis") and self.node.state:
            pass  # Could handle slot/root lookups

        if not self.node.state:
            return web.json_response({"message": "State not found"}, status=404)

        # Parse query params for filtering
        id_param = request.query.getall("id", [])
        status_param = request.query.getall("status", [])

        validators_data = []
        for i, validator in enumerate(self.node.state.validators):
            # Filter by id if specified
            if id_param:
                if str(i) not in id_param and f"0x{bytes(validator.pubkey).hex()}" not in id_param:
                    continue

            # Determine validator status
            balance = int(self.node.state.balances[i])
            current_epoch = int(self.node.state.slot) // 8  # SLOTS_PER_EPOCH for minimal

            if int(validator.activation_epoch) > current_epoch:
                status = "pending_queued"
            elif int(validator.exit_epoch) <= current_epoch:
                if int(validator.withdrawable_epoch) <= current_epoch:
                    status = "withdrawal_done" if balance == 0 else "withdrawal_possible"
                else:
                    status = "exited_slashed" if validator.slashed else "exited_unslashed"
            elif validator.slashed:
                status = "active_slashed"
            elif int(validator.exit_epoch) < 2**64 - 1:
                status = "active_exiting"
            else:
                status = "active_ongoing"

            # Filter by status if specified
            # Supports both exact status (e.g. "active_ongoing") and prefix (e.g. "active")
            if status_param:
                match = False
                for sp in status_param:
                    if status == sp or status.startswith(sp + "_"):
                        match = True
                        break
                if not match:
                    continue

            validators_data.append({
                "index": str(i),
                "balance": str(balance),
                "status": status,
                "validator": {
                    "pubkey": "0x" + bytes(validator.pubkey).hex(),
                    "withdrawal_credentials": "0x" + bytes(validator.withdrawal_credentials).hex(),
                    "effective_balance": str(validator.effective_balance),
                    "slashed": validator.slashed,
                    "activation_eligibility_epoch": str(validator.activation_eligibility_epoch),
                    "activation_epoch": str(validator.activation_epoch),
                    "exit_epoch": str(validator.exit_epoch),
                    "withdrawable_epoch": str(validator.withdrawable_epoch),
                }
            })

        return web.json_response({
            "execution_optimistic": False,
            "finalized": False,
            "data": validators_data
        })

    async def get_validator(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/states/{state_id}/validators/{validator_id}"""
        validator_id = request.match_info["validator_id"]

        if not self.node.state:
            return web.json_response({"message": "State not found"}, status=404)

        # Find validator by index or pubkey
        validator_index = None
        if validator_id.startswith("0x"):
            # Search by pubkey
            pubkey_bytes = bytes.fromhex(validator_id[2:])
            for i, v in enumerate(self.node.state.validators):
                if bytes(v.pubkey) == pubkey_bytes:
                    validator_index = i
                    break
        else:
            try:
                validator_index = int(validator_id)
            except ValueError:
                return web.json_response({"message": "Invalid validator id"}, status=400)

        if validator_index is None or validator_index >= len(self.node.state.validators):
            return web.json_response({"message": "Validator not found"}, status=404)

        validator = self.node.state.validators[validator_index]
        balance = int(self.node.state.balances[validator_index])
        current_epoch = int(self.node.state.slot) // 8

        if int(validator.activation_epoch) > current_epoch:
            status = "pending_queued"
        elif int(validator.exit_epoch) <= current_epoch:
            if int(validator.withdrawable_epoch) <= current_epoch:
                status = "withdrawal_done" if balance == 0 else "withdrawal_possible"
            else:
                status = "exited_slashed" if validator.slashed else "exited_unslashed"
        elif validator.slashed:
            status = "active_slashed"
        elif int(validator.exit_epoch) < 2**64 - 1:
            status = "active_exiting"
        else:
            status = "active_ongoing"

        return web.json_response({
            "execution_optimistic": False,
            "finalized": False,
            "data": {
                "index": str(validator_index),
                "balance": str(balance),
                "status": status,
                "validator": {
                    "pubkey": "0x" + bytes(validator.pubkey).hex(),
                    "withdrawal_credentials": "0x" + bytes(validator.withdrawal_credentials).hex(),
                    "effective_balance": str(validator.effective_balance),
                    "slashed": validator.slashed,
                    "activation_eligibility_epoch": str(validator.activation_eligibility_epoch),
                    "activation_epoch": str(validator.activation_epoch),
                    "exit_epoch": str(validator.exit_epoch),
                    "withdrawable_epoch": str(validator.withdrawable_epoch),
                }
            }
        })

    async def get_validator_balances(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/states/{state_id}/validator_balances"""

        if not self.node.state:
            return web.json_response({"message": "State not found"}, status=404)

        # Parse query params for filtering
        id_param = request.query.getall("id", [])

        balances_data = []
        for i, balance in enumerate(self.node.state.balances):
            # Filter by id if specified
            if id_param:
                validator = self.node.state.validators[i]
                if str(i) not in id_param and f"0x{bytes(validator.pubkey).hex()}" not in id_param:
                    continue

            balances_data.append({
                "index": str(i),
                "balance": str(balance),
            })

        return web.json_response({
            "execution_optimistic": False,
            "finalized": False,
            "data": balances_data
        })

    async def get_committees(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/states/{state_id}/committees"""
        state_id = request.match_info["state_id"]
        state = self._resolve_state(state_id)
        if state is None:
            return web.json_response({"message": "State not found"}, status=404)

        from ..spec.constants import SLOTS_PER_EPOCH
        from ..spec.state_transition.helpers.beacon_committee import get_beacon_committee

        epoch_param = request.query.get("epoch")
        index_param = request.query.get("index")
        slot_param = request.query.get("slot")

        current_epoch = int(state.slot) // SLOTS_PER_EPOCH()
        target_epoch = int(epoch_param) if epoch_param else current_epoch

        committees_data = []
        start_slot = target_epoch * SLOTS_PER_EPOCH()
        end_slot = start_slot + SLOTS_PER_EPOCH()

        for slot in range(start_slot, end_slot):
            if slot_param and int(slot_param) != slot:
                continue

            committee_count = max(1, len(state.validators) // 32 // SLOTS_PER_EPOCH())
            for committee_index in range(committee_count):
                if index_param and int(index_param) != committee_index:
                    continue

                try:
                    committee = get_beacon_committee(state, slot, committee_index)
                    committees_data.append({
                        "index": str(committee_index),
                        "slot": str(slot),
                        "validators": [str(v) for v in committee],
                    })
                except Exception:
                    pass

        return web.json_response({
            "execution_optimistic": False,
            "finalized": False,
            "data": committees_data
        })

    async def get_sync_committees(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/states/{state_id}/sync_committees"""
        state_id = request.match_info["state_id"]
        state = self._resolve_state(state_id)
        if state is None:
            return web.json_response({"message": "State not found"}, status=404)

        if not hasattr(state, "current_sync_committee"):
            return web.json_response({"message": "Sync committees not available for this fork"}, status=400)

        from ..spec.constants import SYNC_COMMITTEE_SIZE, SYNC_COMMITTEE_SUBNET_COUNT

        sync_committee = state.current_sync_committee
        all_pubkeys = {bytes(v.pubkey): i for i, v in enumerate(state.validators)}

        validators = []
        for pubkey in sync_committee.pubkeys:
            pk_bytes = bytes(pubkey)
            if pk_bytes in all_pubkeys:
                validators.append(str(all_pubkeys[pk_bytes]))
            else:
                validators.append("0")

        subcommittee_size = SYNC_COMMITTEE_SIZE() // SYNC_COMMITTEE_SUBNET_COUNT
        validator_aggregates = []
        for i in range(SYNC_COMMITTEE_SUBNET_COUNT):
            start = i * subcommittee_size
            end = start + subcommittee_size
            validator_aggregates.append(validators[start:end])

        return web.json_response({
            "execution_optimistic": False,
            "finalized": False,
            "data": {
                "validators": validators,
                "validator_aggregates": validator_aggregates,
            }
        })

    async def get_randao(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/states/{state_id}/randao"""
        state_id = request.match_info["state_id"]
        state = self._resolve_state(state_id)
        if state is None:
            return web.json_response({"message": "State not found"}, status=404)

        from ..spec.constants import SLOTS_PER_EPOCH, EPOCHS_PER_HISTORICAL_VECTOR

        epoch_param = request.query.get("epoch")
        current_epoch = int(state.slot) // SLOTS_PER_EPOCH()
        target_epoch = int(epoch_param) if epoch_param else current_epoch

        randao_index = target_epoch % EPOCHS_PER_HISTORICAL_VECTOR()
        randao_mix = bytes(state.randao_mixes[randao_index])

        return web.json_response({
            "execution_optimistic": False,
            "finalized": False,
            "data": {
                "randao": "0x" + randao_mix.hex(),
            }
        })

    async def get_headers(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/headers

        Returns the head header when called without parameters.
        """
        if not self.node.state or not self.node.head_root:
            return web.json_response({"data": []})

        header = self.node.state.latest_block_header
        return web.json_response({
            "data": [{
                "root": "0x" + self.node.head_root.hex(),
                "canonical": True,
                "header": {
                    "message": {
                        "slot": str(header.slot),
                        "proposer_index": str(header.proposer_index),
                        "parent_root": "0x" + bytes(header.parent_root).hex(),
                        "state_root": "0x" + bytes(header.state_root).hex(),
                        "body_root": "0x" + bytes(header.body_root).hex(),
                    },
                    "signature": "0x" + "00" * 96,
                }
            }]
        })

    def _resolve_block_id(self, block_id: str) -> tuple[Optional[bytes], Optional[object]]:
        """Resolve a block_id to (root, signed_block).

        Supports: "head", "genesis", "finalized", "0x..." (root), slot number.
        Returns (None, None) if not found.
        """
        if block_id == "head":
            if self.node.head_root:
                block = self.node.store.get_block(self.node.head_root)
                return self.node.head_root, block
            return None, None

        if block_id == "genesis":
            block = self.node.store.get_block_by_slot(0)
            if block:
                from ..crypto import hash_tree_root
                msg = block.message if hasattr(block, "message") else block
                root = hash_tree_root(msg)
                return root, block
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
            block = self.node.store.get_block_by_slot(slot)
            if block:
                from ..crypto import hash_tree_root
                msg = block.message if hasattr(block, "message") else block
                root = hash_tree_root(msg)
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

        # For post-Electra blocks, check slot against fork epochs
        if hasattr(body, "execution_requests"):
            from ..spec.network_config import get_config
            from ..spec.constants import SLOTS_PER_EPOCH
            config = get_config()
            slot = int(block.slot)
            epoch = slot // SLOTS_PER_EPOCH()

            logger.debug(f"Block version detection: slot={slot}, epoch={epoch}, fulu_fork_epoch={config.fulu_fork_epoch}")

            # Check if we're in GLOAS epoch (return gloas for GLOAS blocks)
            if hasattr(config, 'gloas_fork_epoch') and epoch >= config.gloas_fork_epoch:
                logger.debug(f"Returning gloas (epoch {epoch} >= {config.gloas_fork_epoch})")
                return "gloas"

            # Check Fulu epoch
            if hasattr(config, 'fulu_fork_epoch') and epoch >= config.fulu_fork_epoch:
                logger.debug(f"Returning fulu (epoch {epoch} >= fulu_fork_epoch {config.fulu_fork_epoch})")
                return "fulu"

            logger.debug(f"Returning electra (epoch {epoch})")
            return "electra"

        if hasattr(body, "signed_execution_payload_header") or hasattr(body, "signed_execution_payload_bid"):
            if hasattr(body, "signed_execution_payload_bid") and hasattr(
                body.signed_execution_payload_bid.message, "inclusion_list_bits"
            ):
                from ..spec.network_config import get_config
                from ..spec.constants import SLOTS_PER_EPOCH
                return get_config().fork_name_at_epoch(int(block.slot) // SLOTS_PER_EPOCH())
            return "gloas"
        if hasattr(body, "blob_kzg_commitments"):
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

        if root is None or signed_block is None:
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
        logger.info(f"get_block request: block_id={block_id}")
        root, signed_block = self._resolve_block_id(block_id)

        if root is None or signed_block is None:
            logger.warning(f"Block not found: block_id={block_id}")
            return web.json_response({"message": "Block not found"}, status=404)

        accept = request.headers.get("Accept", "application/json")

        if "application/octet-stream" in accept:
            ssz_bytes = signed_block.encode_bytes()
            version = self._get_block_version(signed_block)
            logger.info(f"Returning block SSZ: block_id={block_id}, version={version}, size={len(ssz_bytes)}")
            return web.Response(
                body=ssz_bytes,
                content_type="application/octet-stream",
                headers={"Eth-Consensus-Version": version},
            )

        version = self._get_block_version(signed_block)
        return web.json_response(
            {
                "version": version,
                "execution_optimistic": False,
                "finalized": False,
                "data": to_json(signed_block),
            },
            headers={"Eth-Consensus-Version": version},
        )

    async def get_block_root(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/blocks/{block_id}/root"""
        block_id = request.match_info["block_id"]
        root, signed_block = self._resolve_block_id(block_id)

        if root is None:
            return web.json_response({"message": "Block not found"}, status=404)

        return web.json_response({
            "execution_optimistic": False,
            "finalized": False,
            "data": {
                "root": "0x" + root.hex(),
            }
        })

    async def get_execution_payload_envelope(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/execution_payload_envelope/{block_id}"""
        block_id = request.match_info["block_id"]
        logger.info(f"get_execution_payload_envelope request: block_id={block_id}")

        # Resolve block_id to root
        if block_id.startswith("0x"):
            try:
                root = bytes.fromhex(block_id[2:])
            except ValueError:
                return web.json_response({"message": "Invalid block root"}, status=400)
        elif block_id == "head":
            root = self.node.head_root
        else:
            try:
                slot = int(block_id)
                block = self.node.store.get_block_by_slot(slot)
                if block:
                    from ..crypto import hash_tree_root
                    msg = block.message if hasattr(block, "message") else block
                    root = hash_tree_root(msg)
                else:
                    return web.json_response({"message": "Block not found"}, status=404)
            except ValueError:
                return web.json_response({"message": "Invalid block id"}, status=400)

        if root is None:
            return web.json_response({"message": "Block not found"}, status=404)

        signed_envelope = self.node.store.get_payload(root)
        if signed_envelope is None:
            return web.json_response({"message": "Execution payload envelope not found"}, status=404)

        return self._versioned_response(
            request,
            self._fork_at_slot(self.node._envelope_slot(signed_envelope.message) or self.node.head_slot),
            signed_envelope.encode_bytes(),
            to_json(signed_envelope),
            execution_optimistic=False,
            finalized=False,
        )

    async def _run_on_node_loop(self, coro):
        """Run a coroutine on the node's event loop and await its result."""
        fut = asyncio.run_coroutine_threadsafe(coro, self._node_loop)
        return await asyncio.wrap_future(fut)

    async def post_execution_payload_envelopes(self, request: web.Request) -> web.Response:
        """POST /eth/v1/beacon/execution_payload_envelopes

        Accepts a SignedExecutionPayloadEnvelope, or with
        `Eth-Blob-Data-Included: true` a SignedExecutionPayloadEnvelopeContents,
        broadcasts it on the `execution_payload` topic together with the data
        column sidecars of its blobs, and integrates it locally.
        """
        from ..spec.types.base import Container, List
        from ..spec.types.deneb import Blob
        from ..spec.types import KZGProof
        from ..spec.types.gloas import SignedExecutionPayloadEnvelope as envelope_type
        from ..spec.constants import FIELD_ELEMENTS_PER_EXT_BLOB, MAX_BLOB_COMMITMENTS_PER_BLOCK

        blob_data_included = request.headers.get("Eth-Blob-Data-Included", "false").lower() == "true"
        blobs_bundle = None
        try:
            if self._is_ssz_body(request):
                raw = await request.read()
                if blob_data_included:
                    class SignedExecutionPayloadEnvelopeContents(Container):
                        signed_execution_payload_envelope: envelope_type
                        kzg_proofs: List[KZGProof, FIELD_ELEMENTS_PER_EXT_BLOB * MAX_BLOB_COMMITMENTS_PER_BLOCK]
                        blobs: List[Blob, MAX_BLOB_COMMITMENTS_PER_BLOCK]

                    contents = SignedExecutionPayloadEnvelopeContents.decode_bytes(raw)
                    signed = contents.signed_execution_payload_envelope
                    blobs_bundle = {"blobs": ["0x" + bytes(b).hex() for b in contents.blobs]}
                else:
                    signed = envelope_type.decode_bytes(raw)
            else:
                body = await request.json()
                if blob_data_included or "signed_execution_payload_envelope" in body:
                    signed = from_json(envelope_type, body["signed_execution_payload_envelope"])
                    blobs_bundle = {"blobs": list(body.get("blobs", []))}
                else:
                    signed = from_json(envelope_type, body)
        except Exception as e:
            return self._error(400, f"Invalid signed execution payload envelope: {e}")

        envelope = signed.message
        beacon_block_root = bytes(envelope.beacon_block_root)
        slot = self.node._envelope_slot(envelope)
        if blobs_bundle is None and slot is not None:
            cached = self.node.api_produced_payload(slot, beacon_block_root)
            if cached is not None:
                blobs_bundle = cached["blobs_bundle"]

        ssz_bytes = signed.encode_bytes()
        try:
            if self.node.beacon_gossip:
                await self._run_on_node_loop(
                    self.node.beacon_gossip.publish_execution_payload(ssz_bytes)
                )
                if blobs_bundle and blobs_bundle.get("blobs") and slot is not None:
                    await self._run_on_node_loop(
                        self.node._publish_payload_data_columns(beacon_block_root, slot, blobs_bundle)
                    )
        except Exception as e:
            logger.error(f"Envelope gossip publish failed: {e}")
            return self._error(500, f"Broadcast failed: {e}")

        try:
            await self._run_on_node_loop(
                self.node._on_p2p_execution_payload(ssz_bytes, "beacon-api")
            )
        except Exception as e:
            logger.warning(f"Envelope local integration failed: {e}")
            return web.Response(status=202)
        return web.Response(status=200)

    async def post_execution_payload_bids(self, request: web.Request) -> web.Response:
        """POST /eth/v1/beacon/execution_payload_bids

        Accepts a SignedExecutionPayloadBid and broadcasts it on the
        `execution_payload_bid` gossip topic.
        """
        version = request.headers.get("Eth-Consensus-Version", "gloas").lower()
        bid_type = self._signed_bid_type(version)
        try:
            if self._is_ssz_body(request):
                signed = bid_type.decode_bytes(await request.read())
            else:
                signed = from_json(bid_type, await request.json())
        except Exception as e:
            return self._error(400, f"Invalid signed execution payload bid: {e}")

        try:
            if self.node.beacon_gossip:
                await self._run_on_node_loop(
                    self.node.beacon_gossip.publish_execution_payload_bid(signed.encode_bytes())
                )
        except Exception as e:
            logger.error(f"Bid gossip publish failed: {e}")
            return self._error(500, f"Broadcast failed: {e}")

        await self.emit_execution_payload_bid(signed)
        return web.Response(status=200)

    async def post_proposer_preferences(self, request: web.Request) -> web.Response:
        """POST /eth/v1/validator/proposer_preferences

        Accepts an array of SignedProposerPreferences, validates each
        against the head state, stores them, and publishes them on the
        `proposer_preferences` gossip topic.
        """
        from ..spec.types.gloas import SignedProposerPreferences

        try:
            if self._is_ssz_body(request):
                raw = await request.read()
                size = SignedProposerPreferences.type_byte_length()
                if len(raw) % size:
                    raise ValueError("SSZ body is not a list of SignedProposerPreferences")
                items = [SignedProposerPreferences.decode_bytes(raw[i:i + size]) for i in range(0, len(raw), size)]
            else:
                body = await request.json()
                if not isinstance(body, list):
                    raise ValueError("expected an array of SignedProposerPreferences")
                items = [from_json(SignedProposerPreferences, d) for d in body]
        except Exception as e:
            return self._error(400, f"Invalid signed proposer preferences: {e}")

        failures = []
        for i, signed in enumerate(items):
            try:
                ok, reason = await self._run_on_node_loop(
                    self.node.process_proposer_preferences(signed)
                )
                if not ok and reason == "reject":
                    failures.append({"index": i, "message": "rejected"})
                    continue
                if self.node.beacon_gossip and hasattr(
                    self.node.beacon_gossip, "publish_proposer_preferences"
                ):
                    await self._run_on_node_loop(
                        self.node.beacon_gossip.publish_proposer_preferences(
                            signed.encode_bytes()
                        )
                    )
            except Exception as e:
                failures.append({"index": i, "message": str(e)})

        if failures:
            return self._error(400, "Errors with one or more signed proposer preferences", failures=failures)
        return web.Response(status=200)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _error(status: int, message: str, **extra) -> web.Response:
        return web.json_response({"code": status, "message": message, **extra}, status=status)

    @staticmethod
    def _wants_ssz(request: web.Request) -> bool:
        return "application/octet-stream" in request.headers.get("Accept", "")

    @staticmethod
    def _is_ssz_body(request: web.Request) -> bool:
        return "application/octet-stream" in request.headers.get("Content-Type", "")

    @staticmethod
    def _fork_at_slot(slot: int) -> str:
        return get_network_config().fork_name_at_slot(int(slot))

    @staticmethod
    def _signed_bid_type(version: str):
        if version == "heze":
            from ..spec.types.heze import SignedExecutionPayloadBid
        else:
            from ..spec.types.gloas import SignedExecutionPayloadBid
        return SignedExecutionPayloadBid

    @staticmethod
    def _signed_block_type(version: str):
        from ..spec.types import phase0, altair, bellatrix, capella, deneb, electra, gloas, heze

        types = {
            "phase0": phase0.SignedPhase0BeaconBlock,
            "altair": altair.SignedAltairBeaconBlock,
            "bellatrix": bellatrix.SignedBellatrixBeaconBlock,
            "capella": capella.SignedCapellaBeaconBlock,
            "deneb": deneb.SignedDenebBeaconBlock,
            "electra": electra.SignedElectraBeaconBlock,
            "fulu": electra.SignedElectraBeaconBlock,
            "gloas": gloas.SignedBeaconBlock,
            "heze": heze.SignedBeaconBlock,
        }
        if version not in types:
            raise ValueError(f"unsupported consensus version {version!r}")
        return types[version]

    def _versioned_response(self, request, version: str, ssz_bytes: bytes, data, **extra) -> web.Response:
        headers = {"Eth-Consensus-Version": version}
        if self._wants_ssz(request):
            return web.Response(body=ssz_bytes, content_type="application/octet-stream", headers=headers)
        return web.json_response({"version": version, **extra, "data": data}, headers=headers)

    @staticmethod
    def _block_root_at_slot(state, slot: int) -> bytes:
        """get_block_root_at_slot, using the genesis block on underflow and the
        state's latest block for slots at or past the state's slot."""
        from ..crypto import hash_tree_root
        from ..spec.constants import SLOTS_PER_HISTORICAL_ROOT

        slot = max(0, int(slot))
        if slot < int(state.slot):
            return bytes(state.block_roots[slot % SLOTS_PER_HISTORICAL_ROOT()])
        header = state.latest_block_header.copy()
        if bytes(header.state_root) == b"\x00" * 32:
            header.state_root = hash_tree_root(state)
        return hash_tree_root(header)

    def _dependent_root(self, state, epoch: int) -> bytes:
        from ..spec.constants import SLOTS_PER_EPOCH
        return self._block_root_at_slot(state, epoch * SLOTS_PER_EPOCH() - 1)

    async def _run_on_node_loop_fn(self, fn, *args):
        async def _call():
            return fn(*args)
        return await self._run_on_node_loop(_call())

    # ------------------------------------------------------------------ node

    async def get_version_v2(self, request: web.Request) -> web.Response:
        """GET /eth/v2/node/version"""
        data = {"beacon_node": get_cl_client_version_info()}
        el_info = getattr(self.node.config, "_el_client_info", None)
        if el_info and all(k in el_info for k in ("code", "name", "version", "commit")):
            commit = str(el_info["commit"])
            if not commit.startswith("0x"):
                commit = "0x" + commit
            data["execution_client"] = {
                "code": el_info["code"],
                "name": el_info["name"],
                "version": el_info["version"],
                "commit": commit[:10],
            }
        return web.json_response({"data": data})

    # ------------------------------------------------------------------ state

    async def get_proposer_lookahead(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/states/{state_id}/proposer_lookahead"""
        state = self._resolve_state_id(request.match_info["state_id"])
        if state is None:
            return self._error(404, "State not found")
        if not hasattr(state, "proposer_lookahead"):
            return self._error(400, "State is prior to Fulu")
        return self._versioned_response(
            request,
            self._get_state_version(state),
            state.proposer_lookahead.encode_bytes(),
            [str(int(v)) for v in state.proposer_lookahead],
            execution_optimistic=False,
            finalized=False,
        )

    async def get_state_ptc(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/states/{state_id}/ptc"""
        from ..spec.constants import SLOTS_PER_EPOCH, MIN_SEED_LOOKAHEAD, PTC_SIZE
        from ..spec.state_transition.helpers.ptc import get_ptc
        from ..spec.types.base import Vector, uint64

        state = self._resolve_state_id(request.match_info["state_id"])
        if state is None:
            return self._error(404, "State not found")
        if not hasattr(state, "builders"):
            return self._error(400, "State is prior to Gloas")
        try:
            slot = int(request.query.get("slot", int(state.slot)))
        except ValueError:
            return self._error(400, "Invalid slot parameter")
        epoch = slot // SLOTS_PER_EPOCH()
        state_epoch = int(state.slot) // SLOTS_PER_EPOCH()
        if epoch < get_network_config().gloas_fork_epoch:
            return self._error(400, "Slot is prior to Gloas")
        if not max(0, state_epoch - 1) <= epoch <= state_epoch + MIN_SEED_LOOKAHEAD:
            return self._error(400, "Slot is outside the PTC window of the state")
        ptc = [int(v) for v in get_ptc(state, slot)]
        if self._wants_ssz(request):
            return web.Response(
                body=Vector[uint64, PTC_SIZE()](*ptc).encode_bytes(),
                content_type="application/octet-stream",
            )
        return web.json_response({
            "execution_optimistic": False,
            "finalized": False,
            "data": {"slot": str(slot), "validators": [str(v) for v in ptc]},
        })

    async def post_state_builders(self, request: web.Request) -> web.Response:
        """POST /eth/v1/beacon/states/{state_id}/builders"""
        from ..spec.constants import FAR_FUTURE_EPOCH
        from ..spec.state_transition.helpers.predicates import is_active_builder

        state = self._resolve_state_id(request.match_info["state_id"])
        if state is None:
            return self._error(404, "State not found")
        if not hasattr(state, "builders"):
            return self._error(400, "State is prior to Gloas")
        ids, statuses = [], []
        if request.can_read_body:
            try:
                body = await request.json()
                if body is not None:
                    ids = body.get("ids") or []
                    statuses = body.get("statuses") or []
            except Exception as e:
                return self._error(400, f"Invalid request body: {e}")
        wanted_indices, wanted_pubkeys = set(), set()
        for i in ids:
            if isinstance(i, str) and i.startswith("0x"):
                wanted_pubkeys.add(i.lower())
            else:
                try:
                    wanted_indices.add(int(i))
                except (TypeError, ValueError):
                    return self._error(400, f"Invalid builder id: {i}")

        data = []
        for index, builder in enumerate(state.builders):
            pubkey = "0x" + bytes(builder.pubkey).hex()
            if ids and index not in wanted_indices and pubkey not in wanted_pubkeys:
                continue
            if int(builder.withdrawable_epoch) != FAR_FUTURE_EPOCH:
                status = "exited"
            elif is_active_builder(state, index):
                status = "active"
            else:
                status = "pending"
            if statuses and status not in statuses:
                continue
            data.append({"index": str(index), "status": status, "builder": to_json(builder)})
        return web.json_response({"execution_optimistic": False, "finalized": False, "data": data})

    async def _get_state_field(self, request: web.Request, field: str) -> web.Response:
        state = self._resolve_state_id(request.match_info["state_id"])
        if state is None:
            return self._error(404, "State not found")
        if not hasattr(state, field):
            return self._error(400, "State is prior to Gloas")
        value = getattr(state, field)
        return self._versioned_response(
            request,
            self._get_state_version(state),
            value.encode_bytes(),
            to_json(value),
            execution_optimistic=False,
            finalized=False,
        )

    async def get_builder_pending_payments(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/states/{state_id}/builder_pending_payments"""
        return await self._get_state_field(request, "builder_pending_payments")

    async def get_builder_pending_withdrawals(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/states/{state_id}/builder_pending_withdrawals"""
        return await self._get_state_field(request, "builder_pending_withdrawals")

    # ------------------------------------------------------------------ validator duties

    async def _proposer_duties(self, request: web.Request, dependent_offset: int) -> web.Response:
        from ..spec.constants import SLOTS_PER_EPOCH, MIN_SEED_LOOKAHEAD

        try:
            epoch = int(request.match_info["epoch"])
            if epoch < 0:
                raise ValueError
        except ValueError:
            return self._error(400, f"Invalid epoch: {request.match_info['epoch']}")
        state = self.node.state
        if state is None:
            return self._error(503, "Beacon node is currently syncing")
        if not hasattr(state, "proposer_lookahead"):
            return self._error(400, "Proposer duties require a Fulu or later state")
        state_epoch = int(state.slot) // SLOTS_PER_EPOCH()
        if not 0 <= epoch - state_epoch <= MIN_SEED_LOOKAHEAD:
            return self._error(400, f"Epoch {epoch} is outside the proposer lookahead (current epoch {state_epoch})")
        offset = (epoch - state_epoch) * SLOTS_PER_EPOCH()
        duties = []
        for i in range(SLOTS_PER_EPOCH()):
            vi = int(state.proposer_lookahead[offset + i])
            duties.append({
                "pubkey": "0x" + bytes(state.validators[vi].pubkey).hex(),
                "validator_index": str(vi),
                "slot": str(epoch * SLOTS_PER_EPOCH() + i),
            })
        return web.json_response({
            "dependent_root": "0x" + self._dependent_root(state, epoch - dependent_offset).hex(),
            "execution_optimistic": False,
            "data": duties,
        })

    async def get_proposer_duties_v1(self, request: web.Request) -> web.Response:
        """GET /eth/v1/validator/duties/proposer/{epoch} (deprecated)"""
        return await self._proposer_duties(request, 0)

    async def get_proposer_duties_v2(self, request: web.Request) -> web.Response:
        """GET /eth/v2/validator/duties/proposer/{epoch}"""
        return await self._proposer_duties(request, 1)

    async def post_ptc_duties(self, request: web.Request) -> web.Response:
        """POST /eth/v1/validator/duties/ptc/{epoch}"""
        from ..spec.constants import SLOTS_PER_EPOCH, MIN_SEED_LOOKAHEAD
        from ..spec.state_transition.helpers.ptc import get_ptc

        try:
            epoch = int(request.match_info["epoch"])
            if epoch < 0:
                raise ValueError
        except ValueError:
            return self._error(400, f"Invalid epoch: {request.match_info['epoch']}")
        try:
            body = await request.json()
            if not isinstance(body, list) or not body:
                raise ValueError("expected a non-empty array of validator indices")
            indices = {int(i) for i in body}
        except Exception as e:
            return self._error(400, f"Invalid request body: {e}")
        state = self.node.state
        if state is None:
            return self._error(503, "Beacon node is currently syncing")
        if not hasattr(state, "builders") or epoch < get_network_config().gloas_fork_epoch:
            return self._error(400, "PTC duties are only available from Gloas")
        state_epoch = int(state.slot) // SLOTS_PER_EPOCH()
        if not max(0, state_epoch - 1) <= epoch <= state_epoch + MIN_SEED_LOOKAHEAD:
            return self._error(400, f"Epoch {epoch} is outside the PTC window (current epoch {state_epoch})")
        duties = []
        for slot in range(epoch * SLOTS_PER_EPOCH(), (epoch + 1) * SLOTS_PER_EPOCH()):
            for vi in sorted({int(v) for v in get_ptc(state, slot)} & indices):
                duties.append({
                    "pubkey": "0x" + bytes(state.validators[vi].pubkey).hex(),
                    "validator_index": str(vi),
                    "slot": str(slot),
                })
        return web.json_response({
            "dependent_root": "0x" + self._dependent_root(state, epoch - 1).hex(),
            "execution_optimistic": False,
            "data": duties,
        })

    # ------------------------------------------------------------------ payload attestations

    def _block_root_for_slot(self, slot: int) -> Optional[bytes]:
        from ..crypto import hash_tree_root
        if self.node.head_root is not None and int(self.node.head_slot) == int(slot):
            return bytes(self.node.head_root)
        block = self.node.store.get_block_by_slot(int(slot))
        if block is None or int(block.message.slot) != int(slot):
            return None
        return hash_tree_root(block.message)

    async def get_payload_attestation_data(self, request: web.Request) -> web.Response:
        """GET /eth/v1/validator/payload_attestation_data"""
        from ..spec.types.gloas import PayloadAttestationData
        from ..spec.constants import SLOTS_PER_EPOCH

        try:
            slot = int(request.query["slot"])
        except (KeyError, ValueError):
            return self._error(400, "Invalid or missing slot parameter")
        if self.node.state is None or not self.node._is_synced():
            return self._error(503, "Beacon node is currently syncing")
        if slot // SLOTS_PER_EPOCH() < get_network_config().gloas_fork_epoch:
            return self._error(400, "Payload attestations are only available from Gloas")
        root = self._block_root_for_slot(slot)
        if root is None:
            return web.Response(status=204)
        payload_present = self.node.store.get_payload(root) is not None
        blob_data_available = payload_present
        das = getattr(self.node, "das", None)
        if payload_present and das is not None:
            block = self.node.store.get_block(root)
            commitments = 0
            if block is not None and hasattr(block.message.body, "signed_execution_payload_bid"):
                commitments = len(block.message.body.signed_execution_payload_bid.message.blob_kzg_commitments)
            if commitments:
                blob_data_available = das.is_available(root)
        data = PayloadAttestationData(
            beacon_block_root=root,
            slot=slot,
            payload_present=payload_present,
            blob_data_available=blob_data_available,
        )
        return self._versioned_response(request, self._fork_at_slot(slot), data.encode_bytes(), to_json(data))

    async def get_pool_payload_attestations(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/pool/payload_attestations"""
        slot = request.query.get("slot")
        try:
            slot = int(slot) if slot is not None else None
        except ValueError:
            return self._error(400, "Invalid slot parameter")
        aggregates = await self._run_on_node_loop_fn(
            self.node.payload_attestation_pool.get_aggregates, slot
        )
        version = self._fork_at_slot(slot if slot is not None else self.node.head_slot)
        return web.json_response(
            {"version": version, "data": [to_json(a) for a in aggregates]},
            headers={"Eth-Consensus-Version": version},
        )

    async def post_pool_payload_attestations(self, request: web.Request) -> web.Response:
        """POST /eth/v1/beacon/pool/payload_attestations"""
        from ..spec.types.gloas import PayloadAttestationMessage

        try:
            if self._is_ssz_body(request):
                raw = await request.read()
                size = PayloadAttestationMessage.type_byte_length()
                if len(raw) % size:
                    raise ValueError("SSZ body is not a list of PayloadAttestationMessage")
                messages = [
                    PayloadAttestationMessage.decode_bytes(raw[i:i + size]) for i in range(0, len(raw), size)
                ]
            else:
                body = await request.json()
                if not isinstance(body, list):
                    raise ValueError("expected an array of PayloadAttestationMessage")
                messages = [from_json(PayloadAttestationMessage, m) for m in body]
        except Exception as e:
            return self._error(400, f"Invalid payload attestation messages: {e}")

        failures = []
        for i, msg in enumerate(messages):
            try:
                reason = await self._run_on_node_loop(self._submit_payload_attestation_message(msg))
                if reason:
                    failures.append({"index": i, "message": reason})
            except Exception as e:
                failures.append({"index": i, "message": str(e)})
        if failures:
            return self._error(400, "Some payload attestation messages failed validation", failures=failures)
        return web.Response(status=200)

    async def _submit_payload_attestation_message(self, msg) -> Optional[str]:
        """Gossip-validate, pool, and publish one PayloadAttestationMessage.
        Runs on the node loop. Returns a failure reason or None."""
        from ..spec.constants import DOMAIN_PTC_ATTESTER, SLOTS_PER_EPOCH
        from ..spec.state_transition.helpers.ptc import get_ptc
        from ..spec.state_transition.helpers.domain import get_domain_at_epoch, compute_signing_root
        from ..crypto.crypto import verify_async

        node = self.node
        state = node.state
        if state is None or not hasattr(state, "builders"):
            return "node has no Gloas state"
        slot = int(msg.data.slot)
        if slot != node._current_wall_slot():
            return f"slot {slot} is not the current slot"
        if node.store.get_block(bytes(msg.data.beacon_block_root)) is None:
            return "unknown beacon_block_root"
        ptc = [int(v) for v in get_ptc(state, slot)]
        vi = int(msg.validator_index)
        if vi not in ptc:
            return f"validator {vi} is not in the PTC for slot {slot}"
        domain = get_domain_at_epoch(state, DOMAIN_PTC_ATTESTER, slot // SLOTS_PER_EPOCH())
        signing_root = compute_signing_root(msg.data, domain)
        if not await verify_async(bytes(state.validators[vi].pubkey), signing_root, bytes(msg.signature)):
            return "invalid signature"
        node.payload_attestation_pool.add_message(msg, ptc)
        node._fc_on_payload_attestation(msg)
        if node.beacon_gossip:
            await node.beacon_gossip.publish_payload_attestation_message(msg.encode_bytes())
        self.emit_payload_attestation_message(msg)
        return None

    async def get_proposer_preferences(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/proposer_preferences"""
        try:
            slot = int(request.query["slot"]) if "slot" in request.query else None
            dependent_root = (
                bytes.fromhex(request.query["dependent_root"].removeprefix("0x"))
                if "dependent_root" in request.query else None
            )
        except ValueError:
            return self._error(400, "Invalid slot or dependent_root parameter")
        items = [
            signed for (root, proposal_slot, _), signed in list(self.node.proposer_preferences.items())
            if (slot is None or proposal_slot == slot)
            and (dependent_root is None or root == dependent_root)
        ]
        version = self._fork_at_slot(slot if slot is not None else self.node.head_slot)
        return web.json_response(
            {"version": version, "data": [to_json(p) for p in items]},
            headers={"Eth-Consensus-Version": version},
        )

    # ------------------------------------------------------------------ blocks

    async def post_block_v2(self, request: web.Request) -> web.Response:
        """POST /eth/v2/beacon/blocks"""
        from ..spec.types.base import Container, List
        from ..spec.types.deneb import Blob
        from ..spec.types import KZGProof
        from ..spec.constants import FIELD_ELEMENTS_PER_EXT_BLOB, MAX_BLOB_COMMITMENTS_PER_BLOCK

        version = request.headers.get("Eth-Consensus-Version", "").lower()
        with_blobs = version in ("deneb", "electra", "fulu")
        blobs_bundle = None
        try:
            if not version:
                raise ValueError("missing Eth-Consensus-Version header")
            block_type = self._signed_block_type(version)
            if self._is_ssz_body(request):
                raw = await request.read()
                if with_blobs:
                    class SignedBlockContents(Container):
                        signed_block: block_type
                        kzg_proofs: List[KZGProof, FIELD_ELEMENTS_PER_EXT_BLOB * MAX_BLOB_COMMITMENTS_PER_BLOCK]
                        blobs: List[Blob, MAX_BLOB_COMMITMENTS_PER_BLOCK]

                    contents = SignedBlockContents.decode_bytes(raw)
                    signed_block = contents.signed_block
                    blobs_bundle = {"blobs": ["0x" + bytes(b).hex() for b in contents.blobs]}
                else:
                    signed_block = block_type.decode_bytes(raw)
            else:
                body = await request.json()
                if with_blobs:
                    signed_block = from_json(block_type, body["signed_block"])
                    blobs_bundle = {"blobs": list(body.get("blobs", []))}
                else:
                    signed_block = from_json(block_type, body)
        except Exception as e:
            return self._error(400, f"Invalid block: {e}")

        ssz_bytes = signed_block.encode_bytes()
        try:
            if self.node.beacon_gossip:
                await self._run_on_node_loop(self.node.beacon_gossip.publish_block(ssz_bytes))
                if blobs_bundle and blobs_bundle["blobs"]:
                    await self._run_on_node_loop(self._publish_fulu_columns(signed_block, blobs_bundle))
        except Exception as e:
            logger.error(f"Block gossip publish failed: {e}")
            return self._error(500, f"Broadcast failed: {e}")

        from ..crypto import hash_tree_root
        block_root = hash_tree_root(signed_block.message)
        try:
            await self._run_on_node_loop(self.node._on_p2p_block(ssz_bytes, "beacon-api"))
        except Exception as e:
            logger.warning(f"Block local integration failed: {e}")
        if self.node.store.get_block(block_root) is None:
            return web.Response(status=202)
        return web.Response(status=200)

    async def _publish_fulu_columns(self, signed_block, blobs_bundle: dict) -> None:
        from ..das import compute_subnet_for_data_column_sidecar

        das = getattr(self.node, "das", None)
        if das is None:
            return
        sidecars = await asyncio.get_running_loop().run_in_executor(
            None, das.build_fulu_sidecars, signed_block, blobs_bundle
        )
        for sc in sidecars:
            await self.node.beacon_gossip.publish_data_column_sidecar(
                compute_subnet_for_data_column_sidecar(int(sc.index)), sc.encode_bytes()
            )

    async def post_produce_block_v4(self, request: web.Request) -> web.Response:
        """POST /eth/v4/validator/blocks/{slot}"""
        from ..spec.types.base import Container, List
        from ..spec.types.deneb import Blob
        from ..spec.types import KZGProof
        from ..spec.constants import FIELD_ELEMENTS_PER_EXT_BLOB, MAX_BLOB_COMMITMENTS_PER_BLOCK

        try:
            slot = int(request.match_info["slot"])
            randao_reveal = bytes.fromhex(request.query["randao_reveal"].removeprefix("0x"))
            if len(randao_reveal) != 96:
                raise ValueError("randao_reveal must be 96 bytes")
            graffiti = None
            if "graffiti" in request.query:
                graffiti = bytes.fromhex(request.query["graffiti"].removeprefix("0x")).ljust(32, b"\x00")[:32]
            include_payload = request.query.get("include_payload", "").lower()
            if include_payload not in ("true", "false"):
                raise ValueError("include_payload must be true or false")
            include_payload = include_payload == "true"
        except (KeyError, ValueError) as e:
            return self._error(400, f"Invalid request to produce a block: {e}")
        if not await request.read():
            return self._error(400, "Missing BuilderConfig request body")
        if self.node.state is None or not self.node._is_synced():
            return self._error(503, "Beacon node is currently syncing")
        version = self._fork_at_slot(slot)
        if version in ("phase0", "altair", "bellatrix", "capella", "deneb", "electra", "fulu"):
            return self._error(400, "produceBlockV4 is only available from Gloas")

        try:
            produced = await self._run_on_node_loop(
                self.node.produce_block_for_api(slot, randao_reveal, graffiti)
            )
        except ValueError as e:
            return self._error(400, str(e))
        except Exception as e:
            logger.error(f"produceBlockV4 failed for slot {slot}: {e}")
            return self._error(500, f"Block production failed: {e}")

        block = produced["block"]
        envelope = produced["envelope"]
        bundle = produced["blobs_bundle"] or {}
        blobs = [bytes.fromhex(b.removeprefix("0x")) for b in bundle.get("blobs", [])]
        proofs = [bytes.fromhex(p.removeprefix("0x")) for p in bundle.get("proofs", [])]
        headers = {
            "Eth-Consensus-Version": version,
            "Eth-Consensus-Block-Value": "0",
            "Eth-Execution-Payload-Value": str(produced["execution_payload_value"]),
            "Eth-Execution-Payload-Included": "true" if include_payload else "false",
        }

        if self._wants_ssz(request):
            if include_payload:
                class BlockContents(Container):
                    block: type(block)
                    execution_payload_envelope: type(envelope)
                    kzg_proofs: List[KZGProof, FIELD_ELEMENTS_PER_EXT_BLOB * MAX_BLOB_COMMITMENTS_PER_BLOCK]
                    blobs: List[Blob, MAX_BLOB_COMMITMENTS_PER_BLOCK]

                body = BlockContents(
                    block=block, execution_payload_envelope=envelope, kzg_proofs=proofs, blobs=blobs
                ).encode_bytes()
            else:
                body = block.encode_bytes()
            return web.Response(body=body, content_type="application/octet-stream", headers=headers)

        if include_payload:
            data = {
                "block": to_json(block),
                "execution_payload_envelope": to_json(envelope),
                "kzg_proofs": ["0x" + p.hex() for p in proofs],
                "blobs": ["0x" + b.hex() for b in blobs],
            }
        else:
            data = to_json(block)
        return web.json_response(
            {
                "version": version,
                "consensus_block_value": "0",
                "execution_payload_value": str(produced["execution_payload_value"]),
                "execution_payload_included": include_payload,
                "data": data,
            },
            headers=headers,
        )

    async def get_validator_execution_payload_envelope(self, request: web.Request) -> web.Response:
        """GET /eth/v1/validator/execution_payload_envelopes/{slot}/{beacon_block_root}"""
        try:
            slot = int(request.match_info["slot"])
            root = bytes.fromhex(request.match_info["beacon_block_root"].removeprefix("0x"))
        except ValueError:
            return self._error(400, "Invalid slot or beacon_block_root parameter")
        cached = self.node.api_produced_payload(slot, root)
        if cached is None:
            return self._error(404, "Execution payload envelope not available for slot")
        envelope = cached["envelope"]
        return self._versioned_response(request, self._fork_at_slot(slot), envelope.encode_bytes(), to_json(envelope))

    async def post_builder_preferences(self, request: web.Request) -> web.Response:
        """POST /eth/v1/validator/builder_preferences

        Forwards each entry to the builder-API submitBuilderPreferences
        endpoint at the entry's url.
        """
        import aiohttp
        from urllib.parse import urlparse

        version = request.headers.get("Eth-Consensus-Version", "gloas").lower()
        if self._is_ssz_body(request):
            return self._error(415, "SSZ builder preferences are not supported, use JSON")
        try:
            entries = await request.json()
            if not isinstance(entries, list):
                raise ValueError("expected an array of BuilderPreferencesEntry")
        except Exception as e:
            return self._error(400, f"Invalid builder preferences: {e}")

        failures = []
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for i, entry in enumerate(entries):
                try:
                    url = str(entry["url"]).rstrip("/")
                    if not url or urlparse(url).scheme not in ("http", "https"):
                        raise ValueError("url must be http(s)")
                    pubkey = entry["proposer_pubkey"]
                    payload = {
                        "preferences": {"max_execution_payment": str(int(entry["max_execution_payment"]))},
                        "auth": entry["auth"],
                    }
                    async with session.post(
                        f"{url}/eth/v1/builder/builder_preferences/{pubkey}",
                        json=payload,
                        headers={"Eth-Consensus-Version": version},
                        allow_redirects=False,
                    ) as resp:
                        if resp.status >= 300:
                            text = (await resp.text())[:200]
                            failures.append({"index": i, "message": f"builder returned {resp.status}: {text}"})
                except Exception as e:
                    failures.append({"index": i, "message": str(e)})
        if failures:
            return self._error(400, "Some builder preferences could not be submitted", failures=failures)
        return web.Response(status=200)

    async def get_blobs(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/blobs/{block_id}"""
        from hashlib import sha256
        from ..spec.types.base import List
        from ..spec.types.deneb import Blob
        from ..spec.constants import MAX_BLOB_COMMITMENTS_PER_BLOCK

        root, signed_block = self._resolve_block_id(request.match_info["block_id"])
        if root is None or signed_block is None:
            return self._error(404, "Block not found")
        body = signed_block.message.body
        if hasattr(body, "signed_execution_payload_bid"):
            commitments = [bytes(c) for c in body.signed_execution_payload_bid.message.blob_kzg_commitments]
        else:
            commitments = [bytes(c) for c in getattr(body, "blob_kzg_commitments", [])]

        blobs = self._load_blobs(root, len(commitments))
        if blobs is None:
            return self._error(404, "Blobs not available for block")

        wanted = request.query.getall("versioned_hashes", [])
        if wanted:
            wanted = {bytes.fromhex(h.removeprefix("0x")) for w in wanted for h in w.split(",") if h}
            blobs = [
                b for b, c in zip(blobs, commitments)
                if b"\x01" + sha256(c).digest()[1:] in wanted
            ]
        if self._wants_ssz(request):
            return web.Response(
                body=List[Blob, MAX_BLOB_COMMITMENTS_PER_BLOCK](*blobs).encode_bytes(),
                content_type="application/octet-stream",
            )
        return web.json_response({
            "execution_optimistic": False,
            "finalized": False,
            "data": ["0x" + b.hex() for b in blobs],
        })

    def _load_blobs(self, root: bytes, count: int) -> Optional[list[bytes]]:
        """Blobs for ``root``: stored EL bundles first, else rebuilt from the
        first half of the data columns (the cells of the unextended blob)."""
        if count == 0:
            return []
        stored = self.node.store.get_blobs(root)
        if len(stored) >= count:
            return [bytes.fromhex(sc["blob"].removeprefix("0x")) for sc in stored[:count]]
        from ..spec.constants import NUMBER_OF_COLUMNS
        from ..spec.types.gloas import DataColumnSidecar
        from ..spec.types.fulu import DataColumnSidecar as FuluDataColumnSidecar

        columns = self.node.store.get_data_columns(root)
        half = NUMBER_OF_COLUMNS // 2
        if any(i not in columns for i in range(half)):
            return None
        decoded = []
        for i in range(half):
            try:
                decoded.append(DataColumnSidecar.decode_bytes(columns[i]))
            except Exception:
                decoded.append(FuluDataColumnSidecar.decode_bytes(columns[i]))
        return [b"".join(bytes(sc.column[blob]) for sc in decoded) for blob in range(count)]

    # ------------------------------------------------------------------ fork choice

    async def get_fork_choice_v2(self, request: web.Request) -> web.Response:
        """GET /eth/v2/debug/fork_choice"""
        try:
            data = await self._run_on_node_loop_fn(self._fork_choice_dump)
        except Exception as e:
            return self._error(500, f"Fork choice dump failed: {e}")
        return web.json_response({"data": data})

    async def get_fork_choice_v1(self, request: web.Request) -> web.Response:
        """GET /eth/v1/debug/fork_choice (deprecated)"""
        try:
            data = await self._run_on_node_loop_fn(self._fork_choice_dump)
        except Exception as e:
            return self._error(500, f"Fork choice dump failed: {e}")
        by_root: dict[str, dict] = {}
        for n in data["fork_choice_nodes"]:
            by_root.setdefault(n["block_root"], {})[n["payload_status"]] = n
        nodes = []
        for variants in by_root.values():
            base = variants.get("pending") or variants["full"]
            head = variants.get("full") or base
            nodes.append({
                "slot": base["slot"],
                "block_root": base["block_root"],
                "parent_root": base["parent_root"],
                "justified_epoch": base["justified_checkpoint"]["epoch"],
                "finalized_epoch": base["finalized_checkpoint"]["epoch"],
                "weight": base["weight"],
                "validity": base["validity"],
                "execution_block_hash": head["execution_block_hash"],
                "extra_data": {},
            })
        return web.json_response({
            "justified_checkpoint": data["justified_checkpoint"],
            "finalized_checkpoint": data["finalized_checkpoint"],
            "fork_choice_nodes": nodes,
            "extra_data": {},
        })

    def _chain_fork_choice_dump(self, checkpoint) -> dict:
        """Fork-choice view without the spec Store (fast confirmation off):
        the canonical chain back to the finalized checkpoint, unweighted."""
        from ..spec.constants import SLOTS_PER_EPOCH

        state = self.node.state
        justified = checkpoint(state.current_justified_checkpoint)
        finalized = checkpoint(state.finalized_checkpoint)
        finalized_slot = int(state.finalized_checkpoint.epoch) * SLOTS_PER_EPOCH()

        chain = []
        root = self.node.head_root
        while root is not None and len(chain) < 256:
            signed = self.node.store.get_block(root)
            if signed is None:
                break
            chain.append((bytes(root), signed.message))
            if int(signed.message.slot) <= finalized_slot:
                break
            root = bytes(signed.message.parent_root)
        chain.reverse()

        def bid_of(block):
            body = block.body
            return body.signed_execution_payload_bid.message if hasattr(body, "signed_execution_payload_bid") else None

        known = {root: block for root, block in chain}
        nodes = []
        for root, block in chain:
            parent_root = bytes(block.parent_root)
            parent = known.get(parent_root)
            bid = bid_of(block)
            base = {
                "slot": str(int(block.slot)),
                "block_root": "0x" + root.hex(),
                "justified_checkpoint": justified,
                "finalized_checkpoint": finalized,
                "weight": "0",
                "validity": "valid",
                "payload_attester_count": "0",
                "payload_availability_yes_count": "0",
                "payload_data_availability_yes_count": "0",
                "extra_data": {},
            }
            if parent is None:
                parent_status = None
            elif bid is None or bid_of(parent) is None:
                parent_status = "full"
            else:
                parent_status = "full" if bytes(bid.parent_block_hash) == bytes(bid_of(parent).block_hash) else "empty"
            if bid is None:
                nodes.append({
                    **base,
                    "payload_status": "full",
                    "parent_root": "0x" + parent_root.hex(),
                    "parent_payload_status": parent_status,
                    "execution_block_hash": "0x" + bytes(
                        block.body.execution_payload.block_hash if hasattr(block.body, "execution_payload") else b"\x00" * 32
                    ).hex(),
                })
                continue
            nodes.append({
                **base,
                "payload_status": "pending",
                "parent_root": "0x" + parent_root.hex(),
                "parent_payload_status": parent_status,
                "execution_block_hash": "0x" + bytes(bid.parent_block_hash).hex(),
            })
            nodes.append({
                **base,
                "payload_status": "empty",
                "parent_root": "0x" + root.hex(),
                "parent_payload_status": "pending",
                "execution_block_hash": "0x" + bytes(bid.parent_block_hash).hex(),
            })
            if self.node.store.get_payload(root) is not None:
                nodes.append({
                    **base,
                    "payload_status": "full",
                    "parent_root": "0x" + root.hex(),
                    "parent_payload_status": "pending",
                    "execution_block_hash": "0x" + bytes(bid.block_hash).hex(),
                })
        return {
            "justified_checkpoint": justified,
            "finalized_checkpoint": finalized,
            "fork_choice_nodes": nodes,
            "extra_data": {},
        }

    def _fork_choice_dump(self) -> dict:
        """Snapshot of the fork-choice tree, one node per (block_root, payload_status).
        Runs on the node loop so the Store is not mutated underneath it."""
        from ..spec import fork_choice as fc

        def checkpoint(c) -> dict:
            return {"epoch": str(int(c.epoch)), "root": "0x" + bytes(c.root).hex()}

        status_name = {
            fc.PAYLOAD_STATUS_PENDING: "pending",
            fc.PAYLOAD_STATUS_EMPTY: "empty",
            fc.PAYLOAD_STATUS_FULL: "full",
        }
        store = self.node.fc_store
        if store is None:
            return self._chain_fork_choice_dump(checkpoint)

        nodes = []
        for root, block in list(store.blocks.items()):
            root = bytes(root)
            bid = block.body.signed_execution_payload_bid.message
            parent_root = bytes(block.parent_root)
            state = store.block_states.get(root)
            justified = checkpoint(state.current_justified_checkpoint) if state is not None else checkpoint(store.justified_checkpoint)
            finalized = checkpoint(state.finalized_checkpoint) if state is not None else checkpoint(store.finalized_checkpoint)
            timeliness = store.payload_timeliness_vote.get(root) or []
            availability = store.payload_data_availability_vote.get(root) or []
            counts = {
                "payload_attester_count": str(sum(v is not None for v in timeliness)),
                "payload_availability_yes_count": str(sum(v is True for v in timeliness)),
                "payload_data_availability_yes_count": str(sum(v is True for v in availability)),
            }
            statuses = [fc.PAYLOAD_STATUS_PENDING, fc.PAYLOAD_STATUS_EMPTY]
            if fc.is_payload_verified(store, root):
                statuses.append(fc.PAYLOAD_STATUS_FULL)
            for status in statuses:
                if status == fc.PAYLOAD_STATUS_PENDING:
                    node_parent = parent_root
                    parent_status = (
                        status_name[fc.get_parent_payload_status(store, block)]
                        if parent_root in store.blocks else None
                    )
                else:
                    node_parent = root
                    parent_status = "pending"
                try:
                    weight = fc.get_weight(store, fc.ForkChoiceNode(root=root, payload_status=status))
                except Exception:
                    weight = 0
                nodes.append({
                    "slot": str(int(block.slot)),
                    "block_root": "0x" + root.hex(),
                    "payload_status": status_name[status],
                    "parent_root": "0x" + node_parent.hex(),
                    "parent_payload_status": parent_status,
                    "justified_checkpoint": justified,
                    "finalized_checkpoint": finalized,
                    "weight": str(int(weight)),
                    "validity": "valid",
                    "execution_block_hash": "0x" + bytes(
                        bid.block_hash if status == fc.PAYLOAD_STATUS_FULL else bid.parent_block_hash
                    ).hex(),
                    **counts,
                    "extra_data": {},
                })
        nodes.sort(key=lambda n: int(n["slot"]))
        return {
            "justified_checkpoint": checkpoint(store.justified_checkpoint),
            "finalized_checkpoint": checkpoint(store.finalized_checkpoint),
            "fork_choice_nodes": nodes,
            "extra_data": {},
        }

    async def get_spec(self, request: web.Request) -> web.Response:
        """GET /eth/v1/config/spec"""
        spec = build_spec_response()
        return web.json_response({"data": spec})

    async def get_fork_schedule(self, request: web.Request) -> web.Response:
        """GET /eth/v1/config/fork_schedule"""
        from ..spec.network_config import get_config
        config = get_config()

        forks = []
        prev = config.genesis_fork_version
        for epoch, version, _name in config.get_fork_schedule():
            forks.append({
                "previous_version": "0x" + bytes(prev).hex(),
                "current_version": "0x" + bytes(version).hex(),
                "epoch": str(epoch),
            })
            prev = version
        return web.json_response({"data": forks})

    async def get_deposit_contract(self, request: web.Request) -> web.Response:
        """GET /eth/v1/config/deposit_contract"""
        from ..spec.network_config import get_config
        config = get_config()

        chain_id = getattr(config, 'deposit_chain_id', 1)
        address = getattr(config, 'deposit_contract_address', b'\x00' * 20)

        return web.json_response({
            "data": {
                "chain_id": str(chain_id),
                "address": "0x" + (address.hex() if isinstance(address, bytes) else address),
            }
        })

    def _get_state_version(self, state) -> str:
        """Determine the fork version string for a state."""
        # Check GLOAS before fulu since GLOAS extends fulu
        if hasattr(state, "builders"):
            if hasattr(state.latest_execution_payload_bid, "inclusion_list_bits"):
                from ..spec.network_config import get_config
                from ..spec.constants import SLOTS_PER_EPOCH
                return get_config().fork_name_at_epoch(int(state.slot) // SLOTS_PER_EPOCH())
            return "gloas"
        if hasattr(state, "proposer_lookahead"):
            return "fulu"
        if hasattr(state, "pending_deposits"):
            return "electra"
        if hasattr(state, "latest_execution_payload_header") and hasattr(
            state.latest_execution_payload_header, "blob_gas_used"
        ):
            return "deneb"
        if hasattr(state, "latest_execution_payload_header") and hasattr(
            state.latest_execution_payload_header, "withdrawals_root"
        ):
            return "capella"
        if hasattr(state, "latest_execution_payload_header"):
            return "bellatrix"
        if hasattr(state, "current_sync_committee"):
            return "altair"
        return "phase0"

    def _resolve_state_id(self, state_id: str):
        """Resolve a state_id to the actual state object.

        Supports: "head", "finalized", "justified", "genesis", slot number, state root, or block root.
        Returns None if not found.
        """
        if state_id == "head":
            return self.node.state
        if state_id == "genesis":
            # First try current state if at slot 0
            if self.node.state and int(self.node.state.slot) == 0:
                return self.node.state
            # Then try to fetch from store
            if self.node.store:
                return self.node.store.get_state_by_slot(0)
            return None
        if state_id in ("finalized", "justified"):
            return self.node.state
        if state_id.startswith("0x"):
            try:
                root = bytes.fromhex(state_id[2:])
                if len(root) == 32:
                    if self.node.state:
                        from ..crypto import hash_tree_root
                        current_state_root = hash_tree_root(self.node.state)
                        if current_state_root == root:
                            return self.node.state
                        header = self.node.state.latest_block_header
                        if bytes(header.state_root) == root:
                            return self.node.state
                    if self.node.store:
                        stored_state = self.node.store.get_state(root)
                        if stored_state:
                            return stored_state
                        block = self.node.store.get_block(root)
                        if block and hasattr(block, "message"):
                            block_state_root = bytes(block.message.state_root)
                            stored_state = self.node.store.get_state(block_state_root)
                            if stored_state:
                                return stored_state
                            if self.node.state:
                                from ..crypto import hash_tree_root
                                current_state_root = hash_tree_root(self.node.state)
                                if current_state_root == block_state_root:
                                    return self.node.state
            except ValueError:
                pass
            return None
        try:
            slot = int(state_id)
            if self.node.state and int(self.node.state.slot) == slot:
                return self.node.state
            # Try to fetch from store if current state doesn't match
            if self.node.store:
                return self.node.store.get_state_by_slot(slot)
            return None
        except ValueError:
            return None

    async def get_debug_state(self, request: web.Request) -> web.Response:
        """GET /eth/v2/debug/beacon/states/{state_id}

        Returns the full beacon state for debugging/indexing purposes.
        Supports SSZ (application/octet-stream) or JSON responses.
        """
        state_id = request.match_info["state_id"]
        state = self._resolve_state_id(state_id)

        if state is None:
            return web.json_response({"message": "State not found"}, status=404)

        accept = request.headers.get("Accept", "application/json")
        version = self._get_state_version(state)

        if "application/octet-stream" in accept:
            ssz_bytes = state.encode_bytes()
            return web.Response(
                body=ssz_bytes,
                content_type="application/octet-stream",
                headers={"Eth-Consensus-Version": version},
            )

        return web.json_response(
            {"message": "JSON format not supported for full state, use SSZ"},
            status=406,
        )

    EVENT_TOPICS = frozenset({
        "head", "head_v2", "block", "finalized_checkpoint", "chain_reorg",
        "payload_attributes", "execution_payload", "execution_payload_gossip",
        "execution_payload_available", "execution_payload_bid",
        "payload_attestation_message", "fast_confirmation", "proposer_preferences",
        "inclusion_list",
    })
    KNOWN_EVENT_TOPICS = EVENT_TOPICS | {
        "block_gossip", "attestation", "single_attestation", "voluntary_exit",
        "bls_to_execution_change", "proposer_slashing", "attester_slashing",
        "contribution_and_proof", "light_client_finality_update",
        "light_client_optimistic_update", "data_column_sidecar",
    }

    async def get_events(self, request: web.Request) -> web.StreamResponse:
        """GET /eth/v1/events - SSE endpoint for beacon events."""
        topics = {
            t.strip()
            for param in request.query.getall("topics", [])
            for t in param.split(",")
            if t.strip()
        }
        if not topics:
            return self._error(400, "No topics specified")
        unknown = topics - self.KNOWN_EVENT_TOPICS
        if unknown:
            return self._error(400, f"Invalid topic: {sorted(unknown)[0]}")
        requested_topics = topics

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

    def _head_events(self, slot: int, root: bytes, payload_status: str, new_block: bool) -> list[dict]:
        """head / head_v2 / block events for a new fork-choice head."""
        from ..crypto import hash_tree_root
        from ..spec.constants import SLOTS_PER_EPOCH

        state = self.node.state
        if state is None:
            return []
        state_root = "0x" + hash_tree_root(state).hex()
        epoch = slot // SLOTS_PER_EPOCH()
        prev_dependent = "0x" + self._dependent_root(state, epoch - 1).hex()
        current_dependent = "0x" + self._dependent_root(state, epoch).hex()
        epoch_transition = slot % SLOTS_PER_EPOCH() == 0
        block_hex = "0x" + root.hex()
        events = []
        if new_block:
            events.append({
                "event": "head",
                "data": {
                    "slot": str(slot),
                    "block": block_hex,
                    "state": state_root,
                    "epoch_transition": epoch_transition,
                    "previous_duty_dependent_root": prev_dependent,
                    "current_duty_dependent_root": current_dependent,
                    "execution_optimistic": False,
                },
            })
            block_data = {"slot": str(slot), "block": block_hex, "execution_optimistic": False}
            signed_block = self.node.store.get_block(root)
            if signed_block is not None and hasattr(signed_block.message.body, "signed_execution_payload_bid"):
                bid = signed_block.message.body.signed_execution_payload_bid.message
                block_data["builder_index"] = str(int(bid.builder_index))
                block_data["block_hash"] = "0x" + bytes(bid.block_hash).hex()
            events.append({"event": "block", "data": block_data})
        events.append({
            "event": "head_v2",
            "data": {
                "version": self._fork_at_slot(slot),
                "data": {
                    "slot": str(slot),
                    "block": block_hex,
                    "state": state_root,
                    "payload_status": payload_status,
                    "epoch_transition": epoch_transition,
                    "current_epoch_dependent_root": prev_dependent,
                    "next_epoch_dependent_root": current_dependent,
                    "execution_optimistic": False,
                },
            },
        })
        return events

    def _head_payload_status(self, slot: int, root: bytes) -> str:
        from ..spec.constants import SLOTS_PER_EPOCH
        if slot // SLOTS_PER_EPOCH() < get_network_config().gloas_fork_epoch:
            return "full"
        return "full" if self.node.store.get_payload(root) is not None else "empty"

    async def _emit_events_loop(self) -> None:
        """Background task that checks for state changes and emits events."""
        self._last_head_slot = self.node.head_slot
        self._last_head_root = self.node.head_root
        self._last_head_payload_status: Optional[str] = None
        if self.node.state:
            self._last_finalized_epoch = int(self.node.state.finalized_checkpoint.epoch)

        while True:
            try:
                await asyncio.sleep(0.5)

                current_slot = int(self.node.head_slot)
                current_root = self.node.head_root

                if current_root:
                    new_block = current_slot != self._last_head_slot or current_root != self._last_head_root
                    payload_status = self._head_payload_status(current_slot, current_root)
                    if new_block or payload_status != self._last_head_payload_status:
                        for event in self._head_events(current_slot, current_root, payload_status, new_block):
                            await self._broadcast_event(event)
                    self._last_head_payload_status = payload_status

                self._last_head_slot = current_slot
                self._last_head_root = current_root

                if self.node.state:
                    current_finalized_epoch = int(self.node.state.finalized_checkpoint.epoch)
                    if current_finalized_epoch != self._last_finalized_epoch:
                        finalized_root = bytes(self.node.state.finalized_checkpoint.root)
                        finalized_block = self.node.store.get_block(finalized_root)
                        finalized_state_root = (
                            bytes(finalized_block.message.state_root) if finalized_block is not None else finalized_root
                        )
                        finalized_event = {
                            "event": "finalized_checkpoint",
                            "data": {
                                "block": "0x" + finalized_root.hex(),
                                "state": "0x" + finalized_state_root.hex(),
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

    def emit(self, event: str, data: dict) -> None:
        """Queue an SSE event from any thread."""
        self._cross_loop_events.put({"event": event, "data": data})

    async def emit_fast_confirmation(self, slot: int, block_root: bytes, current_slot: int) -> None:
        """Emit the `fast_confirmation` SSE event: emitted every time the fast
        confirmation rule runs, with the latest confirmed block."""
        self.emit("fast_confirmation", {
            "block": "0x" + block_root.hex(),
            "slot": str(slot),
            "current_slot": str(current_slot),
        })

    async def emit_execution_payload_available(self, slot: int, block_root: bytes) -> None:
        """Emit execution_payload_available SSE event (ePBS)."""
        self.emit("execution_payload_available", {
            "slot": str(slot),
            "block_root": "0x" + block_root.hex(),
        })

    def emit_execution_payload(self, signed_envelope, gossip: bool) -> None:
        """Emit `execution_payload_gossip` (passed gossip validation) or
        `execution_payload` (imported by fork choice)."""
        envelope = signed_envelope.message
        slot = self.node._envelope_slot(envelope)
        data = {
            "slot": str(slot if slot is not None else int(envelope.payload.slot_number)),
            "builder_index": str(int(envelope.builder_index)),
            "block_hash": "0x" + bytes(envelope.payload.block_hash).hex(),
            "block_root": "0x" + bytes(envelope.beacon_block_root).hex(),
        }
        if gossip:
            self.emit("execution_payload_gossip", data)
        else:
            self.emit("execution_payload", {**data, "execution_optimistic": False})

    def emit_payload_attestation_message(self, msg) -> None:
        """Emit a `payload_attestation_message` SSE event."""
        self.emit("payload_attestation_message", {
            "version": self._fork_at_slot(int(msg.data.slot)),
            "data": to_json(msg),
        })

    async def emit_payload_attributes(
        self,
        *,
        proposal_slot: int,
        proposer_index: int,
        parent_block_root: bytes,
        parent_block_hash: bytes,
        version: str,
        payload_attributes: dict,
        parent_block_number: int = 0,
        safe_block_hash: Optional[bytes] = None,
        finalized_block_hash: Optional[bytes] = None,
    ) -> None:
        """Emit a payload_attributes SSE event."""
        data = {
            "proposal_slot": str(proposal_slot),
            "proposer_index": str(proposer_index),
            "parent_block_root": "0x" + parent_block_root.hex(),
            "parent_block_hash": "0x" + parent_block_hash.hex(),
            "payload_attributes": payload_attributes,
        }
        if version in ("bellatrix", "capella", "deneb", "electra", "fulu"):
            data["parent_block_number"] = str(parent_block_number)
        else:
            data["safe_block_hash"] = "0x" + (safe_block_hash or parent_block_hash).hex()
            data["finalized_block_hash"] = "0x" + (finalized_block_hash or b"\x00" * 32).hex()
        self.emit("payload_attributes", {"version": version, "data": data})

    async def emit_proposer_preferences(self, signed) -> None:
        """Emit a proposer_preferences SSE event."""
        self.emit("proposer_preferences", {
            "version": self._fork_at_slot(int(signed.message.proposal_slot)),
            "data": to_json(signed),
        })

    async def emit_execution_payload_bid(self, signed_bid) -> None:
        """Emit execution_payload_bid SSE event (ePBS)."""
        self.emit("execution_payload_bid", {
            "version": self._fork_at_slot(int(signed_bid.message.slot)),
            "data": to_json(signed_bid),
        })

    def emit_inclusion_list(self, signed_inclusion_list) -> None:
        """Emit an inclusion_list SSE event (Heze, EIP-7805). Thread-safe."""
        from ..spec.network_config import get_config
        from ..spec.constants import SLOTS_PER_EPOCH

        il = signed_inclusion_list.message
        self._cross_loop_events.put({
            "event": "inclusion_list",
            "data": {
                "version": get_config().fork_name_at_epoch(int(il.slot) // SLOTS_PER_EPOCH()),
                "data": {
                    "message": {
                        "slot": str(int(il.slot)),
                        "validator_index": str(int(il.validator_index)),
                        "dependent_root": "0x" + bytes(il.dependent_root).hex(),
                        "transactions": ["0x" + bytes(tx).hex() for tx in il.transactions],
                    },
                    "signature": "0x" + bytes(signed_inclusion_list.signature).hex(),
                },
            },
        })
