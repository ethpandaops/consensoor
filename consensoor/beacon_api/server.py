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

FORK_NAMES = ("phase0", "altair", "bellatrix", "capella", "deneb", "electra", "fulu", "gloas", "heze")
VALIDATOR_STATUSES = frozenset({
    "pending_initialized", "pending_queued", "active_ongoing", "active_exiting", "active_slashed",
    "exited_unslashed", "exited_slashed", "withdrawal_possible", "withdrawal_done",
    "active", "pending", "exited", "withdrawal",
})
MAX_QUERY_IDS = 64
BASE58_ALPHABET = frozenset("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")
MAX_REQUEST_LIGHT_CLIENT_UPDATES = 128
MAX_LIGHT_CLIENT_PERIODS = 64


class ApiError(Exception):
    def __init__(self, status: int, message: str, **extra):
        super().__init__(message)
        self.status = status
        self.message = message
        self.extra = extra


@web.middleware
async def _error_middleware(request: web.Request, handler):
    try:
        return await handler(request)
    except ApiError as e:
        return BeaconAPI._error(e.status, e.message, **e.extra)
    except web.HTTPException:
        raise
    except Exception as e:
        logger.exception(f"{request.method} {request.path} failed")
        return BeaconAPI._error(500, f"Internal error: {e}")


class BeaconAPI:
    """Beacon API server."""

    def __init__(self, node, host: str = "0.0.0.0", port: int = 5052):
        self.node = node
        self.host = host
        self.port = port
        self.app = web.Application(client_max_size=256 * 1024 * 1024, middlewares=[_error_middleware])
        self._lc_optimistic_update = None
        self._lc_finality_update = None
        self._lc_best_updates: dict[int, object] = {}
        self._blinded_types: dict[str, tuple] = {}
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

    def _setup_routes(self):
        """Set up API routes."""
        r = self.app.router
        r.add_get("/eth/v1/node/health", self.get_health)
        r.add_get("/eth/v1/node/version", self.get_version)
        r.add_get("/eth/v2/node/version", self.get_version_v2)
        r.add_get("/eth/v1/node/syncing", self.get_syncing)
        r.add_get("/eth/v1/node/identity", self.get_identity)
        r.add_get("/eth/v1/node/peers", self.get_peers)
        r.add_get("/eth/v1/node/peers/{peer_id}", self.get_peer)
        r.add_get("/eth/v1/node/peer_count", self.get_peer_count)
        r.add_get("/consensoor/v1/fast_confirmation", self.get_fast_confirmation)
        r.add_get("/consensoor/v1/custody", self.get_custody)
        r.add_post("/consensoor/v1/custody/backfill", self.post_custody_backfill)

        r.add_get("/eth/v1/beacon/genesis", self.get_genesis)
        r.add_get("/eth/v1/beacon/states/{state_id}/root", self.get_state_root)
        r.add_get("/eth/v1/beacon/states/{state_id}/fork", self.get_state_fork)
        r.add_get("/eth/v1/beacon/states/{state_id}/finality_checkpoints", self.get_finality_checkpoints)
        r.add_get("/eth/v1/beacon/states/{state_id}/validators", self.get_validators)
        r.add_post("/eth/v1/beacon/states/{state_id}/validators", self.post_validators)
        r.add_get("/eth/v1/beacon/states/{state_id}/validators/{validator_id}", self.get_validator)
        r.add_get("/eth/v1/beacon/states/{state_id}/validator_balances", self.get_validator_balances)
        r.add_post("/eth/v1/beacon/states/{state_id}/validator_balances", self.post_validator_balances)
        r.add_post("/eth/v1/beacon/states/{state_id}/validator_identities", self.post_validator_identities)
        r.add_post("/eth/v1/beacon/states/{state_id}/builders", self.post_state_builders)
        r.add_get("/eth/v1/beacon/states/{state_id}/committees", self.get_committees)
        r.add_get("/eth/v1/beacon/states/{state_id}/sync_committees", self.get_sync_committees)
        r.add_get("/eth/v1/beacon/states/{state_id}/ptc", self.get_state_ptc)
        r.add_get("/eth/v1/beacon/states/{state_id}/randao", self.get_randao)
        for field in (
            "pending_consolidations", "pending_deposits", "pending_partial_withdrawals",
            "proposer_lookahead", "builder_pending_withdrawals", "builder_pending_payments",
        ):
            r.add_get(f"/eth/v1/beacon/states/{{state_id}}/{field}", self._state_field_handler(field))

        r.add_get("/eth/v1/beacon/headers", self.get_headers)
        r.add_get("/eth/v1/beacon/headers/{block_id}", self.get_header)
        r.add_post("/eth/v2/beacon/blinded_blocks", self.post_blinded_block_v2)
        r.add_post("/eth/v2/beacon/blocks", self.post_block_v2)
        r.add_post("/eth/v1/beacon/execution_payload_bids", self.post_execution_payload_bids)
        r.add_post("/eth/v1/beacon/execution_payload_envelopes", self.post_execution_payload_envelopes)
        r.add_get("/eth/v1/beacon/execution_payload_envelopes/{block_id}", self.get_execution_payload_envelope)
        r.add_get("/eth/v1/beacon/execution_payload_envelope/{block_id}", self.get_execution_payload_envelope)
        r.add_get("/eth/v1/beacon/proposer_preferences", self.get_proposer_preferences)
        r.add_get("/eth/v2/beacon/blocks/{block_id}", self.get_block)
        r.add_get("/eth/v1/beacon/blocks/{block_id}/root", self.get_block_root)
        r.add_get("/eth/v2/beacon/blocks/{block_id}/attestations", self.get_block_attestations)
        r.add_get("/eth/v1/beacon/blobs/{block_id}", self.get_blobs)
        r.add_get("/eth/v1/debug/beacon/data_column_sidecars/{block_id}", self.get_data_column_sidecars)
        r.add_post("/eth/v1/beacon/rewards/sync_committee/{block_id}", self.post_sync_committee_rewards)
        r.add_get("/eth/v1/beacon/rewards/blocks/{block_id}", self.get_block_rewards)
        r.add_post("/eth/v1/beacon/rewards/attestations/{epoch}", self.post_attestation_rewards)
        r.add_get("/eth/v1/beacon/blinded_blocks/{block_id}", self.get_blinded_block)
        r.add_get("/eth/v1/beacon/light_client/bootstrap/{block_root}", self.get_light_client_bootstrap)
        r.add_get("/eth/v1/beacon/light_client/updates", self.get_light_client_updates)
        r.add_get("/eth/v1/beacon/light_client/finality_update", self.get_light_client_finality_update)
        r.add_get("/eth/v1/beacon/light_client/optimistic_update", self.get_light_client_optimistic_update)
        r.add_get("/eth/v2/beacon/pool/attestations", self.get_pool_attestations)
        r.add_post("/eth/v2/beacon/pool/attestations", self.post_pool_attestations)
        r.add_get("/eth/v1/beacon/pool/payload_attestations", self.get_pool_payload_attestations)
        r.add_post("/eth/v1/beacon/pool/payload_attestations", self.post_pool_payload_attestations)
        r.add_get("/eth/v2/beacon/pool/attester_slashings", self.get_pool_attester_slashings)
        r.add_post("/eth/v2/beacon/pool/attester_slashings", self.post_pool_attester_slashings)
        r.add_get("/eth/v1/beacon/pool/proposer_slashings", self.get_pool_proposer_slashings)
        r.add_post("/eth/v1/beacon/pool/proposer_slashings", self.post_pool_proposer_slashings)
        r.add_post("/eth/v1/beacon/pool/sync_committees", self.post_pool_sync_committees)
        r.add_get("/eth/v1/beacon/pool/voluntary_exits", self.get_pool_voluntary_exits)
        r.add_post("/eth/v1/beacon/pool/voluntary_exits", self.post_pool_voluntary_exits)
        r.add_get("/eth/v1/beacon/pool/bls_to_execution_changes", self.get_pool_bls_to_execution_changes)
        r.add_post("/eth/v1/beacon/pool/bls_to_execution_changes", self.post_pool_bls_to_execution_changes)

        r.add_get("/eth/v2/debug/beacon/states/{state_id}", self.get_debug_state)
        r.add_get("/eth/v2/debug/beacon/heads", self.get_debug_heads)
        r.add_get("/eth/v1/debug/fork_choice", self.get_fork_choice_v1)
        r.add_get("/eth/v2/debug/fork_choice", self.get_fork_choice_v2)

        r.add_get("/eth/v1/config/spec", self.get_spec)
        r.add_get("/eth/v1/config/fork_schedule", self.get_fork_schedule)
        r.add_get("/eth/v1/config/deposit_contract", self.get_deposit_contract)

        r.add_post("/eth/v1/validator/duties/attester/{epoch}", self.post_attester_duties)
        r.add_get("/eth/v1/validator/duties/proposer/{epoch}", self.get_proposer_duties_v1)
        r.add_get("/eth/v2/validator/duties/proposer/{epoch}", self.get_proposer_duties_v2)
        r.add_post("/eth/v1/validator/duties/sync/{epoch}", self.post_sync_duties)
        r.add_post("/eth/v1/validator/duties/ptc/{epoch}", self.post_ptc_duties)
        r.add_get("/eth/v3/validator/blocks/{slot}", self.get_produce_block_v3)
        r.add_post("/eth/v4/validator/blocks/{slot}", self.post_produce_block_v4)
        r.add_get("/eth/v1/validator/attestation_data", self.get_attestation_data)
        r.add_get("/eth/v1/validator/payload_attestation_data", self.get_payload_attestation_data)
        r.add_get("/eth/v2/validator/aggregate_attestation", self.get_aggregate_attestation)
        r.add_post("/eth/v2/validator/aggregate_and_proofs", self.post_aggregate_and_proofs)
        r.add_post("/eth/v1/validator/beacon_committee_subscriptions", self.post_beacon_committee_subscriptions)
        r.add_post("/eth/v1/validator/sync_committee_subscriptions", self.post_sync_committee_subscriptions)
        r.add_post("/eth/v1/validator/beacon_committee_selections", self.post_selections)
        r.add_get("/eth/v1/validator/sync_committee_contribution", self.get_sync_committee_contribution)
        r.add_post("/eth/v1/validator/sync_committee_selections", self.post_selections)
        r.add_post("/eth/v1/validator/contribution_and_proofs", self.post_contribution_and_proofs)
        r.add_post("/eth/v1/validator/prepare_beacon_proposer", self.post_prepare_beacon_proposer)
        r.add_post("/eth/v1/validator/register_validator", self.post_register_validator)
        r.add_post("/eth/v1/validator/proposer_preferences", self.post_proposer_preferences)
        r.add_post("/eth/v1/validator/builder_preferences", self.post_builder_preferences)
        r.add_post("/eth/v1/validator/liveness/{epoch}", self.post_liveness)
        r.add_get(
            "/eth/v1/validator/execution_payload_envelopes/{slot}/{beacon_block_root}",
            self.get_validator_execution_payload_envelope,
        )

        r.add_get("/eth/v1/events", self.get_events)

    # ------------------------------------------------------------------ request parsing

    @staticmethod
    def _uint(value, name: str) -> int:
        try:
            result = int(str(value), 10)
        except (TypeError, ValueError):
            raise ApiError(400, f"Invalid {name}: {value}")
        if not 0 <= result < 2**64:
            raise ApiError(400, f"Invalid {name}: {value}")
        return result

    @classmethod
    def _query_uint(cls, request: web.Request, name: str, required: bool = False) -> Optional[int]:
        if name not in request.query:
            if required:
                raise ApiError(400, f"Missing {name} parameter")
            return None
        return cls._uint(request.query[name], name)

    @staticmethod
    def _hex_bytes(value, name: str, length: Optional[int] = None) -> bytes:
        try:
            if not isinstance(value, str) or not value.startswith("0x"):
                raise ValueError
            raw = bytes.fromhex(value[2:])
        except ValueError:
            raise ApiError(400, f"Invalid {name}: {value}")
        if length is not None and len(raw) != length:
            raise ApiError(400, f"Invalid {name}: {value}")
        return raw

    @staticmethod
    def _query_list(request: web.Request, name: str) -> list[str]:
        return [v.strip() for param in request.query.getall(name, []) for v in param.split(",") if v.strip()]

    @staticmethod
    async def _json_body(request: web.Request, default=None):
        if not request.can_read_body:
            return default
        raw = await request.read()
        if not raw.strip():
            return default
        try:
            return json.loads(raw)
        except ValueError as e:
            raise ApiError(400, f"Invalid JSON body: {e}")

    async def _json_list_body(self, request: web.Request, required: bool = True) -> list:
        body = await self._json_body(request, None)
        if body is None:
            if required:
                raise ApiError(400, "Missing request body")
            return []
        if not isinstance(body, list):
            raise ApiError(400, "Request body must be an array")
        return body

    @staticmethod
    def _content_type(request: web.Request) -> str:
        return request.headers.get("Content-Type", "application/json").split(";")[0].strip().lower()

    async def _decode_body(self, request: web.Request, typ, list_body: bool = False):
        """Decode a JSON or SSZ request body into ``typ`` (or a list of it)."""
        from ..spec.types.base import List

        content_type = self._content_type(request)
        try:
            if content_type == "application/octet-stream":
                raw = await request.read()
                if list_body:
                    return list(List[typ, 2**20].decode_bytes(raw))
                return typ.decode_bytes(raw)
            if content_type not in ("application/json", ""):
                raise ApiError(415, f"Unsupported Content-Type: {content_type}")
            body = await self._json_body(request, None)
            if body is None:
                raise ValueError("missing request body")
            if list_body:
                if not isinstance(body, list):
                    raise ValueError("request body must be an array")
                return [from_json(typ, item) for item in body]
            return from_json(typ, body)
        except ApiError:
            raise
        except Exception as e:
            raise ApiError(400, f"Invalid request body: {e}")

    @staticmethod
    def _consensus_version(request: web.Request, required: bool = True) -> Optional[str]:
        version = request.headers.get("Eth-Consensus-Version", "").strip().lower()
        if not version:
            if required:
                raise ApiError(400, "Missing Eth-Consensus-Version header")
            return None
        if version not in FORK_NAMES:
            raise ApiError(400, f"Invalid Eth-Consensus-Version header: {version}")
        return version

    @staticmethod
    def _parse_epoch(request: web.Request) -> int:
        return BeaconAPI._uint(request.match_info["epoch"], "epoch")

    async def _validator_index_body(self, request: web.Request) -> list[int]:
        body = await self._json_list_body(request)
        return [self._uint(i, "validator index") for i in body]

    # ------------------------------------------------------------------ chain lookups

    @staticmethod
    def _spe() -> int:
        from ..spec.constants import SLOTS_PER_EPOCH
        return SLOTS_PER_EPOCH()

    def _head_state(self):
        state = self.node.state
        if state is None:
            raise ApiError(503, "Beacon node is currently syncing")
        return state

    def _finalized_slot(self) -> int:
        state = self.node.state
        return int(state.finalized_checkpoint.epoch) * self._spe() if state is not None else 0

    def _genesis_root(self) -> Optional[bytes]:
        return getattr(self.node, "_genesis_block_root", None)

    def _block_root_at_or_before(self, slot: int) -> Optional[bytes]:
        """Canonical block root at ``slot``, or the latest canonical block before it."""
        from ..spec.constants import SLOTS_PER_HISTORICAL_ROOT

        node = self.node
        if node.head_root is None or node.state is None:
            return None
        if slot >= int(node.head_slot):
            return bytes(node.head_root)
        state = node.state
        if slot < int(state.slot) <= slot + SLOTS_PER_HISTORICAL_ROOT():
            return bytes(state.block_roots[slot % SLOTS_PER_HISTORICAL_ROOT()])
        root = bytes(node.head_root)
        while True:
            block = node.store.get_block(root)
            if block is None:
                return None
            if int(block.message.slot) <= slot:
                return root
            root = bytes(block.message.parent_root)

    def _canonical_root_at_slot(self, slot: int) -> Optional[bytes]:
        root = self._block_root_at_or_before(slot)
        if root is None:
            return None
        block = self.node.store.get_block(root)
        if block is None or int(block.message.slot) != slot:
            return None
        return root

    def _is_canonical(self, root: bytes, slot: int) -> bool:
        return self._block_root_at_or_before(slot) == root

    def _meta(self, finalized: bool) -> dict:
        return {"execution_optimistic": False, "finalized": bool(finalized)}

    def _block(self, block_id: str) -> tuple[bytes, object, bool, bool]:
        """Resolve a block_id to (root, signed_block, canonical, finalized)."""
        node = self.node
        state = self._head_state()
        if block_id == "head":
            root = node.head_root
        elif block_id == "genesis":
            root = self._genesis_root()
        elif block_id in ("finalized", "justified"):
            checkpoint = state.finalized_checkpoint if block_id == "finalized" else state.current_justified_checkpoint
            root = bytes(checkpoint.root)
            if root == b"\x00" * 32:
                root = self._genesis_root()
        elif block_id.startswith("0x"):
            root = self._hex_bytes(block_id, "block ID", 32)
        else:
            try:
                slot = int(block_id, 10)
            except ValueError:
                raise ApiError(400, f"Invalid block ID: {block_id}")
            root = self._canonical_root_at_slot(slot)
        signed_block = node.store.get_block(bytes(root)) if root is not None else None
        if signed_block is None:
            raise ApiError(404, "Block not found")
        root = bytes(root)
        slot = int(signed_block.message.slot)
        canonical = self._is_canonical(root, slot)
        return root, signed_block, canonical, canonical and slot <= self._finalized_slot()

    @staticmethod
    def _advance(state, slot: int):
        from ..spec.state_transition import process_slots
        if int(state.slot) >= slot:
            return state
        return process_slots(state.copy(), slot)

    def _state_at_block(self, root: bytes):
        node = self.node
        if node.head_root is not None and root == bytes(node.head_root):
            return node.state
        if root == self._genesis_root():
            return node._genesis_state
        return node.store.get_state(root)

    async def _state_at_slot(self, slot: int):
        node = self.node
        state = node.state
        if state is not None and int(state.slot) == slot:
            return state
        root = self._block_root_at_or_before(slot)
        base = self._state_at_block(root) if root is not None else None
        if base is None:
            return None
        return await asyncio.to_thread(self._advance, base, slot)

    async def _state(self, state_id: str) -> tuple[object, bool]:
        """Resolve a state_id to (state, finalized)."""
        from ..crypto import hash_tree_root

        node = self.node
        head = self._head_state()
        finalized_slot = self._finalized_slot()
        if state_id == "head":
            return head, int(head.slot) <= finalized_slot
        if state_id == "genesis":
            return node._genesis_state, True
        if state_id in ("finalized", "justified"):
            checkpoint = head.finalized_checkpoint if state_id == "finalized" else head.current_justified_checkpoint
            root = bytes(checkpoint.root)
            base = node._genesis_state if root == b"\x00" * 32 else self._state_at_block(root)
            if base is None:
                raise ApiError(404, "State not found")
            state = await asyncio.to_thread(self._advance, base, int(checkpoint.epoch) * self._spe())
            return state, state_id == "finalized" or int(state.slot) <= finalized_slot
        if state_id.startswith("0x"):
            root = self._hex_bytes(state_id, "state ID", 32)
            if hash_tree_root(head) == root:
                return head, int(head.slot) <= finalized_slot
            state = node.store.get_state(root)
            if state is None:
                raise ApiError(404, "State not found")
            return state, int(state.slot) <= finalized_slot
        try:
            slot = int(state_id, 10)
        except ValueError:
            raise ApiError(400, f"Invalid state ID: {state_id}")
        if slot > int(head.slot):
            raise ApiError(404, "State not found")
        state = await self._state_at_slot(slot)
        if state is None:
            raise ApiError(404, "State not found")
        return state, slot <= finalized_slot

    async def _request_state(self, request: web.Request) -> tuple[object, bool]:
        return await self._state(request.match_info["state_id"])

    @staticmethod
    def _state_epoch(state) -> int:
        return int(state.slot) // BeaconAPI._spe()

    @staticmethod
    def _validator_status(validator, epoch: int) -> str:
        from ..spec.constants import FAR_FUTURE_EPOCH

        if int(validator.activation_epoch) > epoch:
            if int(validator.activation_eligibility_epoch) == FAR_FUTURE_EPOCH:
                return "pending_initialized"
            return "pending_queued"
        if epoch < int(validator.exit_epoch):
            if int(validator.exit_epoch) == FAR_FUTURE_EPOCH:
                return "active_ongoing"
            return "active_slashed" if validator.slashed else "active_exiting"
        if epoch < int(validator.withdrawable_epoch):
            return "exited_slashed" if validator.slashed else "exited_unslashed"
        return "withdrawal_possible" if int(validator.effective_balance) != 0 else "withdrawal_done"

    @staticmethod
    def _status_matches(status: str, wanted: set[str]) -> bool:
        return not wanted or status in wanted or status.split("_")[0] in wanted

    def _validator_indices(self, state, ids) -> list[int]:
        """Indices of the validators named by ``ids`` (indices or pubkeys) that
        exist in ``state``; all validators when ``ids`` is empty."""
        count = len(state.validators)
        if not ids:
            return list(range(count))
        pubkeys: Optional[dict[bytes, int]] = None
        indices: list[int] = []
        seen: set[int] = set()
        for raw in ids:
            raw = str(raw)
            if raw.startswith("0x"):
                pubkey = self._hex_bytes(raw, "validator ID", 48)
                if pubkeys is None:
                    pubkeys = {bytes(v.pubkey): i for i, v in enumerate(state.validators)}
                index = pubkeys.get(pubkey)
            else:
                index = self._uint(raw, "validator ID")
                if index >= count:
                    index = None
            if index is not None and index not in seen:
                seen.add(index)
                indices.append(index)
        return indices

    @staticmethod
    def _parse_statuses(statuses) -> set[str]:
        wanted = set(statuses or [])
        unknown = wanted - VALIDATOR_STATUSES
        if unknown:
            raise ApiError(400, f"Invalid validator status: {sorted(unknown)[0]}")
        return wanted

    def _validators_data(self, state, ids, statuses: set[str]) -> list[dict]:
        epoch = self._state_epoch(state)
        data = []
        for index in self._validator_indices(state, ids):
            validator = state.validators[index]
            status = self._validator_status(validator, epoch)
            if not self._status_matches(status, statuses):
                continue
            data.append({
                "index": str(index),
                "balance": str(int(state.balances[index])),
                "status": status,
                "validator": to_json(validator),
            })
        return data

    def _ssz_or_json(self, request: web.Request, ssz_bytes, data, headers: Optional[dict] = None, **extra) -> web.Response:
        if self._wants_ssz(request):
            return web.Response(body=ssz_bytes(), content_type="application/octet-stream", headers=headers)
        return web.json_response({**extra, "data": data}, headers=headers)

    # ------------------------------------------------------------------ node

    def _sync_status(self) -> tuple[int, int, bool]:
        node = self.node
        head_slot = int(node.head_slot)
        wall_slot = node._wall_slot() if node._genesis_time else head_slot
        distance = max(0, wall_slot - head_slot)
        return head_slot, distance, node.state is None or distance > self._spe()

    async def get_health(self, request: web.Request) -> web.Response:
        """GET /eth/v1/node/health"""
        syncing_status = 206
        if "syncing_status" in request.query:
            try:
                syncing_status = int(request.query["syncing_status"])
            except ValueError:
                return web.Response(status=400)
            if not 100 <= syncing_status <= 599:
                return web.Response(status=400)
        if self.node.state is None:
            return web.Response(status=503)
        _, _, is_syncing = self._sync_status()
        if is_syncing or self.node.engine is None:
            return web.Response(status=syncing_status)
        return web.Response(status=200)

    async def get_syncing(self, request: web.Request) -> web.Response:
        """GET /eth/v1/node/syncing"""
        head_slot, distance, is_syncing = self._sync_status()
        return web.json_response({
            "data": {
                "head_slot": str(head_slot),
                "sync_distance": str(distance),
                "is_syncing": is_syncing,
                "is_optimistic": False,
                "el_offline": self.node.engine is None,
            }
        })

    def _peer_entries(self) -> list[dict]:
        gossip = self.node.beacon_gossip
        if gossip is None:
            return []
        peers = []
        for peer_info in gossip.connected_peers():
            addrs = peer_info.get("addrs", [])
            entry = {
                "peer_id": peer_info.get("peer_id", ""),
                "enr": peer_info.get("enr") or None,
                "last_seen_p2p_address": addrs[0] if addrs else "",
                "state": "connected",
                "direction": peer_info.get("direction", "outbound"),
            }
            if peer_info.get("agent_version"):
                entry["agent_version"] = peer_info["agent_version"]
            if peer_info.get("score") is not None:
                entry["score"] = peer_info["score"]
            if peer_info.get("downscore_reasons"):
                entry["downscore_reasons"] = peer_info["downscore_reasons"]
            peers.append(entry)
        return peers

    async def get_peers(self, request: web.Request) -> web.Response:
        """GET /eth/v1/node/peers"""
        states = set(self._query_list(request, "state"))
        directions = set(self._query_list(request, "direction"))
        if states - {"disconnected", "connecting", "connected", "disconnecting"}:
            raise ApiError(400, f"Invalid peer state: {sorted(states)[0]}")
        if directions - {"inbound", "outbound"}:
            raise ApiError(400, f"Invalid peer direction: {sorted(directions)[0]}")
        peers = [
            p for p in self._peer_entries()
            if (not states or p["state"] in states) and (not directions or p["direction"] in directions)
        ]
        return web.json_response({"data": peers, "meta": {"count": len(peers)}})

    async def get_peer(self, request: web.Request) -> web.Response:
        """GET /eth/v1/node/peers/{peer_id}"""
        peer_id = request.match_info["peer_id"]
        if not peer_id or set(peer_id) - BASE58_ALPHABET:
            raise ApiError(400, f"Invalid peer ID: {peer_id}")
        for peer in self._peer_entries():
            if peer["peer_id"] == peer_id:
                return web.json_response({"data": peer})
        raise ApiError(404, "Peer not found")

    async def get_peer_count(self, request: web.Request) -> web.Response:
        """GET /eth/v1/node/peer_count"""
        return web.json_response({
            "data": {
                "disconnected": "0",
                "connecting": "0",
                "connected": str(len(self._peer_entries())),
                "disconnecting": "0",
            }
        })

    # ------------------------------------------------------------------ beacon states

    async def get_state_root(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/states/{state_id}/root"""
        from ..crypto import hash_tree_root

        state, finalized = await self._request_state(request)
        root = await asyncio.to_thread(hash_tree_root, state)
        return web.json_response({**self._meta(finalized), "data": {"root": "0x" + bytes(root).hex()}})

    async def get_state_fork(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/states/{state_id}/fork"""
        state, finalized = await self._request_state(request)
        return web.json_response({**self._meta(finalized), "data": to_json(state.fork)})

    async def get_finality_checkpoints(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/states/{state_id}/finality_checkpoints"""
        state, finalized = await self._request_state(request)
        return web.json_response({
            **self._meta(finalized),
            "data": {
                "previous_justified": to_json(state.previous_justified_checkpoint),
                "current_justified": to_json(state.current_justified_checkpoint),
                "finalized": to_json(state.finalized_checkpoint),
            },
        })

    async def get_validators(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/states/{state_id}/validators"""
        ids = self._query_list(request, "id")
        if len(ids) > MAX_QUERY_IDS:
            raise ApiError(414, "Too many validator IDs in request")
        statuses = self._parse_statuses(self._query_list(request, "status"))
        state, finalized = await self._request_state(request)
        data = await asyncio.to_thread(self._validators_data, state, ids, statuses)
        return web.json_response({**self._meta(finalized), "data": data})

    async def post_validators(self, request: web.Request) -> web.Response:
        """POST /eth/v1/beacon/states/{state_id}/validators"""
        body = await self._json_body(request, {}) or {}
        if not isinstance(body, dict):
            raise ApiError(400, "Request body must be an object")
        ids = body.get("ids") or []
        statuses = self._parse_statuses(body.get("statuses") or [])
        if not isinstance(ids, list):
            raise ApiError(400, "ids must be an array")
        state, finalized = await self._request_state(request)
        data = await asyncio.to_thread(self._validators_data, state, ids, statuses)
        return web.json_response({**self._meta(finalized), "data": data})

    async def get_validator(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/states/{state_id}/validators/{validator_id}"""
        validator_id = request.match_info["validator_id"]
        state, finalized = await self._request_state(request)
        data = await asyncio.to_thread(self._validators_data, state, [validator_id], set())
        if not data:
            raise ApiError(404, "Validator not found")
        return web.json_response({**self._meta(finalized), "data": data[0]})

    def _balances_response(self, request: web.Request, state, finalized: bool, ids) -> web.Response:
        from ..spec.types.base import Container, List, uint64
        from ..spec.constants import VALIDATOR_REGISTRY_LIMIT

        class ValidatorBalance(Container):
            index: uint64
            balance: uint64

        indices = self._validator_indices(state, ids)
        return self._ssz_or_json(
            request,
            lambda: List[ValidatorBalance, VALIDATOR_REGISTRY_LIMIT](
                *[ValidatorBalance(index=i, balance=int(state.balances[i])) for i in indices]
            ).encode_bytes(),
            [{"index": str(i), "balance": str(int(state.balances[i]))} for i in indices],
            **self._meta(finalized),
        )

    async def get_validator_balances(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/states/{state_id}/validator_balances"""
        ids = self._query_list(request, "id")
        if len(ids) > MAX_QUERY_IDS:
            raise ApiError(414, "Too many validator IDs in request")
        state, finalized = await self._request_state(request)
        return await asyncio.to_thread(self._balances_response, request, state, finalized, ids)

    async def post_validator_balances(self, request: web.Request) -> web.Response:
        """POST /eth/v1/beacon/states/{state_id}/validator_balances"""
        ids = await self._json_list_body(request, required=False)
        state, finalized = await self._request_state(request)
        return await asyncio.to_thread(self._balances_response, request, state, finalized, ids)

    async def post_validator_identities(self, request: web.Request) -> web.Response:
        """POST /eth/v1/beacon/states/{state_id}/validator_identities"""
        from ..spec.types.base import Container, List, uint64, BLSPubkey
        from ..spec.constants import VALIDATOR_REGISTRY_LIMIT

        class ValidatorIdentity(Container):
            index: uint64
            pubkey: BLSPubkey
            activation_epoch: uint64

        ids = await self._json_list_body(request, required=False)
        state, finalized = await self._request_state(request)
        indices = await asyncio.to_thread(self._validator_indices, state, ids)
        identities = [
            ValidatorIdentity(
                index=i,
                pubkey=state.validators[i].pubkey,
                activation_epoch=state.validators[i].activation_epoch,
            )
            for i in indices
        ]
        return self._ssz_or_json(
            request,
            lambda: List[ValidatorIdentity, VALIDATOR_REGISTRY_LIMIT](*identities).encode_bytes(),
            [to_json(i) for i in identities],
            **self._meta(finalized),
        )

    async def post_state_builders(self, request: web.Request) -> web.Response:
        """POST /eth/v1/beacon/states/{state_id}/builders"""
        from ..spec.constants import FAR_FUTURE_EPOCH
        from ..spec.state_transition.helpers.predicates import is_active_builder

        body = await self._json_body(request, {}) or {}
        if not isinstance(body, dict):
            raise ApiError(400, "Request body must be an object")
        ids = body.get("ids") or []
        statuses = set(body.get("statuses") or [])
        if statuses - {"pending", "active", "exited"}:
            raise ApiError(400, f"Invalid builder status: {sorted(statuses)[0]}")
        state, finalized = await self._request_state(request)
        if not hasattr(state, "builders"):
            raise ApiError(400, "State is prior to Gloas")
        wanted_indices, wanted_pubkeys = set(), set()
        for i in ids:
            if isinstance(i, str) and i.startswith("0x"):
                wanted_pubkeys.add(self._hex_bytes(i, "builder ID", 48))
            else:
                wanted_indices.add(self._uint(i, "builder ID"))

        data = []
        for index, builder in enumerate(state.builders):
            if ids and index not in wanted_indices and bytes(builder.pubkey) not in wanted_pubkeys:
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
        return web.json_response({**self._meta(finalized), "data": data})

    async def get_committees(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/states/{state_id}/committees"""
        from ..spec.constants import MIN_SEED_LOOKAHEAD
        from ..spec.state_transition.helpers.beacon_committee import (
            get_beacon_committee,
            get_committee_count_per_slot,
        )

        epoch = self._query_uint(request, "epoch")
        index = self._query_uint(request, "index")
        slot = self._query_uint(request, "slot")
        state, finalized = await self._request_state(request)
        state_epoch = self._state_epoch(state)
        if epoch is None:
            epoch = state_epoch
        if not max(0, state_epoch - 1) <= epoch <= state_epoch + MIN_SEED_LOOKAHEAD:
            raise ApiError(400, f"Epoch {epoch} is outside the committee window of the state")
        if slot is not None and slot // self._spe() != epoch:
            raise ApiError(400, f"Slot {slot} is not in epoch {epoch}")

        def compute():
            committees_per_slot = get_committee_count_per_slot(state, epoch)
            if index is not None and index >= committees_per_slot:
                raise ApiError(400, f"Committee index {index} out of range")
            data = []
            for s in range(epoch * self._spe(), (epoch + 1) * self._spe()):
                if slot is not None and s != slot:
                    continue
                for i in range(committees_per_slot):
                    if index is not None and i != index:
                        continue
                    data.append({
                        "index": str(i),
                        "slot": str(s),
                        "validators": [str(int(v)) for v in get_beacon_committee(state, s, i)],
                    })
            return data

        data = await asyncio.to_thread(compute)
        return web.json_response({**self._meta(finalized), "data": data})

    def _sync_committee_for_epoch(self, state, epoch: int):
        from ..spec.constants import EPOCHS_PER_SYNC_COMMITTEE_PERIOD

        if not hasattr(state, "current_sync_committee"):
            raise ApiError(400, "Sync committees are not available before Altair")
        period = epoch // EPOCHS_PER_SYNC_COMMITTEE_PERIOD()
        state_period = self._state_epoch(state) // EPOCHS_PER_SYNC_COMMITTEE_PERIOD()
        if period == state_period:
            return state.current_sync_committee
        if period == state_period + 1:
            return state.next_sync_committee
        raise ApiError(400, f"Epoch {epoch} is outside the sync committee periods of the state")

    async def get_sync_committees(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/states/{state_id}/sync_committees"""
        from ..spec.constants import SYNC_COMMITTEE_SIZE, SYNC_COMMITTEE_SUBNET_COUNT

        epoch = self._query_uint(request, "epoch")
        state, finalized = await self._request_state(request)
        committee = self._sync_committee_for_epoch(state, self._state_epoch(state) if epoch is None else epoch)

        def compute():
            pubkeys = {bytes(v.pubkey): i for i, v in enumerate(state.validators)}
            return [str(pubkeys[bytes(pk)]) for pk in committee.pubkeys]

        validators = await asyncio.to_thread(compute)
        size = SYNC_COMMITTEE_SIZE() // SYNC_COMMITTEE_SUBNET_COUNT
        return web.json_response({
            **self._meta(finalized),
            "data": {
                "validators": validators,
                "validator_aggregates": [validators[i * size:(i + 1) * size] for i in range(SYNC_COMMITTEE_SUBNET_COUNT)],
            },
        })

    async def get_state_ptc(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/states/{state_id}/ptc"""
        from ..spec.constants import MIN_SEED_LOOKAHEAD, PTC_SIZE
        from ..spec.state_transition.helpers.ptc import get_ptc
        from ..spec.types.base import Container, Vector, uint64

        slot = self._query_uint(request, "slot")
        state, finalized = await self._request_state(request)
        if not hasattr(state, "builders"):
            raise ApiError(400, "State is prior to Gloas")
        if slot is None:
            slot = int(state.slot)
        epoch = slot // self._spe()
        state_epoch = self._state_epoch(state)
        if epoch < get_network_config().gloas_fork_epoch:
            raise ApiError(400, "Slot is prior to Gloas")
        if not max(0, state_epoch - 1) <= epoch <= state_epoch + MIN_SEED_LOOKAHEAD:
            raise ApiError(400, "Slot is outside the PTC window of the state")
        ptc = [int(v) for v in get_ptc(state, slot)]

        class StatePtc(Container):
            slot: uint64
            validators: Vector[uint64, PTC_SIZE()]

        return self._ssz_or_json(
            request,
            lambda: StatePtc(slot=slot, validators=ptc).encode_bytes(),
            {"slot": str(slot), "validators": [str(v) for v in ptc]},
            **self._meta(finalized),
        )

    async def get_randao(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/states/{state_id}/randao"""
        from ..spec.constants import EPOCHS_PER_HISTORICAL_VECTOR

        epoch = self._query_uint(request, "epoch")
        state, finalized = await self._request_state(request)
        state_epoch = self._state_epoch(state)
        if epoch is None:
            epoch = state_epoch
        if not max(0, state_epoch - EPOCHS_PER_HISTORICAL_VECTOR() + 1) <= epoch <= state_epoch:
            raise ApiError(400, f"Epoch {epoch} is out of range for the state's RANDAO mixes")
        mix = bytes(state.randao_mixes[epoch % EPOCHS_PER_HISTORICAL_VECTOR()])
        return web.json_response({**self._meta(finalized), "data": {"randao": "0x" + mix.hex()}})

    def _state_field_handler(self, field: str):
        since = {
            "pending_consolidations": "Electra",
            "pending_deposits": "Electra",
            "pending_partial_withdrawals": "Electra",
            "proposer_lookahead": "Fulu",
            "builder_pending_withdrawals": "Gloas",
            "builder_pending_payments": "Gloas",
        }[field]

        async def handler(request: web.Request) -> web.Response:
            state, finalized = await self._request_state(request)
            if not hasattr(state, field):
                raise ApiError(400, f"State is prior to {since}")
            value = getattr(state, field)
            return self._versioned_response(
                request, self._fork_at_slot(int(state.slot)), value.encode_bytes(), to_json(value),
                **self._meta(finalized),
            )

        handler.__doc__ = f"GET /eth/v1/beacon/states/{{state_id}}/{field}"
        return handler

    async def get_debug_state(self, request: web.Request) -> web.Response:
        """GET /eth/v2/debug/beacon/states/{state_id}"""
        state, finalized = await self._request_state(request)
        version = self._fork_at_slot(int(state.slot))
        headers = {"Eth-Consensus-Version": version}
        if self._wants_ssz(request):
            body = await asyncio.to_thread(lambda: bytes(state.encode_bytes()))
            return web.Response(body=body, content_type="application/octet-stream", headers=headers)
        data = await asyncio.to_thread(to_json, state)
        return web.json_response({"version": version, **self._meta(finalized), "data": data}, headers=headers)

    # ------------------------------------------------------------------ blocks

    def _header_data(self, root: bytes, signed_block, canonical: bool) -> dict:
        from ..crypto import hash_tree_root

        block = signed_block.message
        return {
            "root": "0x" + root.hex(),
            "canonical": canonical,
            "header": {
                "message": {
                    "slot": str(int(block.slot)),
                    "proposer_index": str(int(block.proposer_index)),
                    "parent_root": "0x" + bytes(block.parent_root).hex(),
                    "state_root": "0x" + bytes(block.state_root).hex(),
                    "body_root": "0x" + bytes(hash_tree_root(block.body)).hex(),
                },
                "signature": "0x" + bytes(signed_block.signature).hex(),
            },
        }

    async def get_headers(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/headers"""
        from ..crypto import hash_tree_root

        slot = self._query_uint(request, "slot")
        parent_root = request.query.get("parent_root")
        parent_root = self._hex_bytes(parent_root, "parent_root", 32) if parent_root is not None else None
        store = self.node.store
        candidates: dict[bytes, object] = {}
        if parent_root is not None:
            for block in store.get_blocks_by_parent(parent_root):
                candidates[bytes(hash_tree_root(block.message))] = block
        elif slot is not None:
            root = self._canonical_root_at_slot(slot)
            if root is not None:
                candidates[root] = store.get_block(root)
            other = store.get_block_by_slot(slot)
            if other is not None and int(other.message.slot) == slot:
                candidates.setdefault(bytes(hash_tree_root(other.message)), other)
        else:
            root, block, _, _ = self._block("head")
            candidates[root] = block
        data = []
        all_finalized = True
        for root, block in candidates.items():
            if block is None or (slot is not None and int(block.message.slot) != slot):
                continue
            canonical = self._is_canonical(root, int(block.message.slot))
            all_finalized &= canonical and int(block.message.slot) <= self._finalized_slot()
            data.append(self._header_data(root, block, canonical))
        return web.json_response({**self._meta(bool(data) and all_finalized), "data": data})

    async def get_header(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/headers/{block_id}"""
        root, signed_block, canonical, finalized = self._block(request.match_info["block_id"])
        return web.json_response({**self._meta(finalized), "data": self._header_data(root, signed_block, canonical)})

    async def get_block(self, request: web.Request) -> web.Response:
        """GET /eth/v2/beacon/blocks/{block_id}"""
        root, signed_block, _, finalized = self._block(request.match_info["block_id"])
        return self._versioned_response(
            request,
            self._fork_at_slot(int(signed_block.message.slot)),
            signed_block.encode_bytes(),
            to_json(signed_block),
            **self._meta(finalized),
        )

    async def get_block_root(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/blocks/{block_id}/root"""
        root, _, _, finalized = self._block(request.match_info["block_id"])
        return web.json_response({**self._meta(finalized), "data": {"root": "0x" + root.hex()}})

    async def get_block_attestations(self, request: web.Request) -> web.Response:
        """GET /eth/v2/beacon/blocks/{block_id}/attestations"""
        _, signed_block, _, finalized = self._block(request.match_info["block_id"])
        version = self._fork_at_slot(int(signed_block.message.slot))
        return web.json_response(
            {
                "version": version,
                **self._meta(finalized),
                "data": [to_json(a) for a in signed_block.message.body.attestations],
            },
            headers={"Eth-Consensus-Version": version},
        )

    def _blinded_block_types(self, version: str) -> tuple:
        """(SignedBlindedBeaconBlock, ExecutionPayloadHeader) types of a
        Bellatrix..Fulu fork, derived from its SignedBeaconBlock type."""
        from ..spec.types.base import Container
        from ..spec.types.bellatrix import ExecutionPayloadHeaderBellatrix
        from ..spec.types.capella import ExecutionPayloadHeaderCapella
        from ..spec.types.deneb import ExecutionPayloadHeader

        if version not in self._blinded_types:
            header_type = {
                "bellatrix": ExecutionPayloadHeaderBellatrix,
                "capella": ExecutionPayloadHeaderCapella,
            }.get(version, ExecutionPayloadHeader)
            signed_type = self._signed_block_type(version)
            block_type = signed_type.fields()["message"]
            body_fields = {
                ("execution_payload_header" if name == "execution_payload" else name): (
                    header_type if name == "execution_payload" else typ
                )
                for name, typ in block_type.fields()["body"].fields().items()
            }
            body = type("BlindedBeaconBlockBody", (Container,), {"__annotations__": body_fields})
            block = type(
                "BlindedBeaconBlock", (Container,),
                {"__annotations__": {**block_type.fields(), "body": body}},
            )
            signed = type(
                "SignedBlindedBeaconBlock", (Container,),
                {"__annotations__": {"message": block, "signature": signed_type.fields()["signature"]}},
            )
            self._blinded_types[version] = (signed, header_type)
        return self._blinded_types[version]

    def _blind(self, signed_block, version: str):
        from ..crypto import hash_tree_root

        signed_type, header_type = self._blinded_block_types(version)
        block = signed_block.message
        payload = block.body.execution_payload
        header_values = {}
        for name in header_type.fields():
            if name in ("transactions_root", "withdrawals_root"):
                header_values[name] = hash_tree_root(getattr(payload, name[:-len("_root")]))
            else:
                header_values[name] = getattr(payload, name)
        body_type = signed_type.fields()["message"].fields()["body"]
        body_values = {
            name: (header_type(**header_values) if name == "execution_payload_header" else getattr(block.body, name))
            for name in body_type.fields()
        }
        message_type = signed_type.fields()["message"]
        message = message_type(**{
            name: (body_type(**body_values) if name == "body" else getattr(block, name))
            for name in message_type.fields()
        })
        return signed_type(message=message, signature=signed_block.signature)

    async def get_blinded_block(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/blinded_blocks/{block_id}"""
        _, signed_block, _, finalized = self._block(request.match_info["block_id"])
        version = self._fork_at_slot(int(signed_block.message.slot))
        if version in ("gloas", "heze"):
            raise ApiError(400, f"Blinded blocks are not available from Gloas (block fork {version})")
        if hasattr(signed_block.message.body, "execution_payload"):
            signed_block = await asyncio.to_thread(self._blind, signed_block, version)
        return self._versioned_response(
            request, version, signed_block.encode_bytes(), to_json(signed_block), **self._meta(finalized)
        )

    async def post_blinded_block_v2(self, request: web.Request) -> web.Response:
        """POST /eth/v2/beacon/blinded_blocks"""
        version = self._consensus_version(request)
        if version in ("gloas", "heze"):
            raise ApiError(400, "Gloas blocks have no blinded form, publish them via POST /eth/v2/beacon/blocks")
        if version not in ("phase0", "altair"):
            raise ApiError(400, "Cannot unblind the block: no builder is connected to this node")
        signed_block = await self._decode_body(request, self._signed_block_type(version))
        return await self._publish_block(request, signed_block, None)

    async def post_block_v2(self, request: web.Request) -> web.Response:
        """POST /eth/v2/beacon/blocks"""
        from ..spec.types.base import Container, List
        from ..spec.types.deneb import Blob
        from ..spec.types import KZGProof
        from ..spec.constants import FIELD_ELEMENTS_PER_EXT_BLOB, MAX_BLOB_COMMITMENTS_PER_BLOCK

        version = self._consensus_version(request)
        block_type = self._signed_block_type(version)
        if version in ("deneb", "electra", "fulu"):
            class SignedBlockContents(Container):
                signed_block: block_type
                kzg_proofs: List[KZGProof, FIELD_ELEMENTS_PER_EXT_BLOB * MAX_BLOB_COMMITMENTS_PER_BLOCK]
                blobs: List[Blob, MAX_BLOB_COMMITMENTS_PER_BLOCK]

            contents = await self._decode_body(request, SignedBlockContents)
            signed_block = contents.signed_block
            blobs_bundle = {"blobs": ["0x" + bytes(b).hex() for b in contents.blobs]}
        else:
            signed_block = await self._decode_body(request, block_type)
            blobs_bundle = None
        return await self._publish_block(request, signed_block, blobs_bundle)

    async def _publish_block(self, request: web.Request, signed_block, blobs_bundle: Optional[dict]) -> web.Response:
        from ..crypto import hash_tree_root
        from ..spec.state_transition import state_transition

        validation = request.query.get("broadcast_validation", "gossip")
        if validation not in ("gossip", "consensus", "consensus_and_equivocation"):
            raise ApiError(400, f"Invalid broadcast_validation: {validation}")
        block = signed_block.message
        slot = int(block.slot)
        block_root = bytes(hash_tree_root(block))
        if slot <= self._finalized_slot() and self.node.state is not None and int(self.node.state.finalized_checkpoint.epoch) > 0:
            raise ApiError(400, f"Block slot {slot} is not after the finalized slot")
        if validation == "consensus_and_equivocation":
            other = self.node.store.get_block_by_slot(slot)
            if (
                other is not None
                and int(other.message.slot) == slot
                and int(other.message.proposer_index) == int(block.proposer_index)
                and bytes(hash_tree_root(other.message)) != block_root
            ):
                raise ApiError(400, "Block is an equivocation of a known block")
        if validation != "gossip":
            parent_state = self._state_at_block(bytes(block.parent_root))
            if parent_state is None:
                raise ApiError(400, "Unknown parent state")
            try:
                await asyncio.to_thread(state_transition, parent_state, signed_block, True)
            except Exception as e:
                raise ApiError(400, f"Block failed consensus validation: {e}")

        ssz_bytes = signed_block.encode_bytes()
        try:
            if self.node.beacon_gossip:
                await self._run_on_node_loop(self.node.beacon_gossip.publish_block(ssz_bytes))
                if blobs_bundle and blobs_bundle["blobs"]:
                    await self._run_on_node_loop(self._publish_fulu_columns(signed_block, blobs_bundle))
        except Exception as e:
            logger.error(f"Block gossip publish failed: {e}")
            raise ApiError(500, f"Broadcast failed: {e}")
        try:
            await self._run_on_node_loop(self.node._on_p2p_block(ssz_bytes, "beacon-api"))
        except Exception as e:
            logger.warning(f"Block local integration failed: {e}")
        if self.node.store.get_block(block_root) is None:
            return web.Response(status=202)
        return web.Response(status=200)

    async def get_execution_payload_envelope(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/execution_payload_envelopes/{block_id}"""
        root, signed_block, _, finalized = self._block(request.match_info["block_id"])
        signed_envelope = self.node.store.get_payload(root)
        if signed_envelope is None:
            raise ApiError(404, "Execution payload envelope not found")
        return self._versioned_response(
            request,
            self._fork_at_slot(int(signed_block.message.slot)),
            signed_envelope.encode_bytes(),
            to_json(signed_envelope),
            **self._meta(finalized),
        )

    async def post_execution_payload_bids(self, request: web.Request) -> web.Response:
        """POST /eth/v1/beacon/execution_payload_bids"""
        version = self._consensus_version(request)
        if version not in ("gloas", "heze"):
            raise ApiError(400, "Execution payload bids are only available from Gloas")
        signed = await self._decode_body(request, self._signed_bid_type(version))
        try:
            if self.node.beacon_gossip:
                await self._run_on_node_loop(
                    self.node.beacon_gossip.publish_execution_payload_bid(signed.encode_bytes())
                )
        except Exception as e:
            logger.error(f"Bid gossip publish failed: {e}")
            raise ApiError(500, f"Broadcast failed: {e}")
        await self.emit_execution_payload_bid(signed)
        return web.Response(status=200)

    async def get_blobs(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/blobs/{block_id}"""
        from hashlib import sha256
        from ..spec.types.base import List
        from ..spec.types.deneb import Blob
        from ..spec.constants import MAX_BLOB_COMMITMENTS_PER_BLOCK

        wanted = {
            self._hex_bytes(h, "versioned hash", 32) for h in self._query_list(request, "versioned_hashes")
        }
        root, signed_block, _, finalized = self._block(request.match_info["block_id"])
        body = signed_block.message.body
        if hasattr(body, "signed_execution_payload_bid"):
            commitments = [bytes(c) for c in body.signed_execution_payload_bid.message.blob_kzg_commitments]
        else:
            commitments = [bytes(c) for c in getattr(body, "blob_kzg_commitments", [])]

        blobs = await asyncio.to_thread(self._load_blobs, root, len(commitments))
        if blobs is None:
            raise ApiError(404, "Blobs not available for block")
        if wanted:
            blobs = [b for b, c in zip(blobs, commitments) if b"\x01" + sha256(c).digest()[1:] in wanted]
        return self._ssz_or_json(
            request,
            lambda: List[Blob, MAX_BLOB_COMMITMENTS_PER_BLOCK](*blobs).encode_bytes(),
            ["0x" + b.hex() for b in blobs],
            **self._meta(finalized),
        )

    async def get_data_column_sidecars(self, request: web.Request) -> web.Response:
        """GET /eth/v1/debug/beacon/data_column_sidecars/{block_id}"""
        from ..spec.constants import NUMBER_OF_COLUMNS
        from ..spec.types.base import List
        from ..spec.types.gloas import DataColumnSidecar
        from ..spec.types.fulu import DataColumnSidecar as FuluDataColumnSidecar

        indices = {self._uint(i, "column index") for i in self._query_list(request, "indices")}
        if any(i >= NUMBER_OF_COLUMNS for i in indices):
            raise ApiError(400, "Column index out of range")
        root, signed_block, _, finalized = self._block(request.match_info["block_id"])
        version = self._fork_at_slot(int(signed_block.message.slot))
        if version in ("phase0", "altair", "bellatrix", "capella", "deneb", "electra"):
            raise ApiError(400, "Data column sidecars are only available from Fulu")
        sidecar_type = FuluDataColumnSidecar if version == "fulu" else DataColumnSidecar
        columns = self.node.store.get_data_columns(root)
        sidecars = [
            sidecar_type.decode_bytes(raw)
            for index, raw in sorted(columns.items())
            if not indices or index in indices
        ]
        return self._versioned_response(
            request,
            version,
            List[sidecar_type, NUMBER_OF_COLUMNS](*sidecars).encode_bytes(),
            [to_json(sc) for sc in sidecars],
            **self._meta(finalized),
        )

    # ------------------------------------------------------------------ rewards

    async def get_block_rewards(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/rewards/blocks/{block_id}"""
        from .rewards import compute_block_rewards

        _, signed_block, _, finalized = self._block(request.match_info["block_id"])
        if int(signed_block.message.slot) == 0:
            raise ApiError(400, "Rewards are not available for the genesis block")
        pre_state = self._state_at_block(bytes(signed_block.message.parent_root))
        if pre_state is None:
            raise ApiError(404, "Parent state not available")
        rewards = await asyncio.to_thread(compute_block_rewards, pre_state, signed_block)
        return web.json_response({**self._meta(finalized), "data": {k: str(v) for k, v in rewards.items()}})

    async def post_sync_committee_rewards(self, request: web.Request) -> web.Response:
        """POST /eth/v1/beacon/rewards/sync_committee/{block_id}"""
        from .rewards import compute_sync_committee_rewards

        ids = await self._json_list_body(request, required=False)
        _, signed_block, _, finalized = self._block(request.match_info["block_id"])
        if not hasattr(signed_block.message.body, "sync_aggregate") or int(signed_block.message.slot) == 0:
            raise ApiError(400, "Block has no sync aggregate")
        pre_state = self._state_at_block(bytes(signed_block.message.parent_root))
        if pre_state is None:
            raise ApiError(404, "Parent state not available")
        wanted = set(self._validator_indices(pre_state, ids)) if ids else None
        rewards = await asyncio.to_thread(compute_sync_committee_rewards, pre_state, signed_block, wanted)
        return web.json_response({
            **self._meta(finalized),
            "data": [{"validator_index": str(r["validator_index"]), "reward": str(r["reward"])} for r in rewards],
        })

    async def post_attestation_rewards(self, request: web.Request) -> web.Response:
        """POST /eth/v1/beacon/rewards/attestations/{epoch}"""
        from .rewards import compute_attestation_rewards

        epoch = self._parse_epoch(request)
        ids = await self._json_list_body(request, required=False)
        head = self._head_state()
        if self._state_epoch(head) < epoch + 1:
            raise ApiError(404, f"Attestation rewards for epoch {epoch} are not available yet")
        slot = min((epoch + 2) * self._spe() - 1, int(head.slot))
        state = await self._state_at_slot(slot)
        if state is None:
            raise ApiError(404, "State not available")
        if not hasattr(state, "previous_epoch_participation"):
            raise ApiError(400, "Attestation rewards are not available before Altair")
        wanted = set(self._validator_indices(state, ids)) if ids else None
        rewards = await asyncio.to_thread(compute_attestation_rewards, state, wanted)

        def stringify(entries):
            return [{k: str(v) for k, v in entry.items()} for entry in entries]

        return web.json_response({
            **self._meta(slot <= self._finalized_slot()),
            "data": {
                "ideal_rewards": stringify(rewards["ideal_rewards"]),
                "total_rewards": stringify(rewards["total_rewards"]),
            },
        })

    # ------------------------------------------------------------------ light client

    def _update_light_client(self, block_root: bytes) -> None:
        """Derive light client updates with ``block_root`` as signature block."""
        from .light_client import (
            compute_sync_committee_period_at_slot,
            create_light_client_finality_update,
            create_light_client_optimistic_update,
            create_light_client_update,
            is_better_update,
            is_finality_update,
            lc_fork_name,
        )
        from ..spec.constants import MIN_SYNC_COMMITTEE_PARTICIPANTS

        store = self.node.store
        block = store.get_block(block_root)
        if block is None or int(block.message.slot) == 0 or lc_fork_name(int(block.message.slot)) is None:
            return
        if not hasattr(block.message.body, "sync_aggregate"):
            return
        participants = sum(1 for b in block.message.body.sync_aggregate.sync_committee_bits if b)
        if participants < MIN_SYNC_COMMITTEE_PARTICIPANTS:
            return
        attested_root = bytes(block.message.parent_root)
        attested_block = store.get_block(attested_root)
        attested_state = self._state_at_block(attested_root)
        state = self._state_at_block(block_root)
        if attested_block is None or attested_state is None or state is None:
            return
        if lc_fork_name(int(attested_block.message.slot)) is None:
            return
        finalized_root = bytes(attested_state.finalized_checkpoint.root)
        finalized_block = store.get_block(finalized_root) if finalized_root != b"\x00" * 32 else None
        update = create_light_client_update(state, block, attested_state, attested_block, finalized_block)

        attested_slot = int(update.attested_header.beacon.slot)
        optimistic = self._lc_optimistic_update
        if optimistic is None or attested_slot > int(optimistic.attested_header.beacon.slot):
            self._lc_optimistic_update = create_light_client_optimistic_update(update)
            self.emit("light_client_optimistic_update", self._lc_event(self._lc_optimistic_update))
        if is_finality_update(update):
            finality = self._lc_finality_update
            finalized_slot = int(update.finalized_header.beacon.slot)
            supermajority = participants * 3 > len(block.message.body.sync_aggregate.sync_committee_bits) * 2
            if finality is None:
                newer = True
            else:
                old_slot = int(finality.finalized_header.beacon.slot)
                old_bits = finality.sync_aggregate.sync_committee_bits
                old_supermajority = sum(1 for b in old_bits if b) * 3 > len(old_bits) * 2
                newer = finalized_slot > old_slot or (
                    finalized_slot == old_slot and supermajority and not old_supermajority
                )
            if newer:
                self._lc_finality_update = create_light_client_finality_update(update)
                self.emit("light_client_finality_update", self._lc_event(self._lc_finality_update))
        period = compute_sync_committee_period_at_slot(attested_slot)
        best = self._lc_best_updates.get(period)
        if best is None or is_better_update(update, best):
            self._lc_best_updates[period] = update
            for old in sorted(self._lc_best_updates)[:-MAX_LIGHT_CLIENT_PERIODS]:
                del self._lc_best_updates[old]

    @staticmethod
    def _lc_version(update_or_bootstrap) -> str:
        from .light_client import lc_fork_name

        if hasattr(update_or_bootstrap, "attested_header"):
            header = update_or_bootstrap.attested_header
        else:
            header = update_or_bootstrap.header
        return lc_fork_name(int(header.beacon.slot))

    def _lc_event(self, update) -> dict:
        return {"version": self._lc_version(update), "data": to_json(update)}

    async def get_light_client_bootstrap(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/light_client/bootstrap/{block_root}"""
        from .light_client import create_light_client_bootstrap, lc_fork_name

        root = self._hex_bytes(request.match_info["block_root"], "block root", 32)
        block = self.node.store.get_block(root)
        state = self._state_at_block(root) if block is not None else None
        if block is None or state is None or lc_fork_name(int(block.message.slot)) is None:
            raise ApiError(404, "Light client bootstrap not available for block root")
        try:
            bootstrap = await asyncio.to_thread(create_light_client_bootstrap, state, block)
        except AssertionError:
            raise ApiError(404, "Light client bootstrap not available for block root")
        return self._versioned_response(
            request, self._lc_version(bootstrap), bootstrap.encode_bytes(), to_json(bootstrap)
        )

    async def get_light_client_updates(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/light_client/updates"""
        import struct

        start_period = self._query_uint(request, "start_period", required=True)
        count = min(self._query_uint(request, "count", required=True), MAX_REQUEST_LIGHT_CLIENT_UPDATES)
        updates = [
            self._lc_best_updates[p]
            for p in range(start_period, start_period + count)
            if p in self._lc_best_updates
        ]
        if self._wants_ssz(request):
            chunks = []
            for update in updates:
                epoch = int(update.attested_header.beacon.slot) // self._spe()
                payload = (self.node._fork_digest_for_epoch(epoch) or b"\x00" * 4) + bytes(update.encode_bytes())
                chunks.append(struct.pack("<Q", len(payload)) + payload)
            return web.Response(body=b"".join(chunks), content_type="application/octet-stream")
        return web.json_response([self._lc_event(u) for u in updates])

    def _lc_latest(self, request: web.Request, update) -> web.Response:
        if update is None:
            raise ApiError(404, "No light client update available")
        return self._versioned_response(request, self._lc_version(update), update.encode_bytes(), to_json(update))

    async def get_light_client_finality_update(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/light_client/finality_update"""
        return self._lc_latest(request, self._lc_finality_update)

    async def get_light_client_optimistic_update(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/light_client/optimistic_update"""
        return self._lc_latest(request, self._lc_optimistic_update)

    # ------------------------------------------------------------------ pools

    @staticmethod
    def _indexed_failures(failures: list[dict], message: str) -> None:
        if failures:
            raise ApiError(400, message, failures=failures)

    async def get_pool_attestations(self, request: web.Request) -> web.Response:
        """GET /eth/v2/beacon/pool/attestations"""
        slot = self._query_uint(request, "slot")
        committee_index = self._query_uint(request, "committee_index")
        attestations = await self._run_on_node_loop_fn(
            self.node.attestation_pool.get_attestations, slot, committee_index
        )
        version = self._fork_at_slot(slot if slot is not None else self.node.head_slot)
        return web.json_response(
            {"version": version, "data": [to_json(a) for a in attestations]},
            headers={"Eth-Consensus-Version": version},
        )

    def _attestation_subnet(self, slot: int, committee_index: int) -> int:
        from ..spec.constants import ATTESTATION_SUBNET_COUNT
        from ..spec.state_transition.helpers.beacon_committee import get_committee_count_per_slot

        committees_per_slot = get_committee_count_per_slot(self._head_state(), slot // self._spe())
        return (committees_per_slot * (slot % self._spe()) + committee_index) % ATTESTATION_SUBNET_COUNT()

    async def post_pool_attestations(self, request: web.Request) -> web.Response:
        """POST /eth/v2/beacon/pool/attestations"""
        from ..spec.types.electra import SingleAttestation

        version = self._consensus_version(request)
        if version in ("phase0", "altair", "bellatrix", "capella", "deneb"):
            raise ApiError(400, f"Attestation submission is not supported for {version}")
        attestations = await self._decode_body(request, SingleAttestation, list_body=True)
        failures = []
        for i, single in enumerate(attestations):
            try:
                subnet = self._attestation_subnet(int(single.data.slot), int(single.committee_index))
                await self._run_on_node_loop(self.node.on_single_attestation(single, subnet))
            except ValueError as e:
                failures.append({"index": i, "message": str(e)})
        self._indexed_failures(failures, "Some attestations failed validation")
        return web.Response(status=200)

    async def get_pool_payload_attestations(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/pool/payload_attestations"""
        from ..spec.types.base import List
        from ..spec.types.gloas import PayloadAttestation

        slot = self._query_uint(request, "slot")
        aggregates = await self._run_on_node_loop_fn(self.node.payload_attestation_pool.get_aggregates, slot)
        version = self._fork_at_slot(slot if slot is not None else self.node.head_slot)
        return self._versioned_response(
            request,
            version,
            List[PayloadAttestation, 2**20](*aggregates).encode_bytes(),
            [to_json(a) for a in aggregates],
        )

    async def _submit_operations(self, kind: str, ops: list) -> None:
        failures = []
        for i, op in enumerate(ops):
            try:
                await self._run_on_node_loop(self.node.submit_operation(kind, op, publish=True))
            except ValueError as e:
                failures.append({"index": i, "message": str(e)})
        if len(ops) == 1 and failures:
            raise ApiError(400, f"Invalid {kind}: {failures[0]['message']}")
        self._indexed_failures(failures, f"Some {kind} submissions failed validation")

    async def get_proposer_preferences(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/proposer_preferences"""
        from ..spec.types.base import List
        from ..spec.types.gloas import SignedProposerPreferences

        slot = self._query_uint(request, "slot")
        dependent_root = request.query.get("dependent_root")
        if dependent_root is not None:
            dependent_root = self._hex_bytes(dependent_root, "dependent_root", 32)
        items = [
            signed for (root, proposal_slot, _), signed in list(self.node.proposer_preferences.items())
            if (slot is None or proposal_slot == slot) and (dependent_root is None or root == dependent_root)
        ]
        return self._versioned_response(
            request,
            self._fork_at_slot(slot if slot is not None else self.node.head_slot),
            List[SignedProposerPreferences, 2**20](*items).encode_bytes(),
            [to_json(p) for p in items],
        )

    async def get_pool_attester_slashings(self, request: web.Request) -> web.Response:
        """GET /eth/v2/beacon/pool/attester_slashings"""
        version = self._fork_at_slot(self.node.head_slot)
        slashings = list(self.node.operation_pool.attester_slashings.values())
        return web.json_response(
            {"version": version, "data": [to_json(s) for s in slashings]},
            headers={"Eth-Consensus-Version": version},
        )

    async def post_pool_attester_slashings(self, request: web.Request) -> web.Response:
        """POST /eth/v2/beacon/pool/attester_slashings"""
        version = self._consensus_version(request)
        slashing = await self._decode_body(request, self.node.attester_slashing_type(version))
        await self._submit_operations("attester_slashing", [slashing])
        return web.Response(status=200)

    async def get_pool_proposer_slashings(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/pool/proposer_slashings"""
        return web.json_response({"data": [to_json(s) for s in list(self.node.operation_pool.proposer_slashings.values())]})

    async def post_pool_proposer_slashings(self, request: web.Request) -> web.Response:
        """POST /eth/v1/beacon/pool/proposer_slashings"""
        from ..spec.types.phase0 import ProposerSlashing

        await self._submit_operations("proposer_slashing", [await self._decode_body(request, ProposerSlashing)])
        return web.Response(status=200)

    async def get_pool_voluntary_exits(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/pool/voluntary_exits"""
        return web.json_response({"data": [to_json(e) for e in list(self.node.operation_pool.voluntary_exits.values())]})

    async def post_pool_voluntary_exits(self, request: web.Request) -> web.Response:
        """POST /eth/v1/beacon/pool/voluntary_exits"""
        from ..spec.types.phase0 import SignedVoluntaryExit

        await self._submit_operations("voluntary_exit", [await self._decode_body(request, SignedVoluntaryExit)])
        return web.Response(status=200)

    async def get_pool_bls_to_execution_changes(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/pool/bls_to_execution_changes"""
        changes = list(self.node.operation_pool.bls_to_execution_changes.values())
        return web.json_response({"data": [to_json(c) for c in changes]})

    async def post_pool_bls_to_execution_changes(self, request: web.Request) -> web.Response:
        """POST /eth/v1/beacon/pool/bls_to_execution_changes"""
        from ..spec.types.capella import SignedBLSToExecutionChange

        changes = await self._decode_body(request, SignedBLSToExecutionChange, list_body=True)
        failures = []
        for i, change in enumerate(changes):
            try:
                await self._run_on_node_loop(self.node.submit_operation("bls_to_execution_change", change, publish=True))
            except ValueError as e:
                failures.append({"index": i, "message": str(e)})
        self._indexed_failures(failures, "Some BLS to execution changes failed validation")
        return web.Response(status=200)

    async def post_pool_sync_committees(self, request: web.Request) -> web.Response:
        """POST /eth/v1/beacon/pool/sync_committees"""
        from ..spec.types.altair import SyncCommitteeMessage

        messages = await self._decode_body(request, SyncCommitteeMessage, list_body=True)
        failures = []
        for i, message in enumerate(messages):
            try:
                await self._run_on_node_loop(self._submit_sync_committee_message(message))
            except ValueError as e:
                failures.append({"index": i, "message": str(e)})
        self._indexed_failures(failures, "Some sync committee messages failed validation")
        return web.Response(status=200)

    def _sync_committee_positions(self, state, validator_index: int, slot: int) -> list[int]:
        committee = self._sync_committee_for_epoch(state, slot // self._spe())
        pubkey = bytes(state.validators[validator_index].pubkey)
        return [i for i, pk in enumerate(committee.pubkeys) if bytes(pk) == pubkey]

    async def _submit_sync_committee_message(self, message) -> None:
        """Validate, pool and publish one SyncCommitteeMessage (node loop)."""
        from ..spec.constants import DOMAIN_SYNC_COMMITTEE, SYNC_COMMITTEE_SIZE, SYNC_COMMITTEE_SUBNET_COUNT
        from ..spec.state_transition.helpers.domain import get_domain_at_epoch, compute_signing_root
        from ..crypto.crypto import verify_async

        node = self.node
        state = node.state
        if state is None:
            raise ValueError("node has no state")
        slot = int(message.slot)
        if abs(slot - node._wall_slot()) > 1:
            raise ValueError(f"slot {slot} is not the current slot")
        index = int(message.validator_index)
        if index >= len(state.validators):
            raise ValueError(f"unknown validator {index}")
        try:
            positions = self._sync_committee_positions(state, index, slot)
        except ApiError as e:
            raise ValueError(e.message)
        if not positions:
            raise ValueError(f"validator {index} is not in the sync committee")
        domain = get_domain_at_epoch(state, DOMAIN_SYNC_COMMITTEE, slot // self._spe())
        signing_root = compute_signing_root(message.beacon_block_root, domain)
        if not await verify_async(bytes(state.validators[index].pubkey), signing_root, bytes(message.signature)):
            raise ValueError("invalid signature")
        size = SYNC_COMMITTEE_SIZE() // SYNC_COMMITTEE_SUBNET_COUNT
        for position in positions:
            node.sync_committee_pool.add(message, position)
        node.record_liveness(slot // self._spe(), [index])
        if node.beacon_gossip:
            for subnet in sorted({p // size for p in positions}):
                await node.beacon_gossip.publish_sync_committee_message(subnet, message.encode_bytes())

    # ------------------------------------------------------------------ debug

    async def get_debug_heads(self, request: web.Request) -> web.Response:
        """GET /eth/v2/debug/beacon/heads"""
        def leaves():
            store = self.node.fc_store
            if store is None:
                return [(bytes(self.node.head_root), int(self.node.head_slot))] if self.node.head_root else []
            blocks = dict(store.blocks)
            parents = {bytes(b.parent_root) for b in blocks.values()}
            return [(bytes(r), int(b.slot)) for r, b in blocks.items() if bytes(r) not in parents]

        heads = await self._run_on_node_loop_fn(leaves)
        return web.json_response({
            "data": [
                {"root": "0x" + root.hex(), "slot": str(slot), "execution_optimistic": False}
                for root, slot in sorted(heads, key=lambda h: h[1], reverse=True)
            ]
        })

    # ------------------------------------------------------------------ validator

    def _require_synced(self) -> None:
        _, _, is_syncing = self._sync_status()
        if is_syncing:
            raise ApiError(503, "Beacon node is currently syncing")

    async def _duty_state(self, epoch: int):
        """A state from which the shuffling of ``epoch`` can be computed."""
        from ..spec.constants import MIN_SEED_LOOKAHEAD

        head = self._head_state()
        head_epoch = self._state_epoch(head)
        if epoch > head_epoch + MIN_SEED_LOOKAHEAD:
            raise ApiError(400, f"Epoch {epoch} is too far in the future (head epoch {head_epoch})")
        if epoch >= head_epoch - 1:
            return head
        state = await self._state_at_slot(epoch * self._spe())
        if state is None:
            raise ApiError(400, f"No state available for epoch {epoch}")
        return state

    async def post_attester_duties(self, request: web.Request) -> web.Response:
        """POST /eth/v1/validator/duties/attester/{epoch}"""
        from ..spec.state_transition.helpers.beacon_committee import (
            get_beacon_committee,
            get_committee_count_per_slot,
        )

        epoch = self._parse_epoch(request)
        indices = set(await self._validator_index_body(request))
        wall_epoch = self.node.current_epoch if self.node._genesis_time else self._state_epoch(self._head_state())
        if epoch > wall_epoch + 1:
            raise ApiError(400, f"Epoch {epoch} is more than one epoch ahead of the current epoch {wall_epoch}")
        state = await self._duty_state(epoch)
        if any(i >= len(state.validators) for i in indices):
            raise ApiError(400, "Unknown validator index")

        def compute():
            duties = []
            committees_per_slot = get_committee_count_per_slot(state, epoch)
            for slot in range(epoch * self._spe(), (epoch + 1) * self._spe()):
                for committee_index in range(committees_per_slot):
                    committee = get_beacon_committee(state, slot, committee_index)
                    for position, vi in enumerate(committee):
                        if int(vi) in indices:
                            duties.append({
                                "pubkey": "0x" + bytes(state.validators[int(vi)].pubkey).hex(),
                                "validator_index": str(int(vi)),
                                "committee_index": str(committee_index),
                                "committee_length": str(len(committee)),
                                "committees_at_slot": str(committees_per_slot),
                                "validator_committee_index": str(position),
                                "slot": str(slot),
                            })
            return duties

        duties = await asyncio.to_thread(compute)
        return web.json_response({
            "dependent_root": "0x" + self._dependent_root(state, epoch - 1).hex(),
            "execution_optimistic": False,
            "data": duties,
        })

    async def post_sync_duties(self, request: web.Request) -> web.Response:
        """POST /eth/v1/validator/duties/sync/{epoch}"""
        epoch = self._parse_epoch(request)
        indices = await self._validator_index_body(request)
        state = self._head_state()
        if any(i >= len(state.validators) for i in indices):
            raise ApiError(400, "Unknown validator index")
        committee = self._sync_committee_for_epoch(state, epoch)
        positions: dict[bytes, list[int]] = {}
        for position, pubkey in enumerate(committee.pubkeys):
            positions.setdefault(bytes(pubkey), []).append(position)
        duties = []
        for index in dict.fromkeys(indices):
            pubkey = bytes(state.validators[index].pubkey)
            if pubkey in positions:
                duties.append({
                    "pubkey": "0x" + pubkey.hex(),
                    "validator_index": str(index),
                    "validator_sync_committee_indices": [str(p) for p in positions[pubkey]],
                })
        return web.json_response({"execution_optimistic": False, "data": duties})

    def _parse_block_production(self, request: web.Request) -> tuple[int, bytes, Optional[bytes]]:
        slot = self._uint(request.match_info["slot"], "slot")
        if "randao_reveal" not in request.query:
            raise ApiError(400, "Missing randao_reveal parameter")
        randao_reveal = self._hex_bytes(request.query["randao_reveal"], "randao_reveal", 96)
        graffiti = None
        if "graffiti" in request.query:
            graffiti = self._hex_bytes(request.query["graffiti"], "graffiti").ljust(32, b"\x00")[:32]
        return slot, randao_reveal, graffiti

    async def _produce_block(self, slot: int, randao_reveal: bytes, graffiti: Optional[bytes]) -> dict:
        if self.node.state is None or not self.node._is_synced():
            raise ApiError(503, "Beacon node is currently syncing")
        try:
            return await self._run_on_node_loop(self.node.produce_block_for_api(slot, randao_reveal, graffiti))
        except ValueError as e:
            raise ApiError(400, str(e))
        except Exception as e:
            logger.error(f"Block production failed for slot {slot}: {e}")
            raise ApiError(500, f"Block production failed: {e}")

    async def get_produce_block_v3(self, request: web.Request) -> web.Response:
        """GET /eth/v3/validator/blocks/{slot} (deprecated)"""
        from ..spec.types.base import Container, List
        from ..spec.types.deneb import Blob
        from ..spec.types import KZGProof
        from ..spec.constants import FIELD_ELEMENTS_PER_EXT_BLOB, MAX_BLOB_COMMITMENTS_PER_BLOCK

        slot, randao_reveal, graffiti = self._parse_block_production(request)
        if "builder_boost_factor" in request.query:
            self._uint(request.query["builder_boost_factor"], "builder_boost_factor")
        version = self._fork_at_slot(slot)
        if version in ("gloas", "heze"):
            raise ApiError(400, "produceBlockV3 is not available from Gloas, use /eth/v4/validator/blocks")
        produced = await self._produce_block(slot, randao_reveal, graffiti)
        block = produced["block"]
        bundle = produced["blobs_bundle"] or {}
        value = str(produced["execution_payload_value"])
        headers = {
            "Eth-Consensus-Version": version,
            "Eth-Execution-Payload-Blinded": "false",
            "Eth-Execution-Payload-Value": value,
            "Eth-Consensus-Block-Value": "0",
        }
        with_blobs = hasattr(block.body, "blob_kzg_commitments")
        if with_blobs:
            class BlockContents(Container):
                block: type(block)
                kzg_proofs: List[KZGProof, FIELD_ELEMENTS_PER_EXT_BLOB * MAX_BLOB_COMMITMENTS_PER_BLOCK]
                blobs: List[Blob, MAX_BLOB_COMMITMENTS_PER_BLOCK]

            contents = BlockContents(
                block=block,
                kzg_proofs=[bytes.fromhex(p.removeprefix("0x")) for p in bundle.get("proofs", [])],
                blobs=[bytes.fromhex(b.removeprefix("0x")) for b in bundle.get("blobs", [])],
            )
            ssz, data = contents.encode_bytes(), to_json(contents)
        else:
            ssz, data = block.encode_bytes(), to_json(block)
        if self._wants_ssz(request):
            return web.Response(body=ssz, content_type="application/octet-stream", headers=headers)
        return web.json_response(
            {
                "version": version,
                "execution_payload_blinded": False,
                "execution_payload_value": value,
                "consensus_block_value": "0",
                "data": data,
            },
            headers=headers,
        )

    async def get_attestation_data(self, request: web.Request) -> web.Response:
        """GET /eth/v1/validator/attestation_data"""
        from ..spec.types.phase0 import AttestationData
        from ..spec.types.base import Checkpoint

        slot = self._query_uint(request, "slot", required=True)
        committee_index = self._query_uint(request, "committee_index")
        self._require_synced()
        node = self.node
        wall_slot = node._wall_slot() if node._genesis_time else int(node.head_slot)
        if slot > wall_slot + 1 or slot + self._spe() < wall_slot:
            raise ApiError(400, f"Slot {slot} is too far from the current slot {wall_slot}")
        epoch = slot // self._spe()
        version = self._fork_at_slot(slot)
        if version in ("phase0", "altair", "bellatrix", "capella", "deneb") and committee_index is None:
            raise ApiError(400, "Missing committee_index parameter")
        head_root = self._block_root_at_or_before(slot)
        head_block = node.store.get_block(head_root) if head_root is not None else None
        if head_block is None:
            raise ApiError(503, "Head block not available")
        state = self._head_state()
        if epoch > self._state_epoch(state):
            state = await self._state_at_slot(slot) or state
        epoch_start = epoch * self._spe()
        target_root = head_root if int(head_block.message.slot) <= epoch_start else self._block_root_at_or_before(epoch_start)
        if version in ("gloas", "heze"):
            if int(head_block.message.slot) == slot:
                index = 0
            else:
                index = 1 if node.store.get_payload(head_root) is not None else 0
        elif version in ("electra", "fulu"):
            index = 0
        else:
            index = committee_index
        data = AttestationData(
            slot=slot,
            index=index,
            beacon_block_root=head_root,
            source=state.current_justified_checkpoint,
            target=Checkpoint(epoch=epoch, root=target_root),
        )
        return self._ssz_or_json(request, data.encode_bytes, to_json(data))

    async def get_aggregate_attestation(self, request: web.Request) -> web.Response:
        """GET /eth/v2/validator/aggregate_attestation"""
        from ..spec.types.gloas import Attestation as GloasAttestation

        if "attestation_data_root" not in request.query:
            raise ApiError(400, "Missing attestation_data_root parameter")
        data_root = self._hex_bytes(request.query["attestation_data_root"], "attestation_data_root", 32)
        slot = self._query_uint(request, "slot", required=True)
        committee_index = self._query_uint(request, "committee_index", required=True)
        aggregate = await self._run_on_node_loop_fn(
            self.node.attestation_pool.get_aggregate, slot, data_root, committee_index
        )
        if aggregate is None:
            raise ApiError(404, "No matching attestations found")
        version = self._fork_at_slot(slot)
        if version in ("gloas", "heze"):
            aggregate = GloasAttestation.decode_bytes(aggregate.encode_bytes())
        return self._versioned_response(request, version, aggregate.encode_bytes(), to_json(aggregate))

    @staticmethod
    def _aggregate_and_proof_type(version: str):
        from ..spec.types.electra import SignedElectraAggregateAndProof
        from ..spec.types.phase0 import SignedAggregateAndProof
        from ..spec.types.gloas import SignedAggregateAndProof as GloasSignedAggregateAndProof

        if version in ("gloas", "heze"):
            return GloasSignedAggregateAndProof
        if version in ("electra", "fulu"):
            return SignedElectraAggregateAndProof
        return SignedAggregateAndProof

    async def post_aggregate_and_proofs(self, request: web.Request) -> web.Response:
        """POST /eth/v2/validator/aggregate_and_proofs"""
        version = self._consensus_version(request)
        items = await self._decode_body(request, self._aggregate_and_proof_type(version), list_body=True)
        failures = []
        for i, signed in enumerate(items):
            try:
                await self._run_on_node_loop(self._submit_aggregate_and_proof(signed))
            except ValueError as e:
                failures.append({"index": i, "message": str(e)})
        self._indexed_failures(failures, "Some aggregate and proofs failed validation")
        return web.Response(status=200)

    async def _submit_aggregate_and_proof(self, signed) -> None:
        """Validate, pool and publish one SignedAggregateAndProof (node loop)."""
        from ..spec.constants import DOMAIN_AGGREGATE_AND_PROOF, DOMAIN_SELECTION_PROOF, TARGET_AGGREGATORS_PER_COMMITTEE
        from ..spec.state_transition.helpers.attestation import get_indexed_attestation
        from ..spec.state_transition.helpers.beacon_committee import get_beacon_committee
        from ..spec.state_transition.helpers.domain import get_domain_at_epoch, compute_signing_root
        from ..spec.state_transition.helpers.predicates import is_valid_indexed_attestation
        from ..spec.types import Slot
        from ..spec.types.electra import Attestation as ElectraAttestation
        from ..crypto import sha256
        from ..crypto.crypto import verify_async

        node = self.node
        state = node.state
        if state is None:
            raise ValueError("node has no state")
        message = signed.message
        aggregate = message.aggregate
        slot = int(aggregate.data.slot)
        epoch = slot // self._spe()
        if int(aggregate.data.target.epoch) != epoch:
            raise ValueError("target epoch does not match slot")
        committee_index = node.attestation_pool.committee_of(aggregate)
        if committee_index < 0:
            raise ValueError("aggregate must have exactly one committee bit set")
        aggregator = int(message.aggregator_index)
        try:
            committee = [int(v) for v in get_beacon_committee(state, slot, committee_index)]
        except Exception as e:
            raise ValueError(f"cannot compute committee: {e}")
        if aggregator not in committee:
            raise ValueError(f"aggregator {aggregator} is not in the committee")
        pubkey = bytes(state.validators[aggregator].pubkey)
        selection_root = compute_signing_root(Slot(slot), get_domain_at_epoch(state, DOMAIN_SELECTION_PROOF, epoch))
        if not await verify_async(pubkey, selection_root, bytes(message.selection_proof)):
            raise ValueError("invalid selection proof")
        modulo = max(1, len(committee) // TARGET_AGGREGATORS_PER_COMMITTEE)
        if int.from_bytes(sha256(bytes(message.selection_proof))[:8], "little") % modulo != 0:
            raise ValueError(f"validator {aggregator} is not an aggregator")
        aggregate_root = compute_signing_root(message, get_domain_at_epoch(state, DOMAIN_AGGREGATE_AND_PROOF, epoch))
        if not await verify_async(pubkey, aggregate_root, bytes(signed.signature)):
            raise ValueError("invalid aggregate and proof signature")
        if not is_valid_indexed_attestation(state, get_indexed_attestation(state, aggregate)):
            raise ValueError("invalid aggregate signature")
        pooled = ElectraAttestation.decode_bytes(aggregate.encode_bytes()) if hasattr(aggregate, "committee_bits") else aggregate
        node.attestation_pool.add(pooled)
        node._fc_on_attestation(pooled)
        node.record_liveness(epoch, [aggregator])
        self.emit_attestation(pooled)
        if node.beacon_gossip:
            await node.beacon_gossip.publish_aggregate(signed.encode_bytes())

    async def post_beacon_committee_subscriptions(self, request: web.Request) -> web.Response:
        """POST /eth/v1/validator/beacon_committee_subscriptions"""
        body = await self._json_list_body(request)
        try:
            subscriptions = [
                {
                    "validator_index": self._uint(s["validator_index"], "validator_index"),
                    "committee_index": self._uint(s["committee_index"], "committee_index"),
                    "committees_at_slot": self._uint(s["committees_at_slot"], "committees_at_slot"),
                    "slot": self._uint(s["slot"], "slot"),
                    "is_aggregator": s["is_aggregator"],
                }
                for s in body
            ]
        except (KeyError, TypeError) as e:
            raise ApiError(400, f"Invalid subscription: {e}")
        if any(not isinstance(s["is_aggregator"], bool) for s in subscriptions):
            raise ApiError(400, "is_aggregator must be a boolean")
        self._require_synced()
        from ..spec.constants import ATTESTATION_SUBNET_COUNT

        await self._run_on_node_loop_fn(
            self.node.attach_api_validators, [s["validator_index"] for s in subscriptions]
        )
        for s in subscriptions:
            if not s["is_aggregator"]:
                continue
            subnet = (
                s["committees_at_slot"] * (s["slot"] % self._spe()) + s["committee_index"]
            ) % ATTESTATION_SUBNET_COUNT()
            await self._run_on_node_loop(self.node.subscribe_attestation_subnet(subnet, s["slot"]))
        return web.Response(status=200)

    async def post_sync_committee_subscriptions(self, request: web.Request) -> web.Response:
        """POST /eth/v1/validator/sync_committee_subscriptions"""
        body = await self._json_list_body(request)
        try:
            indices = [self._uint(s["validator_index"], "validator_index") for s in body]
            for s in body:
                [self._uint(i, "sync_committee_index") for i in s["sync_committee_indices"]]
                self._uint(s["until_epoch"], "until_epoch")
        except (KeyError, TypeError) as e:
            raise ApiError(400, f"Invalid subscription: {e}")
        self._require_synced()
        await self._run_on_node_loop_fn(self.node.attach_api_validators, indices)
        return web.Response(status=200)

    async def post_selections(self, request: web.Request) -> web.Response:
        """POST /eth/v1/validator/{beacon,sync}_committee_selections"""
        raise ApiError(501, "Selection proofs are only combined by distributed validator middleware")

    async def get_sync_committee_contribution(self, request: web.Request) -> web.Response:
        """GET /eth/v1/validator/sync_committee_contribution"""
        from ..spec.constants import SYNC_COMMITTEE_SIZE, SYNC_COMMITTEE_SUBNET_COUNT
        from ..spec.types.altair import SyncCommitteeContribution
        from ..spec.types.base import Bitvector
        from ..crypto import aggregate_signatures

        slot = self._query_uint(request, "slot", required=True)
        subcommittee_index = self._query_uint(request, "subcommittee_index", required=True)
        if subcommittee_index >= SYNC_COMMITTEE_SUBNET_COUNT:
            raise ApiError(400, f"Invalid subcommittee_index: {subcommittee_index}")
        if "beacon_block_root" not in request.query:
            raise ApiError(400, "Missing beacon_block_root parameter")
        root = self._hex_bytes(request.query["beacon_block_root"], "beacon_block_root", 32)
        self._require_synced()
        size = SYNC_COMMITTEE_SIZE() // SYNC_COMMITTEE_SUBNET_COUNT

        def build():
            pool = self.node.sync_committee_pool
            bits = Bitvector[size]()
            signatures = []
            for position, pooled in sorted(pool._messages.get(slot, {}).items()):
                if position // size != subcommittee_index or bytes(pooled.message.beacon_block_root) != root:
                    continue
                if not bits[position % size]:
                    bits[position % size] = True
                    signatures.append(bytes(pooled.message.signature))
            existing = pool._contributions.get(slot, {}).get((subcommittee_index, root))
            if existing is not None and sum(existing.contribution.aggregation_bits) >= len(signatures):
                return existing.contribution
            if not signatures:
                return None
            return SyncCommitteeContribution(
                slot=slot,
                beacon_block_root=root,
                subcommittee_index=subcommittee_index,
                aggregation_bits=bits,
                signature=aggregate_signatures(signatures),
            )

        contribution = await self._run_on_node_loop_fn(build)
        if contribution is None:
            raise ApiError(404, "No sync committee messages for the requested slot, subcommittee and root")
        return web.json_response({"data": to_json(contribution)})

    async def post_contribution_and_proofs(self, request: web.Request) -> web.Response:
        """POST /eth/v1/validator/contribution_and_proofs"""
        from ..spec.types.altair import SignedContributionAndProof

        items = await self._decode_body(request, SignedContributionAndProof, list_body=True)
        failures = []
        for i, signed in enumerate(items):
            try:
                await self._run_on_node_loop(self._submit_contribution_and_proof(signed))
            except ValueError as e:
                failures.append({"index": i, "message": str(e)})
        self._indexed_failures(failures, "Some contribution and proofs failed validation")
        return web.Response(status=200)

    async def _submit_contribution_and_proof(self, signed) -> None:
        """Validate, pool and publish one SignedContributionAndProof (node loop)."""
        from ..spec.constants import (
            DOMAIN_CONTRIBUTION_AND_PROOF,
            DOMAIN_SYNC_COMMITTEE,
            DOMAIN_SYNC_COMMITTEE_SELECTION_PROOF,
            SYNC_COMMITTEE_SIZE,
            SYNC_COMMITTEE_SUBNET_COUNT,
            TARGET_AGGREGATORS_PER_SYNC_SUBCOMMITTEE,
        )
        from ..spec.state_transition.helpers.domain import get_domain_at_epoch, compute_signing_root
        from ..spec.types.altair import SyncAggregatorSelectionData
        from ..crypto import sha256
        from ..crypto.crypto import verify_async, fast_aggregate_verify_async

        node = self.node
        state = node.state
        if state is None:
            raise ValueError("node has no state")
        message = signed.message
        contribution = message.contribution
        slot = int(contribution.slot)
        epoch = slot // self._spe()
        subcommittee_index = int(contribution.subcommittee_index)
        if subcommittee_index >= SYNC_COMMITTEE_SUBNET_COUNT:
            raise ValueError("invalid subcommittee index")
        size = SYNC_COMMITTEE_SIZE() // SYNC_COMMITTEE_SUBNET_COUNT
        aggregator = int(message.aggregator_index)
        if aggregator >= len(state.validators):
            raise ValueError(f"unknown validator {aggregator}")
        try:
            committee = self._sync_committee_for_epoch(state, epoch)
        except ApiError as e:
            raise ValueError(e.message)
        subcommittee = [bytes(pk) for pk in committee.pubkeys[subcommittee_index * size:(subcommittee_index + 1) * size]]
        pubkey = bytes(state.validators[aggregator].pubkey)
        if pubkey not in subcommittee:
            raise ValueError(f"aggregator {aggregator} is not in subcommittee {subcommittee_index}")
        selection_data = SyncAggregatorSelectionData(slot=slot, subcommittee_index=subcommittee_index)
        selection_root = compute_signing_root(
            selection_data, get_domain_at_epoch(state, DOMAIN_SYNC_COMMITTEE_SELECTION_PROOF, epoch)
        )
        if not await verify_async(pubkey, selection_root, bytes(message.selection_proof)):
            raise ValueError("invalid selection proof")
        modulo = max(1, size // TARGET_AGGREGATORS_PER_SYNC_SUBCOMMITTEE)
        if int.from_bytes(sha256(bytes(message.selection_proof))[:8], "little") % modulo != 0:
            raise ValueError(f"validator {aggregator} is not a sync committee aggregator")
        outer_root = compute_signing_root(message, get_domain_at_epoch(state, DOMAIN_CONTRIBUTION_AND_PROOF, epoch))
        if not await verify_async(pubkey, outer_root, bytes(signed.signature)):
            raise ValueError("invalid contribution and proof signature")
        participants = [pk for pk, bit in zip(subcommittee, contribution.aggregation_bits) if bit]
        if not participants:
            raise ValueError("contribution has no participants")
        message_root = compute_signing_root(
            contribution.beacon_block_root, get_domain_at_epoch(state, DOMAIN_SYNC_COMMITTEE, epoch)
        )
        if not await fast_aggregate_verify_async(participants, message_root, bytes(contribution.signature)):
            raise ValueError("invalid contribution signature")
        node.sync_committee_pool.add_contribution(contribution)
        node.record_liveness(epoch, [aggregator])
        self.emit_contribution_and_proof(signed)
        if node.beacon_gossip:
            await node.beacon_gossip.publish_sync_committee_contribution(signed.encode_bytes())

    async def post_prepare_beacon_proposer(self, request: web.Request) -> web.Response:
        """POST /eth/v1/validator/prepare_beacon_proposer (deprecated)"""
        body = await self._json_list_body(request)
        try:
            entries = {
                self._uint(e["validator_index"], "validator_index"): self._hex_bytes(e["fee_recipient"], "fee_recipient", 20)
                for e in body
            }
        except (KeyError, TypeError) as e:
            raise ApiError(400, f"Invalid proposer preparation: {e}")
        self.node.fee_recipients.update(entries)
        await self._run_on_node_loop_fn(self.node.attach_api_validators, list(entries))
        return web.Response(status=200)

    async def post_register_validator(self, request: web.Request) -> web.Response:
        """POST /eth/v1/validator/register_validator (deprecated)

        Registrations are validated and their fee recipients applied to
        locally built payloads; this node has no external builder to forward
        them to.
        """
        from ..spec.types.base import BLSPubkey, BLSSignature, Container, ExecutionAddress, uint64

        class ValidatorRegistrationV1(Container):
            fee_recipient: ExecutionAddress
            gas_limit: uint64
            timestamp: uint64
            pubkey: BLSPubkey

        class SignedValidatorRegistrationV1(Container):
            message: ValidatorRegistrationV1
            signature: BLSSignature

        registrations = await self._decode_body(request, SignedValidatorRegistrationV1, list_body=True)
        state = self._head_state()
        pubkeys = {bytes(v.pubkey): i for i, v in enumerate(state.validators)}
        for registration in registrations:
            index = pubkeys.get(bytes(registration.message.pubkey))
            if index is not None:
                self.node.fee_recipients[index] = bytes(registration.message.fee_recipient)
        return web.Response(status=200)

    async def post_liveness(self, request: web.Request) -> web.Response:
        """POST /eth/v1/validator/liveness/{epoch}"""
        from ..spec.state_transition.helpers.attestation import get_attesting_indices

        epoch = self._parse_epoch(request)
        indices = await self._validator_index_body(request)
        state = self._head_state()
        wall_epoch = self.node.current_epoch if self.node._genesis_time else self._state_epoch(state)
        if epoch > wall_epoch:
            raise ApiError(400, f"Epoch {epoch} is in the future")
        if any(i >= len(state.validators) for i in indices):
            raise ApiError(400, "Unknown validator index")

        def compute() -> set[int]:
            live = set(self.node.liveness.get(epoch, set()))
            head_slot = int(self.node.head_slot)
            for slot in range(epoch * self._spe(), min((epoch + 2) * self._spe(), head_slot + 1)):
                root = self._canonical_root_at_slot(slot)
                block = self.node.store.get_block(root) if root is not None else None
                if block is None:
                    continue
                if slot // self._spe() == epoch:
                    live.add(int(block.message.proposer_index))
                for attestation in block.message.body.attestations:
                    if int(attestation.data.target.epoch) != epoch:
                        continue
                    try:
                        live.update(int(i) for i in get_attesting_indices(state, attestation))
                    except Exception:
                        pass
            return live

        live = await asyncio.to_thread(compute)
        return web.json_response({
            "data": [{"index": str(i), "is_live": i in live} for i in dict.fromkeys(indices)]
        })

    async def get_events(self, request: web.Request) -> web.StreamResponse:
        """GET /eth/v1/events - SSE endpoint for beacon events."""
        topics = set(self._query_list(request, "topics"))
        if not topics:
            raise ApiError(400, "No topics specified")
        unknown = topics - self.EVENT_TOPICS
        if unknown:
            raise ApiError(400, f"Invalid topic: {sorted(unknown)[0]}")

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
        logger.info(f"SSE client connected, topics: {topics}")
        try:
            while True:
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=15.0)
                    if event.get("event") in topics:
                        await response.write(f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n".encode())
                except asyncio.TimeoutError:
                    await response.write(b": keepalive\n\n")
                except asyncio.CancelledError:
                    break
        except ConnectionResetError:
            pass
        finally:
            if event_queue in self._event_subscribers:
                self._event_subscribers.remove(event_queue)
            logger.info("SSE client disconnected")
        return response

    def _head_events(self, slot: int, root: bytes, payload_status: str, new_block: bool) -> list[dict]:
        """head / head_v2 / block events for a new fork-choice head."""
        from ..crypto import hash_tree_root

        state = self.node.state
        if state is None:
            return []
        state_root = "0x" + hash_tree_root(state).hex()
        epoch = slot // self._spe()
        prev_dependent = "0x" + self._dependent_root(state, epoch - 1).hex()
        current_dependent = "0x" + self._dependent_root(state, epoch).hex()
        epoch_transition = slot % self._spe() == 0
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
                    if new_block:
                        try:
                            await asyncio.to_thread(self._update_light_client, bytes(current_root))
                        except Exception as e:
                            logger.debug(f"Light client update for {current_root.hex()[:16]} failed: {e}")
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
                        await self._broadcast_event({
                            "event": "finalized_checkpoint",
                            "data": {
                                "block": "0x" + finalized_root.hex(),
                                "state": "0x" + finalized_state_root.hex(),
                                "epoch": str(current_finalized_epoch),
                                "execution_optimistic": False,
                            },
                        })
                        self._last_finalized_epoch = current_finalized_epoch

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in event emitter loop: {e}")
                await asyncio.sleep(1.0)

    def emit_attestation(self, attestation) -> None:
        """Emit an `attestation` SSE event."""
        self.emit("attestation", to_json(attestation))

    def emit_contribution_and_proof(self, signed_contribution) -> None:
        """Emit a `contribution_and_proof` SSE event."""
        self.emit("contribution_and_proof", to_json(signed_contribution))

    def emit_data_column_sidecar(self, sidecar_ssz: bytes) -> None:
        """Emit a `data_column_sidecar` SSE event for a verified gossip sidecar."""
        from ..spec.types.gloas import DataColumnSidecar

        sidecar = DataColumnSidecar.decode_bytes(sidecar_ssz)
        self.emit("data_column_sidecar", {
            "block_root": "0x" + bytes(sidecar.beacon_block_root).hex(),
            "index": str(int(sidecar.index)),
            "slot": str(int(sidecar.slot)),
        })

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

        self._consensus_version(request)
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

    async def post_proposer_preferences(self, request: web.Request) -> web.Response:
        """POST /eth/v1/validator/proposer_preferences

        Accepts an array of SignedProposerPreferences, validates each
        against the head state, stores them, and publishes them on the
        `proposer_preferences` gossip topic.
        """
        from ..spec.types.gloas import SignedProposerPreferences

        self._consensus_version(request)
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
        root = self._canonical_root_at_slot(slot)
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

    async def post_pool_payload_attestations(self, request: web.Request) -> web.Response:
        """POST /eth/v1/beacon/pool/payload_attestations"""
        from ..spec.types.gloas import PayloadAttestationMessage

        self._consensus_version(request)
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

    # ------------------------------------------------------------------ blocks

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

        slot, randao_reveal, graffiti = self._parse_block_production(request)
        include_payload = request.query.get("include_payload", "").lower()
        if include_payload not in ("true", "false"):
            raise ApiError(400, "include_payload must be true or false")
        include_payload = include_payload == "true"
        self._consensus_version(request)
        if not await request.read():
            raise ApiError(400, "Missing BuilderConfig request body")
        version = self._fork_at_slot(slot)
        if version not in ("gloas", "heze"):
            raise ApiError(400, "produceBlockV4 is only available from Gloas")
        produced = await self._produce_block(slot, randao_reveal, graffiti)

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

        version = self._consensus_version(request)
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
        config = get_network_config()

        address = config.deposit_contract_address
        if isinstance(address, int):
            address = address.to_bytes(20, "big")
        elif isinstance(address, str):
            address = bytes.fromhex(address.removeprefix("0x"))
        return web.json_response({
            "data": {
                "chain_id": str(int(config.deposit_chain_id)),
                "address": "0x" + bytes(address).hex(),
            }
        })

    EVENT_TOPICS = frozenset({
        "head", "head_v2", "block", "block_gossip", "attestation", "single_attestation",
        "voluntary_exit", "bls_to_execution_change", "proposer_slashing", "attester_slashing",
        "finalized_checkpoint", "chain_reorg", "contribution_and_proof",
        "light_client_finality_update", "light_client_optimistic_update", "payload_attributes",
        "data_column_sidecar", "execution_payload", "execution_payload_gossip",
        "execution_payload_available", "execution_payload_bid", "payload_attestation_message",
        "fast_confirmation", "proposer_preferences", "inclusion_list",
    })

    def _head_payload_status(self, slot: int, root: bytes) -> str:
        from ..spec.constants import SLOTS_PER_EPOCH
        if slot // SLOTS_PER_EPOCH() < get_network_config().gloas_fork_epoch:
            return "full"
        return "full" if self.node.store.get_payload(root) is not None else "empty"

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
