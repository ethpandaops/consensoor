"""Beacon API HTTP server."""

import asyncio
import json
import logging
from typing import Optional
from aiohttp import web

from .utils import get_local_ip, generate_peer_id
from .spec import build_spec_response
from ..spec.network_config import get_config as get_network_config
from ..version import get_cl_version, get_cl_commit, CL_CLIENT_NAME

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
        self.app.router.add_get("/eth/v1/beacon/states/{state_id}/fork", self.get_state_fork)
        self.app.router.add_get("/eth/v1/beacon/states/{state_id}/finality_checkpoints", self.get_finality_checkpoints)
        self.app.router.add_get("/eth/v1/beacon/states/{state_id}/validators", self.get_validators)
        self.app.router.add_get("/eth/v1/beacon/states/{state_id}/validators/{validator_id}", self.get_validator)
        self.app.router.add_get("/eth/v1/beacon/states/{state_id}/validator_balances", self.get_validator_balances)
        self.app.router.add_get("/eth/v1/beacon/states/{state_id}/committees", self.get_committees)
        self.app.router.add_get("/eth/v1/beacon/states/{state_id}/sync_committees", self.get_sync_committees)
        self.app.router.add_get("/eth/v1/beacon/states/{state_id}/randao", self.get_randao)
        self.app.router.add_get("/eth/v1/beacon/headers", self.get_headers)
        self.app.router.add_get("/eth/v1/beacon/headers/{block_id}", self.get_header)
        self.app.router.add_get("/eth/v2/beacon/blocks/{block_id}", self.get_block)
        self.app.router.add_get("/eth/v1/beacon/blocks/{block_id}/root", self.get_block_root)
        self.app.router.add_get("/eth/v1/beacon/blob_sidecars/{block_id}", self.get_blob_sidecars)
        self.app.router.add_get("/eth/v2/debug/beacon/states/{state_id}", self.get_debug_state)
        self.app.router.add_get("/eth/v1/config/spec", self.get_spec)
        self.app.router.add_get("/eth/v1/config/fork_schedule", self.get_fork_schedule)
        self.app.router.add_get("/eth/v1/config/deposit_contract", self.get_deposit_contract)
        self.app.router.add_get("/eth/v1/beacon/execution_payload_envelope/{block_id}", self.get_execution_payload_envelope)
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

        # Get actual connected peers from P2P host
        if self.node.beacon_gossip and hasattr(self.node.beacon_gossip, '_host'):
            p2p_host = self.node.beacon_gossip._host
            if p2p_host and hasattr(p2p_host, 'connected_peers'):
                for peer_info in p2p_host.connected_peers:
                    peer_id = peer_info.get("peer_id", "")
                    addrs = peer_info.get("addrs", [])
                    direction = peer_info.get("direction", "unknown")
                    peers.append({
                        "peer_id": peer_id,
                        "enr": "",
                        "last_seen_p2p_address": addrs[0] if addrs else "",
                        "state": "connected",
                        "direction": direction,
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
        state_id = request.match_info["state_id"]
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
        state_id = request.match_info["state_id"]

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
        logger.info(f"Returning block JSON: block_id={block_id}, version={version}")
        block_json = self._signed_block_to_json(signed_block)

        return web.json_response({
            "version": version,
            "execution_optimistic": False,
            "finalized": False,
            "data": block_json,
        })

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

        accept = request.headers.get("Accept", "application/json")

        if "application/octet-stream" in accept:
            ssz_bytes = signed_envelope.encode_bytes()
            logger.info(f"Returning envelope SSZ: block_id={block_id}, size={len(ssz_bytes)}")
            return web.Response(
                body=ssz_bytes,
                content_type="application/octet-stream",
                headers={"Eth-Consensus-Version": "gloas"},
            )

        # JSON response
        envelope = signed_envelope.message
        payload = envelope.payload
        data = {
            "message": {
                "payload": {
                    "parent_hash": "0x" + bytes(payload.parent_hash).hex(),
                    "fee_recipient": "0x" + bytes(payload.fee_recipient).hex(),
                    "state_root": "0x" + bytes(payload.state_root).hex(),
                    "receipts_root": "0x" + bytes(payload.receipts_root).hex(),
                    "logs_bloom": "0x" + bytes(payload.logs_bloom).hex(),
                    "prev_randao": "0x" + bytes(payload.prev_randao).hex(),
                    "block_number": str(int(payload.block_number)),
                    "gas_limit": str(int(payload.gas_limit)),
                    "gas_used": str(int(payload.gas_used)),
                    "timestamp": str(int(payload.timestamp)),
                    "extra_data": "0x" + bytes(payload.extra_data).hex(),
                    "base_fee_per_gas": str(int(payload.base_fee_per_gas)),
                    "block_hash": "0x" + bytes(payload.block_hash).hex(),
                    "transactions": ["0x" + bytes(tx).hex() for tx in payload.transactions],
                },
                "execution_requests": {
                    "deposits": [],
                    "withdrawals": [],
                    "consolidations": [],
                },
                "builder_index": str(int(envelope.builder_index)),
                "beacon_block_root": "0x" + bytes(envelope.beacon_block_root).hex(),
                "slot": str(int(envelope.slot)),
                "blob_kzg_commitments": ["0x" + bytes(c).hex() for c in envelope.blob_kzg_commitments],
                "state_root": "0x" + bytes(envelope.state_root).hex(),
            },
            "signature": "0x" + bytes(signed_envelope.signature).hex(),
        }
        logger.info(f"Returning envelope JSON: block_id={block_id}")
        return web.json_response({
            "version": "gloas",
            "execution_optimistic": False,
            "finalized": False,
            "data": data,
        })

    async def get_blob_sidecars(self, request: web.Request) -> web.Response:
        """GET /eth/v1/beacon/blob_sidecars/{block_id}"""
        block_id = request.match_info["block_id"]
        root, signed_block = self._resolve_block_id(block_id)

        if root is None:
            return web.json_response({"message": "Block not found"}, status=404)

        sidecars = self.node.store.get_blobs(root)

        return web.json_response({
            "data": sidecars,
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
        attestations_json = []
        for att in body.attestations:
            att_json = {
                "aggregation_bits": "0x" + bytes(att.aggregation_bits).hex(),
                "data": {
                    "slot": str(att.data.slot),
                    "index": str(att.data.index),
                    "beacon_block_root": "0x" + bytes(att.data.beacon_block_root).hex(),
                    "source": {
                        "epoch": str(att.data.source.epoch),
                        "root": "0x" + bytes(att.data.source.root).hex(),
                    },
                    "target": {
                        "epoch": str(att.data.target.epoch),
                        "root": "0x" + bytes(att.data.target.root).hex(),
                    },
                },
                "signature": "0x" + bytes(att.signature).hex(),
            }
            if hasattr(att, "committee_bits"):
                att_json["committee_bits"] = "0x" + bytes(att.committee_bits).hex()
            attestations_json.append(att_json)

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
            "attestations": attestations_json,
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

    async def get_fork_schedule(self, request: web.Request) -> web.Response:
        """GET /eth/v1/config/fork_schedule"""
        from ..spec.network_config import get_config
        config = get_config()

        forks = []

        if hasattr(config, 'genesis_fork_version'):
            forks.append({
                "previous_version": "0x" + config.genesis_fork_version.hex(),
                "current_version": "0x" + config.genesis_fork_version.hex(),
                "epoch": "0",
            })

        if hasattr(config, 'altair_fork_epoch') and hasattr(config, 'altair_fork_version'):
            forks.append({
                "previous_version": "0x" + config.genesis_fork_version.hex(),
                "current_version": "0x" + config.altair_fork_version.hex(),
                "epoch": str(config.altair_fork_epoch),
            })

        if hasattr(config, 'bellatrix_fork_epoch') and hasattr(config, 'bellatrix_fork_version'):
            prev = config.altair_fork_version if hasattr(config, 'altair_fork_version') else config.genesis_fork_version
            forks.append({
                "previous_version": "0x" + prev.hex(),
                "current_version": "0x" + config.bellatrix_fork_version.hex(),
                "epoch": str(config.bellatrix_fork_epoch),
            })

        if hasattr(config, 'capella_fork_epoch') and hasattr(config, 'capella_fork_version'):
            prev = config.bellatrix_fork_version if hasattr(config, 'bellatrix_fork_version') else config.genesis_fork_version
            forks.append({
                "previous_version": "0x" + prev.hex(),
                "current_version": "0x" + config.capella_fork_version.hex(),
                "epoch": str(config.capella_fork_epoch),
            })

        if hasattr(config, 'deneb_fork_epoch') and hasattr(config, 'deneb_fork_version'):
            prev = config.capella_fork_version if hasattr(config, 'capella_fork_version') else config.genesis_fork_version
            forks.append({
                "previous_version": "0x" + prev.hex(),
                "current_version": "0x" + config.deneb_fork_version.hex(),
                "epoch": str(config.deneb_fork_epoch),
            })

        if hasattr(config, 'electra_fork_epoch') and hasattr(config, 'electra_fork_version'):
            prev = config.deneb_fork_version if hasattr(config, 'deneb_fork_version') else config.genesis_fork_version
            forks.append({
                "previous_version": "0x" + prev.hex(),
                "current_version": "0x" + config.electra_fork_version.hex(),
                "epoch": str(config.electra_fork_epoch),
            })

        if hasattr(config, 'fulu_fork_epoch') and hasattr(config, 'fulu_fork_version'):
            prev = config.electra_fork_version if hasattr(config, 'electra_fork_version') else config.genesis_fork_version
            forks.append({
                "previous_version": "0x" + prev.hex(),
                "current_version": "0x" + config.fulu_fork_version.hex(),
                "epoch": str(config.fulu_fork_epoch),
            })

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

    async def get_events(self, request: web.Request) -> web.StreamResponse:
        """GET /eth/v1/events - SSE endpoint for beacon events."""
        topics_param = request.query.get("topics", "")
        topics = set(t.strip() for t in topics_param.split(",") if t.strip())

        valid_topics = {"head", "block", "finalized_checkpoint", "chain_reorg",
                        "execution_payload_available", "execution_payload_bid"}
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
                        # Compute actual state_root (not block_root)
                        state_root_hex = "0x" + current_root.hex()  # fallback
                        if self.node.state:
                            from ..crypto import hash_tree_root
                            actual_state_root = hash_tree_root(self.node.state)
                            state_root_hex = "0x" + actual_state_root.hex()

                        head_event = {
                            "event": "head",
                            "data": {
                                "slot": str(current_slot),
                                "block": "0x" + current_root.hex(),
                                "state": state_root_hex,
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

    async def emit_execution_payload_available(self, slot: int, block_root: bytes) -> None:
        """Emit execution_payload_available SSE event (ePBS)."""
        event = {
            "event": "execution_payload_available",
            "data": {
                "slot": str(slot),
                "block_root": "0x" + block_root.hex(),
            },
        }
        await self._broadcast_event(event)

    async def emit_execution_payload_bid(self, signed_bid) -> None:
        """Emit execution_payload_bid SSE event (ePBS)."""
        bid = signed_bid.message
        event = {
            "event": "execution_payload_bid",
            "data": {
                "message": {
                    "parent_block_hash": "0x" + bytes(bid.parent_block_hash).hex(),
                    "parent_block_root": "0x" + bytes(bid.parent_block_root).hex(),
                    "block_hash": "0x" + bytes(bid.block_hash).hex(),
                    "prev_randao": "0x" + bytes(bid.prev_randao).hex(),
                    "fee_recipient": "0x" + bytes(bid.fee_recipient).hex(),
                    "gas_limit": str(int(bid.gas_limit)),
                    "builder_index": str(int(bid.builder_index)),
                    "slot": str(int(bid.slot)),
                    "value": str(int(bid.value)),
                    "execution_payment": str(int(bid.execution_payment)),
                    "blob_kzg_commitments_root": "0x" + bytes(bid.blob_kzg_commitments_root).hex(),
                },
                "signature": "0x" + bytes(signed_bid.signature).hex(),
            },
        }
        await self._broadcast_event(event)
