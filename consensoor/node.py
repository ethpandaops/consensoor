"""Main beacon node orchestration."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional, Union

from .config import Config
from .beacon_sync import RemoteBeaconClient, StateSyncManager
from .p2p import BeaconGossip
from .spec.types import (
    BeaconState,
    FuluBeaconState,
    ElectraBeaconState,
    DenebBeaconState,
    CapellaBeaconState,
    BellatrixBeaconState,
    AltairBeaconState,
    Phase0BeaconState,
    SignedBeaconBlock,
    SignedExecutionPayloadBid,
    SignedExecutionPayloadEnvelope,
    ExecutionPayloadEnvelope,
    PayloadAttestationMessage,
    Attestation,
    SignedElectraBeaconBlock,
)
from .spec.types.bellatrix import SignedBellatrixBeaconBlock
from .spec.types.capella import SignedCapellaBeaconBlock
from .spec.types.deneb import SignedDenebBeaconBlock

AnySignedBeaconBlock = Union[
    SignedBeaconBlock, SignedElectraBeaconBlock,
    SignedDenebBeaconBlock, SignedCapellaBeaconBlock, SignedBellatrixBeaconBlock,
]

AnyBeaconState = Union[
    BeaconState, FuluBeaconState, ElectraBeaconState, DenebBeaconState,
    CapellaBeaconState, BellatrixBeaconState, AltairBeaconState, Phase0BeaconState
]
from .spec.network_config import (
    load_config as load_network_config,
    load_config_from_upstream,
    get_config,
)
from .spec.constants import SLOTS_PER_EPOCH
from .network import Gossip, GossipConfig, MessageType
from .engine import EngineAPIClient, ForkchoiceState, PayloadStatusEnum
from .store import Store
from .beacon_api import BeaconAPI
from .crypto import hash_tree_root, sign, compute_signing_root
from .validator import ValidatorClient, load_keystores_teku_style
from .builder import BlockBuilder

logger = logging.getLogger(__name__)


def decode_signed_beacon_block(ssz_bytes: bytes) -> AnySignedBeaconBlock:
    """Decode a signed beacon block, trying different fork types."""
    block_types = [
        ("Gloas", SignedBeaconBlock),
        ("Electra", SignedElectraBeaconBlock),
        ("Deneb", SignedDenebBeaconBlock),
        ("Capella", SignedCapellaBeaconBlock),
        ("Bellatrix", SignedBellatrixBeaconBlock),
    ]

    for fork_name, block_type in block_types:
        try:
            block = block_type.decode_bytes(ssz_bytes)
            logger.debug(f"Block decoded as {fork_name} format")
            return block
        except Exception as e:
            logger.debug(f"Failed to decode block as {fork_name}: {e}")

    raise ValueError("Failed to decode block as any known format")


class BeaconNode:
    """Main beacon node implementation."""

    def __init__(self, config: Config):
        self.config = config
        self.state: Optional[AnyBeaconState] = None
        self.head_root: Optional[bytes] = None
        self.head_slot: int = 0

        self.store = Store(config.data_dir)

        gossip_config = GossipConfig(
            listen_host=config.listen_host,
            listen_port=config.listen_port,
            peers=config.peers,
        )
        self.gossip = Gossip(gossip_config)
        self.engine: Optional[EngineAPIClient] = None
        self.beacon_api: Optional[BeaconAPI] = None
        self.validator_client: Optional[ValidatorClient] = None
        self.remote_beacon: Optional[RemoteBeaconClient] = None
        self.state_sync: Optional[StateSyncManager] = None
        self.block_builder: Optional[BlockBuilder] = None
        self.beacon_gossip: Optional[BeaconGossip] = None

        self._running = False
        self._genesis_time: int = 0
        self._slot_ticker_task: Optional[asyncio.Task] = None
        self._current_payload_id: Optional[bytes] = None

    async def start(self) -> None:
        """Start the beacon node."""
        logger.info("Starting consensoor beacon node")
        logger.info(f"Using preset: {self.config.preset}")

        if self.config.network_config_path:
            load_network_config(self.config.network_config_path)
            logger.info(f"Loaded network config from {self.config.network_config_path}")
        else:
            logger.info(f"Fetching {self.config.preset} config from upstream")
            load_config_from_upstream(self.config.preset)

        await self._load_genesis_state()
        await self._load_validator_keys()
        await self._init_engine_client()
        self.block_builder = BlockBuilder(self)
        await self._setup_beacon_sync()
        await self._setup_gossip()
        await self._setup_p2p()
        await self._setup_beacon_api()

        self._running = True
        self._slot_ticker_task = asyncio.create_task(self._slot_ticker())

        logger.info(f"Beacon node started at slot {self.head_slot}")

    async def stop(self) -> None:
        """Stop the beacon node."""
        logger.info("Stopping beacon node")
        self._running = False

        if self._slot_ticker_task:
            self._slot_ticker_task.cancel()
            try:
                await self._slot_ticker_task
            except asyncio.CancelledError:
                pass

        await self.gossip.stop()
        if self.beacon_gossip:
            await self.beacon_gossip.stop()
        if self.state_sync:
            await self.state_sync.stop()
        if self.engine:
            await self.engine.close()
        if self.beacon_api:
            await self.beacon_api.stop()

    async def _load_genesis_state(self) -> None:
        """Load the genesis state."""
        if not self.config.genesis_state_path:
            raise ValueError("Genesis state path not configured")

        path = Path(self.config.genesis_state_path)
        if not path.exists():
            raise FileNotFoundError(f"Genesis state not found: {path}")

        logger.info(f"Loading genesis state from {path}")
        with open(path, "rb") as f:
            ssz_bytes = f.read()

        state_types = [
            ("Gloas", BeaconState),
            ("Fulu", FuluBeaconState),
            ("Electra", ElectraBeaconState),
            ("Deneb", DenebBeaconState),
            ("Capella", CapellaBeaconState),
            ("Bellatrix", BellatrixBeaconState),
            ("Altair", AltairBeaconState),
            ("Phase0", Phase0BeaconState),
        ]

        for fork_name, state_type in state_types:
            try:
                self.state = state_type.decode_bytes(ssz_bytes)
                fork_version = bytes(self.state.fork.current_version)
                logger.info(f"Genesis state parsed as {fork_name} format (fork_version={fork_version.hex()})")
                break
            except Exception as e:
                logger.debug(f"Failed to parse as {fork_name}: {e}")
        else:
            raise ValueError("Failed to parse genesis state as any known format")

        # Compute genesis block root per the spec:
        # The state's latest_block_header.state_root is ZERO_HASH at genesis.
        # The genesis block root is computed by filling in the actual state_root.
        from .spec.types import BeaconBlockHeader
        genesis_state_root = hash_tree_root(self.state)
        header = self.state.latest_block_header
        logger.info(
            f"Genesis header from state: slot={header.slot}, proposer_index={header.proposer_index}, "
            f"parent_root={bytes(header.parent_root).hex()[:16]}, "
            f"state_root={bytes(header.state_root).hex()[:16]}, "
            f"body_root={bytes(header.body_root).hex()[:16]}"
        )
        logger.info(f"Computed genesis state root: {genesis_state_root.hex()}")
        genesis_block_header = BeaconBlockHeader(
            slot=header.slot,
            proposer_index=header.proposer_index,
            parent_root=header.parent_root,
            state_root=genesis_state_root,
            body_root=header.body_root,
        )
        self.head_root = hash_tree_root(genesis_block_header)
        self.head_slot = int(self.state.slot)
        self._genesis_time = int(self.state.genesis_time)

        self.store.save_state(self.head_root, self.state)
        self.store.set_head(self.head_root)
        logger.info(f"Genesis block root: {self.head_root.hex()}")

    async def _load_validator_keys(self) -> None:
        """Load validator keys if configured."""
        if not self.config.validator_keys_spec:
            logger.info("No validator keys configured, running as non-validator")
            self.validator_client = ValidatorClient([])
            return

        try:
            keys = load_keystores_teku_style(self.config.validator_keys_spec)
            self.validator_client = ValidatorClient(keys)
            logger.info(f"Loaded {len(keys)} validator keys")

            if self.state:
                self.validator_client.update_validator_indices(self.state)
        except Exception as e:
            logger.error(f"Failed to load validator keys: {e}")
            self.validator_client = ValidatorClient([])

    async def _init_engine_client(self) -> None:
        """Initialize the Engine API client."""
        if not self.config.engine_api_url:
            logger.warning("Engine API URL not configured, running without EL")
            return

        self.engine = EngineAPIClient(
            url=self.config.engine_api_url,
            jwt_secret=self.config.jwt_secret,
        )
        self.engine.set_genesis_time(self._genesis_time)

        try:
            capabilities = await self.engine.exchange_capabilities()
            logger.info(f"Engine API capabilities: {capabilities}")
        except Exception as e:
            logger.error(f"Failed to connect to Engine API: {e}")

    async def _setup_beacon_sync(self) -> None:
        """Set up beacon sync from upstream beacon node if configured."""
        if not self.config.checkpoint_sync_url:
            logger.info("No checkpoint sync URL configured, running standalone")
            return

        self.remote_beacon = RemoteBeaconClient(self.config.checkpoint_sync_url)
        self.state_sync = StateSyncManager(self.remote_beacon, self)

        try:
            await self.state_sync.start()
            logger.info(f"Checkpoint sync enabled from {self.config.checkpoint_sync_url}")
        except Exception as e:
            logger.error(f"Failed to start beacon sync: {e}")
            self.state_sync = None
            self.remote_beacon = None

    async def _setup_gossip(self) -> None:
        """Set up gossip message handlers."""
        self.gossip.subscribe(MessageType.BEACON_BLOCK, self._on_block)
        self.gossip.subscribe(MessageType.EXECUTION_PAYLOAD_BID, self._on_bid)
        self.gossip.subscribe(MessageType.EXECUTION_PAYLOAD, self._on_payload)
        self.gossip.subscribe(MessageType.PAYLOAD_ATTESTATION, self._on_ptc_attestation)
        self.gossip.subscribe(MessageType.ATTESTATION, self._on_attestation)

        await self.gossip.start()

    async def _setup_p2p(self) -> None:
        """Set up libp2p-based gossipsub for Ethereum P2P networking."""
        if not self.state:
            logger.warning("Cannot start P2P: no genesis state loaded")
            return

        try:
            from .spec.network_config import get_config as get_network_config
            from .spec import constants
            from .p2p import extract_fork_digest_from_enr

            fork_version = bytes(self.state.fork.current_version)
            genesis_validators_root = bytes(self.state.genesis_validators_root)
            net_config = get_network_config()

            current_epoch = self.head_slot // constants.SLOTS_PER_EPOCH()
            next_fork_version, next_fork_epoch = self._get_next_fork_info(net_config, current_epoch)

            # Try to extract fork_digest from bootnode ENRs for compatibility
            fork_digest_override = None
            for peer in self.config.peers:
                if peer.startswith("enr:") or peer.startswith("-"):
                    extracted = extract_fork_digest_from_enr(peer)
                    if extracted:
                        fork_digest_override = extracted
                        logger.info(f"Using fork_digest from bootnode ENR: {extracted.hex()}")
                        break

            self.beacon_gossip = BeaconGossip(
                fork_version=fork_version,
                genesis_validators_root=genesis_validators_root,
                listen_port=self.config.listen_port,
                static_peers=self.config.peers,
                next_fork_version=next_fork_version,
                next_fork_epoch=next_fork_epoch,
                fork_digest_override=fork_digest_override,
            )

            self.beacon_gossip.subscribe_blocks(self._on_p2p_block)
            self.beacon_gossip.subscribe_aggregates(self._on_p2p_aggregate)

            await self.beacon_gossip.start()
            await self.beacon_gossip.activate_subscriptions()

            logger.info(
                f"P2P gossipsub started: peer_id={self.beacon_gossip.peer_id}, "
                f"fork_digest={self.beacon_gossip.fork_digest.hex()}"
            )

        except ImportError as e:
            logger.warning(f"libp2p not available, running without P2P: {e}")
            self.beacon_gossip = None
        except Exception as e:
            logger.error(f"Failed to start P2P: {e}")
            self.beacon_gossip = None

    def _get_next_fork_info(self, net_config, current_epoch: int) -> tuple[bytes, int]:
        """Get the next scheduled fork version and epoch.

        Returns (next_fork_version, next_fork_epoch).
        If no future fork is scheduled, returns (current_fork_version, FAR_FUTURE_EPOCH).
        """
        FAR_FUTURE_EPOCH = 2**64 - 1

        forks = [
            (net_config.altair_fork_epoch, net_config.altair_fork_version),
            (net_config.bellatrix_fork_epoch, net_config.bellatrix_fork_version),
            (net_config.capella_fork_epoch, net_config.capella_fork_version),
            (net_config.deneb_fork_epoch, net_config.deneb_fork_version),
            (net_config.electra_fork_epoch, net_config.electra_fork_version),
            (net_config.fulu_fork_epoch, net_config.fulu_fork_version),
            (net_config.gloas_fork_epoch, net_config.gloas_fork_version),
        ]

        for fork_epoch, fork_version in forks:
            if fork_epoch > current_epoch and fork_epoch < FAR_FUTURE_EPOCH:
                return fork_version, fork_epoch

        current_version = net_config.get_fork_version(current_epoch)
        return current_version, FAR_FUTURE_EPOCH

    async def _setup_beacon_api(self) -> None:
        """Set up Beacon API server."""
        self.beacon_api = BeaconAPI(
            node=self,
            host=self.config.listen_host,
            port=self.config.beacon_api_port,
        )
        await self.beacon_api.start()

    async def _slot_ticker(self) -> None:
        """Tick every slot and process duties."""
        network_config = get_config()
        seconds_per_slot = network_config.seconds_per_slot

        while self._running:
            now = int(time.time())
            current_slot = (now - self._genesis_time) // seconds_per_slot

            if current_slot > self.head_slot:
                await self._on_slot(current_slot)

            next_slot_time = self._genesis_time + (current_slot + 1) * seconds_per_slot
            sleep_time = max(0.1, next_slot_time - time.time())
            await asyncio.sleep(sleep_time)

    async def _on_slot(self, slot: int) -> None:
        """Handle a new slot."""
        slots_per_epoch = SLOTS_PER_EPOCH()
        epoch = slot // slots_per_epoch
        logger.info(f"Slot {slot} (epoch {epoch})")

        # NOTE: Don't update head_slot here - it should only be updated when we
        # actually have a block for a slot (either produced locally or received via P2P)
        # The head_slot tracks the slot of the actual chain head, not the current clock

        if slot % slots_per_epoch == 0:
            logger.info(f"New epoch: {epoch}")

        # Update forkchoice to keep EL in sync
        await self._update_forkchoice_for_slot(slot)

        # Check if we should propose a block
        await self._maybe_propose_block(slot)

    async def _update_forkchoice_for_slot(self, slot: int) -> None:
        """Update forkchoice with EL at the start of each slot."""
        if not self.engine or not self.state:
            return

        try:
            # Get the latest block hash
            if hasattr(self.state, "latest_block_hash"):
                head_block_hash = bytes(self.state.latest_block_hash)
            else:
                head_block_hash = bytes(
                    self.state.latest_execution_payload_header.block_hash
                )

            # Finalized block hash
            if self.state.finalized_checkpoint.epoch > 0:
                finalized_hash = bytes(self.state.finalized_checkpoint.root)
            else:
                finalized_hash = b"\x00" * 32

            forkchoice_state = ForkchoiceState(
                head_block_hash=head_block_hash,
                safe_block_hash=head_block_hash,
                finalized_block_hash=finalized_hash,
            )

            # Prepare payload attributes for block building
            network_config = get_config()
            timestamp = self._genesis_time + slot * network_config.seconds_per_slot

            # Get prev_randao from state
            randao_mixes = self.state.randao_mixes
            epoch = slot // SLOTS_PER_EPOCH()
            prev_randao = bytes(randao_mixes[epoch % len(randao_mixes)])

            payload_attributes = {
                "timestamp": hex(timestamp),
                "prevRandao": "0x" + prev_randao.hex(),
                "suggestedFeeRecipient": "0x" + "00" * 20,  # Default fee recipient
                "withdrawals": [],
                "parentBeaconBlockRoot": "0x" + (self.head_root or b"\x00" * 32).hex(),
            }

            response = await self.engine.forkchoice_updated(
                forkchoice_state, payload_attributes, timestamp=timestamp
            )

            if response.payload_id:
                logger.debug(f"Payload building started: id={response.payload_id.hex()[:16]}")
                # Store payload_id for potential block proposal
                self._current_payload_id = response.payload_id
            else:
                self._current_payload_id = None

        except Exception as e:
            logger.error(f"Failed to update forkchoice for slot {slot}: {e}")

    async def _maybe_propose_block(self, slot: int) -> None:
        """Check if we're the proposer and produce a block if so."""
        if not self.validator_client or not self.validator_client.keys:
            return

        if not self.state:
            return

        proposer_key = self.validator_client.is_our_proposer_slot(self.state, slot)
        if not proposer_key:
            return

        logger.info(f"We are proposer for slot {slot}! Building block...")

        try:
            await self._produce_and_broadcast_block(slot, proposer_key)
        except Exception as e:
            logger.error(f"Failed to produce block for slot {slot}: {e}")

    async def _produce_and_broadcast_block(self, slot: int, proposer_key) -> None:
        """Produce and broadcast a block for the given slot."""
        if not self.engine or not self._current_payload_id:
            logger.warning("Cannot produce block: no engine or payload_id")
            return

        if not self.block_builder:
            logger.warning("Cannot produce block: no block builder")
            return

        try:
            network_config = get_config()
            timestamp = self._genesis_time + slot * network_config.seconds_per_slot
            payload_response = await self.engine.get_payload(self._current_payload_id, timestamp=timestamp)
            execution_payload_dict = payload_response.execution_payload
            block_hash = execution_payload_dict.get("blockHash", "unknown")
            block_number = execution_payload_dict.get("blockNumber", "unknown")
            logger.info(
                f"Got execution payload: block_hash={block_hash[:18]}, "
                f"block_number={block_number}, value={payload_response.block_value}"
            )

            signed_block = await self.block_builder.build_block(
                slot, proposer_key, execution_payload_dict
            )
            if signed_block is None:
                logger.error("Failed to build block")
                return

            block = signed_block.message
            block_root = hash_tree_root(block)
            execution_payload = block.body.execution_payload

            versioned_hashes = []
            parent_beacon_root = bytes(block.parent_root)
            execution_requests = []

            status = await self.engine.new_payload(
                execution_payload,
                versioned_hashes,
                parent_beacon_root,
                execution_requests,
                timestamp=int(execution_payload.timestamp),
            )

            if status.status != PayloadStatusEnum.VALID:
                logger.error(f"Execution payload invalid: {status.status}")
                if status.status == PayloadStatusEnum.SYNCING:
                    logger.info("EL is syncing, block may be valid later")
                else:
                    return

            new_block_hash = bytes(execution_payload.block_hash)
            forkchoice_state = ForkchoiceState(
                head_block_hash=new_block_hash,
                safe_block_hash=new_block_hash,
                finalized_block_hash=(
                    bytes(self.state.finalized_checkpoint.root)
                    if self.state and int(self.state.finalized_checkpoint.epoch) > 0
                    else b"\x00" * 32
                ),
            )

            fc_response = await self.engine.forkchoice_updated(forkchoice_state, timestamp=int(execution_payload.timestamp))
            logger.info(f"Forkchoice updated: {fc_response.payload_status.status}")

            self.store.save_block(block_root, signed_block)
            self.head_slot = slot
            self.head_root = block_root
            self.store.set_head(block_root)

            # Apply the produced block to state so forkchoice stays in sync
            await self._apply_block_to_state(block, block_root, signed_block)

            logger.info(
                f"Block produced and applied: slot={slot}, "
                f"root={block_root.hex()[:16]}, block_hash={new_block_hash.hex()[:16]}"
            )

            # Broadcast block via P2P gossipsub
            if self.beacon_gossip:
                try:
                    block_ssz = signed_block.encode_bytes()
                    await self.beacon_gossip.publish_block(block_ssz)
                    logger.info(f"Block published to P2P network: slot={slot}")
                except Exception as e:
                    logger.error(f"Failed to publish block to P2P: {e}")

        except Exception as e:
            logger.error(f"Block production failed: {e}")
            import traceback
            traceback.print_exc()

    async def _on_block(self, payload: bytes, sender: tuple[str, int]) -> None:
        """Handle a received beacon block."""
        try:
            signed_block = decode_signed_beacon_block(payload)
            block = signed_block.message
            block_root = hash_tree_root(block)

            logger.info(
                f"Received block: slot={block.slot}, root={block_root.hex()[:16]}"
            )

            if self.store.get_block(block_root):
                return

            self.store.save_block(block_root, signed_block)

            if int(block.slot) > self.head_slot:
                self.head_slot = int(block.slot)
                self.head_root = block_root
                self.store.set_head(block_root)

                # Apply full state transition
                await self._apply_block_to_state(block, block_root, signed_block)
                await self._update_forkchoice()

        except Exception as e:
            logger.error(f"Error processing block: {e}")

    async def _on_bid(self, payload: bytes, sender: tuple[str, int]) -> None:
        """Handle a received execution payload bid (ePBS)."""
        try:
            signed_bid = SignedExecutionPayloadBid.decode_bytes(payload)
            bid = signed_bid.message

            logger.debug(
                f"Received bid: slot={bid.slot}, builder={bid.builder_index}"
            )

            self.store.save_bid(int(bid.slot), signed_bid)

        except Exception as e:
            logger.error(f"Error processing bid: {e}")

    async def _on_payload(self, payload: bytes, sender: tuple[str, int]) -> None:
        """Handle a received execution payload envelope (ePBS)."""
        try:
            signed_envelope = SignedExecutionPayloadEnvelope.decode_bytes(payload)
            envelope = signed_envelope.message

            logger.info(
                f"Received payload: slot={envelope.slot}, "
                f"block_hash={bytes(envelope.payload.block_hash).hex()[:16]}"
            )

            payload_root = hash_tree_root(envelope)
            self.store.save_payload(payload_root, signed_envelope)

            if self.engine:
                await self._validate_execution_payload(envelope)

        except Exception as e:
            logger.error(f"Error processing payload: {e}")

    async def _on_ptc_attestation(
        self, payload: bytes, sender: tuple[str, int]
    ) -> None:
        """Handle a received payload attestation (ePBS)."""
        try:
            attestation = PayloadAttestationMessage.decode_bytes(payload)
            logger.debug(
                f"Received PTC attestation: slot={attestation.data.slot}, "
                f"validator={attestation.validator_index}"
            )
        except Exception as e:
            logger.error(f"Error processing PTC attestation: {e}")

    async def _on_attestation(self, payload: bytes, sender: tuple[str, int]) -> None:
        """Handle a received attestation."""
        try:
            attestation = Attestation.decode_bytes(payload)
            logger.debug(f"Received attestation: slot={attestation.data.slot}")
        except Exception as e:
            logger.error(f"Error processing attestation: {e}")

    async def _on_p2p_block(self, data: bytes, from_peer: str) -> None:
        """Handle a beacon block received via libp2p gossipsub."""
        try:
            signed_block = decode_signed_beacon_block(data)
            block = signed_block.message
            block_root = hash_tree_root(block)
            parent_root = bytes(block.parent_root)

            logger.info(
                f"P2P: Received block slot={block.slot}, "
                f"root={block_root.hex()[:16]}, parent={parent_root.hex()[:16]}, "
                f"from={from_peer[:16]}"
            )

            if self.store.get_block(block_root):
                logger.debug(f"Block already known: {block_root.hex()[:16]}")
                return

            # Check if this block builds on our current head
            # If not, we can't apply it without state regeneration
            logger.debug(
                f"P2P: Parent check: parent_root={parent_root.hex()[:16]}, "
                f"head_root={self.head_root.hex()[:16] if self.head_root else 'None'}, "
                f"match={parent_root == self.head_root if self.head_root else 'N/A'}"
            )
            if self.head_root and parent_root != self.head_root:
                # Check if the parent is in our store (we know about it)
                if not self.store.get_block(parent_root):
                    logger.warning(
                        f"P2P: Ignoring block slot={block.slot} - parent {parent_root.hex()[:16]} "
                        f"not found (our head={self.head_root.hex()[:16] if self.head_root else 'None'})"
                    )
                    # Store the block anyway for later processing
                    self.store.save_block(block_root, signed_block)
                    return
                else:
                    logger.warning(
                        f"P2P: Block slot={block.slot} builds on different chain "
                        f"(parent={parent_root.hex()[:16]}, our head={self.head_root.hex()[:16]}). "
                        f"Would need state regeneration to apply."
                    )
                    # Store the block anyway for later processing
                    self.store.save_block(block_root, signed_block)
                    return

            self.store.save_block(block_root, signed_block)

            logger.debug(
                f"P2P: Block saved. slot={block.slot}, head_slot={self.head_slot}, "
                f"head_root={self.head_root.hex()[:16] if self.head_root else 'None'}"
            )

            if int(block.slot) > self.head_slot:
                logger.info(f"P2P: Attempting to adopt block slot={block.slot} as new head")
                # Apply state transition FIRST, only update head if successful
                old_head = self.head_root
                try:
                    logger.debug(f"P2P: Calling _apply_block_to_state for slot={block.slot}")
                    await self._apply_block_to_state(block, block_root, signed_block)
                    logger.debug(f"P2P: _apply_block_to_state succeeded for slot={block.slot}")
                    # State transition succeeded, update head
                    self.head_slot = int(block.slot)
                    self.head_root = block_root
                    self.store.set_head(block_root)
                    logger.info(
                        f"P2P: Adopted block as new head: slot={block.slot}, "
                        f"root={block_root.hex()[:16]}"
                    )
                    await self._update_forkchoice()
                except Exception as e:
                    logger.warning(
                        f"P2P: Failed to apply block slot={block.slot}: {e}. "
                        f"Keeping head at {old_head.hex()[:16] if old_head else 'None'}"
                    )
            else:
                logger.debug(f"P2P: Block slot={block.slot} not newer than head_slot={self.head_slot}")

        except Exception as e:
            logger.error(f"Error processing P2P block: {e}")

    async def _on_p2p_aggregate(self, data: bytes, from_peer: str) -> None:
        """Handle an aggregate attestation received via libp2p gossipsub."""
        try:
            attestation = Attestation.decode_bytes(data)
            logger.debug(
                f"P2P: Received aggregate slot={attestation.data.slot}, "
                f"from={from_peer[:16]}"
            )
        except Exception as e:
            logger.error(f"Error processing P2P aggregate: {e}")

    async def _apply_block_to_state(self, block, block_root: bytes, signed_block=None) -> None:
        """Apply a received block to the local state using full state transition.

        Executes the complete state transition function per the consensus specs.
        Raises an exception if state transition fails.

        Args:
            block: Beacon block to apply
            block_root: Root hash of the block
            signed_block: Optional signed block for signature verification

        Raises:
            Exception: If state transition fails for any reason
        """
        if not self.state:
            raise ValueError("No state available")

        from .spec.state_transition import state_transition, process_slots, process_block

        # If we have a signed block, use full state transition with signature verification
        if signed_block is not None:
            new_state = state_transition(
                self.state, signed_block, validate_result=False
            )
            self.state = new_state
            logger.info(
                f"Full state transition applied: slot={block.slot}, "
                f"block_hash={bytes(block.body.execution_payload.block_hash).hex()[:16] if hasattr(block.body, 'execution_payload') else 'N/A'}, "
                f"latest_header_slot={self.state.latest_block_header.slot}"
            )
            return

        # Fallback: process slots to advance state, then process block
        target_slot = int(block.slot)
        if target_slot > int(self.state.slot):
            process_slots(self.state, target_slot)
        process_block(self.state, block)
        logger.info(
            f"Block processed: slot={block.slot}, "
            f"block_hash={bytes(block.body.execution_payload.block_hash).hex()[:16] if hasattr(block.body, 'execution_payload') else 'N/A'}, "
            f"latest_header_slot={self.state.latest_block_header.slot}"
        )

    async def _apply_minimal_block_update(self, block) -> None:
        """Apply minimal block updates when full state transition fails.

        This is a degraded fallback mode. Updates only the essential fields
        needed for forkchoice tracking. The state will be in an inconsistent
        state until a full state sync occurs.

        Per the spec, latest_block_header.state_root is set to zero when a
        block is processed, and filled in by process_slot at the next slot.
        We maintain this invariant even in degraded mode to allow recovery
        via state sync.
        """
        if not self.state:
            return

        from .spec.types import BeaconBlockHeader

        # Update slot
        self.state.slot = block.slot

        # Update latest_block_header with state_root=0 per spec
        # (state_root is filled in by process_slot at next slot boundary)
        self.state.latest_block_header = BeaconBlockHeader(
            slot=block.slot,
            proposer_index=block.proposer_index,
            parent_root=block.parent_root,
            state_root=b"\x00" * 32,  # Zero per spec, filled in by process_slot
            body_root=hash_tree_root(block.body),
        )
        logger.info(
            f"Minimal block update applied: slot={block.slot}, "
            f"latest_header_slot={self.state.latest_block_header.slot}"
        )

        # Update execution payload header if present
        if hasattr(block.body, "execution_payload"):
            payload = block.body.execution_payload
            if hasattr(self.state, "latest_execution_payload_header"):
                def get_view_root(view) -> bytes:
                    if hasattr(view, 'hash_tree_root'):
                        root = view.hash_tree_root()
                        return bytes(root) if not isinstance(root, bytes) else root
                    return hash_tree_root(view)

                base_fields = {
                    "parent_hash": payload.parent_hash,
                    "fee_recipient": payload.fee_recipient,
                    "state_root": payload.state_root,
                    "receipts_root": payload.receipts_root,
                    "logs_bloom": payload.logs_bloom,
                    "prev_randao": payload.prev_randao,
                    "block_number": payload.block_number,
                    "gas_limit": payload.gas_limit,
                    "gas_used": payload.gas_used,
                    "timestamp": payload.timestamp,
                    "extra_data": payload.extra_data,
                    "base_fee_per_gas": payload.base_fee_per_gas,
                    "block_hash": payload.block_hash,
                    "transactions_root": get_view_root(payload.transactions),
                }

                has_withdrawals = hasattr(payload, "withdrawals")
                has_blob_gas = hasattr(payload, "blob_gas_used")

                if not has_withdrawals:
                    from .spec.types.bellatrix import ExecutionPayloadHeaderBellatrix
                    self.state.latest_execution_payload_header = ExecutionPayloadHeaderBellatrix(**base_fields)
                elif not has_blob_gas:
                    from .spec.types.capella import ExecutionPayloadHeaderCapella
                    base_fields["withdrawals_root"] = get_view_root(payload.withdrawals)
                    self.state.latest_execution_payload_header = ExecutionPayloadHeaderCapella(**base_fields)
                else:
                    from .spec.types import ExecutionPayloadHeader
                    base_fields["withdrawals_root"] = get_view_root(payload.withdrawals)
                    base_fields["blob_gas_used"] = payload.blob_gas_used
                    base_fields["excess_blob_gas"] = payload.excess_blob_gas
                    self.state.latest_execution_payload_header = ExecutionPayloadHeader(**base_fields)

        logger.debug(
            f"Minimal state update from block: slot={block.slot}, "
            f"block_hash={bytes(block.body.execution_payload.block_hash).hex()[:16] if hasattr(block.body, 'execution_payload') else 'N/A'}"
        )

    async def _validate_execution_payload(
        self, envelope: ExecutionPayloadEnvelope
    ) -> bool:
        """Validate an execution payload with the EL."""
        if not self.engine:
            return True

        try:
            versioned_hashes = []
            parent_beacon_root = bytes(envelope.beacon_block_root)
            execution_requests = []

            status = await self.engine.new_payload(
                envelope.payload,
                versioned_hashes,
                parent_beacon_root,
                execution_requests,
                timestamp=int(envelope.payload.timestamp),
            )

            if status.status == PayloadStatusEnum.VALID:
                logger.info("Execution payload validated")
                return True
            else:
                logger.warning(f"Payload validation failed: {status.status}")
                return False

        except Exception as e:
            logger.error(f"Failed to validate payload: {e}")
            return False

    async def _update_forkchoice(self) -> None:
        """Update forkchoice with the execution layer."""
        if not self.engine or not self.state:
            return

        try:
            # Get the latest block hash and timestamp based on state type
            # Gloas has latest_block_hash, Fulu/Electra use execution payload header
            if hasattr(self.state, "latest_block_hash"):
                head_block_hash = bytes(self.state.latest_block_hash)
                timestamp = int(self.state.latest_execution_payload_header.timestamp) if hasattr(self.state, "latest_execution_payload_header") else int(time.time())
            else:
                head_block_hash = bytes(
                    self.state.latest_execution_payload_header.block_hash
                )
                timestamp = int(self.state.latest_execution_payload_header.timestamp)

            forkchoice_state = ForkchoiceState(
                head_block_hash=head_block_hash,
                safe_block_hash=head_block_hash,
                finalized_block_hash=bytes(
                    self.state.finalized_checkpoint.root
                )
                if self.state.finalized_checkpoint.epoch > 0
                else b"\x00" * 32,
            )

            response = await self.engine.forkchoice_updated(forkchoice_state, timestamp=timestamp)
            logger.debug(f"Forkchoice updated: {response.payload_status.status}")

        except Exception as e:
            logger.error(f"Failed to update forkchoice: {e}")

    @property
    def current_slot(self) -> int:
        """Get the current slot based on time."""
        network_config = get_config()
        now = int(time.time())
        return (now - self._genesis_time) // network_config.seconds_per_slot

    @property
    def current_epoch(self) -> int:
        """Get the current epoch."""
        return self.current_slot // SLOTS_PER_EPOCH()


async def run_node(config: Config) -> None:
    """Run the beacon node."""
    node = BeaconNode(config)

    try:
        await node.start()
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await node.stop()
