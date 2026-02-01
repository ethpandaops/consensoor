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
    BLSSignature,
    Eth1Data,
)
from .spec.types.electra import (
    ElectraBeaconBlock,
    ElectraBeaconBlockBody,
    ExecutionRequests,
)
from .spec.types.altair import SyncAggregate
from .spec.types.deneb import ExecutionPayload
from .spec.types.base import Bytes32
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
from .spec.state_transition import process_slots
from .network import Gossip, GossipConfig, MessageType
from .engine import EngineAPIClient, ForkchoiceState, PayloadStatusEnum
from .store import Store
from .beacon_api import BeaconAPI
from .crypto import hash_tree_root, sign, compute_signing_root
from .validator import ValidatorClient, load_keystores_teku_style
from .builder import BlockBuilder
from .attestation_pool import AttestationPool
from .sync_committee_pool import SyncCommitteePool
from . import metrics

logger = logging.getLogger(__name__)


def decode_signed_beacon_block(ssz_bytes: bytes) -> AnySignedBeaconBlock:
    """Decode a signed beacon block, trying different fork types.

    Tries forks from newest to oldest (excluding Gloas for now since it has a
    completely different structure with SignedExecutionPayloadBid).
    Electra/Fulu use the same block format.
    """
    from .spec.network_config import get_config
    from .spec.constants import SLOTS_PER_EPOCH

    config = get_config()

    # Try to peek at slot to determine fork (slot is at fixed offset in SSZ)
    # SignedBeaconBlock: message offset (4 bytes) + signature (96 bytes) at start
    # BeaconBlock: slot (8 bytes) is first field
    # So slot is at offset 4 (after the message offset)
    try:
        slot = int.from_bytes(ssz_bytes[4:12], "little")
        epoch = slot // SLOTS_PER_EPOCH()

        # Determine expected fork based on slot
        if hasattr(config, 'gloas_fork_epoch') and epoch >= config.gloas_fork_epoch:
            try:
                block = SignedBeaconBlock.decode_bytes(ssz_bytes)
                logger.debug(f"Block slot={slot} decoded as Gloas format")
                return block
            except Exception:
                pass

        if hasattr(config, 'electra_fork_epoch') and epoch >= config.electra_fork_epoch:
            try:
                block = SignedElectraBeaconBlock.decode_bytes(ssz_bytes)
                logger.debug(f"Block slot={slot} decoded as Electra/Fulu format")
                return block
            except Exception:
                pass

        if hasattr(config, 'deneb_fork_epoch') and epoch >= config.deneb_fork_epoch:
            try:
                block = SignedDenebBeaconBlock.decode_bytes(ssz_bytes)
                logger.debug(f"Block slot={slot} decoded as Deneb format")
                return block
            except Exception:
                pass

        if hasattr(config, 'capella_fork_epoch') and epoch >= config.capella_fork_epoch:
            try:
                block = SignedCapellaBeaconBlock.decode_bytes(ssz_bytes)
                logger.debug(f"Block slot={slot} decoded as Capella format")
                return block
            except Exception:
                pass

        # Bellatrix or earlier
        try:
            block = SignedBellatrixBeaconBlock.decode_bytes(ssz_bytes)
            logger.debug(f"Block slot={slot} decoded as Bellatrix format")
            return block
        except Exception:
            pass

    except Exception as e:
        logger.debug(f"Failed to peek at slot: {e}, falling back to brute force")

    # Fallback: try all types from newest to oldest (excluding Gloas)
    block_types = [
        ("Electra", SignedElectraBeaconBlock),
        ("Deneb", SignedDenebBeaconBlock),
        ("Capella", SignedCapellaBeaconBlock),
        ("Bellatrix", SignedBellatrixBeaconBlock),
    ]

    for fork_name, block_type in block_types:
        try:
            block = block_type.decode_bytes(ssz_bytes)
            logger.debug(f"Block decoded as {fork_name} format (fallback)")
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
        self.attestation_pool = AttestationPool()
        self.sync_committee_pool = SyncCommitteePool()
        self._attester_duties: dict[int, list] = {}  # epoch -> duties
        self._sync_committee_duties: dict[int, list] = {}  # validator_index -> committee positions

        self._running = False
        self._genesis_time: int = 0
        self._slot_ticker_task: Optional[asyncio.Task] = None
        self._current_payload_id: Optional[bytes] = None
        self._current_payload_beacon_root: Optional[bytes] = None  # Store head_root used in forkchoiceUpdated
        self._fork_digest_override: Optional[bytes] = None  # For devnets with custom fork digests

    async def start(self) -> None:
        """Start the beacon node."""
        logger.info("Starting consensoor beacon node")
        logger.info(f"Using preset: {self.config.preset}")

        # Start metrics server
        metrics.start_metrics_server(self.config.metrics_port)
        from .version import get_cl_version
        metrics.set_node_info(
            version=get_cl_version(),
            network=self.config.network_config_path or "kurtosis",
            preset=self.config.preset,
        )

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

        # Prepare initial payload for slot 1 (or current slot + 1)
        await self._prepare_initial_payload()

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

        detected_fork = None
        for fork_name, state_type in state_types:
            try:
                self.state = state_type.decode_bytes(ssz_bytes)
                fork_version = bytes(self.state.fork.current_version)
                logger.info(f"Genesis state parsed as {fork_name} format (fork_version={fork_version.hex()})")
                detected_fork = fork_name.lower()
                break
            except Exception as e:
                logger.debug(f"Failed to parse as {fork_name}: {e}")
        else:
            raise ValueError("Failed to parse genesis state as any known format")

        # Update config fork epochs based on detected genesis fork
        # This ensures correct fork detection even when config YAML isn't loaded
        config = get_config()
        FAR_FUTURE_EPOCH = 2**64 - 1
        if detected_fork == "fulu" and config.fulu_fork_epoch == FAR_FUTURE_EPOCH:
            logger.info("Genesis state is Fulu - setting fulu_fork_epoch=0 in config")
            config.fulu_fork_epoch = 0
            # Also set earlier forks to epoch 0 since Fulu implies all previous forks
            config.electra_fork_epoch = 0
            config.deneb_fork_epoch = 0
            config.capella_fork_epoch = 0
            config.bellatrix_fork_epoch = 0
            config.altair_fork_epoch = 0
        elif detected_fork == "gloas" and config.gloas_fork_epoch == FAR_FUTURE_EPOCH:
            logger.info("Genesis state is GLOAS - setting gloas_fork_epoch=0 in config")
            config.gloas_fork_epoch = 0
            config.fulu_fork_epoch = 0
            config.electra_fork_epoch = 0
            config.deneb_fork_epoch = 0
            config.capella_fork_epoch = 0
            config.bellatrix_fork_epoch = 0
            config.altair_fork_epoch = 0

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

        # Save genesis block root and state for potential reorg handling
        self._genesis_block_root = self.head_root
        self._genesis_state = self.state.__class__.decode_bytes(bytes(self.state.encode_bytes()))

        # Save state by genesis_state_root (original hash) BEFORE modifying latest_block_header
        self.store.save_state(genesis_state_root, self.state)

        # Update state's latest_block_header with actual state_root so that
        # hash_tree_root(state.latest_block_header) == head_root
        # This is critical for Dora/explorers that compute block root from state
        self.state.latest_block_header.state_root = genesis_state_root
        logger.info(f"Updated state.latest_block_header.state_root to {genesis_state_root.hex()[:16]}")

        # Also save state by head_root (block_root) for flexibility
        self.store.save_state(self.head_root, self.state)
        self.store.set_head(self.head_root)
        logger.info(f"Genesis block root: {self.head_root.hex()}")

        # Create and save genesis block for API compatibility
        # The genesis block body is empty with default values
        genesis_body = ElectraBeaconBlockBody(
            randao_reveal=BLSSignature(b'\x00' * 96),
            eth1_data=Eth1Data(),
            graffiti=Bytes32(b'\x00' * 32),
            proposer_slashings=[],
            attester_slashings=[],
            attestations=[],
            deposits=[],
            voluntary_exits=[],
            sync_aggregate=SyncAggregate(),
            execution_payload=ExecutionPayload(),
            bls_to_execution_changes=[],
            blob_kzg_commitments=[],
            execution_requests=ExecutionRequests(),
        )
        genesis_block = ElectraBeaconBlock(
            slot=int(header.slot),
            proposer_index=int(header.proposer_index),
            parent_root=header.parent_root,
            state_root=genesis_state_root,
            body=genesis_body,
        )
        signed_genesis_block = SignedElectraBeaconBlock(
            message=genesis_block,
            signature=BLSSignature(b'\x00' * 96),
        )
        self.store.save_block(self.head_root, signed_genesis_block)
        logger.info(f"Saved genesis block to store")

        # Update metrics
        from .spec import constants
        epoch = self.head_slot // constants.SLOTS_PER_EPOCH()
        metrics.update_head(self.head_slot, epoch)
        metrics.update_checkpoints(
            finalized=int(self.state.finalized_checkpoint.epoch),
            justified=int(self.state.current_justified_checkpoint.epoch),
        )

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

        try:
            el_info = await self.engine.get_client_version()
            if el_info:
                self.config.set_el_client_info(el_info)
                logger.info(f"EL client: {el_info.get('name', 'unknown')} commit={el_info.get('commit', 'unknown')}")
        except Exception as e:
            logger.debug(f"Could not get EL client version: {e}")

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
            blob_params = self._get_blob_params_for_epoch(net_config, current_epoch)

            # Try to extract fork_digest from bootnode ENRs for compatibility
            fork_digest_override = None
            for peer in self.config.peers:
                if peer.startswith("enr:") or peer.startswith("-"):
                    extracted = extract_fork_digest_from_enr(peer)
                    if extracted:
                        fork_digest_override = extracted
                        self._fork_digest_override = extracted  # Store for epoch updates
                        logger.info(f"Using fork_digest from bootnode ENR: {extracted.hex()}")
                        break

            # Compute all fork digests for multi-fork subscription
            all_fork_digests = self._get_all_fork_digests(net_config, genesis_validators_root)
            logger.info(f"All fork digests for subscription: {[d.hex() for d in all_fork_digests]}")

            self.beacon_gossip = BeaconGossip(
                fork_version=fork_version,
                genesis_validators_root=genesis_validators_root,
                listen_port=self.config.listen_port,
                static_peers=self.config.peers,
                next_fork_version=next_fork_version,
                next_fork_epoch=next_fork_epoch,
                fork_digest_override=fork_digest_override,
                blob_params=blob_params,
                supernode=self.config.supernode,
                all_fork_digests=all_fork_digests,
            )

            self.beacon_gossip.subscribe_blocks(self._on_p2p_block)
            self.beacon_gossip.subscribe_aggregates(self._on_p2p_aggregate)
            self.beacon_gossip.subscribe_sync_committee_contributions(self._on_p2p_sync_committee_contribution)
            self.beacon_gossip.subscribe_blob_sidecars(self._on_p2p_blob_sidecar)
            self.beacon_gossip.set_status_provider(self._get_chain_status)

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

    def _get_chain_status(self) -> dict:
        """Get current chain status for P2P status messages.

        Returns a dict with head_slot, head_root, finalized_epoch,
        finalized_root, and earliest_available_slot.
        """
        from .crypto import hash_tree_root

        head_slot = 0
        head_root = b"\x00" * 32
        finalized_epoch = 0
        finalized_root = b"\x00" * 32

        if self.state:
            head_slot = int(self.state.slot)
            finalized_epoch = int(self.state.finalized_checkpoint.epoch)
            finalized_root = bytes(self.state.finalized_checkpoint.root)

        if self.head_root:
            head_root = self.head_root
        elif self.state:
            head_root = hash_tree_root(self.state.latest_block_header)

        if self.store.finalized_root and self.store.finalized_root != b"\x00" * 32:
            finalized_root = self.store.finalized_root
        if self.store.finalized_epoch > 0:
            finalized_epoch = self.store.finalized_epoch

        return {
            "head_slot": head_slot,
            "head_root": head_root,
            "finalized_epoch": finalized_epoch,
            "finalized_root": finalized_root,
            "earliest_available_slot": 0,
        }

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

    def _get_all_fork_digests(self, net_config, genesis_validators_root: bytes) -> list[bytes]:
        """Get fork digests for all scheduled forks.

        Returns a list of fork_digests from genesis through future forks.
        This allows subscribing to gossipsub topics for all forks to handle
        fork transitions correctly.
        """
        from .p2p.encoding import compute_fork_digest
        from .spec import constants

        FAR_FUTURE_EPOCH = 2**64 - 1
        digests = []

        forks = [
            (net_config.genesis_fork_version, "genesis"),
            (net_config.altair_fork_version, "altair"),
            (net_config.bellatrix_fork_version, "bellatrix"),
            (net_config.capella_fork_version, "capella"),
            (net_config.deneb_fork_version, "deneb"),
            (net_config.electra_fork_version, "electra"),
            (net_config.fulu_fork_version, "fulu"),
        ]

        fork_epochs = [
            0,  # genesis
            net_config.altair_fork_epoch,
            net_config.bellatrix_fork_epoch,
            net_config.capella_fork_epoch,
            net_config.deneb_fork_epoch,
            net_config.electra_fork_epoch,
            net_config.fulu_fork_epoch,
        ]

        for i, (fork_version, fork_name) in enumerate(forks):
            fork_epoch = fork_epochs[i]
            if fork_epoch < FAR_FUTURE_EPOCH:
                blob_params = None
                if fork_name == "fulu":
                    # Base Fulu digest uses default max blobs (Electra), not BPO overrides
                    blob_params = (fork_epoch, constants.MAX_BLOBS_PER_BLOCK_ELECTRA)
                digest = compute_fork_digest(
                    fork_version,
                    genesis_validators_root,
                    blob_params=blob_params,
                )
                digests.append(digest)
                logger.debug(f"Fork digest for {fork_name}: {digest.hex()}")

        # Add BPO fork digests (PeerDAS blob param overrides)
        if getattr(net_config, "blob_schedule", None):
            for entry in net_config.blob_schedule:
                try:
                    entry_epoch = entry.get("epoch", entry.get("EPOCH"))
                    max_blobs = entry.get("max_blobs_per_block", entry.get("MAX_BLOBS_PER_BLOCK"))
                    if entry_epoch is None or max_blobs is None:
                        continue
                    entry_epoch = int(entry_epoch)
                    max_blobs = int(max_blobs)
                    if net_config.fulu_fork_epoch < FAR_FUTURE_EPOCH and entry_epoch >= net_config.fulu_fork_epoch:
                        digest = compute_fork_digest(
                            net_config.fulu_fork_version,
                            genesis_validators_root,
                            blob_params=(entry_epoch, max_blobs),
                        )
                        digests.append(digest)
                        logger.debug(f"BPO fork digest for epoch {entry_epoch}: {digest.hex()}")
                except Exception as e:
                    logger.debug(f"Failed to parse blob schedule entry {entry}: {e}")

        return digests

    def _get_blob_params_for_epoch(self, net_config, epoch: int) -> tuple[int, int] | None:
        """Get blob params (epoch, max_blobs) for fork digest modification."""
        from .spec import constants

        FAR_FUTURE_EPOCH = 2**64 - 1
        if epoch < net_config.fulu_fork_epoch or net_config.fulu_fork_epoch >= FAR_FUTURE_EPOCH:
            return None

        if getattr(net_config, "blob_schedule", None):
            selected_epoch = None
            selected_max_blobs = None
            for entry in net_config.blob_schedule:
                entry_epoch = entry.get("epoch", entry.get("EPOCH"))
                max_blobs = entry.get("max_blobs_per_block", entry.get("MAX_BLOBS_PER_BLOCK"))
                if entry_epoch is None or max_blobs is None:
                    continue
                entry_epoch = int(entry_epoch)
                max_blobs = int(max_blobs)
                if entry_epoch <= epoch and (selected_epoch is None or entry_epoch >= selected_epoch):
                    selected_epoch = entry_epoch
                    selected_max_blobs = max_blobs
            if selected_epoch is not None and selected_max_blobs is not None:
                return selected_epoch, selected_max_blobs

        # Default Fulu blob params when no schedule is present
        return net_config.fulu_fork_epoch, constants.MAX_BLOBS_PER_BLOCK_ELECTRA

    async def _update_fork_digest_for_epoch(self, epoch: int) -> None:
        """Update the fork_digest for publishing when crossing fork boundaries.

        Call this at the start of each epoch to ensure messages are published
        to the correct gossipsub topic.
        """
        if not self.beacon_gossip or not self.state:
            return

        # If we have a fork_digest_override (from devnet ENR), always use that
        # Devnets use custom fork digests that don't match computed values
        if self._fork_digest_override:
            self.beacon_gossip.update_fork_digest(self._fork_digest_override)
            return

        try:
            from .spec.network_config import get_config as get_network_config
            from .p2p.encoding import compute_fork_digest

            net_config = get_network_config()
            genesis_validators_root = bytes(self.state.genesis_validators_root)

            # Get the fork version for this epoch
            fork_version = net_config.get_fork_version(epoch)
            blob_params = self._get_blob_params_for_epoch(net_config, epoch)
            new_digest = compute_fork_digest(
                fork_version,
                genesis_validators_root,
                blob_params=blob_params,
            )

            self.beacon_gossip.update_fork_digest(new_digest)
        except Exception as e:
            logger.error(f"Failed to update fork_digest for epoch {epoch}: {e}")

    async def _setup_beacon_api(self) -> None:
        """Set up Beacon API server."""
        self.beacon_api = BeaconAPI(
            node=self,
            host=self.config.listen_host,
            port=self.config.beacon_api_port,
        )
        await self.beacon_api.start()

    async def _prepare_initial_payload(self) -> None:
        """Prepare initial payload for first slot after genesis.

        This ensures we have a payload_id ready when the first slot starts,
        so we can produce a block if we're the proposer.
        """
        if not self.engine or not self.state:
            logger.warning("Cannot prepare initial payload: no engine or state")
            return

        network_config = get_config()
        current_time = int(time.time())
        slot_duration_sec = network_config.slot_duration_ms // 1000
        current_slot = (current_time - self._genesis_time) // slot_duration_sec

        # Prepare for the next slot (current_slot + 1, or slot 1 if before genesis)
        target_slot = max(1, current_slot + 1)

        logger.info(f"Preparing initial payload for slot {target_slot}")

        # Call forkchoice_updated with payload_attributes for target_slot
        # We use slot - 1 as the "current slot" for _update_forkchoice_for_slot
        await self._update_forkchoice_for_slot(target_slot - 1)

    async def _request_payload_for_slot(self, slot: int) -> None:
        """Request a fresh payload for a specific slot.

        This is used when we need to build a block but don't have a valid payload_id,
        or when the existing payload_id might be for a different slot.
        """
        if not self.engine or not self.state:
            logger.warning("Cannot request payload: no engine or state")
            return

        try:
            network_config = get_config()
            timestamp = self._genesis_time + slot * (network_config.slot_duration_ms // 1000)

            # Get the current head block hash
            if hasattr(self.state, "latest_block_hash"):
                head_block_hash = bytes(self.state.latest_block_hash)
            else:
                head_block_hash = bytes(
                    self.state.latest_execution_payload_header.block_hash
                )

            forkchoice_state = ForkchoiceState(
                head_block_hash=head_block_hash,
                safe_block_hash=head_block_hash,
                finalized_block_hash=b"\x00" * 32,
            )

            # Get prev_randao from state for this slot
            # At epoch boundaries, we need special handling:
            # - The current state is at some slot (possibly in the old epoch)
            # - When process_slots advances to a new epoch, process_randao_mixes_reset
            #   copies the CURRENT epoch's mix to the NEW epoch's slot
            # - So for slots in a new epoch, we should use the current state's epoch mix
            randao_mixes = self.state.randao_mixes
            target_epoch = slot // SLOTS_PER_EPOCH()
            current_epoch = int(self.state.slot) // SLOTS_PER_EPOCH()

            if target_epoch > current_epoch:
                # Target slot is in a new epoch - use current epoch's mix
                # (which will be copied to the new epoch during epoch processing)
                prev_randao = bytes(randao_mixes[current_epoch % len(randao_mixes)])
                logger.debug(
                    f"Epoch boundary: using current_epoch={current_epoch} randao for target_epoch={target_epoch}"
                )
            else:
                # Same epoch - use the target epoch's mix directly
                prev_randao = bytes(randao_mixes[target_epoch % len(randao_mixes)])

            payload_attributes = {
                "timestamp": hex(timestamp),
                "prevRandao": "0x" + prev_randao.hex(),
                "suggestedFeeRecipient": "0x" + "00" * 20,
                "withdrawals": [],
                "parentBeaconBlockRoot": "0x" + (self.head_root or b"\x00" * 32).hex(),
            }

            # Debug: compare head_root with hash of latest_block_header
            latest_header_hash = hash_tree_root(self.state.latest_block_header)
            logger.info(
                f"Requesting payload for slot {slot}: head_hash={head_block_hash.hex()[:16]}, "
                f"timestamp={timestamp}, prev_randao={prev_randao.hex()[:16]}, "
                f"state_slot={self.state.slot}, target_epoch={target_epoch}, current_epoch={current_epoch}, "
                f"head_root={self.head_root.hex()[:16] if self.head_root else 'None'}, "
                f"latest_header_hash={latest_header_hash.hex()[:16]}"
            )

            response = await self.engine.forkchoice_updated(
                forkchoice_state, payload_attributes, timestamp=timestamp
            )

            if response.payload_id:
                logger.info(f"Got fresh payload_id for slot {slot}: {response.payload_id.hex()}")
                self._current_payload_id = response.payload_id
                # Store the beacon root used in this forkchoiceUpdated for later use in newPayloadV5
                self._current_payload_beacon_root = self.head_root or b"\x00" * 32
            else:
                logger.warning(
                    f"Failed to get payload_id for slot {slot}: status={response.payload_status.status}"
                )
                self._current_payload_id = None
                self._current_payload_beacon_root = None

        except Exception as e:
            logger.error(f"Failed to request payload for slot {slot}: {e}")
            self._current_payload_id = None
            self._current_payload_beacon_root = None

    async def _slot_ticker(self) -> None:
        """Tick every slot and process duties with proper intra-slot timing.

        Per Ethereum spec, the slot is divided into phases with timing from config:
        - 0: Block proposal window
        - ATTESTATION_DUE_BPS: Attestation production (default 33.33% = 1/3)
        - AGGREGATE_DUE_BPS: Aggregation window (default 66.67% = 2/3)

        Gloas (ePBS) uses different timing:
        - ATTESTATION_DUE_BPS_GLOAS: 25% of slot
        - AGGREGATE_DUE_BPS_GLOAS: 50% of slot
        """
        network_config = get_config()
        slot_duration = network_config.slot_duration_ms / 1000.0

        last_slot_processed = -1
        last_attestation_slot = -1

        while self._running:
            try:
                now = time.time()
                current_slot = int((now - self._genesis_time) // slot_duration)
                slot_start_time = self._genesis_time + current_slot * slot_duration
                time_into_slot = now - slot_start_time

                # Calculate current epoch for fork-aware timing
                slots_per_epoch = SLOTS_PER_EPOCH()
                current_epoch = current_slot // slots_per_epoch

                # Get attestation timing based on active fork (Gloas has different timing)
                attestation_offset = network_config.get_attestation_due_offset(current_epoch)

                # Phase 1: Slot start - propose blocks and update forkchoice
                if current_slot > last_slot_processed:
                    last_slot_processed = current_slot
                    await self._on_slot_start(current_slot)

                # Phase 2: Attestation due time - produce attestations
                if current_slot > last_attestation_slot and time_into_slot >= attestation_offset:
                    last_attestation_slot = current_slot
                    await self._produce_attestations(current_slot)

                # Calculate sleep time - wake up for next event
                if current_slot == last_attestation_slot:
                    # Already attested this slot, sleep until next slot
                    next_slot_time = self._genesis_time + (current_slot + 1) * slot_duration
                    sleep_time = max(0.05, next_slot_time - time.time())
                else:
                    # Need to attest this slot, sleep until attestation due time
                    attestation_time = slot_start_time + attestation_offset
                    sleep_time = max(0.05, attestation_time - time.time())

                await asyncio.sleep(sleep_time)
            except asyncio.CancelledError:
                logger.info("Slot ticker cancelled")
                raise
            except Exception as e:
                logger.error(f"Slot ticker error: {e}", exc_info=True)
                await asyncio.sleep(1)  # Prevent tight loop on errors

    async def _on_slot_start(self, slot: int) -> None:
        """Handle the start of a new slot (0/3 mark).

        This is called at slot start for block proposals and forkchoice updates.
        Attestations are produced separately at the 1/3 mark by the slot ticker.
        """
        slots_per_epoch = SLOTS_PER_EPOCH()
        epoch = slot // slots_per_epoch
        logger.info(f"Slot {slot} (epoch {epoch})")

        # Check if we need to sync blocks via req/resp (fallback for gossipsub issues)
        await self._sync_missing_blocks(slot)

        # NOTE: Don't update head_slot here - it should only be updated when we
        # actually have a block for a slot (either produced locally or received via P2P)
        # The head_slot tracks the slot of the actual chain head, not the current clock

        if slot % slots_per_epoch == 0:
            logger.info(f"New epoch: {epoch}")
            # Prune old attestations from pool
            self.attestation_pool.prune(slot)
            # Prune old sync committee messages
            self.sync_committee_pool.prune(slot)
            # Update fork_digest for publishing if we've crossed a fork boundary
            await self._update_fork_digest_for_epoch(epoch)
            # Recompute sync committee duties at epoch boundary
            await self._compute_sync_committee_duties()

        # Compute attester duties for current epoch if not cached
        await self._ensure_attester_duties(epoch)

        # Produce sync committee messages at slot start (Altair+)
        await self._produce_sync_committee_messages(slot)

        # Check if we should propose a block FIRST (uses payload_id from previous slot's forkchoice)
        await self._maybe_propose_block(slot)

        # Update forkchoice to keep EL in sync and prepare payload for NEXT slot
        await self._update_forkchoice_for_slot(slot)

        # NOTE: Attestations are produced at the 1/3 mark by _slot_ticker, not here

    async def _update_forkchoice_for_slot(self, slot: int) -> None:
        """Update forkchoice with EL and prepare payload for NEXT slot (slot + 1)."""
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

            # Finalized block hash - need the EXECUTION block hash, not beacon root
            # The finalized_checkpoint.root is a beacon block root, but EL needs execution block hash
            # For now, use zeros until finalization (which needs looking up the finalized block)
            # In production, we'd look up the finalized beacon block and get its execution payload hash
            finalized_hash = b"\x00" * 32
            finalized_epoch = int(self.state.finalized_checkpoint.epoch)

            forkchoice_state = ForkchoiceState(
                head_block_hash=head_block_hash,
                safe_block_hash=head_block_hash,
                finalized_block_hash=finalized_hash,
            )

            # Prepare payload attributes for NEXT slot (slot + 1)
            network_config = get_config()
            next_slot = slot + 1
            timestamp = self._genesis_time + next_slot * (network_config.slot_duration_ms // 1000)
            import time as time_mod
            current_time = int(time_mod.time())
            logger.info(
                f"Forkchoice prep for slot {next_slot}: head_hash={head_block_hash.hex()[:16]}, "
                f"finalized_epoch={finalized_epoch}, timestamp={timestamp}, state_slot={self.state.slot}"
            )

            # Get prev_randao from state for the next slot
            # At epoch boundaries, we need special handling:
            # - The current state is at the previous slot (in the old epoch)
            # - When process_slots advances to the new epoch, process_randao_mixes_reset
            #   copies the CURRENT epoch's mix to the NEW epoch's slot
            # - So for slots in a new epoch, we should use the current state's epoch mix
            randao_mixes = self.state.randao_mixes
            target_epoch = next_slot // SLOTS_PER_EPOCH()
            current_epoch = int(self.state.slot) // SLOTS_PER_EPOCH()

            if target_epoch > current_epoch:
                # Next slot is in a new epoch - use current epoch's mix
                # (which will be copied to the new epoch during epoch processing)
                prev_randao = bytes(randao_mixes[current_epoch % len(randao_mixes)])
                logger.debug(
                    f"Epoch boundary: using current_epoch={current_epoch} randao for target_epoch={target_epoch}"
                )
            else:
                # Same epoch - use the target epoch's mix directly
                prev_randao = bytes(randao_mixes[target_epoch % len(randao_mixes)])

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
                logger.info(f"Payload prepared for slot {next_slot}: id={response.payload_id.hex()}")
                self._current_payload_id = response.payload_id
                # Store the beacon root used in this forkchoiceUpdated for later use in newPayloadV5
                self._current_payload_beacon_root = self.head_root or b"\x00" * 32
            else:
                logger.warning(
                    f"No payload_id returned for slot {next_slot}: status={response.payload_status.status}, "
                    f"head_hash={head_block_hash.hex()[:16]}"
                )
                self._current_payload_id = None
                self._current_payload_beacon_root = None

        except Exception as e:
            logger.error(f"Failed to update forkchoice for slot {slot}: {e}")
            import traceback
            traceback.print_exc()

    async def _maybe_propose_block(self, slot: int) -> None:
        """Check if we're the proposer and produce a block if so."""
        if not self.validator_client or not self.validator_client.keys:
            return

        if not self.state:
            return

        # Use a temp copy for proposer duty calculation to avoid mutating self.state
        # This is necessary because state_transition needs to advance state from pre-slot
        temp_state = self.state
        if slot > int(temp_state.slot):
            temp_state = process_slots(
                temp_state.__class__.decode_bytes(bytes(temp_state.encode_bytes())),
                slot
            )

        proposer_key = self.validator_client.is_our_proposer_slot(temp_state, slot)
        if not proposer_key:
            return

        logger.info(
            f"We are proposer for slot {slot}! payload_id={self._current_payload_id.hex() if self._current_payload_id else 'None'}"
        )

        try:
            await self._produce_and_broadcast_block(slot, proposer_key)
        except Exception as e:
            logger.error(f"Failed to produce block for slot {slot}: {e}")

    async def _ensure_attester_duties(self, epoch: int) -> None:
        """Ensure we have attester duties computed for the given epoch."""
        if not self.validator_client or not self.validator_client.keys:
            return
        if not self.state:
            return
        if epoch in self._attester_duties:
            return

        duties = self.validator_client.get_attester_duties(self.state, epoch)
        self._attester_duties[epoch] = duties
        if duties:
            logger.info(f"Computed {len(duties)} attester duties for epoch {epoch}")

        # Clean up old epochs
        old_epochs = [e for e in self._attester_duties if e < epoch - 1]
        for e in old_epochs:
            del self._attester_duties[e]

    async def _produce_attestations(self, slot: int) -> None:
        """Produce attestations for validators with duties at this slot.

        Called at 1/3 into the slot per Ethereum spec to allow time for
        block proposals to arrive before attesting.
        """
        if not self.validator_client or not self.validator_client.keys:
            return
        if not self.state or not self.head_root:
            return

        # Log timing for debugging attestation rate issues
        network_config = get_config()
        slot_duration = network_config.slot_duration_ms / 1000.0
        slots_per_epoch = SLOTS_PER_EPOCH()
        epoch = slot // slots_per_epoch
        slot_start = self._genesis_time + slot * slot_duration
        time_into_slot = time.time() - slot_start
        expected_offset = network_config.get_attestation_due_offset(epoch)

        duties = self._attester_duties.get(epoch, [])
        slot_duties = [d for d in duties if d.slot == slot]

        if not slot_duties:
            return

        # Use temp state advanced to current slot for attestation production
        temp_state = self.state
        if slot > int(temp_state.slot):
            temp_state = process_slots(
                temp_state.__class__.decode_bytes(bytes(temp_state.encode_bytes())),
                slot
            )

        validator_indices = [d.validator_index for d in slot_duties]
        logger.info(
            f"[ATTESTER] slot={slot} count={len(slot_duties)} time={time_into_slot:.2f}s validators={validator_indices}"
        )

        produced_count = 0
        broadcast_count = 0
        for duty in slot_duties:
            try:
                attestation = self.validator_client.produce_attestation(
                    temp_state, duty, self.head_root
                )
                if attestation:
                    self.attestation_pool.add(attestation)
                    produced_count += 1

                    # Broadcast via P2P as aggregate
                    if self.beacon_gossip:
                        try:
                            await self._broadcast_attestation_as_aggregate(
                                attestation, duty.validator_index, duty.pubkey
                            )
                            broadcast_count += 1
                        except Exception as e:
                            logger.error(f"Failed to broadcast attestation: {e}")
            except Exception as e:
                logger.error(f"Failed to produce attestation for duty {duty}: {e}")

        logger.info(f"Attestations: slot={slot}, produced={produced_count}, broadcast={broadcast_count}")

    def _is_electra_fork(self) -> bool:
        """Check if current state is at Electra or later fork."""
        if not self.state:
            return False
        return hasattr(self.state, "pending_deposits")

    async def _compute_sync_committee_duties(self) -> None:
        """Compute which of our validators are in the current sync committee.

        Called at epoch boundaries to refresh sync committee membership.
        """
        if not self.state or not self.validator_client:
            return

        if not hasattr(self.state, "current_sync_committee"):
            return

        self._sync_committee_duties.clear()

        committee_pubkeys = list(self.state.current_sync_committee.pubkeys)

        for position, pubkey in enumerate(committee_pubkeys):
            pubkey_bytes = bytes(pubkey)
            if self.validator_client.has_key(pubkey_bytes):
                validator_index = self.validator_client.get_validator_index(pubkey_bytes)
                if validator_index is not None:
                    if validator_index not in self._sync_committee_duties:
                        self._sync_committee_duties[validator_index] = []
                    self._sync_committee_duties[validator_index].append(position)

        if self._sync_committee_duties:
            logger.info(
                f"Sync committee duties: {len(self._sync_committee_duties)} validators, "
                f"positions: {sum(len(p) for p in self._sync_committee_duties.values())}"
            )

    async def _produce_sync_committee_messages(self, slot: int) -> None:
        """Produce sync committee messages for validators in the current sync committee.

        Called at the start of each slot. Sync committee members sign the
        previous slot's block root.

        Args:
            slot: Current slot
        """
        if not self.state or not self.validator_client:
            return

        if not hasattr(self.state, "current_sync_committee"):
            return

        if not self._sync_committee_duties:
            await self._compute_sync_committee_duties()

        if not self._sync_committee_duties:
            return

        state_for_signing = self.state
        if int(self.state.slot) < slot:
            state_for_signing = process_slots(
                self.state.__class__.decode_bytes(bytes(self.state.encode_bytes())),
                slot
            )

        for validator_index, positions in self._sync_committee_duties.items():
            for pubkey, key in self.validator_client.keys.items():
                if key.validator_index == validator_index:
                    message = await self.validator_client.produce_sync_committee_message(
                        state_for_signing, slot, key
                    )
                    if message:
                        for position in positions:
                            self.sync_committee_pool.add(message, position)
                    break

        logger.debug(
            f"Produced sync committee messages: slot={slot}, "
            f"pool_size={self.sync_committee_pool.size}"
        )

        await self._broadcast_sync_committee_contributions(slot, state_for_signing)

    async def _broadcast_sync_committee_contributions(self, slot: int, state) -> None:
        """Broadcast sync committee contributions for each subcommittee."""
        if not self.beacon_gossip or not self.validator_client:
            return

        from .spec.constants import SYNC_COMMITTEE_SIZE, SYNC_COMMITTEE_SUBNET_COUNT, DOMAIN_SYNC_COMMITTEE_SELECTION_PROOF, DOMAIN_CONTRIBUTION_AND_PROOF
        from .spec.types.altair import (
            SyncCommitteeContribution, ContributionAndProof, SignedContributionAndProof,
            SyncAggregatorSelectionData,
        )
        from .spec.types.base import Bitvector, BLSSignature, uint64
        from .spec.state_transition.helpers.domain import get_domain, compute_signing_root
        from .crypto import sign as bls_sign

        sync_committee_size = SYNC_COMMITTEE_SIZE()
        subcommittee_size = sync_committee_size // SYNC_COMMITTEE_SUBNET_COUNT
        messages = self.sync_committee_pool._messages.get(slot, {})

        if not messages:
            return

        for subcommittee_index in range(SYNC_COMMITTEE_SUBNET_COUNT):
            base_position = subcommittee_index * subcommittee_size

            subcommittee_messages = []
            aggregator_key = None

            for position, pooled in messages.items():
                if base_position <= position < base_position + subcommittee_size:
                    subcommittee_messages.append((position - base_position, pooled))
                    if aggregator_key is None:
                        for pubkey, key in self.validator_client.keys.items():
                            if key.validator_index == pooled.message.validator_index:
                                aggregator_key = key
                                break

            if not subcommittee_messages or not aggregator_key:
                continue

            agg_bits = Bitvector[subcommittee_size]()
            signatures = []
            beacon_block_root = None

            for bit_pos, pooled in subcommittee_messages:
                agg_bits[bit_pos] = True
                signatures.append(bytes(pooled.message.signature))
                if beacon_block_root is None:
                    beacon_block_root = pooled.message.beacon_block_root

            if not signatures:
                continue

            try:
                from .crypto import aggregate_signatures
                if len(signatures) == 1:
                    aggregated_sig = signatures[0]
                else:
                    aggregated_sig = aggregate_signatures(signatures)
            except Exception as e:
                logger.warning(f"Failed to aggregate subcommittee {subcommittee_index}: {e}")
                continue

            contribution = SyncCommitteeContribution(
                slot=slot,
                beacon_block_root=beacon_block_root,
                subcommittee_index=uint64(subcommittee_index),
                aggregation_bits=agg_bits,
                signature=BLSSignature(aggregated_sig),
            )

            epoch = slot // SLOTS_PER_EPOCH()
            selection_data = SyncAggregatorSelectionData(
                slot=slot,
                subcommittee_index=uint64(subcommittee_index),
            )
            selection_domain = get_domain(state, DOMAIN_SYNC_COMMITTEE_SELECTION_PROOF, epoch)
            selection_root = compute_signing_root(selection_data, selection_domain)
            selection_proof = bls_sign(aggregator_key.privkey, selection_root)

            contribution_and_proof = ContributionAndProof(
                aggregator_index=aggregator_key.validator_index,
                contribution=contribution,
                selection_proof=BLSSignature(selection_proof),
            )

            contrib_domain = get_domain(state, DOMAIN_CONTRIBUTION_AND_PROOF, epoch)
            contrib_root = compute_signing_root(contribution_and_proof, contrib_domain)
            contrib_signature = bls_sign(aggregator_key.privkey, contrib_root)

            signed_contribution = SignedContributionAndProof(
                message=contribution_and_proof,
                signature=BLSSignature(contrib_signature),
            )

            try:
                await self.beacon_gossip.publish_sync_committee_contribution(
                    bytes(signed_contribution.encode_bytes())
                )
                logger.debug(
                    f"Broadcast sync committee contribution: slot={slot}, "
                    f"subcommittee={subcommittee_index}, bits={len(signatures)}"
                )
            except Exception as e:
                logger.warning(f"Failed to broadcast contribution: {e}")

    async def _broadcast_attestation_as_aggregate(
        self, attestation, validator_index: int, pubkey: bytes
    ) -> None:
        """Wrap attestation in SignedAggregateAndProof and broadcast."""
        from .spec.types.electra import ElectraAggregateAndProof, SignedElectraAggregateAndProof
        from .spec.types.phase0 import AggregateAndProof, SignedAggregateAndProof
        from .spec.types.base import BLSSignature
        from .spec.constants import DOMAIN_SELECTION_PROOF, DOMAIN_AGGREGATE_AND_PROOF
        from .spec.state_transition.helpers.domain import get_domain, compute_signing_root
        from .spec.types import Slot

        key = self.validator_client.get_key(pubkey)
        if not key:
            return

        slot = int(attestation.data.slot)
        epoch = slot // SLOTS_PER_EPOCH()

        # Create selection proof (sign the slot)
        domain = get_domain(self.state, DOMAIN_SELECTION_PROOF, epoch)
        signing_root = compute_signing_root(Slot(slot), domain)
        selection_proof = sign(key.privkey, signing_root)

        is_electra = self._is_electra_fork()

        if is_electra:
            aggregate_and_proof = ElectraAggregateAndProof(
                aggregator_index=validator_index,
                aggregate=attestation,
                selection_proof=BLSSignature(selection_proof),
            )
        else:
            aggregate_and_proof = AggregateAndProof(
                aggregator_index=validator_index,
                aggregate=attestation,
                selection_proof=BLSSignature(selection_proof),
            )

        # Sign the aggregate and proof
        domain = get_domain(self.state, DOMAIN_AGGREGATE_AND_PROOF, epoch)
        signing_root = compute_signing_root(aggregate_and_proof, domain)
        signature = sign(key.privkey, signing_root)

        if is_electra:
            signed_aggregate = SignedElectraAggregateAndProof(
                message=aggregate_and_proof,
                signature=BLSSignature(signature),
            )
        else:
            signed_aggregate = SignedAggregateAndProof(
                message=aggregate_and_proof,
                signature=BLSSignature(signature),
            )

        # Broadcast
        ssz_bytes = signed_aggregate.encode_bytes()
        await self.beacon_gossip.publish_aggregate(ssz_bytes)
        logger.debug(f"Broadcast attestation as aggregate for slot {slot}, electra={is_electra}")

    async def _broadcast_execution_payload_envelope(
        self,
        slot: int,
        beacon_block_root: bytes,
        execution_payload_dict: dict,
        execution_requests: list,
        blobs_bundle: dict | None,
    ) -> None:
        """Build and broadcast the SignedExecutionPayloadEnvelope for GLOAS self-build.

        In GLOAS ePBS, the execution payload is delivered separately from the beacon block.
        For self-build mode, the proposer acts as the builder and reveals the payload.

        Args:
            slot: Slot number
            beacon_block_root: Root of the beacon block
            execution_payload_dict: Execution payload from EL
            execution_requests: List of execution requests
            blobs_bundle: Optional blobs bundle from EL
        """
        from .spec.types.gloas import (
            ExecutionPayloadEnvelope,
            SignedExecutionPayloadEnvelope,
        )
        from .spec.types import BLSSignature, Root, Slot, KZGCommitment
        from .spec.types.base import List
        from .spec.constants import BUILDER_INDEX_SELF_BUILD
        # MAX_BLOB_COMMITMENTS_PER_BLOCK is 4096
        MAX_BLOB_COMMITMENTS = 4096

        # Build the execution payload using the same logic as block builder
        execution_payload = self.block_builder._build_execution_payload(
            execution_payload_dict, "gloas"
        )

        # Build execution requests
        from .spec.types.electra import ExecutionRequests
        exec_requests_obj = ExecutionRequests()
        # Parse execution_requests if provided as hex strings
        if execution_requests:
            from .spec.types.electra import DepositRequest, WithdrawalRequest, ConsolidationRequest
            for req_hex in execution_requests:
                if isinstance(req_hex, str):
                    req_bytes = bytes.fromhex(req_hex.replace("0x", ""))
                    if req_bytes and len(req_bytes) > 0:
                        req_type = req_bytes[0]
                        req_data = req_bytes[1:]
                        if req_type == 0x00:
                            exec_requests_obj.deposits.append(DepositRequest.decode_bytes(req_data))
                        elif req_type == 0x01:
                            exec_requests_obj.withdrawals.append(WithdrawalRequest.decode_bytes(req_data))
                        elif req_type == 0x02:
                            exec_requests_obj.consolidations.append(ConsolidationRequest.decode_bytes(req_data))

        # Get KZG commitments from blobs bundle
        kzg_commitments = List[KZGCommitment, MAX_BLOB_COMMITMENTS]()
        if blobs_bundle:
            commitments = blobs_bundle.get("commitments", [])
            for commitment_hex in commitments:
                commitment_bytes = bytes.fromhex(commitment_hex.replace("0x", ""))
                kzg_commitments.append(KZGCommitment(commitment_bytes))

        # Compute state root (use the current state's root)
        state_root = hash_tree_root(self.state)

        # Build the envelope
        envelope = ExecutionPayloadEnvelope(
            payload=execution_payload,
            execution_requests=exec_requests_obj,
            builder_index=BUILDER_INDEX_SELF_BUILD,
            beacon_block_root=Root(beacon_block_root),
            slot=Slot(slot),
            state_root=Root(state_root),
        )

        # For self-build, use G2 point-at-infinity signature (no actual signing needed)
        g2_point_at_infinity = b"\xc0" + b"\x00" * 95

        signed_envelope = SignedExecutionPayloadEnvelope(
            message=envelope,
            signature=BLSSignature(g2_point_at_infinity),
        )

        # Broadcast
        ssz_bytes = signed_envelope.encode_bytes()
        await self.beacon_gossip.publish_execution_payload(ssz_bytes)
        logger.info(
            f"Broadcast execution payload envelope for slot {slot}: "
            f"block_hash={bytes(execution_payload.block_hash).hex()[:16]}, "
            f"commitments={len(kzg_commitments)}"
        )

    async def _produce_and_broadcast_block(self, slot: int, proposer_key) -> None:
        """Produce and broadcast a block for the given slot."""
        if not self.engine:
            logger.warning("Cannot produce block: no engine")
            return

        if not self.block_builder:
            logger.warning("Cannot produce block: no block builder")
            return

        try:
            network_config = get_config()
            timestamp = self._genesis_time + slot * (network_config.slot_duration_ms // 1000)

            # Always request a fresh payload for this slot to ensure correct timestamp
            # This is safer than relying on a previously prepared payload which might be stale
            logger.info(f"Requesting fresh payload for slot {slot}")
            await self._request_payload_for_slot(slot)
            if not self._current_payload_id:
                logger.warning("Cannot produce block: failed to get payload_id")
                return

            payload_response = await self.engine.get_payload(self._current_payload_id, timestamp=timestamp)
            execution_payload_dict = payload_response.execution_payload
            block_hash = execution_payload_dict.get("blockHash", "unknown")
            block_number = execution_payload_dict.get("blockNumber", "unknown")
            exec_requests = payload_response.execution_requests
            logger.info(
                f"Got execution payload: block_hash={block_hash[:18]}, "
                f"block_number={block_number}, value={payload_response.block_value}, "
                f"execution_requests_count={len(exec_requests) if exec_requests else 0}"
            )

            # Extract execution_requests from payload response (Electra/Fulu)
            el_execution_requests = payload_response.execution_requests or []

            signed_block = await self.block_builder.build_block(
                slot, proposer_key, execution_payload_dict
            )
            if signed_block is None:
                logger.error("Failed to build block")
                return

            block = signed_block.message
            block_root = hash_tree_root(block)

            # Check if this is a GLOAS block (ePBS) - has signed_execution_payload_bid instead of execution_payload
            is_gloas = hasattr(block.body, 'signed_execution_payload_bid')

            if is_gloas:
                # GLOAS (ePBS): Extract block hash from bid
                bid = block.body.signed_execution_payload_bid.message
                new_block_hash = bytes(bid.block_hash)
                payload_timestamp = int(execution_payload_dict.get("timestamp", "0x0"), 16)
                logger.info(
                    f"GLOAS block with self-build bid: block_hash={new_block_hash.hex()[:16]}, "
                    f"builder_index={bid.builder_index}, timestamp={payload_timestamp}"
                )

                # Extract versioned hashes from blobs bundle (EIP-4844)
                versioned_hashes = []
                if payload_response.blobs_bundle:
                    blobs_bundle = payload_response.blobs_bundle
                    commitments = blobs_bundle.get("commitments", [])
                    if commitments:
                        from hashlib import sha256
                        for commitment in commitments:
                            commitment_bytes = bytes.fromhex(commitment.replace("0x", ""))
                            versioned_hash = b'\x01' + sha256(commitment_bytes).digest()[1:]
                            versioned_hashes.append(versioned_hash)
                        logger.info(f"Extracted {len(versioned_hashes)} versioned hashes from blobs bundle")

                # Use the beacon root that was stored when forkchoiceUpdated was called.
                # This MUST match the parentBeaconBlockRoot used in that call, as Geth uses it
                # to compute the block hash. self.head_root may have changed since then.
                parent_beacon_root = self._current_payload_beacon_root or bytes(block.parent_root)

                # For GLOAS self-build, validate with EL using raw payload dict
                # (bypasses SSZ round-trip to avoid blockhash mismatch)
                logger.info(
                    f"GLOAS newPayloadV5: stored_beacon_root={self._current_payload_beacon_root.hex()[:16] if self._current_payload_beacon_root else 'None'}, "
                    f"head_root={self.head_root.hex()[:16] if self.head_root else 'None'}, "
                    f"block.parent_root={bytes(block.parent_root).hex()[:16]}, "
                    f"using={parent_beacon_root.hex()[:16]}"
                )
                status = await self.engine.new_payload_v5_raw(
                    execution_payload_dict,
                    versioned_hashes,
                    parent_beacon_root,
                    el_execution_requests,
                )

                if status.status != PayloadStatusEnum.VALID:
                    logger.error(f"GLOAS execution payload invalid: {status.status}")
                    if status.status == PayloadStatusEnum.SYNCING:
                        logger.info("EL is syncing, block may be valid later")
                    else:
                        return

            else:
                # Non-GLOAS: execution_payload is directly in block body
                execution_payload = block.body.execution_payload

                # Extract versioned hashes from blobs bundle (EIP-4844)
                versioned_hashes = []
                if payload_response.blobs_bundle:
                    blobs_bundle = payload_response.blobs_bundle
                    commitments = blobs_bundle.get("commitments", [])
                    if commitments:
                        from hashlib import sha256
                        for commitment in commitments:
                            commitment_bytes = bytes.fromhex(commitment.replace("0x", ""))
                            versioned_hash = b'\x01' + sha256(commitment_bytes).digest()[1:]
                            versioned_hashes.append(versioned_hash)
                        logger.info(f"Extracted {len(versioned_hashes)} versioned hashes from blobs bundle")

                # Use the beacon root that was stored when forkchoiceUpdated was called.
                # This MUST match the parentBeaconBlockRoot used in that call.
                parent_beacon_root = self._current_payload_beacon_root or bytes(block.parent_root)

                # Log the SSZ payload's blockHash to verify it matches the original
                ssz_block_hash = bytes(execution_payload.block_hash).hex()
                ssz_timestamp = int(execution_payload.timestamp)
                original_timestamp = int(execution_payload_dict.get("timestamp", "0x0"), 16)
                logger.info(
                    f"Payload round-trip check: "
                    f"original_blockHash={block_hash}, "
                    f"ssz_blockHash=0x{ssz_block_hash}, "
                    f"original_timestamp={original_timestamp}, "
                    f"ssz_timestamp={ssz_timestamp}, "
                    f"original_stateRoot={execution_payload_dict.get('stateRoot')}, "
                    f"ssz_stateRoot=0x{bytes(execution_payload.state_root).hex()}"
                )

                # Detailed payload comparison for debugging blockhash mismatch
                reconstructed_dict = self.engine._payload_to_dict(execution_payload)
                diff_fields = []
                for key in set(execution_payload_dict.keys()) | set(reconstructed_dict.keys()):
                    orig_val = execution_payload_dict.get(key)
                    recon_val = reconstructed_dict.get(key)
                    if orig_val != recon_val:
                        if key in ('transactions', 'withdrawals'):
                            orig_len = len(orig_val) if orig_val else 0
                            recon_len = len(recon_val) if recon_val else 0
                            if orig_len != recon_len:
                                diff_fields.append(f"{key}: len {orig_len} vs {recon_len}")
                            else:
                                for i, (o, r) in enumerate(zip(orig_val or [], recon_val or [])):
                                    if o != r:
                                        diff_fields.append(f"{key}[{i}]: {str(o)[:50]} vs {str(r)[:50]}")
                                        break
                        else:
                            diff_fields.append(f"{key}: {orig_val} vs {recon_val}")
                if diff_fields:
                    logger.warning(f"Payload dict differences: {diff_fields}")
                else:
                    logger.info("Payload dict round-trip: no differences detected")

                # Log execution requests being sent
                if el_execution_requests:
                    logger.info(f"Sending {len(el_execution_requests)} execution_requests to newPayload")
                    for i, req in enumerate(el_execution_requests):
                        if isinstance(req, str):
                            logger.debug(f"  execution_request[{i}]: {req[:66]}...")
                        else:
                            logger.debug(f"  execution_request[{i}]: type={type(req)}")

                status = await self.engine.new_payload(
                    execution_payload,
                    versioned_hashes,
                    parent_beacon_root,
                    el_execution_requests,
                    timestamp=int(execution_payload.timestamp),
                )

                if status.status != PayloadStatusEnum.VALID:
                    logger.error(f"Execution payload invalid: {status.status}")
                    if status.status == PayloadStatusEnum.SYNCING:
                        logger.info("EL is syncing, block may be valid later")
                    else:
                        return

                new_block_hash = bytes(execution_payload.block_hash)
                payload_timestamp = int(execution_payload.timestamp)

            # Apply the produced block to state FIRST so subsequent forkchoice calls
            # use the updated state
            self.store.save_block(block_root, signed_block)

            # Save blob sidecars if present
            if payload_response.blobs_bundle and versioned_hashes:
                kzg_commitments = []
                if hasattr(block.body, "blob_kzg_commitments"):
                    kzg_commitments = [bytes(c) for c in block.body.blob_kzg_commitments]
                self.store.save_blobs(block_root, slot, payload_response.blobs_bundle, kzg_commitments, signed_block)
                logger.info(f"Saved {len(versioned_hashes)} blob sidecars for block at slot {slot}")

            # Apply state transition FIRST (before updating head, to avoid race with SSE events)
            await self._apply_block_to_state(block, block_root, signed_block)

            # Fill in state.latest_block_header.state_root with the actual post-state root.
            # This is normally filled in during process_slot for the NEXT slot, but we need it
            # now so that hash_tree_root(state.latest_block_header) == block_root for correct
            # parentBeaconBlockRoot in forkchoiceUpdated/newPayload.
            if bytes(self.state.latest_block_header.state_root) == b"\x00" * 32:
                self.state.latest_block_header.state_root = bytes(block.state_root)
                logger.debug(f"Filled in latest_block_header.state_root: {bytes(block.state_root).hex()[:16]}")

            # For GLOAS self-build: update state.latest_block_hash after state transition
            # This is needed because process_execution_payload_bid only stores the bid,
            # but doesn't update latest_block_hash (which happens in payload reveal normally)
            if is_gloas and hasattr(self.state, 'latest_block_hash'):
                from .spec.types import Hash32
                self.state.latest_block_hash = Hash32(new_block_hash)
                logger.info(f"GLOAS: Updated state.latest_block_hash to {new_block_hash.hex()[:16]}")

            # Save state for epoch queries (Dora needs historical states by state_root)
            # This MUST happen before updating head_slot/head_root to avoid race condition
            # where SSE event is emitted before state is saved
            state_root = hash_tree_root(self.state)
            self.store.save_state(state_root, self.state)
            self.store.save_state(block_root, self.state)  # Also by block_root for flexibility
            # Also save by the block's state_root field, as that's what Dora looks for
            # (may differ from computed hash if latest_block_header.state_root was filled in)
            block_state_root = bytes(block.state_root)
            if block_state_root != state_root:
                self.store.save_state(block_state_root, self.state)

            # NOW update head_slot/head_root (SSE event loop will see this change)
            self.head_slot = slot
            self.head_root = block_root
            self.store.set_head(block_root)

            # Update metrics
            from .spec import constants
            epoch = slot // constants.SLOTS_PER_EPOCH()
            metrics.update_head(slot, epoch)
            metrics.record_block_proposed(success=True)

            # Now update forkchoice with the new block
            forkchoice_state = ForkchoiceState(
                head_block_hash=new_block_hash,
                safe_block_hash=new_block_hash,
                finalized_block_hash=b"\x00" * 32,  # Use zeros for now
            )

            fc_response = await self.engine.forkchoice_updated(forkchoice_state, timestamp=payload_timestamp)
            logger.info(
                f"Block forkchoice updated: status={fc_response.payload_status.status}, "
                f"new_head={new_block_hash.hex()[:16]}"
            )

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

                # For GLOAS self-build: also broadcast the execution payload envelope
                if is_gloas:
                    try:
                        await self._broadcast_execution_payload_envelope(
                            slot, block_root, execution_payload_dict, el_execution_requests,
                            payload_response.blobs_bundle
                        )
                    except Exception as e:
                        logger.error(f"Failed to broadcast execution payload envelope: {e}")
                        import traceback
                        traceback.print_exc()

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

            metrics.record_block_received()

            if self.store.get_block(block_root):
                return

            self.store.save_block(block_root, signed_block)

            if int(block.slot) > self.head_slot:
                # Apply state transition FIRST (before updating head, to avoid race with SSE events)
                await self._apply_block_to_state(block, block_root, signed_block)

                # Fill in state.latest_block_header.state_root with the actual post-state root
                if bytes(self.state.latest_block_header.state_root) == b"\x00" * 32:
                    self.state.latest_block_header.state_root = bytes(block.state_root)

                # Save state for epoch queries (Dora needs historical states by state_root)
                # This MUST happen before updating head_slot/head_root to avoid race condition
                # where SSE event is emitted before state is saved
                state_root = hash_tree_root(self.state)
                self.store.save_state(state_root, self.state)
                self.store.save_state(block_root, self.state)  # Also by block_root for flexibility
                # Also save by the block's state_root field, as that's what Dora looks for
                block_state_root = bytes(block.state_root)
                if block_state_root != state_root:
                    self.store.save_state(block_state_root, self.state)

                # NOW update head_slot/head_root (SSE event loop will see this change)
                self.head_slot = int(block.slot)
                self.head_root = block_root
                self.store.set_head(block_root)

                # Update metrics
                from .spec import constants
                slot = int(block.slot)
                epoch = slot // constants.SLOTS_PER_EPOCH()
                metrics.update_head(slot, epoch)

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

    async def _sync_missing_blocks(self, current_slot: int) -> None:
        """Sync missing blocks via req/resp if we're behind.

        This is a fallback for when gossipsub doesn't deliver blocks reliably.
        Since py-libp2p gossipsub has issues with rust-libp2p, we aggressively
        sync via req/resp to stay in sync with Lighthouse.
        """
        if not self.beacon_gossip or not self.state:
            return

        # Check if our state is behind the current slot
        state_slot = int(self.state.slot)
        slots_behind = current_slot - state_slot

        # Sync if we're even 1 slot behind (aggressive sync for gossipsub workaround)
        if slots_behind < 1:
            return

        # IMPORTANT: When we're on a different fork (e.g., our slot 1 vs peer's slot 1),
        # we need to request from slot 1, not state_slot+1, because the peer's chain
        # may have different blocks. This allows us to get the peer's complete chain.
        # Request from slot 1 if we might be on a different fork.
        if state_slot <= 1 and slots_behind > 1:
            # We're at genesis or slot 1, peer is ahead - request from slot 1
            start_slot = 1
            count = min(current_slot, 32)  # Get up to 32 blocks to catch up
        else:
            start_slot = state_slot + 1
            count = min(slots_behind, 16)

        logger.info(f"State is {slots_behind} slots behind (state_slot={state_slot}, current={current_slot}). "
                    f"Requesting blocks {start_slot} to {start_slot + count - 1} via req/resp")

        try:
            blocks = await self.beacon_gossip.request_blocks_by_range(start_slot, count)
            if not blocks:
                logger.warning(f"No blocks received from req/resp request")
                return

            logger.info(f"Received {len(blocks)} blocks via req/resp")

            # Process each block
            for block_ssz in blocks:
                try:
                    await self._on_p2p_block(block_ssz, "req/resp")
                except Exception as e:
                    logger.warning(f"Error processing synced block: {e}")

        except Exception as e:
            logger.warning(f"Block sync via req/resp failed: {e}")

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

            metrics.record_block_received()

            if self.store.get_block(block_root):
                logger.debug(f"Block already known: {block_root.hex()[:16]}")
                return

            # Check if this block builds on our current head or a known parent
            logger.debug(
                f"P2P: Parent check: parent_root={parent_root.hex()[:16]}, "
                f"head_root={self.head_root.hex()[:16] if self.head_root else 'None'}, "
                f"match={parent_root == self.head_root if self.head_root else 'N/A'}"
            )

            # Get genesis root for comparison
            genesis_root = self._genesis_block_root if hasattr(self, '_genesis_block_root') else None

            # Check if this is a potential reorg opportunity
            # If block builds on genesis (slot 1) and we're stuck at a low slot, consider reorg
            if self.head_root and parent_root != self.head_root:
                # Check if parent is genesis - this could be an alternate slot 1 block
                is_slot_1_block = int(block.slot) == 1 and parent_root == genesis_root

                if is_slot_1_block and self.head_slot <= 1:
                    # We both have slot 1, but different versions - this is a fork at slot 1
                    # Reset to genesis and adopt this new chain if we're behind
                    logger.warning(
                        f"P2P: Detected fork at slot 1. Resetting to genesis to adopt peer's chain. "
                        f"Our slot 1={self.head_root.hex()[:16]}, peer's slot 1={block_root.hex()[:16]}"
                    )
                    # Reset state to genesis
                    self.state = self._genesis_state.__class__.decode_bytes(
                        bytes(self._genesis_state.encode_bytes())
                    )
                    self.head_slot = 0
                    self.head_root = genesis_root
                    # Now we can process this block
                elif not self.store.get_block(parent_root):
                    # Check if parent is genesis
                    if parent_root == genesis_root:
                        logger.info(f"P2P: Block parent is genesis, will process")
                    else:
                        logger.warning(
                            f"P2P: Ignoring block slot={block.slot} - parent {parent_root.hex()[:16]} "
                            f"not found (our head={self.head_root.hex()[:16] if self.head_root else 'None'})"
                        )
                        self.store.save_block(block_root, signed_block)
                        return
                else:
                    logger.warning(
                        f"P2P: Block slot={block.slot} builds on different chain "
                        f"(parent={parent_root.hex()[:16]}, our head={self.head_root.hex()[:16]}). "
                        f"Would need state regeneration to apply."
                    )
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

                    # Fill in state.latest_block_header.state_root with the actual post-state root
                    if bytes(self.state.latest_block_header.state_root) == b"\x00" * 32:
                        self.state.latest_block_header.state_root = bytes(block.state_root)

                    # Save state for epoch queries (Dora needs historical states by state_root)
                    # This MUST happen before updating head_slot/head_root to avoid race condition
                    # where SSE event is emitted before state is saved
                    state_root = hash_tree_root(self.state)
                    self.store.save_state(state_root, self.state)
                    self.store.save_state(block_root, self.state)  # Also by block_root for flexibility
                    # Also save by the block's state_root field, as that's what Dora looks for
                    block_state_root = bytes(block.state_root)
                    if block_state_root != state_root:
                        self.store.save_state(block_state_root, self.state)

                    # NOW update head_slot/head_root (SSE event loop will see this change)
                    self.head_slot = int(block.slot)
                    self.head_root = block_root
                    self.store.set_head(block_root)

                    # Update metrics
                    from .spec import constants
                    slot = int(block.slot)
                    epoch = slot // constants.SLOTS_PER_EPOCH()
                    metrics.update_head(slot, epoch)

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
            from .spec.types.electra import SignedElectraAggregateAndProof
            from .spec.types.phase0 import SignedAggregateAndProof

            # Try Electra format first, then Phase0
            signed_aggregate = None
            attestation = None

            try:
                signed_aggregate = SignedElectraAggregateAndProof.decode_bytes(data)
                attestation = signed_aggregate.message.aggregate
            except Exception:
                try:
                    signed_aggregate = SignedAggregateAndProof.decode_bytes(data)
                    attestation = signed_aggregate.message.aggregate
                except Exception as e:
                    logger.error(f"Failed to decode aggregate as any known format: {e}")
                    return

            slot = int(attestation.data.slot)

            # Add to attestation pool
            self.attestation_pool.add(attestation)

            logger.info(
                f"P2P: Received aggregate slot={slot}, "
                f"aggregator={signed_aggregate.message.aggregator_index}, "
                f"from={from_peer[:16]}"
            )
        except Exception as e:
            logger.error(f"Error processing P2P aggregate: {e}")

    async def _on_p2p_sync_committee_contribution(self, data: bytes, from_peer: str) -> None:
        """Handle a sync committee contribution received via libp2p gossipsub."""
        try:
            from .spec.types.altair import SignedContributionAndProof

            signed_contribution = SignedContributionAndProof.decode_bytes(data)
            contribution = signed_contribution.message.contribution
            slot = int(contribution.slot)
            subcommittee_index = int(contribution.subcommittee_index)

            self.sync_committee_pool.add_contribution(contribution)

            participant_count = sum(1 for b in contribution.aggregation_bits if b)
            logger.info(
                f"P2P: Received sync committee contribution slot={slot}, "
                f"subcommittee={subcommittee_index}, participants={participant_count}, "
                f"from={from_peer[:16]}"
            )
        except Exception as e:
            logger.error(f"Error processing P2P sync committee contribution: {e}")

    async def _on_p2p_blob_sidecar(self, data: bytes, from_peer: str) -> None:
        """Handle a blob sidecar received via libp2p gossipsub."""
        try:
            from .spec.types.deneb import BlobSidecar

            sidecar = BlobSidecar.decode_bytes(data)
            slot = int(sidecar.signed_block_header.message.slot)
            index = int(sidecar.index)
            block_root = bytes(sidecar.signed_block_header.message.parent_root)

            # Compute the block root from the header
            block_header = sidecar.signed_block_header.message
            block_root = hash_tree_root(block_header)

            logger.info(
                f"P2P: Received blob sidecar slot={slot}, index={index}, "
                f"block_root={block_root.hex()[:16]}, from={from_peer[:16]}"
            )

            # Store the blob sidecar
            self._store_received_blob_sidecar(block_root, slot, sidecar)

        except Exception as e:
            logger.error(f"Error processing P2P blob sidecar: {e}")

    def _store_received_blob_sidecar(self, block_root: bytes, slot: int, sidecar) -> None:
        """Store a received blob sidecar in the store."""
        # Get existing blobs for this block (if any)
        existing_blobs = self.store.get_blobs(block_root)

        # Convert sidecar to JSON format for storage
        index = int(sidecar.index)

        sidecar_json = {
            "index": str(index),
            "blob": "0x" + bytes(sidecar.blob).hex(),
            "kzg_commitment": "0x" + bytes(sidecar.kzg_commitment).hex(),
            "kzg_proof": "0x" + bytes(sidecar.kzg_proof).hex(),
            "signed_block_header": {
                "message": {
                    "slot": str(sidecar.signed_block_header.message.slot),
                    "proposer_index": str(sidecar.signed_block_header.message.proposer_index),
                    "parent_root": "0x" + bytes(sidecar.signed_block_header.message.parent_root).hex(),
                    "state_root": "0x" + bytes(sidecar.signed_block_header.message.state_root).hex(),
                    "body_root": "0x" + bytes(sidecar.signed_block_header.message.body_root).hex(),
                },
                "signature": "0x" + bytes(sidecar.signed_block_header.signature).hex(),
            },
            "kzg_commitment_inclusion_proof": [
                "0x" + bytes(p).hex() for p in sidecar.kzg_commitment_inclusion_proof
            ],
        }

        # Check if we already have this blob index
        existing_indices = {int(b["index"]) for b in existing_blobs}
        if index in existing_indices:
            return  # Already have this blob

        # Add to existing blobs
        updated_blobs = existing_blobs + [sidecar_json]
        updated_blobs.sort(key=lambda b: int(b["index"]))

        # Store directly in the store's blob cache and DB
        import json
        from .store.store import PREFIX_BLOBS

        self.store._blob_cache[block_root] = updated_blobs
        try:
            data = json.dumps(updated_blobs).encode()
            self.store._db.put(PREFIX_BLOBS + block_root, data)
            logger.info(f"Stored blob sidecar: slot={slot}, index={index}, block_root={block_root.hex()[:16]}")
        except Exception as e:
            logger.warning(f"Failed to persist blob sidecar to LevelDB: {e}")

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
            # Update checkpoint metrics
            metrics.update_checkpoints(
                finalized=int(self.state.finalized_checkpoint.epoch),
                justified=int(self.state.current_justified_checkpoint.epoch),
            )
            return

        # Fallback: process slots to advance state, then process block
        target_slot = int(block.slot)
        if target_slot > int(self.state.slot):
            self.state = process_slots(self.state, target_slot)
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
        """Update forkchoice with the execution layer (no payload preparation)."""
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

            # Use zeros for finalized hash (finalized_checkpoint.root is beacon root, not execution hash)
            forkchoice_state = ForkchoiceState(
                head_block_hash=head_block_hash,
                safe_block_hash=head_block_hash,
                finalized_block_hash=b"\x00" * 32,
            )

            response = await self.engine.forkchoice_updated(forkchoice_state, timestamp=timestamp)
            logger.debug(
                f"Forkchoice updated (no payload prep): head={head_block_hash.hex()[:16]}, "
                f"status={response.payload_status.status}"
            )

        except Exception as e:
            logger.error(f"Failed to update forkchoice: {e}")

    @property
    def current_slot(self) -> int:
        """Get the current slot based on time."""
        network_config = get_config()
        now = int(time.time())
        return (now - self._genesis_time) // (network_config.slot_duration_ms // 1000)

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
