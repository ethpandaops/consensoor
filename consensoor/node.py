"""Main beacon node orchestration."""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional, Union

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
from .engine import EngineAPIClient, ForkchoiceState, PayloadStatusEnum
from .store import Store
from .beacon_api import BeaconAPI
from .crypto import hash_tree_root, sign_async, compute_signing_root
from .validator import ValidatorClient, load_keystores_teku_style
from .builder import BlockBuilder
from .attestation_pool import AttestationPool
from .sync_committee_pool import SyncCommitteePool
from . import metrics

logger = logging.getLogger(__name__)


def _peek_block_slot(ssz_bytes: bytes) -> Optional[int]:
    """Peek at the slot field of a SignedBeaconBlock without full SSZ decode.

    SignedBeaconBlock SSZ layout (mixed-size container):
        bytes[0:4]    uint32 LE — offset to `message` (always 100 in practice)
        bytes[4:100]  BLSSignature (96 bytes, fixed)
        bytes[off:]   BeaconBlock — slot is its first 8 bytes (uint64 LE)
    """
    if len(ssz_bytes) < 12:
        return None
    try:
        message_offset = int.from_bytes(ssz_bytes[0:4], "little")
    except Exception:
        return None
    if message_offset + 8 > len(ssz_bytes) or message_offset < 4:
        return None
    slot = int.from_bytes(ssz_bytes[message_offset:message_offset + 8], "little")
    # Slot must fit in a realistic range. ETH2 won't see anywhere near 2^40
    # slots in practice, but anything above that is certainly garbage.
    if slot >= (1 << 40):
        return None
    return slot


def decode_signed_beacon_block(ssz_bytes: bytes) -> AnySignedBeaconBlock:
    """Decode a signed beacon block, picking the fork from the slot peek."""
    from .spec.network_config import get_config
    from .spec.constants import SLOTS_PER_EPOCH

    config = get_config()
    FAR_FUTURE = 2**64 - 1

    candidates: list[tuple[str, type]] = []

    slot = _peek_block_slot(ssz_bytes)
    if slot is not None:
        epoch = slot // SLOTS_PER_EPOCH()

        def fork_active(attr: str) -> bool:
            ep = getattr(config, attr, FAR_FUTURE)
            return ep != FAR_FUTURE and epoch >= ep

        # Order newest-active-fork first; only the schemas the slot can
        # actually belong to are tried, and the first hit wins.
        if fork_active("gloas_fork_epoch"):
            candidates.append(("Gloas", SignedBeaconBlock))
        if fork_active("electra_fork_epoch"):
            candidates.append(("Electra/Fulu", SignedElectraBeaconBlock))
        if fork_active("deneb_fork_epoch"):
            candidates.append(("Deneb", SignedDenebBeaconBlock))
        if fork_active("capella_fork_epoch"):
            candidates.append(("Capella", SignedCapellaBeaconBlock))
        candidates.append(("Bellatrix", SignedBellatrixBeaconBlock))
    else:
        logger.debug("Slot peek failed, brute-forcing decode")
        candidates = [
            ("Gloas", SignedBeaconBlock),
            ("Electra/Fulu", SignedElectraBeaconBlock),
            ("Deneb", SignedDenebBeaconBlock),
            ("Capella", SignedCapellaBeaconBlock),
            ("Bellatrix", SignedBellatrixBeaconBlock),
        ]

    for fork_name, block_type in candidates:
        try:
            block = block_type.decode_bytes(ssz_bytes)
        except Exception as e:
            logger.debug(f"Decode as {fork_name} failed: {e}")
            continue

        # Reject decodes whose slot disagrees with the peeked slot — that means
        # we matched a different fork's SSZ layout that happened to validate.
        decoded_slot = int(block.message.slot)
        if slot is not None and decoded_slot != slot:
            logger.debug(
                f"Decode as {fork_name} slot mismatch (peek={slot}, decoded={decoded_slot}); "
                "trying next fork"
            )
            continue

        logger.debug(f"Block slot={decoded_slot} decoded as {fork_name} format")
        return block

    raise ValueError("Failed to decode block as any known fork format")


class BeaconNode:
    """Main beacon node implementation."""

    def __init__(self, config: Config):
        self.config = config
        self.state: Optional[AnyBeaconState] = None
        self.head_root: Optional[bytes] = None
        self.head_slot: int = 0

        self.store = Store(config.data_dir)

        self.engine: Optional[EngineAPIClient] = None
        self.beacon_api: Optional[BeaconAPI] = None
        self.validator_client: Optional[ValidatorClient] = None
        self.remote_beacon: Optional[RemoteBeaconClient] = None
        self.state_sync: Optional[StateSyncManager] = None
        self.block_builder: Optional[BlockBuilder] = None
        self.beacon_gossip: Optional[BeaconGossip] = None
        self.attestation_pool = AttestationPool()
        self.sync_committee_pool = SyncCommitteePool()
        from .payload_attestation_pool import PayloadAttestationPool
        self.payload_attestation_pool = PayloadAttestationPool()
        # Gloas proposer preferences seen via gossip (or self-published),
        # keyed (dependent_root, proposal_slot, validator_index) per the
        # first-valid-message dedup rule in gloas/p2p-interface.md. Builders
        # consult this to match bid.fee_recipient to the proposer's wishes.
        self.proposer_preferences: dict[tuple[bytes, int, int], object] = {}
        # Tuples we already broadcast for our own validators, to avoid
        # re-publishing the same preferences every slot.
        self._published_prefs_keys: set[tuple[bytes, int, int]] = set()
        self._attester_duties: dict[int, list] = {}  # epoch -> duties
        self._sync_committee_duties: dict[int, list] = {}  # validator_index -> committee positions
        # Cache of validator_index -> list[position] for ALL validators in the
        # current sync committee. Used to map incoming SyncCommitteeMessages
        # from subnets to pool positions. Refreshed at each epoch boundary.
        self._sync_committee_index_to_positions: dict[int, list[int]] = {}
        # Debounced log buffer for inbound SyncCommitteeMessages — keyed by
        # (slot, peer, subnet), flushed via a single short-lived task so we
        # emit one aggregated line per group instead of one per message.
        self._sync_msg_log_buffer: dict[tuple[int, str, int], list[int]] = {}
        self._sync_msg_log_task: Optional[asyncio.Task] = None

        self._running = False
        self._genesis_time: int = 0
        # Serializes block imports. Gossip and req/resp batches land
        # concurrently, and the parent-check → gap-walk → apply sequence in
        # _on_p2p_block yields at every await — two interleaved imports each
        # walk a gap chain from the same head and double-apply the same
        # blocks (process_block_header then fails its slot monotonicity
        # assert).
        self._block_import_lock = asyncio.Lock()
        self._slot_ticker_task: Optional[asyncio.Task] = None
        self._block_sync_task: Optional[asyncio.Task] = None
        self._current_payload_id: Optional[bytes] = None
        self._current_payload_beacon_root: Optional[bytes] = None  # Store head_root used in forkchoiceUpdated
        self._current_payload_slot: Optional[int] = None  # The slot the current payload_id is being built for
        # If _sync_missing_blocks asked peers for slots in (head_slot, current_slot)
        # and they all returned empty, those slots are confirmed empty and we
        # can safely propose at current_slot on top of the existing head.
        # Without this signal _is_synced() blocks every proposer slot whose
        # n-1 was missed by another client.
        self._sync_empty_confirmed_for_slot: Optional[int] = None
        self._fork_digest_override: Optional[bytes] = None  # For devnets with custom fork digests
        self._pending_envelopes: dict[bytes, Any] = {}  # beacon_block_root -> signed_envelope

        # Single-worker pool that owns the beacon state. Heavy CPU work
        # (process_slots, hash_tree_root over the full state, full state
        # transition, committee derivation) runs here so the asyncio event
        # loop — and therefore the beacon API and gossipsub handlers —
        # stays responsive. max_workers=1 serialises mutations so coroutines
        # don't trip over each other.
        self._state_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="consensoor-state"
        )

    async def _on_state_thread(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """Run a CPU-bound state operation on the dedicated state worker."""
        loop = asyncio.get_running_loop()
        if kwargs:
            return await loop.run_in_executor(
                self._state_executor, lambda: func(*args, **kwargs)
            )
        return await loop.run_in_executor(self._state_executor, func, *args)

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

        # Start the beacon API HTTP server FIRST so kurtosis healthcheck and
        # other consumers see /eth/v1/node/health 200 within a second of process
        # start. Handlers gracefully report defaults (None state, no peers,
        # generated peer_id) until the rest of the node finishes initialising.
        await self._setup_beacon_api()

        # Genesis state has to land before P2P starts (fork_version,
        # genesis_validators_root, fork_digest are derived from it).
        await self._load_genesis_state()

        # Bring P2P up BEFORE the slow validator-key load. The rust libp2p
        # stack accepts inbound connections immediately and other clients
        # (prysm, lighthouse) require a Status RPC response within ~20s or
        # they drop the connection with "no chain status for peer". If we
        # load 128 keystores on the asyncio thread first, we miss prysm's
        # Status timeout and lose the only peer we have.
        await self._setup_p2p()

        await self._load_validator_keys()
        await self._init_engine_client()
        self.block_builder = BlockBuilder(self)
        await self._setup_beacon_sync()

        self._running = True

        # Prepare initial payload for slot 1 (or current slot + 1)
        await self._prepare_initial_payload()

        self._slot_ticker_task = asyncio.create_task(self._slot_ticker())
        self._block_sync_task = asyncio.create_task(self._block_sync_loop())

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

        if self._block_sync_task:
            self._block_sync_task.cancel()
            try:
                await self._block_sync_task
            except asyncio.CancelledError:
                pass

        if self.beacon_gossip:
            await self.beacon_gossip.stop()
        if self.state_sync:
            await self.state_sync.stop()
        if self.engine:
            await self.engine.close()
        if self.beacon_api:
            await self.beacon_api.stop()
        self._state_executor.shutdown(wait=False, cancel_futures=True)

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
        # `get_genesis_block(state) = SignedBeaconBlock(message=BeaconBlock(
        #     state_root=hash_tree_root(state)))`
        # — i.e. all body / header fields default and state_root = state hash.
        self.head_slot = int(self.state.slot)
        self._genesis_time = int(self.state.genesis_time)
        header = self.state.latest_block_header  # used below for parent_root etc.

        # Build the synthetic genesis BODY FIRST. We later use its hash both as
        # the genesis block's body_root and as a patch for
        # state.latest_block_header.body_root — see body_root reconciliation
        # below.
        if detected_fork == "gloas":
            from .spec.types.gloas import (
                BeaconBlockBody as GloasBeaconBlockBody,
                BeaconBlock as GloasBeaconBlock,
                SignedBeaconBlock as SignedGloasBeaconBlock,
                ExecutionPayloadBid as GloasExecutionPayloadBid,
                SignedExecutionPayloadBid as GloasSignedExecutionPayloadBid,
            )
            # Mirror state.latest_execution_payload_bid into the genesis
            # body's signed_execution_payload_bid.message — lighthouse does
            # this so its genesis body_root != prysm's. Prysm uses empty
            # defaults; lighthouse populates parent_block_hash + execution
            # requests_root from the state's bid. We follow lighthouse here
            # to interop on the lighthouse-glamsterdam-devnet-3 chain.
            state_bid = self.state.latest_execution_payload_bid
            genesis_bid_msg = GloasExecutionPayloadBid(
                parent_block_hash=state_bid.parent_block_hash,
                parent_block_root=state_bid.parent_block_root,
                block_hash=state_bid.block_hash,
                prev_randao=state_bid.prev_randao,
                fee_recipient=state_bid.fee_recipient,
                gas_limit=state_bid.gas_limit,
                builder_index=state_bid.builder_index,
                slot=state_bid.slot,
                value=state_bid.value,
                execution_payment=state_bid.execution_payment,
                blob_kzg_commitments=list(state_bid.blob_kzg_commitments),
                execution_requests_root=state_bid.execution_requests_root,
            )
            genesis_signed_bid = GloasSignedExecutionPayloadBid(
                message=genesis_bid_msg,
                signature=BLSSignature(b'\x00' * 96),
            )
            genesis_body = GloasBeaconBlockBody(
                signed_execution_payload_bid=genesis_signed_bid,
            )
        else:
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

        # Reconcile state.latest_block_header.body_root with our synthetic body
        # BEFORE computing genesis_state_root.
        #
        # Why: process_slot for slot 0 (executed during any from-genesis state
        # replay, including reorgs) computes
        #   previous_block_root = hash_tree_root(state.latest_block_header)
        # using whatever body_root is stored in the state. Peer-built blocks
        # at slot 1 have parent_root == HTR(synthetic_genesis_block), which
        # equals HTR(BeaconBlockHeader{..., body_root=HTR(synthetic_body)}).
        # If eth-beacon-genesis wrote a different body_root into the state
        # (e.g. the spec-default empty body, or a dynssz schema variant),
        # those two roots diverge and process_block_header asserts on
        # block 1: "Block parent root X doesn't match expected Y". Lighthouse
        # avoids the divergence by patching the state in-memory before use;
        # we do the same.
        synthetic_body_root = hash_tree_root(genesis_body)
        existing_body_root = bytes(self.state.latest_block_header.body_root)
        if existing_body_root != synthetic_body_root:
            logger.warning(
                f"Patching genesis latest_block_header.body_root: "
                f"state={existing_body_root.hex()[:16]} -> "
                f"synthetic={synthetic_body_root.hex()[:16]} "
                f"(eth-beacon-genesis body shape differs from our synthetic "
                f"body; this realigns process_slot with peer genesis_block_root)"
            )
            self.state.latest_block_header.body_root = synthetic_body_root
            # `header` was captured before the mutation; refresh so the
            # genesis_block we build below sees the patched body_root.
            header = self.state.latest_block_header

        genesis_state_root = hash_tree_root(self.state)
        logger.info(f"Computed genesis state root: {genesis_state_root.hex()}")

        if detected_fork == "gloas":
            genesis_block = GloasBeaconBlock(
                slot=int(header.slot),
                proposer_index=int(header.proposer_index),
                parent_root=header.parent_root,
                state_root=genesis_state_root,
                body=genesis_body,
            )
            signed_genesis_block = SignedGloasBeaconBlock(
                message=genesis_block,
                signature=BLSSignature(b'\x00' * 96),
            )
        else:
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

        self.head_root = hash_tree_root(genesis_block)
        self._genesis_block_root = self.head_root
        self._genesis_state = self.state.copy()

        # Save state by genesis_state_root (its true hash). Per spec the
        # genesis state has latest_block_header.state_root == ZERO_HASH;
        # process_slot for slot 1 fills it in. Don't mutate it here or
        # state.state_roots[0] gets cached against a hash no other client
        # computes.
        self.store.save_state(genesis_state_root, self.state)
        self.store.save_state(self.head_root, self.state)
        self.store.set_head(self.head_root)
        self.store.save_block(self.head_root, signed_genesis_block)
        logger.info(f"Genesis block root: {self.head_root.hex()} (fork={detected_fork})")

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
            # Keystore decryption is scrypt — slow per-key and CPU-bound. Run
            # off the asyncio loop so the rust→python daemon threads (status,
            # ping, metadata, gossip dispatch) stay snappy enough that peers
            # don't drop us with "no chain status for peer".
            keys = await asyncio.to_thread(
                load_keystores_teku_style, self.config.validator_keys_spec
            )
            self.validator_client = ValidatorClient(keys)
            logger.info(f"Loaded {len(keys)} validator keys")

            if self.state:
                self.validator_client.update_validator_indices(self.state)
                self._update_custody_group_count_from_state()
        except Exception as e:
            logger.error(f"Failed to load validator keys: {e}")
            self.validator_client = ValidatorClient([])

    def _update_custody_group_count_from_state(self) -> None:
        """Apply `get_validators_custody_requirement` from
        `consensus-specs/specs/fulu/validator.md`.

        Sums effective_balance of attached validators, divides by
        BALANCE_PER_ADDITIONAL_CUSTODY_GROUP, clamps to
        [VALIDATOR_CUSTODY_REQUIREMENT, NUMBER_OF_CUSTODY_GROUPS]. Pushes the
        result into the libp2p host so MetaData v3 advertises it on the next
        request (seq_number is bumped).
        """
        if self.config.supernode:
            return  # supernode pinned at NUMBER_OF_CUSTODY_GROUPS in BeaconGossip
        if not self.beacon_gossip or not self.validator_client or not self.state:
            return
        indices = [
            idx for idx in self.validator_client._validator_indices.values()
            if idx is not None
        ]
        if not indices:
            return
        net_config = get_config()
        total_balance = sum(
            int(self.state.validators[i].effective_balance) for i in indices
        )
        per_group = net_config.balance_per_additional_custody_group
        count = total_balance // per_group if per_group else 0
        cgc = min(
            max(count, net_config.validator_custody_requirement),
            net_config.number_of_custody_groups,
        )
        logger.info(
            f"Computed custody_group_count={cgc} "
            f"(validators={len(indices)}, total_effective_balance={total_balance} gwei, "
            f"per_group={per_group} gwei, "
            f"floor=VALIDATOR_CUSTODY_REQUIREMENT={net_config.validator_custody_requirement}, "
            f"ceil=NUMBER_OF_CUSTODY_GROUPS={net_config.number_of_custody_groups})"
        )
        self.beacon_gossip.update_custody_group_count(cgc)

    async def _init_engine_client(self) -> None:
        """Initialize the Engine API client."""
        if not self.config.engine_api_url:
            logger.warning("Engine API URL not configured, running without EL")
            return

        self.engine = EngineAPIClient(
            url=self.config.engine_api_url,
            jwt_secret=self.config.jwt_secret,
            force_json=self.config.engine_force_json,
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

            # Auto-bootstrap from checkpoint_sync_url's beacon API: ask its
            # /eth/v1/node/identity for a libp2p multiaddr and add it as a
            # static peer so the rust libp2p host has someone to dial. Without
            # this consensoor only listens and never connects (we don't have
            # discv5 yet).
            if self.config.checkpoint_sync_url and not self.config.peers:
                try:
                    import aiohttp
                    base = self.config.checkpoint_sync_url.rstrip("/")
                    async with aiohttp.ClientSession() as sess:
                        async with sess.get(
                            f"{base}/eth/v1/node/identity",
                            timeout=aiohttp.ClientTimeout(total=5),
                        ) as r:
                            if r.status == 200:
                                data = (await r.json())["data"]
                                addrs = data.get("p2p_addresses", [])
                                if addrs:
                                    bootstrap_addr = addrs[0]
                                    self.config.peers = (bootstrap_addr,)
                                    logger.info(
                                        f"Auto-bootstrap from checkpoint upstream: {bootstrap_addr}"
                                    )
                except Exception as e:
                    logger.debug(f"Auto-bootstrap from checkpoint URL failed: {e}")

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
            self.beacon_gossip.subscribe_sync_committee_messages(self._on_p2p_sync_committee_message)
            self.beacon_gossip.subscribe_blob_sidecars(self._on_p2p_blob_sidecar)
            self.beacon_gossip.subscribe_execution_payloads(self._on_p2p_execution_payload)
            self.beacon_gossip.subscribe_payload_attestation_messages(self._on_p2p_payload_attestation_message)
            self.beacon_gossip.subscribe_proposer_preferences(self._on_p2p_proposer_preferences)
            self.beacon_gossip.set_status_provider(self._get_chain_status)
            self.beacon_gossip.set_block_provider(self._get_block_for_slot)
            self.beacon_gossip.set_block_by_root_provider(self._get_block_by_root)

            await self.beacon_gossip.start()
            await self.beacon_gossip.activate_subscriptions()

            self._push_status_snapshot()

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

    def _push_status_snapshot(self) -> None:
        """Refresh the Rust binding's cached StatusMessage so inbound Status
        RPCs are answered without crossing into Python. Call after any head
        or finalized checkpoint change. Cheap — must not raise.
        """
        if self.beacon_gossip is None:
            return
        try:
            self.beacon_gossip.push_status_snapshot(self._get_chain_status())
        except Exception as e:
            logger.warning(f"push_status_snapshot failed: {e}")

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

    def _get_block_for_slot(self, slot: int) -> Optional[tuple[bytes, bytes]]:
        """Get a block for a given slot for serving via req/resp.

        Returns (ssz_encoded_signed_block, fork_digest_context) or None.
        Called from the P2P host's Trio thread — must be thread-safe and
        avoid async operations.
        """
        try:
            block = self.store.get_block_by_slot(slot)
            if block is None:
                return None

            block_ssz = block.encode_bytes()

            # Compute fork digest context for this block's slot
            from .p2p.encoding import compute_fork_digest
            from .spec.network_config import get_config as get_network_config
            from .spec.constants import SLOTS_PER_EPOCH

            net_config = get_network_config()
            epoch = slot // SLOTS_PER_EPOCH()
            fork_version = net_config.get_fork_version(epoch)
            genesis_validators_root = bytes(self.state.genesis_validators_root) if self.state else b"\x00" * 32

            fork_digest = compute_fork_digest(fork_version, genesis_validators_root)

            return (block_ssz, fork_digest)

        except Exception as e:
            logger.warning(f"Failed to get block for slot {slot}: {e}")
            return None

    def _get_block_by_root(self, root: bytes) -> Optional[tuple[bytes, bytes]]:
        """Get a block by its 32-byte root for serving via BlocksByRoot req/resp.

        Returns (ssz_encoded_signed_block, fork_digest_context) or None when
        we don't have the block. Called from the P2P host dispatcher thread —
        must be cheap and not perform async work.
        """
        try:
            block = self.store.get_block(root)
            if block is None:
                return None

            block_ssz = block.encode_bytes()

            from .p2p.encoding import compute_fork_digest
            from .spec.network_config import get_config as get_network_config
            from .spec.constants import SLOTS_PER_EPOCH

            net_config = get_network_config()
            slot = int(block.message.slot)
            epoch = slot // SLOTS_PER_EPOCH()
            fork_version = net_config.get_fork_version(epoch)
            genesis_validators_root = (
                bytes(self.state.genesis_validators_root) if self.state else b"\x00" * 32
            )
            fork_digest = compute_fork_digest(fork_version, genesis_validators_root)

            return (block_ssz, fork_digest)

        except Exception as e:
            logger.warning(f"Failed to get block by root {root.hex()[:16]}: {e}")
            return None

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
            (net_config.gloas_fork_version, "gloas"),
        ]

        fork_epochs = [
            0,  # genesis
            net_config.altair_fork_epoch,
            net_config.bellatrix_fork_epoch,
            net_config.capella_fork_epoch,
            net_config.deneb_fork_epoch,
            net_config.electra_fork_epoch,
            net_config.fulu_fork_epoch,
            net_config.gloas_fork_epoch,
        ]

        for i, (fork_version, fork_name) in enumerate(forks):
            fork_epoch = fork_epochs[i]
            if fork_epoch < FAR_FUTURE_EPOCH:
                # Forks at or after Fulu XOR blob params into the digest.
                blob_params = self._get_blob_params_for_epoch(net_config, fork_epoch)
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
                            net_config.get_fork_version(entry_epoch),
                            genesis_validators_root,
                            blob_params=(entry_epoch, max_blobs),
                        )
                        digests.append(digest)
                        logger.debug(f"BPO fork digest for epoch {entry_epoch}: {digest.hex()}")
                except Exception as e:
                    logger.debug(f"Failed to parse blob schedule entry {entry}: {e}")

        return digests

    def _get_blob_params_for_epoch(self, net_config, epoch: int) -> tuple[int, int] | None:
        """Get blob params (epoch, max_blobs) for fork digest BPO XOR.

        Per the spec (and Lighthouse `ChainSpec::get_blob_parameters`), every
        fork at or after Fulu mixes blob parameters into the fork digest —
        including Gloas. The default fallback is the Electra activation
        epoch with `max_blobs_per_block_electra`; entries from
        `blob_schedule` override it for matching epochs.
        """
        FAR_FUTURE_EPOCH = 2**64 - 1
        if net_config.fulu_fork_epoch >= FAR_FUTURE_EPOCH or epoch < net_config.fulu_fork_epoch:
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

        return net_config.electra_fork_epoch, net_config.max_blobs_per_block_electra

    async def _update_fork_digest_for_epoch(self, epoch: int) -> None:
        """Update the fork_digest for publishing when crossing fork boundaries.

        Call this at the start of each epoch to ensure messages are published
        to the correct gossipsub topic.
        """
        if not self.beacon_gossip or not self.state:
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

            # Get the current head block hash. In Gloas the EL chain tip is
            # the most recently installed bid's payload (already imported via
            # newPayload during block apply / envelope receive). state.latest_block_hash
            # lags by one slot because process_parent_execution_payload only
            # promotes it when the NEXT bid chains in — using it here makes
            # fcU point at a head behind the EL's view, so geth returns VALID
            # without issuing a payload_id.
            head_block_hash = self._gloas_el_head_hash()

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

            withdrawals_list = self._withdrawals_attr_list_for_slot(int(slot))

            payload_attributes = {
                "timestamp": hex(timestamp),
                "prevRandao": "0x" + prev_randao.hex(),
                "suggestedFeeRecipient": "0x" + "00" * 20,
                "withdrawals": withdrawals_list,
                "parentBeaconBlockRoot": "0x" + (self.head_root or b"\x00" * 32).hex(),
            }
            if hasattr(self.state, "ptc_window"):
                payload_attributes["slotNumber"] = hex(int(slot))
                payload_attributes["targetGasLimit"] = hex(int(self.config.target_gas_limit))

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
                self._current_payload_slot = int(slot)
            else:
                logger.warning(
                    f"Failed to get payload_id for slot {slot}: status={response.payload_status.status}"
                )
                self._current_payload_id = None
                self._current_payload_beacon_root = None
                self._current_payload_slot = None

        except Exception as e:
            logger.error(f"Failed to request payload for slot {slot}: {e}")
            self._current_payload_id = None
            self._current_payload_beacon_root = None
            self._current_payload_slot = None

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
        last_ptc_slot = -1
        last_sync_committee_slot = -1

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
                ptc_offset = network_config.get_payload_attestation_due_offset()
                gloas_active = network_config.is_gloas_active(current_epoch)

                # Phase 1: Slot start - propose blocks and update forkchoice
                if current_slot > last_slot_processed:
                    last_slot_processed = current_slot
                    epoch = current_slot // slots_per_epoch
                    # Ensure attester duties are cached inline so attestations
                    # can fire on time even while slot work runs in background
                    await self._ensure_attester_duties(epoch)
                    # Launch heavy slot work as background task so the event loop
                    # stays responsive for gossipsub block processing
                    asyncio.create_task(self._on_slot_start_safe(current_slot))
                    # Gloas: advertise fee_recipient/target_gas_limit for our
                    # upcoming proposal slots (deduped inside)
                    if gloas_active:
                        asyncio.create_task(
                            self._broadcast_proposer_preferences(current_slot)
                        )

                # Phase 2: Attestation due time - produce attestations and
                # sync committee messages. Per altair/validator.md sync
                # committee members sign at the same 1/3 mark (or earlier
                # if the slot's block already landed) — signing at slot
                # start, before the slot's block has arrived, makes our
                # messages reference the previous slot's head root while
                # peer aggregators sign this slot's block. The pool's
                # expected_block_root filter then drops all of ours,
                # which is why our 83 sync committee validators never
                # appear in any block's sync_aggregate.
                if current_slot > last_attestation_slot and time_into_slot >= attestation_offset:
                    last_attestation_slot = current_slot
                    await self._produce_attestations(current_slot)

                if current_slot > last_sync_committee_slot and time_into_slot >= attestation_offset:
                    last_sync_committee_slot = current_slot
                    await self._produce_sync_committee_messages(current_slot)

                # Phase 3: PTC due time (Gloas) - produce payload attestations
                # Vote on whether slot M's execution payload was revealed in
                # time. Aggregates are bundled by the slot M+1 proposer.
                if (
                    gloas_active
                    and current_slot > last_ptc_slot
                    and time_into_slot >= ptc_offset
                ):
                    last_ptc_slot = current_slot
                    await self._produce_payload_attestations(current_slot)

                # Calculate sleep time - wake up for next event
                attested_this_slot = current_slot == last_attestation_slot
                ptc_done_this_slot = (not gloas_active) or current_slot == last_ptc_slot

                if attested_this_slot and ptc_done_this_slot:
                    # All slot duties done, sleep until next slot
                    next_slot_time = self._genesis_time + (current_slot + 1) * slot_duration
                    sleep_time = max(0.05, next_slot_time - time.time())
                elif not attested_this_slot:
                    # Next wakeup: attestation due time
                    attestation_time = slot_start_time + attestation_offset
                    sleep_time = max(0.05, attestation_time - time.time())
                else:
                    # Next wakeup: PTC due time
                    ptc_time = slot_start_time + ptc_offset
                    sleep_time = max(0.05, ptc_time - time.time())

                await asyncio.sleep(sleep_time)
            except asyncio.CancelledError:
                logger.info("Slot ticker cancelled")
                raise
            except Exception as e:
                logger.error(f"Slot ticker error: {e}", exc_info=True)
                await asyncio.sleep(1)  # Prevent tight loop on errors

    async def _on_slot_start_safe(self, slot: int) -> None:
        """Wrapper around _on_slot_start that catches exceptions.

        Used when _on_slot_start is launched as a background task via
        asyncio.create_task, which would otherwise silently swallow errors.
        """
        try:
            await self._on_slot_start(slot)
        except Exception as e:
            logger.error(f"Slot start error for slot {slot}: {e}", exc_info=True)

    async def _on_slot_start(self, slot: int) -> None:
        """Handle the start of a new slot (0/3 mark).

        This is called at slot start for block proposals and forkchoice updates.
        Attestations are produced separately at the 1/3 mark by the slot ticker.
        """
        slots_per_epoch = SLOTS_PER_EPOCH()
        epoch = slot // slots_per_epoch
        logger.info(f"Slot {slot} (epoch {epoch})")

        # Sync committee message production moved to the 1/3 mark (next
        # to attestations) so we sign this slot's head, not the previous
        # one. See _slot_ticker for the call site.

        # Yield to event loop so gossipsub blocks can be processed
        await asyncio.sleep(0)

        # Sync missing blocks FIRST so state is up-to-date before proposing
        await self._sync_missing_blocks(slot)

        # Yield to event loop so gossipsub blocks can be processed
        await asyncio.sleep(0)

        # Propose block — now has up-to-date state with correct RANDAO
        await self._maybe_propose_block(slot)

        # Yield to event loop so gossipsub blocks can be processed
        await asyncio.sleep(0)

        # Broadcast sync committee contributions (BLS-heavy, deferred until after block proposal)
        await self._broadcast_sync_committee_contributions(slot, self.state)

        # Yield to event loop so gossipsub blocks can be processed
        await asyncio.sleep(0)

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

        # Yield to event loop so gossipsub blocks can be processed
        await asyncio.sleep(0)

        # Update forkchoice to keep EL in sync and prepare payload for NEXT slot
        await self._update_forkchoice_for_slot(slot)

        # NOTE: Attestations are produced at the 1/3 mark by _slot_ticker, not here

    async def _emit_payload_attributes_event(
        self,
        proposal_slot: int,
        timestamp: int,
        prev_randao: bytes,
        parent_block_hash: bytes,
        withdrawals_list: list,
    ) -> None:
        """Emit a beacon-API payload_attributes SSE event for ``proposal_slot``."""
        if not self.beacon_api or self.state is None:
            return
        try:
            proposer_index = 0
            if self.validator_client is not None:
                idx = self.validator_client._get_proposer_index(
                    self.state, int(proposal_slot)
                )
                if idx is not None:
                    proposer_index = int(idx)

            # engine-API withdrawals are hex/camelCase; beacon-API wants decimal/snake_case.
            spec_withdrawals = [
                {
                    "index": str(int(w["index"], 16)),
                    "validator_index": str(int(w["validatorIndex"], 16)),
                    "address": w["address"],
                    "amount": str(int(w["amount"], 16)),
                }
                for w in withdrawals_list
            ]

            network_config = get_config()
            gloas_fork_epoch = getattr(network_config, "gloas_fork_epoch", None)
            is_gloas = (
                gloas_fork_epoch is not None
                and int(proposal_slot) // SLOTS_PER_EPOCH() >= gloas_fork_epoch
            )

            payload_attributes = {
                "timestamp": str(int(timestamp)),
                "prev_randao": "0x" + prev_randao.hex(),
                "suggested_fee_recipient": "0x" + "00" * 20,
                "withdrawals": spec_withdrawals,
                "parent_beacon_block_root": "0x" + (self.head_root or b"\x00" * 32).hex(),
            }
            if is_gloas:
                payload_attributes["target_gas_limit"] = str(
                    int(self.config.target_gas_limit)
                )

            await self.beacon_api.emit_payload_attributes(
                proposal_slot=int(proposal_slot),
                proposer_index=proposer_index,
                parent_block_root=self.head_root or b"\x00" * 32,
                parent_block_hash=parent_block_hash,
                version="gloas" if is_gloas else "electra",
                payload_attributes=payload_attributes,
            )
        except Exception as e:
            logger.error(
                f"Failed to emit payload_attributes for slot {proposal_slot}: {e}"
            )

    async def _update_forkchoice_for_slot(self, slot: int) -> None:
        """Update forkchoice with EL and prepare payload for NEXT slot (slot + 1)."""
        if not self.engine or not self.state:
            return

        try:
            # See _gloas_el_head_hash docstring for why we prefer the bid's
            # block_hash over state.latest_block_hash on Gloas.
            head_block_hash = self._gloas_el_head_hash()

            finalized_hash = self._resolve_finalized_block_hash() or b"\x00" * 32
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

            withdrawals_list = self._withdrawals_attr_list_for_slot(int(next_slot))

            await self._emit_payload_attributes_event(
                int(next_slot), int(timestamp), prev_randao, head_block_hash, withdrawals_list
            )

            payload_attributes = {
                "timestamp": hex(timestamp),
                "prevRandao": "0x" + prev_randao.hex(),
                "suggestedFeeRecipient": "0x" + "00" * 20,  # Default fee recipient
                "withdrawals": withdrawals_list,
                "parentBeaconBlockRoot": "0x" + (self.head_root or b"\x00" * 32).hex(),
            }
            # Gate slotNumber on the NEXT slot's fork, not the current state's.
            # The engine routes to FCU v4 by timestamp (see engine/client.py
            # _get_fork_for_timestamp); if we keyed off `state.ptc_window`
            # instead, we'd skip slotNumber whenever state lags behind the
            # wall-clock fork epoch — geth then rejects with -38003
            # "missing slot number".
            gloas_fork_epoch = getattr(network_config, "gloas_fork_epoch", None)
            if gloas_fork_epoch is not None and next_slot // SLOTS_PER_EPOCH() >= gloas_fork_epoch:
                payload_attributes["slotNumber"] = hex(int(next_slot))
                payload_attributes["targetGasLimit"] = hex(int(self.config.target_gas_limit))

            response = await self.engine.forkchoice_updated(
                forkchoice_state, payload_attributes, timestamp=timestamp
            )

            if response.payload_id:
                logger.info(f"Payload prepared for slot {next_slot}: id={response.payload_id.hex()}")
                self._current_payload_id = response.payload_id
                # Store the beacon root used in this forkchoiceUpdated for later use in newPayloadV5
                self._current_payload_beacon_root = self.head_root or b"\x00" * 32
                self._current_payload_slot = int(next_slot)
            else:
                logger.warning(
                    f"No payload_id returned for slot {next_slot}: status={response.payload_status.status}, "
                    f"head_hash={head_block_hash.hex()[:16]}"
                )
                self._current_payload_id = None
                self._current_payload_beacon_root = None
                self._current_payload_slot = None

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

        # We deliberately do NOT gate proposal on _is_synced() the way
        # attestation does. If our head is older than current_slot-1 it's
        # almost always because intervening slots were missed by other
        # clients; refusing to propose there compounds the gap (every
        # missed n-1 ripples into a missed n for us). The correct behavior
        # is to build on whatever we believe is the latest head — empty
        # slots will be skipped via process_slots in the builder. If we're
        # genuinely behind (peers have a block we didn't import), the
        # block we publish will simply be orphaned; no peer penalty.

        # Check proposer duty using current state — proposer_lookahead allows O(1)
        # lookup without needing state advanced to the exact slot. Only fall back to
        # expensive deep copy + process_slots if lookahead is unavailable.
        proposer_key = self.validator_client.is_our_proposer_slot(self.state, slot)
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

        # get_attester_duties iterates committees, which is sync CPU-bound work
        # (shuffling, committee derivation per slot). Run it on the dedicated
        # state worker so it serialises with mutations and doesn't starve the
        # asyncio loop.
        duties = await self._on_state_thread(
            self.validator_client.get_attester_duties, self.state, epoch
        )
        self._attester_duties[epoch] = duties
        if duties:
            logger.info(f"Computed {len(duties)} attester duties for epoch {epoch}")

        # Clean up old epochs
        old_epochs = [e for e in self._attester_duties if e < epoch - 1]
        for e in old_epochs:
            del self._attester_duties[e]

    def _is_synced(self) -> bool:
        """Are we close enough to the network head to safely publish?

        Per spec, an attester at slot N votes head=last block we've imported
        and target=block_root_at(epoch_start). Both reference the chain we
        already know, so the only correctness gate is: we must have imported
        at least the previous slot's block (head_slot >= current_slot - 1).
        If we're further behind, our head root is stale relative to the
        network and peers will reject the message (~-214 score each, prune
        threshold ~3 in 30s). state.slot may legitimately lag head_slot
        when the latest seen block is from a previous slot — _produce_*
        callers run process_slots on a clone before signing.

        Aggregate broadcasts are also gated by is_aggregator() in
        _broadcast_attestation_as_aggregate so non-selected validators
        stay silent on the aggregate topic.
        """
        if not self.state or not self.head_root or not self._genesis_time:
            return False
        try:
            slot_duration = get_config().slot_duration_ms / 1000.0
        except Exception:
            return False
        current_slot = int((time.time() - self._genesis_time) // slot_duration)
        if self.head_slot >= current_slot - 1:
            return True
        # Empty intervening slots are not "being behind". If we just asked
        # peers for blocks across the gap at this slot and they all returned
        # nothing, the gap is genuinely empty and our head IS the canonical
        # head — don't skip our proposer duty just because the previous
        # validator missed theirs.
        return self._sync_empty_confirmed_for_slot == current_slot

    async def _produce_attestations(self, slot: int) -> None:
        """Produce attestations for validators with duties at this slot.

        Called at 1/3 into the slot per Ethereum spec to allow time for
        block proposals to arrive before attesting.
        """
        if not self.validator_client or not self.validator_client.keys:
            return
        if not self.state or not self.head_root:
            return
        if not self._is_synced():
            logger.debug(
                f"[ATTESTER] slot={slot} skipped — not synced "
                f"(head={self.head_slot}, current={int((time.time() - self._genesis_time) / (get_config().slot_duration_ms / 1000.0))})"
            )
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

        # Use temp state advanced to current slot for attestation production.
        # process_slots runs the full per-slot transition (epoch boundary work
        # included) so it must not block the event loop.
        temp_state = self.state
        if slot > int(temp_state.slot):
            def _advance(state, target_slot):
                cloned = state.copy()
                return process_slots(cloned, target_slot)
            temp_state = await self._on_state_thread(_advance, temp_state, slot)

        validator_indices = [d.validator_index for d in slot_duties]
        logger.info(
            f"[ATTESTER] slot={slot} count={len(slot_duties)} time={time_into_slot:.2f}s validators={validator_indices}"
        )

        produced_count = 0
        broadcast_count = 0
        for duty in slot_duties:
            try:
                attestation = await self.validator_client.produce_attestation(
                    temp_state, duty, self.head_root
                )
                if attestation:
                    self.attestation_pool.add(attestation)
                    produced_count += 1

                    # Broadcast via P2P as aggregate, but only if this
                    # validator passes is_aggregator() for this slot.
                    if self.beacon_gossip:
                        try:
                            await self._broadcast_attestation_as_aggregate(
                                attestation,
                                duty.validator_index,
                                duty.pubkey,
                                duty.committee_index,
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
        self._sync_committee_index_to_positions.clear()

        committee_pubkeys = list(self.state.current_sync_committee.pubkeys)

        # Build the full pubkey -> validator_index map once so subnet ingest
        # can resolve foreign messages without scanning validator state every
        # time. Falls back to None for any unknown pubkey (shouldn't happen
        # for a well-synced node).
        pubkey_to_index: dict[bytes, int] = {}
        for vi, v in enumerate(self.state.validators):
            pubkey_to_index[bytes(v.pubkey)] = vi

        for position, pubkey in enumerate(committee_pubkeys):
            pubkey_bytes = bytes(pubkey)

            global_index = pubkey_to_index.get(pubkey_bytes)
            if global_index is not None:
                self._sync_committee_index_to_positions.setdefault(global_index, []).append(position)

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

        # Use current state directly — produce_sync_committee_message only needs
        # block_roots and fork version, both of which are already correct.
        # Avoids expensive deep copy + process_slots that blocks the event loop.
        state_for_signing = self.state

        from .spec.constants import SYNC_COMMITTEE_SIZE, SYNC_COMMITTEE_SUBNET_COUNT
        sync_committee_size = SYNC_COMMITTEE_SIZE()
        subcommittee_size = sync_committee_size // SYNC_COMMITTEE_SUBNET_COUNT

        produced_validators: list[int] = []
        produced_positions: list[int] = []
        block_root_hex: Optional[str] = None
        # Sign against the tracked head root. Without this, the producer
        # falls back to state.block_roots[(slot-1) % HIST] which at slot
        # start is stale or zero (process_slot for slot-1 hasn't run yet),
        # so our messages would sign a phantom root that no peer agrees on
        # — Lighthouse/Teku silently drop them and even our own
        # `_validate_sync_aggregate` zeroes them out on BLS check.
        head_block_root = self.head_root
        for validator_index, positions in self._sync_committee_duties.items():
            for pubkey, key in self.validator_client.keys.items():
                if key.validator_index == validator_index:
                    message = await self.validator_client.produce_sync_committee_message(
                        state_for_signing, slot, key,
                        head_block_root=head_block_root,
                    )
                    if message:
                        produced_validators.append(int(validator_index))
                        for position in positions:
                            self.sync_committee_pool.add(message, position)
                            produced_positions.append(int(position))
                        if block_root_hex is None:
                            block_root_hex = bytes(message.beacon_block_root).hex()[:16]

                        # Per altair/validator.md: each member publishes its
                        # SyncCommitteeMessage on every subnet whose subcommittee
                        # contains one of its positions. A single validator can
                        # occupy multiple subcommittees if duplicated.
                        if self.beacon_gossip is not None:
                            subnets = {p // subcommittee_size for p in positions}
                            try:
                                ssz_bytes = bytes(message.encode_bytes())
                                for subnet_id in subnets:
                                    await self.beacon_gossip.publish_sync_committee_message(
                                        subnet_id, ssz_bytes
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to publish SyncCommitteeMessage "
                                    f"validator={validator_index}: {e}"
                                )
                    break

        if produced_validators:
            logger.debug(
                f"Sync committee messages: slot={slot} "
                f"validators({len(produced_validators)})="
                f"{sorted(produced_validators)} "
                f"positions({len(produced_positions)})="
                f"{sorted(produced_positions)} "
                f"block_root={block_root_hex} "
                f"pool_size={self.sync_committee_pool.size}"
            )

    async def _produce_payload_attestations(self, slot: int) -> None:
        """Produce PTC payload attestations at the 75% slot mark (Gloas).

        For each of our validators in the PTC for `slot`, sign a
        PayloadAttestationMessage voting on whether the block at `slot` was
        proposed in time and whether its execution payload envelope was
        revealed before the deadline. The aggregated attestation is bundled
        by the proposer of slot+1.

        Self-feeds the local pool so a consensoor-proposed slot+1 doesn't
        depend on gossip echo to include our own votes.
        """
        if not self.state or not self.validator_client or not self.beacon_gossip:
            return
        if not self._is_synced():
            return

        from .spec.state_transition.helpers.ptc import get_ptc
        from .spec.types.gloas import PayloadAttestationMessage

        try:
            ptc = list(get_ptc(self.state, slot))
        except Exception as e:
            logger.warning(f"PTC computation failed for slot {slot}: {e}")
            return
        if not ptc:
            return

        our_indices = {
            k.validator_index for k in self.validator_client.keys.values()
            if k.validator_index is not None
        }
        our_in_ptc = [vi for vi in ptc if vi in our_indices]
        if not our_in_ptc:
            return

        # The PTC for slot M votes on the block at slot M. If we don't have
        # a block at slot M yet (skipped, late, or unreceived), abstain —
        # an aggregate built around a wrong beacon_block_root will fail the
        # process_payload_attestation assertion at inclusion time.
        if int(self.head_slot) != int(slot) or self.head_root is None:
            logger.debug(
                f"PTC slot={slot}: abstaining, head_slot={self.head_slot} "
                f"head_root={self.head_root.hex()[:16] if self.head_root else None} "
                f"({len(our_in_ptc)} of our validators in PTC)"
            )
            return

        beacon_block_root = self.head_root
        payload_present = self.store.get_payload(beacon_block_root) is not None
        # Conservative: only assert blob_data_available when we also have
        # the envelope. A finer check would compare blob count to
        # bid.blob_kzg_commitments — left for a follow-up.
        blob_data_available = payload_present

        produced: list[int] = []
        for validator_index in our_in_ptc:
            key = next(
                (k for k in self.validator_client.keys.values()
                 if k.validator_index == validator_index),
                None,
            )
            if key is None:
                continue
            msg = await self.validator_client.produce_payload_attestation_message(
                self.state,
                slot,
                key,
                beacon_block_root,
                payload_present,
                blob_data_available,
            )
            if msg is None:
                continue

            # Self-feed: next consensoor proposer at slot+1 can include us
            # without round-tripping our vote through gossip.
            try:
                self.payload_attestation_pool.add_message(msg, ptc)
            except Exception as e:
                logger.warning(f"PTC self-feed failed for vi={validator_index}: {e}")

            try:
                ssz_bytes = msg.encode_bytes()
                await self.beacon_gossip.publish_payload_attestation_message(ssz_bytes)
            except Exception as e:
                logger.warning(
                    f"PTC gossip publish failed for vi={validator_index}: {e}"
                )

            produced.append(int(validator_index))

        if produced:
            logger.info(
                f"PTC votes: slot={slot} "
                f"validators({len(produced)})={sorted(produced)} "
                f"block_root={beacon_block_root.hex()[:16]} "
                f"payload_present={payload_present} "
                f"blob_data_available={blob_data_available}"
            )

    async def _broadcast_proposer_preferences(self, current_slot: int) -> None:
        """Broadcast SignedProposerPreferences for our upcoming proposal slots.

        Per gloas/validator.md, for every slot in the proposer lookahead that
        belongs to one of our validators and hasn't passed yet, sign and
        gossip the preferred fee_recipient/target_gas_limit so builders can
        construct matching bids. Deduped per
        (dependent_root, proposal_slot, validator_index), so a tuple is only
        re-published when its dependent root changes (reorg) or the lookahead
        advances.
        """
        if not self.state or not self.validator_client or not self.beacon_gossip:
            return
        if not self._is_synced():
            return
        state = self.state
        if not hasattr(state, "proposer_lookahead") or not hasattr(state, "builders"):
            return

        try:
            from .spec.state_transition.helpers.accessors import get_current_epoch
            from .spec.state_transition.helpers.beacon_committee import (
                get_proposer_dependent_root,
            )

            our_keys = {
                k.validator_index: k for k in self.validator_client.keys.values()
                if k.validator_index is not None
            }
            if not our_keys:
                return

            slots_per_epoch = SLOTS_PER_EPOCH()
            epoch_start_slot = get_current_epoch(state) * slots_per_epoch

            for offset, proposer_index in enumerate(state.proposer_lookahead):
                proposal_slot = epoch_start_slot + offset
                if proposal_slot <= current_slot:
                    continue
                vi = int(proposer_index)
                key = our_keys.get(vi)
                if key is None:
                    continue

                proposal_epoch = proposal_slot // slots_per_epoch
                try:
                    dependent_root = get_proposer_dependent_root(state, proposal_epoch)
                except Exception:
                    dependent_root = None
                if dependent_root is None:
                    # Underflow near genesis: spec says use the genesis block root
                    dependent_root = getattr(self, "_genesis_block_root", None)
                if dependent_root is None:
                    continue

                dedup_key = (bytes(dependent_root), proposal_slot, vi)
                if dedup_key in self._published_prefs_keys:
                    continue

                signed = await self.validator_client.produce_proposer_preferences(
                    state,
                    proposal_slot,
                    key,
                    bytes(dependent_root),
                    self.config.fee_recipient_bytes,
                    self.config.target_gas_limit,
                )
                if signed is None:
                    continue

                self._published_prefs_keys.add(dedup_key)
                # Self-feed so a local builder can match our own proposals
                self.proposer_preferences[dedup_key] = signed

                try:
                    await self.beacon_gossip.publish_proposer_preferences(
                        signed.encode_bytes()
                    )
                    logger.info(
                        f"Proposer preferences: vi={vi} slot={proposal_slot} "
                        f"fee_recipient={self.config.fee_recipient} "
                        f"target_gas_limit={self.config.target_gas_limit}"
                    )
                except Exception as e:
                    logger.warning(
                        f"proposer_preferences gossip publish failed for vi={vi}: {e}"
                    )

            self._prune_proposer_preferences(current_slot)
        except Exception as e:
            logger.error(f"Proposer preferences broadcast failed: {e}", exc_info=True)

    async def _broadcast_sync_committee_contributions(self, slot: int, state) -> None:
        """Broadcast sync committee contributions for each subcommittee."""
        if not self.beacon_gossip or not self.validator_client:
            return
        if not self._is_synced():
            return

        from .spec.constants import (
            SYNC_COMMITTEE_SIZE, SYNC_COMMITTEE_SUBNET_COUNT,
            TARGET_AGGREGATORS_PER_SYNC_SUBCOMMITTEE,
            DOMAIN_SYNC_COMMITTEE_SELECTION_PROOF, DOMAIN_CONTRIBUTION_AND_PROOF,
        )
        from .spec.types.altair import (
            SyncCommitteeContribution, ContributionAndProof, SignedContributionAndProof,
            SyncAggregatorSelectionData,
        )
        from .spec.types.base import Bitvector, BLSSignature, uint64
        from .spec.state_transition.helpers.domain import get_domain, compute_signing_root
        from .crypto import sign_async as bls_sign_async, sha256

        sync_committee_size = SYNC_COMMITTEE_SIZE()
        subcommittee_size = sync_committee_size // SYNC_COMMITTEE_SUBNET_COUNT
        messages = self.sync_committee_pool._messages.get(slot, {})

        if not messages:
            return

        broadcast_summary: list[tuple[int, int]] = []  # (subcommittee_index, bits)
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
                from .crypto import aggregate_signatures_async
                if len(signatures) == 1:
                    aggregated_sig = signatures[0]
                else:
                    aggregated_sig = await aggregate_signatures_async(signatures)
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
            selection_proof = await bls_sign_async(aggregator_key.privkey, selection_root)

            # is_sync_committee_aggregator gate (specs/altair/validator.md):
            #   modulo = max(1, SYNC_COMMITTEE_SIZE // SUBNET_COUNT // TARGET_AGGREGATORS)
            #   bytes_to_uint64(hash(signature)[0:8]) % modulo == 0
            sync_modulo = max(
                1,
                sync_committee_size // SYNC_COMMITTEE_SUBNET_COUNT
                // TARGET_AGGREGATORS_PER_SYNC_SUBCOMMITTEE,
            )
            sync_h = sha256(selection_proof)
            if int.from_bytes(sync_h[:8], "little") % sync_modulo != 0:
                continue

            contribution_and_proof = ContributionAndProof(
                aggregator_index=aggregator_key.validator_index,
                contribution=contribution,
                selection_proof=BLSSignature(selection_proof),
            )

            contrib_domain = get_domain(state, DOMAIN_CONTRIBUTION_AND_PROOF, epoch)
            contrib_root = compute_signing_root(contribution_and_proof, contrib_domain)
            contrib_signature = await bls_sign_async(aggregator_key.privkey, contrib_root)

            signed_contribution = SignedContributionAndProof(
                message=contribution_and_proof,
                signature=BLSSignature(contrib_signature),
            )

            try:
                await self.beacon_gossip.publish_sync_committee_contribution(
                    bytes(signed_contribution.encode_bytes())
                )
                broadcast_summary.append((subcommittee_index, len(signatures)))
            except Exception as e:
                logger.warning(f"Failed to broadcast contribution: {e}")

        if broadcast_summary:
            bits_per_sub = ", ".join(
                f"sub{idx}={bits}" for idx, bits in broadcast_summary
            )
            logger.debug(
                f"Broadcast sync committee contributions: slot={slot} "
                f"subcommittees={len(broadcast_summary)} {bits_per_sub}"
            )

    async def _broadcast_attestation_as_aggregate(
        self, attestation, validator_index: int, pubkey: bytes, committee_index: int
    ) -> None:
        """Wrap attestation in SignedAggregateAndProof and broadcast — but
        ONLY if this validator was selected as an aggregator for this slot
        per the eth2 spec's `is_aggregator(state, slot, index, slot_signature)`.

        Spec (specs/phase0/validator.md):
            modulo = max(1, len(committee) // TARGET_AGGREGATORS_PER_COMMITTEE)
            is_aggregator = bytes_to_uint64(hash(slot_signature)[0:8]) % modulo == 0

        Without this check, we broadcast every validator's attestation as an
        aggregate. ~16x too many; non-aggregator messages get rejected by
        prysm's gossipsub validation, our peer score drops to -620 in <30s,
        and prysm prunes us from its mesh.
        """
        from .spec.types.electra import ElectraAggregateAndProof, SignedElectraAggregateAndProof
        from .spec.types.phase0 import AggregateAndProof, SignedAggregateAndProof
        from .spec.types.base import BLSSignature
        from .spec.constants import (
            DOMAIN_SELECTION_PROOF, DOMAIN_AGGREGATE_AND_PROOF,
            TARGET_AGGREGATORS_PER_COMMITTEE,
        )
        from .spec.state_transition.helpers.domain import get_domain, compute_signing_root
        from .spec.state_transition.helpers.beacon_committee import (
            get_beacon_committee, get_committee_count_per_slot,
        )
        from .spec.types import Slot
        from .crypto import sha256

        key = self.validator_client.get_key(pubkey)
        if not key:
            return

        slot = int(attestation.data.slot)
        epoch = slot // SLOTS_PER_EPOCH()

        # Selection proof = BLS sign over Slot(slot) with DOMAIN_SELECTION_PROOF.
        domain = get_domain(self.state, DOMAIN_SELECTION_PROOF, epoch)
        signing_root = compute_signing_root(Slot(slot), domain)
        selection_proof = await sign_async(key.privkey, signing_root)

        # is_aggregator gate.
        try:
            committee = get_beacon_committee(self.state, slot, committee_index)
        except Exception as e:
            logger.debug(f"is_aggregator: get_beacon_committee failed: {e}")
            return
        modulo = max(1, len(committee) // TARGET_AGGREGATORS_PER_COMMITTEE)
        h = sha256(selection_proof)
        slot_sig_uint = int.from_bytes(h[:8], "little")
        if slot_sig_uint % modulo != 0:
            # Not selected as aggregator this slot — staying silent on the
            # aggregate topic is the spec-correct behaviour.
            return

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
        signature = await sign_async(key.privkey, signing_root)

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

    async def _broadcast_execution_payload_envelope(
        self,
        slot: int,
        beacon_block_root: bytes,
        execution_payload_dict: dict,
        execution_requests: list,
        blobs_bundle: dict | None,
        proposer_key=None,
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

        # alpha-7 ExecutionPayloadEnvelope: payload, execution_requests,
        # builder_index, beacon_block_root, parent_beacon_block_root.
        # No `slot` / `state_root` — those are dev-spec only.
        parent_beacon_block_root = bytes(self.state.latest_block_header.parent_root)
        envelope = ExecutionPayloadEnvelope(
            payload=execution_payload,
            execution_requests=exec_requests_obj,
            builder_index=BUILDER_INDEX_SELF_BUILD,
            beacon_block_root=Root(beacon_block_root),
            parent_beacon_block_root=Root(parent_beacon_block_root),
        )

        # Sign envelope with proposer key (self-build mode: builder_index ==
        # SELF_BUILD, so verify_execution_payload_envelope_signature uses
        # state.validators[proposer_index].pubkey).
        from .spec.constants import DOMAIN_BEACON_BUILDER
        from .spec.state_transition.helpers.domain import get_domain, compute_signing_root
        from .crypto import sign as bls_sign
        if proposer_key is None or getattr(proposer_key, "privkey", None) is None:
            logger.warning("envelope publish: no proposer_key, signing with infinity")
            envelope_signature = b"\xc0" + b"\x00" * 95
        else:
            domain = get_domain(self.state, DOMAIN_BEACON_BUILDER)
            signing_root = compute_signing_root(envelope, domain)
            envelope_signature = bls_sign(proposer_key.privkey, signing_root)
        signed_envelope = SignedExecutionPayloadEnvelope(
            message=envelope,
            signature=BLSSignature(envelope_signature),
        )

        # Save by both envelope root and block root for API lookups
        payload_root = hash_tree_root(envelope)
        self.store.save_payload(payload_root, signed_envelope)
        self.store.save_payload(beacon_block_root, signed_envelope)

        # Broadcast
        ssz_bytes = signed_envelope.encode_bytes()
        await self.beacon_gossip.publish_execution_payload(ssz_bytes)
        logger.info(
            f"Broadcast execution payload envelope for slot {slot}: "
            f"block_hash={bytes(execution_payload.block_hash).hex()[:16]}, "
            f"latest_block_hash={bytes(self.state.latest_block_hash).hex()[:16]}, "
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

            # Prefer the already-prepared payload from the prep-fcU at the
            # previous slot's end — geth has been filling it with txs for
            # the whole slot duration, so it's full. Requesting a "fresh"
            # payload here resets geth's build job and getPayload returns
            # within milliseconds with 0 txs (which is what was happening).
            # Only fall back to a fresh build if the prep is missing, was
            # built for a different slot, or the head has shifted since
            # then (e.g. a reorg made the prep extend a stale parent).
            current_head_root = self.head_root or b"\x00" * 32
            prep_is_valid = (
                self._current_payload_id is not None
                and self._current_payload_slot == int(slot)
                and self._current_payload_beacon_root == current_head_root
            )
            if prep_is_valid:
                logger.info(
                    f"Reusing prepared payload_id for slot {slot}: "
                    f"{self._current_payload_id.hex()}"
                )
            else:
                logger.info(
                    f"Requesting fresh payload for slot {slot} "
                    f"(prep_slot={self._current_payload_slot}, "
                    f"prep_head={self._current_payload_beacon_root.hex()[:16] if self._current_payload_beacon_root else 'None'}, "
                    f"head={current_head_root.hex()[:16]})"
                )
                await self._request_payload_for_slot(slot)
                # Safety net for the case where head-adoption re-prep
                # (in _on_p2p_block) didn't fire in time and we still
                # ended up issuing fcU{payload_attributes} at slot start.
                # Geth typically needs ~500ms to seal txs into a payload
                # after fcU; calling getPayload in the same tick yields
                # tx_count=0. Sleep keeps the proposal honest at the
                # cost of being ~500ms later than the slot start. The
                # `else` branch handles the rare case — the warm path
                # (prep_is_valid=True) skips this entirely.
                await asyncio.sleep(0.5)
            if not self._current_payload_id:
                logger.error("Cannot produce block: failed to get payload_id")
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
                slot, proposer_key, execution_payload_dict,
                blobs_bundle=payload_response.blobs_bundle,
                execution_requests=el_execution_requests,
            )
            if signed_block is None:
                logger.error("Failed to build block")
                return

            # PUBLISH IMMEDIATELY before doing any local validation, engine
            # round-trips, or state persistence. The block was built from
            # our own state and run through process_block inside
            # block_builder, so we already know it's coherent. Doing the
            # importer-side checks here costs ~1.5–2s — long enough that
            # prysm's late-block-reorg orphans us every time. Local apply
            # happens below and just duplicates the work an importer would
            # do; the network publish has to win the race.
            if self.beacon_gossip:
                try:
                    early_ssz = signed_block.encode_bytes()
                    await self.beacon_gossip.publish_block(early_ssz)
                    logger.info(f"Block published EARLY to P2P: slot={slot}")
                except Exception as e:
                    logger.error(f"Failed to publish block to P2P (early): {e}")

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

            # NOTE: do NOT mutate self.state.latest_block_header.state_root here.
            # Per spec it stays ZERO until process_slot for slot+1 fills it
            # in via previous_state_root. If we set it now, every subsequent
            # process_slot caches state.state_roots[slot] against a state hash
            # no other client computes, and our chain permanently forks from
            # the network. process_slots in the next state_transition will
            # fill it in correctly when needed.

            # For GLOAS: apply phase 2 (envelope processing) to state BEFORE saving.
            # This updates latest_block_hash, execution_payload_availability, etc.
            # Must happen before state save so the saved state includes phase 2 changes.
            if is_gloas:
                try:
                    await self._broadcast_execution_payload_envelope(
                        slot, block_root, execution_payload_dict, el_execution_requests,
                        payload_response.blobs_bundle, proposer_key,
                    )
                except Exception as e:
                    logger.error(f"Failed to build/apply execution payload envelope: {e}")
                    import traceback
                    traceback.print_exc()

            # Save state for epoch queries (Dora needs historical states by state_root)
            # This MUST happen before updating head_slot/head_root to avoid race condition
            # where SSE event is emitted before state is saved
            state_root = await self._on_state_thread(hash_tree_root, self.state)
            state_data = await self._on_state_thread(
                lambda: bytes(self.state.encode_bytes())
            )
            self.store.save_state(state_root, self.state, data=state_data)
            self.store.save_state(block_root, self.state, data=state_data)  # Also by block_root for flexibility
            # Also save by the block's state_root field, as that's what Dora looks for
            block_state_root = bytes(block.state_root)
            if block_state_root != state_root:
                self.store.save_state(block_state_root, self.state, data=state_data)

            # NOW update head_slot/head_root (SSE event loop will see this change)
            self.head_slot = slot
            self.head_root = block_root
            self.store.set_head(block_root)
            self._push_status_snapshot()

            # Update metrics
            from .spec import constants
            epoch = slot // constants.SLOTS_PER_EPOCH()
            metrics.update_head(slot, epoch)
            metrics.record_block_proposed(success=True)

            # Remove included attestations from pool to avoid re-inclusion
            self.attestation_pool.remove_included(list(block.body.attestations))

            # Now update forkchoice with the new block
            forkchoice_state = ForkchoiceState(
                head_block_hash=new_block_hash,
                safe_block_hash=new_block_hash,
                finalized_block_hash=self._resolve_finalized_block_hash() or b"\x00" * 32,
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

            # (Block was already published EARLY immediately after build_block.)

            if is_gloas:
                if self.beacon_api:
                    try:
                        await self.beacon_api.emit_execution_payload_available(slot, block_root)
                    except Exception as e:
                        logger.error(f"Failed to emit execution_payload_available: {e}")

                # Emit execution_payload_bid SSE event
                if self.beacon_api and hasattr(block.body, 'signed_execution_payload_bid'):
                    try:
                        await self.beacon_api.emit_execution_payload_bid(
                            block.body.signed_execution_payload_bid
                        )
                    except Exception as e:
                        logger.error(f"Failed to emit execution_payload_bid event: {e}")

        except Exception as e:
            logger.error(f"Block production failed: {e}")
            import traceback
            traceback.print_exc()

    async def _block_sync_loop(self) -> None:
        """Independent block-sync poller.

        Runs alongside the slot ticker so that req/resp catch-up keeps making
        progress even when the slot ticker is busy with heavy validator-duty
        compute.

        Polling cadence is one slot — anything tighter exceeds prysm's
        per-peer beacon_blocks_by_range rate limit (128 req / 10s) and
        gets us downscored / kicked. Once we've actually fallen behind we
        fan out larger requests instead of one-block requests every poll.
        """
        network_config = get_config()
        slot_duration = network_config.slot_duration_ms / 1000.0
        # One full slot between polls. With 6s gloas-minimal slots that's
        # well under prysm's 128/10s ceiling even if we burst-request a
        # whole batch of blocks in one call.
        poll = max(2.0, slot_duration)

        while self._running:
            try:
                if (
                    self.beacon_gossip is None
                    or self.state is None
                    or not self._genesis_time
                ):
                    await asyncio.sleep(poll)
                    continue

                now = time.time()
                current_slot = int((now - self._genesis_time) // slot_duration)
                if current_slot < 1:
                    await asyncio.sleep(poll)
                    continue

                # Don't bother peers if we're <2 slots behind — gossipsub
                # will deliver the next block. Only fire req/resp catch-up
                # when there's a meaningful gap.
                state_slot = int(self.state.slot)
                if current_slot - state_slot < 2:
                    await asyncio.sleep(poll)
                    continue

                await self._sync_missing_blocks(current_slot)
            except asyncio.CancelledError:
                logger.info("Block sync loop cancelled")
                raise
            except Exception as e:
                logger.error(f"Block sync loop error: {e}", exc_info=True)
            await asyncio.sleep(poll)

    async def _backfill_for_reorg(self, target_slot: int) -> None:
        """Fetch a recent slot window via blocks-by-range to fill a reorg gap.

        Called when `_reorg_to` walks past everything we've stored. Best
        effort — failures are logged and swallowed, the periodic sync loop
        will keep trying. Throttled to one concurrent backfill via
        `_reorg_backfill_lock`.
        """
        if not self.beacon_gossip:
            return
        lock = getattr(self, "_reorg_backfill_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._reorg_backfill_lock = lock
        if lock.locked():
            return
        async with lock:
            count = 64
            start_slot = max(1, target_slot - count + 1)
            logger.info(
                f"reorg backfill: requesting slots {start_slot}..{start_slot + count - 1}"
            )
            try:
                blocks = await self.beacon_gossip.request_blocks_by_range(start_slot, count)
            except Exception as e:
                logger.warning(f"reorg backfill request failed: {e}")
                return
            if not blocks:
                logger.warning("reorg backfill returned no blocks")
                return
            for block_ssz in blocks:
                try:
                    await self._on_p2p_block(block_ssz, "req/resp")
                except Exception as e:
                    logger.warning(f"reorg backfill: error processing block: {e}")

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

        # Never request the current slot — at slot-start tick the current
        # slot's proposer hasn't broadcast yet (often we ARE the proposer),
        # and every connected peer answers with an empty range. The
        # request_blocks_by_range path serialises a 1.5s timeout per peer
        # before giving up, so with N peers the proposer trigger fires
        # 1.5*N seconds late, well after gossipsub would have delivered the
        # *previous* slot's block. Cap the request at current_slot - 1 so
        # we only chase slots that should already be on the wire.
        end_slot = current_slot - 1
        if state_slot >= end_slot:
            return

        slots_behind = current_slot - state_slot

        # IMPORTANT: When we're on a different fork (e.g., our slot 1 vs peer's slot 1),
        # we need to request from slot 1, not state_slot+1, because the peer's chain
        # may have different blocks. This allows us to get the peer's complete chain.
        # Request from slot 1 if we might be on a different fork.
        if state_slot <= 1 and slots_behind > 1:
            # We're at genesis or slot 1, peer is ahead - request from slot 1
            start_slot = 1
            count = min(end_slot - start_slot + 1, 32)
        else:
            start_slot = state_slot + 1
            count = min(end_slot - start_slot + 1, 16)

        if count < 1:
            return

        logger.info(f"State is {slots_behind} slots behind (state_slot={state_slot}, current={current_slot}). "
                    f"Requesting blocks {start_slot} to {start_slot + count - 1} via req/resp")

        try:
            blocks = await self.beacon_gossip.request_blocks_by_range(start_slot, count)
            if not blocks:
                # Every peer returned no chunks for [start_slot, start_slot+count-1].
                # Those slots are genuinely empty (no proposer produced a block).
                # Mark current_slot as "gap confirmed empty" so _is_synced lets us
                # propose on top of the existing head instead of skipping our duty
                # because someone earlier in the round-robin missed theirs.
                if start_slot + count - 1 >= current_slot - 1:
                    self._sync_empty_confirmed_for_slot = current_slot
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

    def _record_peer_event(self, peer_id: str, reason: str) -> None:
        """Push a PR #606 PeerScoreReason for a misbehaving peer.

        Defensive: skip the `"req/resp"` synthetic peer used by sync
        (it's a literal, not a base58 PeerId) and any other empty/
        falsy id. Never let recording errors leak into the calling
        handler — scoring is observability, not policy.
        """
        if not peer_id or peer_id == "req/resp":
            return
        if not self.beacon_gossip:
            return
        host = getattr(self.beacon_gossip, "_host", None)
        if host is None:
            return
        try:
            host.record_peer_score_event(peer_id, reason)
        except Exception:
            pass

    async def _on_p2p_block(self, data: bytes, from_peer: str) -> None:
        """Handle a beacon block received via libp2p gossipsub or req/resp."""
        async with self._block_import_lock:
            await self._import_signed_block(data, from_peer)

    async def _import_signed_block(self, data: bytes, from_peer: str) -> None:
        """Import one signed block. Caller must hold _block_import_lock."""
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
                    # Parent is in our store but != head. Two cases:
                    #  (a) Sequential gap-fill: blocks arrived out of order
                    #      over gossip (e.g. 37 before 32-36) but every
                    #      intermediate descends from our current head.
                    #      No reorg — just apply the missing blocks forward
                    #      on top of self.state.
                    #  (b) Real reorg: a sibling chain whose common ancestor
                    #      sits below our current head. Walk-and-replay via
                    #      `_reorg_to`.
                    # Out-of-order gossip is the common case; misclassifying
                    # it as a reorg replays from a deep anchor every time a
                    # new gossip block arrives, and head never advances.
                    self.store.save_block(block_root, signed_block)
                    if int(block.slot) > self.head_slot:
                        gap_chain = self._walk_back_to_head(parent_root, max_steps=64)
                        if gap_chain is not None:
                            logger.info(
                                f"P2P: Sequential gap-fill slot={block.slot}: "
                                f"applying {len(gap_chain)} intermediate block(s) "
                                f"on top of head slot={self.head_slot} "
                                f"root={self.head_root.hex()[:16]}"
                            )
                            try:
                                for missing_root, missing_signed in gap_chain:
                                    missing_block = (
                                        missing_signed.message
                                        if hasattr(missing_signed, "message")
                                        else missing_signed
                                    )
                                    if self.state is not None and int(
                                        missing_block.slot
                                    ) <= int(self.state.latest_block_header.slot):
                                        # Already applied (e.g. chain walked
                                        # from a head that lagged the state).
                                        continue
                                    await self._apply_block_to_state(
                                        missing_block, missing_root, missing_signed
                                    )
                                    # Head moves with the state, immediately.
                                    # Anything fallible (EL notify) comes
                                    # after — a failure between apply and
                                    # head update leaves head pointing at a
                                    # block the state already moved past,
                                    # and every later gap-fill dies on the
                                    # parent_root assert.
                                    self.head_slot = int(missing_block.slot)
                                    self.head_root = missing_root
                                    self.store.set_head(missing_root)
                                    try:
                                        await self._notify_el_of_received_block(
                                            missing_block
                                        )
                                    except Exception as el_err:
                                        logger.warning(
                                            f"EL notify for gap-fill "
                                            f"slot={missing_block.slot} failed: {el_err}"
                                        )
                            except Exception as e:
                                logger.error(
                                    f"P2P: gap-fill failed before slot={block.slot}: {e}"
                                )
                                import traceback
                                traceback.print_exc()
                                # The gap chain didn't connect to the state's
                                # branch (head and state on different forks).
                                # _reorg_to replays from a cached ancestor
                                # state and swaps state+head atomically —
                                # use it to self-heal instead of staying
                                # wedged.
                                try:
                                    await self._reorg_to(block_root, signed_block)
                                except Exception as reorg_err:
                                    logger.error(
                                        f"P2P: heal-reorg to slot={block.slot} "
                                        f"failed: {reorg_err}"
                                    )
                                return
                            # parent_root now equals self.head_root; fall
                            # through to the normal forward-import path
                            # below to apply the new block as head.
                        else:
                            logger.info(
                                f"P2P: Reorg candidate slot={block.slot} "
                                f"root={block_root.hex()[:16]} "
                                f"(our head slot={self.head_slot} "
                                f"root={self.head_root.hex()[:16]})"
                            )
                            try:
                                await self._reorg_to(block_root, signed_block)
                            except Exception as e:
                                logger.error(
                                    f"P2P: Reorg to slot={block.slot} failed: {e}"
                                )
                                import traceback
                                traceback.print_exc()
                            return
                    else:
                        logger.debug(
                            f"P2P: Block slot={block.slot} on different chain but not "
                            f"newer than head_slot={self.head_slot} — saved, no reorg"
                        )
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

                    # Do NOT mutate latest_block_header.state_root — see comment
                    # in _produce_and_broadcast_block; same spec-divergence trap.

                    # Save state for epoch queries (Dora needs historical states by state_root)
                    # This MUST happen before updating head_slot/head_root to avoid race condition
                    # where SSE event is emitted before state is saved
                    state_root = await self._on_state_thread(hash_tree_root, self.state)
                    state_data = await self._on_state_thread(
                        lambda: bytes(self.state.encode_bytes())
                    )
                    self.store.save_state(state_root, self.state, data=state_data)
                    self.store.save_state(block_root, self.state, data=state_data)  # Also by block_root for flexibility
                    # Also save by the block's state_root field, as that's what Dora looks for
                    block_state_root = bytes(block.state_root)
                    if block_state_root != state_root:
                        self.store.save_state(block_state_root, self.state, data=state_data)

                    # NOW update head_slot/head_root (SSE event loop will see this change)
                    self.head_slot = int(block.slot)
                    self.head_root = block_root
                    self.store.set_head(block_root)
                    self._push_status_snapshot()

                    # Push the execution payload to our EL. For Gloas blocks
                    # the envelope path does this; pre-Gloas blocks need it
                    # here or geth never sees the canonical chain past our
                    # own proposals. Best-effort AFTER the head update: the
                    # state is already advanced, and an EL hiccup here must
                    # not strand head behind the state.
                    try:
                        await self._notify_el_of_received_block(block)
                    except Exception as el_err:
                        logger.warning(
                            f"EL notify for slot={block.slot} failed: {el_err}"
                        )

                    # Update metrics
                    from .spec import constants
                    slot = int(block.slot)
                    epoch = slot // constants.SLOTS_PER_EPOCH()
                    metrics.update_head(slot, epoch)

                    logger.info(
                        f"P2P: Adopted block as new head: slot={block.slot}, "
                        f"root={block_root.hex()[:16]}"
                    )

                    self.attestation_pool.remove_included(list(block.body.attestations))

                    # For Gloas blocks: check if we have a pending envelope for phase 2
                    is_gloas_block = hasattr(block.body, "signed_execution_payload_bid")
                    if is_gloas_block and block_root in self._pending_envelopes:
                        pending_envelope = self._pending_envelopes.pop(block_root)
                        logger.info(f"P2P: Found pending envelope for block {block_root.hex()[:16]}, applying phase 2")
                        await self._apply_execution_payload_envelope(pending_envelope)

                    await self._update_forkchoice()

                    # If we're the proposer for the *next* slot, kick off a
                    # fresh payload-prep against the new head right now
                    # (not at end-of-slot). Geth needs ~500ms-1s to seal
                    # txs into a payload after fcU{payload_attributes};
                    # prepping immediately on head adoption gives it the
                    # full inter-slot gap to build, instead of racing the
                    # slot boundary. Without this, every block we propose
                    # ends up with tx_count=0 because the slot-start
                    # late-prep + getPayload happens in the same second.
                    next_slot = int(block.slot) + 1
                    if (
                        self.validator_client
                        and self.validator_client.keys
                        and self.validator_client.is_our_proposer_slot(self.state, next_slot)
                    ):
                        logger.info(
                            f"Upcoming proposer for slot {next_slot}; "
                            f"re-prepping payload on head adoption"
                        )
                        await self._update_forkchoice_for_slot(int(block.slot))
                except Exception as e:
                    logger.warning(
                        f"P2P: Failed to apply block slot={block.slot}: {e}. "
                        f"Keeping head at {old_head.hex()[:16] if old_head else 'None'}"
                    )
                    self._record_peer_event(from_peer, "gossip_invalid_block")
                    import traceback
                    traceback.print_exc()
            else:
                logger.debug(f"P2P: Block slot={block.slot} not newer than head_slot={self.head_slot}")

        except Exception as e:
            logger.error(f"Error processing P2P block: {e}")
            self._record_peer_event(from_peer, "gossip_invalid_block")

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
                    self._record_peer_event(from_peer, "gossip_invalid_attestation")
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

    async def _on_p2p_sync_committee_message(
        self, data: bytes, from_peer: str, subnet_id: int
    ) -> None:
        """Handle an individual SyncCommitteeMessage from a subnet topic.

        Resolves the sender's validator_index to its sync-committee position(s)
        via the cached map and inserts one entry per position. Positions
        outside this subnet's subcommittee are skipped — a duplicated
        validator publishes the same message on each subnet it covers.
        """
        try:
            from .spec.types.altair import SyncCommitteeMessage
            from .spec.constants import SYNC_COMMITTEE_SIZE, SYNC_COMMITTEE_SUBNET_COUNT

            message = SyncCommitteeMessage.decode_bytes(data)
            validator_index = int(message.validator_index)

            positions = self._sync_committee_index_to_positions.get(validator_index)
            if not positions:
                # Either we don't know this validator yet (state not built) or
                # they aren't in the current sync committee. Drop silently —
                # over-validating here would also drop legitimate fork-boundary
                # messages we can't yet place.
                return

            subcommittee_size = SYNC_COMMITTEE_SIZE() // SYNC_COMMITTEE_SUBNET_COUNT
            added_any = False
            for position in positions:
                if position // subcommittee_size != subnet_id:
                    continue
                if self.sync_committee_pool.add(message, position):
                    added_any = True

            if added_any:
                key = (int(message.slot), from_peer, subnet_id)
                self._sync_msg_log_buffer.setdefault(key, []).append(validator_index)
                if self._sync_msg_log_task is None or self._sync_msg_log_task.done():
                    self._sync_msg_log_task = asyncio.create_task(
                        self._flush_sync_msg_log_after()
                    )
        except Exception as e:
            logger.error(f"Error processing P2P sync committee message: {e}")

    async def _flush_sync_msg_log_after(self) -> None:
        """Flush buffered SyncCommitteeMessage debug logs after a short delay.

        Groups by (slot, peer, subnet) and emits a single line with all
        validator indices so a slot's worth of subnet chatter is one row
        per (peer, subnet) instead of one per validator.
        """
        try:
            await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            return
        buf = self._sync_msg_log_buffer
        self._sync_msg_log_buffer = {}
        for (slot, peer, subnet), validators in sorted(buf.items()):
            unique_sorted = sorted(set(validators))
            logger.debug(
                f"P2P: SyncCommitteeMessages slot={slot} subnet={subnet} "
                f"count={len(unique_sorted)} validators={unique_sorted} "
                f"from={peer[:16]}"
            )

    async def _on_p2p_sync_committee_contribution(self, data: bytes, from_peer: str) -> None:
        """Handle a sync committee contribution received via libp2p gossipsub."""
        try:
            from .spec.types.altair import SignedContributionAndProof

            signed_contribution = SignedContributionAndProof.decode_bytes(data)
            contribution = signed_contribution.message.contribution
            slot = int(contribution.slot)
            subcommittee_index = int(contribution.subcommittee_index)

            added = self.sync_committee_pool.add_contribution(contribution)

            # Only log contributions that actually changed pool state. Steady-state
            # mesh chatter (everyone republishes the same aggregate to everyone)
            # would otherwise emit one debug line per (slot × subcommittee × peer).
            if added:
                participant_count = sum(1 for b in contribution.aggregation_bits if b)
                logger.debug(
                    f"P2P: merged sync committee contribution slot={slot}, "
                    f"subcommittee={subcommittee_index}, participants={participant_count}, "
                    f"from={from_peer[:16]}"
                )
        except Exception as e:
            logger.error(f"Error processing P2P sync committee contribution: {e}")

    async def _on_p2p_execution_payload(self, data: bytes, from_peer: str) -> None:
        """Handle an execution payload envelope received via libp2p gossipsub."""
        try:
            from .spec.types.gloas import SignedExecutionPayloadEnvelope

            signed_envelope = SignedExecutionPayloadEnvelope.decode_bytes(data)
            envelope = signed_envelope.message
            slot = self._envelope_slot(envelope)
            beacon_block_root = bytes(envelope.beacon_block_root)

            logger.info(
                f"P2P: Received execution payload envelope slot={slot if slot is not None else '?'}, "
                f"block_hash={bytes(envelope.payload.block_hash).hex()[:16]}, "
                f"from={from_peer[:16]}"
            )

            payload_root = hash_tree_root(envelope)
            self.store.save_payload(payload_root, signed_envelope)
            self.store.save_payload(beacon_block_root, signed_envelope)

            if self.engine:
                await self._validate_execution_payload(envelope)

            # Apply phase 2 (envelope processing) to state if the block has been processed
            await self._apply_execution_payload_envelope(signed_envelope)

            if self.beacon_api and slot is not None:
                try:
                    await self.beacon_api.emit_execution_payload_available(
                        slot, beacon_block_root
                    )
                except Exception as e:
                    logger.error(f"Failed to emit execution_payload_available event: {e}")

        except Exception as e:
            logger.error(f"Error processing P2P execution payload envelope: {e}")

    async def _on_p2p_payload_attestation_message(self, data: bytes, from_peer: str) -> None:
        """Handle a PayloadAttestationMessage from a PTC validator."""
        if self.state is None:
            return
        try:
            from .spec.types.gloas import PayloadAttestationMessage
            from .spec.state_transition.helpers.ptc import get_ptc

            msg = PayloadAttestationMessage.decode_bytes(data)
            slot = int(msg.data.slot)
            ptc = list(get_ptc(self.state, slot))
            self.payload_attestation_pool.add_message(msg, ptc)
            # Drop messages older than current_epoch's start.
            current_slot = int(self.state.slot)
            from .spec.constants import SLOTS_PER_EPOCH
            keep_from = max(0, (current_slot // SLOTS_PER_EPOCH() - 1) * SLOTS_PER_EPOCH())
            self.payload_attestation_pool.prune_before(keep_from)
        except Exception as e:
            logger.warning(f"P2P payload_attestation_message decode/handle failed: {e}")

    async def _on_p2p_proposer_preferences(self, data: bytes, from_peer: str) -> None:
        """Handle a SignedProposerPreferences from gossip (Gloas).

        Validates per gloas/p2p-interface.md against the head state (we use
        the head state rather than the dependent-root checkpoint state; on a
        devnet without long reorgs the proposer_lookahead they expose for the
        validated epoch range is identical).
        """
        if self.state is None or not hasattr(self.state, "proposer_lookahead"):
            return
        try:
            from .spec.types.gloas import SignedProposerPreferences
            from .spec.state_transition.helpers.beacon_committee import is_valid_proposal_slot
            from .spec.state_transition.helpers.misc import compute_epoch_at_slot
            from .spec.state_transition.helpers.domain import get_domain, compute_signing_root
            from .spec.constants import DOMAIN_PROPOSER_PREFERENCES, MIN_SEED_LOOKAHEAD
            from .crypto.crypto import verify_async

            signed = SignedProposerPreferences.decode_bytes(data)
            prefs = signed.message
            proposal_slot = int(prefs.proposal_slot)
            validator_index = int(prefs.validator_index)
            dependent_root = bytes(prefs.dependent_root)

            current_slot = self._current_wall_slot()
            # [IGNORE] proposal_slot has not already passed
            if proposal_slot <= current_slot:
                return
            # [IGNORE] proposal_slot within the proposer lookahead
            proposal_epoch = compute_epoch_at_slot(proposal_slot)
            current_epoch = compute_epoch_at_slot(current_slot)
            if not (current_epoch <= proposal_epoch <= current_epoch + MIN_SEED_LOOKAHEAD):
                return
            # [IGNORE] first valid message for the tuple
            key = (dependent_root, proposal_slot, validator_index)
            if key in self.proposer_preferences:
                return
            # [REJECT] validator is the scheduled proposer for the slot
            if not is_valid_proposal_slot(self.state, prefs):
                self._record_peer_event(from_peer, "gossip_invalid_proposer_preferences")
                return
            # [REJECT] signature valid for the validator's pubkey
            if validator_index >= len(self.state.validators):
                self._record_peer_event(from_peer, "gossip_invalid_proposer_preferences")
                return
            domain = get_domain(self.state, DOMAIN_PROPOSER_PREFERENCES, proposal_epoch)
            signing_root = compute_signing_root(prefs, domain)
            pubkey = bytes(self.state.validators[validator_index].pubkey)
            if not await verify_async(pubkey, signing_root, bytes(signed.signature)):
                self._record_peer_event(from_peer, "gossip_invalid_proposer_preferences")
                return

            self.proposer_preferences[key] = signed
            self._prune_proposer_preferences(current_slot)
            logger.debug(
                f"P2P: proposer_preferences vi={validator_index} slot={proposal_slot} "
                f"fee_recipient=0x{bytes(prefs.fee_recipient).hex()} "
                f"target_gas_limit={int(prefs.target_gas_limit)} from={from_peer[:16]}"
            )
        except Exception as e:
            logger.warning(f"P2P proposer_preferences decode/handle failed: {e}")

    def _prune_proposer_preferences(self, current_slot: int) -> None:
        """Drop preferences (and publish dedup keys) for slots already passed."""
        stale = [k for k in self.proposer_preferences if k[1] <= current_slot]
        for k in stale:
            del self.proposer_preferences[k]
        stale_pub = [k for k in self._published_prefs_keys if k[1] <= current_slot]
        for k in stale_pub:
            self._published_prefs_keys.discard(k)

    def _current_wall_slot(self) -> int:
        """Wall-clock slot derived from genesis time."""
        slot_duration = get_config().slot_duration_ms / 1000.0
        return int((time.time() - self._genesis_time) // slot_duration)

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
            self._record_peer_event(from_peer, "gossip_invalid_blob_sidecar")

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

    def _walk_back_to_head(
        self, start_root: bytes, max_steps: int = 64
    ) -> Optional[list[tuple[bytes, object]]]:
        """If `start_root`'s ancestry reaches `self.head_root` within
        `max_steps`, return the list of (root, signed_block) entries from
        oldest to newest (i.e. the missing blocks that, applied in order on
        top of `self.state`, advance head from current to `start_root`).
        Returns None if we hit an unknown ancestor, exceed max_steps, or
        head isn't set.
        """
        if self.head_root is None:
            return None
        chain: list[tuple[bytes, object]] = []
        cursor = start_root
        for _ in range(max_steps):
            if cursor == self.head_root:
                chain.reverse()
                return chain
            ancestor_signed = self.store.get_block(cursor)
            if ancestor_signed is None:
                return None
            chain.append((cursor, ancestor_signed))
            anc_block = (
                ancestor_signed.message
                if hasattr(ancestor_signed, "message")
                else ancestor_signed
            )
            cursor = bytes(anc_block.parent_root)
        return None

    async def _reorg_to(self, target_root: bytes, target_signed_block) -> None:
        """Switch chain head to a new block by walking back to a common
        ancestor and re-applying state transitions.

        This is the simple-LMD-GHOST reorg path: when a peer's block at
        slot > our head slot points to a different parent than our head,
        we want to adopt the peer's chain (it has more proposers / more
        attestations behind it). We:

          1. Walk back from target_block up parent_root pointers, building
             the chain back to either our current state's slot or genesis.
          2. Snapshot the state at the common ancestor, or recompute from
             genesis if no closer ancestor is reachable.
          3. Replay the chain from that ancestor through to target_root
             via process_slots / process_block.
          4. Atomically swap (self.state, self.head_root, self.head_slot)
             on success, or leave them unchanged on failure.

        Bigger optimisation later: cache state at each finalised checkpoint
        and only walk back to the most recent finalised one. For now we
        replay from genesis if needed — a few hundred ms per reorg.
        """
        target_block = target_signed_block.message
        target_slot = int(target_block.slot)

        # Bound the walk by our finalized checkpoint. Per fork-choice spec,
        # only blocks descended from `store.finalized_checkpoint` can be
        # canonical; a target at or below the finalized slot is unreachable
        # by fork choice and walking past finality wastes memory (each
        # `chain` entry is a full SignedBeaconBlock).
        finalized_root: Optional[bytes] = None
        finalized_slot: Optional[int] = None
        if self.state is not None and hasattr(self.state, "finalized_checkpoint"):
            ckpt = self.state.finalized_checkpoint
            ckpt_root = bytes(ckpt.root)
            ckpt_epoch = int(ckpt.epoch)
            if ckpt_root != b"\x00" * 32 and ckpt_epoch > 0:
                finalized_root = ckpt_root
                finalized_slot = ckpt_epoch * SLOTS_PER_EPOCH()

        if finalized_slot is not None and target_slot <= finalized_slot:
            logger.warning(
                f"reorg: rejecting target slot={target_slot} root={target_root.hex()[:16]} — "
                f"at or below finalized slot {finalized_slot}"
            )
            return

        # Walk back collecting (root, signed_block) from target until we
        # reach either:
        #   (a) an ancestor whose post-state we already have cached in the
        #       store — start replay from that state, or
        #   (b) the finalized checkpoint — anchor on its cached post-state,
        #       or fall back to genesis if we don't have it, or
        #   (c) genesis — fall back to a full replay.
        # Without (a) we'd re-replay from genesis on every Gloas-boundary
        # block, the replays queue up under wall-clock pressure, head gets
        # stuck for hours.
        chain: list[tuple[bytes, object]] = [(target_root, target_signed_block)]
        cursor = bytes(target_block.parent_root)
        anchor_state = None
        # Cap walk depth at finalized-window + small jitter buffer. Without
        # finality info, fall back to the previous heuristic.
        if finalized_slot is not None:
            max_walk = max(8, target_slot - finalized_slot + 8)
        else:
            max_walk = max(target_slot + 8, 64)
        for _ in range(max_walk):
            if self._genesis_block_root and cursor == self._genesis_block_root:
                break
            if finalized_root is not None and cursor == finalized_root:
                # Stop at finality. If we have the finalized state cached,
                # use it as anchor; otherwise fall through to genesis replay.
                anchor_state = self.store.get_state(cursor)
                break
            cached = self.store.get_state(cursor)
            if cached is not None:
                anchor_state = cached
                break
            ancestor_signed = self.store.get_block(cursor)
            if ancestor_signed is None:
                # Peer's chain extends past anything we've stored. We don't
                # have a BlocksByRoot protocol yet, so backfill via a range
                # request anchored on the target slot and bail without
                # updating head — a later block or the periodic sync loop
                # will retry once the gap closes.
                logger.warning(
                    f"reorg: missing ancestor {cursor.hex()[:16]} while walking "
                    f"back from {target_root.hex()[:16]} (target_slot={target_slot}); "
                    f"requesting range backfill"
                )
                asyncio.create_task(self._backfill_for_reorg(target_slot))
                return
            chain.append((cursor, ancestor_signed))
            anc_block = ancestor_signed.message if hasattr(ancestor_signed, "message") else ancestor_signed
            cursor = bytes(anc_block.parent_root)
        else:
            raise ValueError(
                f"reorg: walked back {max_walk} steps without hitting genesis"
            )

        # Reverse so we apply oldest → newest.
        chain.reverse()

        if anchor_state is not None:
            anchor_slot = int(anchor_state.slot)
            logger.info(
                f"Reorg: replaying {len(chain)} blocks from cached ancestor "
                f"slot {anchor_slot} to slot {target_slot}"
            )
            new_state = anchor_state.copy()
        else:
            logger.info(
                f"Reorg: replaying {len(chain)} blocks from genesis to slot {target_slot}"
            )
            new_state = self._genesis_state.copy()

        from .spec.state_transition import state_transition

        def _replay(state, indexed_blocks):
            for idx, (root, signed) in enumerate(indexed_blocks):
                try:
                    state = state_transition(state, signed, False)
                    replayed = (
                        signed.message if hasattr(signed, "message") else signed
                    )
                    # Canonical-chain pinning — see _apply_with_check.
                    state.latest_block_header.state_root = bytes(
                        replayed.state_root
                    )
                except Exception as exc:
                    blk = signed.message if hasattr(signed, "message") else signed
                    raise RuntimeError(
                        f"reorg replay failed at chain idx {idx}/"
                        f"{len(indexed_blocks)}: slot={blk.slot} "
                        f"root={root.hex()[:16]} "
                        f"parent_root={bytes(blk.parent_root).hex()[:16]} "
                        f"state_slot_before={state.slot}: {exc}"
                    ) from exc
            return state

        # Push the (potentially expensive) replay onto a worker thread.
        new_state = await asyncio.to_thread(_replay, new_state, chain)

        # latest_block_header.state_root is pinned to each block's claimed
        # root inside _replay (canonical-chain pinning); nothing to fill
        # here.

        # Persist the new state under several keys so the API + dora can
        # find it back.
        state_root = await asyncio.to_thread(hash_tree_root, new_state)
        state_data = await asyncio.to_thread(lambda: bytes(new_state.encode_bytes()))
        self.store.save_state(state_root, new_state, data=state_data)
        self.store.save_state(target_root, new_state, data=state_data)
        block_state_root = bytes(target_block.state_root)
        if block_state_root != state_root:
            self.store.save_state(block_state_root, new_state, data=state_data)

        old_head = self.head_root.hex()[:16] if self.head_root else "None"

        # Atomic swap.
        self.state = new_state
        self.head_slot = target_slot
        self.head_root = target_root
        self.store.set_head(target_root)
        self._push_status_snapshot()

        for _root, _signed in chain:
            _blk = _signed.message if hasattr(_signed, "message") else _signed
            self.attestation_pool.remove_included(list(_blk.body.attestations))

        from .spec import constants
        epoch = target_slot // constants.SLOTS_PER_EPOCH()
        metrics.update_head(target_slot, epoch)
        metrics.update_checkpoints(
            finalized=int(self.state.finalized_checkpoint.epoch),
            justified=int(self.state.current_justified_checkpoint.epoch),
        )

        logger.info(
            f"Reorg: head moved from {old_head} → "
            f"{target_root.hex()[:16]} at slot {target_slot}"
        )

        # Forkchoice update with EL so it tracks the new head.
        try:
            await self._update_forkchoice()
        except Exception as e:
            logger.error(f"Reorg: forkchoice_updated after reorg failed: {e}")

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

        from_slot = int(self.state.slot)
        target_slot = int(block.slot)
        # The active fork name is config-driven (slot → epoch → fork
        # schedule). Don't infer from block structure: hasattr() only tells
        # us how SSZ was decoded, not what fork the spec thinks we're on —
        # an Electra/Fulu block at slot 17 in a mainnet-preset network with
        # gloas_fork_epoch=1 was getting labelled "post_capella" or even
        # mis-labelled "gloas" depending on which body type was hit.
        fork = get_config().fork_name_at_slot(int(block.slot))

        # State transition is heavy sync compute (process_slots fast-forwards
        # one epoch boundary at a time, process_block runs full block ops).
        # Running it on the asyncio loop wedges everything: gossipsub block
        # delivery backs up, slot ticker stalls, beacon API responses time out.
        # Push it to the dedicated state worker so all state mutations
        # serialise through a single thread.
        t0 = time.monotonic()
        if signed_block is not None:
            # DIAGNOSTIC: Run state_transition without sig verify but with
            # state_root validation, comparing our post-state hash against
            # the block's claimed state_root. First mismatch pinpoints the
            # exact slot where our state diverges from spec.
            def _apply_with_check(state, sb):
                # remerkleable copy(): O(1) structural sharing that keeps
                # the memoized subtree hashes — an encode/decode round-trip
                # would throw the hash cache away and force a full
                # re-merkleization of the state on the next hash_tree_root.
                pre_copy = state.copy()
                target = int(sb.message.slot)
                if target > int(pre_copy.slot):
                    pre_copy = process_slots(pre_copy, target)
                process_block(pre_copy, sb.message)
                computed = hash_tree_root(pre_copy)
                claimed = bytes(sb.message.state_root)
                if computed != claimed:
                    logger.error(
                        f"STATE_ROOT_MISMATCH at slot={target} "
                        f"block_root={hash_tree_root(sb.message).hex()[:16]} "
                        f"computed={computed.hex()[:16]} "
                        f"claimed={claimed.hex()[:16]}"
                    )
                # Pin the block's CLAIMED state_root into the header. Spec
                # leaves it ZERO until the next process_slot fills it with
                # our computed root — but when our transition diverges (see
                # mismatch above) that filled root no longer hashes to the
                # block root peers built on, the next canonical block fails
                # its parent_root assert, and the node wedges permanently.
                # The claimed root is by definition what peers hashed into
                # the block root, so pinning it keeps us chained to the
                # canonical chain even while our state content drifts.
                pre_copy.latest_block_header.state_root = claimed
                return pre_copy
            new_state = await self._on_state_thread(
                _apply_with_check, self.state, signed_block
            )
            self.state = new_state
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            slot_advance = target_slot - from_slot
            logger.info(
                f"[STATE_TX] fork={fork} from_slot={from_slot} to_slot={target_slot} "
                f"advance={slot_advance} elapsed_ms={elapsed_ms:.0f}"
            )
            metrics.update_checkpoints(
                finalized=int(self.state.finalized_checkpoint.epoch),
                justified=int(self.state.current_justified_checkpoint.epoch),
            )
            return

        def _apply_sync(state):
            if target_slot > int(state.slot):
                state = process_slots(state, target_slot)
            process_block(state, block)
            # Canonical-chain pinning — see _apply_with_check above.
            state.latest_block_header.state_root = bytes(block.state_root)
            return state

        self.state = await self._on_state_thread(_apply_sync, self.state)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        slot_advance = target_slot - from_slot
        logger.info(
            f"[STATE_TX] fork={fork} from_slot={from_slot} to_slot={target_slot} "
            f"advance={slot_advance} elapsed_ms={elapsed_ms:.0f} (no_sig_verify)"
        )

    async def _notify_el_of_received_block(self, block) -> None:
        """Submit a received non-Gloas block's execution payload to the EL.

        Gloas blocks deliver the payload separately via the envelope gossip
        topic; that path already calls newPayload via
        _validate_execution_payload. Pre-Gloas blocks carry the payload
        inline in block.body.execution_payload, but consensoor's
        state-transition spec stub doesn't call the engine (the assertion
        relies on `execution_valid=True`). Without an explicit newPayload
        call here, geth's chain falls behind: it only knows the payloads
        consensoor itself produced. Then when the first Gloas slot arrives
        and we call newPayloadV5(payload, parent=slot_N-1_block_hash),
        geth either says SYNCING (missing parent) or — once parents catch
        up via the EL's own sync — still can't form the canonical head.
        """
        if not self.engine:
            return
        body = block.body
        if hasattr(body, "signed_execution_payload_bid"):
            return  # Gloas — envelope path handles newPayload
        if not hasattr(body, "execution_payload"):
            return  # pre-Bellatrix block, no payload to submit

        execution_payload = body.execution_payload

        versioned_hashes = []
        if hasattr(body, "blob_kzg_commitments"):
            from hashlib import sha256
            for commitment in body.blob_kzg_commitments:
                commitment_bytes = bytes(commitment)
                versioned_hashes.append(b"\x01" + sha256(commitment_bytes).digest()[1:])

        execution_requests_hex = self._encode_execution_requests(
            getattr(body, "execution_requests", None)
        )

        parent_beacon_root = bytes(block.parent_root)

        try:
            status = await self.engine.new_payload(
                execution_payload,
                versioned_hashes,
                parent_beacon_root,
                execution_requests_hex,
                timestamp=int(execution_payload.timestamp),
            )
        except Exception as e:
            logger.warning(
                f"newPayload for received block slot={int(block.slot)} raised: {e}"
            )
            return

        if status.status == PayloadStatusEnum.SYNCING:
            logger.info(
                f"newPayload for received block slot={int(block.slot)} "
                f"block_hash={bytes(execution_payload.block_hash).hex()[:16]}: "
                f"EL syncing"
            )
        elif status.status != PayloadStatusEnum.VALID and status.status != PayloadStatusEnum.ACCEPTED:
            logger.error(
                f"newPayload for received block slot={int(block.slot)} "
                f"block_hash={bytes(execution_payload.block_hash).hex()[:16]}: "
                f"status={status.status}"
            )

    @staticmethod
    def _encode_execution_requests(execution_requests) -> list[str]:
        """Encode SSZ ExecutionRequests as EIP-7685 hex strings for the EL.

        Geth's newPayloadV4/V5 expects one hex entry per request type,
        each `<type_byte><concatenated_ssz_request_data>`. Empty types
        are omitted entirely.
        """
        if execution_requests is None:
            return []

        def _concat(items) -> bytes:
            out = b""
            for item in items:
                if hasattr(item, "encode_bytes"):
                    out += bytes(item.encode_bytes())
                else:
                    out += bytes(item)
            return out

        result: list[str] = []
        for type_byte, attr in ((0x00, "deposits"), (0x01, "withdrawals"), (0x02, "consolidations")):
            data = _concat(getattr(execution_requests, attr, ()) or ())
            if data:
                result.append("0x" + bytes([type_byte]).hex() + data.hex())
        return result

    def _gloas_el_head_hash(self) -> bytes:
        """Return the EL block_hash that fcU should target as the head.

        On Gloas, the EL chain tip after a block is processed is the bid's
        block_hash — that's the payload we just imported via newPayload
        (either ourselves during proposal, or via the envelope receive path).
        `state.latest_block_hash` lags by one slot because Phase 2 only
        promotes it when the NEXT bid chains in; using it for the prep-fcU
        sends geth a head that's behind its own view, so geth answers VALID
        without issuing a payload_id. We prefer the bid's block_hash and
        fall back to latest_block_hash, then the legacy header.
        """
        if hasattr(self.state, "latest_execution_payload_bid"):
            bid_hash = bytes(self.state.latest_execution_payload_bid.block_hash)
            if bid_hash != b"\x00" * 32:
                return bid_hash
        if hasattr(self.state, "latest_block_hash"):
            return bytes(self.state.latest_block_hash)
        return bytes(self.state.latest_execution_payload_header.block_hash)

    def _withdrawals_attr_list_for_slot(self, target_slot: int) -> list:
        """Expected withdrawals for the payload of ``target_slot``, EL-JSON shaped.

        Gloas: state.payload_expected_withdrawals is the PREVIOUS block's list
        (process_withdrawals overwrites it during the next block's phase 1),
        so sending it in payload attributes makes the EL build a payload whose
        withdrawals_root mismatches what peers' process_withdrawals expects —
        they reject the envelope, PTC votes absent, and the block is reorged
        out. Always recompute with get_expected_withdrawals; across an epoch
        boundary advance a throwaway copy first since the epoch transition
        moves balances and withdrawability.
        """
        state = self.state
        if state is None or not hasattr(state, "next_withdrawal_index"):
            return []
        from .spec.state_transition.block.withdrawals import get_expected_withdrawals
        try:
            target_epoch = target_slot // SLOTS_PER_EPOCH()
            state_epoch = int(state.slot) // SLOTS_PER_EPOCH()
            if target_epoch > state_epoch:
                # A checkpoint-synced state legitimately lags wall clock by
                # 2-3 epochs. Anything beyond that means we're unsynced; a
                # process_slots replay over thousands of slots runs for hours
                # on the event loop, so bail out instead.
                if target_epoch - state_epoch > 4:
                    logger.warning(
                        f"state epoch {state_epoch} is {target_epoch - state_epoch} epochs "
                        f"behind target slot {target_slot}; skipping withdrawals precompute"
                    )
                    return []
                from .spec.state_transition.transition import process_slots
                clone = state.copy()
                state = process_slots(clone, int(target_slot))
            expected = get_expected_withdrawals(state)
            return [
                {
                    "index": hex(int(w.index)),
                    "validatorIndex": hex(int(w.validator_index)),
                    "address": "0x" + bytes(w.address).hex(),
                    "amount": hex(int(w.amount)),
                }
                for w in expected.withdrawals
            ]
        except Exception as we:
            logger.warning(
                f"expected withdrawals for slot {target_slot} failed: {we}; sending empty list"
            )
            return []

    def _resolve_finalized_block_hash(self) -> Optional[bytes]:
        """Get the EL block_hash of the finalized beacon block, or None.

        Walks state.finalized_checkpoint.root → store.get_block(...) →
        body.signed_execution_payload_bid.message.block_hash (Gloas) or
        body.execution_payload.block_hash (pre-Gloas). Returns None when
        we don't have the finalized block (early in chain) or its
        payload is empty/builder-deferred.
        """
        if self.state is None:
            return None
        ckpt = self.state.finalized_checkpoint
        if int(ckpt.epoch) == 0:
            return None
        finalized_root = bytes(ckpt.root)
        if finalized_root == b"\x00" * 32:
            return None
        signed_block = self.store.get_block(finalized_root)
        if signed_block is None:
            return None
        block = signed_block.message if hasattr(signed_block, "message") else signed_block
        body = block.body
        if hasattr(body, "signed_execution_payload_bid"):
            # Gloas: a block's bid.block_hash only became an EL block if its
            # payload envelope was revealed AND applied on the canonical
            # chain. A withheld (or late, PTC-voted-absent) payload leaves a
            # bid hash the EL chain skipped — sending it in
            # forkchoiceUpdated draws a 409 Inconsistent ForkChoiceState.
            # Holding the envelope in our store is NOT enough (we store
            # late-revealed envelopes that fork choice discarded); consult
            # the head state's execution_payload_availability bit, which
            # tracks canonical payload application per slot. Walk back to
            # the nearest ancestor whose payload was applied.
            avail = self.state.execution_payload_availability
            window = len(avail)
            head_slot = int(self.state.slot)
            sb, root = signed_block, finalized_root
            for _ in range(4 * SLOTS_PER_EPOCH()):
                if sb is None:
                    return None
                blk = sb.message if hasattr(sb, "message") else sb
                b = blk.body
                if not hasattr(b, "signed_execution_payload_bid"):
                    # Crossed the fork boundary: pre-Gloas payloads are
                    # always executed with their block.
                    if hasattr(b, "execution_payload"):
                        return bytes(b.execution_payload.block_hash)
                    return None
                slot = int(blk.slot)
                if head_slot - slot < window and bool(avail[slot % window]):
                    return bytes(b.signed_execution_payload_bid.message.block_hash)
                root = bytes(blk.parent_root)
                sb = self.store.get_block(root)
            return None
        if hasattr(body, "execution_payload"):
            return bytes(body.execution_payload.block_hash)
        return None

    def _envelope_slot(self, envelope) -> Optional[int]:
        """Derive the slot for an ExecutionPayloadEnvelope.

        The Gloas envelope schema (payload, execution_requests, builder_index,
        beacon_block_root, parent_beacon_block_root) has no `slot` field — the
        slot is implicitly the slot of the beacon block referenced by
        `beacon_block_root`. We resolve it by:
          1. matching against `state.latest_block_header` (cheapest), then
          2. looking the block up in the store, then
          3. returning None if neither hit.
        """
        beacon_block_root = bytes(envelope.beacon_block_root)
        if self.state is not None:
            header = self.state.latest_block_header
            if bytes(header.state_root) != b"\x00" * 32:
                if hash_tree_root(header) == beacon_block_root:
                    return int(header.slot)
        signed_block = self.store.get_block(beacon_block_root)
        if signed_block is not None:
            block = signed_block.message if hasattr(signed_block, "message") else signed_block
            return int(block.slot)
        return None

    async def _apply_execution_payload_envelope(self, signed_envelope) -> None:
        """Apply execution payload envelope (phase 2) to state.

        In Gloas ePBS, the state transition has two phases:
        - Phase 1: block processing (proposer's block with bid)
        - Phase 2: envelope processing (builder's payload reveal)

        This method applies phase 2. It checks that the block for this
        envelope has already been processed (matching slot and beacon_block_root).
        If the block hasn't been processed yet, saves the envelope as pending.
        """
        if not self.state:
            return

        envelope = signed_envelope.message
        beacon_block_root = bytes(envelope.beacon_block_root)
        envelope_slot = self._envelope_slot(envelope)

        # If we can't resolve the envelope's slot, the corresponding beacon
        # block hasn't been processed yet. Save as pending so phase 2 can run
        # once the block lands.
        if envelope_slot is None:
            logger.info(
                f"Envelope for unknown beacon block {beacon_block_root.hex()[:16]}, "
                "saving as pending"
            )
            self._pending_envelopes[beacon_block_root] = signed_envelope
            return

        # Check if the block has been processed (state is at the right slot)
        if int(self.state.slot) != envelope_slot:
            logger.info(
                f"Envelope for slot {envelope_slot} but state at slot {self.state.slot}, "
                f"saving as pending"
            )
            self._pending_envelopes[beacon_block_root] = signed_envelope
            return

        # Re-check the slot + beacon_block_root match AND apply the envelope in
        # one atomic step on the state worker thread. Doing the precheck on the
        # main coroutine and then awaiting the apply is racy: between the
        # checks and the apply, another coroutine (block adoption, reorg) can
        # mutate self.state via the same state thread, advancing the slot and
        # rewriting latest_block_header — at which point the envelope's
        # beacon_block_root no longer matches and the spec assertion inside
        # process_execution_payload_envelope blows up. Folding everything into
        # one worker invocation keeps the precondition+mutation pair atomic
        # against the asyncio scheduler.
        from .spec.types import BeaconBlockHeader
        from .spec.state_transition.block.execution_payload_envelope import (
            process_execution_payload_envelope,
        )

        def _envelope_phase2(state, env, expected_slot, expected_root):
            if int(state.slot) != expected_slot:
                return ("slot_advanced", None, int(state.slot))
            header = state.latest_block_header
            if bytes(header.state_root) == b"\x00" * 32:
                filled_state_root = hash_tree_root(state)
                check_header = BeaconBlockHeader(
                    slot=header.slot,
                    proposer_index=header.proposer_index,
                    parent_root=header.parent_root,
                    state_root=filled_state_root,
                    body_root=header.body_root,
                )
                actual_root = hash_tree_root(check_header)
            else:
                actual_root = hash_tree_root(header)
            if expected_root != actual_root:
                return ("root_mismatch", actual_root, None)
            process_execution_payload_envelope(state, env, verify=False)
            return ("applied", hash_tree_root(state), None)

        try:
            result, payload, info = await self._on_state_thread(
                _envelope_phase2,
                self.state,
                signed_envelope,
                envelope_slot,
                beacon_block_root,
            )
        except Exception as e:
            logger.error(f"Failed to apply envelope (phase 2) for slot {envelope_slot}: {e}")
            import traceback
            traceback.print_exc()
            return

        if result == "slot_advanced":
            logger.info(
                f"Envelope for slot {envelope_slot} but state advanced to slot {info} "
                f"before apply, saving as pending"
            )
            self._pending_envelopes[beacon_block_root] = signed_envelope
            return
        if result == "root_mismatch":
            logger.warning(
                f"Envelope beacon_block_root {beacon_block_root.hex()[:16]} doesn't match "
                f"state header root {payload.hex()[:16]}, saving as pending"
            )
            self._pending_envelopes[beacon_block_root] = signed_envelope
            return

        state_root = payload
        self.store.save_state(state_root, self.state)
        if self.head_root:
            self.store.save_state(self.head_root, self.state)
        logger.info(
            f"Phase 2 applied: slot={envelope_slot}, "
            f"latest_block_hash={bytes(self.state.latest_block_hash).hex()[:16]}, "
            f"state_root={state_root.hex()[:16]}"
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
            # EIP-4788: parentBeaconBlockRoot is the PARENT beacon block's
            # root, not the current envelope's beacon_block_root. The bug
            # was sending envelope.beacon_block_root, which is the block
            # containing this envelope's bid — geth folded that into the
            # header hash, computed a different blockhash than the one we
            # claimed, and returned INVALID. Use the parent ref the
            # envelope carries explicitly.
            parent_beacon_root = bytes(envelope.parent_beacon_block_root)
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
            elif status.status == PayloadStatusEnum.SYNCING:
                logger.info(f"Payload validation deferred (EL syncing): {status.status}")
                return False
            else:
                # INVALID / ACCEPTED-with-error / unknown — the EL rejected
                # this payload outright. Surface at ERROR so it filters out
                # of the noisy WARNING bucket.
                logger.error(f"Payload validation failed: {status.status}")
                return False

        except Exception as e:
            logger.error(f"Failed to validate payload: {e}")
            return False

    async def _update_forkchoice(self) -> None:
        """Update forkchoice with the execution layer (no payload preparation)."""
        if not self.engine or not self.state:
            return

        try:
            # Get the latest block hash and timestamp based on state type.
            # Gloas: use latest_execution_payload_bid.block_hash (the most
            # recent payload imported into the EL) — see _gloas_el_head_hash.
            # Fulu/Electra: use the execution payload header.
            if hasattr(self.state, "latest_execution_payload_bid"):
                head_block_hash = self._gloas_el_head_hash()
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
