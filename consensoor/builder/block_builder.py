"""Block builder for creating valid beacon blocks."""

import logging
from typing import Optional, TYPE_CHECKING, Union

from ..spec.types import (
    ElectraBeaconBlock,
    ElectraBeaconBlockBody,
    SignedElectraBeaconBlock,
    ExecutionPayload,
    ExecutionRequests,
    SyncAggregate,
    Root,
    Bytes32,
    Hash32,
    BLSSignature,
    Slot,
    Epoch,
    ValidatorIndex,
    Gwei,
    ExecutionAddress,
    Bitvector,
)
from ..spec.types.gloas import (
    BeaconBlock as GloasBeaconBlock,
    BeaconBlockBody as GloasBeaconBlockBody,
    SignedBeaconBlock as SignedGloasBeaconBlock,
    ExecutionPayloadBid,
    SignedExecutionPayloadBid,
    PayloadAttestation,
)
from ..spec.types.phase0 import ProposerSlashing, Deposit, SignedVoluntaryExit
from ..spec.types.bellatrix import (
    ExecutionPayloadBellatrix,
    BellatrixBeaconBlock,
    BellatrixBeaconBlockBody,
    SignedBellatrixBeaconBlock,
)
from ..spec.types.capella import (
    ExecutionPayloadCapella,
    CapellaBeaconBlock,
    CapellaBeaconBlockBody,
    SignedCapellaBeaconBlock,
    SignedBLSToExecutionChange,
    Withdrawal,
)
from ..spec.types.deneb import (
    DenebBeaconBlock,
    DenebBeaconBlockBody,
    SignedDenebBeaconBlock,
)
from ..spec.constants import (
    SLOTS_PER_EPOCH,
    DOMAIN_RANDAO,
    DOMAIN_BEACON_PROPOSER,
    SYNC_COMMITTEE_SIZE,
    MAX_ATTESTATIONS,
    MAX_ATTESTATIONS_ELECTRA,
    BUILDER_INDEX_SELF_BUILD,
    MAX_PAYLOAD_ATTESTATIONS,
)
from ..spec.network_config import get_config
from ..crypto import sign, compute_signing_root, hash_tree_root

AnySignedBeaconBlock = Union[
    SignedBellatrixBeaconBlock,
    SignedCapellaBeaconBlock,
    SignedDenebBeaconBlock,
    SignedElectraBeaconBlock,
    SignedGloasBeaconBlock,
]

if TYPE_CHECKING:
    from ..node import BeaconNode
    from ..validator import ValidatorKey

logger = logging.getLogger(__name__)


def compute_domain(domain_type: bytes, fork_version: bytes, genesis_validators_root: bytes) -> bytes:
    """Compute the domain for signing."""
    from ..spec.types import ForkData
    from ..crypto import sha256

    fork_data = ForkData(
        current_version=fork_version,
        genesis_validators_root=Root(genesis_validators_root),
    )
    fork_data_root = hash_tree_root(fork_data)
    return domain_type + fork_data_root[:28]


def get_domain(state, domain_type: bytes, epoch: Optional[int] = None) -> bytes:
    """Get the domain for signing at the given epoch."""
    if epoch is None:
        epoch = int(state.slot) // SLOTS_PER_EPOCH()

    net_config = get_config()
    fork_version = net_config.get_fork_version(epoch)
    return compute_domain(domain_type, fork_version, bytes(state.genesis_validators_root))


class BlockBuilder:
    """Builds valid beacon blocks."""

    def __init__(self, node: "BeaconNode"):
        self.node = node

    def _get_fork_for_slot(self, slot: int) -> str:
        """Determine which fork is active for a given slot."""
        config = get_config()
        epoch = slot // SLOTS_PER_EPOCH()

        if hasattr(config, 'gloas_fork_epoch') and epoch >= config.gloas_fork_epoch:
            return "gloas"
        if hasattr(config, 'fulu_fork_epoch') and epoch >= config.fulu_fork_epoch:
            return "fulu"
        if hasattr(config, 'electra_fork_epoch') and epoch >= config.electra_fork_epoch:
            return "electra"
        if hasattr(config, 'deneb_fork_epoch') and epoch >= config.deneb_fork_epoch:
            return "deneb"
        if hasattr(config, 'capella_fork_epoch') and epoch >= config.capella_fork_epoch:
            return "capella"
        return "bellatrix"

    async def build_block(
        self,
        slot: int,
        proposer_key: "ValidatorKey",
        execution_payload_dict: dict,
    ) -> Optional[AnySignedBeaconBlock]:
        """Build a complete signed beacon block."""
        import time as time_mod
        from ..spec.state_transition import process_slots, process_block

        build_start = time_mod.time()

        state = self.node.state
        if state is None:
            logger.error("Cannot build block: no state")
            return None

        proposer_index = proposer_key.validator_index
        if proposer_index is None:
            logger.error("Cannot build block: proposer has no validator index")
            return None

        fork = self._get_fork_for_slot(slot)
        logger.debug(f"Building block for fork: {fork}")

        # Work on a copy of the state to compute state_root
        # Use SSZ round-trip instead of copy.deepcopy for deterministic behavior across architectures
        t0 = time_mod.time()
        temp_state = state.__class__.decode_bytes(bytes(state.encode_bytes()))
        t1 = time_mod.time()
        logger.debug(f"State copy took {(t1-t0)*1000:.1f}ms")
        logger.info(
            f"Building block for slot={slot}: state_slot={state.slot}, "
            f"latest_header_slot={state.latest_block_header.slot}"
        )

        # Process slots to advance state (fills in latest_block_header.state_root)
        # May return upgraded state type if crossing a fork boundary
        t0 = time_mod.time()
        if slot > int(temp_state.slot):
            temp_state = process_slots(temp_state, slot)
        t1 = time_mod.time()
        logger.debug(f"process_slots took {(t1-t0)*1000:.1f}ms")

        # Compute parent_root from state.latest_block_header (with state_root filled in)
        t0 = time_mod.time()
        parent_root = hash_tree_root(temp_state.latest_block_header)
        t1 = time_mod.time()
        logger.debug(f"parent_root hash_tree_root took {(t1-t0)*1000:.1f}ms")

        # Use temp_state (which may be upgraded) for RANDAO and body construction
        # This ensures correct fork version is used for domain computation
        t0 = time_mod.time()
        randao_reveal = self._compute_randao_reveal(temp_state, slot, proposer_key)
        t1 = time_mod.time()
        logger.debug(f"RANDAO reveal took {(t1-t0)*1000:.1f}ms")

        # Get attestations from pool for inclusion (fork-specific limits)
        # NOTE: Limiting attestations due to slow BLS verification with py_ecc (~100ms each)
        # With 12s slots, we can handle ~8 attestations (800ms BLS time)
        # Attestation aggregation helps by combining multiple validators into fewer attestations
        if fork in ("electra", "fulu", "gloas"):
            max_attestations = min(8, MAX_ATTESTATIONS_ELECTRA())
        else:
            max_attestations = min(8, MAX_ATTESTATIONS())
        net_config = get_config()
        electra_fork_epoch = getattr(net_config, 'electra_fork_epoch', 2**64 - 1)
        pool_attestations = self.node.attestation_pool.get_attestations_for_block(
            slot, max_attestations * 2, electra_fork_epoch  # Get extra to account for filtering
        )

        # Pre-validate attestations before including in block
        # This catches attestations from forked chains that would fail BLS verification
        from ..spec.state_transition.helpers.attestation import get_indexed_attestation
        from ..spec.state_transition.helpers.predicates import is_valid_indexed_attestation

        valid_attestations = []
        for att in pool_attestations:
            try:
                indexed = get_indexed_attestation(temp_state, att)
                if is_valid_indexed_attestation(temp_state, indexed):
                    valid_attestations.append(att)
                    if len(valid_attestations) >= max_attestations:
                        break
                else:
                    logger.debug(f"Skipping invalid attestation: slot={att.data.slot}")
            except Exception as e:
                logger.debug(f"Error validating attestation: {e}")

        attestations = valid_attestations
        logger.info(f"Pre-validated {len(attestations)} of {len(pool_attestations)} attestations")

        # Build block body - GLOAS uses different structure (ePBS)
        if fork == "gloas":
            # GLOAS (ePBS): Build execution payload bid for self-build mode
            execution_payload_bid = self._build_execution_payload_bid(
                temp_state, slot, execution_payload_dict
            )
            # Self-build uses G2 point-at-infinity signature (no actual signing needed)
            g2_point_at_infinity = b"\xc0" + b"\x00" * 95
            signed_execution_payload_bid = SignedExecutionPayloadBid(
                message=execution_payload_bid,
                signature=BLSSignature(g2_point_at_infinity),
            )
            body = self._build_gloas_block_body(
                temp_state, randao_reveal, signed_execution_payload_bid, attestations
            )
            logger.info(
                f"Built GLOAS block body with self-build bid: "
                f"block_hash={execution_payload_dict['blockHash'][:18]}"
            )
        else:
            # Non-GLOAS: Build execution payload directly
            execution_payload = self._build_execution_payload(execution_payload_dict, fork)
            body = self._build_block_body(temp_state, randao_reveal, execution_payload, fork, attestations)
        logger.debug(f"Block body attestations count after build: {len(body.attestations)}")

        # Build block with placeholder state_root
        block = self._create_block(slot, proposer_index, parent_root, b"\x00" * 32, body, fork)
        logger.debug(f"Block attestations count after create: {len(block.body.attestations)}")

        # Process block on temp state to compute post-state root
        try:
            logger.debug(f"Processing block with {len(block.body.attestations)} attestations")
            t0 = time_mod.time()
            process_block(temp_state, block)
            t1 = time_mod.time()
            logger.debug(f"process_block took {(t1-t0)*1000:.1f}ms")

            t0 = time_mod.time()
            state_root = hash_tree_root(temp_state)
            t1 = time_mod.time()
            logger.debug(f"state_root hash_tree_root took {(t1-t0)*1000:.1f}ms")
            logger.debug(f"State root computed successfully")
        except Exception as e:
            logger.warning(f"Failed to compute state_root: {e}, using zero")
            import traceback
            traceback.print_exc()
            state_root = b"\x00" * 32

        total_time = time_mod.time() - build_start
        logger.info(f"Total block build time: {total_time*1000:.1f}ms")

        # Rebuild block with correct state_root
        block = self._create_block(slot, proposer_index, parent_root, state_root, body, fork)

        domain = get_domain(state, DOMAIN_BEACON_PROPOSER, slot // SLOTS_PER_EPOCH())
        signing_root = compute_signing_root(block, domain)
        signature = sign(proposer_key.privkey, signing_root)

        signed_block = self._create_signed_block(block, signature, fork)

        logger.info(
            f"Built {fork} block: slot={slot}, proposer={proposer_index}, "
            f"parent={parent_root.hex()[:16]}, state_root={state_root.hex()[:16]}"
        )

        return signed_block

    def _create_block(self, slot: int, proposer_index: int, parent_root: bytes, state_root: bytes, body, fork: str):
        """Create the appropriate block type for the fork."""
        if fork == "bellatrix":
            return BellatrixBeaconBlock(
                slot=Slot(slot),
                proposer_index=ValidatorIndex(proposer_index),
                parent_root=Root(parent_root),
                state_root=Root(state_root),
                body=body,
            )
        elif fork == "capella":
            return CapellaBeaconBlock(
                slot=Slot(slot),
                proposer_index=ValidatorIndex(proposer_index),
                parent_root=Root(parent_root),
                state_root=Root(state_root),
                body=body,
            )
        elif fork == "deneb":
            return DenebBeaconBlock(
                slot=Slot(slot),
                proposer_index=ValidatorIndex(proposer_index),
                parent_root=Root(parent_root),
                state_root=Root(state_root),
                body=body,
            )
        elif fork == "gloas":
            return GloasBeaconBlock(
                slot=Slot(slot),
                proposer_index=ValidatorIndex(proposer_index),
                parent_root=Root(parent_root),
                state_root=Root(state_root),
                body=body,
            )
        else:
            return ElectraBeaconBlock(
                slot=Slot(slot),
                proposer_index=ValidatorIndex(proposer_index),
                parent_root=Root(parent_root),
                state_root=Root(state_root),
                body=body,
            )

    def _create_signed_block(self, block, signature: bytes, fork: str) -> AnySignedBeaconBlock:
        """Create the appropriate signed block type for the fork."""
        if fork == "bellatrix":
            return SignedBellatrixBeaconBlock(
                message=block,
                signature=BLSSignature(signature),
            )
        elif fork == "capella":
            return SignedCapellaBeaconBlock(
                message=block,
                signature=BLSSignature(signature),
            )
        elif fork == "deneb":
            return SignedDenebBeaconBlock(
                message=block,
                signature=BLSSignature(signature),
            )
        elif fork == "gloas":
            return SignedGloasBeaconBlock(
                message=block,
                signature=BLSSignature(signature),
            )
        else:
            return SignedElectraBeaconBlock(
                message=block,
                signature=BLSSignature(signature),
            )

    def _compute_randao_reveal(self, state, slot: int, proposer_key: "ValidatorKey") -> BLSSignature:
        """Compute the RANDAO reveal for the block."""
        epoch = slot // SLOTS_PER_EPOCH()
        domain = get_domain(state, DOMAIN_RANDAO, epoch)

        from ..spec.types import Epoch
        epoch_obj = Epoch(epoch)
        signing_root = compute_signing_root(epoch_obj, domain)
        signature = sign(proposer_key.privkey, signing_root)
        return BLSSignature(signature)

    def _build_execution_payload(self, payload_dict: dict, fork: str):
        """Convert EL payload dict to ExecutionPayload SSZ object for the appropriate fork."""
        from ..spec.types.base import uint64, uint256, Hash32, ByteVector
        from ..spec.constants import BYTES_PER_LOGS_BLOOM

        def hex_to_bytes(h: str) -> bytes:
            return bytes.fromhex(h.replace("0x", ""))

        def hex_to_int(h) -> int:
            if h is None:
                return 0
            if isinstance(h, int):
                return h
            return int(h, 16)

        transactions = []
        for tx_hex in payload_dict.get("transactions") or []:
            tx_bytes = hex_to_bytes(tx_hex)
            transactions.append(list(tx_bytes))

        extra_data_bytes = hex_to_bytes(payload_dict.get("extraData", "0x"))

        # Debug: log input values
        logger.debug(
            f"_build_execution_payload: blockHash={payload_dict['blockHash']}, "
            f"stateRoot={payload_dict['stateRoot']}, parentHash={payload_dict['parentHash']}, "
            f"receiptsRoot={payload_dict['receiptsRoot']}"
        )

        base_fields = {
            "parent_hash": Hash32(hex_to_bytes(payload_dict["parentHash"])),
            "fee_recipient": hex_to_bytes(payload_dict["feeRecipient"]),
            "state_root": Bytes32(hex_to_bytes(payload_dict["stateRoot"])),
            "receipts_root": Bytes32(hex_to_bytes(payload_dict["receiptsRoot"])),
            "logs_bloom": ByteVector[BYTES_PER_LOGS_BLOOM](hex_to_bytes(payload_dict["logsBloom"])),
            "prev_randao": Bytes32(hex_to_bytes(payload_dict["prevRandao"])),
            "block_number": uint64(hex_to_int(payload_dict["blockNumber"])),
            "gas_limit": uint64(hex_to_int(payload_dict["gasLimit"])),
            "gas_used": uint64(hex_to_int(payload_dict["gasUsed"])),
            "timestamp": uint64(hex_to_int(payload_dict["timestamp"])),
            "extra_data": list(extra_data_bytes),
            "base_fee_per_gas": uint256(hex_to_int(payload_dict["baseFeePerGas"])),
            "block_hash": Hash32(hex_to_bytes(payload_dict["blockHash"])),
            "transactions": transactions,
        }

        if fork == "bellatrix":
            return ExecutionPayloadBellatrix(**base_fields)

        withdrawals = []
        for w in payload_dict.get("withdrawals") or []:
            withdrawal = Withdrawal(
                index=uint64(hex_to_int(w["index"])),
                validator_index=ValidatorIndex(hex_to_int(w["validatorIndex"])),
                address=hex_to_bytes(w["address"]),
                amount=uint64(hex_to_int(w["amount"])),
            )
            withdrawals.append(withdrawal)
        base_fields["withdrawals"] = withdrawals

        if fork == "capella":
            return ExecutionPayloadCapella(**base_fields)

        base_fields["blob_gas_used"] = uint64(hex_to_int(payload_dict.get("blobGasUsed", "0x0")))
        base_fields["excess_blob_gas"] = uint64(hex_to_int(payload_dict.get("excessBlobGas", "0x0")))

        return ExecutionPayload(**base_fields)

    def _build_block_body(self, state, randao_reveal: BLSSignature, execution_payload, fork: str, attestations=None):
        """Build the beacon block body for the appropriate fork."""
        current_slot = int(state.slot)
        # SyncAggregate: messages produced at slot N sign block at slot N-1
        # Block at slot N includes the sync aggregate from messages produced at slot N
        sync_aggregate_slot = current_slot
        sync_aggregate = self.node.sync_committee_pool.get_sync_aggregate(sync_aggregate_slot)
        participant_count = sum(1 for b in sync_aggregate.sync_committee_bits if b)
        sig_bytes = bytes(sync_aggregate.sync_committee_signature)
        g2_infinity = b'\xc0' + b'\x00' * 95
        logger.info(
            f"Sync aggregate for block at slot {current_slot} (using messages from slot {sync_aggregate_slot}): "
            f"{participant_count}/{SYNC_COMMITTEE_SIZE()} participants, sig_is_infinity={sig_bytes == g2_infinity}"
        )

        # Pre-validate sync aggregate signature before including in block
        # This catches sync aggregates from forked chains that would fail verification
        if participant_count > 0:
            sync_aggregate = self._validate_sync_aggregate(state, sync_aggregate)
            new_count = sum(1 for b in sync_aggregate.sync_committee_bits if b)
            if new_count != participant_count:
                logger.info(f"Sync aggregate failed validation, using empty aggregate")

        if attestations:
            logger.info(f"Including {len(attestations)} attestations in block body")
            for i, att in enumerate(attestations[:3]):  # Log first 3 for debugging
                logger.debug(f"  Attestation {i}: slot={att.data.slot}, committee={att.data.index}, bits={sum(1 for b in att.aggregation_bits if b)}")

        base_fields = {
            "randao_reveal": randao_reveal,
            "eth1_data": state.eth1_data,
            "graffiti": Bytes32(self.node.config.graffiti_bytes),
            "proposer_slashings": [],
            "attester_slashings": [],
            "attestations": attestations or [],
            "deposits": [],
            "voluntary_exits": [],
            "sync_aggregate": sync_aggregate,
            "execution_payload": execution_payload,
        }

        if fork == "bellatrix":
            return BellatrixBeaconBlockBody(**base_fields)

        base_fields["bls_to_execution_changes"] = []

        if fork == "capella":
            return CapellaBeaconBlockBody(**base_fields)

        base_fields["blob_kzg_commitments"] = []

        if fork == "deneb":
            return DenebBeaconBlockBody(**base_fields)

        empty_execution_requests = ExecutionRequests(
            deposits=[],
            withdrawals=[],
            consolidations=[],
        )
        base_fields["execution_requests"] = empty_execution_requests

        return ElectraBeaconBlockBody(**base_fields)

    def _validate_sync_aggregate(self, state, sync_aggregate: SyncAggregate) -> SyncAggregate:
        """Validate sync aggregate signature before including in block.

        If validation fails (e.g., due to fork mismatch or stale contributions),
        returns an empty sync aggregate with G2 infinity signature.

        Args:
            state: Beacon state
            sync_aggregate: Sync aggregate to validate

        Returns:
            Original sync aggregate if valid, empty aggregate if invalid
        """
        from ..spec.state_transition.helpers.accessors import get_block_root_at_slot
        from ..spec.state_transition.helpers.domain import get_domain, compute_signing_root
        from ..spec.state_transition.helpers.misc import compute_epoch_at_slot
        from ..spec.constants import DOMAIN_SYNC_COMMITTEE
        from ..crypto import bls_verify

        try:
            # Get participant pubkeys
            committee_pubkeys = list(state.current_sync_committee.pubkeys)
            participant_pubkeys = [
                committee_pubkeys[i]
                for i in range(SYNC_COMMITTEE_SIZE())
                if sync_aggregate.sync_committee_bits[i]
            ]

            if not participant_pubkeys:
                return sync_aggregate

            # Compute signing root (same as process_sync_aggregate)
            previous_slot = max(int(state.slot), 1) - 1
            domain = get_domain(
                state,
                DOMAIN_SYNC_COMMITTEE,
                compute_epoch_at_slot(previous_slot),
            )
            block_root = get_block_root_at_slot(state, previous_slot)
            signing_root = compute_signing_root(block_root, domain)

            # Verify signature
            sig_bytes = bytes(sync_aggregate.sync_committee_signature)
            is_valid = bls_verify(
                [bytes(pk) for pk in participant_pubkeys],
                signing_root,
                sig_bytes,
            )

            if is_valid:
                return sync_aggregate

            logger.warning(
                f"Sync aggregate pre-validation failed: "
                f"participants={len(participant_pubkeys)}, previous_slot={previous_slot}"
            )

        except Exception as e:
            logger.warning(f"Error validating sync aggregate: {e}")

        # Return empty sync aggregate on failure
        return SyncAggregate(
            sync_committee_bits=Bitvector[SYNC_COMMITTEE_SIZE()](),
            sync_committee_signature=BLSSignature(b"\xc0" + b"\x00" * 95),
        )

    def _build_execution_payload_bid(
        self,
        state,
        slot: int,
        execution_payload_dict: dict,
    ) -> ExecutionPayloadBid:
        """Build an ExecutionPayloadBid for self-build mode (ePBS).

        In self-build mode, the proposer acts as their own builder.
        The builder_index is set to BUILDER_INDEX_SELF_BUILD (2^64 - 1).
        """
        from ..spec.types.base import uint64

        def hex_to_bytes(h: str) -> bytes:
            return bytes.fromhex(h.replace("0x", ""))

        def hex_to_int(h) -> int:
            if h is None:
                return 0
            if isinstance(h, int):
                return h
            return int(h, 16)

        # Get parent block info from state
        parent_block_hash = bytes(state.latest_block_hash) if hasattr(state, "latest_block_hash") else b"\x00" * 32
        parent_block_root = hash_tree_root(state.latest_block_header)

        bid = ExecutionPayloadBid(
            parent_block_hash=Hash32(parent_block_hash),
            parent_block_root=Root(parent_block_root),
            block_hash=Hash32(hex_to_bytes(execution_payload_dict["blockHash"])),
            prev_randao=Bytes32(hex_to_bytes(execution_payload_dict["prevRandao"])),
            fee_recipient=ExecutionAddress(hex_to_bytes(execution_payload_dict["feeRecipient"])),
            gas_limit=uint64(hex_to_int(execution_payload_dict["gasLimit"])),
            builder_index=uint64(BUILDER_INDEX_SELF_BUILD),
            slot=Slot(slot),
            value=Gwei(0),
            execution_payment=Gwei(0),
            blob_kzg_commitments_root=Root(b"\x00" * 32),
        )

        logger.debug(
            f"Built execution payload bid: slot={slot}, block_hash={execution_payload_dict['blockHash'][:18]}, "
            f"builder_index=SELF_BUILD"
        )

        return bid

    def _build_gloas_block_body(
        self,
        state,
        randao_reveal: BLSSignature,
        signed_execution_payload_bid: SignedExecutionPayloadBid,
        attestations=None,
    ) -> GloasBeaconBlockBody:
        """Build a GLOAS (ePBS) block body.

        GLOAS block body contains:
        - signed_execution_payload_bid instead of execution_payload
        - payload_attestations from PTC
        """
        current_slot = int(state.slot)
        sync_aggregate_slot = current_slot
        sync_aggregate = self.node.sync_committee_pool.get_sync_aggregate(sync_aggregate_slot)
        participant_count = sum(1 for b in sync_aggregate.sync_committee_bits if b)
        sig_bytes = bytes(sync_aggregate.sync_committee_signature)
        g2_infinity = b'\xc0' + b'\x00' * 95
        logger.info(
            f"Sync aggregate for GLOAS block at slot {current_slot}: "
            f"{participant_count}/{SYNC_COMMITTEE_SIZE()} participants, sig_is_infinity={sig_bytes == g2_infinity}"
        )

        # Pre-validate sync aggregate
        if participant_count > 0:
            sync_aggregate = self._validate_sync_aggregate(state, sync_aggregate)
            new_count = sum(1 for b in sync_aggregate.sync_committee_bits if b)
            if new_count != participant_count:
                logger.info(f"Sync aggregate failed validation, using empty aggregate")

        if attestations:
            logger.info(f"Including {len(attestations)} attestations in GLOAS block body")

        # TODO: Get payload attestations from PTC pool
        # For now, empty payload_attestations since we're doing self-build
        payload_attestations = []

        return GloasBeaconBlockBody(
            randao_reveal=randao_reveal,
            eth1_data=state.eth1_data,
            graffiti=Bytes32(self.node.config.graffiti_bytes),
            proposer_slashings=[],
            attester_slashings=[],
            attestations=attestations or [],
            deposits=[],
            voluntary_exits=[],
            sync_aggregate=sync_aggregate,
            bls_to_execution_changes=[],
            signed_execution_payload_bid=signed_execution_payload_bid,
            payload_attestations=payload_attestations,
        )
