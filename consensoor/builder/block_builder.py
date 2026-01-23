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
    BLSSignature,
    Slot,
    ValidatorIndex,
    Bitvector,
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
)
from ..spec.network_config import get_config
from ..crypto import sign, compute_signing_root, hash_tree_root

AnySignedBeaconBlock = Union[
    SignedBellatrixBeaconBlock,
    SignedCapellaBeaconBlock,
    SignedDenebBeaconBlock,
    SignedElectraBeaconBlock,
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
        from ..spec.state_transition import process_slots, process_block

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
        temp_state = state.__class__.decode_bytes(bytes(state.encode_bytes()))
        logger.info(
            f"Building block for slot={slot}: state_slot={state.slot}, "
            f"latest_header_slot={state.latest_block_header.slot}"
        )

        # Process slots to advance state (fills in latest_block_header.state_root)
        if slot > int(temp_state.slot):
            process_slots(temp_state, slot)

        # Compute parent_root from state.latest_block_header (with state_root filled in)
        parent_root = hash_tree_root(temp_state.latest_block_header)

        randao_reveal = self._compute_randao_reveal(state, slot, proposer_key)
        execution_payload = self._build_execution_payload(execution_payload_dict, fork)
        body = self._build_block_body(state, randao_reveal, execution_payload, fork)

        # Build block with placeholder state_root
        block = self._create_block(slot, proposer_index, parent_root, b"\x00" * 32, body, fork)

        # Process block on temp state to compute post-state root
        try:
            process_block(temp_state, block)
            state_root = hash_tree_root(temp_state)
        except Exception as e:
            logger.warning(f"Failed to compute state_root: {e}, using zero")
            state_root = b"\x00" * 32

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

    def _build_block_body(self, state, randao_reveal: BLSSignature, execution_payload, fork: str):
        """Build the beacon block body for the appropriate fork."""
        sync_committee_size = SYNC_COMMITTEE_SIZE()
        empty_sync_aggregate = SyncAggregate(
            sync_committee_bits=Bitvector[sync_committee_size](),
            sync_committee_signature=BLSSignature(b"\xc0" + b"\x00" * 95),
        )

        base_fields = {
            "randao_reveal": randao_reveal,
            "eth1_data": state.eth1_data,
            "graffiti": Bytes32(self.node.config.graffiti_bytes),
            "proposer_slashings": [],
            "attester_slashings": [],
            "attestations": [],
            "deposits": [],
            "voluntary_exits": [],
            "sync_aggregate": empty_sync_aggregate,
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
