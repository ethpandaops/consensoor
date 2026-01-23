"""Bellatrix (The Merge) SSZ types."""

from .base import (
    Container, Vector, List,
    Bitvector,
    uint8, uint64, uint256,
    Bytes32, ByteVector, BLSPubkey, BLSSignature,
    Hash32, ExecutionAddress, Transaction,
    Slot, Epoch, ValidatorIndex, Gwei, Root,
    Checkpoint, Fork, ParticipationFlags,
)
from .phase0 import (
    Validator, Eth1Data, BeaconBlockHeader,
    ProposerSlashing, Deposit, SignedVoluntaryExit,
    Phase0Attestation, Phase0AttesterSlashing,
)
from .altair import SyncCommittee, SyncAggregate
from ..constants import (
    BYTES_PER_LOGS_BLOOM,
    MAX_EXTRA_DATA_BYTES,
    SLOTS_PER_EPOCH,
    SLOTS_PER_HISTORICAL_ROOT,
    EPOCHS_PER_HISTORICAL_VECTOR,
    EPOCHS_PER_SLASHINGS_VECTOR,
    EPOCHS_PER_ETH1_VOTING_PERIOD,
    HISTORICAL_ROOTS_LIMIT,
    VALIDATOR_REGISTRY_LIMIT,
    JUSTIFICATION_BITS_LENGTH,
    MAX_PROPOSER_SLASHINGS,
    MAX_ATTESTER_SLASHINGS_PRE_ELECTRA,
    MAX_ATTESTATIONS_PRE_ELECTRA,
    MAX_DEPOSITS,
    MAX_VOLUNTARY_EXITS,
)

MAX_TRANSACTIONS_PER_PAYLOAD = 2**20


class PowBlock(Container):
    block_hash: Hash32
    parent_hash: Hash32
    total_difficulty: uint256


class ExecutionPayloadHeaderBellatrix(Container):
    """Bellatrix ExecutionPayloadHeader (no withdrawals)."""
    parent_hash: Hash32
    fee_recipient: ExecutionAddress
    state_root: Bytes32
    receipts_root: Bytes32
    logs_bloom: ByteVector[BYTES_PER_LOGS_BLOOM]
    prev_randao: Bytes32
    block_number: uint64
    gas_limit: uint64
    gas_used: uint64
    timestamp: uint64
    extra_data: List[uint8, MAX_EXTRA_DATA_BYTES]
    base_fee_per_gas: uint256
    block_hash: Hash32
    transactions_root: Bytes32


class ExecutionPayloadBellatrix(Container):
    """Bellatrix ExecutionPayload (no withdrawals)."""
    parent_hash: Hash32
    fee_recipient: ExecutionAddress
    state_root: Bytes32
    receipts_root: Bytes32
    logs_bloom: ByteVector[BYTES_PER_LOGS_BLOOM]
    prev_randao: Bytes32
    block_number: uint64
    gas_limit: uint64
    gas_used: uint64
    timestamp: uint64
    extra_data: List[uint8, MAX_EXTRA_DATA_BYTES]
    base_fee_per_gas: uint256
    block_hash: Hash32
    transactions: List[Transaction, MAX_TRANSACTIONS_PER_PAYLOAD]


class BellatrixBeaconBlockBody(Container):
    randao_reveal: BLSSignature
    eth1_data: Eth1Data
    graffiti: Bytes32
    proposer_slashings: List[ProposerSlashing, MAX_PROPOSER_SLASHINGS]
    attester_slashings: List[Phase0AttesterSlashing, MAX_ATTESTER_SLASHINGS_PRE_ELECTRA]
    attestations: List[Phase0Attestation, MAX_ATTESTATIONS_PRE_ELECTRA]
    deposits: List[Deposit, MAX_DEPOSITS]
    voluntary_exits: List[SignedVoluntaryExit, MAX_VOLUNTARY_EXITS]
    sync_aggregate: SyncAggregate
    execution_payload: ExecutionPayloadBellatrix


class BellatrixBeaconBlock(Container):
    slot: Slot
    proposer_index: ValidatorIndex
    parent_root: Root
    state_root: Root
    body: BellatrixBeaconBlockBody


class SignedBellatrixBeaconBlock(Container):
    message: BellatrixBeaconBlock
    signature: BLSSignature


class BellatrixBeaconState(Container):
    genesis_time: uint64
    genesis_validators_root: Root
    slot: Slot
    fork: Fork
    latest_block_header: BeaconBlockHeader
    block_roots: Vector[Root, SLOTS_PER_HISTORICAL_ROOT()]
    state_roots: Vector[Root, SLOTS_PER_HISTORICAL_ROOT()]
    historical_roots: List[Root, HISTORICAL_ROOTS_LIMIT]
    eth1_data: Eth1Data
    eth1_data_votes: List[Eth1Data, EPOCHS_PER_ETH1_VOTING_PERIOD() * SLOTS_PER_EPOCH()]
    eth1_deposit_index: uint64
    validators: List[Validator, VALIDATOR_REGISTRY_LIMIT]
    balances: List[Gwei, VALIDATOR_REGISTRY_LIMIT]
    randao_mixes: Vector[Bytes32, EPOCHS_PER_HISTORICAL_VECTOR()]
    slashings: Vector[Gwei, EPOCHS_PER_SLASHINGS_VECTOR()]
    previous_epoch_participation: List[ParticipationFlags, VALIDATOR_REGISTRY_LIMIT]
    current_epoch_participation: List[ParticipationFlags, VALIDATOR_REGISTRY_LIMIT]
    justification_bits: Bitvector[JUSTIFICATION_BITS_LENGTH]
    previous_justified_checkpoint: Checkpoint
    current_justified_checkpoint: Checkpoint
    finalized_checkpoint: Checkpoint
    inactivity_scores: List[uint64, VALIDATOR_REGISTRY_LIMIT]
    current_sync_committee: SyncCommittee
    next_sync_committee: SyncCommittee
    latest_execution_payload_header: ExecutionPayloadHeaderBellatrix


__all__ = [
    "PowBlock",
    "ExecutionPayloadHeaderBellatrix",
    "ExecutionPayloadBellatrix",
    "BellatrixBeaconBlockBody",
    "BellatrixBeaconBlock",
    "SignedBellatrixBeaconBlock",
    "BellatrixBeaconState",
]
