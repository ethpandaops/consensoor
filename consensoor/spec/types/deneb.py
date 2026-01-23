"""Deneb SSZ types (adds blob gas fields)."""

from .base import (
    Container, Vector, List,
    Bitvector,
    uint8, uint64, uint256,
    Bytes32, ByteVector, BLSSignature,
    Hash32, ExecutionAddress, Transaction,
    Slot, ValidatorIndex, Gwei, Root, WithdrawalIndex,
    Checkpoint, Fork, ParticipationFlags, KZGCommitment, KZGProof,
)
from .phase0 import (
    Validator, Eth1Data, BeaconBlockHeader,
    ProposerSlashing, Deposit, SignedVoluntaryExit,
    Phase0Attestation, Phase0AttesterSlashing,
)
from .altair import SyncCommittee, SyncAggregate
from .capella import Withdrawal, SignedBLSToExecutionChange, HistoricalSummary
from .phase0 import SignedBeaconBlockHeader
from ..constants import (
    BYTES_PER_LOGS_BLOOM,
    MAX_EXTRA_DATA_BYTES,
    MAX_WITHDRAWALS_PER_PAYLOAD,
    MAX_BLS_TO_EXECUTION_CHANGES,
    MAX_BLOB_COMMITMENTS_PER_BLOCK,
    FIELD_ELEMENTS_PER_BLOB,
    KZG_COMMITMENT_INCLUSION_PROOF_DEPTH,
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
    EXECUTION_PAYLOAD_DEPTH,
    FINALIZED_ROOT_DEPTH,
    CURRENT_SYNC_COMMITTEE_DEPTH,
    NEXT_SYNC_COMMITTEE_DEPTH,
)

MAX_TRANSACTIONS_PER_PAYLOAD = 2**20
BYTES_PER_BLOB = FIELD_ELEMENTS_PER_BLOB * 32


class Blob(ByteVector[BYTES_PER_BLOB]):
    pass


class BlobIdentifier(Container):
    block_root: Root
    index: uint64


class BlobSidecar(Container):
    index: uint64
    blob: Blob
    kzg_commitment: KZGCommitment
    kzg_proof: KZGProof
    signed_block_header: SignedBeaconBlockHeader
    kzg_commitment_inclusion_proof: Vector[Bytes32, KZG_COMMITMENT_INCLUSION_PROOF_DEPTH]


class ExecutionPayloadHeader(Container):
    """Deneb/Electra ExecutionPayloadHeader (with blob gas fields)."""
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
    withdrawals_root: Bytes32
    blob_gas_used: uint64
    excess_blob_gas: uint64


class DenebLightClientHeader(Container):
    beacon: BeaconBlockHeader
    execution: ExecutionPayloadHeader
    execution_branch: Vector[Bytes32, EXECUTION_PAYLOAD_DEPTH]


class DenebLightClientBootstrap(Container):
    header: DenebLightClientHeader
    current_sync_committee: SyncCommittee
    current_sync_committee_branch: Vector[Bytes32, CURRENT_SYNC_COMMITTEE_DEPTH]


class DenebLightClientUpdate(Container):
    attested_header: DenebLightClientHeader
    next_sync_committee: SyncCommittee
    next_sync_committee_branch: Vector[Bytes32, NEXT_SYNC_COMMITTEE_DEPTH]
    finalized_header: DenebLightClientHeader
    finality_branch: Vector[Bytes32, FINALIZED_ROOT_DEPTH]
    sync_aggregate: SyncAggregate
    signature_slot: Slot


class DenebLightClientFinalityUpdate(Container):
    attested_header: DenebLightClientHeader
    finalized_header: DenebLightClientHeader
    finality_branch: Vector[Bytes32, FINALIZED_ROOT_DEPTH]
    sync_aggregate: SyncAggregate
    signature_slot: Slot


class DenebLightClientOptimisticUpdate(Container):
    attested_header: DenebLightClientHeader
    sync_aggregate: SyncAggregate
    signature_slot: Slot


class ExecutionPayload(Container):
    """Deneb/Electra ExecutionPayload (with blob gas fields)."""
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
    withdrawals: List[Withdrawal, MAX_WITHDRAWALS_PER_PAYLOAD()]
    blob_gas_used: uint64
    excess_blob_gas: uint64


class DenebBeaconBlockBody(Container):
    randao_reveal: BLSSignature
    eth1_data: Eth1Data
    graffiti: Bytes32
    proposer_slashings: List[ProposerSlashing, MAX_PROPOSER_SLASHINGS]
    attester_slashings: List[Phase0AttesterSlashing, MAX_ATTESTER_SLASHINGS_PRE_ELECTRA]
    attestations: List[Phase0Attestation, MAX_ATTESTATIONS_PRE_ELECTRA]
    deposits: List[Deposit, MAX_DEPOSITS]
    voluntary_exits: List[SignedVoluntaryExit, MAX_VOLUNTARY_EXITS]
    sync_aggregate: SyncAggregate
    execution_payload: ExecutionPayload
    bls_to_execution_changes: List[SignedBLSToExecutionChange, MAX_BLS_TO_EXECUTION_CHANGES]
    blob_kzg_commitments: List[KZGCommitment, MAX_BLOB_COMMITMENTS_PER_BLOCK]


class DenebBeaconBlock(Container):
    slot: Slot
    proposer_index: ValidatorIndex
    parent_root: Root
    state_root: Root
    body: DenebBeaconBlockBody


class SignedDenebBeaconBlock(Container):
    message: DenebBeaconBlock
    signature: BLSSignature


class DenebBeaconState(Container):
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
    latest_execution_payload_header: ExecutionPayloadHeader
    next_withdrawal_index: WithdrawalIndex
    next_withdrawal_validator_index: ValidatorIndex
    historical_summaries: List[HistoricalSummary, HISTORICAL_ROOTS_LIMIT]


__all__ = [
    "Blob",
    "BlobIdentifier",
    "BlobSidecar",
    "ExecutionPayloadHeader",
    "DenebLightClientHeader",
    "DenebLightClientBootstrap",
    "DenebLightClientUpdate",
    "DenebLightClientFinalityUpdate",
    "DenebLightClientOptimisticUpdate",
    "ExecutionPayload",
    "DenebBeaconBlockBody",
    "DenebBeaconBlock",
    "SignedDenebBeaconBlock",
    "DenebBeaconState",
]
