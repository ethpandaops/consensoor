"""Gloas SSZ types (ePBS - Enshrined Proposer-Builder Separation).

Updated for v1.7.0-alpha.7.
"""

from .base import (
    Container, Vector, List,
    Bitvector,
    uint8, uint64, uint256, boolean,
    Bytes32, ByteVector, ByteList, BLSPubkey, BLSSignature,
    Slot, Epoch, ValidatorIndex, Gwei, Root, Hash32, ExecutionAddress,
    ParticipationFlags, KZGCommitment, KZGProof, Transaction, WithdrawalIndex,
)
from .phase0 import (
    Validator, Eth1Data, BeaconBlockHeader,
    ProposerSlashing, Deposit, SignedVoluntaryExit,
)
from .altair import SyncCommittee, SyncAggregate
from .capella import Withdrawal, SignedBLSToExecutionChange, HistoricalSummary
from .electra import (
    Attestation, AttesterSlashing, ExecutionRequests,
    PendingDeposit, PendingPartialWithdrawal, PendingConsolidation,
)
from .fulu import proposer_lookahead_length, Cell
from ..constants import (
    SLOTS_PER_EPOCH,
    SLOTS_PER_HISTORICAL_ROOT,
    EPOCHS_PER_HISTORICAL_VECTOR,
    EPOCHS_PER_SLASHINGS_VECTOR,
    EPOCHS_PER_ETH1_VOTING_PERIOD,
    HISTORICAL_ROOTS_LIMIT,
    VALIDATOR_REGISTRY_LIMIT,
    JUSTIFICATION_BITS_LENGTH,
    MAX_PROPOSER_SLASHINGS,
    MAX_ATTESTER_SLASHINGS_ELECTRA,
    MAX_ATTESTATIONS_ELECTRA,
    MAX_DEPOSITS,
    MAX_VOLUNTARY_EXITS,
    MAX_BLS_TO_EXECUTION_CHANGES,
    MAX_BLOB_COMMITMENTS_PER_BLOCK,
    MAX_WITHDRAWALS_PER_PAYLOAD,
    MAX_TRANSACTIONS_PER_PAYLOAD,
    MAX_BYTES_PER_TRANSACTION,
    MAX_EXTRA_DATA_BYTES,
    BYTES_PER_LOGS_BLOOM,
    PENDING_DEPOSITS_LIMIT,
    PENDING_PARTIAL_WITHDRAWALS_LIMIT,
    PENDING_CONSOLIDATIONS_LIMIT,
    PTC_SIZE,
    MAX_PAYLOAD_ATTESTATIONS,
    BUILDER_REGISTRY_LIMIT,
    BUILDER_PENDING_WITHDRAWALS_LIMIT,
    MIN_SEED_LOOKAHEAD,
    EXECUTION_BLOCK_HASH_DEPTH_GLOAS,
    FINALIZED_ROOT_DEPTH_ELECTRA,
    CURRENT_SYNC_COMMITTEE_DEPTH_ELECTRA,
    NEXT_SYNC_COMMITTEE_DEPTH_ELECTRA,
)
from .base import Checkpoint, Fork

# Gloas-specific type aliases
BuilderIndex = uint64
PayloadStatus = uint8


class ForkChoiceNode(Container):
    root: Root
    payload_status: PayloadStatus


class DataColumnSidecar(Container):
    """Gloas DataColumnSidecar - simplified compared to Fulu."""
    index: uint64
    column: List[Cell, MAX_BLOB_COMMITMENTS_PER_BLOCK]
    kzg_proofs: List[KZGProof, MAX_BLOB_COMMITMENTS_PER_BLOCK]
    slot: Slot
    beacon_block_root: Root


class Builder(Container):
    pubkey: BLSPubkey
    version: uint8
    execution_address: ExecutionAddress
    balance: Gwei
    deposit_epoch: Epoch
    withdrawable_epoch: Epoch


class BuilderPendingWithdrawal(Container):
    fee_recipient: ExecutionAddress
    amount: Gwei
    builder_index: BuilderIndex


class BuilderPendingPayment(Container):
    weight: Gwei
    withdrawal: BuilderPendingWithdrawal


class PayloadAttestationData(Container):
    beacon_block_root: Root
    slot: Slot
    payload_present: boolean
    blob_data_available: boolean


class PayloadAttestation(Container):
    aggregation_bits: Bitvector[PTC_SIZE()]
    data: PayloadAttestationData
    signature: BLSSignature


class PayloadAttestationMessage(Container):
    validator_index: ValidatorIndex
    data: PayloadAttestationData
    signature: BLSSignature


class IndexedPayloadAttestation(Container):
    attesting_indices: List[ValidatorIndex, PTC_SIZE()]
    data: PayloadAttestationData
    signature: BLSSignature


class BlockAccessList(ByteList[MAX_BYTES_PER_TRANSACTION]):
    """Block access list bytes (EIP-7928, new in Gloas)."""
    pass


class ExecutionPayload(Container):
    """Gloas ExecutionPayload - adds block_access_list (EIP-7928) and slot_number (EIP-7843)."""
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
    block_access_list: BlockAccessList
    slot_number: uint64


class ExecutionPayloadBid(Container):
    parent_block_hash: Hash32
    parent_block_root: Root
    block_hash: Hash32
    prev_randao: Bytes32
    fee_recipient: ExecutionAddress
    gas_limit: uint64
    builder_index: BuilderIndex
    slot: Slot
    value: Gwei
    execution_payment: Gwei
    blob_kzg_commitments: List[KZGCommitment, MAX_BLOB_COMMITMENTS_PER_BLOCK]
    execution_requests_root: Root


class SignedExecutionPayloadBid(Container):
    message: ExecutionPayloadBid
    signature: BLSSignature


class ExecutionPayloadEnvelope(Container):
    payload: ExecutionPayload
    execution_requests: ExecutionRequests
    builder_index: BuilderIndex
    beacon_block_root: Root
    parent_beacon_block_root: Root


class SignedExecutionPayloadEnvelope(Container):
    message: ExecutionPayloadEnvelope
    signature: BLSSignature


class ProposerPreferences(Container):
    """Proposer preferences for block proposals (ePBS)."""
    dependent_root: Root
    proposal_slot: Slot
    validator_index: ValidatorIndex
    fee_recipient: ExecutionAddress
    gas_limit: uint64


class SignedProposerPreferences(Container):
    """Signed proposer preferences."""
    message: ProposerPreferences
    signature: BLSSignature


class BeaconBlockBody(Container):
    """Gloas BeaconBlockBody (ePBS)."""
    randao_reveal: BLSSignature
    eth1_data: Eth1Data
    graffiti: Bytes32
    proposer_slashings: List[ProposerSlashing, MAX_PROPOSER_SLASHINGS]
    attester_slashings: List[AttesterSlashing, MAX_ATTESTER_SLASHINGS_ELECTRA]
    attestations: List[Attestation, MAX_ATTESTATIONS_ELECTRA()]
    deposits: List[Deposit, MAX_DEPOSITS]
    voluntary_exits: List[SignedVoluntaryExit, MAX_VOLUNTARY_EXITS]
    sync_aggregate: SyncAggregate
    bls_to_execution_changes: List[SignedBLSToExecutionChange, MAX_BLS_TO_EXECUTION_CHANGES]
    signed_execution_payload_bid: SignedExecutionPayloadBid
    payload_attestations: List[PayloadAttestation, MAX_PAYLOAD_ATTESTATIONS]
    parent_execution_requests: ExecutionRequests


class BeaconBlock(Container):
    slot: Slot
    proposer_index: ValidatorIndex
    parent_root: Root
    state_root: Root
    body: BeaconBlockBody


class SignedBeaconBlock(Container):
    message: BeaconBlock
    signature: BLSSignature


class BeaconState(Container):
    """Gloas BeaconState (ePBS).

    Field order matches v1.7.0-alpha.7 exactly. Critical for SSZ correctness.
    """
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
    # [New in Gloas:EIP7732] Replaces latest_execution_payload_header
    latest_block_hash: Hash32
    next_withdrawal_index: WithdrawalIndex
    next_withdrawal_validator_index: ValidatorIndex
    historical_summaries: List[HistoricalSummary, HISTORICAL_ROOTS_LIMIT]
    deposit_requests_start_index: uint64
    deposit_balance_to_consume: Gwei
    exit_balance_to_consume: Gwei
    earliest_exit_epoch: Epoch
    consolidation_balance_to_consume: Gwei
    earliest_consolidation_epoch: Epoch
    pending_deposits: List[PendingDeposit, PENDING_DEPOSITS_LIMIT()]
    pending_partial_withdrawals: List[PendingPartialWithdrawal, PENDING_PARTIAL_WITHDRAWALS_LIMIT()]
    pending_consolidations: List[PendingConsolidation, PENDING_CONSOLIDATIONS_LIMIT()]
    proposer_lookahead: Vector[ValidatorIndex, proposer_lookahead_length()]
    # [New in Gloas:EIP7732] ePBS additions follow
    builders: List[Builder, BUILDER_REGISTRY_LIMIT]
    next_withdrawal_builder_index: BuilderIndex
    execution_payload_availability: Bitvector[SLOTS_PER_HISTORICAL_ROOT()]
    builder_pending_payments: Vector[BuilderPendingPayment, 2 * SLOTS_PER_EPOCH()]
    builder_pending_withdrawals: List[BuilderPendingWithdrawal, BUILDER_PENDING_WITHDRAWALS_LIMIT]
    latest_execution_payload_bid: ExecutionPayloadBid
    payload_expected_withdrawals: List[Withdrawal, MAX_WITHDRAWALS_PER_PAYLOAD()]
    # [New in Gloas:EIP7732] PTC window
    ptc_window: Vector[Vector[ValidatorIndex, PTC_SIZE()], (2 + MIN_SEED_LOOKAHEAD) * SLOTS_PER_EPOCH()]


class LightClientHeader(Container):
    """Gloas LightClientHeader - replaces ExecutionPayloadHeader with execution_block_hash."""
    beacon: BeaconBlockHeader
    execution_block_hash: Hash32
    execution_branch: Vector[Bytes32, EXECUTION_BLOCK_HASH_DEPTH_GLOAS]


class LightClientOptimisticUpdate(Container):
    attested_header: LightClientHeader
    sync_aggregate: SyncAggregate
    signature_slot: Slot


class LightClientFinalityUpdate(Container):
    attested_header: LightClientHeader
    finalized_header: LightClientHeader
    finality_branch: Vector[Bytes32, FINALIZED_ROOT_DEPTH_ELECTRA]
    sync_aggregate: SyncAggregate
    signature_slot: Slot


class LightClientUpdate(Container):
    attested_header: LightClientHeader
    next_sync_committee: SyncCommittee
    next_sync_committee_branch: Vector[Bytes32, NEXT_SYNC_COMMITTEE_DEPTH_ELECTRA]
    finalized_header: LightClientHeader
    finality_branch: Vector[Bytes32, FINALIZED_ROOT_DEPTH_ELECTRA]
    sync_aggregate: SyncAggregate
    signature_slot: Slot


class LightClientBootstrap(Container):
    header: LightClientHeader
    current_sync_committee: SyncCommittee
    current_sync_committee_branch: Vector[Bytes32, CURRENT_SYNC_COMMITTEE_DEPTH_ELECTRA]


__all__ = [
    "DataColumnSidecar",
    "BuilderIndex",
    "PayloadStatus",
    "ForkChoiceNode",
    "Builder",
    "BuilderPendingWithdrawal",
    "BuilderPendingPayment",
    "PayloadAttestationData",
    "PayloadAttestation",
    "PayloadAttestationMessage",
    "IndexedPayloadAttestation",
    "BlockAccessList",
    "ExecutionPayload",
    "ExecutionPayloadBid",
    "SignedExecutionPayloadBid",
    "ExecutionPayloadEnvelope",
    "SignedExecutionPayloadEnvelope",
    "ProposerPreferences",
    "SignedProposerPreferences",
    "BeaconBlockBody",
    "BeaconBlock",
    "SignedBeaconBlock",
    "BeaconState",
    "LightClientHeader",
    "LightClientOptimisticUpdate",
    "LightClientFinalityUpdate",
    "LightClientUpdate",
    "LightClientBootstrap",
]
