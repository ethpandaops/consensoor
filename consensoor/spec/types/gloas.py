"""Gloas SSZ types (ePBS - Enshrined Proposer-Builder Separation).

Updated for consensus-specs master + PR #4630 (EIP-7688: forward compatible
consensus data structures). Gloas containers use ProgressiveContainer
(EIP-7495) and unbounded ProgressiveList/ProgressiveBitlist/ProgressiveByteList
(EIP-7916) instead of bounded List/Bitlist. Per-block limits formerly enforced
by SSZ bounds are now asserted in the state transition (process_operations et
al.).
"""

from .base import (
    Container, Vector, List,
    Bitvector, Bitlist,
    ProgressiveContainer, ProgressiveList, ProgressiveBitlist, ProgressiveByteList,
    uint8, uint64, uint256, boolean,
    Bytes32, ByteVector, ByteList, BLSPubkey, BLSSignature,
    Slot, Epoch, ValidatorIndex, Gwei, Root, Hash32, ExecutionAddress,
    ParticipationFlags, KZGCommitment, KZGProof, WithdrawalIndex,
)
from .phase0 import (
    Validator, Eth1Data, BeaconBlockHeader, AttestationData,
    ProposerSlashing, Deposit, SignedVoluntaryExit,
)
from .altair import SyncCommittee, SyncAggregate
from .capella import Withdrawal, SignedBLSToExecutionChange, HistoricalSummary
from .electra import (
    PendingDeposit, PendingPartialWithdrawal, PendingConsolidation,
    DepositRequest, WithdrawalRequest, ConsolidationRequest,
)
from .fulu import proposer_lookahead_length, Cell
from ..constants import (
    SLOTS_PER_EPOCH,
    SLOTS_PER_HISTORICAL_ROOT,
    EPOCHS_PER_HISTORICAL_VECTOR,
    EPOCHS_PER_SLASHINGS_VECTOR,
    EPOCHS_PER_ETH1_VOTING_PERIOD,
    HISTORICAL_ROOTS_LIMIT,
    JUSTIFICATION_BITS_LENGTH,
    MAX_BLOB_COMMITMENTS_PER_BLOCK,
    MAX_COMMITTEES_PER_SLOT,
    MAX_EXTRA_DATA_BYTES,
    BYTES_PER_LOGS_BLOOM,
    PTC_SIZE,
    MIN_SEED_LOOKAHEAD,
    EXECUTION_BLOCK_HASH_DEPTH_GLOAS,
    FINALIZED_ROOT_DEPTH_GLOAS,
    CURRENT_SYNC_COMMITTEE_DEPTH_GLOAS,
    NEXT_SYNC_COMMITTEE_DEPTH_GLOAS,
)
from .base import Checkpoint, Fork

# Gloas-specific type aliases
BuilderIndex = uint64
PayloadStatus = uint8

# [Modified in Gloas:EIP7688] progressive (unbounded) collection aliases
AggregationBits = ProgressiveBitlist
AttestingIndices = ProgressiveList[ValidatorIndex]
Transaction = ProgressiveByteList
DepositRequests = ProgressiveList[DepositRequest]
WithdrawalRequests = ProgressiveList[WithdrawalRequest]
ConsolidationRequests = ProgressiveList[ConsolidationRequest]


class ForkChoiceNode(Container):
    root: Root
    payload_status: PayloadStatus


class DataColumnSidecar(Container):
    """Gloas DataColumnSidecar - simplified compared to Fulu."""
    index: uint64
    # [Modified in Gloas:EIP7688]
    column: ProgressiveList[Cell]
    kzg_proofs: ProgressiveList[KZGProof]
    slot: Slot
    beacon_block_root: Root


class PartialDataColumnSidecar(Container):
    # [Modified in Gloas:EIP7688]
    cells_present_bitmap: ProgressiveBitlist
    partial_column: ProgressiveList[Cell]
    kzg_proofs: ProgressiveList[KZGProof]


class PartialDataColumnPartsMetadata(Container):
    available: Bitlist[MAX_BLOB_COMMITMENTS_PER_BLOCK]
    requests: Bitlist[MAX_BLOB_COMMITMENTS_PER_BLOCK]


class PartialDataColumnGroupID(Container):
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
    # [New in alpha.11+] lets process_proposer_slashing drop only payments
    # belonging to the slashed proposer
    proposer_index: ValidatorIndex


class PayloadAttestationData(Container):
    beacon_block_root: Root
    slot: Slot
    payload_present: boolean
    blob_data_available: boolean


class PayloadAttestation(ProgressiveContainer(active_fields=[1] * 3)):
    aggregation_bits: Bitvector[PTC_SIZE()]
    data: PayloadAttestationData
    signature: BLSSignature


class PayloadAttestationMessage(Container):
    validator_index: ValidatorIndex
    data: PayloadAttestationData
    signature: BLSSignature


class IndexedPayloadAttestation(ProgressiveContainer(active_fields=[1] * 3)):
    attesting_indices: List[ValidatorIndex, PTC_SIZE()]
    data: PayloadAttestationData
    signature: BLSSignature


# Block access list bytes (EIP-7928). [Modified in Gloas:EIP7688] unbounded.
BlockAccessList = ProgressiveByteList


class BuilderDepositRequest(Container):
    """Builder deposit request (EIP-8282, new in Gloas)."""
    pubkey: BLSPubkey
    withdrawal_credentials: Bytes32
    amount: Gwei
    signature: BLSSignature


class BuilderExitRequest(Container):
    """Builder exit request (EIP-8282, new in Gloas)."""
    source_address: ExecutionAddress
    pubkey: BLSPubkey


BuilderDepositRequests = ProgressiveList[BuilderDepositRequest]
BuilderExitRequests = ProgressiveList[BuilderExitRequest]


# [Modified in Gloas:EIP7688]
class ExecutionRequests(ProgressiveContainer(active_fields=[1] * 5)):
    deposits: DepositRequests
    withdrawals: WithdrawalRequests
    consolidations: ConsolidationRequests
    # [New in Gloas:EIP8282]
    builder_deposits: BuilderDepositRequests
    builder_exits: BuilderExitRequests


# [Modified in Gloas:EIP7688]
class Attestation(ProgressiveContainer(active_fields=[1] * 4)):
    aggregation_bits: AggregationBits
    data: AttestationData
    signature: BLSSignature
    committee_bits: Bitvector[MAX_COMMITTEES_PER_SLOT()]


# [Modified in Gloas:EIP7688]
class IndexedAttestation(ProgressiveContainer(active_fields=[1] * 3)):
    attesting_indices: AttestingIndices
    data: AttestationData
    signature: BLSSignature


class AttesterSlashing(Container):
    attestation_1: IndexedAttestation
    attestation_2: IndexedAttestation


class AggregateAndProof(Container):
    aggregator_index: ValidatorIndex
    aggregate: Attestation
    selection_proof: BLSSignature


class SignedAggregateAndProof(Container):
    message: AggregateAndProof
    signature: BLSSignature


# [Modified in Gloas:EIP7688]
class ExecutionPayload(ProgressiveContainer(active_fields=[1] * 19)):
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
    # [Modified in Gloas:EIP7688]
    transactions: ProgressiveList[Transaction]
    withdrawals: ProgressiveList[Withdrawal]
    blob_gas_used: uint64
    excess_blob_gas: uint64
    block_access_list: BlockAccessList
    slot_number: uint64


class ExecutionPayloadBid(ProgressiveContainer(active_fields=[1] * 12)):
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
    # [Modified in Gloas:EIP7688]
    blob_kzg_commitments: ProgressiveList[KZGCommitment]
    execution_requests_root: Root


class SignedExecutionPayloadBid(Container):
    message: ExecutionPayloadBid
    signature: BLSSignature


class ExecutionPayloadEnvelope(ProgressiveContainer(active_fields=[1] * 5)):
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
    target_gas_limit: uint64


class SignedProposerPreferences(Container):
    """Signed proposer preferences."""
    message: ProposerPreferences
    signature: BLSSignature


# [Modified in Gloas:EIP7688]
class BeaconBlockBody(ProgressiveContainer(active_fields=[1] * 13)):
    """Gloas BeaconBlockBody (ePBS)."""
    randao_reveal: BLSSignature
    eth1_data: Eth1Data
    graffiti: Bytes32
    proposer_slashings: ProgressiveList[ProposerSlashing]
    attester_slashings: ProgressiveList[AttesterSlashing]
    attestations: ProgressiveList[Attestation]
    deposits: ProgressiveList[Deposit]
    voluntary_exits: ProgressiveList[SignedVoluntaryExit]
    sync_aggregate: SyncAggregate
    bls_to_execution_changes: ProgressiveList[SignedBLSToExecutionChange]
    signed_execution_payload_bid: SignedExecutionPayloadBid
    payload_attestations: ProgressiveList[PayloadAttestation]
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


# [Modified in Gloas:EIP7688]
class BeaconState(ProgressiveContainer(active_fields=[1] * 46)):
    """Gloas BeaconState (ePBS).

    Field order matches consensus-specs master + PR #4630 (EIP-7688) exactly.
    Critical for SSZ correctness.
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
    # [Modified in Gloas:EIP7688]
    validators: ProgressiveList[Validator]
    balances: ProgressiveList[Gwei]
    randao_mixes: Vector[Bytes32, EPOCHS_PER_HISTORICAL_VECTOR()]
    slashings: Vector[Gwei, EPOCHS_PER_SLASHINGS_VECTOR()]
    # [Modified in Gloas:EIP7688]
    previous_epoch_participation: ProgressiveList[ParticipationFlags]
    current_epoch_participation: ProgressiveList[ParticipationFlags]
    justification_bits: Bitvector[JUSTIFICATION_BITS_LENGTH]
    previous_justified_checkpoint: Checkpoint
    current_justified_checkpoint: Checkpoint
    finalized_checkpoint: Checkpoint
    # [Modified in Gloas:EIP7688]
    inactivity_scores: ProgressiveList[uint64]
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
    # [Modified in Gloas:EIP7688]
    pending_deposits: ProgressiveList[PendingDeposit]
    pending_partial_withdrawals: ProgressiveList[PendingPartialWithdrawal]
    pending_consolidations: ProgressiveList[PendingConsolidation]
    proposer_lookahead: Vector[ValidatorIndex, proposer_lookahead_length()]
    # [New in Gloas:EIP7732] ePBS additions follow
    builders: ProgressiveList[Builder]
    next_withdrawal_builder_index: BuilderIndex
    execution_payload_availability: Bitvector[SLOTS_PER_HISTORICAL_ROOT()]
    builder_pending_payments: Vector[BuilderPendingPayment, 2 * SLOTS_PER_EPOCH()]
    builder_pending_withdrawals: ProgressiveList[BuilderPendingWithdrawal]
    latest_execution_payload_bid: ExecutionPayloadBid
    payload_expected_withdrawals: ProgressiveList[Withdrawal]
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
    # [Modified in Gloas:EIP7688] progressive BeaconState moves the gindices
    finality_branch: Vector[Bytes32, FINALIZED_ROOT_DEPTH_GLOAS]
    sync_aggregate: SyncAggregate
    signature_slot: Slot


class LightClientUpdate(Container):
    attested_header: LightClientHeader
    next_sync_committee: SyncCommittee
    # [Modified in Gloas:EIP7688]
    next_sync_committee_branch: Vector[Bytes32, NEXT_SYNC_COMMITTEE_DEPTH_GLOAS]
    finalized_header: LightClientHeader
    finality_branch: Vector[Bytes32, FINALIZED_ROOT_DEPTH_GLOAS]
    sync_aggregate: SyncAggregate
    signature_slot: Slot


class LightClientBootstrap(Container):
    header: LightClientHeader
    current_sync_committee: SyncCommittee
    # [Modified in Gloas:EIP7688]
    current_sync_committee_branch: Vector[Bytes32, CURRENT_SYNC_COMMITTEE_DEPTH_GLOAS]


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
    "AggregationBits",
    "AttestingIndices",
    "Transaction",
    "Attestation",
    "IndexedAttestation",
    "AttesterSlashing",
    "AggregateAndProof",
    "SignedAggregateAndProof",
    "DepositRequests",
    "WithdrawalRequests",
    "ConsolidationRequests",
    "BuilderDepositRequest",
    "BuilderExitRequest",
    "BuilderDepositRequests",
    "BuilderExitRequests",
    "ExecutionRequests",
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
