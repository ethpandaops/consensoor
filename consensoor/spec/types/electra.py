"""Electra SSZ types."""

from .base import (
    Container, Vector, List,
    Bitvector, Bitlist,
    uint64, boolean,
    Bytes32, BLSPubkey, BLSSignature,
    Slot, Epoch, ValidatorIndex, Gwei, Root, Hash32, ExecutionAddress,
    ParticipationFlags, KZGCommitment,
)
from .phase0 import (
    Validator, AttestationData, Eth1Data, BeaconBlockHeader,
    SignedBeaconBlockHeader, ProposerSlashing, Deposit, SignedVoluntaryExit,
)
from .altair import SyncCommittee, SyncAggregate
from .capella import Withdrawal, SignedBLSToExecutionChange, HistoricalSummary
from .deneb import ExecutionPayloadHeader, ExecutionPayload, DenebLightClientHeader
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
    MAX_VALIDATORS_PER_COMMITTEE,
    MAX_COMMITTEES_PER_SLOT,
    PENDING_DEPOSITS_LIMIT,
    PENDING_PARTIAL_WITHDRAWALS_LIMIT,
    PENDING_CONSOLIDATIONS_LIMIT,
    MAX_BLOB_COMMITMENTS_PER_BLOCK,
    MAX_DEPOSIT_REQUESTS_PER_PAYLOAD,
    MAX_WITHDRAWAL_REQUESTS_PER_PAYLOAD,
    MAX_CONSOLIDATION_REQUESTS_PER_PAYLOAD,
    EXECUTION_PAYLOAD_DEPTH,
    FINALIZED_ROOT_DEPTH_ELECTRA,
    CURRENT_SYNC_COMMITTEE_DEPTH_ELECTRA,
    NEXT_SYNC_COMMITTEE_DEPTH_ELECTRA,
)
from .base import Checkpoint, Fork


class Attestation(Container):
    aggregation_bits: Bitlist[MAX_VALIDATORS_PER_COMMITTEE() * MAX_COMMITTEES_PER_SLOT()]
    data: AttestationData
    signature: BLSSignature
    committee_bits: Bitvector[MAX_COMMITTEES_PER_SLOT()]


class IndexedAttestation(Container):
    attesting_indices: List[ValidatorIndex, MAX_VALIDATORS_PER_COMMITTEE() * MAX_COMMITTEES_PER_SLOT()]
    data: AttestationData
    signature: BLSSignature


class AttesterSlashing(Container):
    attestation_1: IndexedAttestation
    attestation_2: IndexedAttestation


class SingleAttestation(Container):
    committee_index: uint64
    attester_index: ValidatorIndex
    data: AttestationData
    signature: BLSSignature


class ElectraAggregateAndProof(Container):
    aggregator_index: ValidatorIndex
    aggregate: Attestation
    selection_proof: BLSSignature


class SignedElectraAggregateAndProof(Container):
    message: ElectraAggregateAndProof
    signature: BLSSignature


class PendingDeposit(Container):
    pubkey: BLSPubkey
    withdrawal_credentials: Bytes32
    amount: Gwei
    signature: BLSSignature
    slot: Slot


class PendingPartialWithdrawal(Container):
    validator_index: ValidatorIndex
    amount: Gwei
    withdrawable_epoch: Epoch


class PendingConsolidation(Container):
    source_index: ValidatorIndex
    target_index: ValidatorIndex


class DepositRequest(Container):
    pubkey: BLSPubkey
    withdrawal_credentials: Bytes32
    amount: uint64
    signature: BLSSignature
    index: uint64


class WithdrawalRequest(Container):
    source_address: ExecutionAddress
    validator_pubkey: BLSPubkey
    amount: uint64


class ConsolidationRequest(Container):
    source_address: ExecutionAddress
    source_pubkey: BLSPubkey
    target_pubkey: BLSPubkey


class ExecutionRequests(Container):
    deposits: List[DepositRequest, MAX_DEPOSIT_REQUESTS_PER_PAYLOAD]
    withdrawals: List[WithdrawalRequest, MAX_WITHDRAWAL_REQUESTS_PER_PAYLOAD]
    consolidations: List[ConsolidationRequest, MAX_CONSOLIDATION_REQUESTS_PER_PAYLOAD]


class ElectraBeaconBlockBody(Container):
    """Electra/Fulu BeaconBlockBody (with execution payload)."""
    randao_reveal: BLSSignature
    eth1_data: Eth1Data
    graffiti: Bytes32
    proposer_slashings: List[ProposerSlashing, MAX_PROPOSER_SLASHINGS]
    attester_slashings: List[AttesterSlashing, MAX_ATTESTER_SLASHINGS_ELECTRA]
    attestations: List[Attestation, MAX_ATTESTATIONS_ELECTRA]
    deposits: List[Deposit, MAX_DEPOSITS]
    voluntary_exits: List[SignedVoluntaryExit, MAX_VOLUNTARY_EXITS]
    sync_aggregate: SyncAggregate
    execution_payload: ExecutionPayload
    bls_to_execution_changes: List[SignedBLSToExecutionChange, MAX_BLS_TO_EXECUTION_CHANGES]
    blob_kzg_commitments: List[KZGCommitment, MAX_BLOB_COMMITMENTS_PER_BLOCK]
    execution_requests: ExecutionRequests


class ElectraBeaconBlock(Container):
    """Electra/Fulu BeaconBlock."""
    slot: Slot
    proposer_index: ValidatorIndex
    parent_root: Root
    state_root: Root
    body: ElectraBeaconBlockBody


class SignedElectraBeaconBlock(Container):
    """Signed Electra/Fulu BeaconBlock."""
    message: ElectraBeaconBlock
    signature: BLSSignature


class ElectraBeaconState(Container):
    """Electra BeaconState."""
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
    next_withdrawal_index: uint64
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


# Electra LightClient types (use Deneb header, but Electra branch depths)
class ElectraLightClientHeader(Container):
    """Electra LightClientHeader (same as Deneb, reuses ExecutionPayloadHeader)."""
    beacon: BeaconBlockHeader
    execution: ExecutionPayloadHeader
    execution_branch: Vector[Bytes32, EXECUTION_PAYLOAD_DEPTH]


class ElectraLightClientBootstrap(Container):
    header: ElectraLightClientHeader
    current_sync_committee: SyncCommittee
    current_sync_committee_branch: Vector[Bytes32, CURRENT_SYNC_COMMITTEE_DEPTH_ELECTRA]


class ElectraLightClientUpdate(Container):
    attested_header: ElectraLightClientHeader
    next_sync_committee: SyncCommittee
    next_sync_committee_branch: Vector[Bytes32, NEXT_SYNC_COMMITTEE_DEPTH_ELECTRA]
    finalized_header: ElectraLightClientHeader
    finality_branch: Vector[Bytes32, FINALIZED_ROOT_DEPTH_ELECTRA]
    sync_aggregate: SyncAggregate
    signature_slot: Slot


class ElectraLightClientFinalityUpdate(Container):
    attested_header: ElectraLightClientHeader
    finalized_header: ElectraLightClientHeader
    finality_branch: Vector[Bytes32, FINALIZED_ROOT_DEPTH_ELECTRA]
    sync_aggregate: SyncAggregate
    signature_slot: Slot


class ElectraLightClientOptimisticUpdate(Container):
    attested_header: ElectraLightClientHeader
    sync_aggregate: SyncAggregate
    signature_slot: Slot


__all__ = [
    "Attestation",
    "IndexedAttestation",
    "AttesterSlashing",
    "SingleAttestation",
    "ElectraAggregateAndProof",
    "SignedElectraAggregateAndProof",
    "PendingDeposit",
    "PendingPartialWithdrawal",
    "PendingConsolidation",
    "DepositRequest",
    "WithdrawalRequest",
    "ConsolidationRequest",
    "ExecutionRequests",
    "ElectraBeaconBlockBody",
    "ElectraBeaconBlock",
    "SignedElectraBeaconBlock",
    "ElectraBeaconState",
    "ElectraLightClientHeader",
    "ElectraLightClientBootstrap",
    "ElectraLightClientUpdate",
    "ElectraLightClientFinalityUpdate",
    "ElectraLightClientOptimisticUpdate",
]
