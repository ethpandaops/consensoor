"""PQ SSZ types (leanXMSS signatures, built on Gloas + EIP-8483)."""

from .base import (
    Container, Vector, List, Bitvector,
    ProgressiveContainer, ProgressiveList,
    uint8, uint64, boolean,
    Bytes32, ByteVector, ByteList,
    Slot, Epoch, ValidatorIndex, Gwei, Root, Hash32, ExecutionAddress,
    ParticipationFlags, WithdrawalIndex,
    Checkpoint, Fork,
)
from .phase0 import AttestationData, Eth1Data, BeaconBlockHeader, Deposit
from .capella import Withdrawal, HistoricalSummary
from .electra import (
    PendingDeposit, PendingPartialWithdrawal, PendingConsolidation,
    DepositRequest, WithdrawalRequest, ConsolidationRequest,
)
from .fulu import proposer_lookahead_length
from .gloas import (
    BuilderIndex, BuilderPendingWithdrawal, BuilderPendingPayment,
    PayloadAttestationData, ExecutionPayload, ExecutionPayloadBid,
    ProposerPreferences, BuilderDepositRequest, BuilderExitRequest,
    AggregationBits, AttestingIndices,
    DepositRequests, WithdrawalRequests, ConsolidationRequests,
    BuilderDepositRequests, BuilderExitRequests,
)
from ..constants import (
    SLOTS_PER_EPOCH,
    SLOTS_PER_HISTORICAL_ROOT,
    EPOCHS_PER_HISTORICAL_VECTOR,
    EPOCHS_PER_SLASHINGS_VECTOR,
    EPOCHS_PER_ETH1_VOTING_PERIOD,
    HISTORICAL_ROOTS_LIMIT,
    JUSTIFICATION_BITS_LENGTH,
    PTC_SIZE,
    MIN_SEED_LOOKAHEAD,
    MAX_COMMITTEES_PER_SLOT_PQ,
    XMSS_SIGNATURE_SIZE,
    MAX_STARK_PROOF_SIZE,
    VRF_TREE_DEPTH,
)


def _active_fields(width: int, gaps: tuple[int, ...] = ()) -> list[int]:
    return [0 if index in gaps else 1 for index in range(width)]


XMSSPubkey = Bytes32
XMSSSignature = ByteVector[XMSS_SIGNATURE_SIZE]
VRFProof = Vector[Bytes32, VRF_TREE_DEPTH + 1]
AttestationProof = ByteList[MAX_STARK_PROOF_SIZE]
CommitteeBits = Bitvector[MAX_COMMITTEES_PER_SLOT_PQ()]

AggregationCommittee = ProgressiveList[ValidatorIndex]
AggregationCommittees = ProgressiveList[AggregationCommittee]
AggregationCommitteeWindow = Vector[
    AggregationCommittees, (MIN_SEED_LOOKAHEAD + 2) * SLOTS_PER_EPOCH()
]


def aggregation_committees(committees) -> AggregationCommittees:
    return AggregationCommittees(*[AggregationCommittee(*committee) for committee in committees])


class Validator(Container):
    pubkey: XMSSPubkey
    vrf_root: Root
    withdrawal_credentials: Bytes32
    effective_balance: Gwei
    slashed: boolean
    activation_eligibility_epoch: Epoch
    activation_epoch: Epoch
    exit_epoch: Epoch
    withdrawable_epoch: Epoch


class Builder(Container):
    pubkey: XMSSPubkey
    version: uint8
    execution_address: ExecutionAddress
    balance: Gwei
    deposit_epoch: Epoch
    withdrawable_epoch: Epoch


class ValidatorPreregistrationRequest(Container):
    pubkey: XMSSPubkey
    withdrawal_credentials: Bytes32
    vrf_root: Root


class ValidatorPreregistration(Container):
    pubkey: XMSSPubkey
    withdrawal_credentials: Bytes32
    vrf_root: Root
    expiry_slot: Slot


class BuilderPreregistrationRequest(Container):
    pubkey: XMSSPubkey
    withdrawal_credentials: Bytes32


class BuilderPreregistration(Container):
    pubkey: XMSSPubkey
    withdrawal_credentials: Bytes32
    expiry_slot: Slot


class VoluntaryExitRequest(Container):
    pubkey: XMSSPubkey


ValidatorPreregistrationRequests = ProgressiveList[ValidatorPreregistrationRequest]
BuilderPreregistrationRequests = ProgressiveList[BuilderPreregistrationRequest]
ValidatorPreregistrations = ProgressiveList[ValidatorPreregistration]
BuilderPreregistrations = ProgressiveList[BuilderPreregistration]
VoluntaryExitRequests = ProgressiveList[VoluntaryExitRequest]


class ExecutionRequests(ProgressiveContainer(active_fields=_active_fields(10, (5, 6)))):
    deposits: DepositRequests
    withdrawals: WithdrawalRequests
    consolidations: ConsolidationRequests
    builder_deposits: BuilderDepositRequests
    builder_exits: BuilderExitRequests
    validator_preregistrations: ValidatorPreregistrationRequests
    builder_preregistrations: BuilderPreregistrationRequests
    voluntary_exits: VoluntaryExitRequests


class Attestation(ProgressiveContainer(active_fields=_active_fields(4, (2,)))):
    aggregation_bits: AggregationBits
    data: AttestationData
    committee_bits: CommitteeBits


class IndexedAttestation(ProgressiveContainer(active_fields=_active_fields(3))):
    attesting_indices: AttestingIndices
    data: AttestationData
    proof: AttestationProof


class AttesterSlashing(Container):
    attestation_1: IndexedAttestation
    attestation_2: IndexedAttestation


class SingleAttestation(Container):
    committee_index: uint64
    attester_index: ValidatorIndex
    data: AttestationData
    signature: XMSSSignature


class AggregateAndProof(Container):
    aggregator_index: ValidatorIndex
    aggregate: Attestation
    proof: AttestationProof


class SignedAggregateAndProof(Container):
    message: AggregateAndProof
    signature: XMSSSignature


class PayloadAttestationMessage(Container):
    validator_index: ValidatorIndex
    data: PayloadAttestationData
    signature: XMSSSignature


class SignedBeaconBlockHeader(Container):
    message: BeaconBlockHeader
    signature: XMSSSignature


class ProposerSlashing(Container):
    signed_header_1: SignedBeaconBlockHeader
    signed_header_2: SignedBeaconBlockHeader


class SignedExecutionPayloadBid(Container):
    message: ExecutionPayloadBid
    signature: XMSSSignature


class ExecutionPayloadEnvelope(ProgressiveContainer(active_fields=_active_fields(5))):
    payload: ExecutionPayload
    execution_requests: ExecutionRequests
    builder_index: BuilderIndex
    beacon_block_root: Root
    parent_beacon_block_root: Root


class SignedExecutionPayloadEnvelope(Container):
    message: ExecutionPayloadEnvelope
    signature: XMSSSignature


class SignedProposerPreferences(Container):
    message: ProposerPreferences
    signature: XMSSSignature


class BeaconBlockBody(ProgressiveContainer(active_fields=_active_fields(14, (7, 8, 9, 11)))):
    randao_reveal: VRFProof
    eth1_data: Eth1Data
    graffiti: Bytes32
    proposer_slashings: ProgressiveList[ProposerSlashing]
    attester_slashings: ProgressiveList[AttesterSlashing]
    attestations: ProgressiveList[Attestation]
    deposits: ProgressiveList[Deposit]
    signed_execution_payload_bid: SignedExecutionPayloadBid
    parent_execution_requests: ExecutionRequests
    proof: AttestationProof


class BeaconBlock(Container):
    slot: Slot
    proposer_index: ValidatorIndex
    parent_root: Root
    state_root: Root
    body: BeaconBlockBody


class SignedBeaconBlock(Container):
    message: BeaconBlock
    signature: XMSSSignature


class BeaconState(ProgressiveContainer(active_fields=_active_fields(51, (22, 23, 46, 47)))):
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
    validators: ProgressiveList[Validator]
    balances: ProgressiveList[Gwei]
    randao_mixes: Vector[Bytes32, EPOCHS_PER_HISTORICAL_VECTOR()]
    slashings: Vector[Gwei, EPOCHS_PER_SLASHINGS_VECTOR()]
    previous_epoch_participation: ProgressiveList[ParticipationFlags]
    current_epoch_participation: ProgressiveList[ParticipationFlags]
    justification_bits: Bitvector[JUSTIFICATION_BITS_LENGTH]
    previous_justified_checkpoint: Checkpoint
    current_justified_checkpoint: Checkpoint
    finalized_checkpoint: Checkpoint
    inactivity_scores: ProgressiveList[uint64]
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
    pending_deposits: ProgressiveList[PendingDeposit]
    pending_partial_withdrawals: ProgressiveList[PendingPartialWithdrawal]
    pending_consolidations: ProgressiveList[PendingConsolidation]
    proposer_lookahead: Vector[ValidatorIndex, proposer_lookahead_length()]
    builders: ProgressiveList[Builder]
    next_withdrawal_builder_index: BuilderIndex
    execution_payload_availability: Bitvector[SLOTS_PER_HISTORICAL_ROOT()]
    builder_pending_payments: Vector[BuilderPendingPayment, 2 * SLOTS_PER_EPOCH()]
    builder_pending_withdrawals: ProgressiveList[BuilderPendingWithdrawal]
    latest_execution_payload_bid: ExecutionPayloadBid
    payload_expected_withdrawals: ProgressiveList[Withdrawal]
    ptc_window: Vector[Vector[ValidatorIndex, PTC_SIZE()], (2 + MIN_SEED_LOOKAHEAD) * SLOTS_PER_EPOCH()]
    validator_preregistrations: ValidatorPreregistrations
    builder_preregistrations: BuilderPreregistrations
    aggregation_committee_window: AggregationCommitteeWindow


__all__ = [
    "XMSSPubkey",
    "XMSSSignature",
    "VRFProof",
    "AttestationProof",
    "CommitteeBits",
    "AggregationCommittee",
    "AggregationCommittees",
    "AggregationCommitteeWindow",
    "Validator",
    "Builder",
    "ValidatorPreregistrationRequest",
    "ValidatorPreregistration",
    "BuilderPreregistrationRequest",
    "BuilderPreregistration",
    "VoluntaryExitRequest",
    "ValidatorPreregistrationRequests",
    "BuilderPreregistrationRequests",
    "ValidatorPreregistrations",
    "BuilderPreregistrations",
    "VoluntaryExitRequests",
    "ExecutionRequests",
    "Attestation",
    "IndexedAttestation",
    "AttesterSlashing",
    "SingleAttestation",
    "AggregateAndProof",
    "SignedAggregateAndProof",
    "PayloadAttestationMessage",
    "SignedBeaconBlockHeader",
    "ProposerSlashing",
    "SignedExecutionPayloadBid",
    "ExecutionPayloadEnvelope",
    "SignedExecutionPayloadEnvelope",
    "SignedProposerPreferences",
    "BeaconBlockBody",
    "BeaconBlock",
    "SignedBeaconBlock",
    "BeaconState",
]
