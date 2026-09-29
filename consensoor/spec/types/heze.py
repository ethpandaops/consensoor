"""Heze SSZ types (EIP-7805: Fork-choice enforced Inclusion Lists).

Heze only modifies ``ExecutionPayloadBid`` (adds ``inclusion_list_bits``).
Every container that transitively embeds the bid (``SignedExecutionPayloadBid``,
``BeaconBlockBody``, ``BeaconBlock``, ``SignedBeaconBlock``, ``BeaconState``)
gets a Heze-local class so its SSZ schema follows. Everything else is the
Gloas type re-exported unchanged.

EIP-8198 (quick slots) is built on Heze and does not change any container,
so these are also the EIP-8198 types.
"""

from .base import (
    Container, Vector, List, Bitvector,
    ProgressiveContainer, ProgressiveList,
    uint64, Bytes32, ByteVector, BLSSignature,
    Slot, Epoch, ValidatorIndex, Gwei, Root, Hash32, ExecutionAddress,
    ParticipationFlags, KZGCommitment, WithdrawalIndex,
)
from .phase0 import (
    Validator, Eth1Data, BeaconBlockHeader,
    ProposerSlashing, Deposit, SignedVoluntaryExit,
)
from .altair import SyncCommittee, SyncAggregate
from .capella import Withdrawal, SignedBLSToExecutionChange, HistoricalSummary
from .electra import PendingDeposit, PendingPartialWithdrawal, PendingConsolidation
from .fulu import proposer_lookahead_length
from .gloas import (
    BuilderIndex, Builder, BuilderPendingWithdrawal, BuilderPendingPayment,
    PayloadAttestation, AttesterSlashing, Attestation, ExecutionRequests,
    Transaction,
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
    INCLUSION_LIST_COMMITTEE_SIZE,
    MAX_REQUEST_INCLUSION_LIST,
)
from .base import Checkpoint, Fork

# [New in Heze:EIP7805]
InclusionListBits = Bitvector[INCLUSION_LIST_COMMITTEE_SIZE]
InclusionListCommittee = Vector[ValidatorIndex, INCLUSION_LIST_COMMITTEE_SIZE]
Transactions = ProgressiveList[Transaction]


class InclusionList(Container):
    slot: Slot
    validator_index: ValidatorIndex
    dependent_root: Root
    transactions: Transactions


class SignedInclusionList(Container):
    message: InclusionList
    signature: BLSSignature


# p2p: InclusionListsByIndices v1
SignedInclusionLists = List[SignedInclusionList, MAX_REQUEST_INCLUSION_LIST]


class InclusionListsByIndicesRequest(Container):
    slot: Slot
    dependent_root: Root
    indices: InclusionListBits


# [Modified in Heze:EIP7805]
class ExecutionPayloadBid(ProgressiveContainer(active_fields=[1] * 13)):
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
    blob_kzg_commitments: ProgressiveList[KZGCommitment]
    execution_requests_root: Root
    # [New in Heze:EIP7805]
    inclusion_list_bits: InclusionListBits


# [Modified in Heze:EIP7805]
class SignedExecutionPayloadBid(Container):
    message: ExecutionPayloadBid
    signature: BLSSignature


class BeaconBlockBody(ProgressiveContainer(active_fields=[1] * 13)):
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
    # [Modified in Heze:EIP7805]
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


class BeaconState(ProgressiveContainer(active_fields=[1] * 46)):
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
    current_sync_committee: SyncCommittee
    next_sync_committee: SyncCommittee
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
    # [Modified in Heze:EIP7805]
    latest_execution_payload_bid: ExecutionPayloadBid
    payload_expected_withdrawals: ProgressiveList[Withdrawal]
    ptc_window: Vector[Vector[ValidatorIndex, PTC_SIZE()], (2 + MIN_SEED_LOOKAHEAD) * SLOTS_PER_EPOCH()]


__all__ = [
    "InclusionListBits",
    "InclusionListCommittee",
    "Transactions",
    "InclusionList",
    "SignedInclusionList",
    "SignedInclusionLists",
    "InclusionListsByIndicesRequest",
    "ExecutionPayloadBid",
    "SignedExecutionPayloadBid",
    "BeaconBlockBody",
    "BeaconBlock",
    "SignedBeaconBlock",
    "BeaconState",
]
