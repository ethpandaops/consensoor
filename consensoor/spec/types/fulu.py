"""Fulu SSZ types (adds PeerDAS and proposer lookahead)."""

from .base import (
    Container, Vector, List,
    Bitvector, ByteVector,
    uint64,
    Bytes32, BLSPubkey,
    Slot, Epoch, ValidatorIndex, Gwei, Root,
    ParticipationFlags, KZGCommitment, KZGProof,
)
from .phase0 import Validator, Eth1Data, BeaconBlockHeader, SignedBeaconBlockHeader
from .altair import SyncCommittee
from .capella import HistoricalSummary
from .deneb import ExecutionPayloadHeader
from .electra import PendingDeposit, PendingPartialWithdrawal, PendingConsolidation
from ..constants import (
    SLOTS_PER_EPOCH,
    SLOTS_PER_HISTORICAL_ROOT,
    EPOCHS_PER_HISTORICAL_VECTOR,
    EPOCHS_PER_SLASHINGS_VECTOR,
    EPOCHS_PER_ETH1_VOTING_PERIOD,
    HISTORICAL_ROOTS_LIMIT,
    VALIDATOR_REGISTRY_LIMIT,
    JUSTIFICATION_BITS_LENGTH,
    PENDING_DEPOSITS_LIMIT,
    PENDING_PARTIAL_WITHDRAWALS_LIMIT,
    PENDING_CONSOLIDATIONS_LIMIT,
    MIN_SEED_LOOKAHEAD,
    FIELD_ELEMENTS_PER_CELL,
    NUMBER_OF_COLUMNS,
    MAX_BLOB_COMMITMENTS_PER_BLOCK,
    KZG_COMMITMENTS_INCLUSION_PROOF_DEPTH_ELECTRA,
)
from .base import Checkpoint, Fork

BYTES_PER_CELL = FIELD_ELEMENTS_PER_CELL * 32


class Cell(ByteVector[BYTES_PER_CELL]):
    pass


class DataColumnSidecar(Container):
    index: uint64
    column: List[Cell, MAX_BLOB_COMMITMENTS_PER_BLOCK]
    kzg_commitments: List[KZGCommitment, MAX_BLOB_COMMITMENTS_PER_BLOCK]
    kzg_proofs: List[KZGProof, MAX_BLOB_COMMITMENTS_PER_BLOCK]
    signed_block_header: SignedBeaconBlockHeader
    kzg_commitments_inclusion_proof: Vector[Bytes32, KZG_COMMITMENTS_INCLUSION_PROOF_DEPTH_ELECTRA]


class DataColumnsByRootIdentifier(Container):
    block_root: Root
    columns: List[uint64, NUMBER_OF_COLUMNS]


class MatrixEntry(Container):
    cell: Cell
    kzg_proof: KZGProof
    column_index: uint64
    row_index: uint64


def proposer_lookahead_length() -> int:
    """Length of proposer_lookahead Vector: (MIN_SEED_LOOKAHEAD + 1) * SLOTS_PER_EPOCH."""
    return (MIN_SEED_LOOKAHEAD + 1) * SLOTS_PER_EPOCH()


class FuluBeaconState(Container):
    """Fulu BeaconState (adds proposer_lookahead as Vector)."""
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
    # Fulu addition: proposer lookahead as Vector (fixed size)
    proposer_lookahead: Vector[ValidatorIndex, proposer_lookahead_length()]


__all__ = [
    "Cell",
    "DataColumnSidecar",
    "DataColumnsByRootIdentifier",
    "MatrixEntry",
    "FuluBeaconState",
    "proposer_lookahead_length",
]
