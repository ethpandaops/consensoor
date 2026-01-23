"""Capella SSZ types (adds withdrawals)."""

from .base import (
    Container, Vector, List,
    Bitvector,
    uint8, uint64, uint256,
    Bytes32, ByteVector,
    Hash32, ExecutionAddress, ValidatorIndex, Gwei, BLSPubkey, BLSSignature,
    WithdrawalIndex, Transaction, Slot, Root,
    Checkpoint, Fork, ParticipationFlags,
)
from .phase0 import (
    Validator, Eth1Data, BeaconBlockHeader,
    ProposerSlashing, Deposit, SignedVoluntaryExit,
    Phase0Attestation, Phase0AttesterSlashing,
)
from .altair import (
    SyncCommittee, SyncAggregate,
    LightClientHeader as AltairLightClientHeader,
)
from ..constants import (
    BYTES_PER_LOGS_BLOOM,
    MAX_EXTRA_DATA_BYTES,
    MAX_WITHDRAWALS_PER_PAYLOAD,
    MAX_BLS_TO_EXECUTION_CHANGES,
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


class Withdrawal(Container):
    index: WithdrawalIndex
    validator_index: ValidatorIndex
    address: ExecutionAddress
    amount: Gwei


class BLSToExecutionChange(Container):
    validator_index: ValidatorIndex
    from_bls_pubkey: BLSPubkey
    to_execution_address: ExecutionAddress


class SignedBLSToExecutionChange(Container):
    message: BLSToExecutionChange
    signature: BLSSignature


class HistoricalSummary(Container):
    block_summary_root: Bytes32
    state_summary_root: Bytes32


class ExecutionPayloadHeaderCapella(Container):
    """Capella ExecutionPayloadHeader (with withdrawals_root)."""
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


class CapellaLightClientHeader(Container):
    beacon: BeaconBlockHeader
    execution: ExecutionPayloadHeaderCapella
    execution_branch: Vector[Bytes32, EXECUTION_PAYLOAD_DEPTH]


class CapellaLightClientBootstrap(Container):
    header: CapellaLightClientHeader
    current_sync_committee: SyncCommittee
    current_sync_committee_branch: Vector[Bytes32, CURRENT_SYNC_COMMITTEE_DEPTH]


class CapellaLightClientUpdate(Container):
    attested_header: CapellaLightClientHeader
    next_sync_committee: SyncCommittee
    next_sync_committee_branch: Vector[Bytes32, NEXT_SYNC_COMMITTEE_DEPTH]
    finalized_header: CapellaLightClientHeader
    finality_branch: Vector[Bytes32, FINALIZED_ROOT_DEPTH]
    sync_aggregate: SyncAggregate
    signature_slot: Slot


class CapellaLightClientFinalityUpdate(Container):
    attested_header: CapellaLightClientHeader
    finalized_header: CapellaLightClientHeader
    finality_branch: Vector[Bytes32, FINALIZED_ROOT_DEPTH]
    sync_aggregate: SyncAggregate
    signature_slot: Slot


class CapellaLightClientOptimisticUpdate(Container):
    attested_header: CapellaLightClientHeader
    sync_aggregate: SyncAggregate
    signature_slot: Slot


class ExecutionPayloadCapella(Container):
    """Capella ExecutionPayload (with withdrawals)."""
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


class CapellaBeaconBlockBody(Container):
    randao_reveal: BLSSignature
    eth1_data: Eth1Data
    graffiti: Bytes32
    proposer_slashings: List[ProposerSlashing, MAX_PROPOSER_SLASHINGS]
    attester_slashings: List[Phase0AttesterSlashing, MAX_ATTESTER_SLASHINGS_PRE_ELECTRA]
    attestations: List[Phase0Attestation, MAX_ATTESTATIONS_PRE_ELECTRA]
    deposits: List[Deposit, MAX_DEPOSITS]
    voluntary_exits: List[SignedVoluntaryExit, MAX_VOLUNTARY_EXITS]
    sync_aggregate: SyncAggregate
    execution_payload: ExecutionPayloadCapella
    bls_to_execution_changes: List[SignedBLSToExecutionChange, MAX_BLS_TO_EXECUTION_CHANGES]


class CapellaBeaconBlock(Container):
    slot: Slot
    proposer_index: ValidatorIndex
    parent_root: Root
    state_root: Root
    body: CapellaBeaconBlockBody


class SignedCapellaBeaconBlock(Container):
    message: CapellaBeaconBlock
    signature: BLSSignature


class CapellaBeaconState(Container):
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
    latest_execution_payload_header: ExecutionPayloadHeaderCapella
    next_withdrawal_index: WithdrawalIndex
    next_withdrawal_validator_index: ValidatorIndex
    historical_summaries: List[HistoricalSummary, HISTORICAL_ROOTS_LIMIT]


__all__ = [
    "Withdrawal",
    "BLSToExecutionChange",
    "SignedBLSToExecutionChange",
    "HistoricalSummary",
    "ExecutionPayloadHeaderCapella",
    "CapellaLightClientHeader",
    "CapellaLightClientBootstrap",
    "CapellaLightClientUpdate",
    "CapellaLightClientFinalityUpdate",
    "CapellaLightClientOptimisticUpdate",
    "ExecutionPayloadCapella",
    "CapellaBeaconBlockBody",
    "CapellaBeaconBlock",
    "SignedCapellaBeaconBlock",
    "CapellaBeaconState",
]
