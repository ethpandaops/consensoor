"""Phase 0 SSZ types."""

from .base import (
    Container, Vector, List,
    Bitvector, Bitlist,
    uint64, boolean,
    Bytes32, BLSPubkey, BLSSignature,
    Slot, Epoch, ValidatorIndex, Gwei, Root, Hash32,
    Checkpoint, Fork,
)
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
    MAX_ATTESTER_SLASHINGS_PRE_ELECTRA,
    MAX_ATTESTATIONS,
    MAX_DEPOSITS,
    MAX_VOLUNTARY_EXITS,
    MAX_VALIDATORS_PER_COMMITTEE,
)


class Validator(Container):
    pubkey: BLSPubkey
    withdrawal_credentials: Bytes32
    effective_balance: Gwei
    slashed: boolean
    activation_eligibility_epoch: Epoch
    activation_epoch: Epoch
    exit_epoch: Epoch
    withdrawable_epoch: Epoch


class AttestationData(Container):
    slot: Slot
    index: uint64  # CommitteeIndex
    beacon_block_root: Root
    source: Checkpoint
    target: Checkpoint


class Eth1Data(Container):
    deposit_root: Root
    deposit_count: uint64
    block_hash: Hash32


class BeaconBlockHeader(Container):
    slot: Slot
    proposer_index: ValidatorIndex
    parent_root: Root
    state_root: Root
    body_root: Root


class SignedBeaconBlockHeader(Container):
    message: BeaconBlockHeader
    signature: BLSSignature


class ProposerSlashing(Container):
    signed_header_1: SignedBeaconBlockHeader
    signed_header_2: SignedBeaconBlockHeader


class DepositData(Container):
    pubkey: BLSPubkey
    withdrawal_credentials: Bytes32
    amount: Gwei
    signature: BLSSignature


class Deposit(Container):
    proof: Vector[Bytes32, 33]
    data: DepositData


class VoluntaryExit(Container):
    epoch: Epoch
    validator_index: ValidatorIndex


class SignedVoluntaryExit(Container):
    message: VoluntaryExit
    signature: BLSSignature


class Phase0Attestation(Container):
    aggregation_bits: Bitlist[MAX_VALIDATORS_PER_COMMITTEE()]
    data: AttestationData
    signature: BLSSignature


class Phase0IndexedAttestation(Container):
    attesting_indices: List[ValidatorIndex, MAX_VALIDATORS_PER_COMMITTEE()]
    data: AttestationData
    signature: BLSSignature


class Phase0AttesterSlashing(Container):
    attestation_1: Phase0IndexedAttestation
    attestation_2: Phase0IndexedAttestation


class AggregateAndProof(Container):
    aggregator_index: ValidatorIndex
    aggregate: Phase0Attestation
    selection_proof: BLSSignature


class SignedAggregateAndProof(Container):
    message: AggregateAndProof
    signature: BLSSignature


class DepositMessage(Container):
    pubkey: BLSPubkey
    withdrawal_credentials: Bytes32
    amount: Gwei


class Eth1Block(Container):
    timestamp: uint64
    deposit_root: Root
    deposit_count: uint64


class HistoricalBatch(Container):
    block_roots: Vector[Root, SLOTS_PER_HISTORICAL_ROOT()]
    state_roots: Vector[Root, SLOTS_PER_HISTORICAL_ROOT()]


class PendingAttestation(Container):
    aggregation_bits: Bitlist[MAX_VALIDATORS_PER_COMMITTEE()]
    data: AttestationData
    inclusion_delay: Slot
    proposer_index: ValidatorIndex


class Phase0BeaconBlockBody(Container):
    randao_reveal: BLSSignature
    eth1_data: Eth1Data
    graffiti: Bytes32
    proposer_slashings: List[ProposerSlashing, MAX_PROPOSER_SLASHINGS]
    attester_slashings: List[Phase0AttesterSlashing, MAX_ATTESTER_SLASHINGS_PRE_ELECTRA]
    attestations: List[Phase0Attestation, MAX_ATTESTATIONS()]
    deposits: List[Deposit, MAX_DEPOSITS]
    voluntary_exits: List[SignedVoluntaryExit, MAX_VOLUNTARY_EXITS]


class Phase0BeaconBlock(Container):
    slot: Slot
    proposer_index: ValidatorIndex
    parent_root: Root
    state_root: Root
    body: Phase0BeaconBlockBody


class SignedPhase0BeaconBlock(Container):
    message: Phase0BeaconBlock
    signature: BLSSignature


class Phase0BeaconState(Container):
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
    previous_epoch_attestations: List[PendingAttestation, MAX_ATTESTATIONS() * SLOTS_PER_EPOCH()]
    current_epoch_attestations: List[PendingAttestation, MAX_ATTESTATIONS() * SLOTS_PER_EPOCH()]
    justification_bits: Bitvector[JUSTIFICATION_BITS_LENGTH]
    previous_justified_checkpoint: Checkpoint
    current_justified_checkpoint: Checkpoint
    finalized_checkpoint: Checkpoint


__all__ = [
    "Validator",
    "AttestationData",
    "Eth1Data",
    "BeaconBlockHeader",
    "SignedBeaconBlockHeader",
    "ProposerSlashing",
    "DepositData",
    "Deposit",
    "VoluntaryExit",
    "SignedVoluntaryExit",
    "Phase0Attestation",
    "Phase0IndexedAttestation",
    "Phase0AttesterSlashing",
    "AggregateAndProof",
    "SignedAggregateAndProof",
    "DepositMessage",
    "Eth1Block",
    "HistoricalBatch",
    "PendingAttestation",
    "Phase0BeaconBlockBody",
    "Phase0BeaconBlock",
    "SignedPhase0BeaconBlock",
    "Phase0BeaconState",
]
