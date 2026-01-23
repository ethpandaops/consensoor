"""Altair SSZ types."""

from .base import (
    Container, Vector, List,
    Bitvector, Bitlist,
    uint64, boolean,
    Bytes32, BLSPubkey, BLSSignature,
    Slot, Epoch, ValidatorIndex, Gwei, Root, Hash32,
    Checkpoint, Fork, ParticipationFlags,
)
from .phase0 import (
    Validator, AttestationData, Eth1Data, BeaconBlockHeader,
    ProposerSlashing, Deposit, SignedVoluntaryExit,
    Phase0Attestation, Phase0AttesterSlashing,
)
from ..constants import (
    SYNC_COMMITTEE_SIZE,
    SYNC_COMMITTEE_SUBNET_COUNT,
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
    FINALIZED_ROOT_DEPTH,
    CURRENT_SYNC_COMMITTEE_DEPTH,
    NEXT_SYNC_COMMITTEE_DEPTH,
)


class SyncCommittee(Container):
    pubkeys: Vector[BLSPubkey, SYNC_COMMITTEE_SIZE()]
    aggregate_pubkey: BLSPubkey


class SyncAggregate(Container):
    sync_committee_bits: Bitvector[SYNC_COMMITTEE_SIZE()]
    sync_committee_signature: BLSSignature


class SyncCommitteeMessage(Container):
    slot: Slot
    beacon_block_root: Root
    validator_index: ValidatorIndex
    signature: BLSSignature


def _sync_subcommittee_size() -> int:
    return SYNC_COMMITTEE_SIZE() // SYNC_COMMITTEE_SUBNET_COUNT


class SyncCommitteeContribution(Container):
    slot: Slot
    beacon_block_root: Root
    subcommittee_index: uint64
    aggregation_bits: Bitvector[_sync_subcommittee_size()]
    signature: BLSSignature


class ContributionAndProof(Container):
    aggregator_index: ValidatorIndex
    contribution: SyncCommitteeContribution
    selection_proof: BLSSignature


class SignedContributionAndProof(Container):
    message: ContributionAndProof
    signature: BLSSignature


class SyncAggregatorSelectionData(Container):
    slot: Slot
    subcommittee_index: uint64


class LightClientHeader(Container):
    beacon: BeaconBlockHeader


class LightClientBootstrap(Container):
    header: LightClientHeader
    current_sync_committee: SyncCommittee
    current_sync_committee_branch: Vector[Bytes32, CURRENT_SYNC_COMMITTEE_DEPTH]


class LightClientUpdate(Container):
    attested_header: LightClientHeader
    next_sync_committee: SyncCommittee
    next_sync_committee_branch: Vector[Bytes32, NEXT_SYNC_COMMITTEE_DEPTH]
    finalized_header: LightClientHeader
    finality_branch: Vector[Bytes32, FINALIZED_ROOT_DEPTH]
    sync_aggregate: SyncAggregate
    signature_slot: Slot


class LightClientFinalityUpdate(Container):
    attested_header: LightClientHeader
    finalized_header: LightClientHeader
    finality_branch: Vector[Bytes32, FINALIZED_ROOT_DEPTH]
    sync_aggregate: SyncAggregate
    signature_slot: Slot


class LightClientOptimisticUpdate(Container):
    attested_header: LightClientHeader
    sync_aggregate: SyncAggregate
    signature_slot: Slot


class AltairBeaconBlockBody(Container):
    randao_reveal: BLSSignature
    eth1_data: Eth1Data
    graffiti: Bytes32
    proposer_slashings: List[ProposerSlashing, MAX_PROPOSER_SLASHINGS]
    attester_slashings: List[Phase0AttesterSlashing, MAX_ATTESTER_SLASHINGS_PRE_ELECTRA]
    attestations: List[Phase0Attestation, MAX_ATTESTATIONS_PRE_ELECTRA]
    deposits: List[Deposit, MAX_DEPOSITS]
    voluntary_exits: List[SignedVoluntaryExit, MAX_VOLUNTARY_EXITS]
    sync_aggregate: SyncAggregate


class AltairBeaconBlock(Container):
    slot: Slot
    proposer_index: ValidatorIndex
    parent_root: Root
    state_root: Root
    body: AltairBeaconBlockBody


class SignedAltairBeaconBlock(Container):
    message: AltairBeaconBlock
    signature: BLSSignature


class AltairBeaconState(Container):
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


__all__ = [
    "SyncCommittee",
    "SyncAggregate",
    "SyncCommitteeMessage",
    "SyncCommitteeContribution",
    "ContributionAndProof",
    "SignedContributionAndProof",
    "SyncAggregatorSelectionData",
    "LightClientHeader",
    "LightClientBootstrap",
    "LightClientUpdate",
    "LightClientFinalityUpdate",
    "LightClientOptimisticUpdate",
    "AltairBeaconBlockBody",
    "AltairBeaconBlock",
    "SignedAltairBeaconBlock",
    "AltairBeaconState",
]
