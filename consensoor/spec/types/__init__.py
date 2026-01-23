"""SSZ types for consensus layer.

Types are organized by the fork that introduced them:
- base.py: Basic types and primitives
- phase0.py: Phase 0 types
- altair.py: Altair types (sync committees)
- bellatrix.py: Bellatrix types (execution layer)
- capella.py: Capella types (withdrawals)
- deneb.py: Deneb types (blob gas)
- electra.py: Electra types (pending deposits, consolidations)
- fulu.py: Fulu types (proposer lookahead)
- gloas.py: Gloas types (ePBS)
"""

# Base types
from .base import (
    uint8, uint64, uint256, boolean,
    Bytes4, Bytes20, Bytes32, Bytes48, Bytes96, ByteVector,
    Container, Vector, List,
    Bitvector, Bitlist,
    Slot, Epoch, CommitteeIndex, ValidatorIndex, Gwei,
    Root, Hash32, Version, DomainType, ForkDigest, Domain,
    BLSPubkey, BLSSignature, ExecutionAddress, WithdrawalIndex,
    ParticipationFlags, KZGCommitment, KZGProof,
    Transaction,
    Fork, ForkData, Checkpoint, SigningData,
)

# Phase 0
from .phase0 import (
    Validator,
    AttestationData,
    Eth1Data,
    BeaconBlockHeader,
    SignedBeaconBlockHeader,
    ProposerSlashing,
    DepositData,
    Deposit,
    VoluntaryExit,
    SignedVoluntaryExit,
    Phase0Attestation,
    Phase0IndexedAttestation,
    Phase0AttesterSlashing,
    AggregateAndProof,
    SignedAggregateAndProof,
    DepositMessage,
    Eth1Block,
    HistoricalBatch,
    PendingAttestation,
    Phase0BeaconBlockBody,
    Phase0BeaconBlock,
    SignedPhase0BeaconBlock,
    Phase0BeaconState,
)

# Altair
from .altair import (
    SyncCommittee,
    SyncAggregate,
    SyncCommitteeMessage,
    SyncCommitteeContribution,
    ContributionAndProof,
    SignedContributionAndProof,
    SyncAggregatorSelectionData,
    LightClientHeader,
    LightClientBootstrap,
    LightClientUpdate,
    LightClientFinalityUpdate,
    LightClientOptimisticUpdate,
    AltairBeaconBlockBody,
    AltairBeaconBlock,
    SignedAltairBeaconBlock,
    AltairBeaconState,
)

# Bellatrix
from .bellatrix import (
    PowBlock,
    ExecutionPayloadHeaderBellatrix,
    ExecutionPayloadBellatrix,
    BellatrixBeaconBlockBody,
    BellatrixBeaconBlock,
    SignedBellatrixBeaconBlock,
    BellatrixBeaconState,
)

# Capella
from .capella import (
    Withdrawal,
    BLSToExecutionChange,
    SignedBLSToExecutionChange,
    HistoricalSummary,
    ExecutionPayloadHeaderCapella,
    CapellaLightClientHeader,
    CapellaLightClientBootstrap,
    CapellaLightClientUpdate,
    CapellaLightClientFinalityUpdate,
    CapellaLightClientOptimisticUpdate,
    ExecutionPayloadCapella,
    CapellaBeaconBlockBody,
    CapellaBeaconBlock,
    SignedCapellaBeaconBlock,
    CapellaBeaconState,
)

# Deneb
from .deneb import (
    Blob,
    BlobIdentifier,
    BlobSidecar,
    ExecutionPayloadHeader,
    DenebLightClientHeader,
    DenebLightClientBootstrap,
    DenebLightClientUpdate,
    DenebLightClientFinalityUpdate,
    DenebLightClientOptimisticUpdate,
    ExecutionPayload,
    DenebBeaconBlockBody,
    DenebBeaconBlock,
    SignedDenebBeaconBlock,
    DenebBeaconState,
)

# Electra
from .electra import (
    Attestation,
    IndexedAttestation,
    AttesterSlashing,
    SingleAttestation,
    ElectraAggregateAndProof,
    SignedElectraAggregateAndProof,
    PendingDeposit,
    PendingPartialWithdrawal,
    PendingConsolidation,
    DepositRequest,
    WithdrawalRequest,
    ConsolidationRequest,
    ExecutionRequests,
    ElectraBeaconBlockBody,
    ElectraBeaconBlock,
    SignedElectraBeaconBlock,
    ElectraBeaconState,
    ElectraLightClientHeader,
    ElectraLightClientBootstrap,
    ElectraLightClientUpdate,
    ElectraLightClientFinalityUpdate,
    ElectraLightClientOptimisticUpdate,
)

# Fulu
from .fulu import (
    Cell,
    DataColumnSidecar,
    DataColumnsByRootIdentifier,
    MatrixEntry,
    FuluBeaconState,
)

# Gloas (ePBS)
from .gloas import (
    BuilderIndex,
    Builder,
    BuilderPendingWithdrawal,
    BuilderPendingPayment,
    PayloadAttestationData,
    PayloadAttestation,
    PayloadAttestationMessage,
    IndexedPayloadAttestation,
    ExecutionPayloadBid,
    SignedExecutionPayloadBid,
    ExecutionPayloadEnvelope,
    SignedExecutionPayloadEnvelope,
    ProposerPreferences,
    SignedProposerPreferences,
    BeaconBlockBody,
    BeaconBlock,
    SignedBeaconBlock,
    BeaconState,
)

__all__ = [
    # Base
    "Slot", "Epoch", "CommitteeIndex", "ValidatorIndex", "Gwei",
    "Root", "Hash32", "Version", "Domain",
    "BLSPubkey", "BLSSignature", "ExecutionAddress",
    "KZGCommitment",
    "Fork", "ForkData", "Checkpoint", "SigningData",
    # Phase 0
    "Validator", "AttestationData", "Eth1Data",
    "BeaconBlockHeader", "SignedBeaconBlockHeader",
    "ProposerSlashing", "DepositData", "Deposit",
    "VoluntaryExit", "SignedVoluntaryExit",
    "Phase0Attestation", "Phase0IndexedAttestation", "Phase0AttesterSlashing",
    "AggregateAndProof", "SignedAggregateAndProof",
    "DepositMessage", "Eth1Block", "HistoricalBatch",
    "PendingAttestation",
    "Phase0BeaconBlockBody", "Phase0BeaconBlock", "SignedPhase0BeaconBlock",
    "Phase0BeaconState",
    # Altair
    "SyncCommittee", "SyncAggregate",
    "SyncCommitteeMessage", "SyncCommitteeContribution",
    "ContributionAndProof", "SignedContributionAndProof",
    "SyncAggregatorSelectionData",
    "LightClientHeader", "LightClientBootstrap", "LightClientUpdate",
    "LightClientFinalityUpdate", "LightClientOptimisticUpdate",
    "AltairBeaconBlockBody", "AltairBeaconBlock", "SignedAltairBeaconBlock",
    "AltairBeaconState",
    # Bellatrix
    "PowBlock",
    "ExecutionPayloadHeaderBellatrix", "ExecutionPayloadBellatrix",
    "BellatrixBeaconBlockBody", "BellatrixBeaconBlock", "SignedBellatrixBeaconBlock",
    "BellatrixBeaconState",
    # Capella
    "Withdrawal", "SignedBLSToExecutionChange", "HistoricalSummary",
    "ExecutionPayloadHeaderCapella", "ExecutionPayloadCapella",
    "CapellaBeaconBlockBody", "CapellaBeaconBlock", "SignedCapellaBeaconBlock",
    "CapellaBeaconState",
    # Deneb
    "Blob", "BlobIdentifier", "BlobSidecar",
    "ExecutionPayloadHeader", "ExecutionPayload",
    "DenebBeaconBlockBody", "DenebBeaconBlock", "SignedDenebBeaconBlock",
    "DenebBeaconState",
    # Electra
    "Attestation", "IndexedAttestation", "AttesterSlashing", "SingleAttestation",
    "PendingDeposit", "PendingPartialWithdrawal", "PendingConsolidation",
    "ExecutionRequests",
    "ElectraBeaconBlockBody", "ElectraBeaconBlock", "SignedElectraBeaconBlock",
    "ElectraBeaconState",
    # Fulu
    "Cell", "DataColumnSidecar", "DataColumnsByRootIdentifier", "MatrixEntry",
    "FuluBeaconState",
    # Gloas
    "BuilderIndex", "Builder",
    "BuilderPendingPayment", "BuilderPendingWithdrawal",
    "PayloadAttestationData", "PayloadAttestation", "PayloadAttestationMessage",
    "IndexedPayloadAttestation",
    "ExecutionPayloadBid", "SignedExecutionPayloadBid",
    "ExecutionPayloadEnvelope", "SignedExecutionPayloadEnvelope",
    "ProposerPreferences", "SignedProposerPreferences",
    "BeaconBlockBody", "BeaconBlock", "SignedBeaconBlock",
    "BeaconState",
]
