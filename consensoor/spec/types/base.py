"""Base SSZ types and primitives used across all forks."""

from remerkleable.basic import uint8, uint64, uint256, boolean
from remerkleable.byte_arrays import Bytes4, Bytes32, Bytes48, Bytes96, ByteVector
from remerkleable.complex import Container, Vector, List
from remerkleable.bitfields import Bitvector, Bitlist

Bytes20 = ByteVector[20]

# Type aliases
Slot = uint64
Epoch = uint64
CommitteeIndex = uint64
ValidatorIndex = uint64
Gwei = uint64
Root = Bytes32
Hash32 = Bytes32
Version = Bytes4
DomainType = Bytes4
ForkDigest = Bytes4
Domain = Bytes32
BLSPubkey = Bytes48
BLSSignature = Bytes96
ExecutionAddress = Bytes20
WithdrawalIndex = uint64
ParticipationFlags = uint8
KZGCommitment = ByteVector[48]
KZGProof = ByteVector[48]

# Transaction type
MAX_BYTES_PER_TRANSACTION = 2**30
Transaction = List[uint8, MAX_BYTES_PER_TRANSACTION]


class Fork(Container):
    previous_version: Version
    current_version: Version
    epoch: Epoch


class ForkData(Container):
    current_version: Version
    genesis_validators_root: Root


class Checkpoint(Container):
    epoch: Epoch
    root: Root


class SigningData(Container):
    object_root: Root
    domain: Domain


__all__ = [
    "uint8", "uint64", "uint256", "boolean",
    "Bytes4", "Bytes20", "Bytes32", "Bytes48", "Bytes96", "ByteVector",
    "Container", "Vector", "List",
    "Bitvector", "Bitlist",
    "Slot", "Epoch", "CommitteeIndex", "ValidatorIndex", "Gwei",
    "Root", "Hash32", "Version", "DomainType", "ForkDigest", "Domain",
    "BLSPubkey", "BLSSignature", "ExecutionAddress", "WithdrawalIndex",
    "ParticipationFlags", "KZGCommitment", "KZGProof",
    "Transaction",
    "Fork", "ForkData", "Checkpoint", "SigningData",
]
