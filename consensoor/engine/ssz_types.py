"""SSZ container definitions for the Engine API v2 REST transport.

Per execution-apis PR #793 (`src/engine/refactor.md` + `refactor-ssz.md`):
the engine API moves from JSON-RPC to a resource-oriented REST API where
hot-path request/response bodies are SSZ-encoded and scoped under
``/engine/v2/{fork}/...``.

Conventions (refactor.md § SSZ encoding conventions):
- ``Optional[T]`` ≡ ``List[T, 1]`` — empty list = absent, single element = present.
- ``String`` ≡ ``List[byte, MAX_ERROR_BYTES]`` (UTF-8).
- uints are little-endian on the wire.
- field names are ``snake_case`` to match consensus-specs.
"""

from remerkleable.basic import boolean, uint8, uint64, uint256
from remerkleable.bitfields import Bitvector
from remerkleable.byte_arrays import ByteList, ByteVector, Bytes32
from remerkleable.complex import Container, List

from ..spec.types.base import (
    Bytes20,
    ExecutionAddress,
    Hash32,
    MAX_BYTES_PER_TRANSACTION,
)
from ..spec.types.capella import Withdrawal as WithdrawalV1

# --- MAX_* constants (refactor-ssz.md § MAX_* constants) ---
BYTES_PER_LOGS_BLOOM = 256
MAX_EXTRA_DATA_BYTES = 2**5
MAX_TRANSACTIONS_PER_PAYLOAD = 2**20
MAX_WITHDRAWALS_PER_PAYLOAD = 2**4
MAX_BLOB_COMMITMENTS_PER_BLOCK = 2**12
FIELD_ELEMENTS_PER_BLOB = 4096
BYTES_PER_FIELD_ELEMENT = 32
BLOB_SIZE = FIELD_ELEMENTS_PER_BLOB * BYTES_PER_FIELD_ELEMENT
CELLS_PER_EXT_BLOB = 128
BYTES_PER_CELL = BLOB_SIZE // CELLS_PER_EXT_BLOB  # 1024
MAX_BLOBS_REQUEST = 128  # MAX_VERSIONED_HASHES_PER_REQUEST
MAX_BODIES_REQUEST = 2**5
MAX_EXECUTION_REQUESTS_PER_PAYLOAD = 2**8
# Placeholders per refactor-ssz.md "Open sketch questions" — reuse the tx bound
# until EIP-7928 / EIP-7685 pin tighter numbers.
MAX_BAL_BYTES = MAX_BYTES_PER_TRANSACTION
MAX_BYTES_PER_EXECUTION_REQUEST = MAX_BYTES_PER_TRANSACTION
MAX_ERROR_MESSAGE_LENGTH = 1024  # MAX_ERROR_BYTES

# --- primitive aliases ---
Bytes4 = ByteVector[4]
Bytes8 = ByteVector[8]
Bytes48 = ByteVector[48]
Root = Bytes32
VersionedHash = Bytes32
LogsBloom = ByteVector[BYTES_PER_LOGS_BLOOM]
ExtraData = ByteList[MAX_EXTRA_DATA_BYTES]
TransactionBytes = ByteList[MAX_BYTES_PER_TRANSACTION]
ExecutionRequestBytes = ByteList[MAX_BYTES_PER_EXECUTION_REQUEST]
Blob = ByteVector[BLOB_SIZE]
Cell = ByteVector[BYTES_PER_CELL]
ErrorString = ByteList[MAX_ERROR_MESSAGE_LENGTH]
CustodyColumns = Bitvector[CELLS_PER_EXT_BLOB]


# ---------------------------------------------------------------------------
# ExecutionPayload per fork (refactor-ssz.md § ExecutionPayload per fork)
# ---------------------------------------------------------------------------
class ExecutionPayloadV1(Container):  # Paris
    parent_hash: Hash32
    fee_recipient: ExecutionAddress
    state_root: Bytes32
    receipts_root: Bytes32
    logs_bloom: LogsBloom
    prev_randao: Bytes32
    block_number: uint64
    gas_limit: uint64
    gas_used: uint64
    timestamp: uint64
    extra_data: ExtraData
    base_fee_per_gas: uint256
    block_hash: Hash32
    transactions: List[TransactionBytes, MAX_TRANSACTIONS_PER_PAYLOAD]


class ExecutionPayloadV2(Container):  # Shanghai (+ withdrawals)
    parent_hash: Hash32
    fee_recipient: ExecutionAddress
    state_root: Bytes32
    receipts_root: Bytes32
    logs_bloom: LogsBloom
    prev_randao: Bytes32
    block_number: uint64
    gas_limit: uint64
    gas_used: uint64
    timestamp: uint64
    extra_data: ExtraData
    base_fee_per_gas: uint256
    block_hash: Hash32
    transactions: List[TransactionBytes, MAX_TRANSACTIONS_PER_PAYLOAD]
    withdrawals: List[WithdrawalV1, MAX_WITHDRAWALS_PER_PAYLOAD]


class ExecutionPayloadV3(Container):  # Cancun / Prague / Osaka (+ blob gas)
    parent_hash: Hash32
    fee_recipient: ExecutionAddress
    state_root: Bytes32
    receipts_root: Bytes32
    logs_bloom: LogsBloom
    prev_randao: Bytes32
    block_number: uint64
    gas_limit: uint64
    gas_used: uint64
    timestamp: uint64
    extra_data: ExtraData
    base_fee_per_gas: uint256
    block_hash: Hash32
    transactions: List[TransactionBytes, MAX_TRANSACTIONS_PER_PAYLOAD]
    withdrawals: List[WithdrawalV1, MAX_WITHDRAWALS_PER_PAYLOAD]
    blob_gas_used: uint64
    excess_blob_gas: uint64


class ExecutionPayloadV4(Container):  # Amsterdam (+ block_access_list + slot_number)
    parent_hash: Hash32
    fee_recipient: ExecutionAddress
    state_root: Bytes32
    receipts_root: Bytes32
    logs_bloom: LogsBloom
    prev_randao: Bytes32
    block_number: uint64
    gas_limit: uint64
    gas_used: uint64
    timestamp: uint64
    extra_data: ExtraData
    base_fee_per_gas: uint256
    block_hash: Hash32
    transactions: List[TransactionBytes, MAX_TRANSACTIONS_PER_PAYLOAD]
    withdrawals: List[WithdrawalV1, MAX_WITHDRAWALS_PER_PAYLOAD]
    blob_gas_used: uint64
    excess_blob_gas: uint64
    block_access_list: ByteList[MAX_BAL_BYTES]
    slot_number: uint64


# ---------------------------------------------------------------------------
# Shared structures
# ---------------------------------------------------------------------------
class ForkchoiceStateV1(Container):
    head_block_hash: Bytes32
    safe_block_hash: Bytes32
    finalized_block_hash: Bytes32


class PayloadStatus(Container):
    """refactor-ssz.md § PayloadStatus.

    ``status`` is a uint8 enum (VALID=0, INVALID=1, SYNCING=2, ACCEPTED=3);
    ``INVALID_BLOCK_HASH`` is removed in #793. ``validation_error`` is an
    ``Optional[String]`` (length-0-or-1 list of an error ByteList).
    """
    status: uint8
    latest_valid_hash: List[Bytes32, 1]
    validation_error: List[ErrorString, 1]


PAYLOAD_STATUS_TO_INT = {"VALID": 0, "INVALID": 1, "SYNCING": 2, "ACCEPTED": 3}
INT_TO_PAYLOAD_STATUS = {v: k for k, v in PAYLOAD_STATUS_TO_INT.items()}


# --- PayloadAttributes per fork (refactor-ssz.md § PayloadAttributes per fork) ---
class PayloadAttributesV1(Container):  # Paris
    timestamp: uint64
    prev_randao: Bytes32
    suggested_fee_recipient: Bytes20


class PayloadAttributesV2(Container):  # Shanghai (+ withdrawals)
    timestamp: uint64
    prev_randao: Bytes32
    suggested_fee_recipient: Bytes20
    withdrawals: List[WithdrawalV1, MAX_WITHDRAWALS_PER_PAYLOAD]


class PayloadAttributesV3(Container):  # Cancun / Prague / Osaka (+ parent_beacon_block_root)
    timestamp: uint64
    prev_randao: Bytes32
    suggested_fee_recipient: Bytes20
    withdrawals: List[WithdrawalV1, MAX_WITHDRAWALS_PER_PAYLOAD]
    parent_beacon_block_root: Bytes32


class PayloadAttributesV4(Container):  # Amsterdam = Cancun + slot_number + target_gas_limit
    timestamp: uint64
    prev_randao: Bytes32
    suggested_fee_recipient: Bytes20
    withdrawals: List[WithdrawalV1, MAX_WITHDRAWALS_PER_PAYLOAD]
    parent_beacon_block_root: Bytes32
    slot_number: uint64
    # Not in the #793 draft (which has no target_gas_limit), but geth's
    # post-Amsterdam payload-build path requires it, so we keep it.
    target_gas_limit: uint64


# ---------------------------------------------------------------------------
# BlobsBundle per revision (refactor-ssz.md § BlobsBundle per revision)
# ---------------------------------------------------------------------------
class BlobsBundleV1(Container):  # Cancun / Prague — one proof per blob
    commitments: List[Bytes48, MAX_BLOB_COMMITMENTS_PER_BLOCK]
    proofs: List[Bytes48, MAX_BLOB_COMMITMENTS_PER_BLOCK]
    blobs: List[Blob, MAX_BLOB_COMMITMENTS_PER_BLOCK]


class BlobsBundleV2(Container):  # Osaka / Amsterdam — cell proofs
    commitments: List[Bytes48, MAX_BLOB_COMMITMENTS_PER_BLOCK]
    proofs: List[Bytes48, MAX_BLOB_COMMITMENTS_PER_BLOCK * CELLS_PER_EXT_BLOB]
    blobs: List[Blob, MAX_BLOB_COMMITMENTS_PER_BLOCK]


# ---------------------------------------------------------------------------
# POST /engine/v2/{fork}/payloads  (replaces engine_newPayload)
# Request body: ExecutionPayloadEnvelope; response body: PayloadStatus.
# `expected_blob_versioned_hashes` is removed in #793.
# ---------------------------------------------------------------------------
class ExecutionPayloadEnvelopeParis(Container):
    payload: ExecutionPayloadV1


class ExecutionPayloadEnvelopeShanghai(Container):
    payload: ExecutionPayloadV2


class ExecutionPayloadEnvelopeCancun(Container):
    payload: ExecutionPayloadV3
    parent_beacon_block_root: Root


class ExecutionPayloadEnvelopePrague(Container):  # Prague / Osaka
    payload: ExecutionPayloadV3
    parent_beacon_block_root: Root
    execution_requests: List[ExecutionRequestBytes, MAX_EXECUTION_REQUESTS_PER_PAYLOAD]


class ExecutionPayloadEnvelopeAmsterdam(Container):
    payload: ExecutionPayloadV4
    parent_beacon_block_root: Root
    execution_requests: List[ExecutionRequestBytes, MAX_EXECUTION_REQUESTS_PER_PAYLOAD]


# ---------------------------------------------------------------------------
# POST /engine/v2/{fork}/forkchoice  (replaces engine_forkchoiceUpdated)
# ---------------------------------------------------------------------------
class ForkchoiceUpdateParis(Container):
    forkchoice_state: ForkchoiceStateV1
    payload_attributes: List[PayloadAttributesV1, 1]


class ForkchoiceUpdateShanghai(Container):
    forkchoice_state: ForkchoiceStateV1
    payload_attributes: List[PayloadAttributesV2, 1]


class ForkchoiceUpdateCancun(Container):  # Cancun / Prague / Osaka
    forkchoice_state: ForkchoiceStateV1
    payload_attributes: List[PayloadAttributesV3, 1]


class ForkchoiceUpdateAmsterdam(Container):
    forkchoice_state: ForkchoiceStateV1
    payload_attributes: List[PayloadAttributesV4, 1]
    custody_columns: List[CustodyColumns, 1]


class ForkchoiceUpdateResponse(Container):
    payload_status: PayloadStatus  # restricted enum (VALID|INVALID|SYNCING)
    payload_id: List[Bytes8, 1]


# ---------------------------------------------------------------------------
# GET /engine/v2/{fork}/payloads/{payloadId}  (replaces engine_getPayload)
# Field order per refactor.md: payload, block_value, blobs_bundle,
# execution_requests, should_override_builder.
# ---------------------------------------------------------------------------
class BuiltPayloadParis(Container):
    payload: ExecutionPayloadV1
    block_value: uint256


class BuiltPayloadShanghai(Container):
    payload: ExecutionPayloadV2
    block_value: uint256


class BuiltPayloadCancun(Container):
    payload: ExecutionPayloadV3
    block_value: uint256
    blobs_bundle: BlobsBundleV1
    should_override_builder: boolean


class BuiltPayloadPrague(Container):
    payload: ExecutionPayloadV3
    block_value: uint256
    blobs_bundle: BlobsBundleV1
    execution_requests: List[ExecutionRequestBytes, MAX_EXECUTION_REQUESTS_PER_PAYLOAD]
    should_override_builder: boolean


class BuiltPayloadOsaka(Container):
    payload: ExecutionPayloadV3
    block_value: uint256
    blobs_bundle: BlobsBundleV2
    execution_requests: List[ExecutionRequestBytes, MAX_EXECUTION_REQUESTS_PER_PAYLOAD]
    should_override_builder: boolean


class BuiltPayloadAmsterdam(Container):
    payload: ExecutionPayloadV4
    block_value: uint256
    blobs_bundle: BlobsBundleV2
    execution_requests: List[ExecutionRequestBytes, MAX_EXECUTION_REQUESTS_PER_PAYLOAD]
    should_override_builder: boolean


# ---------------------------------------------------------------------------
# Historical bodies (replaces engine_getPayloadBodiesBy{Hash,Range})
# refactor.md: request = List[Hash32, MAX_BODIES_REQUEST] (POST /bodies/hash);
# response = List[BodyEntry, MAX_BODIES_REQUEST].
# ---------------------------------------------------------------------------
BodiesByHashRequest = List[Hash32, MAX_BODIES_REQUEST]


class ExecutionPayloadBodyParis(Container):
    transactions: List[TransactionBytes, MAX_TRANSACTIONS_PER_PAYLOAD]


class ExecutionPayloadBodyShanghai(Container):  # Shanghai / Cancun / Prague / Osaka
    transactions: List[TransactionBytes, MAX_TRANSACTIONS_PER_PAYLOAD]
    withdrawals: List[WithdrawalV1, MAX_WITHDRAWALS_PER_PAYLOAD]


class ExecutionPayloadBodyAmsterdam(Container):
    transactions: List[TransactionBytes, MAX_TRANSACTIONS_PER_PAYLOAD]
    withdrawals: List[WithdrawalV1, MAX_WITHDRAWALS_PER_PAYLOAD]
    block_access_list: ByteList[MAX_BAL_BYTES]


class BodyEntryParis(Container):
    available: boolean
    body: ExecutionPayloadBodyParis


class BodyEntryShanghai(Container):
    available: boolean
    body: ExecutionPayloadBodyShanghai


class BodyEntryAmsterdam(Container):
    available: boolean
    body: ExecutionPayloadBodyAmsterdam


BodiesResponseParis = List[BodyEntryParis, MAX_BODIES_REQUEST]
BodiesResponseShanghai = List[BodyEntryShanghai, MAX_BODIES_REQUEST]
BodiesResponseAmsterdam = List[BodyEntryAmsterdam, MAX_BODIES_REQUEST]


# ---------------------------------------------------------------------------
# Blob pool (replaces engine_getBlobsV{1..4}); independently versioned.
# Request (v1/v2/v3): List[VersionedHash, MAX_BLOBS_REQUEST].
# Response: List[BlobEntry, MAX_BLOBS_REQUEST].
# ---------------------------------------------------------------------------
VersionedHashList = List[VersionedHash, MAX_BLOBS_REQUEST]


class BlobAndProofV1(Container):
    blob: Blob
    proof: Bytes48


class BlobV1Entry(Container):
    available: boolean
    contents: BlobAndProofV1


BlobsV1Response = List[BlobV1Entry, MAX_BLOBS_REQUEST]


class BlobAndProofV2(Container):
    blob: Blob
    proofs: List[Bytes48, CELLS_PER_EXT_BLOB]


class BlobV2Entry(Container):
    available: boolean
    contents: BlobAndProofV2


BlobsV2Response = List[BlobV2Entry, MAX_BLOBS_REQUEST]
BlobsV3Response = BlobsV2Response  # same per-entry shape


class BlobsV4Request(Container):
    versioned_hashes: List[VersionedHash, MAX_BLOBS_REQUEST]
    indices_bitarray: Bitvector[CELLS_PER_EXT_BLOB]


class BlobCellsAndProofs(Container):
    # Optional[Cell] / Optional[Bytes48] per index — [] where unavailable.
    blob_cells: List[List[Cell, 1], CELLS_PER_EXT_BLOB]
    proofs: List[List[Bytes48, 1], CELLS_PER_EXT_BLOB]


class BlobV4Entry(Container):
    available: boolean
    contents: BlobCellsAndProofs


BlobsV4Response = List[BlobV4Entry, MAX_BLOBS_REQUEST]


# ---------------------------------------------------------------------------
# Per-fork dispatch tables. Keyed by the execution-layer fork name that
# appears in the `/engine/v2/{fork}/...` URL segment.
# ---------------------------------------------------------------------------
EL_FORKS = ("paris", "shanghai", "cancun", "prague", "osaka", "amsterdam")

EXECUTION_PAYLOAD_BY_FORK = {
    "paris": ExecutionPayloadV1,
    "shanghai": ExecutionPayloadV2,
    "cancun": ExecutionPayloadV3,
    "prague": ExecutionPayloadV3,
    "osaka": ExecutionPayloadV3,
    "amsterdam": ExecutionPayloadV4,
}

ENVELOPE_BY_FORK = {
    "paris": ExecutionPayloadEnvelopeParis,
    "shanghai": ExecutionPayloadEnvelopeShanghai,
    "cancun": ExecutionPayloadEnvelopeCancun,
    "prague": ExecutionPayloadEnvelopePrague,
    "osaka": ExecutionPayloadEnvelopePrague,
    "amsterdam": ExecutionPayloadEnvelopeAmsterdam,
}

FORKCHOICE_UPDATE_BY_FORK = {
    "paris": ForkchoiceUpdateParis,
    "shanghai": ForkchoiceUpdateShanghai,
    "cancun": ForkchoiceUpdateCancun,
    "prague": ForkchoiceUpdateCancun,
    "osaka": ForkchoiceUpdateCancun,
    "amsterdam": ForkchoiceUpdateAmsterdam,
}

PAYLOAD_ATTRIBUTES_BY_FORK = {
    "paris": PayloadAttributesV1,
    "shanghai": PayloadAttributesV2,
    "cancun": PayloadAttributesV3,
    "prague": PayloadAttributesV3,
    "osaka": PayloadAttributesV3,
    "amsterdam": PayloadAttributesV4,
}

BUILT_PAYLOAD_BY_FORK = {
    "paris": BuiltPayloadParis,
    "shanghai": BuiltPayloadShanghai,
    "cancun": BuiltPayloadCancun,
    "prague": BuiltPayloadPrague,
    "osaka": BuiltPayloadOsaka,
    "amsterdam": BuiltPayloadAmsterdam,
}

BODIES_RESPONSE_BY_FORK = {
    "paris": BodiesResponseParis,
    "shanghai": BodiesResponseShanghai,
    "cancun": BodiesResponseShanghai,
    "prague": BodiesResponseShanghai,
    "osaka": BodiesResponseShanghai,
    "amsterdam": BodiesResponseAmsterdam,
}
