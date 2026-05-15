"""SSZ container definitions for the Engine API binary transport.

Per execution-apis PR #764 (`src/engine/ssz-encoding.md`).
Nullable JSON fields are encoded as ``List[T, 1]`` in SSZ — empty list
denotes absence, single element denotes presence.
"""

from remerkleable.basic import boolean, uint8, uint64, uint256
from remerkleable.byte_arrays import ByteList, ByteVector, Bytes32
from remerkleable.complex import Container, List

from ..spec.types.base import (
    Bytes20,
    ExecutionAddress,
    Hash32,
    MAX_BYTES_PER_TRANSACTION,
)
from ..spec.types.capella import Withdrawal as WithdrawalV1

BYTES_PER_LOGS_BLOOM = 256
MAX_EXTRA_DATA_BYTES = 2**5
MAX_TRANSACTIONS_PER_PAYLOAD = 2**20
MAX_WITHDRAWALS_PER_PAYLOAD = 2**4
MAX_BLOB_COMMITMENTS_PER_BLOCK = 2**12
FIELD_ELEMENTS_PER_BLOB = 4096
BYTES_PER_FIELD_ELEMENT = 32
BLOB_SIZE = FIELD_ELEMENTS_PER_BLOB * BYTES_PER_FIELD_ELEMENT
CELLS_PER_EXT_BLOB = 128
MAX_BLOB_HASHES_REQUEST = 128
MAX_PAYLOAD_BODIES_REQUEST = 2**5
MAX_EXECUTION_REQUESTS = 2**8
MAX_ERROR_MESSAGE_LENGTH = 1024
MAX_CLIENT_CODE_LENGTH = 2
MAX_CLIENT_NAME_LENGTH = 64
MAX_CLIENT_VERSION_LENGTH = 64
MAX_CLIENT_VERSIONS = 4
MAX_CAPABILITY_NAME_LENGTH = 64
MAX_CAPABILITIES = 64


Bytes8 = ByteVector[8]
Bytes48 = ByteVector[48]
LogsBloom = ByteVector[BYTES_PER_LOGS_BLOOM]
ExtraData = ByteList[MAX_EXTRA_DATA_BYTES]
TransactionBytes = ByteList[MAX_BYTES_PER_TRANSACTION]
Blob = ByteVector[BLOB_SIZE]


class ExecutionPayloadV1(Container):
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


class ExecutionPayloadV2(Container):
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


class ExecutionPayloadV3(Container):
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


class ExecutionPayloadV4(Container):
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
    block_access_list: ByteList[MAX_BYTES_PER_TRANSACTION]
    slot_number: uint64


class PayloadStatusV1(Container):
    status: uint8
    latest_valid_hash: List[Bytes32, 1]
    validation_error: ByteList[MAX_ERROR_MESSAGE_LENGTH]


PAYLOAD_STATUS_TO_INT = {"VALID": 0, "INVALID": 1, "SYNCING": 2, "ACCEPTED": 3}
INT_TO_PAYLOAD_STATUS = {v: k for k, v in PAYLOAD_STATUS_TO_INT.items()}


class ForkchoiceStateV1(Container):
    head_block_hash: Bytes32
    safe_block_hash: Bytes32
    finalized_block_hash: Bytes32


class PayloadAttributesV1(Container):
    timestamp: uint64
    prev_randao: Bytes32
    suggested_fee_recipient: Bytes20


class PayloadAttributesV2(Container):
    timestamp: uint64
    prev_randao: Bytes32
    suggested_fee_recipient: Bytes20
    withdrawals: List[WithdrawalV1, MAX_WITHDRAWALS_PER_PAYLOAD]


class PayloadAttributesV3(Container):
    timestamp: uint64
    prev_randao: Bytes32
    suggested_fee_recipient: Bytes20
    withdrawals: List[WithdrawalV1, MAX_WITHDRAWALS_PER_PAYLOAD]
    parent_beacon_block_root: Bytes32


class PayloadAttributesV4(Container):
    timestamp: uint64
    prev_randao: Bytes32
    suggested_fee_recipient: Bytes20
    withdrawals: List[WithdrawalV1, MAX_WITHDRAWALS_PER_PAYLOAD]
    parent_beacon_block_root: Bytes32
    slot_number: uint64


class ForkchoiceUpdatedResponseV1(Container):
    payload_status: PayloadStatusV1
    payload_id: List[Bytes8, 1]


class BlobsBundleV1(Container):
    commitments: List[Bytes48, MAX_BLOB_COMMITMENTS_PER_BLOCK]
    proofs: List[Bytes48, MAX_BLOB_COMMITMENTS_PER_BLOCK]
    blobs: List[Blob, MAX_BLOB_COMMITMENTS_PER_BLOCK]


class BlobsBundleV2(Container):
    commitments: List[Bytes48, MAX_BLOB_COMMITMENTS_PER_BLOCK]
    proofs: List[Bytes48, MAX_BLOB_COMMITMENTS_PER_BLOCK * CELLS_PER_EXT_BLOB]
    blobs: List[Blob, MAX_BLOB_COMMITMENTS_PER_BLOCK]


class GetPayloadResponseV2(Container):
    execution_payload: ExecutionPayloadV2
    block_value: uint256


class GetPayloadResponseV3(Container):
    execution_payload: ExecutionPayloadV3
    block_value: uint256
    blobs_bundle: BlobsBundleV1
    should_override_builder: boolean


class GetPayloadResponseV4(Container):
    execution_payload: ExecutionPayloadV3
    block_value: uint256
    blobs_bundle: BlobsBundleV1
    should_override_builder: boolean
    execution_requests: List[TransactionBytes, MAX_EXECUTION_REQUESTS]


class GetPayloadResponseV5(Container):
    execution_payload: ExecutionPayloadV3
    block_value: uint256
    blobs_bundle: BlobsBundleV2
    should_override_builder: boolean
    execution_requests: List[TransactionBytes, MAX_EXECUTION_REQUESTS]


class GetPayloadResponseV6(Container):
    execution_payload: ExecutionPayloadV4
    block_value: uint256
    blobs_bundle: BlobsBundleV2
    should_override_builder: boolean
    execution_requests: List[TransactionBytes, MAX_EXECUTION_REQUESTS]


class NewPayloadV1Request(Container):
    execution_payload: ExecutionPayloadV1


class NewPayloadV2Request(Container):
    execution_payload: ExecutionPayloadV2


class NewPayloadV3Request(Container):
    execution_payload: ExecutionPayloadV3
    expected_blob_versioned_hashes: List[Bytes32, MAX_BLOB_COMMITMENTS_PER_BLOCK]
    parent_beacon_block_root: Bytes32


class NewPayloadV4Request(Container):
    execution_payload: ExecutionPayloadV3
    expected_blob_versioned_hashes: List[Bytes32, MAX_BLOB_COMMITMENTS_PER_BLOCK]
    parent_beacon_block_root: Bytes32
    execution_requests: List[TransactionBytes, MAX_EXECUTION_REQUESTS]


class NewPayloadV5Request(Container):
    execution_payload: ExecutionPayloadV4
    expected_blob_versioned_hashes: List[Bytes32, MAX_BLOB_COMMITMENTS_PER_BLOCK]
    parent_beacon_block_root: Bytes32
    execution_requests: List[TransactionBytes, MAX_EXECUTION_REQUESTS]


class ForkchoiceUpdatedV1Request(Container):
    forkchoice_state: ForkchoiceStateV1
    payload_attributes: List[PayloadAttributesV1, 1]


class ForkchoiceUpdatedV2Request(Container):
    forkchoice_state: ForkchoiceStateV1
    payload_attributes: List[PayloadAttributesV2, 1]


class ForkchoiceUpdatedV3Request(Container):
    forkchoice_state: ForkchoiceStateV1
    payload_attributes: List[PayloadAttributesV3, 1]


class ForkchoiceUpdatedV4Request(Container):
    forkchoice_state: ForkchoiceStateV1
    payload_attributes: List[PayloadAttributesV4, 1]


class ExchangeCapabilitiesRequest(Container):
    capabilities: List[ByteList[MAX_CAPABILITY_NAME_LENGTH], MAX_CAPABILITIES]


class ExchangeCapabilitiesResponse(Container):
    capabilities: List[ByteList[MAX_CAPABILITY_NAME_LENGTH], MAX_CAPABILITIES]


# Capability strings advertised in `engine_exchangeCapabilities` per PR #764.
# A CL advertises the SSZ REST endpoints it supports; if the EL responds with
# the same string, the corresponding call MAY use the binary transport.
SSZ_CAPABILITIES: list[str] = [
    "POST /engine/v1/payloads",
    "POST /engine/v2/payloads",
    "POST /engine/v3/payloads",
    "POST /engine/v4/payloads",
    "POST /engine/v5/payloads",
    "GET /engine/v1/payloads/{payload_id}",
    "GET /engine/v2/payloads/{payload_id}",
    "GET /engine/v3/payloads/{payload_id}",
    "GET /engine/v4/payloads/{payload_id}",
    "GET /engine/v5/payloads/{payload_id}",
    "GET /engine/v6/payloads/{payload_id}",
    "POST /engine/v1/forkchoice",
    "POST /engine/v2/forkchoice",
    "POST /engine/v3/forkchoice",
    "POST /engine/v4/forkchoice",
    "POST /engine/v1/blobs",
    "POST /engine/v2/blobs",
    "POST /engine/v3/blobs",
    "POST /engine/v1/capabilities",
    "POST /engine/v1/client/version",
]
