"""Engine API data types."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PayloadStatusEnum(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    SYNCING = "SYNCING"
    ACCEPTED = "ACCEPTED"
    INVALID_BLOCK_HASH = "INVALID_BLOCK_HASH"


@dataclass
class PayloadStatus:
    """Response from newPayload."""

    status: PayloadStatusEnum
    latest_valid_hash: Optional[bytes] = None
    validation_error: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "PayloadStatus":
        return cls(
            status=PayloadStatusEnum(data["status"]),
            latest_valid_hash=(
                bytes.fromhex(data["latestValidHash"][2:])
                if data.get("latestValidHash")
                else None
            ),
            validation_error=data.get("validationError"),
        )


@dataclass
class ForkchoiceState:
    """Forkchoice state for forkchoiceUpdated."""

    head_block_hash: bytes
    safe_block_hash: bytes
    finalized_block_hash: bytes

    def to_dict(self) -> dict:
        return {
            "headBlockHash": "0x" + self.head_block_hash.hex(),
            "safeBlockHash": "0x" + self.safe_block_hash.hex(),
            "finalizedBlockHash": "0x" + self.finalized_block_hash.hex(),
        }


@dataclass
class ForkchoiceUpdateResponse:
    """Response from forkchoiceUpdated."""

    payload_status: PayloadStatus
    payload_id: Optional[bytes] = None

    @classmethod
    def from_dict(cls, data: dict) -> "ForkchoiceUpdateResponse":
        return cls(
            payload_status=PayloadStatus.from_dict(data["payloadStatus"]),
            payload_id=(
                bytes.fromhex(data["payloadId"][2:])
                if data.get("payloadId")
                else None
            ),
        )


@dataclass
class GetPayloadResponse:
    """Response from getPayload."""

    execution_payload: dict
    block_value: int
    blobs_bundle: Optional[dict] = None
    should_override_builder: bool = False
    execution_requests: Optional[list] = None  # Electra/Fulu: EIP-7685 requests

    @classmethod
    def from_dict(cls, data: dict) -> "GetPayloadResponse":
        if "executionPayload" in data:
            return cls(
                execution_payload=data["executionPayload"],
                block_value=int(data.get("blockValue", "0x0"), 16),
                blobs_bundle=data.get("blobsBundle"),
                should_override_builder=data.get("shouldOverrideBuilder", False),
                execution_requests=data.get("executionRequests"),
            )
        else:
            return cls(
                execution_payload=data,
                block_value=0,
                blobs_bundle=None,
                should_override_builder=False,
                execution_requests=None,
            )


class EngineAPIError(Exception):
    """Error from Engine API."""

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"Engine API error {code}: {message}")
