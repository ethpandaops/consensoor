"""Engine API client for communication with execution layer."""

import logging
import time
from typing import Optional, Any

import aiohttp
import jwt

from .types import (
    PayloadStatus,
    ForkchoiceState,
    ForkchoiceUpdateResponse,
    GetPayloadResponse,
    EngineAPIError,
)

logger = logging.getLogger(__name__)


def get_fork_for_timestamp(timestamp: int) -> str:
    """Determine which fork is active for a given timestamp."""
    from ..spec.network_config import get_config
    config = get_config()

    if hasattr(config, 'fulu_fork_epoch') and timestamp >= _epoch_to_timestamp(config.fulu_fork_epoch, config):
        return "fulu"
    if hasattr(config, 'electra_fork_epoch') and timestamp >= _epoch_to_timestamp(config.electra_fork_epoch, config):
        return "electra"
    if hasattr(config, 'deneb_fork_epoch') and timestamp >= _epoch_to_timestamp(config.deneb_fork_epoch, config):
        return "deneb"
    if hasattr(config, 'capella_fork_epoch') and timestamp >= _epoch_to_timestamp(config.capella_fork_epoch, config):
        return "capella"
    return "bellatrix"


def _epoch_to_timestamp(epoch: int, config) -> int:
    """Convert epoch to timestamp."""
    from ..spec.constants import SLOTS_PER_EPOCH
    if epoch == 2**64 - 1:
        return 2**63
    genesis_time = getattr(config, 'min_genesis_time', 0)
    return genesis_time + epoch * SLOTS_PER_EPOCH() * config.seconds_per_slot


class EngineAPIClient:
    """Client for Ethereum Engine API."""

    def __init__(self, url: str, jwt_secret: bytes):
        self.url = url
        self.jwt_secret = jwt_secret
        self._session: Optional[aiohttp.ClientSession] = None
        self._request_id = 0
        self._genesis_time: Optional[int] = None

    def set_genesis_time(self, genesis_time: int) -> None:
        """Set the genesis time for fork calculations."""
        self._genesis_time = genesis_time

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure we have an active session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _create_jwt_token(self) -> str:
        """Create a JWT token for authentication."""
        now = int(time.time())
        payload = {"iat": now}
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")

    async def _call(self, method: str, params: list) -> Any:
        """Make a JSON-RPC call to the Engine API."""
        session = await self._ensure_session()
        self._request_id += 1

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._create_jwt_token()}",
        }

        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": self._request_id,
        }

        logger.debug(f"Engine API call: {method}")

        try:
            async with session.post(
                self.url, json=payload, headers=headers
            ) as response:
                data = await response.json()

                if "error" in data:
                    error = data["error"]
                    raise EngineAPIError(error.get("code", -1), error.get("message", ""))

                return data.get("result")
        except aiohttp.ClientError as e:
            logger.error(f"Engine API connection error: {e}")
            raise

    def _get_fork_for_timestamp(self, timestamp: int) -> str:
        """Determine which fork is active for a given timestamp."""
        from ..spec.network_config import get_config
        from ..spec.constants import SLOTS_PER_EPOCH

        config = get_config()
        genesis_time = self._genesis_time or 0

        def epoch_start_time(epoch: int) -> int:
            if epoch == 2**64 - 1:
                return 2**63
            return genesis_time + epoch * SLOTS_PER_EPOCH() * config.seconds_per_slot

        if hasattr(config, 'fulu_fork_epoch') and timestamp >= epoch_start_time(config.fulu_fork_epoch):
            return "fulu"
        if hasattr(config, 'electra_fork_epoch') and timestamp >= epoch_start_time(config.electra_fork_epoch):
            return "electra"
        if hasattr(config, 'deneb_fork_epoch') and timestamp >= epoch_start_time(config.deneb_fork_epoch):
            return "deneb"
        if hasattr(config, 'capella_fork_epoch') and timestamp >= epoch_start_time(config.capella_fork_epoch):
            return "capella"
        return "bellatrix"

    async def new_payload_v4(
        self,
        execution_payload,
        versioned_hashes: list[bytes],
        parent_beacon_block_root: bytes,
        execution_requests: list,
    ) -> PayloadStatus:
        """Send a new payload to the execution layer (Engine API v4)."""
        payload_dict = self._payload_to_dict(execution_payload)

        params = [
            payload_dict,
            ["0x" + h.hex() for h in versioned_hashes],
            "0x" + parent_beacon_block_root.hex(),
            execution_requests,
        ]

        result = await self._call("engine_newPayloadV4", params)
        return PayloadStatus.from_dict(result)

    async def forkchoice_updated_v3(
        self,
        forkchoice_state: ForkchoiceState,
        payload_attributes: Optional[dict] = None,
    ) -> ForkchoiceUpdateResponse:
        """Update the forkchoice state (Deneb/Cancun - requires parentBeaconBlockRoot)."""
        params = [
            forkchoice_state.to_dict(),
            payload_attributes,
        ]

        result = await self._call("engine_forkchoiceUpdatedV3", params)
        return ForkchoiceUpdateResponse.from_dict(result)

    async def forkchoice_updated_v2(
        self,
        forkchoice_state: ForkchoiceState,
        payload_attributes: Optional[dict] = None,
    ) -> ForkchoiceUpdateResponse:
        """Update the forkchoice state (Capella - has withdrawals, no parentBeaconBlockRoot)."""
        attrs = None
        if payload_attributes:
            attrs = {k: v for k, v in payload_attributes.items() if k != "parentBeaconBlockRoot"}
        params = [
            forkchoice_state.to_dict(),
            attrs,
        ]

        result = await self._call("engine_forkchoiceUpdatedV2", params)
        return ForkchoiceUpdateResponse.from_dict(result)

    async def forkchoice_updated_v1(
        self,
        forkchoice_state: ForkchoiceState,
        payload_attributes: Optional[dict] = None,
    ) -> ForkchoiceUpdateResponse:
        """Update the forkchoice state (Bellatrix - no withdrawals, no parentBeaconBlockRoot)."""
        attrs = None
        if payload_attributes:
            attrs = {k: v for k, v in payload_attributes.items() if k not in ("parentBeaconBlockRoot", "withdrawals")}
        params = [
            forkchoice_state.to_dict(),
            attrs,
        ]

        result = await self._call("engine_forkchoiceUpdatedV1", params)
        return ForkchoiceUpdateResponse.from_dict(result)

    async def forkchoice_updated(
        self,
        forkchoice_state: ForkchoiceState,
        payload_attributes: Optional[dict] = None,
        timestamp: Optional[int] = None,
    ) -> ForkchoiceUpdateResponse:
        """Update the forkchoice state using the appropriate version for the current fork."""
        if timestamp is None:
            timestamp = int(time.time())

        fork = self._get_fork_for_timestamp(timestamp)
        logger.debug(f"forkchoice_updated: timestamp={timestamp}, fork={fork}")

        if fork in ("fulu", "electra", "deneb"):
            return await self.forkchoice_updated_v3(forkchoice_state, payload_attributes)
        elif fork == "capella":
            return await self.forkchoice_updated_v2(forkchoice_state, payload_attributes)
        else:
            return await self.forkchoice_updated_v1(forkchoice_state, payload_attributes)

    async def get_payload_v5(self, payload_id: bytes) -> GetPayloadResponse:
        """Get an execution payload by ID (Osaka/Fulu and beyond)."""
        result = await self._call("engine_getPayloadV5", ["0x" + payload_id.hex()])
        return GetPayloadResponse.from_dict(result)

    async def get_payload_v4(self, payload_id: bytes) -> GetPayloadResponse:
        """Get an execution payload by ID (Electra/Prague)."""
        result = await self._call("engine_getPayloadV4", ["0x" + payload_id.hex()])
        return GetPayloadResponse.from_dict(result)

    async def get_payload_v3(self, payload_id: bytes) -> GetPayloadResponse:
        """Get an execution payload by ID (Deneb/Cancun)."""
        result = await self._call("engine_getPayloadV3", ["0x" + payload_id.hex()])
        return GetPayloadResponse.from_dict(result)

    async def get_payload_v2(self, payload_id: bytes) -> GetPayloadResponse:
        """Get an execution payload by ID (Capella/Shanghai)."""
        result = await self._call("engine_getPayloadV2", ["0x" + payload_id.hex()])
        return GetPayloadResponse.from_dict(result)

    async def get_payload_v1(self, payload_id: bytes) -> GetPayloadResponse:
        """Get an execution payload by ID (Bellatrix/Paris)."""
        result = await self._call("engine_getPayloadV1", ["0x" + payload_id.hex()])
        return GetPayloadResponse.from_dict(result)

    async def get_payload(self, payload_id: bytes, timestamp: Optional[int] = None) -> GetPayloadResponse:
        """Get an execution payload using the appropriate version for the current fork."""
        if timestamp is None:
            timestamp = int(time.time())

        fork = self._get_fork_for_timestamp(timestamp)
        logger.debug(f"get_payload: timestamp={timestamp}, fork={fork}")

        if fork == "fulu":
            return await self.get_payload_v5(payload_id)
        elif fork == "electra":
            return await self.get_payload_v4(payload_id)
        elif fork == "deneb":
            return await self.get_payload_v3(payload_id)
        elif fork == "capella":
            return await self.get_payload_v2(payload_id)
        else:
            return await self.get_payload_v1(payload_id)

    async def new_payload_v3(
        self,
        execution_payload,
        versioned_hashes: list[bytes],
        parent_beacon_block_root: bytes,
    ) -> PayloadStatus:
        """Send a new payload to the execution layer (Engine API v3 - Deneb)."""
        payload_dict = self._payload_to_dict(execution_payload)

        params = [
            payload_dict,
            ["0x" + h.hex() for h in versioned_hashes],
            "0x" + parent_beacon_block_root.hex(),
        ]

        result = await self._call("engine_newPayloadV3", params)
        return PayloadStatus.from_dict(result)

    async def new_payload_v2(
        self,
        execution_payload,
    ) -> PayloadStatus:
        """Send a new payload to the execution layer (Engine API v2 - Capella)."""
        payload_dict = self._payload_to_dict(execution_payload)
        result = await self._call("engine_newPayloadV2", [payload_dict])
        return PayloadStatus.from_dict(result)

    async def new_payload_v1(
        self,
        execution_payload,
    ) -> PayloadStatus:
        """Send a new payload to the execution layer (Engine API v1 - Bellatrix)."""
        payload_dict = self._payload_to_dict(execution_payload)
        if "withdrawals" in payload_dict:
            del payload_dict["withdrawals"]
        if "blobGasUsed" in payload_dict:
            del payload_dict["blobGasUsed"]
        if "excessBlobGas" in payload_dict:
            del payload_dict["excessBlobGas"]
        result = await self._call("engine_newPayloadV1", [payload_dict])
        return PayloadStatus.from_dict(result)

    async def new_payload(
        self,
        execution_payload,
        versioned_hashes: list[bytes] = None,
        parent_beacon_block_root: bytes = None,
        execution_requests: list = None,
        timestamp: int = None,
    ) -> PayloadStatus:
        """Send a new payload using the appropriate version for the current fork."""
        if timestamp is None:
            timestamp = int(time.time())

        fork = self._get_fork_for_timestamp(timestamp)
        logger.debug(f"new_payload: timestamp={timestamp}, fork={fork}")

        if fork == "fulu":
            return await self.new_payload_v4(execution_payload, versioned_hashes or [], parent_beacon_block_root or b"\x00" * 32, execution_requests or [])
        elif fork == "electra":
            return await self.new_payload_v4(execution_payload, versioned_hashes or [], parent_beacon_block_root or b"\x00" * 32, execution_requests or [])
        elif fork == "deneb":
            return await self.new_payload_v3(execution_payload, versioned_hashes or [], parent_beacon_block_root or b"\x00" * 32)
        elif fork == "capella":
            return await self.new_payload_v2(execution_payload)
        else:
            return await self.new_payload_v1(execution_payload)

    async def exchange_capabilities(self) -> list[str]:
        """Exchange capabilities with the execution layer."""
        capabilities = [
            "engine_newPayloadV4",
            "engine_newPayloadV3",
            "engine_newPayloadV2",
            "engine_newPayloadV1",
            "engine_forkchoiceUpdatedV3",
            "engine_forkchoiceUpdatedV2",
            "engine_forkchoiceUpdatedV1",
            "engine_getPayloadV4",
            "engine_getPayloadV3",
            "engine_getPayloadV2",
            "engine_getPayloadV1",
        ]
        result = await self._call("engine_exchangeCapabilities", [capabilities])
        return result

    async def close(self) -> None:
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _payload_to_dict(self, payload) -> dict:
        """Convert an ExecutionPayload to Engine API format."""
        result = {
            "parentHash": "0x" + bytes(payload.parent_hash).hex(),
            "feeRecipient": "0x" + bytes(payload.fee_recipient).hex(),
            "stateRoot": "0x" + bytes(payload.state_root).hex(),
            "receiptsRoot": "0x" + bytes(payload.receipts_root).hex(),
            "logsBloom": "0x" + bytes(payload.logs_bloom).hex(),
            "prevRandao": "0x" + bytes(payload.prev_randao).hex(),
            "blockNumber": hex(int(payload.block_number)),
            "gasLimit": hex(int(payload.gas_limit)),
            "gasUsed": hex(int(payload.gas_used)),
            "timestamp": hex(int(payload.timestamp)),
            "extraData": "0x" + bytes(payload.extra_data).hex(),
            "baseFeePerGas": hex(int(payload.base_fee_per_gas)),
            "blockHash": "0x" + bytes(payload.block_hash).hex(),
            "transactions": ["0x" + bytes(tx).hex() for tx in payload.transactions],
        }
        if hasattr(payload, 'withdrawals'):
            result["withdrawals"] = [
                {
                    "index": hex(int(w.index)),
                    "validatorIndex": hex(int(w.validator_index)),
                    "address": "0x" + bytes(w.address).hex(),
                    "amount": hex(int(w.amount)),
                }
                for w in payload.withdrawals
            ]
        if hasattr(payload, 'blob_gas_used'):
            result["blobGasUsed"] = hex(int(payload.blob_gas_used))
        if hasattr(payload, 'excess_blob_gas'):
            result["excessBlobGas"] = hex(int(payload.excess_blob_gas))
        return result

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
