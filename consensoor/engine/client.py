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
from .. import metrics

logger = logging.getLogger(__name__)


def get_fork_for_timestamp(timestamp: int) -> str:
    """Determine which fork is active for a given timestamp."""
    from ..spec.network_config import get_config
    config = get_config()

    if hasattr(config, 'gloas_fork_epoch') and timestamp >= _epoch_to_timestamp(config.gloas_fork_epoch, config):
        return "gloas"
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
    return genesis_time + epoch * SLOTS_PER_EPOCH() * (config.slot_duration_ms // 1000)


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

        start_time = time.time()
        error_type = None

        try:
            async with session.post(
                self.url, json=payload, headers=headers
            ) as response:
                data = await response.json()

                if "error" in data:
                    error = data["error"]
                    error_type = str(error.get("code", "unknown"))
                    raise EngineAPIError(error.get("code", -1), error.get("message", ""))

                result = data.get("result")
                if method.startswith("engine_forkchoice"):
                    logger.debug(f"Engine API response for {method}: {result}")
                return result
        except aiohttp.ClientError as e:
            error_type = "connection_error"
            logger.error(f"Engine API connection error: {e}")
            raise
        finally:
            latency = time.time() - start_time
            metrics.record_engine_api_call(method, latency, error_type)

    def _get_fork_for_timestamp(self, timestamp: int) -> str:
        """Determine which fork is active for a given timestamp."""
        from ..spec.network_config import get_config
        from ..spec.constants import SLOTS_PER_EPOCH

        config = get_config()
        genesis_time = self._genesis_time or 0
        FAR_FUTURE_EPOCH = 2**64 - 1

        def epoch_start_time(epoch: int) -> int:
            if epoch >= FAR_FUTURE_EPOCH:
                return 2**63
            return genesis_time + epoch * SLOTS_PER_EPOCH() * (config.slot_duration_ms // 1000)

        def is_fork_active(attr_name: str) -> bool:
            if not hasattr(config, attr_name):
                return False
            epoch = getattr(config, attr_name)
            if epoch >= FAR_FUTURE_EPOCH:
                return False
            return timestamp >= epoch_start_time(epoch)

        if is_fork_active('gloas_fork_epoch'):
            return "gloas"
        if is_fork_active('fulu_fork_epoch'):
            return "fulu"
        if is_fork_active('electra_fork_epoch'):
            return "electra"
        if is_fork_active('deneb_fork_epoch'):
            return "deneb"
        if is_fork_active('capella_fork_epoch'):
            return "capella"
        return "bellatrix"

    async def new_payload_v5(
        self,
        execution_payload,
        versioned_hashes: list[bytes],
        parent_beacon_block_root: bytes,
        execution_requests: list,
    ) -> PayloadStatus:
        """Send a new payload to the execution layer (Engine API v5 - Osaka/Fulu)."""
        payload_dict = self._payload_to_dict(execution_payload)

        params = [
            payload_dict,
            ["0x" + h.hex() for h in versioned_hashes],
            "0x" + parent_beacon_block_root.hex(),
            execution_requests,
        ]

        logger.debug(
            f"newPayloadV5: blockHash={payload_dict.get('blockHash')}, "
            f"execution_requests={execution_requests}, "
            f"parent_beacon_root={parent_beacon_block_root.hex()[:16]}"
        )

        result = await self._call("engine_newPayloadV5", params)
        return PayloadStatus.from_dict(result)

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

        logger.info(
            f"newPayloadV4: blockHash={payload_dict.get('blockHash')}, "
            f"stateRoot={payload_dict.get('stateRoot')}, "
            f"timestamp={payload_dict.get('timestamp')}, "
            f"execution_requests={execution_requests}, "
            f"parent_beacon_root={parent_beacon_block_root.hex()[:16]}, "
            f"full_params_len={len(params)}"
        )

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

        if fork in ("gloas", "fulu", "electra", "deneb"):
            return await self.forkchoice_updated_v3(forkchoice_state, payload_attributes)
        elif fork == "capella":
            return await self.forkchoice_updated_v2(forkchoice_state, payload_attributes)
        else:
            return await self.forkchoice_updated_v1(forkchoice_state, payload_attributes)

    async def get_payload_v6(self, payload_id: bytes) -> GetPayloadResponse:
        """Get an execution payload by ID (Amsterdam/Gloas)."""
        result = await self._call("engine_getPayloadV6", ["0x" + payload_id.hex()])
        exec_payload = result.get('executionPayload', {})
        exec_requests = result.get('executionRequests')
        logger.debug(
            f"getPayloadV6 raw response: executionRequests={exec_requests}, "
            f"blockHash={exec_payload.get('blockHash')}, "
            f"timestamp={exec_payload.get('timestamp')}, "
            f"stateRoot={exec_payload.get('stateRoot')}"
        )
        return GetPayloadResponse.from_dict(result)

    async def get_payload_v5(self, payload_id: bytes) -> GetPayloadResponse:
        """Get an execution payload by ID (Osaka/Fulu)."""
        result = await self._call("engine_getPayloadV5", ["0x" + payload_id.hex()])
        exec_payload = result.get('executionPayload', {})
        exec_requests = result.get('executionRequests')
        logger.debug(
            f"getPayloadV5 raw response: executionRequests={exec_requests}, "
            f"blockHash={exec_payload.get('blockHash')}, "
            f"timestamp={exec_payload.get('timestamp')}, "
            f"stateRoot={exec_payload.get('stateRoot')}"
        )
        return GetPayloadResponse.from_dict(result)

    async def get_payload_v4(self, payload_id: bytes) -> GetPayloadResponse:
        """Get an execution payload by ID (Electra/Prague)."""
        result = await self._call("engine_getPayloadV4", ["0x" + payload_id.hex()])
        logger.debug(
            f"getPayloadV4 raw response: executionRequests={result.get('executionRequests')}, "
            f"blockHash={result.get('executionPayload', {}).get('blockHash')}"
        )
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
        logger.info(f"get_payload: timestamp={timestamp}, fork={fork}, payload_id={payload_id.hex()}")

        if fork == "gloas":
            return await self.get_payload_v6(payload_id)
        elif fork == "fulu":
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

        if fork == "gloas":
            return await self.new_payload_v5(execution_payload, versioned_hashes or [], parent_beacon_block_root or b"\x00" * 32, execution_requests or [])
        elif fork in ("fulu", "electra"):
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
            "engine_newPayloadV5",
            "engine_newPayloadV4",
            "engine_newPayloadV3",
            "engine_newPayloadV2",
            "engine_newPayloadV1",
            "engine_forkchoiceUpdatedV3",
            "engine_forkchoiceUpdatedV2",
            "engine_forkchoiceUpdatedV1",
            "engine_getPayloadV6",
            "engine_getPayloadV5",
            "engine_getPayloadV4",
            "engine_getPayloadV3",
            "engine_getPayloadV2",
            "engine_getPayloadV1",
        ]
        result = await self._call("engine_exchangeCapabilities", [capabilities])
        return result

    async def get_client_version(self) -> Optional[dict]:
        """Get execution layer client version info via engine_getClientVersionV1."""
        try:
            from ..version import get_cl_client_version_info
            cl_info = get_cl_client_version_info()
            result = await self._call("engine_getClientVersionV1", [cl_info])
            if result and isinstance(result, list) and len(result) > 0:
                return result[0]
            return result
        except Exception as e:
            logger.debug(f"engine_getClientVersionV1 not supported or failed: {e}")
            return None

    async def close(self) -> None:
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _payload_to_dict(self, payload) -> dict:
        """Convert an ExecutionPayload to Engine API format."""
        txs = []
        for tx in payload.transactions:
            tx_bytes = bytes(tx)
            txs.append("0x" + tx_bytes.hex())

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
            "transactions": txs,
        }
        logger.debug(
            f"_payload_to_dict: blockHash={result['blockHash']}, "
            f"stateRoot={result['stateRoot']}, "
            f"receiptsRoot={result['receiptsRoot']}, "
            f"parentHash={result['parentHash']}, "
            f"prevRandao={result['prevRandao']}, "
            f"extra_data_len={len(bytes(payload.extra_data))}, tx_count={len(txs)}"
        )
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
