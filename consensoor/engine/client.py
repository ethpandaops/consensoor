"""Engine API client for communication with execution layer."""

import json
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
from . import ssz_types as ssz
from .. import metrics

logger = logging.getLogger(__name__)

SSZ_CONTENT_TYPE = "application/octet-stream"


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

    def __init__(self, url: str, jwt_secret: bytes, force_json: bool = False):
        self.url = url.rstrip("/")
        self.jwt_secret = jwt_secret
        self._session: Optional[aiohttp.ClientSession] = None
        self._request_id = 0
        self._genesis_time: Optional[int] = None
        # When True, never advertise or use SSZ-over-REST transport — every
        # Engine API call goes via JSON-RPC. Useful for visual debugging.
        self._force_json = force_json
        # SSZ REST endpoints (e.g. "POST /engine/v4/payloads") the EL has
        # advertised via engine_exchangeCapabilities. Populated lazily on
        # the first capabilities exchange; empty means JSON-RPC for all calls.
        self._ssz_endpoints: set[str] = set()

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

    def _ssz_supported(self, endpoint: str) -> bool:
        """Whether the EL has advertised the given SSZ REST endpoint."""
        if self._force_json:
            return False
        return endpoint in self._ssz_endpoints

    async def _ssz_request(
        self,
        http_method: str,
        path: str,
        body: Optional[bytes],
        metric_label: str,
    ) -> Optional[bytes]:
        """Issue an SSZ-over-REST Engine API request.

        Returns the raw response body bytes on 200, or ``None`` on 204.
        Raises ``EngineAPIError`` for non-success status codes.
        """
        session = await self._ensure_session()
        headers = {
            "Authorization": f"Bearer {self._create_jwt_token()}",
            "Accept": SSZ_CONTENT_TYPE,
        }
        if body is not None:
            headers["Content-Type"] = SSZ_CONTENT_TYPE

        url = f"{self.url}{path}"
        start = time.time()
        error_type: Optional[str] = None
        try:
            async with session.request(
                http_method, url, data=body, headers=headers
            ) as response:
                raw = await response.read()
                if response.status == 200:
                    return raw
                if response.status == 204:
                    return None
                error_type = str(response.status)
                text = raw.decode("utf-8", errors="replace")
                raise EngineAPIError(response.status, text)
        except aiohttp.ClientError as e:
            error_type = "connection_error"
            logger.error(f"Engine SSZ {http_method} {path} connection error: {e}")
            raise
        finally:
            metrics.record_engine_api_call(metric_label, time.time() - start, error_type)

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
                raw = await response.read()
                text = raw.decode("utf-8", errors="replace")

                if response.status != 200:
                    error_type = str(response.status)
                    logger.error(
                        f"Engine API {method} HTTP {response.status}: {text[:500]}"
                    )
                    raise EngineAPIError(response.status, text[:500])

                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    error_type = "invalid_json"
                    content_type = response.headers.get("Content-Type", "")
                    logger.error(
                        f"Engine API {method} returned non-JSON body "
                        f"(content-type={content_type!r}): {text[:500]}"
                    )
                    raise EngineAPIError(-1, f"Invalid JSON response: {text[:500]}")

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
        """Send a new payload to the execution layer (Engine API v5 - Amsterdam)."""
        payload_dict = self._payload_to_dict(execution_payload)
        return await self._new_payload_v5_via_dict(
            payload_dict, versioned_hashes, parent_beacon_block_root, execution_requests
        )

    async def new_payload_v5_raw(
        self,
        payload_dict: dict,
        versioned_hashes: list[bytes],
        parent_beacon_block_root: bytes,
        execution_requests: list,
    ) -> PayloadStatus:
        """Send a new payload to the execution layer using raw dict (no SSZ round-trip).

        This is used for GLOAS/ePBS where the payload comes directly from getPayloadV6
        and should be passed through without modification to avoid blockhash mismatch.
        """
        logger.info(
            f"newPayloadV5_raw: blockHash={payload_dict.get('blockHash')}, "
            f"execution_requests={execution_requests}, "
            f"parent_beacon_root={parent_beacon_block_root.hex()[:16]}"
        )
        return await self._new_payload_v5_via_dict(
            payload_dict, versioned_hashes, parent_beacon_block_root, execution_requests
        )

    async def _new_payload_v5_via_dict(
        self,
        payload_dict: dict,
        versioned_hashes: list[bytes],
        parent_beacon_block_root: bytes,
        execution_requests: list,
    ) -> PayloadStatus:
        if self._ssz_supported("POST /engine/v5/payloads"):
            req = ssz.NewPayloadV5Request(
                execution_payload=self._payload_dict_to_v4_ssz(payload_dict),
                expected_blob_versioned_hashes=[ssz.Bytes32(h) for h in versioned_hashes],
                parent_beacon_block_root=ssz.Bytes32(parent_beacon_block_root),
                execution_requests=[ssz.TransactionBytes(self._hex_bytes(r)) for r in execution_requests],
            )
            raw = await self._ssz_request(
                "POST", "/engine/v5/payloads", req.encode_bytes(), "engine_newPayloadV5_ssz"
            )
            if raw is None:
                return PayloadStatus.from_dict({"status": "SYNCING", "latestValidHash": None})
            decoded = ssz.PayloadStatusV1.decode_bytes(raw)
            return PayloadStatus.from_dict(self._payload_status_ssz_to_dict(decoded))

        params = [
            payload_dict,
            ["0x" + h.hex() for h in versioned_hashes],
            "0x" + parent_beacon_block_root.hex(),
            execution_requests,
        ]
        result = await self._call("engine_newPayloadV5", params)
        return PayloadStatus.from_dict(result)

    async def new_payload_v4(
        self,
        execution_payload,
        versioned_hashes: list[bytes],
        parent_beacon_block_root: bytes,
        execution_requests: list,
    ) -> PayloadStatus:
        """Send a new payload to the execution layer (Engine API v4 - Prague/Electra/Fulu)."""
        payload_dict = self._payload_to_dict(execution_payload)
        logger.info(
            f"newPayloadV4: blockHash={payload_dict.get('blockHash')}, "
            f"stateRoot={payload_dict.get('stateRoot')}, "
            f"timestamp={payload_dict.get('timestamp')}, "
            f"execution_requests={execution_requests}, "
            f"parent_beacon_root={parent_beacon_block_root.hex()[:16]}"
        )

        if self._ssz_supported("POST /engine/v4/payloads"):
            req = ssz.NewPayloadV4Request(
                execution_payload=self._payload_dict_to_v3_ssz(payload_dict),
                expected_blob_versioned_hashes=[ssz.Bytes32(h) for h in versioned_hashes],
                parent_beacon_block_root=ssz.Bytes32(parent_beacon_block_root),
                execution_requests=[ssz.TransactionBytes(self._hex_bytes(r)) for r in execution_requests],
            )
            raw = await self._ssz_request(
                "POST", "/engine/v4/payloads", req.encode_bytes(), "engine_newPayloadV4_ssz"
            )
            if raw is None:
                return PayloadStatus.from_dict({"status": "SYNCING", "latestValidHash": None})
            decoded = ssz.PayloadStatusV1.decode_bytes(raw)
            return PayloadStatus.from_dict(self._payload_status_ssz_to_dict(decoded))

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
        attrs = None
        if payload_attributes:
            attrs = {k: v for k, v in payload_attributes.items() if k != "slotNumber"}

        if self._ssz_supported("POST /engine/v3/forkchoice"):
            req = ssz.ForkchoiceUpdatedV3Request(
                forkchoice_state=self._fc_state_to_ssz(forkchoice_state),
                payload_attributes=[self._attrs_v3_dict_to_ssz(attrs)] if attrs else [],
            )
            raw = await self._ssz_request(
                "POST", "/engine/v3/forkchoice", req.encode_bytes(), "engine_forkchoiceUpdatedV3_ssz"
            )
            if raw is None:
                return ForkchoiceUpdateResponse.from_dict({
                    "payloadStatus": {"status": "SYNCING", "latestValidHash": None},
                    "payloadId": None,
                })
            decoded = ssz.ForkchoiceUpdatedResponseV1.decode_bytes(raw)
            return ForkchoiceUpdateResponse.from_dict(self._fcu_response_ssz_to_dict(decoded))

        params = [
            forkchoice_state.to_dict(),
            attrs,
        ]
        result = await self._call("engine_forkchoiceUpdatedV3", params)
        return ForkchoiceUpdateResponse.from_dict(result)

    async def forkchoice_updated_v4(
        self,
        forkchoice_state: ForkchoiceState,
        payload_attributes: Optional[dict] = None,
    ) -> ForkchoiceUpdateResponse:
        """Update the forkchoice state (Amsterdam / Gloas).

        Per `execution-apis/src/engine/amsterdam.md`, `PayloadAttributesV4`
        is `PayloadAttributesV3` plus a required `slotNumber` field. Strip
        nothing — pass attrs through verbatim. (Earlier revisions stripped
        `slotNumber` based on a misread; that produced a `PayloadAttributesV3`
        shape and geth replied -38003 "Invalid payload attributes".)
        """
        if self._ssz_supported("POST /engine/v4/forkchoice"):
            req = ssz.ForkchoiceUpdatedV4Request(
                forkchoice_state=self._fc_state_to_ssz(forkchoice_state),
                payload_attributes=[self._attrs_v4_dict_to_ssz(payload_attributes)] if payload_attributes else [],
            )
            raw = await self._ssz_request(
                "POST", "/engine/v4/forkchoice", req.encode_bytes(), "engine_forkchoiceUpdatedV4_ssz"
            )
            if raw is None:
                return ForkchoiceUpdateResponse.from_dict({
                    "payloadStatus": {"status": "SYNCING", "latestValidHash": None},
                    "payloadId": None,
                })
            decoded = ssz.ForkchoiceUpdatedResponseV1.decode_bytes(raw)
            return ForkchoiceUpdateResponse.from_dict(self._fcu_response_ssz_to_dict(decoded))

        params = [
            forkchoice_state.to_dict(),
            payload_attributes,
        ]
        result = await self._call("engine_forkchoiceUpdatedV4", params)
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

        if fork == "gloas":
            # Self-build mode: WE are both proposer and builder, so we DO
            # need a payload from the EL. forkchoiceUpdatedV4 with full
            # PayloadAttributesV4 (incl. slotNumber) is the right call.
            # Earlier revisions dropped attrs assuming "builder owns
            # payload" — true for an external-builder PBS flow, but not for
            # our self-build setup, and the drop meant we never received a
            # payload_id and never proposed any block.
            return await self.forkchoice_updated_v4(forkchoice_state, payload_attributes)
        if fork in ("fulu", "electra", "deneb"):
            return await self.forkchoice_updated_v3(forkchoice_state, payload_attributes)
        elif fork == "capella":
            return await self.forkchoice_updated_v2(forkchoice_state, payload_attributes)
        else:
            return await self.forkchoice_updated_v1(forkchoice_state, payload_attributes)

    async def get_payload_v6(self, payload_id: bytes) -> GetPayloadResponse:
        """Get an execution payload by ID (Amsterdam/Gloas)."""
        if self._ssz_supported("GET /engine/v6/payloads/{payload_id}"):
            raw = await self._ssz_request(
                "GET", f"/engine/v6/payloads/0x{payload_id.hex()}", None, "engine_getPayloadV6_ssz"
            )
            if raw is None:
                raise EngineAPIError(204, "EL returned 204 No Content for getPayloadV6")
            decoded = ssz.GetPayloadResponseV6.decode_bytes(raw)
            return GetPayloadResponse.from_dict({
                "executionPayload": self._v4_payload_ssz_to_dict(decoded.execution_payload),
                "blockValue": hex(int(decoded.block_value)),
                "blobsBundle": self._blobs_bundle_v2_to_dict(decoded.blobs_bundle),
                "shouldOverrideBuilder": bool(decoded.should_override_builder),
                "executionRequests": ["0x" + bytes(r).hex() for r in decoded.execution_requests],
            })

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
        if self._ssz_supported("GET /engine/v5/payloads/{payload_id}"):
            raw = await self._ssz_request(
                "GET", f"/engine/v5/payloads/0x{payload_id.hex()}", None, "engine_getPayloadV5_ssz"
            )
            if raw is None:
                raise EngineAPIError(204, "EL returned 204 No Content for getPayloadV5")
            decoded = ssz.GetPayloadResponseV5.decode_bytes(raw)
            return GetPayloadResponse.from_dict({
                "executionPayload": self._v3_payload_ssz_to_dict(decoded.execution_payload),
                "blockValue": hex(int(decoded.block_value)),
                "blobsBundle": self._blobs_bundle_v2_to_dict(decoded.blobs_bundle),
                "shouldOverrideBuilder": bool(decoded.should_override_builder),
                "executionRequests": ["0x" + bytes(r).hex() for r in decoded.execution_requests],
            })

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
        if self._ssz_supported("GET /engine/v4/payloads/{payload_id}"):
            raw = await self._ssz_request(
                "GET", f"/engine/v4/payloads/0x{payload_id.hex()}", None, "engine_getPayloadV4_ssz"
            )
            if raw is None:
                raise EngineAPIError(204, "EL returned 204 No Content for getPayloadV4")
            decoded = ssz.GetPayloadResponseV4.decode_bytes(raw)
            return GetPayloadResponse.from_dict({
                "executionPayload": self._v3_payload_ssz_to_dict(decoded.execution_payload),
                "blockValue": hex(int(decoded.block_value)),
                "blobsBundle": self._blobs_bundle_v1_to_dict(decoded.blobs_bundle),
                "shouldOverrideBuilder": bool(decoded.should_override_builder),
                "executionRequests": ["0x" + bytes(r).hex() for r in decoded.execution_requests],
            })

        result = await self._call("engine_getPayloadV4", ["0x" + payload_id.hex()])
        logger.debug(
            f"getPayloadV4 raw response: executionRequests={result.get('executionRequests')}, "
            f"blockHash={result.get('executionPayload', {}).get('blockHash')}"
        )
        return GetPayloadResponse.from_dict(result)

    async def get_payload_v3(self, payload_id: bytes) -> GetPayloadResponse:
        """Get an execution payload by ID (Deneb/Cancun)."""
        if self._ssz_supported("GET /engine/v3/payloads/{payload_id}"):
            raw = await self._ssz_request(
                "GET", f"/engine/v3/payloads/0x{payload_id.hex()}", None, "engine_getPayloadV3_ssz"
            )
            if raw is None:
                raise EngineAPIError(204, "EL returned 204 No Content for getPayloadV3")
            decoded = ssz.GetPayloadResponseV3.decode_bytes(raw)
            return GetPayloadResponse.from_dict({
                "executionPayload": self._v3_payload_ssz_to_dict(decoded.execution_payload),
                "blockValue": hex(int(decoded.block_value)),
                "blobsBundle": self._blobs_bundle_v1_to_dict(decoded.blobs_bundle),
                "shouldOverrideBuilder": bool(decoded.should_override_builder),
            })

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

        if self._ssz_supported("POST /engine/v3/payloads"):
            req = ssz.NewPayloadV3Request(
                execution_payload=self._payload_dict_to_v3_ssz(payload_dict),
                expected_blob_versioned_hashes=[ssz.Bytes32(h) for h in versioned_hashes],
                parent_beacon_block_root=ssz.Bytes32(parent_beacon_block_root),
            )
            raw = await self._ssz_request(
                "POST", "/engine/v3/payloads", req.encode_bytes(), "engine_newPayloadV3_ssz"
            )
            if raw is None:
                return PayloadStatus.from_dict({"status": "SYNCING", "latestValidHash": None})
            decoded = ssz.PayloadStatusV1.decode_bytes(raw)
            return PayloadStatus.from_dict(self._payload_status_ssz_to_dict(decoded))

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
        """Exchange capabilities with the execution layer.

        Advertises both JSON-RPC method names and the SSZ REST endpoints from
        execution-apis PR #764. The intersection of advertised SSZ endpoints
        with what the EL returns determines which calls use binary transport;
        all others fall back to JSON-RPC.
        """
        json_rpc_capabilities = [
            "engine_newPayloadV5",
            "engine_newPayloadV4",
            "engine_newPayloadV3",
            "engine_newPayloadV2",
            "engine_newPayloadV1",
            "engine_forkchoiceUpdatedV4",
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
        capabilities = json_rpc_capabilities if self._force_json else json_rpc_capabilities + ssz.SSZ_CAPABILITIES
        result = await self._call("engine_exchangeCapabilities", [capabilities])

        if self._force_json:
            self._ssz_endpoints = set()
            logger.info("Engine SSZ transport disabled by --engine-force-json; using JSON-RPC for all calls")
            return result

        el_caps = result if isinstance(result, list) else []
        offered = set(ssz.SSZ_CAPABILITIES)
        self._ssz_endpoints = {c for c in el_caps if c in offered}
        if self._ssz_endpoints:
            logger.info(
                f"Engine SSZ transport negotiated: {len(self._ssz_endpoints)} endpoints — "
                f"{sorted(self._ssz_endpoints)}"
            )
        else:
            logger.info("Engine SSZ transport not advertised by EL; using JSON-RPC for all calls")
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
        # Gloas/amsterdam fields — geth bal-devnet-6 rejects with
        # "nil slotnumber post-amsterdam" if these are missing.
        if hasattr(payload, 'slot_number'):
            result["slotNumber"] = hex(int(payload.slot_number))
        if hasattr(payload, 'block_access_list'):
            result["blockAccessList"] = "0x" + bytes(payload.block_access_list).hex()
        return result

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    # --- SSZ helpers ---

    @staticmethod
    def _hex_bytes(s: Optional[str]) -> bytes:
        if not s:
            return b""
        return bytes.fromhex(s[2:] if s.startswith("0x") else s)

    @staticmethod
    def _hex_int(s: Any) -> int:
        if s is None:
            return 0
        if isinstance(s, int):
            return s
        return int(s, 16) if s.startswith("0x") else int(s)

    @classmethod
    def _withdrawal_dict_to_ssz(cls, w: dict) -> Any:
        from ..spec.types.capella import Withdrawal
        return Withdrawal(
            index=cls._hex_int(w["index"]),
            validator_index=cls._hex_int(w["validatorIndex"]),
            address=ssz.Bytes20(cls._hex_bytes(w["address"])),
            amount=cls._hex_int(w["amount"]),
        )

    @classmethod
    def _payload_dict_to_v3_ssz(cls, p: dict) -> "ssz.ExecutionPayloadV3":
        return ssz.ExecutionPayloadV3(
            parent_hash=ssz.Hash32(cls._hex_bytes(p["parentHash"])),
            fee_recipient=ssz.ExecutionAddress(cls._hex_bytes(p["feeRecipient"])),
            state_root=ssz.Bytes32(cls._hex_bytes(p["stateRoot"])),
            receipts_root=ssz.Bytes32(cls._hex_bytes(p["receiptsRoot"])),
            logs_bloom=ssz.LogsBloom(cls._hex_bytes(p["logsBloom"])),
            prev_randao=ssz.Bytes32(cls._hex_bytes(p["prevRandao"])),
            block_number=cls._hex_int(p["blockNumber"]),
            gas_limit=cls._hex_int(p["gasLimit"]),
            gas_used=cls._hex_int(p["gasUsed"]),
            timestamp=cls._hex_int(p["timestamp"]),
            extra_data=ssz.ExtraData(cls._hex_bytes(p["extraData"])),
            base_fee_per_gas=cls._hex_int(p["baseFeePerGas"]),
            block_hash=ssz.Hash32(cls._hex_bytes(p["blockHash"])),
            transactions=[ssz.TransactionBytes(cls._hex_bytes(tx)) for tx in p["transactions"]],
            withdrawals=[cls._withdrawal_dict_to_ssz(w) for w in p.get("withdrawals", [])],
            blob_gas_used=cls._hex_int(p.get("blobGasUsed", "0x0")),
            excess_blob_gas=cls._hex_int(p.get("excessBlobGas", "0x0")),
        )

    @classmethod
    def _payload_dict_to_v4_ssz(cls, p: dict) -> "ssz.ExecutionPayloadV4":
        return ssz.ExecutionPayloadV4(
            parent_hash=ssz.Hash32(cls._hex_bytes(p["parentHash"])),
            fee_recipient=ssz.ExecutionAddress(cls._hex_bytes(p["feeRecipient"])),
            state_root=ssz.Bytes32(cls._hex_bytes(p["stateRoot"])),
            receipts_root=ssz.Bytes32(cls._hex_bytes(p["receiptsRoot"])),
            logs_bloom=ssz.LogsBloom(cls._hex_bytes(p["logsBloom"])),
            prev_randao=ssz.Bytes32(cls._hex_bytes(p["prevRandao"])),
            block_number=cls._hex_int(p["blockNumber"]),
            gas_limit=cls._hex_int(p["gasLimit"]),
            gas_used=cls._hex_int(p["gasUsed"]),
            timestamp=cls._hex_int(p["timestamp"]),
            extra_data=ssz.ExtraData(cls._hex_bytes(p["extraData"])),
            base_fee_per_gas=cls._hex_int(p["baseFeePerGas"]),
            block_hash=ssz.Hash32(cls._hex_bytes(p["blockHash"])),
            transactions=[ssz.TransactionBytes(cls._hex_bytes(tx)) for tx in p["transactions"]],
            withdrawals=[cls._withdrawal_dict_to_ssz(w) for w in p.get("withdrawals", [])],
            blob_gas_used=cls._hex_int(p.get("blobGasUsed", "0x0")),
            excess_blob_gas=cls._hex_int(p.get("excessBlobGas", "0x0")),
            block_access_list=ssz.TransactionBytes(cls._hex_bytes(p.get("blockAccessList", "0x"))),
            slot_number=cls._hex_int(p.get("slotNumber", "0x0")),
        )

    @classmethod
    def _v3_payload_ssz_to_dict(cls, ep: "ssz.ExecutionPayloadV3") -> dict:
        return {
            "parentHash": "0x" + bytes(ep.parent_hash).hex(),
            "feeRecipient": "0x" + bytes(ep.fee_recipient).hex(),
            "stateRoot": "0x" + bytes(ep.state_root).hex(),
            "receiptsRoot": "0x" + bytes(ep.receipts_root).hex(),
            "logsBloom": "0x" + bytes(ep.logs_bloom).hex(),
            "prevRandao": "0x" + bytes(ep.prev_randao).hex(),
            "blockNumber": hex(int(ep.block_number)),
            "gasLimit": hex(int(ep.gas_limit)),
            "gasUsed": hex(int(ep.gas_used)),
            "timestamp": hex(int(ep.timestamp)),
            "extraData": "0x" + bytes(ep.extra_data).hex(),
            "baseFeePerGas": hex(int(ep.base_fee_per_gas)),
            "blockHash": "0x" + bytes(ep.block_hash).hex(),
            "transactions": ["0x" + bytes(tx).hex() for tx in ep.transactions],
            "withdrawals": [
                {
                    "index": hex(int(w.index)),
                    "validatorIndex": hex(int(w.validator_index)),
                    "address": "0x" + bytes(w.address).hex(),
                    "amount": hex(int(w.amount)),
                }
                for w in ep.withdrawals
            ],
            "blobGasUsed": hex(int(ep.blob_gas_used)),
            "excessBlobGas": hex(int(ep.excess_blob_gas)),
        }

    @classmethod
    def _v4_payload_ssz_to_dict(cls, ep: "ssz.ExecutionPayloadV4") -> dict:
        d = {
            "parentHash": "0x" + bytes(ep.parent_hash).hex(),
            "feeRecipient": "0x" + bytes(ep.fee_recipient).hex(),
            "stateRoot": "0x" + bytes(ep.state_root).hex(),
            "receiptsRoot": "0x" + bytes(ep.receipts_root).hex(),
            "logsBloom": "0x" + bytes(ep.logs_bloom).hex(),
            "prevRandao": "0x" + bytes(ep.prev_randao).hex(),
            "blockNumber": hex(int(ep.block_number)),
            "gasLimit": hex(int(ep.gas_limit)),
            "gasUsed": hex(int(ep.gas_used)),
            "timestamp": hex(int(ep.timestamp)),
            "extraData": "0x" + bytes(ep.extra_data).hex(),
            "baseFeePerGas": hex(int(ep.base_fee_per_gas)),
            "blockHash": "0x" + bytes(ep.block_hash).hex(),
            "transactions": ["0x" + bytes(tx).hex() for tx in ep.transactions],
            "withdrawals": [
                {
                    "index": hex(int(w.index)),
                    "validatorIndex": hex(int(w.validator_index)),
                    "address": "0x" + bytes(w.address).hex(),
                    "amount": hex(int(w.amount)),
                }
                for w in ep.withdrawals
            ],
            "blobGasUsed": hex(int(ep.blob_gas_used)),
            "excessBlobGas": hex(int(ep.excess_blob_gas)),
            "blockAccessList": "0x" + bytes(ep.block_access_list).hex(),
            "slotNumber": hex(int(ep.slot_number)),
        }
        return d

    @staticmethod
    def _blobs_bundle_v1_to_dict(b: "ssz.BlobsBundleV1") -> dict:
        return {
            "commitments": ["0x" + bytes(c).hex() for c in b.commitments],
            "proofs": ["0x" + bytes(p).hex() for p in b.proofs],
            "blobs": ["0x" + bytes(blob).hex() for blob in b.blobs],
        }

    @staticmethod
    def _blobs_bundle_v2_to_dict(b: "ssz.BlobsBundleV2") -> dict:
        return {
            "commitments": ["0x" + bytes(c).hex() for c in b.commitments],
            "proofs": ["0x" + bytes(p).hex() for p in b.proofs],
            "blobs": ["0x" + bytes(blob).hex() for blob in b.blobs],
        }

    @staticmethod
    def _payload_status_ssz_to_dict(ps: "ssz.PayloadStatusV1") -> dict:
        status_str = ssz.INT_TO_PAYLOAD_STATUS.get(int(ps.status))
        if status_str is None:
            raise EngineAPIError(-1, f"Unknown PayloadStatus enum: {int(ps.status)}")
        result: dict = {"status": status_str}
        if len(ps.latest_valid_hash) == 1:
            result["latestValidHash"] = "0x" + bytes(ps.latest_valid_hash[0]).hex()
        else:
            result["latestValidHash"] = None
        if len(ps.validation_error) > 0:
            result["validationError"] = bytes(ps.validation_error).decode("utf-8", errors="replace")
        else:
            result["validationError"] = None
        return result

    @classmethod
    def _fcu_response_ssz_to_dict(cls, r: "ssz.ForkchoiceUpdatedResponseV1") -> dict:
        result = {"payloadStatus": cls._payload_status_ssz_to_dict(r.payload_status)}
        if len(r.payload_id) == 1:
            result["payloadId"] = "0x" + bytes(r.payload_id[0]).hex()
        else:
            result["payloadId"] = None
        return result

    @classmethod
    def _fc_state_to_ssz(cls, state: ForkchoiceState) -> "ssz.ForkchoiceStateV1":
        return ssz.ForkchoiceStateV1(
            head_block_hash=ssz.Bytes32(state.head_block_hash),
            safe_block_hash=ssz.Bytes32(state.safe_block_hash),
            finalized_block_hash=ssz.Bytes32(state.finalized_block_hash),
        )

    @classmethod
    def _attrs_v1_dict_to_ssz(cls, a: dict) -> "ssz.PayloadAttributesV1":
        return ssz.PayloadAttributesV1(
            timestamp=cls._hex_int(a["timestamp"]),
            prev_randao=ssz.Bytes32(cls._hex_bytes(a["prevRandao"])),
            suggested_fee_recipient=ssz.Bytes20(cls._hex_bytes(a["suggestedFeeRecipient"])),
        )

    @classmethod
    def _attrs_v2_dict_to_ssz(cls, a: dict) -> "ssz.PayloadAttributesV2":
        return ssz.PayloadAttributesV2(
            timestamp=cls._hex_int(a["timestamp"]),
            prev_randao=ssz.Bytes32(cls._hex_bytes(a["prevRandao"])),
            suggested_fee_recipient=ssz.Bytes20(cls._hex_bytes(a["suggestedFeeRecipient"])),
            withdrawals=[cls._withdrawal_dict_to_ssz(w) for w in a.get("withdrawals", [])],
        )

    @classmethod
    def _attrs_v3_dict_to_ssz(cls, a: dict) -> "ssz.PayloadAttributesV3":
        return ssz.PayloadAttributesV3(
            timestamp=cls._hex_int(a["timestamp"]),
            prev_randao=ssz.Bytes32(cls._hex_bytes(a["prevRandao"])),
            suggested_fee_recipient=ssz.Bytes20(cls._hex_bytes(a["suggestedFeeRecipient"])),
            withdrawals=[cls._withdrawal_dict_to_ssz(w) for w in a.get("withdrawals", [])],
            parent_beacon_block_root=ssz.Bytes32(cls._hex_bytes(a["parentBeaconBlockRoot"])),
        )

    @classmethod
    def _attrs_v4_dict_to_ssz(cls, a: dict) -> "ssz.PayloadAttributesV4":
        return ssz.PayloadAttributesV4(
            timestamp=cls._hex_int(a["timestamp"]),
            prev_randao=ssz.Bytes32(cls._hex_bytes(a["prevRandao"])),
            suggested_fee_recipient=ssz.Bytes20(cls._hex_bytes(a["suggestedFeeRecipient"])),
            withdrawals=[cls._withdrawal_dict_to_ssz(w) for w in a.get("withdrawals", [])],
            parent_beacon_block_root=ssz.Bytes32(cls._hex_bytes(a["parentBeaconBlockRoot"])),
            slot_number=cls._hex_int(a["slotNumber"]),
        )
