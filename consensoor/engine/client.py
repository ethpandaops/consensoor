"""Engine API client for communication with execution layer.

Implements the REST + SSZ Engine API v2 (execution-apis PR #793): hot-path
calls go to fork-scoped REST endpoints under ``/engine/v2/{fork}/...`` with
SSZ bodies over HTTP/2 (h2c), with structured JSON for capabilities, identity
and RFC-7807 errors. The legacy JSON-RPC engine API is kept as the
transition-window fallback: if the EL does not expose ``/engine/v2/...`` (or
doesn't advertise the URL fork), every call falls back to JSON-RPC for the
lifetime of the connection.
"""

import json
import logging
import time
from typing import Optional, Any

import httpx
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
JSON_CONTENT_TYPE = "application/json"

# Maps a consensus-layer fork name (what `_get_fork_for_timestamp` returns) to
# the execution-layer fork segment used in `/engine/v2/{fork}/...` URLs.
CL_TO_EL_FORK = {
    "bellatrix": "paris",
    "capella": "shanghai",
    "deneb": "cancun",
    "electra": "prague",
    "fulu": "osaka",
    "gloas": "amsterdam",
}


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
    """Client for Ethereum Engine API (v2 REST + SSZ, JSON-RPC fallback)."""

    def __init__(self, url: str, jwt_secret: bytes, force_json: bool = False):
        self.url = url.rstrip("/")
        self.jwt_secret = jwt_secret
        self._jsonrpc_client: Optional[httpx.AsyncClient] = None
        self._v2_client: Optional[httpx.AsyncClient] = None
        self._request_id = 0
        self._genesis_time: Optional[int] = None
        self._client_version: Optional[str] = None
        # When True, never use the v2 REST transport — every Engine API call
        # goes via legacy JSON-RPC. Useful for visual debugging.
        self._force_json = force_json
        # Populated by exchange_capabilities() from GET /engine/v2/capabilities.
        self._v2_enabled = False
        self._v2_supported_forks: set[str] = set()
        self._v2_blob_revisions: set[str] = set()
        self._v2_limits: dict = {}

    def set_genesis_time(self, genesis_time: int) -> None:
        """Set the genesis time for fork calculations."""
        self._genesis_time = genesis_time

    # --- transport ---------------------------------------------------------

    async def _ensure_jsonrpc_client(self) -> httpx.AsyncClient:
        if self._jsonrpc_client is None or self._jsonrpc_client.is_closed:
            self._jsonrpc_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        return self._jsonrpc_client

    async def _ensure_v2_client(self) -> httpx.AsyncClient:
        if self._v2_client is None or self._v2_client.is_closed:
            # h2c prior-knowledge: disabling HTTP/1.1 makes httpx speak
            # cleartext HTTP/2 directly, as #793 § Transport requires.
            self._v2_client = httpx.AsyncClient(
                http1=False, http2=True, timeout=httpx.Timeout(30.0)
            )
        return self._v2_client

    def _create_jwt_token(self) -> str:
        """Create a JWT token for authentication (iat only; clv removed in #793)."""
        now = int(time.time())
        payload = {"iat": now}
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")

    def _client_version_header(self) -> str:
        """Value for the X-Engine-Client-Version request header (#793)."""
        if self._client_version is None:
            try:
                from ..version import get_cl_client_version_info
                info = get_cl_client_version_info()
                self._client_version = f"{info.get('code', 'CO')}/{info.get('version', '0')}"
            except Exception:
                self._client_version = "CO/unknown"
        return self._client_version

    def _raise_problem(self, status: int, raw: bytes, content_type: str) -> None:
        """Raise EngineAPIError from an RFC-7807 application/problem+json body."""
        ptype: Optional[str] = None
        detail: Optional[str] = None
        if "json" in (content_type or "").lower():
            try:
                j = json.loads(raw.decode("utf-8", errors="replace"))
                ptype = j.get("type")
                detail = j.get("detail")
            except Exception:
                pass
        msg = ptype or f"HTTP {status}"
        if detail:
            msg = f"{msg}: {detail}"
        raise EngineAPIError(status, msg)

    async def _v2_request(
        self,
        http_method: str,
        path: str,
        body: Optional[bytes],
        metric_label: str,
        accept: str = "ssz",
    ) -> Optional[bytes]:
        """Issue a v2 REST Engine API request.

        Returns the raw response body on 200, ``None`` on 204. Raises
        ``EngineAPIError`` for RFC-7807 error responses.
        """
        client = await self._ensure_v2_client()
        headers = {
            "Authorization": f"Bearer {self._create_jwt_token()}",
            "X-Engine-Client-Version": self._client_version_header(),
            "Accept": JSON_CONTENT_TYPE if accept == "json" else SSZ_CONTENT_TYPE,
        }
        if body is not None:
            headers["Content-Type"] = SSZ_CONTENT_TYPE

        url = f"{self.url}{path}"
        start = time.time()
        error_type: Optional[str] = None
        try:
            resp = await client.request(http_method, url, content=body, headers=headers)
            raw = resp.content
            if resp.status_code == 200:
                return raw
            if resp.status_code == 204:
                return None
            error_type = str(resp.status_code)
            self._raise_problem(
                resp.status_code, raw, resp.headers.get("content-type", "")
            )
        except httpx.HTTPError as e:
            error_type = error_type or "connection_error"
            logger.error(f"Engine v2 {http_method} {path} error: {e}")
            raise
        finally:
            metrics.record_engine_api_call(metric_label, time.time() - start, error_type)

    async def _call(self, method: str, params: list) -> Any:
        """Make a JSON-RPC call to the legacy Engine API (fallback transport)."""
        client = await self._ensure_jsonrpc_client()
        self._request_id += 1

        headers = {
            "Content-Type": JSON_CONTENT_TYPE,
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
            resp = await client.post(self.url, json=payload, headers=headers)
            text = resp.text
            if resp.status_code != 200:
                error_type = str(resp.status_code)
                logger.error(f"Engine API {method} HTTP {resp.status_code}: {text[:500]}")
                raise EngineAPIError(resp.status_code, text[:500])

            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                error_type = "invalid_json"
                content_type = resp.headers.get("Content-Type", "")
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
        except httpx.HTTPError as e:
            error_type = error_type or "connection_error"
            logger.error(f"Engine API connection error: {e}")
            raise
        finally:
            metrics.record_engine_api_call(method, time.time() - start_time, error_type)

    # --- fork resolution ---------------------------------------------------

    def _get_fork_for_timestamp(self, timestamp: int) -> str:
        """Determine which (consensus-layer) fork is active for a timestamp."""
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

    def _el_fork_for_timestamp(self, timestamp: int) -> str:
        """Execution-layer fork segment (paris..amsterdam) for the URL."""
        return CL_TO_EL_FORK[self._get_fork_for_timestamp(timestamp)]

    def _use_v2(self, el_fork: str) -> bool:
        """Whether the v2 REST transport should be used for the given fork."""
        return (
            self._v2_enabled
            and not self._force_json
            and el_fork in self._v2_supported_forks
        )

    def _use_blobs_v2(self, revision: str) -> bool:
        return (
            self._v2_enabled
            and not self._force_json
            and revision in self._v2_blob_revisions
        )

    # --- capabilities & identity ------------------------------------------

    async def exchange_capabilities(self) -> Any:
        """Probe v2 capabilities, falling back to legacy JSON-RPC negotiation.

        #793 replaces ``engine_exchangeCapabilities`` with a single
        ``GET /engine/v2/capabilities`` returning structured JSON. If that
        endpoint is unavailable (legacy EL → 404 / connection error) the
        client disables v2 and negotiates over JSON-RPC instead.
        """
        if self._force_json:
            self._v2_enabled = False
            logger.info("Engine v2 REST disabled by --engine-force-json; using JSON-RPC")
            return await self._legacy_exchange_capabilities()

        try:
            raw = await self._v2_request(
                "GET", "/engine/v2/capabilities", None, "engine_v2_capabilities", accept="json"
            )
            caps = json.loads(raw.decode("utf-8")) if raw else {}
            self._v2_supported_forks = set(caps.get("supported_forks", []))
            independently = caps.get("independently_versioned", {}) or {}
            self._v2_blob_revisions = set(independently.get("blobs", []))
            self._v2_limits = caps.get("limits", {}) or {}
            self._v2_enabled = bool(self._v2_supported_forks)
            if self._v2_enabled:
                logger.info(
                    "Engine API v2 REST enabled — forks="
                    f"{sorted(self._v2_supported_forks)}, blobs="
                    f"{sorted(self._v2_blob_revisions)}, limits={self._v2_limits}"
                )
                return caps
            logger.info("Engine v2 /capabilities advertised no forks; falling back to JSON-RPC")
        except Exception as e:
            logger.info(f"Engine v2 REST not available ({e}); falling back to JSON-RPC engine API")
            self._v2_enabled = False

        return await self._legacy_exchange_capabilities()

    async def _legacy_exchange_capabilities(self) -> list[str]:
        """Legacy JSON-RPC engine_exchangeCapabilities handshake."""
        capabilities = [
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
        try:
            return await self._call("engine_exchangeCapabilities", [capabilities])
        except Exception as e:
            logger.error(f"engine_exchangeCapabilities failed: {e}")
            return []

    async def get_client_version(self) -> Optional[dict]:
        """Get EL client version info (GET /engine/v2/identity, else JSON-RPC)."""
        if self._use_identity_v2():
            try:
                raw = await self._v2_request(
                    "GET", "/engine/v2/identity", None, "engine_v2_identity", accept="json"
                )
                data = json.loads(raw.decode("utf-8")) if raw else []
                if isinstance(data, list) and data:
                    return data[0]
                return data or None
            except Exception as e:
                logger.debug(f"GET /engine/v2/identity failed: {e}")
                return None

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

    def _use_identity_v2(self) -> bool:
        return self._v2_enabled and not self._force_json

    # --- newPayload --------------------------------------------------------

    async def _new_payload_v2(
        self,
        el_fork: str,
        payload_dict: dict,
        parent_beacon_block_root: bytes,
        execution_requests: list,
    ) -> PayloadStatus:
        """POST /engine/v2/{fork}/payloads."""
        envelope_cls = ssz.ENVELOPE_BY_FORK[el_fork]
        kwargs: dict = {"payload": self._payload_dict_to_ssz(el_fork, payload_dict)}
        if el_fork in ("cancun", "prague", "osaka", "amsterdam"):
            kwargs["parent_beacon_block_root"] = ssz.Root(parent_beacon_block_root)
        if el_fork in ("prague", "osaka", "amsterdam"):
            kwargs["execution_requests"] = [
                ssz.ExecutionRequestBytes(self._hex_bytes(r)) for r in execution_requests
            ]
        envelope = envelope_cls(**kwargs)
        raw = await self._v2_request(
            "POST", f"/engine/v2/{el_fork}/payloads", envelope.encode_bytes(),
            "engine_v2_newPayload",
        )
        if raw is None:
            return PayloadStatus.from_dict({"status": "SYNCING", "latestValidHash": None})
        decoded = ssz.PayloadStatus.decode_bytes(raw)
        return PayloadStatus.from_dict(self._payload_status_ssz_to_dict(decoded))

    async def new_payload_v5(
        self, execution_payload, versioned_hashes, parent_beacon_block_root, execution_requests
    ) -> PayloadStatus:
        """JSON-RPC engine_newPayloadV5 (Amsterdam / Gloas)."""
        payload_dict = self._payload_to_dict(execution_payload)
        return await self._new_payload_v5_via_dict(
            payload_dict, versioned_hashes, parent_beacon_block_root, execution_requests
        )

    async def new_payload_v5_raw(
        self, payload_dict: dict, versioned_hashes, parent_beacon_block_root, execution_requests
    ) -> PayloadStatus:
        """Send a new payload from a raw dict (GLOAS self-build pass-through)."""
        timestamp = self._hex_int(payload_dict.get("timestamp", "0x0"))
        el_fork = self._el_fork_for_timestamp(timestamp) if timestamp else "amsterdam"
        logger.info(
            f"newPayloadV5_raw: blockHash={payload_dict.get('blockHash')}, "
            f"execution_requests={execution_requests}, "
            f"parent_beacon_root={parent_beacon_block_root.hex()[:16]}, el_fork={el_fork}"
        )
        if self._use_v2(el_fork):
            return await self._new_payload_v2(
                el_fork, payload_dict, parent_beacon_block_root, execution_requests
            )
        return await self._new_payload_v5_via_dict(
            payload_dict, versioned_hashes, parent_beacon_block_root, execution_requests
        )

    async def _new_payload_v5_via_dict(
        self, payload_dict, versioned_hashes, parent_beacon_block_root, execution_requests
    ) -> PayloadStatus:
        params = [
            payload_dict,
            ["0x" + h.hex() for h in versioned_hashes],
            "0x" + parent_beacon_block_root.hex(),
            execution_requests,
        ]
        result = await self._call("engine_newPayloadV5", params)
        return PayloadStatus.from_dict(result)

    async def new_payload_v4(
        self, execution_payload, versioned_hashes, parent_beacon_block_root, execution_requests
    ) -> PayloadStatus:
        """JSON-RPC engine_newPayloadV4 (Prague / Electra / Fulu)."""
        payload_dict = self._payload_to_dict(execution_payload)
        params = [
            payload_dict,
            ["0x" + h.hex() for h in versioned_hashes],
            "0x" + parent_beacon_block_root.hex(),
            execution_requests,
        ]
        result = await self._call("engine_newPayloadV4", params)
        return PayloadStatus.from_dict(result)

    async def new_payload_v3(
        self, execution_payload, versioned_hashes, parent_beacon_block_root
    ) -> PayloadStatus:
        """JSON-RPC engine_newPayloadV3 (Deneb)."""
        payload_dict = self._payload_to_dict(execution_payload)
        params = [
            payload_dict,
            ["0x" + h.hex() for h in versioned_hashes],
            "0x" + parent_beacon_block_root.hex(),
        ]
        result = await self._call("engine_newPayloadV3", params)
        return PayloadStatus.from_dict(result)

    async def new_payload_v2(self, execution_payload) -> PayloadStatus:
        """JSON-RPC engine_newPayloadV2 (Capella)."""
        payload_dict = self._payload_to_dict(execution_payload)
        result = await self._call("engine_newPayloadV2", [payload_dict])
        return PayloadStatus.from_dict(result)

    async def new_payload_v1(self, execution_payload) -> PayloadStatus:
        """JSON-RPC engine_newPayloadV1 (Bellatrix)."""
        payload_dict = self._payload_to_dict(execution_payload)
        for key in ("withdrawals", "blobGasUsed", "excessBlobGas"):
            payload_dict.pop(key, None)
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
        """Send a new payload using v2 REST when available, else JSON-RPC."""
        if timestamp is None:
            timestamp = int(time.time())

        el_fork = self._el_fork_for_timestamp(timestamp)
        if self._use_v2(el_fork):
            payload_dict = self._payload_to_dict(execution_payload)
            return await self._new_payload_v2(
                el_fork, payload_dict,
                parent_beacon_block_root or b"\x00" * 32,
                execution_requests or [],
            )

        fork = self._get_fork_for_timestamp(timestamp)
        logger.debug(f"new_payload (JSON-RPC): timestamp={timestamp}, fork={fork}")
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

    # --- forkchoiceUpdated --------------------------------------------------

    async def _forkchoice_updated_v2(
        self,
        el_fork: str,
        forkchoice_state: ForkchoiceState,
        payload_attributes: Optional[dict],
        custody_columns: Optional[Any] = None,
    ) -> ForkchoiceUpdateResponse:
        """POST /engine/v2/{fork}/forkchoice."""
        fcu_cls = ssz.FORKCHOICE_UPDATE_BY_FORK[el_fork]
        attrs_list = (
            [self._attrs_dict_to_ssz(el_fork, payload_attributes)] if payload_attributes else []
        )
        kwargs: dict = {
            "forkchoice_state": self._fc_state_to_ssz(forkchoice_state),
            "payload_attributes": attrs_list,
        }
        if el_fork == "amsterdam":
            # Custody set unchanged when the optional field is absent.
            kwargs["custody_columns"] = [custody_columns] if custody_columns is not None else []
        req = fcu_cls(**kwargs)
        raw = await self._v2_request(
            "POST", f"/engine/v2/{el_fork}/forkchoice", req.encode_bytes(),
            "engine_v2_forkchoiceUpdated",
        )
        if raw is None:
            return ForkchoiceUpdateResponse.from_dict({
                "payloadStatus": {"status": "SYNCING", "latestValidHash": None},
                "payloadId": None,
            })
        decoded = ssz.ForkchoiceUpdateResponse.decode_bytes(raw)
        return ForkchoiceUpdateResponse.from_dict(self._fcu_response_ssz_to_dict(decoded))

    async def forkchoice_updated_v4(
        self, forkchoice_state: ForkchoiceState, payload_attributes: Optional[dict] = None
    ) -> ForkchoiceUpdateResponse:
        """JSON-RPC engine_forkchoiceUpdatedV4 (Amsterdam / Gloas)."""
        params = [forkchoice_state.to_dict(), payload_attributes]
        result = await self._call("engine_forkchoiceUpdatedV4", params)
        return ForkchoiceUpdateResponse.from_dict(result)

    async def forkchoice_updated_v3(
        self, forkchoice_state: ForkchoiceState, payload_attributes: Optional[dict] = None
    ) -> ForkchoiceUpdateResponse:
        """JSON-RPC engine_forkchoiceUpdatedV3 (Deneb / Electra / Fulu)."""
        attrs = None
        if payload_attributes:
            attrs = {k: v for k, v in payload_attributes.items() if k != "slotNumber"}
        params = [forkchoice_state.to_dict(), attrs]
        result = await self._call("engine_forkchoiceUpdatedV3", params)
        return ForkchoiceUpdateResponse.from_dict(result)

    async def forkchoice_updated_v2(
        self, forkchoice_state: ForkchoiceState, payload_attributes: Optional[dict] = None
    ) -> ForkchoiceUpdateResponse:
        """JSON-RPC engine_forkchoiceUpdatedV2 (Capella)."""
        attrs = None
        if payload_attributes:
            attrs = {k: v for k, v in payload_attributes.items() if k != "parentBeaconBlockRoot"}
        params = [forkchoice_state.to_dict(), attrs]
        result = await self._call("engine_forkchoiceUpdatedV2", params)
        return ForkchoiceUpdateResponse.from_dict(result)

    async def forkchoice_updated_v1(
        self, forkchoice_state: ForkchoiceState, payload_attributes: Optional[dict] = None
    ) -> ForkchoiceUpdateResponse:
        """JSON-RPC engine_forkchoiceUpdatedV1 (Bellatrix)."""
        attrs = None
        if payload_attributes:
            attrs = {k: v for k, v in payload_attributes.items() if k not in ("parentBeaconBlockRoot", "withdrawals")}
        params = [forkchoice_state.to_dict(), attrs]
        result = await self._call("engine_forkchoiceUpdatedV1", params)
        return ForkchoiceUpdateResponse.from_dict(result)

    async def forkchoice_updated(
        self,
        forkchoice_state: ForkchoiceState,
        payload_attributes: Optional[dict] = None,
        timestamp: Optional[int] = None,
    ) -> ForkchoiceUpdateResponse:
        """Update forkchoice via v2 REST when available, else JSON-RPC."""
        if timestamp is None:
            timestamp = int(time.time())

        el_fork = self._el_fork_for_timestamp(timestamp)
        if self._use_v2(el_fork):
            return await self._forkchoice_updated_v2(el_fork, forkchoice_state, payload_attributes)

        fork = self._get_fork_for_timestamp(timestamp)
        logger.debug(f"forkchoice_updated (JSON-RPC): timestamp={timestamp}, fork={fork}")
        if fork == "gloas":
            return await self.forkchoice_updated_v4(forkchoice_state, payload_attributes)
        if fork in ("fulu", "electra", "deneb"):
            return await self.forkchoice_updated_v3(forkchoice_state, payload_attributes)
        elif fork == "capella":
            return await self.forkchoice_updated_v2(forkchoice_state, payload_attributes)
        else:
            return await self.forkchoice_updated_v1(forkchoice_state, payload_attributes)

    # --- getPayload --------------------------------------------------------

    async def _get_payload_v2(self, el_fork: str, payload_id: bytes) -> GetPayloadResponse:
        """GET /engine/v2/{fork}/payloads/{payloadId}."""
        raw = await self._v2_request(
            "GET", f"/engine/v2/{el_fork}/payloads/0x{payload_id.hex()}", None,
            "engine_v2_getPayload",
        )
        if raw is None:
            raise EngineAPIError(204, f"EL returned 204 for GET /engine/v2/{el_fork}/payloads")
        decoded = ssz.BUILT_PAYLOAD_BY_FORK[el_fork].decode_bytes(raw)
        return GetPayloadResponse.from_dict(self._built_payload_ssz_to_dict(el_fork, decoded))

    async def get_payload_v6(self, payload_id: bytes) -> GetPayloadResponse:
        """JSON-RPC engine_getPayloadV6 (Amsterdam / Gloas)."""
        result = await self._call("engine_getPayloadV6", ["0x" + payload_id.hex()])
        return GetPayloadResponse.from_dict(result)

    async def get_payload_v5(self, payload_id: bytes) -> GetPayloadResponse:
        """JSON-RPC engine_getPayloadV5 (Osaka / Fulu)."""
        result = await self._call("engine_getPayloadV5", ["0x" + payload_id.hex()])
        return GetPayloadResponse.from_dict(result)

    async def get_payload_v4(self, payload_id: bytes) -> GetPayloadResponse:
        """JSON-RPC engine_getPayloadV4 (Electra / Prague)."""
        result = await self._call("engine_getPayloadV4", ["0x" + payload_id.hex()])
        return GetPayloadResponse.from_dict(result)

    async def get_payload_v3(self, payload_id: bytes) -> GetPayloadResponse:
        """JSON-RPC engine_getPayloadV3 (Deneb / Cancun)."""
        result = await self._call("engine_getPayloadV3", ["0x" + payload_id.hex()])
        return GetPayloadResponse.from_dict(result)

    async def get_payload_v2(self, payload_id: bytes) -> GetPayloadResponse:
        """JSON-RPC engine_getPayloadV2 (Capella / Shanghai)."""
        result = await self._call("engine_getPayloadV2", ["0x" + payload_id.hex()])
        return GetPayloadResponse.from_dict(result)

    async def get_payload_v1(self, payload_id: bytes) -> GetPayloadResponse:
        """JSON-RPC engine_getPayloadV1 (Bellatrix / Paris)."""
        result = await self._call("engine_getPayloadV1", ["0x" + payload_id.hex()])
        return GetPayloadResponse.from_dict(result)

    async def get_payload(self, payload_id: bytes, timestamp: Optional[int] = None) -> GetPayloadResponse:
        """Get a built payload via v2 REST when available, else JSON-RPC."""
        if timestamp is None:
            timestamp = int(time.time())

        el_fork = self._el_fork_for_timestamp(timestamp)
        if self._use_v2(el_fork):
            logger.info(f"get_payload (v2): fork={el_fork}, payload_id={payload_id.hex()}")
            return await self._get_payload_v2(el_fork, payload_id)

        fork = self._get_fork_for_timestamp(timestamp)
        logger.info(f"get_payload (JSON-RPC): fork={fork}, payload_id={payload_id.hex()}")
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

    # --- blob pool (POST /engine/v2/blobs/vN) -----------------------------

    async def get_blobs_v1(self, versioned_hashes: list[bytes]) -> Optional[list[dict]]:
        """POST /engine/v2/blobs/v1 (Cancun whole-blob, single proof)."""
        if not self._use_blobs_v2("v1"):
            return None
        req = ssz.VersionedHashList([ssz.VersionedHash(h) for h in versioned_hashes])
        raw = await self._v2_request("POST", "/engine/v2/blobs/v1", req.encode_bytes(), "engine_v2_getBlobsV1")
        if raw is None:
            return None
        resp = ssz.BlobsV1Response.decode_bytes(raw)
        return [
            {
                "available": bool(e.available),
                "blob": "0x" + bytes(e.contents.blob).hex(),
                "proof": "0x" + bytes(e.contents.proof).hex(),
            }
            for e in resp
        ]

    async def get_blobs_v2(self, versioned_hashes: list[bytes]) -> Optional[list[dict]]:
        """POST /engine/v2/blobs/v2 (Osaka, all-or-nothing cell proofs)."""
        return await self._get_blobs_cell("v2", versioned_hashes)

    async def get_blobs_v3(self, versioned_hashes: list[bytes]) -> Optional[list[dict]]:
        """POST /engine/v2/blobs/v3 (Osaka, partial-response cell proofs)."""
        return await self._get_blobs_cell("v3", versioned_hashes)

    async def _get_blobs_cell(self, revision: str, versioned_hashes: list[bytes]) -> Optional[list[dict]]:
        if not self._use_blobs_v2(revision):
            return None
        req = ssz.VersionedHashList([ssz.VersionedHash(h) for h in versioned_hashes])
        raw = await self._v2_request("POST", f"/engine/v2/blobs/{revision}", req.encode_bytes(), f"engine_v2_getBlobs{revision.upper()}")
        if raw is None:
            return None
        resp = ssz.BlobsV2Response.decode_bytes(raw)
        return [
            {
                "available": bool(e.available),
                "blob": "0x" + bytes(e.contents.blob).hex(),
                "proofs": ["0x" + bytes(p).hex() for p in e.contents.proofs],
            }
            for e in resp
        ]

    async def get_blobs_v4(self, versioned_hashes: list[bytes], indices_bitarray: Optional[Any] = None) -> Optional[list[dict]]:
        """POST /engine/v2/blobs/v4 (Amsterdam, cell-range selection)."""
        if not self._use_blobs_v2("v4"):
            return None
        bitarray = indices_bitarray if indices_bitarray is not None else ssz.CustodyColumns()
        req = ssz.BlobsV4Request(
            versioned_hashes=[ssz.VersionedHash(h) for h in versioned_hashes],
            indices_bitarray=bitarray,
        )
        raw = await self._v2_request("POST", "/engine/v2/blobs/v4", req.encode_bytes(), "engine_v2_getBlobsV4")
        if raw is None:
            return None
        resp = ssz.BlobsV4Response.decode_bytes(raw)
        out = []
        for e in resp:
            cells = ["0x" + bytes(c[0]).hex() if len(c) == 1 else None for c in e.contents.blob_cells]
            proofs = ["0x" + bytes(p[0]).hex() if len(p) == 1 else None for p in e.contents.proofs]
            out.append({"available": bool(e.available), "blob_cells": cells, "proofs": proofs})
        return out

    async def get_blobs(
        self, versioned_hashes: list[bytes], timestamp: Optional[int] = None
    ) -> Optional[list[dict]]:
        """Fetch blobs from the EL pool using the fork-appropriate revision.

        Picks the highest ``/blobs/vN`` revision the EL advertised for the
        active fork era. Returns the per-entry dicts (see ``get_blobs_vN``),
        or ``None`` when the EL serves no matching revision / can't serve.
        """
        el_fork = self._el_fork_for_timestamp(timestamp or int(time.time()))
        candidates = {
            "cancun": ["v1"],
            "prague": ["v1"],
            "osaka": ["v3", "v2", "v1"],
            "amsterdam": ["v4", "v3", "v2", "v1"],
        }.get(el_fork, ["v1"])
        for rev in candidates:
            if not self._use_blobs_v2(rev):
                continue
            if rev == "v1":
                return await self.get_blobs_v1(versioned_hashes)
            if rev == "v2":
                return await self.get_blobs_v2(versioned_hashes)
            if rev == "v3":
                return await self.get_blobs_v3(versioned_hashes)
            if rev == "v4":
                return await self.get_blobs_v4(versioned_hashes)
        return None

    # --- historical bodies -------------------------------------------------

    async def get_payload_bodies_by_hash(
        self, block_hashes: list[bytes], timestamp: Optional[int] = None
    ) -> Optional[list[dict]]:
        """POST /engine/v2/{fork}/bodies/hash."""
        el_fork = self._el_fork_for_timestamp(timestamp or int(time.time()))
        if not self._use_v2(el_fork):
            return None
        req = ssz.BodiesByHashRequest([ssz.Hash32(h) for h in block_hashes])
        raw = await self._v2_request(
            "POST", f"/engine/v2/{el_fork}/bodies/hash", req.encode_bytes(),
            "engine_v2_getPayloadBodiesByHash",
        )
        if raw is None:
            return None
        decoded = ssz.BODIES_RESPONSE_BY_FORK[el_fork].decode_bytes(raw)
        return [self._body_entry_to_dict(e) for e in decoded]

    async def get_payload_bodies_by_range(
        self, start: int, count: int, timestamp: Optional[int] = None
    ) -> Optional[list[dict]]:
        """GET /engine/v2/{fork}/bodies?from=N&count=M."""
        el_fork = self._el_fork_for_timestamp(timestamp or int(time.time()))
        if not self._use_v2(el_fork):
            return None
        raw = await self._v2_request(
            "GET", f"/engine/v2/{el_fork}/bodies?from={start}&count={count}", None,
            "engine_v2_getPayloadBodiesByRange",
        )
        if raw is None:
            return None
        decoded = ssz.BODIES_RESPONSE_BY_FORK[el_fork].decode_bytes(raw)
        return [self._body_entry_to_dict(e) for e in decoded]

    async def close(self) -> None:
        """Close the client sessions."""
        for client in (self._jsonrpc_client, self._v2_client):
            if client is not None and not client.is_closed:
                await client.aclose()
        self._jsonrpc_client = None
        self._v2_client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    # --- payload dict conversion ------------------------------------------

    def _payload_to_dict(self, payload) -> dict:
        """Convert an ExecutionPayload to Engine API (camelCase hex) format."""
        txs = ["0x" + bytes(tx).hex() for tx in payload.transactions]
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
        if hasattr(payload, 'slot_number'):
            result["slotNumber"] = hex(int(payload.slot_number))
        if hasattr(payload, 'block_access_list'):
            result["blockAccessList"] = "0x" + bytes(payload.block_access_list).hex()
        return result

    # --- SSZ helpers -------------------------------------------------------

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

    def _payload_dict_to_ssz(self, el_fork: str, p: dict) -> Any:
        if el_fork == "paris":
            return self._payload_dict_to_v1_ssz(p)
        if el_fork == "shanghai":
            return self._payload_dict_to_v2_ssz(p)
        if el_fork in ("cancun", "prague", "osaka"):
            return self._payload_dict_to_v3_ssz(p)
        return self._payload_dict_to_v4_ssz(p)

    @classmethod
    def _payload_common_fields(cls, p: dict) -> dict:
        return dict(
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
        )

    @classmethod
    def _payload_dict_to_v1_ssz(cls, p: dict) -> "ssz.ExecutionPayloadV1":
        return ssz.ExecutionPayloadV1(**cls._payload_common_fields(p))

    @classmethod
    def _payload_dict_to_v2_ssz(cls, p: dict) -> "ssz.ExecutionPayloadV2":
        return ssz.ExecutionPayloadV2(
            **cls._payload_common_fields(p),
            withdrawals=[cls._withdrawal_dict_to_ssz(w) for w in p.get("withdrawals", [])],
        )

    @classmethod
    def _payload_dict_to_v3_ssz(cls, p: dict) -> "ssz.ExecutionPayloadV3":
        return ssz.ExecutionPayloadV3(
            **cls._payload_common_fields(p),
            withdrawals=[cls._withdrawal_dict_to_ssz(w) for w in p.get("withdrawals", [])],
            blob_gas_used=cls._hex_int(p.get("blobGasUsed", "0x0")),
            excess_blob_gas=cls._hex_int(p.get("excessBlobGas", "0x0")),
        )

    @classmethod
    def _payload_dict_to_v4_ssz(cls, p: dict) -> "ssz.ExecutionPayloadV4":
        return ssz.ExecutionPayloadV4(
            **cls._payload_common_fields(p),
            withdrawals=[cls._withdrawal_dict_to_ssz(w) for w in p.get("withdrawals", [])],
            blob_gas_used=cls._hex_int(p.get("blobGasUsed", "0x0")),
            excess_blob_gas=cls._hex_int(p.get("excessBlobGas", "0x0")),
            block_access_list=ssz.ByteList[ssz.MAX_BAL_BYTES](cls._hex_bytes(p.get("blockAccessList", "0x"))),
            slot_number=cls._hex_int(p.get("slotNumber", "0x0")),
        )

    @staticmethod
    def _withdrawals_ssz_to_list(withdrawals) -> list:
        return [
            {
                "index": hex(int(w.index)),
                "validatorIndex": hex(int(w.validator_index)),
                "address": "0x" + bytes(w.address).hex(),
                "amount": hex(int(w.amount)),
            }
            for w in withdrawals
        ]

    @classmethod
    def _payload_common_ssz_to_dict(cls, ep) -> dict:
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
        }

    @classmethod
    def _v1_payload_ssz_to_dict(cls, ep) -> dict:
        return cls._payload_common_ssz_to_dict(ep)

    @classmethod
    def _v2_payload_ssz_to_dict(cls, ep) -> dict:
        d = cls._payload_common_ssz_to_dict(ep)
        d["withdrawals"] = cls._withdrawals_ssz_to_list(ep.withdrawals)
        return d

    @classmethod
    def _v3_payload_ssz_to_dict(cls, ep) -> dict:
        d = cls._payload_common_ssz_to_dict(ep)
        d["withdrawals"] = cls._withdrawals_ssz_to_list(ep.withdrawals)
        d["blobGasUsed"] = hex(int(ep.blob_gas_used))
        d["excessBlobGas"] = hex(int(ep.excess_blob_gas))
        return d

    @classmethod
    def _v4_payload_ssz_to_dict(cls, ep) -> dict:
        d = cls._v3_payload_ssz_to_dict(ep)
        d["blockAccessList"] = "0x" + bytes(ep.block_access_list).hex()
        d["slotNumber"] = hex(int(ep.slot_number))
        return d

    @staticmethod
    def _blobs_bundle_to_dict(b) -> dict:
        return {
            "commitments": ["0x" + bytes(c).hex() for c in b.commitments],
            "proofs": ["0x" + bytes(p).hex() for p in b.proofs],
            "blobs": ["0x" + bytes(blob).hex() for blob in b.blobs],
        }

    def _built_payload_ssz_to_dict(self, el_fork: str, d) -> dict:
        if el_fork == "paris":
            return {"executionPayload": self._v1_payload_ssz_to_dict(d.payload), "blockValue": hex(int(d.block_value))}
        if el_fork == "shanghai":
            return {"executionPayload": self._v2_payload_ssz_to_dict(d.payload), "blockValue": hex(int(d.block_value))}
        if el_fork == "cancun":
            return {
                "executionPayload": self._v3_payload_ssz_to_dict(d.payload),
                "blockValue": hex(int(d.block_value)),
                "blobsBundle": self._blobs_bundle_to_dict(d.blobs_bundle),
                "shouldOverrideBuilder": bool(d.should_override_builder),
            }
        payload_dict = (
            self._v4_payload_ssz_to_dict(d.payload) if el_fork == "amsterdam"
            else self._v3_payload_ssz_to_dict(d.payload)
        )
        return {
            "executionPayload": payload_dict,
            "blockValue": hex(int(d.block_value)),
            "blobsBundle": self._blobs_bundle_to_dict(d.blobs_bundle),
            "shouldOverrideBuilder": bool(d.should_override_builder),
            "executionRequests": ["0x" + bytes(r).hex() for r in d.execution_requests],
        }

    def _body_entry_to_dict(self, entry) -> dict:
        body = entry.body
        d: dict = {
            "available": bool(entry.available),
            "transactions": ["0x" + bytes(t).hex() for t in body.transactions],
        }
        if hasattr(body, "withdrawals"):
            d["withdrawals"] = self._withdrawals_ssz_to_list(body.withdrawals)
        if hasattr(body, "block_access_list"):
            d["blockAccessList"] = "0x" + bytes(body.block_access_list).hex()
        return d

    @staticmethod
    def _payload_status_ssz_to_dict(ps) -> dict:
        status_str = ssz.INT_TO_PAYLOAD_STATUS.get(int(ps.status))
        if status_str is None:
            raise EngineAPIError(-1, f"Unknown PayloadStatus enum: {int(ps.status)}")
        result: dict = {"status": status_str}
        if len(ps.latest_valid_hash) == 1:
            result["latestValidHash"] = "0x" + bytes(ps.latest_valid_hash[0]).hex()
        else:
            result["latestValidHash"] = None
        if len(ps.validation_error) == 1:
            result["validationError"] = bytes(ps.validation_error[0]).decode("utf-8", errors="replace")
        else:
            result["validationError"] = None
        return result

    @classmethod
    def _fcu_response_ssz_to_dict(cls, r) -> dict:
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

    def _attrs_dict_to_ssz(self, el_fork: str, a: dict) -> Any:
        if el_fork == "paris":
            return self._attrs_v1_dict_to_ssz(a)
        if el_fork == "shanghai":
            return self._attrs_v2_dict_to_ssz(a)
        if el_fork in ("cancun", "prague", "osaka"):
            return self._attrs_v3_dict_to_ssz(a)
        return self._attrs_v4_dict_to_ssz(a)

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
            slot_number=cls._hex_int(a.get("slotNumber", "0x0")),
            target_gas_limit=cls._hex_int(a.get("targetGasLimit", "0x0")),
        )
