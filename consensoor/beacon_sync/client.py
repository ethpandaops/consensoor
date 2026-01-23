"""Remote Beacon API client for syncing from an upstream beacon node."""

import asyncio
import json
import logging
from typing import Callable, Optional, Any

import aiohttp

from .exceptions import BeaconAPIError, BlockNotFoundError, StateNotFoundError

logger = logging.getLogger(__name__)


class RemoteBeaconClient:
    """Client for syncing from a remote Beacon API (any conformant client)."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None
        self._sse_task: Optional[asyncio.Task] = None
        self._running = False
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 30.0

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def subscribe_to_events(
        self,
        topics: list[str],
        callback: Callable[[str, dict], Any],
    ) -> None:
        """Subscribe to SSE events from the Beacon API.

        Args:
            topics: Event topics to subscribe to (e.g., ["block", "finalized_checkpoint"])
            callback: Async callback function that receives (event_type, event_data)
        """
        self._running = True
        self._sse_task = asyncio.create_task(
            self._sse_loop(topics, callback)
        )

    async def _sse_loop(
        self,
        topics: list[str],
        callback: Callable[[str, dict], Any],
    ) -> None:
        """Internal SSE event loop with reconnection logic."""
        topics_param = ",".join(topics)
        url = f"{self.base_url}/eth/v1/events?topics={topics_param}"

        while self._running:
            try:
                session = await self._ensure_session()
                logger.info(f"Connecting to SSE stream: {url}")

                async with session.get(
                    url,
                    headers={"Accept": "text/event-stream"},
                    timeout=aiohttp.ClientTimeout(total=None, sock_read=None),
                ) as response:
                    if response.status != 200:
                        logger.error(f"SSE connection failed: {response.status}")
                        await asyncio.sleep(self._reconnect_delay)
                        self._reconnect_delay = min(
                            self._reconnect_delay * 2,
                            self._max_reconnect_delay,
                        )
                        continue

                    self._reconnect_delay = 1.0
                    logger.info("SSE connection established")

                    event_type = None
                    event_data = ""

                    async for line in response.content:
                        if not self._running:
                            break

                        line = line.decode("utf-8").strip()

                        if not line:
                            if event_type and event_data:
                                try:
                                    data = json.loads(event_data)
                                    await callback(event_type, data)
                                except Exception as e:
                                    logger.error(f"Error processing SSE event: {e}")
                            event_type = None
                            event_data = ""
                            continue

                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                        elif line.startswith("data:"):
                            event_data = line[5:].strip()

            except asyncio.CancelledError:
                break
            except aiohttp.ClientError as e:
                logger.error(f"SSE connection error: {e}")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2,
                    self._max_reconnect_delay,
                )
            except Exception as e:
                logger.error(f"Unexpected SSE error: {e}")
                await asyncio.sleep(self._reconnect_delay)

    async def stop_events(self) -> None:
        """Stop the SSE event subscription."""
        self._running = False
        if self._sse_task:
            self._sse_task.cancel()
            try:
                await self._sse_task
            except asyncio.CancelledError:
                pass

    async def get_block(self, block_id: str) -> bytes:
        """Fetch a block as SSZ bytes."""
        session = await self._ensure_session()
        url = f"{self.base_url}/eth/v2/beacon/blocks/{block_id}"

        async with session.get(
            url,
            headers={"Accept": "application/octet-stream"},
        ) as response:
            if response.status == 404:
                raise BlockNotFoundError(f"Block not found: {block_id}")
            if response.status != 200:
                text = await response.text()
                raise BeaconAPIError(response.status, text)
            return await response.read()

    async def get_block_json(self, block_id: str) -> dict:
        """Fetch a block as JSON."""
        session = await self._ensure_session()
        url = f"{self.base_url}/eth/v2/beacon/blocks/{block_id}"

        async with session.get(
            url,
            headers={"Accept": "application/json"},
        ) as response:
            if response.status == 404:
                raise BlockNotFoundError(f"Block not found: {block_id}")
            if response.status != 200:
                text = await response.text()
                raise BeaconAPIError(response.status, text)
            return await response.json()

    async def get_state(self, state_id: str) -> bytes:
        """Fetch beacon state as SSZ bytes."""
        session = await self._ensure_session()
        url = f"{self.base_url}/eth/v2/debug/beacon/states/{state_id}"

        async with session.get(
            url,
            headers={"Accept": "application/octet-stream"},
            timeout=aiohttp.ClientTimeout(total=300),
        ) as response:
            if response.status == 404:
                raise StateNotFoundError(f"State not found: {state_id}")
            if response.status != 200:
                text = await response.text()
                raise BeaconAPIError(response.status, text)
            return await response.read()

    async def get_fork(self, state_id: str) -> dict:
        """Get fork information for a state."""
        session = await self._ensure_session()
        url = f"{self.base_url}/eth/v1/beacon/states/{state_id}/fork"

        async with session.get(url) as response:
            if response.status != 200:
                text = await response.text()
                raise BeaconAPIError(response.status, text)
            data = await response.json()
            return data.get("data", {})

    async def get_finality_checkpoints(self, state_id: str) -> dict:
        """Get finality checkpoints for a state."""
        session = await self._ensure_session()
        url = f"{self.base_url}/eth/v1/beacon/states/{state_id}/finality_checkpoints"

        async with session.get(url) as response:
            if response.status != 200:
                text = await response.text()
                raise BeaconAPIError(response.status, text)
            data = await response.json()
            return data.get("data", {})

    async def get_header(self, block_id: str) -> dict:
        """Get block header."""
        session = await self._ensure_session()
        url = f"{self.base_url}/eth/v1/beacon/headers/{block_id}"

        async with session.get(url) as response:
            if response.status == 404:
                raise BlockNotFoundError(f"Block header not found: {block_id}")
            if response.status != 200:
                text = await response.text()
                raise BeaconAPIError(response.status, text)
            data = await response.json()
            return data.get("data", {})

    async def get_genesis(self) -> dict:
        """Get genesis information."""
        session = await self._ensure_session()
        url = f"{self.base_url}/eth/v1/beacon/genesis"

        async with session.get(url) as response:
            if response.status != 200:
                text = await response.text()
                raise BeaconAPIError(response.status, text)
            data = await response.json()
            return data.get("data", {})

    async def get_spec(self) -> dict:
        """Get the chain spec/config."""
        session = await self._ensure_session()
        url = f"{self.base_url}/eth/v1/config/spec"

        async with session.get(url) as response:
            if response.status != 200:
                text = await response.text()
                raise BeaconAPIError(response.status, text)
            data = await response.json()
            return data.get("data", {})

    async def get_version(self) -> str:
        """Get the beacon node version string."""
        session = await self._ensure_session()
        url = f"{self.base_url}/eth/v1/node/version"

        async with session.get(url) as response:
            if response.status != 200:
                text = await response.text()
                raise BeaconAPIError(response.status, text)
            data = await response.json()
            return data.get("data", {}).get("version", "unknown")

    async def close(self) -> None:
        """Close the client and cleanup resources."""
        await self.stop_events()
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
