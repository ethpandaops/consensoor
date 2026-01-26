"""Prometheus metrics for consensoor."""

import logging
import threading
from typing import Optional

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    start_http_server,
    REGISTRY,
    CollectorRegistry,
)

logger = logging.getLogger(__name__)

DEFAULT_METRICS_PORT = 8008

# Node info
node_info = Info(
    "consensoor_node",
    "Node information",
)

# Slot/epoch metrics
head_slot = Gauge(
    "consensoor_head_slot",
    "Current head slot",
)

head_epoch = Gauge(
    "consensoor_head_epoch",
    "Current head epoch",
)

finalized_epoch = Gauge(
    "consensoor_finalized_epoch",
    "Current finalized epoch",
)

justified_epoch = Gauge(
    "consensoor_justified_epoch",
    "Current justified epoch",
)

# Sync status
is_syncing = Gauge(
    "consensoor_syncing",
    "Whether the node is syncing (1) or synced (0)",
)

sync_distance = Gauge(
    "consensoor_sync_distance",
    "Number of slots behind the head",
)

# P2P metrics
peers_connected = Gauge(
    "consensoor_peers_connected",
    "Number of connected peers",
)

gossip_messages_received = Counter(
    "consensoor_gossip_messages_received_total",
    "Total gossip messages received",
    ["topic"],
)

gossip_messages_sent = Counter(
    "consensoor_gossip_messages_sent_total",
    "Total gossip messages sent",
    ["topic"],
)

# Validator metrics
attestations_produced = Counter(
    "consensoor_attestations_produced_total",
    "Total attestations produced",
)

attestations_included = Counter(
    "consensoor_attestations_included_total",
    "Total attestations included in blocks",
)

blocks_proposed = Counter(
    "consensoor_blocks_proposed_total",
    "Total blocks proposed",
)

blocks_proposed_success = Counter(
    "consensoor_blocks_proposed_success_total",
    "Total blocks successfully proposed",
)

blocks_received = Counter(
    "consensoor_blocks_received_total",
    "Total blocks received from network",
)

# Block processing metrics
block_processing_time = Histogram(
    "consensoor_block_processing_seconds",
    "Time to process a block",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

state_transition_time = Histogram(
    "consensoor_state_transition_seconds",
    "Time for state transition",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# Engine API metrics
engine_api_requests = Counter(
    "consensoor_engine_api_requests_total",
    "Total Engine API requests",
    ["method"],
)

engine_api_errors = Counter(
    "consensoor_engine_api_errors_total",
    "Total Engine API errors",
    ["method", "error_type"],
)

engine_api_latency = Histogram(
    "consensoor_engine_api_latency_seconds",
    "Engine API request latency",
    ["method"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# Store metrics
store_states_count = Gauge(
    "consensoor_store_states_count",
    "Number of states in store",
)

store_blocks_count = Gauge(
    "consensoor_store_blocks_count",
    "Number of blocks in store",
)

# Beacon API metrics
beacon_api_requests = Counter(
    "consensoor_beacon_api_requests_total",
    "Total Beacon API requests",
    ["endpoint", "method"],
)


_server_started = False
_server_lock = threading.Lock()


def start_metrics_server(port: int = DEFAULT_METRICS_PORT) -> bool:
    """Start the Prometheus metrics HTTP server.

    Args:
        port: Port to listen on (default 8008)

    Returns:
        True if server started successfully, False if already running
    """
    global _server_started

    with _server_lock:
        if _server_started:
            logger.warning(f"Metrics server already running")
            return False

        try:
            start_http_server(port)
            _server_started = True
            logger.info(f"Prometheus metrics server started on port {port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            return False


def set_node_info(version: str, network: str, preset: str) -> None:
    """Set node information metric."""
    node_info.info({
        "version": version,
        "network": network,
        "preset": preset,
    })


def update_head(slot: int, epoch: int) -> None:
    """Update head slot and epoch metrics."""
    head_slot.set(slot)
    head_epoch.set(epoch)


def update_checkpoints(finalized: int, justified: int) -> None:
    """Update checkpoint epoch metrics."""
    finalized_epoch.set(finalized)
    justified_epoch.set(justified)


def update_sync_status(syncing: bool, distance: int = 0) -> None:
    """Update sync status metrics."""
    is_syncing.set(1 if syncing else 0)
    sync_distance.set(distance)


def update_peers(count: int) -> None:
    """Update peer count metric."""
    peers_connected.set(count)


def record_gossip_received(topic: str) -> None:
    """Record a gossip message received."""
    gossip_messages_received.labels(topic=topic).inc()


def record_gossip_sent(topic: str) -> None:
    """Record a gossip message sent."""
    gossip_messages_sent.labels(topic=topic).inc()


def record_attestation_produced() -> None:
    """Record an attestation produced."""
    attestations_produced.inc()


def record_block_proposed(success: bool = True) -> None:
    """Record a block proposal."""
    blocks_proposed.inc()
    if success:
        blocks_proposed_success.inc()


def record_block_received() -> None:
    """Record a block received from the network."""
    blocks_received.inc()


def record_engine_api_call(method: str, latency: float, error: Optional[str] = None) -> None:
    """Record an Engine API call.

    Args:
        method: API method name (e.g., 'forkchoiceUpdated', 'newPayload')
        latency: Request latency in seconds
        error: Error type if the call failed, None if successful
    """
    engine_api_requests.labels(method=method).inc()
    engine_api_latency.labels(method=method).observe(latency)
    if error:
        engine_api_errors.labels(method=method, error_type=error).inc()


def record_beacon_api_request(endpoint: str, method: str = "GET") -> None:
    """Record a Beacon API request."""
    beacon_api_requests.labels(endpoint=endpoint, method=method).inc()


def update_store_stats(states: int, blocks: int) -> None:
    """Update store statistics."""
    store_states_count.set(states)
    store_blocks_count.set(blocks)
