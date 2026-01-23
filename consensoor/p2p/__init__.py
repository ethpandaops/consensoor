"""P2P networking for Ethereum consensus using libp2p.

This module provides gossipsub-based pub/sub for beacon chain messages.
"""

from .host import P2PHost, P2PConfig
from .gossip import BeaconGossip, extract_fork_digest_from_enr
from .encoding import (
    encode_message,
    decode_message,
    compute_fork_digest,
    get_topic_name,
    BEACON_BLOCK_TOPIC,
    BEACON_AGGREGATE_AND_PROOF_TOPIC,
    VOLUNTARY_EXIT_TOPIC,
    PROPOSER_SLASHING_TOPIC,
    ATTESTER_SLASHING_TOPIC,
)

__all__ = [
    "P2PHost",
    "P2PConfig",
    "BeaconGossip",
    "extract_fork_digest_from_enr",
    "encode_message",
    "decode_message",
    "compute_fork_digest",
    "get_topic_name",
    "BEACON_BLOCK_TOPIC",
    "BEACON_AGGREGATE_AND_PROOF_TOPIC",
    "VOLUNTARY_EXIT_TOPIC",
    "PROPOSER_SLASHING_TOPIC",
    "ATTESTER_SLASHING_TOPIC",
]
