"""Message encoding for Ethereum consensus P2P protocol.

Ethereum consensus uses SSZ encoding with snappy compression for all gossip messages.
Topic strings follow the format: /eth2/{fork_digest}/{topic_name}/{encoding}
"""


from ..crypto import sha256


def compute_fork_digest(
    current_version: bytes,
    genesis_validators_root: bytes,
    blob_params: tuple[int, int] | None = None,
) -> bytes:
    """Compute the fork digest for topic encoding.

    fork_digest = ForkDigest(compute_fork_data_root(current_version, genesis_validators_root)[:4])
    """
    from ..spec.types import ForkData, Root
    from ..crypto import hash_tree_root

    fork_data = ForkData(
        current_version=current_version,
        genesis_validators_root=Root(genesis_validators_root),
    )
    fork_data_root = hash_tree_root(fork_data)

    if blob_params:
        epoch, max_blobs_per_block = blob_params
        # BPO fork digest: xor base digest with hash(epoch || max_blobs) per DAS Guardian
        from ..crypto import sha256

        blob_param_bytes = (
            epoch.to_bytes(8, "little") +
            max_blobs_per_block.to_bytes(8, "little")
        )
        blob_param_hash = sha256(blob_param_bytes)
        return bytes(
            fork_data_root[i] ^ blob_param_hash[i]
            for i in range(4)
        )

    return fork_data_root[:4]


def get_topic_name(base_topic: str, fork_digest: bytes, encoding: str = "ssz_snappy") -> str:
    """Get the full topic name for a gossip topic.

    Format: /eth2/{fork_digest}/{topic_name}/{encoding}
    """
    return f"/eth2/{fork_digest.hex()}/{base_topic}/{encoding}"


BEACON_BLOCK_TOPIC = "beacon_block"
BEACON_AGGREGATE_AND_PROOF_TOPIC = "beacon_aggregate_and_proof"
VOLUNTARY_EXIT_TOPIC = "voluntary_exit"
PROPOSER_SLASHING_TOPIC = "proposer_slashing"
ATTESTER_SLASHING_TOPIC = "attester_slashing"
BLS_TO_EXECUTION_CHANGE_TOPIC = "bls_to_execution_change"
SYNC_COMMITTEE_CONTRIBUTION_AND_PROOF_TOPIC = "sync_committee_contribution_and_proof"
SYNC_COMMITTEE_SUBNET_TOPIC_PREFIX = "sync_committee_"  # sync_committee_{subnet_id}
BLOB_SIDECAR_TOPIC_PREFIX = "blob_sidecar_"  # blob_sidecar_{subnet_id}
BEACON_ATTESTATION_TOPIC_PREFIX = "beacon_attestation_"  # beacon_attestation_{subnet_id}
EXECUTION_PAYLOAD_TOPIC = "execution_payload"  # GLOAS/ePBS execution payload envelope
EXECUTION_PAYLOAD_BID_TOPIC = "execution_payload_bid"  # GLOAS/ePBS builder bid
PAYLOAD_ATTESTATION_MESSAGE_TOPIC = "payload_attestation_message"  # GLOAS/ePBS PTC vote
PROPOSER_PREFERENCES_TOPIC = "proposer_preferences"  # GLOAS/ePBS proposer fee_recipient/gas_limit prefs


def get_blob_sidecar_topic(subnet_id: int, fork_digest: bytes, encoding: str = "ssz_snappy") -> str:
    """Get the full topic name for a blob sidecar subnet.

    Format: /eth2/{fork_digest}/blob_sidecar_{subnet_id}/{encoding}
    """
    return f"/eth2/{fork_digest.hex()}/{BLOB_SIDECAR_TOPIC_PREFIX}{subnet_id}/{encoding}"


def get_sync_committee_subnet_topic(subnet_id: int, fork_digest: bytes, encoding: str = "ssz_snappy") -> str:
    """Get the full topic name for a sync committee subnet.

    Format: /eth2/{fork_digest}/sync_committee_{subnet_id}/{encoding}

    Per Altair `specs/altair/validator.md`, each sync committee member
    publishes their SyncCommitteeMessage on the subnet whose id matches
    their `subcommittee_index = position // (SYNC_COMMITTEE_SIZE / SYNC_COMMITTEE_SUBNET_COUNT)`.
    """
    return f"/eth2/{fork_digest.hex()}/{SYNC_COMMITTEE_SUBNET_TOPIC_PREFIX}{subnet_id}/{encoding}"


def get_attestation_subnet_topic(subnet_id: int, fork_digest: bytes, encoding: str = "ssz_snappy") -> str:
    """Get the full topic name for an unaggregated attestation subnet.

    Format: /eth2/{fork_digest}/beacon_attestation_{subnet_id}/{encoding}
    """
    return f"/eth2/{fork_digest.hex()}/{BEACON_ATTESTATION_TOPIC_PREFIX}{subnet_id}/{encoding}"


def encode_message(data: bytes) -> bytes:
    """Pass-through: the Rust p2p stack does snappy compression on publish."""
    return data


def decode_message(data: bytes) -> bytes:
    """Pass-through: the Rust p2p stack decompresses on receive."""
    return data


def compute_message_id(message_data: bytes) -> bytes:
    """Compute the message ID for gossipsub.

    For Ethereum, message_id = sha256(message)[:20]
    """
    return sha256(message_data)[:20]
