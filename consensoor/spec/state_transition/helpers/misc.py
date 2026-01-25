"""Miscellaneous helper functions for state transition.

Implements epoch/slot computation and fork digest functions.
Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/phase0/beacon-chain.md
"""

from ...constants import (
    SLOTS_PER_EPOCH,
    MAX_SEED_LOOKAHEAD,
    MIN_SEED_LOOKAHEAD,
    BUILDER_INDEX_FLAG,
)
from ...types import ForkData, Root
from ....crypto import hash_tree_root


def compute_epoch_at_slot(slot: int) -> int:
    """Return the epoch number at the given slot.

    Args:
        slot: Slot number

    Returns:
        Epoch containing the slot
    """
    return slot // SLOTS_PER_EPOCH()


def compute_start_slot_at_epoch(epoch: int) -> int:
    """Return the start slot of the given epoch.

    Args:
        epoch: Epoch number

    Returns:
        First slot of the epoch
    """
    return epoch * SLOTS_PER_EPOCH()


def compute_activation_exit_epoch(epoch: int) -> int:
    """Return the epoch during which validator activations and exits initiated
    in the given epoch take effect.

    Args:
        epoch: Current epoch

    Returns:
        Activation/exit epoch (epoch + 1 + MAX_SEED_LOOKAHEAD)
    """
    return epoch + 1 + MAX_SEED_LOOKAHEAD


def compute_fork_data_root(
    current_version: bytes,
    genesis_validators_root: bytes,
) -> bytes:
    """Return the 32-byte fork data root for the current_version and
    genesis_validators_root.

    This is used as the domain separator in signatures.

    Args:
        current_version: 4-byte fork version
        genesis_validators_root: 32-byte genesis validators root

    Returns:
        32-byte fork data root
    """
    fork_data = ForkData(
        current_version=current_version,
        genesis_validators_root=Root(genesis_validators_root),
    )
    return hash_tree_root(fork_data)


def compute_fork_digest(
    current_version: bytes,
    genesis_validators_root: bytes,
) -> bytes:
    """Return the 4-byte fork digest for the current_version and
    genesis_validators_root.

    This is used in network message topics.

    Args:
        current_version: 4-byte fork version
        genesis_validators_root: 32-byte genesis validators root

    Returns:
        4-byte fork digest (first 4 bytes of fork data root)
    """
    return compute_fork_data_root(current_version, genesis_validators_root)[:4]


def compute_time_at_slot(genesis_time: int, slot: int, slot_duration_ms: int) -> int:
    """Return the Unix timestamp at the start of the given slot.

    Args:
        genesis_time: Genesis Unix timestamp
        slot: Slot number
        slot_duration_ms: Slot duration in milliseconds from network config

    Returns:
        Unix timestamp at start of slot
    """
    return genesis_time + slot * (slot_duration_ms // 1000)


def convert_builder_index_to_validator_index(builder_index: int) -> int:
    """Convert a builder index to a validator index (with builder flag)."""
    return int(builder_index) | BUILDER_INDEX_FLAG


def convert_validator_index_to_builder_index(validator_index: int) -> int:
    """Convert a validator index to a builder index (strip builder flag)."""
    return int(validator_index) & ~BUILDER_INDEX_FLAG
