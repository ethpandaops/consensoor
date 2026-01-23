"""Domain and signing helper functions for state transition.

Implements domain computation and signing root generation.
Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/phase0/beacon-chain.md
"""

from typing import TYPE_CHECKING, Optional, Any

from ...constants import DOMAIN_BEACON_PROPOSER
from .misc import compute_fork_data_root, compute_epoch_at_slot
from .accessors import get_current_epoch
from ....crypto import hash_tree_root, bls_verify

if TYPE_CHECKING:
    from ...types import BeaconState, SignedBeaconBlock


def get_domain(
    state: "BeaconState",
    domain_type: bytes,
    epoch: Optional[int] = None,
) -> bytes:
    """Return the domain for the given domain_type and epoch.

    Args:
        state: Beacon state
        domain_type: 4-byte domain type
        epoch: Target epoch (defaults to current epoch)

    Returns:
        32-byte domain
    """
    if epoch is None:
        epoch = get_current_epoch(state)

    # Get fork version for the epoch
    fork = state.fork
    if epoch < int(fork.epoch):
        fork_version = bytes(fork.previous_version)
    else:
        fork_version = bytes(fork.current_version)

    return compute_domain(
        domain_type, fork_version, bytes(state.genesis_validators_root)
    )


def compute_domain(
    domain_type: bytes,
    fork_version: Optional[bytes] = None,
    genesis_validators_root: Optional[bytes] = None,
) -> bytes:
    """Return the domain for signing.

    Args:
        domain_type: 4-byte domain type
        fork_version: 4-byte fork version (defaults to zeros)
        genesis_validators_root: 32-byte genesis validators root (defaults to zeros)

    Returns:
        32-byte domain
    """
    if fork_version is None:
        fork_version = b"\x00\x00\x00\x00"
    if genesis_validators_root is None:
        genesis_validators_root = b"\x00" * 32

    fork_data_root = compute_fork_data_root(fork_version, genesis_validators_root)
    return domain_type + fork_data_root[:28]


def compute_signing_root(ssz_object: Any, domain: bytes) -> bytes:
    """Return the signing root for an object and domain.

    Args:
        ssz_object: SSZ object to sign
        domain: 32-byte domain

    Returns:
        32-byte signing root
    """
    from ...types import SigningData, Root

    signing_data = SigningData(
        object_root=Root(hash_tree_root(ssz_object)),
        domain=domain,
    )
    return hash_tree_root(signing_data)


def verify_block_signature(
    state: "BeaconState", signed_block: "SignedBeaconBlock"
) -> bool:
    """Verify the signature of a signed beacon block.

    Args:
        state: Beacon state
        signed_block: Signed beacon block

    Returns:
        True if signature is valid
    """
    proposer_index = int(signed_block.message.proposer_index)
    proposer = state.validators[proposer_index]
    domain = get_domain(
        state,
        DOMAIN_BEACON_PROPOSER,
        compute_epoch_at_slot(int(signed_block.message.slot)),
    )
    signing_root = compute_signing_root(signed_block.message, domain)
    return bls_verify(
        [bytes(proposer.pubkey)], signing_root, bytes(signed_block.signature)
    )
