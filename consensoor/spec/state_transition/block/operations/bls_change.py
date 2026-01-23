"""BLS to execution change processing (Capella+).

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/capella/beacon-chain.md
"""

from typing import TYPE_CHECKING

from ....constants import (
    DOMAIN_BLS_TO_EXECUTION_CHANGE,
    BLS_WITHDRAWAL_PREFIX,
    ETH1_ADDRESS_WITHDRAWAL_PREFIX,
)
from ....network_config import get_config
from ...helpers.domain import compute_domain, compute_signing_root
from .....crypto import bls_verify, sha256

if TYPE_CHECKING:
    from ....types import BeaconState
    from ....types.capella import SignedBLSToExecutionChange


def process_bls_to_execution_change(
    state: "BeaconState", signed_address_change: "SignedBLSToExecutionChange"
) -> None:
    """Process a BLS to execution address change.

    Updates the validator's withdrawal credentials from BLS (0x00) to
    execution address (0x01).

    Args:
        state: Beacon state (modified in place)
        signed_address_change: Signed address change to process

    Raises:
        AssertionError: If validation fails
    """
    address_change = signed_address_change.message
    validator_index = int(address_change.validator_index)
    validator = state.validators[validator_index]

    # Verify the validator has BLS withdrawal credentials
    withdrawal_credentials = bytes(validator.withdrawal_credentials)
    assert withdrawal_credentials[0] == BLS_WITHDRAWAL_PREFIX, (
        "Validator does not have BLS withdrawal credentials"
    )

    # Verify the from_bls_pubkey matches the withdrawal credentials
    # BLS withdrawal credentials format: BLS_WITHDRAWAL_PREFIX || hash(pubkey)[1:]
    # The last 31 bytes of the hash are used (bytes 1-31)
    pubkey_hash = sha256(bytes(address_change.from_bls_pubkey))
    assert withdrawal_credentials[1:] == pubkey_hash[1:], (
        "BLS pubkey does not match withdrawal credentials"
    )

    # Verify signature
    # Fork-agnostic domain since address changes are valid across forks
    # Use genesis fork version per spec (not current fork version)
    domain = compute_domain(
        DOMAIN_BLS_TO_EXECUTION_CHANGE,
        get_config().genesis_fork_version,
        bytes(state.genesis_validators_root),
    )
    signing_root = compute_signing_root(address_change, domain)
    assert bls_verify(
        [bytes(address_change.from_bls_pubkey)],
        signing_root,
        bytes(signed_address_change.signature),
    ), "Invalid BLS to execution change signature"

    # Update withdrawal credentials to execution address
    new_withdrawal_credentials = (
        bytes([ETH1_ADDRESS_WITHDRAWAL_PREFIX])
        + b"\x00" * 11
        + bytes(address_change.to_execution_address)
    )
    validator.withdrawal_credentials = new_withdrawal_credentials
