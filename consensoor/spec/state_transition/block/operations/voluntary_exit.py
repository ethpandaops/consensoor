"""Voluntary exit processing.

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/phase0/beacon-chain.md
Reference (Electra): https://github.com/ethereum/consensus-specs/blob/master/specs/electra/beacon-chain.md
"""

from typing import TYPE_CHECKING

from ....constants import (
    DOMAIN_VOLUNTARY_EXIT,
    SHARD_COMMITTEE_PERIOD,
    FAR_FUTURE_EPOCH,
)
from ...helpers.predicates import is_active_validator
from ...helpers.accessors import get_current_epoch, get_pending_balance_to_withdraw
from ...helpers.mutators import initiate_validator_exit
from ...helpers.domain import get_domain, compute_domain, compute_signing_root
from ....network_config import get_config
from .....crypto import bls_verify

if TYPE_CHECKING:
    from ....types import BeaconState
    from ....types.phase0 import SignedVoluntaryExit


def process_voluntary_exit(
    state: "BeaconState", signed_voluntary_exit: "SignedVoluntaryExit"
) -> None:
    """Process a voluntary exit.

    Validates the exit request and initiates the validator's exit.

    Args:
        state: Beacon state (modified in place)
        signed_voluntary_exit: Signed voluntary exit to process

    Raises:
        AssertionError: If validation fails
    """
    voluntary_exit = signed_voluntary_exit.message
    validator_index = int(voluntary_exit.validator_index)
    validator = state.validators[validator_index]
    current_epoch = get_current_epoch(state)

    # Verify the validator is active
    assert is_active_validator(validator, current_epoch), (
        "Validator is not active"
    )

    # Verify exit has not been initiated
    assert int(validator.exit_epoch) == FAR_FUTURE_EPOCH, (
        "Validator has already initiated exit"
    )

    # Verify exit epoch has passed
    assert current_epoch >= int(voluntary_exit.epoch), (
        "Exit epoch has not been reached"
    )

    # Verify the validator has been active long enough
    assert current_epoch >= int(validator.activation_epoch) + SHARD_COMMITTEE_PERIOD(), (
        "Validator has not been active long enough"
    )

    # Fork detection for EIP-7044 (Deneb) and EIP-7251 (Electra)
    config = get_config()
    current_fork_version = bytes(state.fork.current_version)
    is_deneb_or_later = current_fork_version >= config.deneb_fork_version
    is_electra_or_later = current_fork_version >= config.electra_fork_version

    # EIP-7251: Only exit validator if it has no pending withdrawals in the queue
    if is_electra_or_later:
        assert get_pending_balance_to_withdraw(state, validator_index) == 0, (
            "Validator has pending withdrawals"
        )

    # Verify signature
    # EIP-7044 (Deneb): voluntary exits use CAPELLA_FORK_VERSION for the domain
    # This allows voluntary exits to be signed once and remain valid across fork upgrades
    if is_deneb_or_later:
        domain = compute_domain(
            DOMAIN_VOLUNTARY_EXIT,
            config.capella_fork_version,
            bytes(state.genesis_validators_root),
        )
    else:
        domain = get_domain(state, DOMAIN_VOLUNTARY_EXIT, int(voluntary_exit.epoch))
    signing_root = compute_signing_root(voluntary_exit, domain)
    assert bls_verify(
        [bytes(validator.pubkey)],
        signing_root,
        bytes(signed_voluntary_exit.signature),
    ), "Invalid voluntary exit signature"

    # Initiate exit
    initiate_validator_exit(state, validator_index)
