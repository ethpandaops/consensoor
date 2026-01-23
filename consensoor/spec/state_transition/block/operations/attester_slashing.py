"""Attester slashing processing.

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/phase0/beacon-chain.md
"""

from typing import TYPE_CHECKING

from ...helpers.predicates import (
    is_slashable_validator,
    is_slashable_attestation_data,
    is_valid_indexed_attestation,
)
from ...helpers.accessors import get_current_epoch
from ...helpers.mutators import slash_validator

if TYPE_CHECKING:
    from ....types import BeaconState
    from ....types.electra import AttesterSlashing


def process_attester_slashing(
    state: "BeaconState", attester_slashing: "AttesterSlashing"
) -> None:
    """Process an attester slashing.

    Validates the slashing evidence and slashes all attesting validators
    that participated in both attestations.

    Args:
        state: Beacon state (modified in place)
        attester_slashing: Attester slashing to process

    Raises:
        AssertionError: If validation fails
    """
    attestation_1 = attester_slashing.attestation_1
    attestation_2 = attester_slashing.attestation_2

    # Verify attestation data is slashable (double vote or surround vote)
    assert is_slashable_attestation_data(attestation_1.data, attestation_2.data), (
        "Attestation data is not slashable"
    )

    # Verify both attestations are valid
    assert is_valid_indexed_attestation(state, attestation_1), (
        "Invalid indexed attestation 1"
    )
    assert is_valid_indexed_attestation(state, attestation_2), (
        "Invalid indexed attestation 2"
    )

    # Find validators that attested to both
    indices_1 = set(attestation_1.attesting_indices)
    indices_2 = set(attestation_2.attesting_indices)
    slashable_indices = sorted(indices_1 & indices_2)

    # Slash all validators that are slashable
    slashed_any = False
    for index in slashable_indices:
        if is_slashable_validator(state.validators[index], get_current_epoch(state)):
            slash_validator(state, index)
            slashed_any = True

    # Must slash at least one validator
    assert slashed_any, "No validators slashed"
