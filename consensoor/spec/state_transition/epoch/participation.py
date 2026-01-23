"""Participation flag updates processing.

Handles both Phase0 (attestation list rotation) and Altair+ (participation flags).
Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/phase0/beacon-chain.md
Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/altair/beacon-chain.md
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...types import BeaconState


def process_participation_flag_updates(state: "BeaconState") -> None:
    """Rotate participation tracking for the new epoch.

    For Altair+: Rotates participation flags.
    For Phase0: Rotates attestation lists.

    Args:
        state: Beacon state (modified in place)
    """
    if hasattr(state, "previous_epoch_participation"):
        # Altair+ path: rotate participation flags
        state.previous_epoch_participation = state.current_epoch_participation.copy()
        validator_count = len(state.validators)
        state.current_epoch_participation = [0] * validator_count
    else:
        # Phase0 path: rotate attestation lists
        process_participation_record_updates_phase0(state)


def process_participation_record_updates_phase0(state: "BeaconState") -> None:
    """Rotate attestation lists for the new epoch (Phase0).

    Current epoch attestations become previous epoch attestations.
    Current epoch attestations are cleared.

    Args:
        state: Beacon state (modified in place)
    """
    # Move current epoch attestations to previous
    state.previous_epoch_attestations = list(state.current_epoch_attestations)

    # Clear current epoch attestations
    state.current_epoch_attestations = []
