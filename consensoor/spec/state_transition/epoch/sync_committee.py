"""Sync committee updates processing (Altair+).

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/altair/beacon-chain.md
"""

from typing import TYPE_CHECKING

from ...constants import EPOCHS_PER_SYNC_COMMITTEE_PERIOD
from ..helpers.accessors import get_current_epoch
from ..helpers.attestation import get_next_sync_committee

if TYPE_CHECKING:
    from ...types import BeaconState


def process_sync_committee_updates(state: "BeaconState") -> None:
    """Update sync committees at sync committee period boundaries (Altair+).

    At the start of each sync committee period, rotate the committees:
    - Current becomes previous (conceptually, though we only store current/next)
    - Next becomes current
    - Compute new next committee

    Args:
        state: Beacon state (modified in place)
    """
    next_epoch = get_current_epoch(state) + 1

    # Check if we're at a sync committee period boundary
    if next_epoch % EPOCHS_PER_SYNC_COMMITTEE_PERIOD() == 0:
        # Rotate committees
        state.current_sync_committee = state.next_sync_committee
        state.next_sync_committee = get_next_sync_committee(state)
