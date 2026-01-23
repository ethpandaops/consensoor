"""Epoch reset functions for state transition.

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/phase0/beacon-chain.md
"""

from typing import TYPE_CHECKING

from ...constants import (
    EPOCHS_PER_HISTORICAL_VECTOR,
    EPOCHS_PER_SLASHINGS_VECTOR,
    EPOCHS_PER_ETH1_VOTING_PERIOD,
    SLOTS_PER_HISTORICAL_ROOT,
)
from ..helpers.accessors import get_current_epoch, get_block_root, get_randao_mix
from ....crypto import hash_tree_root

if TYPE_CHECKING:
    from ...types import BeaconState


def process_participation_record_updates(state: "BeaconState") -> None:
    """Rotate current/previous epoch attestations (Phase0).

    In Phase0, attestations are tracked via PendingAttestation objects.
    This function rotates the attestation lists at the end of each epoch.

    In Altair+, this is replaced by process_participation_flag_updates.

    Args:
        state: Beacon state (modified in place)
    """
    # Check if this is a Phase0 state (has attestation lists, not participation flags)
    if not hasattr(state, "previous_epoch_attestations"):
        return

    # Rotate attestation lists
    state.previous_epoch_attestations = list(state.current_epoch_attestations)
    state.current_epoch_attestations = []


def process_eth1_data_reset(state: "BeaconState") -> None:
    """Reset Eth1 data votes at the end of the voting period.

    Args:
        state: Beacon state (modified in place)
    """
    next_epoch = get_current_epoch(state) + 1

    # Reset Eth1 data votes at the end of the voting period
    if next_epoch % EPOCHS_PER_ETH1_VOTING_PERIOD() == 0:
        state.eth1_data_votes = []


def process_slashings_reset(state: "BeaconState") -> None:
    """Reset the slashings accumulator for the next epoch.

    Args:
        state: Beacon state (modified in place)
    """
    next_epoch = get_current_epoch(state) + 1

    # Reset slashings for the upcoming epoch slot in the circular buffer
    state.slashings[next_epoch % EPOCHS_PER_SLASHINGS_VECTOR()] = 0


def process_randao_mixes_reset(state: "BeaconState") -> None:
    """Copy the current RANDAO mix to the next epoch slot.

    Args:
        state: Beacon state (modified in place)
    """
    current_epoch = get_current_epoch(state)
    next_epoch = current_epoch + 1

    # Copy current mix to next epoch's slot
    state.randao_mixes[next_epoch % EPOCHS_PER_HISTORICAL_VECTOR()] = get_randao_mix(
        state, current_epoch
    )


def process_historical_summaries_update(state: "BeaconState") -> None:
    """Update historical summaries/roots at the end of each historical period.

    Phase0 uses historical_roots with HistoricalBatch.
    Capella+ uses historical_summaries with HistoricalSummary.

    Args:
        state: Beacon state (modified in place)
    """
    from ...constants import SLOTS_PER_EPOCH

    next_epoch = get_current_epoch(state) + 1

    # Calculate number of epochs per historical root
    slots_per_epoch = SLOTS_PER_EPOCH()
    epochs_per_historical_root = SLOTS_PER_HISTORICAL_ROOT() // slots_per_epoch

    # Check if we're at a historical roots boundary
    if next_epoch % epochs_per_historical_root != 0:
        return

    # Check if this is Phase0/Altair/Bellatrix (has historical_roots) or Capella+ (has historical_summaries)
    if hasattr(state, "historical_summaries"):
        # Capella+ path: use HistoricalSummary
        from ...types.capella import HistoricalSummary
        historical_summary = HistoricalSummary(
            block_summary_root=hash_tree_root(state.block_roots),
            state_summary_root=hash_tree_root(state.state_roots),
        )
        state.historical_summaries.append(historical_summary)
    else:
        # Phase0/Altair/Bellatrix path: use HistoricalBatch
        from ...types.phase0 import HistoricalBatch
        historical_batch = HistoricalBatch(
            block_roots=list(state.block_roots),
            state_roots=list(state.state_roots),
        )
        state.historical_roots.append(hash_tree_root(historical_batch))
