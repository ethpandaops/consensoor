"""Aggregation committee window processing (PQ)."""

from typing import TYPE_CHECKING

from ...constants import MIN_SEED_LOOKAHEAD, SLOTS_PER_EPOCH
from ..helpers.accessors import get_current_epoch
from ..helpers.misc import compute_start_slot_at_epoch
from ..helpers.pq import compute_aggregation_committees
from ...types.pq import aggregation_committees

if TYPE_CHECKING:
    from ...types.pq import BeaconState


def process_aggregation_committee_window(state: "BeaconState") -> None:
    spe = SLOTS_PER_EPOCH()
    window_length = len(state.aggregation_committee_window)
    for i in range(window_length - spe):
        state.aggregation_committee_window[i] = state.aggregation_committee_window[i + spe]

    next_epoch = get_current_epoch(state) + MIN_SEED_LOOKAHEAD + 1
    start_slot = compute_start_slot_at_epoch(next_epoch)
    for j, slot in enumerate(range(start_slot, start_slot + spe)):
        state.aggregation_committee_window[window_length - spe + j] = aggregation_committees(
            compute_aggregation_committees(state, slot)
        )
