"""PTC window processing (Gloas EIP-7732).

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/gloas/beacon-chain.md
"""

from typing import TYPE_CHECKING

from ...constants import SLOTS_PER_EPOCH, MIN_SEED_LOOKAHEAD
from ..helpers.accessors import get_current_epoch
from ..helpers.misc import compute_start_slot_at_epoch
from ..helpers.ptc import compute_ptc

if TYPE_CHECKING:
    from ...types import BeaconState


def process_ptc_window(state: "BeaconState") -> None:
    """Update the cached PTC window.

    Shifts all epochs forward by one and computes the new last epoch.
    """
    spe = SLOTS_PER_EPOCH()
    window_len = len(state.ptc_window)

    new_window = [None] * window_len
    for i in range(window_len - spe):
        new_window[i] = state.ptc_window[i + spe]

    next_epoch = get_current_epoch(state) + MIN_SEED_LOOKAHEAD + 1
    start_slot = compute_start_slot_at_epoch(next_epoch)
    for j, slot in enumerate(range(start_slot, start_slot + spe)):
        new_window[window_len - spe + j] = compute_ptc(state, slot)

    for i in range(window_len):
        state.ptc_window[i] = new_window[i]
