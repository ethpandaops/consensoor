"""Ethereum consensus layer state transition implementation.

This module implements the complete state transition function according to the
Ethereum consensus specs (Phase 0 through Gloas/ePBS).
"""

from .transition import (
    state_transition,
    process_slots,
    process_slot,
    process_epoch,
    process_block,
    upgrade_fork_if_needed,
)
from .fork_upgrade import (
    upgrade_to_capella,
    upgrade_to_deneb,
    upgrade_to_electra,
    upgrade_to_fulu,
    maybe_upgrade_state,
)

# Gloas (ePBS) - execution payload envelope processing (separate from block)
from .block import process_execution_payload_envelope

__all__ = [
    "state_transition",
    "process_slots",
    "process_slot",
    "process_epoch",
    "process_block",
    "upgrade_fork_if_needed",
    # Fork upgrades
    "upgrade_to_capella",
    "upgrade_to_deneb",
    "upgrade_to_electra",
    "upgrade_to_fulu",
    "maybe_upgrade_state",
    # Gloas (ePBS)
    "process_execution_payload_envelope",
]
