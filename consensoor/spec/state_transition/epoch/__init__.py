"""Epoch processing functions for state transition."""

from .justification import process_justification_and_finalization
from .inactivity import process_inactivity_updates
from .rewards import process_rewards_and_penalties
from .registry import process_registry_updates
from .slashings import process_slashings
from .effective_balance import process_effective_balance_updates
from .participation import process_participation_flag_updates
from .sync_committee import process_sync_committee_updates
from .resets import (
    process_eth1_data_reset,
    process_slashings_reset,
    process_randao_mixes_reset,
    process_historical_summaries_update,
    process_participation_record_updates,
)
from .pending_deposits import process_pending_deposits
from .pending_consolidations import process_pending_consolidations
from .proposer_lookahead import process_proposer_lookahead
from .builder_pending_payments import process_builder_pending_payments

__all__ = [
    "process_justification_and_finalization",
    "process_inactivity_updates",
    "process_rewards_and_penalties",
    "process_registry_updates",
    "process_slashings",
    "process_effective_balance_updates",
    "process_participation_flag_updates",
    "process_participation_record_updates",
    "process_sync_committee_updates",
    "process_eth1_data_reset",
    "process_slashings_reset",
    "process_randao_mixes_reset",
    "process_historical_summaries_update",
    "process_pending_deposits",
    "process_pending_consolidations",
    "process_proposer_lookahead",
    "process_builder_pending_payments",
]
