"""Eth1 data processing.

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/phase0/beacon-chain.md
"""

from typing import TYPE_CHECKING

from ...constants import EPOCHS_PER_ETH1_VOTING_PERIOD, SLOTS_PER_EPOCH

if TYPE_CHECKING:
    from ...types import BeaconState, BeaconBlockBody


def process_eth1_data(state: "BeaconState", body: "BeaconBlockBody") -> None:
    """Process the Eth1 data vote from the block.

    Adds the vote to the list and updates eth1_data if a majority is reached.

    Args:
        state: Beacon state (modified in place)
        body: Block body containing the eth1_data vote
    """
    # Add the vote
    state.eth1_data_votes.append(body.eth1_data)

    # Check if we have a majority for this vote
    vote_count = sum(
        1 for v in state.eth1_data_votes if v == body.eth1_data
    )

    # Majority threshold: more than half of votes in the period
    votes_in_period = EPOCHS_PER_ETH1_VOTING_PERIOD() * SLOTS_PER_EPOCH()
    if vote_count * 2 > votes_in_period:
        state.eth1_data = body.eth1_data
