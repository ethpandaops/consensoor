"""Proposer slashing processing.

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/phase0/beacon-chain.md
"""

from typing import TYPE_CHECKING

from ....constants import DOMAIN_BEACON_PROPOSER, SLOTS_PER_EPOCH
from ...helpers.predicates import is_slashable_validator
from ...helpers.accessors import get_current_epoch, get_previous_epoch
from ...helpers.domain import get_domain, compute_signing_root
from ...helpers.mutators import slash_validator
from ...helpers.misc import compute_epoch_at_slot
from .....crypto import bls_verify, hash_tree_root

if TYPE_CHECKING:
    from ....types import BeaconState
    from ....types.phase0 import ProposerSlashing


def process_proposer_slashing(
    state: "BeaconState", proposer_slashing: "ProposerSlashing"
) -> None:
    """Process a proposer slashing.

    Validates the slashing evidence and slashes the proposer.

    Args:
        state: Beacon state (modified in place)
        proposer_slashing: Proposer slashing to process

    Raises:
        AssertionError: If validation fails
    """
    header_1 = proposer_slashing.signed_header_1.message
    header_2 = proposer_slashing.signed_header_2.message

    # Verify headers are for the same slot
    assert int(header_1.slot) == int(header_2.slot), "Slashing headers not for same slot"

    # Verify headers have the same proposer
    assert int(header_1.proposer_index) == int(header_2.proposer_index), (
        "Slashing headers not from same proposer"
    )

    # Verify headers are different
    assert header_1 != header_2, "Slashing headers are identical"

    # Verify proposer is slashable
    proposer_index = int(header_1.proposer_index)
    proposer = state.validators[proposer_index]
    assert is_slashable_validator(proposer, get_current_epoch(state)), (
        "Proposer is not slashable"
    )

    # Verify signatures
    for signed_header in [
        proposer_slashing.signed_header_1,
        proposer_slashing.signed_header_2,
    ]:
        domain = get_domain(
            state,
            DOMAIN_BEACON_PROPOSER,
            compute_epoch_at_slot(int(signed_header.message.slot)),
        )
        signing_root = compute_signing_root(signed_header.message, domain)
        assert bls_verify(
            [bytes(proposer.pubkey)],
            signing_root,
            bytes(signed_header.signature),
        ), "Invalid proposer slashing signature"

    if hasattr(state, "builder_pending_payments"):
        from ....types.gloas import BuilderPendingPayment

        slot = int(header_1.slot)
        proposal_epoch = compute_epoch_at_slot(slot)
        if proposal_epoch == get_current_epoch(state):
            payment_index = SLOTS_PER_EPOCH() + slot % SLOTS_PER_EPOCH()
            state.builder_pending_payments[payment_index] = BuilderPendingPayment()
        elif proposal_epoch == get_previous_epoch(state):
            payment_index = slot % SLOTS_PER_EPOCH()
            state.builder_pending_payments[payment_index] = BuilderPendingPayment()

    # Slash the proposer
    slash_validator(state, proposer_index)
