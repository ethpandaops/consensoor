"""RANDAO processing.

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/phase0/beacon-chain.md
"""

from typing import TYPE_CHECKING

from ...constants import DOMAIN_RANDAO, EPOCHS_PER_HISTORICAL_VECTOR
from ..helpers.accessors import get_current_epoch, get_randao_mix
from ..helpers.domain import get_domain, compute_signing_root
from ..helpers.beacon_committee import get_beacon_proposer_index
from ..helpers.math import xor
from ....crypto import bls_verify, sha256

if TYPE_CHECKING:
    from ...types import BeaconState, BeaconBlockBody


def process_randao(state: "BeaconState", body: "BeaconBlockBody") -> None:
    """Process the RANDAO reveal from the block.

    Verifies the reveal signature and mixes it into the RANDAO mix.

    Args:
        state: Beacon state (modified in place)
        body: Block body containing the RANDAO reveal

    Raises:
        AssertionError: If signature verification fails
    """
    epoch = get_current_epoch(state)
    proposer_index = get_beacon_proposer_index(state)
    proposer = state.validators[proposer_index]

    # Verify RANDAO reveal signature
    # The reveal is the proposer's signature of the current epoch
    domain = get_domain(state, DOMAIN_RANDAO, epoch)

    # Epoch as signing root (just the uint64 encoded)
    from ...types import Root
    from ...types.base import uint64
    from remerkleable.basic import uint64 as ssz_uint64

    signing_root = compute_signing_root(ssz_uint64(epoch), domain)

    assert bls_verify(
        [bytes(proposer.pubkey)],
        signing_root,
        bytes(body.randao_reveal),
    ), "Invalid RANDAO reveal signature"

    # Mix in the RANDAO reveal
    mix = xor(get_randao_mix(state, epoch), sha256(bytes(body.randao_reveal)))
    state.randao_mixes[epoch % EPOCHS_PER_HISTORICAL_VECTOR()] = mix
