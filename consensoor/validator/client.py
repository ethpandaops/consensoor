"""Validator client for producing blocks and attestations."""

import logging
from typing import Optional

from ..spec.constants import SLOTS_PER_EPOCH
from .types import ValidatorKey, ProposerDuty, AttesterDuty
from .shuffling import get_beacon_proposer_index

logger = logging.getLogger(__name__)


class ValidatorClient:
    """Validator client for producing blocks and attestations."""

    def __init__(self, keys: list[ValidatorKey]):
        self.keys = {k.pubkey: k for k in keys}
        self.pubkeys = set(k.pubkey for k in keys)
        self._validator_indices: dict[bytes, int] = {}

    def has_key(self, pubkey: bytes) -> bool:
        """Check if we have a key for this pubkey."""
        return pubkey in self.pubkeys

    def get_key(self, pubkey: bytes) -> Optional[ValidatorKey]:
        """Get the validator key for a pubkey."""
        return self.keys.get(pubkey)

    def update_validator_indices(self, state) -> None:
        """Update validator indices from state."""
        validators = state.validators
        for i, validator in enumerate(validators):
            pubkey = bytes(validator.pubkey)
            if pubkey in self.pubkeys:
                self._validator_indices[pubkey] = i
                key = self.keys[pubkey]
                key.validator_index = i
                logger.info(f"Validator {pubkey.hex()[:16]}... has index {i}")

    def get_validator_index(self, pubkey: bytes) -> Optional[int]:
        """Get validator index for a pubkey."""
        return self._validator_indices.get(pubkey)

    def _get_proposer_index(self, state, slot: int) -> Optional[int]:
        """Get proposer index for a slot.

        Uses proposer_lookahead if available and the state is in the same epoch.
        Falls back to compute_proposer_index (via randao) otherwise.

        Per the Fulu spec, proposer_lookahead is indexed by slot % SLOTS_PER_EPOCH
        and is only valid for the current epoch stored in state.
        """
        slots_per_epoch = SLOTS_PER_EPOCH()
        slot_epoch = slot // slots_per_epoch
        state_epoch = int(state.slot) // slots_per_epoch

        if hasattr(state, "proposer_lookahead"):
            # proposer_lookahead is only valid for the state's current epoch
            # If the requested slot is in a different epoch, fall back to computing
            if slot_epoch != state_epoch:
                logger.debug(
                    f"Slot {slot} (epoch {slot_epoch}) differs from state epoch {state_epoch}, "
                    f"falling back to randao-based computation"
                )
                return get_beacon_proposer_index(state, slot)

            lookahead = state.proposer_lookahead
            slot_in_epoch = slot % slots_per_epoch
            if slot_in_epoch < len(lookahead):
                return int(lookahead[slot_in_epoch])

        return get_beacon_proposer_index(state, slot)

    def get_our_proposer_duties(self, state, epoch: int) -> list[ProposerDuty]:
        """Get proposer duties for our validators in an epoch.

        Uses proposer_lookahead when available.
        """
        duties = []
        slots_per_epoch = SLOTS_PER_EPOCH()
        start_slot = epoch * slots_per_epoch
        end_slot = start_slot + slots_per_epoch

        for slot in range(start_slot, end_slot):
            proposer_index = self._get_proposer_index(state, slot)
            if proposer_index is None:
                continue
            for pubkey, key in self.keys.items():
                if key.validator_index == proposer_index:
                    duties.append(ProposerDuty(
                        validator_index=proposer_index,
                        slot=slot,
                        pubkey=pubkey,
                    ))
                    logger.info(f"We have proposer duty at slot {slot}")

        return duties

    def is_our_proposer_slot(self, state, slot: int) -> Optional[ValidatorKey]:
        """Check if the given slot is our proposer duty and return the key if so."""
        proposer_index = self._get_proposer_index(state, slot)
        if proposer_index is None:
            return None
        for pubkey, key in self.keys.items():
            if key.validator_index == proposer_index:
                return key
        return None

    async def get_attester_duties(self, state, epoch: int) -> list[AttesterDuty]:
        """Get attester duties for an epoch."""
        duties = []
        return duties

    async def produce_attestation(self, state, slot: int, committee_index: int) -> object:
        """Produce an attestation for the given slot and committee."""
        pass

    async def produce_sync_committee_message(self, state, slot: int) -> object:
        """Produce a sync committee message."""
        pass
