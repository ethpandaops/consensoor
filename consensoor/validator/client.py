"""Validator client for producing blocks and attestations."""

import logging
from typing import Optional

from ..spec.constants import SLOTS_PER_EPOCH, DOMAIN_BEACON_ATTESTER
from ..spec.state_transition.helpers.beacon_committee import (
    get_beacon_committee,
    get_committee_count_per_slot,
)
from ..spec.state_transition.helpers.accessors import (
    get_block_root,
    get_block_root_at_slot,
)
from ..spec.state_transition.helpers.misc import compute_epoch_at_slot
from ..spec.state_transition.helpers.domain import get_domain, compute_signing_root
from ..spec.types import AttestationData
from ..spec.types.phase0 import Phase0Attestation
from ..spec.types.electra import Attestation as ElectraAttestation
from ..spec.types.base import Checkpoint, Bitlist, Bitvector
from ..spec.constants import MAX_VALIDATORS_PER_COMMITTEE, MAX_COMMITTEES_PER_SLOT

Phase0AggregationBits = Bitlist[MAX_VALIDATORS_PER_COMMITTEE()]
ElectraAggregationBits = Bitlist[MAX_VALIDATORS_PER_COMMITTEE() * MAX_COMMITTEES_PER_SLOT()]
ElectraCommitteeBits = Bitvector[MAX_COMMITTEES_PER_SLOT()]
from ..crypto import sign as bls_sign, hash_tree_root
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

        Uses proposer_lookahead if available and the slot is within the lookahead range.
        The lookahead contains proposers for (MIN_SEED_LOOKAHEAD + 1) epochs starting
        from the state's current epoch.

        Falls back to compute_proposer_index (via randao) otherwise.
        """
        slots_per_epoch = SLOTS_PER_EPOCH()
        state_epoch = int(state.slot) // slots_per_epoch

        if hasattr(state, "proposer_lookahead"):
            lookahead = state.proposer_lookahead
            current_epoch_start_slot = state_epoch * slots_per_epoch
            slot_offset = slot - current_epoch_start_slot

            if 0 <= slot_offset < len(lookahead):
                return int(lookahead[slot_offset])

            logger.debug(
                f"Slot {slot} is outside proposer_lookahead range "
                f"(state epoch {state_epoch}, offset {slot_offset}, lookahead len {len(lookahead)})"
            )

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

    def get_attester_duties(self, state, epoch: int) -> list[AttesterDuty]:
        """Get attester duties for our validators in an epoch.

        Args:
            state: Beacon state
            epoch: Target epoch

        Returns:
            List of attester duties for our validators
        """
        duties = []
        slots_per_epoch = SLOTS_PER_EPOCH()
        start_slot = epoch * slots_per_epoch
        end_slot = start_slot + slots_per_epoch

        committees_per_slot = get_committee_count_per_slot(state, epoch)

        for slot in range(start_slot, end_slot):
            for committee_index in range(committees_per_slot):
                committee = get_beacon_committee(state, slot, committee_index)

                for validator_position, validator_index in enumerate(committee):
                    for pubkey, key in self.keys.items():
                        if key.validator_index == validator_index:
                            duty = AttesterDuty(
                                validator_index=validator_index,
                                slot=slot,
                                committee_index=committee_index,
                                committee_length=len(committee),
                                committees_at_slot=committees_per_slot,
                                validator_committee_index=validator_position,
                                pubkey=pubkey,
                            )
                            duties.append(duty)
                            logger.debug(
                                f"Attester duty: slot={slot}, committee={committee_index}, "
                                f"position={validator_position}, validator={validator_index}"
                            )

        return duties

    def _is_electra_fork(self, state) -> bool:
        """Check if the state is at or after Electra fork."""
        # Electra states have pending_deposits field
        return hasattr(state, "pending_deposits")

    def produce_attestation(
        self, state, duty: AttesterDuty, head_root: bytes
    ):
        """Produce an attestation for the given duty.

        Args:
            state: Beacon state
            duty: The attester duty
            head_root: Current head block root

        Returns:
            Signed attestation or None if production fails
        """
        try:
            slot = duty.slot
            committee_index = duty.committee_index
            epoch = compute_epoch_at_slot(slot)

            # Get target root - the block root at the start of the target epoch
            # If we're at the first slot of the epoch, we can't get that slot's root yet
            # In that case, use the head_root (most recent known block)
            epoch_start_slot = epoch * SLOTS_PER_EPOCH()
            if epoch_start_slot < int(state.slot):
                target_root = get_block_root(state, epoch)
            else:
                # At or before epoch start - use head_root as target
                target_root = head_root

            source = state.current_justified_checkpoint

            target = Checkpoint(
                epoch=epoch,
                root=target_root,
            )

            key = self.keys.get(duty.pubkey)
            if key is None:
                logger.error(f"No key found for pubkey {duty.pubkey.hex()[:16]}...")
                return None

            domain = get_domain(state, DOMAIN_BEACON_ATTESTER, epoch)

            # Debug: log domain computation details for signing
            fork_version = bytes(state.fork.current_version) if epoch >= int(state.fork.epoch) else bytes(state.fork.previous_version)
            genesis_root = bytes(state.genesis_validators_root)
            logger.debug(
                f"Attestation sign: epoch={epoch}, state_slot={state.slot}, "
                f"fork_epoch={state.fork.epoch}, fork_version={fork_version.hex()}, "
                f"genesis_root={genesis_root.hex()[:16]}, domain={domain.hex()[:16]}"
            )

            if self._is_electra_fork(state):
                # Electra+ attestations use committee_bits
                attestation_data = AttestationData(
                    slot=slot,
                    index=0,  # Electra attestations use committee_bits, not index in data
                    beacon_block_root=head_root,
                    source=source,
                    target=target,
                )

                aggregation_bits = ElectraAggregationBits()
                for i in range(duty.committee_length):
                    aggregation_bits.append(i == duty.validator_committee_index)

                committee_bits = ElectraCommitteeBits()
                committee_bits[committee_index] = True

                signing_root = compute_signing_root(attestation_data, domain)
                signature = bls_sign(key.privkey, signing_root)

                attestation = ElectraAttestation(
                    aggregation_bits=aggregation_bits,
                    data=attestation_data,
                    signature=signature,
                    committee_bits=committee_bits,
                )
            else:
                # Pre-Electra attestations (Phase0 through Deneb)
                attestation_data = AttestationData(
                    slot=slot,
                    index=committee_index,  # Pre-Electra uses index in data
                    beacon_block_root=head_root,
                    source=source,
                    target=target,
                )

                aggregation_bits = Phase0AggregationBits()
                for i in range(duty.committee_length):
                    aggregation_bits.append(i == duty.validator_committee_index)

                signing_root = compute_signing_root(attestation_data, domain)
                signature = bls_sign(key.privkey, signing_root)

                attestation = Phase0Attestation(
                    aggregation_bits=aggregation_bits,
                    data=attestation_data,
                    signature=signature,
                )

            logger.info(
                f"Produced attestation: slot={slot}, committee={committee_index}, "
                f"target_epoch={epoch}, source_epoch={int(source.epoch)}, "
                f"electra={self._is_electra_fork(state)}"
            )

            return attestation

        except Exception as e:
            logger.error(f"Failed to produce attestation: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def produce_sync_committee_message(self, state, slot: int) -> object:
        """Produce a sync committee message."""
        pass
