"""Validator client for producing blocks and attestations."""

import logging
from typing import Optional

from ..spec.constants import (
    SLOTS_PER_EPOCH,
    DOMAIN_BEACON_ATTESTER,
    DOMAIN_SYNC_COMMITTEE,
    DOMAIN_PTC_ATTESTER,
)
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
from ..spec.types.altair import SyncCommitteeMessage
from ..spec.types.gloas import (
    PayloadAttestationData,
    PayloadAttestationMessage,
)
from ..spec.types.base import Checkpoint, Bitlist, Bitvector
from ..spec.constants import MAX_VALIDATORS_PER_COMMITTEE, MAX_COMMITTEES_PER_SLOT

Phase0AggregationBits = Bitlist[MAX_VALIDATORS_PER_COMMITTEE()]
ElectraAggregationBits = Bitlist[MAX_VALIDATORS_PER_COMMITTEE() * MAX_COMMITTEES_PER_SLOT()]
ElectraCommitteeBits = Bitvector[MAX_COMMITTEES_PER_SLOT()]
from ..crypto import sign_async as bls_sign_async, hash_tree_root
from .types import ValidatorKey, ProposerDuty, AttesterDuty
from .shuffling import get_beacon_proposer_index
from .. import metrics

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
        newly_resolved: list[int] = []
        for i, validator in enumerate(validators):
            pubkey = bytes(validator.pubkey)
            if pubkey in self.pubkeys and pubkey not in self._validator_indices:
                self._validator_indices[pubkey] = i
                key = self.keys[pubkey]
                key.validator_index = i
                newly_resolved.append(i)
        if newly_resolved:
            logger.info(
                f"Resolved validator indices ({len(newly_resolved)}/{len(self.pubkeys)}): "
                f"{sorted(newly_resolved)}"
            )

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

        if duties and logger.isEnabledFor(logging.DEBUG):
            # One rollup line per epoch instead of one line per duty.
            by_slot: dict[int, list[int]] = {}
            for d in duties:
                by_slot.setdefault(d.slot, []).append(int(d.validator_index))
            slot_summary = " ".join(
                f"s{slot}=[{','.join(str(v) for v in sorted(vs))}]"
                for slot, vs in sorted(by_slot.items())
            )
            logger.debug(
                f"Attester duties epoch={epoch} count={len(duties)} {slot_summary}"
            )

        return duties

    def _is_electra_fork(self, state) -> bool:
        """Check if the state is at or after Electra fork."""
        # Electra states have pending_deposits field
        return hasattr(state, "pending_deposits")

    async def produce_attestation(
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
                signature = await bls_sign_async(key.privkey, signing_root)

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
                signature = await bls_sign_async(key.privkey, signing_root)

                attestation = Phase0Attestation(
                    aggregation_bits=aggregation_bits,
                    data=attestation_data,
                    signature=signature,
                )

            metrics.record_attestation_produced()
            return attestation

        except Exception as e:
            logger.error(f"Failed to produce attestation: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def produce_sync_committee_message(
        self,
        state,
        slot: int,
        validator_key: ValidatorKey,
        head_block_root: Optional[bytes] = None,
    ) -> Optional[SyncCommitteeMessage]:
        """Produce a sync committee message for a validator.

        Sync committee members sign the head block root at the start of each
        slot. The caller MUST pass the head block's actual root via
        `head_block_root`: at slot start, state.slot is still slot-1 and
        process_slot for slot-1 hasn't run yet, so
        state.block_roots[(slot-1) % HIST] is stale (or zero) and would
        produce a different signing root than the block builder later uses.
        Using the tracked head root keeps validator + builder in sync.

        Args:
            state: Beacon state
            slot: Current slot
            validator_key: The validator's key
            head_block_root: The current head block root (= block at slot-1
                with its state_root filled in). Required for correct
                signing.

        Returns:
            SyncCommitteeMessage or None if production fails
        """
        try:
            if validator_key.validator_index is None:
                logger.warning("Cannot produce sync committee message: no validator index")
                return None

            previous_slot = max(0, slot - 1)
            state_slot = int(state.slot)
            from ..spec.constants import SLOTS_PER_HISTORICAL_ROOT

            if head_block_root is not None and head_block_root != b"\x00" * 32:
                beacon_block_root = head_block_root
            elif previous_slot < state_slot:
                beacon_block_root = get_block_root_at_slot(state, previous_slot)
            else:
                block_root_entry = bytes(state.block_roots[previous_slot % SLOTS_PER_HISTORICAL_ROOT()])
                if block_root_entry == b'\x00' * 32:
                    beacon_block_root = hash_tree_root(state.latest_block_header)
                else:
                    beacon_block_root = block_root_entry

            # Per altair/validator.md `get_sync_committee_message`, the domain
            # uses the message's own slot epoch (epoch = get_current_epoch(state)
            # = epoch_at_slot(state.slot) = epoch_at_slot(message.slot)).
            # process_sync_aggregate for the block at slot M+1 verifies with
            # epoch_at_slot(M) — matches when message.slot = M.
            epoch = compute_epoch_at_slot(slot)
            domain = get_domain(state, DOMAIN_SYNC_COMMITTEE, epoch)
            signing_root = compute_signing_root(beacon_block_root, domain)
            signature = await bls_sign_async(validator_key.privkey, signing_root)

            message = SyncCommitteeMessage(
                slot=slot,
                beacon_block_root=beacon_block_root,
                validator_index=validator_key.validator_index,
                signature=signature,
            )
            return message

        except Exception as e:
            logger.error(f"Failed to produce sync committee message: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def produce_payload_attestation_message(
        self,
        state,
        slot: int,
        validator_key: ValidatorKey,
        beacon_block_root: bytes,
        payload_present: bool,
        blob_data_available: bool,
    ) -> Optional[PayloadAttestationMessage]:
        """Produce a PTC payload attestation message for one of our validators.

        PTC members at slot M sign `PayloadAttestationData{beacon_block_root,
        slot=M, payload_present, blob_data_available}` near the 75% slot mark
        and gossip it. The proposer of slot M+1 includes the aggregate in
        its block body (`block.body.payload_attestations`).

        Domain is computed for the epoch of `slot` (the slot whose payload
        is being voted on).
        """
        try:
            if validator_key.validator_index is None:
                return None

            data = PayloadAttestationData(
                beacon_block_root=beacon_block_root,
                slot=slot,
                payload_present=payload_present,
                blob_data_available=blob_data_available,
            )
            epoch = compute_epoch_at_slot(slot)
            domain = get_domain(state, DOMAIN_PTC_ATTESTER, epoch)
            signing_root = compute_signing_root(data, domain)
            signature = await bls_sign_async(validator_key.privkey, signing_root)

            return PayloadAttestationMessage(
                validator_index=validator_key.validator_index,
                data=data,
                signature=signature,
            )

        except Exception as e:
            logger.error(f"Failed to produce payload attestation message: {e}")
            import traceback
            traceback.print_exc()
            return None
