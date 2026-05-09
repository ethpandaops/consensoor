"""Payload Timeliness Committee (PTC) attestation pool.

Collects gossip-received `PayloadAttestationMessage` from PTC validators
and aggregates them by `PayloadAttestationData` so the next-slot proposer
can include up to MAX_PAYLOAD_ATTESTATIONS aggregated `PayloadAttestation`
in the block body. Per gloas/EIP-7732.
"""

import logging
from typing import Optional

from .spec.constants import MAX_PAYLOAD_ATTESTATIONS, PTC_SIZE
from .spec.types.gloas import (
    PayloadAttestationMessage,
    PayloadAttestation,
    PayloadAttestationData,
)
from .spec.types.base import Bitvector, BLSSignature
from .crypto import aggregate_signatures, hash_tree_root

logger = logging.getLogger(__name__)


class PayloadAttestationPool:
    """In-memory pool of received PTC attestation messages."""

    def __init__(self) -> None:
        # data_root → list[(ptc_index_within_committee, signature)]
        self._messages: dict[bytes, list[tuple[int, bytes]]] = {}
        # data_root → PayloadAttestationData (cached so we can rebuild aggregates)
        self._data: dict[bytes, PayloadAttestationData] = {}
        # slot → set[validator_index] of senders we've already accepted
        self._seen: dict[int, set[int]] = {}

    def add_message(self, msg: PayloadAttestationMessage, ptc: list[int]) -> None:
        """Add a PayloadAttestationMessage if it's from a valid PTC member.

        `ptc` must be the canonical PTC list for `msg.data.slot`. The
        validator's bit position is its index within `ptc`. We dedupe by
        validator_index so a misbehaving sender can't inflate the
        aggregate's bit count.
        """
        validator_index = int(msg.validator_index)
        slot = int(msg.data.slot)

        seen_for_slot = self._seen.setdefault(slot, set())
        if validator_index in seen_for_slot:
            return

        try:
            ptc_index = ptc.index(validator_index)
        except ValueError:
            # Sender isn't in this slot's PTC — drop silently.
            return
        seen_for_slot.add(validator_index)

        data_root = bytes(hash_tree_root(msg.data))
        self._messages.setdefault(data_root, []).append((ptc_index, bytes(msg.signature)))
        self._data.setdefault(data_root, msg.data)

    def get_aggregates_for_slot(self, parent_slot: int) -> list[PayloadAttestation]:
        """Return aggregated PayloadAttestations for messages that vote on
        the parent slot's payload. Capped at MAX_PAYLOAD_ATTESTATIONS.

        One aggregate per distinct PayloadAttestationData (so the same
        beacon_block_root + payload_status group all collapse into one).
        """
        aggregates: list[PayloadAttestation] = []
        for data_root, entries in self._messages.items():
            data = self._data.get(data_root)
            if data is None or int(data.slot) != int(parent_slot):
                continue
            if not entries:
                continue
            bits = Bitvector[PTC_SIZE()]()
            sigs: list[bytes] = []
            for ptc_index, sig in entries:
                if 0 <= ptc_index < PTC_SIZE() and not bits[ptc_index]:
                    bits[ptc_index] = True
                    sigs.append(sig)
            if not sigs:
                continue
            try:
                aggregated_sig = (
                    sigs[0] if len(sigs) == 1 else aggregate_signatures(sigs)
                )
            except Exception as e:
                logger.warning(
                    f"PTC aggregate skipped for data_root={data_root.hex()[:16]}: {e}"
                )
                continue
            aggregates.append(
                PayloadAttestation(
                    aggregation_bits=bits,
                    data=data,
                    signature=BLSSignature(aggregated_sig),
                )
            )
            if len(aggregates) >= MAX_PAYLOAD_ATTESTATIONS:
                break
        return aggregates

    def prune_before(self, keep_from_slot: int) -> None:
        """Drop messages older than keep_from_slot."""
        stale_data_roots = [
            r
            for r, d in self._data.items()
            if int(d.slot) < keep_from_slot
        ]
        for r in stale_data_roots:
            self._messages.pop(r, None)
            self._data.pop(r, None)
        for slot in list(self._seen.keys()):
            if slot < keep_from_slot:
                del self._seen[slot]
