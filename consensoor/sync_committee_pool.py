"""Sync committee message pool for collecting and aggregating sync committee signatures."""

import logging
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional, Union

from .spec.constants import SYNC_COMMITTEE_SIZE, SYNC_COMMITTEE_SUBNET_COUNT
from .spec.types import BLSSignature, Root
from .spec.types.altair import SyncAggregate, SyncCommitteeMessage, SyncCommitteeContribution
from .spec.types.base import Bitvector
from .crypto import aggregate_signatures

logger = logging.getLogger(__name__)


@dataclass
class PooledSyncMessage:
    """A sync committee message in the pool."""
    message: SyncCommitteeMessage
    committee_position: int


@dataclass
class PooledContribution:
    """A sync committee contribution in the pool."""
    contribution: SyncCommitteeContribution
    subcommittee_index: int


class SyncCommitteePool:
    """Pool for collecting sync committee messages to aggregate into SyncAggregate.

    Supports both individual messages (from local validators) and contributions
    (from P2P gossip with pre-aggregated signatures).
    """

    def __init__(self, max_slots: int = 8):
        self.max_slots = max_slots
        self._messages: dict[int, dict[int, PooledSyncMessage]] = defaultdict(dict)
        self._contributions: dict[int, dict[int, PooledContribution]] = defaultdict(dict)

    def add(
        self, message: SyncCommitteeMessage, committee_position: int
    ) -> bool:
        """Add a sync committee message to the pool.

        Args:
            message: The sync committee message
            committee_position: Position in the sync committee (0 to SYNC_COMMITTEE_SIZE-1)

        Returns:
            True if the message was added, False if duplicate
        """
        slot = int(message.slot)

        if committee_position in self._messages[slot]:
            return False

        self._messages[slot][committee_position] = PooledSyncMessage(
            message=message,
            committee_position=committee_position,
        )

        logger.debug(
            f"Added sync committee message: slot={slot}, position={committee_position}, "
            f"validator={message.validator_index}"
        )
        return True

    def add_contribution(self, contribution: SyncCommitteeContribution) -> bool:
        """Add a sync committee contribution to the pool.

        Contributions contain pre-aggregated signatures for a subcommittee.
        If a contribution already exists for this subcommittee, merge the bits
        and aggregate the signatures (if bits are non-overlapping).

        Args:
            contribution: The sync committee contribution

        Returns:
            True if the contribution was added/merged, False if no new bits
        """
        slot = int(contribution.slot)
        subcommittee_index = int(contribution.subcommittee_index)

        existing = self._contributions[slot].get(subcommittee_index)
        if existing:
            logger.debug(f"Checking existing contribution for slot={slot}, subcommittee={subcommittee_index}")
            try:
                existing_bits = existing.contribution.aggregation_bits
                new_bits = contribution.aggregation_bits

                # Check for new bits not in existing
                has_new_bits = False
                has_overlap = False
                overlap_count = 0
                new_only_count = 0
                for i, (eb, nb) in enumerate(zip(existing_bits, new_bits)):
                    if nb and not eb:
                        has_new_bits = True
                        new_only_count += 1
                    if nb and eb:
                        has_overlap = True
                        overlap_count += 1

                existing_count = sum(1 for b in existing_bits if b)
                new_count = sum(1 for b in new_bits if b)
                logger.info(
                    f"Contribution merge check: slot={slot}, subcommittee={subcommittee_index}, "
                    f"existing={existing_count}, new={new_count}, overlap={overlap_count}, new_only={new_only_count}"
                )

                if not has_new_bits:
                    logger.info(f"No new bits in contribution for slot={slot} subcommittee={subcommittee_index}, skipping")
                    return False
            except Exception as e:
                logger.error(f"Error checking contribution merge: {e}")
                import traceback
                traceback.print_exc()
                return False

            if has_overlap:
                # Overlapping bits - can't safely merge, keep the one with more bits
                existing_count = sum(1 for b in existing_bits if b)
                new_count = sum(1 for b in new_bits if b)
                if new_count <= existing_count:
                    return False
                # Replace with new contribution
                self._contributions[slot][subcommittee_index] = PooledContribution(
                    contribution=contribution,
                    subcommittee_index=subcommittee_index,
                )
                logger.debug(
                    f"Replaced sync committee contribution (overlap): slot={slot}, "
                    f"subcommittee={subcommittee_index}, participants={new_count}"
                )
                return True

            # No overlap - merge the contributions
            try:
                merged_sig = aggregate_signatures([
                    bytes(existing.contribution.signature),
                    bytes(contribution.signature)
                ])

                # Create merged bits
                from .spec.constants import SYNC_COMMITTEE_SIZE, SYNC_COMMITTEE_SUBNET_COUNT
                subcommittee_size = SYNC_COMMITTEE_SIZE() // SYNC_COMMITTEE_SUBNET_COUNT
                from .spec.types.base import Bitvector
                merged_bits = Bitvector[subcommittee_size]()
                for i in range(len(existing_bits)):
                    merged_bits[i] = existing_bits[i] or new_bits[i]

                merged_contribution = SyncCommitteeContribution(
                    slot=contribution.slot,
                    beacon_block_root=contribution.beacon_block_root,
                    subcommittee_index=contribution.subcommittee_index,
                    aggregation_bits=merged_bits,
                    signature=BLSSignature(merged_sig),
                )

                self._contributions[slot][subcommittee_index] = PooledContribution(
                    contribution=merged_contribution,
                    subcommittee_index=subcommittee_index,
                )

                merged_count = sum(1 for b in merged_bits if b)
                logger.debug(
                    f"Merged sync committee contribution: slot={slot}, "
                    f"subcommittee={subcommittee_index}, participants={merged_count}"
                )
                return True

            except Exception as e:
                logger.warning(f"Failed to merge contributions: {e}")
                # Fall back to keeping the one with more bits
                existing_count = sum(1 for b in existing_bits if b)
                new_count = sum(1 for b in new_bits if b)
                if new_count > existing_count:
                    self._contributions[slot][subcommittee_index] = PooledContribution(
                        contribution=contribution,
                        subcommittee_index=subcommittee_index,
                    )
                return True

        self._contributions[slot][subcommittee_index] = PooledContribution(
            contribution=contribution,
            subcommittee_index=subcommittee_index,
        )

        participant_count = sum(1 for b in contribution.aggregation_bits if b)
        logger.debug(
            f"Added sync committee contribution: slot={slot}, "
            f"subcommittee={subcommittee_index}, participants={participant_count}"
        )
        return True

    def get_sync_aggregate(self, slot: int) -> SyncAggregate:
        """Get aggregated SyncAggregate for a slot.

        Combines individual messages and contributions into a single SyncAggregate.
        Contributions take precedence for their subcommittees to avoid duplicate signatures.

        Args:
            slot: The slot to aggregate for

        Returns:
            SyncAggregate with aggregated bits and signature
        """
        sync_committee_size = SYNC_COMMITTEE_SIZE()
        subcommittee_size = sync_committee_size // SYNC_COMMITTEE_SUBNET_COUNT

        sync_bits = Bitvector[sync_committee_size]()
        signatures = []
        covered_positions: set[int] = set()

        contributions = self._contributions.get(slot, {})
        for subcommittee_index, pooled in contributions.items():
            contribution = pooled.contribution
            base_position = subcommittee_index * subcommittee_size

            has_bits = False
            for bit_index, bit_set in enumerate(contribution.aggregation_bits):
                if bit_set:
                    position = base_position + bit_index
                    if position < sync_committee_size:
                        sync_bits[position] = True
                        covered_positions.add(position)
                        has_bits = True

            if has_bits:
                signatures.append(bytes(contribution.signature))

        messages = self._messages.get(slot, {})
        uncovered_messages = []
        for position, pooled in messages.items():
            if position not in covered_positions and 0 <= position < sync_committee_size:
                sync_bits[position] = True
                uncovered_messages.append(pooled)

        if uncovered_messages:
            msg_signatures = [bytes(p.message.signature) for p in uncovered_messages]
            if len(msg_signatures) == 1:
                signatures.append(msg_signatures[0])
            elif msg_signatures:
                try:
                    aggregated_msg_sig = aggregate_signatures(msg_signatures)
                    signatures.append(aggregated_msg_sig)
                except Exception as e:
                    logger.warning(f"Failed to aggregate message signatures: {e}")

        if not signatures:
            return self._empty_sync_aggregate()

        try:
            if len(signatures) == 1:
                final_signature = signatures[0]
            else:
                final_signature = aggregate_signatures(signatures)
        except Exception as e:
            logger.warning(f"Failed to aggregate sync committee signatures: {e}")
            return self._empty_sync_aggregate()

        total_bits = sum(1 for b in sync_bits if b)
        logger.info(
            f"Aggregated sync committee for slot {slot}: "
            f"{total_bits}/{sync_committee_size} participation "
            f"({len(contributions)} contributions, {len(uncovered_messages)} individual)"
        )

        return SyncAggregate(
            sync_committee_bits=sync_bits,
            sync_committee_signature=BLSSignature(final_signature),
        )

    def _empty_sync_aggregate(self) -> SyncAggregate:
        """Return an empty sync aggregate with G2 infinity point signature."""
        sync_committee_size = SYNC_COMMITTEE_SIZE()
        return SyncAggregate(
            sync_committee_bits=Bitvector[sync_committee_size](),
            sync_committee_signature=BLSSignature(b"\xc0" + b"\x00" * 95),
        )

    def prune(self, current_slot: int) -> int:
        """Remove old messages from the pool.

        Args:
            current_slot: Current slot

        Returns:
            Number of slots pruned
        """
        min_slot = max(0, current_slot - self.max_slots)

        slots_to_remove = [s for s in self._messages if s < min_slot]
        contrib_slots_to_remove = [s for s in self._contributions if s < min_slot]

        for slot in slots_to_remove:
            del self._messages[slot]
        for slot in contrib_slots_to_remove:
            del self._contributions[slot]

        total_pruned = len(slots_to_remove) + len(contrib_slots_to_remove)
        if total_pruned > 0:
            logger.debug(f"Pruned sync committee data for {total_pruned} slots")

        return total_pruned

    @property
    def size(self) -> int:
        """Return the total number of messages in the pool."""
        msg_count = sum(len(msgs) for msgs in self._messages.values())
        contrib_count = sum(len(c) for c in self._contributions.values())
        return msg_count + contrib_count
