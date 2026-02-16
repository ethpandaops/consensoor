"""Attestation pool for collecting and aggregating attestations."""

import logging
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

from typing import Union
from .spec.constants import SLOTS_PER_EPOCH, MAX_VALIDATORS_PER_COMMITTEE, MAX_COMMITTEES_PER_SLOT
from .spec.types import AttestationData
from .spec.types.electra import Attestation as ElectraAttestation
from .spec.types.phase0 import Phase0Attestation
from .crypto import hash_tree_root, aggregate_signatures

AnyAttestation = Union[Phase0Attestation, ElectraAttestation]

logger = logging.getLogger(__name__)


@dataclass
class PooledAttestation:
    """An attestation in the pool with metadata."""

    attestation: AnyAttestation
    data_root: bytes
    slot: int
    committee_index: int


class AttestationPool:
    """Pool for collecting attestations to include in blocks.

    Attestations are indexed by (slot, committee_index) for efficient lookup.
    Aggregation is performed when retrieving attestations for block building.
    """

    def __init__(self, max_slots: int = 64):
        self.max_slots = max_slots
        self._attestations: dict[tuple[int, int], list[PooledAttestation]] = defaultdict(list)
        self._seen_data_roots: set[bytes] = set()

    def add(self, attestation: AnyAttestation) -> bool:
        """Add an attestation to the pool.

        Args:
            attestation: The attestation to add

        Returns:
            True if the attestation was added, False if it was a duplicate
        """
        data = attestation.data
        slot = int(data.slot)
        committee_index = int(data.index)
        data_root = hash_tree_root(data)

        key = (slot, committee_index)

        pooled = PooledAttestation(
            attestation=attestation,
            data_root=data_root,
            slot=slot,
            committee_index=committee_index,
        )

        self._attestations[key].append(pooled)
        logger.debug(
            f"Added attestation to pool: slot={slot}, committee={committee_index}, "
            f"bits={sum(1 for b in attestation.aggregation_bits if b)}"
        )
        return True

    def get_attestations_for_block(
        self, current_slot: int, max_attestations: int = 128, electra_fork_epoch: int = 2**64 - 1
    ) -> list[AnyAttestation]:
        """Get attestations suitable for inclusion in a block.

        Returns attestations from previous slots that can be included.
        Attestations are aggregated by data_root (same AttestationData).

        Args:
            current_slot: The slot of the block being built
            max_attestations: Maximum number of attestations to return
            electra_fork_epoch: Epoch at which Electra fork occurs (for filtering)

        Returns:
            List of aggregated attestations for block inclusion
        """
        slots_per_epoch = SLOTS_PER_EPOCH()
        min_slot = max(0, current_slot - slots_per_epoch)
        current_epoch = current_slot // slots_per_epoch
        is_electra_block = current_epoch >= electra_fork_epoch
        electra_start_slot = electra_fork_epoch * slots_per_epoch

        # Group attestations for aggregation
        # Pre-Electra: group by data_root only
        # Electra: group by (data_root, committee_bits) because aggregation_bits positions
        # depend on which committees are included
        by_key: dict[bytes, list[PooledAttestation]] = defaultdict(list)

        for (slot, committee_index), pooled_list in self._attestations.items():
            if slot >= current_slot or slot < min_slot:
                continue

            # Fork compatibility filtering
            if is_electra_block and slot < electra_start_slot:
                continue
            if not is_electra_block and slot >= electra_start_slot:
                continue

            for pooled in pooled_list:
                if is_electra_block and hasattr(pooled.attestation, 'committee_bits'):
                    # Electra: include committee_bits in the grouping key
                    # Convert committee_bits to a hashable tuple
                    committee_bits_tuple = tuple(bool(b) for b in pooled.attestation.committee_bits)
                    key = pooled.data_root + bytes(committee_bits_tuple)
                else:
                    # Pre-Electra: just use data_root
                    key = pooled.data_root
                by_key[key].append(pooled)

        # Phase 1: Within-committee aggregation (same data_root + same committee_bits)
        within_committee = []
        for key, pooled_list in by_key.items():
            try:
                aggregated = self._aggregate_attestations(pooled_list, is_electra_block)
                within_committee.append(aggregated)
            except Exception as e:
                logger.warning(f"Failed to aggregate attestations for key {key[:16].hex()}: {e}")
                if pooled_list:
                    within_committee.append(pooled_list[0].attestation)

        # Phase 2: Cross-committee aggregation for Electra
        # Merge single-committee attestations with same data into one multi-committee attestation
        if is_electra_block:
            result = self._merge_across_committees(within_committee)
        else:
            result = within_committee

        # Sort by slot (newest first) and limit
        result.sort(key=lambda a: int(a.data.slot), reverse=True)

        if len(result) > max_attestations:
            result = result[:max_attestations]

        logger.info(
            f"Returning {len(result)} aggregated attestations for block at slot {current_slot} "
            f"(is_electra={is_electra_block}, max={max_attestations}, groups={len(by_key)})"
        )
        return result

    def _aggregate_attestations(
        self, pooled_list: list[PooledAttestation], is_electra: bool
    ) -> AnyAttestation:
        """Aggregate multiple attestations with the same AttestationData.

        Only aggregates attestations with DISJOINT bit sets to avoid double-counting
        validators' signatures (which would cause BLS verification to fail).

        Args:
            pooled_list: List of attestations with identical data
            is_electra: Whether these are Electra-format attestations

        Returns:
            Aggregated attestation
        """
        if len(pooled_list) == 1:
            return pooled_list[0].attestation

        # Use the first attestation as template
        first = pooled_list[0].attestation
        data = first.data

        # Sort by number of bits set (descending) to prefer attestations with more validators
        sorted_pooled = sorted(
            pooled_list,
            key=lambda p: sum(1 for b in p.attestation.aggregation_bits if b),
            reverse=True
        )

        # Track which bit positions we've already included
        # Only include attestations with DISJOINT bit sets (no overlap)
        included_bits: set[int] = set()
        aggregatable: list[PooledAttestation] = []

        for pooled in sorted_pooled:
            att = pooled.attestation
            att_bits = {i for i, bit in enumerate(att.aggregation_bits) if bit}

            # Check for ANY overlap with already included bits
            if att_bits & included_bits:
                # Has overlapping bits - skip this attestation to avoid signature issues
                continue

            # No overlap - we can include this attestation
            aggregatable.append(pooled)
            included_bits |= att_bits

        if len(aggregatable) == 0:
            return first

        # Debug: log committee_bits for Electra attestations being aggregated
        if is_electra and len(aggregatable) > 1:
            cb_list = []
            for p in aggregatable:
                if hasattr(p.attestation, 'committee_bits'):
                    cb = [i for i, b in enumerate(p.attestation.committee_bits) if b]
                    cb_list.append(cb)
            logger.debug(f"Aggregating {len(aggregatable)} Electra attestations, committee_bits: {cb_list}")

        if len(aggregatable) == 1:
            return aggregatable[0].attestation

        # Aggregate BLS signatures from disjoint attestations
        signatures = [bytes(p.attestation.signature) for p in aggregatable]

        # Log signature details for debugging
        for i, p in enumerate(aggregatable):
            att = p.attestation
            att_bits = {j for j, bit in enumerate(att.aggregation_bits) if bit}
            logger.info(
                f"Aggregation signature {i}: slot={att.data.slot}, "
                f"target_epoch={att.data.target.epoch}, "
                f"source_epoch={att.data.source.epoch}, "
                f"beacon_block_root={bytes(att.data.beacon_block_root).hex()[:16]}, "
                f"target_root={bytes(att.data.target.root).hex()[:16]}, "
                f"source_root={bytes(att.data.source.root).hex()[:16]}, "
                f"data_root={p.data_root.hex()[:16]}, "
                f"agg_bits={att_bits}, "
                f"sig={bytes(att.signature).hex()[:32]}"
            )

        try:
            aggregated_sig = aggregate_signatures(signatures)
        except Exception as e:
            logger.warning(f"Failed to aggregate {len(signatures)} signatures: {e}, using first")
            return aggregatable[0].attestation

        logger.debug(
            f"Aggregated {len(aggregatable)} attestations with {len(included_bits)} total bits"
        )

        if is_electra or hasattr(first, 'committee_bits'):
            # Electra attestation - merge aggregation_bits and committee_bits
            merged_agg_bits = self._merge_electra_aggregation_bits(aggregatable)
            merged_committee_bits = self._merge_electra_committee_bits(aggregatable)

            from .spec.types.electra import Attestation as ElectraAttestation
            from .spec.types import BLSSignature

            return ElectraAttestation(
                aggregation_bits=merged_agg_bits,
                data=data,
                signature=BLSSignature(aggregated_sig),
                committee_bits=merged_committee_bits,
            )
        else:
            # Phase0 attestation - just merge aggregation_bits
            merged_bits = self._merge_phase0_aggregation_bits(aggregatable)

            from .spec.types.phase0 import Phase0Attestation
            from .spec.types import BLSSignature

            return Phase0Attestation(
                aggregation_bits=merged_bits,
                data=data,
                signature=BLSSignature(aggregated_sig),
            )

    def _merge_across_committees(self, attestations: list[AnyAttestation]) -> list[AnyAttestation]:
        """Merge Electra attestations from different committees with same AttestationData.

        In Electra, each validator produces a single-committee attestation. After
        within-committee aggregation, we have one attestation per (data, committee).
        This merges those into one attestation per data by concatenating
        aggregation_bits in committee index order per the spec.
        """
        by_data: dict[bytes, list[AnyAttestation]] = defaultdict(list)
        for att in attestations:
            data_root = hash_tree_root(att.data)
            by_data[data_root].append(att)

        result = []
        for data_root, atts in by_data.items():
            if len(atts) == 1:
                result.append(atts[0])
                continue

            # Collect per-committee data from single-committee attestations
            # committee_index -> (aggregation_bits, signature)
            committee_map: dict[int, tuple] = {}
            multi_committee = []

            for att in atts:
                if not hasattr(att, 'committee_bits'):
                    result.append(att)
                    continue
                committees = [i for i, b in enumerate(att.committee_bits) if b]
                if len(committees) == 1:
                    ci = committees[0]
                    if ci not in committee_map:
                        committee_map[ci] = (att.aggregation_bits, att.signature)
                    else:
                        existing_count = sum(1 for b in committee_map[ci][0] if b)
                        new_count = sum(1 for b in att.aggregation_bits if b)
                        if new_count > existing_count:
                            committee_map[ci] = (att.aggregation_bits, att.signature)
                else:
                    multi_committee.append(att)

            if not committee_map:
                result.extend(atts)
                continue

            # Build merged attestation: concatenate aggregation_bits in committee order
            from .spec.types.base import Bitlist, Bitvector
            from .spec.types import BLSSignature

            sorted_committees = sorted(committee_map.keys())

            CommitteeBitsType = Bitvector[MAX_COMMITTEES_PER_SLOT()]
            merged_committee_bits = CommitteeBitsType()
            for ci in sorted_committees:
                merged_committee_bits[ci] = True

            AggBitsType = Bitlist[MAX_VALIDATORS_PER_COMMITTEE() * MAX_COMMITTEES_PER_SLOT()]
            merged_agg_bits = AggBitsType()
            for ci in sorted_committees:
                bits = committee_map[ci][0]
                for b in bits:
                    merged_agg_bits.append(bool(b))

            sigs = [bytes(committee_map[ci][1]) for ci in sorted_committees]
            try:
                aggregated_sig = aggregate_signatures(sigs)
            except Exception as e:
                logger.warning(f"Cross-committee signature aggregation failed: {e}")
                result.extend(atts)
                continue

            merged = ElectraAttestation(
                aggregation_bits=merged_agg_bits,
                data=atts[0].data,
                signature=BLSSignature(aggregated_sig),
                committee_bits=merged_committee_bits,
            )
            result.append(merged)
            result.extend(multi_committee)

            logger.debug(
                f"Cross-committee merge: {len(committee_map)} committees -> 1 attestation, "
                f"slot={atts[0].data.slot}, committees={sorted_committees}, "
                f"total_bits={len(merged_agg_bits)}"
            )

        return result

    def _merge_phase0_aggregation_bits(self, pooled_list: list[PooledAttestation]):
        """Merge aggregation bits for Phase0 attestations (OR operation)."""
        if not pooled_list:
            return None

        # Get the length from first attestation
        first_bits = pooled_list[0].attestation.aggregation_bits
        bit_len = len(first_bits)

        # Create merged bits
        from .spec.types.base import Bitlist
        AggBitsType = Bitlist[MAX_VALIDATORS_PER_COMMITTEE()]
        merged = AggBitsType()

        # Initialize with False values
        for _ in range(bit_len):
            merged.append(False)

        # OR all bits together
        for pooled in pooled_list:
            bits = pooled.attestation.aggregation_bits
            for i in range(min(len(bits), len(merged))):
                if bits[i]:
                    merged[i] = True

        return merged

    def _merge_electra_aggregation_bits(self, pooled_list: list[PooledAttestation]):
        """Merge aggregation bits for Electra attestations (OR operation)."""
        if not pooled_list:
            return None

        first_bits = pooled_list[0].attestation.aggregation_bits
        bit_len = len(first_bits)

        from .spec.types.base import Bitlist
        AggBitsType = Bitlist[MAX_VALIDATORS_PER_COMMITTEE() * MAX_COMMITTEES_PER_SLOT()]
        merged = AggBitsType()

        for _ in range(bit_len):
            merged.append(False)

        for pooled in pooled_list:
            bits = pooled.attestation.aggregation_bits
            for i in range(min(len(bits), len(merged))):
                if bits[i]:
                    merged[i] = True

        return merged

    def _merge_electra_committee_bits(self, pooled_list: list[PooledAttestation]):
        """Merge committee bits for Electra attestations (OR operation)."""
        if not pooled_list:
            return None

        from .spec.types.base import Bitvector
        CommitteeBitsType = Bitvector[MAX_COMMITTEES_PER_SLOT()]
        merged = CommitteeBitsType()

        for pooled in pooled_list:
            if hasattr(pooled.attestation, 'committee_bits'):
                bits = pooled.attestation.committee_bits
                for i in range(len(bits)):
                    if bits[i]:
                        merged[i] = True

        return merged

    def remove_included(self, attestations: list) -> int:
        """Remove attestations that have been included in a block.

        Matches by (slot, committee_index) and data_root to remove
        attestations already on-chain, preventing re-inclusion in future blocks.

        Args:
            attestations: List of attestations that were included in a block

        Returns:
            Number of attestations removed
        """
        removed = 0
        for att in attestations:
            slot = int(att.data.slot)
            committee_index = int(att.data.index)
            key = (slot, committee_index)

            if key not in self._attestations:
                continue

            data_root = hash_tree_root(att.data)
            before = len(self._attestations[key])
            self._attestations[key] = [
                p for p in self._attestations[key] if p.data_root != data_root
            ]
            after = len(self._attestations[key])
            removed += before - after

            if not self._attestations[key]:
                del self._attestations[key]

        if removed > 0:
            logger.debug(f"Removed {removed} included attestations from pool")
        return removed

    def prune(self, current_slot: int) -> int:
        """Remove old attestations from the pool.

        Args:
            current_slot: Current slot

        Returns:
            Number of attestations pruned
        """
        slots_per_epoch = SLOTS_PER_EPOCH()
        min_slot = max(0, current_slot - slots_per_epoch * 2)

        pruned = 0
        keys_to_remove = []

        for key in self._attestations:
            slot, _ = key
            if slot < min_slot:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            pruned += len(self._attestations[key])
            del self._attestations[key]

        if pruned > 0:
            logger.debug(f"Pruned {pruned} old attestations from pool")

        return pruned

    @property
    def size(self) -> int:
        """Return the total number of attestations in the pool."""
        return sum(len(v) for v in self._attestations.values())
