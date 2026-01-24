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
from .spec.types.base import Bitlist
from .crypto import hash_tree_root

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
        Attestations are aggregated by (slot, committee_index, data_root).

        Args:
            current_slot: The slot of the block being built
            max_attestations: Maximum number of attestations to return
            electra_fork_epoch: Epoch at which Electra fork occurs (for filtering)

        Returns:
            List of attestations for block inclusion
        """
        result = []

        slots_per_epoch = SLOTS_PER_EPOCH()
        min_slot = max(0, current_slot - slots_per_epoch)
        current_epoch = current_slot // slots_per_epoch
        is_electra_block = current_epoch >= electra_fork_epoch
        electra_start_slot = electra_fork_epoch * slots_per_epoch

        # Collect individual attestations (skip aggregation for simplicity)
        for (slot, committee_index), pooled_list in self._attestations.items():
            if slot >= current_slot or slot < min_slot:
                continue

            # For Electra blocks, only include attestations from Electra epoch onwards
            # (Phase0 and Electra attestations have incompatible formats)
            if is_electra_block and slot < electra_start_slot:
                continue

            # For pre-Electra blocks, only include attestations from before Electra
            if not is_electra_block and slot >= electra_start_slot:
                continue

            for pooled in pooled_list:
                result.append(pooled.attestation)

        # Sort by slot (newest first) and limit
        result.sort(key=lambda a: int(a.data.slot), reverse=True)

        if len(result) > max_attestations:
            result = result[:max_attestations]

        logger.info(
            f"Returning {len(result)} attestations for block at slot {current_slot} "
            f"(electra_fork_epoch={electra_fork_epoch}, is_electra={is_electra_block}, "
            f"electra_start_slot={electra_start_slot}, max={max_attestations})"
        )
        return result

    def _merge_aggregation_bits(
        self, bits1: Bitlist, bits2: Bitlist
    ) -> Bitlist:
        """Merge two aggregation bitlists (OR operation)."""
        max_len = max(len(bits1), len(bits2))
        AggBitsType = Attestation.fields()['aggregation_bits']

        merged = AggBitsType(*([False] * max_len))
        for i in range(max_len):
            val1 = bits1[i] if i < len(bits1) else False
            val2 = bits2[i] if i < len(bits2) else False
            merged[i] = val1 or val2
        return merged

    def _merge_committee_bits(self, bits1, bits2):
        """Merge two committee bitvectors (OR operation)."""
        CommitteeBitsType = Attestation.fields()['committee_bits']
        max_committees = MAX_COMMITTEES_PER_SLOT()

        merged = CommitteeBitsType()
        for i in range(max_committees):
            merged[i] = bits1[i] or bits2[i]
        return merged

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
