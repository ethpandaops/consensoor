"""Attestation pool for PQ: leanXMSS signatures and aggregate proofs, grouped by committee."""

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

from .crypto import hash_tree_root
from .spec.constants import MIN_ATTESTATION_INCLUSION_DELAY_PQ, SLOTS_PER_EPOCH
from .spec.types.pq import Attestation, CommitteeBits

logger = logging.getLogger(__name__)


@dataclass
class CommitteeVotes:
    committee: list[int]
    signatures: dict[int, bytes] = field(default_factory=dict)
    proven_bits: tuple[bool, ...] = ()
    proof: bytes = b""

    def raw_bits(self) -> tuple[bool, ...]:
        return tuple(position in self.signatures for position in range(len(self.committee)))

    def bits(self) -> tuple[bool, ...]:
        if sum(self.proven_bits) >= len(self.signatures):
            return self.proven_bits
        return self.raw_bits()


@dataclass
class PooledData:
    data: object
    slot: int
    committees: dict[int, CommitteeVotes] = field(default_factory=dict)


class PQAttestationPool:
    def __init__(self) -> None:
        self._entries: dict[bytes, PooledData] = {}

    def _votes(self, data, committee_index: int, committee: list[int]) -> CommitteeVotes:
        data_root = bytes(hash_tree_root(data))
        entry = self._entries.get(data_root)
        if entry is None:
            entry = PooledData(data=data, slot=int(data.slot))
            self._entries[data_root] = entry
        votes = entry.committees.get(committee_index)
        if votes is None:
            votes = CommitteeVotes(committee=[int(v) for v in committee])
            entry.committees[committee_index] = votes
        return votes

    def add_single(self, single, committee: list[int]) -> bool:
        votes = self._votes(single.data, int(single.committee_index), committee)
        attester = int(single.attester_index)
        if attester not in votes.committee:
            return False
        position = votes.committee.index(attester)
        if position in votes.signatures:
            return False
        votes.signatures[position] = bytes(single.signature)
        return True

    def add_aggregate(self, attestation, proof: bytes, committee: list[int]) -> bool:
        committee_index = self.committee_of(attestation)
        if committee_index < 0:
            return False
        bits = tuple(bool(b) for b in attestation.aggregation_bits)
        proof = bytes(proof)
        if len(bits) != len(committee) or not any(bits) or not proof:
            return False
        votes = self._votes(attestation.data, committee_index, committee)
        if sum(bits) <= sum(votes.proven_bits):
            return False
        votes.proven_bits = bits
        votes.proof = proof
        return True

    @staticmethod
    def committee_of(attestation) -> int:
        bits = [i for i, b in enumerate(attestation.committee_bits) if b]
        return bits[0] if len(bits) == 1 else -1

    @staticmethod
    def _attestation(data, committees: list[tuple[int, tuple[bool, ...]]]) -> Attestation:
        aggregation_bits: list[bool] = []
        committee_bits = CommitteeBits()
        for committee_index, bits in committees:
            committee_bits[committee_index] = True
            aggregation_bits.extend(bits)
        return Attestation(
            aggregation_bits=aggregation_bits,
            data=data,
            committee_bits=committee_bits,
        )

    def get_aggregate_inputs(self, slot: int, data_root: bytes, committee_index: int):
        entry = self._entries.get(bytes(data_root))
        if entry is None or entry.slot != int(slot):
            return None
        votes = entry.committees.get(int(committee_index))
        if votes is None or not votes.signatures:
            return None
        attestation = self._attestation(entry.data, [(int(committee_index), votes.raw_bits())])
        return attestation, [votes.signatures[position] for position in sorted(votes.signatures)]

    def get_block_candidates(self, current_slot: int) -> list[bytes]:
        min_slot = max(0, current_slot - 2 * SLOTS_PER_EPOCH())
        candidates = [
            (entry.slot, root)
            for root, entry in self._entries.items()
            if min_slot <= entry.slot <= current_slot - MIN_ATTESTATION_INCLUSION_DELAY_PQ
            and any(any(votes.bits()) for votes in entry.committees.values())
        ]
        candidates.sort(reverse=True)
        return [root for _, root in candidates]

    def build_for_block(
        self,
        data_root: bytes,
        prove: Callable[[Attestation, list[bytes]], bytes],
        verify: Optional[Callable[[Attestation, bytes], bool]] = None,
    ):
        entry = self._entries.get(bytes(data_root))
        if entry is None:
            return None
        children: list[tuple[Attestation, bytes]] = []
        committees: list[tuple[int, tuple[bool, ...]]] = []
        for committee_index, votes in sorted(entry.committees.items()):
            if len(votes.signatures) > sum(votes.proven_bits):
                bits = votes.raw_bits()
                child = self._attestation(entry.data, [(committee_index, bits)])
                try:
                    votes.proof = prove(
                        child, [votes.signatures[position] for position in sorted(votes.signatures)]
                    )
                    votes.proven_bits = bits
                except Exception as e:
                    logger.warning(f"Cannot prove attestations of committee {committee_index}: {e}")
            if not any(votes.proven_bits):
                continue
            child = self._attestation(entry.data, [(committee_index, votes.proven_bits)])
            if verify is not None and not verify(child, votes.proof):
                votes.proven_bits, votes.proof = (), b""
                continue
            committees.append((committee_index, votes.proven_bits))
            children.append((child, votes.proof))
        if not children:
            return None
        return self._attestation(entry.data, committees), children

    def get_attestations(self, slot: int | None = None, committee_index: int | None = None) -> list:
        result = []
        for entry in self._entries.values():
            if slot is not None and entry.slot != int(slot):
                continue
            for index, votes in sorted(entry.committees.items()):
                if committee_index is not None and index != int(committee_index):
                    continue
                bits = votes.bits()
                if any(bits):
                    result.append(self._attestation(entry.data, [(index, bits)]))
        return result

    def remove_included(self, attestations: list, state) -> int:
        from .spec.state_transition.helpers.attestation import get_attesting_indices

        removed = 0
        for attestation in attestations:
            entry = self._entries.get(bytes(hash_tree_root(attestation.data)))
            if entry is None:
                continue
            try:
                included = get_attesting_indices(state, attestation)
            except Exception:
                continue
            for votes in entry.committees.values():
                for position, validator_index in enumerate(votes.committee):
                    if validator_index in included and votes.signatures.pop(position, None) is not None:
                        removed += 1
                if any(votes.proven_bits) and all(
                    validator_index in included
                    for validator_index, bit in zip(votes.committee, votes.proven_bits)
                    if bit
                ):
                    removed += sum(votes.proven_bits)
                    votes.proven_bits, votes.proof = (), b""
        return removed

    def prune(self, current_slot: int) -> int:
        min_slot = max(0, current_slot - 2 * SLOTS_PER_EPOCH())
        stale = [root for root, entry in self._entries.items() if entry.slot < min_slot]
        for root in stale:
            del self._entries[root]
        return len(stale)

    @property
    def size(self) -> int:
        return sum(
            sum(votes.bits())
            for entry in self._entries.values()
            for votes in entry.committees.values()
        )
