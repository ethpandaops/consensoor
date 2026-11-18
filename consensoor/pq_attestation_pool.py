"""Attestation pool for PQ: per-validator leanXMSS signatures, grouped by committee."""

import logging
from dataclasses import dataclass, field

from .crypto import hash_tree_root
from .spec.constants import (
    MAX_ATTESTATIONS_PQ,
    MIN_ATTESTATION_INCLUSION_DELAY_PQ,
    SLOTS_PER_EPOCH,
    XMSS_SIGNATURE_SIZE,
)
from .spec.types.pq import Attestation, CommitteeBits

logger = logging.getLogger(__name__)


@dataclass
class CommitteeVotes:
    committee: list[int]
    signatures: dict[int, bytes] = field(default_factory=dict)


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

    def add_aggregate(self, attestation, proof: bytes, committee: list[int]) -> int:
        committee_index = self.committee_of(attestation)
        if committee_index < 0:
            return 0
        bits = [bool(b) for b in attestation.aggregation_bits]
        if len(bits) != len(committee):
            return 0
        positions = [i for i, bit in enumerate(bits) if bit]
        proof = bytes(proof)
        if len(proof) != len(positions) * XMSS_SIGNATURE_SIZE:
            return 0
        votes = self._votes(attestation.data, committee_index, committee)
        added = 0
        for n, position in enumerate(positions):
            if position not in votes.signatures:
                votes.signatures[position] = proof[n * XMSS_SIGNATURE_SIZE:(n + 1) * XMSS_SIGNATURE_SIZE]
                added += 1
        return added

    @staticmethod
    def committee_of(attestation) -> int:
        bits = [i for i, b in enumerate(attestation.committee_bits) if b]
        return bits[0] if len(bits) == 1 else -1

    @staticmethod
    def _build(data, committees: list[tuple[int, CommitteeVotes]]):
        aggregation_bits: list[bool] = []
        committee_bits = CommitteeBits()
        proof = b""
        for committee_index, votes in committees:
            committee_bits[committee_index] = True
            for position in range(len(votes.committee)):
                signature = votes.signatures.get(position)
                aggregation_bits.append(signature is not None)
                if signature is not None:
                    proof += signature
        attestation = Attestation(
            aggregation_bits=aggregation_bits,
            data=data,
            committee_bits=committee_bits,
        )
        return attestation, proof

    def get_aggregate(self, slot: int, data_root: bytes, committee_index: int):
        entry = self._entries.get(bytes(data_root))
        if entry is None or entry.slot != int(slot):
            return None
        votes = entry.committees.get(int(committee_index))
        if votes is None or not votes.signatures:
            return None
        return self._build(entry.data, [(int(committee_index), votes)])

    def get_attestations_for_block(self, current_slot: int) -> list[tuple[Attestation, bytes]]:
        spe = SLOTS_PER_EPOCH()
        min_slot = max(0, current_slot - 2 * spe)
        result = []
        for entry in self._entries.values():
            if entry.slot + MIN_ATTESTATION_INCLUSION_DELAY_PQ > current_slot or entry.slot < min_slot:
                continue
            committees = [
                (index, votes)
                for index, votes in sorted(entry.committees.items())
                if votes.signatures
            ]
            if committees:
                result.append(self._build(entry.data, committees))
        result.sort(key=lambda pair: int(pair[0].data.slot), reverse=True)
        return result[: 4 * MAX_ATTESTATIONS_PQ]

    def get_attestations(self, slot: int | None = None, committee_index: int | None = None) -> list:
        result = []
        for entry in self._entries.values():
            if slot is not None and entry.slot != int(slot):
                continue
            for index, votes in sorted(entry.committees.items()):
                if committee_index is not None and index != int(committee_index):
                    continue
                if votes.signatures:
                    result.append(self._build(entry.data, [(index, votes)])[0])
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
            len(votes.signatures)
            for entry in self._entries.values()
            for votes in entry.committees.values()
        )
