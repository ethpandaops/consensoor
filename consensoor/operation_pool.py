"""Pool of voluntary exits, slashings and BLS-to-execution changes.

Operations are validated by running their state-transition processing on a
copy of the given state; the block builder packs the ones that still apply
to the proposal state.
"""

import logging

from .crypto import hash_tree_root
from .spec.constants import (
    MAX_ATTESTER_SLASHINGS_ELECTRA,
    MAX_ATTESTER_SLASHINGS_PRE_ELECTRA,
    MAX_BLS_TO_EXECUTION_CHANGES,
    MAX_PROPOSER_SLASHINGS,
    MAX_VOLUNTARY_EXITS,
)
from .spec.state_transition.block.operations import (
    process_attester_slashing,
    process_bls_to_execution_change,
    process_proposer_slashing,
    process_voluntary_exit,
)

logger = logging.getLogger(__name__)


class OperationPool:
    """In-memory pool of operations awaiting block inclusion."""

    def __init__(self) -> None:
        self.voluntary_exits: dict[int, object] = {}
        self.proposer_slashings: dict[int, object] = {}
        self.attester_slashings: dict[bytes, object] = {}
        self.bls_to_execution_changes: dict[int, object] = {}

    @staticmethod
    def _check(process, state, op) -> None:
        try:
            process(state.copy(), op)
        except (AssertionError, IndexError) as e:
            raise ValueError(str(e) or f"invalid {type(op).__name__}") from e

    def add_voluntary_exit(self, state, signed_exit) -> bool:
        index = int(signed_exit.message.validator_index)
        if index in self.voluntary_exits:
            return False
        self._check(process_voluntary_exit, state, signed_exit)
        self.voluntary_exits[index] = signed_exit
        return True

    def add_proposer_slashing(self, state, slashing) -> bool:
        index = int(slashing.signed_header_1.message.proposer_index)
        if index in self.proposer_slashings:
            return False
        self._check(process_proposer_slashing, state, slashing)
        self.proposer_slashings[index] = slashing
        return True

    def add_attester_slashing(self, state, slashing) -> bool:
        root = bytes(hash_tree_root(slashing))
        if root in self.attester_slashings:
            return False
        self._check(process_attester_slashing, state, slashing)
        self.attester_slashings[root] = slashing
        return True

    def add_bls_to_execution_change(self, state, signed_change) -> bool:
        index = int(signed_change.message.validator_index)
        if index in self.bls_to_execution_changes:
            return False
        self._check(process_bls_to_execution_change, state, signed_change)
        self.bls_to_execution_changes[index] = signed_change
        return True

    def get_for_block(self, state, attester_slashing_type=None) -> dict[str, list]:
        """Operations that apply in order on top of ``state`` (not mutated)."""
        work = state.copy()
        is_electra = hasattr(state, "pending_deposits")
        groups = (
            ("proposer_slashings", self.proposer_slashings, process_proposer_slashing, MAX_PROPOSER_SLASHINGS),
            (
                "attester_slashings",
                self.attester_slashings,
                process_attester_slashing,
                MAX_ATTESTER_SLASHINGS_ELECTRA if is_electra else MAX_ATTESTER_SLASHINGS_PRE_ELECTRA,
            ),
            ("voluntary_exits", self.voluntary_exits, process_voluntary_exit, MAX_VOLUNTARY_EXITS),
            (
                "bls_to_execution_changes",
                self.bls_to_execution_changes,
                process_bls_to_execution_change,
                MAX_BLS_TO_EXECUTION_CHANGES,
            ),
        )
        result: dict[str, list] = {}
        for name, ops, process, limit in groups:
            picked = []
            for key, op in list(ops.items()):
                if len(picked) >= limit:
                    break
                if name == "attester_slashings" and attester_slashing_type is not None:
                    op = attester_slashing_type.decode_bytes(op.encode_bytes())
                candidate = work.copy()
                try:
                    process(candidate, op)
                except Exception:
                    continue
                work = candidate
                picked.append(op)
            result[name] = picked
        return result

    def remove_included(self, body) -> None:
        """Drop operations included in a block body."""
        for slashing in body.proposer_slashings:
            self.proposer_slashings.pop(int(slashing.signed_header_1.message.proposer_index), None)
        for slashing in body.attester_slashings:
            self.attester_slashings.pop(bytes(hash_tree_root(slashing)), None)
        for signed_exit in body.voluntary_exits:
            self.voluntary_exits.pop(int(signed_exit.message.validator_index), None)
        for signed_change in getattr(body, "bls_to_execution_changes", []):
            self.bls_to_execution_changes.pop(int(signed_change.message.validator_index), None)

    def prune(self, state) -> None:
        """Drop operations that no longer apply to ``state``."""
        for ops, process in (
            (self.voluntary_exits, process_voluntary_exit),
            (self.proposer_slashings, process_proposer_slashing),
            (self.attester_slashings, process_attester_slashing),
            (self.bls_to_execution_changes, process_bls_to_execution_change),
        ):
            for key, op in list(ops.items()):
                try:
                    process(state.copy(), op)
                except Exception:
                    del ops[key]
