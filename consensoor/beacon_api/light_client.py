"""Light client data derivation (altair..heze light-client/full-node.md)."""

from typing import Optional

from ..crypto import hash_tree_root
from ..spec.constants import (
    EPOCHS_PER_SYNC_COMMITTEE_PERIOD,
    EXECUTION_BLOCK_HASH_DEPTH_GLOAS,
    MIN_SYNC_COMMITTEE_PARTICIPANTS,
    SLOTS_PER_EPOCH,
)
from ..spec.network_config import get_config
from ..spec.types import altair, capella, deneb, electra, gloas
from ..spec.types.phase0 import BeaconBlockHeader


_FAMILY = {
    "altair": "altair",
    "bellatrix": "altair",
    "capella": "capella",
    "deneb": "deneb",
    "electra": "electra",
    "fulu": "electra",
    "gloas": "gloas",
    "heze": "gloas",
}
_FAMILY_ORDER = ["altair", "capella", "deneb", "electra", "gloas"]

_TYPES = {
    "altair": (
        altair.LightClientHeader, altair.LightClientBootstrap, altair.LightClientUpdate,
        altair.LightClientFinalityUpdate, altair.LightClientOptimisticUpdate,
    ),
    "capella": (
        capella.CapellaLightClientHeader, capella.CapellaLightClientBootstrap, capella.CapellaLightClientUpdate,
        capella.CapellaLightClientFinalityUpdate, capella.CapellaLightClientOptimisticUpdate,
    ),
    "deneb": (
        deneb.DenebLightClientHeader, deneb.DenebLightClientBootstrap, deneb.DenebLightClientUpdate,
        deneb.DenebLightClientFinalityUpdate, deneb.DenebLightClientOptimisticUpdate,
    ),
    "electra": (
        electra.ElectraLightClientHeader, electra.ElectraLightClientBootstrap, electra.ElectraLightClientUpdate,
        electra.ElectraLightClientFinalityUpdate, electra.ElectraLightClientOptimisticUpdate,
    ),
    "gloas": (
        gloas.LightClientHeader, gloas.LightClientBootstrap, gloas.LightClientUpdate,
        gloas.LightClientFinalityUpdate, gloas.LightClientOptimisticUpdate,
    ),
}
_TYPE_NAMES = (
    "LightClientHeader", "LightClientBootstrap", "LightClientUpdate",
    "LightClientFinalityUpdate", "LightClientOptimisticUpdate",
)


def lc_fork_name(slot: int) -> Optional[str]:
    """Fork name of the light client data at ``slot``, or None before Altair."""
    fork = get_config().fork_name_at_slot(int(slot))
    return fork if fork in _FAMILY else None


def light_client_types(fork: str) -> dict:
    """{"LightClientHeader": ..., "LightClientBootstrap": ..., ...} for ``fork``."""
    return dict(zip(_TYPE_NAMES, _TYPES[_FAMILY[fork]]))


def _family_at_slot(slot: int) -> str:
    fork = lc_fork_name(slot)
    if fork is None:
        raise ValueError(f"no light client data before Altair (slot {slot})")
    return _FAMILY[fork]


def get_generalized_index(typ, *path) -> int:
    gindex = 1
    for key in path:
        sub = int(typ.key_to_static_gindex(key))
        depth = sub.bit_length() - 1
        gindex = (gindex << depth) | (sub - (1 << depth))
        typ = typ.navigate_type(key)
    return gindex


def compute_merkle_proof(obj, gindex: int) -> list[bytes]:
    node = obj.get_backing()
    branch = []
    for bit in bin(gindex)[3:]:
        if bit == "1":
            branch.append(node.get_left().merkle_root())
            node = node.get_right()
        else:
            branch.append(node.get_right().merkle_root())
            node = node.get_left()
    return [bytes(b) for b in reversed(branch)]


def normalize_merkle_branch(branch, depth: int) -> list[bytes]:
    branch = [bytes(b) for b in branch]
    return [b"\x00" * 32] * (depth - len(branch)) + branch


def compute_sync_committee_period_at_slot(slot: int) -> int:
    return int(slot) // SLOTS_PER_EPOCH() // EPOCHS_PER_SYNC_COMMITTEE_PERIOD()


def _beacon_header(block) -> BeaconBlockHeader:
    return BeaconBlockHeader(
        slot=block.slot,
        proposer_index=block.proposer_index,
        parent_root=block.parent_root,
        state_root=block.state_root,
        body_root=hash_tree_root(block.body),
    )


def _native_header(signed_block):
    """Header of the block's own light client fork."""
    block = signed_block.message
    family = _family_at_slot(int(block.slot))
    header_type = _TYPES[family][0]
    beacon = _beacon_header(block)
    if family == "altair":
        return header_type(beacon=beacon)
    if family == "gloas":
        gindex = get_generalized_index(
            type(block.body), "signed_execution_payload_bid", "message", "parent_block_hash"
        )
        return header_type(
            beacon=beacon,
            execution_block_hash=block.body.signed_execution_payload_bid.message.parent_block_hash,
            execution_branch=compute_merkle_proof(block.body, gindex),
        )
    payload = block.body.execution_payload
    fields = dict(
        parent_hash=payload.parent_hash,
        fee_recipient=payload.fee_recipient,
        state_root=payload.state_root,
        receipts_root=payload.receipts_root,
        logs_bloom=payload.logs_bloom,
        prev_randao=payload.prev_randao,
        block_number=payload.block_number,
        gas_limit=payload.gas_limit,
        gas_used=payload.gas_used,
        timestamp=payload.timestamp,
        extra_data=payload.extra_data,
        base_fee_per_gas=payload.base_fee_per_gas,
        block_hash=payload.block_hash,
        transactions_root=hash_tree_root(payload.transactions),
        withdrawals_root=hash_tree_root(payload.withdrawals),
    )
    if family == "capella":
        execution = capella.ExecutionPayloadHeaderCapella(**fields)
    else:
        execution = deneb.ExecutionPayloadHeader(
            **fields, blob_gas_used=payload.blob_gas_used, excess_blob_gas=payload.excess_blob_gas
        )
    return header_type(
        beacon=beacon,
        execution=execution,
        execution_branch=compute_merkle_proof(block.body, get_generalized_index(type(block.body), "execution_payload")),
    )


def _upgrade_header(header, family: str):
    """Bring a header forward to ``family`` per the light-client fork.md upgrades."""
    current = next(f for f in _FAMILY_ORDER if isinstance(header, _TYPES[f][0]))
    while current != family:
        nxt = _FAMILY_ORDER[_FAMILY_ORDER.index(current) + 1]
        header_type = _TYPES[nxt][0]
        if nxt == "capella":
            header = header_type(beacon=header.beacon)
        elif nxt == "deneb":
            header = header_type(
                beacon=header.beacon,
                execution=deneb.ExecutionPayloadHeader(
                    **{k: getattr(header.execution, k) for k in capella.ExecutionPayloadHeaderCapella.fields()}
                ),
                execution_branch=header.execution_branch,
            )
        elif nxt == "electra":
            header = header_type(
                beacon=header.beacon, execution=header.execution, execution_branch=header.execution_branch
            )
        else:
            if hash_tree_root(header.execution) == hash_tree_root(type(header.execution)()):
                header = header_type(beacon=header.beacon)
            else:
                execution = header.execution
                if int(header.beacon.slot) // SLOTS_PER_EPOCH() < get_config().deneb_fork_epoch:
                    execution = capella.ExecutionPayloadHeaderCapella(
                        **{k: getattr(execution, k) for k in capella.ExecutionPayloadHeaderCapella.fields()}
                    )
                branch = compute_merkle_proof(execution, get_generalized_index(type(execution), "block_hash"))
                header = header_type(
                    beacon=header.beacon,
                    execution_block_hash=execution.block_hash,
                    execution_branch=normalize_merkle_branch(
                        branch + [bytes(b) for b in header.execution_branch], EXECUTION_BLOCK_HASH_DEPTH_GLOAS
                    ),
                )
        current = nxt
    return header


def block_to_light_client_header(signed_block, fork: Optional[str] = None):
    """LightClientHeader for ``signed_block``, upgraded to ``fork`` (default: the block's fork)."""
    header = _native_header(signed_block)
    if fork is None:
        return header
    return _upgrade_header(header, _FAMILY[fork])


def _post_state_header_root(state) -> bytes:
    assert int(state.slot) == int(state.latest_block_header.slot)
    header = state.latest_block_header.copy()
    header.state_root = hash_tree_root(state)
    return hash_tree_root(header)


def create_light_client_bootstrap(state, signed_block):
    family = _family_at_slot(int(state.slot))
    assert _post_state_header_root(state) == hash_tree_root(signed_block.message)
    bootstrap_type = _TYPES[family][1]
    return bootstrap_type(
        header=block_to_light_client_header(signed_block),
        current_sync_committee=state.current_sync_committee,
        current_sync_committee_branch=compute_merkle_proof(
            state, get_generalized_index(type(state), "current_sync_committee")
        ),
    )


def create_light_client_update(state, signed_block, attested_state, attested_block, finalized_block):
    block = signed_block.message
    fork = lc_fork_name(int(attested_state.slot))
    assert fork is not None
    update_type = _TYPES[_FAMILY[fork]][2]
    assert sum(bool(b) for b in block.body.sync_aggregate.sync_committee_bits) >= MIN_SYNC_COMMITTEE_PARTICIPANTS
    assert _post_state_header_root(state) == hash_tree_root(block)
    update_signature_period = compute_sync_committee_period_at_slot(int(block.slot))
    assert (
        _post_state_header_root(attested_state)
        == hash_tree_root(attested_block.message)
        == bytes(block.parent_root)
    )
    update_attested_period = compute_sync_committee_period_at_slot(int(attested_block.message.slot))

    update = update_type()
    update.attested_header = block_to_light_client_header(attested_block, fork)
    if update_attested_period == update_signature_period:
        update.next_sync_committee = attested_state.next_sync_committee
        update.next_sync_committee_branch = compute_merkle_proof(
            attested_state, get_generalized_index(type(attested_state), "next_sync_committee")
        )
    if finalized_block is not None:
        if int(finalized_block.message.slot) != 0:
            update.finalized_header = block_to_light_client_header(finalized_block, fork)
            assert hash_tree_root(update.finalized_header.beacon) == bytes(attested_state.finalized_checkpoint.root)
        else:
            assert bytes(attested_state.finalized_checkpoint.root) == b"\x00" * 32
        update.finality_branch = compute_merkle_proof(
            attested_state, get_generalized_index(type(attested_state), "finalized_checkpoint", "root")
        )
    update.sync_aggregate = block.body.sync_aggregate
    update.signature_slot = block.slot
    return update


def _update_fork(update) -> str:
    return next(f for f in _FAMILY_ORDER if isinstance(update, (_TYPES[f][2],)))


def create_light_client_finality_update(update):
    return _TYPES[_update_fork(update)][3](
        attested_header=update.attested_header,
        finalized_header=update.finalized_header,
        finality_branch=update.finality_branch,
        sync_aggregate=update.sync_aggregate,
        signature_slot=update.signature_slot,
    )


def create_light_client_optimistic_update(update):
    return _TYPES[_update_fork(update)][4](
        attested_header=update.attested_header,
        sync_aggregate=update.sync_aggregate,
        signature_slot=update.signature_slot,
    )


def _is_zero_branch(branch) -> bool:
    return all(bytes(b) == b"\x00" * 32 for b in branch)


def is_sync_committee_update(update) -> bool:
    return not _is_zero_branch(update.next_sync_committee_branch)


def is_finality_update(update) -> bool:
    return not _is_zero_branch(update.finality_branch)


def is_better_update(new_update, old_update) -> bool:
    max_active_participants = len(new_update.sync_aggregate.sync_committee_bits)
    new_num_active_participants = sum(bool(b) for b in new_update.sync_aggregate.sync_committee_bits)
    old_num_active_participants = sum(bool(b) for b in old_update.sync_aggregate.sync_committee_bits)
    new_has_supermajority = new_num_active_participants * 3 >= max_active_participants * 2
    old_has_supermajority = old_num_active_participants * 3 >= max_active_participants * 2
    if new_has_supermajority != old_has_supermajority:
        return new_has_supermajority
    if not new_has_supermajority and new_num_active_participants != old_num_active_participants:
        return new_num_active_participants > old_num_active_participants

    def has_relevant_sync_committee(update) -> bool:
        return is_sync_committee_update(update) and (
            compute_sync_committee_period_at_slot(int(update.attested_header.beacon.slot))
            == compute_sync_committee_period_at_slot(int(update.signature_slot))
        )

    new_has_relevant_sync_committee = has_relevant_sync_committee(new_update)
    old_has_relevant_sync_committee = has_relevant_sync_committee(old_update)
    if new_has_relevant_sync_committee != old_has_relevant_sync_committee:
        return new_has_relevant_sync_committee

    new_has_finality = is_finality_update(new_update)
    old_has_finality = is_finality_update(old_update)
    if new_has_finality != old_has_finality:
        return new_has_finality

    if new_has_finality:
        def has_sync_committee_finality(update) -> bool:
            return compute_sync_committee_period_at_slot(
                int(update.finalized_header.beacon.slot)
            ) == compute_sync_committee_period_at_slot(int(update.attested_header.beacon.slot))

        new_has_sync_committee_finality = has_sync_committee_finality(new_update)
        old_has_sync_committee_finality = has_sync_committee_finality(old_update)
        if new_has_sync_committee_finality != old_has_sync_committee_finality:
            return new_has_sync_committee_finality

    if new_num_active_participants != old_num_active_participants:
        return new_num_active_participants > old_num_active_participants

    if int(new_update.attested_header.beacon.slot) != int(old_update.attested_header.beacon.slot):
        return int(new_update.attested_header.beacon.slot) < int(old_update.attested_header.beacon.slot)

    return int(new_update.signature_slot) < int(old_update.signature_slot)
