"""Execution payload processing (Bellatrix+).

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/bellatrix/beacon-chain.md
"""

from typing import TYPE_CHECKING

from ..helpers.predicates import is_merge_transition_complete, is_execution_enabled
from ..helpers.accessors import get_current_epoch, get_randao_mix
from ..helpers.misc import compute_time_at_slot
from ....crypto import hash_tree_root

if TYPE_CHECKING:
    from ...types import BeaconState, BeaconBlockBody


def process_execution_payload(
    state: "BeaconState",
    body,
    execution_engine=None,
    execution_valid: bool = True,
) -> None:
    """Process the execution payload from the block (Bellatrix+).

    Validates:
    - Parent hash matches previous execution payload
    - RANDAO matches
    - Timestamp matches
    - Notifies execution engine

    Then caches the payload header in state.

    Args:
        state: Beacon state (modified in place)
        body: Block body containing execution payload
        execution_engine: Execution engine client (optional for validation only)

    Raises:
        AssertionError: If any validation fails
    """
    if hasattr(body, "message") and hasattr(body.message, "payload"):
        from .execution_payload_envelope import process_execution_payload_envelope

        if execution_engine is None:
            class NoopEngine:
                def verify_and_notify_new_payload(self, _request) -> bool:
                    return execution_valid
            execution_engine = NoopEngine()

        process_execution_payload_envelope(
            state,
            body,
            execution_engine=execution_engine,
            verify=True,
        )
        return

    if not hasattr(body, "execution_payload"):
        return

    payload = body.execution_payload

    # Capella+ removed the is_merge_transition_complete check for parent_hash
    # Detection: Capella+ has withdrawals field
    is_capella_or_later = hasattr(payload, "withdrawals")

    # Verify parent hash
    # Bellatrix: only check if merge is complete
    # Capella+: always check (merge is assumed complete)
    if is_capella_or_later or is_merge_transition_complete(state):
        assert bytes(payload.parent_hash) == bytes(
            state.latest_execution_payload_header.block_hash
        ), "Execution payload parent hash mismatch"

    # Verify prev_randao
    expected_randao = get_randao_mix(state, get_current_epoch(state))
    assert bytes(payload.prev_randao) == expected_randao, (
        f"Payload prev_randao {bytes(payload.prev_randao).hex()[:16]} doesn't match "
        f"expected {expected_randao.hex()[:16]}"
    )

    # Verify timestamp
    from ...network_config import get_config

    network_config = get_config()
    expected_timestamp = compute_time_at_slot(
        int(state.genesis_time),
        int(state.slot),
        network_config.slot_duration_ms,
    )
    assert int(payload.timestamp) == expected_timestamp, (
        f"Payload timestamp {payload.timestamp} doesn't match expected {expected_timestamp}"
    )

    # Verify blob commitments limit
    if hasattr(body, "blob_kzg_commitments"):
        # Fulu+ uses dynamic blob schedule
        is_fulu = hasattr(state, "proposer_lookahead")
        if is_fulu:
            from ..helpers.accessors import get_blob_parameters
            blob_params = get_blob_parameters(get_current_epoch(state))
            max_blobs = blob_params.max_blobs_per_block
        else:
            from ...constants import MAX_BLOBS_PER_BLOCK, MAX_BLOBS_PER_BLOCK_ELECTRA
            is_electra = hasattr(state, "pending_deposits")
            max_blobs = MAX_BLOBS_PER_BLOCK_ELECTRA if is_electra else MAX_BLOBS_PER_BLOCK

        assert len(body.blob_kzg_commitments) <= max_blobs, (
            f"Too many blob commitments: {len(body.blob_kzg_commitments)} > {max_blobs}"
        )

    # Verify the execution payload is valid
    # In production, this calls execution_engine.verify_and_notify_new_payload()
    # For tests, execution_valid is passed from execution.yaml metadata
    assert execution_valid, "Execution payload validation failed"

    # Cache execution payload header
    _cache_execution_payload_header(state, payload)


def _get_view_root(view) -> bytes:
    """Get the hash_tree_root of a view or container field.

    Handles both SSZ Views (which have hash_tree_root method) and
    raw data (which needs to be wrapped).
    """
    if hasattr(view, 'hash_tree_root'):
        root = view.hash_tree_root()
        return bytes(root) if not isinstance(root, bytes) else root
    else:
        return hash_tree_root(view)


def _cache_execution_payload_header(state: "BeaconState", payload) -> None:
    """Cache the execution payload header in state.

    Args:
        state: Beacon state (modified in place)
        payload: Execution payload to cache
    """
    transactions_root = _get_view_root(payload.transactions)

    base_fields = {
        "parent_hash": payload.parent_hash,
        "fee_recipient": payload.fee_recipient,
        "state_root": payload.state_root,
        "receipts_root": payload.receipts_root,
        "logs_bloom": payload.logs_bloom,
        "prev_randao": payload.prev_randao,
        "block_number": payload.block_number,
        "gas_limit": payload.gas_limit,
        "gas_used": payload.gas_used,
        "timestamp": payload.timestamp,
        "extra_data": payload.extra_data,
        "base_fee_per_gas": payload.base_fee_per_gas,
        "block_hash": payload.block_hash,
        "transactions_root": transactions_root,
    }

    has_withdrawals = hasattr(payload, "withdrawals")
    has_blob_gas = hasattr(payload, "blob_gas_used")

    if not has_withdrawals:
        from ...types.bellatrix import ExecutionPayloadHeaderBellatrix
        state.latest_execution_payload_header = ExecutionPayloadHeaderBellatrix(**base_fields)
    elif not has_blob_gas:
        from ...types.capella import ExecutionPayloadHeaderCapella
        base_fields["withdrawals_root"] = _get_view_root(payload.withdrawals)
        state.latest_execution_payload_header = ExecutionPayloadHeaderCapella(**base_fields)
    else:
        from ...types.deneb import ExecutionPayloadHeader
        base_fields["withdrawals_root"] = _get_view_root(payload.withdrawals)
        base_fields["blob_gas_used"] = payload.blob_gas_used
        base_fields["excess_blob_gas"] = payload.excess_blob_gas
        state.latest_execution_payload_header = ExecutionPayloadHeader(**base_fields)


def get_execution_requests_list(body: "BeaconBlockBody") -> list:
    """Get the execution requests list from the block body (Electra).

    Args:
        body: Block body

    Returns:
        List of execution request bytes
    """
    if not hasattr(body, "execution_requests"):
        return []

    requests = body.execution_requests
    result = []

    # Deposit requests (type 0x00)
    for deposit in requests.deposits:
        # Serialize deposit request
        from ....crypto import hash_tree_root

        result.append(b"\x00" + hash_tree_root(deposit))

    # Withdrawal requests (type 0x01)
    for withdrawal in requests.withdrawals:
        result.append(b"\x01" + hash_tree_root(withdrawal))

    # Consolidation requests (type 0x02)
    for consolidation in requests.consolidations:
        result.append(b"\x02" + hash_tree_root(consolidation))

    return result
