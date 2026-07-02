"""Gloas (ePBS) beacon-API JSON codecs.

JSON <-> SSZ conversion helpers for the Gloas API objects introduced by
beacon-APIs PRs #552/#580/#608, updated for EIP-7688 (progressive
containers) and EIP-8282 (builder execution requests).

Conventions follow the beacon-APIs spec: byte fields are 0x-prefixed hex,
integer fields are decimal strings.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _hex(b) -> str:
    return "0x" + bytes(b).hex()


def _to_bytes(h: str) -> bytes:
    return bytes.fromhex(h[2:] if h.startswith("0x") else h)


# ---------------------------------------------------------------------------
# SSZ -> JSON
# ---------------------------------------------------------------------------

def execution_payload_to_json(payload) -> dict:
    """Serialize a Gloas ExecutionPayload to beacon-API JSON."""
    return {
        "parent_hash": _hex(payload.parent_hash),
        "fee_recipient": _hex(payload.fee_recipient),
        "state_root": _hex(payload.state_root),
        "receipts_root": _hex(payload.receipts_root),
        "logs_bloom": _hex(payload.logs_bloom),
        "prev_randao": _hex(payload.prev_randao),
        "block_number": str(int(payload.block_number)),
        "gas_limit": str(int(payload.gas_limit)),
        "gas_used": str(int(payload.gas_used)),
        "timestamp": str(int(payload.timestamp)),
        "extra_data": _hex(payload.extra_data),
        "base_fee_per_gas": str(int(payload.base_fee_per_gas)),
        "block_hash": _hex(payload.block_hash),
        "transactions": [_hex(tx) for tx in payload.transactions],
        "withdrawals": [
            {
                "index": str(int(w.index)),
                "validator_index": str(int(w.validator_index)),
                "address": _hex(w.address),
                "amount": str(int(w.amount)),
            }
            for w in payload.withdrawals
        ],
        "blob_gas_used": str(int(payload.blob_gas_used)),
        "excess_blob_gas": str(int(payload.excess_blob_gas)),
        "block_access_list": _hex(payload.block_access_list),
        "slot_number": str(int(payload.slot_number)),
    }


def execution_requests_to_json(requests) -> dict:
    """Serialize a Gloas ExecutionRequests (incl. EIP-8282 fields) to JSON."""
    return {
        "deposits": [
            {
                "pubkey": _hex(r.pubkey),
                "withdrawal_credentials": _hex(r.withdrawal_credentials),
                "amount": str(int(r.amount)),
                "signature": _hex(r.signature),
                "index": str(int(r.index)),
            }
            for r in requests.deposits
        ],
        "withdrawals": [
            {
                "source_address": _hex(r.source_address),
                "validator_pubkey": _hex(r.validator_pubkey),
                "amount": str(int(r.amount)),
            }
            for r in requests.withdrawals
        ],
        "consolidations": [
            {
                "source_address": _hex(r.source_address),
                "source_pubkey": _hex(r.source_pubkey),
                "target_pubkey": _hex(r.target_pubkey),
            }
            for r in requests.consolidations
        ],
        "builder_deposits": [
            {
                "pubkey": _hex(r.pubkey),
                "withdrawal_credentials": _hex(r.withdrawal_credentials),
                "amount": str(int(r.amount)),
                "signature": _hex(r.signature),
            }
            for r in requests.builder_deposits
        ],
        "builder_exits": [
            {
                "source_address": _hex(r.source_address),
                "pubkey": _hex(r.pubkey),
            }
            for r in requests.builder_exits
        ],
    }


def signed_envelope_to_json(signed_envelope) -> dict:
    """Serialize a SignedExecutionPayloadEnvelope to beacon-API JSON."""
    envelope = signed_envelope.message
    return {
        "message": {
            "payload": execution_payload_to_json(envelope.payload),
            "execution_requests": execution_requests_to_json(envelope.execution_requests),
            "builder_index": str(int(envelope.builder_index)),
            "beacon_block_root": _hex(envelope.beacon_block_root),
            "parent_beacon_block_root": _hex(envelope.parent_beacon_block_root),
        },
        "signature": _hex(signed_envelope.signature),
    }


def signed_bid_to_json(signed_bid) -> dict:
    """Serialize a SignedExecutionPayloadBid to beacon-API JSON."""
    bid = signed_bid.message
    return {
        "message": {
            "parent_block_hash": _hex(bid.parent_block_hash),
            "parent_block_root": _hex(bid.parent_block_root),
            "block_hash": _hex(bid.block_hash),
            "prev_randao": _hex(bid.prev_randao),
            "fee_recipient": _hex(bid.fee_recipient),
            "gas_limit": str(int(bid.gas_limit)),
            "builder_index": str(int(bid.builder_index)),
            "slot": str(int(bid.slot)),
            "value": str(int(bid.value)),
            "execution_payment": str(int(bid.execution_payment)),
            "blob_kzg_commitments": [_hex(c) for c in bid.blob_kzg_commitments],
            "execution_requests_root": _hex(bid.execution_requests_root),
        },
        "signature": _hex(signed_bid.signature),
    }


def signed_proposer_preferences_to_json(signed) -> dict:
    """Serialize a SignedProposerPreferences to beacon-API JSON."""
    prefs = signed.message
    return {
        "message": {
            "dependent_root": _hex(prefs.dependent_root),
            "proposal_slot": str(int(prefs.proposal_slot)),
            "validator_index": str(int(prefs.validator_index)),
            "fee_recipient": _hex(prefs.fee_recipient),
            "target_gas_limit": str(int(prefs.target_gas_limit)),
        },
        "signature": _hex(signed.signature),
    }


# ---------------------------------------------------------------------------
# JSON -> SSZ
# ---------------------------------------------------------------------------

def json_to_execution_payload(d: dict):
    """Parse beacon-API JSON into a Gloas ExecutionPayload."""
    from ..spec.types.gloas import ExecutionPayload, BlockAccessList
    from ..spec.types.capella import Withdrawal

    return ExecutionPayload(
        parent_hash=_to_bytes(d["parent_hash"]),
        fee_recipient=_to_bytes(d["fee_recipient"]),
        state_root=_to_bytes(d["state_root"]),
        receipts_root=_to_bytes(d["receipts_root"]),
        logs_bloom=_to_bytes(d["logs_bloom"]),
        prev_randao=_to_bytes(d["prev_randao"]),
        block_number=int(d["block_number"]),
        gas_limit=int(d["gas_limit"]),
        gas_used=int(d["gas_used"]),
        timestamp=int(d["timestamp"]),
        extra_data=list(_to_bytes(d.get("extra_data", "0x"))),
        base_fee_per_gas=int(d["base_fee_per_gas"]),
        block_hash=_to_bytes(d["block_hash"]),
        transactions=[_to_bytes(tx) for tx in d.get("transactions", [])],
        withdrawals=[
            Withdrawal(
                index=int(w["index"]),
                validator_index=int(w["validator_index"]),
                address=_to_bytes(w["address"]),
                amount=int(w["amount"]),
            )
            for w in d.get("withdrawals", [])
        ],
        blob_gas_used=int(d.get("blob_gas_used", 0)),
        excess_blob_gas=int(d.get("excess_blob_gas", 0)),
        block_access_list=BlockAccessList(_to_bytes(d.get("block_access_list", "0x"))),
        slot_number=int(d.get("slot_number", 0)),
    )


def json_to_execution_requests(d: dict):
    """Parse beacon-API JSON into a Gloas ExecutionRequests."""
    from ..spec.types.gloas import (
        ExecutionRequests,
        BuilderDepositRequest,
        BuilderExitRequest,
    )
    from ..spec.types.electra import (
        DepositRequest,
        WithdrawalRequest,
        ConsolidationRequest,
    )

    return ExecutionRequests(
        deposits=[
            DepositRequest(
                pubkey=_to_bytes(r["pubkey"]),
                withdrawal_credentials=_to_bytes(r["withdrawal_credentials"]),
                amount=int(r["amount"]),
                signature=_to_bytes(r["signature"]),
                index=int(r["index"]),
            )
            for r in d.get("deposits", [])
        ],
        withdrawals=[
            WithdrawalRequest(
                source_address=_to_bytes(r["source_address"]),
                validator_pubkey=_to_bytes(r["validator_pubkey"]),
                amount=int(r["amount"]),
            )
            for r in d.get("withdrawals", [])
        ],
        consolidations=[
            ConsolidationRequest(
                source_address=_to_bytes(r["source_address"]),
                source_pubkey=_to_bytes(r["source_pubkey"]),
                target_pubkey=_to_bytes(r["target_pubkey"]),
            )
            for r in d.get("consolidations", [])
        ],
        builder_deposits=[
            BuilderDepositRequest(
                pubkey=_to_bytes(r["pubkey"]),
                withdrawal_credentials=_to_bytes(r["withdrawal_credentials"]),
                amount=int(r["amount"]),
                signature=_to_bytes(r["signature"]),
            )
            for r in d.get("builder_deposits", [])
        ],
        builder_exits=[
            BuilderExitRequest(
                source_address=_to_bytes(r["source_address"]),
                pubkey=_to_bytes(r["pubkey"]),
            )
            for r in d.get("builder_exits", [])
        ],
    )


def json_to_signed_envelope(d: dict):
    """Parse beacon-API JSON into a SignedExecutionPayloadEnvelope."""
    from ..spec.types.gloas import (
        ExecutionPayloadEnvelope,
        SignedExecutionPayloadEnvelope,
    )

    m = d["message"]
    envelope = ExecutionPayloadEnvelope(
        payload=json_to_execution_payload(m["payload"]),
        execution_requests=json_to_execution_requests(m.get("execution_requests", {})),
        builder_index=int(m["builder_index"]),
        beacon_block_root=_to_bytes(m["beacon_block_root"]),
        parent_beacon_block_root=_to_bytes(m["parent_beacon_block_root"]),
    )
    return SignedExecutionPayloadEnvelope(
        message=envelope,
        signature=_to_bytes(d["signature"]),
    )


def json_to_signed_bid(d: dict):
    """Parse beacon-API JSON into a SignedExecutionPayloadBid."""
    from ..spec.types.gloas import ExecutionPayloadBid, SignedExecutionPayloadBid

    m = d["message"]
    bid = ExecutionPayloadBid(
        parent_block_hash=_to_bytes(m["parent_block_hash"]),
        parent_block_root=_to_bytes(m["parent_block_root"]),
        block_hash=_to_bytes(m["block_hash"]),
        prev_randao=_to_bytes(m["prev_randao"]),
        fee_recipient=_to_bytes(m["fee_recipient"]),
        gas_limit=int(m["gas_limit"]),
        builder_index=int(m["builder_index"]),
        slot=int(m["slot"]),
        value=int(m["value"]),
        execution_payment=int(m.get("execution_payment", 0)),
        blob_kzg_commitments=[_to_bytes(c) for c in m.get("blob_kzg_commitments", [])],
        execution_requests_root=_to_bytes(m["execution_requests_root"]),
    )
    return SignedExecutionPayloadBid(
        message=bid,
        signature=_to_bytes(d["signature"]),
    )


def json_to_signed_proposer_preferences(d: dict):
    """Parse beacon-API JSON into a SignedProposerPreferences."""
    from ..spec.types.gloas import ProposerPreferences, SignedProposerPreferences

    m = d["message"]
    prefs = ProposerPreferences(
        dependent_root=_to_bytes(m["dependent_root"]),
        proposal_slot=int(m["proposal_slot"]),
        validator_index=int(m["validator_index"]),
        fee_recipient=_to_bytes(m["fee_recipient"]),
        target_gas_limit=int(m["target_gas_limit"]),
    )
    return SignedProposerPreferences(
        message=prefs,
        signature=_to_bytes(d["signature"]),
    )
