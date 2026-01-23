"""Block operations processing functions."""

from .proposer_slashing import process_proposer_slashing
from .attester_slashing import process_attester_slashing
from .attestation import process_attestation
from .deposit import process_deposit, apply_deposit, add_validator_to_registry
from .voluntary_exit import process_voluntary_exit
from .bls_change import process_bls_to_execution_change
from .deposit_request import process_deposit_request
from .withdrawal_request import process_withdrawal_request
from .consolidation_request import process_consolidation_request
from .gloas import process_execution_payload_bid, process_payload_attestation

__all__ = [
    "process_proposer_slashing",
    "process_attester_slashing",
    "process_attestation",
    "process_deposit",
    "apply_deposit",
    "add_validator_to_registry",
    "process_voluntary_exit",
    "process_bls_to_execution_change",
    "process_deposit_request",
    "process_withdrawal_request",
    "process_consolidation_request",
    "process_execution_payload_bid",
    "process_payload_attestation",
]
