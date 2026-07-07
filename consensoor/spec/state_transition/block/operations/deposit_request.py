"""Deposit request processing (Electra+).

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/electra/beacon-chain.md
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ....types import BeaconState
    from ....types.electra import DepositRequest


def _is_gloas_state(state) -> bool:
    """Check if state is a gloas (ePBS) state.

    Uses try/except because remerkleable containers may raise exceptions
    for unknown attributes instead of returning AttributeError.
    """
    try:
        _ = state.builders
        return True
    except Exception:
        return False


def process_deposit_request(
    state: "BeaconState", deposit_request: "DepositRequest"
) -> None:
    """Process a deposit request from the execution layer (Electra+).

    Queues the deposit for processing via pending_deposits.

    Args:
        state: Beacon state (modified in place)
        deposit_request: Deposit request from execution layer
    """
    from ....types.electra import PendingDeposit
    from ....constants import UNSET_DEPOSIT_REQUESTS_START_INDEX

    # Gloas: builder onboarding moved to BuilderDepositRequest (EIP-8282);
    # all deposit requests are queued as pending deposits, without setting
    # deposit_requests_start_index.
    if _is_gloas_state(state):
        state.pending_deposits.append(
            PendingDeposit(
                pubkey=deposit_request.pubkey,
                withdrawal_credentials=deposit_request.withdrawal_credentials,
                amount=int(deposit_request.amount),
                signature=deposit_request.signature,
                slot=int(state.slot),
            )
        )
        return

    # Set deposit request start index on first deposit request (Electra/Fulu only)
    if int(state.deposit_requests_start_index) == UNSET_DEPOSIT_REQUESTS_START_INDEX:
        state.deposit_requests_start_index = int(deposit_request.index)

    state.pending_deposits.append(
        PendingDeposit(
            pubkey=deposit_request.pubkey,
            withdrawal_credentials=deposit_request.withdrawal_credentials,
            amount=int(deposit_request.amount),
            signature=deposit_request.signature,
            slot=int(state.slot),
        )
    )
