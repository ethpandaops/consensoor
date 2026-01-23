"""State mutator helper functions for state transition.

Implements functions to modify state values.
Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/phase0/beacon-chain.md
"""

from typing import TYPE_CHECKING, Optional

from ...constants import (
    FAR_FUTURE_EPOCH,
    MIN_SLASHING_PENALTY_QUOTIENT,
    MIN_SLASHING_PENALTY_QUOTIENT_ALTAIR,
    MIN_SLASHING_PENALTY_QUOTIENT_BELLATRIX,
    MIN_SLASHING_PENALTY_QUOTIENT_ELECTRA,
    WHISTLEBLOWER_REWARD_QUOTIENT,
    WHISTLEBLOWER_REWARD_QUOTIENT_ELECTRA,
    PROPOSER_REWARD_QUOTIENT,
    PROPOSER_WEIGHT,
    WEIGHT_DENOMINATOR,
    EFFECTIVE_BALANCE_INCREMENT,
    MAX_EFFECTIVE_BALANCE,
    MAX_EFFECTIVE_BALANCE_ELECTRA,
    MIN_VALIDATOR_WITHDRAWABILITY_DELAY,
    EPOCHS_PER_SLASHINGS_VECTOR,
)
from .misc import compute_activation_exit_epoch, compute_epoch_at_slot
from .accessors import (
    get_current_epoch,
    get_validator_churn_limit,
    get_activation_exit_churn_limit,
    get_consolidation_churn_limit,
)
from .beacon_committee import get_beacon_proposer_index
from .predicates import has_compounding_withdrawal_credential

if TYPE_CHECKING:
    from ...types import BeaconState, PendingPartialWithdrawal


def increase_balance(state: "BeaconState", index: int, delta: int) -> None:
    """Increase the balance of a validator.

    Args:
        state: Beacon state (modified in place)
        index: Validator index
        delta: Amount to increase in Gwei
    """
    state.balances[index] = int(state.balances[index]) + delta


def decrease_balance(state: "BeaconState", index: int, delta: int) -> None:
    """Decrease the balance of a validator (saturates at 0).

    Args:
        state: Beacon state (modified in place)
        index: Validator index
        delta: Amount to decrease in Gwei
    """
    balance = int(state.balances[index])
    state.balances[index] = 0 if delta > balance else balance - delta


def initiate_validator_exit(state: "BeaconState", index: int) -> None:
    """Initiate the exit of a validator.

    Sets the exit_epoch and withdrawable_epoch for the validator,
    respecting the churn limit.

    Args:
        state: Beacon state (modified in place)
        index: Validator index
    """
    validator = state.validators[index]

    # Return if validator already initiated exit
    if int(validator.exit_epoch) != FAR_FUTURE_EPOCH:
        return

    # Electra+ uses balance-based churn tracking
    is_electra_or_later = hasattr(state, "exit_balance_to_consume")
    if is_electra_or_later:
        exit_queue_epoch = compute_exit_epoch_and_update_churn(
            state, int(validator.effective_balance)
        )
    else:
        # Pre-Electra: count-based exit queue
        current_epoch = get_current_epoch(state)
        exit_epochs = [
            int(v.exit_epoch)
            for v in state.validators
            if int(v.exit_epoch) != FAR_FUTURE_EPOCH
        ]
        exit_queue_epoch = max(
            exit_epochs + [compute_activation_exit_epoch(current_epoch)]
        )
        exit_queue_churn = len(
            [v for v in state.validators if int(v.exit_epoch) == exit_queue_epoch]
        )
        if exit_queue_churn >= get_validator_churn_limit(state):
            exit_queue_epoch += 1

    # Set validator exit epoch and withdrawable epoch
    validator.exit_epoch = exit_queue_epoch
    validator.withdrawable_epoch = (
        exit_queue_epoch + MIN_VALIDATOR_WITHDRAWABILITY_DELAY
    )


def get_min_slashing_penalty_quotient(state: "BeaconState") -> int:
    """Get the minimum slashing penalty quotient for the current fork.

    Returns:
        - Phase0: MIN_SLASHING_PENALTY_QUOTIENT (64)
        - Altair: MIN_SLASHING_PENALTY_QUOTIENT_ALTAIR (64)
        - Bellatrix+Capella+Deneb: MIN_SLASHING_PENALTY_QUOTIENT_BELLATRIX (32)
        - Electra+: MIN_SLASHING_PENALTY_QUOTIENT_ELECTRA (4096)
    """
    if hasattr(state, "pending_deposits"):
        return MIN_SLASHING_PENALTY_QUOTIENT_ELECTRA
    if hasattr(state, "latest_execution_payload_header"):
        return MIN_SLASHING_PENALTY_QUOTIENT_BELLATRIX
    if hasattr(state, "previous_epoch_participation"):
        return MIN_SLASHING_PENALTY_QUOTIENT_ALTAIR
    return MIN_SLASHING_PENALTY_QUOTIENT


def get_whistleblower_reward_quotient(state: "BeaconState") -> int:
    """Get the whistleblower reward quotient for the current fork.

    Returns:
        - Pre-Electra: WHISTLEBLOWER_REWARD_QUOTIENT (512)
        - Electra+: WHISTLEBLOWER_REWARD_QUOTIENT_ELECTRA (4096)
    """
    if hasattr(state, "pending_deposits"):
        return WHISTLEBLOWER_REWARD_QUOTIENT_ELECTRA
    return WHISTLEBLOWER_REWARD_QUOTIENT


def slash_validator(
    state: "BeaconState",
    slashed_index: int,
    whistleblower_index: Optional[int] = None,
) -> None:
    """Slash a validator.

    Marks the validator as slashed, initiates exit, applies penalties,
    and distributes rewards.

    Args:
        state: Beacon state (modified in place)
        slashed_index: Index of validator being slashed
        whistleblower_index: Index of whistleblower (defaults to proposer)
    """
    from .beacon_committee import get_beacon_proposer_index

    epoch = get_current_epoch(state)
    initiate_validator_exit(state, slashed_index)

    validator = state.validators[slashed_index]
    validator.slashed = True
    validator.withdrawable_epoch = max(
        int(validator.withdrawable_epoch),
        epoch + EPOCHS_PER_SLASHINGS_VECTOR(),
    )

    state.slashings[epoch % EPOCHS_PER_SLASHINGS_VECTOR()] = (
        int(state.slashings[epoch % EPOCHS_PER_SLASHINGS_VECTOR()])
        + int(validator.effective_balance)
    )

    # Slash penalty (fork-aware)
    slashing_penalty = (
        int(validator.effective_balance) // get_min_slashing_penalty_quotient(state)
    )
    decrease_balance(state, slashed_index, slashing_penalty)

    # Whistleblower and proposer rewards
    proposer_index = get_beacon_proposer_index(state)
    if whistleblower_index is None:
        whistleblower_index = proposer_index

    whistleblower_reward = (
        int(validator.effective_balance) // get_whistleblower_reward_quotient(state)
    )

    # Phase0: proposer gets 1/PROPOSER_REWARD_QUOTIENT of whistleblower reward
    # Altair+: proposer gets PROPOSER_WEIGHT/WEIGHT_DENOMINATOR of whistleblower reward
    if hasattr(state, "previous_epoch_participation"):
        proposer_reward = whistleblower_reward * PROPOSER_WEIGHT // WEIGHT_DENOMINATOR
    else:
        proposer_reward = whistleblower_reward // PROPOSER_REWARD_QUOTIENT

    increase_balance(state, proposer_index, proposer_reward)
    increase_balance(state, whistleblower_index, whistleblower_reward - proposer_reward)


# Electra mutators


def switch_to_compounding_validator(state: "BeaconState", index: int) -> None:
    """Switch a validator to compounding withdrawal credentials (Electra).

    Updates the withdrawal credentials prefix from 0x01 to 0x02,
    and queues any excess balance.

    Args:
        state: Beacon state (modified in place)
        index: Validator index
    """
    from ...constants import COMPOUNDING_WITHDRAWAL_PREFIX

    validator = state.validators[index]
    # Change first byte from 0x01 to 0x02, keep the rest (execution address)
    old_credentials = bytes(validator.withdrawal_credentials)
    new_credentials = bytes([COMPOUNDING_WITHDRAWAL_PREFIX]) + old_credentials[1:]
    validator.withdrawal_credentials = new_credentials
    queue_excess_active_balance(state, index)


def queue_excess_active_balance(state: "BeaconState", index: int) -> None:
    """Queue excess balance above MIN_ACTIVATION_BALANCE as pending deposit (Electra).

    Transfers excess balance into the pending deposits queue.

    Args:
        state: Beacon state (modified in place)
        index: Validator index
    """
    from ...types.electra import PendingDeposit
    from ...constants import MIN_ACTIVATION_BALANCE, GENESIS_SLOT

    balance = int(state.balances[index])
    if balance > MIN_ACTIVATION_BALANCE:
        excess_balance = balance - MIN_ACTIVATION_BALANCE
        state.balances[index] = MIN_ACTIVATION_BALANCE
        validator = state.validators[index]
        # Use G2 point at infinity as signature placeholder
        # and GENESIS_SLOT to distinguish from a pending deposit request
        g2_point_at_infinity = b"\xc0" + b"\x00" * 95
        state.pending_deposits.append(
            PendingDeposit(
                pubkey=validator.pubkey,
                withdrawal_credentials=validator.withdrawal_credentials,
                amount=excess_balance,
                signature=g2_point_at_infinity,
                slot=GENESIS_SLOT,
            )
        )


def compute_exit_epoch_and_update_churn(state: "BeaconState", exit_balance: int) -> int:
    """Compute exit epoch and update churn tracker (Electra).

    Args:
        state: Beacon state (modified in place)
        exit_balance: Balance being exited in Gwei

    Returns:
        Exit epoch
    """
    from .misc import compute_activation_exit_epoch

    earliest_exit_epoch = max(
        int(state.earliest_exit_epoch),
        compute_activation_exit_epoch(get_current_epoch(state)),
    )
    per_epoch_churn = get_activation_exit_churn_limit(state)

    # New epoch for exits - reset churn
    if int(state.earliest_exit_epoch) < earliest_exit_epoch:
        exit_balance_to_consume = per_epoch_churn
    else:
        exit_balance_to_consume = int(state.exit_balance_to_consume)

    # Exit doesn't fit in the current earliest epoch
    if exit_balance > exit_balance_to_consume:
        balance_to_process = exit_balance - exit_balance_to_consume
        additional_epochs = (balance_to_process - 1) // per_epoch_churn + 1
        earliest_exit_epoch += additional_epochs
        exit_balance_to_consume += additional_epochs * per_epoch_churn

    # Consume the balance and update state variables
    state.exit_balance_to_consume = exit_balance_to_consume - exit_balance
    state.earliest_exit_epoch = earliest_exit_epoch

    return earliest_exit_epoch


def compute_consolidation_epoch_and_update_churn(
    state: "BeaconState", consolidation_balance: int
) -> int:
    """Compute consolidation epoch and update churn tracker (Electra).

    Args:
        state: Beacon state (modified in place)
        consolidation_balance: Balance being consolidated in Gwei

    Returns:
        Consolidation epoch
    """
    from .misc import compute_activation_exit_epoch

    earliest_consolidation_epoch = max(
        int(state.earliest_consolidation_epoch),
        compute_activation_exit_epoch(get_current_epoch(state)),
    )
    per_epoch_churn = get_consolidation_churn_limit(state)

    # New epoch for consolidations - reset churn
    if int(state.earliest_consolidation_epoch) < earliest_consolidation_epoch:
        consolidation_balance_to_consume = per_epoch_churn
    else:
        consolidation_balance_to_consume = int(state.consolidation_balance_to_consume)

    # Consolidation doesn't fit in the current earliest epoch
    if consolidation_balance > consolidation_balance_to_consume:
        balance_to_process = consolidation_balance - consolidation_balance_to_consume
        additional_epochs = (balance_to_process - 1) // per_epoch_churn + 1
        earliest_consolidation_epoch += additional_epochs
        consolidation_balance_to_consume += additional_epochs * per_epoch_churn

    # Consume the balance and update state variables
    state.consolidation_balance_to_consume = (
        consolidation_balance_to_consume - consolidation_balance
    )
    state.earliest_consolidation_epoch = earliest_consolidation_epoch

    return int(state.earliest_consolidation_epoch)


def get_beacon_proposer_index(state: "BeaconState") -> int:
    """Get the beacon proposer index for the current slot.

    For Fulu+, uses the proposer_lookahead vector.
    For earlier forks, computes on demand.

    Args:
        state: Beacon state

    Returns:
        Proposer validator index
    """
    # Check if we have Fulu proposer_lookahead
    if hasattr(state, "proposer_lookahead") and len(state.proposer_lookahead) > 0:
        from ...constants import SLOTS_PER_EPOCH, MIN_SEED_LOOKAHEAD

        slot = int(state.slot)
        epoch = compute_epoch_at_slot(slot)
        # Calculate index into lookahead vector
        # Lookahead covers (MIN_SEED_LOOKAHEAD + 1) epochs
        lookahead_epoch_start = epoch  # Current epoch is at start of lookahead
        slot_in_epoch = slot % SLOTS_PER_EPOCH()
        epoch_offset = epoch - lookahead_epoch_start
        lookahead_index = epoch_offset * SLOTS_PER_EPOCH() + slot_in_epoch
        if 0 <= lookahead_index < len(state.proposer_lookahead):
            return int(state.proposer_lookahead[lookahead_index])

    # Fall back to computing on demand
    from .beacon_committee import _compute_proposer_index_on_demand

    return _compute_proposer_index_on_demand(state)
