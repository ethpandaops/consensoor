"""Withdrawals processing (Capella+).

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/capella/beacon-chain.md
Reference (Electra): https://github.com/ethereum/consensus-specs/blob/master/specs/electra/beacon-chain.md
"""

from typing import TYPE_CHECKING, NamedTuple, List, Tuple, Sequence

from ...constants import (
    MAX_VALIDATORS_PER_WITHDRAWALS_SWEEP,
    MAX_WITHDRAWALS_PER_PAYLOAD,
    MAX_PENDING_PARTIALS_PER_WITHDRAWALS_SWEEP,
    MAX_BUILDERS_PER_WITHDRAWALS_SWEEP,
    FAR_FUTURE_EPOCH,
    MIN_ACTIVATION_BALANCE,
    MAX_EFFECTIVE_BALANCE,
)
from ..helpers.accessors import get_current_epoch
from ..helpers.predicates import (
    is_fully_withdrawable_validator,
    is_partially_withdrawable_validator,
    has_execution_withdrawal_credential,
    has_eth1_withdrawal_credential,
)
from ..helpers.mutators import decrease_balance

if TYPE_CHECKING:
    from ...types import BeaconState, Validator
    from ...types.capella import Withdrawal


class ExpectedWithdrawals(NamedTuple):
    """Expected withdrawals result."""

    withdrawals: List["Withdrawal"]
    processed_builder_withdrawals_count: int
    processed_partial_withdrawals_count: int
    processed_builders_sweep_count: int
    processed_validators_sweep_count: int


def is_eligible_for_partial_withdrawals(validator: "Validator", balance: int) -> bool:
    """Check if validator can process a pending partial withdrawal (Electra).

    Args:
        validator: Validator
        balance: Current balance after prior withdrawals

    Returns:
        True if validator is eligible for partial withdrawal
    """
    has_sufficient_effective_balance = int(validator.effective_balance) >= MIN_ACTIVATION_BALANCE
    has_excess_balance = balance > MIN_ACTIVATION_BALANCE
    return (
        int(validator.exit_epoch) == FAR_FUTURE_EPOCH
        and has_sufficient_effective_balance
        and has_excess_balance
    )


def get_balance_after_withdrawals(
    state: "BeaconState",
    validator_index: int,
    withdrawals: Sequence["Withdrawal"],
) -> int:
    """Get validator's balance after applying prior withdrawals.

    Args:
        state: Beacon state
        validator_index: Validator index
        withdrawals: Prior withdrawals to subtract

    Returns:
        Balance after withdrawals
    """
    balance = int(state.balances[validator_index])
    for w in withdrawals:
        if int(w.validator_index) == validator_index:
            balance -= int(w.amount)
    return balance


def get_pending_partial_withdrawals(
    state: "BeaconState",
    withdrawal_index: int,
    prior_withdrawals: List["Withdrawal"],
) -> Tuple[List["Withdrawal"], int, int]:
    """Get pending partial withdrawals (Electra).

    Args:
        state: Beacon state
        withdrawal_index: Starting withdrawal index
        prior_withdrawals: Prior withdrawals already processed

    Returns:
        Tuple of (withdrawals, new withdrawal_index, processed_count)
    """
    from ...types.capella import Withdrawal

    epoch = get_current_epoch(state)
    processed_count = 0
    withdrawals: List[Withdrawal] = []

    for partial in state.pending_partial_withdrawals:
        is_withdrawable = int(partial.withdrawable_epoch) <= epoch
        has_reached_limit = len(withdrawals) == MAX_PENDING_PARTIALS_PER_WITHDRAWALS_SWEEP()

        if not is_withdrawable or has_reached_limit:
            break

        validator_index = int(partial.validator_index)
        validator = state.validators[validator_index]
        # Get balance after accounting for prior withdrawals in this batch
        all_withdrawals = prior_withdrawals + withdrawals
        balance = get_balance_after_withdrawals(state, validator_index, all_withdrawals)

        has_sufficient_effective_balance = int(validator.effective_balance) >= MIN_ACTIVATION_BALANCE
        has_excess_balance = balance > MIN_ACTIVATION_BALANCE

        if (
            int(validator.exit_epoch) == FAR_FUTURE_EPOCH
            and has_sufficient_effective_balance
            and has_excess_balance
        ):
            withdrawal_amount = min(balance - MIN_ACTIVATION_BALANCE, int(partial.amount))
            withdrawals.append(
                Withdrawal(
                    index=withdrawal_index,
                    validator_index=validator_index,
                    address=bytes(validator.withdrawal_credentials)[12:],
                    amount=withdrawal_amount,
                )
            )
            withdrawal_index += 1

        processed_count += 1

    return withdrawals, withdrawal_index, processed_count


def get_builder_withdrawals(
    state: "BeaconState",
    withdrawal_index: int,
    prior_withdrawals: List["Withdrawal"],
) -> Tuple[List["Withdrawal"], int, int]:
    """Get builder pending withdrawals (Gloas)."""
    from ...types.capella import Withdrawal
    from ..helpers.misc import convert_builder_index_to_validator_index

    withdrawals_limit = MAX_WITHDRAWALS_PER_PAYLOAD() - 1
    processed_count = 0
    withdrawals: List[Withdrawal] = []

    for withdrawal in state.builder_pending_withdrawals:
        all_withdrawals = prior_withdrawals + withdrawals
        if len(all_withdrawals) >= withdrawals_limit:
            break

        builder_index = int(withdrawal.builder_index)
        withdrawals.append(
            Withdrawal(
                index=withdrawal_index,
                validator_index=convert_builder_index_to_validator_index(builder_index),
                address=withdrawal.fee_recipient,
                amount=int(withdrawal.amount),
            )
        )
        withdrawal_index += 1
        processed_count += 1

    return withdrawals, withdrawal_index, processed_count


def get_builders_sweep_withdrawals(
    state: "BeaconState",
    withdrawal_index: int,
    prior_withdrawals: List["Withdrawal"],
) -> Tuple[List["Withdrawal"], int, int]:
    """Get builder sweep withdrawals (Gloas)."""
    from ...types.capella import Withdrawal
    from ..helpers.misc import convert_builder_index_to_validator_index

    epoch = get_current_epoch(state)
    builders_limit = min(len(state.builders), MAX_BUILDERS_PER_WITHDRAWALS_SWEEP)
    withdrawals_limit = MAX_WITHDRAWALS_PER_PAYLOAD() - 1
    processed_count = 0
    withdrawals: List[Withdrawal] = []

    if len(state.builders) == 0:
        return withdrawals, withdrawal_index, processed_count

    builder_index = int(state.next_withdrawal_builder_index)
    for _ in range(builders_limit):
        all_withdrawals = prior_withdrawals + withdrawals
        if len(all_withdrawals) >= withdrawals_limit:
            break

        builder = state.builders[builder_index]
        if int(builder.withdrawable_epoch) <= epoch and int(builder.balance) > 0:
            withdrawals.append(
                Withdrawal(
                    index=withdrawal_index,
                    validator_index=convert_builder_index_to_validator_index(builder_index),
                    address=builder.execution_address,
                    amount=int(builder.balance),
                )
            )
            withdrawal_index += 1

        builder_index = (builder_index + 1) % len(state.builders)
        processed_count += 1

    return withdrawals, withdrawal_index, processed_count


def get_validators_sweep_withdrawals(
    state: "BeaconState",
    withdrawal_index: int,
    prior_withdrawals: List["Withdrawal"],
) -> Tuple[List["Withdrawal"], int, int]:
    """Get validators sweep withdrawals (Electra).

    Args:
        state: Beacon state
        withdrawal_index: Starting withdrawal index
        prior_withdrawals: Prior withdrawals already processed

    Returns:
        Tuple of (withdrawals, new withdrawal_index, processed_count)
    """
    from ...types.capella import Withdrawal
    from ..helpers.accessors import get_max_effective_balance

    epoch = get_current_epoch(state)
    validators_limit = min(len(state.validators), MAX_VALIDATORS_PER_WITHDRAWALS_SWEEP())
    withdrawals_limit = MAX_WITHDRAWALS_PER_PAYLOAD()

    processed_count = 0
    withdrawals: List[Withdrawal] = []
    validator_index = int(state.next_withdrawal_validator_index)

    while processed_count < validators_limit and len(prior_withdrawals) + len(withdrawals) < withdrawals_limit:
        validator = state.validators[validator_index]
        all_withdrawals = prior_withdrawals + withdrawals
        balance = get_balance_after_withdrawals(state, validator_index, all_withdrawals)

        if is_fully_withdrawable_validator(validator, balance, epoch):
            withdrawals.append(
                Withdrawal(
                    index=withdrawal_index,
                    validator_index=validator_index,
                    address=bytes(validator.withdrawal_credentials)[12:],
                    amount=balance,
                )
            )
            withdrawal_index += 1
        elif is_partially_withdrawable_validator(validator, balance):
            max_effective = get_max_effective_balance(validator)
            excess = balance - max_effective
            withdrawals.append(
                Withdrawal(
                    index=withdrawal_index,
                    validator_index=validator_index,
                    address=bytes(validator.withdrawal_credentials)[12:],
                    amount=excess,
                )
            )
            withdrawal_index += 1

        validator_index = (validator_index + 1) % len(state.validators)
        processed_count += 1

    return withdrawals, withdrawal_index, processed_count


def get_expected_withdrawals(state: "BeaconState") -> ExpectedWithdrawals:
    """Get the expected withdrawals for the current state.

    Args:
        state: Beacon state

    Returns:
        ExpectedWithdrawals with list of withdrawals and count of partial withdrawals
    """
    from ...types.capella import Withdrawal

    withdrawal_index = int(state.next_withdrawal_index)
    withdrawals: List[Withdrawal] = []

    is_gloas = hasattr(state, "builder_pending_withdrawals")
    is_electra = hasattr(state, "pending_partial_withdrawals")

    if is_gloas:
        builder_withdrawals, withdrawal_index, processed_builder_count = (
            get_builder_withdrawals(state, withdrawal_index, withdrawals)
        )
        withdrawals.extend(builder_withdrawals)

        partial_withdrawals, withdrawal_index, processed_partial_count = (
            get_pending_partial_withdrawals(state, withdrawal_index, withdrawals)
        )
        withdrawals.extend(partial_withdrawals)

        builders_sweep_withdrawals, withdrawal_index, processed_builders_sweep_count = (
            get_builders_sweep_withdrawals(state, withdrawal_index, withdrawals)
        )
        withdrawals.extend(builders_sweep_withdrawals)

        validators_sweep_withdrawals, withdrawal_index, processed_validators_sweep_count = (
            get_validators_sweep_withdrawals(state, withdrawal_index, withdrawals)
        )
        withdrawals.extend(validators_sweep_withdrawals)

        return ExpectedWithdrawals(
            withdrawals,
            processed_builder_count,
            processed_partial_count,
            processed_builders_sweep_count,
            processed_validators_sweep_count,
        )

    if is_electra:
        partial_withdrawals, withdrawal_index, processed_partial_count = (
            get_pending_partial_withdrawals(state, withdrawal_index, withdrawals)
        )
        withdrawals.extend(partial_withdrawals)

        sweep_withdrawals, withdrawal_index, processed_validators_sweep_count = (
            get_validators_sweep_withdrawals(state, withdrawal_index, withdrawals)
        )
        withdrawals.extend(sweep_withdrawals)

        return ExpectedWithdrawals(
            withdrawals,
            0,
            processed_partial_count,
            0,
            processed_validators_sweep_count,
        )

    # Pre-Electra: just sweep validators
    validator_index = int(state.next_withdrawal_validator_index)
    validators_checked = 0
    bound = min(len(state.validators), MAX_VALIDATORS_PER_WITHDRAWALS_SWEEP())
    epoch = get_current_epoch(state)

    while len(withdrawals) < MAX_WITHDRAWALS_PER_PAYLOAD() and validators_checked < bound:
        validator = state.validators[validator_index]
        balance = int(state.balances[validator_index])

        if (
            has_eth1_withdrawal_credential(validator)
            and int(validator.withdrawable_epoch) <= epoch
            and balance > 0
        ):
            withdrawals.append(
                Withdrawal(
                    index=withdrawal_index,
                    validator_index=validator_index,
                    address=bytes(validator.withdrawal_credentials)[12:],
                    amount=balance,
                )
            )
            withdrawal_index += 1
        elif (
            has_eth1_withdrawal_credential(validator)
            and int(validator.effective_balance) == MAX_EFFECTIVE_BALANCE
            and balance > MAX_EFFECTIVE_BALANCE
        ):
            excess = balance - MAX_EFFECTIVE_BALANCE
            withdrawals.append(
                Withdrawal(
                    index=withdrawal_index,
                    validator_index=validator_index,
                    address=bytes(validator.withdrawal_credentials)[12:],
                    amount=excess,
                )
            )
            withdrawal_index += 1

        validator_index = (validator_index + 1) % len(state.validators)
        validators_checked += 1

    return ExpectedWithdrawals(withdrawals, 0, 0, 0, validators_checked)


def update_payload_expected_withdrawals(
    state: "BeaconState", withdrawals: Sequence["Withdrawal"]
) -> None:
    """Update payload_expected_withdrawals (Gloas)."""
    state.payload_expected_withdrawals = state.payload_expected_withdrawals.__class__(
        withdrawals
    )


def update_builder_pending_withdrawals(
    state: "BeaconState", processed_builder_withdrawals_count: int
) -> None:
    """Drop processed builder pending withdrawals (Gloas)."""
    state.builder_pending_withdrawals = list(
        state.builder_pending_withdrawals[processed_builder_withdrawals_count:]
    )


def update_next_withdrawal_builder_index(
    state: "BeaconState", processed_builders_sweep_count: int
) -> None:
    """Update next_withdrawal_builder_index after builder sweep (Gloas)."""
    if len(state.builders) > 0:
        next_index = int(state.next_withdrawal_builder_index) + processed_builders_sweep_count
        state.next_withdrawal_builder_index = next_index % len(state.builders)


def process_withdrawals(state: "BeaconState", payload=None) -> None:
    """Process withdrawals from the execution payload (Capella+).

    Validates that payload withdrawals match expected withdrawals and applies them.

    Args:
        state: Beacon state (modified in place)
        payload: Execution payload with withdrawals

    Raises:
        AssertionError: If withdrawals don't match expected
    """
    expected = get_expected_withdrawals(state)

    is_gloas = hasattr(state, "payload_expected_withdrawals")
    if not is_gloas:
        assert payload is not None, "Execution payload required"

        assert len(payload.withdrawals) == len(expected.withdrawals), (
            f"Withdrawal count mismatch: got {len(payload.withdrawals)}, "
            f"expected {len(expected.withdrawals)}"
        )

        for i, (actual, expected_w) in enumerate(
            zip(payload.withdrawals, expected.withdrawals)
        ):
            assert int(actual.index) == int(expected_w.index), (
                f"Withdrawal {i} index mismatch"
            )
            assert int(actual.validator_index) == int(expected_w.validator_index), (
                f"Withdrawal {i} validator_index mismatch"
            )
            assert bytes(actual.address) == bytes(expected_w.address), (
                f"Withdrawal {i} address mismatch"
            )
            assert int(actual.amount) == int(expected_w.amount), (
                f"Withdrawal {i} amount mismatch"
            )

    from ..helpers.predicates import is_builder_index
    from ..helpers.misc import convert_validator_index_to_builder_index

    for withdrawal in expected.withdrawals:
        if is_builder_index(int(withdrawal.validator_index)):
            builder_index = convert_validator_index_to_builder_index(withdrawal.validator_index)
            builder_balance = int(state.builders[builder_index].balance)
            state.builders[builder_index].balance = builder_balance - min(
                builder_balance, int(withdrawal.amount)
            )
        else:
            decrease_balance(
                state, int(withdrawal.validator_index), int(withdrawal.amount)
            )

    if len(expected.withdrawals) > 0:
        last_withdrawal = expected.withdrawals[-1]
        state.next_withdrawal_index = int(last_withdrawal.index) + 1

    if len(expected.withdrawals) == MAX_WITHDRAWALS_PER_PAYLOAD():
        next_index = (
            int(expected.withdrawals[-1].validator_index) + 1
        ) % len(state.validators)
        state.next_withdrawal_validator_index = next_index
    elif len(state.validators) > 0:
        state.next_withdrawal_validator_index = (
            int(state.next_withdrawal_validator_index) + MAX_VALIDATORS_PER_WITHDRAWALS_SWEEP()
        ) % len(state.validators)

    if hasattr(state, "pending_partial_withdrawals") and expected.processed_partial_withdrawals_count > 0:
        state.pending_partial_withdrawals = list(
            state.pending_partial_withdrawals[expected.processed_partial_withdrawals_count:]
        )

    if is_gloas:
        update_payload_expected_withdrawals(state, expected.withdrawals)
        update_builder_pending_withdrawals(state, expected.processed_builder_withdrawals_count)
        update_next_withdrawal_builder_index(state, expected.processed_builders_sweep_count)


