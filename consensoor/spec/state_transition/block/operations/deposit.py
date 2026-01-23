"""Deposit processing.

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/phase0/beacon-chain.md
Reference (Electra): https://github.com/ethereum/consensus-specs/blob/master/specs/electra/beacon-chain.md
"""

from typing import TYPE_CHECKING, Optional

from ....constants import (
    FAR_FUTURE_EPOCH,
    DEPOSIT_CONTRACT_TREE_DEPTH,
    MAX_EFFECTIVE_BALANCE,
    EFFECTIVE_BALANCE_INCREMENT,
    DOMAIN_DEPOSIT,
    BLS_WITHDRAWAL_PREFIX,
)
from ....network_config import get_config
from ...helpers.predicates import is_valid_merkle_branch
from ...helpers.mutators import increase_balance
from ...helpers.domain import compute_domain, compute_signing_root
from .....crypto import bls_verify, hash_tree_root

if TYPE_CHECKING:
    from ....types import BeaconState
    from ....types.phase0 import Deposit


def process_deposit(state: "BeaconState", deposit: "Deposit") -> None:
    """Process a deposit.

    Validates the deposit proof and adds the deposit to the validator set
    or increases an existing validator's balance.

    Note: In Electra+, this function queues deposits rather than applying immediately.

    Args:
        state: Beacon state (modified in place)
        deposit: Deposit to process

    Raises:
        AssertionError: If validation fails
    """
    # Verify Merkle proof (deposit index and proof)
    leaf = hash_tree_root(deposit.data)
    assert is_valid_merkle_branch(
        leaf=leaf,
        branch=list(deposit.proof),
        depth=DEPOSIT_CONTRACT_TREE_DEPTH + 1,
        index=int(state.eth1_deposit_index),
        root=bytes(state.eth1_data.deposit_root),
    ), "Invalid deposit proof"

    # Increment deposit index
    state.eth1_deposit_index = int(state.eth1_deposit_index) + 1

    # Apply the deposit (handles both pre-Electra and Electra+)
    apply_deposit(
        state,
        pubkey=bytes(deposit.data.pubkey),
        withdrawal_credentials=bytes(deposit.data.withdrawal_credentials),
        amount=int(deposit.data.amount),
        signature=bytes(deposit.data.signature),
    )


def is_valid_deposit_signature(
    pubkey: bytes,
    withdrawal_credentials: bytes,
    amount: int,
    signature: bytes,
) -> bool:
    """Verify a deposit signature (proof of possession).

    Args:
        pubkey: Validator public key
        withdrawal_credentials: Withdrawal credentials
        amount: Deposit amount in Gwei
        signature: BLS signature

    Returns:
        True if signature is valid
    """
    from ....types.phase0 import DepositMessage

    deposit_message = DepositMessage(
        pubkey=pubkey,
        withdrawal_credentials=withdrawal_credentials,
        amount=amount,
    )

    domain = compute_domain(DOMAIN_DEPOSIT, get_config().genesis_fork_version, b"\x00" * 32)
    signing_root = compute_signing_root(deposit_message, domain)
    return bls_verify([pubkey], signing_root, signature)


def apply_deposit(
    state: "BeaconState",
    pubkey: bytes,
    withdrawal_credentials: bytes,
    amount: int,
    signature: bytes,
) -> None:
    """Apply a deposit to the state.

    Pre-Electra: Creates validator and credits balance immediately.
    Electra+: Creates validator with 0 balance and queues deposit.

    Args:
        state: Beacon state (modified in place)
        pubkey: Validator public key
        withdrawal_credentials: Withdrawal credentials
        amount: Deposit amount in Gwei
        signature: BLS signature (only verified for new validators)
    """
    is_electra = hasattr(state, "pending_deposits")
    validator_pubkeys = [bytes(v.pubkey) for v in state.validators]

    is_new = pubkey not in validator_pubkeys
    print(f"      DEBUG deposit: pubkey={pubkey.hex()[:16]}, is_new={is_new}, validators={len(validator_pubkeys)}")

    if is_new:
        # New validator - verify signature first
        sig_valid = is_valid_deposit_signature(pubkey, withdrawal_credentials, amount, signature)
        print(f"      DEBUG deposit: new validator, sig_valid={sig_valid}")
        if sig_valid:
            if is_electra:
                add_validator_to_registry(state, pubkey, withdrawal_credentials, 0)
            else:
                add_validator_to_registry(state, pubkey, withdrawal_credentials, amount)
        else:
            # Invalid signature for new validator: ignore deposit (all forks)
            return

    if is_electra:
        from ....types.electra import PendingDeposit
        from ....constants import GENESIS_SLOT

        state.pending_deposits.append(
            PendingDeposit(
                pubkey=pubkey,
                withdrawal_credentials=withdrawal_credentials,
                amount=amount,
                signature=signature,
                slot=GENESIS_SLOT,
            )
        )
    elif pubkey in validator_pubkeys:
        index = validator_pubkeys.index(pubkey)
        print(f"      DEBUG deposit: existing validator index={index}, adding {amount}")
        increase_balance(state, index, amount)


def add_validator_to_registry(
    state: "BeaconState",
    pubkey: bytes,
    withdrawal_credentials: bytes,
    amount: int,
) -> None:
    """Add a new validator to the registry.

    Args:
        state: Beacon state (modified in place)
        pubkey: Validator public key
        withdrawal_credentials: Withdrawal credentials
        amount: Initial deposit amount in Gwei
    """
    from ....types.phase0 import Validator
    from ...helpers.accessors import get_current_epoch

    # Create validator
    effective_balance = min(
        amount - (amount % EFFECTIVE_BALANCE_INCREMENT),
        MAX_EFFECTIVE_BALANCE,
    )

    validator = Validator(
        pubkey=pubkey,
        withdrawal_credentials=withdrawal_credentials,
        activation_eligibility_epoch=FAR_FUTURE_EPOCH,
        activation_epoch=FAR_FUTURE_EPOCH,
        exit_epoch=FAR_FUTURE_EPOCH,
        withdrawable_epoch=FAR_FUTURE_EPOCH,
        effective_balance=effective_balance,
        slashed=False,
    )

    # Add to state
    state.validators.append(validator)
    state.balances.append(amount)

    # Add participation flags (Altair+)
    if hasattr(state, "previous_epoch_participation"):
        state.previous_epoch_participation.append(0)
    if hasattr(state, "current_epoch_participation"):
        state.current_epoch_participation.append(0)

    # Add inactivity score (Altair+)
    if hasattr(state, "inactivity_scores"):
        state.inactivity_scores.append(0)
