"""Predicate helper functions for state transition.

Implements validator state checks and attestation validation.
Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/phase0/beacon-chain.md
"""

from typing import TYPE_CHECKING, Sequence

from ...constants import (
    FAR_FUTURE_EPOCH,
    BLS_WITHDRAWAL_PREFIX,
    ETH1_ADDRESS_WITHDRAWAL_PREFIX,
    COMPOUNDING_WITHDRAWAL_PREFIX,
    MAX_EFFECTIVE_BALANCE,
    MAX_EFFECTIVE_BALANCE_ELECTRA,
    EFFECTIVE_BALANCE_INCREMENT,
    BUILDER_INDEX_FLAG,
    BUILDER_WITHDRAWAL_PREFIX,
    DOMAIN_PTC_ATTESTER,
)
from ....crypto import hash_tree_root, bls_verify, sha256

if TYPE_CHECKING:
    from ...types import (
        Validator,
        BeaconState,
        AttestationData,
        IndexedAttestation,
        BeaconBlockBody,
    )


def is_active_validator(validator: "Validator", epoch: int) -> bool:
    """Check if validator is active at the given epoch.

    A validator is active if activation_epoch <= epoch < exit_epoch.

    Args:
        validator: Validator to check
        epoch: Epoch to check at

    Returns:
        True if validator is active at epoch
    """
    return int(validator.activation_epoch) <= epoch < int(validator.exit_epoch)


def is_builder_index(validator_index: int) -> bool:
    """Check if validator_index is a builder index (flagged)."""
    return (int(validator_index) & BUILDER_INDEX_FLAG) != 0


def is_active_builder(state: "BeaconState", builder_index: int) -> bool:
    """Check if the builder at builder_index is active."""
    builder = state.builders[builder_index]
    return (
        int(builder.deposit_epoch) < int(state.finalized_checkpoint.epoch)
        and int(builder.withdrawable_epoch) == FAR_FUTURE_EPOCH
    )


def is_builder_withdrawal_credential(withdrawal_credentials: bytes) -> bool:
    """Check if withdrawal_credentials is a builder credential."""
    return withdrawal_credentials[:1] == BUILDER_WITHDRAWAL_PREFIX


def is_eligible_for_activation_queue(validator: "Validator") -> bool:
    """Check if validator is eligible to join the activation queue.

    A validator is eligible if not yet activated and has sufficient effective balance.

    Args:
        validator: Validator to check

    Returns:
        True if eligible for activation queue
    """
    return (
        int(validator.activation_eligibility_epoch) == FAR_FUTURE_EPOCH
        and int(validator.effective_balance) == MAX_EFFECTIVE_BALANCE
    )


def is_eligible_for_activation(state: "BeaconState", validator: "Validator") -> bool:
    """Check if validator is eligible for activation.

    A validator is eligible if in the activation queue and the queue epoch
    has been finalized.

    Args:
        state: Beacon state
        validator: Validator to check

    Returns:
        True if eligible for activation
    """
    return (
        int(validator.activation_eligibility_epoch)
        <= int(state.finalized_checkpoint.epoch)
        and int(validator.activation_epoch) == FAR_FUTURE_EPOCH
    )


def is_slashable_validator(validator: "Validator", epoch: int) -> bool:
    """Check if validator is slashable at the given epoch.

    A validator is slashable if not already slashed, is active, and
    within the withdrawable epoch window.

    Args:
        validator: Validator to check
        epoch: Current epoch

    Returns:
        True if validator is slashable
    """
    return (
        not validator.slashed
        and int(validator.activation_epoch) <= epoch
        and epoch < int(validator.withdrawable_epoch)
    )


def is_slashable_attestation_data(
    data_1: "AttestationData", data_2: "AttestationData"
) -> bool:
    """Check if attestation data is slashable (double vote or surround vote).

    Args:
        data_1: First attestation data
        data_2: Second attestation data

    Returns:
        True if the attestations are slashable
    """
    # Double vote: same target epoch but different data
    double_vote = (
        int(data_1.target.epoch) == int(data_2.target.epoch)
        and data_1 != data_2
    )
    # Surround vote: data_1 surrounds data_2
    surround_vote = (
        int(data_1.source.epoch) < int(data_2.source.epoch)
        and int(data_2.target.epoch) < int(data_1.target.epoch)
    )
    return double_vote or surround_vote


def is_valid_indexed_attestation(
    state: "BeaconState", indexed_attestation: "IndexedAttestation"
) -> bool:
    """Check if an indexed attestation is valid.

    Validates:
    - Indices are sorted and unique
    - At least one index
    - Indices are within validator set
    - Aggregate signature is valid

    Args:
        state: Beacon state
        indexed_attestation: Indexed attestation to validate

    Returns:
        True if indexed attestation is valid
    """
    indices = list(indexed_attestation.attesting_indices)

    # Verify indices are sorted and unique
    if len(indices) == 0:
        return False
    if indices != sorted(set(indices)):
        return False

    # Verify all indices are valid validator indices
    if any(i >= len(state.validators) for i in indices):
        return False

    # Verify aggregate signature
    from .domain import get_domain, compute_signing_root
    from ...constants import DOMAIN_BEACON_ATTESTER
    from .misc import compute_epoch_at_slot

    pubkeys = [bytes(state.validators[i].pubkey) for i in indices]
    domain = get_domain(
        state,
        DOMAIN_BEACON_ATTESTER,
        int(indexed_attestation.data.target.epoch),
    )
    signing_root = compute_signing_root(indexed_attestation.data, domain)

    return bls_verify(pubkeys, signing_root, bytes(indexed_attestation.signature))


def is_valid_indexed_payload_attestation(
    state: "BeaconState", indexed_attestation
) -> bool:
    """Check if an indexed payload attestation is valid."""
    indices = list(indexed_attestation.attesting_indices)
    if len(indices) == 0 or indices != sorted(indices):
        return False

    if any(i >= len(state.validators) for i in indices):
        return False

    from .domain import get_domain, compute_signing_root
    from .misc import compute_epoch_at_slot

    pubkeys = [bytes(state.validators[i].pubkey) for i in indices]
    domain = get_domain(
        state,
        DOMAIN_PTC_ATTESTER,
        compute_epoch_at_slot(int(indexed_attestation.data.slot)),
    )
    signing_root = compute_signing_root(indexed_attestation.data, domain)

    return bls_verify(pubkeys, signing_root, bytes(indexed_attestation.signature))


def is_valid_merkle_branch(
    leaf: bytes,
    branch: Sequence[bytes],
    depth: int,
    index: int,
    root: bytes,
) -> bool:
    """Check if a Merkle branch is valid.

    Args:
        leaf: Leaf value
        branch: Merkle proof branch
        depth: Depth of the tree
        index: Index of the leaf
        root: Expected root

    Returns:
        True if the Merkle branch is valid
    """
    value = leaf
    for i in range(depth):
        if index // (2**i) % 2:
            value = sha256(bytes(branch[i]) + value)
        else:
            value = sha256(value + bytes(branch[i]))
    return value == bytes(root)


# Bellatrix predicates


def is_merge_transition_complete(state: "BeaconState") -> bool:
    """Check if the merge transition is complete.

    The merge is complete when the latest execution payload header is not empty.

    Args:
        state: Beacon state

    Returns:
        True if merge transition is complete
    """
    if not hasattr(state, "latest_execution_payload_header"):
        return False
    header = state.latest_execution_payload_header
    # Check if parent_hash is non-zero (empty header has all zeros)
    return bytes(header.parent_hash) != b"\x00" * 32


def is_execution_enabled(state: "BeaconState", body: "BeaconBlockBody") -> bool:
    """Check if execution is enabled for this block.

    Execution is enabled if merge is complete or this is a merge transition block.

    Args:
        state: Beacon state
        body: Block body

    Returns:
        True if execution is enabled
    """
    if is_merge_transition_complete(state):
        return True
    # Check if this is a merge transition block (first non-empty payload)
    if hasattr(body, "execution_payload"):
        payload = body.execution_payload
        return bytes(payload.parent_hash) != b"\x00" * 32
    return False


# Capella predicates


def has_eth1_withdrawal_credential(validator: "Validator") -> bool:
    """Check if validator has ETH1 withdrawal credentials (0x01 prefix).

    Args:
        validator: Validator to check

    Returns:
        True if has ETH1 withdrawal credentials
    """
    return bytes(validator.withdrawal_credentials)[0] == ETH1_ADDRESS_WITHDRAWAL_PREFIX


def is_fully_withdrawable_validator(
    validator: "Validator", balance: int, epoch: int
) -> bool:
    """Check if validator is fully withdrawable.

    A validator is fully withdrawable if they have ETH1 withdrawal credentials,
    have reached their withdrawable epoch, and have a positive balance.

    Args:
        validator: Validator to check
        balance: Validator's balance
        epoch: Current epoch

    Returns:
        True if fully withdrawable
    """
    return (
        has_execution_withdrawal_credential(validator)
        and int(validator.withdrawable_epoch) <= epoch
        and balance > 0
    )


def is_partially_withdrawable_validator(validator: "Validator", balance: int) -> bool:
    """Check if validator is partially withdrawable.

    A validator is partially withdrawable if they have ETH1 withdrawal credentials,
    have max effective balance, and have excess balance.

    Args:
        validator: Validator to check
        balance: Validator's balance

    Returns:
        True if partially withdrawable
    """
    max_effective = get_max_effective_balance_for_validator(validator)
    has_max_effective_balance = int(validator.effective_balance) == max_effective
    has_excess_balance = balance > max_effective
    return (
        has_execution_withdrawal_credential(validator)
        and has_max_effective_balance
        and has_excess_balance
    )


# Electra predicates


def is_compounding_withdrawal_credential(withdrawal_credentials: bytes) -> bool:
    """Check if withdrawal credentials are compounding (0x02 prefix).

    Args:
        withdrawal_credentials: 32-byte withdrawal credentials

    Returns:
        True if compounding withdrawal credentials
    """
    return bytes(withdrawal_credentials)[0] == COMPOUNDING_WITHDRAWAL_PREFIX


def has_compounding_withdrawal_credential(validator: "Validator") -> bool:
    """Check if validator has compounding withdrawal credentials.

    Args:
        validator: Validator to check

    Returns:
        True if has compounding withdrawal credentials
    """
    return is_compounding_withdrawal_credential(validator.withdrawal_credentials)


def has_execution_withdrawal_credential(validator: "Validator") -> bool:
    """Check if validator has execution withdrawal credentials (0x01 or 0x02 prefix).

    Args:
        validator: Validator to check

    Returns:
        True if has execution withdrawal credentials
    """
    return (
        has_eth1_withdrawal_credential(validator)
        or has_compounding_withdrawal_credential(validator)
    )


def get_max_effective_balance_for_validator(validator: "Validator") -> int:
    """Get the maximum effective balance for a validator.

    Compounding validators can have up to MAX_EFFECTIVE_BALANCE_ELECTRA,
    others have the standard MAX_EFFECTIVE_BALANCE.

    Args:
        validator: Validator to check

    Returns:
        Maximum effective balance in Gwei
    """
    if has_compounding_withdrawal_credential(validator):
        return MAX_EFFECTIVE_BALANCE_ELECTRA
    return MAX_EFFECTIVE_BALANCE


def is_valid_switch_to_compounding_request(
    state: "BeaconState", consolidation_request
) -> bool:
    """Check if a consolidation request is a valid switch-to-compounding request.

    A valid switch-to-compounding request has source == target pubkey,
    valid credentials, and an active validator.

    Args:
        state: Beacon state
        consolidation_request: Consolidation request to check

    Returns:
        True if this is a valid switch-to-compounding request
    """
    from .accessors import get_current_epoch
    from ...constants import FAR_FUTURE_EPOCH

    # Switch to compounding requires source and target be equal
    if bytes(consolidation_request.source_pubkey) != bytes(consolidation_request.target_pubkey):
        return False

    # Verify pubkey exists
    source_pubkey = bytes(consolidation_request.source_pubkey)
    validator_pubkeys = [bytes(v.pubkey) for v in state.validators]
    if source_pubkey not in validator_pubkeys:
        return False

    source_index = validator_pubkeys.index(source_pubkey)
    source_validator = state.validators[source_index]

    # Verify request has been authorized
    if bytes(source_validator.withdrawal_credentials)[12:] != bytes(consolidation_request.source_address):
        return False

    # Verify source withdrawal credentials (must be ETH1, not already compounding)
    if not has_eth1_withdrawal_credential(source_validator):
        return False

    # Verify the source is active
    current_epoch = get_current_epoch(state)
    if not is_active_validator(source_validator, current_epoch):
        return False

    # Verify exit for source has not been initiated
    if int(source_validator.exit_epoch) != FAR_FUTURE_EPOCH:
        return False

    return True
