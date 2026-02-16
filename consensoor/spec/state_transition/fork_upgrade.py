"""Fork upgrade functions for state transitions between consensus layer forks.

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/capella/fork.md
Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/deneb/fork.md
Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/electra/fork.md

Each upgrade function converts a pre-fork state to a post-fork state,
adding new fields with appropriate default values.
"""

import logging
from typing import TYPE_CHECKING

from ..types import Fork
from ..types.base import Version, Epoch, Gwei, ValidatorIndex, uint64, Bytes32, Hash32, Root, Bitvector
from ..types.bellatrix import BellatrixBeaconState, ExecutionPayloadHeaderBellatrix
from ..types.capella import (
    CapellaBeaconState,
    ExecutionPayloadHeaderCapella,
    HistoricalSummary,
    Withdrawal,
)
from ..types.deneb import DenebBeaconState, ExecutionPayloadHeader
from ..types.electra import ElectraBeaconState
from ..types.fulu import FuluBeaconState, proposer_lookahead_length
from ..types.gloas import (
    BeaconState as GloasBeaconState,
    ExecutionPayloadBid,
    BuilderPendingPayment,
    BuilderPendingWithdrawal,
)
from ..constants import SLOTS_PER_EPOCH, SLOTS_PER_HISTORICAL_ROOT, FAR_FUTURE_EPOCH, MIN_SEED_LOOKAHEAD, MAX_WITHDRAWALS_PER_PAYLOAD
from ..network_config import get_config
from .helpers.accessors import get_activation_exit_churn_limit, get_current_epoch
from .helpers.beacon_committee import get_beacon_proposer_indices

if TYPE_CHECKING:
    from ..types import BeaconState

logger = logging.getLogger(__name__)


def upgrade_to_capella(pre: BellatrixBeaconState, fork_version: bytes, epoch: int) -> CapellaBeaconState:
    """Upgrade a Bellatrix state to Capella.

    Adds withdrawal tracking fields and historical summaries.
    """
    pre_header = pre.latest_execution_payload_header

    post_header = ExecutionPayloadHeaderCapella(
        parent_hash=pre_header.parent_hash,
        fee_recipient=pre_header.fee_recipient,
        state_root=pre_header.state_root,
        receipts_root=pre_header.receipts_root,
        logs_bloom=pre_header.logs_bloom,
        prev_randao=pre_header.prev_randao,
        block_number=pre_header.block_number,
        gas_limit=pre_header.gas_limit,
        gas_used=pre_header.gas_used,
        timestamp=pre_header.timestamp,
        extra_data=list(pre_header.extra_data),
        base_fee_per_gas=pre_header.base_fee_per_gas,
        block_hash=pre_header.block_hash,
        transactions_root=pre_header.transactions_root,
        withdrawals_root=Bytes32(b"\x00" * 32),
    )

    post = CapellaBeaconState(
        genesis_time=pre.genesis_time,
        genesis_validators_root=pre.genesis_validators_root,
        slot=pre.slot,
        fork=Fork(
            previous_version=Version(bytes(pre.fork.current_version)),
            current_version=Version(fork_version),
            epoch=Epoch(epoch),
        ),
        latest_block_header=pre.latest_block_header,
        block_roots=pre.block_roots,
        state_roots=pre.state_roots,
        historical_roots=list(pre.historical_roots),
        eth1_data=pre.eth1_data,
        eth1_data_votes=list(pre.eth1_data_votes),
        eth1_deposit_index=pre.eth1_deposit_index,
        validators=list(pre.validators),
        balances=list(pre.balances),
        randao_mixes=pre.randao_mixes,
        slashings=pre.slashings,
        previous_epoch_participation=list(pre.previous_epoch_participation),
        current_epoch_participation=list(pre.current_epoch_participation),
        justification_bits=pre.justification_bits,
        previous_justified_checkpoint=pre.previous_justified_checkpoint,
        current_justified_checkpoint=pre.current_justified_checkpoint,
        finalized_checkpoint=pre.finalized_checkpoint,
        inactivity_scores=list(pre.inactivity_scores),
        current_sync_committee=pre.current_sync_committee,
        next_sync_committee=pre.next_sync_committee,
        latest_execution_payload_header=post_header,
        next_withdrawal_index=uint64(0),
        next_withdrawal_validator_index=ValidatorIndex(0),
        historical_summaries=[],
    )

    logger.info(f"Upgraded state to Capella at epoch {epoch}")
    return post


def upgrade_to_deneb(pre: CapellaBeaconState, fork_version: bytes, epoch: int) -> DenebBeaconState:
    """Upgrade a Capella state to Deneb.

    Adds blob gas fields to execution payload header.
    """
    pre_header = pre.latest_execution_payload_header

    post_header = ExecutionPayloadHeader(
        parent_hash=pre_header.parent_hash,
        fee_recipient=pre_header.fee_recipient,
        state_root=pre_header.state_root,
        receipts_root=pre_header.receipts_root,
        logs_bloom=pre_header.logs_bloom,
        prev_randao=pre_header.prev_randao,
        block_number=pre_header.block_number,
        gas_limit=pre_header.gas_limit,
        gas_used=pre_header.gas_used,
        timestamp=pre_header.timestamp,
        extra_data=list(pre_header.extra_data),
        base_fee_per_gas=pre_header.base_fee_per_gas,
        block_hash=pre_header.block_hash,
        transactions_root=pre_header.transactions_root,
        withdrawals_root=pre_header.withdrawals_root,
        blob_gas_used=uint64(0),
        excess_blob_gas=uint64(0),
    )

    post = DenebBeaconState(
        genesis_time=pre.genesis_time,
        genesis_validators_root=pre.genesis_validators_root,
        slot=pre.slot,
        fork=Fork(
            previous_version=Version(bytes(pre.fork.current_version)),
            current_version=Version(fork_version),
            epoch=Epoch(epoch),
        ),
        latest_block_header=pre.latest_block_header,
        block_roots=pre.block_roots,
        state_roots=pre.state_roots,
        historical_roots=list(pre.historical_roots),
        eth1_data=pre.eth1_data,
        eth1_data_votes=list(pre.eth1_data_votes),
        eth1_deposit_index=pre.eth1_deposit_index,
        validators=list(pre.validators),
        balances=list(pre.balances),
        randao_mixes=pre.randao_mixes,
        slashings=pre.slashings,
        previous_epoch_participation=list(pre.previous_epoch_participation),
        current_epoch_participation=list(pre.current_epoch_participation),
        justification_bits=pre.justification_bits,
        previous_justified_checkpoint=pre.previous_justified_checkpoint,
        current_justified_checkpoint=pre.current_justified_checkpoint,
        finalized_checkpoint=pre.finalized_checkpoint,
        inactivity_scores=list(pre.inactivity_scores),
        current_sync_committee=pre.current_sync_committee,
        next_sync_committee=pre.next_sync_committee,
        latest_execution_payload_header=post_header,
        next_withdrawal_index=pre.next_withdrawal_index,
        next_withdrawal_validator_index=pre.next_withdrawal_validator_index,
        historical_summaries=list(pre.historical_summaries),
    )

    logger.info(f"Upgraded state to Deneb at epoch {epoch}")
    return post


def upgrade_to_electra(pre: DenebBeaconState, fork_version: bytes, epoch: int) -> ElectraBeaconState:
    """Upgrade a Deneb state to Electra.

    Adds deposit/withdrawal/consolidation queue fields.
    """
    earliest_exit_epoch = _compute_earliest_exit_epoch(pre, epoch)

    post = ElectraBeaconState(
        genesis_time=pre.genesis_time,
        genesis_validators_root=pre.genesis_validators_root,
        slot=pre.slot,
        fork=Fork(
            previous_version=Version(bytes(pre.fork.current_version)),
            current_version=Version(fork_version),
            epoch=Epoch(epoch),
        ),
        latest_block_header=pre.latest_block_header,
        block_roots=pre.block_roots,
        state_roots=pre.state_roots,
        historical_roots=list(pre.historical_roots),
        eth1_data=pre.eth1_data,
        eth1_data_votes=list(pre.eth1_data_votes),
        eth1_deposit_index=pre.eth1_deposit_index,
        validators=list(pre.validators),
        balances=list(pre.balances),
        randao_mixes=pre.randao_mixes,
        slashings=pre.slashings,
        previous_epoch_participation=list(pre.previous_epoch_participation),
        current_epoch_participation=list(pre.current_epoch_participation),
        justification_bits=pre.justification_bits,
        previous_justified_checkpoint=pre.previous_justified_checkpoint,
        current_justified_checkpoint=pre.current_justified_checkpoint,
        finalized_checkpoint=pre.finalized_checkpoint,
        inactivity_scores=list(pre.inactivity_scores),
        current_sync_committee=pre.current_sync_committee,
        next_sync_committee=pre.next_sync_committee,
        latest_execution_payload_header=pre.latest_execution_payload_header,
        next_withdrawal_index=pre.next_withdrawal_index,
        next_withdrawal_validator_index=pre.next_withdrawal_validator_index,
        historical_summaries=list(pre.historical_summaries),
        deposit_requests_start_index=uint64(2**64 - 1),
        deposit_balance_to_consume=Gwei(0),
        exit_balance_to_consume=Gwei(0),
        earliest_exit_epoch=Epoch(earliest_exit_epoch),
        consolidation_balance_to_consume=Gwei(0),
        earliest_consolidation_epoch=Epoch(max(epoch, earliest_exit_epoch)),
        pending_deposits=[],
        pending_partial_withdrawals=[],
        pending_consolidations=[],
    )

    logger.info(f"Upgraded state to Electra at epoch {epoch}")
    return post


def initialize_proposer_lookahead(state: "BeaconState") -> list[ValidatorIndex]:
    """Initialize the proposer lookahead for Fulu.

    Return the proposer indices for the full available lookahead starting from current epoch.
    Used to initialize the proposer_lookahead field at genesis and after forks.

    Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/fulu/fork.md
    """
    current_epoch = get_current_epoch(state)
    lookahead = []
    for i in range(MIN_SEED_LOOKAHEAD + 1):
        epoch_proposers = get_beacon_proposer_indices(state, current_epoch + i)
        lookahead.extend([ValidatorIndex(p) for p in epoch_proposers])
    return lookahead


def upgrade_to_fulu(pre: ElectraBeaconState, fork_version: bytes, epoch: int) -> FuluBeaconState:
    """Upgrade an Electra state to Fulu.

    Adds proposer lookahead.
    """
    proposer_lookahead = initialize_proposer_lookahead(pre)

    post = FuluBeaconState(
        genesis_time=pre.genesis_time,
        genesis_validators_root=pre.genesis_validators_root,
        slot=pre.slot,
        fork=Fork(
            previous_version=Version(bytes(pre.fork.current_version)),
            current_version=Version(fork_version),
            epoch=Epoch(epoch),
        ),
        latest_block_header=pre.latest_block_header,
        block_roots=pre.block_roots,
        state_roots=pre.state_roots,
        historical_roots=list(pre.historical_roots),
        eth1_data=pre.eth1_data,
        eth1_data_votes=list(pre.eth1_data_votes),
        eth1_deposit_index=pre.eth1_deposit_index,
        validators=list(pre.validators),
        balances=list(pre.balances),
        randao_mixes=pre.randao_mixes,
        slashings=pre.slashings,
        previous_epoch_participation=list(pre.previous_epoch_participation),
        current_epoch_participation=list(pre.current_epoch_participation),
        justification_bits=pre.justification_bits,
        previous_justified_checkpoint=pre.previous_justified_checkpoint,
        current_justified_checkpoint=pre.current_justified_checkpoint,
        finalized_checkpoint=pre.finalized_checkpoint,
        inactivity_scores=list(pre.inactivity_scores),
        current_sync_committee=pre.current_sync_committee,
        next_sync_committee=pre.next_sync_committee,
        latest_execution_payload_header=pre.latest_execution_payload_header,
        next_withdrawal_index=pre.next_withdrawal_index,
        next_withdrawal_validator_index=pre.next_withdrawal_validator_index,
        historical_summaries=list(pre.historical_summaries),
        deposit_requests_start_index=pre.deposit_requests_start_index,
        deposit_balance_to_consume=pre.deposit_balance_to_consume,
        exit_balance_to_consume=pre.exit_balance_to_consume,
        earliest_exit_epoch=pre.earliest_exit_epoch,
        consolidation_balance_to_consume=pre.consolidation_balance_to_consume,
        earliest_consolidation_epoch=pre.earliest_consolidation_epoch,
        pending_deposits=list(pre.pending_deposits),
        pending_partial_withdrawals=list(pre.pending_partial_withdrawals),
        pending_consolidations=list(pre.pending_consolidations),
        proposer_lookahead=proposer_lookahead,
    )

    logger.info(f"Upgraded state to Fulu at epoch {epoch}")
    return post


def upgrade_to_gloas(pre: FuluBeaconState, fork_version: bytes, epoch: int) -> GloasBeaconState:
    """Upgrade a Fulu state to Gloas.

    Adds ePBS (enshrined Proposer-Builder Separation) fields:
    - latest_execution_payload_bid (replaces latest_execution_payload_header)
    - builders registry
    - builder pending payments and withdrawals
    - execution payload availability tracking
    """
    pre_header = pre.latest_execution_payload_header

    empty_bid = ExecutionPayloadBid(
        parent_block_hash=pre_header.parent_hash,
        parent_block_root=Root(b"\x00" * 32),
        block_hash=pre_header.block_hash,
        prev_randao=pre_header.prev_randao,
        fee_recipient=pre_header.fee_recipient,
        gas_limit=pre_header.gas_limit,
        builder_index=uint64(0),
        slot=pre.slot,
        value=Gwei(0),
        execution_payment=Gwei(0),
        blob_kzg_commitments_root=Root(b"\x00" * 32),
    )

    slots_per_hist = SLOTS_PER_HISTORICAL_ROOT()
    slots_2x_epoch = 2 * SLOTS_PER_EPOCH()

    empty_pending_payment = BuilderPendingPayment(
        weight=Gwei(0),
        withdrawal=BuilderPendingWithdrawal(
            fee_recipient=b"\x00" * 20,
            amount=Gwei(0),
            builder_index=uint64(0),
        ),
    )

    post = GloasBeaconState(
        genesis_time=pre.genesis_time,
        genesis_validators_root=pre.genesis_validators_root,
        slot=pre.slot,
        fork=Fork(
            previous_version=Version(bytes(pre.fork.current_version)),
            current_version=Version(fork_version),
            epoch=Epoch(epoch),
        ),
        latest_block_header=pre.latest_block_header,
        block_roots=pre.block_roots,
        state_roots=pre.state_roots,
        historical_roots=list(pre.historical_roots),
        eth1_data=pre.eth1_data,
        eth1_data_votes=list(pre.eth1_data_votes),
        eth1_deposit_index=pre.eth1_deposit_index,
        validators=list(pre.validators),
        balances=list(pre.balances),
        randao_mixes=pre.randao_mixes,
        slashings=pre.slashings,
        previous_epoch_participation=list(pre.previous_epoch_participation),
        current_epoch_participation=list(pre.current_epoch_participation),
        justification_bits=pre.justification_bits,
        previous_justified_checkpoint=pre.previous_justified_checkpoint,
        current_justified_checkpoint=pre.current_justified_checkpoint,
        finalized_checkpoint=pre.finalized_checkpoint,
        inactivity_scores=list(pre.inactivity_scores),
        current_sync_committee=pre.current_sync_committee,
        next_sync_committee=pre.next_sync_committee,
        latest_execution_payload_bid=empty_bid,
        next_withdrawal_index=pre.next_withdrawal_index,
        next_withdrawal_validator_index=pre.next_withdrawal_validator_index,
        historical_summaries=list(pre.historical_summaries),
        deposit_requests_start_index=pre.deposit_requests_start_index,
        deposit_balance_to_consume=pre.deposit_balance_to_consume,
        exit_balance_to_consume=pre.exit_balance_to_consume,
        earliest_exit_epoch=pre.earliest_exit_epoch,
        consolidation_balance_to_consume=pre.consolidation_balance_to_consume,
        earliest_consolidation_epoch=pre.earliest_consolidation_epoch,
        pending_deposits=list(pre.pending_deposits),
        pending_partial_withdrawals=list(pre.pending_partial_withdrawals),
        pending_consolidations=list(pre.pending_consolidations),
        proposer_lookahead=pre.proposer_lookahead,
        builders=[],
        next_withdrawal_builder_index=uint64(0),
        execution_payload_availability=Bitvector[slots_per_hist](),
        builder_pending_payments=[empty_pending_payment] * slots_2x_epoch,
        builder_pending_withdrawals=[],
        latest_block_hash=Hash32(pre_header.block_hash),
        payload_expected_withdrawals=[Withdrawal()] * MAX_WITHDRAWALS_PER_PAYLOAD(),
    )

    logger.info(f"Upgraded state to Gloas at epoch {epoch}")
    return post


def _compute_earliest_exit_epoch(state: DenebBeaconState, epoch: int) -> int:
    """Compute earliest exit epoch for Electra upgrade.

    Per the spec, this is the maximum exit_epoch among all validators
    that have exit_epoch != FAR_FUTURE_EPOCH, or current epoch if none.
    """
    earliest = epoch

    for validator in state.validators:
        exit_epoch = int(validator.exit_epoch)
        if exit_epoch != FAR_FUTURE_EPOCH and exit_epoch > earliest:
            earliest = exit_epoch

    return earliest


def maybe_upgrade_state(state: "BeaconState", target_epoch: int) -> "BeaconState":
    """Check if state needs upgrading and perform the upgrade if needed.

    This function checks the network config for any fork scheduled at
    target_epoch and upgrades the state accordingly.

    Args:
        state: Current beacon state
        target_epoch: The epoch we're transitioning into

    Returns:
        Upgraded state if a fork occurs, otherwise the original state
    """
    config = get_config()
    fork_info = config.get_fork_at_epoch(target_epoch)

    if fork_info is None:
        return state

    fork_epoch, fork_version, fork_name = fork_info
    current_version = bytes(state.fork.current_version)

    if fork_version == current_version:
        return state

    logger.info(
        f"Fork upgrade at epoch {target_epoch}: {fork_name} "
        f"(version {fork_version.hex()})"
    )

    if fork_name == "capella" and isinstance(state, BellatrixBeaconState):
        return upgrade_to_capella(state, fork_version, target_epoch)
    elif fork_name == "deneb" and isinstance(state, CapellaBeaconState):
        return upgrade_to_deneb(state, fork_version, target_epoch)
    elif fork_name == "electra" and isinstance(state, DenebBeaconState):
        return upgrade_to_electra(state, fork_version, target_epoch)
    elif fork_name == "fulu" and isinstance(state, ElectraBeaconState):
        return upgrade_to_fulu(state, fork_version, target_epoch)
    elif fork_name == "gloas" and isinstance(state, FuluBeaconState):
        return upgrade_to_gloas(state, fork_version, target_epoch)
    else:
        logger.warning(
            f"No upgrade function for {fork_name} from state type {type(state).__name__}. "
            f"Only updating state.fork."
        )
        state.fork = Fork(
            previous_version=Version(current_version),
            current_version=Version(fork_version),
            epoch=Epoch(target_epoch),
        )
        return state
