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
from ..types.bellatrix import BellatrixBeaconState
from ..types.capella import (
    CapellaBeaconState,
    ExecutionPayloadHeaderCapella,
)
from ..types.deneb import DenebBeaconState, ExecutionPayloadHeader
from ..types.electra import ElectraBeaconState
from ..types.fulu import FuluBeaconState
from ..types.gloas import (
    BeaconState as GloasBeaconState,
    ExecutionPayloadBid,
    BuilderPendingPayment,
    BuilderPendingWithdrawal,
)
from ..constants import SLOTS_PER_EPOCH, SLOTS_PER_HISTORICAL_ROOT, FAR_FUTURE_EPOCH, MIN_SEED_LOOKAHEAD
from ..network_config import get_config
from .helpers.accessors import get_current_epoch
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

    Adds ePBS (enshrined Proposer-Builder Separation) fields per alpha 7 spec.
    """
    import time as _time
    # The empty-requests root must come from the Gloas ExecutionRequests
    # (EIP-8282 builder request fields), not the Electra container.
    from ..types.gloas import ExecutionRequests
    from ...crypto import hash_tree_root
    from .helpers.ptc import compute_ptc
    from .helpers.misc import compute_start_slot_at_epoch

    _t0 = _time.monotonic()
    pre_header = pre.latest_execution_payload_header

    # Spec: latest_execution_payload_bid sets block_hash, gas_limit (alpha 8),
    # and execution_requests_root from the pre-fork execution payload header.
    empty_requests_root = hash_tree_root(ExecutionRequests())
    empty_bid = ExecutionPayloadBid(
        parent_block_hash=Hash32(),
        parent_block_root=Root(b"\x00" * 32),
        block_hash=pre_header.block_hash,
        prev_randao=Bytes32(),
        fee_recipient=b"\x00" * 20,
        gas_limit=uint64(pre_header.gas_limit),
        builder_index=uint64(0),
        slot=uint64(0),
        value=Gwei(0),
        execution_payment=Gwei(0),
        blob_kzg_commitments=[],
        execution_requests_root=empty_requests_root,
    )

    slots_per_hist = SLOTS_PER_HISTORICAL_ROOT()
    slots_2x_epoch = 2 * SLOTS_PER_EPOCH()
    spe = SLOTS_PER_EPOCH()

    empty_pending_payment = BuilderPendingPayment(
        weight=Gwei(0),
        withdrawal=BuilderPendingWithdrawal(
            fee_recipient=b"\x00" * 20,
            amount=Gwei(0),
            builder_index=uint64(0),
        ),
    )

    # Initialize execution_payload_availability to all 1s
    availability = Bitvector[slots_per_hist]()
    for i in range(slots_per_hist):
        availability[i] = True

    # Initialize ptc_window. The spec defines initialize_ptc_window over the
    # POST-upgrade state, so the "current_epoch" must be the Gloas fork epoch
    # (the epoch we're transitioning into) — not pre.slot // SPE, which is
    # still the pre-fork epoch because process_slots calls fork_upgrade BEFORE
    # incrementing state.slot. Using the pre-fork epoch here makes us cache
    # PTCs for epoch 0+1 while prysm caches epoch 1+2; the resulting state
    # diverges by one epoch worth of PTCs and every block from slot 8 onward
    # is invalid relative to peers.
    current_epoch = epoch
    from ..constants import PTC_SIZE
    ptc_size_val = PTC_SIZE()
    empty_ptc = [ValidatorIndex(0)] * ptc_size_val

    _t_ptc_start = _time.monotonic()
    ptc_window = []
    # Empty for previous epoch
    for _ in range(spe):
        ptc_window.append(list(empty_ptc))
    # Compute for current epoch through current + MIN_SEED_LOOKAHEAD
    for e in range(MIN_SEED_LOOKAHEAD + 1):
        target_epoch = current_epoch + e
        start_slot = compute_start_slot_at_epoch(target_epoch)
        for i in range(spe):
            _ts = _time.monotonic()
            ptc_window.append(list(compute_ptc(pre, start_slot + i)))
            logger.info(
                f"upgrade_to_gloas: compute_ptc slot={start_slot + i} took "
                f"{(_time.monotonic() - _ts) * 1000:.0f}ms"
            )
    logger.info(
        f"upgrade_to_gloas: PTC window built in "
        f"{(_time.monotonic() - _t_ptc_start) * 1000:.0f}ms "
        f"(prefix={(_t_ptc_start - _t0) * 1000:.0f}ms)"
    )

    _t_ctor = _time.monotonic()

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
        latest_block_hash=Hash32(pre_header.block_hash),
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
        execution_payload_availability=availability,
        builder_pending_payments=[empty_pending_payment] * slots_2x_epoch,
        builder_pending_withdrawals=[],
        latest_execution_payload_bid=empty_bid,
        payload_expected_withdrawals=[],
        ptc_window=ptc_window,
    )

    logger.info(
        f"upgrade_to_gloas: GloasBeaconState ctor took "
        f"{(_time.monotonic() - _t_ctor) * 1000:.0f}ms"
    )
    logger.info(
        f"Upgraded state to Gloas at epoch {epoch} (total "
        f"{(_time.monotonic() - _t0) * 1000:.0f}ms)"
    )
    return post


def upgrade_attestation_to_gloas(pre):
    """Upgrade a pre-Gloas (Electra/Fulu) attestation to the Gloas type.

    A Gloas BeaconBlockBody's attestations list requires the Gloas class;
    pool attestations are Electra-typed and must be locally upgraded before
    packing into a block.
    Returns ``pre`` unchanged if it is already a Gloas attestation.
    """
    from ..types.gloas import Attestation, AggregationBits

    if isinstance(pre, Attestation):
        return pre
    return Attestation(
        aggregation_bits=AggregationBits(list(pre.aggregation_bits)),
        data=pre.data,
        signature=pre.signature,
        committee_bits=pre.committee_bits,
    )


def upgrade_indexed_attestation_to_gloas(pre):
    """Upgrade a pre-Gloas indexed attestation to the Gloas type."""
    from ..types.gloas import IndexedAttestation, AttestingIndices

    if isinstance(pre, IndexedAttestation):
        return pre
    return IndexedAttestation(
        attesting_indices=AttestingIndices(list(pre.attesting_indices)),
        data=pre.data,
        signature=pre.signature,
    )


def upgrade_attester_slashing_to_gloas(pre):
    """Upgrade a pre-Gloas attester slashing to the Gloas type."""
    from ..types.gloas import AttesterSlashing

    if isinstance(pre, AttesterSlashing):
        return pre
    return AttesterSlashing(
        attestation_1=upgrade_indexed_attestation_to_gloas(pre.attestation_1),
        attestation_2=upgrade_indexed_attestation_to_gloas(pre.attestation_2),
    )


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
