# State Transition Implementation Plan

## Overview

This document outlines the complete implementation plan for the Ethereum consensus layer state transition function according to the official specs (master branch). The implementation targets Electra fork compatibility.

## Architecture

```
consensoor/spec/
├── __init__.py
├── constants.py          # Already exists
├── network_config.py     # Already exists
├── types/                # Already exists
│   └── ...
└── state_transition/     # NEW - All state transition logic
    ├── __init__.py
    ├── helpers/
    │   ├── __init__.py
    │   ├── math.py              # integer_squareroot, xor, saturating_sub
    │   ├── predicates.py        # is_active_validator, is_slashable_validator, etc.
    │   ├── accessors.py         # get_active_validator_indices, get_total_balance, etc.
    │   ├── mutators.py          # increase_balance, decrease_balance, initiate_validator_exit, etc.
    │   ├── beacon_committee.py  # compute_shuffled_index, compute_committee, get_beacon_committee
    │   ├── attestation.py       # get_attesting_indices, get_indexed_attestation
    │   ├── domain.py            # get_domain, compute_domain, compute_signing_root
    │   └── misc.py              # compute_epoch_at_slot, compute_fork_digest, etc.
    ├── epoch/
    │   ├── __init__.py
    │   ├── justification.py     # process_justification_and_finalization
    │   ├── rewards.py           # process_rewards_and_penalties, get_flag_index_deltas
    │   ├── registry.py          # process_registry_updates
    │   ├── slashings.py         # process_slashings
    │   ├── effective_balance.py # process_effective_balance_updates
    │   ├── inactivity.py        # process_inactivity_updates
    │   ├── sync_committee.py    # process_sync_committee_updates
    │   ├── participation.py     # process_participation_flag_updates
    │   ├── resets.py            # process_eth1_data_reset, process_slashings_reset, etc.
    │   ├── pending_deposits.py  # process_pending_deposits (Electra)
    │   └── pending_consolidations.py  # process_pending_consolidations (Electra)
    ├── block/
    │   ├── __init__.py
    │   ├── header.py            # process_block_header
    │   ├── randao.py            # process_randao
    │   ├── eth1_data.py         # process_eth1_data
    │   ├── execution_payload.py # process_execution_payload
    │   ├── withdrawals.py       # process_withdrawals, get_expected_withdrawals
    │   ├── sync_aggregate.py    # process_sync_aggregate
    │   └── operations/
    │       ├── __init__.py
    │       ├── attestation.py       # process_attestation
    │       ├── proposer_slashing.py # process_proposer_slashing
    │       ├── attester_slashing.py # process_attester_slashing
    │       ├── deposit.py           # process_deposit
    │       ├── voluntary_exit.py    # process_voluntary_exit
    │       ├── bls_change.py        # process_bls_to_execution_change
    │       ├── deposit_request.py   # process_deposit_request (Electra)
    │       ├── withdrawal_request.py # process_withdrawal_request (Electra)
    │       └── consolidation_request.py # process_consolidation_request (Electra)
    └── transition.py            # state_transition, process_slots, process_slot, process_epoch, process_block
```

## Implementation Order

### Phase 1: Core Helper Functions (Foundation)

These must be implemented first as everything else depends on them.

#### 1.1 Math Utilities (`helpers/math.py`)
- [ ] `integer_squareroot(n: uint64) -> uint64`
- [ ] `xor(bytes_1: Bytes32, bytes_2: Bytes32) -> Bytes32`
- [ ] `saturating_sub(a: int, b: int) -> int`

#### 1.2 Miscellaneous Helpers (`helpers/misc.py`)
- [ ] `compute_epoch_at_slot(slot: Slot) -> Epoch`
- [ ] `compute_start_slot_at_epoch(epoch: Epoch) -> Slot`
- [ ] `compute_activation_exit_epoch(epoch: Epoch) -> Epoch`
- [ ] `compute_fork_data_root(current_version: Version, genesis_validators_root: Root) -> Root`
- [ ] `compute_fork_digest(current_version: Version, genesis_validators_root: Root) -> ForkDigest`
- [ ] `compute_time_at_slot(state: BeaconState, slot: Slot) -> uint64`

#### 1.3 Predicate Functions (`helpers/predicates.py`)
- [ ] `is_active_validator(validator: Validator, epoch: Epoch) -> bool`
- [ ] `is_eligible_for_activation_queue(validator: Validator) -> bool`
- [ ] `is_eligible_for_activation(state: BeaconState, validator: Validator) -> bool`
- [ ] `is_slashable_validator(validator: Validator, epoch: Epoch) -> bool`
- [ ] `is_slashable_attestation_data(data_1: AttestationData, data_2: AttestationData) -> bool`
- [ ] `is_valid_indexed_attestation(state: BeaconState, indexed_attestation: IndexedAttestation) -> bool`
- [ ] `is_valid_merkle_branch(leaf: Bytes32, branch: Sequence[Bytes32], depth: uint64, index: uint64, root: Root) -> bool`
- [ ] `is_merge_transition_complete(state: BeaconState) -> bool` (Bellatrix)
- [ ] `is_execution_enabled(state: BeaconState, body: BeaconBlockBody) -> bool` (Bellatrix)
- [ ] `has_eth1_withdrawal_credential(validator: Validator) -> bool` (Capella)
- [ ] `is_fully_withdrawable_validator(validator: Validator, balance: Gwei, epoch: Epoch) -> bool` (Capella)
- [ ] `is_partially_withdrawable_validator(validator: Validator, balance: Gwei) -> bool` (Capella)
- [ ] `is_compounding_withdrawal_credential(withdrawal_credentials: Bytes32) -> bool` (Electra)
- [ ] `has_compounding_withdrawal_credential(validator: Validator) -> bool` (Electra)
- [ ] `has_execution_withdrawal_credential(validator: Validator) -> bool` (Electra)

#### 1.4 Beacon Committee Functions (`helpers/beacon_committee.py`)
- [ ] `compute_shuffled_index(index: uint64, index_count: uint64, seed: Bytes32) -> uint64`
- [ ] `compute_proposer_index(state: BeaconState, indices: Sequence[ValidatorIndex], seed: Bytes32) -> ValidatorIndex`
- [ ] `compute_committee(indices: Sequence[ValidatorIndex], seed: Bytes32, index: uint64, count: uint64) -> Sequence[ValidatorIndex]`
- [ ] `get_committee_count_per_slot(state: BeaconState, epoch: Epoch) -> uint64`
- [ ] `get_beacon_committee(state: BeaconState, slot: Slot, index: CommitteeIndex) -> Sequence[ValidatorIndex]`
- [ ] `get_beacon_proposer_index(state: BeaconState) -> ValidatorIndex`

#### 1.5 State Accessors (`helpers/accessors.py`)
- [ ] `get_current_epoch(state: BeaconState) -> Epoch`
- [ ] `get_previous_epoch(state: BeaconState) -> Epoch`
- [ ] `get_block_root(state: BeaconState, epoch: Epoch) -> Root`
- [ ] `get_block_root_at_slot(state: BeaconState, slot: Slot) -> Root`
- [ ] `get_randao_mix(state: BeaconState, epoch: Epoch) -> Bytes32`
- [ ] `get_active_validator_indices(state: BeaconState, epoch: Epoch) -> Sequence[ValidatorIndex]`
- [ ] `get_validator_churn_limit(state: BeaconState) -> uint64`
- [ ] `get_balance_churn_limit(state: BeaconState) -> Gwei` (Electra)
- [ ] `get_activation_exit_churn_limit(state: BeaconState) -> Gwei` (Electra)
- [ ] `get_consolidation_churn_limit(state: BeaconState) -> Gwei` (Electra)
- [ ] `get_seed(state: BeaconState, epoch: Epoch, domain_type: DomainType) -> Bytes32`
- [ ] `get_total_balance(state: BeaconState, indices: Set[ValidatorIndex]) -> Gwei`
- [ ] `get_total_active_balance(state: BeaconState) -> Gwei`
- [ ] `get_base_reward_per_increment(state: BeaconState) -> Gwei` (Altair)
- [ ] `get_base_reward(state: BeaconState, index: ValidatorIndex) -> Gwei`
- [ ] `get_proposer_reward(state: BeaconState, attester_index: ValidatorIndex) -> Gwei`
- [ ] `get_finality_delay(state: BeaconState) -> uint64`
- [ ] `is_in_inactivity_leak(state: BeaconState) -> bool`
- [ ] `get_eligible_validator_indices(state: BeaconState) -> Sequence[ValidatorIndex]`
- [ ] `get_max_effective_balance(validator: Validator) -> Gwei` (Electra)
- [ ] `get_pending_balance_to_withdraw(state: BeaconState, validator_index: ValidatorIndex) -> Gwei` (Electra)

#### 1.6 State Mutators (`helpers/mutators.py`)
- [ ] `increase_balance(state: BeaconState, index: ValidatorIndex, delta: Gwei) -> None`
- [ ] `decrease_balance(state: BeaconState, index: ValidatorIndex, delta: Gwei) -> None`
- [ ] `initiate_validator_exit(state: BeaconState, index: ValidatorIndex) -> None`
- [ ] `slash_validator(state: BeaconState, slashed_index: ValidatorIndex, whistleblower_index: ValidatorIndex | None) -> None`
- [ ] `switch_to_compounding_validator(state: BeaconState, index: ValidatorIndex) -> None` (Electra)
- [ ] `queue_excess_active_balance(state: BeaconState, index: ValidatorIndex) -> None` (Electra)
- [ ] `compute_exit_epoch_and_update_churn(state: BeaconState, exit_balance: Gwei) -> Epoch` (Electra)
- [ ] `compute_consolidation_epoch_and_update_churn(state: BeaconState, consolidation_balance: Gwei) -> Epoch` (Electra)

#### 1.7 Domain & Signing (`helpers/domain.py`)
- [ ] `get_domain(state: BeaconState, domain_type: DomainType, epoch: Epoch | None) -> Domain`
- [ ] `compute_domain(domain_type: DomainType, fork_version: Version | None, genesis_validators_root: Root | None) -> Domain`
- [ ] `compute_signing_root(ssz_object: SSZObject, domain: Domain) -> Root`
- [ ] `verify_block_signature(state: BeaconState, signed_block: SignedBeaconBlock) -> bool`

#### 1.8 Attestation Functions (`helpers/attestation.py`)
- [ ] `get_attesting_indices(state: BeaconState, attestation: Attestation) -> Set[ValidatorIndex]`
- [ ] `get_indexed_attestation(state: BeaconState, attestation: Attestation) -> IndexedAttestation`
- [ ] `get_attestation_participation_flag_indices(state: BeaconState, data: AttestationData, inclusion_delay: uint64) -> Sequence[int]` (Altair)
- [ ] `get_unslashed_participating_indices(state: BeaconState, flag_index: int, epoch: Epoch) -> Set[ValidatorIndex]` (Altair)
- [ ] `add_flag(flags: ParticipationFlags, flag_index: int) -> ParticipationFlags` (Altair)
- [ ] `has_flag(flags: ParticipationFlags, flag_index: int) -> bool` (Altair)
- [ ] `get_committee_indices(committee_bits: Bitvector) -> Sequence[CommitteeIndex]` (Electra)

#### 1.9 Sync Committee Functions (`helpers/sync_committee.py`)
- [ ] `get_sync_committee_indices(state: BeaconState, epoch: Epoch) -> Sequence[ValidatorIndex]` (Altair)
- [ ] `get_next_sync_committee(state: BeaconState) -> SyncCommittee` (Altair)

### Phase 2: Per-Slot Processing

#### 2.1 Slot Processing (`transition.py`)
- [ ] `process_slot(state: BeaconState) -> None`
  - Cache state root in state_roots
  - Update latest_block_header.state_root if empty
  - Cache block root in block_roots

- [ ] `process_slots(state: BeaconState, slot: Slot) -> None`
  - Loop from state.slot to target slot
  - Call process_slot() each iteration
  - Call process_epoch() at epoch boundaries

### Phase 3: Per-Epoch Processing

All these run at the end of an epoch (when (slot + 1) % SLOTS_PER_EPOCH == 0).

#### 3.1 Justification & Finalization (`epoch/justification.py`)
- [ ] `process_justification_and_finalization(state: BeaconState) -> None`
- [ ] `weigh_justification_and_finalization(state: BeaconState, total_active_balance: Gwei, previous_epoch_target_balance: Gwei, current_epoch_target_balance: Gwei) -> None`

#### 3.2 Inactivity Updates (`epoch/inactivity.py`)
- [ ] `process_inactivity_updates(state: BeaconState) -> None` (Altair)

#### 3.3 Rewards & Penalties (`epoch/rewards.py`)
- [ ] `process_rewards_and_penalties(state: BeaconState) -> None`
- [ ] `get_flag_index_deltas(state: BeaconState, flag_index: int) -> Tuple[Sequence[Gwei], Sequence[Gwei]]` (Altair)
- [ ] `get_inactivity_penalty_deltas(state: BeaconState) -> Tuple[Sequence[Gwei], Sequence[Gwei]]`

#### 3.4 Registry Updates (`epoch/registry.py`)
- [ ] `process_registry_updates(state: BeaconState) -> None`
- [ ] `get_validator_activation_churn_limit(state: BeaconState) -> uint64` (Deneb)

#### 3.5 Slashings (`epoch/slashings.py`)
- [ ] `process_slashings(state: BeaconState) -> None`

#### 3.6 Eth1 Data Reset (`epoch/resets.py`)
- [ ] `process_eth1_data_reset(state: BeaconState) -> None`

#### 3.7 Pending Deposits (`epoch/pending_deposits.py`)
- [ ] `process_pending_deposits(state: BeaconState) -> None` (Electra)
- [ ] `apply_pending_deposit(state: BeaconState, deposit: PendingDeposit) -> None` (Electra)

#### 3.8 Pending Consolidations (`epoch/pending_consolidations.py`)
- [ ] `process_pending_consolidations(state: BeaconState) -> None` (Electra)

#### 3.9 Effective Balance Updates (`epoch/effective_balance.py`)
- [ ] `process_effective_balance_updates(state: BeaconState) -> None`

#### 3.10 Slashings Reset (`epoch/resets.py`)
- [ ] `process_slashings_reset(state: BeaconState) -> None`

#### 3.11 Randao Mixes Reset (`epoch/resets.py`)
- [ ] `process_randao_mixes_reset(state: BeaconState) -> None`

#### 3.12 Historical Summaries Update (`epoch/resets.py`)
- [ ] `process_historical_summaries_update(state: BeaconState) -> None` (Capella)

#### 3.13 Participation Flag Updates (`epoch/participation.py`)
- [ ] `process_participation_flag_updates(state: BeaconState) -> None` (Altair)

#### 3.14 Sync Committee Updates (`epoch/sync_committee.py`)
- [ ] `process_sync_committee_updates(state: BeaconState) -> None` (Altair)

### Phase 4: Per-Block Processing

#### 4.1 Block Header (`block/header.py`)
- [ ] `process_block_header(state: BeaconState, block: BeaconBlock) -> None`

#### 4.2 Withdrawals (`block/withdrawals.py`)
- [ ] `get_expected_withdrawals(state: BeaconState) -> ExpectedWithdrawals` (Capella/Electra)
- [ ] `process_withdrawals(state: BeaconState, payload: ExecutionPayload) -> None` (Capella)

#### 4.3 Execution Payload (`block/execution_payload.py`)
- [ ] `process_execution_payload(state: BeaconState, body: BeaconBlockBody, execution_engine: ExecutionEngine) -> None` (Bellatrix)
- [ ] `get_execution_requests_list(body: BeaconBlockBody) -> Sequence[bytes]` (Electra)

#### 4.4 Randao (`block/randao.py`)
- [ ] `process_randao(state: BeaconState, body: BeaconBlockBody) -> None`

#### 4.5 Eth1 Data (`block/eth1_data.py`)
- [ ] `process_eth1_data(state: BeaconState, body: BeaconBlockBody) -> None`

#### 4.6 Sync Aggregate (`block/sync_aggregate.py`)
- [ ] `process_sync_aggregate(state: BeaconState, sync_aggregate: SyncAggregate) -> None` (Altair)

### Phase 5: Operations Processing

#### 5.1 Proposer Slashing (`block/operations/proposer_slashing.py`)
- [ ] `process_proposer_slashing(state: BeaconState, proposer_slashing: ProposerSlashing) -> None`

#### 5.2 Attester Slashing (`block/operations/attester_slashing.py`)
- [ ] `process_attester_slashing(state: BeaconState, attester_slashing: AttesterSlashing) -> None`

#### 5.3 Attestation (`block/operations/attestation.py`)
- [ ] `process_attestation(state: BeaconState, attestation: Attestation) -> None`

#### 5.4 Deposit (`block/operations/deposit.py`)
- [ ] `process_deposit(state: BeaconState, deposit: Deposit) -> None`
- [ ] `apply_deposit(state: BeaconState, pubkey: BLSPubkey, withdrawal_credentials: Bytes32, amount: Gwei, signature: BLSSignature) -> None`
- [ ] `add_validator_to_registry(state: BeaconState, pubkey: BLSPubkey, withdrawal_credentials: Bytes32, amount: Gwei) -> None`
- [ ] `is_valid_deposit_signature(pubkey: BLSPubkey, withdrawal_credentials: Bytes32, amount: Gwei, signature: BLSSignature) -> bool` (Electra)

#### 5.5 Voluntary Exit (`block/operations/voluntary_exit.py`)
- [ ] `process_voluntary_exit(state: BeaconState, signed_voluntary_exit: SignedVoluntaryExit) -> None`

#### 5.6 BLS to Execution Change (`block/operations/bls_change.py`)
- [ ] `process_bls_to_execution_change(state: BeaconState, signed_change: SignedBLSToExecutionChange) -> None` (Capella)

#### 5.7 Deposit Request (`block/operations/deposit_request.py`)
- [ ] `process_deposit_request(state: BeaconState, deposit_request: DepositRequest) -> None` (Electra)

#### 5.8 Withdrawal Request (`block/operations/withdrawal_request.py`)
- [ ] `process_withdrawal_request(state: BeaconState, withdrawal_request: WithdrawalRequest) -> None` (Electra)

#### 5.9 Consolidation Request (`block/operations/consolidation_request.py`)
- [ ] `process_consolidation_request(state: BeaconState, consolidation_request: ConsolidationRequest) -> None` (Electra)
- [ ] `is_valid_switch_to_compounding_request(state: BeaconState, consolidation_request: ConsolidationRequest) -> bool` (Electra)

### Phase 6: Main State Transition

#### 6.1 State Transition (`transition.py`)
- [ ] `state_transition(state: BeaconState, signed_block: SignedBeaconBlock, validate_result: bool = True) -> None`
- [ ] `process_block(state: BeaconState, block: BeaconBlock) -> None`
- [ ] `process_operations(state: BeaconState, body: BeaconBlockBody) -> None`
- [ ] `process_epoch(state: BeaconState) -> None`

### Phase 7: Integration with Node

#### 7.1 Update `_apply_block_to_state` in node.py
- [ ] Replace simplified state update with full `state_transition` call
- [ ] Handle validation errors properly
- [ ] Update head tracking

## Dependencies

### External Libraries
- `py_ecc` - BLS signature verification (already installed)
- `remerkleable` - SSZ serialization (already installed)

### Internal Dependencies
- `consensoor.crypto` - hash_tree_root, sha256
- `consensoor.spec.types` - All SSZ type definitions
- `consensoor.spec.constants` - All spec constants

## Testing Strategy

1. **Unit tests** for each helper function
2. **Integration tests** for epoch/block processing
3. **Spec compliance tests** against consensus-spec-tests vectors
4. **Live testing** with Kurtosis/Lighthouse

### Phase 8: Fulu-Specific Functions

#### 8.1 Blob Parameters (`helpers/blob.py`)
- [ ] `get_blob_parameters(epoch: Epoch) -> BlobParameters`

#### 8.2 Proposer Lookahead (`helpers/proposer_lookahead.py`)
- [ ] `compute_proposer_indices(state: BeaconState, epoch: Epoch, seed: Bytes32, indices: Sequence[ValidatorIndex]) -> Sequence[ValidatorIndex]`
- [ ] `get_beacon_proposer_indices(state: BeaconState, epoch: Epoch) -> Sequence[ValidatorIndex]`

#### 8.3 Epoch Processing (`epoch/proposer_lookahead.py`)
- [ ] `process_proposer_lookahead(state: BeaconState) -> None`

#### 8.4 Modified Functions (Fulu)
- [ ] `get_beacon_proposer_index(state: BeaconState) -> ValidatorIndex` - Use proposer_lookahead vector
- [ ] `process_execution_payload` - Validate blob commitments against dynamic max
- [ ] `process_epoch` - Call process_proposer_lookahead

### Phase 9: Gloas (ePBS) - Future

Note: Gloas/ePBS (EIP-7732) is not yet merged to master branch. Implementation will be added when spec is finalized.

#### Planned Functions (when available):
- [ ] Builder registration and management
- [ ] Payload attestation processing
- [ ] Execution payload bid handling
- [ ] Modified block processing for ePBS

## Notes

- All functions must handle both mainnet and minimal presets
- State is mutated in-place (following spec convention)
- BLS signature verification is critical for security
- Proper error handling for invalid blocks/attestations
- Fulu is the current target fork (Electra + PeerDAS + proposer lookahead)
