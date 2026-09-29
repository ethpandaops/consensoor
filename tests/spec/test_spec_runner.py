"""Consensus spec test runner.

Runs the Ethereum consensus spec tests against consensoor's implementation.
Supports all test categories: ssz_static, operations, epoch_processing, sanity, etc.
"""

import pytest
import snappy
import yaml
import copy
from pathlib import Path
from typing import Optional, Type, Callable, Any


def load_yaml(path: Path) -> Optional[dict]:
    """Load a YAML file."""
    if not path.exists():
        return None
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_ssz_snappy(path: Path, ssz_type: Type) -> Any:
    """Load and decode a snappy-compressed SSZ file."""
    with open(path, "rb") as f:
        decompressed = snappy.decompress(f.read())
    return ssz_type.decode_bytes(decompressed)


def ssz_copy(obj: Any) -> Any:
    """Create a deterministic copy of an SSZ object via encode/decode."""
    if hasattr(obj, "encode_bytes"):
        return obj.__class__.decode_bytes(bytes(obj.encode_bytes()))
    return copy.deepcopy(obj)


def get_spec_tests_dir(config) -> Path:
    """Get the spec tests directory, using preset-based default if not specified."""
    spec_dir = config.getoption("--spec-tests-dir")
    if spec_dir is None:
        preset = config.getoption("--preset")
        spec_dir = f"tests/spec-tests/tests/{preset}"
    return Path(spec_dir)


def get_state_type_for_fork(fork: str) -> Type:
    """Get the BeaconState type for a given fork."""
    from consensoor.spec import types

    state_types = {
        "phase0": types.Phase0BeaconState,
        "altair": types.AltairBeaconState,
        "bellatrix": types.BellatrixBeaconState,
        "capella": types.CapellaBeaconState,
        "deneb": types.DenebBeaconState,
        "electra": types.ElectraBeaconState,
        "fulu": types.FuluBeaconState,
        "gloas": types.BeaconState,
    }
    return state_types.get(fork)


def get_block_type_for_fork(fork: str) -> Type:
    """Get the SignedBeaconBlock type for a given fork."""
    from consensoor.spec import types

    block_types = {
        "phase0": types.SignedPhase0BeaconBlock,
        "altair": types.SignedAltairBeaconBlock,
        "bellatrix": types.SignedBellatrixBeaconBlock,
        "capella": types.SignedCapellaBeaconBlock,
        "deneb": types.SignedDenebBeaconBlock,
        "electra": types.SignedElectraBeaconBlock,
        "fulu": types.SignedElectraBeaconBlock,
        "gloas": types.SignedBeaconBlock,
    }
    return block_types.get(fork)


def get_execution_payload_type_for_fork(fork: str) -> Type:
    """Get the ExecutionPayload type for a given fork."""
    from consensoor.spec.types.bellatrix import ExecutionPayloadBellatrix
    from consensoor.spec.types.capella import ExecutionPayloadCapella
    from consensoor.spec.types import ExecutionPayload
    from consensoor.spec.types.gloas import ExecutionPayload as GloasExecutionPayload

    payload_types = {
        "bellatrix": ExecutionPayloadBellatrix,
        "capella": ExecutionPayloadCapella,
        "deneb": ExecutionPayload,
        "electra": ExecutionPayload,
        "fulu": ExecutionPayload,
        "gloas": GloasExecutionPayload,
    }
    return payload_types.get(fork)


def get_unsigned_block_type_for_fork(fork: str) -> Type:
    """Get the unsigned BeaconBlock type for a given fork."""
    from consensoor.spec import types

    block_types = {
        "phase0": types.Phase0BeaconBlock,
        "altair": types.AltairBeaconBlock,
        "bellatrix": types.BellatrixBeaconBlock,
        "capella": types.CapellaBeaconBlock,
        "deneb": types.DenebBeaconBlock,
        "electra": types.ElectraBeaconBlock,
        "fulu": types.ElectraBeaconBlock,
        "gloas": types.BeaconBlock,
    }
    return block_types.get(fork)


def get_block_body_type_for_fork(fork: str) -> Type:
    """Get the BeaconBlockBody type for a given fork."""
    from consensoor.spec import types

    body_types = {
        "phase0": types.Phase0BeaconBlockBody,
        "altair": types.AltairBeaconBlockBody,
        "bellatrix": types.BellatrixBeaconBlockBody,
        "capella": types.CapellaBeaconBlockBody,
        "deneb": types.DenebBeaconBlockBody,
        "electra": types.ElectraBeaconBlockBody,
        "fulu": types.ElectraBeaconBlockBody,
        "gloas": types.BeaconBlockBody,
    }
    return body_types.get(fork)


def get_ssz_type_by_name(fork: str, type_name: str) -> Optional[Type]:
    """Get SSZ type class by name for a given fork."""
    from consensoor.spec import types

    fork_prefix_map = {
        "phase0": "Phase0",
        "altair": "Altair",
        "bellatrix": "Bellatrix",
        "capella": "Capella",
        "deneb": "Deneb",
        "electra": "Electra",
        "fulu": "Fulu",
        "gloas": "",
    }

    pre_electra_forks = {"phase0", "altair", "bellatrix", "capella", "deneb"}
    phase0_types = {"Attestation", "IndexedAttestation", "AttesterSlashing"}

    if fork in pre_electra_forks and type_name in phase0_types:
        prefixed_name = f"Phase0{type_name}"
        if hasattr(types, prefixed_name):
            return getattr(types, prefixed_name)

    # Gloas is handled by gloas_only_types below: its AggregateAndProof wraps
    # the progressive (EIP-7688) Attestation, so the Electra type's root differs.
    electra_aggregate_types = {"AggregateAndProof", "SignedAggregateAndProof"}
    if fork in {"electra", "fulu"} and type_name in electra_aggregate_types:
        electra_name = f"Electra{type_name}"
        if hasattr(types, electra_name):
            return getattr(types, electra_name)
        if type_name.startswith("Signed"):
            alt_name = f"SignedElectra{type_name[6:]}"
            if hasattr(types, alt_name):
                return getattr(types, alt_name)

    light_client_types = {
        "LightClientHeader", "LightClientBootstrap", "LightClientUpdate",
        "LightClientFinalityUpdate", "LightClientOptimisticUpdate",
    }
    if type_name in light_client_types:
        if fork == "gloas":
            from consensoor.spec.types import gloas as gloas_mod
            if hasattr(gloas_mod, type_name):
                return getattr(gloas_mod, type_name)
        if fork in {"electra", "fulu"}:
            electra_name = f"Electra{type_name}"
            if hasattr(types, electra_name):
                return getattr(types, electra_name)
        elif fork == "deneb":
            deneb_name = f"Deneb{type_name}"
            if hasattr(types, deneb_name):
                return getattr(types, deneb_name)
        elif fork == "capella":
            capella_name = f"Capella{type_name}"
            if hasattr(types, capella_name):
                return getattr(types, capella_name)

    fulu_electra_types = {
        "BeaconBlockBody", "BeaconBlock", "SignedBeaconBlock",
        "Attestation", "IndexedAttestation", "AttesterSlashing",
        "ExecutionRequests", "PendingDeposit", "PendingPartialWithdrawal",
        "PendingConsolidation", "DepositRequest", "WithdrawalRequest",
        "ConsolidationRequest", "SingleAttestation",
    }
    if fork == "fulu" and type_name in fulu_electra_types:
        electra_name = f"Electra{type_name}"
        if hasattr(types, electra_name):
            return getattr(types, electra_name)
        if type_name.startswith("Signed"):
            alt_name = f"Signed{'Electra'}{type_name[6:]}"
            if hasattr(types, alt_name):
                return getattr(types, alt_name)
        if hasattr(types, type_name):
            return getattr(types, type_name)

    prefix = fork_prefix_map.get(fork, "")

    if prefix:
        prefixed_name = f"{prefix}{type_name}"
        if hasattr(types, prefixed_name):
            return getattr(types, prefixed_name)

        suffixed_name = f"{type_name}{prefix}"
        if hasattr(types, suffixed_name):
            return getattr(types, suffixed_name)

        if type_name.startswith("Signed"):
            alt_name = f"Signed{prefix}{type_name[6:]}"
            if hasattr(types, alt_name):
                return getattr(types, alt_name)

    if fork == "gloas" and type_name == "DataColumnSidecar":
        from consensoor.spec.types.gloas import DataColumnSidecar as GloasDataColumnSidecar
        return GloasDataColumnSidecar

    # Gloas-specific overrides for types that exist in multiple forks.
    # [Modified in Gloas:EIP7688] Attestation/IndexedAttestation/
    # AttesterSlashing/ExecutionRequests and the aggregate wrappers get
    # Gloas-local progressive definitions.
    if fork == "gloas":
        gloas_only_types = {
            "ExecutionPayload",
            "PartialDataColumnSidecar",
            "PartialDataColumnGroupID",
            "PartialDataColumnPartsMetadata",
            "Attestation",
            "IndexedAttestation",
            "AttesterSlashing",
            "AggregateAndProof",
            "SignedAggregateAndProof",
            "ExecutionRequests",
            "BuilderDepositRequest",
            "BuilderExitRequest",
            "NewPayloadRequest",
        }
        if type_name in gloas_only_types:
            from consensoor.spec.types import gloas as gloas_mod
            return getattr(gloas_mod, type_name, None)

    if hasattr(types, type_name):
        t = getattr(types, type_name)
        if fork != "gloas":
            from consensoor.spec.types.gloas import BeaconState, BeaconBlock, SignedBeaconBlock
            from consensoor.spec.types.gloas import BeaconBlockBody
            if t in (BeaconState, BeaconBlock, SignedBeaconBlock, BeaconBlockBody):
                return None
        return t

    return None


def get_operation_type_for_test(fork: str, op_name: str) -> Optional[Type]:
    """Get the SSZ type for an operation based on test directory name."""
    from consensoor.spec import types

    pre_electra_forks = {"phase0", "altair", "bellatrix", "capella", "deneb"}

    if fork == "gloas":
        # [Modified in Gloas:EIP7688] progressive attestation types
        from consensoor.spec.types import gloas as gloas_mod
        attestation_type = gloas_mod.Attestation
        attester_slashing_type = gloas_mod.AttesterSlashing
    elif fork in pre_electra_forks:
        attestation_type = types.Phase0Attestation
        attester_slashing_type = types.Phase0AttesterSlashing
    else:
        attestation_type = types.Attestation
        attester_slashing_type = types.AttesterSlashing

    op_type_map = {
        "attestation": attestation_type,
        "attester_slashing": attester_slashing_type,
        "proposer_slashing": types.ProposerSlashing,
        "deposit": types.Deposit,
        "voluntary_exit": types.SignedVoluntaryExit,
        "voluntary_exit_churn": types.SignedVoluntaryExit,
        "block_header": None,
        "bls_to_execution_change": types.SignedBLSToExecutionChange if hasattr(types, "SignedBLSToExecutionChange") else None,
        "sync_aggregate": types.SyncAggregate if hasattr(types, "SyncAggregate") else None,
        "execution_payload": None,
        "withdrawals": None,
        "deposit_request": types.DepositRequest if hasattr(types, "DepositRequest") else None,
        "withdrawal_request": types.WithdrawalRequest if hasattr(types, "WithdrawalRequest") else None,
        "consolidation_request": types.ConsolidationRequest if hasattr(types, "ConsolidationRequest") else None,
        "execution_payload_bid": types.SignedExecutionPayloadBid if hasattr(types, "SignedExecutionPayloadBid") else None,
        "parent_execution_payload": None,  # Uses block.ssz_snappy, special handling
        "payload_attestation": types.PayloadAttestation if hasattr(types, "PayloadAttestation") else None,
        "builder_deposit_request": types.BuilderDepositRequest if hasattr(types, "BuilderDepositRequest") else None,
        "builder_exit_request": types.BuilderExitRequest if hasattr(types, "BuilderExitRequest") else None,
    }
    return op_type_map.get(op_name)


def get_operation_processor(op_name: str) -> Optional[Callable]:
    """Get the processing function for an operation."""
    from consensoor.spec.state_transition.block.operations import (
        process_attestation,
        process_attester_slashing,
        process_proposer_slashing,
        process_deposit,
        process_voluntary_exit,
        process_bls_to_execution_change,
        process_deposit_request,
        process_withdrawal_request,
        process_consolidation_request,
        process_execution_payload_bid,
        process_parent_execution_payload,
        process_payload_attestation,
    )
    from consensoor.spec.state_transition.block.operations.builder_request import (
        process_builder_deposit_request,
        process_builder_exit_request,
    )
    from consensoor.spec.state_transition.block import (
        process_block_header,
        process_sync_aggregate,
        process_execution_payload,
        process_withdrawals,
    )

    processors = {
        "attestation": process_attestation,
        "attester_slashing": process_attester_slashing,
        "proposer_slashing": process_proposer_slashing,
        "deposit": process_deposit,
        "voluntary_exit": process_voluntary_exit,
        "voluntary_exit_churn": process_voluntary_exit,
        "block_header": process_block_header,
        "bls_to_execution_change": process_bls_to_execution_change,
        "sync_aggregate": process_sync_aggregate,
        "execution_payload": process_execution_payload,
        "withdrawals": process_withdrawals,
        "deposit_request": process_deposit_request,
        "withdrawal_request": process_withdrawal_request,
        "consolidation_request": process_consolidation_request,
        "execution_payload_bid": process_execution_payload_bid,
        "parent_execution_payload": process_parent_execution_payload,
        "payload_attestation": process_payload_attestation,
        "builder_deposit_request": process_builder_deposit_request,
        "builder_exit_request": process_builder_exit_request,
    }
    return processors.get(op_name)


def get_epoch_processor(function_name: str) -> Optional[Callable]:
    """Get the processing function for an epoch processing test."""
    from consensoor.spec.state_transition.epoch import (
        process_justification_and_finalization,
        process_inactivity_updates,
        process_rewards_and_penalties,
        process_registry_updates,
        process_slashings,
        process_effective_balance_updates,
        process_participation_flag_updates,
        process_participation_record_updates,
        process_sync_committee_updates,
        process_eth1_data_reset,
        process_slashings_reset,
        process_randao_mixes_reset,
        process_historical_summaries_update,
        process_pending_deposits,
        process_pending_consolidations,
        process_proposer_lookahead,
        process_builder_pending_payments,
    )

    from consensoor.spec.state_transition.epoch.ptc_window import process_ptc_window

    processors = {
        "justification_and_finalization": process_justification_and_finalization,
        "inactivity_updates": process_inactivity_updates,
        "rewards_and_penalties": process_rewards_and_penalties,
        "registry_updates": process_registry_updates,
        "slashings": process_slashings,
        "effective_balance_updates": process_effective_balance_updates,
        "participation_flag_updates": process_participation_flag_updates,
        "participation_record_updates": process_participation_record_updates,
        "sync_committee_updates": process_sync_committee_updates,
        "eth1_data_reset": process_eth1_data_reset,
        "slashings_reset": process_slashings_reset,
        "randao_mixes_reset": process_randao_mixes_reset,
        "historical_roots_update": process_historical_summaries_update,
        "historical_summaries_update": process_historical_summaries_update,
        "pending_deposits": process_pending_deposits,
        "pending_deposits_churn": process_pending_deposits,
        "pending_consolidations": process_pending_consolidations,
        "proposer_lookahead": process_proposer_lookahead,
        "builder_pending_payments": process_builder_pending_payments,
        "ptc_window": process_ptc_window,
    }
    return processors.get(function_name)


def discover_ssz_static_tests(spec_tests_dir: Path):
    """Discover all ssz_static test cases."""
    supported_forks = {"phase0", "altair", "bellatrix", "capella", "deneb", "electra", "fulu", "gloas"}
    test_cases = []
    for fork_dir in sorted(spec_tests_dir.iterdir()):
        if not fork_dir.is_dir():
            continue
        fork = fork_dir.name
        if fork not in supported_forks:
            continue
        ssz_dir = fork_dir / "ssz_static"
        if not ssz_dir.exists():
            continue
        for type_dir in sorted(ssz_dir.iterdir()):
            if not type_dir.is_dir():
                continue
            type_name = type_dir.name
            for ssz_file in sorted(type_dir.rglob("serialized.ssz_snappy")):
                case_path = ssz_file.parent
                case_id = f"{fork}/{type_name}/{case_path.parent.name}/{case_path.name}"
                test_cases.append((case_id, fork, type_name, case_path))
    return test_cases


def discover_operations_tests(spec_tests_dir: Path):
    """Discover all operations test cases."""
    supported_forks = {"phase0", "altair", "bellatrix", "capella", "deneb", "electra", "fulu", "gloas"}
    test_cases = []
    for fork_dir in sorted(spec_tests_dir.iterdir()):
        if not fork_dir.is_dir():
            continue
        fork = fork_dir.name
        if fork not in supported_forks:
            continue
        ops_dir = fork_dir / "operations"
        if not ops_dir.exists():
            continue
        for op_dir in sorted(ops_dir.iterdir()):
            if not op_dir.is_dir():
                continue
            op_name = op_dir.name
            pyspec_dir = op_dir / "pyspec_tests"
            if not pyspec_dir.exists():
                continue
            for case_dir in sorted(pyspec_dir.iterdir()):
                if not case_dir.is_dir():
                    continue
                pre_file = case_dir / "pre.ssz_snappy"
                if not pre_file.exists():
                    continue
                case_id = f"{fork}/operations/{op_name}/{case_dir.name}"
                test_cases.append((case_id, fork, op_name, case_dir))
    return test_cases


def discover_epoch_processing_tests(spec_tests_dir: Path):
    """Discover all epoch_processing test cases."""
    supported_forks = {"phase0", "altair", "bellatrix", "capella", "deneb", "electra", "fulu", "gloas"}
    test_cases = []
    for fork_dir in sorted(spec_tests_dir.iterdir()):
        if not fork_dir.is_dir():
            continue
        fork = fork_dir.name
        if fork not in supported_forks:
            continue
        epoch_dir = fork_dir / "epoch_processing"
        if not epoch_dir.exists():
            continue
        for func_dir in sorted(epoch_dir.iterdir()):
            if not func_dir.is_dir():
                continue
            func_name = func_dir.name
            pyspec_dir = func_dir / "pyspec_tests"
            if not pyspec_dir.exists():
                continue
            for case_dir in sorted(pyspec_dir.iterdir()):
                if not case_dir.is_dir():
                    continue
                pre_file = case_dir / "pre.ssz_snappy"
                if not pre_file.exists():
                    continue
                case_id = f"{fork}/epoch_processing/{func_name}/{case_dir.name}"
                test_cases.append((case_id, fork, func_name, case_dir))
    return test_cases


def discover_sanity_blocks_tests(spec_tests_dir: Path):
    """Discover all sanity/blocks test cases."""
    supported_forks = {"phase0", "altair", "bellatrix", "capella", "deneb", "electra", "fulu", "gloas"}
    test_cases = []
    for fork_dir in sorted(spec_tests_dir.iterdir()):
        if not fork_dir.is_dir():
            continue
        fork = fork_dir.name
        if fork not in supported_forks:
            continue
        blocks_dir = fork_dir / "sanity" / "blocks" / "pyspec_tests"
        if not blocks_dir.exists():
            continue
        for case_dir in sorted(blocks_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            pre_file = case_dir / "pre.ssz_snappy"
            if not pre_file.exists():
                continue
            case_id = f"{fork}/sanity/blocks/{case_dir.name}"
            test_cases.append((case_id, fork, case_dir))
    return test_cases


def discover_sanity_slots_tests(spec_tests_dir: Path):
    """Discover all sanity/slots test cases."""
    supported_forks = {"phase0", "altair", "bellatrix", "capella", "deneb", "electra", "fulu", "gloas"}
    test_cases = []
    for fork_dir in sorted(spec_tests_dir.iterdir()):
        if not fork_dir.is_dir():
            continue
        fork = fork_dir.name
        if fork not in supported_forks:
            continue
        slots_dir = fork_dir / "sanity" / "slots" / "pyspec_tests"
        if not slots_dir.exists():
            continue
        for case_dir in sorted(slots_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            pre_file = case_dir / "pre.ssz_snappy"
            if not pre_file.exists():
                continue
            case_id = f"{fork}/sanity/slots/{case_dir.name}"
            test_cases.append((case_id, fork, case_dir))
    return test_cases


def discover_finality_tests(spec_tests_dir: Path):
    """Discover all finality test cases."""
    supported_forks = {"phase0", "altair", "bellatrix", "capella", "deneb", "electra", "fulu", "gloas"}
    test_cases = []
    for fork_dir in sorted(spec_tests_dir.iterdir()):
        if not fork_dir.is_dir():
            continue
        fork = fork_dir.name
        if fork not in supported_forks:
            continue
        finality_dir = fork_dir / "finality" / "finality" / "pyspec_tests"
        if not finality_dir.exists():
            continue
        for case_dir in sorted(finality_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            pre_file = case_dir / "pre.ssz_snappy"
            if not pre_file.exists():
                continue
            case_id = f"{fork}/finality/{case_dir.name}"
            test_cases.append((case_id, fork, case_dir))
    return test_cases


def discover_rewards_tests(spec_tests_dir: Path):
    """Discover all rewards test cases."""
    supported_forks = {"phase0", "altair", "bellatrix", "capella", "deneb", "electra", "fulu", "gloas"}
    test_cases = []
    for fork_dir in sorted(spec_tests_dir.iterdir()):
        if not fork_dir.is_dir():
            continue
        fork = fork_dir.name
        if fork not in supported_forks:
            continue
        rewards_dir = fork_dir / "rewards"
        if not rewards_dir.exists():
            continue
        for reward_type_dir in sorted(rewards_dir.iterdir()):
            if not reward_type_dir.is_dir():
                continue
            reward_type = reward_type_dir.name
            pyspec_dir = reward_type_dir / "pyspec_tests"
            if not pyspec_dir.exists():
                continue
            for case_dir in sorted(pyspec_dir.iterdir()):
                if not case_dir.is_dir():
                    continue
                pre_file = case_dir / "pre.ssz_snappy"
                if not pre_file.exists():
                    continue
                case_id = f"{fork}/rewards/{reward_type}/{case_dir.name}"
                test_cases.append((case_id, fork, reward_type, case_dir))
    return test_cases


def discover_shuffling_tests(spec_tests_dir: Path):
    """Discover all shuffling test cases."""
    supported_forks = {"phase0", "altair", "bellatrix", "capella", "deneb", "electra", "fulu", "gloas"}
    test_cases = []
    for fork_dir in sorted(spec_tests_dir.iterdir()):
        if not fork_dir.is_dir():
            continue
        fork = fork_dir.name
        if fork not in supported_forks:
            continue
        shuffling_dir = fork_dir / "shuffling" / "core" / "shuffle"
        if not shuffling_dir.exists():
            continue
        for case_dir in sorted(shuffling_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            case_file = case_dir / "mapping.yaml"
            if not case_file.exists():
                continue
            case_id = f"{fork}/shuffling/{case_dir.name}"
            test_cases.append((case_id, fork, case_file))
    return test_cases


def discover_fork_choice_compliance_tests(spec_tests_dir: Path):
    """Discover all fork-choice compliance test cases.

    Layout: <preset>/<fork>/fork_choice_compliance/<test_name>/pyspec_tests/<case>/{
        steps.yaml, meta.yaml, anchor_block.ssz_snappy, anchor_state.ssz_snappy,
        block_0x*.ssz_snappy, attestation_0x*.ssz_snappy,
        payload_attestation_0x*.ssz_snappy,
        execution_payload_envelope_0x*.ssz_snappy,
        attester_slashing_0x*.ssz_snappy
    }
    """
    supported_forks = {"fulu", "gloas"}
    test_cases = []
    for fork_dir in sorted(spec_tests_dir.iterdir()):
        if not fork_dir.is_dir():
            continue
        fork = fork_dir.name
        if fork not in supported_forks:
            continue
        fc_dir = fork_dir / "fork_choice_compliance"
        if not fc_dir.exists():
            continue
        for test_name_dir in sorted(fc_dir.iterdir()):
            if not test_name_dir.is_dir():
                continue
            pyspec_dir = test_name_dir / "pyspec_tests"
            if not pyspec_dir.exists():
                continue
            for case_dir in sorted(pyspec_dir.iterdir()):
                if not case_dir.is_dir():
                    continue
                if not (case_dir / "steps.yaml").exists():
                    continue
                case_id = f"{fork}/fork_choice_compliance/{test_name_dir.name}/{case_dir.name}"
                test_cases.append((case_id, fork, test_name_dir.name, case_dir))
    return test_cases


def discover_random_tests(spec_tests_dir: Path):
    """Discover all random test cases."""
    supported_forks = {"phase0", "altair", "bellatrix", "capella", "deneb", "electra", "fulu", "gloas"}
    test_cases = []
    for fork_dir in sorted(spec_tests_dir.iterdir()):
        if not fork_dir.is_dir():
            continue
        fork = fork_dir.name
        if fork not in supported_forks:
            continue
        random_dir = fork_dir / "random" / "random" / "pyspec_tests"
        if not random_dir.exists():
            continue
        for case_dir in sorted(random_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            pre_file = case_dir / "pre.ssz_snappy"
            if not pre_file.exists():
                continue
            case_id = f"{fork}/random/{case_dir.name}"
            test_cases.append((case_id, fork, case_dir))
    return test_cases


CASE_FIXTURES = [
    ("ssz_case", discover_ssz_static_tests),
    ("fork_choice_case", lambda d: discover_fork_choice_tests(d, "fork_choice")),
    ("fast_confirmation_case", lambda d: discover_fork_choice_tests(d, "fast_confirmation")),
    ("operations_case", discover_operations_tests),
    ("epoch_case", discover_epoch_processing_tests),
    ("sanity_blocks_case", discover_sanity_blocks_tests),
    ("sanity_slots_case", discover_sanity_slots_tests),
    ("finality_case", discover_finality_tests),
    ("rewards_case", discover_rewards_tests),
    ("shuffling_case", discover_shuffling_tests),
    ("random_case", discover_random_tests),
    ("compliance_case", discover_fork_choice_compliance_tests),
]


def pytest_generate_tests(metafunc):
    """Generate test cases from spec test directories.

    Always parametrize (with an empty list if fixtures aren't downloaded)
    so pytest collects zero tests rather than raising "fixture not found"
    at setup time.
    """
    spec_tests_dir = get_spec_tests_dir(metafunc.config)
    for fixture_name, discover in CASE_FIXTURES:
        if fixture_name in metafunc.fixturenames:
            test_cases = discover(spec_tests_dir) if spec_tests_dir.exists() else []
            ids = [tc[0] for tc in test_cases]
            metafunc.parametrize(fixture_name, test_cases, ids=ids)


class TestSSZStatic:
    """SSZ static tests - verify SSZ encode/decode and hash_tree_root."""

    def test_ssz_roundtrip(self, ssz_case, preset):
        case_id, fork, type_name, case_path = ssz_case

        type_class = get_ssz_type_by_name(fork, type_name)
        if type_class is None:
            pytest.skip(f"Type {type_name} not implemented for {fork}")

        ssz_file = case_path / "serialized.ssz_snappy"
        roots_file = case_path / "roots.yaml"

        with open(ssz_file, "rb") as f:
            decompressed = snappy.decompress(f.read())

        obj = type_class.decode_bytes(decompressed)

        encoded = bytes(obj.encode_bytes())
        assert encoded == decompressed, f"Encode/decode roundtrip failed for {case_id}"

        if roots_file.exists():
            roots = load_yaml(roots_file)
            if roots and "root" in roots:
                expected_root = bytes.fromhex(roots["root"][2:])
                actual_root = obj.hash_tree_root()
                assert actual_root == expected_root, \
                    f"Root mismatch for {case_id}: {actual_root.hex()} != {expected_root.hex()}"


class TestOperations:
    """Operations tests - verify individual block operation processing."""

    def test_operation(self, operations_case, preset):
        case_id, fork, op_name, case_path = operations_case

        state_type = get_state_type_for_fork(fork)
        if state_type is None:
            pytest.skip(f"State type not implemented for {fork}")

        processor = get_operation_processor(op_name)
        if processor is None:
            pytest.skip(f"Operation processor not implemented for {op_name}")

        pre_file = case_path / "pre.ssz_snappy"
        post_file = case_path / "post.ssz_snappy"
        expects_failure = not post_file.exists()

        pre_state = load_ssz_snappy(pre_file, state_type)

        op_type = get_operation_type_for_test(fork, op_name)
        op_file_name = f"{op_name}.ssz_snappy"
        op_file = case_path / op_file_name

        if not op_file.exists():
            alt_names = {
                "voluntary_exit": "voluntary_exit.ssz_snappy",
                "voluntary_exit_churn": "voluntary_exit.ssz_snappy",
                "bls_to_execution_change": "address_change.ssz_snappy",
            }
            if op_name in alt_names:
                op_file = case_path / alt_names[op_name]

        if op_type is None or not op_file.exists():
            if op_name == "block_header":
                # Block header tests use unsigned BeaconBlock
                block_type = get_unsigned_block_type_for_fork(fork)
                if block_type is None:
                    pytest.skip(f"Block type not implemented for {fork}")
                block_file = case_path / "block.ssz_snappy"
                if not block_file.exists():
                    pytest.skip(f"Block file not found for {case_id}")
                operation = load_ssz_snappy(block_file, block_type)
            elif op_name in ("execution_payload", "withdrawals", "parent_execution_payload"):
                # These have special handling below - operation loaded differently
                operation = None
            else:
                pytest.skip(f"Operation type/file not found for {op_name} in {fork}")
        else:
            operation = load_ssz_snappy(op_file, op_type)

        state_copy = ssz_copy(pre_state)

        try:
            if op_name == "block_header":
                processor(state_copy, operation)
            elif op_name == "sync_aggregate":
                processor(state_copy, operation)
            elif op_name == "execution_payload":
                execution_file = case_path / "execution.yaml"
                execution_valid = True
                if execution_file.exists():
                    execution_meta = load_yaml(execution_file)
                    if execution_meta:
                        execution_valid = execution_meta.get("execution_valid", True)
                if fork == "gloas":
                    from consensoor.spec import types
                    signed_envelope_file = case_path / "signed_envelope.ssz_snappy"
                    signed_envelope = load_ssz_snappy(
                        signed_envelope_file, types.SignedExecutionPayloadEnvelope
                    )

                    class TestEngine:
                        def verify_and_notify_new_payload(self, _request) -> bool:
                            return execution_valid

                    processor(
                        state_copy,
                        signed_envelope,
                        execution_engine=TestEngine(),
                        execution_valid=execution_valid,
                    )
                else:
                    body_type = get_block_body_type_for_fork(fork)
                    body_file = case_path / "body.ssz_snappy"
                    body = load_ssz_snappy(body_file, body_type)
                    processor(state_copy, body, execution_valid=execution_valid)
            elif op_name == "withdrawals":
                if fork == "gloas":
                    processor(state_copy)
                else:
                    payload_type = get_execution_payload_type_for_fork(fork)
                    payload_file = case_path / "execution_payload.ssz_snappy"
                    payload = load_ssz_snappy(payload_file, payload_type)
                    processor(state_copy, payload)
            elif op_name == "parent_execution_payload":
                block_type = get_unsigned_block_type_for_fork(fork)
                block_file = case_path / "block.ssz_snappy"
                block = load_ssz_snappy(block_file, block_type)
                processor(state_copy, block)
            else:
                processor(state_copy, operation)

            if expects_failure:
                pytest.fail(f"Expected operation to fail but it succeeded: {case_id}")
        except (AssertionError, Exception) as e:
            if expects_failure:
                return
            raise AssertionError(f"Operation failed unexpectedly: {case_id}: {e}") from e

        if post_file.exists():
            expected_state = load_ssz_snappy(post_file, state_type)
            actual_root = state_copy.hash_tree_root()
            expected_root = expected_state.hash_tree_root()
            assert actual_root == expected_root, \
                f"State root mismatch for {case_id}: {actual_root.hex()} != {expected_root.hex()}"


class TestEpochProcessing:
    """Epoch processing tests - verify individual epoch processing functions."""

    def test_epoch_processing(self, epoch_case, preset):
        case_id, fork, func_name, case_path = epoch_case

        state_type = get_state_type_for_fork(fork)
        if state_type is None:
            pytest.skip(f"State type not implemented for {fork}")

        processor = get_epoch_processor(func_name)
        if processor is None:
            pytest.skip(f"Epoch processor not implemented for {func_name}")

        pre_file = case_path / "pre.ssz_snappy"
        post_file = case_path / "post.ssz_snappy"
        expects_failure = not post_file.exists()

        pre_state = load_ssz_snappy(pre_file, state_type)
        state_copy = ssz_copy(pre_state)

        try:
            processor(state_copy)

            if expects_failure:
                pytest.fail(f"Expected epoch processing to fail but it succeeded: {case_id}")
        except (AssertionError, Exception) as e:
            if expects_failure:
                return
            raise AssertionError(f"Epoch processing failed unexpectedly: {case_id}: {e}") from e

        if post_file.exists():
            expected_state = load_ssz_snappy(post_file, state_type)
            actual_root = state_copy.hash_tree_root()
            expected_root = expected_state.hash_tree_root()
            assert actual_root == expected_root, \
                f"State root mismatch for {case_id}: {actual_root.hex()} != {expected_root.hex()}"


class TestSanityBlocks:
    """Sanity/blocks tests - verify full block state transitions."""

    def test_sanity_blocks(self, sanity_blocks_case, preset):
        case_id, fork, case_path = sanity_blocks_case

        state_type = get_state_type_for_fork(fork)
        block_type = get_block_type_for_fork(fork)
        if state_type is None or block_type is None:
            pytest.skip(f"Types not implemented for {fork}")

        from consensoor.spec.state_transition import state_transition

        pre_file = case_path / "pre.ssz_snappy"
        post_file = case_path / "post.ssz_snappy"
        expects_failure = not post_file.exists()

        meta_file = case_path / "meta.yaml"
        meta = load_yaml(meta_file) or {}
        bls_setting = meta.get("bls_setting", 1)

        pre_state = load_ssz_snappy(pre_file, state_type)
        state = ssz_copy(pre_state)

        block_files = sorted(
            case_path.glob("blocks_*.ssz_snappy"),
            key=lambda p: int(p.stem.split("_")[1])
        )

        try:
            for block_file in block_files:
                block = load_ssz_snappy(block_file, block_type)
                state = state_transition(state, block, validate_result=(bls_setting != 2))

            if expects_failure:
                pytest.fail(f"Expected block transition to fail but it succeeded: {case_id}")
        except (AssertionError, Exception) as e:
            if expects_failure:
                return
            raise AssertionError(f"Block transition failed unexpectedly: {case_id}: {e}") from e

        if post_file.exists():
            expected_state = load_ssz_snappy(post_file, state_type)
            actual_root = state.hash_tree_root()
            expected_root = expected_state.hash_tree_root()
            assert actual_root == expected_root, \
                f"State root mismatch for {case_id}: {actual_root.hex()} != {expected_root.hex()}"


class TestSanitySlots:
    """Sanity/slots tests - verify slot-only state transitions."""

    def test_sanity_slots(self, sanity_slots_case, preset):
        case_id, fork, case_path = sanity_slots_case

        state_type = get_state_type_for_fork(fork)
        if state_type is None:
            pytest.skip(f"State type not implemented for {fork}")

        from consensoor.spec.state_transition import process_slots

        pre_file = case_path / "pre.ssz_snappy"
        post_file = case_path / "post.ssz_snappy"
        slots_file = case_path / "slots.yaml"

        pre_state = load_ssz_snappy(pre_file, state_type)
        state = ssz_copy(pre_state)

        slots_data = load_yaml(slots_file)
        target_slot = int(state.slot) + slots_data

        try:
            state = process_slots(state, target_slot)
        except (AssertionError, Exception) as e:
            if not post_file.exists():
                return
            raise AssertionError(f"Slot transition failed unexpectedly: {case_id}: {e}") from e

        if post_file.exists():
            expected_state = load_ssz_snappy(post_file, state_type)
            actual_root = state.hash_tree_root()
            expected_root = expected_state.hash_tree_root()
            assert actual_root == expected_root, \
                f"State root mismatch for {case_id}: {actual_root.hex()} != {expected_root.hex()}"


class TestFinality:
    """Finality tests - verify finality transitions."""

    def test_finality(self, finality_case, preset):
        case_id, fork, case_path = finality_case

        state_type = get_state_type_for_fork(fork)
        block_type = get_block_type_for_fork(fork)
        if state_type is None or block_type is None:
            pytest.skip(f"Types not implemented for {fork}")

        from consensoor.spec.state_transition import state_transition

        pre_file = case_path / "pre.ssz_snappy"
        post_file = case_path / "post.ssz_snappy"

        meta_file = case_path / "meta.yaml"
        meta = load_yaml(meta_file) or {}
        blocks_count = meta.get("blocks_count", 0)

        pre_state = load_ssz_snappy(pre_file, state_type)
        state = ssz_copy(pre_state)

        for i in range(blocks_count):
            block_file = case_path / f"blocks_{i}.ssz_snappy"
            if block_file.exists():
                block = load_ssz_snappy(block_file, block_type)
                state = state_transition(state, block, validate_result=False)

        if post_file.exists():
            expected_state = load_ssz_snappy(post_file, state_type)
            actual_root = state.hash_tree_root()
            expected_root = expected_state.hash_tree_root()
            assert actual_root == expected_root, \
                f"State root mismatch for {case_id}: {actual_root.hex()} != {expected_root.hex()}"


class TestRandom:
    """Random tests - verify random state transitions."""

    def test_random(self, random_case, preset):
        case_id, fork, case_path = random_case

        state_type = get_state_type_for_fork(fork)
        block_type = get_block_type_for_fork(fork)
        if state_type is None or block_type is None:
            pytest.skip(f"Types not implemented for {fork}")

        from consensoor.spec.state_transition import state_transition

        pre_file = case_path / "pre.ssz_snappy"
        post_file = case_path / "post.ssz_snappy"

        pre_state = load_ssz_snappy(pre_file, state_type)
        state = ssz_copy(pre_state)

        block_files = sorted(
            case_path.glob("blocks_*.ssz_snappy"),
            key=lambda p: int(p.stem.split("_")[1])
        )

        try:
            for block_file in block_files:
                block = load_ssz_snappy(block_file, block_type)
                state = state_transition(state, block, validate_result=False)
        except (AssertionError, Exception) as e:
            if not post_file.exists():
                return
            raise AssertionError(f"Random transition failed: {case_id}: {e}") from e

        if post_file.exists():
            expected_state = load_ssz_snappy(post_file, state_type)
            actual_root = state.hash_tree_root()
            expected_root = expected_state.hash_tree_root()
            assert actual_root == expected_root, \
                f"State root mismatch for {case_id}: {actual_root.hex()} != {expected_root.hex()}"


class TestShuffling:
    """Shuffling tests - verify validator shuffling."""

    def test_shuffling(self, shuffling_case, preset):
        case_id, fork, case_file = shuffling_case

        from consensoor.spec.state_transition.helpers.beacon_committee import compute_shuffled_index

        data = load_yaml(case_file)
        if data is None:
            pytest.skip(f"Could not load shuffling test: {case_id}")

        seed = bytes.fromhex(data["seed"][2:])
        count = data["count"]
        expected_mapping = data["mapping"]

        for index, expected in enumerate(expected_mapping):
            actual = compute_shuffled_index(index, count, seed)
            assert actual == expected, \
                f"Shuffling mismatch at index {index}: got {actual}, expected {expected}"


class TestRewards:
    """Rewards tests - verify reward calculations."""

    def test_rewards(self, rewards_case, preset):
        case_id, fork, reward_type, case_path = rewards_case

        state_type = get_state_type_for_fork(fork)
        if state_type is None:
            pytest.skip(f"State type not implemented for {fork}")

        pre_file = case_path / "pre.ssz_snappy"
        if not pre_file.exists():
            pytest.skip(f"Pre-state file not found for {case_id}")

        pre_state = load_ssz_snappy(pre_file, state_type)

        source_deltas_file = case_path / "source_deltas.ssz_snappy"
        target_deltas_file = case_path / "target_deltas.ssz_snappy"
        head_deltas_file = case_path / "head_deltas.ssz_snappy"
        inactivity_penalty_deltas_file = case_path / "inactivity_penalty_deltas.ssz_snappy"
        inclusion_delay_deltas_file = case_path / "inclusion_delay_deltas.ssz_snappy"

        from consensoor.spec.types.base import List, uint64
        from consensoor.spec.constants import VALIDATOR_REGISTRY_LIMIT

        DeltasList = List[List[uint64, VALIDATOR_REGISTRY_LIMIT], 2]

        def check_deltas(name, actual_rewards, actual_penalties, expected_file):
            if not expected_file.exists():
                return
            expected = load_ssz_snappy(expected_file, DeltasList)
            for i in range(len(pre_state.validators)):
                if i < len(expected[0]) and i < len(expected[1]):
                    assert actual_rewards[i] == int(expected[0][i]), \
                        f"{name} reward mismatch at {i}: {actual_rewards[i]} != {expected[0][i]}"
                    assert actual_penalties[i] == int(expected[1][i]), \
                        f"{name} penalty mismatch at {i}: {actual_penalties[i]} != {expected[1][i]}"

        if fork in {"phase0"}:
            from consensoor.spec.state_transition.epoch.rewards import (
                get_source_deltas_phase0,
                get_target_deltas_phase0,
                get_head_deltas_phase0,
                get_inclusion_delay_deltas_phase0,
                get_inactivity_penalty_deltas_phase0,
            )

            source_rewards, source_penalties = get_source_deltas_phase0(pre_state)
            check_deltas("Source", source_rewards, source_penalties, source_deltas_file)

            target_rewards, target_penalties = get_target_deltas_phase0(pre_state)
            check_deltas("Target", target_rewards, target_penalties, target_deltas_file)

            head_rewards, head_penalties = get_head_deltas_phase0(pre_state)
            check_deltas("Head", head_rewards, head_penalties, head_deltas_file)

            inclusion_rewards, inclusion_penalties = get_inclusion_delay_deltas_phase0(pre_state)
            check_deltas("Inclusion delay", inclusion_rewards, inclusion_penalties, inclusion_delay_deltas_file)

            inactivity_rewards, inactivity_penalties = get_inactivity_penalty_deltas_phase0(pre_state)
            check_deltas("Inactivity", inactivity_rewards, inactivity_penalties, inactivity_penalty_deltas_file)
        else:
            from consensoor.spec.state_transition.epoch.rewards import (
                get_flag_index_deltas,
                get_inactivity_penalty_deltas,
            )
            from consensoor.spec.constants import (
                TIMELY_SOURCE_FLAG_INDEX,
                TIMELY_TARGET_FLAG_INDEX,
                TIMELY_HEAD_FLAG_INDEX,
            )

            source_rewards, source_penalties = get_flag_index_deltas(pre_state, TIMELY_SOURCE_FLAG_INDEX)
            check_deltas("Source", source_rewards, source_penalties, source_deltas_file)

            target_rewards, target_penalties = get_flag_index_deltas(pre_state, TIMELY_TARGET_FLAG_INDEX)
            check_deltas("Target", target_rewards, target_penalties, target_deltas_file)

            head_rewards, head_penalties = get_flag_index_deltas(pre_state, TIMELY_HEAD_FLAG_INDEX)
            check_deltas("Head", head_rewards, head_penalties, head_deltas_file)

            inactivity_rewards, inactivity_penalties = get_inactivity_penalty_deltas(pre_state)
            check_deltas("Inactivity", inactivity_rewards, inactivity_penalties, inactivity_penalty_deltas_file)


def _pyspec_module(fork: str, preset: str):
    """Return the pyspec module (eth-consensus-specs) for the given fork+preset.

    Used by the fork-choice compliance runner. Returns None if pyspec is not
    installed or doesn't carry this fork.
    """
    try:
        import eth_consensus_specs  # noqa: F401
    except ImportError:
        return None
    import importlib
    try:
        return importlib.import_module(f"eth_consensus_specs.{fork}.{preset}")
    except ImportError:
        return None


def _load_signed_block_pyspec(spec_mod, path: Path):
    with open(path, "rb") as f:
        return spec_mod.SignedBeaconBlock.decode_bytes(snappy.decompress(f.read()))


def _load_attestation_pyspec(spec_mod, path: Path):
    with open(path, "rb") as f:
        return spec_mod.Attestation.decode_bytes(snappy.decompress(f.read()))


def _load_attester_slashing_pyspec(spec_mod, path: Path):
    with open(path, "rb") as f:
        return spec_mod.AttesterSlashing.decode_bytes(snappy.decompress(f.read()))


def _load_payload_attestation_pyspec(spec_mod, path: Path):
    with open(path, "rb") as f:
        return spec_mod.PayloadAttestationMessage.decode_bytes(snappy.decompress(f.read()))


def _load_signed_envelope_pyspec(spec_mod, path: Path):
    with open(path, "rb") as f:
        return spec_mod.SignedExecutionPayloadEnvelope.decode_bytes(snappy.decompress(f.read()))


class TestForkChoiceCompliance:
    """Fork-choice compliance tests (from `comptests.yml`).

    Drives the pyspec reference fork-choice store with the recorded events.
    This validates the compliance fixtures and the test infrastructure; it is
    not yet wired to consensoor's runtime fork choice (consensoor does not yet
    expose a spec-style Store with on_block/on_attestation/get_head).
    """

    def test_fork_choice_compliance(self, compliance_case, preset):
        case_id, fork, test_name, case_path = compliance_case

        spec_mod = _pyspec_module(fork, preset)
        if spec_mod is None:
            pytest.skip(f"pyspec module not available for {fork}/{preset}")

        meta = load_yaml(case_path / "meta.yaml") or {}
        bls_setting = int(meta.get("bls_setting", 0))
        if bls_setting != 1:
            from eth_consensus_specs.utils import bls as pyspec_bls
            pyspec_bls.bls_active = False

        steps = load_yaml(case_path / "steps.yaml")
        if not steps:
            pytest.skip(f"No steps for {case_id}")

        with open(case_path / "anchor_state.ssz_snappy", "rb") as f:
            anchor_state = spec_mod.BeaconState.decode_bytes(snappy.decompress(f.read()))
        with open(case_path / "anchor_block.ssz_snappy", "rb") as f:
            anchor_block = spec_mod.BeaconBlock.decode_bytes(snappy.decompress(f.read()))

        store = spec_mod.get_forkchoice_store(anchor_state, anchor_block)

        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                pytest.fail(f"{case_id} step {i}: unexpected step shape {step!r}")

            if "tick" in step:
                spec_mod.on_tick(store, step["tick"])
            elif "block" in step:
                ref = step["block"]
                valid = step.get("valid", True)
                block_file = case_path / f"{ref}.ssz_snappy"
                try:
                    signed = _load_signed_block_pyspec(spec_mod, block_file)
                    spec_mod.on_block(store, signed)
                    if not valid:
                        pytest.fail(f"{case_id} step {i}: block accepted but expected invalid")
                    # Mirror upstream compliance runner: replay block's body attestations
                    # and attester_slashings via on_*, silently skip individual failures.
                    for block_att in signed.message.body.attestations:
                        try:
                            spec_mod.on_attestation(store, block_att, is_from_block=True)
                        except AssertionError:
                            pass
                    for block_slash in signed.message.body.attester_slashings:
                        try:
                            spec_mod.on_attester_slashing(store, block_slash)
                        except AssertionError:
                            pass
                    # Post-Gloas: an on_block step also implies receiving the
                    # block's payload attestations, unpacked into individual
                    # PayloadAttestationMessages (mirrors upstream add_block).
                    if hasattr(signed.message.body, "payload_attestations"):
                        block_state = store.block_states[signed.message.hash_tree_root()]
                        for pa in signed.message.body.payload_attestations:
                            pa_slot = pa.data.slot
                            ptc = spec_mod.get_ptc(block_state, pa_slot)
                            bits = pa.aggregation_bits
                            for bit_i, v_index in enumerate(ptc):
                                if not bits[bit_i]:
                                    continue
                                ptc_message = spec_mod.PayloadAttestationMessage(
                                    validator_index=v_index,
                                    data=pa.data,
                                    signature=spec_mod.BLSSignature(),
                                )
                                try:
                                    spec_mod.on_payload_attestation_message(
                                        store, ptc_message, is_from_block=True
                                    )
                                except AssertionError:
                                    pass
                except (AssertionError, Exception) as e:
                    if valid:
                        raise AssertionError(f"{case_id} step {i}: block rejected unexpectedly: {e}") from e
            elif "attestation" in step:
                ref = step["attestation"]
                valid = step.get("valid", True)
                att_file = case_path / f"{ref}.ssz_snappy"
                try:
                    att = _load_attestation_pyspec(spec_mod, att_file)
                    spec_mod.on_attestation(store, att, is_from_block=False)
                    if not valid:
                        pytest.fail(f"{case_id} step {i}: attestation accepted but expected invalid")
                except (AssertionError, Exception) as e:
                    if valid:
                        raise AssertionError(f"{case_id} step {i}: attestation rejected unexpectedly: {e}") from e
            elif "attester_slashing" in step:
                ref = step["attester_slashing"]
                valid = step.get("valid", True)
                slashing_file = case_path / f"{ref}.ssz_snappy"
                try:
                    slashing = _load_attester_slashing_pyspec(spec_mod, slashing_file)
                    spec_mod.on_attester_slashing(store, slashing)
                    if not valid:
                        pytest.fail(f"{case_id} step {i}: attester_slashing accepted but expected invalid")
                except (AssertionError, Exception) as e:
                    if valid:
                        raise AssertionError(f"{case_id} step {i}: attester_slashing rejected unexpectedly: {e}") from e
            elif "payload_attestation" in step or "payload_attestation_message" in step:
                # Verb renamed payload_attestation -> payload_attestation_message
                # in newer compliance fixtures; both carry a PayloadAttestationMessage.
                ref = step.get("payload_attestation_message", step.get("payload_attestation"))
                valid = step.get("valid", True)
                pa_file = case_path / f"{ref}.ssz_snappy"
                try:
                    pa = _load_payload_attestation_pyspec(spec_mod, pa_file)
                    spec_mod.on_payload_attestation_message(store, pa, is_from_block=False)
                    if not valid:
                        pytest.fail(f"{case_id} step {i}: payload_attestation accepted but expected invalid")
                except (AssertionError, Exception) as e:
                    if valid:
                        raise AssertionError(f"{case_id} step {i}: payload_attestation rejected unexpectedly: {e}") from e
            elif "execution_payload" in step:
                ref = step["execution_payload"]
                valid = step.get("valid", True)
                envelope_file = case_path / f"{ref}.ssz_snappy"
                try:
                    envelope = _load_signed_envelope_pyspec(spec_mod, envelope_file)
                    spec_mod.on_execution_payload_envelope(store, envelope)
                    if not valid:
                        pytest.fail(f"{case_id} step {i}: execution_payload accepted but expected invalid")
                except (AssertionError, Exception) as e:
                    if valid:
                        raise AssertionError(f"{case_id} step {i}: execution_payload rejected unexpectedly: {e}") from e
            elif "checks" in step:
                checks = step["checks"]
                head_node = spec_mod.get_head(store)
                if isinstance(head_node, tuple):
                    head_root = bytes(head_node[0])
                    head_payload_status = int(head_node[1]) if len(head_node) > 1 else None
                else:
                    head_root = bytes(getattr(head_node, "root", head_node))
                    head_payload_status = int(getattr(head_node, "payload_status", 0))

                if "head" in checks:
                    expected_root = bytes.fromhex(checks["head"]["root"][2:])
                    assert head_root == expected_root, \
                        f"{case_id} step {i}: head root {head_root.hex()} != expected {expected_root.hex()}"
                    expected_slot = int(checks["head"]["slot"])
                    head_block = store.blocks[bytes(head_root)]
                    assert int(head_block.slot) == expected_slot, \
                        f"{case_id} step {i}: head slot {int(head_block.slot)} != expected {expected_slot}"

                if "head_payload_status" in checks and head_payload_status is not None:
                    expected_ps = int(checks["head_payload_status"])
                    assert head_payload_status == expected_ps, \
                        f"{case_id} step {i}: head_payload_status {head_payload_status} != expected {expected_ps}"

                if "justified_checkpoint" in checks:
                    jc = store.justified_checkpoint
                    exp = checks["justified_checkpoint"]
                    assert int(jc.epoch) == int(exp["epoch"]) and bytes(jc.root) == bytes.fromhex(exp["root"][2:]), \
                        f"{case_id} step {i}: justified_checkpoint mismatch ({int(jc.epoch)},{bytes(jc.root).hex()}) vs ({exp['epoch']},{exp['root']})"

                if "finalized_checkpoint" in checks:
                    fc = store.finalized_checkpoint
                    exp = checks["finalized_checkpoint"]
                    assert int(fc.epoch) == int(exp["epoch"]) and bytes(fc.root) == bytes.fromhex(exp["root"][2:]), \
                        f"{case_id} step {i}: finalized_checkpoint mismatch ({int(fc.epoch)},{bytes(fc.root).hex()}) vs ({exp['epoch']},{exp['root']})"

                if "proposer_boost_root" in checks:
                    expected_pbr = bytes.fromhex(checks["proposer_boost_root"][2:])
                    actual_pbr = bytes(store.proposer_boost_root)
                    assert actual_pbr == expected_pbr, \
                        f"{case_id} step {i}: proposer_boost_root {actual_pbr.hex()} != expected {expected_pbr.hex()}"
            else:
                pytest.fail(f"{case_id} step {i}: unknown step verb in {step!r}")



# ---------------------------------------------------------------------------
# Fork choice + fast confirmation vectors driven by consensoor's own Store
# ---------------------------------------------------------------------------

def discover_fork_choice_tests(spec_tests_dir: Path, handler_group: str):
    """Yield (case_id, case_path, fork, preset) for gloas fork_choice /
    fast_confirmation vectors (consensoor's Store is Gloas-only)."""
    cases = []
    base = spec_tests_dir / "gloas" / handler_group
    if not base.exists():
        return cases
    for handler_dir in sorted(base.iterdir()):
        tests_dir = handler_dir / "pyspec_tests"
        if not tests_dir.exists():
            continue
        for case_path in sorted(tests_dir.iterdir()):
            if (case_path / "steps.yaml").exists():
                cases.append((f"gloas/{handler_group}/{handler_dir.name}/{case_path.name}", case_path, "gloas", None))
    return cases


def _run_consensoor_fork_choice_case(case_id: str, case_path: Path, preset: str, with_fcr: bool):
    import snappy
    from consensoor.spec.constants import set_preset
    from consensoor.spec.network_config import load_config_from_upstream, set_config
    set_preset(preset)
    set_config(load_config_from_upstream(preset))
    from consensoor.spec import fork_choice as fc
    from consensoor.spec import fast_confirmation as fcr
    from consensoor.spec.types.gloas import (
        BeaconState, BeaconBlock, SignedBeaconBlock, Attestation, AttesterSlashing,
        SignedExecutionPayloadEnvelope, PayloadAttestationMessage,
    )

    def load(name, typ):
        with open(case_path / f"{name}.ssz_snappy", "rb") as f:
            return typ.decode_bytes(snappy.decompress(f.read()))

    with open(case_path / "anchor_state.ssz_snappy", "rb") as f:
        anchor_state = BeaconState.decode_bytes(snappy.decompress(f.read()))
    with open(case_path / "anchor_block.ssz_snappy", "rb") as f:
        anchor_block = BeaconBlock.decode_bytes(snappy.decompress(f.read()))
    store = fc.get_forkchoice_store(anchor_state, anchor_block)
    fcr_store = fcr.get_fast_confirmation_store(store) if with_fcr else None
    steps = load_yaml(case_path / "steps.yaml")
    assert steps, f"{case_id}: no steps"
    meta = load_yaml(case_path / "meta.yaml") or {}
    from consensoor.crypto import set_bls_verification
    bls_setting = int(meta.get("bls_setting", 1))
    set_bls_verification(bls_setting != 2)
    try:
        _run_steps(case_id, case_path, store, fcr_store, steps, with_fcr, load, fc, fcr,
                   SignedBeaconBlock, Attestation, AttesterSlashing, SignedExecutionPayloadEnvelope, PayloadAttestationMessage)
    finally:
        set_bls_verification(True)


def _run_steps(case_id, case_path, store, fcr_store, steps, with_fcr, load, fc, fcr,
               SignedBeaconBlock, Attestation, AttesterSlashing, SignedExecutionPayloadEnvelope, PayloadAttestationMessage):

    def expect(valid, fn, what, i):
        try:
            fn()
        except (AssertionError, Exception) as e:
            if valid:
                raise AssertionError(f"{case_id} step {i}: {what} rejected unexpectedly: {e}") from e
            return
        if not valid:
            pytest.fail(f"{case_id} step {i}: {what} accepted but expected invalid")

    # Fast-confirmation vectors list attestation steps in apply order, after
    # the tick that makes them past-slot (consensus-specs #5627), so every
    # step is replayed exactly as written.
    for i, step in enumerate(steps):
        if "tick" in step:
            fc.on_tick(store, int(step["tick"]))
        elif "block" in step:
            signed = load(step["block"], SignedBeaconBlock)
            def do_block():
                fc.on_block(store, signed)
                for att in signed.message.body.attestations:
                    try:
                        fc.on_attestation(store, att, is_from_block=True)
                    except AssertionError:
                        pass
                for sl in signed.message.body.attester_slashings:
                    try:
                        fc.on_attester_slashing(store, sl)
                    except AssertionError:
                        pass
            expect(step.get("valid", True), do_block, "block", i)
        elif "attestation" in step:
            att = load(step["attestation"], Attestation)
            expect(step.get("valid", True), lambda: fc.on_attestation(store, att, is_from_block=False), "attestation", i)
        elif "attester_slashing" in step:
            sl = load(step["attester_slashing"], AttesterSlashing)
            expect(step.get("valid", True), lambda: fc.on_attester_slashing(store, sl), "attester_slashing", i)
        elif "payload_attestation" in step or "payload_attestation_message" in step:
            ref = step.get("payload_attestation_message", step.get("payload_attestation"))
            pa = load(ref, PayloadAttestationMessage)
            expect(step.get("valid", True), lambda: fc.on_payload_attestation_message(store, pa, is_from_block=False), "payload_attestation", i)
        elif "execution_payload" in step:
            env = load(step["execution_payload"], SignedExecutionPayloadEnvelope)
            expect(step.get("valid", True), lambda: fc.on_execution_payload_envelope(store, env), "execution_payload", i)
        elif "checks" in step:
            checks = step["checks"]
            is_fcr_check = "confirmed_root" in checks
            if is_fcr_check and fcr_store is not None:
                fcr.on_fast_confirmation(fcr_store)
            head = fc.get_head(store)
            if "time" in checks:
                assert store.time == int(checks["time"]), f"{case_id} step {i}: time {store.time} != {checks['time']}"
            if "head" in checks:
                exp_root = bytes.fromhex(checks["head"]["root"][2:])
                assert head.root == exp_root, f"{case_id} step {i}: head {head.root.hex()} != {exp_root.hex()}"
                assert int(store.blocks[head.root].slot) == int(checks["head"]["slot"]), f"{case_id} step {i}: head slot"
                if "payload_status" in checks["head"]:
                    assert head.payload_status == int(checks["head"]["payload_status"]), \
                        f"{case_id} step {i}: head payload_status {head.payload_status} != {checks['head']['payload_status']}"
            for key, attr in (("justified_checkpoint", "justified_checkpoint"), ("finalized_checkpoint", "finalized_checkpoint")):
                if key in checks:
                    c = getattr(store, attr); exp = checks[key]
                    assert c.epoch == int(exp["epoch"]) and c.root == bytes.fromhex(exp["root"][2:]), \
                        f"{case_id} step {i}: {key} ({c.epoch},{c.root.hex()}) != ({exp['epoch']},{exp['root']})"
            if "proposer_boost_root" in checks:
                assert store.proposer_boost_root == bytes.fromhex(checks["proposer_boost_root"][2:]), \
                    f"{case_id} step {i}: proposer_boost_root {store.proposer_boost_root.hex()}"
            if "get_proposer_head" in checks:
                exp = bytes.fromhex(checks["get_proposer_head"][2:])
                got = fc.get_proposer_head(store, head, fc.get_current_slot(store)).root
                assert got == exp, f"{case_id} step {i}: get_proposer_head {got.hex()} != {exp.hex()}"
            if "should_override_forkchoice_update" in checks:
                pass  # not implemented in consensoor (builder override is n/a for self-build)
            if "viable_for_head_roots_and_weights" in checks:
                # Mirror pyspec's get_viable_for_head_checks: the leaves of the
                # filtered (root, payload_status) node tree with their weights.
                filtered = fc.get_filtered_node_tree(store)
                pending = [fc.ForkChoiceNode(root=store.justified_checkpoint.root, payload_status=fc.PAYLOAD_STATUS_PENDING)]
                leaves = []
                while pending:
                    node = pending.pop()
                    children = [c for c in fc.get_node_children(store, node) if c in filtered]
                    if not children:
                        leaves.append(node)
                    else:
                        pending.extend(children)
                got = {(n.root, n.payload_status): fc.get_weight(store, n) for n in leaves}
                exp = {(bytes.fromhex(e["root"][2:]), int(e.get("payload_status", fc.PAYLOAD_STATUS_PENDING))): int(e["weight"])
                       for e in checks["viable_for_head_roots_and_weights"]}
                assert got == exp, f"{case_id} step {i}: viable leaves {[(r.hex()[:8], ps, w) for (r, ps), w in got.items()]} != {[(r.hex()[:8], ps, w) for (r, ps), w in exp.items()]}"
            if is_fcr_check and fcr_store is not None:
                for key in ("previous_epoch_observed_justified_checkpoint", "current_epoch_observed_justified_checkpoint",
                            "previous_epoch_greatest_unrealized_checkpoint"):
                    c = getattr(fcr_store, key); exp = checks[key]
                    assert c.epoch == int(exp["epoch"]) and c.root == bytes.fromhex(exp["root"][2:]), \
                        f"{case_id} step {i}: {key} ({c.epoch},{c.root.hex()[:12]}) != ({exp['epoch']},{exp['root'][:14]})"
                for key in ("previous_slot_head", "current_slot_head", "confirmed_root"):
                    exp = bytes.fromhex(checks[key][2:]); got = getattr(fcr_store, key)
                    assert got == exp, f"{case_id} step {i}: {key} {got.hex()[:12]} != {exp.hex()[:12]}"
                if "safe_execution_block_hash" in checks:
                    exp = bytes.fromhex(checks["safe_execution_block_hash"][2:])
                    got = fcr.get_safe_execution_block_hash(fcr_store)
                    assert got == exp, f"{case_id} step {i}: safe_execution_block_hash {got.hex()[:12]} != {exp.hex()[:12]}"
        else:
            pytest.fail(f"{case_id} step {i}: unknown step {step!r}")


@pytest.mark.fork_choice
class TestForkChoiceConsensoor:
    """Gloas fork_choice vectors against consensoor.spec.fork_choice."""

    def test_fork_choice(self, fork_choice_case, preset):
        case_id, case_path, fork, _ = fork_choice_case
        _run_consensoor_fork_choice_case(case_id, case_path, preset, with_fcr=False)


@pytest.mark.fork_choice
class TestFastConfirmationConsensoor:
    """Gloas fast_confirmation vectors against consensoor.spec.fast_confirmation."""

    def test_fast_confirmation(self, fast_confirmation_case, preset):
        case_id, case_path, fork, _ = fast_confirmation_case
        _run_consensoor_fork_choice_case(case_id, case_path, preset, with_fcr=True)


# ---------------------------------------------------------------------------
# Mid-chain anchor replay (node scenario, not a spec vector)
#
# A running node anchors its Store on the first block it fully imports —
# typically the first block of an epoch — not on genesis. The dependent root
# of that epoch and the PTC votes for the anchor slot then reference state
# *below* the anchor, which the spec functions never see in the reference
# vectors. Replay the tail of a fast_confirmation vector from such an anchor
# and require that get_head / on_fast_confirmation keep working and agree with
# the genesis-anchored reference store.
# ---------------------------------------------------------------------------

def _select_mid_chain_anchor_cases(spec_tests_dir: Path, limit: int = 3):
    """The FCR vectors with the most block steps (longest linear chains)."""
    from consensoor.spec.constants import SLOTS_PER_EPOCH
    scored = []
    for case_id, case_path, _fork, _p in discover_fork_choice_tests(spec_tests_dir, "fast_confirmation"):
        steps = load_yaml(case_path / "steps.yaml") or []
        n_blocks = sum(1 for s in steps if "block" in s)
        if n_blocks >= 3 * SLOTS_PER_EPOCH():
            scored.append((n_blocks, case_id, case_path))
    scored.sort(reverse=True)
    return [(cid, path) for _n, cid, path in scored[:limit]]


def _replay_from_mid_chain_anchor(case_id: str, case_path: Path, preset: str):
    from consensoor.spec.constants import set_preset
    from consensoor.spec.network_config import load_config_from_upstream, set_config
    set_preset(preset)
    set_config(load_config_from_upstream(preset))
    from consensoor.crypto import hash_tree_root, set_bls_verification
    from consensoor.spec import fork_choice as fc
    from consensoor.spec import fast_confirmation as fcr
    from consensoor.spec.types.gloas import (
        BeaconState, BeaconBlock, SignedBeaconBlock, Attestation, AttesterSlashing,
        SignedExecutionPayloadEnvelope, PayloadAttestationMessage,
    )

    def load(name, typ):
        with open(case_path / f"{name}.ssz_snappy", "rb") as f:
            return typ.decode_bytes(snappy.decompress(f.read()))

    anchor_state = load("anchor_state", BeaconState)
    anchor_block = load("anchor_block", BeaconBlock)
    steps = load_yaml(case_path / "steps.yaml")
    meta = load_yaml(case_path / "meta.yaml") or {}
    set_bls_verification(int(meta.get("bls_setting", 1)) != 2)
    try:
        # Pass 1: reference store from genesis (also yields every post-state).
        ref = fc.get_forkchoice_store(anchor_state, anchor_block)
        ref_fcr = fcr.get_fast_confirmation_store(ref)
        _run_steps(case_id, case_path, ref, ref_fcr, steps, True, load, fc, fcr,
                   SignedBeaconBlock, Attestation, AttesterSlashing, SignedExecutionPayloadEnvelope,
                   PayloadAttestationMessage)

        # Anchor: the first epoch-start block that still leaves >= 1 epoch to replay.
        from consensoor.spec.constants import SLOTS_PER_EPOCH
        spe = SLOTS_PER_EPOCH()
        max_slot = max(int(b.slot) for b in ref.blocks.values())
        candidates = sorted(
            (int(b.slot), root) for root, b in ref.blocks.items()
            if int(b.slot) >= spe and int(b.slot) % spe == 0 and int(b.slot) + spe <= max_slot
        )
        if not candidates:
            pytest.skip(f"{case_id}: no epoch-start block with an epoch of chain after it")
        anchor_slot, anchor_root = candidates[0]

        # Pass 2: replay the same steps into a store anchored at that block.
        store = fcr_store = None
        queued = []
        ticks_after_anchor = 0
        for i, step in enumerate(steps):
            if "tick" in step:
                if store is None:
                    continue
                fc.on_tick(store, int(step["tick"]))
                for att in queued:
                    try:
                        fc.on_attestation(store, att, is_from_block=False)
                    except AssertionError:
                        pass  # pre-anchor target / future slot — same as the node drops them
                queued = []
                # This is exactly what the node does on every slot tick.
                fcr.on_fast_confirmation(fcr_store)
                head = fc.get_head(store)
                assert head.root in store.blocks, f"{case_id} step {i}: head not in store"
                ticks_after_anchor += 1
            elif "block" in step:
                signed = load(step["block"], SignedBeaconBlock)
                root = hash_tree_root(signed.message)
                post_state = ref.block_states.get(root)
                if post_state is None:
                    continue  # invalid block in the vector
                if store is None:
                    if root != anchor_root:
                        continue
                    store = fc.get_forkchoice_store(post_state, signed.message)
                    store.time = ref.time
                    fcr_store = fcr.get_fast_confirmation_store(store)
                    continue
                assert fc.on_block_with_state(store, signed, post_state), f"{case_id} step {i}: parent unknown"
                for att in signed.message.body.attestations:
                    try:
                        fc.on_attestation(store, att, is_from_block=True)
                    except AssertionError:
                        pass
            elif "attestation" in step:
                if store is not None and step.get("valid", True):
                    queued.append(load(step["attestation"], Attestation))
            elif "payload_attestation" in step or "payload_attestation_message" in step:
                if store is not None and step.get("valid", True):
                    pa = load(step.get("payload_attestation_message", step.get("payload_attestation")),
                              PayloadAttestationMessage)
                    try:
                        fc.on_payload_attestation_message(store, pa, is_from_block=False)
                    except AssertionError:
                        pass
            elif "execution_payload" in step:
                if store is not None and step.get("valid", True):
                    fc.on_execution_payload_envelope_trusted(store, load(step["execution_payload"], SignedExecutionPayloadEnvelope))

        assert store is not None and ticks_after_anchor > 0, f"{case_id}: anchor block never replayed"
        ref_head, head = fc.get_head(ref), fc.get_head(store)
        assert (head.root, head.payload_status) == (ref_head.root, ref_head.payload_status), (
            f"{case_id}: anchored@{anchor_slot} head {head} != reference head {ref_head}"
        )
        confirmed_slot = int(store.blocks[fcr_store.confirmed_root].slot)
        assert confirmed_slot >= anchor_slot, f"{case_id}: confirmed root left the anchored store"
        if int(ref.blocks[ref_fcr.confirmed_root].slot) >= anchor_slot:
            assert fcr_store.confirmed_root == ref_fcr.confirmed_root, f"{case_id}: confirmed root differs from reference"
    finally:
        set_bls_verification(True)


@pytest.mark.fork_choice
class TestMidChainAnchorReplay:
    """Store anchored on an epoch-start block mid-chain (how the node anchors)."""

    def test_replay_from_mid_chain_anchor(self, spec_tests_dir, preset):
        cases = _select_mid_chain_anchor_cases(spec_tests_dir)
        if not cases:
            pytest.skip("no gloas fast_confirmation vectors with >= 3 epochs of blocks")
        for case_id, case_path in cases:
            _replay_from_mid_chain_anchor(case_id, case_path, preset)
