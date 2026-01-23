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

    payload_types = {
        "bellatrix": ExecutionPayloadBellatrix,
        "capella": ExecutionPayloadCapella,
        "deneb": ExecutionPayload,
        "electra": ExecutionPayload,
        "fulu": ExecutionPayload,
        "gloas": ExecutionPayload,
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

    electra_aggregate_types = {"AggregateAndProof", "SignedAggregateAndProof"}
    if fork in {"electra", "fulu", "gloas"} and type_name in electra_aggregate_types:
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
        if fork in {"electra", "fulu", "gloas"}:
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

    op_type_map = {
        "attestation": types.Phase0Attestation if fork in pre_electra_forks else types.Attestation,
        "attester_slashing": types.Phase0AttesterSlashing if fork in pre_electra_forks else types.AttesterSlashing,
        "proposer_slashing": types.ProposerSlashing,
        "deposit": types.Deposit,
        "voluntary_exit": types.SignedVoluntaryExit,
        "block_header": None,
        "bls_to_execution_change": types.SignedBLSToExecutionChange if hasattr(types, "SignedBLSToExecutionChange") else None,
        "sync_aggregate": types.SyncAggregate if hasattr(types, "SyncAggregate") else None,
        "execution_payload": None,
        "withdrawals": None,
        "deposit_request": types.DepositRequest if hasattr(types, "DepositRequest") else None,
        "withdrawal_request": types.WithdrawalRequest if hasattr(types, "WithdrawalRequest") else None,
        "consolidation_request": types.ConsolidationRequest if hasattr(types, "ConsolidationRequest") else None,
        "execution_payload_bid": None,  # Uses block.ssz_snappy, special handling
        "payload_attestation": types.PayloadAttestation if hasattr(types, "PayloadAttestation") else None,
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
        process_payload_attestation,
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
        "block_header": process_block_header,
        "bls_to_execution_change": process_bls_to_execution_change,
        "sync_aggregate": process_sync_aggregate,
        "execution_payload": process_execution_payload,
        "withdrawals": process_withdrawals,
        "deposit_request": process_deposit_request,
        "withdrawal_request": process_withdrawal_request,
        "consolidation_request": process_consolidation_request,
        "execution_payload_bid": process_execution_payload_bid,
        "payload_attestation": process_payload_attestation,
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
        "pending_consolidations": process_pending_consolidations,
        "proposer_lookahead": process_proposer_lookahead,
        "builder_pending_payments": process_builder_pending_payments,
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
            for ssz_file in type_dir.rglob("serialized.ssz_snappy"):
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


def pytest_generate_tests(metafunc):
    """Generate test cases from spec test directories."""
    spec_tests_dir = get_spec_tests_dir(metafunc.config)
    if not spec_tests_dir.exists():
        return

    if "ssz_case" in metafunc.fixturenames:
        test_cases = discover_ssz_static_tests(spec_tests_dir)
        if test_cases:
            ids = [tc[0] for tc in test_cases]
            metafunc.parametrize("ssz_case", test_cases, ids=ids)

    if "operations_case" in metafunc.fixturenames:
        test_cases = discover_operations_tests(spec_tests_dir)
        if test_cases:
            ids = [tc[0] for tc in test_cases]
            metafunc.parametrize("operations_case", test_cases, ids=ids)

    if "epoch_case" in metafunc.fixturenames:
        test_cases = discover_epoch_processing_tests(spec_tests_dir)
        if test_cases:
            ids = [tc[0] for tc in test_cases]
            metafunc.parametrize("epoch_case", test_cases, ids=ids)

    if "sanity_blocks_case" in metafunc.fixturenames:
        test_cases = discover_sanity_blocks_tests(spec_tests_dir)
        if test_cases:
            ids = [tc[0] for tc in test_cases]
            metafunc.parametrize("sanity_blocks_case", test_cases, ids=ids)

    if "sanity_slots_case" in metafunc.fixturenames:
        test_cases = discover_sanity_slots_tests(spec_tests_dir)
        if test_cases:
            ids = [tc[0] for tc in test_cases]
            metafunc.parametrize("sanity_slots_case", test_cases, ids=ids)

    if "finality_case" in metafunc.fixturenames:
        test_cases = discover_finality_tests(spec_tests_dir)
        if test_cases:
            ids = [tc[0] for tc in test_cases]
            metafunc.parametrize("finality_case", test_cases, ids=ids)

    if "rewards_case" in metafunc.fixturenames:
        test_cases = discover_rewards_tests(spec_tests_dir)
        if test_cases:
            ids = [tc[0] for tc in test_cases]
            metafunc.parametrize("rewards_case", test_cases, ids=ids)

    if "shuffling_case" in metafunc.fixturenames:
        test_cases = discover_shuffling_tests(spec_tests_dir)
        if test_cases:
            ids = [tc[0] for tc in test_cases]
            metafunc.parametrize("shuffling_case", test_cases, ids=ids)

    if "random_case" in metafunc.fixturenames:
        test_cases = discover_random_tests(spec_tests_dir)
        if test_cases:
            ids = [tc[0] for tc in test_cases]
            metafunc.parametrize("random_case", test_cases, ids=ids)


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
            elif op_name in ("execution_payload", "withdrawals", "execution_payload_bid"):
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
            elif op_name == "execution_payload_bid":
                # Load unsigned block and extract signed_execution_payload_bid from body
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
            process_slots(state, target_slot)
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
        from consensoor.spec.constants import SHUFFLE_ROUND_COUNT

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

        from consensoor.spec.types.base import List, uint64
        from consensoor.spec.constants import VALIDATOR_REGISTRY_LIMIT

        class Deltas:
            def __init__(self, rewards, penalties):
                self.rewards = rewards
                self.penalties = penalties

        if fork in {"phase0"}:
            pytest.skip(f"Rewards tests for {fork} require legacy reward calculation")
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

            if source_deltas_file.exists():
                DeltasList = List[List[uint64, VALIDATOR_REGISTRY_LIMIT], 2]
                expected = load_ssz_snappy(source_deltas_file, DeltasList)
                actual_rewards, actual_penalties = get_flag_index_deltas(pre_state, TIMELY_SOURCE_FLAG_INDEX)
                for i in range(len(pre_state.validators)):
                    if i < len(expected[0]) and i < len(expected[1]):
                        assert actual_rewards[i] == int(expected[0][i]), \
                            f"Source reward mismatch at {i}: {actual_rewards[i]} != {expected[0][i]}"
                        assert actual_penalties[i] == int(expected[1][i]), \
                            f"Source penalty mismatch at {i}: {actual_penalties[i]} != {expected[1][i]}"
