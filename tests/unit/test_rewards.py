"""Beacon-API rewards computations checked against the state transition."""
import json
import os
from functools import reduce
from math import gcd
from pathlib import Path

import pytest
import snappy

FIX = Path(__file__).parent / "fixtures" / "eip8198"
SPEC_TESTS = Path(os.environ.get("SPEC_TESTS_DIR", Path(__file__).parents[1] / "spec-tests" / "tests" / "minimal"))
FORKS = ("altair", "bellatrix", "capella", "deneb", "electra", "fulu", "gloas")


@pytest.fixture
def minimal():
    from consensoor.spec.constants import set_preset
    from consensoor.spec.network_config import NetworkConfig, set_config, get_config

    set_preset("minimal")
    from consensoor.spec.state_transition.helpers import clear_spec_caches
    prev = get_config()
    config = NetworkConfig.from_yaml(Path(__file__).parents[1] / "spec" / "configs" / "minimal.yaml")
    for attr in FORKS:
        setattr(config, f"{attr}_fork_epoch", 0)
    set_config(config)
    clear_spec_caches()
    yield config
    clear_spec_caches()
    set_config(prev)


def _load(path: Path, typ):
    return typ.decode_bytes(snappy.decompress(path.read_bytes()))


def test_attestation_rewards_match_epoch_processing(minimal):
    from consensoor.beacon_api.rewards import compute_attestation_rewards
    from consensoor.spec.types.heze import BeaconState
    from consensoor.spec.state_transition.epoch import (
        process_inactivity_updates,
        process_justification_and_finalization,
        process_rewards_and_penalties,
    )

    meta = json.loads((FIX / "meta.json").read_text())
    minimal.slot_duration_ms_heze = meta["slot_duration_ms_heze"]
    minimal.heze_fork_epoch = meta["heze_fork_epoch"]
    cases = [c for c in meta["cases"] if c["kind"] == "rewards"]
    assert cases
    for case in cases:
        state = _load(FIX / case["name"] / "pre.ssz_snappy", BeaconState)
        rewards = compute_attestation_rewards(state, None)

        expected = state.copy()
        process_justification_and_finalization(expected)
        process_inactivity_updates(expected)
        before = [int(b) for b in expected.balances]
        process_rewards_and_penalties(expected)

        totals = {r["validator_index"]: r for r in rewards["total_rewards"]}
        assert totals
        for index, entry in totals.items():
            delta = entry["head"] + entry["target"] + entry["source"] + entry["inactivity"]
            assert max(0, before[index] + delta) == int(expected.balances[index]), (case["name"], index)
            assert "inclusion_delay" not in entry

        ideal = {r["effective_balance"]: r for r in rewards["ideal_rewards"]}
        for index, entry in totals.items():
            best = ideal[int(state.validators[index].effective_balance)]
            for name in ("head", "target", "source"):
                assert entry[name] <= best[name]

        subset = compute_attestation_rewards(state, {next(iter(totals))})
        assert [r["validator_index"] for r in subset["total_rewards"]] == [next(iter(totals))]


def _sanity_case(name: str) -> Path:
    path = SPEC_TESTS / "gloas" / "sanity" / "blocks" / "pyspec_tests" / name
    if not path.exists():
        pytest.skip(f"spec test vectors not found at {path}")
    return path


@pytest.mark.parametrize("name", ["sync_committee_committee__half", "sync_committee_committee__full"])
def test_block_and_sync_committee_rewards(minimal, name):
    from consensoor.beacon_api.rewards import compute_block_rewards, compute_sync_committee_rewards
    from consensoor.spec.types.gloas import BeaconState, SignedBeaconBlock
    from consensoor.spec.state_transition import process_slots
    from consensoor.spec.state_transition.transition import process_block

    case = _sanity_case(name)
    pre = _load(case / "pre.ssz_snappy", BeaconState)
    signed_block = _load(case / "blocks_0.ssz_snappy", SignedBeaconBlock)
    block = signed_block.message
    proposer = int(block.proposer_index)

    rewards = compute_block_rewards(pre, signed_block)
    sync_rewards = compute_sync_committee_rewards(pre, signed_block, None)
    assert int(pre.slot) < int(block.slot)

    state = process_slots(pre.copy(), int(block.slot))
    before = int(state.balances[proposer])
    process_block(state, block)
    own_sync = sum(r["reward"] for r in sync_rewards if r["validator_index"] == proposer)

    participants = sum(1 for bit in block.body.sync_aggregate.sync_committee_bits if bit)
    assert participants
    assert rewards["proposer_index"] == proposer
    assert rewards["sync_aggregate"] > 0
    assert rewards["sync_aggregate"] % participants == 0
    assert rewards["total"] == (
        rewards["attestations"] + rewards["sync_aggregate"]
        + rewards["proposer_slashings"] + rewards["attester_slashings"]
    )
    assert int(state.balances[proposer]) - before == rewards["total"] + own_sync

    participant_reward = reduce(gcd, (abs(r["reward"]) for r in sync_rewards))
    assert sum(r["reward"] for r in sync_rewards) == (2 * participants - len(
        block.body.sync_aggregate.sync_committee_bits
    )) * participant_reward

    one = sync_rewards[0]["validator_index"]
    assert [r["validator_index"] for r in compute_sync_committee_rewards(pre, signed_block, {one})] == [one]
