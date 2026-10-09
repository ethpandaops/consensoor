"""EIP-8198 (quick slots) differential tests against pyspec master.

Fixtures come from tests/tools/gen_eip8198_vectors.py run inside a
consensus-specs checkout (``uv run python tests/tools/gen_eip8198_vectors.py
<out>``): 6000 ms slots until EIP8198_FORK_EPOCH = 3, then SLOT_DURATION_MS_EIP8198 = 4000 ms.
Upstream does not publish eip8198 reference tests yet.
"""
import json
from pathlib import Path

import pytest
import snappy

FIX = Path(__file__).parent / "fixtures" / "eip8198"


@pytest.fixture(scope="module")
def cfg():
    from consensoor.spec.constants import set_preset
    from consensoor.spec.network_config import NetworkConfig, set_config, get_config

    set_preset("minimal")
    prev = get_config()
    meta = json.loads((FIX / "meta.json").read_text())
    config = NetworkConfig.from_yaml(Path(__file__).parents[1] / "spec" / "configs" / "minimal.yaml")
    config.slot_duration_ms_eip8198 = meta["slot_duration_ms_eip8198"]
    config.min_blob_data_retention_ms = meta["min_blob_data_retention_ms"]
    for attr in ("altair", "bellatrix", "capella", "deneb", "electra", "fulu", "gloas", "heze"):
        setattr(config, f"{attr}_fork_epoch", 0)
    config.eip8198_fork_epoch = meta["eip8198_fork_epoch"]
    set_config(config)
    yield meta
    set_config(prev)


def _load(name, typ):
    return typ.decode_bytes(snappy.decompress((FIX / name).read_bytes()))


def test_time_helpers(cfg):
    from consensoor.spec.network_config import get_config
    config = get_config()
    g = 1_000_000
    for kind, arg, expected in cfg["times"]:
        if kind == "time_at_slot":
            assert config.compute_time_at_slot_ms(g, arg) == expected, (kind, arg)
        elif kind == "slot_at_time":
            assert config.compute_slot_at_time_ms(g, arg) == expected, (kind, arg)
        else:
            assert config.compute_blob_data_retention_start_epoch(arg) == expected, (kind, arg)


def test_rewards_and_penalties(cfg):
    from consensoor.spec.types.heze import BeaconState
    from consensoor.spec.state_transition.epoch import process_rewards_and_penalties
    from consensoor.spec.state_transition.helpers import clear_spec_caches

    cases = [c for c in cfg["cases"] if c["kind"] == "rewards"]
    assert cases
    for case in cases:
        clear_spec_caches()
        state = _load(f"{case['name']}/pre.ssz_snappy", BeaconState)
        expected = _load(f"{case['name']}/post.ssz_snappy", BeaconState)
        process_rewards_and_penalties(state)
        assert state.hash_tree_root() == expected.hash_tree_root(), case["name"]


def test_churn_and_base_reward(cfg):
    from consensoor.spec.types.heze import BeaconState
    from consensoor.spec.state_transition.helpers import clear_spec_caches
    from consensoor.spec.state_transition.helpers.accessors import (
        get_exit_churn_limit, get_activation_churn_limit_gloas,
        get_consolidation_churn_limit, get_base_reward_per_increment,
    )

    for case in (c for c in cfg["cases"] if c["kind"] == "churn"):
        clear_spec_caches()
        state = _load(f"{case['pre']}/pre.ssz_snappy", BeaconState)
        assert get_exit_churn_limit(state) == case["exit"], case["name"]
        assert get_activation_churn_limit_gloas(state) == case["activation"], case["name"]
        assert get_consolidation_churn_limit(state) == case["consolidation"], case["name"]
        assert get_base_reward_per_increment(state) == case["base_reward_per_increment"], case["name"]


def test_sync_aggregate_rewards(cfg):
    from consensoor.crypto import set_bls_verification, bls_verification_enabled
    from consensoor.spec.types.heze import BeaconState
    from consensoor.spec.types import SyncAggregate
    from consensoor.spec.state_transition.block import process_sync_aggregate
    from consensoor.spec.state_transition.helpers import clear_spec_caches

    was = bls_verification_enabled()
    set_bls_verification(False)
    try:
        for case in (c for c in cfg["cases"] if c["kind"] == "sync_aggregate"):
            clear_spec_caches()
            state = _load(f"{case['name']}/pre.ssz_snappy", BeaconState)
            agg = _load(f"{case['name']}/sync_aggregate.ssz_snappy", SyncAggregate)
            expected = _load(f"{case['name']}/post.ssz_snappy", BeaconState)
            process_sync_aggregate(state, agg)
            assert state.hash_tree_root() == expected.hash_tree_root(), case["name"]
    finally:
        set_bls_verification(was)
