"""Generate Heze EIP-8198 differential vectors from pyspec (minimal)."""
import json, random, sys
from pathlib import Path
from eth_consensus_specs.heze import minimal as base_spec
from eth_consensus_specs.test.context import spec_with_config_overrides, get_copy_of_spec
from eth_consensus_specs.test.helpers.genesis import create_genesis_state
from eth_consensus_specs.utils import bls

bls.bls_active = False
OUT = Path(sys.argv[1])
FORK_EPOCH = 3
SLOT_DURATION_MS_HEZE = 4000
# Override in place: get_copy_of_spec() leaves cache_this-wrapped functions
# (get_base_reward, ...) closed over the original config, so they would
# silently ignore the slot duration overrides.
spec, _ = spec_with_config_overrides(base_spec, {
    "SLOT_DURATION_MS_HEZE": SLOT_DURATION_MS_HEZE, "HEZE_FORK_EPOCH": FORK_EPOCH,
    "MIN_BLOB_DATA_RETENTION_MS": 196608000 // 4096 * 4,  # small window to exercise it
})
SPE = spec.SLOTS_PER_EPOCH
meta = {
    "heze_fork_epoch": FORK_EPOCH,
    "slot_duration_ms_heze": SLOT_DURATION_MS_HEZE,
    "min_blob_data_retention_ms": int(spec.config.MIN_BLOB_DATA_RETENTION_MS),
}

# --- time helpers ---------------------------------------------------------
times = []
g = 1_000_000
for slot in list(range(0, 10 * SPE)) + [1000, 12345]:
    times.append(["time_at_slot", slot, int(spec.compute_time_at_slot_ms(g, slot))])
for t in range(g, g + 10 * SPE * 6000, 777):
    times.append(["slot_at_time", t, int(spec.compute_slot_at_time_ms(g, t))])
for e in range(0, 40):
    times.append(["retention_start", e, int(spec.compute_blob_data_retention_start_epoch(e))])
meta["times"] = times

# --- state-based cases ------------------------------------------------------
rng = random.Random(8198)
balances = [spec.MAX_EFFECTIVE_BALANCE] * 64
from eth_consensus_specs.heze import minimal as heze_spec
base = spec.BeaconState.decode_bytes(
    create_genesis_state(heze_spec, balances, spec.MAX_EFFECTIVE_BALANCE).encode_bytes())

def prep(epoch, leak):
    st = base.copy()
    st.slot = epoch * SPE
    for i in range(len(st.validators)):
        st.previous_epoch_participation[i] = rng.randrange(8)
        st.current_epoch_participation[i] = rng.randrange(8)
        st.inactivity_scores[i] = rng.randrange(0, 200)
        st.balances[i] = spec.MAX_EFFECTIVE_BALANCE - rng.randrange(0, 10**9)
    if not leak:
        st.finalized_checkpoint = spec.Checkpoint(epoch=max(0, epoch - 2), root=b"\x11" * 32)
    return st

cases = []
for epoch in (2, 3, 4, 6, 7, 12):
    for leak in (False, True):
        name = f"rewards_epoch{epoch}_{'leak' if leak else 'noleak'}"
        pre = prep(epoch, leak)
        post = pre.copy()
        spec.process_rewards_and_penalties(post)
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "pre.ssz").write_bytes(pre.encode_bytes())
        (d / "post.ssz").write_bytes(post.encode_bytes())
        cases.append({"name": name, "kind": "rewards"})
        # churn values at this state
        cases.append({
            "name": f"churn_epoch{epoch}_{leak}", "kind": "churn", "pre": name,
            "exit": int(spec.get_exit_churn_limit(pre)),
            "activation": int(spec.get_activation_churn_limit(pre)),
            "consolidation": int(spec.get_consolidation_churn_limit(pre)),
            "base_reward_per_increment": int(spec.get_base_reward_per_increment(pre, spec.get_current_epoch(pre))),
        })
    # sync aggregate (bls off): mixed participation
    name = f"sync_aggregate_epoch{epoch}"
    pre = prep(epoch, False)
    pre.slot = epoch * SPE + 1
    bits = [rng.random() < 0.7 for _ in range(spec.SYNC_COMMITTEE_SIZE)]
    agg = spec.SyncAggregate(sync_committee_bits=spec.SyncCommitteeBits(data=bits), sync_committee_signature=spec.G2_POINT_AT_INFINITY)
    post = pre.copy()
    spec.process_sync_aggregate(post, agg)
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "pre.ssz").write_bytes(pre.encode_bytes())
    (d / "post.ssz").write_bytes(post.encode_bytes())
    (d / "sync_aggregate.ssz").write_bytes(agg.encode_bytes())
    cases.append({"name": name, "kind": "sync_aggregate"})

meta["cases"] = cases
(OUT / "meta.json").write_text(json.dumps(meta))
print("ok", len(cases), len(times))
