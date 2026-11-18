"""PQ fork (leanXMSS) types, helpers and state transition."""

from pathlib import Path

import pytest

from consensoor.crypto import xmss

pytestmark = pytest.mark.skipif(not xmss.available(), reason="leanXMSS library not built")


@pytest.fixture(scope="module")
def pq_config():
    from consensoor.spec.constants import set_preset
    from consensoor.spec.network_config import NetworkConfig, get_config, set_config

    set_preset("minimal")
    prev = get_config()
    config = NetworkConfig.from_yaml(Path(__file__).parents[1] / "spec" / "configs" / "minimal.yaml")
    for attr in ("altair", "bellatrix", "capella", "deneb", "electra", "fulu", "gloas", "pq"):
        setattr(config, f"{attr}_fork_epoch", 0)
    set_config(config)
    yield config
    set_config(prev)


@pytest.fixture(scope="module")
def genesis(pq_config):
    from . import pq_helpers

    return pq_helpers.build_genesis_state()


def test_chain_finalizes(genesis):
    from consensoor.crypto import hash_tree_root
    from consensoor.spec.constants import SLOTS_PER_EPOCH
    from consensoor.spec.state_transition import state_transition
    from . import pq_helpers

    state = genesis
    pending = {}
    for slot in range(1, 4 * SLOTS_PER_EPOCH() + 2):
        attestations = pending.pop(slot - 2, [])
        signed_block, post = pq_helpers.build_block(state, slot, attestations)
        state = state_transition(state, signed_block, validate_result=True)
        assert hash_tree_root(state) == hash_tree_root(post)
        head_root = hash_tree_root(signed_block.message)
        pending[slot] = pq_helpers.sign_attestations(state, slot, head_root)

    assert int(state.finalized_checkpoint.epoch) >= 2
