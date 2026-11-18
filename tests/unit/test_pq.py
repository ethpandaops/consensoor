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


def test_block_builder_and_validator_client(genesis, pq_config, tmp_path):
    import asyncio
    import hashlib

    from consensoor.builder import BlockBuilder
    from consensoor.crypto import hash_tree_root
    from consensoor.operation_pool import OperationPool
    from consensoor.pq_attestation_pool import PQAttestationPool
    from consensoor.spec.constants import SLOTS_PER_EPOCH
    from consensoor.spec.state_transition import process_slots, state_transition
    from consensoor.spec.state_transition.helpers.accessors import get_current_epoch, get_randao_mix
    from consensoor.spec.state_transition.helpers.beacon_committee import get_beacon_committee
    from consensoor.spec.state_transition.helpers.pq import (
        get_aggregation_committee,
        get_attestation_statements,
    )
    from consensoor.validator import ValidatorClient, ValidatorKey, xmss_signer
    from . import pq_helpers

    xmss_signer.configure(str(tmp_path))
    keys = [
        ValidatorKey(pubkey=i.to_bytes(48, "big"), privkey=pq_helpers.privkey(i))
        for i in range(pq_helpers.VALIDATOR_COUNT)
    ]
    client = ValidatorClient(keys)
    client.use_xmss_keys()
    client.update_validator_indices(genesis)
    assert sorted(k.validator_index for k in client.keys.values()) == list(range(len(keys)))

    class Store:
        @staticmethod
        def get_payload(root):
            return None

    class Config:
        graffiti_bytes = b"\x00" * 32

    class Node:
        state = genesis
        config = Config()
        store = Store()
        operation_pool = OperationPool()
        pq_attestation_pool = PQAttestationPool()

        @staticmethod
        def _advanced_state_for(state, slot):
            return None

    node = Node()
    builder = BlockBuilder(node)

    async def run():
        for slot in range(1, 4 * SLOTS_PER_EPOCH() + 2):
            pre = process_slots(pq_helpers.copy_state(node.state), slot)
            proposer_key = client.is_our_proposer_slot(node.state, slot)
            payload = {
                "blockHash": "0x" + hashlib.sha256(b"payload" + slot.to_bytes(8, "little")).hexdigest(),
                "prevRandao": "0x" + bytes(get_randao_mix(pre, get_current_epoch(pre))).hex(),
                "feeRecipient": "0x" + "11" * 20,
                "gasLimit": hex(30_000_000),
            }
            signed_block = await builder.build_block(
                slot,
                proposer_key,
                payload,
                execution_requests=[],
                execution_head_hash=bytes(node.state.latest_block_hash),
            )
            assert signed_block is not None
            if slot > 3:
                assert len(signed_block.message.body.attestations) > 0
            node.state = state_transition(node.state, signed_block, validate_result=True)
            node.pq_attestation_pool.remove_included(
                list(signed_block.message.body.attestations), node.state
            )
            head_root = hash_tree_root(signed_block.message)
            epoch = slot // SLOTS_PER_EPOCH()
            for duty in client.get_attester_duties(node.state, epoch):
                if duty.slot != slot:
                    continue
                single = await client.produce_attestation(node.state, duty, head_root)
                committee = [int(v) for v in get_beacon_committee(node.state, slot, duty.committee_index)]
                assert node.pq_attestation_pool.add_single(single, committee)

            aggregators = get_aggregation_committee(node.state, slot, 0)
            assert 0 < len(aggregators) <= 4
            attestation, proof = node.pq_attestation_pool.get_aggregate(
                slot, hash_tree_root(single.data), 0
            )
            from consensoor.crypto import xmss as xmss_lib

            assert xmss_lib.verify_aggregates(get_attestation_statements(node.state, attestation), proof)

    asyncio.run(run())
    assert int(node.state.finalized_checkpoint.epoch) >= 2

    key = next(iter(client.keys.values()))
    with pytest.raises(xmss_signer.LeafReuseError):
        xmss_signer.sign(key, 2**15, b"\x01" * 32)
        xmss_signer.sign(key, 2**15, b"\x02" * 32)
    assert (tmp_path / "xmss_leaves.txt").read_text().count("\n") > 0
