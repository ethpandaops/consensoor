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

    from consensoor.spec.constants import (
        BUILDER_INDEX_SELF_BUILD,
        DOMAIN_BEACON_BUILDER,
        LEAF_OFFSET_EXECUTION_PAYLOAD_ENVELOPE,
    )
    from consensoor.spec.state_transition.block.execution_payload_envelope import (
        verify_execution_payload_envelope_signature,
    )
    from consensoor.spec.state_transition.helpers.domain import compute_signing_root, get_domain
    from consensoor.spec.state_transition.helpers.pq import compute_xmss_leaf_index
    from consensoor.spec.types import pq

    state = node.state
    proposer_index = int(state.latest_block_header.proposer_index)
    proposer_key = next(k for k in client.keys.values() if k.validator_index == proposer_index)
    envelope = pq.ExecutionPayloadEnvelope(builder_index=BUILDER_INDEX_SELF_BUILD)
    signing_root = compute_signing_root(envelope, get_domain(state, DOMAIN_BEACON_BUILDER))
    slot = int(state.latest_block_header.slot)
    signature = xmss_signer.sign(
        proposer_key,
        compute_xmss_leaf_index(slot, LEAF_OFFSET_EXECUTION_PAYLOAD_ENVELOPE),
        signing_root,
    )
    signed_envelope = pq.SignedExecutionPayloadEnvelope(message=envelope, signature=signature)
    assert verify_execution_payload_envelope_signature(state, signed_envelope)
    wrong_leaf = xmss_signer.sign(
        proposer_key,
        compute_xmss_leaf_index(slot + 100, LEAF_OFFSET_EXECUTION_PAYLOAD_ENVELOPE),
        signing_root,
    )
    assert not verify_execution_payload_envelope_signature(
        state, pq.SignedExecutionPayloadEnvelope(message=envelope, signature=wrong_leaf)
    )

    key = next(iter(client.keys.values()))
    with pytest.raises(xmss_signer.LeafReuseError):
        xmss_signer.sign(key, 2**15, b"\x01" * 32)
        xmss_signer.sign(key, 2**15, b"\x02" * 32)
    assert (tmp_path / "xmss_leaves.txt").read_text().count("\n") > 0


def test_container_shapes(pq_config):
    from consensoor.spec.types import pq

    assert list(pq.BeaconBlockBody.fields()) == [
        "randao_reveal",
        "eth1_data",
        "graffiti",
        "proposer_slashings",
        "attester_slashings",
        "attestations",
        "deposits",
        "signed_execution_payload_bid",
        "parent_execution_requests",
        "proof",
    ]
    state_fields = list(pq.BeaconState.fields())
    assert len(state_fields) == 47
    assert "current_sync_committee" not in state_fields
    assert state_fields[-3:] == [
        "validator_preregistrations",
        "builder_preregistrations",
        "aggregation_committee_window",
    ]
    assert list(pq.Attestation.fields()) == ["aggregation_bits", "data", "committee_bits"]
    assert list(pq.ExecutionRequests.fields())[-3:] == [
        "validator_preregistrations",
        "builder_preregistrations",
        "voluntary_exits",
    ]
    assert len(pq.Validator().encode_bytes()) == 32 + 32 + 32 + 8 + 1 + 8 * 4
    assert len(pq.SingleAttestation().encode_bytes()) == 8 + 8 + 128 + 2404
    assert len(pq.PayloadAttestationMessage().encode_bytes()) == 8 + 42 + 2404
    assert len(pq.VRFProof().encode_bytes()) == 32 * 32
    assert len(pq.CommitteeBits()) == 4


def test_generator_genesis_fixture(pq_config):
    from consensoor.spec.state_transition.helpers.pq import initialize_aggregation_committee_window
    from consensoor.spec.types import pq

    state = pq.BeaconState.decode_bytes((Path(__file__).parent / "fixtures" / "pq" / "genesis.ssz").read_bytes())
    assert bytes(state.hash_tree_root()).hex() == (
        "cc0e5639191cf4d1b7d459b8e1d3a7d6aa86eaa37cc63b6f02a9ea6145c15736"
    )
    body = pq.BeaconBlockBody(
        signed_execution_payload_bid=pq.SignedExecutionPayloadBid(
            message=state.latest_execution_payload_bid
        )
    )
    assert bytes(body.hash_tree_root()) == bytes(state.latest_block_header.body_root)
    assert bytes(state.latest_execution_payload_bid.execution_requests_root) == bytes(
        pq.ExecutionRequests().hash_tree_root()
    )
    window = [[[int(v) for v in c] for c in cs] for cs in state.aggregation_committee_window]
    assert window == initialize_aggregation_committee_window(state)


def test_leaf_indexes():
    from consensoor.spec.state_transition.helpers import pq as helpers

    assert helpers.compute_xmss_attestation_leaf_index(0) == 2**15
    assert helpers.compute_xmss_attestation_leaf_index(7) == 2**15 + 7
    base = 2**15 + 2**29 + 2**30
    assert helpers.compute_xmss_leaf_index(0, 0) == base
    assert helpers.compute_xmss_leaf_index(5, 4) == base + 44
    assert not helpers.is_builder_bid_leaf_index(2**15 + 2**29 - 1)
    assert helpers.is_builder_bid_leaf_index(2**15 + 2**29)
    assert helpers.is_builder_bid_leaf_index(base - 1)
    assert not helpers.is_builder_bid_leaf_index(base)
    assert helpers.is_legacy_pubkey(b"\x80" + b"\x00" * 47)
    assert not helpers.is_legacy_pubkey(b"\x00" * 16 + b"\x11" * 32)
    assert helpers.compute_deposit_pubkey(b"\x00" * 16 + b"\x11" * 32) == b"\x11" * 32


def test_vrf_proof_opens_root():
    import hashlib

    from consensoor.spec.state_transition.helpers.predicates import is_valid_merkle_branch

    seed = xmss.vrf_seed_from_privkey(42)
    root = xmss.vrf_root(seed)
    for slot in (0, 1, 1023, 1024, 2**20 - 1, 2**20, 2**31 - 1):
        proof = xmss.vrf_proof(seed, slot)
        assert len(proof) == 32
        expected_leaf = (
            hashlib.sha256(seed + slot.to_bytes(8, "little")).digest() if slot < 2**20 else bytes(32)
        )
        assert proof[0] == expected_leaf
        assert is_valid_merkle_branch(proof[0], proof[1:], 31, slot, root)
        if slot < 2**20:
            assert not is_valid_merkle_branch(proof[0], proof[1:], 31, slot ^ 1, root)


def test_stark_stand_in_checks_leaf_and_signature():
    seeds = [xmss.seed_from_privkey(sk) for sk in (1, 2)]
    pubkeys = [xmss.public_key(seed) for seed in seeds]
    message = b"\x07" * 32
    leaf = 2**15 + 3
    signatures = [xmss.sign(seed, leaf, message) for seed in seeds]
    proof = xmss.prove_aggregates([signatures])
    assert xmss.get_leaf_index(signatures[0]) == leaf
    assert xmss.verify_aggregates([(pubkeys, leaf, message)], proof)
    assert xmss.verify_aggregates([(pubkeys[:1], leaf, message), (pubkeys[1:], leaf, message)], proof)
    assert not xmss.verify_aggregates([(pubkeys, leaf + 1, message)], proof)
    assert not xmss.verify_aggregates([(pubkeys, leaf, b"\x08" * 32)], proof)
    assert not xmss.verify_aggregates([(pubkeys[::-1], leaf, message)], proof)
    assert not xmss.verify_aggregates([(pubkeys, leaf, message)], proof[:-1])
    assert not xmss.verify_aggregates([(pubkeys, leaf, message)], proof + signatures[0])
    tampered = bytearray(proof)
    tampered[100] ^= 1
    assert not xmss.verify_aggregates([(pubkeys, leaf, message)], bytes(tampered))
    other_leaf = xmss.sign(seeds[0], leaf + 1, message)
    assert not xmss.verify(pubkeys[0], leaf, message, other_leaf)
    assert xmss.verify(pubkeys[0], leaf + 1, message, other_leaf)


def test_deposits_and_requests(genesis, pq_config):
    from consensoor.builder import BlockBuilder
    from consensoor.node import BeaconNode
    from consensoor.spec.constants import FAR_FUTURE_EPOCH
    from consensoor.spec.state_transition.block.operations import process_deposit_request
    from consensoor.spec.state_transition.block.operations.pq import (
        process_validator_preregistration_request,
        process_voluntary_exit_request,
    )
    from consensoor.spec.state_transition.epoch.pending_deposits import apply_pending_deposit
    from consensoor.spec.types import pq
    from consensoor.spec.types.electra import DepositRequest
    from . import pq_helpers

    state = pq_helpers.copy_state(genesis)
    credentials = b"\x01" + b"\x00" * 11 + b"\x22" * 20
    new_key = b"\x33" * 32
    vrf_root = b"\x44" * 32

    process_validator_preregistration_request(
        state,
        pq.ValidatorPreregistrationRequest(
            pubkey=new_key, withdrawal_credentials=credentials, vrf_root=vrf_root
        ),
    )
    assert len(state.validator_preregistrations) == 1

    def deposit(pubkey48, creds, sig):
        return DepositRequest(
            pubkey=pubkey48, withdrawal_credentials=creds, amount=32 * 10**9, signature=sig, index=0
        )

    process_deposit_request(state, deposit(b"\x00" * 16 + new_key, b"\x01" + b"\x00" * 31, vrf_root + b"\x00" * 64))
    assert len(state.pending_deposits) == 0
    process_deposit_request(state, deposit(b"\x00" * 16 + new_key, credentials, vrf_root + b"\x00" * 64))
    assert len(state.pending_deposits) == 1
    assert len(state.validator_preregistrations) == 0

    count = len(state.validators)
    apply_pending_deposit(state, state.pending_deposits[0])
    assert len(state.validators) == count + 1
    assert bytes(state.validators[count].pubkey) == new_key
    assert bytes(state.validators[count].vrf_root) == vrf_root
    assert int(state.validators[count].exit_epoch) == FAR_FUTURE_EPOCH

    legacy = b"\xa0" + b"\x55" * 47
    process_deposit_request(state, deposit(legacy, credentials, b"\x00" * 96))
    apply_pending_deposit(state, state.pending_deposits[1])
    assert len(state.validators) == count + 2
    assert int(state.validators[count + 1].exit_epoch) == 0

    process_voluntary_exit_request(state, pq.VoluntaryExitRequest(pubkey=state.validators[0].pubkey))
    assert int(state.validators[0].exit_epoch) == FAR_FUTURE_EPOCH

    requests = pq.ExecutionRequests(
        validator_preregistrations=[
            pq.ValidatorPreregistrationRequest(pubkey=new_key, withdrawal_credentials=credentials, vrf_root=vrf_root)
        ],
        voluntary_exits=[pq.VoluntaryExitRequest(pubkey=new_key)],
    )
    encoded = BeaconNode._encode_execution_requests(requests)
    assert [entry[:4] for entry in encoded] == ["0x07", "0x09"]
    decoded = BlockBuilder._decode_execution_requests_hex(encoded, gloas=True, pq=True)
    assert decoded.hash_tree_root() == requests.hash_tree_root()


def test_pool_caps_signatures_per_proof(pq_config):
    from consensoor.pq_attestation_pool import MAX_PROOF_SIGNATURES, PQAttestationPool
    from consensoor.spec.types import pq
    from consensoor.spec.types.phase0 import AttestationData

    assert MAX_PROOF_SIGNATURES == 436
    pool = PQAttestationPool()
    data = AttestationData(slot=3)
    for committee_index, committee in ((0, [10, 11, 12]), (1, [20, 21])):
        for validator_index in committee:
            single = pq.SingleAttestation(
                committee_index=committee_index,
                attester_index=validator_index,
                data=data,
                signature=bytes([validator_index]) * 2404,
            )
            assert pool.add_single(single, committee)
            assert not pool.add_single(single, committee)

    assert pool.get_block_candidates(4) == []
    (root,) = pool.get_block_candidates(5)
    attestation, proof = pool.build_for_block(root, 100)
    assert list(attestation.aggregation_bits) == [True] * 5
    assert list(attestation.committee_bits) == [True, True, False, False]
    assert len(proof) == 5 * 2404

    attestation, proof = pool.build_for_block(root, 2)
    assert list(attestation.aggregation_bits) == [True, True, False]
    assert list(attestation.committee_bits) == [True, False, False, False]
    assert proof == bytes([10]) * 2404 + bytes([11]) * 2404

    attestation, proof = pool.build_for_block(root, 4)
    assert list(attestation.aggregation_bits) == [True, True, True, True, False]
    assert pool.build_for_block(root, 0) is None

    single_committee, committee_proof = pool.get_aggregate(3, root, 1)
    assert pool.committee_of(single_committee) == 1
    other = PQAttestationPool()
    assert other.add_aggregate(single_committee, committee_proof, [20, 21]) == 2
    assert other.size == 2
