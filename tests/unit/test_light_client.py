"""Light client data derivation against the light_client spec vectors."""
from pathlib import Path

import pytest
import snappy
import yaml

VECTORS = Path(__file__).parents[1] / "spec-tests" / "tests" / "minimal"
FORKS = ("altair", "bellatrix", "capella", "deneb", "electra", "fulu", "gloas", "heze")


def _state_type(fork):
    from consensoor.spec.types import altair, bellatrix, capella, deneb, electra, fulu, gloas, heze
    return {
        "altair": altair.AltairBeaconState,
        "bellatrix": bellatrix.BellatrixBeaconState,
        "capella": capella.CapellaBeaconState,
        "deneb": deneb.DenebBeaconState,
        "electra": electra.ElectraBeaconState,
        "fulu": fulu.FuluBeaconState,
        "gloas": gloas.BeaconState,
        "heze": heze.BeaconState,
    }[fork]


def _body_type(fork):
    from consensoor.spec.types import capella, deneb, electra, gloas, heze
    return {
        "capella": capella.CapellaBeaconBlockBody,
        "deneb": deneb.DenebBeaconBlockBody,
        "electra": electra.ElectraBeaconBlockBody,
        "fulu": electra.ElectraBeaconBlockBody,
        "gloas": gloas.BeaconBlockBody,
        "heze": heze.BeaconBlockBody,
    }[fork]


@pytest.fixture(scope="module")
def minimal():
    from consensoor.spec.constants import set_preset
    from consensoor.spec.network_config import NetworkConfig, get_config, set_config

    set_preset("minimal")
    prev = get_config()
    config = NetworkConfig()
    for fork in FORKS:
        setattr(config, f"{fork}_fork_epoch", 0)
    set_config(config)
    yield config
    set_config(prev)


def _cases(fork, kind):
    base = VECTORS / fork / "light_client" / "single_merkle_proof" / kind
    return sorted(base.glob("*")) if base.exists() else []


@pytest.mark.skipif(not VECTORS.exists(), reason="spec vectors not downloaded")
@pytest.mark.parametrize("fork", FORKS)
def test_single_merkle_proof(minimal, fork):
    from consensoor.beacon_api.light_client import compute_merkle_proof, get_generalized_index

    paths = {
        "current_sync_committee_merkle_proof": ("current_sync_committee",),
        "next_sync_committee_merkle_proof": ("next_sync_committee",),
        "finality_root_merkle_proof": ("finalized_checkpoint", "root"),
        "execution_merkle_proof": ("execution_payload",),
        "execution_block_hash_merkle_proof": ("signed_execution_payload_bid", "message", "parent_block_hash"),
    }
    cases = [("BeaconState", c) for c in _cases(fork, "BeaconState")]
    cases += [("BeaconBlockBody", c) for c in _cases(fork, "BeaconBlockBody")]
    assert cases
    for kind, case in cases:
        typ = _state_type(fork) if kind == "BeaconState" else _body_type(fork)
        obj = typ.decode_bytes(snappy.decompress((case / "object.ssz_snappy").read_bytes()))
        proof = yaml.safe_load((case / "proof.yaml").read_text())
        gindex = get_generalized_index(typ, *paths[case.name])
        assert gindex == proof["leaf_index"], case
        assert ["0x" + b.hex() for b in compute_merkle_proof(obj, gindex)] == proof["branch"], case


@pytest.mark.skipif(not VECTORS.exists(), reason="spec vectors not downloaded")
@pytest.mark.parametrize("fork", FORKS)
def test_update_ranking(minimal, fork):
    from consensoor.beacon_api.light_client import is_better_update, light_client_types

    case = VECTORS / fork / "light_client" / "update_ranking" / "pyspec_tests" / "update_ranking"
    if not case.exists():
        pytest.skip("no update_ranking vectors")
    count = yaml.safe_load((case / "meta.yaml").read_text())["updates_count"]
    update_type = light_client_types(fork)["LightClientUpdate"]
    updates = [
        update_type.decode_bytes(snappy.decompress((case / f"updates_{i}.ssz_snappy").read_bytes()))
        for i in range(count)
    ]
    for i in range(count - 1):
        assert not is_better_update(updates[i + 1], updates[i]), (fork, i)


def _gloas_chain():
    from consensoor.crypto import hash_tree_root
    from consensoor.spec.types import gloas
    from consensoor.spec.types.phase0 import BeaconBlockHeader

    state = gloas.BeaconState()
    state.slot = 9
    state.finalized_checkpoint.epoch = 1
    state.current_sync_committee.aggregate_pubkey = b"\x11" * 48
    state.next_sync_committee.aggregate_pubkey = b"\x22" * 48
    body = gloas.BeaconBlockBody()
    body.signed_execution_payload_bid.message.parent_block_hash = b"\x33" * 32
    body.graffiti = b"\x44" * 32
    state.latest_block_header = BeaconBlockHeader(
        slot=9, proposer_index=3, parent_root=b"\x55" * 32, body_root=hash_tree_root(body)
    )
    block = gloas.SignedBeaconBlock(
        message=gloas.BeaconBlock(
            slot=9, proposer_index=3, parent_root=b"\x55" * 32, state_root=hash_tree_root(state), body=body
        )
    )
    return state, block


def test_gloas_bootstrap_branches(minimal):
    from consensoor.beacon_api.light_client import create_light_client_bootstrap, light_client_types
    from consensoor.crypto import hash_tree_root
    from consensoor.spec.constants import CURRENT_SYNC_COMMITTEE_GINDEX_GLOAS, EXECUTION_BLOCK_HASH_GINDEX_GLOAS
    from consensoor.spec.state_transition.helpers.predicates import is_valid_merkle_branch

    state, block = _gloas_chain()
    bootstrap = create_light_client_bootstrap(state, block)
    header = bootstrap.header
    assert bytes(header.execution_block_hash) == b"\x33" * 32
    assert is_valid_merkle_branch(
        bytes(header.execution_block_hash), header.execution_branch,
        EXECUTION_BLOCK_HASH_GINDEX_GLOAS.bit_length() - 1,
        EXECUTION_BLOCK_HASH_GINDEX_GLOAS % 2 ** (EXECUTION_BLOCK_HASH_GINDEX_GLOAS.bit_length() - 1),
        bytes(header.beacon.body_root),
    )
    assert is_valid_merkle_branch(
        hash_tree_root(bootstrap.current_sync_committee), bootstrap.current_sync_committee_branch,
        CURRENT_SYNC_COMMITTEE_GINDEX_GLOAS.bit_length() - 1,
        CURRENT_SYNC_COMMITTEE_GINDEX_GLOAS % 2 ** (CURRENT_SYNC_COMMITTEE_GINDEX_GLOAS.bit_length() - 1),
        bytes(header.beacon.state_root),
    )
    bootstrap_type = light_client_types("gloas")["LightClientBootstrap"]
    assert bootstrap_type.decode_bytes(bootstrap.encode_bytes()) == bootstrap
    assert hash_tree_root(bootstrap_type.decode_bytes(bootstrap.encode_bytes())) == hash_tree_root(bootstrap)


def test_gloas_update_and_derived_updates(minimal):
    from consensoor.beacon_api.light_client import (
        create_light_client_finality_update,
        create_light_client_optimistic_update,
        create_light_client_update,
        is_finality_update,
        is_sync_committee_update,
    )
    from consensoor.crypto import hash_tree_root
    from consensoor.spec.constants import FINALIZED_ROOT_GINDEX_GLOAS, NEXT_SYNC_COMMITTEE_GINDEX_GLOAS
    from consensoor.spec.state_transition.helpers.predicates import is_valid_merkle_branch
    from consensoor.spec.types import gloas
    from consensoor.spec.types.phase0 import BeaconBlockHeader

    attested_state, attested_block = _gloas_chain()
    attested_root = hash_tree_root(attested_block.message)

    state = gloas.BeaconState()
    state.slot = 10
    body = gloas.BeaconBlockBody()
    body.sync_aggregate.sync_committee_bits[0] = True
    state.latest_block_header = BeaconBlockHeader(
        slot=10, proposer_index=4, parent_root=attested_root, body_root=hash_tree_root(body)
    )
    block = gloas.SignedBeaconBlock(
        message=gloas.BeaconBlock(
            slot=10, proposer_index=4, parent_root=attested_root, state_root=hash_tree_root(state), body=body
        )
    )

    update = create_light_client_update(state, block, attested_state, attested_block, None)
    assert is_sync_committee_update(update)
    assert not is_finality_update(update)
    depth = NEXT_SYNC_COMMITTEE_GINDEX_GLOAS.bit_length() - 1
    assert is_valid_merkle_branch(
        hash_tree_root(update.next_sync_committee), update.next_sync_committee_branch,
        depth, NEXT_SYNC_COMMITTEE_GINDEX_GLOAS % 2 ** depth, bytes(update.attested_header.beacon.state_root),
    )
    assert int(update.signature_slot) == 10

    finality_update = create_light_client_finality_update(update)
    optimistic_update = create_light_client_optimistic_update(update)
    assert finality_update.attested_header == update.attested_header
    assert int(optimistic_update.signature_slot) == 10

    finalized_state, finalized_block = _gloas_chain()
    attested_state.finalized_checkpoint.root = hash_tree_root(finalized_block.message)
    attested_block.message.state_root = hash_tree_root(attested_state)
    attested_root = hash_tree_root(attested_block.message)
    state.latest_block_header.parent_root = attested_root
    block.message.parent_root = attested_root
    block.message.state_root = hash_tree_root(state)
    update = create_light_client_update(state, block, attested_state, attested_block, finalized_block)
    assert is_finality_update(update)
    depth = FINALIZED_ROOT_GINDEX_GLOAS.bit_length() - 1
    assert is_valid_merkle_branch(
        hash_tree_root(update.finalized_header.beacon), update.finality_branch,
        depth, FINALIZED_ROOT_GINDEX_GLOAS % 2 ** depth, bytes(update.attested_header.beacon.state_root),
    )


def test_electra_header_upgraded_to_gloas(minimal):
    from consensoor.beacon_api.light_client import block_to_light_client_header, get_generalized_index
    from consensoor.crypto import hash_tree_root
    from consensoor.spec.constants import EXECUTION_BLOCK_HASH_GINDEX_GLOAS
    from consensoor.spec.state_transition.helpers.predicates import is_valid_merkle_branch
    from consensoor.spec.types import electra

    minimal.gloas_fork_epoch = minimal.heze_fork_epoch = 2
    try:
        block = electra.SignedElectraBeaconBlock()
        block.message.slot = 3
        block.message.body.execution_payload.block_hash = b"\x66" * 32
        native = block_to_light_client_header(block)
        assert isinstance(native, electra.ElectraLightClientHeader)
        header = block_to_light_client_header(block, "gloas")
        assert bytes(header.execution_block_hash) == b"\x66" * 32
        gindex = get_generalized_index(type(block.message.body), "execution_payload", "block_hash")
        depth = gindex.bit_length() - 1
        padding = EXECUTION_BLOCK_HASH_GINDEX_GLOAS.bit_length() - 1 - depth
        assert all(bytes(b) == b"\x00" * 32 for b in list(header.execution_branch)[:padding])
        assert is_valid_merkle_branch(
            b"\x66" * 32, list(header.execution_branch)[padding:], depth,
            gindex % 2 ** depth, hash_tree_root(block.message.body),
        )
        assert bytes(header.beacon.body_root) == hash_tree_root(block.message.body)
    finally:
        minimal.gloas_fork_epoch = minimal.heze_fork_epoch = 0
