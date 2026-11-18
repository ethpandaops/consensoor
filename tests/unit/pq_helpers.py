"""Helpers building a PQ genesis state and PQ blocks for unit tests."""

import hashlib

VALIDATOR_COUNT = 16


def privkey(index: int) -> int:
    return int.from_bytes(hashlib.sha256(b"pq-test-key" + index.to_bytes(8, "little")).digest(), "big") >> 8


def build_genesis_state(validator_count: int = VALIDATOR_COUNT):
    from consensoor.crypto import xmss
    from consensoor.spec.constants import (
        FAR_FUTURE_EPOCH,
        MIN_SEED_LOOKAHEAD,
        SLOTS_PER_EPOCH,
    )
    from consensoor.spec.network_config import get_config
    from consensoor.spec.state_transition.fork_upgrade import initialize_proposer_lookahead
    from consensoor.spec.state_transition.helpers.pq import initialize_aggregation_committee_window
    from consensoor.spec.state_transition.helpers.ptc import compute_ptc
    from consensoor.spec.types import pq
    from consensoor.spec.types.base import Fork
    from consensoor.spec.types.gloas import ExecutionPayloadBid
    from consensoor.spec.types.phase0 import BeaconBlockHeader

    config = get_config()
    validators = []
    for index in range(validator_count):
        sk = privkey(index)
        validators.append(
            pq.Validator(
                pubkey=xmss.public_key(xmss.seed_from_privkey(sk)),
                vrf_root=xmss.vrf_root(xmss.vrf_seed_from_privkey(sk)),
                withdrawal_credentials=b"\x01" + b"\x00" * 11 + b"\x11" * 20,
                effective_balance=32 * 10**9,
                slashed=False,
                activation_eligibility_epoch=0,
                activation_epoch=0,
                exit_epoch=FAR_FUTURE_EPOCH,
                withdrawable_epoch=FAR_FUTURE_EPOCH,
            )
        )

    state = pq.BeaconState(
        genesis_time=1_000_000,
        fork=Fork(
            previous_version=config.pq_fork_version,
            current_version=config.pq_fork_version,
            epoch=0,
        ),
        validators=validators,
        balances=[32 * 10**9] * validator_count,
        previous_epoch_participation=[0] * validator_count,
        current_epoch_participation=[0] * validator_count,
        inactivity_scores=[0] * validator_count,
        latest_block_hash=b"\x42" * 32,
        latest_execution_payload_bid=ExecutionPayloadBid(
            block_hash=b"\x42" * 32,
            execution_requests_root=pq.ExecutionRequests().hash_tree_root(),
        ),
    )
    state.latest_block_header = BeaconBlockHeader(
        body_root=pq.BeaconBlockBody().hash_tree_root()
    )
    state.genesis_validators_root = state.validators.hash_tree_root()
    for i in range(len(state.randao_mixes)):
        state.randao_mixes[i] = b"\x42" * 32

    lookahead = initialize_proposer_lookahead(state)
    for i, proposer in enumerate(lookahead):
        state.proposer_lookahead[i] = proposer

    spe = SLOTS_PER_EPOCH()
    for slot in range((MIN_SEED_LOOKAHEAD + 1) * spe):
        state.ptc_window[spe + slot] = compute_ptc(state, slot)

    window = initialize_aggregation_committee_window(state)
    for i, committees in enumerate(window):
        state.aggregation_committee_window[i] = pq.aggregation_committees(committees)
    return state


def copy_state(state):
    return state.__class__.decode_bytes(bytes(state.encode_bytes()))


def sign_attestations(state, slot: int, head_root: bytes):
    """Return (attestation, proof) pairs, one per committee of ``slot``, signed by every member."""
    from consensoor.crypto import xmss
    from consensoor.spec.constants import DOMAIN_BEACON_ATTESTER, SLOTS_PER_EPOCH
    from consensoor.spec.state_transition.helpers.accessors import get_block_root
    from consensoor.spec.state_transition.helpers.beacon_committee import (
        get_beacon_committee,
        get_committee_count_per_slot,
    )
    from consensoor.spec.state_transition.helpers.domain import (
        compute_signing_root,
        get_domain_at_epoch,
    )
    from consensoor.spec.state_transition.helpers.pq import compute_xmss_attestation_leaf_index
    from consensoor.spec.types import pq
    from consensoor.spec.types.base import Checkpoint
    from consensoor.spec.types.phase0 import AttestationData

    epoch = slot // SLOTS_PER_EPOCH()
    target_root = (
        get_block_root(state, epoch) if epoch * SLOTS_PER_EPOCH() < int(state.slot) else head_root
    )
    data = AttestationData(
        slot=slot,
        index=0,
        beacon_block_root=head_root,
        source=state.current_justified_checkpoint,
        target=Checkpoint(epoch=epoch, root=target_root),
    )
    signing_root = compute_signing_root(data, get_domain_at_epoch(state, DOMAIN_BEACON_ATTESTER, epoch))
    leaf_index = compute_xmss_attestation_leaf_index(epoch)
    pubkey_to_index = {bytes(v.pubkey): i for i, v in enumerate(state.validators)}
    seeds = {
        pubkey_to_index[xmss.public_key(xmss.seed_from_privkey(privkey(i)))]: xmss.seed_from_privkey(privkey(i))
        for i in range(len(state.validators))
    }

    result = []
    for committee_index in range(get_committee_count_per_slot(state, epoch)):
        committee = get_beacon_committee(state, slot, committee_index)
        committee_bits = pq.CommitteeBits()
        committee_bits[committee_index] = True
        attestation = pq.Attestation(
            aggregation_bits=[True] * len(committee),
            data=data,
            committee_bits=committee_bits,
        )
        proof = xmss.prove_aggregates(
            [[xmss.sign(seeds[int(v)], leaf_index, signing_root) for v in committee]]
        )
        result.append((attestation, proof))
    return result


def build_block(state, slot: int, attestations=()):
    """Build and sign a self-built block for ``slot`` on top of ``state``; return (signed_block, post_state)."""
    from consensoor.crypto import hash_tree_root, xmss
    from consensoor.spec.constants import (
        BUILDER_INDEX_SELF_BUILD,
        DOMAIN_BEACON_PROPOSER,
        LEAF_OFFSET_BEACON_PROPOSER,
    )
    from consensoor.spec.state_transition import process_block, process_slots
    from consensoor.spec.state_transition.helpers.accessors import (
        get_block_root_at_slot,
        get_current_epoch,
        get_randao_mix,
    )
    from consensoor.spec.state_transition.helpers.beacon_committee import get_beacon_proposer_index
    from consensoor.spec.state_transition.helpers.domain import compute_signing_root, get_domain
    from consensoor.spec.state_transition.helpers.pq import compute_xmss_leaf_index
    from consensoor.spec.types import pq
    from consensoor.spec.types.gloas import ExecutionPayloadBid

    pre = process_slots(copy_state(state), slot)
    proposer_index = get_beacon_proposer_index(pre)
    sk = next(
        privkey(i)
        for i in range(len(pre.validators))
        if xmss.public_key(xmss.seed_from_privkey(privkey(i))) == bytes(pre.validators[proposer_index].pubkey)
    )
    bid = ExecutionPayloadBid(
        parent_block_hash=pre.latest_block_hash,
        parent_block_root=get_block_root_at_slot(pre, slot - 1),
        block_hash=hashlib.sha256(b"payload" + slot.to_bytes(8, "little")).digest(),
        prev_randao=get_randao_mix(pre, get_current_epoch(pre)),
        builder_index=BUILDER_INDEX_SELF_BUILD,
        slot=slot,
        execution_requests_root=hash_tree_root(pq.ExecutionRequests()),
    )
    body = pq.BeaconBlockBody(
        randao_reveal=xmss.vrf_proof(xmss.vrf_seed_from_privkey(sk), slot),
        signed_execution_payload_bid=pq.SignedExecutionPayloadBid(message=bid),
        attestations=[attestation for attestation, _ in attestations],
        proof=xmss.merge_proofs([proof for _, proof in attestations]),
    )
    block = pq.BeaconBlock(
        slot=slot,
        proposer_index=proposer_index,
        parent_root=hash_tree_root(pre.latest_block_header),
        body=body,
    )
    post = copy_state(pre)
    process_block(post, block)
    block.state_root = hash_tree_root(post)
    signing_root = compute_signing_root(block, get_domain(pre, DOMAIN_BEACON_PROPOSER))
    signature = xmss.sign(
        xmss.seed_from_privkey(sk),
        compute_xmss_leaf_index(slot, LEAF_OFFSET_BEACON_PROPOSER),
        signing_root,
    )
    return pq.SignedBeaconBlock(message=block, signature=signature), post
