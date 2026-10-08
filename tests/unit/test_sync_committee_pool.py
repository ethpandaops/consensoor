import pytest


@pytest.fixture(scope="module", autouse=True)
def minimal():
    from consensoor.spec.constants import set_preset
    set_preset("minimal")


def _contribution(root: bytes, bits: list[int], sk: int):
    from py_ecc.bls import G2ProofOfPossession as bls
    from consensoor.spec.constants import SYNC_COMMITTEE_SIZE, SYNC_COMMITTEE_SUBNET_COUNT
    from consensoor.spec.types.altair import SyncCommitteeContribution
    from consensoor.spec.types.base import Bitvector

    size = SYNC_COMMITTEE_SIZE() // SYNC_COMMITTEE_SUBNET_COUNT
    agg_bits = Bitvector[size]()
    for b in bits:
        agg_bits[b] = True
    return SyncCommitteeContribution(
        slot=5, beacon_block_root=root, subcommittee_index=0,
        aggregation_bits=agg_bits, signature=bls.Sign(sk, root),
    )


def test_contributions_for_different_roots_are_not_merged():
    from consensoor.sync_committee_pool import SyncCommitteePool

    root_a, root_b = b"\xaa" * 32, b"\xbb" * 32
    pool = SyncCommitteePool()
    a = _contribution(root_a, [0, 1], 1)
    assert pool.add_contribution(a)
    assert pool.add_contribution(_contribution(root_b, [2, 3], 2))

    agg = pool.get_sync_aggregate(5, expected_block_root=root_a)
    assert [i for i, b in enumerate(agg.sync_committee_bits) if b] == [0, 1]
    assert bytes(agg.sync_committee_signature) == bytes(a.signature)


def test_contributions_for_same_root_are_merged():
    from py_ecc.bls import G2ProofOfPossession as bls
    from consensoor.sync_committee_pool import SyncCommitteePool

    root = b"\xaa" * 32
    pool = SyncCommitteePool()
    pool.add_contribution(_contribution(root, [0], 1))
    pool.add_contribution(_contribution(root, [1], 2))

    agg = pool.get_sync_aggregate(5, expected_block_root=root)
    assert [i for i, b in enumerate(agg.sync_committee_bits) if b] == [0, 1]
    assert bls.FastAggregateVerify([bls.SkToPk(1), bls.SkToPk(2)], root, bytes(agg.sync_committee_signature))
