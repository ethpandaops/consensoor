"""kzg_commitments_inclusion_proof against the fulu merkle_proof vectors."""
from pathlib import Path

import pytest
import snappy
import yaml

VECTORS = Path(__file__).parents[1] / "spec-tests" / "tests" / "minimal" / "fulu" / "merkle_proof" / "single_merkle_proof" / "BeaconBlockBody"


@pytest.mark.skipif(not VECTORS.exists(), reason="spec vectors not downloaded")
def test_kzg_commitments_inclusion_proof_matches_vectors():
    from consensoor.spec.constants import set_preset
    set_preset("minimal")
    from consensoor.spec.types import ElectraBeaconBlockBody
    from consensoor.das import kzg_commitments_inclusion_proof

    cases = sorted(VECTORS.glob("blob_kzg_commitments_merkle_proof__*"))
    assert cases
    for case in cases:
        body = ElectraBeaconBlockBody.decode_bytes(snappy.decompress((case / "object.ssz_snappy").read_bytes()))
        proof = yaml.safe_load((case / "proof.yaml").read_text())
        assert proof["leaf"] == "0x" + bytes(body.blob_kzg_commitments.hash_tree_root()).hex(), case.name
        got = ["0x" + bytes(b).hex() for b in kzg_commitments_inclusion_proof(body)]
        assert got == proof["branch"], case.name
