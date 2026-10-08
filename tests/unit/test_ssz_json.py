import pytest

from consensoor.beacon_api.ssz_json import from_json, to_json


@pytest.fixture(scope="module", autouse=True)
def minimal():
    from consensoor.spec.constants import set_preset
    set_preset("minimal")


def test_gloas_block_round_trip():
    from consensoor.spec.types.gloas import SignedBeaconBlock

    block = SignedBeaconBlock()
    block.message.slot = 7
    block.message.body.graffiti = b"\x01" * 32
    obj = to_json(block)
    assert obj["message"]["slot"] == "7"
    assert from_json(SignedBeaconBlock, obj).hash_tree_root() == block.hash_tree_root()


def test_execution_payload_byte_lists_are_hex():
    from consensoor.spec.types.gloas import ExecutionPayload

    payload = ExecutionPayload(extra_data=[1, 2, 3], transactions=[b"\x01\x02", b"\x03"], block_access_list=b"\xaa")
    obj = to_json(payload)
    assert obj["extra_data"] == "0x010203"
    assert obj["transactions"] == ["0x0102", "0x03"]
    assert obj["block_access_list"] == "0xaa"
    assert from_json(ExecutionPayload, obj).hash_tree_root() == payload.hash_tree_root()


def test_bitvector_and_booleans():
    from consensoor.spec.types.gloas import PayloadAttestation

    att = PayloadAttestation()
    att.aggregation_bits[1] = True
    att.data.payload_present = True
    obj = to_json(att)
    assert obj["data"]["payload_present"] is True
    assert obj["data"]["blob_data_available"] is False
    assert obj["aggregation_bits"].startswith("0x02")
    assert from_json(PayloadAttestation, obj).hash_tree_root() == att.hash_tree_root()


def test_missing_field_rejected():
    from consensoor.spec.types.gloas import PayloadAttestationMessage

    with pytest.raises(ValueError):
        from_json(PayloadAttestationMessage, {"validator_index": "1"})
