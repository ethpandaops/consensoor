import pytest


@pytest.fixture(scope="module", autouse=True)
def minimal():
    from consensoor.spec.constants import set_preset
    set_preset("minimal")


FAR = 2**64 - 1


@pytest.mark.parametrize("fields,epoch,expected", [
    (dict(activation_eligibility_epoch=FAR, activation_epoch=FAR), 5, "pending_initialized"),
    (dict(activation_eligibility_epoch=1, activation_epoch=FAR), 5, "pending_queued"),
    (dict(activation_epoch=0, exit_epoch=FAR), 5, "active_ongoing"),
    (dict(activation_epoch=0, exit_epoch=9), 5, "active_exiting"),
    (dict(activation_epoch=0, exit_epoch=9, slashed=True), 5, "active_slashed"),
    (dict(activation_epoch=0, exit_epoch=4, withdrawable_epoch=9), 5, "exited_unslashed"),
    (dict(activation_epoch=0, exit_epoch=4, withdrawable_epoch=9, slashed=True), 5, "exited_slashed"),
    (dict(activation_epoch=0, exit_epoch=4, withdrawable_epoch=5, effective_balance=32), 5, "withdrawal_possible"),
    (dict(activation_epoch=0, exit_epoch=4, withdrawable_epoch=5, effective_balance=0), 5, "withdrawal_done"),
])
def test_validator_status(fields, epoch, expected):
    from consensoor.beacon_api.server import BeaconAPI
    from consensoor.spec.types.phase0 import Validator

    validator = Validator(**{"exit_epoch": FAR, "withdrawable_epoch": FAR, **fields})
    assert BeaconAPI._validator_status(validator, epoch) == expected
    assert BeaconAPI._status_matches(expected, {expected.split("_")[0]})


def test_blinded_block_keeps_block_root():
    from consensoor.beacon_api.server import BeaconAPI
    from consensoor.crypto import hash_tree_root
    from consensoor.spec.types.electra import SignedElectraBeaconBlock

    block = SignedElectraBeaconBlock()
    block.message.slot = 3
    block.message.body.execution_payload.transactions = [b"\x01\x02"]
    blinded = BeaconAPI(object())._blind(block, "electra")
    assert hash_tree_root(blinded.message) == hash_tree_root(block.message)
    assert "execution_payload_header" in type(blinded.message.body).fields()
