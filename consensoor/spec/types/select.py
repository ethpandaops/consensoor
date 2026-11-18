"""Fork-dependent type selection for the Gloas family of forks (Gloas, Heze, PQ)."""

from . import electra, gloas, heze, phase0, pq


class ForkTypes:
    def __init__(self, name: str, *modules):
        self.name = name
        self._modules = modules

    def __getattr__(self, item: str):
        for module in self._modules:
            if hasattr(module, item):
                return getattr(module, item)
        raise AttributeError(item)


GLOAS = ForkTypes("gloas", gloas, electra, phase0)
HEZE = ForkTypes("heze", heze, gloas, electra, phase0)
PQ = ForkTypes("pq", pq, gloas, electra, phase0)


def types_at_epoch(epoch: int) -> ForkTypes:
    from ..network_config import get_config

    config = get_config()
    if config.is_pq_active(epoch):
        return PQ
    if config.is_heze_active(epoch):
        return HEZE
    return GLOAS


def types_at_slot(slot: int) -> ForkTypes:
    from ..constants import SLOTS_PER_EPOCH

    return types_at_epoch(int(slot) // SLOTS_PER_EPOCH())


def types_for_state(state) -> ForkTypes:
    if hasattr(state, "aggregation_committee_window"):
        return PQ
    if not hasattr(state, "eth1_data"):
        return HEZE
    return GLOAS
