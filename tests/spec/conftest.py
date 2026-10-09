"""Pytest configuration for consensus spec tests."""

import sys
import pytest
from pathlib import Path


def pytest_addoption(parser):
    """Add command line options for spec tests."""
    parser.addoption(
        "--spec-tests-dir",
        action="store",
        default=None,
        help="Path to the extracted spec tests directory (defaults to tests/spec-tests/tests/{preset})",
    )
    parser.addoption(
        "--preset",
        action="store",
        default="minimal",
        choices=["minimal", "mainnet"],
        help="Preset to use for tests (minimal or mainnet)",
    )


def pytest_configure(config):
    """Set preset BEFORE any imports happen during test collection.

    This is critical because SSZ types use Vector[T, N()] where N() is
    evaluated at class definition time. We must set the preset before
    any type modules are imported.
    """
    preset = config.getoption("--preset", default="minimal")

    type_modules = [
        "consensoor.spec.types",
        "consensoor.spec.types.phase0",
        "consensoor.spec.types.altair",
        "consensoor.spec.types.bellatrix",
        "consensoor.spec.types.capella",
        "consensoor.spec.types.deneb",
        "consensoor.spec.types.electra",
        "consensoor.spec.types.fulu",
        "consensoor.spec.types.gloas",
        "consensoor.spec.types.heze",
        "consensoor.spec.types.base",
    ]
    for mod in type_modules:
        if mod in sys.modules:
            del sys.modules[mod]

    from consensoor.spec.constants import set_preset as do_set_preset
    do_set_preset(preset)

    from consensoor.spec.network_config import NetworkConfig, set_config
    config_path = Path(__file__).parent / "configs" / f"{preset}.yaml"
    if config_path.exists():
        net_config = NetworkConfig.from_yaml(config_path)
    elif preset == "minimal":
        net_config = NetworkConfig.minimal()
    else:
        net_config = NetworkConfig()
    set_config(net_config)


@pytest.fixture(scope="session")
def spec_tests_dir(request):
    """Return the path to spec tests."""
    spec_dir = request.config.getoption("--spec-tests-dir")
    if spec_dir is None:
        preset = request.config.getoption("--preset")
        spec_dir = f"tests/spec-tests/tests/{preset}"
    path = Path(spec_dir)
    if not path.exists():
        pytest.skip(f"Spec tests not found at {path}. Run 'make fetch-tests' first.")
    return path


@pytest.fixture(scope="session")
def preset(request):
    """Return the preset being used."""
    return request.config.getoption("--preset")


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear spec caches before each test to prevent cross-test pollution.

    This is necessary because the caches are keyed by slot/epoch, not by
    state identity. Different tests with states at the same slot would
    otherwise get incorrect cached values.
    """
    from consensoor.spec.state_transition.helpers import clear_spec_caches
    clear_spec_caches()
    yield
    clear_spec_caches()


_FORK_ORDER = ["phase0", "altair", "bellatrix", "capella", "deneb", "electra", "fulu", "gloas", "heze"]


@pytest.fixture(autouse=True)
def fork_epochs_for_case(request):
    """Mirror pyspec's vector config: the fork under test and every earlier
    fork activate at genesis, later forks stay unscheduled.

    Only applied from Gloas on (earlier forks' code paths do not consult
    fork epochs for state-transition rules).
    """
    callspec = getattr(request.node, "callspec", None)
    case = None
    case_path = None
    if callspec is not None:
        for v in callspec.params.values():
            if isinstance(v, tuple) and v and isinstance(v[0], str) and "/" in v[0]:
                case = v[0]
                case_path = next((x for x in v if isinstance(x, Path)), None)
                break
    fork = case.split("/", 1)[0] if case else None
    if fork not in ("gloas", "heze"):
        yield
        return
    from consensoor.spec.network_config import get_config
    config = get_config()
    far = 2**64 - 1
    saved = {}
    idx = _FORK_ORDER.index(fork)
    for i, name in enumerate(_FORK_ORDER[1:], start=1):
        attr = f"{name}_fork_epoch"
        if not hasattr(config, attr):
            continue
        saved[attr] = getattr(config, attr)
        setattr(config, attr, 0 if i <= idx else far)
    # A case may carry its own config.yaml (e.g. GLOAS_FORK_EPOCH: 1).
    if case_path is not None and (case_path / "config.yaml").exists():
        import yaml
        with open(case_path / "config.yaml") as f:
            case_cfg = yaml.safe_load(f) or {}
        for key, value in case_cfg.items():
            attr = key.lower()
            if attr.endswith("_fork_epoch") and hasattr(config, attr):
                saved.setdefault(attr, getattr(config, attr))
                setattr(config, attr, int(value))
    try:
        yield
    finally:
        for attr, v in saved.items():
            setattr(config, attr, v)
