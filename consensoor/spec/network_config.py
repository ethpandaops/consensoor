"""Network configuration loaded from yaml (runtime values, not presets)."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

import yaml

logger = logging.getLogger(__name__)

UPSTREAM_MAINNET_CONFIG_URL = (
    "https://raw.githubusercontent.com/ethereum/consensus-specs/master/configs/mainnet.yaml"
)
UPSTREAM_MINIMAL_CONFIG_URL = (
    "https://raw.githubusercontent.com/ethereum/consensus-specs/master/configs/minimal.yaml"
)


@dataclass
class NetworkConfig:
    """Network-specific configuration loaded from yaml."""

    config_name: str = "mainnet"
    preset_base: str = "mainnet"
    blob_schedule: list = field(default_factory=list)

    seconds_per_slot: int = 12
    seconds_per_eth1_block: int = 14
    min_validator_withdrawability_delay: int = 256
    shard_committee_period: int = 256
    eth1_follow_distance: int = 2048
    min_builder_withdrawability_delay: int = 4096

    min_genesis_active_validator_count: int = 16384
    min_genesis_time: int = 0
    genesis_delay: int = 604800
    genesis_fork_version: bytes = field(default_factory=lambda: bytes.fromhex("00000000"))

    altair_fork_version: bytes = field(default_factory=lambda: bytes.fromhex("01000000"))
    altair_fork_epoch: int = 2**64 - 1
    bellatrix_fork_version: bytes = field(default_factory=lambda: bytes.fromhex("02000000"))
    bellatrix_fork_epoch: int = 2**64 - 1
    capella_fork_version: bytes = field(default_factory=lambda: bytes.fromhex("03000000"))
    capella_fork_epoch: int = 2**64 - 1
    deneb_fork_version: bytes = field(default_factory=lambda: bytes.fromhex("04000000"))
    deneb_fork_epoch: int = 2**64 - 1
    electra_fork_version: bytes = field(default_factory=lambda: bytes.fromhex("05000000"))
    electra_fork_epoch: int = 2**64 - 1
    fulu_fork_version: bytes = field(default_factory=lambda: bytes.fromhex("06000000"))
    fulu_fork_epoch: int = 2**64 - 1
    gloas_fork_version: bytes = field(default_factory=lambda: bytes.fromhex("07000000"))
    gloas_fork_epoch: int = 2**64 - 1

    terminal_total_difficulty: int = 2**256 - 1
    terminal_block_hash: bytes = field(default_factory=lambda: b"\x00" * 32)
    terminal_block_hash_activation_epoch: int = 2**64 - 1

    inactivity_score_bias: int = 4
    inactivity_score_recovery_rate: int = 16
    ejection_balance: int = 16 * 10**9
    min_per_epoch_churn_limit: int = 4
    churn_limit_quotient: int = 65536

    proposer_score_boost: int = 40
    reorg_head_weight_threshold: int = 20
    reorg_parent_weight_threshold: int = 160
    reorg_max_epochs_since_finalization: int = 2

    deposit_chain_id: int = 1
    deposit_network_id: int = 1
    deposit_contract_address: bytes = field(
        default_factory=lambda: bytes.fromhex("00000000219ab540356cBB839Cbe05303d7705Fa")
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NetworkConfig":
        """Load network config from yaml file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls._from_dict(data)

    @classmethod
    def from_upstream(cls, preset: str = "mainnet") -> "NetworkConfig":
        """Fetch network config from upstream consensus-specs repository."""
        if preset == "minimal":
            url = UPSTREAM_MINIMAL_CONFIG_URL
        else:
            url = UPSTREAM_MAINNET_CONFIG_URL

        logger.info(f"Fetching {preset} config from {url}")
        try:
            with urlopen(url, timeout=10) as response:
                data = yaml.safe_load(response.read().decode())
            return cls._from_dict(data)
        except URLError as e:
            logger.warning(f"Failed to fetch upstream config: {e}")
            raise

    @classmethod
    def minimal(cls) -> "NetworkConfig":
        """Create a minimal preset configuration for testing."""
        config = cls()
        config.config_name = "minimal"
        config.preset_base = "minimal"
        config.seconds_per_slot = 6
        config.min_genesis_active_validator_count = 64
        config.genesis_delay = 300
        config.min_validator_withdrawability_delay = 256
        config.shard_committee_period = 64
        config.eth1_follow_distance = 16
        return config

    @classmethod
    def _from_dict(cls, data: dict) -> "NetworkConfig":
        """Create config from dictionary."""
        config = cls()
        fork_version_fields = {
            "genesis_fork_version",
            "altair_fork_version",
            "bellatrix_fork_version",
            "capella_fork_version",
            "deneb_fork_version",
            "electra_fork_version",
            "fulu_fork_version",
            "gloas_fork_version",
        }
        for key, value in data.items():
            attr_name = key.lower()
            if hasattr(config, attr_name):
                if isinstance(value, str) and value.startswith("0x"):
                    value = bytes.fromhex(value[2:])
                elif attr_name in fork_version_fields and isinstance(value, int):
                    value = value.to_bytes(4, "big")
                elif attr_name == "blob_schedule" and isinstance(value, list):
                    pass
                setattr(config, attr_name, value)

        return config

    def get_fork_version(self, epoch: int) -> bytes:
        """Get the fork version active at the given epoch."""
        if epoch >= self.gloas_fork_epoch:
            return self.gloas_fork_version
        if epoch >= self.fulu_fork_epoch:
            return self.fulu_fork_version
        if epoch >= self.electra_fork_epoch:
            return self.electra_fork_version
        if epoch >= self.deneb_fork_epoch:
            return self.deneb_fork_version
        if epoch >= self.capella_fork_epoch:
            return self.capella_fork_version
        if epoch >= self.bellatrix_fork_epoch:
            return self.bellatrix_fork_version
        if epoch >= self.altair_fork_epoch:
            return self.altair_fork_version
        return self.genesis_fork_version


_config: NetworkConfig | None = None


def get_config() -> NetworkConfig:
    """Get the current network config (singleton)."""
    global _config
    if _config is None:
        _config = NetworkConfig()
    return _config


def load_config(path: str | Path) -> NetworkConfig:
    """Load network config from yaml and set as current."""
    global _config
    from .constants import set_preset

    _config = NetworkConfig.from_yaml(path)
    set_preset(_config.preset_base)
    return _config


def load_config_from_upstream(preset: str = "mainnet") -> NetworkConfig:
    """Load network config from upstream and set as current."""
    global _config
    from .constants import set_preset

    set_preset(preset)
    _config = NetworkConfig.from_upstream(preset)
    return _config


def set_config(config: NetworkConfig) -> None:
    """Set the current network config."""
    global _config
    _config = config
