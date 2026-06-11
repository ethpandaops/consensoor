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

    slot_duration_ms: int = 12000
    seconds_per_eth1_block: int = 14

    # Intra-slot timing (basis points = hundredths of a percent)
    proposer_reorg_cutoff_bps: int = 1667  # ~17% of slot
    confirmation_byzantine_threshold: int = 25  # fast-confirmation rule, percent
    attestation_due_bps: int = 3333  # ~33% of slot (1/3 mark)
    aggregate_due_bps: int = 6667  # ~67% of slot (2/3 mark)
    sync_message_due_bps: int = 3333
    contribution_due_bps: int = 6667

    # Gloas (ePBS) timing - different values for ePBS slots
    attestation_due_bps_gloas: int = 2500  # 25% of slot
    aggregate_due_bps_gloas: int = 5000  # 50% of slot
    sync_message_due_bps_gloas: int = 2500
    contribution_due_bps_gloas: int = 5000
    payload_due_bps: int = 7500  # 75% of slot (Gloas alpha 8)
    payload_attestation_due_bps: int = 7500  # 75% of slot

    # EIP-7805 timing
    view_freeze_cutoff_bps: int = 7500
    inclusion_list_submission_due_bps: int = 6667
    proposer_inclusion_list_cutoff_bps: int = 9167
    min_validator_withdrawability_delay: int = 256
    shard_committee_period: int = 256
    eth1_follow_distance: int = 2048
    min_builder_withdrawability_delay: int = 8192

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

    # Deneb churn (deprecated, still in spec config)
    max_per_epoch_activation_churn_limit: int = 8

    # Electra churn
    min_per_epoch_churn_limit_electra: int = 128 * 10**9
    max_per_epoch_activation_exit_churn_limit: int = 256 * 10**9

    # Gloas churn (EIP-8061)
    churn_limit_quotient_gloas: int = 32768
    consolidation_churn_limit_quotient: int = 65536
    max_per_epoch_activation_churn_limit_gloas: int = 256 * 10**9

    proposer_score_boost: int = 40
    reorg_head_weight_threshold: int = 20
    reorg_parent_weight_threshold: int = 160
    reorg_max_epochs_since_finalization: int = 2

    deposit_chain_id: int = 1
    deposit_network_id: int = 1
    deposit_contract_address: bytes = field(
        default_factory=lambda: bytes.fromhex("00000000219ab540356cBB839Cbe05303d7705Fa")
    )

    # Networking config
    max_payload_size: int = 10 * 2**20
    max_request_blocks: int = 1024
    epochs_per_subnet_subscription: int = 256
    attestation_propagation_slot_range: int = 32
    maximum_gossip_clock_disparity: int = 500
    message_domain_invalid_snappy: bytes = field(default_factory=lambda: b"\x00\x00\x00\x00")
    message_domain_valid_snappy: bytes = field(default_factory=lambda: b"\x01\x00\x00\x00")
    subnets_per_node: int = 2
    attestation_subnet_count: int = 64
    attestation_subnet_extra_bits: int = 0

    # Deneb networking
    max_request_blocks_deneb: int = 128
    min_epochs_for_blob_sidecars_requests: int = 4096
    blob_sidecar_subnet_count: int = 6
    max_blobs_per_block: int = 6

    # Electra networking
    blob_sidecar_subnet_count_electra: int = 9
    max_blobs_per_block_electra: int = 9

    # Fulu (PeerDAS) networking
    number_of_custody_groups: int = 128
    data_column_sidecar_subnet_count: int = 128
    samples_per_slot: int = 8
    custody_requirement: int = 4
    validator_custody_requirement: int = 8
    balance_per_additional_custody_group: int = 32 * 10**9
    min_epochs_for_data_column_sidecars_requests: int = 4096

    # Gloas networking
    max_request_payloads: int = 128

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
        config.slot_duration_ms = 6000
        config.min_genesis_active_validator_count = 64
        config.genesis_delay = 300
        config.min_validator_withdrawability_delay = 256
        config.shard_committee_period = 64
        config.eth1_follow_distance = 16
        # Minimal preset uses different fork versions (end in 01 instead of 00)
        config.genesis_fork_version = bytes.fromhex("00000001")
        config.altair_fork_version = bytes.fromhex("01000001")
        config.bellatrix_fork_version = bytes.fromhex("02000001")
        config.capella_fork_version = bytes.fromhex("03000001")
        config.deneb_fork_version = bytes.fromhex("04000001")
        config.electra_fork_version = bytes.fromhex("05000001")
        config.fulu_fork_version = bytes.fromhex("06000001")
        config.gloas_fork_version = bytes.fromhex("07000001")
        # Timing values (same as mainnet by default, loaded from config)
        config.attestation_due_bps = 3333
        config.aggregate_due_bps = 6667
        config.attestation_due_bps_gloas = 2500
        config.aggregate_due_bps_gloas = 5000
        config.payload_due_bps = 7500
        config.payload_attestation_due_bps = 7500
        # Validator cycle (per minimal.yaml)
        config.min_per_epoch_churn_limit = 2
        config.churn_limit_quotient = 32
        config.max_per_epoch_activation_churn_limit = 4
        config.min_per_epoch_churn_limit_electra = 64 * 10**9
        config.max_per_epoch_activation_exit_churn_limit = 128 * 10**9
        # Gloas churn quotients are the same as mainnet in minimal.yaml
        # Gloas
        config.min_builder_withdrawability_delay = 2
        # Deposit contract (per minimal.yaml)
        config.deposit_chain_id = 5
        config.deposit_network_id = 5
        config.deposit_contract_address = bytes.fromhex(
            "1234567890123456789012345678901234567890"
        )
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
                    value = [
                        {k.lower(): v for k, v in entry.items()} for entry in value
                    ]
                setattr(config, attr_name, value)

        # Log fork epochs for debugging
        logger.info(
            f"Config loaded: fulu_fork_epoch={config.fulu_fork_epoch}, "
            f"gloas_fork_epoch={config.gloas_fork_epoch}, "
            f"electra_fork_epoch={config.electra_fork_epoch}"
        )
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

    def get_fork_schedule(self) -> list[tuple[int, bytes, str]]:
        """Get all forks in chronological order.

        Returns a list of (epoch, version, name) tuples sorted by epoch.
        Only includes forks that are scheduled (epoch < FAR_FUTURE_EPOCH).
        """
        FAR_FUTURE_EPOCH = 2**64 - 1
        forks = [
            (0, self.genesis_fork_version, "phase0"),
            (self.altair_fork_epoch, self.altair_fork_version, "altair"),
            (self.bellatrix_fork_epoch, self.bellatrix_fork_version, "bellatrix"),
            (self.capella_fork_epoch, self.capella_fork_version, "capella"),
            (self.deneb_fork_epoch, self.deneb_fork_version, "deneb"),
            (self.electra_fork_epoch, self.electra_fork_version, "electra"),
            (self.fulu_fork_epoch, self.fulu_fork_version, "fulu"),
            (self.gloas_fork_epoch, self.gloas_fork_version, "gloas"),
        ]
        scheduled = [(e, v, n) for e, v, n in forks if e < FAR_FUTURE_EPOCH]
        return sorted(scheduled, key=lambda x: x[0])

    def get_fork_at_epoch(self, epoch: int) -> tuple[int, bytes, str] | None:
        """Get the fork that activates exactly at the given epoch.

        Returns (epoch, version, name) if a fork activates at this epoch,
        or None if no fork activates at this epoch.
        """
        schedule = self.get_fork_schedule()
        for fork_epoch, fork_version, fork_name in schedule:
            if fork_epoch == epoch:
                return (fork_epoch, fork_version, fork_name)
        return None

    def is_gloas_active(self, epoch: int) -> bool:
        """Check if Gloas (ePBS) fork is active at the given epoch."""
        return epoch >= self.gloas_fork_epoch

    def fork_name_at_epoch(self, epoch: int) -> str:
        """Return the spec fork name active at the given epoch.

        Walks the config-declared fork schedule (genesis/altair/.../gloas)
        and returns the name of the most recent fork whose activation
        epoch is <= `epoch`. Use this anywhere we need to know which fork
        we're on without inspecting block/state structure — the structure
        check only tells us how SSZ was decoded, not what the spec
        thinks the active fork is.
        """
        if epoch >= self.gloas_fork_epoch:
            return "gloas"
        if epoch >= self.fulu_fork_epoch:
            return "fulu"
        if epoch >= self.electra_fork_epoch:
            return "electra"
        if epoch >= self.deneb_fork_epoch:
            return "deneb"
        if epoch >= self.capella_fork_epoch:
            return "capella"
        if epoch >= self.bellatrix_fork_epoch:
            return "bellatrix"
        if epoch >= self.altair_fork_epoch:
            return "altair"
        return "phase0"

    def fork_name_at_slot(self, slot: int) -> str:
        """Return the spec fork name active at the given slot."""
        from .constants import SLOTS_PER_EPOCH
        return self.fork_name_at_epoch(slot // SLOTS_PER_EPOCH())

    def get_attestation_due_offset(self, epoch: int) -> float:
        """Get attestation due time offset in seconds for the given epoch.

        Returns the time offset from slot start when attestations should be produced.
        Uses Gloas timing if ePBS is active.
        """
        slot_duration = self.slot_duration_ms / 1000.0
        if self.is_gloas_active(epoch):
            bps = self.attestation_due_bps_gloas
        else:
            bps = self.attestation_due_bps
        return slot_duration * (bps / 10000.0)

    def get_aggregate_due_offset(self, epoch: int) -> float:
        """Get aggregate due time offset in seconds for the given epoch.

        Returns the time offset from slot start when aggregates should be published.
        Uses Gloas timing if ePBS is active.
        """
        slot_duration = self.slot_duration_ms / 1000.0
        if self.is_gloas_active(epoch):
            bps = self.aggregate_due_bps_gloas
        else:
            bps = self.aggregate_due_bps
        return slot_duration * (bps / 10000.0)

    def get_payload_attestation_due_offset(self) -> float:
        """Get payload attestation (PTC) due time offset in seconds.

        Only applicable in Gloas (ePBS) for Payload Timeliness Committee attestations.
        """
        slot_duration = self.slot_duration_ms / 1000.0
        return slot_duration * (self.payload_attestation_due_bps / 10000.0)

    def get_payload_due_offset(self) -> float:
        """Payload due offset in seconds (PTC validators consider payloads seen after this stale)."""
        slot_duration = self.slot_duration_ms / 1000.0
        return slot_duration * (self.payload_due_bps / 10000.0)


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
