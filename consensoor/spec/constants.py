"""Consensus spec presets (hardcoded, affect SSZ type sizes).

These are preset values that define SSZ container sizes and must be consistent
across the network. Config values (fork epochs, timing, etc.) are loaded from
network_config.yaml at runtime.

Organized by fork for maintainability when spec changes occur.

Supports both mainnet and minimal presets.
"""

from typing import Final

# =============================================================================
# Preset Selection
# =============================================================================

class _PresetConfig:
    """Singleton to hold preset configuration."""
    _instance = None
    _preset: str = "mainnet"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def preset(self) -> str:
        return self._preset

    @preset.setter
    def preset(self, value: str) -> None:
        if value not in ("mainnet", "minimal"):
            raise ValueError(f"Unknown preset: {value}")
        self._preset = value

_config = _PresetConfig()


def set_preset(preset: str) -> None:
    """Set the active preset (mainnet or minimal)."""
    _config.preset = preset


def get_preset() -> str:
    """Get the active preset."""
    return _config.preset


# =============================================================================
# Phase 0
# =============================================================================

GENESIS_SLOT: Final[int] = 0
GENESIS_EPOCH: Final[int] = 0
FAR_FUTURE_EPOCH: Final[int] = 2**64 - 1
UNSET_DEPOSIT_REQUESTS_START_INDEX: Final[int] = 2**64 - 1


def SLOTS_PER_EPOCH() -> int:
    return 32 if _config.preset == "mainnet" else 8


def TARGET_COMMITTEE_SIZE() -> int:
    return 128 if _config.preset == "mainnet" else 4


def EPOCHS_PER_HISTORICAL_VECTOR() -> int:
    return 65536 if _config.preset == "mainnet" else 64


def EPOCHS_PER_SLASHINGS_VECTOR() -> int:
    return 8192 if _config.preset == "mainnet" else 64


def EPOCHS_PER_ETH1_VOTING_PERIOD() -> int:
    return 64 if _config.preset == "mainnet" else 4


def SLOTS_PER_HISTORICAL_ROOT() -> int:
    return 8192 if _config.preset == "mainnet" else 64


MAX_SEED_LOOKAHEAD: Final[int] = 4
MIN_SEED_LOOKAHEAD: Final[int] = 1


def SHUFFLE_ROUND_COUNT() -> int:
    return 90 if _config.preset == "mainnet" else 10


JUSTIFICATION_BITS_LENGTH: Final[int] = 4
HISTORICAL_ROOTS_LIMIT: Final[int] = 2**24
VALIDATOR_REGISTRY_LIMIT: Final[int] = 2**40


def MAX_VALIDATORS_PER_COMMITTEE() -> int:
    return 2048 if _config.preset == "mainnet" else 2048


def MAX_COMMITTEES_PER_SLOT() -> int:
    return 64 if _config.preset == "mainnet" else 4


MAX_PROPOSER_SLASHINGS: Final[int] = 16
MAX_ATTESTER_SLASHINGS: Final[int] = 2  # Pre-Electra (Phase0-Deneb)
MAX_ATTESTER_SLASHINGS_PRE_ELECTRA: Final[int] = 2
MAX_DEPOSITS: Final[int] = 16
MAX_VOLUNTARY_EXITS: Final[int] = 16


def MAX_ATTESTATIONS() -> int:
    """Phase0 MAX_ATTESTATIONS - mainnet=128, minimal=128."""
    return 128

MIN_DEPOSIT_AMOUNT: Final[int] = 10**9
MAX_EFFECTIVE_BALANCE: Final[int] = 32 * 10**9
EFFECTIVE_BALANCE_INCREMENT: Final[int] = 10**9

DEPOSIT_CONTRACT_TREE_DEPTH: Final[int] = 32

MIN_ATTESTATION_INCLUSION_DELAY: Final[int] = 1
MIN_EPOCHS_TO_INACTIVITY_PENALTY: Final[int] = 4

HYSTERESIS_QUOTIENT: Final[int] = 4
HYSTERESIS_DOWNWARD_MULTIPLIER: Final[int] = 1
HYSTERESIS_UPWARD_MULTIPLIER: Final[int] = 5
BASE_REWARD_FACTOR: Final[int] = 64
WHISTLEBLOWER_REWARD_QUOTIENT: Final[int] = 512
PROPOSER_REWARD_QUOTIENT: Final[int] = 8


def INACTIVITY_PENALTY_QUOTIENT() -> int:
    return 2**26 if _config.preset == "mainnet" else 2**25


def MIN_SLASHING_PENALTY_QUOTIENT() -> int:
    return 128 if _config.preset == "mainnet" else 64


def PROPORTIONAL_SLASHING_MULTIPLIER() -> int:
    """Mainnet=1, Minimal=2 (higher for faster testing)."""
    return 1 if _config.preset == "mainnet" else 2


DOMAIN_BEACON_PROPOSER: Final[bytes] = b"\x00\x00\x00\x00"
DOMAIN_BEACON_ATTESTER: Final[bytes] = b"\x01\x00\x00\x00"
DOMAIN_RANDAO: Final[bytes] = b"\x02\x00\x00\x00"
DOMAIN_DEPOSIT: Final[bytes] = b"\x03\x00\x00\x00"
DOMAIN_VOLUNTARY_EXIT: Final[bytes] = b"\x04\x00\x00\x00"
DOMAIN_SELECTION_PROOF: Final[bytes] = b"\x05\x00\x00\x00"
DOMAIN_AGGREGATE_AND_PROOF: Final[bytes] = b"\x06\x00\x00\x00"

BLS_WITHDRAWAL_PREFIX: Final[int] = 0x00

MIN_VALIDATOR_WITHDRAWABILITY_DELAY: Final[int] = 256
def SHARD_COMMITTEE_PERIOD() -> int:
    return 256 if _config.preset == "mainnet" else 64

TARGET_AGGREGATORS_PER_COMMITTEE: Final[int] = 16

BASE_REWARDS_PER_EPOCH: Final[int] = 4


def CHURN_LIMIT_QUOTIENT() -> int:
    return 65536 if _config.preset == "mainnet" else 32


def MIN_PER_EPOCH_CHURN_LIMIT() -> int:
    return 4 if _config.preset == "mainnet" else 2
EJECTION_BALANCE: Final[int] = 16 * 10**9

# =============================================================================
# Altair
# =============================================================================

TIMELY_SOURCE_FLAG_INDEX: Final[int] = 0
TIMELY_TARGET_FLAG_INDEX: Final[int] = 1
TIMELY_HEAD_FLAG_INDEX: Final[int] = 2

TIMELY_SOURCE_WEIGHT: Final[int] = 14
TIMELY_TARGET_WEIGHT: Final[int] = 26
TIMELY_HEAD_WEIGHT: Final[int] = 14
PARTICIPATION_FLAG_WEIGHTS: Final[list[int]] = [TIMELY_SOURCE_WEIGHT, TIMELY_TARGET_WEIGHT, TIMELY_HEAD_WEIGHT]
SYNC_REWARD_WEIGHT: Final[int] = 2
PROPOSER_WEIGHT: Final[int] = 8
WEIGHT_DENOMINATOR: Final[int] = 64


def SYNC_COMMITTEE_SIZE() -> int:
    return 512 if _config.preset == "mainnet" else 32


def EPOCHS_PER_SYNC_COMMITTEE_PERIOD() -> int:
    return 256 if _config.preset == "mainnet" else 8


TARGET_AGGREGATORS_PER_SYNC_SUBCOMMITTEE: Final[int] = 16
SYNC_COMMITTEE_SUBNET_COUNT: Final[int] = 4
MIN_SYNC_COMMITTEE_PARTICIPANTS: Final[int] = 1


def UPDATE_TIMEOUT() -> int:
    """SLOTS_PER_EPOCH * EPOCHS_PER_SYNC_COMMITTEE_PERIOD (mainnet=8192, minimal=64)."""
    return SLOTS_PER_EPOCH() * EPOCHS_PER_SYNC_COMMITTEE_PERIOD()

INACTIVITY_PENALTY_QUOTIENT_ALTAIR: Final[int] = 3 * 2**24  # 50331648
MIN_SLASHING_PENALTY_QUOTIENT_ALTAIR: Final[int] = 64
PROPORTIONAL_SLASHING_MULTIPLIER_ALTAIR: Final[int] = 2
INACTIVITY_SCORE_BIAS: Final[int] = 4
INACTIVITY_SCORE_RECOVERY_RATE: Final[int] = 16

DOMAIN_SYNC_COMMITTEE: Final[bytes] = b"\x07\x00\x00\x00"
DOMAIN_SYNC_COMMITTEE_SELECTION_PROOF: Final[bytes] = b"\x08\x00\x00\x00"
DOMAIN_CONTRIBUTION_AND_PROOF: Final[bytes] = b"\x09\x00\x00\x00"

FINALIZED_ROOT_DEPTH: Final[int] = 6
CURRENT_SYNC_COMMITTEE_DEPTH: Final[int] = 5
NEXT_SYNC_COMMITTEE_DEPTH: Final[int] = 5

# =============================================================================
# Bellatrix (The Merge)
# =============================================================================

BYTES_PER_LOGS_BLOOM: Final[int] = 256
MAX_EXTRA_DATA_BYTES: Final[int] = 32
MAX_BYTES_PER_TRANSACTION: Final[int] = 2**30  # 1073741824
MAX_TRANSACTIONS_PER_PAYLOAD: Final[int] = 2**20  # 1048576

INACTIVITY_PENALTY_QUOTIENT_BELLATRIX: Final[int] = 2**24  # 16777216
MIN_SLASHING_PENALTY_QUOTIENT_BELLATRIX: Final[int] = 32
PROPORTIONAL_SLASHING_MULTIPLIER_BELLATRIX: Final[int] = 3

# =============================================================================
# Capella
# =============================================================================

MAX_BLS_TO_EXECUTION_CHANGES: Final[int] = 16


def MAX_VALIDATORS_PER_WITHDRAWALS_SWEEP() -> int:
    return 2**14 if _config.preset == "mainnet" else 16  # 16384 for mainnet, 16 for minimal


def MAX_WITHDRAWALS_PER_PAYLOAD() -> int:
    return 16 if _config.preset == "mainnet" else 4


DOMAIN_BLS_TO_EXECUTION_CHANGE: Final[bytes] = b"\x0a\x00\x00\x00"

ETH1_ADDRESS_WITHDRAWAL_PREFIX: Final[int] = 0x01

EXECUTION_PAYLOAD_DEPTH: Final[int] = 4

# =============================================================================
# Deneb
# =============================================================================

MAX_BLOB_COMMITMENTS_PER_BLOCK: Final[int] = 4096
FIELD_ELEMENTS_PER_BLOB: Final[int] = 4096
KZG_COMMITMENT_INCLUSION_PROOF_DEPTH: Final[int] = 17


def MAX_BLOBS_PER_BLOCK() -> int:
    from .network_config import get_config
    return get_config().max_blobs_per_block


def BLOB_SIDECAR_SUBNET_COUNT() -> int:
    from .network_config import get_config
    return get_config().blob_sidecar_subnet_count


def MAX_REQUEST_BLOCKS_DENEB() -> int:
    from .network_config import get_config
    return get_config().max_request_blocks_deneb


def MIN_EPOCHS_FOR_BLOB_SIDECARS_REQUESTS() -> int:
    from .network_config import get_config
    return get_config().min_epochs_for_blob_sidecars_requests


def MAX_REQUEST_BLOB_SIDECARS() -> int:
    from .network_config import get_config
    cfg = get_config()
    return cfg.max_request_blocks_deneb * cfg.max_blobs_per_block


# =============================================================================
# Electra
# =============================================================================

MAX_ATTESTER_SLASHINGS_ELECTRA: Final[int] = 1


def MAX_ATTESTATIONS_ELECTRA() -> int:
    """Electra MAX_ATTESTATIONS - mainnet=8, minimal=8."""
    return 8

MIN_ACTIVATION_BALANCE: Final[int] = 32 * 10**9
MAX_EFFECTIVE_BALANCE_ELECTRA: Final[int] = 2048 * 10**9

MIN_SLASHING_PENALTY_QUOTIENT_ELECTRA: Final[int] = 4096
WHISTLEBLOWER_REWARD_QUOTIENT_ELECTRA: Final[int] = 4096


def PENDING_DEPOSITS_LIMIT() -> int:
    return 2**27 if _config.preset == "mainnet" else 2**27


def PENDING_PARTIAL_WITHDRAWALS_LIMIT() -> int:
    return 2**27 if _config.preset == "mainnet" else 64


def PENDING_CONSOLIDATIONS_LIMIT() -> int:
    return 2**18 if _config.preset == "mainnet" else 64


MAX_DEPOSIT_REQUESTS_PER_PAYLOAD: Final[int] = 8192
MAX_WITHDRAWAL_REQUESTS_PER_PAYLOAD: Final[int] = 16
MAX_CONSOLIDATION_REQUESTS_PER_PAYLOAD: Final[int] = 2
def MAX_PENDING_PARTIALS_PER_WITHDRAWALS_SWEEP() -> int:
    return 8 if _config.preset == "mainnet" else 2
MAX_PENDING_DEPOSITS_PER_EPOCH: Final[int] = 16

KZG_COMMITMENTS_INCLUSION_PROOF_DEPTH_ELECTRA: Final[int] = 4

# Electra LightClient depths (changed due to new BeaconState fields)
FINALIZED_ROOT_DEPTH_ELECTRA: Final[int] = 7
CURRENT_SYNC_COMMITTEE_DEPTH_ELECTRA: Final[int] = 6
NEXT_SYNC_COMMITTEE_DEPTH_ELECTRA: Final[int] = 6

# Gloas LightClient (EIP-7688 progressive BeaconState/BeaconBlockBody move the gindices)
FINALIZED_ROOT_GINDEX_GLOAS: Final[int] = 735
CURRENT_SYNC_COMMITTEE_GINDEX_GLOAS: Final[int] = 2945
NEXT_SYNC_COMMITTEE_GINDEX_GLOAS: Final[int] = 2946
EXECUTION_BLOCK_HASH_GINDEX_GLOAS: Final[int] = 2856
FINALIZED_ROOT_DEPTH_GLOAS: Final[int] = 9  # floorlog2(735)
CURRENT_SYNC_COMMITTEE_DEPTH_GLOAS: Final[int] = 11  # floorlog2(2945)
NEXT_SYNC_COMMITTEE_DEPTH_GLOAS: Final[int] = 11  # floorlog2(2946)
EXECUTION_BLOCK_HASH_DEPTH_GLOAS: Final[int] = 11  # floorlog2(2856)

def MIN_PER_EPOCH_CHURN_LIMIT_ELECTRA() -> int:
    from .network_config import get_config
    return get_config().min_per_epoch_churn_limit_electra


def MAX_PER_EPOCH_ACTIVATION_EXIT_CHURN_LIMIT() -> int:
    from .network_config import get_config
    return get_config().max_per_epoch_activation_exit_churn_limit


def MAX_BLOBS_PER_BLOCK_ELECTRA() -> int:
    from .network_config import get_config
    return get_config().max_blobs_per_block_electra


def BLOB_SIDECAR_SUBNET_COUNT_ELECTRA() -> int:
    from .network_config import get_config
    return get_config().blob_sidecar_subnet_count_electra


def MAX_REQUEST_BLOB_SIDECARS_ELECTRA() -> int:
    from .network_config import get_config
    cfg = get_config()
    return cfg.max_request_blocks_deneb * cfg.max_blobs_per_block_electra

COMPOUNDING_WITHDRAWAL_PREFIX: Final[int] = 0x02

FULL_EXIT_REQUEST_AMOUNT: Final[int] = 0

# =============================================================================
# Fulu (PeerDAS)
# =============================================================================

# Fulu adds PeerDAS (data availability sampling) and deterministic proposer lookahead
# BeaconState gains proposer_lookahead field

FIELD_ELEMENTS_PER_CELL: Final[int] = 64
FIELD_ELEMENTS_PER_EXT_BLOB: Final[int] = 8192
CELLS_PER_EXT_BLOB: Final[int] = 128
NUMBER_OF_COLUMNS: Final[int] = 128
MAX_REQUEST_DATA_COLUMN_SIDECARS: Final[int] = 16384


def NUMBER_OF_CUSTODY_GROUPS() -> int:
    from .network_config import get_config
    return get_config().number_of_custody_groups


def DATA_COLUMN_SIDECAR_SUBNET_COUNT() -> int:
    from .network_config import get_config
    return get_config().data_column_sidecar_subnet_count


def SAMPLES_PER_SLOT() -> int:
    from .network_config import get_config
    return get_config().samples_per_slot


def CUSTODY_REQUIREMENT() -> int:
    from .network_config import get_config
    return get_config().custody_requirement


def VALIDATOR_CUSTODY_REQUIREMENT() -> int:
    from .network_config import get_config
    return get_config().validator_custody_requirement


def BALANCE_PER_ADDITIONAL_CUSTODY_GROUP() -> int:
    from .network_config import get_config
    return get_config().balance_per_additional_custody_group


def MIN_EPOCHS_FOR_DATA_COLUMN_SIDECARS_REQUESTS() -> int:
    from .network_config import get_config
    return get_config().min_epochs_for_data_column_sidecars_requests

# =============================================================================
# Gloas (ePBS - Enshrined Proposer-Builder Separation)
# =============================================================================


def PTC_SIZE() -> int:
    return 512 if _config.preset == "mainnet" else 16


MAX_PAYLOAD_ATTESTATIONS: Final[int] = 4

# EIP-8282 (builder deposit/exit requests)
MAX_BUILDER_DEPOSIT_REQUESTS_PER_PAYLOAD: Final[int] = 2**8
MAX_BUILDER_EXIT_REQUESTS_PER_PAYLOAD: Final[int] = 2**4
BUILDER_DEPOSIT_REQUEST_TYPE: Final[bytes] = b"\x03"
BUILDER_EXIT_REQUEST_TYPE: Final[bytes] = b"\x04"


def MAX_BUILDERS_PER_WITHDRAWALS_SWEEP() -> int:
    return 2**14 if _config.preset == "mainnet" else 16
BUILDER_INDEX_FLAG: Final[int] = 2**40
BUILDER_INDEX_SELF_BUILD: Final[int] = 2**64 - 1
BUILDER_PAYMENT_THRESHOLD_NUMERATOR: Final[int] = 6
BUILDER_PAYMENT_THRESHOLD_DENOMINATOR: Final[int] = 10

DOMAIN_BEACON_BUILDER: Final[bytes] = b"\x0b\x00\x00\x00"
DOMAIN_PTC_ATTESTER: Final[bytes] = b"\x0c\x00\x00\x00"
DOMAIN_PROPOSER_PREFERENCES: Final[bytes] = b"\x0d\x00\x00\x00"
DOMAIN_BUILDER_DEPOSIT: Final[bytes] = b"\x0e\x00\x00\x00"

BUILDER_WITHDRAWAL_PREFIX: Final[bytes] = b"\xb0"


# Gloas churn limits (EIP-8061): split exit and activation churn limits
def CHURN_LIMIT_QUOTIENT_GLOAS() -> int:
    from .network_config import get_config
    return get_config().churn_limit_quotient_gloas


def MAX_PER_EPOCH_ACTIVATION_CHURN_LIMIT_GLOAS() -> int:
    from .network_config import get_config
    return get_config().max_per_epoch_activation_churn_limit_gloas


def CONSOLIDATION_CHURN_LIMIT_QUOTIENT() -> int:
    from .network_config import get_config
    return get_config().consolidation_churn_limit_quotient


def MIN_BUILDER_WITHDRAWABILITY_DELAY() -> int:
    from .network_config import get_config
    return get_config().min_builder_withdrawability_delay

# =============================================================================
# Networking Constants
# =============================================================================

ATTESTATION_SUBNET_PREFIX_BITS: Final[int] = 6


def MAX_PAYLOAD_SIZE() -> int:
    from .network_config import get_config
    return get_config().max_payload_size


def MAX_REQUEST_BLOCKS() -> int:
    from .network_config import get_config
    return get_config().max_request_blocks


def EPOCHS_PER_SUBNET_SUBSCRIPTION() -> int:
    from .network_config import get_config
    return get_config().epochs_per_subnet_subscription


def ATTESTATION_PROPAGATION_SLOT_RANGE() -> int:
    from .network_config import get_config
    return get_config().attestation_propagation_slot_range


def MAXIMUM_GOSSIP_CLOCK_DISPARITY() -> int:
    from .network_config import get_config
    return get_config().maximum_gossip_clock_disparity


def MESSAGE_DOMAIN_INVALID_SNAPPY() -> bytes:
    from .network_config import get_config
    return get_config().message_domain_invalid_snappy


def MESSAGE_DOMAIN_VALID_SNAPPY() -> bytes:
    from .network_config import get_config
    return get_config().message_domain_valid_snappy


def SUBNETS_PER_NODE() -> int:
    from .network_config import get_config
    return get_config().subnets_per_node


def ATTESTATION_SUBNET_COUNT() -> int:
    from .network_config import get_config
    return get_config().attestation_subnet_count


def ATTESTATION_SUBNET_EXTRA_BITS() -> int:
    from .network_config import get_config
    return get_config().attestation_subnet_extra_bits


def MIN_EPOCHS_FOR_BLOCK_REQUESTS() -> int:
    return 33024 if _config.preset == "mainnet" else 272


def MAX_PER_EPOCH_ACTIVATION_CHURN_LIMIT() -> int:
    from .network_config import get_config
    return get_config().max_per_epoch_activation_churn_limit


