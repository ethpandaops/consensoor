# consensoor

Lightweight Python consensus layer client for local testing and prototyping.

## Features

- Implements Gloas consensus spec (EIP-7732 ePBS)
- libp2p-based P2P networking with gossipsub
- ENR generation with eth2 field for network identification
- State synchronization from upstream beacon node (checkpoint sync)
- Engine API client for execution layer integration
- Validator key loading (EIP-2335 keystores)
- Block production when assigned as proposer
- Supports both mainnet and minimal presets
- Auto-fetches upstream config from ethereum/consensus-specs if not provided
- Designed for Kurtosis local devnets
- Dynamic graffiti with EL+CL version encoding
- Automatic git version injection via setuptools_scm
- Prometheus metrics for monitoring
- LevelDB-backed persistent storage

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Platform-specific dependencies:
  - **Linux**: `apt install libgmp-dev libleveldb-dev`
  - **macOS**: `brew install gmp leveldb`

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `blspy` | Fast BLS12-381 cryptography (C/assembly, 100x faster than py_ecc) |
| `plyvel` | LevelDB bindings for state/block storage |
| `remerkleable` | SSZ serialization and Merkleization |
| `libp2p` | P2P networking with gossipsub |
| `aiohttp` | Async HTTP for Engine API and Beacon API |

## Installation

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Usage

```bash
consensoor run \
  --genesis-state /path/to/genesis.ssz \
  --engine-api-url http://localhost:8551 \
  --jwt-secret /path/to/jwt.hex \
  --bootnodes enr:-... \
  --p2p-port 9000 \
  --beacon-api-port 5052 \
  --preset mainnet \
  --checkpoint-sync-url http://lighthouse:5052
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--engine-api-url` | Engine API URL | `http://localhost:8551` |
| `--jwt-secret` | Path to JWT secret file | - |
| `--genesis-state` | Path to genesis state SSZ | Required |
| `--network-config` | Path to network config YAML | Fetched from upstream |
| `--preset` | Preset (mainnet/minimal) | `mainnet` |
| `--p2p-port` | TCP/UDP port for P2P | `9000` |
| `--p2p-host` | Host to bind P2P | `0.0.0.0` |
| `--beacon-api-port` | Beacon API HTTP port | `5052` |
| `--metrics-port` | Prometheus metrics HTTP port | `8008` |
| `--bootnodes` | Bootnode ENRs (repeatable) | - |
| `--checkpoint-sync-url` | Upstream beacon URL for state sync | - |
| `--validator-keys` | Validator keystores (format: keystores:secrets) | - |
| `--data-dir` | Data directory | `./data` |
| `--log-level` | Logging level | `INFO` |

All options can also be set via environment variables with the `CONSENSOOR_` prefix.

## Docker

```bash
docker build -t consensoor .

docker run consensoor run \
  --genesis-state /data/genesis.ssz \
  --engine-api-url http://el:8551 \
  --preset minimal
```

The Docker build automatically embeds the git commit hash via `setuptools_scm`. This enables:
- Automatic version tracking in logs and metrics
- EL+CL version graffiti (see below)

## Graffiti with Version Encoding

Consensoor encodes EL and CL client version info in the block graffiti:

```
GEabcdCOxxxx consensoor
│ │    │ │
│ │    │ └── CL commit (first 4 chars)
│ │    └──── CL client code (CO = consensoor)
│ └───────── EL commit (first 4 chars)
└─────────── EL client code (GE = Geth, RH = Reth, etc.)
```

**Client codes:**
| Code | EL Client | Code | CL Client |
|------|-----------|------|-----------|
| GE | Geth | CO | Consensoor |
| NM | Nethermind | LH | Lighthouse |
| BU | Besu | PR | Prysm |
| ER | Erigon | TK | Teku |
| RH | Reth | NB | Nimbus |

The EL client info is obtained via `engine_getClientVersionV1`. If unavailable, only the CL info is included.

## Architecture

```
consensoor/
├── spec/               # Consensus spec implementation
│   ├── types/          # SSZ containers (BeaconState, BeaconBlock, etc.)
│   ├── constants.py    # Preset values organized by fork
│   └── network_config.py   # Runtime config from YAML or upstream
├── crypto/             # BLS signatures (blspy), hashing
│   └── crypto.py       # BLS sign/verify, hash_tree_root
├── p2p/                # libp2p networking
│   ├── host.py         # P2P host with ENR generation
│   ├── gossip.py       # Beacon gossip topics
│   └── encoding.py     # Message encoding (snappy)
├── network/            # UDP gossip layer (legacy)
│   └── gossip.py       # Simple UDP gossip for local testing
├── engine/             # Engine API client
│   ├── types.py        # Payload types and responses
│   └── client.py       # JSON-RPC client
├── store/              # State and block storage (LevelDB)
│   └── store.py        # LevelDB-backed persistent store
├── metrics/            # Prometheus metrics
│   └── metrics.py      # Metric definitions and helpers
├── validator/          # Validator duties
│   ├── types.py        # ValidatorKey, ProposerDuty, etc.
│   ├── shuffling.py    # Proposer selection algorithms
│   ├── keystore.py     # EIP-2335 keystore loading
│   └── client.py       # ValidatorClient
├── builder/            # Block building
├── beacon_api/         # Beacon API HTTP server
│   ├── server.py       # HTTP routes
│   ├── spec.py         # /eth/v1/config/spec builder
│   └── utils.py        # Helper functions
├── beacon_sync/        # State synchronization
│   ├── client.py       # Remote beacon client (SSE)
│   └── sync.py         # State sync manager
├── version.py          # Version info and graffiti builder
├── node.py             # Main orchestration
├── config.py           # Node configuration
└── cli.py              # CLI entry point

tests/
├── spec/               # Consensus spec tests
│   ├── conftest.py     # Pytest fixtures
│   └── test_spec_runner.py  # Test runner for all forks
└── spec-tests/         # Downloaded spec tests (gitignored)
```

## P2P Networking

Consensoor uses libp2p with gossipsub for P2P networking:

- ENR includes `eth2` field for network identification
- Supports bootnode discovery via ENR
- Subscribes to beacon block and aggregate attestation topics
- Message encoding uses snappy framing format
- Dynamic fork digest calculation with blob parameters
- Req/resp protocol support (Status, Ping, Metadata)

## State Synchronization

When `--checkpoint-sync-url` is provided, consensoor syncs state from an upstream beacon node:

- Subscribes to SSE events for block and finality notifications
- Periodically syncs state at epoch boundaries
- Updates randao_mixes, validators, checkpoints from upstream
- Enables accurate proposer calculation via synced state

## Spec Tests

Run consensus spec tests against consensoor:

```bash
# Download spec tests and run a specific fork
make test phase0 minimal
make test altair minimal
make test bellatrix minimal
make test capella minimal
make test deneb minimal
make test electra minimal
make test fulu minimal
make test gloas minimal

# Run all forks
make test all minimal
make test all mainnet

# Default preset is mainnet
make test electra         # equivalent to: make test electra mainnet
```

Tests download from consensus-specs releases and cache locally.

## Spec Forks

Full BeaconState types implemented for all forks:

| Fork | SSZ Types | Description |
|------|-----------|-------------|
| Phase 0 | `Phase0BeaconState`, `Phase0BeaconBlock` | Base state with attestations |
| Altair | `AltairBeaconState`, `AltairBeaconBlock` | Sync committees, participation |
| Bellatrix | `BellatrixBeaconState`, `BellatrixBeaconBlock` | The Merge, execution payload |
| Capella | `CapellaBeaconState`, `CapellaBeaconBlock` | Withdrawals |
| Deneb | `DenebBeaconState`, `DenebBeaconBlock` | Blob gas |
| Electra | `ElectraBeaconState`, `ElectraBeaconBlock` | MaxEB, consolidations |
| Fulu | `FuluBeaconState` | PeerDAS, proposer lookahead |
| Gloas | `BeaconState`, `BeaconBlock` | ePBS (EIP-7732) |

Constants are organized by fork in `spec/constants.py`.

## Presets

Two presets are supported:

- **mainnet**: Production parameters (32 slots/epoch, etc.)
- **minimal**: Testing parameters (8 slots/epoch, etc.)

Config is automatically fetched from upstream consensus-specs if `--network-config` is not provided.

## Beacon API

Implements subset of standard Beacon API:

**Node:**
- `GET /eth/v1/node/health`
- `GET /eth/v1/node/version`
- `GET /eth/v1/node/syncing`
- `GET /eth/v1/node/identity`
- `GET /eth/v1/node/peers`

**Beacon:**
- `GET /eth/v1/beacon/genesis`
- `GET /eth/v1/beacon/headers`
- `GET /eth/v1/beacon/headers/{block_id}`
- `GET /eth/v2/beacon/blocks/{block_id}`
- `GET /eth/v1/beacon/blocks/{block_id}/root`
- `GET /eth/v1/beacon/blob_sidecars/{block_id}`
- `GET /eth/v1/beacon/states/{state_id}/root`
- `GET /eth/v1/beacon/states/{state_id}/fork`
- `GET /eth/v1/beacon/states/{state_id}/finality_checkpoints`
- `GET /eth/v1/beacon/states/{state_id}/validators`
- `GET /eth/v1/beacon/states/{state_id}/validators/{validator_id}`
- `GET /eth/v1/beacon/states/{state_id}/validator_balances`
- `GET /eth/v1/beacon/states/{state_id}/committees`
- `GET /eth/v1/beacon/states/{state_id}/sync_committees`
- `GET /eth/v1/beacon/states/{state_id}/randao`

**Config:**
- `GET /eth/v1/config/spec`
- `GET /eth/v1/config/fork_schedule`
- `GET /eth/v1/config/deposit_contract`

**Debug:**
- `GET /eth/v2/debug/beacon/states/{state_id}`

**Events:**
- `GET /eth/v1/events` (SSE: head, finalized_checkpoint)

Supports `genesis`, `head`, `finalized`, `justified`, and slot/root identifiers for state_id and block_id parameters.

## Prometheus Metrics

Consensoor exposes Prometheus-compatible metrics on port 8008 (configurable via `--metrics-port`).

**Node metrics:**
- `consensoor_head_slot` - Current head slot
- `consensoor_head_epoch` - Current head epoch
- `consensoor_finalized_epoch` - Current finalized epoch
- `consensoor_justified_epoch` - Current justified epoch
- `consensoor_syncing` - Whether the node is syncing (1) or synced (0)

**P2P metrics:**
- `consensoor_peers_connected` - Number of connected peers
- `consensoor_gossip_messages_received_total` - Messages received by topic
- `consensoor_gossip_messages_sent_total` - Messages sent by topic

**Validator metrics:**
- `consensoor_attestations_produced_total` - Attestations created
- `consensoor_blocks_proposed_total` - Blocks proposed
- `consensoor_blocks_received_total` - Blocks received from network

**Engine API metrics:**
- `consensoor_engine_api_requests_total` - Requests by method
- `consensoor_engine_api_errors_total` - Errors by method and type
- `consensoor_engine_api_latency_seconds` - Request latency histogram

**Performance metrics:**
- `consensoor_block_processing_seconds` - Block processing time histogram
- `consensoor_state_transition_seconds` - State transition time histogram

## Gloas ePBS Flow

In Gloas (EIP-7732), block production is separated from payload production:

```
1. Builder submits SignedExecutionPayloadBid
2. Proposer includes bid reference in BeaconBlock
3. Builder reveals SignedExecutionPayloadEnvelope
4. PTC (Payload Timeliness Committee) attests to payload availability
```

Consensoor implements the builder role with:
- Execution payload bid generation via `_build_execution_payload_bid`
- Separate execution payload building path for GLOAS blocks
- GLOAS-specific constants and domains exposed in `/eth/v1/config/spec`

## Limitations

This is a prototype for local testing. Not intended for production use.

- No full fork choice implementation
- Simplified state transition
- No slashing protection
- Falls back to py_ecc (slow) if blspy unavailable
- No proper sync protocol (relies on checkpoint sync)
- Limited validator/builder logic

## License

MIT
