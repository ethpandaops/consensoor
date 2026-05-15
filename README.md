# consensoor

Lightweight Python consensus layer client for local testing and prototyping.

## Features

- Implements Gloas consensus spec (EIP-7732 ePBS)
- Native Rust libp2p stack (`consensoor-p2p-rs`) — TCP+Noise+Yamux/Mplex, gossipsub, identify, ping, req/resp; replaces py-libp2p, which was too slow and unreliable under the GIL to keep up with gossip mesh management on a live devnet
- ENR generation with eth2 field for network identification
- State synchronization from upstream beacon node (checkpoint sync) plus gossipsub block sync from peers
- Engine API client with binary SSZ-over-REST transport (execution-apis [#764](https://github.com/ethereum/execution-apis/pull/764)) — auto-negotiated via `engine_exchangeCapabilities`, used by default when the EL advertises support; JSON-RPC is the fallback
- Validator key loading (EIP-2335 keystores), attestation/sync-committee/payload-attestation pools, block production when assigned as proposer
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
| `consensoor_p2p` | In-tree Rust libp2p binding (`consensoor-p2p-rs`); pyo3 wheel built with maturin, ships TCP+Noise+Yamux/Mplex, gossipsub, identify, ping and req/resp configured to Lighthouse's defaults |
| `aiohttp` | Async HTTP for Engine API (JSON-RPC + SSZ-REST) and Beacon API |

## Installation

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"

# Build & install the Rust p2p extension into the active venv
cd consensoor-p2p-rs && python3 -m maturin develop --release && cd ..
```

The Docker build does both wheels in a multi-stage image — see [Docker](#docker).

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
├── spec/                       # Consensus spec implementation
│   ├── types/                  # SSZ containers per fork (phase0…gloas)
│   ├── state_transition/       # Block / epoch processing, fork upgrades
│   ├── constants.py            # Preset values organized by fork
│   └── network_config.py       # Runtime config from YAML or upstream
├── ssz/                        # In-tree SSZ helpers
├── crypto/                     # BLS signatures (blspy), hashing
├── p2p/                        # Thin Python shim over the Rust libp2p stack
│   ├── host.py                 # Wraps consensoor_p2p.Network (Rust)
│   ├── gossip.py               # Beacon gossip topics + handlers
│   └── encoding.py             # Snappy-framed req/resp encoding
├── engine/                     # Engine API client (JSON-RPC + SSZ-REST)
│   ├── client.py               # Versioned methods, capability negotiation
│   ├── ssz_types.py            # SSZ containers from execution-apis #764
│   └── types.py                # Dataclasses returned to callers
├── store/                      # LevelDB-backed persistent state/block store
├── metrics/                    # Prometheus metrics
├── validator/                  # Validator duties, keystores, shuffling
├── builder/                    # Block building (incl. ePBS payload bids)
├── beacon_api/                 # Beacon API HTTP server
│   ├── server.py               # HTTP routes
│   ├── spec.py                 # /eth/v1/config/spec builder
│   └── utils.py
├── beacon_sync/                # SSE+REST sync from an upstream beacon node
├── attestation_pool.py         # Unaggregated/aggregate attestation pool
├── sync_committee_pool.py      # Sync committee messages + contributions
├── payload_attestation_pool.py # PTC (Gloas) payload-attestation pool
├── version.py                  # Version info and graffiti builder
├── node.py                     # Main orchestration
├── config.py                   # Node configuration
└── cli.py                      # CLI entry point

consensoor-p2p-rs/              # Rust crate (pyo3/maturin) — see its README
└── src/                        # libp2p host, gossipsub, req/resp wired like Lighthouse

tests/
├── spec/                       # Consensus spec tests
│   ├── conftest.py             # Pytest fixtures
│   └── test_spec_runner.py     # Spec + fork-choice compliance runner
└── spec-tests/                 # Downloaded spec tests (gitignored)
```

## P2P Networking

P2P runs on a native Rust libp2p host (`consensoor-p2p-rs`) exposed to Python via pyo3. The transport stack is configured the same way Lighthouse configures `lighthouse_network`: TCP + Noise + Yamux (with Mplex as required by `consensus-specs/specs/phase0/p2p-interface.md`), gossipsub with the Eth2 message-id rules, identify, ping, and request/response.

**Why a Rust binding instead of pure Python?** py-libp2p was the original implementation, but on a live devnet it could not keep up: gossipsub mesh management (GRAFT/PRUNE bookkeeping, message-cache scans) stalled under the GIL, peers timed out the Status/Metadata exchanges and the mesh fell apart within a couple of slots. The Rust host runs on a tokio runtime entirely outside the GIL and only crosses into Python to hand decoded messages up to the node; the slow path is gone.

- ENR includes the `eth2` field for network identification, with dynamic fork digest calculation (blob parameters folded in for Fulu+)
- Bootnode discovery via ENR
- Subscribes to beacon block and aggregate attestation topics, plus payload-attestation topics on Gloas
- Snappy framing format on req/resp; gossipsub uses snappy-block compression
- Req/resp protocol support (Status, Ping, Metadata v2, BeaconBlocksByRange/Root, BlobSidecars, DataColumnSidecars)
- Cached `StatusMessage` is mirrored into the Rust binding so the host can answer inbound `/eth2/beacon_chain/req/status/1/` without round-tripping into Python

## State Synchronization

Consensoor has two paths into a running network:

**Checkpoint sync (initial state)** — when `--checkpoint-sync-url` is provided, the node fetches a recent state SSZ from an upstream beacon node, subscribes to its SSE event stream, and re-syncs state at epoch boundaries (`randao_mixes`, validators, checkpoints). This is what gets the node to a usable head before peering up.

**Gossipsub block sync (steady state)** — once peered, the node receives `beacon_block` messages over the Rust libp2p host, verifies and applies them through the state-transition functions, and updates the EL via `engine_newPayload` + `engine_forkchoiceUpdated`. Per project policy this is the default sync path; full beacon-API checkpoint-sync replay is intentionally not used in steady state.

## Engine API

Two transports, negotiated automatically on startup via JSON-RPC `engine_exchangeCapabilities`:

- **Binary SSZ over REST** (default when available, per execution-apis [#764](https://github.com/ethereum/execution-apis/pull/764)) — requests/responses are raw SSZ over `application/octet-stream` on the existing 8551 port. The CL advertises strings like `"POST /engine/v4/payloads"` alongside its JSON-RPC method names; for any endpoint the EL also advertises in that form, the CL switches to binary. Eliminates hex-encoding, JSON parsing and the SSZ↔JSON round-trip that the CL already pays internally.
- **JSON-RPC** — used at all times for `engine_exchangeCapabilities` itself, and as the fallback for any endpoint where SSZ was not negotiated.

The implementation covers `engine_newPayloadV{1-5}`, `engine_getPayloadV{1-6}`, `engine_forkchoiceUpdatedV{1-4}`, `engine_getBlobsV{1-3}`, `engine_getClientVersionV1` and `engine_exchangeCapabilities`. Nullable JSON fields are encoded as `List[T, 1]` in SSZ per the spec.

## Spec Tests

Run consensus spec tests against consensoor:

```bash
make test                              # Run all tests (minimal preset)
make test preset=mainnet               # Run all tests (mainnet preset)
make test fork=electra                 # Run electra tests (minimal)
make test fork=electra preset=mainnet  # Run electra tests (mainnet)
```

Tests download from consensus-specs releases and cache locally. Specify with `SPEC_VERSION`:

```bash
make test SPEC_VERSION=nightly-2026-03-19           # last successful run for that date
make test SPEC_VERSION=nightly-23467328019          # pin to a specific run
make test SPEC_VERSION=v1.7.0-alpha.2               # release tag
```

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
- `GET /eth/v1/beacon/execution_payload_envelope/{block_id}` (Gloas)
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

- Simplified, not-fully-spec-compliant fork choice
- State transition is implemented for all forks but has known divergence cases (see `CLAUDE.md` for the `debug_state_diff.py` workflow)
- No slashing protection
- Falls back to py_ecc (slow) if blspy is unavailable
- Validator/builder logic is intentionally minimal — enough to propose and attest on a devnet, not enough to operate on mainnet

## License

MIT
