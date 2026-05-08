//! Tokio-driven libp2p host that consensoor drives from Python.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;

use async_channel::{bounded, Receiver, Sender};
use enr::{CombinedKey, Enr};
use futures::StreamExt;
use libp2p::{
    core::{upgrade, ConnectedPoint},
    gossipsub::{self, IdentTopic, ValidationMode},
    identify, identity, noise, ping,
    request_response,
    swarm::{NetworkBehaviour, SwarmEvent},
    tcp, yamux, Multiaddr, Swarm, Transport,
};
use parking_lot::Mutex;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use tokio::runtime::Runtime;

use crate::blocks_by_range::{
    self, BlocksByRangeBehaviour, BlocksByRangeEvent, BlocksByRangeRequest, BlocksByRangeResponse,
};
use crate::bootnode;
use crate::gossip;
use crate::rpc::{
    self, GoodbyeEvent, GoodbyeMessage, MetaDataMessage, MetaDataRequest, MetadataEvent,
    PingEvent, PingMessage, StatusEvent, StatusMessage,
};

const ETH2_AGENT_VERSION: &str = "consensoor/0.1.0";

#[derive(NetworkBehaviour)]
struct Eth2Behaviour {
    gossipsub: gossipsub::Behaviour,
    identify: identify::Behaviour,
    ping_libp2p: ping::Behaviour,
    status: rpc::StatusBehaviour,
    eth2_ping: rpc::PingBehaviour,
    goodbye: rpc::GoodbyeBehaviour,
    metadata: rpc::MetadataBehaviour,
    blocks_by_range: BlocksByRangeBehaviour,
}

#[pyclass]
#[derive(Clone)]
pub struct NetworkConfig {
    #[pyo3(get, set)]
    pub listen_addr: String,
    #[pyo3(get, set)]
    pub external_addr: Option<String>,
    #[pyo3(get, set)]
    pub bootnodes: Vec<String>,
    #[pyo3(get, set)]
    pub fork_digest: Vec<u8>,
    #[pyo3(get, set)]
    pub seed_phrase: Option<String>,
    #[pyo3(get, set)]
    pub max_peers: usize,
    #[pyo3(get, set)]
    pub agent_version: String,
    /// SSZ-encoded `next_fork_version` (4 bytes) for the `eth2` ENR field.
    #[pyo3(get, set)]
    pub next_fork_version: Vec<u8>,
    /// `next_fork_epoch` for the `eth2` ENR field.
    #[pyo3(get, set)]
    pub next_fork_epoch: u64,
    /// SSZ-encoded `attnets` Bitvector[ATTESTATION_SUBNET_COUNT].
    #[pyo3(get, set)]
    pub attnets: Vec<u8>,
    /// SSZ-encoded `syncnets` Bitvector[SYNC_COMMITTEE_SUBNET_COUNT].
    #[pyo3(get, set)]
    pub syncnets: Vec<u8>,
    /// PeerDAS `custody_group_count` (`cgc` ENR field).
    #[pyo3(get, set)]
    pub cgc: u64,
    /// External IPv4 to advertise in the ENR. None = omit ip field.
    #[pyo3(get, set)]
    pub external_ip: Option<String>,
}

#[pymethods]
impl NetworkConfig {
    #[new]
    #[pyo3(signature = (
        listen_addr="/ip4/0.0.0.0/tcp/9000".to_string(),
        external_addr=None,
        bootnodes=Vec::new(),
        fork_digest=Vec::new(),
        seed_phrase=None,
        max_peers=64,
        agent_version=ETH2_AGENT_VERSION.to_string(),
        next_fork_version=vec![0u8; 4],
        next_fork_epoch=u64::MAX,
        attnets=vec![0u8; 8],
        syncnets=vec![0u8; 1],
        cgc=4,
        external_ip=None,
    ))]
    pub fn new(
        listen_addr: String,
        external_addr: Option<String>,
        bootnodes: Vec<String>,
        fork_digest: Vec<u8>,
        seed_phrase: Option<String>,
        max_peers: usize,
        agent_version: String,
        next_fork_version: Vec<u8>,
        next_fork_epoch: u64,
        attnets: Vec<u8>,
        syncnets: Vec<u8>,
        cgc: u64,
        external_ip: Option<String>,
    ) -> Self {
        Self {
            listen_addr,
            external_addr,
            bootnodes,
            fork_digest,
            seed_phrase,
            max_peers,
            agent_version,
            next_fork_version,
            next_fork_epoch,
            attnets,
            syncnets,
            cgc,
            external_ip,
        }
    }
}

/// Extract the `/tcp/<port>` component from a libp2p Multiaddr.
fn parse_tcp_port(addr: &Multiaddr) -> Option<u16> {
    use libp2p::multiaddr::Protocol;
    addr.iter().find_map(|p| match p {
        Protocol::Tcp(port) => Some(port),
        _ => None,
    })
}

/// Build (or rebuild) our local Eth2 ENR using the libp2p secp256k1 key.
///
/// Eth2 custom fields follow Lighthouse's encoding (see
/// `lighthouse_network/src/discovery/enr.rs`):
///   - `eth2`  : RLP byte string of SSZ(`ENRForkID { fork_digest,
///               next_fork_version, next_fork_epoch }`).
///   - `attnets` / `syncnets`: RLP byte string of the SSZ-serialised
///     Bitvector (caller passes the already-SSZ'd bytes).
///   - `cgc`   : RLP-encoded u64 (PeerDAS / EIP-7594).
fn build_local_enr(
    enr_key: &CombinedKey,
    cfg: &NetworkConfig,
    tcp_port: u16,
    seq: u64,
) -> Result<Enr<CombinedKey>, Box<dyn std::error::Error>> {
    use bytes::Bytes;
    let mut builder = Enr::builder();
    builder.seq(seq);
    builder.tcp4(tcp_port);
    builder.udp4(tcp_port);
    if let Some(ip_str) = &cfg.external_ip {
        if let Ok(ip) = ip_str.parse::<IpAddr>() {
            builder.ip(ip);
        }
    }
    // eth2 = SSZ(ENRForkID): fork_digest (4) || next_fork_version (4) || next_fork_epoch (u64 LE) = 16 bytes.
    let mut eth2_bytes = Vec::with_capacity(16);
    eth2_bytes.extend_from_slice(if cfg.fork_digest.len() >= 4 {
        &cfg.fork_digest[..4]
    } else {
        &[0u8; 4]
    });
    eth2_bytes.extend_from_slice(if cfg.next_fork_version.len() >= 4 {
        &cfg.next_fork_version[..4]
    } else {
        &[0u8; 4]
    });
    eth2_bytes.extend_from_slice(&cfg.next_fork_epoch.to_le_bytes());
    builder.add_value::<Bytes>("eth2", &Bytes::from(eth2_bytes));
    builder.add_value::<Bytes>("attnets", &Bytes::from(cfg.attnets.clone()));
    builder.add_value::<Bytes>("syncnets", &Bytes::from(cfg.syncnets.clone()));
    builder.add_value("cgc", &cfg.cgc);
    Ok(builder.build(enr_key)?)
}

/// Decompressed gossipsub message (Python sees the original SSZ payload).
#[pyclass]
pub struct GossipMessage {
    #[pyo3(get)]
    pub topic: String,
    #[pyo3(get)]
    pub data: Vec<u8>,
    #[pyo3(get)]
    pub from_peer: String,
    #[pyo3(get)]
    pub message_id: Vec<u8>,
}

#[pyfunction]
pub fn generate_keypair() -> PyResult<(Vec<u8>, String)> {
    let key = identity::Keypair::generate_secp256k1();
    let peer_id = key.public().to_peer_id();
    let bytes = key
        .to_protobuf_encoding()
        .map_err(|e| PyRuntimeError::new_err(format!("encoding key: {e}")))?;
    Ok((bytes, peer_id.to_string()))
}

enum Command {
    Subscribe(String),
    Publish { topic: String, data: Vec<u8> },
    Dial(Multiaddr),
    SendStatus { peer: String, status: StatusMessage },
    AnswerStatus { id: u64, status: StatusMessage },
    SendPing { peer: String, ping: PingMessage },
    AnswerPing { id: u64, pong: PingMessage },
    SendGoodbye { peer: String, goodbye: GoodbyeMessage },
    AnswerGoodbye { id: u64, goodbye: GoodbyeMessage },
    RequestMetadata { peer: String },
    AnswerMetadata { id: u64, metadata: MetaDataMessage },
    RequestBlocksByRange { peer: String, request: BlocksByRangeRequest },
    AnswerBlocksByRange { id: u64, response: BlocksByRangeResponse },
    Shutdown,
}

#[pyclass]
pub struct Network {
    runtime: Arc<Runtime>,
    cmd_tx: Sender<Command>,
    msg_rx: Mutex<Receiver<GossipMessage>>,
    status_rx: Mutex<Receiver<StatusEvent>>,
    ping_rx: Mutex<Receiver<PingEvent>>,
    goodbye_rx: Mutex<Receiver<GoodbyeEvent>>,
    metadata_rx: Mutex<Receiver<MetadataEvent>>,
    by_range_rx: Mutex<Receiver<BlocksByRangeEvent>>,
    local_peer_id: String,
    listen_addrs: Arc<Mutex<Vec<String>>>,
    connected_peers: Arc<Mutex<HashMap<String, &'static str>>>,
    /// secp256k1 key for ENR signing (cloned from the libp2p keypair).
    enr_key: Arc<CombinedKey>,
    /// The latest signed local ENR. Replaced on update_enr_*.
    local_enr: Arc<Mutex<Enr<CombinedKey>>>,
    /// ENR seq_number — bumped on every rebuild so peers re-fetch.
    enr_seq: Arc<Mutex<u64>>,
    /// Cached for ENR rebuilds.
    cfg: Arc<Mutex<NetworkConfig>>,
    tcp_port: u16,
}

#[pymethods]
impl Network {
    #[staticmethod]
    pub fn start(config: NetworkConfig) -> PyResult<Self> {
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .thread_name("consensoor-p2p")
                .build()
                .map_err(|e| PyRuntimeError::new_err(format!("tokio: {e}")))?,
        );

        let key = identity::Keypair::generate_secp256k1();
        let local_peer_id = key.public().to_peer_id();
        let local_peer_id_str = local_peer_id.to_string();

        // Bridge libp2p secp256k1 secret bytes into an enr::CombinedKey so
        // we can sign the local ENR with the same identity that drives the
        // libp2p PeerId. libp2p's SecretKey is a wrapper around
        // k256::ecdsa::SigningKey, and `secp256k1_from_bytes` accepts the
        // 32-byte raw form.
        let secp_kp = key
            .clone()
            .try_into_secp256k1()
            .map_err(|e| PyRuntimeError::new_err(format!("secp256k1 keypair: {e}")))?;
        let mut secret_bytes = secp_kp.secret().to_bytes();
        let enr_key = CombinedKey::secp256k1_from_bytes(&mut secret_bytes)
            .map_err(|e| PyRuntimeError::new_err(format!("enr secp256k1: {e}")))?;
        // Second copy for discv5 — `secp256k1_from_bytes` zeroizes its input,
        // and `CombinedKey` itself isn't Clone.
        let mut secret_bytes_for_discv5 = secp_kp.secret().to_bytes();
        let discv5_enr_key = CombinedKey::secp256k1_from_bytes(&mut secret_bytes_for_discv5)
            .map_err(|e| PyRuntimeError::new_err(format!("discv5 secp256k1: {e}")))?;

        let listen_addr: Multiaddr = config
            .listen_addr
            .parse()
            .map_err(|e| PyValueError::new_err(format!("listen_addr parse: {e}")))?;

        // Pull out the tcp port for the ENR (Multiaddr like /ip4/.../tcp/9000).
        let tcp_port = parse_tcp_port(&listen_addr).unwrap_or(9000);

        let initial_enr = build_local_enr(&enr_key, &config, tcp_port, 1)
            .map_err(|e| PyRuntimeError::new_err(format!("build local ENR: {e}")))?;
        tracing::info!("local ENR: {}", initial_enr.to_base64());

        let dial_targets: Vec<Multiaddr> = config
            .bootnodes
            .iter()
            .map(|b| {
                bootnode::parse_dial_target(b)
                    .map_err(|e| PyValueError::new_err(format!("bootnode {b}: {e}")))
            })
            .collect::<Result<_, _>>()?;

        // Parse bootnodes that look like ENRs into actual Enr<CombinedKey>
        // values for the discv5 routing table. Multiaddr-only bootnodes are
        // skipped (discv5 needs an ENR to walk the network from).
        let bootnode_enrs: Vec<Enr<CombinedKey>> = config
            .bootnodes
            .iter()
            .filter(|s| s.starts_with("enr:"))
            .filter_map(|s| s.parse::<Enr<CombinedKey>>().ok())
            .collect();

        let agent_version = config.agent_version.clone();

        let (cmd_tx, cmd_rx) = bounded::<Command>(64);
        let (msg_tx, msg_rx) = bounded::<GossipMessage>(2048);
        let (status_tx, status_rx) = bounded::<StatusEvent>(256);
        let (ping_tx, ping_rx) = bounded::<PingEvent>(256);
        let (goodbye_tx, goodbye_rx) = bounded::<GoodbyeEvent>(256);
        let (metadata_tx, metadata_rx) = bounded::<MetadataEvent>(256);
        let (by_range_tx, by_range_rx) = bounded::<BlocksByRangeEvent>(64);
        let listen_addrs = Arc::new(Mutex::new(Vec::<String>::new()));
        let listen_addrs_clone = listen_addrs.clone();
        let connected_peers = Arc::new(Mutex::new(HashMap::<String, &'static str>::new()));
        let connected_peers_clone = connected_peers.clone();

        let runtime_clone = runtime.clone();
        let discv5_local_enr = initial_enr.clone();
        runtime_clone.spawn(async move {
            if let Err(e) = run_swarm(
                key,
                listen_addr,
                dial_targets,
                agent_version,
                cmd_rx,
                msg_tx,
                status_tx,
                ping_tx,
                goodbye_tx,
                metadata_tx,
                by_range_tx,
                listen_addrs_clone,
                connected_peers_clone,
                discv5_local_enr,
                discv5_enr_key,
                bootnode_enrs,
                tcp_port,
            )
            .await
            {
                tracing::error!("p2p swarm exited: {e}");
            }
        });

        Ok(Self {
            runtime,
            cmd_tx,
            msg_rx: Mutex::new(msg_rx),
            status_rx: Mutex::new(status_rx),
            ping_rx: Mutex::new(ping_rx),
            goodbye_rx: Mutex::new(goodbye_rx),
            metadata_rx: Mutex::new(metadata_rx),
            by_range_rx: Mutex::new(by_range_rx),
            local_peer_id: local_peer_id_str,
            listen_addrs,
            connected_peers,
            enr_key: Arc::new(enr_key),
            local_enr: Arc::new(Mutex::new(initial_enr)),
            enr_seq: Arc::new(Mutex::new(1)),
            cfg: Arc::new(Mutex::new(config)),
            tcp_port,
        })
    }

    /// Return the current local ENR as a base64 `enr:` string.
    pub fn enr(&self) -> String {
        self.local_enr.lock().to_base64()
    }

    /// Rebuild + re-sign the local ENR with new Eth2 fields, bumping seq.
    /// Mirrors the dynamic adjustments described in
    /// `consensus-specs/specs/fulu/validator.md` (cgc bumps, fork digest
    /// rotations).
    #[pyo3(signature = (fork_digest=None, next_fork_version=None, next_fork_epoch=None, attnets=None, syncnets=None, cgc=None, external_ip=None))]
    pub fn update_enr(
        &self,
        fork_digest: Option<Vec<u8>>,
        next_fork_version: Option<Vec<u8>>,
        next_fork_epoch: Option<u64>,
        attnets: Option<Vec<u8>>,
        syncnets: Option<Vec<u8>>,
        cgc: Option<u64>,
        external_ip: Option<String>,
    ) -> PyResult<String> {
        let mut cfg = self.cfg.lock();
        if let Some(v) = fork_digest {
            cfg.fork_digest = v;
        }
        if let Some(v) = next_fork_version {
            cfg.next_fork_version = v;
        }
        if let Some(v) = next_fork_epoch {
            cfg.next_fork_epoch = v;
        }
        if let Some(v) = attnets {
            cfg.attnets = v;
        }
        if let Some(v) = syncnets {
            cfg.syncnets = v;
        }
        if let Some(v) = cgc {
            cfg.cgc = v;
        }
        if let Some(v) = external_ip {
            cfg.external_ip = Some(v);
        }
        let mut seq = self.enr_seq.lock();
        *seq += 1;
        let new_enr = build_local_enr(&self.enr_key, &cfg, self.tcp_port, *seq)
            .map_err(|e| PyRuntimeError::new_err(format!("rebuild ENR: {e}")))?;
        let b64 = new_enr.to_base64();
        *self.local_enr.lock() = new_enr;
        Ok(b64)
    }

    /// Snapshot of currently-connected peer IDs (as base58 strings).
    pub fn connected_peers(&self) -> Vec<String> {
        self.connected_peers.lock().keys().cloned().collect()
    }

    /// Snapshot of currently-connected peers as `(peer_id, direction)` pairs,
    /// where direction is `"inbound"` or `"outbound"` per the libp2p
    /// `ConnectedPoint` for the established connection. Surfaced via the
    /// beacon API `/eth/v1/node/peers` endpoint.
    pub fn connected_peers_with_direction(&self) -> Vec<(String, String)> {
        self.connected_peers
            .lock()
            .iter()
            .map(|(peer_id, direction)| (peer_id.clone(), (*direction).to_string()))
            .collect()
    }

    pub fn peer_id(&self) -> String {
        self.local_peer_id.clone()
    }

    pub fn listen_addresses(&self) -> Vec<String> {
        self.listen_addrs.lock().clone()
    }

    pub fn subscribe(&self, topic: String) -> PyResult<()> {
        self.cmd_tx
            .send_blocking(Command::Subscribe(topic))
            .map_err(|e| PyRuntimeError::new_err(format!("subscribe: {e}")))
    }

    /// Publish raw uncompressed SSZ payload. The Rust side handles snappy
    /// compression and Eth2 message-id calculation.
    pub fn publish(&self, topic: String, data: Vec<u8>) -> PyResult<()> {
        self.cmd_tx
            .send_blocking(Command::Publish { topic, data })
            .map_err(|e| PyRuntimeError::new_err(format!("publish: {e}")))
    }

    pub fn dial(&self, addr: String) -> PyResult<()> {
        let addr = bootnode::parse_dial_target(&addr)
            .map_err(|e| PyValueError::new_err(format!("dial addr: {e}")))?;
        self.cmd_tx
            .send_blocking(Command::Dial(addr))
            .map_err(|e| PyRuntimeError::new_err(format!("dial: {e}")))
    }

    pub fn send_status(&self, peer: String, status: StatusMessage) -> PyResult<()> {
        self.cmd_tx
            .send_blocking(Command::SendStatus { peer, status })
            .map_err(|e| PyRuntimeError::new_err(format!("send_status: {e}")))
    }

    pub fn answer_status(&self, request_id: u64, status: StatusMessage) -> PyResult<()> {
        self.cmd_tx
            .send_blocking(Command::AnswerStatus {
                id: request_id,
                status,
            })
            .map_err(|e| PyRuntimeError::new_err(format!("answer_status: {e}")))
    }

    pub fn send_ping(&self, peer: String, ping: PingMessage) -> PyResult<()> {
        self.cmd_tx
            .send_blocking(Command::SendPing { peer, ping })
            .map_err(|e| PyRuntimeError::new_err(format!("send_ping: {e}")))
    }

    pub fn answer_ping(&self, request_id: u64, pong: PingMessage) -> PyResult<()> {
        self.cmd_tx
            .send_blocking(Command::AnswerPing { id: request_id, pong })
            .map_err(|e| PyRuntimeError::new_err(format!("answer_ping: {e}")))
    }

    pub fn send_goodbye(&self, peer: String, goodbye: GoodbyeMessage) -> PyResult<()> {
        self.cmd_tx
            .send_blocking(Command::SendGoodbye { peer, goodbye })
            .map_err(|e| PyRuntimeError::new_err(format!("send_goodbye: {e}")))
    }

    pub fn answer_goodbye(&self, request_id: u64, goodbye: GoodbyeMessage) -> PyResult<()> {
        self.cmd_tx
            .send_blocking(Command::AnswerGoodbye { id: request_id, goodbye })
            .map_err(|e| PyRuntimeError::new_err(format!("answer_goodbye: {e}")))
    }

    pub fn request_metadata(&self, peer: String) -> PyResult<()> {
        self.cmd_tx
            .send_blocking(Command::RequestMetadata { peer })
            .map_err(|e| PyRuntimeError::new_err(format!("request_metadata: {e}")))
    }

    pub fn answer_metadata(&self, request_id: u64, metadata: MetaDataMessage) -> PyResult<()> {
        self.cmd_tx
            .send_blocking(Command::AnswerMetadata {
                id: request_id,
                metadata,
            })
            .map_err(|e| PyRuntimeError::new_err(format!("answer_metadata: {e}")))
    }

    pub fn request_blocks_by_range(
        &self,
        peer: String,
        start_slot: u64,
        count: u64,
    ) -> PyResult<()> {
        self.cmd_tx
            .send_blocking(Command::RequestBlocksByRange {
                peer,
                request: BlocksByRangeRequest::new(start_slot, count),
            })
            .map_err(|e| PyRuntimeError::new_err(format!("request_blocks_by_range: {e}")))
    }

    pub fn answer_blocks_by_range(
        &self,
        request_id: u64,
        response: BlocksByRangeResponse,
    ) -> PyResult<()> {
        self.cmd_tx
            .send_blocking(Command::AnswerBlocksByRange {
                id: request_id,
                response,
            })
            .map_err(|e| PyRuntimeError::new_err(format!("answer_blocks_by_range: {e}")))
    }

    #[pyo3(signature = (timeout_ms=None))]
    pub fn next_blocks_by_range<'py>(
        &self,
        py: Python<'py>,
        timeout_ms: Option<u64>,
    ) -> PyResult<Option<Py<BlocksByRangeEvent>>> {
        let rx = self.by_range_rx.lock().clone();
        let runtime = self.runtime.clone();
        let result = py.allow_threads(|| match timeout_ms {
            Some(ms) => runtime.block_on(async move {
                tokio::time::timeout(Duration::from_millis(ms), rx.recv())
                    .await
                    .ok()
                    .and_then(|r| r.ok())
            }),
            None => runtime.block_on(async move { rx.recv().await.ok() }),
        });
        match result {
            Some(ev) => Ok(Some(Py::new(py, ev)?)),
            None => Ok(None),
        }
    }

    /// Block until a gossipsub message arrives or `timeout_ms` elapses.
    #[pyo3(signature = (timeout_ms=None))]
    pub fn next_message<'py>(
        &self,
        py: Python<'py>,
        timeout_ms: Option<u64>,
    ) -> PyResult<Option<Py<GossipMessage>>> {
        let rx = self.msg_rx.lock().clone();
        let runtime = self.runtime.clone();
        // Release the GIL while we sit on the rust channel — otherwise a
        // 1-second poll across four daemon threads serialises every other
        // python thread (incl. the asyncio loop and beacon API thread) for
        // up to 4 seconds at a time.
        let result = py.allow_threads(|| match timeout_ms {
            Some(ms) => runtime.block_on(async move {
                tokio::time::timeout(Duration::from_millis(ms), rx.recv())
                    .await
                    .ok()
                    .and_then(|r| r.ok())
            }),
            None => runtime.block_on(async move { rx.recv().await.ok() }),
        });
        match result {
            Some(msg) => Ok(Some(Py::new(py, msg)?)),
            None => Ok(None),
        }
    }

    /// Block until a Status RPC event arrives.
    #[pyo3(signature = (timeout_ms=None))]
    pub fn next_status<'py>(
        &self,
        py: Python<'py>,
        timeout_ms: Option<u64>,
    ) -> PyResult<Option<Py<StatusEvent>>> {
        let rx = self.status_rx.lock().clone();
        let runtime = self.runtime.clone();
        let result = py.allow_threads(|| match timeout_ms {
            Some(ms) => runtime.block_on(async move {
                tokio::time::timeout(Duration::from_millis(ms), rx.recv())
                    .await
                    .ok()
                    .and_then(|r| r.ok())
            }),
            None => runtime.block_on(async move { rx.recv().await.ok() }),
        });
        match result {
            Some(ev) => Ok(Some(Py::new(py, ev)?)),
            None => Ok(None),
        }
    }

    #[pyo3(signature = (timeout_ms=None))]
    pub fn next_ping<'py>(
        &self,
        py: Python<'py>,
        timeout_ms: Option<u64>,
    ) -> PyResult<Option<Py<PingEvent>>> {
        let rx = self.ping_rx.lock().clone();
        let runtime = self.runtime.clone();
        let result = py.allow_threads(|| match timeout_ms {
            Some(ms) => runtime.block_on(async move {
                tokio::time::timeout(Duration::from_millis(ms), rx.recv())
                    .await
                    .ok()
                    .and_then(|r| r.ok())
            }),
            None => runtime.block_on(async move { rx.recv().await.ok() }),
        });
        match result {
            Some(ev) => Ok(Some(Py::new(py, ev)?)),
            None => Ok(None),
        }
    }

    #[pyo3(signature = (timeout_ms=None))]
    pub fn next_goodbye<'py>(
        &self,
        py: Python<'py>,
        timeout_ms: Option<u64>,
    ) -> PyResult<Option<Py<GoodbyeEvent>>> {
        let rx = self.goodbye_rx.lock().clone();
        let runtime = self.runtime.clone();
        let result = py.allow_threads(|| match timeout_ms {
            Some(ms) => runtime.block_on(async move {
                tokio::time::timeout(Duration::from_millis(ms), rx.recv())
                    .await
                    .ok()
                    .and_then(|r| r.ok())
            }),
            None => runtime.block_on(async move { rx.recv().await.ok() }),
        });
        match result {
            Some(ev) => Ok(Some(Py::new(py, ev)?)),
            None => Ok(None),
        }
    }

    #[pyo3(signature = (timeout_ms=None))]
    pub fn next_metadata<'py>(
        &self,
        py: Python<'py>,
        timeout_ms: Option<u64>,
    ) -> PyResult<Option<Py<MetadataEvent>>> {
        let rx = self.metadata_rx.lock().clone();
        let runtime = self.runtime.clone();
        let result = py.allow_threads(|| match timeout_ms {
            Some(ms) => runtime.block_on(async move {
                tokio::time::timeout(Duration::from_millis(ms), rx.recv())
                    .await
                    .ok()
                    .and_then(|r| r.ok())
            }),
            None => runtime.block_on(async move { rx.recv().await.ok() }),
        });
        match result {
            Some(ev) => Ok(Some(Py::new(py, ev)?)),
            None => Ok(None),
        }
    }

    pub fn shutdown(&self) -> PyResult<()> {
        let _ = self.cmd_tx.send_blocking(Command::Shutdown);
        Ok(())
    }
}

async fn run_swarm(
    key: identity::Keypair,
    listen_addr: Multiaddr,
    dial_targets: Vec<Multiaddr>,
    agent_version: String,
    cmd_rx: Receiver<Command>,
    msg_tx: Sender<GossipMessage>,
    status_tx: Sender<StatusEvent>,
    ping_tx: Sender<PingEvent>,
    goodbye_tx: Sender<GoodbyeEvent>,
    metadata_tx: Sender<MetadataEvent>,
    by_range_tx: Sender<BlocksByRangeEvent>,
    listen_addrs: Arc<Mutex<Vec<String>>>,
    connected_peers: Arc<Mutex<HashMap<String, &'static str>>>,
    discv5_local_enr: Enr<CombinedKey>,
    discv5_enr_key: CombinedKey,
    bootnode_enrs: Vec<Enr<CombinedKey>>,
    udp_port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let local_peer_id = key.public().to_peer_id();

    let transport = tcp::tokio::Transport::default()
        .upgrade(upgrade::Version::V1)
        .authenticate(noise::Config::new(&key)?)
        .multiplex(yamux::Config::default())
        .boxed();

    // Eth2 gossipsub configuration per consensus-specs phase0/p2p-interface.md
    // (D=8, D_low=6, D_high=12, D_lazy=6, heartbeat 700ms, mcache 6/3, history
    // 12/3, fanout_ttl 60s, seen_ttl 550 heartbeats ≈ 385s, GOSSIP_MAX_SIZE
    // 10 MiB). Without these, libp2p-gossipsub uses its own defaults
    // (D=6, mesh_outbound_min=4, etc.) and prysm/lighthouse/teku score-prune
    // us out of their mesh, so messages never propagate even though the TCP
    // connection and Status RPC handshake succeed.
    let gossipsub_config = gossipsub::ConfigBuilder::default()
        .heartbeat_interval(Duration::from_millis(700))
        .heartbeat_initial_delay(Duration::from_millis(500))
        .mesh_n(8)
        .mesh_n_low(6)
        .mesh_n_high(12)
        .mesh_outbound_min(2)
        .gossip_lazy(6)
        .history_length(12)
        .history_gossip(3)
        .fanout_ttl(Duration::from_secs(60))
        // 550 heartbeats × 700ms = 385s
        .duplicate_cache_time(Duration::from_secs(385))
        .validation_mode(ValidationMode::Anonymous)
        .max_transmit_size(10 * 1024 * 1024)
        .message_id_fn(gossip::message_id_eth2)
        .build()
        .map_err(|e| -> Box<dyn std::error::Error> { Box::from(e.to_string()) })?;

    let gossipsub = gossipsub::Behaviour::new(
        gossipsub::MessageAuthenticity::Anonymous,
        gossipsub_config,
    )
    .map_err(|e| -> Box<dyn std::error::Error> { Box::from(e.to_string()) })?;
    // NOTE: Eth2 peer-scoring is intentionally NOT enabled here yet. With
    // libp2p scoring on but per-topic params empty (we don't have proper
    // P1..P4 wired yet), our local mesh maintenance prunes peers that
    // haven't accrued positive topic score — i.e. every peer at startup.
    // Result: our publishes find zero mesh peers and go nowhere, while
    // we still receive their messages. Re-enable once per-topic params are
    // populated; see crate::peer_score for the threshold/global params.

    let identify = identify::Behaviour::new(
        identify::Config::new(format!("/eth2/beacon_chain/req/status/2"), key.public())
            .with_agent_version(agent_version),
    );

    let ping_libp2p = ping::Behaviour::new(ping::Config::new());

    let behaviour = Eth2Behaviour {
        gossipsub,
        identify,
        ping_libp2p,
        status: rpc::new_status_behaviour(),
        eth2_ping: rpc::new_ping_behaviour(),
        goodbye: rpc::new_goodbye_behaviour(),
        metadata: rpc::new_metadata_behaviour(),
        blocks_by_range: blocks_by_range::new_blocks_by_range_behaviour(),
    };

    let mut swarm = Swarm::new(
        transport,
        behaviour,
        local_peer_id,
        libp2p::swarm::Config::with_tokio_executor()
            .with_idle_connection_timeout(Duration::from_secs(60)),
    );

    swarm.listen_on(listen_addr)?;

    for addr in dial_targets {
        if let Err(e) = swarm.dial(addr) {
            tracing::warn!("initial dial failed: {e}");
        }
    }

    // Spawn discv5 alongside libp2p. As it discovers ENRs, it pushes the
    // libp2p multiaddr we'd dial onto `discovered_rx`, which we drain in
    // the main select! loop and feed to swarm.dial().
    let (discovered_tx, discovered_rx) = bounded::<Multiaddr>(64);
    if let Err(e) = crate::discovery::spawn_discovery(
        discv5_local_enr,
        discv5_enr_key,
        udp_port,
        bootnode_enrs,
        discovered_tx,
    )
    .await
    {
        tracing::warn!("discv5 failed to start: {e} (continuing without discovery)");
    }
    // Track which peers we've already dialed via discovery so we don't
    // re-issue dial commands every time the same ENR re-surfaces.
    let mut dialed_via_discovery: std::collections::HashSet<libp2p::PeerId> =
        std::collections::HashSet::new();

    let mut topic_subs: HashMap<String, IdentTopic> = HashMap::new();
    let mut pending_status: HashMap<u64, request_response::ResponseChannel<StatusMessage>> =
        HashMap::new();
    let mut pending_ping: HashMap<u64, request_response::ResponseChannel<PingMessage>> =
        HashMap::new();
    let mut pending_goodbye: HashMap<u64, request_response::ResponseChannel<GoodbyeMessage>> =
        HashMap::new();
    let mut pending_metadata: HashMap<u64, request_response::ResponseChannel<MetaDataMessage>> =
        HashMap::new();
    let mut pending_by_range: HashMap<
        u64,
        request_response::ResponseChannel<BlocksByRangeResponse>,
    > = HashMap::new();
    let mut next_response_id: u64 = 1;

    loop {
        tokio::select! {
            // Drain peers discovered by discv5 and dial them via libp2p.
            // Skip peers we already dialed (or that match our local id).
            disc = discovered_rx.recv() => match disc {
                Ok(addr) => {
                    use libp2p::multiaddr::Protocol;
                    let pid = addr.iter().find_map(|p| match p {
                        Protocol::P2p(id) => Some(id),
                        _ => None,
                    });
                    if let Some(pid) = pid {
                        if pid == local_peer_id {
                            continue;
                        }
                        if !dialed_via_discovery.insert(pid) {
                            continue;
                        }
                    }
                    if let Err(e) = swarm.dial(addr.clone()) {
                        tracing::debug!("discv5-dial failed for {addr}: {e}");
                    } else {
                        tracing::info!("discv5: dialing {addr}");
                    }
                }
                Err(_) => {
                    // discovery channel closed — keep the swarm running.
                }
            },
            event = swarm.select_next_some() => match event {
                SwarmEvent::NewListenAddr { address, .. } => {
                    tracing::info!("listening on {address}");
                    listen_addrs.lock().push(address.to_string());
                }
                SwarmEvent::ConnectionEstablished { peer_id, endpoint, .. } => {
                    let direction = match endpoint {
                        ConnectedPoint::Dialer { .. } => "outbound",
                        ConnectedPoint::Listener { .. } => "inbound",
                    };
                    tracing::info!("connection established with {peer_id} ({direction})");
                    connected_peers.lock().insert(peer_id.to_string(), direction);
                }
                SwarmEvent::ConnectionClosed { peer_id, num_established, .. } => {
                    if num_established == 0 {
                        tracing::info!("connection closed with {peer_id}");
                        connected_peers.lock().remove(&peer_id.to_string());
                    }
                }
                SwarmEvent::Behaviour(Eth2BehaviourEvent::Gossipsub(gossipsub::Event::Message {
                    propagation_source,
                    message_id,
                    message,
                })) => {
                    let topic = message.topic.to_string();
                    // Decompress before handing to Python so consensoor sees the
                    // raw SSZ-encoded payload exactly the way it would have via
                    // py-libp2p.
                    let data = match gossip::snappy_decompress(&message.data) {
                        Ok(d) => d,
                        Err(e) => {
                            tracing::warn!("dropping gossip msg from {propagation_source}: snappy decompress failed: {e}");
                            continue;
                        }
                    };
                    let _ = msg_tx
                        .send(GossipMessage {
                            topic,
                            data,
                            from_peer: propagation_source.to_string(),
                            message_id: message_id.0.into(),
                        })
                        .await;
                }
                SwarmEvent::Behaviour(Eth2BehaviourEvent::Status(rr_event)) => {
                    handle_symmetric_rpc_event(
                        rr_event,
                        &status_tx,
                        &mut pending_status,
                        &mut next_response_id,
                        |peer, kind, msg, err| StatusEvent { peer, kind, message: msg, error: err },
                    ).await;
                }
                SwarmEvent::Behaviour(Eth2BehaviourEvent::Eth2Ping(rr_event)) => {
                    handle_symmetric_rpc_event(
                        rr_event,
                        &ping_tx,
                        &mut pending_ping,
                        &mut next_response_id,
                        |peer, kind, msg, err| PingEvent { peer, kind, message: msg, error: err },
                    ).await;
                }
                SwarmEvent::Behaviour(Eth2BehaviourEvent::Goodbye(rr_event)) => {
                    handle_symmetric_rpc_event(
                        rr_event,
                        &goodbye_tx,
                        &mut pending_goodbye,
                        &mut next_response_id,
                        |peer, kind, msg, err| GoodbyeEvent { peer, kind, message: msg, error: err },
                    ).await;
                }
                SwarmEvent::Behaviour(Eth2BehaviourEvent::Metadata(rr_event)) => {
                    handle_metadata_event(
                        rr_event,
                        &metadata_tx,
                        &mut pending_metadata,
                        &mut next_response_id,
                    ).await;
                }
                SwarmEvent::Behaviour(Eth2BehaviourEvent::BlocksByRange(rr_event)) => {
                    handle_blocks_by_range_event(
                        rr_event,
                        &by_range_tx,
                        &mut pending_by_range,
                        &mut next_response_id,
                    ).await;
                }
                _ => {}
            },
            cmd = cmd_rx.recv() => match cmd {
                Ok(Command::Subscribe(topic)) => {
                    let t = IdentTopic::new(&topic);
                    if let Err(e) = swarm.behaviour_mut().gossipsub.subscribe(&t) {
                        tracing::warn!("subscribe failed: {e}");
                    }
                    topic_subs.insert(topic, t);
                }
                Ok(Command::Publish { topic, data }) => {
                    let t = topic_subs
                        .entry(topic.clone())
                        .or_insert_with(|| IdentTopic::new(&topic))
                        .clone();
                    let compressed = match gossip::snappy_compress(&data) {
                        Ok(c) => c,
                        Err(e) => {
                            tracing::warn!("publish snappy compress failed: {e}");
                            continue;
                        }
                    };
                    if let Err(e) = swarm.behaviour_mut().gossipsub.publish(t, compressed) {
                        tracing::warn!("publish failed: {e}");
                    }
                }
                Ok(Command::Dial(addr)) => {
                    if let Err(e) = swarm.dial(addr) {
                        tracing::warn!("dial failed: {e}");
                    }
                }
                Ok(Command::SendStatus { peer, status }) => {
                    match peer.parse::<libp2p::PeerId>() {
                        Ok(peer_id) => {
                            let _ = swarm.behaviour_mut().status.send_request(&peer_id, status);
                        }
                        Err(e) => tracing::warn!("send_status: bad peer id {peer}: {e}"),
                    }
                }
                Ok(Command::AnswerStatus { id, status }) => {
                    if let Some(channel) = pending_status.remove(&id) {
                        if swarm.behaviour_mut().status.send_response(channel, status).is_err() {
                            tracing::warn!("answer_status: channel dropped for id {id}");
                        }
                    } else {
                        tracing::warn!("answer_status: no pending status request id {id}");
                    }
                }
                Ok(Command::SendPing { peer, ping }) => {
                    match peer.parse::<libp2p::PeerId>() {
                        Ok(peer_id) => {
                            let _ = swarm.behaviour_mut().eth2_ping.send_request(&peer_id, ping);
                        }
                        Err(e) => tracing::warn!("send_ping: bad peer id {peer}: {e}"),
                    }
                }
                Ok(Command::AnswerPing { id, pong }) => {
                    if let Some(channel) = pending_ping.remove(&id) {
                        if swarm.behaviour_mut().eth2_ping.send_response(channel, pong).is_err() {
                            tracing::warn!("answer_ping: channel dropped for id {id}");
                        }
                    } else {
                        tracing::warn!("answer_ping: no pending ping request id {id}");
                    }
                }
                Ok(Command::SendGoodbye { peer, goodbye }) => {
                    match peer.parse::<libp2p::PeerId>() {
                        Ok(peer_id) => {
                            let _ = swarm.behaviour_mut().goodbye.send_request(&peer_id, goodbye);
                        }
                        Err(e) => tracing::warn!("send_goodbye: bad peer id {peer}: {e}"),
                    }
                }
                Ok(Command::AnswerGoodbye { id, goodbye }) => {
                    if let Some(channel) = pending_goodbye.remove(&id) {
                        if swarm.behaviour_mut().goodbye.send_response(channel, goodbye).is_err() {
                            tracing::warn!("answer_goodbye: channel dropped for id {id}");
                        }
                    } else {
                        tracing::warn!("answer_goodbye: no pending goodbye request id {id}");
                    }
                }
                Ok(Command::RequestMetadata { peer }) => {
                    match peer.parse::<libp2p::PeerId>() {
                        Ok(peer_id) => {
                            let _ = swarm.behaviour_mut().metadata.send_request(&peer_id, MetaDataRequest);
                        }
                        Err(e) => tracing::warn!("request_metadata: bad peer id {peer}: {e}"),
                    }
                }
                Ok(Command::AnswerMetadata { id, metadata }) => {
                    if let Some(channel) = pending_metadata.remove(&id) {
                        if swarm.behaviour_mut().metadata.send_response(channel, metadata).is_err() {
                            tracing::warn!("answer_metadata: channel dropped for id {id}");
                        }
                    } else {
                        tracing::warn!("answer_metadata: no pending metadata request id {id}");
                    }
                }
                Ok(Command::RequestBlocksByRange { peer, request }) => {
                    match peer.parse::<libp2p::PeerId>() {
                        Ok(peer_id) => {
                            let _ = swarm.behaviour_mut().blocks_by_range.send_request(&peer_id, request);
                        }
                        Err(e) => tracing::warn!("request_blocks_by_range: bad peer id {peer}: {e}"),
                    }
                }
                Ok(Command::AnswerBlocksByRange { id, response }) => {
                    if let Some(channel) = pending_by_range.remove(&id) {
                        if swarm.behaviour_mut().blocks_by_range.send_response(channel, response).is_err() {
                            tracing::warn!("answer_blocks_by_range: channel dropped for id {id}");
                        }
                    } else {
                        tracing::warn!("answer_blocks_by_range: no pending request id {id}");
                    }
                }
                Ok(Command::Shutdown) => break,
                Err(_) => break,
            }
        }
    }

    Ok(())
}

/// Handle a symmetric RPC event (Req == Resp). Used for status, ping, goodbye.
async fn handle_symmetric_rpc_event<M, EvT, Mk>(
    ev: request_response::Event<M, M>,
    tx: &Sender<EvT>,
    pending: &mut HashMap<u64, request_response::ResponseChannel<M>>,
    next_id: &mut u64,
    make_event: Mk,
) where
    M: Clone + std::fmt::Debug,
    EvT: Send,
    Mk: Fn(String, String, Option<M>, Option<String>) -> EvT,
{
    use request_response::Event;
    use request_response::Message;
    match ev {
        Event::Message { peer, message, .. } => match message {
            Message::Request {
                request, channel, ..
            } => {
                let id = *next_id;
                *next_id = next_id.wrapping_add(1);
                pending.insert(id, channel);
                let _ = tx
                    .send(make_event(
                        peer.to_string(),
                        format!("request:{id}"),
                        Some(request),
                        None,
                    ))
                    .await;
            }
            Message::Response { response, .. } => {
                let _ = tx
                    .send(make_event(
                        peer.to_string(),
                        "response".into(),
                        Some(response),
                        None,
                    ))
                    .await;
            }
        },
        Event::OutboundFailure { peer, error, .. } => {
            let _ = tx
                .send(make_event(
                    peer.to_string(),
                    "failure".into(),
                    None,
                    Some(format!("{error}")),
                ))
                .await;
        }
        Event::InboundFailure { peer, error, .. } => {
            let _ = tx
                .send(make_event(
                    peer.to_string(),
                    "failure".into(),
                    None,
                    Some(format!("{error}")),
                ))
                .await;
        }
        Event::ResponseSent { .. } => {}
    }
}

async fn handle_blocks_by_range_event(
    ev: request_response::Event<BlocksByRangeRequest, BlocksByRangeResponse>,
    tx: &Sender<BlocksByRangeEvent>,
    pending: &mut HashMap<u64, request_response::ResponseChannel<BlocksByRangeResponse>>,
    next_id: &mut u64,
) {
    use request_response::Event;
    use request_response::Message;
    match ev {
        Event::Message { peer, message, .. } => match message {
            Message::Request {
                request, channel, ..
            } => {
                let id = *next_id;
                *next_id = next_id.wrapping_add(1);
                pending.insert(id, channel);
                let _ = tx
                    .send(BlocksByRangeEvent {
                        peer: peer.to_string(),
                        kind: format!("request:{id}"),
                        request: Some(request),
                        response: None,
                        error: None,
                    })
                    .await;
            }
            Message::Response { response, .. } => {
                let _ = tx
                    .send(BlocksByRangeEvent {
                        peer: peer.to_string(),
                        kind: "response".into(),
                        request: None,
                        response: Some(response),
                        error: None,
                    })
                    .await;
            }
        },
        Event::OutboundFailure { peer, error, .. } => {
            let _ = tx
                .send(BlocksByRangeEvent {
                    peer: peer.to_string(),
                    kind: "failure".into(),
                    request: None,
                    response: None,
                    error: Some(format!("{error}")),
                })
                .await;
        }
        Event::InboundFailure { peer, error, .. } => {
            let _ = tx
                .send(BlocksByRangeEvent {
                    peer: peer.to_string(),
                    kind: "failure".into(),
                    request: None,
                    response: None,
                    error: Some(format!("{error}")),
                })
                .await;
        }
        Event::ResponseSent { .. } => {}
    }
}

async fn handle_metadata_event(
    ev: request_response::Event<MetaDataRequest, MetaDataMessage>,
    metadata_tx: &Sender<MetadataEvent>,
    pending: &mut HashMap<u64, request_response::ResponseChannel<MetaDataMessage>>,
    next_id: &mut u64,
) {
    use request_response::Event;
    use request_response::Message;
    match ev {
        Event::Message { peer, message, .. } => match message {
            Message::Request { channel, .. } => {
                let id = *next_id;
                *next_id = next_id.wrapping_add(1);
                pending.insert(id, channel);
                let _ = metadata_tx
                    .send(MetadataEvent {
                        peer: peer.to_string(),
                        kind: format!("request:{id}"),
                        message: None,
                        error: None,
                    })
                    .await;
            }
            Message::Response { response, .. } => {
                let _ = metadata_tx
                    .send(MetadataEvent {
                        peer: peer.to_string(),
                        kind: "response".into(),
                        message: Some(response),
                        error: None,
                    })
                    .await;
            }
        },
        Event::OutboundFailure { peer, error, .. } => {
            let _ = metadata_tx
                .send(MetadataEvent {
                    peer: peer.to_string(),
                    kind: "failure".into(),
                    message: None,
                    error: Some(format!("{error}")),
                })
                .await;
        }
        Event::InboundFailure { peer, error, .. } => {
            let _ = metadata_tx
                .send(MetadataEvent {
                    peer: peer.to_string(),
                    kind: "failure".into(),
                    message: None,
                    error: Some(format!("{error}")),
                })
                .await;
        }
        Event::ResponseSent { .. } => {}
    }
}
