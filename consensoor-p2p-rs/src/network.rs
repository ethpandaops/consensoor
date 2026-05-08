//! Tokio-driven libp2p host that consensoor drives from Python.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_channel::{bounded, Receiver, Sender};
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
}

#[pymethods]
impl NetworkConfig {
    #[new]
    #[pyo3(signature = (listen_addr="/ip4/0.0.0.0/tcp/9000".to_string(), external_addr=None, bootnodes=Vec::new(), fork_digest=Vec::new(), seed_phrase=None, max_peers=64, agent_version=ETH2_AGENT_VERSION.to_string()))]
    pub fn new(
        listen_addr: String,
        external_addr: Option<String>,
        bootnodes: Vec<String>,
        fork_digest: Vec<u8>,
        seed_phrase: Option<String>,
        max_peers: usize,
        agent_version: String,
    ) -> Self {
        Self {
            listen_addr,
            external_addr,
            bootnodes,
            fork_digest,
            seed_phrase,
            max_peers,
            agent_version,
        }
    }
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

        let listen_addr: Multiaddr = config
            .listen_addr
            .parse()
            .map_err(|e| PyValueError::new_err(format!("listen_addr parse: {e}")))?;

        let dial_targets: Vec<Multiaddr> = config
            .bootnodes
            .iter()
            .map(|b| {
                bootnode::parse_dial_target(b)
                    .map_err(|e| PyValueError::new_err(format!("bootnode {b}: {e}")))
            })
            .collect::<Result<_, _>>()?;

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
        })
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

    let mut gossipsub = gossipsub::Behaviour::new(
        gossipsub::MessageAuthenticity::Anonymous,
        gossipsub_config,
    )
    .map_err(|e| -> Box<dyn std::error::Error> { Box::from(e.to_string()) })?;

    // Eth2 peer scoring — required so mesh GRAFT negotiation with prysm /
    // lighthouse / teku actually completes. Without it our gossipsub treats
    // every peer neutrally; prysm GRAFTs us, scores us at default 0, and
    // since we never reciprocate scoring properly, prysm prunes us back out.
    if let Err(e) = gossipsub.with_peer_score(
        crate::peer_score::eth2_peer_score_params(),
        crate::peer_score::eth2_thresholds(),
    ) {
        tracing::warn!("with_peer_score failed: {e} (continuing without scoring)");
    }

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
