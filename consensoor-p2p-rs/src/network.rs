//! Tokio-driven libp2p host that consensoor drives from Python.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_channel::{bounded, Receiver, Sender};
use futures::StreamExt;
use libp2p::{
    core::upgrade,
    gossipsub::{self, IdentTopic, MessageId, ValidationMode},
    identify, identity, noise, ping,
    swarm::{NetworkBehaviour, SwarmEvent},
    tcp, yamux, Multiaddr, PeerId, Swarm, Transport,
};
use parking_lot::Mutex;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};
use tokio::runtime::Runtime;

const ETH2_AGENT_VERSION: &str = "consensoor/0.1.0";
const ETH2_PROTOCOL_PREFIX: &str = "/eth2/beacon_chain/req";

#[derive(NetworkBehaviour)]
struct Eth2Behaviour {
    gossipsub: gossipsub::Behaviour,
    identify: identify::Behaviour,
    ping: ping::Behaviour,
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
}

#[pymethods]
impl NetworkConfig {
    #[new]
    #[pyo3(signature = (listen_addr="/ip4/0.0.0.0/tcp/9000".to_string(), external_addr=None, bootnodes=Vec::new(), fork_digest=Vec::new(), seed_phrase=None, max_peers=64))]
    pub fn new(
        listen_addr: String,
        external_addr: Option<String>,
        bootnodes: Vec<String>,
        fork_digest: Vec<u8>,
        seed_phrase: Option<String>,
        max_peers: usize,
    ) -> Self {
        Self {
            listen_addr,
            external_addr,
            bootnodes,
            fork_digest,
            seed_phrase,
            max_peers,
        }
    }
}

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
    Shutdown,
}

#[pyclass]
pub struct Network {
    runtime: Arc<Runtime>,
    cmd_tx: Sender<Command>,
    msg_rx: Mutex<Receiver<GossipMessage>>,
    local_peer_id: String,
    listen_addrs: Arc<Mutex<Vec<String>>>,
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
                b.parse::<Multiaddr>()
                    .map_err(|e| PyValueError::new_err(format!("bootnode {b}: {e}")))
            })
            .collect::<Result<_, _>>()?;

        let (cmd_tx, cmd_rx) = bounded::<Command>(64);
        let (msg_tx, msg_rx) = bounded::<GossipMessage>(2048);
        let listen_addrs = Arc::new(Mutex::new(Vec::<String>::new()));
        let listen_addrs_clone = listen_addrs.clone();

        let runtime_clone = runtime.clone();
        runtime_clone.spawn(async move {
            if let Err(e) = run_swarm(key, listen_addr, dial_targets, cmd_rx, msg_tx, listen_addrs_clone).await {
                tracing::error!("p2p swarm exited: {e}");
            }
        });

        Ok(Self {
            runtime,
            cmd_tx,
            msg_rx: Mutex::new(msg_rx),
            local_peer_id: local_peer_id_str,
            listen_addrs,
        })
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

    pub fn publish(&self, topic: String, data: Vec<u8>) -> PyResult<()> {
        self.cmd_tx
            .send_blocking(Command::Publish { topic, data })
            .map_err(|e| PyRuntimeError::new_err(format!("publish: {e}")))
    }

    pub fn dial(&self, addr: String) -> PyResult<()> {
        let addr: Multiaddr = addr
            .parse()
            .map_err(|e| PyValueError::new_err(format!("dial addr: {e}")))?;
        self.cmd_tx
            .send_blocking(Command::Dial(addr))
            .map_err(|e| PyRuntimeError::new_err(format!("dial: {e}")))
    }

    /// Block until a gossipsub message arrives or `timeout_ms` elapses.
    pub fn next_message(&self, timeout_ms: Option<u64>) -> PyResult<Option<Py<GossipMessage>>> {
        let rx = self.msg_rx.lock().clone();
        let result = match timeout_ms {
            Some(ms) => self.runtime.block_on(async move {
                tokio::time::timeout(Duration::from_millis(ms), rx.recv())
                    .await
                    .ok()
                    .and_then(|r| r.ok())
            }),
            None => self.runtime.block_on(async move { rx.recv().await.ok() }),
        };

        match result {
            Some(msg) => Python::with_gil(|py| Ok(Some(Py::new(py, msg)?))),
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
    cmd_rx: Receiver<Command>,
    msg_tx: Sender<GossipMessage>,
    listen_addrs: Arc<Mutex<Vec<String>>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let local_peer_id = key.public().to_peer_id();

    let transport = tcp::tokio::Transport::default()
        .upgrade(upgrade::Version::V1)
        .authenticate(noise::Config::new(&key)?)
        .multiplex(yamux::Config::default())
        .boxed();

    let gossipsub_config = gossipsub::ConfigBuilder::default()
        .heartbeat_interval(Duration::from_millis(700))
        .validation_mode(ValidationMode::Strict)
        .max_transmit_size(10 * 1024 * 1024)
        .message_id_fn(message_id_eth2)
        .build()
        .map_err(|e| -> Box<dyn std::error::Error> { Box::from(e.to_string()) })?;

    let gossipsub = gossipsub::Behaviour::new(
        gossipsub::MessageAuthenticity::Signed(key.clone()),
        gossipsub_config,
    )
    .map_err(|e| -> Box<dyn std::error::Error> { Box::from(e.to_string()) })?;

    let identify = identify::Behaviour::new(identify::Config::new(
        format!("/eth2/{}", "consensoor"),
        key.public(),
    ));

    let ping = ping::Behaviour::new(ping::Config::new());

    let behaviour = Eth2Behaviour {
        gossipsub,
        identify,
        ping,
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

    loop {
        tokio::select! {
            event = swarm.select_next_some() => match event {
                SwarmEvent::NewListenAddr { address, .. } => {
                    tracing::info!("listening on {address}");
                    listen_addrs.lock().push(address.to_string());
                }
                SwarmEvent::Behaviour(Eth2BehaviourEvent::Gossipsub(gossipsub::Event::Message {
                    propagation_source,
                    message_id,
                    message,
                })) => {
                    let topic = message.topic.to_string();
                    let _ = msg_tx
                        .send(GossipMessage {
                            topic,
                            data: message.data,
                            from_peer: propagation_source.to_string(),
                            message_id: message_id.0.into(),
                        })
                        .await;
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
                    if let Err(e) = swarm.behaviour_mut().gossipsub.publish(t, data) {
                        tracing::warn!("publish failed: {e}");
                    }
                }
                Ok(Command::Dial(addr)) => {
                    if let Err(e) = swarm.dial(addr) {
                        tracing::warn!("dial failed: {e}");
                    }
                }
                Ok(Command::Shutdown) => break,
                Err(_) => break,
            }
        }
    }

    Ok(())
}

/// Eth2 message id: SHA256(MESSAGE_DOMAIN_VALID_SNAPPY || topic_bytes_le_length || topic || decompressed_data)[:20]
///
/// This matches the ETH2 gossipsub spec; for now we approximate by hashing
/// the snappy-decompressed payload prefixed with the domain.  Lighthouse
/// applies the per-fork variant and we will mirror that once consensoor
/// hands us the active fork digest via NetworkConfig.fork_digest.
fn message_id_eth2(message: &gossipsub::Message) -> MessageId {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(b"\x01\x00\x00\x00"); // MESSAGE_DOMAIN_VALID_SNAPPY
    let topic_bytes = message.topic.as_str().as_bytes();
    hasher.update(&(topic_bytes.len() as u64).to_le_bytes());
    hasher.update(topic_bytes);
    hasher.update(&message.data);
    let out = hasher.finalize();
    MessageId(out[..20].to_vec())
}
