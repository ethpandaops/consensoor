//! discv5-based peer discovery.
//!
//! Lighthouse-style. We run a discv5 service alongside libp2p:
//!
//!   - libp2p on /ip4/.../tcp/<port> handles gossipsub, identify, RPCs
//!   - discv5 on /ip4/.../udp/<port> handles peer discovery via ENRs
//!
//! On startup we feed every bootnode ENR into discv5's routing table, kick
//! the service, and then spin a loop that periodically issues `find_node`
//! against random target NodeIds. Each result is a list of ENRs we didn't
//! know about; we convert each to a libp2p Multiaddr (`/ip4/.../tcp/<port>/
//! p2p/<peer_id>`) and forward it to the swarm task to dial. New peer →
//! libp2p connection → identify → status RPC → mesh GRAFT, the same chain
//! that already works for the bootnode-supplied peer.
//!
//! Without this, consensoor only ever connects to whichever bootnode it
//! was handed at startup; the rest of the mesh is invisible.

use std::time::Duration;

use async_channel::Sender;
use discv5::{
    enr::{CombinedKey, NodeId},
    ConfigBuilder, Discv5, Enr, ListenConfig,
};
use libp2p::{identity, multiaddr::Protocol, Multiaddr, PeerId};

const FIND_NODE_INTERVAL: Duration = Duration::from_secs(10);

/// Spawn the discv5 service. Returns once the service has started; the
/// background discovery loop keeps running on the caller's tokio runtime.
///
/// `discovered_tx` is sent one `(PeerId, base64 ENR string, Multiaddr)`
/// tuple per newly-seen ENR after every `find_node` query — the swarm task
/// receives those, remembers the ENR keyed by peer id (so we can surface
/// it via `/eth/v1/node/peers`), and issues a libp2p `Dial` for each.
pub async fn spawn_discovery(
    local_enr: Enr,
    enr_key: CombinedKey,
    udp_port: u16,
    bootnodes: Vec<Enr>,
    discovered_tx: Sender<(PeerId, String, Multiaddr)>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let listen_config = ListenConfig::Ipv4 {
        ip: "0.0.0.0".parse()?,
        port: udp_port,
    };
    let config = ConfigBuilder::new(listen_config).build();

    let mut discv5 = Discv5::new(local_enr, enr_key, config).map_err(|e| -> Box<
        dyn std::error::Error + Send + Sync,
    > { Box::from(e.to_string()) })?;

    // Echo bootnode ENRs back so the swarm task can stash them keyed by
    // peer id. Without this, the very first peer we connect to has no
    // ENR surfaced via /eth/v1/node/peers — find_node only yields ENRs
    // we didn't already know about.
    for enr in &bootnodes {
        let id = enr.node_id();
        match discv5.add_enr(enr.clone()) {
            Ok(_) => tracing::info!("discv5: added bootnode {id}"),
            Err(e) => tracing::warn!("discv5: add_enr {id} failed: {e}"),
        }
        if let Some((peer_id, addr)) = enr_to_peer_id_and_multiaddr(enr) {
            let enr_b64 = enr.to_base64();
            if discovered_tx.send((peer_id, enr_b64, addr)).await.is_err() {
                tracing::debug!("discv5: discovered_tx closed during bootstrap");
                return Ok(());
            }
        }
    }

    discv5.start().await.map_err(|e| -> Box<
        dyn std::error::Error + Send + Sync,
    > { Box::from(e.to_string()) })?;
    tracing::info!("discv5: service started on udp/{udp_port}");

    tokio::spawn(async move {
        let mut interval = tokio::time::interval(FIND_NODE_INTERVAL);
        // First tick fires immediately; we want to skip that and let the
        // swarm settle for one beat.
        interval.tick().await;
        loop {
            interval.tick().await;
            let target = NodeId::random();
            match discv5.find_node(target).await {
                Ok(enrs) => {
                    if enrs.is_empty() {
                        continue;
                    }
                    tracing::debug!(
                        "discv5: find_node returned {} ENR(s)",
                        enrs.len()
                    );
                    for enr in enrs {
                        if let Some((peer_id, addr)) = enr_to_peer_id_and_multiaddr(&enr) {
                            let enr_b64 = enr.to_base64();
                            // Best-effort send; if the receiver is dropped
                            // (swarm shut down), break the loop.
                            if discovered_tx.send((peer_id, enr_b64, addr)).await.is_err() {
                                tracing::debug!("discv5: discovered_tx closed, stopping");
                                return;
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::debug!("discv5: find_node failed: {e:?}");
                }
            }
        }
    });

    Ok(())
}

/// Convert an Eth2 ENR to the (PeerId, libp2p multiaddr) pair we'd dial.
///
/// The ENR's `secp256k1` field gives us the libp2p PeerId; ip4/ip6 + tcp4/
/// tcp6 give us the transport. Returns None for ENRs without a usable
/// (ip, tcp) pair (e.g. UDP-only peers, or legacy ENRs without `secp256k1`).
pub fn enr_to_peer_id_and_multiaddr(enr: &Enr) -> Option<(PeerId, Multiaddr)> {
    let tcp_port = enr.tcp4().or_else(|| enr.tcp6())?;
    let ip = if let Some(v4) = enr.ip4() {
        Protocol::Ip4(v4)
    } else if let Some(v6) = enr.ip6() {
        Protocol::Ip6(v6)
    } else {
        return None;
    };

    // Encode the secp256k1 ENR pubkey as a libp2p PeerId.
    use enr::EnrPublicKey;
    let pk_bytes = enr.public_key().encode();
    let peer_id = if pk_bytes.len() == 33 {
        let pk = identity::secp256k1::PublicKey::try_from_bytes(&pk_bytes).ok()?;
        PeerId::from_public_key(&identity::PublicKey::from(pk))
    } else if pk_bytes.len() == 32 {
        let pk = identity::ed25519::PublicKey::try_from_bytes(&pk_bytes).ok()?;
        PeerId::from_public_key(&identity::PublicKey::from(pk))
    } else {
        return None;
    };

    let mut addr = Multiaddr::empty();
    addr.push(ip);
    addr.push(Protocol::Tcp(tcp_port));
    addr.push(Protocol::P2p(peer_id));
    Some((peer_id, addr))
}
