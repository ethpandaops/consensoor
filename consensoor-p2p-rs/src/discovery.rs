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
//! know about; we convert each to its dialable libp2p Multiaddrs (QUIC
//! first: `/ip4/.../udp/<port>/quic-v1/p2p/<peer_id>`, then the TCP
//! fallback) and forward them to the swarm task to dial. New peer →
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
/// `discovered_tx` is sent one `(PeerId, base64 ENR string, Vec<Multiaddr>)`
/// tuple per newly-seen ENR after every `find_node` query — the swarm task
/// receives those, remembers the ENR keyed by peer id (so we can surface
/// it via `/eth/v1/node/peers`), and issues a QUIC-first libp2p dial.
pub async fn spawn_discovery(
    local_enr: Enr,
    enr_key: CombinedKey,
    udp_port: u16,
    bootnodes: Vec<Enr>,
    discovered_tx: Sender<(PeerId, String, Vec<Multiaddr>)>,
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
        if let Some((peer_id, addrs)) = enr_to_peer_id_and_multiaddrs(enr) {
            let enr_b64 = enr.to_base64();
            if discovered_tx.send((peer_id, enr_b64, addrs)).await.is_err() {
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
                        if let Some((peer_id, addrs)) = enr_to_peer_id_and_multiaddrs(&enr) {
                            let enr_b64 = enr.to_base64();
                            // Best-effort send; if the receiver is dropped
                            // (swarm shut down), break the loop.
                            if discovered_tx.send((peer_id, enr_b64, addrs)).await.is_err() {
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

/// Convert an Eth2 ENR to the (PeerId, dialable multiaddrs) pair.
///
/// The ENR's `secp256k1` field gives us the libp2p PeerId; ip4/ip6 plus
/// the `quic`/`quic6`/tcp4/tcp6 ports give us the transports. QUIC
/// addresses come first — p2p-interface makes QUIC the primary transport
/// and says clients SHOULD prioritise a peer's QUIC addresses; the
/// swarm's dial_concurrency_factor=1 turns the returned order into a
/// sequential try-QUIC-then-TCP fallback. Returns None for ENRs without
/// any usable (ip, port) endpoint or without a decodable pubkey.
pub fn enr_to_peer_id_and_multiaddrs(enr: &Enr) -> Option<(PeerId, Vec<Multiaddr>)> {
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

    let build = |ip: Protocol<'static>, port_proto: Protocol<'static>, quic: bool| {
        let mut addr = Multiaddr::empty();
        addr.push(ip);
        addr.push(port_proto);
        if quic {
            addr.push(Protocol::QuicV1);
        }
        addr.push(Protocol::P2p(peer_id));
        addr
    };

    let mut quic_addrs = Vec::new();
    let mut tcp_addrs = Vec::new();
    if let Some(ip) = enr.ip4() {
        // The enr crate has no quic4()/quic6() helpers — read the raw
        // keys the way Lighthouse writes them (RLP-encoded u16).
        if let Some(port) = enr.get_decodable::<u16>("quic").and_then(Result::ok) {
            quic_addrs.push(build(Protocol::Ip4(ip), Protocol::Udp(port), true));
        }
        if let Some(port) = enr.tcp4() {
            tcp_addrs.push(build(Protocol::Ip4(ip), Protocol::Tcp(port), false));
        }
    }
    if let Some(ip) = enr.ip6() {
        if let Some(port) = enr.get_decodable::<u16>("quic6").and_then(Result::ok) {
            quic_addrs.push(build(Protocol::Ip6(ip), Protocol::Udp(port), true));
        }
        if let Some(port) = enr.tcp6() {
            tcp_addrs.push(build(Protocol::Ip6(ip), Protocol::Tcp(port), false));
        }
    }

    quic_addrs.extend(tcp_addrs);
    if quic_addrs.is_empty() {
        return None;
    }
    Some((peer_id, quic_addrs))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_enr(quic: Option<u16>, tcp: Option<u16>) -> Enr {
        let key = CombinedKey::generate_secp256k1();
        let mut b = Enr::builder();
        b.ip("192.0.2.7".parse().unwrap());
        if let Some(p) = tcp {
            b.tcp4(p);
        }
        if let Some(p) = quic {
            b.add_value("quic", &p);
        }
        b.build(&key).unwrap()
    }

    #[test]
    fn quic_addr_comes_before_tcp_fallback() {
        let enr = test_enr(Some(9001), Some(9000));
        let (peer_id, addrs) = enr_to_peer_id_and_multiaddrs(&enr).unwrap();
        assert_eq!(addrs.len(), 2);
        assert_eq!(
            addrs[0].to_string(),
            format!("/ip4/192.0.2.7/udp/9001/quic-v1/p2p/{peer_id}")
        );
        assert_eq!(
            addrs[1].to_string(),
            format!("/ip4/192.0.2.7/tcp/9000/p2p/{peer_id}")
        );
    }

    #[test]
    fn tcp_only_enr_still_dialable() {
        let enr = test_enr(None, Some(9000));
        let (peer_id, addrs) = enr_to_peer_id_and_multiaddrs(&enr).unwrap();
        assert_eq!(addrs.len(), 1);
        assert_eq!(
            addrs[0].to_string(),
            format!("/ip4/192.0.2.7/tcp/9000/p2p/{peer_id}")
        );
    }

    #[test]
    fn quic_only_enr_is_dialable() {
        let enr = test_enr(Some(9001), None);
        let (_, addrs) = enr_to_peer_id_and_multiaddrs(&enr).unwrap();
        assert_eq!(addrs.len(), 1);
        assert!(addrs[0].to_string().contains("/udp/9001/quic-v1"));
    }

    #[test]
    fn portless_enr_is_skipped() {
        let enr = test_enr(None, None);
        assert!(enr_to_peer_id_and_multiaddrs(&enr).is_none());
    }
}
