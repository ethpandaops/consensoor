//! Parsing of dial targets that may arrive as either a libp2p multiaddr or an
//! Ethereum Node Record (ENR). Eth2 clients (Lighthouse, Lodestar, Prysm, ...)
//! advertise themselves via ENR, so consensoor needs to accept those directly
//! until we grow our own discv5.

use enr::{CombinedKey, CombinedPublicKey, Enr, EnrPublicKey};
use libp2p::{identity, multiaddr::Protocol, Multiaddr, PeerId};

/// Parse a bootnode/dial target. Accepts either a multiaddr string
/// (`/ip4/.../tcp/.../p2p/...`) or an ENR (`enr:-...`).
pub fn parse_dial_target(s: &str) -> Result<Multiaddr, String> {
    if s.starts_with("enr:") {
        enr_to_multiaddr(s)
    } else {
        s.parse::<Multiaddr>()
            .map_err(|e| format!("multiaddr parse: {e}"))
    }
}

fn enr_to_multiaddr(s: &str) -> Result<Multiaddr, String> {
    let enr: Enr<CombinedKey> = s.parse().map_err(|e| format!("enr decode: {e}"))?;

    let ip_proto = if let Some(ip) = enr.ip4() {
        Protocol::Ip4(ip)
    } else if let Some(ip) = enr.ip6() {
        Protocol::Ip6(ip)
    } else {
        return Err("enr has no ip4/ip6".to_string());
    };

    let tcp_port = enr
        .tcp4()
        .or_else(|| enr.tcp6())
        .ok_or_else(|| "enr has no tcp port".to_string())?;

    let peer_id = pubkey_to_peer_id(&enr.public_key())?;

    let mut addr = Multiaddr::empty();
    addr.push(ip_proto);
    addr.push(Protocol::Tcp(tcp_port));
    addr.push(Protocol::P2p(peer_id));
    Ok(addr)
}

fn pubkey_to_peer_id(pk: &CombinedPublicKey) -> Result<PeerId, String> {
    let bytes = pk.encode();
    match pk {
        CombinedPublicKey::Secp256k1(_) => {
            let pk = identity::secp256k1::PublicKey::try_from_bytes(&bytes)
                .map_err(|e| format!("secp256k1 decode: {e}"))?;
            Ok(PeerId::from_public_key(&identity::PublicKey::from(pk)))
        }
        CombinedPublicKey::Ed25519(_) => {
            let pk = identity::ed25519::PublicKey::try_from_bytes(&bytes)
                .map_err(|e| format!("ed25519 decode: {e}"))?;
            Ok(PeerId::from_public_key(&identity::PublicKey::from(pk)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_plain_multiaddr() {
        let s = "/ip4/127.0.0.1/tcp/9000";
        let addr = parse_dial_target(s).expect("multiaddr should parse");
        assert_eq!(addr.to_string(), s);
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse_dial_target("not a real address").is_err());
        assert!(parse_dial_target("enr:not-base64!").is_err());
    }
}
