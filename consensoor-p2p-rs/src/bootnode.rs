//! Parsing of dial targets that may arrive as either a libp2p multiaddr or an
//! Ethereum Node Record (ENR). Eth2 clients (Lighthouse, Lodestar, Prysm, ...)
//! advertise themselves via ENR, so consensoor needs to accept those directly
//! until we grow our own discv5.

use enr::{CombinedKey, Enr};
use libp2p::Multiaddr;

/// Parse a bootnode/dial target. Accepts either a multiaddr string
/// (`/ip4/.../tcp/.../p2p/...`) or an ENR (`enr:-...`).
///
/// ENRs may yield several addresses, ordered QUIC-first per p2p-interface
/// ("clients SHOULD prioritise peer's QUIC addresses") with TCP as the
/// fallback; a plain multiaddr yields exactly one.
pub fn parse_dial_targets(s: &str) -> Result<Vec<Multiaddr>, String> {
    if s.starts_with("enr:") {
        let enr: Enr<CombinedKey> = s.parse().map_err(|e| format!("enr decode: {e}"))?;
        let (_peer_id, addrs) = crate::discovery::enr_to_peer_id_and_multiaddrs(&enr)
            .ok_or_else(|| "enr has no dialable (ip, quic/tcp) endpoint".to_string())?;
        Ok(addrs)
    } else {
        Ok(vec![s
            .parse::<Multiaddr>()
            .map_err(|e| format!("multiaddr parse: {e}"))?])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_plain_multiaddr() {
        let s = "/ip4/127.0.0.1/tcp/9000";
        let addrs = parse_dial_targets(s).expect("multiaddr should parse");
        assert_eq!(addrs.len(), 1);
        assert_eq!(addrs[0].to_string(), s);
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse_dial_targets("not a real address").is_err());
        assert!(parse_dial_targets("enr:not-base64!").is_err());
    }
}
