//! Native Rust libp2p stack for consensoor.
//!
//! This is a thin Python binding around `rust-libp2p` configured the same way
//! Lighthouse configures its `lighthouse_network` crate (TCP + Noise + Yamux,
//! gossipsub with the Eth2 message-id rules, identify, ping, request/response).
//!
//! The Python API exposed here mirrors what consensoor needs from a gossipsub
//! host: subscribe to a topic, publish to a topic, get a callback for each
//! incoming message, plus a request/response client/handler API for the Eth2
//! Status protocol.

mod blocks_by_range;
mod blocks_by_root;
mod bootnode;
mod discovery;
mod gossip;
mod network;
mod peer_score;
mod rpc;

use pyo3::prelude::*;

#[pymodule]
fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Default filter: info+ globally, but libp2p_gossipsub gets bumped
    // to error+ because its WARN-level chatter is dominated by benign
    // "Not publishing a message that has already been published" lines
    // (one per local validator * subnet, every slot). Operators can
    // still override with RUST_LOG=libp2p_gossipsub=warn if they need
    // to see those during debugging.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| {
                    tracing_subscriber::EnvFilter::new("info,libp2p_gossipsub=error")
                }),
        )
        .with_target(false)
        .try_init()
        .ok();

    m.add_class::<network::Network>()?;
    m.add_class::<network::NetworkConfig>()?;
    m.add_class::<network::GossipMessage>()?;
    m.add_class::<rpc::StatusMessage>()?;
    m.add_class::<rpc::StatusEvent>()?;
    m.add_class::<rpc::PingMessage>()?;
    m.add_class::<rpc::PingEvent>()?;
    m.add_class::<rpc::GoodbyeMessage>()?;
    m.add_class::<rpc::GoodbyeEvent>()?;
    m.add_class::<rpc::MetaDataMessage>()?;
    m.add_class::<rpc::MetadataEvent>()?;
    m.add_class::<blocks_by_range::BlocksByRangeRequest>()?;
    m.add_class::<blocks_by_range::BlockChunk>()?;
    m.add_class::<blocks_by_range::BlocksByRangeResponse>()?;
    m.add_class::<blocks_by_range::BlocksByRangeEvent>()?;
    m.add_class::<blocks_by_root::BlocksByRootRequest>()?;
    m.add_class::<blocks_by_root::BlocksByRootResponse>()?;
    m.add_class::<blocks_by_root::BlocksByRootEvent>()?;
    m.add_function(wrap_pyfunction!(network::generate_keypair, m)?)?;
    Ok(())
}
