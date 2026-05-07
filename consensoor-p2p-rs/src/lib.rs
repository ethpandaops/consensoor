//! Native Rust libp2p stack for consensoor.
//!
//! This is a thin Python binding around `rust-libp2p` configured the same way
//! Lighthouse configures its `lighthouse_network` crate (TCP + Noise + Yamux,
//! gossipsub with the Eth2 message-id rules, request/response, identify, ping).
//!
//! The goal is to replace the slow / buggy py-libp2p stack used by consensoor
//! today.  The Python API exposed here mirrors what consensoor needs from a
//! gossipsub host: subscribe to a topic, publish to a topic, get a callback for
//! each incoming message, plus a request/response client/handler API.

mod network;

use pyo3::prelude::*;

#[pymodule]
fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .try_init()
        .ok();

    m.add_class::<network::Network>()?;
    m.add_class::<network::NetworkConfig>()?;
    m.add_class::<network::GossipMessage>()?;
    m.add_function(wrap_pyfunction!(network::generate_keypair, m)?)?;
    Ok(())
}
