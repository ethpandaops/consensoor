//! Eth2 gossipsub peer-scoring parameters.
//!
//! Without an active scoring system enabled, libp2p-gossipsub treats peers
//! neutrally: that's fine for *us*, but prysm/lighthouse/teku run with the
//! Eth2 scoring system *on*, and they expect remote peers to do the same.
//! In practice some implementations are lenient about peers that don't
//! implement scoring, but enabling scoring with the spec's threshold values
//! makes our gossipsub instance behave the same way the Eth2 reference
//! clients do — which is what allows mesh GRAFT to actually progress
//! between us and them.
//!
//! Reference: <https://github.com/ethereum/consensus-specs/blob/dev/specs/phase0/p2p-interface.md#topic-validation>
//!
//! The numbers here are the canonical Eth2 thresholds. The per-topic score
//! params are intentionally left at defaults here — running without
//! per-topic params disables P1..P4 contributions but keeps the IP
//! colocation factor (P6) and behaviour penalty (P7) active, which is
//! enough to negotiate mesh-graft cleanly with peers that score on the
//! same scale.

use std::time::Duration;

use libp2p::gossipsub::{PeerScoreParams, PeerScoreThresholds};

/// Eth2 spec thresholds (the ones lighthouse/prysm/teku all use).
#[allow(dead_code)]
pub fn eth2_thresholds() -> PeerScoreThresholds {
    PeerScoreThresholds {
        gossip_threshold: -4000.0,
        publish_threshold: -8000.0,
        graylist_threshold: -16000.0,
        accept_px_threshold: 100.0,
        opportunistic_graft_threshold: 5.0,
    }
}

/// Eth2-shaped peer score params. Decays once per slot (700ms heartbeat ×
/// SLOTS_PER_EPOCH... we just pin to a 12s decay interval which is what
/// lighthouse uses on mainnet — the decay model doesn't break on shorter
/// devnet slots, it just decays slower in slot-relative terms).
#[allow(dead_code)]
pub fn eth2_peer_score_params() -> PeerScoreParams {
    PeerScoreParams {
        // Aggregate cap on positive topic contributions. Eth2 uses 3600.0
        // (the libp2p default), which keeps a well-behaved peer well above
        // the publish_threshold without runaway positives.
        topic_score_cap: 3600.0,

        // P5 (application-specific). We don't run an app-scoring layer yet,
        // but leave the weight at the libp2p default so app scores would
        // contribute if/when we add them.
        app_specific_weight: 10.0,

        // P6: IP colocation. Eth2 uses a non-trivial weight so that one IP
        // can't open many connections and dominate the mesh. We're not in a
        // multi-tenant network on the devnet but the value is harmless.
        ip_colocation_factor_weight: -35.11,
        ip_colocation_factor_threshold: 10.0,
        ip_colocation_factor_whitelist: Default::default(),

        // P7: behaviour penalty for re-grafting too fast / not following up
        // on IWANTs. Without this, a peer that misbehaves on GRAFT/PRUNE
        // never gets penalised and our mesh fills with churn.
        behaviour_penalty_weight: -15.92,
        behaviour_penalty_threshold: 6.0,
        behaviour_penalty_decay: 0.9928,

        // Counter decay cadence. ~one slot of 12s; on minimal-preset 6s
        // devnets this is two slots, which is fine — scores still decay
        // monotonically over the same number of seconds.
        decay_interval: Duration::from_secs(12),
        decay_to_zero: 0.01,

        // Retain a peer's score for an hour after disconnect; if they
        // reconnect within that window we re-use the score.
        retain_score: Duration::from_secs(3600),

        // Slow-peer penalty (libp2p default).
        slow_peer_weight: -0.2,
        slow_peer_threshold: 0.0,
        slow_peer_decay: 0.2,

        // Per-topic params: empty for now — P1..P4 contributions are off,
        // which is what lighthouse does on a fresh devnet too. This still
        // satisfies the threshold negotiation that prysm/lighthouse run
        // when deciding whether to GRAFT us.
        topics: Default::default(),
    }
}
