# consensoor-p2p

Native Rust libp2p stack used by [consensoor](../). Replaces the slow / unreliable
py-libp2p with a minimal Rust binding configured the same way Lighthouse
configures `lighthouse_network` (TCP + Noise + Yamux, gossipsub with the Eth2
message-id rules, identify, ping, request/response).

## Build

```
cd consensoor-p2p-rs
python3 -m maturin develop --release
```
