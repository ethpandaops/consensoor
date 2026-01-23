# Consensoor Development Context

## Project Overview

Consensoor is a Python Ethereum consensus layer client implementing the Fulu fork specification. It's being tested in Kurtosis devnets alongside Lighthouse.

## Current Work: P2P Block Sync from Lighthouse

### Goal
Enable consensoor to receive and process beacon blocks from Lighthouse via P2P gossipsub so it stays in sync with the network. **Do not use checkpoint sync** - the user explicitly rejected this approach.

### Test Environment
- Kurtosis enclave with minimal preset (8 slots per epoch)
- Lighthouse as supernode (67 validators)
- Consensoor (32 validators)
- Fork: Fulu (fork_version `0x70000038`, fork_digest `b2f55651`)

Run test with:
```bash
cd /Users/bbusa/Ethereum/temp/ethereum-package
kurtosis run . --args-file .github/tests/consensoor-basic.yaml --enclave test
```

### P2P Fixes Applied (in `consensoor/p2p/host.py`)

1. **Snappy framing format** - Eth2 req/resp requires snappy framing format, not raw snappy blocks
   ```python
   def _snappy_frame_compress(self, data: bytes) -> bytes:
       import snappy
       compressor = snappy.StreamCompressor()
       return compressor.add_chunk(data)
   ```

2. **Varint decode helper** - Python `varint` package lacks decode function
   ```python
   @staticmethod
   def _decode_varint(buf: bytes) -> tuple[int, int]:
       value = 0
       shift = 0
       for i, byte in enumerate(buf):
           value |= (byte & 0x7f) << shift
           shift += 7
           if (byte & 0x80) == 0:
               return value, i + 1
       raise ValueError("Truncated varint")
   ```

3. **Status request format** - Send varint-encoded UNCOMPRESSED length + snappy framed data
   ```python
   compressed = self._snappy_frame_compress(status_ssz)
   length_prefix = varint.encode(len(status_ssz))  # Length of UNCOMPRESSED data
   message = length_prefix + compressed
   ```

4. **Metadata v2 format** - Use 17-byte Altair format, not 25-byte Fulu format
   ```python
   # MetaData v2 (Altair+): seq_number (8) + attnets (8) + syncnets (1) = 17 bytes
   # Note: Fulu/PeerDAS would use metadata/3 with custody_group_count
   ```

### Current Status (What Works)
- ✅ Status exchange with Lighthouse
- ✅ Ping/pong protocol
- ✅ Gossipsub topic subscription for `beacon_block`

### Current Issue: Gossipsub Mesh Reliability
- py-libp2p gossipsub only delivered 2 blocks (slot 1 and 8), then stopped
- Lighthouse and consensoor end up on different forks (different head roots)
- Consensoor builds its own blocks with INVALID execution payloads

Example state observed:
- Lighthouse: slot 36, head `0x4cfe2b15b9...`
- Consensoor: slot 36, head `0xc0ce23a121...` (different fork!)

### Next Steps
1. Debug py-libp2p gossipsub mesh management
   - Check if GRAFT/PRUNE messages are being handled correctly
   - Verify mesh membership is maintained with Lighthouse peer
   - May require upstream py-libp2p fixes

2. Alternative: Consider ReqResp block fetching
   - Use `/eth/v2/beacon/blocks/{block_id}` via req/resp instead of gossipsub
   - Could poll for blocks or request on slot tick

### Key Files
- `consensoor/p2p/host.py` - P2P networking, gossipsub, req/resp handlers
- `consensoor/node.py` - Main beacon node logic
- `consensoor/validator/client.py` - Validator duties (proposer_lookahead fix applied)
- `consensoor/beacon_sync/sync.py` - HTTP-based state sync (currently used for initial sync)

### Debugging Commands
```bash
# Check enclave status
kurtosis enclave inspect test

# Get consensoor logs
kurtosis service logs test cl-2-consensoor-geth -f

# Get Lighthouse logs
kurtosis service logs test cl-1-lighthouse-geth -f

# Check consensoor head
kurtosis service exec test cl-2-consensoor-geth -- curl -s http://localhost:4000/eth/v1/beacon/headers/head

# Check Lighthouse head
kurtosis service exec test cl-1-lighthouse-geth -- curl -s http://localhost:4000/eth/v1/beacon/headers/head
```

### Related Plan File
A plan file exists at `~/.claude/plans/expressive-enchanting-boole.md` describing HTTP Beacon API sync approach, but user rejected checkpoint sync. The state sync manager (`beacon_sync/sync.py`) is used for initial state only.
