//! BeaconBlocksByRange v2 + v3.
//!
//! Wire format:
//!
//!   v2 (`/eth2/beacon_chain/req/beacon_blocks_by_range/2/ssz_snappy`):
//!     request   := varint(uncompressed_len) ‖ snappy_framed(start_slot ‖ count ‖ step)
//!                  // 24-byte payload — `step` is deprecated since Altair but
//!                  // is still part of the v2 SSZ schema and MUST be `1`.
//!   v3 (`/eth2/beacon_chain/req/beacon_blocks_by_range/3/ssz_snappy`):
//!     request   := varint(uncompressed_len) ‖ snappy_framed(start_slot ‖ count)
//!                  // 16-byte payload — `step` removed entirely.
//!
//!   response (both versions, identical):
//!     stream of chunks, where each chunk is:
//!       <result_byte=0> ‖ <context=fork_digest:4> ‖
//!       varint(uncompressed_len) ‖ snappy_framed(SignedBeaconBlock_ssz)
//!     stream is closed by the responder after the last chunk.
//!
//! Up to MAX_REQUEST_BLOCKS (currently 1024) chunks may be returned.

use std::io::{self, Cursor, Read, Write};

use libp2p::request_response::{self, Codec, ProtocolSupport};
use libp2p::StreamProtocol;
use pyo3::prelude::*;
use unsigned_varint::{decode as varint_decode, encode as varint_encode};

const MAX_BLOCK_BYTES: usize = 10 * 1024 * 1024;
const MAX_REQUEST_BLOCKS: usize = 1024;

const PROTO_V2: &str = "/eth2/beacon_chain/req/beacon_blocks_by_range/2/ssz_snappy";
const PROTO_V3: &str = "/eth2/beacon_chain/req/beacon_blocks_by_range/3/ssz_snappy";

/// SSZ-fixed Eth2 ByRange request body (16 bytes for v3, 24 bytes for v2 with
/// the trailing deprecated `step` always set to 1 on the wire).
#[pyclass]
#[derive(Clone, Debug)]
pub struct BlocksByRangeRequest {
    #[pyo3(get, set)]
    pub start_slot: u64,
    #[pyo3(get, set)]
    pub count: u64,
}

#[pymethods]
impl BlocksByRangeRequest {
    #[new]
    pub fn new(start_slot: u64, count: u64) -> Self {
        Self { start_slot, count }
    }
    pub fn __repr__(&self) -> String {
        format!(
            "BlocksByRangeRequest(start_slot={}, count={})",
            self.start_slot, self.count
        )
    }
}

impl BlocksByRangeRequest {
    /// Encode for v3 (16 bytes).
    pub fn encode_ssz_v3(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(16);
        out.extend_from_slice(&self.start_slot.to_le_bytes());
        out.extend_from_slice(&self.count.to_le_bytes());
        out
    }
    /// Encode for v2 (24 bytes; trailing `step` is the deprecated field, set
    /// to 1 per spec — the only value v2 peers will accept).
    pub fn encode_ssz_v2(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(24);
        out.extend_from_slice(&self.start_slot.to_le_bytes());
        out.extend_from_slice(&self.count.to_le_bytes());
        out.extend_from_slice(&1u64.to_le_bytes()); // step (deprecated, MUST be 1)
        out
    }
    /// Decode either v2 (24 bytes) or v3 (16 bytes). For v2 we ignore the
    /// trailing `step` field — the spec deprecates it and we never honor
    /// values other than 1.
    pub fn decode_ssz(bytes: &[u8]) -> Result<Self, String> {
        match bytes.len() {
            16 => Ok(Self {
                start_slot: u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
                count: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            }),
            24 => Ok(Self {
                start_slot: u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
                count: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            }),
            n => Err(format!("blocks_by_range request expects 16 or 24 bytes, got {n}")),
        }
    }
}

/// One chunk of a ByRange response: a fork-digest plus the raw SSZ block bytes.
#[pyclass]
#[derive(Clone, Debug)]
pub struct BlockChunk {
    /// 4-byte ForkDigest (Eth2 RPC context bytes).
    #[pyo3(get)]
    pub fork_digest: Vec<u8>,
    /// Raw SSZ-encoded SignedBeaconBlock.
    #[pyo3(get)]
    pub ssz_block: Vec<u8>,
}

#[pymethods]
impl BlockChunk {
    pub fn __repr__(&self) -> String {
        format!(
            "BlockChunk(fork_digest=0x{}, ssz_block={}B)",
            hex::encode(&self.fork_digest),
            self.ssz_block.len()
        )
    }
}

/// Whole response: the ordered sequence of block chunks the responder produced.
#[pyclass]
#[derive(Clone, Debug)]
pub struct BlocksByRangeResponse {
    #[pyo3(get)]
    pub chunks: Vec<BlockChunk>,
    /// True if the responder explicitly errored or terminated early. Empty
    /// chunk list with `error=None` just means no blocks in that range.
    #[pyo3(get)]
    pub error: Option<String>,
}

#[pymethods]
impl BlocksByRangeResponse {
    pub fn __repr__(&self) -> String {
        format!(
            "BlocksByRangeResponse(chunks={}, error={:?})",
            self.chunks.len(),
            self.error
        )
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct BlocksByRangeEvent {
    #[pyo3(get)]
    pub peer: String,
    #[pyo3(get)]
    pub kind: String, // "request:<id>" | "response" | "failure"
    #[pyo3(get)]
    pub request: Option<BlocksByRangeRequest>,
    #[pyo3(get)]
    pub response: Option<BlocksByRangeResponse>,
    #[pyo3(get)]
    pub error: Option<String>,
}

// ============================================================================
// Codec
// ============================================================================

#[derive(Clone, Default)]
pub struct BlocksByRangeCodec;

fn encode_one_chunk(fork_digest: &[u8], ssz_block: &[u8]) -> io::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(ssz_block.len() + 16);
    out.push(0u8); // result = success
    if fork_digest.len() != 4 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("fork_digest must be 4 bytes, got {}", fork_digest.len()),
        ));
    }
    out.extend_from_slice(fork_digest);

    let mut varint_buf = varint_encode::usize_buffer();
    let varint_bytes = varint_encode::usize(ssz_block.len(), &mut varint_buf);
    out.extend_from_slice(varint_bytes);

    let mut compressed = Vec::new();
    {
        let mut writer = snap::write::FrameEncoder::new(&mut compressed);
        writer.write_all(ssz_block)?;
        writer.flush()?;
    }
    out.extend_from_slice(&compressed);
    Ok(out)
}

#[async_trait::async_trait]
impl Codec for BlocksByRangeCodec {
    type Protocol = StreamProtocol;
    type Request = BlocksByRangeRequest;
    type Response = BlocksByRangeResponse;

    async fn read_request<T>(
        &mut self,
        protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Request>
    where
        T: futures::AsyncRead + Unpin + Send,
    {
        use futures::AsyncReadExt;
        let expected = expected_request_len(protocol.as_ref());
        let mut buf = Vec::new();
        io.take(64).read_to_end(&mut buf).await?;
        let (declared_len, rest) = varint_decode::usize(&buf).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("varint: {e}"))
        })?;
        if declared_len != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "by_range request expects {expected} bytes on {}, declared {declared_len}",
                    protocol.as_ref()
                ),
            ));
        }
        let mut decoder = snap::read::FrameDecoder::new(Cursor::new(rest));
        let mut payload = Vec::with_capacity(expected);
        decoder.read_to_end(&mut payload)?;
        BlocksByRangeRequest::decode_ssz(&payload)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    async fn read_response<T>(
        &mut self,
        _: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Response>
    where
        T: futures::AsyncRead + Unpin + Send,
    {
        // Slurp the whole stream then split into chunks. Lighthouse always
        // closes the stream after the last chunk, so this is correct (even
        // if higher-latency than real streaming).
        use futures::AsyncReadExt;
        let mut all = Vec::new();
        io.take((MAX_BLOCK_BYTES * MAX_REQUEST_BLOCKS / 32) as u64)
            .read_to_end(&mut all)
            .await?;

        let mut chunks = Vec::new();
        let mut cursor = &all[..];
        while !cursor.is_empty() {
            if cursor.len() < 1 + 4 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("truncated chunk header, remaining={}", cursor.len()),
                ));
            }
            let result_byte = cursor[0];
            cursor = &cursor[1..];

            if result_byte != 0 {
                // First byte != 0: terminal error chunk. Body is an
                // ErrorMessage (variable-len ASCII string up to 256 bytes,
                // framed identically). We surface the raw bytes as the
                // error string and stop parsing.
                let (err_len, rest) = varint_decode::usize(cursor).map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("varint(err_len): {e}"))
                })?;
                let mut decoder = snap::read::FrameDecoder::new(Cursor::new(rest));
                let mut payload = Vec::with_capacity(err_len);
                let _ = decoder.read_to_end(&mut payload);
                return Ok(BlocksByRangeResponse {
                    chunks,
                    error: Some(format!(
                        "result={result_byte}: {}",
                        String::from_utf8_lossy(&payload)
                    )),
                });
            }

            // Success chunk: 4-byte fork_digest, varint(len), snappy-framed body.
            let fork_digest = cursor[0..4].to_vec();
            cursor = &cursor[4..];

            let (block_len, rest) = varint_decode::usize(cursor).map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("varint(block_len): {e}"))
            })?;
            if block_len > MAX_BLOCK_BYTES {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("declared block size {block_len} > 10 MiB"),
                ));
            }
            let varint_consumed = (cursor.len() - rest.len()) as usize;
            cursor = &cursor[varint_consumed..];

            // The framed-snappy reader consumes from a Cursor over `cursor`
            // until it has decompressed `block_len` bytes. Read EXACTLY
            // `block_len` (not `read_to_end`) — the next bytes after the
            // last data frame belong to the *next* RPC chunk, not to this
            // snappy stream, and there is no explicit end-of-stream marker
            // in the snappy framed format. After read_exact, `inner.position()`
            // tells us how many of `cursor`'s bytes were actually consumed
            // off the wire; advance past that.
            use std::io::Read as _;
            let mut decoder = snap::read::FrameDecoder::new(Cursor::new(cursor));
            let mut ssz_block = vec![0u8; block_len];
            decoder.read_exact(&mut ssz_block)?;
            let consumed = decoder.into_inner().position() as usize;
            cursor = &cursor[consumed..];

            chunks.push(BlockChunk {
                fork_digest,
                ssz_block,
            });

            if chunks.len() >= MAX_REQUEST_BLOCKS {
                break;
            }
        }

        Ok(BlocksByRangeResponse {
            chunks,
            error: None,
        })
    }

    async fn write_request<T>(
        &mut self,
        protocol: &Self::Protocol,
        io: &mut T,
        req: Self::Request,
    ) -> io::Result<()>
    where
        T: futures::AsyncWrite + Unpin + Send,
    {
        use futures::AsyncWriteExt;
        // v2 wants 24 bytes (start_slot ‖ count ‖ step=1); v3 wants 16 bytes.
        // Sending the wrong size on v2 makes prysm/lighthouse register a
        // bad-response strike against us and disconnect after 5 strikes.
        let payload = if protocol.as_ref() == PROTO_V2 {
            req.encode_ssz_v2()
        } else {
            req.encode_ssz_v3()
        };

        let mut framed = Vec::new();
        let mut varint_buf = varint_encode::usize_buffer();
        let varint_bytes = varint_encode::usize(payload.len(), &mut varint_buf);
        framed.extend_from_slice(varint_bytes);

        let mut compressed = Vec::new();
        {
            let mut writer = snap::write::FrameEncoder::new(&mut compressed);
            writer.write_all(&payload)?;
            writer.flush()?;
        }
        framed.extend_from_slice(&compressed);

        io.write_all(&framed).await?;
        io.close().await?;
        Ok(())
    }

    async fn write_response<T>(
        &mut self,
        _: &Self::Protocol,
        io: &mut T,
        resp: Self::Response,
    ) -> io::Result<()>
    where
        T: futures::AsyncWrite + Unpin + Send,
    {
        use futures::AsyncWriteExt;
        for chunk in resp.chunks.iter() {
            let bytes = encode_one_chunk(&chunk.fork_digest, &chunk.ssz_block)?;
            io.write_all(&bytes).await?;
        }
        // Closing the stream signals end-of-response.
        io.close().await?;
        Ok(())
    }
}

/// Walk a snappy-framed byte slice and return how many input bytes a single
/// stream's worth of frames occupies (i.e. up to and including the `END` /
/// implicit termination at the next chunk boundary).
///
/// Snappy frame format:
///   chunk_type(1) | length(3 LE) | payload(length)
///   stream_identifier first chunk has type 0xff
fn consumed_snappy_frame_bytes(input: &[u8]) -> io::Result<usize> {
    let mut idx = 0;
    let mut saw_data = false;
    while idx + 4 <= input.len() {
        let chunk_type = input[idx];
        let len = u32::from_le_bytes([input[idx + 1], input[idx + 2], input[idx + 3], 0]) as usize;
        let total = 4 + len;
        if idx + total > input.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "snappy frame truncated: idx={idx} type={chunk_type:#x} len={len} buf={}",
                    input.len()
                ),
            ));
        }
        idx += total;
        if chunk_type == 0xff {
            // stream identifier — keep going
            continue;
        }
        // 0x00 = compressed data, 0x01 = uncompressed data
        if chunk_type == 0x00 || chunk_type == 0x01 {
            saw_data = true;
            // After the first data chunk, the next byte that's not part of a
            // data/padding/skippable frame is the start of the next RPC chunk.
            // We peek without consuming.
            if idx >= input.len() {
                return Ok(idx);
            }
            let next = input[idx];
            if !is_snappy_chunk_continuation(next) {
                return Ok(idx);
            }
            continue;
        }
        // Padding / skippable / unskippable frames
        if (0x80..=0xfe).contains(&chunk_type) {
            // Skippable, keep going.
            continue;
        }
        // Reserved unskippable
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown snappy frame type {chunk_type:#x}"),
        ));
    }
    if !saw_data {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "ran out of input before any snappy data chunk",
        ));
    }
    Ok(idx)
}

fn is_snappy_chunk_continuation(byte: u8) -> bool {
    matches!(byte, 0x00 | 0x01 | 0x80..=0xfe | 0xff)
}

pub type BlocksByRangeBehaviour = request_response::Behaviour<BlocksByRangeCodec>;

fn expected_request_len(protocol: &str) -> usize {
    if protocol == PROTO_V2 {
        24
    } else {
        16
    }
}

pub fn new_blocks_by_range_behaviour() -> BlocksByRangeBehaviour {
    // Advertise both v2 and v3. The codec serialises the body according to
    // whichever protocol libp2p negotiates with the peer:
    //   - prysm currently registers only v2 — uses 24-byte body with step=1
    //   - lighthouse advertises both — typically picks v3 (16 bytes)
    let v2 = StreamProtocol::new(PROTO_V2);
    let v3 = StreamProtocol::new(PROTO_V3);
    let cfg = request_response::Config::default()
        .with_request_timeout(std::time::Duration::from_secs(30));
    request_response::Behaviour::with_codec(
        BlocksByRangeCodec,
        [
            (v2, ProtocolSupport::Full),
            (v3, ProtocolSupport::Full),
        ],
        cfg,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_v3_roundtrip() {
        let r = BlocksByRangeRequest::new(123, 32);
        let b = r.encode_ssz_v3();
        assert_eq!(b.len(), 16);
        let d = BlocksByRangeRequest::decode_ssz(&b).unwrap();
        assert_eq!(d.start_slot, 123);
        assert_eq!(d.count, 32);
    }

    #[test]
    fn request_v2_roundtrip() {
        let r = BlocksByRangeRequest::new(123, 32);
        let b = r.encode_ssz_v2();
        assert_eq!(b.len(), 24);
        // Trailing step field MUST be 1 on the wire.
        assert_eq!(&b[16..24], &1u64.to_le_bytes());
        let d = BlocksByRangeRequest::decode_ssz(&b).unwrap();
        assert_eq!(d.start_slot, 123);
        assert_eq!(d.count, 32);
    }

    #[test]
    fn chunk_encode() {
        let bytes = encode_one_chunk(&[0x4d, 0x21, 0xf1, 0x63], b"fake-block-bytes").unwrap();
        assert_eq!(bytes[0], 0); // success
        assert_eq!(&bytes[1..5], &[0x4d, 0x21, 0xf1, 0x63]);
    }
}
