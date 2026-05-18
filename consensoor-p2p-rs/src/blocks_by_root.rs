//! BeaconBlocksByRoot v2.
//!
//! Wire format (`/eth2/beacon_chain/req/beacon_blocks_by_root/2/ssz_snappy`):
//!
//!   request   := varint(uncompressed_len) ‖ snappy_framed(roots)
//!                where `roots` is the SSZ encoding of `List[Root, MAX_REQUEST_BLOCKS]`,
//!                i.e. N * 32 bytes (N <= MAX_REQUEST_BLOCKS).
//!
//!   response  := stream of chunks, each chunk:
//!                  <result_byte=0> ‖ <context=fork_digest:4> ‖
//!                  varint(uncompressed_len) ‖ snappy_framed(SignedBeaconBlock_ssz)
//!                The responder closes the stream after the last chunk.
//!
//! This is the protocol prysm/lighthouse use to backfill blocks they're
//! missing by-root (e.g. when gossipsub dropped a parent before the child
//! arrived). Without it, peers that miss our gossip can never recover and
//! orphan all our blocks.

use std::io::{self, Cursor, Read, Write};

use libp2p::request_response::{self, Codec, ProtocolSupport};
use libp2p::StreamProtocol;
use pyo3::prelude::*;
use unsigned_varint::{decode as varint_decode, encode as varint_encode};

use crate::blocks_by_range::BlockChunk;

const MAX_BLOCK_BYTES: usize = 10 * 1024 * 1024;
/// MAX_REQUEST_BLOCKS from consensus-specs (deneb+). Larger than the
/// older phase0 value of 1024; we use the more permissive bound for
/// inbound requests so we don't reject otherwise-valid peers.
const MAX_REQUEST_BLOCKS: usize = 1024;
const ROOT_LEN: usize = 32;
const MAX_REQUEST_BYTES: usize = MAX_REQUEST_BLOCKS * ROOT_LEN;

const PROTO_V2: &str = "/eth2/beacon_chain/req/beacon_blocks_by_root/2/ssz_snappy";

/// SSZ List[Root, MAX_REQUEST_BLOCKS]. Since `Root` is fixed-size (32 bytes),
/// the on-wire encoding is just the concatenation of every root.
#[pyclass]
#[derive(Clone, Debug)]
pub struct BlocksByRootRequest {
    /// Flat byte buffer of N * 32 bytes (one 32-byte root per entry).
    #[pyo3(get, set)]
    pub roots: Vec<u8>,
}

#[pymethods]
impl BlocksByRootRequest {
    #[new]
    pub fn new(roots: Vec<u8>) -> PyResult<Self> {
        if roots.len() % ROOT_LEN != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "BlocksByRootRequest: roots length {} is not a multiple of 32",
                roots.len()
            )));
        }
        Ok(Self { roots })
    }

    pub fn __repr__(&self) -> String {
        format!("BlocksByRootRequest(n_roots={})", self.roots.len() / ROOT_LEN)
    }

    /// Number of 32-byte roots in this request.
    pub fn count(&self) -> usize {
        self.roots.len() / ROOT_LEN
    }
}

impl BlocksByRootRequest {
    pub fn encode_ssz(&self) -> Vec<u8> {
        self.roots.clone()
    }

    pub fn decode_ssz(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() % ROOT_LEN != 0 {
            return Err(format!(
                "blocks_by_root request length {} not a multiple of 32",
                bytes.len()
            ));
        }
        let n = bytes.len() / ROOT_LEN;
        if n > MAX_REQUEST_BLOCKS {
            return Err(format!(
                "blocks_by_root request has {n} roots, exceeds MAX_REQUEST_BLOCKS={MAX_REQUEST_BLOCKS}"
            ));
        }
        Ok(Self {
            roots: bytes.to_vec(),
        })
    }
}

/// Whole response: ordered sequence of block chunks the responder produced.
/// Same chunk type as blocks_by_range (`BlockChunk`).
#[pyclass]
#[derive(Clone, Debug)]
pub struct BlocksByRootResponse {
    #[pyo3(get)]
    pub chunks: Vec<BlockChunk>,
    #[pyo3(get)]
    pub error: Option<String>,
}

#[pymethods]
impl BlocksByRootResponse {
    #[new]
    #[pyo3(signature = (chunks=Vec::new(), error=None))]
    pub fn new(chunks: Vec<BlockChunk>, error: Option<String>) -> Self {
        Self { chunks, error }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "BlocksByRootResponse(chunks={}, error={:?})",
            self.chunks.len(),
            self.error
        )
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct BlocksByRootEvent {
    #[pyo3(get)]
    pub peer: String,
    #[pyo3(get)]
    pub kind: String, // "request:<id>" | "response" | "failure"
    #[pyo3(get)]
    pub request: Option<BlocksByRootRequest>,
    #[pyo3(get)]
    pub response: Option<BlocksByRootResponse>,
    #[pyo3(get)]
    pub error: Option<String>,
}

// ============================================================================
// Codec
// ============================================================================

#[derive(Clone, Default)]
pub struct BlocksByRootCodec;

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
impl Codec for BlocksByRootCodec {
    type Protocol = StreamProtocol;
    type Request = BlocksByRootRequest;
    type Response = BlocksByRootResponse;

    async fn read_request<T>(
        &mut self,
        _: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Request>
    where
        T: futures::AsyncRead + Unpin + Send,
    {
        use futures::AsyncReadExt;
        // Cap the read to the max request size plus framing overhead.
        let cap = (MAX_REQUEST_BYTES + 64) as u64;
        let mut buf = Vec::new();
        io.take(cap).read_to_end(&mut buf).await?;
        let (declared_len, rest) = varint_decode::usize(&buf).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("varint: {e}"))
        })?;
        if declared_len > MAX_REQUEST_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("blocks_by_root request declares {declared_len} bytes > max {MAX_REQUEST_BYTES}"),
            ));
        }
        if declared_len % ROOT_LEN != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("blocks_by_root declared_len {declared_len} not a multiple of 32"),
            ));
        }
        let mut decoder = snap::read::FrameDecoder::new(Cursor::new(rest));
        let mut payload = Vec::with_capacity(declared_len);
        decoder.read_to_end(&mut payload)?;
        if payload.len() != declared_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "blocks_by_root decompressed {} bytes, declared {}",
                    payload.len(),
                    declared_len,
                ),
            ));
        }
        BlocksByRootRequest::decode_ssz(&payload)
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
                let (err_len, rest) = varint_decode::usize(cursor).map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("varint(err_len): {e}"))
                })?;
                let mut decoder = snap::read::FrameDecoder::new(Cursor::new(rest));
                let mut payload = Vec::with_capacity(err_len);
                let _ = decoder.read_to_end(&mut payload);
                return Ok(BlocksByRootResponse {
                    chunks,
                    error: Some(format!(
                        "result={result_byte}: {}",
                        String::from_utf8_lossy(&payload)
                    )),
                });
            }

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
            let varint_consumed = cursor.len() - rest.len();
            cursor = &cursor[varint_consumed..];

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

        Ok(BlocksByRootResponse {
            chunks,
            error: None,
        })
    }

    async fn write_request<T>(
        &mut self,
        _: &Self::Protocol,
        io: &mut T,
        req: Self::Request,
    ) -> io::Result<()>
    where
        T: futures::AsyncWrite + Unpin + Send,
    {
        use futures::AsyncWriteExt;
        let payload = req.encode_ssz();

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
        io.close().await?;
        Ok(())
    }
}

pub type BlocksByRootBehaviour = request_response::Behaviour<BlocksByRootCodec>;

pub fn new_blocks_by_root_behaviour() -> BlocksByRootBehaviour {
    let v2 = StreamProtocol::new(PROTO_V2);
    let cfg = request_response::Config::default()
        .with_request_timeout(std::time::Duration::from_secs(30));
    request_response::Behaviour::with_codec(
        BlocksByRootCodec,
        std::iter::once((v2, ProtocolSupport::Full)),
        cfg,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_roundtrip() {
        let roots: Vec<u8> = (0..96).map(|i| (i % 256) as u8).collect();
        let r = BlocksByRootRequest::new(roots.clone()).unwrap();
        assert_eq!(r.count(), 3);
        let bytes = r.encode_ssz();
        assert_eq!(bytes, roots);
        let d = BlocksByRootRequest::decode_ssz(&bytes).unwrap();
        assert_eq!(d.roots, roots);
    }

    #[test]
    fn request_rejects_non_multiple_of_32() {
        let bad = vec![0u8; 33];
        assert!(BlocksByRootRequest::decode_ssz(&bad).is_err());
    }

    #[test]
    fn request_rejects_oversized() {
        let big = vec![0u8; (MAX_REQUEST_BLOCKS + 1) * ROOT_LEN];
        assert!(BlocksByRootRequest::decode_ssz(&big).is_err());
    }

    #[test]
    fn chunk_encode() {
        let bytes = encode_one_chunk(&[0xb2, 0xf5, 0x56, 0x51], b"fake-block").unwrap();
        assert_eq!(bytes[0], 0);
        assert_eq!(&bytes[1..5], &[0xb2, 0xf5, 0x56, 0x51]);
    }
}
