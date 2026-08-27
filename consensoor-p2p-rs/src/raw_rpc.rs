//! Generic Eth2 req/resp transport for protocols whose SSZ is handled in
//! Python: the request is an opaque SSZ payload, the response a sequence of
//! context-tagged SSZ chunks.
//!
//! Wire format (identical to BeaconBlocksByRoot):
//!
//!   request   := varint(uncompressed_len) ‖ snappy_framed(request_ssz)
//!   response  := chunks of <result_byte=0> ‖ <context:4> ‖
//!                varint(uncompressed_len) ‖ snappy_framed(chunk_ssz)
//!
//! One behaviour instance per protocol (outbound negotiation offers every
//! protocol a behaviour supports, so mixing protocols in one instance would
//! make the negotiated protocol ambiguous). Used for
//! `execution_payload_envelopes_by_root/1`, `data_column_sidecars_by_root/1`
//! and `data_column_sidecars_by_range/1`.

use std::io::{self, Cursor, Read, Write};

use libp2p::request_response::{self, Codec, ProtocolSupport};
use libp2p::StreamProtocol;
use pyo3::prelude::*;
use unsigned_varint::{decode as varint_decode, encode as varint_encode};

pub const PROTO_ENVELOPES_BY_ROOT: &str =
    "/eth2/beacon_chain/req/execution_payload_envelopes_by_root/1/ssz_snappy";
pub const PROTO_COLUMNS_BY_ROOT: &str =
    "/eth2/beacon_chain/req/data_column_sidecars_by_root/1/ssz_snappy";
pub const PROTO_COLUMNS_BY_RANGE: &str =
    "/eth2/beacon_chain/req/data_column_sidecars_by_range/1/ssz_snappy";

/// Largest request we accept (DataColumnsByRootIdentifiers for 128 blocks x
/// 128 columns is ~135 KiB).
const MAX_REQUEST_BYTES: usize = 1024 * 1024;
/// Largest single chunk (a SignedExecutionPayloadEnvelope can approach the
/// 10 MiB gossip limit).
const MAX_CHUNK_BYTES: usize = 10 * 1024 * 1024;
/// Largest response we buffer.
const MAX_RESPONSE_BYTES: u64 = 64 * 1024 * 1024;
/// Upper bound on chunks per response (compute_max_request_data_column_sidecars
/// on minimal/mainnet is 128 * 128).
const MAX_CHUNKS: usize = 16384;

#[pyclass]
#[derive(Clone, Debug)]
pub struct RawRpcRequest {
    /// Protocol id the request arrived on / should be sent on.
    #[pyo3(get, set)]
    pub protocol: String,
    /// Raw SSZ request payload (uncompressed).
    #[pyo3(get, set)]
    pub payload: Vec<u8>,
}

#[pymethods]
impl RawRpcRequest {
    #[new]
    pub fn new(protocol: String, payload: Vec<u8>) -> Self {
        Self { protocol, payload }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "RawRpcRequest(protocol={:?}, payload_len={})",
            self.protocol,
            self.payload.len()
        )
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct RawChunk {
    /// 4-byte context (fork digest of the chunk's slot).
    #[pyo3(get, set)]
    pub context: Vec<u8>,
    /// Uncompressed SSZ chunk.
    #[pyo3(get, set)]
    pub ssz: Vec<u8>,
}

#[pymethods]
impl RawChunk {
    #[new]
    pub fn new(context: Vec<u8>, ssz: Vec<u8>) -> Self {
        Self { context, ssz }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct RawRpcResponse {
    #[pyo3(get)]
    pub chunks: Vec<RawChunk>,
    #[pyo3(get)]
    pub error: Option<String>,
}

#[pymethods]
impl RawRpcResponse {
    #[new]
    #[pyo3(signature = (chunks=Vec::new(), error=None))]
    pub fn new(chunks: Vec<RawChunk>, error: Option<String>) -> Self {
        Self { chunks, error }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "RawRpcResponse(chunks={}, error={:?})",
            self.chunks.len(),
            self.error
        )
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct RawRpcEvent {
    #[pyo3(get)]
    pub peer: String,
    /// Protocol id this event belongs to.
    #[pyo3(get)]
    pub protocol: String,
    /// "request:<id>" | "response" | "failure"
    #[pyo3(get)]
    pub kind: String,
    #[pyo3(get)]
    pub request: Option<RawRpcRequest>,
    #[pyo3(get)]
    pub response: Option<RawRpcResponse>,
    #[pyo3(get)]
    pub error: Option<String>,
}

// ============================================================================
// Codec
// ============================================================================

#[derive(Clone, Default)]
pub struct RawRpcCodec;

fn snappy_frame(payload: &[u8]) -> io::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(payload.len() + 16);
    let mut varint_buf = varint_encode::usize_buffer();
    out.extend_from_slice(varint_encode::usize(payload.len(), &mut varint_buf));
    let mut compressed = Vec::new();
    {
        let mut writer = snap::write::FrameEncoder::new(&mut compressed);
        writer.write_all(payload)?;
        writer.flush()?;
    }
    out.extend_from_slice(&compressed);
    Ok(out)
}

fn encode_one_chunk(context: &[u8], ssz: &[u8]) -> io::Result<Vec<u8>> {
    if context.len() != 4 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("context must be 4 bytes, got {}", context.len()),
        ));
    }
    let mut out = Vec::with_capacity(ssz.len() + 16);
    out.push(0u8);
    out.extend_from_slice(context);
    out.extend_from_slice(&snappy_frame(ssz)?);
    Ok(out)
}

#[async_trait::async_trait]
impl Codec for RawRpcCodec {
    type Protocol = StreamProtocol;
    type Request = RawRpcRequest;
    type Response = RawRpcResponse;

    async fn read_request<T>(&mut self, proto: &Self::Protocol, io: &mut T) -> io::Result<Self::Request>
    where
        T: futures::AsyncRead + Unpin + Send,
    {
        use futures::AsyncReadExt;
        let mut buf = Vec::new();
        io.take((MAX_REQUEST_BYTES + 64) as u64).read_to_end(&mut buf).await?;
        let (declared_len, rest) = varint_decode::usize(&buf)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("varint: {e}")))?;
        if declared_len > MAX_REQUEST_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("raw_rpc request declares {declared_len} bytes > max {MAX_REQUEST_BYTES}"),
            ));
        }
        let mut decoder = snap::read::FrameDecoder::new(Cursor::new(rest));
        let mut payload = Vec::with_capacity(declared_len);
        decoder.read_to_end(&mut payload)?;
        if payload.len() != declared_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("raw_rpc decompressed {} bytes, declared {}", payload.len(), declared_len),
            ));
        }
        Ok(RawRpcRequest {
            protocol: proto.as_ref().to_string(),
            payload,
        })
    }

    async fn read_response<T>(&mut self, _: &Self::Protocol, io: &mut T) -> io::Result<Self::Response>
    where
        T: futures::AsyncRead + Unpin + Send,
    {
        use futures::AsyncReadExt;
        let mut all = Vec::new();
        io.take(MAX_RESPONSE_BYTES).read_to_end(&mut all).await?;

        let mut chunks = Vec::new();
        let mut cursor = &all[..];
        while !cursor.is_empty() {
            let result_byte = cursor[0];
            cursor = &cursor[1..];
            if result_byte != 0 {
                let (err_len, rest) = varint_decode::usize(cursor).map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("varint(err_len): {e}"))
                })?;
                let mut decoder = snap::read::FrameDecoder::new(Cursor::new(rest));
                let mut payload = Vec::with_capacity(err_len);
                let _ = decoder.read_to_end(&mut payload);
                return Ok(RawRpcResponse {
                    chunks,
                    error: Some(format!(
                        "result={result_byte}: {}",
                        String::from_utf8_lossy(&payload)
                    )),
                });
            }
            if cursor.len() < 4 {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "truncated chunk context"));
            }
            let context = cursor[0..4].to_vec();
            cursor = &cursor[4..];
            let (chunk_len, rest) = varint_decode::usize(cursor).map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("varint(chunk_len): {e}"))
            })?;
            if chunk_len > MAX_CHUNK_BYTES {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("declared chunk size {chunk_len} > max {MAX_CHUNK_BYTES}"),
                ));
            }
            let varint_consumed = cursor.len() - rest.len();
            cursor = &cursor[varint_consumed..];
            let mut decoder = snap::read::FrameDecoder::new(Cursor::new(cursor));
            let mut ssz = vec![0u8; chunk_len];
            decoder.read_exact(&mut ssz)?;
            let consumed = decoder.into_inner().position() as usize;
            cursor = &cursor[consumed..];
            chunks.push(RawChunk { context, ssz });
            if chunks.len() >= MAX_CHUNKS {
                break;
            }
        }
        Ok(RawRpcResponse { chunks, error: None })
    }

    async fn write_request<T>(&mut self, _: &Self::Protocol, io: &mut T, req: Self::Request) -> io::Result<()>
    where
        T: futures::AsyncWrite + Unpin + Send,
    {
        use futures::AsyncWriteExt;
        io.write_all(&snappy_frame(&req.payload)?).await?;
        io.close().await?;
        Ok(())
    }

    async fn write_response<T>(&mut self, _: &Self::Protocol, io: &mut T, resp: Self::Response) -> io::Result<()>
    where
        T: futures::AsyncWrite + Unpin + Send,
    {
        use futures::AsyncWriteExt;
        for chunk in resp.chunks.iter() {
            io.write_all(&encode_one_chunk(&chunk.context, &chunk.ssz)?).await?;
        }
        io.close().await?;
        Ok(())
    }
}

pub type RawRpcBehaviour = request_response::Behaviour<RawRpcCodec>;

pub fn new_raw_rpc_behaviour(protocol: &'static str) -> RawRpcBehaviour {
    let cfg = request_response::Config::default()
        .with_request_timeout(std::time::Duration::from_secs(30));
    request_response::Behaviour::with_codec(
        RawRpcCodec,
        std::iter::once((StreamProtocol::new(protocol), ProtocolSupport::Full)),
        cfg,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_encode_has_result_and_context() {
        let bytes = encode_one_chunk(&[1, 2, 3, 4], b"payload").unwrap();
        assert_eq!(bytes[0], 0);
        assert_eq!(&bytes[1..5], &[1, 2, 3, 4]);
    }

    #[test]
    fn chunk_encode_rejects_bad_context() {
        assert!(encode_one_chunk(&[1, 2, 3], b"x").is_err());
    }
}
