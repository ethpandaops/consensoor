//! Eth2 ReqResp implementation.
//!
//! Wire format per consensus-specs p2p-interface.md:
//!   request := <encoding-dependent-header> | <encoded-payload>
//!   For ssz_snappy:
//!     - varint length (length of UNCOMPRESSED payload, in bytes)
//!     - snappy-FRAMED compressed payload (snap::read::FrameDecoder)
//!
//! Response chunks are:
//!     - 1 byte status code (0 = success)
//!     - then the same length-prefixed framed-snappy body
//!
//! This module implements only the Status (v2) protocol for now. Goodbye,
//! Ping, Metadata, BeaconBlocksByRange/ByRoot, blob & column variants
//! follow the same shape and can be added incrementally.

use std::io::{self, Cursor, Read, Write};

use bytes::{BufMut, Bytes, BytesMut};
use libp2p::request_response::{self, Codec, ProtocolSupport};
use libp2p::StreamProtocol;
use pyo3::prelude::*;
use unsigned_varint::{decode as varint_decode, encode as varint_encode};

/// Eth2 Status v2 message (92 bytes uncompressed).
///
/// Spec layout:
///   ForkDigest      4
///   finalized_root 32
///   finalized_epoch 8
///   head_root      32
///   head_slot       8
///   earliest_available_slot 8 (added in v2)
#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StatusMessage {
    #[pyo3(get, set)]
    pub fork_digest: Vec<u8>,
    #[pyo3(get, set)]
    pub finalized_root: Vec<u8>,
    #[pyo3(get, set)]
    pub finalized_epoch: u64,
    #[pyo3(get, set)]
    pub head_root: Vec<u8>,
    #[pyo3(get, set)]
    pub head_slot: u64,
    #[pyo3(get, set)]
    pub earliest_available_slot: u64,
}

#[pymethods]
impl StatusMessage {
    #[new]
    #[pyo3(signature = (fork_digest, finalized_root, finalized_epoch, head_root, head_slot, earliest_available_slot=0))]
    pub fn new(
        fork_digest: Vec<u8>,
        finalized_root: Vec<u8>,
        finalized_epoch: u64,
        head_root: Vec<u8>,
        head_slot: u64,
        earliest_available_slot: u64,
    ) -> Self {
        Self {
            fork_digest,
            finalized_root,
            finalized_epoch,
            head_root,
            head_slot,
            earliest_available_slot,
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "StatusMessage(fork_digest=0x{}, finalized_epoch={}, head_slot={}, earliest_available_slot={})",
            hex::encode(&self.fork_digest),
            self.finalized_epoch,
            self.head_slot,
            self.earliest_available_slot,
        )
    }
}

impl StatusMessage {
    pub const SSZ_LEN_V2: usize = 4 + 32 + 8 + 32 + 8 + 8;

    /// Encode into the 92-byte SSZ-fixed layout (Status is fully fixed-size).
    pub fn encode_ssz(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(Self::SSZ_LEN_V2);
        out.extend_from_slice(&self.fork_digest);
        out.extend_from_slice(&self.finalized_root);
        out.extend_from_slice(&self.finalized_epoch.to_le_bytes());
        out.extend_from_slice(&self.head_root);
        out.extend_from_slice(&self.head_slot.to_le_bytes());
        out.extend_from_slice(&self.earliest_available_slot.to_le_bytes());
        debug_assert_eq!(out.len(), Self::SSZ_LEN_V2);
        out
    }

    pub fn decode_ssz(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != Self::SSZ_LEN_V2 {
            return Err(format!(
                "Status v2 expects {} bytes, got {}",
                Self::SSZ_LEN_V2,
                bytes.len(),
            ));
        }
        Ok(Self {
            fork_digest: bytes[0..4].to_vec(),
            finalized_root: bytes[4..36].to_vec(),
            finalized_epoch: u64::from_le_bytes(bytes[36..44].try_into().unwrap()),
            head_root: bytes[44..76].to_vec(),
            head_slot: u64::from_le_bytes(bytes[76..84].try_into().unwrap()),
            earliest_available_slot: u64::from_le_bytes(bytes[84..92].try_into().unwrap()),
        })
    }
}

/// Encode an Eth2 RPC payload: varint(uncompressed_len) || framed_snappy(payload).
pub fn encode_eth2_rpc(payload: &[u8]) -> io::Result<Bytes> {
    let mut out = BytesMut::new();

    let mut varint_buf = varint_encode::usize_buffer();
    let varint_bytes = varint_encode::usize(payload.len(), &mut varint_buf);
    out.put_slice(varint_bytes);

    let mut compressed = Vec::new();
    {
        let mut writer = snap::write::FrameEncoder::new(&mut compressed);
        writer.write_all(payload)?;
        writer.flush()?;
    }
    out.put_slice(&compressed);

    Ok(out.freeze())
}

/// Decode an Eth2 RPC payload: varint(uncompressed_len) || framed_snappy(payload).
/// Returns the raw uncompressed payload.
pub fn decode_eth2_rpc(input: &[u8]) -> io::Result<Vec<u8>> {
    let (expected_len, rest) = varint_decode::usize(input)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("varint: {e}")))?;

    if expected_len > 10 * 1024 * 1024 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("declared rpc payload {expected_len} exceeds 10 MiB"),
        ));
    }

    let mut decoder = snap::read::FrameDecoder::new(Cursor::new(rest));
    let mut decompressed = Vec::with_capacity(expected_len);
    decoder.read_to_end(&mut decompressed)?;

    if decompressed.len() != expected_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "decompressed length mismatch: got {}, expected {}",
                decompressed.len(),
                expected_len,
            ),
        ));
    }
    Ok(decompressed)
}

#[derive(Debug, Clone)]
pub struct StatusProtocol;

impl AsRef<str> for StatusProtocol {
    fn as_ref(&self) -> &str {
        "/eth2/beacon_chain/req/status/2/ssz_snappy"
    }
}

#[derive(Clone, Default)]
pub struct StatusCodec;

#[async_trait::async_trait]
impl Codec for StatusCodec {
    type Protocol = StreamProtocol;
    type Request = StatusMessage;
    type Response = StatusMessage;

    async fn read_request<T>(
        &mut self,
        _: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Request>
    where
        T: futures::AsyncRead + Unpin + Send,
    {
        use futures::AsyncReadExt;
        let mut buf = Vec::with_capacity(StatusMessage::SSZ_LEN_V2 + 16);
        io.take(2048).read_to_end(&mut buf).await?;
        let payload = decode_eth2_rpc(&buf)?;
        StatusMessage::decode_ssz(&payload)
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
        let mut status_byte = [0u8; 1];
        io.read_exact(&mut status_byte).await?;
        if status_byte[0] != 0 {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("rpc status code {}", status_byte[0]),
            ));
        }
        let mut buf = Vec::with_capacity(StatusMessage::SSZ_LEN_V2 + 16);
        io.take(2048).read_to_end(&mut buf).await?;
        let payload = decode_eth2_rpc(&buf)?;
        StatusMessage::decode_ssz(&payload)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
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
        let framed = encode_eth2_rpc(&payload)?;
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
        let payload = resp.encode_ssz();
        let framed = encode_eth2_rpc(&payload)?;
        io.write_all(&[0u8]).await?;
        io.write_all(&framed).await?;
        io.close().await?;
        Ok(())
    }
}

pub type StatusBehaviour = request_response::Behaviour<StatusCodec>;

pub fn new_status_behaviour() -> StatusBehaviour {
    let proto = StreamProtocol::new("/eth2/beacon_chain/req/status/2/ssz_snappy");
    let cfg = request_response::Config::default()
        .with_request_timeout(std::time::Duration::from_secs(10));
    request_response::Behaviour::with_codec(
        StatusCodec,
        std::iter::once((proto, ProtocolSupport::Full)),
        cfg,
    )
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct StatusEvent {
    #[pyo3(get)]
    pub peer: String,
    #[pyo3(get)]
    pub kind: String, // "request" | "response" | "failure"
    #[pyo3(get)]
    pub message: Option<StatusMessage>,
    #[pyo3(get)]
    pub error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rpc_roundtrip() {
        let payload = b"deadbeef".repeat(16);
        let framed = encode_eth2_rpc(&payload).unwrap();
        let decoded = decode_eth2_rpc(&framed).unwrap();
        assert_eq!(decoded, payload);
    }

    #[test]
    fn status_roundtrip() {
        let s = StatusMessage::new(
            vec![0xb2, 0xf5, 0x56, 0x51],
            vec![1u8; 32],
            42,
            vec![2u8; 32],
            1024,
            512,
        );
        let bytes = s.encode_ssz();
        assert_eq!(bytes.len(), StatusMessage::SSZ_LEN_V2);
        let decoded = StatusMessage::decode_ssz(&bytes).unwrap();
        assert_eq!(decoded, s);
    }

    #[test]
    fn status_rpc_full_roundtrip() {
        let s = StatusMessage::new(
            vec![0xb2, 0xf5, 0x56, 0x51],
            vec![1u8; 32],
            42,
            vec![2u8; 32],
            1024,
            512,
        );
        let payload = s.encode_ssz();
        let framed = encode_eth2_rpc(&payload).unwrap();
        let decoded_payload = decode_eth2_rpc(&framed).unwrap();
        let decoded_status = StatusMessage::decode_ssz(&decoded_payload).unwrap();
        assert_eq!(decoded_status, s);
    }
}
