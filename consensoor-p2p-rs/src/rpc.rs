//! Eth2 ReqResp implementation.
//!
//! Wire format (per consensus-specs p2p-interface.md, ssz_snappy encoding):
//!
//!     request   := varint(uncompressed_len) ‖ snappy_framed(payload)
//!     response  := <status_byte> ‖ varint(uncompressed_len) ‖ snappy_framed(payload)
//!
//! The framed-snappy encoding is `snap::write::FrameEncoder` /
//! `snap::read::FrameDecoder` (NOT raw snappy — that's gossipsub).

use std::io::{self, Cursor, Read, Write};

use bytes::{BufMut, Bytes, BytesMut};
use libp2p::request_response::{self, Codec, ProtocolSupport};
use libp2p::StreamProtocol;
use pyo3::prelude::*;
use unsigned_varint::{decode as varint_decode, encode as varint_encode};

const MAX_RPC_PAYLOAD: usize = 10 * 1024 * 1024;

// ============================================================================
// Eth2 wire framing
// ============================================================================

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

pub fn decode_eth2_rpc(input: &[u8]) -> io::Result<Vec<u8>> {
    let (expected_len, rest) = varint_decode::usize(input)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("varint: {e}")))?;

    if expected_len > MAX_RPC_PAYLOAD {
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

// ============================================================================
// Status v2 (92 bytes)
// ============================================================================

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

// ============================================================================
// Ping v1 (8 bytes — uint64 sequence number)
// ============================================================================

#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PingMessage {
    #[pyo3(get, set)]
    pub seq_number: u64,
}

#[pymethods]
impl PingMessage {
    #[new]
    pub fn new(seq_number: u64) -> Self {
        Self { seq_number }
    }
    pub fn __repr__(&self) -> String {
        format!("PingMessage(seq_number={})", self.seq_number)
    }
}

impl PingMessage {
    pub const SSZ_LEN: usize = 8;
    pub fn encode_ssz(&self) -> Vec<u8> {
        self.seq_number.to_le_bytes().to_vec()
    }
    pub fn decode_ssz(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != Self::SSZ_LEN {
            return Err(format!("ping expects 8 bytes, got {}", bytes.len()));
        }
        Ok(Self {
            seq_number: u64::from_le_bytes(bytes.try_into().unwrap()),
        })
    }
}

// ============================================================================
// Goodbye v1 (8 bytes — uint64 reason)
// ============================================================================

#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GoodbyeMessage {
    #[pyo3(get, set)]
    pub reason: u64,
}

#[pymethods]
impl GoodbyeMessage {
    #[new]
    pub fn new(reason: u64) -> Self {
        Self { reason }
    }
    pub fn __repr__(&self) -> String {
        format!("GoodbyeMessage(reason={})", self.reason)
    }
}

impl GoodbyeMessage {
    pub const SSZ_LEN: usize = 8;
    pub fn encode_ssz(&self) -> Vec<u8> {
        self.reason.to_le_bytes().to_vec()
    }
    pub fn decode_ssz(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != Self::SSZ_LEN {
            return Err(format!("goodbye expects 8 bytes, got {}", bytes.len()));
        }
        Ok(Self {
            reason: u64::from_le_bytes(bytes.try_into().unwrap()),
        })
    }
}

// ============================================================================
// MetaData v3 (Gloas / PeerDAS): seq_number(8) + attnets(8) + syncnets(1) + custody_group_count(8) = 25 bytes
// ============================================================================

#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetaDataMessage {
    #[pyo3(get, set)]
    pub seq_number: u64,
    #[pyo3(get, set)]
    pub attnets: Vec<u8>, // 8 bytes
    #[pyo3(get, set)]
    pub syncnets: u8, // 1 byte
    #[pyo3(get, set)]
    pub custody_group_count: u64, // 8 bytes (v3 only)
}

#[pymethods]
impl MetaDataMessage {
    #[new]
    #[pyo3(signature = (seq_number=0, attnets=vec![0u8;8], syncnets=0u8, custody_group_count=0u64))]
    pub fn new(
        seq_number: u64,
        attnets: Vec<u8>,
        syncnets: u8,
        custody_group_count: u64,
    ) -> Self {
        let mut attnets = attnets;
        attnets.resize(8, 0);
        Self {
            seq_number,
            attnets,
            syncnets,
            custody_group_count,
        }
    }
    pub fn __repr__(&self) -> String {
        format!(
            "MetaDataMessage(seq_number={}, attnets=0x{}, syncnets=0x{:02x}, custody_group_count={})",
            self.seq_number,
            hex::encode(&self.attnets),
            self.syncnets,
            self.custody_group_count
        )
    }
}

impl MetaDataMessage {
    pub const SSZ_LEN_V2: usize = 8 + 8 + 1; // 17
    pub const SSZ_LEN_V3: usize = 8 + 8 + 1 + 8; // 25

    pub fn encode_ssz_v3(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(Self::SSZ_LEN_V3);
        out.extend_from_slice(&self.seq_number.to_le_bytes());
        out.extend_from_slice(&self.attnets);
        out.push(self.syncnets);
        out.extend_from_slice(&self.custody_group_count.to_le_bytes());
        out
    }
    pub fn encode_ssz_v2(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(Self::SSZ_LEN_V2);
        out.extend_from_slice(&self.seq_number.to_le_bytes());
        out.extend_from_slice(&self.attnets);
        out.push(self.syncnets);
        out
    }
    pub fn decode_ssz(bytes: &[u8]) -> Result<Self, String> {
        match bytes.len() {
            Self::SSZ_LEN_V3 => Ok(Self {
                seq_number: u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
                attnets: bytes[8..16].to_vec(),
                syncnets: bytes[16],
                custody_group_count: u64::from_le_bytes(bytes[17..25].try_into().unwrap()),
            }),
            Self::SSZ_LEN_V2 => Ok(Self {
                seq_number: u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
                attnets: bytes[8..16].to_vec(),
                syncnets: bytes[16],
                custody_group_count: 0,
            }),
            n => Err(format!("metadata expects 17 or 25 bytes, got {n}")),
        }
    }
}

/// Marker request for the empty Metadata body (the wire body is zero-length
/// but we still need a Rust value for the codec).
#[derive(Clone, Debug, Default)]
pub struct MetaDataRequest;

// ============================================================================
// Generic codec helpers
// ============================================================================

trait Eth2RpcMsg: Sized + Send + Sync + 'static {
    fn encode(&self) -> Vec<u8>;
    fn decode(bytes: &[u8]) -> io::Result<Self>;
}

impl Eth2RpcMsg for StatusMessage {
    fn encode(&self) -> Vec<u8> {
        self.encode_ssz()
    }
    fn decode(bytes: &[u8]) -> io::Result<Self> {
        Self::decode_ssz(bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}

impl Eth2RpcMsg for PingMessage {
    fn encode(&self) -> Vec<u8> {
        self.encode_ssz()
    }
    fn decode(bytes: &[u8]) -> io::Result<Self> {
        Self::decode_ssz(bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}

impl Eth2RpcMsg for GoodbyeMessage {
    fn encode(&self) -> Vec<u8> {
        self.encode_ssz()
    }
    fn decode(bytes: &[u8]) -> io::Result<Self> {
        Self::decode_ssz(bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}

impl Eth2RpcMsg for MetaDataMessage {
    fn encode(&self) -> Vec<u8> {
        self.encode_ssz_v3()
    }
    fn decode(bytes: &[u8]) -> io::Result<Self> {
        Self::decode_ssz(bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}

impl Eth2RpcMsg for MetaDataRequest {
    fn encode(&self) -> Vec<u8> {
        Vec::new()
    }
    fn decode(bytes: &[u8]) -> io::Result<Self> {
        if !bytes.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("metadata request must be empty, got {} bytes", bytes.len()),
            ));
        }
        Ok(MetaDataRequest)
    }
}

/// Generic Eth2 ReqResp codec parameterized over request/response message types.
#[derive(Clone)]
pub struct Eth2Codec<Req, Resp> {
    _phantom: std::marker::PhantomData<(Req, Resp)>,
}

impl<Req, Resp> Default for Eth2Codec<Req, Resp> {
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

#[async_trait::async_trait]
impl<Req, Resp> Codec for Eth2Codec<Req, Resp>
where
    Req: Eth2RpcMsg + Clone + std::fmt::Debug,
    Resp: Eth2RpcMsg + Clone + std::fmt::Debug,
{
    type Protocol = StreamProtocol;
    type Request = Req;
    type Response = Resp;

    async fn read_request<T>(&mut self, _: &Self::Protocol, io: &mut T) -> io::Result<Req>
    where
        T: futures::AsyncRead + Unpin + Send,
    {
        use futures::AsyncReadExt;
        let mut buf = Vec::new();
        io.take(MAX_RPC_PAYLOAD as u64 + 16).read_to_end(&mut buf).await?;
        // Empty body means a marker request (e.g. Metadata).
        if buf.is_empty() {
            return Req::decode(&[]);
        }
        let payload = decode_eth2_rpc(&buf)?;
        Req::decode(&payload)
    }

    async fn read_response<T>(&mut self, _: &Self::Protocol, io: &mut T) -> io::Result<Resp>
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
        let mut buf = Vec::new();
        io.take(MAX_RPC_PAYLOAD as u64 + 16).read_to_end(&mut buf).await?;
        let payload = decode_eth2_rpc(&buf)?;
        Resp::decode(&payload)
    }

    async fn write_request<T>(
        &mut self,
        _: &Self::Protocol,
        io: &mut T,
        req: Req,
    ) -> io::Result<()>
    where
        T: futures::AsyncWrite + Unpin + Send,
    {
        use futures::AsyncWriteExt;
        let payload = req.encode();
        if payload.is_empty() {
            // Empty body for marker requests (Metadata).
            io.close().await?;
            return Ok(());
        }
        let framed = encode_eth2_rpc(&payload)?;
        io.write_all(&framed).await?;
        io.close().await?;
        Ok(())
    }

    async fn write_response<T>(
        &mut self,
        _: &Self::Protocol,
        io: &mut T,
        resp: Resp,
    ) -> io::Result<()>
    where
        T: futures::AsyncWrite + Unpin + Send,
    {
        use futures::AsyncWriteExt;
        let payload = resp.encode();
        let framed = encode_eth2_rpc(&payload)?;
        io.write_all(&[0u8]).await?;
        io.write_all(&framed).await?;
        io.close().await?;
        Ok(())
    }
}

// ============================================================================
// Behaviour constructors per protocol
// ============================================================================

pub type StatusBehaviour = request_response::Behaviour<Eth2Codec<StatusMessage, StatusMessage>>;
pub type PingBehaviour = request_response::Behaviour<Eth2Codec<PingMessage, PingMessage>>;
pub type GoodbyeBehaviour = request_response::Behaviour<Eth2Codec<GoodbyeMessage, GoodbyeMessage>>;
pub type MetadataBehaviour =
    request_response::Behaviour<Eth2Codec<MetaDataRequest, MetaDataMessage>>;

fn rpc_config() -> request_response::Config {
    request_response::Config::default().with_request_timeout(std::time::Duration::from_secs(10))
}

pub fn new_status_behaviour() -> StatusBehaviour {
    let proto = StreamProtocol::new("/eth2/beacon_chain/req/status/2/ssz_snappy");
    request_response::Behaviour::with_codec(
        Eth2Codec::default(),
        std::iter::once((proto, ProtocolSupport::Full)),
        rpc_config(),
    )
}

pub fn new_ping_behaviour() -> PingBehaviour {
    let proto = StreamProtocol::new("/eth2/beacon_chain/req/ping/1/ssz_snappy");
    request_response::Behaviour::with_codec(
        Eth2Codec::default(),
        std::iter::once((proto, ProtocolSupport::Full)),
        rpc_config(),
    )
}

pub fn new_goodbye_behaviour() -> GoodbyeBehaviour {
    let proto = StreamProtocol::new("/eth2/beacon_chain/req/goodbye/1/ssz_snappy");
    request_response::Behaviour::with_codec(
        Eth2Codec::default(),
        std::iter::once((proto, ProtocolSupport::Full)),
        rpc_config(),
    )
}

pub fn new_metadata_behaviour() -> MetadataBehaviour {
    let proto = StreamProtocol::new("/eth2/beacon_chain/req/metadata/3/ssz_snappy");
    request_response::Behaviour::with_codec(
        Eth2Codec::default(),
        std::iter::once((proto, ProtocolSupport::Full)),
        rpc_config(),
    )
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct StatusEvent {
    #[pyo3(get)]
    pub peer: String,
    #[pyo3(get)]
    pub kind: String,
    #[pyo3(get)]
    pub message: Option<StatusMessage>,
    #[pyo3(get)]
    pub error: Option<String>,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct PingEvent {
    #[pyo3(get)]
    pub peer: String,
    #[pyo3(get)]
    pub kind: String,
    #[pyo3(get)]
    pub message: Option<PingMessage>,
    #[pyo3(get)]
    pub error: Option<String>,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct GoodbyeEvent {
    #[pyo3(get)]
    pub peer: String,
    #[pyo3(get)]
    pub kind: String,
    #[pyo3(get)]
    pub message: Option<GoodbyeMessage>,
    #[pyo3(get)]
    pub error: Option<String>,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct MetadataEvent {
    #[pyo3(get)]
    pub peer: String,
    #[pyo3(get)]
    pub kind: String,
    #[pyo3(get)]
    pub message: Option<MetaDataMessage>,
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
    fn ping_roundtrip() {
        let p = PingMessage::new(99);
        let bytes = p.encode_ssz();
        let decoded = PingMessage::decode_ssz(&bytes).unwrap();
        assert_eq!(decoded, p);
    }

    #[test]
    fn metadata_v3_roundtrip() {
        let m = MetaDataMessage::new(7, vec![0xff; 8], 0x01, 128);
        let bytes = m.encode_ssz_v3();
        assert_eq!(bytes.len(), MetaDataMessage::SSZ_LEN_V3);
        let decoded = MetaDataMessage::decode_ssz(&bytes).unwrap();
        assert_eq!(decoded, m);
    }
}
