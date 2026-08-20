//! The migration protocol: what one worker says to another to hand over a
//! decoding sequence.
//!
//! Framing is a 16-byte header and a payload. Page data is deliberately *not*
//! wrapped in a serialised structure — the run descriptors go out first and the
//! bytes follow them straight from the tracked region, so a round costs no copy
//! on the sending side and lands directly in the destination's KV cache on the
//! receiving side. That matters: the stop-and-copy phase is the number this
//! whole project is measured by, and a memcpy of the residual set would be
//! part of it.

use hs_engine::kv::{KvLayout, KvSpec};
use hs_engine::sampler::{Rng, SamplerCfg};
use hs_track::{PageSet, Region};
use std::io::{self, Read, Write};
use std::net::TcpStream;

pub mod codec;
use codec::{Reader, Writer};

pub const MAGIC: u32 = 0x3148_534D; // "MSH1"
pub const PROTO: u32 = 1;
pub const HDR_LEN: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    Hello = 1,
    HelloAck = 2,
    Pages = 3,
    Final = 4,
    ResumeAck = 5,
    Bye = 6,
    Error = 7,
}

impl Kind {
    fn from_u8(v: u8) -> Option<Kind> {
        Some(match v {
            1 => Kind::Hello,
            2 => Kind::HelloAck,
            3 => Kind::Pages,
            4 => Kind::Final,
            5 => Kind::ResumeAck,
            6 => Kind::Bye,
            7 => Kind::Error,
            _ => return None,
        })
    }
}

/// What the source claims about itself and the sequence it wants to hand over.
///
/// The destination checks every field. Resuming a sequence against different
/// weights, a differently shaped cache or a different KV layout would not fail
/// loudly — it would produce fluent, wrong text — so none of it is assumed.
#[derive(Debug, Clone)]
pub struct Hello {
    pub proto: u32,
    pub token: String,
    pub model_fingerprint: u64,
    pub model_name: String,
    pub spec: KvSpec,
    pub page_size: u64,
    pub seq_id: u64,
}

impl Hello {
    fn encode(&self, w: &mut Writer) {
        w.u32(self.proto)
            .str(&self.token)
            .u64(self.model_fingerprint)
            .str(&self.model_name)
            .u32(self.spec.n_layers as u32)
            .u32(self.spec.n_kv_heads as u32)
            .u32(self.spec.head_dim as u32)
            .u32(self.spec.max_ctx as u32)
            .u8(self.spec.layout.as_u8())
            .u64(self.page_size)
            .u64(self.seq_id);
    }
    fn decode(r: &mut Reader) -> io::Result<Hello> {
        let e = |_| io::Error::new(io::ErrorKind::InvalidData, "malformed Hello");
        Ok(Hello {
            proto: r.u32().map_err(e)?,
            token: r.str().map_err(e)?,
            model_fingerprint: r.u64().map_err(e)?,
            model_name: r.str().map_err(e)?,
            spec: KvSpec {
                n_layers: r.u32().map_err(e)? as usize,
                n_kv_heads: r.u32().map_err(e)? as usize,
                head_dim: r.u32().map_err(e)? as usize,
                max_ctx: r.u32().map_err(e)? as usize,
                layout: KvLayout::from_u8(r.u8().map_err(e)?)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bad KV layout"))?,
            },
            page_size: r.u64().map_err(e)?,
            seq_id: r.u64().map_err(e)?,
        })
    }
}

/// Everything about the sequence that is not the KV cache.
///
/// It is small — a few kilobytes — and it is the part that must be exactly
/// right. The RNG state in particular: drop it and the destination produces
/// perfectly reasonable text that is not the text the source would have
/// produced.
#[derive(Debug, Clone)]
pub struct FinalState {
    /// Positions already present in the KV cache.
    pub filled: u32,
    /// Every token so far. `tokens[filled..]` still has to be fed.
    pub tokens: Vec<u32>,
    /// Index in `tokens` where generation began; everything before it is prompt.
    pub prompt_len: u32,
    pub rng: Rng,
    pub cfg: SamplerCfg,
    pub max_tokens: u32,
    /// Hash of the live KV bytes, as the source sees them.
    pub kv_hash: u64,
    /// Wall-clock nanoseconds the source spent in each migration phase.
    pub precopy_ns: u64,
    pub precopy_bytes: u64,
    pub rounds: u32,
}

impl FinalState {
    fn encode(&self, w: &mut Writer) {
        w.u32(self.filled)
            .u32s(&self.tokens)
            .u32(self.prompt_len)
            .u64(self.rng.s[0])
            .u64(self.rng.s[1])
            .u64(self.rng.s[2])
            .u64(self.rng.s[3])
            .u64(self.rng.draws)
            .f32(self.cfg.temperature)
            .f32(self.cfg.top_p)
            .u32(self.cfg.top_k)
            .f32(self.cfg.repeat_penalty)
            .u32(self.cfg.repeat_last_n)
            .u64(self.cfg.seed)
            .u32(self.max_tokens)
            .u64(self.kv_hash)
            .u64(self.precopy_ns)
            .u64(self.precopy_bytes)
            .u32(self.rounds);
    }
    fn decode(r: &mut Reader) -> io::Result<FinalState> {
        let e = |_| io::Error::new(io::ErrorKind::InvalidData, "malformed Final");
        Ok(FinalState {
            filled: r.u32().map_err(e)?,
            tokens: r.u32s().map_err(e)?,
            prompt_len: r.u32().map_err(e)?,
            rng: Rng {
                s: [
                    r.u64().map_err(e)?,
                    r.u64().map_err(e)?,
                    r.u64().map_err(e)?,
                    r.u64().map_err(e)?,
                ],
                draws: r.u64().map_err(e)?,
            },
            cfg: SamplerCfg {
                temperature: r.f32().map_err(e)?,
                top_p: r.f32().map_err(e)?,
                top_k: r.u32().map_err(e)?,
                repeat_penalty: r.f32().map_err(e)?,
                repeat_last_n: r.u32().map_err(e)?,
                seed: r.u64().map_err(e)?,
            },
            max_tokens: r.u32().map_err(e)?,
            kv_hash: r.u64().map_err(e)?,
            precopy_ns: r.u64().map_err(e)?,
            precopy_bytes: r.u64().map_err(e)?,
            rounds: r.u32().map_err(e)?,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ResumeAck {
    pub ok: bool,
    /// The destination's own hash of the cache it now holds. If this does not
    /// match the source's, the two caches differ and the handover is refused.
    pub kv_hash: u64,
    pub resume_ns: u64,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub enum Msg {
    Hello(Hello),
    HelloAck { ok: bool, reason: String },
    /// Header only; the caller reads or writes the page bytes separately.
    Pages { round: u32, runs: Vec<(u64, u64)> },
    Final(FinalState),
    ResumeAck(ResumeAck),
    Bye,
    Error(String),
}

/// A framed connection to the other worker.
pub struct Conn {
    pub sock: TcpStream,
    pub peer: String,
}

impl Conn {
    pub fn new(sock: TcpStream) -> io::Result<Conn> {
        // Migration is a sequence of small control messages interleaved with
        // large writes, and every control message is on the critical path of
        // the stop-the-world window. Nagle would add up to 40 ms to it.
        sock.set_nodelay(true)?;
        let peer = sock.peer_addr().map(|a| a.to_string()).unwrap_or_else(|_| "?".into());
        Ok(Conn { sock, peer })
    }

    pub fn connect(addr: &str) -> io::Result<Conn> {
        Conn::new(TcpStream::connect(addr)?)
    }

    fn write_header(&mut self, kind: Kind, len: u64) -> io::Result<()> {
        let mut h = [0u8; HDR_LEN];
        h[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        h[4] = kind as u8;
        h[8..16].copy_from_slice(&len.to_le_bytes());
        self.sock.write_all(&h)
    }

    pub fn send(&mut self, msg: &Msg) -> io::Result<()> {
        let mut w = Writer::new();
        let kind = match msg {
            Msg::Hello(h) => {
                h.encode(&mut w);
                Kind::Hello
            }
            Msg::HelloAck { ok, reason } => {
                w.u8(*ok as u8).str(reason);
                Kind::HelloAck
            }
            Msg::Pages { round, runs } => {
                w.u32(*round).u32(runs.len() as u32);
                for &(off, len) in runs {
                    w.u64(off).u64(len);
                }
                Kind::Pages
            }
            Msg::Final(f) => {
                f.encode(&mut w);
                Kind::Final
            }
            Msg::ResumeAck(a) => {
                w.u8(a.ok as u8).u64(a.kv_hash).u64(a.resume_ns).str(&a.reason);
                Kind::ResumeAck
            }
            Msg::Bye => Kind::Bye,
            Msg::Error(s) => {
                w.str(s);
                Kind::Error
            }
        };
        self.write_header(kind, w.buf.len() as u64)?;
        if !w.buf.is_empty() {
            self.sock.write_all(&w.buf)?;
        }
        Ok(())
    }

    pub fn recv(&mut self) -> io::Result<Msg> {
        let mut h = [0u8; HDR_LEN];
        self.sock.read_exact(&mut h)?;
        let magic = u32::from_le_bytes(h[0..4].try_into().unwrap());
        if magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("bad frame magic {magic:#x} from {}", self.peer),
            ));
        }
        let kind = Kind::from_u8(h[4])
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("bad kind {}", h[4])))?;
        let len = u64::from_le_bytes(h[8..16].try_into().unwrap()) as usize;
        // A control frame is never large; refusing an absurd length keeps a
        // corrupt or hostile peer from making us allocate a gigabyte.
        if len > 1 << 24 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("frame too large: {len}")));
        }
        let mut body = vec![0u8; len];
        self.sock.read_exact(&mut body)?;
        let mut r = Reader::new(&body);
        let e = |_| io::Error::new(io::ErrorKind::InvalidData, "malformed message");
        Ok(match kind {
            Kind::Hello => Msg::Hello(Hello::decode(&mut r)?),
            Kind::HelloAck => Msg::HelloAck { ok: r.u8().map_err(e)? != 0, reason: r.str().map_err(e)? },
            Kind::Pages => {
                let round = r.u32().map_err(e)?;
                let n = r.u32().map_err(e)? as usize;
                let mut runs = Vec::with_capacity(n);
                for _ in 0..n {
                    runs.push((r.u64().map_err(e)?, r.u64().map_err(e)?));
                }
                Msg::Pages { round, runs }
            }
            Kind::Final => Msg::Final(FinalState::decode(&mut r)?),
            Kind::ResumeAck => Msg::ResumeAck(ResumeAck {
                ok: r.u8().map_err(e)? != 0,
                kv_hash: r.u64().map_err(e)?,
                resume_ns: r.u64().map_err(e)?,
                reason: r.str().map_err(e)?,
            }),
            Kind::Bye => Msg::Bye,
            Kind::Error => Msg::Error(r.str().map_err(e)?),
        })
    }

    /// Send the bytes of `runs` straight out of `region`, no intermediate copy.
    pub fn send_page_bytes(&mut self, region: &Region, runs: &[(u64, u64)]) -> io::Result<u64> {
        let mut total = 0u64;
        for &(off, len) in runs {
            // SAFETY: the decode loop is either paused or writing elsewhere;
            // in a pre-copy round a concurrent write to these bytes is exactly
            // what the tracker will report, and the round after this one
            // re-sends them.
            let bytes = unsafe { region.bytes(off as usize, len as usize) };
            self.sock.write_all(bytes)?;
            total += len;
        }
        Ok(total)
    }

    /// Read page bytes directly into `region`.
    pub fn recv_page_bytes(&mut self, region: &Region, runs: &[(u64, u64)]) -> io::Result<u64> {
        let mut total = 0u64;
        for &(off, len) in runs {
            let end = off.checked_add(len).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "page run overflows")
            })?;
            if end > region.len() as u64 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("page run {off}..{end} is outside the {}-byte cache", region.len()),
                ));
            }
            // SAFETY: bounds checked above; this connection is the only writer
            // of the destination cache until the sequence resumes.
            let bytes = unsafe { region.bytes_mut(off as usize, len as usize) };
            self.sock.read_exact(bytes)?;
            total += len;
        }
        Ok(total)
    }
}

/// Turn a set of dirty pages into byte runs, clipped to the part of the cache
/// that actually holds data.
///
/// Clipping matters more than it looks. A KV cache is allocated for `max_ctx`
/// positions and a sequence normally occupies a small prefix of it, so the
/// dirty set and the *live* set are very different things — sending the union
/// would move a hundred megabytes to hand over a sequence that is using four.
pub fn runs_to_byte_runs(set: &PageSet, page_size: usize, live: &[(usize, usize)]) -> Vec<(u64, u64)> {
    let mut out: Vec<(u64, u64)> = Vec::new();
    for (first, count) in set.runs() {
        let (a, b) = (first * page_size, (first + count) * page_size);
        for &(lo, llen) in live {
            let (c, d) = (a.max(lo), b.min(lo + llen));
            if c < d {
                // Whole pages, so the destination's cache stays byte-identical
                // to the source's over everything the model will read.
                let (c, d) = (c / page_size * page_size, d.div_ceil(page_size) * page_size);
                push_merged(&mut out, c as u64, (d - c) as u64);
            }
        }
    }
    out.sort_by_key(|r| r.0);
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(out.len());
    for (off, len) in out {
        push_merged(&mut merged, off, len);
    }
    merged
}

fn push_merged(v: &mut Vec<(u64, u64)>, off: u64, len: u64) {
    if let Some(last) = v.last_mut() {
        if last.0 + last.1 >= off {
            let end = (last.0 + last.1).max(off + len);
            last.1 = end - last.0;
            return;
        }
    }
    v.push((off, len));
}

pub fn total_bytes(runs: &[(u64, u64)]) -> u64 {
    runs.iter().map(|r| r.1).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_runs_are_clipped_to_the_live_prefix() {
        let ps = 4096;
        let set = PageSet::new(100);
        set.mark_range(0, 100);
        // Live: only the first 10000 bytes.
        let runs = runs_to_byte_runs(&set, ps, &[(0, 10_000)]);
        assert_eq!(runs, vec![(0, 12_288)], "rounded out to whole pages");
        assert_eq!(total_bytes(&runs), 12_288);
    }

    #[test]
    fn byte_runs_merge_adjacent_pages() {
        let ps = 4096;
        let set = PageSet::new(100);
        set.mark_range(3, 4);
        set.mark_range(7, 1);
        let runs = runs_to_byte_runs(&set, ps, &[(0, 100 * ps)]);
        assert_eq!(runs, vec![(3 * 4096, 5 * 4096)], "pages 3..8 are one run");
    }

    #[test]
    fn byte_runs_handle_several_live_stripes() {
        // What head-major layout looks like: the live set is strided.
        let ps = 4096;
        let set = PageSet::new(64);
        set.mark_range(0, 64);
        let live = vec![(0usize, 8192usize), (40 * ps, 8192)];
        let runs = runs_to_byte_runs(&set, ps, &live);
        assert_eq!(runs, vec![(0, 8192), (40 * 4096, 8192)]);
    }

    #[test]
    fn a_clean_set_sends_nothing() {
        let set = PageSet::new(64);
        assert!(runs_to_byte_runs(&set, 4096, &[(0, 1 << 20)]).is_empty());
    }
}
