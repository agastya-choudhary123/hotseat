//! Pre-copy migration of a live decoding sequence.
//!
//! The algorithm is the one QEMU and CRIU use for virtual machines, applied to
//! a KV cache:
//!
//! ```text
//!   round 0            everything the sequence currently occupies
//!   round 1..n         only what changed while the previous round was in flight
//!   stop-and-copy      park the decode loop, send the residue, hand over
//! ```
//!
//! Rounds stop when a round gets small enough to afford, when the round budget
//! runs out, or when a round stops shrinking — the last of which is the
//! interesting failure mode: if the sequence dirties pages faster than the link
//! can move them, iterating forever makes the handover worse, not better, and
//! the right answer is to stop early and eat a longer pause.
//!
//! Two properties of a KV cache make this converge much better than it does for
//! a general-purpose VM, and the round table in a migration report shows both:
//! the cache is append-mostly, so the dirty set is a small moving tail rather
//! than a working set scattered over the whole allocation; and the tail's size
//! is set by the token rate, which is orders of magnitude below the link rate.

use hs_engine::kv::KvSpec;
use hs_track::{PageSet, Region};
use hs_wire::{runs_to_byte_runs, total_bytes, Conn, FinalState, Hello, Msg, PROTO};
use std::io;
use std::sync::Arc;
use std::time::Instant;

use crate::seq::{decode_loop, new_sampler, Seq};
use crate::worker::Worker;

#[derive(Debug, Clone, Copy)]
pub struct Opts {
    pub max_rounds: u32,
    /// Stop iterating once a round is this small; the residue is cheap enough
    /// to send with the sequence stopped.
    pub target_bytes: u64,
    /// Have the destination re-hash the live cache before acknowledging.
    /// Costs real time inside the stop-the-world window, so it is off unless
    /// asked for; the report prints the window both ways.
    pub verify: bool,
    /// Test affordance: hand over a *re-seeded* sampler instead of the live
    /// one. The transfer succeeds, the KV cache matches, and the transcript
    /// diverges anyway — which is how the determinism test proves it is
    /// capable of failing. Never useful in production; the report says so.
    pub drop_rng: bool,
}

impl Default for Opts {
    fn default() -> Self {
        Opts { max_rounds: 8, target_bytes: 1 << 20, verify: false, drop_rng: false }
    }
}

#[derive(Debug, Clone)]
pub struct RoundStat {
    pub index: u32,
    pub bytes: u64,
    pub runs: usize,
    pub ms: f64,
    /// Tokens the sequence produced while this round was in flight.
    pub tokens_during: usize,
}

#[derive(Debug, Clone, Default)]
pub struct Report {
    pub target: String,
    pub rounds: Vec<RoundStat>,
    pub precopy_bytes: u64,
    pub precopy_ms: f64,
    /// Time between asking the decode loop to stop and it actually stopping —
    /// i.e. the remainder of the token that was in flight.
    pub park_us: f64,
    /// The headline: decode stopped on the source until the destination said it
    /// was ready.
    pub stw_us: f64,
    /// The part of that window spent sending the residual pages.
    pub stw_send_us: f64,
    /// The part the destination spent verifying and starting up.
    pub dest_resume_us: f64,
    pub final_bytes: u64,
    pub live_bytes: u64,
    pub filled: usize,
    pub tokens_total: usize,
    pub kv_hash_src: u64,
    pub kv_hash_dst: u64,
    pub verified: bool,
    pub tracker: String,
    pub faults: u64,
    pub fault_us_avg: f64,
}

impl Report {
    pub fn render(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "migrated sequence to {} | {} positions in cache, {} tokens total\n",
            self.target, self.filled, self.tokens_total
        ));
        s.push_str(&format!(
            "tracker {} | {} faults, {:.2} us each\n",
            self.tracker, self.faults, self.fault_us_avg
        ));
        s.push_str("round        bytes     runs       ms   tokens\n");
        for r in &self.rounds {
            s.push_str(&format!(
                "{:>5} {:>12} {:>8} {:>8.2} {:>8}\n",
                r.index,
                fmt_bytes(r.bytes),
                r.runs,
                r.ms,
                r.tokens_during
            ));
        }
        s.push_str(&format!(
            "pre-copy {} in {:.1} ms over {} rounds ({:.0} MiB/s)\n",
            fmt_bytes(self.precopy_bytes),
            self.precopy_ms,
            self.rounds.len(),
            self.precopy_bytes as f64 / (1 << 20) as f64 / (self.precopy_ms / 1000.0).max(1e-9)
        ));
        s.push_str(&format!(
            "stop-the-world {:.1} us  (park {:.1} us, send {} in {:.1} us, destination {:.1} us)\n",
            self.stw_us,
            self.park_us,
            fmt_bytes(self.final_bytes),
            self.stw_send_us,
            self.dest_resume_us
        ));
        s.push_str(&format!(
            "live cache {} | kv hash src {:#018x} dst {:#018x}{}\n",
            fmt_bytes(self.live_bytes),
            self.kv_hash_src,
            self.kv_hash_dst,
            if self.verified { " (verified on destination)" } else { " (source-side only)" }
        ));
        s
    }
}

pub fn fmt_bytes(b: u64) -> String {
    const U: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut v = b as f64;
    let mut i = 0;
    while v >= 1024.0 && i < 3 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{b} B")
    } else {
        format!("{v:.2} {}", U[i])
    }
}

/// Whole pages covering `live`, as byte runs.
fn live_page_runs(spec: &KvSpec, filled: usize, region: &Region) -> Vec<(u64, u64)> {
    let live = spec.live_ranges(filled);
    let ps = region.page_size();
    let set = PageSet::new(region.n_pages());
    for &(off, len) in &live {
        if len == 0 {
            continue;
        }
        let first = off / ps;
        let last = (off + len - 1) / ps;
        set.mark_range(first, last - first + 1);
    }
    runs_to_byte_runs(&set, ps, &live)
}

/// Source half.
pub fn migrate_out(
    seq: &Arc<Seq>,
    target: &str,
    token: &str,
    model_name: &str,
    opts: &Opts,
) -> io::Result<Report> {
    let mut conn = Conn::connect(target)?;
    let mut rep = Report { target: target.to_string(), ..Default::default() };

    conn.send(&Msg::Hello(Hello {
        proto: PROTO,
        token: token.to_string(),
        model_fingerprint: seq.model.fingerprint,
        model_name: model_name.to_string(),
        spec: seq.spec,
        page_size: seq.region.page_size() as u64,
        seq_id: seq.id,
    }))?;
    match conn.recv()? {
        Msg::HelloAck { ok: true, .. } => {}
        Msg::HelloAck { reason, .. } => {
            return Err(io::Error::other(format!("destination refused: {reason}")))
        }
        Msg::Error(e) => return Err(io::Error::other(format!("destination error: {e}"))),
        other => return Err(io::Error::other(format!("unexpected reply: {other:?}"))),
    }

    // From here on, every write the model makes is observed.
    seq.tracker.arm()?;
    let precopy_start = Instant::now();

    // ---- round 0: everything the sequence occupies right now --------------
    let tokens_at = || seq.state.lock().unwrap().tokens.len();
    let mut t_mark = tokens_at();
    let filled0 = seq.filled();
    let runs = live_page_runs(&seq.spec, filled0, &seq.region);
    let t0 = Instant::now();
    conn.send(&Msg::Pages { round: 0, runs: runs.clone() })?;
    let sent = conn.send_page_bytes(&seq.region, &runs)?;
    let now_tokens = tokens_at();
    rep.rounds.push(RoundStat {
        index: 0,
        bytes: sent,
        runs: runs.len(),
        ms: t0.elapsed().as_secs_f64() * 1e3,
        tokens_during: now_tokens - t_mark,
    });
    t_mark = now_tokens;
    rep.precopy_bytes += sent;

    // ---- iterative rounds --------------------------------------------------
    let dirty = PageSet::new(seq.region.n_pages());
    let mut prev = u64::MAX;
    for round in 1..=opts.max_rounds {
        dirty.clear();
        seq.tracker.harvest(&dirty)?;
        let filled = seq.filled();
        let runs = runs_to_byte_runs(&dirty, seq.region.page_size(), &seq.spec.live_ranges(filled));
        let bytes = total_bytes(&runs);
        if bytes == 0 {
            break;
        }
        let t0 = Instant::now();
        conn.send(&Msg::Pages { round, runs: runs.clone() })?;
        let sent = conn.send_page_bytes(&seq.region, &runs)?;
        let now_tokens = tokens_at();
        rep.rounds.push(RoundStat {
            index: round,
            bytes: sent,
            runs: runs.len(),
            ms: t0.elapsed().as_secs_f64() * 1e3,
            tokens_during: now_tokens - t_mark,
        });
        t_mark = now_tokens;
        rep.precopy_bytes += sent;

        if bytes <= opts.target_bytes {
            break; // small enough to finish with the sequence stopped
        }
        if bytes >= prev {
            // The sequence is dirtying pages at least as fast as we can ship
            // them. More rounds would only delay the inevitable pause.
            break;
        }
        prev = bytes;
    }
    rep.precopy_ms = precopy_start.elapsed().as_secs_f64() * 1e3;

    // ---- stop and copy -----------------------------------------------------
    let t_req = Instant::now();
    seq.pause();
    let t_parked = Instant::now();
    rep.park_us = (t_parked - t_req).as_secs_f64() * 1e6;
    assert!(
        seq.state.lock().unwrap().at_token_boundary(),
        "decode loop parked somewhere a handover cannot describe"
    );

    dirty.clear();
    seq.tracker.harvest(&dirty)?;
    let (tokens, prompt_len, filled, rng, cfg, max_tokens) = {
        let st = seq.state.lock().unwrap();
        (
            st.tokens.clone(),
            st.prompt_len,
            st.filled,
            st.sampler.rng,
            st.sampler.cfg,
            st.max_tokens,
        )
    };
    let live = seq.spec.live_ranges(filled);
    let runs = runs_to_byte_runs(&dirty, seq.region.page_size(), &live);
    rep.final_bytes = total_bytes(&runs);
    rep.live_bytes = live.iter().map(|r| r.1 as u64).sum();
    rep.filled = filled;
    rep.tokens_total = tokens.len();

    let kv = seq.kv();
    rep.kv_hash_src = kv.hash(filled);

    let t_send = Instant::now();
    conn.send(&Msg::Pages { round: u32::MAX, runs: runs.clone() })?;
    conn.send_page_bytes(&seq.region, &runs)?;
    conn.send(&Msg::Final(FinalState {
        filled: filled as u32,
        tokens,
        prompt_len: prompt_len as u32,
        rng: if opts.drop_rng { hs_engine::sampler::Rng::seed(cfg.seed ^ 0xDEAD) } else { rng },
        cfg,
        max_tokens: max_tokens as u32,
        kv_hash: if opts.verify { rep.kv_hash_src } else { 0 },
        precopy_ns: (rep.precopy_ms * 1e6) as u64,
        precopy_bytes: rep.precopy_bytes,
        rounds: rep.rounds.len() as u32,
    }))?;
    rep.stw_send_us = t_send.elapsed().as_secs_f64() * 1e6;

    let ack = match conn.recv()? {
        Msg::ResumeAck(a) => a,
        Msg::Error(e) => {
            seq.resume(); // the destination did not take it; keep decoding here
            return Err(io::Error::other(format!("destination error: {e}")));
        }
        other => {
            seq.resume();
            return Err(io::Error::other(format!("unexpected reply: {other:?}")));
        }
    };
    rep.stw_us = t_parked.elapsed().as_secs_f64() * 1e6;
    rep.dest_resume_us = ack.resume_ns as f64 / 1e3;
    rep.kv_hash_dst = ack.kv_hash;
    rep.verified = opts.verify;

    if !ack.ok {
        seq.resume();
        return Err(io::Error::other(format!("destination refused handover: {}", ack.reason)));
    }
    if opts.verify && ack.kv_hash != rep.kv_hash_src {
        seq.resume();
        return Err(io::Error::other(format!(
            "KV cache differs after transfer: source {:#018x}, destination {:#018x}",
            rep.kv_hash_src, ack.kv_hash
        )));
    }

    seq.hand_off();
    let _ = conn.send(&Msg::Bye);

    let st = seq.tracker.stats();
    seq.tracker.disarm()?;
    rep.tracker = seq.tracker.name().to_string();
    rep.faults = st.faults;
    rep.fault_us_avg = st.ns_per_fault() / 1e3;
    Ok(rep)
}

/// Destination half: take a sequence from a peer and start running it.
pub fn migrate_in(worker: &Arc<Worker>, mut conn: Conn) -> io::Result<()> {
    let hello = match conn.recv()? {
        Msg::Hello(h) => h,
        other => {
            let _ = conn.send(&Msg::Error(format!("expected Hello, got {other:?}")));
            return Err(io::Error::other("bad handshake"));
        }
    };

    // Every one of these is a way to end up with fluent, wrong output rather
    // than an error, so none of them is taken on trust.
    let refuse = |c: &mut Conn, why: String| -> io::Result<()> {
        c.send(&Msg::HelloAck { ok: false, reason: why.clone() })?;
        Err(io::Error::other(why))
    };
    if hello.proto != PROTO {
        return refuse(&mut conn, format!("protocol {} != {PROTO}", hello.proto));
    }
    if !worker.check_token(&hello.token) {
        return refuse(&mut conn, "bad token".into());
    }
    if hello.model_fingerprint != worker.model.fingerprint {
        return refuse(
            &mut conn,
            format!(
                "model fingerprint {:#018x} != local {:#018x} ({} vs {})",
                hello.model_fingerprint, worker.model.fingerprint, hello.model_name, worker.model_name
            ),
        );
    }
    let want = worker.model.cfg.kv_spec(worker.max_ctx, worker.layout);
    if hello.spec != want {
        return refuse(&mut conn, format!("KV spec mismatch: source {}, local {}", hello.spec, want));
    }
    {
        let slot = worker.slot.lock().unwrap();
        if let Some(old) = slot.as_ref() {
            let st = old.state.lock().unwrap();
            if !st.done && !st.migrated_away {
                return refuse(&mut conn, format!("sequence {} is still running here", old.id));
            }
        }
    }

    let region = Arc::new(Region::new(hello.spec.bytes())?);
    region.prefault();
    let tracker = hs_track::open(region.clone(), worker.tracker_kind, worker.cluster)?;
    conn.send(&Msg::HelloAck { ok: true, reason: String::new() })?;

    // Page rounds land straight in the destination cache.
    let mut received = 0u64;
    let fin = loop {
        match conn.recv()? {
            Msg::Pages { runs, .. } => {
                received += conn.recv_page_bytes(&region, &runs)?;
            }
            Msg::Final(f) => break f,
            Msg::Error(e) => return Err(io::Error::other(format!("source error: {e}"))),
            other => return Err(io::Error::other(format!("unexpected message: {other:?}"))),
        }
    };
    let t_resume = Instant::now();

    let seq = {
        // SAFETY: nothing else references this region yet.
        let kv = unsafe { hs_engine::kv::Kv::from_raw(region.as_ptr(), hello.spec) };
        let our_hash = if fin.kv_hash != 0 { kv.hash(fin.filled as usize) } else { 0 };
        if fin.kv_hash != 0 && our_hash != fin.kv_hash {
            conn.send(&Msg::ResumeAck(hs_wire::ResumeAck {
                ok: false,
                kv_hash: our_hash,
                resume_ns: t_resume.elapsed().as_nanos() as u64,
                reason: "KV hash mismatch".into(),
            }))?;
            return Err(io::Error::other("KV hash mismatch"));
        }

        let mut sampler = new_sampler(fin.cfg);
        sampler.rng = fin.rng; // the whole reason the transcript continues
        let seq = Seq::new(
            hello.seq_id,
            worker.name.clone(),
            worker.model.clone(),
            worker.pool.clone(),
            hello.spec,
            region.clone(),
            tracker,
            fin.tokens.clone(),
            fin.prompt_len as usize,
            fin.filled as usize,
            sampler,
            fin.max_tokens as usize,
        );
        worker.take_slot(seq.clone()).map_err(io::Error::other)?;
        let s = seq.clone();
        std::thread::Builder::new()
            .name(format!("hs-decode-{}", seq.id))
            .spawn(move || decode_loop(s))?;
        conn.send(&Msg::ResumeAck(hs_wire::ResumeAck {
            ok: true,
            kv_hash: our_hash,
            resume_ns: t_resume.elapsed().as_nanos() as u64,
            reason: String::new(),
        }))?;
        seq
    };

    eprintln!(
        "[{}] took over sequence {} at position {} ({} tokens), {} received",
        worker.name,
        seq.id,
        fin.filled,
        fin.tokens.len(),
        fmt_bytes(received)
    );
    Ok(())
}
