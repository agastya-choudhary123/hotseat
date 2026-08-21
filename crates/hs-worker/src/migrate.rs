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
use hs_engine::sampler::SamplerCfg;

/// Releases a pre-spawned decode thread if the handover does not complete.
struct AbandonOnDrop(Option<Arc<Seq>>);

impl Drop for AbandonOnDrop {
    fn drop(&mut self) {
        if let Some(s) = self.0.take() {
            s.abandon();
        }
    }
}
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
    /// Shape the link for bulk page data, in megabits per second. Zero means
    /// unlimited.
    pub mbps: f64,
    /// Test affordance: hand over a *re-seeded* sampler instead of the live
    /// one. The transfer succeeds, the KV cache matches, and the transcript
    /// diverges anyway — which is how the determinism test proves it is
    /// capable of failing. Never useful in production; the report says so.
    pub drop_rng: bool,
}

impl Default for Opts {
    fn default() -> Self {
        Opts { max_rounds: 8, target_bytes: 1 << 20, verify: false, mbps: 0.0, drop_rng: false }
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
    /// False when the rounds were stopped because they were not shrinking.
    pub converged: bool,
    pub stop_reason: String,
    pub mbps: f64,
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
            "pre-copy {} in {:.1} ms over {} rounds ({:.0} MiB/s{})\n",
            fmt_bytes(self.precopy_bytes),
            self.precopy_ms,
            self.rounds.len(),
            self.precopy_bytes as f64 / (1 << 20) as f64 / (self.precopy_ms / 1000.0).max(1e-9),
            if self.mbps > 0.0 { format!(", link shaped to {:.0} Mbit/s", self.mbps) } else { String::new() }
        ));
        s.push_str(&format!(
            "rounds stopped: {} [{}]\n",
            if self.stop_reason.is_empty() { "converged" } else { &self.stop_reason },
            if self.converged { "converged" } else { "DID NOT CONVERGE" }
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

/// Hash each live range separately, for finding *where* two caches differ.
/// Enabled by `HS_DEBUG_RANGES`; both ends print, and the two lists are diffed.
fn debug_ranges(who: &str, region: &Region, live: &[(usize, usize)]) {
    if std::env::var_os("HS_DEBUG_RANGES").is_none() {
        return;
    }
    // Plane 0 only, one line per position, so the first divergent position is
    // obvious rather than "all 48 ranges differ".
    let (off, len) = live[0];
    let per = 512usize.min(len.max(1));
    let n = len / per;
    for p in n.saturating_sub(24)..n {
        let b = unsafe { region.bytes(off + p * per, per) };
        eprintln!("pos {p} hash {:#018x} [{who}]", hs_engine::hash::hash_wide(b));
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

/// How far to clip a *pre-copy* round: one position past what the sequence has
/// committed.
///
/// The decode loop writes all of position p's K and V and only then increments
/// `filled`, so at any moment a position past `filled` may be partly written.
/// Those bytes must be sent even though nothing reads them yet, because the
/// alternative is to lose them: writes made before the tracker was armed are
/// invisible to it, and the model writes each (layer, position) slot exactly
/// once, so a slot skipped here is never offered again.
///
/// Sending a partly-written position is harmless — its remaining layers fault
/// after the arm and arrive in a later round, and nothing reads past `filled`.
fn precopy_clip(filled: usize, max_ctx: usize) -> usize {
    (filled + 1).min(max_ctx)
}

/// Everything in positions `from..to`, as merged byte runs.
fn live_page_runs(spec: &KvSpec, from: usize, to: usize, region: &Region) -> Vec<(u64, u64)> {
    let live = spec.live_ranges_between(from, to);
    let ps = region.page_size();
    // Every page holding a live byte, then clipped straight back to the live
    // bytes: the round-trip through the page set is what merges runs that the
    // layout leaves adjacent.
    let set = PageSet::new(region.n_pages());
    for &(off, len) in &live {
        if len == 0 {
            continue;
        }
        set.mark_range(off / ps, (off + len - 1) / ps - off / ps + 1);
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
    conn.set_rate_mbps(opts.mbps);
    let mut rep = Report {
        target: target.to_string(),
        mbps: opts.mbps,
        converged: true,
        ..Default::default()
    };

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
    let mut clip = precopy_clip(filled0, seq.spec.max_ctx);
    let runs = live_page_runs(&seq.spec, 0, clip, &seq.region);
    // Positions below this have been handed over in full and, because the cache
    // is append-only, cannot change again. Every later round starts here.
    // Without it each round re-sends the whole of every dirty page, which for a
    // 16 KiB page holding 32 positions is most of a round's bytes.
    let mut low = clip.saturating_sub(1);
    if std::env::var_os("HS_DEBUG_RANGES").is_some() {
        eprintln!("round 0 filled={filled0} runs[0..4]={:?} total={}", &runs[..4.min(runs.len())], total_bytes(&runs));
    }
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
    //
    // `dirty` is cleared only after a round has actually been *sent*. A round
    // that is harvested and then abandoned -- which is what the convergence
    // check does -- must leave its pages in the set, or they are lost: the
    // harvest re-armed them, so nothing will report them again. `harvest` ORs
    // into the set, so carrying one forward is just not clearing it.
    let dirty = PageSet::new(seq.region.n_pages());
    let mut prev = u64::MAX;
    for round in 1..=opts.max_rounds {
        seq.tracker.harvest(&dirty)?;

        let filled = seq.filled();
        clip = precopy_clip(filled, seq.spec.max_ctx);
        let runs = runs_to_byte_runs(
            &dirty,
            seq.region.page_size(),
            &seq.spec.live_ranges_between(low, clip),
        );
        let bytes = total_bytes(&runs);
        if bytes == 0 {
            rep.stop_reason = "nothing left to send".into();
            break;
        }
        // Decide *before* spending the round. If the sequence is dirtying pages
        // at least as fast as the link ships them, another round costs its
        // whole duration and leaves at least as much to do at the end of it;
        // the residue only grows. Stop and take the pause.
        if bytes >= prev {
            rep.converged = false;
            rep.stop_reason =
                format!("round {round} would send {} after {}, no smaller -- link cannot outrun the sequence",
                    fmt_bytes(bytes), fmt_bytes(prev));
            break;
        }
        let t0 = Instant::now();
        conn.send(&Msg::Pages { round, runs: runs.clone() })?;
        let sent = conn.send_page_bytes(&seq.region, &runs)?;
        dirty.clear(); // these pages are now on the far side
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
        low = clip.saturating_sub(1);

        if bytes <= opts.target_bytes {
            rep.stop_reason = format!("round fits in the {} budget", fmt_bytes(opts.target_bytes));
            break;
        }
        prev = bytes;
        if round == opts.max_rounds {
            rep.stop_reason = format!("round budget of {} used up", opts.max_rounds);
        }
    }
    rep.precopy_ms = precopy_start.elapsed().as_secs_f64() * 1e3;

    // ---- stop and copy -----------------------------------------------------
    let t_req = Instant::now();
    seq.pause();
    let t_parked = Instant::now();
    rep.park_us = (t_parked - t_req).as_secs_f64() * 1e6;

    // A long pre-copy can outlast the sequence. `pause` returns when the loop
    // parks *or* when it finishes, and a finished sequence has nothing left to
    // feed the destination, so there is nothing to hand over.
    {
        let st = seq.state.lock().unwrap();
        if !st.can_hand_over() {
            let why = if st.done {
                format!("sequence finished ({}) while the pre-copy was running", st.done_reason)
            } else {
                "sequence is no longer in a handover-able state".to_string()
            };
            drop(st);
            let _ = conn.send(&Msg::Error(why.clone()));
            seq.resume();
            let _ = seq.tracker.disarm();
            return Err(io::Error::other(why));
        }
    }

    // Not cleared: this unions with any round that was harvested but abandoned.
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
    // The decode loop is parked at the top of its iteration, so nothing is
    // half-written and `filled` is exact -- no +1 here. The transfer covers
    // only positions at or after the low-water mark; the hash still covers
    // everything, because everything below it was already sent.
    let live = seq.spec.live_ranges_between(low, filled);
    let runs = runs_to_byte_runs(&dirty, seq.region.page_size(), &live);
    rep.final_bytes = total_bytes(&runs);
    rep.live_bytes = seq.spec.live_ranges(filled).iter().map(|r| r.1 as u64).sum();
    rep.filled = filled;
    rep.tokens_total = tokens.len();

    let kv = seq.kv();
    rep.kv_hash_src = kv.hash(filled);
    debug_ranges("src", &seq.region, &live);

    if std::env::var_os("HS_DEBUG_RANGES").is_some() {
        eprintln!("final filled={filled} dirty_pages={} runs[0..4]={:?} total={}",
            dirty.count(), &runs[..4.min(runs.len())], total_bytes(&runs));
    }
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
    // The decode thread and the sequence object are created *before* the last
    // page arrives, and the thread parks on `ready`. Spawning a thread was
    // measured at 48 us on a good day and 943 us on a bad one, and every one of
    // those microseconds used to land inside the stop-the-world window.
    let mut sampler = new_sampler(SamplerCfg::default());
    sampler.rng = hs_engine::sampler::Rng::seed(0);
    let seq = Seq::new(
        hello.seq_id,
        worker.name.clone(),
        worker.model.clone(),
        worker.pool.clone(),
        hello.spec,
        region.clone(),
        tracker,
        Vec::new(),
        0,
        0,
        sampler,
        0,
    );
    seq.state.lock().unwrap().ready = false;
    worker.take_slot(seq.clone()).map_err(io::Error::other)?;
    {
        let s = seq.clone();
        std::thread::Builder::new()
            .name(format!("hs-decode-{}", seq.id))
            .spawn(move || decode_loop(s))?;
    }
    // From here on, any early return must release the parked thread.
    let guard = AbandonOnDrop(Some(seq.clone()));

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

    if fin.filled as usize >= fin.tokens.len() {
        // The source should never offer this; check anyway, because accepting
        // it means a decode thread with nothing to feed it.
        conn.send(&Msg::Error("handover has no token left to feed".into()))?;
        return Err(io::Error::other(format!(
            "refused handover: filled {} but only {} tokens",
            fin.filled,
            fin.tokens.len()
        )));
    }

    // ---- the stop-the-world window on this side starts here ----------------
    let t_resume = Instant::now();
    let kv = seq.kv();
    let h0 = Instant::now();
    let our_hash = if fin.kv_hash != 0 { kv.hash(fin.filled as usize) } else { 0 };
    let t_hash = h0.elapsed().as_secs_f64() * 1e6;
    debug_ranges("dst", &region, &hello.spec.live_ranges(fin.filled as usize));
    if fin.kv_hash != 0 && our_hash != fin.kv_hash {
        conn.send(&Msg::ResumeAck(hs_wire::ResumeAck {
            ok: false,
            kv_hash: our_hash,
            resume_ns: t_resume.elapsed().as_nanos() as u64,
            reason: "KV hash mismatch".into(),
        }))?;
        return Err(io::Error::other("KV hash mismatch"));
    }

    let b0 = Instant::now();
    {
        let mut st = seq.state.lock().unwrap();
        st.tokens = fin.tokens.clone();
        st.prompt_len = fin.prompt_len as usize;
        st.filled = fin.filled as usize;
        st.max_tokens = fin.max_tokens as usize;
        st.sampler = new_sampler(fin.cfg);
        st.sampler.rng = fin.rng; // the whole reason the transcript continues
        st.ready = true;
        seq.cv.notify_all();
    }
    let t_build = b0.elapsed().as_secs_f64() * 1e6;
    conn.send(&Msg::ResumeAck(hs_wire::ResumeAck {
        ok: true,
        kv_hash: our_hash,
        resume_ns: t_resume.elapsed().as_nanos() as u64,
        reason: String::new(),
    }))?;
    std::mem::forget(guard); // handover complete; the sequence lives on

    eprintln!(
        "[{}] took over sequence {} at position {} ({} tokens), {} received \
         | resume {:.1} us = hash {:.1} + build/spawn {:.1}",
        worker.name,
        seq.id,
        fin.filled,
        fin.tokens.len(),
        fmt_bytes(received),
        t_resume.elapsed().as_secs_f64() * 1e6,
        t_hash,
        t_build,
    );
    Ok(())
}
