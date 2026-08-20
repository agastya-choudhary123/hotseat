//! A decoding sequence and the thread that drives it.
//!
//! The state a migration has to move is exactly the fields of `SeqState` plus
//! the KV region. The loop maintains one invariant that makes that true:
//!
//! > at the top of every iteration, `filled < tokens.len()`
//!
//! meaning there is always at least one token that has been decided but not yet
//! run through the model. A sequence paused there needs no logits, no partial
//! layer state and no in-flight anything — feed `tokens[filled]` at position
//! `filled` and carry on. That is why the handover can be a few kilobytes of
//! metadata plus pages, and why it can happen mid-prefill just as easily as
//! mid-generation.

use hs_engine::kv::{Kv, KvSpec};
use hs_engine::model::Model;
use hs_engine::pool::Pool;
use hs_engine::sampler::{argmax, Sampler, SamplerCfg};
use hs_track::{Region, Tracker};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// One emitted token, with enough context to line two hosts' transcripts up.
#[derive(Debug, Clone)]
pub struct TokenEvent {
    pub index: usize,
    pub token: u32,
    /// Wall clock, so a transcript that spans two workers on one machine shows
    /// the real gap across the handover.
    pub epoch_ns: u128,
    pub host: String,
}

pub struct SeqState {
    /// Prompt followed by everything generated so far.
    pub tokens: Vec<u32>,
    pub prompt_len: usize,
    /// Positions already present in the KV cache.
    pub filled: usize,
    pub sampler: Sampler,
    pub max_tokens: usize,
    pub done: bool,
    pub done_reason: String,
    /// Set once this worker has handed the sequence to someone else.
    pub migrated_away: bool,
    pub pause_req: bool,
    pub paused: bool,
    pub log: Vec<TokenEvent>,
    /// Tokens decoded on this host (not counting any inherited from a source).
    pub decoded_here: usize,
}

impl SeqState {
    /// The invariant the migration point relies on.
    pub fn at_token_boundary(&self) -> bool {
        self.done || self.filled < self.tokens.len()
    }
}

pub struct Seq {
    pub id: u64,
    pub host: String,
    pub spec: KvSpec,
    pub region: Arc<Region>,
    pub tracker: Box<dyn Tracker>,
    pub model: Arc<Model>,
    pub pool: Arc<Pool>,
    pub state: Mutex<SeqState>,
    pub cv: Condvar,
    /// Nanoseconds the decode loop spent parked for a migration.
    pub parked_ns: AtomicU64,
}

pub fn now_ns() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
}

impl Seq {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: u64,
        host: String,
        model: Arc<Model>,
        pool: Arc<Pool>,
        spec: KvSpec,
        region: Arc<Region>,
        tracker: Box<dyn Tracker>,
        tokens: Vec<u32>,
        prompt_len: usize,
        filled: usize,
        sampler: Sampler,
        max_tokens: usize,
    ) -> Arc<Seq> {
        Arc::new(Seq {
            id,
            host,
            spec,
            region,
            tracker,
            model,
            pool,
            state: Mutex::new(SeqState {
                tokens,
                prompt_len,
                filled,
                sampler,
                max_tokens,
                done: false,
                done_reason: String::new(),
                migrated_away: false,
                pause_req: false,
                paused: false,
                log: Vec::new(),
                decoded_here: 0,
            }),
            cv: Condvar::new(),
            parked_ns: AtomicU64::new(0),
        })
    }

    pub fn kv(&self) -> Kv {
        // SAFETY: the region is sized from `spec` and outlives every `Kv` made
        // from it; only the decode thread writes it, and only while running.
        unsafe { Kv::from_raw(self.region.as_ptr(), self.spec) }
    }

    pub fn filled(&self) -> usize {
        self.state.lock().unwrap().filled
    }

    /// Park the decode loop at the next token boundary and return once it is
    /// parked. This is the instant the stop-the-world window opens.
    pub fn pause(&self) {
        let mut st = self.state.lock().unwrap();
        if st.done || st.migrated_away {
            return;
        }
        st.pause_req = true;
        while !st.paused && !st.done {
            st = self.cv.wait(st).unwrap();
        }
    }

    pub fn resume(&self) {
        let mut st = self.state.lock().unwrap();
        st.pause_req = false;
        self.cv.notify_all();
        drop(st);
    }

    /// Tell the decode loop the sequence now belongs to another worker.
    pub fn hand_off(&self) {
        let mut st = self.state.lock().unwrap();
        st.migrated_away = true;
        st.pause_req = false;
        self.cv.notify_all();
    }

    /// Block until at least `n` positions are in the cache, or the sequence
    /// ends. Lets a test pin the migration to an exact point in the sequence
    /// instead of racing it with a sleep.
    pub fn wait_filled(&self, n: usize) {
        let mut st = self.state.lock().unwrap();
        while st.filled < n && !st.done && !st.migrated_away {
            st = self.cv.wait(st).unwrap();
        }
    }

    pub fn wait_done(&self) {
        let mut st = self.state.lock().unwrap();
        while !st.done && !st.migrated_away {
            st = self.cv.wait(st).unwrap();
        }
    }
}

/// Run the sequence to completion, or until it is handed to another worker.
pub fn decode_loop(seq: Arc<Seq>) {
    let kv = seq.kv();
    let mut scratch = seq.model.scratch();
    let eos = seq.model.tok.eos;

    loop {
        // ---- pause point -------------------------------------------------
        // The only place the loop can be interrupted, and the reason a
        // handover never has to serialise anything but tokens and pages.
        let (token, pos) = {
            let mut st = seq.state.lock().unwrap();
            if st.pause_req {
                let t0 = Instant::now();
                st.paused = true;
                seq.cv.notify_all();
                while st.pause_req && !st.migrated_away {
                    st = seq.cv.wait(st).unwrap();
                }
                st.paused = false;
                seq.parked_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
            if st.migrated_away || st.done {
                seq.cv.notify_all();
                return;
            }
            debug_assert!(st.filled < st.tokens.len(), "loop invariant broken");
            (st.tokens[st.filled], st.filled)
        };

        // ---- one forward pass, outside the lock --------------------------
        seq.model.forward(&kv, token, pos, &mut scratch, &seq.pool);

        // ---- commit ------------------------------------------------------
        let mut st = seq.state.lock().unwrap();
        st.filled += 1;
        if st.filled == st.tokens.len() {
            let hist = st.tokens.clone();
            let next = if st.sampler.cfg.temperature <= 0.0 {
                // Greedy still runs through the sampler for the repetition
                // penalty, but must not consume the RNG.
                let mut l = std::mem::take(&mut scratch.logits);
                let t = penalised_argmax(&mut l, &hist, &st.sampler.cfg);
                scratch.logits = l;
                t
            } else {
                let mut l = std::mem::take(&mut scratch.logits);
                let t = st.sampler.sample(&mut l, &hist);
                scratch.logits = l;
                t
            };
            let generated = st.tokens.len() - st.prompt_len;
            if next == eos {
                st.done = true;
                st.done_reason = "eos".into();
            } else if generated >= st.max_tokens {
                st.done = true;
                st.done_reason = "max_tokens".into();
            } else if st.tokens.len() + 1 >= seq.spec.max_ctx {
                st.done = true;
                st.done_reason = "context full".into();
            } else {
                let index = st.tokens.len();
                st.tokens.push(next);
                st.log.push(TokenEvent {
                    index,
                    token: next,
                    epoch_ns: now_ns(),
                    host: seq.host.clone(),
                });
                st.decoded_here += 1;
            }
        }
        seq.cv.notify_all();
        if st.done {
            return;
        }
    }
}

/// Greedy pick, after applying the same repetition penalty sampling would.
fn penalised_argmax(logits: &mut [f32], history: &[u32], cfg: &SamplerCfg) -> u32 {
    if cfg.repeat_penalty != 1.0 && cfg.repeat_last_n > 0 {
        let n = (cfg.repeat_last_n as usize).min(history.len());
        for &t in &history[history.len() - n..] {
            let l = &mut logits[t as usize];
            *l = if *l > 0.0 { *l / cfg.repeat_penalty } else { *l * cfg.repeat_penalty };
        }
    }
    argmax(logits)
}

pub fn new_sampler(cfg: SamplerCfg) -> Sampler {
    Sampler::new(cfg)
}
