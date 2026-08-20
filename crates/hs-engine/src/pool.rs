//! A fixed-size worker pool with a spin barrier.
//!
//! A decode step issues on the order of two hundred parallel regions (seven
//! matvecs per layer, twenty-eight layers). At that rate the cost of *starting*
//! parallel work is part of the token latency, so the workers spin on an atomic
//! rather than sleeping on a condvar: a dispatch costs a few hundred
//! nanoseconds instead of the few microseconds a futex wake would.
//!
//! Work is split by fixed row ranges, never work-stealing. That is not a
//! performance choice, it is the determinism requirement: row `r` must be
//! computed by the same accumulation on every host regardless of how the
//! threads happened to be scheduled.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

struct Shared {
    seq: AtomicU64,
    done: AtomicUsize,
    stop: AtomicBool,
    /// Fat pointer to the current job, split into its two halves. Valid only
    /// between the `seq` bump and `done` reaching `n - 1`.
    job: [AtomicUsize; 2],
    n: usize,
}

pub struct Pool {
    shared: Arc<Shared>,
    handles: Vec<JoinHandle<()>>,
}

/// How many times to spin before yielding. Long enough to cover a matvec
/// dispatch on an idle machine, short enough not to burn a core when the
/// process is descheduled.
const SPINS: u32 = 20_000;

#[inline]
fn spin_until(mut cond: impl FnMut() -> bool) {
    let mut i = 0u32;
    while !cond() {
        if i < SPINS {
            std::hint::spin_loop();
            i += 1;
        } else {
            std::thread::yield_now();
        }
    }
}

impl Pool {
    /// `n` is the total number of participating threads, counting the caller.
    pub fn new(n: usize) -> Pool {
        assert!(n >= 1);
        let shared = Arc::new(Shared {
            seq: AtomicU64::new(0),
            done: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            job: [AtomicUsize::new(0), AtomicUsize::new(0)],
            n,
        });
        let mut handles = Vec::with_capacity(n - 1);
        for tid in 1..n {
            let s = shared.clone();
            handles.push(
                std::thread::Builder::new()
                    .name(format!("hs-pool-{tid}"))
                    .spawn(move || worker(s, tid))
                    .expect("spawn pool worker"),
            );
        }
        Pool { shared, handles }
    }

    pub fn threads(&self) -> usize {
        self.shared.n
    }

    /// Run `f(tid)` on every thread, including this one, and return when all
    /// have finished.
    pub fn broadcast<F: Fn(usize) + Sync>(&self, f: F) {
        let s = &self.shared;
        if s.n == 1 {
            f(0);
            return;
        }
        let dyn_ref: &(dyn Fn(usize) + Sync) = &f;
        // SAFETY: the fat pointer is only observed by workers between the `seq`
        // bump below and the `done` count reaching n-1, and this function does
        // not return until that has happened, so `f` outlives every use.
        let parts: [usize; 2] = unsafe { std::mem::transmute(dyn_ref) };
        s.job[0].store(parts[0], Ordering::Relaxed);
        s.job[1].store(parts[1], Ordering::Relaxed);
        s.done.store(0, Ordering::Relaxed);
        s.seq.fetch_add(1, Ordering::Release);

        f(0);

        spin_until(|| s.done.load(Ordering::Acquire) == s.n - 1);
    }

    /// Split `rows` into `threads` contiguous ranges. The split depends only on
    /// `rows` and the thread count, so a given row is always in a given range.
    #[inline]
    pub fn range(&self, rows: usize, tid: usize) -> (usize, usize) {
        split(rows, self.shared.n, tid)
    }
}

#[inline]
pub fn split(rows: usize, n: usize, tid: usize) -> (usize, usize) {
    let per = rows / n;
    let rem = rows % n;
    let start = tid * per + tid.min(rem);
    let len = per + if tid < rem { 1 } else { 0 };
    (start, start + len)
}

fn worker(s: Arc<Shared>, tid: usize) {
    let mut last = 0u64;
    loop {
        spin_until(|| s.stop.load(Ordering::Acquire) || s.seq.load(Ordering::Acquire) != last);
        if s.stop.load(Ordering::Acquire) {
            return;
        }
        last = s.seq.load(Ordering::Acquire);
        let parts = [s.job[0].load(Ordering::Relaxed), s.job[1].load(Ordering::Relaxed)];
        // SAFETY: mirror of the transmute in `broadcast`; the referent is alive
        // until we bump `done`.
        let f: &(dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(parts) };
        f(tid);
        s.done.fetch_add(1, Ordering::Release);
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    #[test]
    fn ranges_tile_exactly() {
        for rows in [0usize, 1, 7, 8, 1536, 151936] {
            for n in 1..=9 {
                let mut covered = 0;
                let mut prev_end = 0;
                for tid in 0..n {
                    let (a, b) = split(rows, n, tid);
                    assert_eq!(a, prev_end, "gap at rows={rows} n={n} tid={tid}");
                    covered += b - a;
                    prev_end = b;
                }
                assert_eq!(covered, rows);
                assert_eq!(prev_end, rows);
            }
        }
    }

    #[test]
    fn broadcast_runs_every_thread_once() {
        let p = Pool::new(4);
        let hits: Vec<AtomicU32> = (0..4).map(|_| AtomicU32::new(0)).collect();
        for _ in 0..100 {
            p.broadcast(|tid| {
                hits[tid].fetch_add(1, Ordering::Relaxed);
            });
        }
        for h in &hits {
            assert_eq!(h.load(Ordering::Relaxed), 100);
        }
    }

    #[test]
    fn broadcast_writes_are_visible_after_return() {
        let p = Pool::new(4);
        let mut buf = vec![0u32; 4096];
        for round in 1..50u32 {
            let ptr = buf.as_mut_ptr() as usize;
            p.broadcast(|tid| {
                let (a, b) = split(4096, 4, tid);
                for i in a..b {
                    // SAFETY: ranges are disjoint by construction.
                    unsafe { *(ptr as *mut u32).add(i) = round }
                }
            });
            assert!(buf.iter().all(|&v| v == round), "round {round}");
        }
    }
}
