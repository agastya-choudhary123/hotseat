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
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

struct Shared {
    seq: AtomicU64,
    done: AtomicUsize,
    stop: AtomicBool,
    /// Fat pointer to the current job, split into its two halves. Valid only
    /// between the `seq` bump and `done` reaching `n - 1`.
    job: [AtomicUsize; 2],
    n: usize,
    /// Workers currently blocked rather than spinning. Checked on the dispatch
    /// path, which is why it is a counter and not just a flag.
    sleepers: AtomicUsize,
    sleep_lock: Mutex<()>,
    wake: Condvar,
}

pub struct Pool {
    shared: Arc<Shared>,
    handles: Vec<JoinHandle<()>>,
    /// One dispatch at a time.
    ///
    /// The pool has a single job slot, so two threads calling `broadcast`
    /// concurrently would have the workers run one closure with the other's
    /// bounds. A worker hosting a sequence while receiving another has exactly
    /// two decode threads sharing one pool, and the symptom was a pool worker
    /// indexing a 128-element bias at 688. An uncontended lock costs tens of
    /// nanoseconds against a dispatch that does microseconds of work.
    dispatch: Mutex<()>,
}

/// How many times to spin before yielding, and how many times to yield before
/// blocking. The first stage keeps a dispatch under a microsecond while tokens
/// are flowing; the last stage is what stops an idle worker from burning a core.
///
/// That last stage is not a nicety. Two workers on one machine with three
/// spinning threads each will happily consume six cores doing nothing, and the
/// visible symptom was a migration thread waiting 112 ms to be scheduled for
/// work that takes 11 ms.
const SPINS: u32 = 20_000;
const YIELDS: u32 = 200;

#[inline]
fn spin_until(mut cond: impl FnMut() -> bool) {
    let mut i = 0u32;
    while !cond() {
        if i < SPINS {
            std::hint::spin_loop();
        } else {
            std::thread::yield_now();
        }
        i = i.saturating_add(1);
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
            sleepers: AtomicUsize::new(0),
            sleep_lock: Mutex::new(()),
            wake: Condvar::new(),
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
        Pool { shared, handles, dispatch: Mutex::new(()) }
    }

    pub fn threads(&self) -> usize {
        self.shared.n
    }

    /// Run `f(tid)` on every thread, including this one, and return when all
    /// have finished.
    pub fn broadcast<F: Fn(usize) + Sync>(&self, f: F) {
        let s = &self.shared;
        let _dispatch = self.dispatch.lock().unwrap_or_else(|e| e.into_inner());
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
        if s.sleepers.load(Ordering::Acquire) != 0 {
            let _g = s.sleep_lock.lock().unwrap();
            s.wake.notify_all();
        }

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
        let ready = || s.stop.load(Ordering::Acquire) || s.seq.load(Ordering::Acquire) != last;
        let mut i = 0u32;
        while !ready() {
            if i < SPINS {
                std::hint::spin_loop();
                i += 1;
            } else if i < SPINS + YIELDS {
                std::thread::yield_now();
                i += 1;
            } else {
                // Block. The dispatcher bumps `seq` before it looks at
                // `sleepers`, and this side registers as a sleeper before it
                // re-checks `seq`, so a dispatch cannot slip between the two.
                s.sleepers.fetch_add(1, Ordering::AcqRel);
                let g = s.sleep_lock.lock().unwrap();
                if !ready() {
                    let _ = s.wake.wait_timeout(g, std::time::Duration::from_millis(2));
                }
                s.sleepers.fetch_sub(1, Ordering::AcqRel);
            }
        }
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
        {
            let _g = self.shared.sleep_lock.lock().unwrap();
            self.shared.wake.notify_all();
        }
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
    fn concurrent_broadcasts_do_not_interleave_their_jobs() {
        // Two threads dispatching different-sized jobs at once. Without the
        // dispatch lock, a worker runs one closure while another has already
        // replaced the job pointer.
        let p = Arc::new(Pool::new(4));
        let bad = Arc::new(AtomicU32::new(0));
        let mut hs = Vec::new();
        for which in 0..2u32 {
            let p = p.clone();
            let bad = bad.clone();
            hs.push(std::thread::spawn(move || {
                let n = if which == 0 { 64usize } else { 4096 };
                for _ in 0..2000 {
                    p.broadcast(|tid| {
                        let (a, b) = split(n, 4, tid);
                        if b > n || a > b {
                            bad.fetch_add(1, Ordering::Relaxed);
                        }
                    });
                }
            }));
        }
        for h in hs {
            h.join().unwrap();
        }
        assert_eq!(bad.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn broadcast_still_works_after_the_workers_have_gone_to_sleep() {
        let p = Pool::new(4);
        let hits: Vec<AtomicU32> = (0..4).map(|_| AtomicU32::new(0)).collect();
        p.broadcast(|tid| {
            hits[tid].fetch_add(1, Ordering::Relaxed);
        });
        // Long enough that every worker has passed the spin and yield stages.
        std::thread::sleep(std::time::Duration::from_millis(50));
        for _ in 0..20 {
            p.broadcast(|tid| {
                hits[tid].fetch_add(1, Ordering::Relaxed);
            });
        }
        for h in &hits {
            assert_eq!(h.load(Ordering::Relaxed), 21);
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
