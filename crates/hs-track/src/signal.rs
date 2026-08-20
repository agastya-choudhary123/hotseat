//! Dirty-page tracking with `mprotect` and a signal handler.
//!
//! The portable one. Take write permission away with `mprotect`; the store
//! traps; the kernel delivers `SIGSEGV` (Linux) or `SIGBUS` (macOS) to the
//! faulting thread with the address in `siginfo`; the handler records the page,
//! restores permission and returns, and the hardware re-executes the store.
//! Every Unix has had this since the eighties, and it is how garbage collectors
//! have implemented write barriers for about as long.
//!
//! It exists here for two reasons, one practical and one for comparison.
//!
//! **Practical.** Docker Desktop's LinuxKit kernel — the one an aarch64 Mac
//! runs containers on — has `CONFIG_USERFAULTFD` off (the syscall returns
//! `ENOSYS` even under `--privileged`) and `CONFIG_MEM_SOFT_DIRTY` off (writes
//! to `clear_refs` succeed and no bit is ever set, which is worse). Both of the
//! textbook Linux mechanisms are simply absent. This one needs no capability,
//! no sysctl and no kernel option.
//!
//! **Comparison.** It differs from the Mach backend in exactly one interesting
//! way: the fault is handled *on the faulting thread*, with no message, no
//! second thread and no context switch. That is worth a number, and on the same
//! machine it is a large one.
//!
//! # Async-signal-safety
//!
//! The handler does three things: scan a fixed array of atomics for the region
//! containing the address, set bits with `fetch_or`, and call `mprotect`. No
//! allocation, no locks, no `std::io`. `mprotect` is a bare syscall — not on
//! POSIX's async-signal-safe list, but with no userspace state to corrupt, and
//! it is what every write-barrier implementation in this style does.
//!
//! Deregistration cannot free a region's state while a handler might still be
//! reading it, so a slot is first marked empty and then drained: the tracker
//! waits for the in-handler count to reach zero before releasing its `Arc`.

use crate::pageset::PageSet;
use crate::region::Region;
use crate::{Stats, Tracker};
use std::io;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

/// Regions trackable at once. Small and fixed so the handler can scan it
/// without a lock.
const MAX_SLOTS: usize = 16;

struct Slot {
    /// Base address, or 0 for an empty slot. Written last on registration and
    /// first on removal, so a nonzero base implies a valid `inner`.
    base: AtomicUsize,
    len: AtomicUsize,
    inner: AtomicPtr<Inner>,
}

impl Slot {
    const fn empty() -> Slot {
        Slot {
            base: AtomicUsize::new(0),
            len: AtomicUsize::new(0),
            inner: AtomicPtr::new(std::ptr::null_mut()),
        }
    }
}

#[allow(clippy::declare_interior_mutable_const)]
const EMPTY: Slot = Slot::empty();
static SLOTS: [Slot; MAX_SLOTS] = [EMPTY; MAX_SLOTS];
/// Handlers currently running. Deregistration waits for this to fall to zero.
static IN_HANDLER: AtomicUsize = AtomicUsize::new(0);

struct Inner {
    region: Arc<Region>,
    bufs: [PageSet; 2],
    epoch: AtomicU64,
    cluster: usize,
    armed: AtomicBool,
    faults: AtomicU64,
    fault_ns: AtomicU64,
    pages_marked: AtomicU64,
    harvests: AtomicU64,
    rearm_ns: AtomicU64,
    foreign: AtomicU64,
}

pub struct SignalTracker {
    inner: Arc<Inner>,
    slot: usize,
}

// ---------------------------------------------------------------------------
// handler installation
// ---------------------------------------------------------------------------

static OLD_SEGV: OnceLock<libc::sigaction> = OnceLock::new();
static OLD_BUS: OnceLock<libc::sigaction> = OnceLock::new();
static INSTALLED: OnceLock<()> = OnceLock::new();

fn install_handlers() {
    INSTALLED.get_or_init(|| unsafe {
        for (sig, saved) in [(libc::SIGSEGV, &OLD_SEGV), (libc::SIGBUS, &OLD_BUS)] {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = handler as *const () as usize;
            sa.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK | libc::SA_RESTART;
            libc::sigemptyset(&mut sa.sa_mask);
            let mut old: libc::sigaction = std::mem::zeroed();
            libc::sigaction(sig, &sa, &mut old);
            let _ = saved.set(old);
        }
    });
}

#[cfg(target_os = "linux")]
#[inline(always)]
unsafe fn fault_addr(info: *mut libc::siginfo_t) -> usize {
    (*info).si_addr() as usize
}

#[cfg(target_os = "macos")]
#[inline(always)]
unsafe fn fault_addr(info: *mut libc::siginfo_t) -> usize {
    (*info).si_addr as usize
}

extern "C" fn handler(sig: libc::c_int, info: *mut libc::siginfo_t, ctx: *mut libc::c_void) {
    let addr = unsafe { fault_addr(info) };
    IN_HANDLER.fetch_add(1, Ordering::AcqRel);
    let t0 = std::time::Instant::now();
    let handled = handle(addr, t0);
    IN_HANDLER.fetch_sub(1, Ordering::AcqRel);
    if handled {
        return; // the store is re-executed
    }
    // Not our memory. This is somebody's real crash; give it back to whoever
    // was handling it before us, so a null dereference elsewhere in the process
    // still looks like a null dereference.
    chain(sig, info, ctx);
}

fn chain(sig: libc::c_int, info: *mut libc::siginfo_t, ctx: *mut libc::c_void) {
    let saved = if sig == libc::SIGSEGV { OLD_SEGV.get() } else { OLD_BUS.get() };
    let Some(old) = saved else { return };
    unsafe {
        if old.sa_flags & libc::SA_SIGINFO != 0 && old.sa_sigaction != libc::SIG_DFL
            && old.sa_sigaction != libc::SIG_IGN
        {
            let f: extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut libc::c_void) =
                std::mem::transmute(old.sa_sigaction);
            f(sig, info, ctx);
            return;
        }
        if old.sa_sigaction != libc::SIG_DFL && old.sa_sigaction != libc::SIG_IGN {
            let f: extern "C" fn(libc::c_int) = std::mem::transmute(old.sa_sigaction);
            f(sig);
            return;
        }
        // Default action: restore it and return, so the faulting instruction
        // re-runs and kills the process the way it would have.
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = libc::SIG_DFL;
        libc::sigemptyset(&mut sa.sa_mask);
        libc::sigaction(sig, &sa, std::ptr::null_mut());
    }
}

/// Returns true if the address belonged to a tracked, armed region.
fn handle(addr: usize, t0: std::time::Instant) -> bool {
    for slot in SLOTS.iter() {
        let base = slot.base.load(Ordering::Acquire);
        if base == 0 || addr < base || addr >= base + slot.len.load(Ordering::Relaxed) {
            continue;
        }
        let p = slot.inner.load(Ordering::Acquire);
        if p.is_null() {
            continue;
        }
        // SAFETY: a nonzero base means the Arc behind `inner` is held by a live
        // tracker, and deregistration waits for IN_HANDLER to drain before
        // dropping it.
        let inner: &Inner = unsafe { &*p };
        if !inner.armed.load(Ordering::Acquire) {
            inner.foreign.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        let ps = inner.region.page_size();
        let page = (addr - base) / ps;
        let first = page / inner.cluster * inner.cluster;
        let count = inner.cluster.min(inner.region.n_pages() - first);

        // Same two-buffer epoch as the Mach backend; see `machtrack`.
        let e0 = inner.epoch.load(Ordering::Acquire);
        inner.bufs[(e0 & 1) as usize].mark_range(first, count);
        let r = unsafe {
            libc::mprotect(
                (base + first * ps) as *mut libc::c_void,
                count * ps,
                libc::PROT_READ | libc::PROT_WRITE,
            )
        };
        if r != 0 {
            return false;
        }
        let e1 = inner.epoch.load(Ordering::Acquire);
        if e1 != e0 {
            inner.bufs[(e1 & 1) as usize].mark_range(first, count);
        }
        inner.faults.fetch_add(1, Ordering::Relaxed);
        inner.pages_marked.fetch_add(count as u64, Ordering::Relaxed);
        inner.fault_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        return true;
    }
    false
}

// ---------------------------------------------------------------------------

impl SignalTracker {
    pub fn new(region: Arc<Region>, cluster: usize) -> io::Result<SignalTracker> {
        assert!(cluster >= 1, "fault cluster must be at least one page");
        install_handlers();

        let n_pages = region.n_pages();
        let base = region.addr();
        let len = region.len();
        let inner = Arc::new(Inner {
            region,
            bufs: [PageSet::new(n_pages), PageSet::new(n_pages)],
            epoch: AtomicU64::new(0),
            cluster,
            armed: AtomicBool::new(false),
            faults: AtomicU64::new(0),
            fault_ns: AtomicU64::new(0),
            pages_marked: AtomicU64::new(0),
            harvests: AtomicU64::new(0),
            rearm_ns: AtomicU64::new(0),
            foreign: AtomicU64::new(0),
        });

        // Publish `inner` before `base`: the handler treats a nonzero base as
        // the signal that the rest of the slot is valid.
        let raw = Arc::into_raw(inner.clone()) as *mut Inner;
        for (i, slot) in SLOTS.iter().enumerate() {
            if slot
                .inner
                .compare_exchange(std::ptr::null_mut(), raw, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                slot.len.store(len, Ordering::Relaxed);
                slot.base.store(base, Ordering::Release);
                return Ok(SignalTracker { inner, slot: i });
            }
        }
        // SAFETY: nothing took the pointer.
        unsafe { drop(Arc::from_raw(raw)) };
        Err(io::Error::other(format!("more than {MAX_SLOTS} tracked regions at once")))
    }

    fn protect_all(&self, prot: libc::c_int) -> io::Result<()> {
        let r = &self.inner.region;
        let rc = unsafe { libc::mprotect(r.as_ptr() as *mut libc::c_void, r.len(), prot) };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
}

impl Tracker for SignalTracker {
    fn name(&self) -> &'static str {
        "mprotect-signal"
    }

    fn region(&self) -> &Arc<Region> {
        &self.inner.region
    }

    fn arm(&self) -> io::Result<()> {
        self.inner.bufs[0].clear();
        self.inner.bufs[1].clear();
        self.protect_all(libc::PROT_READ)?;
        self.inner.armed.store(true, Ordering::Release);
        Ok(())
    }

    fn harvest(&self, out: &PageSet) -> io::Result<()> {
        let t0 = std::time::Instant::now();
        let old = self.inner.epoch.fetch_add(1, Ordering::AcqRel);
        self.protect_all(libc::PROT_READ)?;
        self.inner.bufs[(old & 1) as usize].drain_into(out);
        self.inner.harvests.fetch_add(1, Ordering::Relaxed);
        self.inner.rearm_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Ok(())
    }

    fn disarm(&self) -> io::Result<()> {
        self.inner.armed.store(false, Ordering::Release);
        self.protect_all(libc::PROT_READ | libc::PROT_WRITE)
    }

    fn stats(&self) -> Stats {
        let i = &self.inner;
        Stats {
            faults: i.faults.load(Ordering::Relaxed),
            // Time inside the handler only -- the trap and return around it are
            // the kernel's, and show up in `faultbench`'s end-to-end number.
            fault_ns: i.fault_ns.load(Ordering::Relaxed),
            pages_marked: i.pages_marked.load(Ordering::Relaxed),
            harvests: i.harvests.load(Ordering::Relaxed),
            rearm_ns: i.rearm_ns.load(Ordering::Relaxed),
            foreign_faults: i.foreign.load(Ordering::Relaxed),
        }
    }
}

impl Drop for SignalTracker {
    fn drop(&mut self) {
        let _ = self.disarm();
        let slot = &SLOTS[self.slot];
        // Stop new handlers from finding it...
        slot.base.store(0, Ordering::Release);
        // ...then wait for any already inside to leave before releasing the Arc.
        while IN_HANDLER.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
        let raw = slot.inner.swap(std::ptr::null_mut(), Ordering::AcqRel);
        if !raw.is_null() {
            // SAFETY: balances the `Arc::into_raw` in `new`.
            unsafe { drop(Arc::from_raw(raw)) };
        }
    }
}
