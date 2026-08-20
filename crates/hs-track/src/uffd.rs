//! Dirty-page tracking on Linux with `userfaultfd` in write-protect mode.
//!
//! The Linux counterpart of the Mach backend, and structurally the same idea:
//! take write permission away, let the hardware tell you when the model tries
//! to store, hand the permission back. The differences are in the plumbing —
//! faults arrive as 32-byte records on a file descriptor rather than as Mach
//! messages, and the protection change is an `ioctl` on that descriptor rather
//! than a VM call — and in one detail that matters for the numbers: the
//! faulting thread is woken by the same `ioctl` that clears the protection, so
//! there is one round trip per fault instead of two.
//!
//! The round-boundary scheme is identical to the Mach backend's two-buffer
//! epoch; see `machtrack` for the argument that no write can be lost.
//!
//! Requires a kernel with `UFFD_FEATURE_PAGEFAULT_FLAG_WP` (5.19+ for
//! anonymous memory) and permission to call `userfaultfd(2)`: either
//! `vm.unprivileged_userfaultfd=1` or `CAP_SYS_PTRACE`. Docker's default
//! seccomp profile blocks the syscall without that capability, so the
//! container demo passes `--cap-add SYS_PTRACE`.

use crate::pageset::PageSet;
use crate::region::Region;
use crate::{Stats, Tracker};
use std::io;
use std::os::unix::io::RawFd;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

const UFFD_API: u64 = 0xAA;
const UFFD_FEATURE_PAGEFAULT_FLAG_WP: u64 = 1 << 0;

const UFFDIO_REGISTER_MODE_WP: u64 = 1 << 1;
const UFFDIO_WRITEPROTECT_MODE_WP: u64 = 1 << 0;

const UFFD_EVENT_PAGEFAULT: u8 = 0x12;
const UFFD_PAGEFAULT_FLAG_WP: u64 = 1 << 1;

// _IOC(dir, type, nr, size) for asm-generic: dir<<30 | size<<16 | type<<8 | nr
const fn ioc(dir: u64, nr: u64, size: u64) -> u64 {
    (dir << 30) | (size << 16) | (0xAA << 8) | nr
}
const DIR_WR: u64 = 3; // _IOC_READ | _IOC_WRITE
const DIR_R: u64 = 2;

const UFFDIO_API: u64 = ioc(DIR_WR, 0x3F, 24);
const UFFDIO_REGISTER: u64 = ioc(DIR_WR, 0x00, 32);
const UFFDIO_UNREGISTER: u64 = ioc(DIR_R, 0x01, 16);
const UFFDIO_WRITEPROTECT: u64 = ioc(DIR_WR, 0x06, 24);

#[repr(C)]
struct UffdioApi {
    api: u64,
    features: u64,
    ioctls: u64,
}

#[repr(C)]
struct UffdioRange {
    start: u64,
    len: u64,
}

#[repr(C)]
struct UffdioRegister {
    range: UffdioRange,
    mode: u64,
    ioctls: u64,
}

#[repr(C)]
struct UffdioWriteprotect {
    range: UffdioRange,
    mode: u64,
}

/// `struct uffd_msg`, 32 bytes. Only the pagefault arm is used.
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct UffdMsg {
    event: u8,
    _r1: u8,
    _r2: u16,
    _r3: u32,
    flags: u64,
    address: u64,
    ptid_and_pad: u64,
}

struct Inner {
    region: Arc<Region>,
    uffd: RawFd,
    bufs: [PageSet; 2],
    epoch: AtomicU64,
    cluster: usize,
    armed: AtomicBool,
    stop: AtomicBool,
    faults: AtomicU64,
    fault_ns: AtomicU64,
    pages_marked: AtomicU64,
    harvests: AtomicU64,
    rearm_ns: AtomicU64,
    foreign: AtomicU64,
}

pub struct UffdTracker {
    inner: Arc<Inner>,
    handler: Option<std::thread::JoinHandle<()>>,
}

fn last_err(what: &str) -> io::Error {
    io::Error::new(io::Error::last_os_error().kind(), format!("{what}: {}", io::Error::last_os_error()))
}

impl UffdTracker {
    pub fn new(region: Arc<Region>, cluster: usize) -> io::Result<UffdTracker> {
        assert!(cluster >= 1, "fault cluster must be at least one page");

        // SYS_userfaultfd is 282 on aarch64 and x86_64 alike.
        let uffd = unsafe { libc::syscall(libc::SYS_userfaultfd, libc::O_CLOEXEC) } as RawFd;
        if uffd < 0 {
            let e = io::Error::last_os_error();
            return Err(io::Error::new(
                e.kind(),
                format!(
                    "userfaultfd(2) failed: {e}. The syscall needs \
                     vm.unprivileged_userfaultfd=1 or CAP_SYS_PTRACE, and Docker's default \
                     seccomp profile blocks it without the capability -- try --cap-add SYS_PTRACE"
                ),
            ));
        }

        let mut api = UffdioApi { api: UFFD_API, features: UFFD_FEATURE_PAGEFAULT_FLAG_WP, ioctls: 0 };
        if unsafe { libc::ioctl(uffd, UFFDIO_API as _, &mut api) } < 0 {
            unsafe { libc::close(uffd) };
            return Err(last_err("UFFDIO_API (kernel lacks write-protect mode?)"));
        }
        if api.features & UFFD_FEATURE_PAGEFAULT_FLAG_WP == 0 {
            unsafe { libc::close(uffd) };
            return Err(io::Error::other(
                "this kernel's userfaultfd has no PAGEFAULT_FLAG_WP; \
                 write-protect tracking needs 5.19 or newer for anonymous memory",
            ));
        }

        let mut reg = UffdioRegister {
            range: UffdioRange { start: region.addr() as u64, len: region.len() as u64 },
            mode: UFFDIO_REGISTER_MODE_WP,
            ioctls: 0,
        };
        if unsafe { libc::ioctl(uffd, UFFDIO_REGISTER as _, &mut reg) } < 0 {
            unsafe { libc::close(uffd) };
            return Err(last_err("UFFDIO_REGISTER(WP)"));
        }

        let n_pages = region.n_pages();
        let inner = Arc::new(Inner {
            region,
            uffd,
            bufs: [PageSet::new(n_pages), PageSet::new(n_pages)],
            epoch: AtomicU64::new(0),
            cluster,
            armed: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            faults: AtomicU64::new(0),
            fault_ns: AtomicU64::new(0),
            pages_marked: AtomicU64::new(0),
            harvests: AtomicU64::new(0),
            rearm_ns: AtomicU64::new(0),
            foreign: AtomicU64::new(0),
        });
        let h = inner.clone();
        let handler = std::thread::Builder::new()
            .name("hs-uffd".into())
            .spawn(move || handler_loop(h))?;
        Ok(UffdTracker { inner, handler: Some(handler) })
    }

    fn write_protect(&self, start: usize, len: usize, on: bool) -> io::Result<()> {
        write_protect_fd(self.inner.uffd, start, len, on)
    }
}

fn write_protect_fd(uffd: RawFd, start: usize, len: usize, on: bool) -> io::Result<()> {
    let mut wp = UffdioWriteprotect {
        range: UffdioRange { start: start as u64, len: len as u64 },
        mode: if on { UFFDIO_WRITEPROTECT_MODE_WP } else { 0 },
    };
    if unsafe { libc::ioctl(uffd, UFFDIO_WRITEPROTECT as _, &mut wp) } < 0 {
        return Err(last_err("UFFDIO_WRITEPROTECT"));
    }
    Ok(())
}

impl Tracker for UffdTracker {
    fn name(&self) -> &'static str {
        "uffd-wp"
    }

    fn region(&self) -> &Arc<Region> {
        &self.inner.region
    }

    fn arm(&self) -> io::Result<()> {
        self.inner.bufs[0].clear();
        self.inner.bufs[1].clear();
        self.write_protect(self.inner.region.addr(), self.inner.region.len(), true)?;
        self.inner.armed.store(true, Ordering::Release);
        Ok(())
    }

    fn harvest(&self, out: &PageSet) -> io::Result<()> {
        let t0 = std::time::Instant::now();
        let old = self.inner.epoch.fetch_add(1, Ordering::AcqRel);
        self.write_protect(self.inner.region.addr(), self.inner.region.len(), true)?;
        self.inner.bufs[(old & 1) as usize].drain_into(out);
        self.inner.harvests.fetch_add(1, Ordering::Relaxed);
        self.inner.rearm_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Ok(())
    }

    fn disarm(&self) -> io::Result<()> {
        self.inner.armed.store(false, Ordering::Release);
        self.write_protect(self.inner.region.addr(), self.inner.region.len(), false)
    }

    fn stats(&self) -> Stats {
        let i = &self.inner;
        Stats {
            faults: i.faults.load(Ordering::Relaxed),
            fault_ns: i.fault_ns.load(Ordering::Relaxed),
            pages_marked: i.pages_marked.load(Ordering::Relaxed),
            harvests: i.harvests.load(Ordering::Relaxed),
            rearm_ns: i.rearm_ns.load(Ordering::Relaxed),
            foreign_faults: i.foreign.load(Ordering::Relaxed),
        }
    }
}

impl Drop for UffdTracker {
    fn drop(&mut self) {
        let _ = self.disarm();
        let mut range = UffdioRange {
            start: self.inner.region.addr() as u64,
            len: self.inner.region.len() as u64,
        };
        unsafe {
            libc::ioctl(self.inner.uffd, UFFDIO_UNREGISTER as _, &mut range);
        }
        self.inner.stop.store(true, Ordering::Release);
        if let Some(h) = self.handler.take() {
            let _ = h.join();
        }
        unsafe {
            libc::close(self.inner.uffd);
        }
    }
}

fn handler_loop(inner: Arc<Inner>) {
    let mut pfd = libc::pollfd { fd: inner.uffd, events: libc::POLLIN, revents: 0 };
    while !inner.stop.load(Ordering::Acquire) {
        // A timeout rather than a blocking read, so the thread notices `stop`
        // without needing the fd closed underneath it.
        let n = unsafe { libc::poll(&mut pfd, 1, 50) };
        if n <= 0 {
            continue;
        }
        let mut msg = UffdMsg::default();
        let r = unsafe {
            libc::read(inner.uffd, &mut msg as *mut _ as *mut libc::c_void, std::mem::size_of::<UffdMsg>())
        };
        if r != std::mem::size_of::<UffdMsg>() as isize {
            if r < 0 && io::Error::last_os_error().kind() == io::ErrorKind::WouldBlock {
                continue;
            }
            if inner.stop.load(Ordering::Acquire) {
                return;
            }
            eprintln!("hs-track: short read from userfaultfd ({r})");
            continue;
        }
        if msg.event != UFFD_EVENT_PAGEFAULT || msg.flags & UFFD_PAGEFAULT_FLAG_WP == 0 {
            continue;
        }
        let t0 = std::time::Instant::now();
        handle_fault(&inner, msg.address as usize);
        inner.fault_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

fn handle_fault(inner: &Inner, addr: usize) {
    let Some(page) = inner.region.page_of(addr) else {
        inner.foreign.fetch_add(1, Ordering::Relaxed);
        return;
    };
    let ps = inner.region.page_size();
    let first = page / inner.cluster * inner.cluster;
    let count = inner.cluster.min(inner.region.n_pages() - first);

    let e0 = inner.epoch.load(Ordering::Acquire);
    inner.bufs[(e0 & 1) as usize].mark_range(first, count);

    // Clearing the protection also wakes the faulting thread.
    if let Err(e) = write_protect_fd(inner.uffd, inner.region.addr() + first * ps, count * ps, false) {
        eprintln!("hs-track: un-protect of page {page} failed: {e}");
        return;
    }

    let e1 = inner.epoch.load(Ordering::Acquire);
    if e1 != e0 {
        inner.bufs[(e1 & 1) as usize].mark_range(first, count);
    }
    inner.faults.fetch_add(1, Ordering::Relaxed);
    inner.pages_marked.fetch_add(count as u64, Ordering::Relaxed);
}
