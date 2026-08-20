//! Dirty-page tracking on Linux by scanning the soft-dirty bit.
//!
//! The other shape the problem can take. Instead of arranging for writes to
//! trap, the kernel already records, per page-table entry, whether the page has
//! been written since the last `clear_refs`; you clear the bits, let the
//! process run at full speed, then read them back out of `/proc/self/pagemap`.
//! This is what CRIU's pre-copy uses.
//!
//! It has the opposite cost profile to write protection, and that is the whole
//! reason it is here:
//!
//! * writes cost **nothing** — no fault, no ioctl, no handler thread;
//! * asking the question costs a walk of every page-table entry in the region,
//!   whether or not anything was written.
//!
//! So which one wins is entirely a question of the ratio between pages written
//! and pages watched. A KV cache is a large allocation with a small moving
//! write set, which is the regime that favours faulting. `docs/` has the
//! measurement.
//!
//! # The race, stated plainly
//!
//! `clear_refs` is process-wide and there is no way to read the bits and clear
//! them in one operation. A write landing between the read and the clear sets a
//! bit that the clear then wipes, and that write is lost from the dirty set —
//! silently, and forever. Write protection has no such window, because the
//! permission and the record are changed under the same fault.
//!
//! CRIU gets away with it by freezing the process across the boundary. This
//! backend does not freeze anything, so it is offered for measurement and
//! comparison and `is_exact()` says so. The tracker acceptance tests run the
//! concurrent-writer case against it precisely to demonstrate the difference.

use crate::pageset::PageSet;
use crate::region::Region;
use crate::{Stats, Tracker};
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Bit 55 of a pagemap entry: written since the last `clear_refs`.
const PM_SOFT_DIRTY: u64 = 1 << 55;
/// Bit 63: the page is present in RAM.
const PM_PRESENT: u64 = 1 << 63;
/// `4` means "clear the soft-dirty bits and start tracking".
const CLEAR_REFS_SOFT_DIRTY: &[u8] = b"4\n";

pub struct SoftDirtyTracker {
    region: Arc<Region>,
    pagemap: Mutex<File>,
    clear_refs: Mutex<File>,
    harvests: AtomicU64,
    rearm_ns: AtomicU64,
    pages_marked: AtomicU64,
}

impl SoftDirtyTracker {
    pub fn new(region: Arc<Region>) -> io::Result<SoftDirtyTracker> {
        let pagemap = File::open("/proc/self/pagemap").map_err(|e| {
            io::Error::new(
                e.kind(),
                format!("/proc/self/pagemap: {e} (soft-dirty tracking needs procfs mounted)"),
            )
        })?;
        let clear_refs = OpenOptions::new().write(true).open("/proc/self/clear_refs").map_err(|e| {
            io::Error::new(e.kind(), format!("/proc/self/clear_refs: {e}"))
        })?;
        let t = SoftDirtyTracker {
            region,
            pagemap: Mutex::new(pagemap),
            clear_refs: Mutex::new(clear_refs),
            harvests: AtomicU64::new(0),
            rearm_ns: AtomicU64::new(0),
            pages_marked: AtomicU64::new(0),
        };
        t.self_test()?;
        Ok(t)
    }

    /// Prove the kernel actually implements soft-dirty before anyone relies on
    /// it.
    ///
    /// `CONFIG_MEM_SOFT_DIRTY` is off in some common kernels — Docker
    /// Desktop's LinuxKit build among them — and the failure mode is silent:
    /// the write to `clear_refs` succeeds, the pagemap reads fine, and no bit
    /// is ever set. A tracker that reports "nothing changed" does not look
    /// broken, it looks converged, and a migration built on it would hand over
    /// a stale cache. So: dirty a page on purpose and refuse to exist if the
    /// kernel does not notice.
    fn self_test(&self) -> io::Result<()> {
        let probe = Region::new(4 * self.region.page_size())?;
        probe.prefault();
        self.clear()?;
        unsafe { probe.bytes_mut(2 * probe.page_size(), 8)[0] = 1 };

        let ps = probe.page_size();
        let mut f = self.pagemap.lock().unwrap();
        f.seek(SeekFrom::Start((probe.addr() / ps) as u64 * 8))?;
        let mut buf = [0u8; 8 * 4];
        f.read_exact(&mut buf)?;
        drop(f);
        let dirty = (0..4).any(|i| {
            let e = u64::from_le_bytes(buf[i * 8..i * 8 + 8].try_into().unwrap());
            e & PM_PRESENT != 0 && e & PM_SOFT_DIRTY != 0
        });
        if !dirty {
            return Err(io::Error::other(
                "this kernel does not implement the soft-dirty bit: a page was written \
                 immediately after clear_refs and no bit was set. Rebuild with \
                 CONFIG_MEM_SOFT_DIRTY=y, or use --tracker signal.",
            ));
        }
        Ok(())
    }

    fn clear(&self) -> io::Result<()> {
        let mut f = self.clear_refs.lock().unwrap();
        f.write_all(CLEAR_REFS_SOFT_DIRTY)?;
        f.flush()
    }

    /// Read the pagemap entries for the region and mark every soft-dirty page.
    fn scan(&self, out: &PageSet) -> io::Result<usize> {
        let ps = self.region.page_size();
        let n = self.region.n_pages();
        let first_index = self.region.addr() / ps;

        let mut f = self.pagemap.lock().unwrap();
        f.seek(SeekFrom::Start(first_index as u64 * 8))?;

        // 64 Ki entries at a time: half a megabyte of reads for a 112 MiB
        // region at 16 KiB pages, and far fewer syscalls than one per page.
        let mut buf = vec![0u8; 65536 * 8];
        let mut done = 0usize;
        let mut marked = 0usize;
        while done < n {
            let want = (n - done).min(65536) * 8;
            f.read_exact(&mut buf[..want])?;
            for i in 0..want / 8 {
                let e = u64::from_le_bytes(buf[i * 8..i * 8 + 8].try_into().unwrap());
                if e & PM_PRESENT != 0 && e & PM_SOFT_DIRTY != 0 {
                    out.mark(done + i);
                    marked += 1;
                }
            }
            done += want / 8;
        }
        Ok(marked)
    }
}

impl Tracker for SoftDirtyTracker {
    fn name(&self) -> &'static str {
        "soft-dirty"
    }

    fn region(&self) -> &Arc<Region> {
        &self.region
    }

    fn arm(&self) -> io::Result<()> {
        self.clear()
    }

    fn harvest(&self, out: &PageSet) -> io::Result<()> {
        let t0 = std::time::Instant::now();
        let marked = self.scan(out)?;
        // See the module comment: a write landing between these two lines is
        // lost. There is no ordering of them that closes the window.
        self.clear()?;
        self.pages_marked.fetch_add(marked as u64, Ordering::Relaxed);
        self.harvests.fetch_add(1, Ordering::Relaxed);
        self.rearm_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Ok(())
    }

    fn disarm(&self) -> io::Result<()> {
        Ok(()) // nothing was ever protected
    }

    fn is_exact(&self) -> bool {
        false // see the race described at the top of this file
    }

    fn stats(&self) -> Stats {
        Stats {
            faults: 0,
            fault_ns: 0,
            pages_marked: self.pages_marked.load(Ordering::Relaxed),
            harvests: self.harvests.load(Ordering::Relaxed),
            rearm_ns: self.rearm_ns.load(Ordering::Relaxed),
            foreign_faults: 0,
        }
    }
}
