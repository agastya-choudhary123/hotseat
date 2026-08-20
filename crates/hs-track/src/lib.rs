//! Dirty-page tracking: which pages of a region has this process written since
//! I last asked?
//!
//! This is the mechanism live VM migration is built on, and it is the mechanism
//! this project applies to a decoding sequence's KV cache. Three real backends
//! are implemented, because the answer to "how do I know what changed" is
//! genuinely different on each platform and the differences show up in the
//! numbers:
//!
//! | backend | how | cost model |
//! |---|---|---|
//! | `mach-vm-protect` (macOS) | write-protect + Mach exception messages | per *first write* to a page |
//! | `uffd-wp` (Linux) | `userfaultfd` write-protect mode | per first write to a page |
//! | `soft-dirty` (Linux) | `clear_refs` + `pagemap` bits | per *scan*, proportional to region size |
//! | `none` | assume everything changed | free, and useless |
//!
//! The first two are fault-driven: they cost nothing for a page that is not
//! written, and a few microseconds for one that is. The third is scan-driven:
//! writes are free, and asking the question costs a walk of the whole region's
//! page table entries. CRIU's pre-copy uses soft-dirty; QEMU's uses write
//! protection. Which one wins depends entirely on the ratio of pages written to
//! pages watched, so both are here and both are measured.

use std::io;
use std::sync::Arc;

pub mod pageset;
pub mod region;

#[cfg(target_os = "macos")]
pub mod machtrack;
#[cfg(target_os = "linux")]
pub mod softdirty;
#[cfg(target_os = "linux")]
pub mod uffd;

pub use pageset::PageSet;
pub use region::{page_size, Region};

#[derive(Default, Clone, Copy, Debug)]
pub struct Stats {
    /// Write faults taken on the tracked region.
    pub faults: u64,
    /// Wall time spent inside the fault handler, in nanoseconds.
    pub fault_ns: u64,
    /// Pages marked dirty. Exceeds `faults` when the fault cluster is > 1.
    pub pages_marked: u64,
    pub harvests: u64,
    /// Wall time spent in `harvest`, including re-arming.
    pub rearm_ns: u64,
    /// Faults at addresses this tracker does not own. Should be zero; a nonzero
    /// count means the process was about to crash for an unrelated reason.
    pub foreign_faults: u64,
}

impl Stats {
    pub fn ns_per_fault(&self) -> f64 {
        if self.faults == 0 {
            0.0
        } else {
            self.fault_ns as f64 / self.faults as f64
        }
    }
    pub fn ns_per_harvest(&self) -> f64 {
        if self.harvests == 0 {
            0.0
        } else {
            self.rearm_ns as f64 / self.harvests as f64
        }
    }
}

pub trait Tracker: Send + Sync {
    /// Begin a tracking session: every page becomes clean, and the next write
    /// to any of them is observable.
    fn arm(&self) -> io::Result<()>;

    /// Move the pages written since the last `arm`/`harvest` into `out`, and
    /// start a new round. Pages may be reported more than once across rounds;
    /// they are never missed.
    fn harvest(&self, out: &PageSet) -> io::Result<()>;

    /// Stop tracking. The region becomes ordinary writable memory again.
    fn disarm(&self) -> io::Result<()>;

    fn stats(&self) -> Stats;
    fn name(&self) -> &'static str;
    fn region(&self) -> &Arc<Region>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// The best fault-driven tracker this platform has.
    Auto,
    Mach,
    Uffd,
    SoftDirty,
    None,
}

impl Kind {
    pub fn parse(s: &str) -> Option<Kind> {
        Some(match s {
            "auto" => Kind::Auto,
            "mach" => Kind::Mach,
            "uffd" => Kind::Uffd,
            "soft-dirty" | "softdirty" => Kind::SoftDirty,
            "none" => Kind::None,
            _ => return None,
        })
    }
}

/// Build a tracker over `region`.
///
/// `cluster` is how many pages a single fault makes writable. One is exact.
/// Larger values cut the number of faults at the cost of a coarser dirty set —
/// pages in the cluster must then be assumed dirty, because after the
/// permission is restored their writes are invisible.
pub fn open(region: Arc<Region>, kind: Kind, cluster: usize) -> io::Result<Box<dyn Tracker>> {
    match kind {
        Kind::None => Ok(Box::new(NoTracker { region })),
        Kind::Auto => {
            #[cfg(target_os = "macos")]
            {
                Ok(Box::new(machtrack::MachTracker::new(region, cluster)?))
            }
            #[cfg(target_os = "linux")]
            {
                uffd::UffdTracker::new(region, cluster).map(|t| Box::new(t) as Box<dyn Tracker>)
            }
            #[cfg(not(any(target_os = "macos", target_os = "linux")))]
            {
                let _ = cluster;
                Err(io::Error::other("no dirty-page tracker for this platform"))
            }
        }
        #[cfg(target_os = "macos")]
        Kind::Mach => Ok(Box::new(machtrack::MachTracker::new(region, cluster)?)),
        #[cfg(target_os = "linux")]
        Kind::Uffd => Ok(Box::new(uffd::UffdTracker::new(region, cluster)?)),
        #[cfg(target_os = "linux")]
        Kind::SoftDirty => Ok(Box::new(softdirty::SoftDirtyTracker::new(region)?)),
        other => Err(io::Error::other(format!(
            "{other:?} tracking is not available on {}",
            std::env::consts::OS
        ))),
    }
}

/// No tracking at all: every harvest reports the whole region.
///
/// This is not a degraded mode that silently kicks in — it has to be asked for
/// by name. It is the control case. Migrating with `none` is what pre-copy
/// looks like when you cannot tell what changed: correct, and it never
/// converges, which is the point of measuring it.
pub struct NoTracker {
    region: Arc<Region>,
}

impl Tracker for NoTracker {
    fn arm(&self) -> io::Result<()> {
        Ok(())
    }
    fn harvest(&self, out: &PageSet) -> io::Result<()> {
        out.mark_range(0, self.region.n_pages());
        Ok(())
    }
    fn disarm(&self) -> io::Result<()> {
        Ok(())
    }
    fn stats(&self) -> Stats {
        Stats::default()
    }
    fn name(&self) -> &'static str {
        "none"
    }
    fn region(&self) -> &Arc<Region> {
        &self.region
    }
}
