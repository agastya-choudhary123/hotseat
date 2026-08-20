//! Behavioural tests for the dirty-page trackers.
//!
//! These run against whatever backend the platform provides, so the same file
//! is the acceptance test for the Mach implementation on macOS and the
//! userfaultfd one inside a container.
//!
//! A tracker takes over process-wide state (the task's exception ports on
//! macOS), so only one may exist at a time; every test holds `SERIAL` for as
//! long as its tracker is alive.

use hs_track::{open, page_size, Kind, PageSet, Region, Tracker};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

static SERIAL: Mutex<()> = Mutex::new(());

fn setup(pages: usize, cluster: usize) -> (Arc<Region>, Box<dyn Tracker>) {
    let region = Arc::new(Region::new(pages * page_size()).unwrap());
    region.prefault();
    let t = open(region.clone(), Kind::Auto, cluster).expect("open tracker");
    (region, t)
}

/// Store one byte into `page`, the way the model stores into its KV cache.
fn poke(region: &Region, page: usize, val: u8) {
    let ps = region.page_size();
    unsafe { region.bytes_mut(page * ps, ps)[7] = val };
}

#[test]
fn reports_exactly_the_pages_written() {
    let _g = SERIAL.lock().unwrap();
    let (region, t) = setup(64, 1);
    t.arm().unwrap();

    let written = [0usize, 1, 17, 63];
    for &p in &written {
        poke(&region, p, 0xAA);
    }

    let out = PageSet::new(region.n_pages());
    t.harvest(&out).unwrap();

    let got: Vec<usize> = out.runs().iter().flat_map(|&(a, n)| a..a + n).collect();
    assert_eq!(got, written, "backend {}", t.name());
    assert_eq!(t.stats().faults, 4);
}

#[test]
fn reads_never_fault() {
    let _g = SERIAL.lock().unwrap();
    let (region, t) = setup(32, 1);
    t.arm().unwrap();

    let mut acc = 0u64;
    let bytes = unsafe { region.bytes(0, region.len()) };
    for b in bytes.iter().step_by(97) {
        acc += *b as u64;
    }
    assert_eq!(acc, 0);

    let out = PageSet::new(region.n_pages());
    t.harvest(&out).unwrap();
    assert!(out.is_empty(), "a read dirtied a page on {}", t.name());
    assert_eq!(t.stats().faults, 0);
}

#[test]
fn each_round_reports_only_its_own_writes() {
    let _g = SERIAL.lock().unwrap();
    let (region, t) = setup(64, 1);
    t.arm().unwrap();

    poke(&region, 5, 1);
    let r1 = PageSet::new(region.n_pages());
    t.harvest(&r1).unwrap();
    assert_eq!(r1.runs(), vec![(5, 1)]);

    poke(&region, 9, 1);
    let r2 = PageSet::new(region.n_pages());
    t.harvest(&r2).unwrap();
    assert_eq!(r2.runs(), vec![(9, 1)], "round two should not repeat page 5");

    // Page 5 must fault again if it is written again — the harvest re-armed it.
    poke(&region, 5, 2);
    let r3 = PageSet::new(region.n_pages());
    t.harvest(&r3).unwrap();
    assert_eq!(r3.runs(), vec![(5, 1)]);
}

#[test]
fn a_quiet_round_is_empty() {
    let _g = SERIAL.lock().unwrap();
    let (region, t) = setup(16, 1);
    t.arm().unwrap();
    poke(&region, 3, 1);
    let a = PageSet::new(region.n_pages());
    t.harvest(&a).unwrap();
    assert_eq!(a.count(), 1);
    let b = PageSet::new(region.n_pages());
    t.harvest(&b).unwrap();
    assert!(b.is_empty());
}

#[test]
fn repeated_writes_to_one_page_fault_once() {
    let _g = SERIAL.lock().unwrap();
    let (region, t) = setup(16, 1);
    t.arm().unwrap();
    for i in 0..1000u32 {
        poke(&region, 4, i as u8);
    }
    let out = PageSet::new(region.n_pages());
    t.harvest(&out).unwrap();
    assert_eq!(out.runs(), vec![(4, 1)]);
    assert_eq!(t.stats().faults, 1, "only the first write should fault");
}

#[test]
fn disarm_stops_tracking_and_leaves_memory_writable() {
    let _g = SERIAL.lock().unwrap();
    let (region, t) = setup(16, 1);
    t.arm().unwrap();
    poke(&region, 2, 1);
    let a = PageSet::new(region.n_pages());
    t.harvest(&a).unwrap();
    assert_eq!(a.count(), 1);

    t.disarm().unwrap();
    let before = t.stats().faults;
    for p in 0..16 {
        poke(&region, p, 9);
    }
    assert_eq!(t.stats().faults, before, "disarmed tracker still took faults");
}

#[test]
fn clustering_reports_the_whole_cluster() {
    let _g = SERIAL.lock().unwrap();
    let (region, t) = setup(64, 8);
    if t.name() == "soft-dirty" {
        return; // scan-based tracking has no cluster knob
    }
    t.arm().unwrap();
    poke(&region, 10, 1); // cluster 8..16
    let out = PageSet::new(region.n_pages());
    t.harvest(&out).unwrap();
    assert_eq!(out.runs(), vec![(8, 8)]);
    assert_eq!(t.stats().faults, 1);

    // The rest of the cluster is now writable, so writing it costs no fault.
    t.arm().unwrap();
    let before = t.stats().faults;
    poke(&region, 10, 2);
    assert_eq!(t.stats().faults, before + 1, "arm should re-protect the cluster");
}

/// The one that matters: a writer running flat out while rounds are taken
/// underneath it. Every page it touched must appear in some round.
#[test]
fn no_write_is_lost_across_a_round_boundary() {
    let _g = SERIAL.lock().unwrap();
    let (region, t) = setup(512, 1);
    let t: Arc<Box<dyn Tracker>> = Arc::new(t);
    t.arm().unwrap();

    let stop = Arc::new(AtomicBool::new(false));
    let progress = Arc::new(AtomicUsize::new(0));
    let n_pages = region.n_pages();

    let w_region = region.clone();
    let w_stop = stop.clone();
    let w_progress = progress.clone();
    let writer = std::thread::spawn(move || {
        // A deterministic walk that revisits pages, so pages go clean->dirty
        // repeatedly and every round boundary has writes in flight.
        let mut x = 0x243F_6A88_85A3_08D3u64;
        let mut touched = vec![false; n_pages];
        let mut i = 0usize;
        while !w_stop.load(Ordering::Relaxed) {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let p = (x >> 33) as usize % n_pages;
            poke(&w_region, p, i as u8);
            touched[p] = true;
            i += 1;
            w_progress.store(i, Ordering::Relaxed);
        }
        touched
    });

    // Harvest on a round cadence, the way a migration does. A tight harvest
    // loop would also pass, but it spends all its time re-protecting a heavily
    // fragmented mapping and tells us nothing extra about the race.
    let union = PageSet::new(n_pages);
    let mut rounds = 0;
    while progress.load(Ordering::Relaxed) < 30_000 {
        let round = PageSet::new(n_pages);
        t.harvest(&round).unwrap();
        round.drain_into(&union);
        rounds += 1;
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    stop.store(true, Ordering::Relaxed);
    let touched = writer.join().unwrap();

    // Final round, after the writer is definitely finished.
    let last = PageSet::new(n_pages);
    t.harvest(&last).unwrap();
    last.drain_into(&union);

    assert!(rounds > 5, "test did not take enough rounds to be meaningful: {rounds}");
    let missed: Vec<usize> =
        (0..n_pages).filter(|&p| touched[p] && !union.contains(p)).collect();
    assert!(
        missed.is_empty(),
        "{} pages were written but never reported by {} over {rounds} rounds: {:?}",
        missed.len(),
        t.name(),
        &missed[..missed.len().min(20)]
    );
}

#[test]
fn none_backend_reports_everything() {
    let _g = SERIAL.lock().unwrap();
    let region = Arc::new(Region::new(16 * page_size()).unwrap());
    let t = open(region.clone(), Kind::None, 1).unwrap();
    t.arm().unwrap();
    let out = PageSet::new(region.n_pages());
    t.harvest(&out).unwrap();
    assert_eq!(out.count(), 16);
}
