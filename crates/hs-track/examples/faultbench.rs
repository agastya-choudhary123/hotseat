//! What does dirty-page tracking actually cost?
//!
//! Three numbers matter for pre-copy migration:
//!   1. the cost of the *first* write to a clean page (the fault),
//!   2. the cost of re-arming a region at a round boundary,
//!   3. how much (1) is reduced by unprotecting more than one page per fault.

use hs_track::{open, page_size, Kind, PageSet, Region};
use std::sync::Arc;
use std::time::Instant;

fn touch_all(region: &Region, stride_pages: usize) {
    let ps = region.page_size();
    let n = region.n_pages();
    let mut p = 0;
    while p < n {
        unsafe { region.bytes_mut(p * ps, ps)[0] = 1 };
        p += stride_pages;
    }
}

fn main() {
    let mb: usize = std::env::args().nth(1).map_or(112, |s| s.parse().unwrap());
    let bytes = mb * (1 << 20);
    let ps = page_size();
    println!("region {mb} MiB, page size {ps} B, {} pages\n", bytes / ps);

    println!("{:<10} {:>12} {:>12} {:>12} {:>14} {:>12}", "cluster", "faults", "us/fault", "pages/s", "rearm us", "MiB/s dirty");
    for cluster in [1usize, 2, 4, 8, 16, 64] {
        let region = Arc::new(Region::new(bytes).unwrap());
        region.prefault();
        let t = open(region.clone(), Kind::Auto, cluster).unwrap();
        t.arm().unwrap();

        let t0 = Instant::now();
        touch_all(&region, 1);
        let el = t0.elapsed();

        let out = PageSet::new(region.n_pages());
        let r0 = Instant::now();
        t.harvest(&out).unwrap();
        let rearm = r0.elapsed();

        let s = t.stats();
        assert_eq!(out.count(), region.n_pages(), "every page was written");
        println!(
            "{:<10} {:>12} {:>12.2} {:>12.0} {:>14.1} {:>12.0}",
            cluster,
            s.faults,
            el.as_secs_f64() * 1e6 / s.faults.max(1) as f64,
            region.n_pages() as f64 / el.as_secs_f64(),
            rearm.as_secs_f64() * 1e6,
            (region.n_pages() * ps) as f64 / el.as_secs_f64() / (1 << 20) as f64,
        );
        drop(t);
    }

    // Re-arm cost on its own, against a region with no fragmentation at all.
    println!();
    let region = Arc::new(Region::new(bytes).unwrap());
    region.prefault();
    let t = open(region.clone(), Kind::Auto, 1).unwrap();
    t.arm().unwrap();
    let out = PageSet::new(region.n_pages());
    let mut clean = Vec::new();
    for _ in 0..20 {
        let r0 = Instant::now();
        t.harvest(&out).unwrap();
        clean.push(r0.elapsed().as_secs_f64() * 1e6);
    }
    clean.sort_by(f64::total_cmp);
    println!("re-arm of an untouched {mb} MiB region: median {:.1} us", clean[clean.len() / 2]);

    // And after every page has been individually unprotected.
    touch_all(&region, 1);
    let mut frag = Vec::new();
    for _ in 0..20 {
        let r0 = Instant::now();
        t.harvest(&out).unwrap();
        frag.push(r0.elapsed().as_secs_f64() * 1e6);
        touch_all(&region, 1);
    }
    frag.sort_by(f64::total_cmp);
    println!("re-arm after all {} pages faulted: median {:.1} us", region.n_pages(), frag[frag.len() / 2]);

    // Cost of a write that does NOT fault (page already dirty this round).
    let t1 = Instant::now();
    let reps = 200;
    for _ in 0..reps {
        touch_all(&region, 1);
    }
    let per = t1.elapsed().as_secs_f64() * 1e9 / (reps * region.n_pages()) as f64;
    println!("write to an already-dirty page: {per:.1} ns");
}
