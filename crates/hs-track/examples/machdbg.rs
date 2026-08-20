//! Minimal trace of the Mach exception path, used to bring the tracker up.
use hs_track::{open, page_size, Kind, PageSet, Region};
use std::sync::Arc;

fn main() {
    let region = Arc::new(Region::new(8 * page_size()).unwrap());
    region.prefault();
    println!("region at {:#x}, {} pages of {}", region.addr(), region.n_pages(), region.page_size());
    let t = open(region.clone(), Kind::Auto, 1).unwrap();
    println!("tracker: {}", t.name());
    t.arm().unwrap();
    println!("armed; about to write page 3");
    std::thread::sleep(std::time::Duration::from_millis(100));
    unsafe { region.bytes_mut(3 * page_size(), page_size())[0] = 1 };
    println!("write returned");
    let out = PageSet::new(region.n_pages());
    t.harvest(&out).unwrap();
    println!("harvest: {:?}, stats {:?}", out.runs(), t.stats());
}
