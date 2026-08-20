fn main(){
  let d = vec![7u8; 64<<20];
  let t = std::time::Instant::now();
  let h = hs_engine::hash::hash_wide(&d);
  let el = t.elapsed().as_secs_f64();
  println!("wide: {:.2} GiB/s (h={h:#x})", 64.0/1024.0/el);
  let t = std::time::Instant::now();
  let h = hs_engine::hash::hash(&d[..8<<20]);
  let el = t.elapsed().as_secs_f64();
  println!("fnv:  {:.2} GiB/s (h={h:#x})", 8.0/1024.0/el);
}
