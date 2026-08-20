//! The KV cache — the state that a migration actually moves.
//!
//! The cache is a flat, page-aligned byte range supplied by the caller, because
//! the whole point of this project is that something else (a dirty-page
//! tracker) owns the mapping and watches it. The engine only ever indexes into
//! it, so the tracker sees every write the model makes through ordinary stores,
//! with no cooperation from the engine and no software write log.
//!
//! # Layout, and why it is a migration decision
//!
//! Two layouts are supported, and the choice trades decode bandwidth against
//! migration cost:
//!
//! * `TokenMajor` — `[layer][k|v][pos][head][dim]`. One token's K for one layer
//!   is `n_kv_heads * head_dim` contiguous floats, so a decode step touches
//!   `2 * n_layers` distinct pages and each page absorbs the next several
//!   tokens before moving on. Attention then reads a head with a stride,
//!   fetching `head_dim` floats out of every `n_kv_heads * head_dim`.
//!
//! * `HeadMajor` — `[layer][k|v][head][pos][dim]`. Attention over positions is
//!   fully sequential, which is what the decode step wants; but a token's write
//!   is now `n_kv_heads` separate slices megabytes apart, so a step dirties
//!   `2 * n_layers * n_kv_heads` pages instead.
//!
//! Both are implemented and both are measured; see `docs/`.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvLayout {
    TokenMajor,
    HeadMajor,
}

impl KvLayout {
    pub fn parse(s: &str) -> Option<KvLayout> {
        match s {
            "token" | "token-major" => Some(KvLayout::TokenMajor),
            "head" | "head-major" => Some(KvLayout::HeadMajor),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            KvLayout::TokenMajor => "token-major",
            KvLayout::HeadMajor => "head-major",
        }
    }
    pub fn as_u8(self) -> u8 {
        match self {
            KvLayout::TokenMajor => 0,
            KvLayout::HeadMajor => 1,
        }
    }
    pub fn from_u8(v: u8) -> Option<KvLayout> {
        match v {
            0 => Some(KvLayout::TokenMajor),
            1 => Some(KvLayout::HeadMajor),
            _ => None,
        }
    }
}

/// Shape of a KV cache. Two workers must agree on this exactly before a
/// migration can mean anything, so it travels in the handshake.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvSpec {
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub max_ctx: usize,
    pub layout: KvLayout,
}

impl KvSpec {
    /// Floats in one (layer, k-or-v) plane.
    #[inline]
    pub const fn plane(&self) -> usize {
        self.max_ctx * self.n_kv_heads * self.head_dim
    }
    #[inline]
    pub const fn layer_stride(&self) -> usize {
        2 * self.plane()
    }
    #[inline]
    pub const fn floats(&self) -> usize {
        self.n_layers * self.layer_stride()
    }
    #[inline]
    pub const fn bytes(&self) -> usize {
        self.floats() * 4
    }

    /// Float offset of one (layer, plane, pos, head) row.
    #[inline(always)]
    pub fn row_off(&self, layer: usize, is_v: bool, pos: usize, head: usize) -> usize {
        debug_assert!(layer < self.n_layers && pos < self.max_ctx && head < self.n_kv_heads);
        let plane = layer * self.layer_stride() + if is_v { self.plane() } else { 0 };
        let within = match self.layout {
            KvLayout::TokenMajor => (pos * self.n_kv_heads + head) * self.head_dim,
            KvLayout::HeadMajor => (head * self.max_ctx + pos) * self.head_dim,
        };
        plane + within
    }

    /// The byte ranges a single decode step at `pos` writes.
    ///
    /// Used to check the dirty-page tracker against ground truth: if the
    /// hardware reports a page dirty, it had better appear here.
    pub fn written_ranges(&self, pos: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for layer in 0..self.n_layers {
            for &is_v in &[false, true] {
                match self.layout {
                    KvLayout::TokenMajor => {
                        // All heads for this position are contiguous.
                        out.push((
                            self.row_off(layer, is_v, pos, 0) * 4,
                            self.n_kv_heads * self.head_dim * 4,
                        ));
                    }
                    KvLayout::HeadMajor => {
                        for h in 0..self.n_kv_heads {
                            out.push((self.row_off(layer, is_v, pos, h) * 4, self.head_dim * 4));
                        }
                    }
                }
            }
        }
        out
    }

    /// Byte ranges covering positions `0..n_pos` — everything a migration has
    /// to send at least once. With `TokenMajor` the live set is a prefix of
    /// each plane; with `HeadMajor` it is `n_kv_heads` strided slices, which is
    /// why the two layouts do not cost the same to move.
    pub fn live_ranges(&self, n_pos: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        if n_pos == 0 {
            return out;
        }
        for layer in 0..self.n_layers {
            for &is_v in &[false, true] {
                let plane_off =
                    (layer * self.layer_stride() + if is_v { self.plane() } else { 0 }) * 4;
                match self.layout {
                    KvLayout::TokenMajor => {
                        out.push((plane_off, n_pos * self.n_kv_heads * self.head_dim * 4));
                    }
                    KvLayout::HeadMajor => {
                        for h in 0..self.n_kv_heads {
                            out.push((
                                plane_off + h * self.max_ctx * self.head_dim * 4,
                                n_pos * self.head_dim * 4,
                            ));
                        }
                    }
                }
            }
        }
        out
    }
}

impl fmt::Display for KvSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} layers x {} kv-heads x {} dim x {} ctx, {}, {:.1} MiB",
            self.n_layers,
            self.n_kv_heads,
            self.head_dim,
            self.max_ctx,
            self.layout.name(),
            self.bytes() as f64 / (1 << 20) as f64
        )
    }
}

/// A view over externally owned KV memory.
///
/// Not the owner: `hs-track` allocates the region so it can write-protect it.
pub struct Kv {
    pub spec: KvSpec,
    ptr: *mut f32,
}

// The engine drives one sequence at a time; the pool threads only read the
// cache during attention, and they read positions the writer has already
// finished with.
unsafe impl Send for Kv {}
unsafe impl Sync for Kv {}

impl Kv {
    /// # Safety
    /// `ptr` must point to at least `spec.bytes()` writable bytes, aligned to 4,
    /// and outlive the `Kv`.
    pub unsafe fn from_raw(ptr: *mut u8, spec: KvSpec) -> Kv {
        assert_eq!(ptr as usize % 4, 0, "KV region must be 4-byte aligned");
        Kv { spec, ptr: ptr as *mut f32 }
    }

    #[inline]
    pub fn base(&self) -> *mut u8 {
        self.ptr as *mut u8
    }

    #[inline(always)]
    fn row_off(&self, layer: usize, is_v: bool, pos: usize, head: usize) -> usize {
        self.spec.row_off(layer, is_v, pos, head)
    }

    #[inline(always)]
    pub fn k(&self, layer: usize, pos: usize, head: usize) -> &[f32] {
        unsafe {
            std::slice::from_raw_parts(
                self.ptr.add(self.row_off(layer, false, pos, head)),
                self.spec.head_dim,
            )
        }
    }

    #[inline(always)]
    pub fn v(&self, layer: usize, pos: usize, head: usize) -> &[f32] {
        unsafe {
            std::slice::from_raw_parts(
                self.ptr.add(self.row_off(layer, true, pos, head)),
                self.spec.head_dim,
            )
        }
    }

    /// # Safety
    /// The caller must not hand out two overlapping mutable rows at once.
    #[inline(always)]
    pub unsafe fn k_mut(&self, layer: usize, pos: usize, head: usize) -> &mut [f32] {
        std::slice::from_raw_parts_mut(
            self.ptr.add(self.row_off(layer, false, pos, head)),
            self.spec.head_dim,
        )
    }

    /// # Safety
    /// As `k_mut`.
    #[inline(always)]
    pub unsafe fn v_mut(&self, layer: usize, pos: usize, head: usize) -> &mut [f32] {
        std::slice::from_raw_parts_mut(
            self.ptr.add(self.row_off(layer, true, pos, head)),
            self.spec.head_dim,
        )
    }

    /// Content hash over the live prefix, for proving two caches are identical.
    pub fn hash(&self, n_pos: usize) -> u64 {
        let mut h = crate::hash::Wide::new();
        for (off, len) in self.spec.live_ranges(n_pos) {
            let bytes = unsafe { std::slice::from_raw_parts(self.base().add(off), len) };
            h.write(bytes);
        }
        h.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(layout: KvLayout) -> KvSpec {
        KvSpec { n_layers: 3, n_kv_heads: 2, head_dim: 8, max_ctx: 16, layout }
    }

    #[test]
    fn rows_are_disjoint_and_in_bounds() {
        for layout in [KvLayout::TokenMajor, KvLayout::HeadMajor] {
            let s = spec(layout);
            let mut seen = vec![0u32; s.floats()];
            let mut tag = 0u32;
            for l in 0..s.n_layers {
                for p in 0..s.max_ctx {
                    for h in 0..s.n_kv_heads {
                        for is_v in [false, true] {
                            tag += 1;
                            let off = s.row_off(l, is_v, p, h);
                            for i in 0..s.head_dim {
                                assert_eq!(seen[off + i], 0, "overlap at {}", off + i);
                                seen[off + i] = tag;
                            }
                        }
                    }
                }
            }
            assert!(seen.iter().all(|&t| t != 0), "{} left holes", layout.name());
        }
    }

    #[test]
    fn written_ranges_cover_exactly_the_step() {
        for layout in [KvLayout::TokenMajor, KvLayout::HeadMajor] {
            let s = spec(layout);
            let mut mem = vec![0f32; s.floats()];
            let kv = unsafe { Kv::from_raw(mem.as_mut_ptr() as *mut u8, s) };
            let pos = 5;
            // Write the step the way the model would.
            for l in 0..s.n_layers {
                for h in 0..s.n_kv_heads {
                    unsafe {
                        kv.k_mut(l, pos, h).fill(1.0);
                        kv.v_mut(l, pos, h).fill(1.0);
                    }
                }
            }
            // Every nonzero float must fall inside a declared range.
            let ranges = s.written_ranges(pos);
            let mut covered = vec![false; s.floats()];
            for (off, len) in &ranges {
                for i in 0..len / 4 {
                    covered[off / 4 + i] = true;
                }
            }
            let declared: usize = ranges.iter().map(|(_, l)| l / 4).sum();
            let actual = mem.iter().filter(|&&v| v != 0.0).count();
            assert_eq!(declared, actual, "{}", layout.name());
            for (i, &v) in mem.iter().enumerate() {
                assert_eq!(v != 0.0, covered[i], "{} float {i}", layout.name());
            }
        }
    }

    #[test]
    fn live_ranges_cover_every_written_position() {
        for layout in [KvLayout::TokenMajor, KvLayout::HeadMajor] {
            let s = spec(layout);
            let mut mem = vec![0f32; s.floats()];
            let kv = unsafe { Kv::from_raw(mem.as_mut_ptr() as *mut u8, s) };
            let n = 7;
            for p in 0..n {
                for l in 0..s.n_layers {
                    for h in 0..s.n_kv_heads {
                        unsafe {
                            kv.k_mut(l, p, h).fill(p as f32 + 1.0);
                            kv.v_mut(l, p, h).fill(p as f32 + 1.0);
                        }
                    }
                }
            }
            let mut covered = vec![false; s.floats()];
            for (off, len) in s.live_ranges(n) {
                for i in 0..len / 4 {
                    covered[off / 4 + i] = true;
                }
            }
            for (i, &v) in mem.iter().enumerate() {
                if v != 0.0 {
                    assert!(covered[i], "{} missed live float {i}", layout.name());
                }
            }
        }
    }

    #[test]
    fn token_major_dirties_fewer_pages_per_step() {
        // The claim the layout choice rests on, checked on a realistic shape.
        let base = KvSpec {
            n_layers: 28,
            n_kv_heads: 2,
            head_dim: 128,
            max_ctx: 2048,
            layout: KvLayout::TokenMajor,
        };
        let page = 16384usize;
        let pages = |s: KvSpec| {
            let mut set = std::collections::HashSet::new();
            for (off, len) in s.written_ranges(1000) {
                for p in (off / page)..=((off + len - 1) / page) {
                    set.insert(p);
                }
            }
            set.len()
        };
        let tm = pages(base);
        let hm = pages(KvSpec { layout: KvLayout::HeadMajor, ..base });
        assert!(tm < hm, "token-major {tm} pages, head-major {hm} pages");
    }
}
