//! A set of page indices, as a bitmap.
//!
//! Kept as a bitmap rather than a list because the fault handler has to mark a
//! page from inside a Mach exception message with no allocation and no lock —
//! one atomic `fetch_or` on a word does it. A 112 MiB cache at 16 KiB pages is
//! 7168 pages, so the whole set is under a kilobyte and clearing it is trivial.

use std::sync::atomic::{AtomicU64, Ordering};

pub struct PageSet {
    words: Vec<AtomicU64>,
    n_pages: usize,
}

impl PageSet {
    pub fn new(n_pages: usize) -> PageSet {
        PageSet {
            words: (0..n_pages.div_ceil(64)).map(|_| AtomicU64::new(0)).collect(),
            n_pages,
        }
    }

    #[inline]
    pub fn n_pages(&self) -> usize {
        self.n_pages
    }

    /// Mark `page` dirty. Safe from a signal or exception handler: one atomic
    /// read-modify-write, no allocation, no lock.
    #[inline]
    pub fn mark(&self, page: usize) {
        debug_assert!(page < self.n_pages);
        self.words[page / 64].fetch_or(1u64 << (page % 64), Ordering::Release);
    }

    pub fn mark_range(&self, first: usize, count: usize) {
        for p in first..(first + count).min(self.n_pages) {
            self.mark(p);
        }
    }

    #[inline]
    pub fn contains(&self, page: usize) -> bool {
        page < self.n_pages
            && self.words[page / 64].load(Ordering::Acquire) & (1u64 << (page % 64)) != 0
    }

    pub fn clear(&self) {
        for w in &self.words {
            w.store(0, Ordering::Release);
        }
    }

    pub fn count(&self) -> usize {
        self.words.iter().map(|w| w.load(Ordering::Acquire).count_ones() as usize).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|w| w.load(Ordering::Acquire) == 0)
    }

    /// Move every set bit into `dst`, leaving this set empty. The two sets must
    /// have the same shape.
    pub fn drain_into(&self, dst: &PageSet) {
        assert_eq!(self.n_pages, dst.n_pages);
        for (i, w) in self.words.iter().enumerate() {
            let v = w.swap(0, Ordering::AcqRel);
            if v != 0 {
                dst.words[i].fetch_or(v, Ordering::Release);
            }
        }
    }

    /// Set bits as maximal runs of consecutive pages: `(first_page, count)`.
    ///
    /// Runs, not individual pages, because both the sender and the receiver
    /// want to move one large contiguous chunk rather than thousands of small
    /// ones — and a KV cache dirties long runs.
    pub fn runs(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut start: Option<usize> = None;
        for i in 0..self.words.len() {
            let mut w = self.words[i].load(Ordering::Acquire);
            if w == 0 {
                if let Some(s) = start.take() {
                    out.push((s, i * 64 - s));
                }
                continue;
            }
            // Walk the word bit by bit; the bitmap is small enough that the
            // clever version would not pay for itself.
            for b in 0..64 {
                let page = i * 64 + b;
                if page >= self.n_pages {
                    break;
                }
                let set = w & 1 == 1;
                w >>= 1;
                match (set, start) {
                    (true, None) => start = Some(page),
                    (false, Some(s)) => {
                        out.push((s, page - s));
                        start = None;
                    }
                    _ => {}
                }
            }
        }
        if let Some(s) = start {
            out.push((s, self.n_pages - s));
        }
        out
    }

    /// Total bytes the set covers at the given page size.
    pub fn bytes(&self, page_size: usize) -> usize {
        self.count() * page_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marks_and_counts() {
        let s = PageSet::new(200);
        assert!(s.is_empty());
        s.mark(0);
        s.mark(63);
        s.mark(64);
        s.mark(199);
        assert_eq!(s.count(), 4);
        assert!(s.contains(64));
        assert!(!s.contains(65));
        s.clear();
        assert!(s.is_empty());
    }

    #[test]
    fn runs_coalesce_across_word_boundaries() {
        let s = PageSet::new(200);
        for p in 60..70 {
            s.mark(p);
        }
        s.mark(100);
        s.mark(102);
        assert_eq!(s.runs(), vec![(60, 10), (100, 1), (102, 1)]);
    }

    #[test]
    fn a_full_set_is_one_run() {
        let s = PageSet::new(130);
        s.mark_range(0, 130);
        assert_eq!(s.runs(), vec![(0, 130)]);
        assert_eq!(s.count(), 130);
    }

    #[test]
    fn runs_reconstruct_the_membership_exactly() {
        let s = PageSet::new(500);
        let mut want = vec![false; 500];
        let mut x = 12345u64;
        for p in 0..500 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            if x >> 62 == 0 {
                s.mark(p);
                want[p] = true;
            }
        }
        let mut got = vec![false; 500];
        for (first, n) in s.runs() {
            for p in first..first + n {
                assert!(!got[p], "page {p} in two runs");
                got[p] = true;
            }
        }
        assert_eq!(got, want);
    }

    #[test]
    fn drain_moves_every_bit() {
        let a = PageSet::new(300);
        let b = PageSet::new(300);
        a.mark(1);
        a.mark(299);
        b.mark(7);
        a.drain_into(&b);
        assert!(a.is_empty());
        assert_eq!(b.count(), 3);
        assert!(b.contains(1) && b.contains(7) && b.contains(299));
    }
}
