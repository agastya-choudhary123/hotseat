//! A page-aligned anonymous mapping that a tracker can write-protect.
//!
//! The engine's KV cache lives in one of these. It is allocated here rather
//! than by the engine because the tracker has to own the protection state of
//! the mapping, and because the migration path wants to hand the kernel whole
//! pages without caring where a `Vec` happened to land.

use std::io;

pub struct Region {
    ptr: *mut u8,
    len: usize,
    page: usize,
}

// The region is a plain byte range; synchronisation of its *contents* is the
// caller's problem, exactly as it would be for a raw pointer.
unsafe impl Send for Region {}
unsafe impl Sync for Region {}

impl Region {
    /// Reserve at least `len` bytes, rounded up to a whole number of pages.
    pub fn new(len: usize) -> io::Result<Region> {
        let page = page_size();
        let len = len.div_ceil(page) * page;
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        debug_assert_eq!(ptr as usize % page, 0, "mmap returned an unaligned page");
        Ok(Region { ptr: ptr as *mut u8, len, page })
    }

    #[inline]
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }
    #[inline]
    pub fn addr(&self) -> usize {
        self.ptr as usize
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[inline]
    pub fn page_size(&self) -> usize {
        self.page
    }
    #[inline]
    pub fn n_pages(&self) -> usize {
        self.len / self.page
    }

    /// Whether `addr` falls inside this region, and if so at which page.
    #[inline]
    pub fn page_of(&self, addr: usize) -> Option<usize> {
        let base = self.ptr as usize;
        if addr >= base && addr < base + self.len {
            Some((addr - base) / self.page)
        } else {
            None
        }
    }

    /// # Safety
    /// The caller must ensure no other reference to `off..off+len` is live.
    #[inline]
    pub unsafe fn bytes(&self, off: usize, len: usize) -> &[u8] {
        debug_assert!(off + len <= self.len);
        std::slice::from_raw_parts(self.ptr.add(off), len)
    }

    /// # Safety
    /// As `bytes`.
    #[inline]
    pub unsafe fn bytes_mut(&self, off: usize, len: usize) -> &mut [u8] {
        debug_assert!(off + len <= self.len);
        std::slice::from_raw_parts_mut(self.ptr.add(off), len)
    }

    /// Fault every page in now, so that later timings measure the tracker and
    /// not first-touch zero-fill.
    pub fn prefault(&self) {
        unsafe {
            std::ptr::write_bytes(self.ptr, 0, self.len);
        }
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
    }
}

pub fn page_size() -> usize {
    // 16 KiB on Apple silicon, 4 KiB on most Linux arm64 and x86_64 builds.
    // Nothing in this project assumes which.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounds_up_to_pages_and_is_writable() {
        let r = Region::new(1).unwrap();
        assert_eq!(r.len(), r.page_size());
        assert_eq!(r.n_pages(), 1);
        unsafe { r.bytes_mut(0, r.len())[r.len() - 1] = 0xAB };
        assert_eq!(unsafe { r.bytes(0, r.len()) }[r.len() - 1], 0xAB);
    }

    #[test]
    fn page_of_maps_addresses_and_rejects_outsiders() {
        let r = Region::new(4 * page_size()).unwrap();
        assert_eq!(r.page_of(r.addr()), Some(0));
        assert_eq!(r.page_of(r.addr() + r.page_size()), Some(1));
        assert_eq!(r.page_of(r.addr() + r.len() - 1), Some(3));
        assert_eq!(r.page_of(r.addr() + r.len()), None);
        assert_eq!(r.page_of(r.addr() - 1), None);
    }

    #[test]
    fn fresh_pages_are_zero() {
        let r = Region::new(64 * 1024).unwrap();
        assert!(unsafe { r.bytes(0, r.len()) }.iter().all(|&b| b == 0));
    }
}
