//! A read-only file mapping. No crate for this — it is twenty lines of libc and
//! the rest of the project already lives at that level.

use std::os::unix::io::AsRawFd;
use std::path::Path;

pub struct Mmap {
    ptr: *const u8,
    len: usize,
}

// The mapping is read-only and never mutated after construction.
unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

impl Mmap {
    pub fn open(path: &Path) -> std::io::Result<Mmap> {
        let f = std::fs::File::open(path)?;
        let len = f.metadata()?.len() as usize;
        if len == 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "empty file"));
        }
        let ptr = unsafe {
            libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_PRIVATE, f.as_raw_fd(), 0)
        };
        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Mmap { ptr: ptr as *const u8, len })
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Tell the kernel we are about to stream the whole file. Load time on a
    /// cold page cache is dominated by faulting the weights in one 16 KiB page
    /// at a time; this turns that into readahead.
    pub fn will_need(&self) {
        unsafe {
            libc::madvise(self.ptr as *mut libc::c_void, self.len, libc::MADV_WILLNEED);
        }
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
    }
}
