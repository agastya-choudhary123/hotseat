//! FNV-1a, 64-bit.
//!
//! Used to answer one question: are these two byte ranges the same? It is not a
//! cryptographic hash and nothing here needs one — the peers authenticate
//! separately, and this is checking for corruption and for divergence, not for
//! an adversary. It is here rather than from a crate so that the digest is
//! stable forever and identical on both ends of a migration.

pub const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Clone, Copy, Debug)]
pub struct Fnv(u64);

impl Default for Fnv {
    fn default() -> Self {
        Fnv::new()
    }
}

impl Fnv {
    #[inline]
    pub const fn new() -> Fnv {
        Fnv(OFFSET)
    }
    #[inline]
    pub fn write(&mut self, bytes: &[u8]) {
        let mut h = self.0;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(PRIME);
        }
        self.0 = h;
    }
    #[inline]
    pub fn write_u64(&mut self, v: u64) {
        self.write(&v.to_le_bytes());
    }
    #[inline]
    pub fn write_u32(&mut self, v: u32) {
        self.write(&v.to_le_bytes());
    }
    #[inline]
    pub const fn finish(&self) -> u64 {
        self.0
    }
}

pub fn hash(bytes: &[u8]) -> u64 {
    let mut h = Fnv::new();
    h.write(bytes);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_published_vectors() {
        assert_eq!(hash(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(hash(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(hash(b"foobar"), 0x85944171f73967e8);
    }

    #[test]
    fn streaming_equals_one_shot() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 7) as u8).collect();
        let mut h = Fnv::new();
        for c in data.chunks(37) {
            h.write(c);
        }
        assert_eq!(h.finish(), hash(&data));
    }
}
