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

/// A wide hash for large buffers.
///
/// Byte-at-a-time FNV runs at a few hundred MiB/s, and the destination of a
/// migration wants to check tens of megabytes of KV cache *inside* the
/// stop-the-world window. This walks four independent lanes of 64-bit words,
/// which vectorises and runs at memory speed, then folds them. Same reasoning
/// as `Fnv`: not cryptographic, just has to be stable and identical on both
/// ends, so it is written out rather than imported.
pub struct Wide {
    lanes: [u64; 4],
    len: u64,
}

impl Default for Wide {
    fn default() -> Self {
        Wide::new()
    }
}

impl Wide {
    pub const fn new() -> Wide {
        Wide {
            lanes: [
                0x243f_6a88_85a3_08d3,
                0x1319_8a2e_0370_7344,
                0xa409_3822_299f_31d0,
                0x082e_fa98_ec4e_6c89,
            ],
            len: 0,
        }
    }

    pub fn write(&mut self, bytes: &[u8]) {
        self.len = self.len.wrapping_add(bytes.len() as u64);
        let mut l = self.lanes;
        let mut chunks = bytes.chunks_exact(32);
        for c in &mut chunks {
            for k in 0..4 {
                let w = u64::from_le_bytes(c[k * 8..k * 8 + 8].try_into().unwrap());
                l[k] = (l[k] ^ w).wrapping_mul(PRIME);
                l[k] ^= l[k] >> 29;
            }
        }
        // Tail: fold the remaining bytes into lane 0 one at a time. The lane
        // assignment depends only on the byte offset, never on how the input
        // was chunked, so streaming and one-shot agree.
        let rem = chunks.remainder();
        for (i, &b) in rem.iter().enumerate() {
            let k = i % 4;
            l[k] = (l[k] ^ b as u64).wrapping_mul(PRIME);
        }
        self.lanes = l;
    }

    pub fn finish(&self) -> u64 {
        let mut h = self.lanes[0]
            ^ self.lanes[1].rotate_left(17)
            ^ self.lanes[2].rotate_left(34)
            ^ self.lanes[3].rotate_left(51);
        h = h.wrapping_add(self.len);
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        h
    }
}

pub fn hash_wide(bytes: &[u8]) -> u64 {
    let mut h = Wide::new();
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
    fn wide_hash_detects_single_bit_flips() {
        let mut data: Vec<u8> = (0..100_000u32).map(|i| (i * 31) as u8).collect();
        let a = hash_wide(&data);
        for at in [0usize, 1, 7, 8, 31, 32, 99_999] {
            data[at] ^= 1;
            assert_ne!(hash_wide(&data), a, "flip at {at} went unnoticed");
            data[at] ^= 1;
        }
        assert_eq!(hash_wide(&data), a);
    }

    #[test]
    fn wide_hash_is_length_sensitive() {
        assert_ne!(hash_wide(&[0u8; 64]), hash_wide(&[0u8; 96]));
        assert_ne!(hash_wide(b""), hash_wide(b"\0"));
    }

    #[test]
    fn wide_streaming_matches_one_shot_on_aligned_chunks() {
        let data: Vec<u8> = (0..8192u32).map(|i| (i * 7) as u8).collect();
        let mut h = Wide::new();
        for c in data.chunks(32 * 13) {
            h.write(c);
        }
        assert_eq!(h.finish(), hash_wide(&data));
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
