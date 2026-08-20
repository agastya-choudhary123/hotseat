//! Dequantisation of the GGML k-quant block formats into f16 weights.
//!
//! Weights are stored as f16 rather than kept quantised. That is a deliberate
//! trade: a quantised dot product would be faster per byte, but this project's
//! subject is migration, and f16 weights give one thing that matters more here —
//! a single, simple, bit-exact numeric path that is identical on every host, so
//! "the token stream is identical across the migration boundary" is a claim
//! about the migration and not about two matmul kernels agreeing.
//!
//! The f16 rounding error (~2^-11 relative) sits well under the Q4_K
//! quantisation error it is applied on top of, so model quality is unchanged.

use crate::f16::f32_to_f16;
use crate::f16::f16_to_f32;
use crate::gguf::GgmlType;

/// Dequantise `n` elements of `src` into `dst` as f16 bit patterns.
pub fn dequant_to_f16(ty: GgmlType, src: &[u8], n: usize, dst: &mut [u16]) {
    assert_eq!(dst.len(), n, "destination must be exactly n elements");
    match ty {
        GgmlType::F16 => {
            for i in 0..n {
                dst[i] = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
            }
        }
        GgmlType::F32 => {
            for i in 0..n {
                let b = [src[i * 4], src[i * 4 + 1], src[i * 4 + 2], src[i * 4 + 3]];
                dst[i] = f32_to_f16(f32::from_le_bytes(b));
            }
        }
        GgmlType::Q5_0 => dequant_q5_0(src, n, dst),
        GgmlType::Q8_0 => dequant_q8_0(src, n, dst),
        GgmlType::Q4K => dequant_q4_k(src, n, dst),
        GgmlType::Q5K => dequant_q5_k(src, n, dst),
        GgmlType::Q6K => dequant_q6_k(src, n, dst),
    }
}

/// Dequantise into f32 (used for the few tensors kept at full precision, and by
/// the unit tests that check a block format against a hand-computed value).
pub fn dequant_to_f32(ty: GgmlType, src: &[u8], n: usize, dst: &mut [f32]) {
    let mut tmp = vec![0u16; n];
    match ty {
        GgmlType::F32 => {
            for i in 0..n {
                let b = [src[i * 4], src[i * 4 + 1], src[i * 4 + 2], src[i * 4 + 3]];
                dst[i] = f32::from_le_bytes(b);
            }
            return;
        }
        _ => dequant_to_f16(ty, src, n, &mut tmp),
    }
    for i in 0..n {
        dst[i] = f16_to_f32(tmp[i]);
    }
}

#[inline(always)]
fn half(b: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[at], b[at + 1]]))
}

fn dequant_q8_0(src: &[u8], n: usize, dst: &mut [u16]) {
    const QK: usize = 32;
    const BS: usize = 34;
    for (bi, blk) in src.chunks_exact(BS).take(n / QK).enumerate() {
        let d = half(blk, 0);
        for l in 0..QK {
            let q = blk[2 + l] as i8 as f32;
            dst[bi * QK + l] = f32_to_f16(d * q);
        }
    }
}

/// Q5_0: a 32-element block whose fifth bits live in a separate 32-bit word.
/// The two nibbles of byte `j` are elements `j` and `j + 16` — the halves are
/// interleaved, not consecutive.
fn dequant_q5_0(src: &[u8], n: usize, dst: &mut [u16]) {
    const QK: usize = 32;
    const BS: usize = 22;
    for (bi, blk) in src.chunks_exact(BS).take(n / QK).enumerate() {
        let d = half(blk, 0);
        let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
        let qs = &blk[6..22];
        let base = bi * QK;
        for j in 0..QK / 2 {
            let h0 = ((qh >> j) << 4) as u8 & 0x10;
            let h1 = (qh >> (j + 12)) as u8 & 0x10;
            let x0 = ((qs[j] & 0x0F) | h0) as i32 - 16;
            let x1 = ((qs[j] >> 4) | h1) as i32 - 16;
            dst[base + j] = f32_to_f16(d * x0 as f32);
            dst[base + j + QK / 2] = f32_to_f16(d * x1 as f32);
        }
    }
}

/// The 6-bit packed scale/min pair for sub-block `j` of a Q4_K super-block.
/// This is ggml's `get_scale_min_k4`, spelled out.
#[inline(always)]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

fn dequant_q4_k(src: &[u8], n: usize, dst: &mut [u16]) {
    const QK: usize = 256;
    const BS: usize = 144; // 2 (d) + 2 (dmin) + 12 (scales) + 128 (quants)
    for (bi, blk) in src.chunks_exact(BS).take(n / QK).enumerate() {
        let d = half(blk, 0);
        let dmin = half(blk, 2);
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        let base = bi * QK;
        let mut o = 0usize; // element within the super-block
        let mut is = 0usize; // sub-block index
        for g in 0..4 {
            // Each group of 64 elements shares one 32-byte slice of `qs`, with
            // the low nibbles forming the first 32 values and the high nibbles
            // the next 32 — each half carrying its own scale and min.
            let q = &qs[g * 32..g * 32 + 32];
            let (sc1, m1) = scale_min_k4(is, scales);
            let (sc2, m2) = scale_min_k4(is + 1, scales);
            let (d1, mm1) = (d * sc1 as f32, dmin * m1 as f32);
            let (d2, mm2) = (d * sc2 as f32, dmin * m2 as f32);
            for l in 0..32 {
                dst[base + o + l] = f32_to_f16(d1 * (q[l] & 0x0F) as f32 - mm1);
            }
            for l in 0..32 {
                dst[base + o + 32 + l] = f32_to_f16(d2 * (q[l] >> 4) as f32 - mm2);
            }
            o += 64;
            is += 2;
        }
    }
}

/// Q5_K: Q4_K plus one extra bit per element, held in a 32-byte plane whose
/// bit position advances by two every 64 elements.
fn dequant_q5_k(src: &[u8], n: usize, dst: &mut [u16]) {
    const QK: usize = 256;
    const BS: usize = 176; // 2 + 2 + 12 (scales) + 32 (high bits) + 128 (nibbles)
    for (bi, blk) in src.chunks_exact(BS).take(n / QK).enumerate() {
        let d = half(blk, 0);
        let dmin = half(blk, 2);
        let scales = &blk[4..16];
        let qh = &blk[16..48];
        let ql = &blk[48..176];
        let base = bi * QK;
        let mut o = 0usize;
        let mut is = 0usize;
        let (mut u1, mut u2) = (1u8, 2u8);
        for g in 0..4 {
            let q = &ql[g * 32..g * 32 + 32];
            let (sc1, m1) = scale_min_k4(is, scales);
            let (sc2, m2) = scale_min_k4(is + 1, scales);
            let (d1, mm1) = (d * sc1 as f32, dmin * m1 as f32);
            let (d2, mm2) = (d * sc2 as f32, dmin * m2 as f32);
            for l in 0..32 {
                let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
                dst[base + o + l] = f32_to_f16(d1 * ((q[l] & 0x0F) + hi) as f32 - mm1);
            }
            for l in 0..32 {
                let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
                dst[base + o + 32 + l] = f32_to_f16(d2 * ((q[l] >> 4) + hi) as f32 - mm2);
            }
            o += 64;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

fn dequant_q6_k(src: &[u8], n: usize, dst: &mut [u16]) {
    const QK: usize = 256;
    const BS: usize = 210; // 128 (low nibbles) + 64 (high bits) + 16 (scales) + 2 (d)
    for (bi, blk) in src.chunks_exact(BS).take(n / QK).enumerate() {
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let sc = &blk[192..208];
        let d = half(blk, 208);
        let base = bi * QK;
        for h in 0..2 {
            // Two halves of 128 elements; each consumes 64 ql, 32 qh, 8 scales.
            let ql = &ql[h * 64..h * 64 + 64];
            let qh = &qh[h * 32..h * 32 + 32];
            let sc = &sc[h * 8..h * 8 + 8];
            let out = base + h * 128;
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                dst[out + l] = f32_to_f16(d * sc[is] as i8 as f32 * q1 as f32);
                dst[out + l + 32] = f32_to_f16(d * sc[is + 2] as i8 as f32 * q2 as f32);
                dst[out + l + 64] = f32_to_f16(d * sc[is + 4] as i8 as f32 * q3 as f32);
                dst[out + l + 96] = f32_to_f16(d * sc[is + 6] as i8 as f32 * q4 as f32);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_round_trips_a_known_block() {
        let mut blk = vec![0u8; 34];
        blk[0..2].copy_from_slice(&crate::f16::f32_to_f16(0.5).to_le_bytes());
        for l in 0..32 {
            blk[2 + l] = (l as i8 - 16) as u8;
        }
        let mut out = vec![0u16; 32];
        dequant_q8_0(&blk, 32, &mut out);
        for l in 0..32 {
            assert_eq!(f16_to_f32(out[l]), 0.5 * (l as i32 - 16) as f32, "lane {l}");
        }
    }

    #[test]
    fn q4_k_scale_min_unpacking_matches_reference() {
        // Reference values computed from ggml's get_scale_min_k4 by hand.
        let q: [u8; 12] = [0b11_000001, 0b10_000010, 0b01_000011, 0b00_000100, 0x55, 0x66, 0x77, 0x88, 0x9A, 0xBC, 0xDE, 0xF0];
        assert_eq!(scale_min_k4(0, &q), (1, 0x55 & 63));
        assert_eq!(scale_min_k4(3, &q), (4, 0x88 & 63));
        // j >= 4 splices the top two bits of q[j-4] / q[j] onto a low nibble.
        assert_eq!(scale_min_k4(4, &q), ((0x9A & 0x0F) | (0b11 << 4), (0x9A >> 4) | ((0x55 >> 6) << 4)));
        assert_eq!(scale_min_k4(7, &q), ((0xF0 & 0x0F) | (0b00 << 4), (0xF0 >> 4) | ((0x88 >> 6) << 4)));
    }

    #[test]
    fn q4_k_block_is_affine_in_the_nibble() {
        // d = 1.0, dmin = 0 -> every value should be scale * nibble.
        let mut blk = vec![0u8; 144];
        blk[0..2].copy_from_slice(&crate::f16::f32_to_f16(1.0).to_le_bytes());
        blk[2..4].copy_from_slice(&crate::f16::f32_to_f16(0.0).to_le_bytes());
        for i in 0..12 {
            blk[4 + i] = 0; // all scales zero except the ones we set
        }
        blk[4] = 3; // sub-block 0 scale = 3
        for l in 0..128 {
            blk[16 + l] = 0x21; // low nibble 1, high nibble 2
        }
        let mut out = vec![0u16; 256];
        dequant_q4_k(&blk, 256, &mut out);
        assert_eq!(f16_to_f32(out[0]), 3.0); // low nibble 1 * scale 3
        assert_eq!(f16_to_f32(out[31]), 3.0);
        assert_eq!(f16_to_f32(out[32]), 0.0); // sub-block 1 scale is 0
        assert_eq!(f16_to_f32(out[64]), 0.0);
    }

    #[test]
    fn q5_0_interleaves_its_halves_and_uses_the_fifth_bit() {
        let mut blk = vec![0u8; 22];
        blk[0..2].copy_from_slice(&crate::f16::f32_to_f16(1.0).to_le_bytes());
        // The fifth bit of element `i` is qh bit `i`: bit 0 for element 0,
        // bit 16 (byte 4, bit 0) for element 16.
        blk[2] = 0b0000_0001;
        blk[4] = 0b0000_0001;
        for j in 0..16 {
            blk[6 + j] = 0x10; // low nibble 0, high nibble 1
        }
        let mut out = vec![0u16; 32];
        dequant_q5_0(&blk, 32, &mut out);
        assert_eq!(f16_to_f32(out[0]), (0 + 16 - 16) as f32);
        assert_eq!(f16_to_f32(out[1]), (0 - 16) as f32);
        assert_eq!(f16_to_f32(out[16]), (1 + 16 - 16) as f32);
        assert_eq!(f16_to_f32(out[17]), (1 - 16) as f32);
    }

    #[test]
    fn q5_k_adds_sixteen_when_the_high_bit_is_set() {
        let mut blk = vec![0u8; 176];
        blk[0..2].copy_from_slice(&crate::f16::f32_to_f16(1.0).to_le_bytes());
        blk[2..4].copy_from_slice(&crate::f16::f32_to_f16(0.0).to_le_bytes());
        blk[4] = 1; // sub-block 0 scale = 1
        blk[16] = 0b0000_0001; // element 0 high bit set
        for l in 0..128 {
            blk[48 + l] = 0x03;
        }
        let mut out = vec![0u16; 256];
        dequant_q5_k(&blk, 256, &mut out);
        assert_eq!(f16_to_f32(out[0]), 19.0, "3 + 16");
        assert_eq!(f16_to_f32(out[1]), 3.0);
    }

    #[test]
    fn q6_k_centres_on_zero() {
        // All quant bits zero -> every value is (0 - 32) * scale * d.
        let mut blk = vec![0u8; 210];
        for i in 0..16 {
            blk[192 + i] = 1i8 as u8;
        }
        blk[208..210].copy_from_slice(&crate::f16::f32_to_f16(0.25).to_le_bytes());
        let mut out = vec![0u16; 256];
        dequant_q6_k(&blk, 256, &mut out);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(f16_to_f32(v), -8.0, "lane {i}");
        }
    }
}
