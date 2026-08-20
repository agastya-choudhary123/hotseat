//! IEEE-754 binary16 <-> binary32, done in integer arithmetic.
//!
//! Every worker in a migration must agree bit-for-bit on what a weight is, and
//! the two ends may be running different operating systems and different libm
//! implementations. Anything that touches numerics here is written out by hand
//! so the answer depends only on the instruction set, never on the host.

/// f16 bits -> f32. Lossless; every f16 is exactly representable in f32.
#[inline(always)]
pub const fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h as u32) >> 10) & 0x1f;
    let man = (h as u32) & 0x3ff;
    if exp == 0 {
        if man == 0 {
            return f32::from_bits(sign); // +/- 0
        }
        // Subnormal f16: renormalise into f32's exponent range. `shift` is the
        // number of leading zeros *within the 10-bit field*, so the mantissa's
        // top set bit lands on the implicit one.
        let shift = man.leading_zeros() - 22; // man < 2^10, so this is 0..=9
        let e = 127 - 15 - shift;
        let m = (man << (shift + 1)) & 0x3ff;
        return f32::from_bits(sign | (e << 23) | (m << 13));
    }
    if exp == 0x1f {
        // inf / NaN
        return f32::from_bits(sign | 0x7f80_0000 | (man << 13));
    }
    f32::from_bits(sign | ((exp + (127 - 15)) << 23) | (man << 13))
}

/// f32 -> f16 bits, round-to-nearest-even, with overflow to infinity.
#[inline(always)]
pub const fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let mut exp = ((x >> 23) & 0xff) as i32;
    let man = x & 0x7f_ffff;

    if exp == 0xff {
        // inf / NaN. Preserve NaN-ness (a zero mantissa would turn NaN into inf).
        let m = (man >> 13) as u16;
        return sign | 0x7c00 | if man != 0 && m == 0 { 1 } else { m };
    }

    exp -= 127 - 15;
    if exp >= 0x1f {
        return sign | 0x7c00; // overflow -> inf
    }
    if exp <= 0 {
        if exp < -10 {
            return sign; // underflow -> zero
        }
        // Subnormal: shift the implicit 1 back in, then round to nearest even.
        let m = man | 0x80_0000;
        let shift = (14 - exp) as u32;
        let half = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let tie = 1u32 << (shift - 1);
        let round = (rem > tie || (rem == tie && (half & 1) == 1)) as u32;
        return sign | (half + round) as u16;
    }

    let half = ((exp as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    let round = (rem > 0x1000 || (rem == 0x1000 && (man >> 13) & 1 == 1)) as u32;
    (half + round) as u16 | sign
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_exact_values() {
        for &v in &[0.0f32, -0.0, 1.0, -1.0, 0.5, 2.0, 65504.0, -65504.0, 6.1035156e-5] {
            assert_eq!(f16_to_f32(f32_to_f16(v)), v, "{v}");
        }
    }

    #[test]
    fn saturates_and_underflows() {
        assert_eq!(f32_to_f16(1.0e30) & 0x7fff, 0x7c00);
        assert_eq!(f32_to_f16(-1.0e30), 0xfc00);
        assert_eq!(f32_to_f16(1.0e-30), 0x0000);
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
    }

    #[test]
    fn subnormals_survive_the_round_trip() {
        // Smallest f16 subnormal and a few multiples of it.
        for k in 1..16u16 {
            let f = f16_to_f32(k);
            assert_eq!(f32_to_f16(f), k, "subnormal {k}");
        }
    }

    #[test]
    fn rounds_to_nearest_even() {
        // 2049 sits exactly halfway between 2048 and 2050 in f16; ties go to even.
        assert_eq!(f16_to_f32(f32_to_f16(2049.0)), 2048.0);
        assert_eq!(f16_to_f32(f32_to_f16(2051.0)), 2052.0);
    }
}
