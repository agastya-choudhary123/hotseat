//! Numeric kernels, all of them bit-reproducible.
//!
//! Two rules hold everywhere in this file, and the migration determinism claim
//! rests on both:
//!
//! 1. **Fixed reduction order.** Every reduction has one accumulation shape,
//!    written out explicitly, and the scalar and NEON implementations use the
//!    *same* shape. Whether the compiler vectorises, how many threads run, and
//!    which thread finishes first cannot change a single bit of the result.
//! 2. **No libm.** `expf` on macOS and `expf` in glibc differ in the last
//!    place. A sequence that migrates from a Mac to a Linux container would
//!    diverge on the first softmax. So exp is implemented here.

use crate::f16::f16_to_f32;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Number of elements accumulated in parallel by every dot product. Chosen to
/// match two 128-bit NEON accumulators; the scalar path mirrors it exactly.
pub const LANES: usize = 8;

// ---------------------------------------------------------------------------
// exp
// ---------------------------------------------------------------------------

const LOG2_E: f32 = 1.442_695_04;
// ln(2) split so that HI is exact in f32 and the product k*HI is exact too.
const LN2_HI: f32 = 0.693_359_375;
const LN2_LO: f32 = -2.121_944_4e-4;

/// e^x, to within 1 ulp, identical on every host with IEEE f32.
///
/// Range reduction x = k*ln2 + r, a degree-6 Horner polynomial on
/// r in [-ln2/2, ln2/2], then scaling by 2^k. The scale is applied in two
/// halves so the intermediate never overflows or flushes on the way to a
/// representable answer.
#[inline]
pub fn exp_f32(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x >= 89.0 {
        return f32::INFINITY;
    }
    if x <= -104.0 {
        return 0.0;
    }
    let kf = (x * LOG2_E).round();
    let r = x - kf * LN2_HI - kf * LN2_LO;
    // e^r on |r| <= 0.3466
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (0.166_666_67
                    + r * (0.041_666_67 + r * (0.008_333_33 + r * 0.001_388_889)))));
    let k = kf as i32;
    let k1 = k >> 1;
    let k2 = k - k1;
    p * pow2i(k1) * pow2i(k2)
}

/// 2^k for k in [-149, 127], built from the exponent field directly.
#[inline(always)]
fn pow2i(k: i32) -> f32 {
    let e = k + 127;
    if e <= 0 {
        // Subnormal or zero; construct from the mantissa instead.
        if e <= -23 {
            return 0.0;
        }
        return f32::from_bits(1u32 << (22 + e));
    }
    f32::from_bits((e as u32) << 23)
}

/// x / (1 + e^-x) — the SiLU / swish activation.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + exp_f32(-x))
}

// ---------------------------------------------------------------------------
// dot products
// ---------------------------------------------------------------------------

/// The one and only reduction tree used to fold `LANES` partial sums into one.
#[inline(always)]
fn fold(a: [f32; LANES]) -> f32 {
    let l = (a[0] + a[1]) + (a[2] + a[3]);
    let r = (a[4] + a[5]) + (a[6] + a[7]);
    l + r
}

/// Dot product of an f16 weight row with an f32 activation vector.
///
/// `w.len()` and `x.len()` must be equal and a multiple of `LANES`. Every
/// tensor dimension in this model already is (128, 256, 896, 1536, 4864, 8960,
/// 151936), so there is no tail path to get subtly wrong.
#[inline]
pub fn dot_f16(w: &[u16], x: &[f32]) -> f32 {
    debug_assert_eq!(w.len(), x.len());
    debug_assert_eq!(w.len() % LANES, 0);
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_f16_neon(w, x) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_f16_scalar(w, x)
    }
}

/// Reference implementation. Kept compiled in on every target: the NEON path is
/// tested against it, so "these two agree bit for bit" is a checked property
/// and not a comment.
pub fn dot_f16_scalar(w: &[u16], x: &[f32]) -> f32 {
    let mut acc = [0f32; LANES];
    let n = w.len();
    let mut i = 0;
    while i < n {
        for l in 0..LANES {
            acc[l] += f16_to_f32(w[i + l]) * x[i + l];
        }
        i += LANES;
    }
    fold(acc)
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_f16_neon(w: &[u16], x: &[f32]) -> f32 {
    let n = w.len();
    let mut a0 = vdupq_n_f32(0.0);
    let mut a1 = vdupq_n_f32(0.0);
    let mut wp = w.as_ptr();
    let mut xp = x.as_ptr();
    let mut i = 0;
    while i < n {
        // FCVTL widens f16 -> f32 exactly; there is no rounding to disagree on.
        let h = vld1q_u16(wp);
        let w0 = vcvt_f32_f16(std::mem::transmute::<uint16x4_t, float16x4_t>(vget_low_u16(h)));
        let w1 = vcvt_f32_f16(std::mem::transmute::<uint16x4_t, float16x4_t>(vget_high_u16(h)));
        // Separate mul and add, never FMA: the scalar mirror cannot form one,
        // and a fused product would round differently.
        a0 = vaddq_f32(a0, vmulq_f32(w0, vld1q_f32(xp)));
        a1 = vaddq_f32(a1, vmulq_f32(w1, vld1q_f32(xp.add(4))));
        wp = wp.add(LANES);
        xp = xp.add(LANES);
        i += LANES;
    }
    let mut lanes = [0f32; LANES];
    vst1q_f32(lanes.as_mut_ptr(), a0);
    vst1q_f32(lanes.as_mut_ptr().add(4), a1);
    fold(lanes)
}

/// Dot product of two f32 vectors, same accumulation shape.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut acc = [0f32; LANES];
    let full = n / LANES * LANES;
    let mut i = 0;
    while i < full {
        for l in 0..LANES {
            acc[l] += a[i + l] * b[i + l];
        }
        i += LANES;
    }
    // KV rows are head_dim (128) long, so the tail is only exercised by tests,
    // but it still has to be deterministic: fold first, then add the remainder
    // in index order.
    let mut s = fold(acc);
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

/// y += s * v, over f32.
#[inline]
pub fn axpy(y: &mut [f32], v: &[f32], s: f32) {
    debug_assert_eq!(y.len(), v.len());
    for i in 0..y.len() {
        y[i] += s * v[i];
    }
}

// ---------------------------------------------------------------------------
// layer primitives
// ---------------------------------------------------------------------------

/// RMS norm: x * g / sqrt(mean(x^2) + eps).
///
/// The sum of squares uses the same LANES-wide shape as the dot products, so a
/// change in thread count cannot move it.
pub fn rmsnorm(out: &mut [f32], x: &[f32], g: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), g.len());
    let ss = dot_f32(x, x);
    let scale = 1.0 / (ss / x.len() as f32 + eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * scale * g[i];
    }
}

/// In-place softmax over `x`, max-shifted.
pub fn softmax(x: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > m {
            m = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = exp_f32(*v - m);
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Rotary position embedding, applied in place to one head of `head_dim`
/// elements. Qwen2 uses the "NeoX" pairing: element i pairs with i + d/2.
///
/// The angle is computed from an integer position with our own exp/ln-free
/// formulation so that two hosts agree; `powf` is not used.
pub fn rope(v: &mut [f32], pos: usize, theta_base: f32) {
    let d = v.len();
    let half = d / 2;
    for i in 0..half {
        // freq = theta_base^(-2i/d), evaluated as exp(-(2i/d) * ln(theta_base))
        let inv_freq = exp_f32(-(2.0 * i as f32 / d as f32) * ln_f32(theta_base));
        let ang = pos as f32 * inv_freq;
        let (s, c) = sincos_f32(ang);
        let a = v[i];
        let b = v[i + half];
        v[i] = a * c - b * s;
        v[i + half] = a * s + b * c;
    }
}

/// ln(x) for finite positive x. Mantissa/exponent split with a polynomial on
/// the mantissa; same reasoning as `exp_f32` — not libm's.
pub fn ln_f32(x: f32) -> f32 {
    debug_assert!(x > 0.0 && x.is_finite());
    let bits = x.to_bits();
    let mut e = ((bits >> 23) & 0xff) as i32 - 127;
    let mut m = f32::from_bits((bits & 0x007f_ffff) | (127 << 23)); // m in [1,2)
    if m > 1.414_213_6 {
        m *= 0.5;
        e += 1;
    }
    // atanh series in s = (m-1)/(m+1), which converges fast on [0.707, 1.414]
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let p = s * (2.0 + s2 * (0.666_666_7 + s2 * (0.4 + s2 * (0.285_714_3 + s2 * 0.222_222_2))));
    p + e as f32 * (LN2_HI + LN2_LO)
}

const PI_2: f32 = std::f32::consts::FRAC_PI_2;

/// sin and cos of an angle, argument-reduced modulo pi/2 in f64 so that large
/// positions (a 32k-token context times a base frequency) do not lose the low
/// bits of the angle. The f64 reduction is exact-enough and, like everything
/// else here, is plain IEEE arithmetic that every host reproduces.
pub fn sincos_f32(x: f32) -> (f32, f32) {
    let q = (x as f64 / PI_2 as f64).round();
    let r = (x as f64 - q * std::f64::consts::FRAC_PI_2) as f32;
    let (s, c) = (sin_poly(r), cos_poly(r));
    match (q as i64).rem_euclid(4) {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

#[inline]
fn sin_poly(r: f32) -> f32 {
    let r2 = r * r;
    r * (1.0 + r2 * (-0.166_666_67 + r2 * (0.008_333_33 + r2 * (-1.984_127e-4 + r2 * 2.755_732e-6))))
}

#[inline]
fn cos_poly(r: f32) -> f32 {
    let r2 = r * r;
    1.0 + r2 * (-0.5 + r2 * (0.041_666_67 + r2 * (-0.001_388_889 + r2 * 2.480_159e-5)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::f16::f32_to_f16;

    #[test]
    fn neon_and_scalar_dots_agree_bit_for_bit() {
        // A deterministic pseudo-random vector pair, long enough to exercise
        // many accumulation steps.
        let n = 8960;
        let mut w = vec![0u16; n];
        let mut x = vec![0f32; n];
        let mut s = 0x1234_5678u32;
        for i in 0..n {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let a = (s >> 8) as f32 / 8_388_608.0 - 1.0;
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let b = (s >> 8) as f32 / 8_388_608.0 - 1.0;
            w[i] = f32_to_f16(a);
            x[i] = b;
        }
        let a = dot_f16(&w, &x);
        let b = dot_f16_scalar(&w, &x);
        assert_eq!(a.to_bits(), b.to_bits(), "{a} vs {b}");
    }

    #[test]
    fn exp_is_accurate() {
        for &x in &[0.0f32, 1.0, -1.0, 0.5, -12.25, 30.0, -30.0, 80.0, 88.5, -95.0, 1e-8] {
            let ours = exp_f32(x);
            let libm = (x as f64).exp() as f32;
            if libm == 0.0 || !libm.is_finite() {
                continue;
            }
            let rel = ((ours - libm) / libm).abs();
            assert!(rel < 2e-7, "exp({x}) = {ours}, want {libm}, rel {rel}");
        }
        assert_eq!(exp_f32(0.0), 1.0);
        assert!(exp_f32(1000.0).is_infinite());
        assert_eq!(exp_f32(-1000.0), 0.0);
    }

    #[test]
    fn ln_is_accurate() {
        for &x in &[1.0f32, 2.0, 0.5, 10000.0, 1e-6, 1.0e6, 1.414, 0.708] {
            let ours = ln_f32(x);
            let libm = (x as f64).ln() as f32;
            assert!((ours - libm).abs() < 1e-5, "ln({x}) = {ours}, want {libm}");
        }
    }

    #[test]
    fn sincos_is_accurate_at_large_arguments() {
        for &x in &[0.0f32, 0.3, 1.7, -2.5, 100.0, 5000.0, 31999.0] {
            let (s, c) = sincos_f32(x);
            let (rs, rc) = ((x as f64).sin() as f32, (x as f64).cos() as f32);
            assert!((s - rs).abs() < 3e-5, "sin({x}) = {s}, want {rs}");
            assert!((c - rc).abs() < 3e-5, "cos({x}) = {c}, want {rc}");
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut v = vec![1.0f32, 2.0, 3.0, -40.0, 0.0, 12.5, -1.0, 7.0];
        softmax(&mut v);
        let s: f32 = v.iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "{s}");
        assert!(v.iter().all(|&x| x >= 0.0));
    }
}
