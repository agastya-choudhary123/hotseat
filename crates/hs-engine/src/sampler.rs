//! Token sampling, and the random stream that drives it.
//!
//! The RNG is deliberately *stateful* — a xoshiro256** stream that advances
//! every time a token is drawn. A counter-based generator keyed on the position
//! would have been easier to migrate (there would be nothing to migrate), but
//! it would also have made the determinism test vacuous. With a real stream,
//! forgetting to move the sampler state produces a divergent transcript, which
//! is exactly the failure this project has to be able to demonstrate it does
//! not have. `tests/` includes a case that skips the RNG on purpose and asserts
//! the streams come apart.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerCfg {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub repeat_penalty: f32,
    pub repeat_last_n: u32,
    pub seed: u64,
}

impl Default for SamplerCfg {
    fn default() -> Self {
        SamplerCfg {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            seed: 0xC0FF_EE00_1234_5678,
        }
    }
}

/// xoshiro256**, seeded through splitmix64. Four words of state; those four
/// words are part of what a migration carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rng {
    pub s: [u64; 4],
    /// How many values have been drawn. Not needed to reproduce the stream —
    /// it is carried so the receiving end can assert it resumed at the same
    /// point the sender left off.
    pub draws: u64,
}

impl Rng {
    pub fn seed(seed: u64) -> Rng {
        let mut z = seed;
        let mut s = [0u64; 4];
        for w in s.iter_mut() {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *w = x ^ (x >> 31);
        }
        Rng { s, draws: 0 }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        self.draws += 1;
        result
    }

    /// Uniform in [0, 1). Built from the top 24 bits so the result is exactly
    /// representable in f32 and the mapping is identical on every host.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / 16_777_216.0)
    }
}

/// Sampling state that has to survive a migration.
#[derive(Debug, Clone)]
pub struct Sampler {
    pub cfg: SamplerCfg,
    pub rng: Rng,
    scratch: Vec<(f32, u32)>,
}

impl Sampler {
    pub fn new(cfg: SamplerCfg) -> Sampler {
        Sampler { rng: Rng::seed(cfg.seed), cfg, scratch: Vec::new() }
    }

    /// Pick the next token from `logits`, penalising tokens that appear in the
    /// tail of `history`.
    ///
    /// `logits` is modified in place.
    pub fn sample(&mut self, logits: &mut [f32], history: &[u32]) -> u32 {
        let cfg = self.cfg;

        if cfg.repeat_penalty != 1.0 && cfg.repeat_last_n > 0 {
            let n = (cfg.repeat_last_n as usize).min(history.len());
            for &t in &history[history.len() - n..] {
                let l = &mut logits[t as usize];
                // The usual asymmetric form: divide positive logits, multiply
                // negative ones, so the penalty always moves them downward.
                *l = if *l > 0.0 { *l / cfg.repeat_penalty } else { *l * cfg.repeat_penalty };
            }
        }

        if cfg.temperature <= 0.0 {
            return argmax(logits);
        }

        // Rank candidates. Ties break on the lower token id so that the order
        // is total and identical on both ends of a migration — a comparison
        // that returns Equal would leave the outcome up to the sort's internals.
        let k = if cfg.top_k == 0 {
            logits.len()
        } else {
            (cfg.top_k as usize).min(logits.len())
        };
        self.scratch.clear();
        self.scratch.extend(logits.iter().copied().zip(0u32..));
        let cand = &mut self.scratch;
        if k < cand.len() {
            cand.select_nth_unstable_by(k - 1, |a, b| cmp_desc(a, b));
            cand.truncate(k);
        }
        cand.sort_unstable_by(|a, b| cmp_desc(a, b));

        // Softmax over the kept candidates at the requested temperature.
        let inv_t = 1.0 / cfg.temperature;
        let max = cand[0].0;
        let mut sum = 0.0f32;
        for c in cand.iter_mut() {
            c.0 = crate::math::exp_f32((c.0 - max) * inv_t);
            sum += c.0;
        }

        // Nucleus: keep the shortest prefix whose mass reaches top_p.
        let mut cut = cand.len();
        if cfg.top_p < 1.0 {
            let target = cfg.top_p * sum;
            let mut acc = 0.0f32;
            for (i, c) in cand.iter().enumerate() {
                acc += c.0;
                if acc >= target {
                    cut = i + 1;
                    break;
                }
            }
            sum = cand[..cut].iter().map(|c| c.0).sum();
        }

        let r = self.rng.next_f32() * sum;
        let mut acc = 0.0f32;
        for c in &cand[..cut] {
            acc += c.0;
            if r < acc {
                return c.1;
            }
        }
        cand[cut - 1].1
    }
}

#[inline]
fn cmp_desc(a: &(f32, u32), b: &(f32, u32)) -> std::cmp::Ordering {
    // total_cmp gives a total order over f32 including NaN, so no comparison
    // ever returns Equal for distinct logits and the id breaks exact ties.
    b.0.total_cmp(&a.0).then(a.1.cmp(&b.1))
}

pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i as u32;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_matches_the_reference_stream() {
        // xoshiro256** from state [1,2,3,4], cross-checked against an
        // independent implementation of Blackman & Vigna's reference C.
        let mut r = Rng { s: [1, 2, 3, 4], draws: 0 };
        let got: Vec<u64> = (0..5).map(|_| r.next_u64()).collect();
        assert_eq!(
            got,
            vec![
                11520u64,
                0,
                1_509_978_240,
                1_215_971_899_390_074_240,
                1_216_172_134_540_287_360,
            ]
        );
        assert_eq!(r.draws, 5);
    }

    #[test]
    fn rng_state_fully_determines_the_future() {
        let mut a = Rng::seed(42);
        for _ in 0..1000 {
            a.next_u64();
        }
        let snapshot = a;
        let want: Vec<u64> = (0..50).map(|_| a.next_u64()).collect();
        let mut b = snapshot; // as if restored on another host
        let got: Vec<u64> = (0..50).map(|_| b.next_u64()).collect();
        assert_eq!(want, got);
        assert_eq!(a, b);
    }

    #[test]
    fn next_f32_stays_in_range() {
        let mut r = Rng::seed(7);
        for _ in 0..100_000 {
            let v = r.next_f32();
            assert!((0.0..1.0).contains(&v), "{v}");
        }
    }

    #[test]
    fn greedy_sampling_ignores_the_rng() {
        let cfg = SamplerCfg { temperature: 0.0, ..Default::default() };
        let mut s = Sampler::new(cfg);
        let mut l = vec![0.1, 5.0, -3.0, 4.9];
        assert_eq!(s.sample(&mut l, &[]), 1);
        assert_eq!(s.rng.draws, 0);
    }

    #[test]
    fn top_k_one_is_greedy_even_with_temperature() {
        let cfg = SamplerCfg {
            temperature: 2.0,
            top_k: 1,
            top_p: 1.0,
            repeat_penalty: 1.0,
            ..Default::default()
        };
        let mut s = Sampler::new(cfg);
        for _ in 0..100 {
            let mut l = vec![0.1, 5.0, -3.0, 4.9];
            assert_eq!(s.sample(&mut l, &[]), 1);
        }
    }

    #[test]
    fn same_state_gives_the_same_token() {
        let cfg = SamplerCfg { temperature: 1.0, top_k: 0, top_p: 1.0, repeat_penalty: 1.0, ..Default::default() };
        let mut a = Sampler::new(cfg);
        let logits: Vec<f32> = (0..512).map(|i| ((i * 37 % 91) as f32) / 13.0).collect();
        for _ in 0..20 {
            a.sample(&mut logits.clone(), &[]);
        }
        let mut b = Sampler::new(cfg);
        b.rng = a.rng; // the only thing a migration moves
        let x = a.sample(&mut logits.clone(), &[]);
        let y = b.sample(&mut logits.clone(), &[]);
        assert_eq!(x, y);
    }

    #[test]
    fn repetition_penalty_pushes_logits_down() {
        let cfg = SamplerCfg { temperature: 0.0, repeat_penalty: 2.0, repeat_last_n: 8, ..Default::default() };
        let mut s = Sampler::new(cfg);
        let mut l = vec![0.0, 10.0, 9.0, 0.0];
        assert_eq!(s.sample(&mut l, &[1]), 2, "the repeated token should lose");
        let mut l2 = vec![0.0, -1.0, -2.0, 0.0];
        s.sample(&mut l2, &[1]);
        assert_eq!(l2[1], -2.0, "negative logits are multiplied, not divided");
    }
}
