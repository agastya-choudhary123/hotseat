//! The Qwen2 decoder.
//!
//! One token at a time — there is no batched prefill. That is a deliberate
//! simplification for this project: a decode step and a prefill step then dirty
//! the KV cache through the identical code path, so the migration numbers
//! describe one mechanism rather than two.
//!
//! The cache itself is never allocated here. `forward` is handed a `&Kv` that
//! points into memory somebody else owns and is watching.

use crate::f16::f16_to_f32;
use crate::gguf::{Error, Gguf, GgmlType};
use crate::hash::Fnv;
use crate::kv::{Kv, KvLayout, KvSpec};
use crate::math::{self, dot_f16, dot_f32};
use crate::pool::Pool;
use crate::quant::dequant_to_f16;
use crate::tok::Tokenizer;
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub vocab: usize,
    pub eps: f32,
    pub rope_base: f32,
    pub train_ctx: usize,
}

impl Config {
    pub fn kv_spec(&self, max_ctx: usize, layout: KvLayout) -> KvSpec {
        KvSpec {
            n_layers: self.n_layers,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            max_ctx,
            layout,
        }
    }
}

struct Layer {
    attn_norm: Vec<f32>,
    wq: Vec<u16>,
    bq: Vec<f32>,
    wk: Vec<u16>,
    bk: Vec<f32>,
    wv: Vec<u16>,
    bv: Vec<f32>,
    wo: Vec<u16>,
    ffn_norm: Vec<f32>,
    w_gate: Vec<u16>,
    w_up: Vec<u16>,
    w_down: Vec<u16>,
}

pub struct Model {
    pub cfg: Config,
    pub tok: Tokenizer,
    /// Identifies the weights. Both ends of a migration must report the same
    /// value or the transfer is refused — resuming a sequence against different
    /// weights would produce plausible text and a silently wrong answer.
    pub fingerprint: u64,
    embd: Vec<u16>,
    out_norm: Vec<f32>,
    /// Present only when the model does not tie the LM head to the embedding.
    out_w: Option<Vec<u16>>,
    layers: Vec<Layer>,
    /// cos/sin for every (position, frequency) pair, computed once.
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    max_ctx: usize,
}

/// A destination address smuggled into a `broadcast` closure.
///
/// It holds a `usize` rather than a `*mut T` on purpose: edition-2021 closures
/// capture individual fields, so a raw-pointer field would be captured on its
/// own and raw pointers are not `Sync`. An address is, and the pool's row
/// ranges are disjoint, so writes through it never overlap.
#[derive(Clone, Copy)]
struct Ptr(usize);

impl Ptr {
    #[inline(always)]
    fn new<T>(p: *mut T) -> Ptr {
        Ptr(p as usize)
    }
    /// # Safety
    /// `i` must be in bounds of the originating allocation, and no other thread
    /// may be writing the same element.
    #[inline(always)]
    unsafe fn f32_at(self, i: usize) -> *mut f32 {
        (self.0 as *mut f32).add(i)
    }
    /// # Safety
    /// As `f32_at`.
    #[inline(always)]
    unsafe fn u16_at(self, i: usize) -> *mut u16 {
        (self.0 as *mut u16).add(i)
    }
}

impl Model {
    pub fn load(path: &Path, max_ctx: usize, pool: &Pool) -> Result<Model, Error> {
        let g = Gguf::open(path)?;
        g.bytes(); // touch
        let arch = g.get_str("general.architecture")?.to_string();
        if arch != "qwen2" {
            return Err(Error::WrongKeyType {
                key: format!("general.architecture = {arch:?}; only qwen2 is implemented"),
                want: "qwen2",
            });
        }
        let k = |s: &str| format!("{arch}.{s}");
        let n_layers = g.get_u32(&k("block_count"))? as usize;
        let d_model = g.get_u32(&k("embedding_length"))? as usize;
        let d_ff = g.get_u32(&k("feed_forward_length"))? as usize;
        let n_heads = g.get_u32(&k("attention.head_count"))? as usize;
        let n_kv_heads = g.get_u32(&k("attention.head_count_kv"))? as usize;
        let eps = g.get_f32(&k("attention.layer_norm_rms_epsilon"))?;
        let rope_base = g.get_f32(&k("rope.freq_base")).unwrap_or(10_000.0);
        let train_ctx = g.get_u32(&k("context_length"))? as usize;
        let head_dim = d_model / n_heads;

        let tok = Tokenizer::from_gguf(&g)?;
        let cfg = Config {
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            d_model,
            d_ff,
            vocab: tok.vocab_size(),
            eps,
            rope_base,
            train_ctx,
        };

        // Fingerprint: the tensor table plus the file length. Cheap, and it
        // changes if any tensor moves, is retyped, or is resized.
        let mut fp = Fnv::new();
        fp.write_u64(g.bytes().len() as u64);
        for name in &g.tensor_order {
            let t = g.tensor(name)?;
            fp.write(name.as_bytes());
            fp.write_u32(t.ty as u32);
            fp.write_u64(t.off as u64);
            fp.write_u64(t.len as u64);
        }
        let fingerprint = fp.finish();

        let f16 = |n: &str| -> Result<Vec<u16>, Error> { load_f16(&g, n, pool) };
        let f32v = |n: &str| -> Result<Vec<f32>, Error> { load_f32(&g, n) };

        let embd = f16("token_embd.weight")?;
        let out_norm = f32v("output_norm.weight")?;
        let out_w = match g.tensors.contains_key("output.weight") {
            true => Some(f16("output.weight")?),
            false => None, // tied to the embedding
        };

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let b = |s: &str| format!("blk.{i}.{s}");
            layers.push(Layer {
                attn_norm: f32v(&b("attn_norm.weight"))?,
                wq: f16(&b("attn_q.weight"))?,
                bq: f32v(&b("attn_q.bias"))?,
                wk: f16(&b("attn_k.weight"))?,
                bk: f32v(&b("attn_k.bias"))?,
                wv: f16(&b("attn_v.weight"))?,
                bv: f32v(&b("attn_v.bias"))?,
                wo: f16(&b("attn_output.weight"))?,
                ffn_norm: f32v(&b("ffn_norm.weight"))?,
                w_gate: f16(&b("ffn_gate.weight"))?,
                w_up: f16(&b("ffn_up.weight"))?,
                w_down: f16(&b("ffn_down.weight"))?,
            });
        }

        // Rotary tables. Computing these once removes every transcendental
        // from the decode step, which is both faster and one less place for
        // two hosts to disagree.
        let half = head_dim / 2;
        let mut rope_cos = vec![0f32; max_ctx * half];
        let mut rope_sin = vec![0f32; max_ctx * half];
        let ln_base = math::ln_f32(rope_base);
        for i in 0..half {
            let inv_freq = math::exp_f32(-(2.0 * i as f32 / head_dim as f32) * ln_base);
            for p in 0..max_ctx {
                let (s, c) = math::sincos_f32(p as f32 * inv_freq);
                rope_cos[p * half + i] = c;
                rope_sin[p * half + i] = s;
            }
        }

        Ok(Model {
            cfg,
            tok,
            fingerprint,
            embd,
            out_norm,
            out_w,
            layers,
            rope_cos,
            rope_sin,
            max_ctx,
        })
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    pub fn scratch(&self) -> Scratch {
        let c = &self.cfg;
        Scratch {
            x: vec![0.0; c.d_model],
            xb: vec![0.0; c.d_model],
            xb2: vec![0.0; c.d_model],
            q: vec![0.0; c.n_heads * c.head_dim],
            k: vec![0.0; c.n_kv_heads * c.head_dim],
            v: vec![0.0; c.n_kv_heads * c.head_dim],
            att: vec![0.0; c.n_heads * c.head_dim],
            scores: vec![0.0; c.n_heads * self.max_ctx],
            hb: vec![0.0; c.d_ff],
            hb2: vec![0.0; c.d_ff],
            logits: vec![0.0; c.vocab],
        }
    }

    /// Run one token through the model at position `pos`, appending its K and V
    /// to `kv` and leaving the next-token distribution in `s.logits`.
    pub fn forward(&self, kv: &Kv, token: u32, pos: usize, s: &mut Scratch, pool: &Pool) {
        let c = &self.cfg;
        assert!(pos < self.max_ctx, "position {pos} past max_ctx {}", self.max_ctx);
        assert_eq!(kv.spec.n_layers, c.n_layers, "KV cache shape does not match the model");

        let hd = c.head_dim;
        let half = hd / 2;
        let row = token as usize * c.d_model;
        for i in 0..c.d_model {
            s.x[i] = f16_to_f32(self.embd[row + i]);
        }

        for (l, layer) in self.layers.iter().enumerate() {
            math::rmsnorm(&mut s.xb, &s.x, &layer.attn_norm, c.eps);

            matvec(pool, &layer.wq, &s.xb, Some(&layer.bq), &mut s.q);
            matvec(pool, &layer.wk, &s.xb, Some(&layer.bk), &mut s.k);
            matvec(pool, &layer.wv, &s.xb, Some(&layer.bv), &mut s.v);

            let (cos, sin) =
                (&self.rope_cos[pos * half..], &self.rope_sin[pos * half..]);
            for h in 0..c.n_heads {
                rope_apply(&mut s.q[h * hd..(h + 1) * hd], cos, sin);
            }
            for h in 0..c.n_kv_heads {
                rope_apply(&mut s.k[h * hd..(h + 1) * hd], cos, sin);
            }

            // The only writes to tracked memory. Plain stores: the dirty-page
            // tracker learns about them from the hardware, not from us.
            for h in 0..c.n_kv_heads {
                unsafe {
                    kv.k_mut(l, pos, h).copy_from_slice(&s.k[h * hd..(h + 1) * hd]);
                    kv.v_mut(l, pos, h).copy_from_slice(&s.v[h * hd..(h + 1) * hd]);
                }
            }

            let scale = 1.0 / (hd as f32).sqrt();
            let groups = c.n_heads / c.n_kv_heads;
            let att_p = Ptr::new(s.att.as_mut_ptr());
            let sc_p = Ptr::new(s.scores.as_mut_ptr());
            let q_ref = &s.q;
            let max_ctx = self.max_ctx;
            pool.broadcast(|tid| {
                let (h0, h1) = pool.range(c.n_heads, tid);
                for h in h0..h1 {
                    let kvh = h / groups;
                    let q = &q_ref[h * hd..(h + 1) * hd];
                    // SAFETY: head ranges are disjoint, so these two slices are
                    // exclusive to this thread.
                    let sc = unsafe {
                        std::slice::from_raw_parts_mut(sc_p.f32_at(h * max_ctx), pos + 1)
                    };
                    let out = unsafe {
                        std::slice::from_raw_parts_mut(att_p.f32_at(h * hd), hd)
                    };
                    for p in 0..=pos {
                        sc[p] = dot_f32(q, kv.k(l, p, kvh)) * scale;
                    }
                    math::softmax(sc);
                    out.fill(0.0);
                    for p in 0..=pos {
                        math::axpy(out, kv.v(l, p, kvh), sc[p]);
                    }
                }
            });

            matvec(pool, &layer.wo, &s.att, None, &mut s.xb2);
            for i in 0..c.d_model {
                s.x[i] += s.xb2[i];
            }

            math::rmsnorm(&mut s.xb, &s.x, &layer.ffn_norm, c.eps);
            matvec(pool, &layer.w_gate, &s.xb, None, &mut s.hb);
            matvec(pool, &layer.w_up, &s.xb, None, &mut s.hb2);
            for i in 0..c.d_ff {
                s.hb[i] = math::silu(s.hb[i]) * s.hb2[i];
            }
            matvec(pool, &layer.w_down, &s.hb, None, &mut s.xb2);
            for i in 0..c.d_model {
                s.x[i] += s.xb2[i];
            }
        }

        math::rmsnorm(&mut s.xb, &s.x, &self.out_norm, c.eps);
        let head = self.out_w.as_ref().unwrap_or(&self.embd);
        matvec(pool, head, &s.xb, None, &mut s.logits);
    }
}

pub struct Scratch {
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att: Vec<f32>,
    scores: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    pub logits: Vec<f32>,
}

/// y[r] = dot(W[r], x) (+ bias[r]), rows split across the pool.
fn matvec(pool: &Pool, w: &[u16], x: &[f32], bias: Option<&[f32]>, y: &mut [f32]) {
    let n_in = x.len();
    let rows = y.len();
    debug_assert_eq!(w.len(), rows * n_in);
    let yp = Ptr::new(y.as_mut_ptr());
    pool.broadcast(|tid| {
        let (a, b) = pool.range(rows, tid);
        for r in a..b {
            let mut v = dot_f16(&w[r * n_in..r * n_in + n_in], x);
            if let Some(bs) = bias {
                v += bs[r];
            }
            // SAFETY: row ranges are disjoint across threads.
            unsafe { *yp.f32_at(r) = v }
        }
    });
}

/// Rotary embedding, NeoX pairing: element i rotates with element i + d/2.
#[inline]
fn rope_apply(v: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = v.len() / 2;
    for i in 0..half {
        let (c, s) = (cos[i], sin[i]);
        let (a, b) = (v[i], v[i + half]);
        v[i] = a * c - b * s;
        v[i + half] = a * s + b * c;
    }
}

fn load_f16(g: &Gguf, name: &str, pool: &Pool) -> Result<Vec<u16>, Error> {
    let t = g.tensor(name)?;
    let n = t.elems();
    let src = g.tensor_bytes(t);
    let mut out = vec![0u16; n];
    let (blk_elems, blk_bytes) = t.ty.block();
    let n_blocks = n / blk_elems;
    let op = Ptr::new(out.as_mut_ptr());
    let ty = t.ty;
    pool.broadcast(|tid| {
        let (b0, b1) = pool.range(n_blocks, tid);
        if b0 == b1 {
            return;
        }
        // SAFETY: block ranges are disjoint and block-aligned in both buffers.
        let dst = unsafe {
            std::slice::from_raw_parts_mut(op.u16_at(b0 * blk_elems), (b1 - b0) * blk_elems)
        };
        dequant_to_f16(ty, &src[b0 * blk_bytes..b1 * blk_bytes], (b1 - b0) * blk_elems, dst);
    });
    Ok(out)
}

fn load_f32(g: &Gguf, name: &str) -> Result<Vec<f32>, Error> {
    let t = g.tensor(name)?;
    let n = t.elems();
    let mut out = vec![0f32; n];
    match t.ty {
        GgmlType::F32 => crate::quant::dequant_to_f32(t.ty, g.tensor_bytes(t), n, &mut out),
        other => {
            let mut h = vec![0u16; n];
            dequant_to_f16(other, g.tensor_bytes(t), n, &mut h);
            for i in 0..n {
                out[i] = f16_to_f32(h[i]);
            }
        }
    }
    Ok(out)
}
