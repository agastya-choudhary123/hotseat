//! Mean negative log-likelihood over a text, for comparison against
//! `llama-perplexity` on the same file and the same context window.

use hs_engine::{kv::{Kv, KvLayout}, math, model::Model, pool::Pool};
use std::path::Path;

fn main() {
    let mut a = std::env::args().skip(1);
    let model_path = a.next().expect("usage: ppl <model.gguf> <text> [ctx] [threads]");
    let text_path = a.next().expect("text file");
    let ctx: usize = a.next().map_or(256, |s| s.parse().unwrap());
    let threads: usize = a.next().map_or(6, |s| s.parse().unwrap());

    let pool = Pool::new(threads);
    let model = Model::load(Path::new(&model_path), ctx, &pool).expect("load");
    let text = std::fs::read_to_string(&text_path).expect("read text");
    let ids = model.tok.encode(&text);

    let spec = model.cfg.kv_spec(ctx, KvLayout::TokenMajor);
    let mut mem = vec![0f32; spec.floats()];
    let kv = unsafe { Kv::from_raw(mem.as_mut_ptr() as *mut u8, spec) };
    let mut s = model.scratch();

    // Same protocol as llama-perplexity: independent windows of `ctx` tokens,
    // scoring only the second half of each window so that every scored token
    // has at least ctx/2 tokens of context behind it.
    let mut nll = 0.0f64;
    let mut counted = 0usize;
    let chunks = ids.len() / ctx;
    for c in 0..chunks {
        let w = &ids[c * ctx..(c + 1) * ctx];
        for (p, &t) in w.iter().enumerate() {
            model.forward(&kv, t, p, &mut s, &pool);
            if p >= ctx / 2 && p + 1 < w.len() {
                let next = w[p + 1] as usize;
                // log softmax, max-shifted
                let max = s.logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f64;
                for &l in s.logits.iter() {
                    sum += math::exp_f32(l - max) as f64;
                }
                nll -= ((s.logits[next] - max) as f64) - sum.ln();
                counted += 1;
            }
        }
    }
    println!(
        "tokens {} | windows {} of {} | scored {} | mean NLL {:.5} | ppl {:.5}",
        ids.len(), chunks, ctx, counted, nll / counted as f64, (nll / counted as f64).exp()
    );
}
