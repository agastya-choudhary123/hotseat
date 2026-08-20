//! Smoke test: load a GGUF, prefill a ChatML prompt, decode greedily.
//! Used to check the engine against llama.cpp before any migration exists.

use hs_engine::{kv::{Kv, KvLayout}, model::Model, pool::Pool, sampler::{argmax, Sampler, SamplerCfg}};
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: gen <model.gguf> [prompt] [n] [threads] [temp]");
    let prompt = args.next().unwrap_or_else(|| "Give me three facts about the moon.".into());
    let n: usize = args.next().map_or(48, |s| s.parse().unwrap());
    let threads: usize = args.next().map_or(6, |s| s.parse().unwrap());
    let temp: f32 = args.next().map_or(0.0, |s| s.parse().unwrap());

    let pool = Pool::new(threads);
    let t0 = Instant::now();
    let model = Model::load(std::path::Path::new(&path), 2048, &pool).expect("load");
    eprintln!(
        "loaded in {:.2}s | {} layers, d={}, ff={}, heads={}/{} kv, vocab={}, rope={}",
        t0.elapsed().as_secs_f64(),
        model.cfg.n_layers, model.cfg.d_model, model.cfg.d_ff,
        model.cfg.n_heads, model.cfg.n_kv_heads, model.cfg.vocab, model.cfg.rope_base
    );

    let spec = model.cfg.kv_spec(2048, KvLayout::TokenMajor);
    eprintln!("kv: {spec}");
    let mut mem = vec![0f32; spec.floats()];
    let kv = unsafe { Kv::from_raw(mem.as_mut_ptr() as *mut u8, spec) };

    let text = format!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    );
    let ids = model.tok.encode(&text);
    eprintln!("prompt: {} tokens, first 12 = {:?}", ids.len(), &ids[..12.min(ids.len())]);

    let mut s = model.scratch();
    let mut sampler = Sampler::new(SamplerCfg { temperature: temp, ..Default::default() });
    let mut hist = ids.clone();

    let t = Instant::now();
    for (p, &id) in ids.iter().enumerate() {
        model.forward(&kv, id, p, &mut s, &pool);
    }
    let prefill = t.elapsed();

    let mut pos = ids.len();
    let t = Instant::now();
    let mut out = Vec::new();
    for _ in 0..n {
        let next = if temp <= 0.0 {
            argmax(&s.logits)
        } else {
            sampler.sample(&mut s.logits, &hist)
        };
        if next == model.tok.eos {
            break;
        }
        out.push(next);
        hist.push(next);
        model.forward(&kv, next, pos, &mut s, &pool);
        pos += 1;
    }
    let decode = t.elapsed();

    println!("{}", model.tok.decode(&out));
    eprintln!(
        "\nprefill {} tok in {:.2}s ({:.1} tok/s) | decode {} tok in {:.2}s ({:.1} tok/s) | kv hash {:#018x}",
        ids.len(), prefill.as_secs_f64(), ids.len() as f64 / prefill.as_secs_f64(),
        out.len(), decode.as_secs_f64(), out.len() as f64 / decode.as_secs_f64(),
        kv.hash(pos)
    );
}
