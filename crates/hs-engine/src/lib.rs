//! The inference engine: GGUF loading, a Qwen2 decoder, and the per-sequence
//! state that a migration moves.
//!
//! Everything numeric here is bit-reproducible by construction — see `math`.

pub mod f16;
pub mod gguf;
pub mod hash;
pub mod kv;
pub mod math;
pub mod mmap;
pub mod model;
pub mod pool;
pub mod quant;
pub mod sampler;
pub mod tok;
