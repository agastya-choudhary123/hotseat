//! `hs-worker` — hosts a decoding sequence and can hand it to another worker.
//!
//! One TCP port serves both roles. A connection whose first four bytes are the
//! migration magic is an incoming handover; anything else is a line-oriented
//! control session, which keeps the thing debuggable with `nc` and keeps the
//! demo scripts free of a second port to plumb through.

mod migrate;
mod seq;
mod worker;

use hs_engine::kv::KvLayout;
use hs_engine::model::Model;
use hs_engine::pool::Pool;
use hs_engine::sampler::SamplerCfg;
use hs_track::{Kind, Region};
use hs_wire::{Conn, MAGIC};
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::os::unix::io::AsRawFd;
use std::sync::Arc;
use std::time::Instant;

use seq::{decode_loop, new_sampler, Seq};
use worker::Worker;

const USAGE: &str = "\
hs-worker --model <gguf> --listen <addr> [options]

  --name    <s>     label used in logs and transcripts (default: the listen addr)
  --threads <n>     compute threads, including the caller (default: 6)
  --max-ctx <n>     KV cache capacity in positions (default: 2048)
  --layout  <s>     token-major | head-major (default: token-major)
  --tracker <s>     auto | mach | uffd | soft-dirty | none (default: auto)
  --cluster <n>     pages made writable per fault (default: 1)
  --token   <s>     shared secret required of a migration peer
";

fn main() {
    let mut model_path = String::new();
    let mut listen = String::new();
    let mut name = String::new();
    let mut threads = 6usize;
    let mut max_ctx = 2048usize;
    let mut layout = KvLayout::TokenMajor;
    let mut tracker_kind = Kind::Auto;
    let mut cluster = 1usize;
    let mut token = "hotseat".to_string();

    let args: Vec<String> = std::env::args().skip(1).collect();
    // Each flag that takes a value consumes the argument after it.
    fn value(args: &[String], i: &mut usize) -> String {
        *i += 1;
        args.get(*i).cloned().unwrap_or_else(|| die(&format!("{} needs a value", args[*i - 1])))
    }
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].clone();
        let mut next = || value(&args, &mut i);
        match flag.as_str() {
            "--model" => model_path = next(),
            "--listen" => listen = next(),
            "--name" => name = next(),
            "--threads" => threads = next().parse().unwrap_or_else(|_| die("bad --threads")),
            "--max-ctx" => max_ctx = next().parse().unwrap_or_else(|_| die("bad --max-ctx")),
            "--layout" => {
                let v = next();
                layout = KvLayout::parse(&v).unwrap_or_else(|| die("bad --layout"))
            }
            "--tracker" => {
                let v = next();
                tracker_kind = Kind::parse(&v).unwrap_or_else(|| die("bad --tracker"))
            }
            "--cluster" => cluster = next().parse().unwrap_or_else(|_| die("bad --cluster")),
            "--token" => token = next(),
            "-h" | "--help" => {
                print!("{USAGE}");
                return;
            }
            other => die(&format!("unknown argument {other:?}\n\n{USAGE}")),
        }
        i += 1;
    }
    if model_path.is_empty() || listen.is_empty() {
        die(&format!("--model and --listen are required\n\n{USAGE}"));
    }
    if name.is_empty() {
        name = listen.clone();
    }

    let pool = Arc::new(Pool::new(threads));
    let t0 = Instant::now();
    let model = match Model::load(std::path::Path::new(&model_path), max_ctx, &pool) {
        Ok(m) => Arc::new(m),
        Err(e) => die(&format!("loading {model_path}: {e}")),
    };
    let spec = model.cfg.kv_spec(max_ctx, layout);
    let model_name = std::path::Path::new(&model_path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| model_path.clone());

    // Resolve the tracker now and print what it actually is. `auto` picks the
    // best mechanism the kernel has, and which one that turned out to be is the
    // first thing you want to know from a log.
    let tracker_name = probe_tracker(tracker_kind);
    eprintln!(
        "[{name}] {model_name} loaded in {:.2}s | fingerprint {:#018x}\n\
         [{name}] {spec}\n\
         [{name}] {threads} threads, tracker {tracker_name}, cluster {cluster}, page size {} B",
        t0.elapsed().as_secs_f64(),
        model.fingerprint,
        hs_track::page_size(),
    );

    let w = Arc::new(Worker::new(
        name.clone(),
        model,
        model_name,
        pool,
        max_ctx,
        layout,
        tracker_kind,
        cluster,
        token,
    ));

    let listener = match TcpListener::bind(&listen) {
        Ok(l) => l,
        Err(e) => die(&format!("binding {listen}: {e}")),
    };
    eprintln!("[{name}] listening on {listen}");

    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let w = w.clone();
        std::thread::spawn(move || {
            if let Err(e) = serve(&w, stream) {
                if e.kind() != std::io::ErrorKind::UnexpectedEof {
                    eprintln!("[{}] connection error: {e}", w.name);
                }
            }
        });
    }
}

/// Open a one-page tracker just to learn which backend `kind` resolves to.
fn probe_tracker(kind: Kind) -> String {
    match Region::new(hs_track::page_size())
        .map_err(|e| e.to_string())
        .and_then(|r| hs_track::open(Arc::new(r), kind, 1).map_err(|e| e.to_string()))
    {
        Ok(t) => format!("{} ({}exact)", t.name(), if t.is_exact() { "" } else { "IN" }),
        Err(e) => die(&format!("tracker {kind:?} is not usable here: {e}")),
    }
}

fn die(msg: &str) -> ! {
    eprintln!("hs-worker: {msg}");
    std::process::exit(2);
}

/// Look at the first four bytes without consuming them.
fn peek_magic(s: &TcpStream) -> std::io::Result<u32> {
    let fd = s.as_raw_fd();
    let mut b = [0u8; 4];
    loop {
        let n = unsafe {
            libc::recv(fd, b.as_mut_ptr() as *mut libc::c_void, 4, libc::MSG_PEEK)
        };
        if n < 0 {
            return Err(std::io::Error::last_os_error());
        }
        if n == 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "peer closed"));
        }
        if n == 4 {
            return Ok(u32::from_le_bytes(b));
        }
        // Fewer than four bytes are buffered yet; wait for the rest.
        std::thread::yield_now();
    }
}

fn serve(w: &Arc<Worker>, stream: TcpStream) -> std::io::Result<()> {
    if peek_magic(&stream)? == MAGIC {
        return migrate::migrate_in(w, Conn::new(stream)?);
    }
    control(w, stream)
}

fn control(w: &Arc<Worker>, stream: TcpStream) -> std::io::Result<()> {
    stream.set_nodelay(true)?;
    let mut out = stream.try_clone()?;
    let mut lines = BufReader::new(stream).lines();
    while let Some(line) = lines.next() {
        let line = line?;
        let reply = command(w, line.trim());
        out.write_all(reply.as_bytes())?;
        out.write_all(b"--end--\n")?;
        if line.trim() == "quit" {
            break;
        }
    }
    Ok(())
}

fn command(w: &Arc<Worker>, line: &str) -> String {
    let (cmd, rest) = line.split_once(' ').unwrap_or((line, ""));
    match cmd {
        "" | "quit" => String::new(),
        "info" => cmd_info(w),
        "start" => cmd_start(w, rest),
        "status" => cmd_status(w),
        "transcript" => cmd_transcript(w),
        "wait" => match w.current() {
            Some(s) => {
                s.wait_done();
                cmd_transcript(w)
            }
            None => "error: no sequence\n".into(),
        },
        "migrate" => cmd_migrate(w, rest),
        "shutdown" => {
            std::thread::spawn(|| {
                std::thread::sleep(std::time::Duration::from_millis(50));
                std::process::exit(0);
            });
            "ok shutting down\n".into()
        }
        other => format!("error: unknown command {other:?}\n"),
    }
}

fn cmd_info(w: &Arc<Worker>) -> String {
    let spec = w.model.cfg.kv_spec(w.max_ctx, w.layout);
    format!(
        "name {}\nmodel {}\nfingerprint {:#018x}\nlayers {} heads {}/{} kv head_dim {} d_model {}\n\
         kv {}\npage_size {}\ntracker {} cluster {}\nthreads {}\nos {} arch {}\n",
        w.name,
        w.model_name,
        w.model.fingerprint,
        w.model.cfg.n_layers,
        w.model.cfg.n_heads,
        w.model.cfg.n_kv_heads,
        w.model.cfg.head_dim,
        w.model.cfg.d_model,
        spec,
        hs_track::page_size(),
        probe_tracker(w.tracker_kind),
        w.cluster,
        w.pool.threads(),
        std::env::consts::OS,
        std::env::consts::ARCH,
    )
}

/// `start max=<n> temp=<f> top_p=<f> top_k=<n> seed=<n> [raw] <prompt...>`
fn cmd_start(w: &Arc<Worker>, rest: &str) -> String {
    let mut cfg = SamplerCfg::default();
    let mut max_tokens = 128usize;
    let mut raw = false;
    let mut prompt = String::new();

    for (i, tok) in rest.split_whitespace().enumerate() {
        let consumed = match tok.split_once('=') {
            Some(("max", v)) => {
                max_tokens = v.parse().unwrap_or(max_tokens);
                true
            }
            Some(("temp", v)) => {
                cfg.temperature = v.parse().unwrap_or(cfg.temperature);
                true
            }
            Some(("top_p", v)) => {
                cfg.top_p = v.parse().unwrap_or(cfg.top_p);
                true
            }
            Some(("top_k", v)) => {
                cfg.top_k = v.parse().unwrap_or(cfg.top_k);
                true
            }
            Some(("seed", v)) => {
                cfg.seed = v.parse().unwrap_or(cfg.seed);
                true
            }
            Some(("penalty", v)) => {
                cfg.repeat_penalty = v.parse().unwrap_or(cfg.repeat_penalty);
                true
            }
            _ if tok == "raw" => {
                raw = true;
                true
            }
            _ => false,
        };
        if !consumed {
            // Everything from here on is the prompt.
            let at = rest
                .split_whitespace()
                .take(i)
                .map(|t| t.len() + 1)
                .sum::<usize>();
            prompt = rest[at..].to_string();
            break;
        }
    }
    if prompt.is_empty() {
        return "error: no prompt\n".into();
    }
    let prompt = prompt.replace("\\n", "\n");

    let text = if raw {
        prompt
    } else {
        format!(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
             <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
    };
    let tokens = w.model.tok.encode(&text);
    if tokens.len() + max_tokens >= w.max_ctx {
        return format!(
            "error: prompt is {} tokens and max is {max_tokens}, over the {} position cache\n",
            tokens.len(),
            w.max_ctx
        );
    }

    let spec = w.model.cfg.kv_spec(w.max_ctx, w.layout);
    let region = match Region::new(spec.bytes()) {
        Ok(r) => Arc::new(r),
        Err(e) => return format!("error: allocating KV cache: {e}\n"),
    };
    region.prefault();
    let tracker = match hs_track::open(region.clone(), w.tracker_kind, w.cluster) {
        Ok(t) => t,
        Err(e) => return format!("error: opening tracker: {e}\n"),
    };

    let prompt_len = tokens.len();
    let id = w.new_id();
    let s = Seq::new(
        id,
        w.name.clone(),
        w.model.clone(),
        w.pool.clone(),
        spec,
        region,
        tracker,
        tokens,
        prompt_len,
        0,
        new_sampler(cfg),
        max_tokens,
    );
    if let Err(e) = w.take_slot(s.clone()) {
        return format!("error: {e}\n");
    }
    let d = s.clone();
    if let Err(e) =
        std::thread::Builder::new().name(format!("hs-decode-{id}")).spawn(move || decode_loop(d))
    {
        return format!("error: spawning decode thread: {e}\n");
    }
    format!("ok seq {id} prompt {prompt_len} tokens max {max_tokens}\n")
}

fn cmd_status(w: &Arc<Worker>) -> String {
    let Some(s) = w.current() else { return "error: no sequence\n".into() };
    let st = s.state.lock().unwrap();
    format!(
        "seq {}\nhost {}\nfilled {}\ntokens {}\nprompt_len {}\ngenerated {}\ndecoded_here {}\n\
         done {}{}\nmigrated_away {}\nrng_draws {}\nparked_us {:.1}\n",
        s.id,
        s.host,
        st.filled,
        st.tokens.len(),
        st.prompt_len,
        st.tokens.len() - st.prompt_len,
        st.decoded_here,
        st.done,
        if st.done { format!(" ({})", st.done_reason) } else { String::new() },
        st.migrated_away,
        st.sampler.rng.draws,
        s.parked_ns.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e3,
    )
}

fn cmd_transcript(w: &Arc<Worker>) -> String {
    let Some(s) = w.current() else { return "error: no sequence\n".into() };
    let st = s.state.lock().unwrap();
    let mut out = String::new();
    out.push_str(&format!("seq {}\n", s.id));
    out.push_str(&format!(
        "tokens {}\n",
        st.tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(",")
    ));
    out.push_str(&format!("prompt_len {}\n", st.prompt_len));
    out.push_str(&format!("filled {}\n", st.filled));
    out.push_str(&format!("done {}\n", st.done));
    for e in &st.log {
        out.push_str(&format!("emit {} {} {} {}\n", e.index, e.token, e.epoch_ns, e.host));
    }
    let kv = s.kv();
    out.push_str(&format!("kv_hash {:#018x}\n", kv.hash(st.filled)));
    out.push_str(&format!("rng {:?} draws {}\n", st.sampler.rng.s, st.sampler.rng.draws));
    out.push_str("text ");
    out.push_str(&w.model.tok.decode(&st.tokens[st.prompt_len..]).replace('\n', "\\n"));
    out.push('\n');
    out
}

fn cmd_migrate(w: &Arc<Worker>, rest: &str) -> String {
    let Some(s) = w.current() else { return "error: no sequence\n".into() };
    let mut target = String::new();
    let mut opts = migrate::Opts::default();
    let mut at: Option<usize> = None;
    for tok in rest.split_whitespace() {
        match tok.split_once('=') {
            Some(("rounds", v)) => opts.max_rounds = v.parse().unwrap_or(opts.max_rounds),
            Some(("target", v)) => opts.target_bytes = v.parse().unwrap_or(opts.target_bytes),
            Some(("at", v)) => at = v.parse().ok(),
            _ if tok == "verify" => opts.verify = true,
            _ if tok == "norng" => opts.drop_rng = true,
            _ => target = tok.to_string(),
        }
    }
    if target.is_empty() {
        return "error: migrate <host:port> [at=n] [rounds=n] [target=bytes] [verify] [norng]\n"
            .into();
    }
    if let Some(n) = at {
        s.wait_filled(n);
    }
    {
        let st = s.state.lock().unwrap();
        if st.done {
            return format!("error: sequence already finished ({})\n", st.done_reason);
        }
        if st.migrated_away {
            return "error: sequence already migrated away\n".into();
        }
    }
    match migrate::migrate_out(&s, &target, &w.token, &w.model_name, &opts) {
        Ok(r) => format!("ok\n{}", r.render()),
        Err(e) => format!("error: {e}\n"),
    }
}
