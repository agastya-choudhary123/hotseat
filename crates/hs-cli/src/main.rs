//! `hotseat` — send one control command to a worker and print the reply.
//!
//! Deliberately thin. The worker's control protocol is line-oriented text so
//! that anything, including `nc`, can drive it; this exists so the demo scripts
//! do not have to.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;

const USAGE: &str = "\
hotseat <host:port> <command> [args...]

  info                      model, cache shape, tracker
  start [k=v...] <prompt>   begin a sequence  (max, temp, top_p, top_k, seed, penalty, raw)
  status                    where the sequence has got to
  transcript                tokens, per-token timings, KV hash, decoded text
  wait                      block until the sequence finishes, then print it
  migrate <host:port> [rounds=n] [target=bytes] [verify]
  shutdown
";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 || args[0] == "-h" || args[0] == "--help" {
        print!("{USAGE}");
        std::process::exit(if args.is_empty() { 2 } else { 0 });
    }
    let addr = &args[0];
    let cmd = args[1..].join(" ");

    let stream = match TcpStream::connect(addr) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("hotseat: connecting to {addr}: {e}");
            std::process::exit(1);
        }
    };
    stream.set_nodelay(true).ok();
    let mut out = stream.try_clone().expect("clone socket");
    writeln!(out, "{cmd}").expect("send command");
    out.flush().ok();

    let mut failed = false;
    for line in BufReader::new(stream).lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("hotseat: {e}");
                std::process::exit(1);
            }
        };
        if line == "--end--" {
            break;
        }
        if line.starts_with("error:") {
            failed = true;
        }
        println!("{line}");
    }
    if failed {
        std::process::exit(1);
    }
}
