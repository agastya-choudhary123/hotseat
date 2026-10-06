# hotseat

Live migration for LLM inference. hotseat can move a sequence that is
generating tokens from one machine to another while it keeps running. The
KV cache, sampler state, and RNG state all move with it, and the destination
produces exactly the tokens the source would have produced if the sequence
had never moved.

It uses the same approach as live VM migration. The cache pages are copied
over in rounds while decoding continues, and the OS reports which pages were
written in the meantime. On macOS that's done by write-protecting the cache
and catching `EXC_BAD_ACCESS`; on Linux it's `mprotect` and `SIGSEGV`. When
the remaining dirty set is small enough, decoding stops briefly and the rest
is sent.

```
$ hotseat a:7400 start max=120 temp=0.8 seed=99 "Explain why a cache is faster than main memory."
ok seq 1 prompt 36 tokens max 120

$ hotseat a:7400 migrate b:7400 verify
migrated sequence to b:7400 | 88 positions in cache, 89 tokens total
tracker mprotect-signal | 10 faults, 9.28 us each
round        bytes     runs       ms   tokens
    0     2.51 MiB       48     2.49        0
    1     3.00 KiB        6     0.02        0
pre-copy 2.51 MiB in 2.7 ms over 2 rounds (945 MiB/s)
rounds stopped: round fits in the 1.00 MiB budget [converged]
stop-the-world 473.0 us  (park 12035.0 us, send 7.00 KiB in 89.7 us, destination 165.1 us)
live cache 2.51 MiB | kv hash src 0x0913b2b2fd071270 dst 0x0913b2b2fd071270 (verified on destination)
```

`libc` is the only dependency. The inference engine, tokenizer, math
functions, hashing, wire format, thread pool, and syscall bindings are all in
this repo, about 7,400 lines of Rust. Design notes and bug write-ups are in
[NOTES.md](NOTES.md).

## Quick start

```sh
cargo build --release

mkdir -p models
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF \
    qwen2.5-0.5b-instruct-q4_k_m.gguf --local-dir models
# the scripts expect this name
mv models/qwen2.5-0.5b-instruct-q4_k_m.gguf models/qwen2.5-0.5b-instruct-q4km.gguf
```

Both workers need the exact same GGUF file. Two different quantizations of
the same model would silently produce different tokens, so the handshake
compares fingerprints and refuses if they don't match.

Two workers on one machine:

```sh
W=./target/release/hs-worker
H=./target/release/hotseat
M=models/qwen2.5-0.5b-instruct-q4km.gguf

$W --model $M --listen 127.0.0.1:7401 --name A &
$W --model $M --listen 127.0.0.1:7402 --name B &

$H 127.0.0.1:7401 start max=120 temp=0.8 seed=99 "Write a paragraph about virtual memory."
$H 127.0.0.1:7401 migrate 127.0.0.1:7402 at=60 verify
$H 127.0.0.1:7402 wait
```

`migrate` options:

- `at=N` hands over at exactly cache position N, which keeps the tests
  repeatable.
- `rounds=N` and `target=BYTES` set the pre-copy limits.
- `mbps=N` throttles the transfer.
- `verify` re-hashes the KV cache on the destination.
- `norng` leaves out the RNG state. This is a control for the determinism
  test.

The scripts in `scripts/`:

```sh
scripts/verify-determinism.sh   # migrate at five positions, compare transcripts
scripts/demo-precopy.sh         # pre-copy at four throttled link rates
scripts/demo-containers.sh      # container to container
scripts/demo-crosshost.sh       # macOS host to a Linux container
```

## How a handover works

The decode loop keeps one invariant: at the start of each iteration,
`filled < tokens.len()`. Here `filled` is the number of positions in the KV
cache, and `tokens` is the prompt plus everything generated so far. That
means there's always at least one token that has been chosen but not yet run
through the model. A sequence paused at that point has no logits or partial
layer state to save. The destination just feeds `tokens[filled]` at position
`filled` and keeps going.

So a handover is the cache pages, the token list, the sampler config, and
four words of RNG state. The invariant holds during prefill too, so a
sequence can even move in the middle of its prompt.

```
source                                             destination
  |-- Hello: fingerprint, cache shape, page size -->|
  |<-- HelloAck -----------------------------------| allocate cache, start
  | arm dirty tracking                              | decode thread (parked)
  |-- round 0: all live pages --------------------->|
  |     ...decoding continues...                    |
  |-- round k: pages dirtied during round k-1 ----->|
  | park decode loop             <-- window opens   |
  |-- remaining pages + tokens + sampler + RNG ---->|
  |                                                 | unpark
  |<-- ResumeAck ----------------  window closes    |
```

The destination's decode thread is created during pre-copy. I measured thread
spawn at anywhere from 48 to 943 µs, and I didn't want that inside the pause.

## Benchmarks

Measured on an Apple M4 with 16 GB, Qwen2.5-0.5B-Instruct Q4_K_M (24 KiB of
KV per token):

| | |
|---|---|
| Pause, loopback | 174–330 µs |
| Pause, container to container | 473 µs |
| Data sent while paused | ~5–20 KiB |
| Dirty-page fault, `mprotect` + signal (macOS / Linux) | 3.10 µs / 1.24 µs |
| Dirty-page fault, Mach exception port (macOS) | 13.0 µs |
| Write to a page that's already dirty | 6.4 ns |

The pause runs from the moment the decode loop parks until the destination
reports it's running. Finishing the token that was in progress is reported
separately as `park`.

Perplexity compared with llama.cpp, using the same files and 256-token
windows:

| model | hotseat | llama.cpp | diff |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct Q4_K_M | 18.140 | 18.113 | +0.15% |
| Qwen2.5-1.5B-Instruct Q4_K_M | 12.486 | 12.487 | −0.01% |

The small gap is because hotseat dequantizes the weights to f16 at load time.
That's slower than quantized matmuls, but it gives the same numeric path on
every machine, which is what this project cares about.

### Pre-copy

On loopback the copy is so much faster than decoding dirties pages that
pre-copy finishes in one round. With `mbps=N` throttling, you can see how
it behaves on slower links. This run uses a 563-token prompt and about 7 MiB
of live cache:

| link | rounds | round sizes | pause |
|---|---|---|---|
| 1000 Mbit/s | 2 | 7.05 MiB → 96 KiB | 0.97 ms |
| 200 Mbit/s | 2 | 7.05 MiB → 480 KiB | 1.11 ms |
| 60 Mbit/s | 3 | 7.08 MiB → 1.34 MiB → 312 KiB | 8.3 ms |
| 10 Mbit/s | 2, gave up | 7.05 MiB → 8.77 MiB → (9.84 MiB) | 8.25 s |

Each round sends whatever was dirtied during the previous round, so the
rounds shrink by roughly (dirty rate / link rate). At 10 Mbit/s that ratio is
above 1 and the rounds grow instead. hotseat notices before it sends a round
that wouldn't be smaller, and switches to stop-and-copy:

```
rounds stopped: round 2 would send 9.84 MiB after 8.77 MiB, no smaller
                -- link cannot outrun the sequence [DID NOT CONVERGE]
```

## Dirty-page tracking

There are four backends. The same test suite runs against each one:
`HS_TRACKER=signal cargo test`.

| backend | mechanism |
|---|---|
| `mach-vm-protect` | `mach_vm_protect` + `EXC_BAD_ACCESS` on a Mach exception port |
| `mprotect-signal` | `mprotect` + `SIGSEGV`/`SIGBUS`, handled on the faulting thread |
| `uffd-wp` | `userfaultfd` write-protect |
| `soft-dirty` | `clear_refs` + `pagemap` bit 55 (cost is a scan of the whole region) |

On a 112 MiB region (median of 5 runs):

| | macOS, 16 KiB pages | Linux container, 4 KiB pages |
|---|---|---|
| `mach-vm-protect` per fault | 13.0 µs | — |
| `mprotect-signal` per fault | 3.10 µs | 1.24 µs |
| ...of which in our handler | 0.35 µs | 0.60 µs |
| write to an already-dirty page | 6.4 ns | — |
| re-arm an untouched region | 0.2 µs | 0.6 µs |
| re-arm after every page faulted | 134 µs (7,168 pages) | 1,245 µs (28,672 pages) |

The signal path costs a trap and a return. The Mach path costs two
`mach_msg` round trips and a context switch, which is why it's about 4x
slower.

Re-arming gets expensive after many faults, because each per-page permission
change splits a VM map entry. `--cluster N` unprotects N pages per fault,
which gives a coarser dirty set but fewer faults and fewer map entries. At
cluster 8, the macOS backend handles 760k pages/s instead of 222k.

## Determinism

`scripts/verify-determinism.sh` runs a reference sequence on one worker. Then
it reruns the sequence with a handover at five fixed positions (one of them
inside the prompt) and once as an A→B→A→B round trip:

```
PASS  migrate at position 12   identical transcript   (stop-the-world 219.5 us)
PASS  migrate at position 26   identical transcript   (stop-the-world 329.5 us)
PASS  migrate at position 40   identical transcript   (stop-the-world 239.4 us)
PASS  migrate at position 53   identical transcript   (stop-the-world 173.9 us)
PASS  migrate at position 63   identical transcript   (stop-the-world 261.3 us)
PASS  A->B->A->B round trip      identical transcript

control: hand over the KV cache but not the sampler state
PASS  transcripts diverge as they must (first 43 tokens shared, then they part)
```

The control run (`norng`) moves everything except the RNG state. The KV
hashes still match, but the transcripts diverge, which shows the test can
actually catch a broken handover.

Getting identical output took three things: the same accumulation order
everywhere, no libm, and moving the sampler RNG. NOTES.md covers each one.

### Across OSes

`scripts/demo-crosshost.sh` moves a sequence from macOS into a Linux
container:

```
  macos-host       macos/aarch64  16384 B pages  mach-vm-protect
  linux-container  linux/aarch64  4096 B pages   mprotect-signal

PASS  identical token stream across macOS -> Linux container
```

Page size doesn't matter to the protocol, because transfers are byte ranges
rather than pages.

`scripts/demo-containers.sh` runs two workers on a Docker network, using
[mincontainer](https://github.com/agastya-choudhary123/mincontainer)'s dev
image with the model mounted read-only. Docker Desktop's LinuxKit kernel has
both `CONFIG_USERFAULTFD` and `CONFIG_MEM_SOFT_DIRTY` turned off, and
soft-dirty fails silently there (see NOTES.md).

## Limitations

- One sequence per worker. The macOS tracker is process-wide, so supporting
  more would mean dispatching faults by address. The signal backend already
  does that.
- No batching, including during prefill. This keeps prefill and decode on the
  same code path, but it makes prefill slow.
- No post-copy. When pre-copy can't converge, you get a long pause.
- `uffd-wp` is written but hasn't been tested, because I haven't had a kernel
  with `CONFIG_USERFAULTFD` enabled.
- Only the Qwen2 architecture is supported. The loader handles
  F32/F16/Q4_K/Q5_K/Q5_0/Q6_K/Q8_0.
- The KV hash catches corruption and divergence. It isn't meant to stop an
  attacker.

## Tests and benchmarks

```sh
cargo test --release                                   # 72 tests
HS_TRACKER=signal cargo test --release -p hs-track     # same suite, other backend

cargo run --release -p hs-track --example faultbench -- 112 mach
cargo run --release -p hs-track --example faultbench -- 112 signal

cargo run --release -p hs-engine --example ppl -- models/<model>.gguf bench/ppl.txt 256
llama-perplexity -m models/<model>.gguf -f bench/ppl.txt --ctx-size 256
```

## Layout

```
crates/hs-engine   GGUF loader, Qwen2 forward pass, tokenizer, sampler, KV cache
crates/hs-track    dirty-page tracking backends
crates/hs-wire     wire format
crates/hs-worker   the worker: decode loop, migration sender and receiver
crates/hs-cli      `hotseat`, a thin client for the worker's text control protocol
```
