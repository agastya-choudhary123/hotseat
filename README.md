hotseat
-------

hotseat moves a running LLM request between machines without stopping it. A
decoding sequence is frozen on one worker, its KV cache, sampler state and
random stream are shipped to another, and it resumes there mid-sentence and
finishes the token stream it started, producing bit-for-bit the tokens it would
have produced had it never moved.

The transfer is not the hard part. The sequence keeps decoding while the
transfer happens, which requires the machine to tell you which pages of the
cache changed under you: on macOS, write-protecting the cache and catching
`EXC_BAD_ACCESS` on a Mach exception port; on Linux, `mprotect` and a `SIGSEGV`
handler. Then iterative pre-copy rounds and a short stop-and-copy at the end.
It is the algorithm live VM migration has used for twenty years, applied to
something nobody applies it to.

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

`hs-a` decoded 72 tokens, `hs-b` decoded the other 48, and the paragraph reads
as one paragraph.

The only dependency in the workspace is `libc`. The tokenizer, transcendentals,
hashes, wire format, thread pool and every syscall binding are in this tree.
About 6,800 lines of Rust and 1,150 of tests.

### Documentation quick links

* [Quick start](#quick-start)
* [Benchmarks](#benchmarks)
* [Dirty-page tracking](#dirty-page-tracking)
* [Determinism](#determinism)
* [Limitations](#limitations)
* [NOTES.md](NOTES.md) — design decisions, the epoch protocol, and the hardest bugs

### Quick start

```
$ cargo build --release
```

The GGUF weights are not in this repo. Both demo models are stock quantizations
from Hugging Face:

```
$ mkdir -p models
$ huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF \
      qwen2.5-0.5b-instruct-q4_k_m.gguf --local-dir models
$ mv models/qwen2.5-0.5b-instruct-q4_k_m.gguf models/qwen2.5-0.5b-instruct-q4km.gguf
```

Any GGUF llama.cpp can load will work. Both workers in a migration must be given
the byte-identical file: a handover replays cache positions against the model's
own weights, so two different quantizations of the same model diverge rather
than failing loudly. The handshake refuses the transfer if the fingerprints
differ.

Two workers on one machine:

```
$ ./target/release/hs-worker --model models/qwen2.5-0.5b-instruct-q4km.gguf \
      --listen 127.0.0.1:7401 --name A &
$ ./target/release/hs-worker --model models/qwen2.5-0.5b-instruct-q4km.gguf \
      --listen 127.0.0.1:7402 --name B &

$ H=./target/release/hotseat
$ $H 127.0.0.1:7401 start max=120 temp=0.8 seed=99 "Write a paragraph about virtual memory."
$ $H 127.0.0.1:7401 migrate 127.0.0.1:7402 at=60 verify
$ $H 127.0.0.1:7402 wait
```

`at=60` pins the handover to an exact cache position instead of racing it with a
sleep, which is what makes the determinism tests repeatable.

Four scripts run the whole story:

```
$ scripts/verify-determinism.sh   # transcripts identical at five migration points
$ scripts/demo-precopy.sh         # pre-copy across four shaped link rates
$ scripts/demo-containers.sh      # container to container
$ scripts/demo-crosshost.sh       # macOS host -> Linux container
```

### Benchmarks

Apple M4, 16 GB, Qwen2.5-0.5B-Instruct Q4_K_M, 24 KiB of KV per token.

| | |
|---|---|
| Stop-the-world window, loopback | 174 – 330 µs |
| Stop-the-world, container to container | 473 µs |
| Residual bytes moved with the sequence stopped | ~5 – 20 KiB |
| Dirty-page fault, `mprotect` + signal (macOS / Linux) | 3.10 µs / 1.24 µs |
| Dirty-page fault, Mach exception port (macOS) | 13.0 µs |
| Write to a page already dirty this round | 6.4 ns |
| Transcript after a handover | token-identical to a run that never moved |

The window is measured from the instant the decode loop parks to the instant the
destination says it is running. It excludes the tail of the token in flight when
the migration asked to stop; that is reported separately as `park`, because the
sequence was still producing during it.

Engine correctness against llama.cpp, same files, same 256-token windows, same
second-half scoring:

| model | hotseat | llama.cpp | delta |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct Q4_K_M | 18.140 | 18.113 | +0.15% |
| Qwen2.5-1.5B-Instruct Q4_K_M | 12.486 | 12.487 | −0.01% |

The residual is deliberate: weights are dequantised to f16 at load rather than
kept quantised. A quantised dot product would be faster per byte, but the
subject here is migration, and f16 gives one numeric path that is identical
everywhere, so "the token stream survives the boundary" is a claim about the
migration rather than about two matmul kernels agreeing.

#### Pre-copy

Loopback moves a KV cache far faster than a sequence can dirty it, so rounds
converge immediately and the algorithm never shows its work. `migrate ...
mbps=N` shapes the bulk transfer so they do. Same sequence, 563-token prompt,
~7 MiB live cache:

| link | rounds | round sizes | stop-the-world |
|---|---|---|---|
| 1000 Mbit/s | 2 | 7.05 MiB → 96 KiB | 0.97 ms |
| 200 Mbit/s | 2 | 7.05 MiB → 480 KiB | 1.11 ms |
| 60 Mbit/s | 3 | 7.08 MiB → 1.34 MiB → 312 KiB | 8.3 ms |
| 10 Mbit/s | 2, aborted | 7.05 MiB → 8.77 MiB → would be 9.84 MiB | 8.25 s |

Each round carries what the sequence dirtied while the previous one was in
flight, so the ratio between consecutive rounds is (dirty rate / link rate). At
60 Mbit/s that is about 0.19 and three rounds suffice. At 10 Mbit/s it is above
1, the rounds grow, and no number of them converges:

```
rounds stopped: round 2 would send 9.84 MiB after 8.77 MiB, no smaller
                -- link cannot outrun the sequence [DID NOT CONVERGE]
```

The rule fires before spending the round, because a round that will not shrink
costs its whole duration and leaves at least as much to do afterwards. The
8.25-second pause that follows is the honest answer: at 10 Mbit/s a sequence
generating 1.4 MB/s of KV cannot be migrated live, and pretending otherwise
only makes the pause longer.

### How a handover works

The decode loop maintains one invariant and the whole design rests on it: at the
top of every iteration, `filled < tokens.len()`. `filled` is how many positions
are in the KV cache and `tokens` is the prompt plus everything generated, so
there is always at least one token that has been decided but not yet run through
the model. A sequence paused there needs no logits, no half-finished layer state
and no in-flight anything. Feed `tokens[filled]` at position `filled` and carry
on.

So a handover is the pages, the token list, the sampler configuration and four
words of RNG state: a few kilobytes plus the cache. Because the invariant holds
during prefill exactly as during generation, a sequence can be handed over in
the middle of its prompt, and the determinism suite does that on purpose.

```
source                                             destination
  |-- Hello: fingerprint, cache shape, page size -->|
  |<-- HelloAck ---------------------------------- | allocate cache,
  | arm dirty tracking                              | start decode thread
  |-- round 0: everything live right now ---------->| (parked on `ready`)
  |     ...decode continues throughout...           |
  |-- round k: only what changed since round k-1 -->|
  | park the decode loop        <--- window opens   |
  |-- residue + tokens + sampler + RNG ------------>|
  |                                                 | fill in state, unpark
  |<-- ResumeAck ---------------  window closes     | decode resumes here
```

Two things are deliberately outside the window. The destination's decode thread
is created during pre-copy and parks on a flag, because spawning a thread was
measured at 48 µs on a good day and 943 µs on a bad one. And the destination's
KV hash check is off unless asked for, because re-hashing a live cache costs
real microseconds; `verify` turns it on.

### Dirty-page tracking

Four backends, one acceptance suite. `HS_TRACKER=signal cargo test` runs the
whole suite against a chosen one, because a table of mechanisms is a claim and a
table of mechanisms that all pass the same tests is a result.

| backend | mechanism | cost model |
|---|---|---|
| `mach-vm-protect` | `mach_vm_protect` + `EXC_BAD_ACCESS` on a Mach exception port | per first write to a page |
| `mprotect-signal` | `mprotect` + `SIGSEGV`/`SIGBUS`, handled on the faulting thread | per first write to a page |
| `uffd-wp` | `userfaultfd` write-protect mode | per first write to a page |
| `soft-dirty` | `clear_refs` + `pagemap` bit 55 | per scan, proportional to region size |

Measured on a 112 MiB region, median of five runs:

| | macOS, 16 KiB pages | Linux container, 4 KiB pages |
|---|---|---|
| `mach-vm-protect`, per fault | 13.0 µs | — |
| `mprotect-signal`, per fault | 3.10 µs | 1.24 µs |
| of which inside our handler | 0.35 µs | 0.60 µs |
| write to an already-dirty page | 6.4 ns | — |
| re-arm an untouched region | 0.2 µs | 0.6 µs |
| re-arm after every page has faulted | 134 µs (7,168 pages) | 1,245 µs (28,672 pages) |

Handling the fault on the faulting thread is 4x cheaper than messaging it to
another one: the Mach path costs two `mach_msg` round trips and a context
switch, the signal path costs a trap and a return.

Re-arming is not free once the mapping is fragmented. Every per-page permission
change splits a VM map entry, so the bulk `mprotect` that starts a round goes
from 0.2 µs to 134 µs after 7,168 individual faults. That is the argument for
`--cluster`: unprotecting N pages per fault trades a coarser dirty set for fewer
faults and a less shredded map. At cluster 8 the macOS backend moves 760k
pages/s instead of 222k.

Not writing costs nothing and neither does writing twice. 6.4 ns is a plain
store, so tracking is expensive only at the boundary between clean and dirty,
which is exactly the property pre-copy needs.

### Determinism

`scripts/verify-determinism.sh` runs a reference sequence start to finish on one
worker, then re-runs it handing over at five pinned positions, one of them
inside the prompt, and once as an A→B→A→B round trip:

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

The control is the point. `migrate ... norng` transfers everything except the
sampler state; the KV hashes still match, the handover still succeeds, and the
transcripts come apart. A determinism test that cannot fail is not a test.

Three things had to be true to get here: one accumulation order everywhere, no
libm, and a sampler RNG that actually moves. [NOTES.md](NOTES.md) explains each.

### Cross-OS and containers

`scripts/demo-crosshost.sh` moves a sequence from a native macOS worker into a
Linux container:

```
  macos-host       macos/aarch64  16384 B pages  mach-vm-protect
  linux-container  linux/aarch64  4096 B pages   mprotect-signal

PASS  identical token stream across macOS -> Linux container
```

Different OS, page size, dirty-tracking mechanism and C library, and the same
tokens, with the destination's own hash matching the source's. Page size never
enters the protocol: the tracker's unit is a page, but the transfer's unit is a
byte range, so a 16 KiB-page sender and a 4 KiB-page receiver need no
translation.

`scripts/demo-containers.sh` puts two workers on a Docker network with the model
mounted read-only, on `mincontainer`'s dev image. Weights are never migrated and
never should be. Note that Docker Desktop's LinuxKit kernel has both
`CONFIG_USERFAULTFD` and `CONFIG_MEM_SOFT_DIRTY` off, and the soft-dirty failure
is silent; see [NOTES.md](NOTES.md).

### Limitations

One sequence per worker. Not a protocol limit: the macOS tracker is
process-wide, so supporting several means one tracker owning several regions and
dispatching faults by address. The signal backend already works that way.

No batching, one token at a time, prefill included. That is deliberate, so
prefill and decode dirty the cache through the identical code path and the
migration numbers describe one mechanism rather than two. It also makes prefill
slow.

Post-copy is not implemented. When pre-copy cannot converge the honest options
are a long pause, which is what this does, or demand-paging the cache from the
source after the switch. The second is the better answer and the tracking
machinery is most of what it needs.

`uffd-wp` is written but untested on real hardware, because no kernel available
here has `CONFIG_USERFAULTFD` on. It passes review, not tests.

Qwen2 only. The loader handles F32/F16/Q4_K/Q5_K/Q5_0/Q6_K/Q8_0, but the forward
pass is one architecture.

The hash detects corruption and divergence, not an adversary.

### Building and testing

```
$ cargo test --release                                 # 72 tests
$ HS_TRACKER=signal cargo test --release -p hs-track   # same suite, other backend

$ cargo run --release -p hs-track --example faultbench -- 112 mach
$ cargo run --release -p hs-track --example faultbench -- 112 signal

$ cargo run --release -p hs-engine --example ppl -- models/<model>.gguf bench/ppl.txt 256
$ llama-perplexity -m models/<model>.gguf -f bench/ppl.txt --ctx-size 256
```
