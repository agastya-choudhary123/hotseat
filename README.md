# hotseat

Move a *running* LLM request between machines without stopping it.

A decoding sequence is frozen on one worker, its KV cache, sampler state and
random stream are shipped to another, and it resumes there mid-sentence and
finishes the token stream it started — bit for bit the same tokens it would have
produced had it never moved.

The interesting part is not the transfer. It is that the sequence keeps decoding
*while* the transfer happens. That needs the machine to tell you which pages of
the cache changed under you, which on macOS means write-protecting the cache and
catching `EXC_BAD_ACCESS` on a Mach exception port, and on Linux means
`mprotect` and a `SIGSEGV` handler — then iterative pre-copy rounds and a short
stop-and-copy at the end. It is the algorithm live VM migration has used for
twenty years, applied to something nobody applies it to.

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

## Contents

- [The headline numbers](#the-headline-numbers)
- [What's here](#whats-here)
- [Quick start](#quick-start)
- [How a handover works](#how-a-handover-works)
- [Dirty-page tracking](#dirty-page-tracking)
- [Why the tokens come out identical](#why-the-tokens-come-out-identical)
- [Pre-copy, and when it stops helping](#pre-copy-and-when-it-stops-helping)
- [Cross-OS migration](#cross-os-migration)
- [Containers](#containers)
- [The engine](#the-engine)
- [Design decisions](#design-decisions)
- [The hardest bugs](#the-hardest-bugs)
- [Limitations](#limitations)
- [Building and testing](#building-and-testing)

## The headline numbers

Apple M4, 16 GB, Qwen2.5-0.5B-Instruct Q4_K_M, 24 KiB of KV per token.

| | |
|---|---|
| **Stop-the-world window**, loopback | **174 – 330 µs** |
| Stop-the-world, container to container | 473 µs |
| Residual bytes moved with the sequence stopped | ~5 – 20 KiB |
| Dirty-page fault, `mprotect` + signal (macOS / Linux) | 3.10 µs / 1.24 µs |
| Dirty-page fault, Mach exception port (macOS) | 13.0 µs |
| Write to a page already dirty this round | 6.4 ns |
| Transcript after a handover | token-identical to a run that never moved |

The window is measured from the instant the decode loop parks to the instant the
destination says it is running. It excludes the tail of the token that was in
flight when the migration asked to stop — that time is reported separately as
`park`, because the sequence was still producing during it.

## What's here

| | |
|---|---|
| **Engine** | GGUF loader, Qwen2 decoder, byte-level BPE, sampler — no ML dependencies |
| **Dirty tracking** | Mach exception ports, `mprotect`+signal, `userfaultfd`, soft-dirty |
| **Migration** | iterative pre-copy over raw TCP, convergence rules, stop-and-copy |
| **Proof** | transcript equality against a never-migrated reference, with a control |
| **Deployment** | two containers on a Docker network; macOS host to Linux container |
| **Size** | ~6,800 lines of Rust, ~1,150 of tests |

The only dependency in the whole workspace is `libc`. The tokenizer, the
transcendentals, the hashes, the wire format, the thread pool and every syscall
binding are in this tree, and there is a reason for each one in the file that
holds it.

## Quick start

```bash
cargo build --release

# two workers, same model file, on one machine
./target/release/hs-worker --model models/qwen2.5-0.5b-instruct-q4km.gguf \
    --listen 127.0.0.1:7401 --name A &
./target/release/hs-worker --model models/qwen2.5-0.5b-instruct-q4km.gguf \
    --listen 127.0.0.1:7402 --name B &

H=./target/release/hotseat
$H 127.0.0.1:7401 start max=120 temp=0.8 seed=99 "Write a paragraph about virtual memory."
$H 127.0.0.1:7401 migrate 127.0.0.1:7402 at=60 verify   # hand it over at position 60
$H 127.0.0.1:7402 wait                                  # B finishes what A started
```

`at=60` pins the handover to an exact cache position instead of racing it with a
sleep, which is what makes the determinism tests repeatable.

Three scripts run the whole story:

```bash
scripts/verify-determinism.sh   # transcripts identical at five migration points
scripts/demo-precopy.sh         # pre-copy across four shaped link rates
scripts/demo-containers.sh      # container to container
scripts/demo-crosshost.sh       # macOS host -> Linux container
```

## How a handover works

The decode loop maintains one invariant, and the whole design rests on it:

> at the top of every iteration, `filled < tokens.len()`

`filled` is how many positions are in the KV cache; `tokens` is the prompt plus
everything generated. There is always at least one token that has been *decided*
but not yet *run through the model*. A sequence paused there needs no logits, no
half-finished layer state, no in-flight anything. Feed `tokens[filled]` at
position `filled` and carry on.

So a handover is: the pages, the token list, the sampler configuration and the
four words of RNG state. A few kilobytes plus the cache. And because the
invariant holds during prefill exactly as it does during generation, a sequence
can be handed over in the middle of its prompt — the determinism suite does that
on purpose.

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

Two things are deliberately *outside* the window. The destination's decode
thread is created during pre-copy and parks on a flag, because spawning a thread
was measured at 48 µs on a good day and 943 µs on a bad one. And the
destination's KV hash check is off unless asked for, because re-hashing a live
cache costs real microseconds; `verify` turns it on and the report prints both.

## Dirty-page tracking

Four backends, one acceptance suite. `HS_TRACKER=signal cargo test` runs the
whole suite against a chosen one, because a table of mechanisms is a claim and a
table of mechanisms that all pass the same tests is a result.

| backend | mechanism | cost model |
|---|---|---|
| `mach-vm-protect` | `mach_vm_protect` + `EXC_BAD_ACCESS` on a Mach exception port | per first write to a page |
| `mprotect-signal` | `mprotect` + `SIGSEGV`/`SIGBUS`, handled on the faulting thread | per first write to a page |
| `uffd-wp` | `userfaultfd` write-protect mode | per first write to a page |
| `soft-dirty` | `clear_refs` + `pagemap` bit 55 | per *scan*, proportional to region size |

Measured on a 112 MiB region, median of five runs:

| | macOS, 16 KiB pages | Linux container, 4 KiB pages |
|---|---|---|
| `mach-vm-protect`, per fault | 13.0 µs | — |
| `mprotect-signal`, per fault | 3.10 µs | 1.24 µs |
| ...of which inside our handler | 0.35 µs | 0.60 µs |
| write to an already-dirty page | 6.4 ns | — |
| re-arm an untouched region | 0.2 µs | 0.6 µs |
| re-arm after every page has faulted | 134 µs (7,168 pages) | 1,245 µs (28,672 pages) |

Three things fall out of that table.

**Handling the fault on the faulting thread is 4x cheaper than messaging it to
another one.** The Mach path costs two `mach_msg` round trips and a context
switch; the signal path costs a trap and a return. Same machine, same region,
same work.

**Re-arming is not free once the mapping is fragmented.** Every per-page
permission change splits a VM map entry, so the bulk `mprotect` that starts a
round goes from 0.2 µs to 134 µs after 7,168 individual faults. That is the
argument for the `--cluster` knob: unprotecting N pages per fault trades a
coarser dirty set for fewer faults and a less shredded map. At cluster 8 the
macOS backend moves 760k pages/s instead of 222k.

**Not writing costs nothing, and neither does writing twice.** 6.4 ns is a plain
store. Tracking is only expensive at the boundary between clean and dirty, which
is exactly the property pre-copy needs.

### Round boundaries without losing a write

Re-arming has to (a) make every page fault again and (b) take the accumulated
set, and a write can land between those two steps. Both fault-driven backends
use a two-buffer epoch:

```
handler:   e0 = epoch                  harvest:  epoch += 1
           mark(buf[e0 & 1], page)               protect(whole region)
           unprotect(page)                       drain buf[old & 1]
           e1 = epoch
           if e1 != e0 { mark(buf[e1 & 1], page) }
```

If the handler's `unprotect` lands after the bulk `protect`, the epoch bump —
which happens first — is already visible, so `e1 != e0` and the page is marked
into the round that is now current. If it lands before, the bulk protect re-arms
it anyway. A page that is writable is a page that is marked. Reporting a page
twice is free; missing one corrupts the transfer.

`soft-dirty` cannot do this, and says so. There is no way to read the bits and
clear them atomically, so a write in that window is lost silently and forever.
CRIU gets away with it by freezing the process first. `Tracker::is_exact()`
returns false for it, and the concurrent-writer test prints how many writes it
dropped rather than pretending to pass.

## Why the tokens come out identical

Three things had to be true.

**One accumulation order, everywhere.** Every reduction in the engine has a
single explicit shape — eight lanes, one fixed fold — and the NEON and scalar
dot products use the *same* shape, with a test asserting they agree bit for bit.
Thread count, compiler vectorisation decisions and scheduling cannot move a
result, because work is split by fixed row ranges and never stolen.

**No libm.** `expf` on Apple's libSystem and `expf` in glibc differ in the last
place. A sequence migrating from a Mac to a Linux container would diverge on the
first softmax. So `exp`, `ln` and `sincos` are implemented in `math.rs` as plain
IEEE f32 arithmetic with no fused multiply-adds, and rotary tables are computed
once at load so the decode step contains no transcendental at all.

**The RNG moves.** The sampler uses a *stateful* xoshiro256\*\* stream that
advances on every draw. A counter-based generator keyed on position would have
been easier — there would be nothing to migrate — but it would also make the
test vacuous. With a real stream, forgetting the four state words produces
fluent, plausible, different text.

`scripts/verify-determinism.sh` runs a reference sequence start to finish on one
worker, then re-runs it handing over at five pinned positions (one inside the
prompt) and once as an A→B→A→B round trip, and diffs the token lists:

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

## Pre-copy, and when it stops helping

Loopback moves a KV cache far faster than a sequence can dirty it, so the rounds
converge immediately and the algorithm never shows its work. `migrate ... mbps=N`
shapes the bulk transfer so they do. Same sequence, 563-token prompt, ~7 MiB live
cache, `scripts/demo-precopy.sh`:

| link | rounds | round sizes | stop-the-world |
|---|---|---|---|
| 1000 Mbit/s | 2 | 7.05 MiB → 96 KiB | **0.97 ms** |
| 200 Mbit/s | 2 | 7.05 MiB → 480 KiB | **1.11 ms** |
| 60 Mbit/s | 3 | 7.08 MiB → 1.34 MiB → 312 KiB | **8.3 ms** |
| 10 Mbit/s | 2, aborted | 7.05 MiB → 8.77 MiB → *would be 9.84 MiB* | **8.25 s** |

Read the round column. Each round carries what the sequence dirtied while the
previous one was in flight, so the ratio between consecutive rounds is
(dirty rate / link rate). At 60 Mbit/s that ratio is about 0.19 and three rounds
suffice. At 10 Mbit/s it is above 1, the rounds *grow*, and no number of them
will ever converge:

```
rounds stopped: round 2 would send 9.84 MiB after 8.77 MiB, no smaller
                -- link cannot outrun the sequence [DID NOT CONVERGE]
```

The rule fires *before* spending the round, because a round that will not shrink
costs its whole duration and leaves at least as much to do at the end of it. The
8.25-second pause that follows is the honest answer: at 10 Mbit/s, a sequence
generating 1.4 MB/s of KV cannot be migrated live, and pretending otherwise just
makes the pause longer.

A KV cache converges far better than a general-purpose VM would, for a reason
worth stating: it is append-mostly. The dirty set is a small moving tail, not a
working set scattered over the whole allocation. The migration exploits that
explicitly with a low-water mark — positions already handed over cannot change,
so later rounds never re-send them. Without it, a dirty 16 KiB page drags along
the 32 settled positions sharing it, and the residual round was 391 KiB instead
of 16 KiB.

## Cross-OS migration

`scripts/demo-crosshost.sh` moves a sequence from a native macOS worker into a
Linux container:

```
the two ends:
  macos-host       macos/aarch64  16384 B pages  mach-vm-protect
  linux-container  linux/aarch64  4096 B pages   mprotect-signal

reference: 120 tokens, never left the Mac
  live cache 1.43 MiB | kv hash src 0x9703d293caa76228 dst 0x9703d293caa76228

PASS  identical token stream across macOS -> Linux container
```

Different operating system, different page size, different dirty-tracking
mechanism, different C library — and the same tokens, with the destination's own
hash of the cache matching the source's. Page size never enters the protocol:
the tracker's unit is a page, but the transfer's unit is a byte range, so a
16 KiB-page sender and a 4 KiB-page receiver need no translation.

## Containers

`mincontainer`'s dev image is the base — it already carries the Rust toolchain
this needs — and `scripts/demo-containers.sh` puts two workers on a Docker
network with the model mounted read-only. Weights are never migrated and never
should be: each worker loads its own copy and the handshake refuses the transfer
if the fingerprints differ.

Getting Linux tracking working turned up something worth recording. Docker
Desktop's LinuxKit kernel — what an Apple Silicon Mac actually runs containers
on — has **`CONFIG_USERFAULTFD` off**; the syscall returns `ENOSYS` even under
`--privileged`. It also has **`CONFIG_MEM_SOFT_DIRTY` off**, and that failure is
silent: the write to `clear_refs` succeeds, `pagemap` reads fine, and no bit is
ever set. A tracker built on it does not look broken. It looks *converged*, and
it would hand over a stale cache.

So `SoftDirtyTracker` now proves the kernel implements soft-dirty before it will
construct — dirty a probe page, check the bit, refuse to exist if it is not
there — and the portable `mprotect`+signal backend is what actually runs in
containers. It needs no capability, no sysctl and no kernel config.

## The engine

There is no inference library here. Loading a GGUF, running Qwen2 and sampling
from it are about 1,500 lines, and writing them was the only way to own the two
things this project needs: where the KV cache lives (someone else's allocation,
so a tracker can watch it) and what the arithmetic does (bit-identically, on
every host).

Correctness is checked against llama.cpp on the same files, the same 256-token
windows and the same second-half scoring:

| model | hotseat | llama.cpp | delta |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct Q4_K_M | 18.140 | 18.113 | +0.15% |
| Qwen2.5-1.5B-Instruct Q4_K_M | 12.486 | 12.487 | −0.01% |

The residual is a deliberate choice: weights are dequantised to f16 at load
rather than kept quantised. A quantised dot product would be faster per byte,
but this project's subject is migration, and f16 gives one simple numeric path
that is identical everywhere — so "the token stream survives the boundary" is a
claim about the migration and not about two matmul kernels agreeing. The f16
rounding error sits well under the Q4_K error it is applied on top of.

## Design decisions

**The KV cache belongs to the tracker, not the engine.** `hs-track` allocates
the region; the engine only indexes into it. The engine has no write log, no
barriers and no idea it is being watched — the dirty set comes from the
hardware. That is what makes the measurement mean something.

**Layout is a migration decision, not just a compute one.** `token-major`
(`[layer][k|v][pos][head][dim]`) makes one decode step touch `2 × n_layers`
pages, each absorbing the next several tokens. `head-major` makes attention read
positions sequentially — better for the matmul — but a step then dirties
`2 × n_layers × n_kv_heads` pages, twice as many. Both are implemented and
selectable with `--layout`; the tests assert the page-count difference.

**One dispatch at a time in the thread pool.** The pool has a single job slot,
so two threads calling `broadcast` concurrently would have workers run one
closure with the other's bounds. A worker hosting a sequence while receiving
another has exactly two decode threads sharing one pool. See the bugs below.

**Workers block after spinning.** Dispatches are hundreds of nanoseconds apart
inside a token, so the pool spins first; but spinning forever meant two idle
workers burned six cores between them, and a migration thread waited 112 ms to
be scheduled for work that takes 11 ms. Spin, then yield, then block. Measured
A/B: no throughput cost, and idle CPU went from ~300% to 0.6%.

**Shared-secret auth, private network.** The migration port accepts memory into
another process's address space. The token is not authentication worth the name;
the deployment story is a private network, as it is for the container runtime
this grew out of.

## The hardest bugs

**The one page-rounding was hiding.** Transfers originally rounded dirty pages
out to whole pages. Clipping them to exactly the live bytes — a 32× reduction in
the residual round — immediately broke the KV hash check. The cause was not the
clipping. Round 0 clipped at `filled`, but the decode loop is *mid-forward* on
position `filled` when the tracker is armed, so the layers it had already
written were neither tracked (they predate the arm) nor sent (past the clip) —
and since each (layer, position) slot is written exactly once, they were never
offered again. Rounding out to page boundaries had been dragging those bytes
along by accident for weeks of wall time. Pre-copy rounds now clip one position
past `filled`, and the fix is a two-line change sitting under a twelve-line
comment explaining why.

**A harvest thrown away.** The convergence check harvests a round, decides it is
not worth sending, and breaks — and stop-and-copy then *cleared* the dirty set
before its own harvest. Those pages had already been re-armed, so nothing would
ever report them again: about 10 MB of KV silently lost, on the slow-link path
only. The fix is that `dirty` is cleared after a round is *sent*, never after
one is merely harvested. Both of these were caught by the destination re-hashing
the cache, which is the argument for `verify` existing at all.

**Two decode threads, one pool.** Pre-spawning the destination's decode thread
took thread creation out of the stop-the-world window and introduced a window
where a worker has two decode threads. `wait_done` returned when the *flag* was
set, not when the thread had left the loop, so a finished sequence could still
be inside a forward pass while the next one started. The symptom was a pool
worker indexing a 128-element bias at 688. Fixed twice over: the pool serialises
dispatch, and a sequence now reports when its thread has actually exited.

**A Mach message layout off by four bytes.** MIG lays `mach_exception_raise` out
at four-byte alignment, so the two 64-bit codes straddle an eight-byte boundary.
Letting Rust align them naturally shifted everything past `codeCnt`, and the
faulting address came through as `0x1` — the real address rotated. `packed(4)`,
and a diagram in the comment.

**A tracker per region.** `task_set_exception_ports` is task-wide. One port per
tracker looks fine until two overlap for an instant: the second `set` displaces
the first, and the first one's drop restores the ports *it* saved, deregistering
the second. Starting a second sequence in a worker did exactly that, and an
ordinary store took SIGBUS. The port and handler are now a process singleton
owning a registry of regions.

## Limitations

- **One sequence per worker.** Not a protocol limit — the macOS tracker is
  process-wide, so supporting several would mean one tracker owning several
  regions and dispatching faults by address. The signal backend already works
  that way; the Mach one would need the same treatment.
- **No batching.** One token at a time, prefill included. That is deliberate:
  prefill and decode then dirty the cache through the identical code path, so
  the migration numbers describe one mechanism rather than two. It also makes
  prefill slow.
- **Post-copy is not implemented.** When pre-copy cannot converge, the honest
  options are a long pause (what this does) or demand-paging the cache from the
  source after the switch. The second is the better answer and the tracking
  machinery is most of what it needs.
- **`uffd-wp` is written but untested on real hardware**, because no kernel
  available here has `CONFIG_USERFAULTFD` on. It passes review, not tests; treat
  it as such.
- **Qwen2 only.** The loader handles F32/F16/Q4_K/Q5_K/Q5_0/Q6_K/Q8_0, but the
  forward pass is one architecture.
- **The hash is not cryptographic.** It detects corruption and divergence, not
  an adversary.

## Building and testing

```bash
cargo test --release            # 72 tests
HS_TRACKER=signal cargo test --release -p hs-track   # same suite, other backend

# what tracking costs, on this machine
cargo run --release -p hs-track --example faultbench -- 112 mach
cargo run --release -p hs-track --example faultbench -- 112 signal

# engine correctness against llama.cpp
cargo run --release -p hs-engine --example ppl -- models/<model>.gguf bench/ppl.txt 256
llama-perplexity -m models/<model>.gguf -f bench/ppl.txt --ctx-size 256

# containers
docker build -t hotseat .
docker run --rm -e HS_TRACKER=signal --entrypoint /usr/local/bin/tracker-tests hotseat
```

The model is any Qwen2 GGUF; the demos default to Qwen2.5-0.5B-Instruct Q4_K_M
and everything has also been run against Qwen2.5-1.5B-Instruct Q4_K_M.
