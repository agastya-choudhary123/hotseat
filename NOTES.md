# hotseat — implementation notes

Material that used to live in the README: the design decisions, the round-epoch
protocol, and the bugs that were worth writing down.

## Round boundaries without losing a write

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

If the handler's `unprotect` lands after the bulk `protect`, the epoch bump
happens first and is already visible, so `e1 != e0` and the page is marked into
the round that is now current. If it lands before, the bulk protect re-arms it
anyway. A page that is writable is a page that is marked. Reporting a page twice
is free; missing one corrupts the transfer.

`soft-dirty` cannot do this and says so. There is no way to read the bits and
clear them atomically, so a write in that window is lost silently and forever.
CRIU gets away with it by freezing the process first. `Tracker::is_exact()`
returns false for it, and the concurrent-writer test prints how many writes it
dropped rather than pretending to pass.

## Design decisions

The KV cache belongs to the tracker, not the engine. `hs-track` allocates the
region and the engine only indexes into it. The engine has no write log, no
barriers and no idea it is being watched; the dirty set comes from the hardware.
That is what makes the measurement mean anything.

Layout is a migration decision, not only a compute one. `token-major`
(`[layer][k|v][pos][head][dim]`) makes one decode step touch `2 × n_layers`
pages, each absorbing the next several tokens. `head-major` makes attention read
positions sequentially, which is better for the matmul, but a step then dirties
`2 × n_layers × n_kv_heads` pages, twice as many. Both are implemented and
selectable with `--layout`, and the tests assert the page-count difference.

One dispatch at a time in the thread pool. The pool has a single job slot, so
two threads calling `broadcast` concurrently would have workers run one closure
with the other's bounds. A worker hosting a sequence while receiving another has
exactly two decode threads sharing one pool.

Workers block after spinning. Dispatches are hundreds of nanoseconds apart
inside a token, so the pool spins first. Spinning forever meant two idle workers
burned six cores between them, and a migration thread waited 112 ms to be
scheduled for work that takes 11 ms. Spin, then yield, then block. Measured A/B:
no throughput cost, and idle CPU went from ~300% to 0.6%.

Shared-secret auth on a private network. The migration port accepts memory into
another process's address space. The token is not authentication worth the name;
the deployment story is a private network, as it is for the container runtime
this grew out of.

## Why the tokens come out identical

One accumulation order, everywhere. Every reduction in the engine has a single
explicit shape, eight lanes and one fixed fold, and the NEON and scalar dot
products use the same shape with a test asserting they agree bit for bit. Thread
count, compiler vectorisation decisions and scheduling cannot move a result,
because work is split by fixed row ranges and never stolen.

No libm. `expf` on Apple's libSystem and `expf` in glibc differ in the last
place, so a sequence migrating from a Mac to a Linux container would diverge on
the first softmax. `exp`, `ln` and `sincos` are implemented in `math.rs` as
plain IEEE f32 arithmetic with no fused multiply-adds, and rotary tables are
computed once at load so the decode step contains no transcendental at all.

The RNG moves. The sampler uses a stateful xoshiro256** stream that advances on
every draw. A counter-based generator keyed on position would have been easier,
since there would be nothing to migrate, but it would also make the test
vacuous. With a real stream, forgetting the four state words produces fluent,
plausible, different text.

## Why a KV cache pre-copies well

It is append-mostly. The dirty set is a small moving tail rather than a working
set scattered over the whole allocation, which is why it converges far better
than a general-purpose VM would. The migration exploits that explicitly with a
low-water mark: positions already handed over cannot change, so later rounds
never re-send them. Without it, a dirty 16 KiB page drags along the 32 settled
positions sharing it, and the residual round was 391 KiB instead of 16 KiB.

## LinuxKit has no usable dirty tracking

Docker Desktop's LinuxKit kernel, which is what an Apple Silicon Mac actually
runs containers on, has `CONFIG_USERFAULTFD` off; the syscall returns `ENOSYS`
even under `--privileged`. It also has `CONFIG_MEM_SOFT_DIRTY` off, and that
failure is silent: the write to `clear_refs` succeeds, `pagemap` reads fine, and
no bit is ever set. A tracker built on it does not look broken. It looks
converged, and it would hand over a stale cache.

`SoftDirtyTracker` now proves the kernel implements soft-dirty before it will
construct: dirty a probe page, check the bit, refuse to exist if it is not
there. The portable `mprotect`+signal backend is what actually runs in
containers, and it needs no capability, no sysctl and no kernel config.

## The hardest bugs

The one page-rounding was hiding. Transfers originally rounded dirty pages out
to whole pages. Clipping them to exactly the live bytes, a 32x reduction in the
residual round, immediately broke the KV hash check. The cause was not the
clipping. Round 0 clipped at `filled`, but the decode loop is mid-forward on
position `filled` when the tracker is armed, so the layers it had already
written were neither tracked, because they predate the arm, nor sent, because
they are past the clip. Since each (layer, position) slot is written exactly
once, they were never offered again. Rounding out to page boundaries had been
dragging those bytes along by accident for weeks of wall time. Pre-copy rounds
now clip one position past `filled`, and the fix is a two-line change under a
twelve-line comment explaining why.

A harvest thrown away. The convergence check harvests a round, decides it is not
worth sending, and breaks, and stop-and-copy then cleared the dirty set before
its own harvest. Those pages had already been re-armed, so nothing would ever
report them again: about 10 MB of KV silently lost, on the slow-link path only.
`dirty` is now cleared after a round is sent, never after one is merely
harvested. Both of these were caught by the destination re-hashing the cache,
which is the argument for `verify` existing at all.

Two decode threads, one pool. Pre-spawning the destination's decode thread took
thread creation out of the stop-the-world window and introduced a window where a
worker has two decode threads. `wait_done` returned when the flag was set, not
when the thread had left the loop, so a finished sequence could still be inside
a forward pass while the next one started. The symptom was a pool worker
indexing a 128-element bias at 688. Fixed twice over: the pool serialises
dispatch, and a sequence now reports when its thread has actually exited.

A Mach message layout off by four bytes. MIG lays `mach_exception_raise` out at
four-byte alignment, so the two 64-bit codes straddle an eight-byte boundary.
Letting Rust align them naturally shifted everything past `codeCnt`, and the
faulting address came through as `0x1`, the real address rotated. `packed(4)`,
and a diagram in the comment.

A tracker per region. `task_set_exception_ports` is task-wide. One port per
tracker looks fine until two overlap for an instant: the second `set` displaces
the first, and the first one's drop restores the ports it saved, deregistering
the second. Starting a second sequence in a worker did exactly that, and an
ordinary store took SIGBUS. The port and handler are now a process singleton
owning a registry of regions.
