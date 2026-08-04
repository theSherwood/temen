# GO.md — running Go on SVM: feasibility & work breakdown

Status: **scoping doc, pre-implementation**, written 2026-08-04. Answers "how much work
would it be to get Go running on SVM?" against what the tree provides today: migratable
stackful fibers (`cont.*`, D57/§23), conservative root enumeration (`gc.roots`, `GC.md`),
N-vCPU parallelism (`thread.spawn`, futex, C11 atomics), and two mature untrusted
on-ramps (`svm-llvm`, `svm-wasm`). It leans on the frontend trust model (`DESIGN.md`
§2a, `FRONTEND.md` §1) and the language-port precedents (`NIM.md`, `TYPESCRIPT.md`,
JACL). Fold settled sections into `DESIGN.md` and drop this file once work lands — the
repo convention.

## 0. TL;DR

- **The runtime substrate is a strong fit — deliberately so.** DESIGN.md §12 names
  goroutines as an intended guest threading model; `demos/steal_fibers` already runs
  work-stealing M:N scheduling of migratable stackful fibers across OS threads as pure
  guest code; `GC.md` §2 specifies the stop-the-world handshake as "Go-style STW"; fuel
  yield-point preemption is described in DESIGN.md §12 as "Go-style async preemption".
  Goroutines → fibers, M:N scheduler → guest code over `thread.spawn` + `cont.resume` +
  futex, GC stack roots → `gc.roots`. The primitives were shaped for exactly this.
- **The official `gc` toolchain ported natively is the wrong target.** Three structural
  mismatches, not effort gaps: (1) Go grows goroutine stacks by *copying*, but the SVM
  control stack is out-of-band, fixed-size, and guest-unaddressable by design (§3d/§5 —
  DESIGN.md §23 notes the contrast explicitly); (2) Go's GC is *precise*, and precise
  stack maps are deliberately deferred (`GC.md` §6) — conservative `gc.roots` forbids
  moving/copying anything a root points at; (3) the `gc` runtime is full of assembly
  and `g`-register TLS that no on-ramp lowers. A native `gc` port is person-years *and*
  would fight the design. Not recommended.
- **Two recommended routes, cheap-first, both bounded:**
  - **Route A — stock Go via `GOOS=wasip1 GOARCH=wasm` → `svm-wasm`.** The `gc`
    compiler's wasm output is plain linear-memory core wasm (not wasm-GC): the Go
    runtime brings its own allocator, GC, and goroutine multiplexing *inside* the
    module. svm-wasm is feature-complete for typical toolchain output; the gap is
    widening the `svm-wasi` preview1 shim from 2 syscalls to the ~20 Go uses.
    **Order: 1–2 weeks.** Buys: unmodified full Go (reflection, maps, channels,
    stdlib) running single-threaded — a compatibility oracle for Route B. Uses
    neither fibers nor `gc.roots`.
  - **Route B — TinyGo → LLVM IR → `svm-llvm`, runtime retargeted to SVM
    primitives.** TinyGo is the Go-semantics compiler whose runtime already matches
    SVM's contract: **conservative, non-moving mark-sweep GC** and **fixed-size
    goroutine stacks** — the exact shape `GC.md` §1 hosts. The port replaces TinyGo's
    per-arch task-switch assembly with `__vm_fiber_new/resume/suspend`, wires its GC's
    stack scan to `__vm_gc_roots`, and its allocator to the window heap. **Order:
    4–8 weeks to single-vCPU goroutines + GC; +4–6 weeks for parallel M:N work-stealing
    goroutines with cross-vCPU STW** (the `steal_fibers` + `gc_quiesce` patterns,
    already proven in C). This is the route that uses what SVM uniquely provides.
- **Known svm-side gaps Route B hits (all small, listed in §4):** `fence` lowering in
  svm-llvm (parses, `Unsupported`), the LLVM TLS gap (avoidable in a retargeted
  runtime), a per-`cont.new` control-stack-size knob for goroutine-scale fiber counts
  (today a host constant, 256 KiB/fiber), and the setjmp×fibers JIT decline
  (avoidable — TinyGo doesn't need setjmp).
- **Route D — multi-threaded self-hosting Go (§6, added 2026-08-04):** a direct
  **Go-SSA → svm-ir backend** (a pure-Go compiler; TinyGo's LLVM dependency is
  exactly what blocks self-hosting it, so the backend replaces LLVM rather than
  feeding it) + the Route B2 runtime. Because the compiler is pure Go with no cgo,
  self-hosting is then *by construction*: whatever the backend can compile includes
  the backend. **Order: the largest language project in the tree so far — roughly
  3–5× the svm-leng effort** (backend ≈ months of slices, the go/types+go/ssa
  stdlib tail is the long pole), with the gc-via-wasm lane (§6.5) closing a
  toolchain-on-svm loop early for near-zero compiler work.

## 1. What Go needs, and what SVM answers

| Go runtime need | SVM answer | Status |
|---|---|---|
| Goroutines (cheap stackful contexts) | Fibers: `cont.new`/`cont.resume`/`suspend`; stack pair per fiber (in-window data stack + out-of-band control stack); ~ns switch; ~16.7M handle space, quota-metered | Built, loom-verified single-owner resume (D57) |
| M:N scheduler, work stealing | Guest policy by design (Invariant 4, D56 — the in-VM scheduler was deliberately removed). `demos/steal_fibers`: injector + per-worker deques, idle worker steals a *suspended fiber* and resumes it on its own OS thread | Proven as guest C, both engines |
| Parallelism (`GOMAXPROCS`) | `thread.spawn`/`join` vCPUs = real OS threads; full C11 atomics; `memory.wait`/`notify` futex | Built, incl. browser Workers (`THREADS.md`) |
| GC stack roots | `gc.roots(heap_lo, heap_hi, mask, buf, cap)` — conservative, range-filtered, deduped enumeration of every non-mutating fiber's control stack + saved registers, incl. the caller | Built on interp + Cranelift JIT (`GC.md` §7) |
| Stop-the-world | Guest-coordinated cooperative STW over safepoint polls + futex barrier; reference `quiesce` barrier tested for N=2/4 (`gc_quiesce.rs`); "coordinate the N vCPUs, not the M fibers" | Built (reference), Go-style by name |
| Preemption of tight loops | Fuel-inserted yield points — guest policy; host preempts vCPUs via fuel/epoch (undisableable) | Built (mechanism) |
| Channels, `select`, `sync` | Guest code over atomics + futex (the same substrate Go's own runtime uses on Linux) | Substrate built |
| Timers, `time.Now` | Clock capability | Built |
| Syscalls / `os`, `net` | POSIX personality named imports + capability powerbox (`POSIX.md`); async ring + blocking-offload pool (the "Tokio `spawn_blocking` / Go" shape, DESIGN.md) for STW-safe blocking I/O | Built for C/Nim corpus; surface grows on demand |
| `defer`/`panic`/`recover` | Frontend lowering (TinyGo lowers these itself); `SetJmp`/`LongJmp` core ops and SJLJ EH exist in svm-llvm if needed | Built |
| Stack growth (`morestack`) | **None, by design** — see §3 | Structural mismatch (fixed-size stacks instead) |
| Precise moving GC | **None, by design** — `gc.roots` is conservative; precise stack maps deferred (`GC.md` §6) | Structural mismatch (non-moving conservative instead) |
| `g`-register / TLS | `vcpu.tls.get/set` (per-vCPU i64, migration-aware for per-*CPU* state); per-*goroutine* state via fiber-data-stack-base derivation (§4.3) | Adequate for a retargeted runtime; LLVM `thread_local` lowering absent |

## 2. Routes considered

### Route A — stock `gc` Go via wasm/WASI (the cheap compatibility lane)

`GOOS=wasip1 GOARCH=wasm go build` emits core linear-memory wasm. The Go runtime is
*inside* the module: its own heap, its own (precise, but self-contained) GC over its own
linear-memory stacks, goroutine switching compiled into the functions themselves
(entry `br_table` re-entry dispatch — svm-wasm handles `br_table` natively). Nothing
about wasm-GC (svm-wasm's permanent non-goal) is involved.

Work:
1. **Widen `svm-wasi`** from `fd_write`/`proc_exit` to the preview1 surface Go's
   wasip1 runtime imports (~20: `args_*`, `environ_*`, `clock_time_get`, `random_get`,
   `fd_*` on stdio + preopens, `poll_oneoff`, `sched_yield`, `proc_exit`). Same
   embedder-`HostFn` pattern the existing shim uses; WASI semantics stay outside the
   TCB.
2. Run the Go test-suite-shaped corpus through it; differential against native
   (`stdout`/exit byte-match — the `c_frontend` two-tier pattern).

Estimate: **1–2 weeks.** Caveats: single-threaded (`gc` Go's wasm target has no
threads), large modules (~2 MB hello world), interpreter-tier performance typical of
Go-on-wasm. Value: *unmodified* full-language Go (reflection, maps, GC, channels — all
of it, cooperatively scheduled by Go's own runtime), and a running **oracle** to
differential-test Route B against (Invariant 9 instincts, applied to the frontend).

### Route B — TinyGo retargeted to SVM primitives (the route that uses the machine)

TinyGo compiles Go (a large, real subset: goroutines, channels, `defer`/`panic`/
`recover`, interfaces, maps, slices, most of the stdlib, much of reflection) through
LLVM, with a runtime *already shaped like SVM's contract*:

- **GC:** conservative, non-moving mark-sweep — literally the `GC.md` §1 model. Its
  stack scan is the piece the guest cannot do on SVM (SSA spills live on the
  unaddressable control stack) — replaced by one `__vm_gc_roots` call, which svm-llvm
  already lowers (LLVM.md slice AC; `vm_gc_roots_smoke`).
- **Goroutines:** fixed-size stacks (no `morestack`), a cooperative scheduler over an
  arch-specific context switch — replaced by `__vm_fiber_new/resume/suspend`. This
  deletes the assembly instead of porting it.
- **Allocator:** simple heap over `malloc`-ish primitives → window heap grown via the
  Memory cap (the chibicc/nimony seam).

svm-llvm is the mature on-ramp (QuickJS/SQLite/Lua/Postgres-scale; no libLLVM link;
reads LLVM-21 output; fail-closed `unsup(...)` on anything outside the subset), and
already exposes threads, fibers, futex, atomics, and `gc.roots` to LLVM-emitting
guests — LLVM.md motivates that surface with "a guest language… reaches the VM's
fibers, threads, atomics, futex, conservative GC roots".

Phased work (each phase lands running + differentially tested, per AGENTS.md):

- **B0 — probe (days).** Feed TinyGo `-opt=2` LLVM IR for hello-world →
  `try_translate`. The `unsup(...)` chokepoint yields the exact gap list — the same
  scoping move every capstone (QuickJS, SQLite, Postgres) started with. Expected
  early hits: `fence` (see §4.1), possibly `llvm.stacksave`, TLS if the runtime isn't
  yet retargeted.
- **B1 — single-vCPU goroutines + GC (order 4–8 weeks).** A TinyGo custom target
  (`svm.json` + `GOOS`-ish build tags) whose runtime package binds: task switch →
  fibers; `gc_conservative` stack scan → `gc.roots` (single-fiber caller coverage is
  already sound — §3.1's caller clause); allocator → window; `time`/`syscall` bottom
  edge → POSIX personality imports (the ~15-function nimony pattern, `NIM.md` §3b).
  Exit criteria: a goroutine/channel/GC-stressing corpus runs on interp + Cranelift
  JIT, byte-identical to native TinyGo and to Route A output where the subset overlaps.
- **B2 — parallel M:N (order +4–6 weeks).** N vCPUs via `thread.spawn`, per-worker
  deques + stealing of suspended fibers (transplant `steal_fibers`' scheduler shape
  into the runtime), channels/`sync` over futex, and cross-vCPU STW: safepoint polls
  at back-edges/call sites (piggybacked on the epoch/kill poll, `GC.md` §5) + the
  §2.1 `quiesce` barrier. `GOMAXPROCS` = vCPU count, passed in (svm exposes no count
  intrinsic by design); per-P state indexed by `vcpu.tls.get`.
- **B3 — breadth, on demand.** `net` over the async ring, bigger stdlib surface,
  browser (the bytecode tier runs `gc.roots` modules; `thread.spawn`+`gc.roots`
  currently falls back to the reference interp — see §4.5).

Deliverable shape mirrors the repo's asset lanes: a `demos/go/` corpus +
`build_tinygo.sh`, CI-gated differential tests, toolchain provisioned like nimony's
(`scripts/ci/provision-*.sh`, tests skip when absent).

**How fibers survive the LLVM transform (recorded because it's the question every
reviewer asks).** There is no coroutine/CPS/asyncify rewrite anywhere — that
machinery exists for targets *without* stack switching, and it is what fights the
optimizer. Here the continuation never enters LLVM at all: `__vm_fiber_suspend`
et al. are opaque extern declarations, and LLVM's mandatory conservatism about
unknown calls (no inline, no DCE, no memory motion across, all live state
preserved per the calling convention — a suspend is indistinguishable from
`read()`) is exactly the contract a stackful switch needs. svm-llvm recognizes
the call *by name* and emits the `cont.*`/`suspend` op (slice AC); the backends
define those ops as call-clobbering (§3d), so Cranelift spills live values to the
control stack around them; and since a fiber owns its whole stack pair, resuming
on another OS thread moves the executing thread, not the stack. Empirically
closed, not just argued: all five concurrency demos — `steal_fibers` included,
whose 121920 total is specifically a locals-live-across-migration integrity
check — compile via `clang -O2` → svm-llvm and match the chibicc-frontend build
(LLVM.md, the concurrency-demos slice). Two runtime-authoring disciplines follow:
never hold per-vCPU state (a `vcpu.tls` read, a per-P pointer) in a value live
across a suspend — the compiler will *correctly preserve* the stale value, since
the fiber may resume on a different vCPU (re-read after resume, Go's own parking
discipline); and keep panic/recover on the error-flag path, not setjmp (§4.4).

**Multithreading, precisely:** B1 is single-vCPU (stock TinyGo's scheduler shape
— one OS thread multiplexing tasks). Real parallelism is B2, and the parallel
scheduler is **runtime code we write**, not something TinyGo brings: `thread.spawn`
vCPUs are real OS threads, atomics are genuinely atomic (no single-threaded gate),
the futex is cross-thread, and stolen-fiber migration is proven — but the M:N
policy over them is guest code by design (Invariant 4).

### Route C — native `gc` toolchain port (rejected; recorded so it stays rejected)

A new `GOARCH`/`GOOS` backend in the `gc` compiler emitting SVM IR, plus a runtime
port. Rejected on three structural grounds (§0), any one of which forces deep runtime
surgery: contiguous-stack copying (`morestack`/`stackguard0`) has no counterpart —
control stacks are fixed and unaddressable; the precise bitmap GC would run
*conservative-stacks/precise-heap* at best (a shape Go abandoned at 1.4) and must
never move anything conservatively rooted; runtime assembly and `g`-register TLS need
wholesale replacement. Cost is person-years for a worse fit than Route B. Revisit only
if `GC.md` §6's precise stack maps ever land *and* someone needs gc-exact semantics.

## 3. The stack question, answered plainly

Go's signature trick — start goroutines at 2–8 KiB and *copy* the stack to grow — is
unavailable, permanently: the control stack is out-of-band precisely so the guest can
never name it (Invariant 2's escape posture; §3d two-stack split), and conservative
roots forbid relocation anyway. So goroutines on SVM are **fixed-stack**, sized at
spawn, overflow = guard-page fault = domain trap (Invariant 6). TinyGo semantics,
not gc semantics. Two practical consequences:

1. **Pick sizes honestly.** Data stack: guest-chosen per fiber (16 KiB default is the
   existing demo norm). Control stack: today a host constant (`FIBER_STACK = 256 KiB`,
   `svm-jit/src/fiber_rt.rs`) — fine for thousands of goroutines, hostile to hundreds
   of thousands (§4.2 proposes the knob).
2. **Deep recursion is the app's problem.** Same posture as every SVM guest: the
   guard page catches it; a runtime that wants more spawns bigger-stack fibers.

## 4. svm-side gaps this surfaces (each small, each its own slice)

1. **`fence` in svm-llvm.** Parses but `Unsupported`; the IR has thread fences and
   the atomics model is C11 wholesale, so this is a translator arm, not new substrate.
   Go/TinyGo runtimes emit fences. *(Small; do first — it also unblocks other
   threaded-Rust/C++ corpus.)*
2. **Per-`cont.new` control-stack size.** Make the fiber control-stack size a
   creation parameter (host-clamped) instead of a constant, so a goroutine-scale
   runtime can trade depth for count. Mechanism-not-policy compatible: the guest
   states a size, the host clamps and allocates. *(Small-medium; touches
   `svm-fiber`/`fiber_rt`; needed for B2 at scale, not for B1.)*
3. **Per-goroutine "TLS" needs no new feature.** The LLVM `thread_local` gap
   (LLVM.md's known follow-up) is avoided by not using `__thread` in the retargeted
   runtime: current-goroutine is derivable from any local's address masked to the
   (aligned, guest-allocated) data-stack base, or threaded explicitly; per-vCPU state
   uses `vcpu.tls.get`. Record the pattern, don't build lowering. *(Doc-only.)*
4. **setjmp × fibers JIT decline.** Modules mixing `SetJmp` with fibers/threads
   decline the Cranelift JIT (fail-closed, LLVM.md). TinyGo's panic path must lower
   via its own scheme (it does) rather than setjmp, or B1 runs interp-only. *(Watch
   item for B0's gap list, likely free.)*
5. **`gc.roots` × `thread.spawn` on the bytecode tier** falls back to the reference
   interp today (`GC.md` §3.2); wasm-JIT declines `gc.roots` entirely. Fine for B1/B2
   on native (interp + Cranelift JIT are the targets); a browser-parallel Go guest is
   B3-and-measured. *(No action now.)*

## 5. Invariants respected

- **Untrusted frontends, zero escape-TCB** (§2a / Invariant 9's frontier): TinyGo,
  the retargeted runtime, and the Go-emitted wasm are all frontend artifacts; the
  verifier re-checks every module; the masking lowering confines every access. A
  compiler or runtime bug corrupts the guest's own world, never the host.
- **Host = mechanism, guest = policy** (Invariant 4): the goroutine scheduler, STW,
  channels, and preemption polls are all guest code over existing primitives — no new
  host scheduling surface, no vCPU-count intrinsic, no priorities.
- **Small core** (Invariant 1): proposed svm changes are two translator/parameter
  slices (§4.1, §4.2), each demanded by a named consumer, nothing speculative.
- **Interpreter is the oracle** (Invariant 9): every phase gates on interp==JIT
  differentials, plus native-toolchain output oracles (Route A as the semantics
  oracle for Route B's overlap).
- **One world per domain / errors are values** (Invariants 5, 6): goroutine overflow
  and runtime bugs trap the domain; blocking host calls take async-form capabilities
  so STW never stalls on a parked syscall (`GC.md` §5).

## 6. Route D — multi-threaded self-hosting: a direct Go→svm-ir backend

Added 2026-08-04, answering the follow-up: *what would it take to run Go
multi-threaded and self-host it — a TinyGo-style backend that targets svm-ir, with
the compiler itself compiled to svm-ir?* "Self-hosting" here follows SELFHOST_C.md
§1: **the toolchain runs on the platform it targets**, with the true
compiler-compiles-itself fixpoint kept as the conformance differential — the
chibicc pattern, which reached per-TU byte-identity fixpoint 2026-07-30.

### 6.0 The crux: LLVM is the self-host blocker, so the backend must replace it

TinyGo as-is cannot self-host on SVM: it links **libLLVM** (a multi-million-line
C++ dependency with pervasive TLS, threads, and EH) for its mid-end and codegen.
On-ramping libLLVM as a guest is a mega-project an order beyond Postgres, for no
architectural payoff — and this tree has already walked away from libLLVM once
(`svm-llvm` §8 Q1b dropped the link in favor of an in-house `.ll` reader).

The move that dissolves the blocker: a backend that consumes **Go SSA**
(`go/parser` + `go/types` + `golang.org/x/tools/go/ssa` — all pure Go, no cgo) and
emits **svm-ir directly**, plus the Route B runtime as ordinary Go+intrinsics
code. Call it `svmgo`. Whether it lives as a TinyGo fork with the LLVM emission
layer swapped out (inheriting TinyGo's map/interface/runtime lowering designs) or
as a fresh compiler reusing those designs is an implementation choice, not a
strategic one — either way the resulting compiler is **pure Go**, and pure Go is
exactly what the backend compiles. Self-hosting stops being a separate project and
becomes a corollary: *the compiler is a member of its own input language subset.*

The IR fit is the favorable kind (`DESIGN.md` §20a's table applies verbatim):
go/ssa is typed SSA with φ-nodes → block params, svm-ir takes irreducible CFGs,
multi-value returns (Go's bread and butter), and true tail calls natively. This is
the svm-llvm situation — "we already have SSA" — not the svm-wasm reconstruction
one. Like every frontend, `svmgo` sits outside the escape-TCB (§2a): the verifier
re-checks its output; a compiler bug is a clean error, never an escape.

### 6.1 What the backend must lower (the real size of the work)

| Go construct | Lowering | Calibration |
|---|---|---|
| Scalars, structs, arrays, pointers | Window offsets + masking, §3d SysV-pinned layout | The chibicc/svm-leng model, proven twice |
| Slices, strings, maps | Fat pointers / runtime hash table in guest Go (TinyGo's designs) | Runtime code, not backend code |
| Interfaces, type switches, type assertions | Interned type-descriptor + itable; **structural interning per Invariant 10** | Same shape as TYPESCRIPT.md §4's descriptors |
| Closures, `defer`, `panic`/`recover` | Env structs; per-frame defer chains; unwind via the error-flag pattern (nimony's proven shape) or `SetJmp`/`LongJmp` — **flag preferred** (avoids the setjmp×fibers JIT decline, §4.4) | nimony exceptions landed as one slice |
| `go` / channels / `select` / `sync` | Fibers + guest scheduler + futex — the Route B2 runtime | `steal_fibers` is the template |
| Generics | go/ssa instantiation (monomorphize) | Handled upstream of the backend |
| Reflection | Partial, TinyGo-style: type metadata emitted per program, `reflect` over it | The long tail; `fmt` needs a working core |
| GC | Conservative non-moving mark-sweep over `gc.roots` + window heap | GC.md's contract, JACL-proven shape |
| Goroutine-local (`g`) | Data-stack-base derivation + `vcpu.tls` for per-P state (§4.3) | No new svm feature |

The backend proper is bounded and slice-able exactly like `svm-leng` was
(fail-closed `unsup(...)` from day one, grow arm by arm against real go/ssa
output). The honest long pole is not the backend — it is **compiling
`go/types` + `go/ssa` themselves** (order 150k+ lines of interface-heavy,
map-heavy, closure-heavy Go) well enough that the compiler runs as a guest. That
is a breadth/bug-tail grind, not a design risk: every construct it exercises is
core semantics the backend must support anyway, so "the compiler compiles" is the
natural capstone test rather than extra scope, and it doubles as the GC/allocator
stress test (go/types is allocation-heavy — the window heap and STW get exercised
for real).

### 6.2 Multi-threaded, in both senses

- **Compiled programs** are multi-threaded via the Route B2 runtime: N vCPUs
  (`thread.spawn`), work-stealing of suspended fibers, channels/`sync` over
  futex+atomics, cross-vCPU STW via the §2.1 quiesce barrier. Nothing here is
  Route-D-specific — B2 is a prerequisite and its own deliverable.
- **The compiler itself** parallelizes the way gc does (concurrent per-function
  compilation over a work queue) — on SVM that is goroutines over vCPUs, i.e. the
  runtime eating its own dog food. One discipline to adopt from gc on day one:
  **deterministic output under parallelism** (sorted emission, no map-iteration
  order leaks) — the fixpoint differential (§6.4) is only meaningful if stage2
  bytes are schedule-independent, and Invariant 9's oracle culture demands it.

### 6.3 Avoiding nimony's W4: one binary, in-process

NIM.md flags multi-binary architecture (`nifler` → `nimony` → `hexer` → `lengc`
subprocesses) as its "biggest unknown" for self-hosting. `svmgo` sidesteps this by
construction — unlike nimony, we control the architecture: **one binary,
parse→types→ssa→emit→link in-process**, importing the linker as a library (the
`.svmo` narrow waist and `svm_ir::link` already exist for exactly this). File I/O
bottoms out on the POSIX personality + `svm-fs` memfs; no `exec`, no process tree.
(`go build`'s real process model, if ever wanted, is EXEC.md/§14-children
territory — deliberately out of scope for the loop.)

### 6.4 The bootstrap loop, in the tree's own convention

1. **Stage 0:** host Go toolchain builds `svmgo` native. This is the build/dev
   oracle binary, exactly like chibicc's native form (SELFHOST_C.md §2) — it
   never ships, it gates.
2. **Stage 1:** native `svmgo` compiles `svmgo` + runtime + its stdlib closure →
   `svmgo.svmb`. Runs as a guest on interp + Cranelift JIT.
3. **Stage 2:** `svmgo.svmb` (on SVM, multi-vCPU) compiles the same sources.
   **Fixpoint gate: stage1 == stage2 byte-identical** — the chibicc criterion.
4. **Differentials throughout:** every corpus program byte-matches (a) native
   TinyGo/gc execution where semantics overlap, and (b) Route A's
   stock-Go-via-wasm output — the frontend-level oracle Route A exists to provide.

**The SELFHOST_C.md §3 trade, updated for Go.** chibicc ships the on-ramp-built
artifact because LLVM's `-O2` beats its own codegen. The analog here: the shipping
`svmgo.svmb` could be built by **stock gc → `GOOS=wasip1` wasm → svm-wasm**
(inheriting gc's optimizer for the compiler binary), with the self-compiled form
kept as the conformance differential. But the trade is genuinely different this
time: the gc-wasm form is **single-threaded** (gc's wasm runtime multiplexes
cooperatively), while the self-compiled form runs the B2 runtime with real vCPU
parallelism. gc's better per-function code vs. N-way parallel compilation —
measure, don't assume; the benchmark harness is the arbiter (AGENTS.md).

### 6.5 The near-free lane: the gc toolchain *on* SVM before any backend exists

Worth naming because it closes a toolchain-on-svm loop for ~zero compiler work,
as soon as Route A lands: the gc compiler (`cmd/compile`, `cmd/link`) is pure Go,
so it builds as a `wasip1` binary and runs on SVM through svm-wasm + the widened
WASI shim + memfs. Output targets `wasip1` too → feed the result back through
svm-wasm. That is *Go compiling Go on SVM, running the result on SVM* — no new
backend, no runtime port. Constraints inherited from Route A: single-threaded,
interpreter-class speed, and the emitted modules carry gc's own in-module runtime
rather than fibers/`gc.roots`. Not the destination — but a working oracle for
stage-1 bring-up, a stress corpus for the WASI/fs surface, and an honest demo
months before Route D's fixpoint.

### 6.6 Work breakdown & calibration

Prerequisites: Route A (oracle + §6.5 lane) and Route B2's runtime (shared
verbatim). Then, in dependency order:

1. **Backend skeleton** — go/ssa → svm-text for scalars/control-flow/calls,
   fail-closed elsewhere; differential from the first commit. *(The svm-leng
   walking-skeleton move; that took days of slices.)*
2. **Core semantics** — slices/strings/maps/interfaces/closures/defer/panic,
   runtime in guest Go. *(The bulk of backend work; svm-leng's W1 analog, which
   closed in about a week of slices — Go's surface is larger; budget several
   times that.)*
3. **Concurrency + GC wiring** — `go`/channels/`select` onto the B2 runtime;
   type-metadata + `gc.roots` collector integration.
4. **Stdlib closure for self-compile** — os/io/fmt/strings/sort/strconv +
   go/token/parser/types + x/tools/go/ssa. *(The long pole — a breadth grind
   measured by "how much of go/types compiles today".)*
5. **Fixpoint** — stage1==stage2, then multi-vCPU stage2 and the determinism
   gate.

By conventional accounting this is person-quarters, not person-weeks — the
largest language project in the tree so far, sized roughly **3–5× svm-leng**
(which went from empty crate to linking real cross-module nimony programs in
about a week of slices, per NIM.md §3). It decomposes into the same
fail-closed, differential-gated slices as every frontend before it, with no
single high-risk unknown: the runtime substrate is proven (B2/`steal_fibers`),
the GC contract is proven (JACL), the frontend pattern is proven three times
over (chibicc, svm-wasm/svm-llvm, svm-leng), and the self-host convention is
proven (chibicc fixpoint). The one genuinely new bet is compiler-scale Go
(go/types) running over the conservative-GC runtime — which is why it's staged
as the capstone with oracles on both sides of it.

### 6.7 Risks, stated

- **go/types+go/ssa breadth.** Interface-heavy stdlib code will find every
  backend gap; mitigated by fail-closed discipline + the §6.5 oracle, but it is
  the schedule risk.
- **GC pressure at compiler scale.** A conservative collector retaining
  compiler-sized heaps (false roots pinning large subgraphs) is a perf risk, not
  a correctness one (GC.md §3.2's over-approximation argument); the top-byte
  payload mask exists if tagging becomes worthwhile.
- **x/tools/go/ssa version coupling.** The backend couples to go/ssa's API — a
  vendored pin, refreshed deliberately (the LLVM-18→21 lesson from svm-llvm
  applies: tolerate versions, don't chase them).
- **Fiber-count economics** — §4.2's control-stack knob stops being optional at
  compiler-workload goroutine counts.

## 7. Non-goals

- Bug-for-bug `gc` runtime semantics (contiguous stack growth, precise GC pause
  targets, `runtime` package internals). TinyGo-level semantics are the bar.
- cgo. Foreign C rides the same on-ramps as everything else if ever needed.
- Beating native Go on throughput. The bar is the usual one: within the
  wasm/Wasmtime-relative envelope (§1a), with confinement.
- wasm-GC. Irrelevant here (Go doesn't emit it) and a standing svm non-goal.
