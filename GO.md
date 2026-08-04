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

## 6. Non-goals

- Bug-for-bug `gc` runtime semantics (contiguous stack growth, precise GC pause
  targets, `runtime` package internals). TinyGo-level semantics are the bar.
- cgo. Foreign C rides the same on-ramps as everything else if ever needed.
- Beating native Go on throughput. The bar is the usual one: within the
  wasm/Wasmtime-relative envelope (§1a), with confinement.
- wasm-GC. Irrelevant here (Go doesn't emit it) and a standing svm non-goal.
