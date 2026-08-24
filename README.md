# Sandbox VM

[![CI](https://github.com/thesherwood/vm/actions/workflows/ci.yml/badge.svg)](https://github.com/thesherwood/vm/actions/workflows/ci.yml)

A compilation target and sandbox VM — a WebAssembly alternative for running
untrusted native code, aimed at being **as secure for the host as wasm, faster on
the axes that matter, with a simpler interface and real virtual memory.**

> ⚠️ **Status: early research build, heavy WIP.** A lot works end-to-end today, but
> nothing here is stable, certified, or ready to depend on. "Appears to work" is
> reachable; "is certified secure" is explicitly future work. See
> [Status](#status) below.

The full design lives in [`DESIGN.md`](DESIGN.md); the working agreement (keep it
simple; test/fuzz/bench early; data-oriented design) is in [`AGENTS.md`](AGENTS.md),
and the security rules a change must not break are in [`INVARIANTS.md`](INVARIANTS.md).

## Why this exists

WebAssembly proved you can run untrusted code safely at near-native speed, and it's
the bar we measure against. But wasm carries baggage: a flat 32-bit linear memory, a
heavy interface (WASI + the component model + WIT + lift/lower marshalling), no
runtime nesting, awkward concurrency, and design choices — a built-in GC, `externref`,
UTF-16, JS interop — shaped by the browser and a broad managed-language market rather
than by the needs of code running natively on a host.

This project asks the opposite question: build the sandbox for **running untrusted
native (and native-ish) code on a host** — aimed first at systems languages (C, C++,
Rust, Zig, Swift), though not limited to them — and see how much simpler, faster, and
more capable it can be. It is **not tied to JavaScript's needs**, and while it stays
committed to *not* absorbing wasm's GC complexity, it isn't hostile to managed
languages either: a guest can bring its own collector and the VM helps it (conservative
root enumeration), instead of baking a garbage collector into the platform.

What we're chasing:

- **Security parity with wasm, honestly scoped.** The bar is "as secure for the host
  as [Wasmtime](https://wasmtime.dev)," not a proof of escape-impossibility. We share
  Wasmtime's most security-critical component — the [Cranelift](https://cranelift.dev)
  code generator — so the trust boundary we actually own is small: a tiny,
  single-pass **verifier** plus one isolated **memory-confinement** lowering pass.
  Both are kept dependency-free and fuzzed continuously.
- **A radically simpler interface.** Scalars, `(ptr, len)` own/borrow buffers, and
  capability handles — no IDL, no lift/lower, structured data is just bytes. The host
  exposes an **open capability surface** (the "powerbox") instead of a fixed WASI menu,
  and it's the *only* channel out of the sandbox.
- **Faster where wasm is weak.** Sharing Cranelift means tight scalar compute runs at
  *parity* with wasm by construction — we don't pretend to beat the same backend. The
  speed budget is spent *around* compute, and this is where a lot of it lives:
    - **Zero-copy host calls / I/O** — borrow buffers are read in place through the
      page table, so a guest region can go straight to a device or GPU with no
      copy-out (vs wasm's mandatory linear-memory→host hop), and `cap.call` is a
      devirtualized register-to-register call, not component-model marshalling.
    - **A clean 64-bit address space** — a real 64-bit window, no 32-bit index type,
      so large or sparse programs are a first-class target instead of a fight.
    - **Faster startup** — SSA is already on the wire (no reconstruction from a stack
      machine), and decls-before-bodies let per-function verify+JIT run in parallel.
    - **Native irregular control flow** — irreducible CFGs (which wasm can't express
      without a relooper), plus tail calls, multi-return, and first-class stack
      switching as a single primitive.
    - **Host-native SIMD** — the LLVM on-ramp targets the actual CPU's 128-bit SIMD,
      richer than wasm's portable `simd128`.
- **Real, guest-visible virtual memory.** The guest holds an attenuable address-space
  capability (`map`/`unmap`/`protect` within its window) — sparse address spaces,
  demand paging, and lending sub-ranges out — not just `memory.grow` on one blob.
- **Simpler concurrency, all the way up.** Fibers, 1:1 threads, and cross-domain
  processes are exposed as clean primitives — plus atomics and a futex — so building
  M:N schedulers, async runtimes, and multithreaded code is far less awkward than on
  wasm. The VM ships the primitives, not a scheduler.
- **Nested sandboxes with no extra virtualization cost.** A guest can spawn a child
  domain in a sub-window with an attenuated subset of its own capabilities;
  confinement composes to any depth at depth-independent per-access cost. Multi-tenant
  hosts and plugin-in-plugin fall out for free.
- **A JIT *inside* the sandbox.** A guest (say, a language runtime) can build IR at
  run time, hand it across a capability, and have the host verify and Cranelift-compile
  it into the guest's *own* domain — verification, not isolation, is the trust
  boundary. wasm handles this poorly.
- **Durable domains.** A running domain can be quiesced, serialized, and restored
  bytewise — recompile-survivable and backend-independent — so guests can be
  snapshotted, migrated, or persisted.
- **Time-travel debugging.** An interpreter-backed debugger (Debug Adapter Protocol)
  gives source-level breakpoints, stepping, and backtraces over the IR's debug info —
  no DWARF or JIT needed — with time-travel (step backward) as the WIP headline.

The through-line: leaving out the browser-shaped surface is what keeps the verifier
and ABI small enough to trust — and that lean core is what makes the memory,
concurrency, nesting, and tooling wins above affordable.

## What it aims to be

The end state is a small, trustworthy core with several interchangeable pieces:

- **One IR** — block-local typed SSA over a CFG — that source frontends target and
  every backend consumes.
- **Frontends** that lower real code to the IR: a C compiler, a core-wasm
  transpiler, and an LLVM-bitcode translator (so anything the LLVM toolchain emits
  can be sandboxed).
- **Backends** that all must agree, byte-for-byte, on every program (the parity
  invariant): a tree-walk interpreter (the differential oracle), a portable bytecode
  interpreter, a Cranelift JIT (the native-speed path), and a wasm-JIT (for the
  browser).
- **A capability-based host interface** (the "powerbox") as the *only* channel out of
  the sandbox, plus host-provided capabilities for memory, I/O, concurrency, nesting,
  durability, and more.
- **Tooling** built on the same core: a debugger, an optimizer/partial evaluator,
  snapshot/restore for durable domains, and instrumentation hooks.

## Status

**This is a research build under active, rapid development.** Much of the above
already runs end-to-end and is differentially tested backend-against-backend, but
APIs, formats, and internals churn constantly, and the security claims are a
*target*, not a finished guarantee.

Roughly where things stand:

**Working today**
- The full scalar IR (integer/float ops, linear memory with confinement masking,
  direct/indirect/tail calls, function table, `select`, `br_table`) flows through
  text ⇄ binary ⇄ verifier ⇄ interpreter ⇄ JIT.
- The **Cranelift JIT** lowers the entire IR, differentially tested against the
  interpreter oracle (results, trap kinds, and host side effects).
- A **C frontend** (a vendored [chibicc](https://github.com/rui314/chibicc) fork)
  compiles a broad C subset — structs/unions by value, function pointers, varargs +
  `printf`, `goto`, recursion, `malloc`/`free` over the Memory capability — and real
  third-party C libraries run sandboxed byte-identically to a native build (Clay,
  jsmn, SHA-256, xxHash, miniz/tinfl, stb_perlin, tiny-regex-c, and more; see
  [`demos/`](crates/svm-run/demos)).
- **Two more frontends**: `svm-wasm` (core-wasm → IR, incl. v128 SIMD and
  wasi-threads) and `svm-llvm` (LLVM-bitcode → IR) — the LLVM on-ramp runs the
  **unmodified SQLite amalgamation** (in-memory and disk-backed via the Fs
  capability) and a **QuickJS** embedding byte-identically to native builds.
- **Real virtual memory**: a reserved window with guard page + fault handler turns an
  out-of-window access into a clean trap on Linux / macOS / Windows; guest-controlled
  demand-paged growth via the Memory capability.
- **Concurrency primitives**: stackful fibers, 1:1 threads, C11 atomics, a
  `wait`/`notify` futex, and a C `<pthread.h>` built over them — no built-in
  scheduler (guests build their own M:N runtimes).
- **Nesting (VM-in-VM)** on both backends, cross-domain shared regions, a
  host-enforced **fuel/epoch kill-path** for runaway guests, and spawn quotas for
  DoS containment.
- A **guest-driven JIT** capability (a guest builds IR at runtime, the host verifies
  and Cranelift-compiles it into the guest's own domain).
- Tooling: durable domains (freeze/thaw + snapshot codec), a DAP debug server,
  memory-access hooks, a partial evaluator, and a wasm64 browser build of the
  interpreter hosting a live playground (Doom, Lua, Postgres, SQLite, QuickJS,
  Tcl, and more, in the browser).
- **Continuous fuzzing** of the security-critical invariants (see [Fuzzing](#fuzzing)).

**Still ahead**
- Narrow-scalar promotion, honoring weak memory orderings (both backends are seq-cst
  today), wider SIMD (`v256`/`v512`), isolation tiers, Spectre hardening, source-level
  DWARF for JIT code, and broader LLVM/wasm frontend coverage.
- The bring-ups in flight: **GNU bash** on the POSIX personality, and QuickJS
  through full test262 (see the READMEs under [`demos/`](crates/svm-run/demos)).
- The security-certification workstream: today's bar is "appears to work," not
  "certified secure" (see `DESIGN.md` §2a/§18).

For a blow-by-blow of what landed and why each demo mattered, see the git history and
the per-subsystem design docs referenced from [`DESIGN.md`](DESIGN.md).

## Layout & trust boundary

The workspace is ~27 crates; the full crate map lives in
[`ARCHITECTURE.md`](ARCHITECTURE.md). What matters most is how few of them you have
to trust. The **escape-TCB** — the code that must be correct for "verified ⇒ cannot
escape" to hold — is a short, closed list: `svm-ir` (the IR), `svm-encode` (decode,
the untrusted-input face), `svm-verify` (the verifier), `svm-mask` (the confinement
masking unit), `svm-mem` (the guest-memory substrate), `svm-fiber` (the stack-switch
primitive), and the JIT tiers (`svm-jit`, plus the browser pair
`svm-wasm-jit`/`svm-wasmjit`).

The audit-critical core (`svm-ir`/`svm-mask`/`svm-encode`/`svm-verify`) is
deliberately **dependency-free** — small, fast to compile, auditable. The JITs are
the designed exception: `svm-jit` shares Wasmtime's codegen (Cranelift) so that a
codegen escape bug is *their* bug class too, not a new one we invented.

Everything else is outside the trust boundary: frontends are untrusted and their
output re-verified; interpreters and JITs are held to the "all backends agree"
parity invariant by differential testing; capability backends and tooling are
ordinary host code. The host is Rust; the C frontend (`frontend/chibicc`) is C.

## Build & test

```sh
cargo build --workspace
cargo test  --workspace          # pipeline + differential + 250k-iter smoke fuzz
cargo fmt   --all --check
cargo clippy --workspace --all-targets
cargo run --release --bin svm-bench   # decode / verify / interp throughput
```

## Run a program in the sandbox

The `svm-run` CLI compiles (if needed), verifies, and runs a guest program on the JIT under
the MVP powerbox (§3e) — `stdout`/`stderr` go to the real streams and it exits with the
guest's code:

```sh
cargo run -p svm-run -- crates/svm-run/demos/hello.svmt   # text IR → "hello, sandbox!"
cargo run -p svm-run -- crates/svm-run/demos/hello.c      # C source (via the chibicc frontend)
cargo run -p svm-run -- crates/svm-run/demos/clay/clay_demo.c        # the Clay UI layout library
cargo run -p svm-run -- crates/svm-run/demos/raytrace/raytrace.c     # ASCII raytracer (guest-side libm)
cargo run -p svm-run -- crates/svm-run/demos/mat4/mat4.c             # 128-bit SIMD matrix math
cargo run -p svm-run -- crates/svm-run/demos/heapgrow/heapgrow.c     # malloc heap growth via the Memory cap
cargo run -p svm-run -- crates/svm-run/demos/jit/jit_demo.c          # a guest interpreter that JITs itself
cargo run -p svm-run -- crates/svm-run/demos/steal_fibers/steal_fibers.c  # work-stealing over migratable fibers
echo 'int main(){ return 42; }' > /tmp/r.c
cargo run -p svm-run -- /tmp/r.c ; echo "exit $?"         # → exit 42
```

The CLI accepts `.svm` (text IR), `.svmb` (binary), or `.c` (compiled through
`frontend/chibicc`, located via `$SVM_CHIBICC` or the in-repo build).

That's a sample — [`demos/`](crates/svm-run/demos) holds dozens more, each picked to
stress a different shape and checked byte-for-byte against a native `cc` build:
real third-party C libraries (jsmn, SHA-256, xxHash, miniz/tinfl, stb_perlin,
tiny-regex-c, monocypher…), guest M:N schedulers and async runtimes over the
concurrency primitives, and more. See [`FRONTEND.md`](FRONTEND.md) for the story
behind getting Clay, jsmn, and friends to run.

The heavyweights run through the **LLVM on-ramp** (`svm-llvm`): the **unmodified
SQLite amalgamation** — in-memory and disk-backed via the Fs capability — plus
**LMDB** and a **QuickJS** embedding all run sandboxed, byte-identical to the same
sources built natively (full test262 for QuickJS is still in progress). And the
**browser playground** (`browser/`) runs real programs live, client-side:
**shareware Doom** (playable — arrow keys, Ctrl fires), **Lua 5.4.7**, **single-user
PostgreSQL 17.5**, SQLite, QuickJS, **Tcl 8.6**, chibicc compiling its own source,
a Nim toolchain, and a shell over the POSIX personality. The big bring-up still in
flight is GNU **bash**; each demo directory's README states honestly where it
stands.

Embedders can call the same path directly — `svm_run::run_powerbox(&module, stdin)`
returns the outcome plus captured output. It's the one reusable piece of host glue
(the `cap.call` trampoline + powerbox grant), and it is *not* escape-TCB: the
verifier, run first, is what makes a module safe.

## Fuzzing

Stable CI runs the smoke fuzz as ordinary tests (`crates/svm/tests/fuzz_smoke.rs`,
`spec_fuzz_smoke.rs`). The coverage-guided targets all gate nightly (the `cargo-fuzz` CI
matrix runs every target in `fuzz/fuzz_targets/` — no built-but-unwired fuzzer):

```sh
cargo install cargo-fuzz
cargo +nightly fuzz run decode_verify   # decode/verify/interp never crash
cargo +nightly fuzz run mask            # the confinement-masking invariant (I1)
cargo +nightly fuzz run roundtrip       # binary + text round-trip identity
cargo +nightly fuzz run diff            # interp-vs-JIT differential (§18)
cargo +nightly fuzz run onramp_diff     # LLVM on-ramp vs its source-semantics oracle (§18)
cargo +nightly fuzz run wasm_transpile  # core-wasm → IR transpile (re-verified)
cargo +nightly fuzz run spec_ops        # every backend matches the executable spec's eval (SPEC.md)
cargo +nightly fuzz run spec_verify     # svm-verify vs the reference verifier agree (SPEC.md)
cargo +nightly fuzz run opt_sccp        # optimizer preserves semantics (SCCP)
cargo +nightly fuzz run opt_ssa_roundtrip   # SSA construct/destruct identity
cargo +nightly fuzz run durable         # freeze → serialize → restore → thaw equivalence (+ _jit / _fiber / _loop / _recycle variants)
cargo +nightly fuzz run coverage_walk   # verifier coverage walker
```

The invariants under test are the security hinge (§2a/§4): on arbitrary bytes,
`decode` fails closed (never panics/OOMs/hangs), `verify` never panics, any *verified*
module is safe to interpret, the masking unit confines every access into its window,
and the formats round-trip without changing the IR. The two `spec_*` targets extend
the executable ISA spec (`SPEC.md`) into unbounded exploration: `spec_ops` drives
random operands through each op and checks all backends against the reference
semantics, and `spec_verify` holds the production verifier and an independent
reference verifier in accept/reject agreement over generated + mutated modules.

## Example IR (text form)

```text
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 0
  br block1(v0, v1)
block1(v2: i32, v3: i32):     ; v2 = i, v3 = sum
  v4 = i32.add v3 v2
  v5 = i32.const -1
  v6 = i32.add v2 v5
  br_if v6 block1(v6, v4) block2(v4)
block2(v7: i32):
  return v7                   ; sum of 1..=N
}
```

---

**A note on authorship.** This project is largely written by LLM agents — the code,
the tests, the design docs, and this README included — working under human direction
and review. That context matters when weighing the claims above: the confidence here
comes from the differential tests, fuzzers, and native-build byte-comparisons that
gate CI, not from expert eyeballs, and the security posture is a *target* until the
validation plan in `DESIGN.md` §18 is executed. Read accordingly.
