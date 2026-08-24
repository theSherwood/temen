# Architecture — crate map

Where everything lives, what each crate does, and — the column that matters most —
whether it is part of the **escape-TCB**: the code that must be correct for the
"verified ⇒ cannot escape" claim to hold (`DESIGN.md` §2a). Everything not marked
escape-TCB is either untrusted input that gets re-verified (the frontends), held to
the parity invariant by differential testing (the backends), or ordinary host
tooling.

Section references (§) point into [`DESIGN.md`](DESIGN.md); D-numbers are its
Decision Log. Per-subsystem docs (`WASM.md`, `LLVM.md`, `BROWSER.md`, …) are linked
from the rows that have them.

| Crate | Role | TCB? |
|---|---|---|
| `temen-ir` | Core IR: block-local typed SSA over a CFG (§3a/§3b) | escape-TCB |
| `temen-mask` | Confinement masking — the isolated, separately-fuzzed unit (§4, I1) | escape-TCB |
| `temen-mem` | Shared guest-memory substrate (§12/§13) — owns the memory `unsafe` behind a safe API (audited in isolation, like `temen-mask`), so the interpreter stays `forbid(unsafe_code)` | escape-TCB |
| `temen-encode` | Binary encode + **decode** (untrusted-input-facing) (§3a) | escape-TCB |
| `temen-verify` | The verifier — single linear pass, fail-closed (§2a I2/I3/I4; §3b) | escape-TCB |
| `temen-interp` | Two of the four IR backends: the **tree-walk interpreter** (the differential oracle, §18) and the **bytecode interpreter** (`bytecode.rs` — the portable / JIT-not-viable path, incl. the wasm64 browser platform). All four backends must agree (§3 parity invariant) | — |
| `temen-fiber` | Native stack-switch primitive for fibers / green threads (§3d/§6/§12); the lone home for that `unsafe`, tiny and auditable (x86-64 + aarch64 unix, x86-64 Windows) | escape-TCB |
| `temen-jit` | **Cranelift JIT** (backend 3, the native speed path) — CLIF lowering + the §4 masking lowering + guard page/signal (§9). By convention bare "JIT" means *this* one | escape-TCB† |
| `temen-wasm-jit` | **wasm-JIT** (backend 4) — emits WebAssembly from the IR so hot compute runs on a wasm engine (§21, `BROWSER.md`); a leaf accelerator under the bytecode interpreter, fail-closed to it. Held against the tree-walk oracle (`tests/differential.rs`) | escape-TCB† |
| `temen-text` | Text format ⇄ IR (dev/debug; 1:1 with binary) (§3a) | — |
| `temen-wasm` | **Core-wasm → IR transpiler** — a second frontend (untrusted, re-verified); stack→SSA reconstruction (`WASM.md`) | — |
| `temen-llvm` | **LLVM-bitcode → IR translator** — the AOT LLVM on-ramp (untrusted, re-verified); dominance-SSA → block-args (§20a, D54; `LLVM.md`) | — |
| `temen-peval` | **Partial evaluator** — semantics-preserving IR→IR optimizer + the first Futamura projection (§20c) | — |
| `temen-durable` | IR→IR **freeze/thaw** transform for durable domains (tooling-tier, +0 TCB; §21, D60; `DURABILITY.md`) | — |
| `temen-snapshot` | Durable-domain **snapshot artifact codec** (window image + handle table + identity gate; §21; `DURABILITY.md`) | — |
| `temen-dap` | Interpreter-backed **Debug Adapter Protocol** server (breakpoints/stepping/locals; §19; `DEBUGGING.md`) | — |
| `temen-wasmjit` | **Temen IR → WebAssembly emitter** — the browser wasm-JIT tier; carries the §4 masking lowering *in emitted wasm* + cap-call outlining (`BROWSER.md`) | escape-TCB |
| `temen-spec` | The **executable ISA spec** (`SPEC.md`): one machine-readable op table (typing + reference semantics) — the `spec_*` conformance/fuzz oracle the backends and `temen-verify` are checked against (§18) | — (spec oracle) |
| `temen-opt` | Generic closed-module **IR→IR optimizer** (SSA construct/destruct, SCCP; `OPT.md`, §20a) — the pass library `temen-peval` builds on | — |
| `temen-exec` | Deterministic **`exec` capability** backend + the wasm-safe exec-cap wire protocol (`EXEC.md`) | — (host cap) |
| `temen-fs` | In-memory **`fs` capability** backend + the fs-cap wire protocol | — (host cap) |
| `temen-posix` | A **POSIX personality** delivered as a §7 host capability (libc-as-a-capability; `POSIX.md`) | — (host shim) |
| `temen-capi` | The **C ABI** over the `temen-run` embedding surface (`temen.h`; `POWERBOX.md` Phase 5) | — |
| `temen-webgpu` | Headless **WebGPU compute** capability — host holds a real GPU via `wgpu` (`LLVM.md`; workspace-excluded) | — |
| `temen` | Umbrella: pipeline (`assemble`/`load`/`run`) + tests + bench | — |
| `temen-run` | Embedding runtime + **`temen-run` CLI**: instantiate with the powerbox, run on the JIT | — |
| `browser/` | The **browser platform**: the bytecode interpreter (backend 2) compiled to **wasm64** to run Temen guests client-side, hosting the **wasm-JIT** (backend 4) for hot compute (`BROWSER.md`). Not a separate backend — backend 2 on a wasm host | — |
| `fuzz/` | cargo-fuzz targets (nightly); mirror the stable smoke fuzz | — |

†`temen-jit` is escape-TCB but, by design (§1), shares Wasmtime's codegen — so unlike
the other TCB crates it *does* take a dependency (Cranelift). The dependency-free rule
covers only the small audit-critical crates (`temen-ir`/`temen-mask`/`temen-encode`/`temen-verify`).

The escape-TCB crates are deliberately **dependency-free** (small, fast to compile,
auditable). The host is Rust; the frontend (`frontend/chibicc`) is C; codegen lowers to
Cranelift (`DESIGN.md` D49 / D36).
