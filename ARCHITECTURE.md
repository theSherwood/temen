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
| `svm-ir` | Core IR: block-local typed SSA over a CFG (§3a/§3b) | escape-TCB |
| `svm-mask` | Confinement masking — the isolated, separately-fuzzed unit (§4, I1) | escape-TCB |
| `svm-mem` | Shared guest-memory substrate (§12/§13) — owns the memory `unsafe` behind a safe API (audited in isolation, like `svm-mask`), so the interpreter stays `forbid(unsafe_code)` | escape-TCB |
| `svm-encode` | Binary encode + **decode** (untrusted-input-facing) (§3a) | escape-TCB |
| `svm-verify` | The verifier — single linear pass, fail-closed (§2a I2/I3/I4; §3b) | escape-TCB |
| `svm-interp` | Two of the four IR backends: the **tree-walk interpreter** (the differential oracle, §18) and the **bytecode interpreter** (`bytecode.rs` — the portable / JIT-not-viable path, incl. the wasm64 browser platform). All four backends must agree (§3 parity invariant) | — |
| `svm-fiber` | Native stack-switch primitive for fibers / green threads (§3d/§6/§12); the lone home for that `unsafe`, tiny and auditable (x86-64 + aarch64 unix, x86-64 Windows) | escape-TCB |
| `svm-jit` | **Cranelift JIT** (backend 3, the native speed path) — CLIF lowering + the §4 masking lowering + guard page/signal (§9). By convention bare "JIT" means *this* one | escape-TCB† |
| `svm-wasm-jit` | **wasm-JIT** (backend 4) — emits WebAssembly from the IR so hot compute runs on a wasm engine (§21, `BROWSER.md`); a leaf accelerator under the bytecode interpreter, fail-closed to it. Held against the tree-walk oracle (`tests/differential.rs`) | escape-TCB† |
| `svm-text` | Text format ⇄ IR (dev/debug; 1:1 with binary) (§3a) | — |
| `svm-wasm` | **Core-wasm → IR transpiler** — a second frontend (untrusted, re-verified); stack→SSA reconstruction (`WASM.md`) | — |
| `svm-llvm` | **LLVM-bitcode → IR translator** — the AOT LLVM on-ramp (untrusted, re-verified); dominance-SSA → block-args (§20a, D54; `LLVM.md`) | — |
| `svm-peval` | **Partial evaluator** — semantics-preserving IR→IR optimizer + the first Futamura projection (§20c) | — |
| `svm-durable` | IR→IR **freeze/thaw** transform for durable domains (tooling-tier, +0 TCB; §21, D60; `DURABILITY.md`) | — |
| `svm-snapshot` | Durable-domain **snapshot artifact codec** (window image + handle table + identity gate; §21; `DURABILITY.md`) | — |
| `svm-dap` | Interpreter-backed **Debug Adapter Protocol** server (breakpoints/stepping/locals; §19; `DEBUGGING.md`) | — |
| `svm-wasmjit` | **SVM IR → WebAssembly emitter** — the browser wasm-JIT tier; carries the §4 masking lowering *in emitted wasm* + cap-call outlining (`BROWSER.md`) | escape-TCB |
| `svm-spec` | The **executable ISA spec** (`SPEC.md`): one machine-readable op table (typing + reference semantics) — the `spec_*` conformance/fuzz oracle the backends and `svm-verify` are checked against (§18) | — (spec oracle) |
| `svm-opt` | Generic closed-module **IR→IR optimizer** (SSA construct/destruct, SCCP; `OPT.md`, §20a) — the pass library `svm-peval` builds on | — |
| `svm-exec` | Deterministic **`exec` capability** backend + the wasm-safe exec-cap wire protocol (`EXEC.md`) | — (host cap) |
| `svm-fs` | In-memory **`fs` capability** backend + the fs-cap wire protocol | — (host cap) |
| `svm-posix` | A **POSIX personality** delivered as a §7 host capability (libc-as-a-capability; `POSIX.md`) | — (host shim) |
| `svm-capi` | The **C ABI** over the `svm-run` embedding surface (`svm.h`; `POWERBOX.md` Phase 5) | — |
| `svm-webgpu` | Headless **WebGPU compute** capability — host holds a real GPU via `wgpu` (`LLVM.md`; workspace-excluded) | — |
| `svm` | Umbrella: pipeline (`assemble`/`load`/`run`) + tests + bench | — |
| `svm-run` | Embedding runtime + **`svm-run` CLI**: instantiate with the powerbox, run on the JIT | — |
| `browser/` | The **browser platform**: the bytecode interpreter (backend 2) compiled to **wasm64** to run SVM guests client-side, hosting the **wasm-JIT** (backend 4) for hot compute (`BROWSER.md`). Not a separate backend — backend 2 on a wasm host | — |
| `fuzz/` | cargo-fuzz targets (nightly); mirror the stable smoke fuzz | — |

†`svm-jit` is escape-TCB but, by design (§1), shares Wasmtime's codegen — so unlike
the other TCB crates it *does* take a dependency (Cranelift). The dependency-free rule
covers only the small audit-critical crates (`svm-ir`/`svm-mask`/`svm-encode`/`svm-verify`).

The escape-TCB crates are deliberately **dependency-free** (small, fast to compile,
auditable). The host is Rust; the frontend (`frontend/chibicc`) is C; codegen lowers to
Cranelift (`DESIGN.md` D49 / D36).
