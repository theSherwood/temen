# wasm-AOT viability & playground speed plan

**Status: assessment settled (no fifth backend); slices proposed, unstarted.** This answers "should
we build a wasm-AOT backend to make the playground fast?" with evidence from the tree, and lays out
the work that actually recovers the speed. Fold into `DESIGN.md`/`BROWSER.md` and drop this file
once the slices land (repo convention).

## 1. Verdict

**Do not build a fifth backend.** Both halves of "wasm-AOT" already exist:

- **"AOT" is the offline lane we already ship.** `svm-llvm`/`svm-opt`/`svm-peval` are ahead-of-time,
  host-side, and produce verified IR; every playground demo module (Lua, SQLite, DOOM, QuickJS,
  Postgres) is *already* AOT-compiled that way into `.svmb` (`LLVM.md`, `OPT.md`, DESIGN §20).
- **"Emit wasm" is backend 4.** `svm-wasm-jit` turns verified IR into a real `WebAssembly.Module`
  and, where it engages, already lands **at or below native Cranelift** on compute (`xorshift`
  2.0 ns emitted vs 1.9 ns native; 16–112× over the interpreter-in-wasm —
  `bench/cross-engine/README.md` § "SVM-in-wasm, the JIT tier"). Emitting *earlier* ("AOT-at-
  `svm_par_compile`", BROWSER.md) is a compile-point choice, not a new backend.

What a new backend *cannot* fix is the leaf-accelerator wall: a wasm frame can't unwind for a stack
switch, so fibers, `thread.spawn`/`join`, `memory.wait`, page ops, and every `cap.call` must bounce
to the bytecode interpreter (`env.call_interp`), with the interpreter owning the top frame
(DESIGN §3; NESTED_JIT.md Tracks 2–3; OPS_PARITY.md ⛔ column). A truly standalone "pure wasm"
artifact therefore exists only for **pure-compute guests** — no I/O, no caps, no threads — which is
not what the slow playground demos are.

## 2. Where playground time actually goes

Three separate causes, none of which a new backend addresses:

1. **Eligible code not on the JIT tier.** The interpreter-in-wasm is ~20–50× off the JIT
   (INTERP_PERF.md; BROWSER.md). The wasm-JIT is a per-demo opt-in checkbox, and demos with
   caps/concurrency in the reachable set are pinned to the interpreter entirely.
2. **No compiled-output reuse.** Every Run re-emits and re-instantiates; JIT break-even is
   ~10⁵–10⁶ iterations, so *light scripts run net slower under the JIT* (`play.js`; the
   cross-engine README calls a compiled-module cache "the dominant inefficiency the steady-state
   table omits"). Only `.svmb` *bytes* are cached today, not emitted wasm or instances.
3. **Per-access confinement in giant functions.** For Lua/SQLite the JIT engages but yields only
   ~3×/~1.3×: the cost is `emit_confine`'s ~7-op bounds-check-and-mask on **every** guest access,
   on top of wasm's own linear-memory bounds check (the double indirection the bench measures as
   `chase_rand` ~3.4×). Not V8 compile time (~6 ms for 3 MB), not the giant function itself
   (TurboFan compiles it; relooper A/B'd at zero gain and reverted). BROWSER.md § slice-8 already
   names the real levers: **redundant-check elimination with proof**, or function splitting.

## 3. What "pure wasm" would buy, and its sanctioned form

The one genuine win hiding in "output pure wasm" is cause 3: emit a module whose linear memory *is*
the guest window, and the engine's own bounds check subsumes ours — deleting the ~7-op inline
sequence per access. That is a real ~2× on memory-bound guests, and post-D63 the semantic gap is
small (both SVM and wasm now **trap** OOB; what differs is the boundary — SVM's power-of-two
`mapped` vs wasm's page-granular size — see the updated WASM.md divergence note).

But it moves confinement out of the one fuzzed masking pass and into "trust the engine," which is
exactly what INVARIANTS #2 exists to reject, and what the NESTED_JIT Track 3 decision declined for
the same reason ((c)+(a): scope out + fail closed, zero TCB growth). The sanctioned form of the same
win is the lever BROWSER.md already names: **elide checks we can prove redundant, inside the
existing masking regime** (slice 3 below). A standalone-memory pure-wasm *export* stays a
possibility behind INVARIANTS #1's bar — a named compute-only consumer — and is deliberately **not**
scheduled here.

## 4. The JACL consumer finding (adopted as slice 0)

`jacl_impl/docs/SVM_BROWSER_TIERUP_FINDINGS.md` is a named consumer hitting the exact path slice 2
depends on: per-function tier-up (`compile_module_tierup` + `VcpuEvent::TierUp` via `svm_par_run`)
of a **mainline** `InterpDriven` guest trapped `MemoryFault → unreachable` inside the emitted body,
before any `env.call_interp`. Their ranked hypotheses — all consistent with our code:

1. **Window/data-materialization mismatch.** The tier-up lane was built for `thread.spawn` bodies
   (fresh activation, argv marshalled). Mainline tier-up requires the emitted function to run over
   the *live, mid-computation* window with `m.data` already materialized and the same `win` base.
   The whole-module paths have explicit materialization (`svm_wasmjit_init_window`,
   `browser/src/lib.rs`); the `svm_par_run`/`PAR_TIERUP` lane's window provenance needs a diff
   against `svm_run_onramp`'s init.
2. **Baked `size_log2` mismatch** — `mapped` is frozen at emit time; a window instantiated or grown
   past it faults legitimate accesses.
3. **Subset-classification hole** (least likely — the trap preceded any cross-tier call).

This came first: "default the JIT on" (slice 2) would otherwise ship a path a real consumer has
already shown to fault. Slice 0 is now diagnosed and the confirmed defect is fixed (below).

## 5. Slices

Ordered by leverage ÷ risk; each lands with tests per AGENTS.md.

### Slice 0 — mainline tier-up over a live window (DIAGNOSED; gate fix landed)

**Diagnosis (empirical, native).** A new native harness (`svm-wasm-jit/tests/tierup_live_window.rs`)
drives the real loop — `bytecode::VcpuReactor` opens over a caller-owned window, runs `_start`
(materializing `m.data`), and tiers up a mainline `Call` onto emitted wasm mirrored over that live
window. It is the native twin of the browser `svm_par_run`/`PAR_TIERUP` lane, which had **no native
coverage** (every prior `tierup.rs` test ran each `f{func}` in isolation over a hand-seeded window).

Against the JACL doc's ranked hypotheses:
- **Hypothesis 1 (data/window-materialization) — ruled out for the native path.** With a
  window sized to `1 << size_log2` and data materialized by the interpreter, mainline tier-up over
  the live window (incl. a data-segment read and a write-then-read round-trip) matches the
  interpreter oracle exactly. The interpreter's `init_data` + the shared `win` base already satisfy
  the contract; the harness pins it.
- **Hypothesis 2/3 (baked `size_log2` vs a page-managed window) — confirmed as a real defect.**
  `compile_module_tierup` (a **public** entry) did **not** apply the `module_uses_page_ops` gate that
  `compile_jit`/`compile_nested` apply (NESTED_JIT Track 3). So a guest that manages its own pages —
  which grows/remaps its window at runtime, exactly the JACL compiler's heap shape — got hot leaves
  emitted with a `mapped = 1 << size_log2` baked at emit time; an emitted mask-only access then
  ignores the live page state (a grown/remapped region), diverging from the interpreter → the
  `MemoryFault → unreachable` mid-body, before any `env.call_interp`, that the spike saw. The JACL
  spike called `compile_module_tierup` directly, bypassing `compile_jit`'s gate.

**Fix (landed).** `compile_module_tierup_caps` now self-applies the page-op gate: a page-op module
emits nothing (all-`false` bitmap, valid imports-only wasm), identical to `compile_interp_only` and
to `compile_jit(Threaded)`. The gate now holds regardless of which public entry a host calls.
Pinned by `tierup_page_op_module_emits_nothing` (red→green: was `[false, true]`, now `[false,
false]`), a control that still emits without the page op, and a cross-entry agreement test.

**Scope honesty.** This closes the confirmed native-reproducible defect. One JACL hypothesis-1
residue can only be checked with the browser Worker harness (node + the wasm cdylib), out of reach
here: whether the `svm_par_run`/`PAR_TIERUP` lane's `win`/`size_log2` provenance matches
`svm_run_onramp` for a guest that *grows* its window mid-run. The native evidence says the shared-
`win` contract is sound when sizes match; the remaining risk is purely the grow/remap case, which
the landed gate now routes to the interpreter anyway (a page-managing guest emits nothing). If a
future consumer needs a *page-managing* guest accelerated, that is the gated-(b)-with-elision
escalation NESTED_JIT Track 3 documents — not in scope here.

### Slice 1 — compiled-output cache (the biggest playground feel-fix)

Two halves sharing **one content key** (the module's encoded-image digest — `svm_encode::digest256`
over `encode_module`, the same identity the durable module-grant registry already uses). Cache
**code**, never window/guest state.

- **Native — LANDED (`svm-run` embedder seam).** `svm_run::CompiledCache`: a content-keyed map of
  `PowerboxProgram`s (the existing build-once/run-many split, now dedup'd by module identity rather
  than object identity). `run(&module, stdin)` compiles a module on first sighting and reuses the
  native code on every later identical module — byte-identical to `run_powerbox` either way. Fills
  the gap `ISSUES.md` named ("no such API today"). Fresh-window safety is inherited from
  `PowerboxProgram` (fresh window + host reset per run) and pinned: `tests/compiled_cache.rs` proves
  reuse-without-recompile, content-not-object keying, `run_powerbox` parity across inputs, no
  state-leak across reuses, distinct-module isolation, and that a refused (concurrent) module is not
  cached and does not poison the cache.
- **Browser — NEXT (two steps, low-risk first).** The playground's dominant pattern is re-Running
  the same module (edit stdin, re-Run Lua/SQLite/chibicc). Today every Run re-emits (cdylib) and
  re-compiles (`WebAssembly.compile`, `wasmjit-module.js:32`) with no cross-Run reuse.
  - **Step 1 (JS-only, no TCB/concurrency change):** a JS `Map` from the module's content digest →
    the compiled `WebAssembly.Module`, consulted in `driveJitRun` (`wasmjit-module.js`) before
    `WebAssembly.compile`. Key on the *source/module* the Run was launched with (the same bytes
    `moduleCache` already holds by URL, or the editor text for editable cards) — not the emitted
    bytes, so the lookup precedes emit. This skips V8 compile (and, if we also memoize the emitted
    bytes, the cdylib emit) on a re-Run. It touches no `static mut`/`CODEGEN_LOCK`/`PAR_RUN_GEN`
    state, so it can't introduce a cross-Worker race — the reason to do it first.
  - **Step 2 (cdylib, only if step 1's emit cost still shows):** replace the per-Run `PAR_RUN_GEN`
    emit-dedup key with the content digest (`svm_encode::digest256` over the decoded module — the
    *same* key `svm_run::CompiledCache` uses), so the cdylib itself skips re-emit across Runs, not
    just across Workers within a Run. Higher-risk (it edits the I22 shared-stash lifetime), so gated
    on a measured need.
  - **Validation:** not locally runnable here (no wasm toolchain in this environment). Rides CI's
    `browser-real` Chromium differential suite for correctness (a stale-cache bug shows as a
    render/output divergence there), plus a first-vs-second-Run timing assertion added to the
    playground recorder / a `node` bench (`bench_jit.mjs`-style) for the win itself.
- **Gate:** (native, met) `compiled_cache.rs` green + `run_powerbox` parity; (browser) second Run of
  a light script ≥ interpreter-only time — kills the "net slower under JIT" footgun.

### Slice 2 — default the JIT tier on where eligible (after slice 0)

- Flip the per-demo checkbox default to on when eligibility passes (`compile_tier_eligibility` /
  `analyze` stay the single routing predicate — INVARIANTS #9's one-veto rule). Fail-closed
  behavior unchanged; the checkbox remains as an off-switch and for parity "prove it" runs.
- Includes the SVM-text editor path where the recipe is compute-only (today it has no JIT toggle at
  all).
- **Gate:** the existing per-demo parity assertions run in both default states in `browser-test.mjs`.

### Slice 3 — redundant confinement-check elimination (the Lua/SQLite lever)

- The BROWSER.md-named lever, scoped conservatively: elide or hoist `emit_confine` only where the
  invariant is *proven* — e.g. repeated accesses off one base+bounded-offset within a block (the
  same reasoning `svm-opt`'s D63-sound offset-disjointness already uses), or a dominating check of
  the same `(base, range)`. Stays entirely inside the masking lowering; every elision form gets its
  own fuzz corpus entry (this is the security hinge — INVARIANTS #2; AGENTS.md fuzzing rule).
- **Measured target, not vibes:** Lua 5M-loop and SQLite REPL throughput on the playground path;
  the bench README's `chase`/`chase_rand` rows for the honest memory-kernel view.
- If the win plateaus, **function splitting** is the named fallback (BROWSER.md slice-8 findings);
  stackification and the relooper stay rejected on the recorded A/B evidence.

### Non-slice — standalone pure-wasm export (deliberately deferred)

Gated on a named compute-only consumer (INVARIANTS #1) and an owner renegotiation of #2's boundary
(engine-bounds-check-as-confinement). Post-D63 the parity story is closer than it used to be; the
JACL macro-staging fast path (AOT `jacl_emit.wasm` + tiny staged macro bodies) is also evidence
that consumers can often sidestep the need entirely. Revisit if one shows up.

## 6. Doc fixes landed with this assessment

- `WASM.md` § semantic divergence + spec-runner caveat: updated the stale pre-D63 "OOB masks,
  doesn't trap" notes to the D63 trap-confinement semantics (trap at the window boundary; the
  residual divergence is boundary granularity, not mask-vs-trap).

## Invariants check

| Invariant | How the plan holds it |
| --- | --- |
| #1 small core | no new backend; cache + default-flip are embedder policy; elision is evidence-gated |
| #2 confinement = masking pass | slice 3 stays inside the pass, fuzzed per elision form; standalone export deferred behind renegotiation |
| #6 one world / fresh activation | cache holds code only, never window state; pinned by test |
| #9 oracle; decline, never diverge | routing predicate unchanged and single; parity gates run in both toggle-default states; slice 0 fixes a decline-path fault |
