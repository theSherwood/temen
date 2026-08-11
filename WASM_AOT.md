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

Until this is fixed, "default the JIT on" (slice 2) would ship a path a real consumer has already
shown to fault. So the bug comes first.

## 5. Slices

Ordered by leverage ÷ risk; each lands with tests per AGENTS.md.

### Slice 0 — mainline tier-up over a live window (bug fix; unblocks slice 2)

- **Repro:** a minimal `InterpDriven` guest that (a) has a data segment and (b) is forced through
  per-function `TierUp` from *mainline* code (not `thread.spawn`), driven through the
  `svm_par_run` loop in the browser test harness. Per the JACL doc, if it faults the problem is
  "mainline TierUp over a live window", independent of guest size.
- **Diagnose:** instrument `env.trap` (test-only) to log trap code + `eff`/`mapped`/`win`; diff the
  window setup between `svm_run_onramp` and the `svm_par_run`/`PAR_TIERUP` lane (data
  materialization, provenance of `win` and `size_log2`).
- **Fix direction:** per the JACL doc's option (a)/(b) — either make mainline tier-up run emitted
  code over the interpreter's *current* live window as a first-class contract, or enforce (and
  assert at the seam) that the par lane materializes data and shares the exact window. Whichever it
  is, the contract gets stated in `svm-wasm-jit`'s module docs and pinned by the repro test.
- **Gate:** repro test red→green; existing tier-up/threads differentials stay green.

### Slice 1 — compiled-output cache (the biggest playground feel-fix)

- **Browser:** cache per module content-hash, across Runs: emitted wasm bytes → the compiled
  `WebAssembly.Module` (V8 shares compiled code on structured clone; the per-code *instance* cache
  per Worker already exists for §22 units — extend the pattern to the top-level emitted module).
  First Run pays emit+compile once; every later Run of the same module (the playground's dominant
  pattern: edit stdin, re-Run Lua/SQLite/chibicc) skips both.
- **Native:** the primitive exists (`svm_jit::compile → CompiledModule::run`); add the caching
  policy at the embedder seam (`svm-run`), keyed the same way.
- **Fresh-window semantics are untouched:** we cache *code*, never window state — each Run still
  builds a fresh window (INVARIANTS #6 / DESIGN §12 activation model). A cached-code run must be
  differentially indistinguishable from a cold one; add that as a test.
- **Gate:** second Run of a light script ≥ interpreter-only time (kills the "net slower under JIT"
  footgun); bench the first-vs-second Run delta in the playground recorder.

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
