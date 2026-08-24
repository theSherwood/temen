# wasm-AOT viability & playground speed plan

**Status: assessment settled (no fifth backend); slices proposed, unstarted.** This answers "should
we build a wasm-AOT backend to make the playground fast?" with evidence from the tree, and lays out
the work that actually recovers the speed. Fold into `DESIGN.md`/`BROWSER.md` and drop this file
once the slices land (repo convention).

## 1. Verdict

**Do not build a fifth backend.** Both halves of "wasm-AOT" already exist:

- **"AOT" is the offline lane we already ship.** `temen-llvm`/`temen-opt`/`temen-peval` are ahead-of-time,
  host-side, and produce verified IR; every playground demo module (Lua, SQLite, DOOM, QuickJS,
  Postgres) is *already* AOT-compiled that way into `.temen` (`LLVM.md`, `OPT.md`, DESIGN §20).
- **"Emit wasm" is backend 4.** `temen-wasm-jit` turns verified IR into a real `WebAssembly.Module`
  and, where it engages, already lands **at or below native Cranelift** on compute (`xorshift`
  2.0 ns emitted vs 1.9 ns native; 16–112× over the interpreter-in-wasm —
  `bench/cross-engine/README.md` § "TEMEN-in-wasm, the JIT tier"). Emitting *earlier* ("AOT-at-
  `temen_par_compile`", BROWSER.md) is a compile-point choice, not a new backend.

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
   table omits"). Only `.temen` *bytes* are cached today, not emitted wasm or instances.
3. **Per-access confinement in giant functions.** For Lua/SQLite the JIT engages but yields only
   ~3×/~1.3×: the cost is `emit_confine`'s ~7-op bounds-check-and-mask on **every** guest access,
   on top of wasm's own linear-memory bounds check (the double indirection the bench measures as
   `chase_rand` ~3.4×). Not V8 compile time (~6 ms for 3 MB), not the giant function itself
   (TurboFan compiles it; relooper A/B'd at zero gain and reverted). BROWSER.md § slice-8 already
   names the real levers: **redundant-check elimination with proof**, or function splitting.
4. **Program-independent runtime init, re-run every Run.** For a language on-ramp (QuickJS, Lua,
   Postgres…) each Run rebuilds the whole guest runtime before it touches the user's code —
   `JS_NewRuntime` + `JS_NewContext` + every intrinsic for QuickJS. It is *fixed* (identical for a
   trivial program and a heavy one) and it dominates the wall clock of a light script: in Chromium
   the qjs warm floor is ~380 ms whether the input is empty, `1;`, or the user's fib/sort/JSON
   program — the program itself contributes ~0. Neither a new backend nor the slice-1 code cache
   touches this: the cache keeps V8's *compiled code* warm, but `JS_NewRuntime` still runs from
   scratch every Run. The lever is a **warm-runtime snapshot** (follow-on to slice 1 below).

## 3. What "pure wasm" would buy, and its sanctioned form

The one genuine win hiding in "output pure wasm" is cause 3: emit a module whose linear memory *is*
the guest window, and the engine's own bounds check subsumes ours — deleting the ~7-op inline
sequence per access. That is a real ~2× on memory-bound guests, and post-D63 the semantic gap is
small (both Temen and wasm now **trap** OOB; what differs is the boundary — Temen's power-of-two
`mapped` vs wasm's page-granular size — see the updated WASM.md divergence note).

But it moves confinement out of the one fuzzed masking pass and into "trust the engine," which is
exactly what INVARIANTS #2 exists to reject, and what the NESTED_JIT Track 3 decision declined for
the same reason ((c)+(a): scope out + fail closed, zero TCB growth). The sanctioned form of the same
win is the lever BROWSER.md already names: **elide checks we can prove redundant, inside the
existing masking regime** (slice 3 below). A standalone-memory pure-wasm *export* stays a
possibility behind INVARIANTS #1's bar — a named compute-only consumer — and is deliberately **not**
scheduled here.

## 4. The JACL consumer finding (adopted as slice 0)

`jacl_impl/docs/TEMEN_BROWSER_TIERUP_FINDINGS.md` is a named consumer hitting the exact path slice 2
depends on: per-function tier-up (`compile_module_tierup` + `VcpuEvent::TierUp` via `temen_par_run`)
of a **mainline** `InterpDriven` guest trapped `MemoryFault → unreachable` inside the emitted body,
before any `env.call_interp`. Their ranked hypotheses — all consistent with our code:

1. **Window/data-materialization mismatch.** The tier-up lane was built for `thread.spawn` bodies
   (fresh activation, argv marshalled). Mainline tier-up requires the emitted function to run over
   the *live, mid-computation* window with `m.data` already materialized and the same `win` base.
   The whole-module paths have explicit materialization (`temen_wasmjit_init_window`,
   `browser/src/lib.rs`); the `temen_par_run`/`PAR_TIERUP` lane's window provenance needs a diff
   against `temen_run_onramp`'s init.
2. **Baked `size_log2` mismatch** — `mapped` is frozen at emit time; a window instantiated or grown
   past it faults legitimate accesses.
3. **Subset-classification hole** (least likely — the trap preceded any cross-tier call).

This came first: "default the JIT on" (slice 2) would otherwise ship a path a real consumer has
already shown to fault. Slice 0 is now diagnosed and the confirmed defect is fixed (below).

## 5. Slices

Ordered by leverage ÷ risk; each lands with tests per AGENTS.md.

### Slice 0 — mainline tier-up over a live window (DIAGNOSED; gate fix landed)

**Diagnosis (empirical, native).** A new native harness (`temen-wasm-jit/tests/tierup_live_window.rs`)
drives the real loop — `bytecode::VcpuReactor` opens over a caller-owned window, runs `_start`
(materializing `m.data`), and tiers up a mainline `Call` onto emitted wasm mirrored over that live
window. It is the native twin of the browser `temen_par_run`/`PAR_TIERUP` lane, which had **no native
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
here: whether the `temen_par_run`/`PAR_TIERUP` lane's `win`/`size_log2` provenance matches
`temen_run_onramp` for a guest that *grows* its window mid-run. The native evidence says the shared-
`win` contract is sound when sizes match; the remaining risk is purely the grow/remap case, which
the landed gate now routes to the interpreter anyway (a page-managing guest emits nothing). If a
future consumer needs a *page-managing* guest accelerated, that is the gated-(b)-with-elision
escalation NESTED_JIT Track 3 documents — not in scope here.

### Slice 1 — compiled-output cache (the biggest playground feel-fix)

Two halves sharing **one content key** (the module's encoded-image digest — `temen_encode::digest256`
over `encode_module`, the same identity the durable module-grant registry already uses). Cache
**code**, never window/guest state.

- **Native — LANDED (`temen-run` embedder seam).** `temen_run::CompiledCache`: a content-keyed map of
  `PowerboxProgram`s (the existing build-once/run-many split, now dedup'd by module identity rather
  than object identity). `run(&module, stdin)` compiles a module on first sighting and reuses the
  native code on every later identical module — byte-identical to `run_powerbox` either way. Fills
  the gap `ISSUES.md` named ("no such API today"). Fresh-window safety is inherited from
  `PowerboxProgram` (fresh window + host reset per run) and pinned: `tests/compiled_cache.rs` proves
  reuse-without-recompile, content-not-object keying, `run_powerbox` parity across inputs, no
  state-leak across reuses, distinct-module isolation, and that a refused (concurrent) module is not
  cached and does not poison the cache.
- **Browser — LANDED (step 1, JS-only).** `wasmjit-module.js`'s `driveJitRun` now consults a
  cross-Run `Map` (`jitModuleCache`) keyed by a caller-supplied **stable module identity** before
  `WebAssembly.compile`: on a hit the compiled `WebAssembly.Module` is reused verbatim, skipping V8
  codegen; a miss compiles and caches. `play.js` passes the module's content-addressed URL for on-ramp
  cards and stable keys for the chibicc compiler/self-host paths (`ex.url` / `'chibicc-compiler'` /
  `'chibicc-selfhost'`). **Code only** — a fresh instance/window/env cell is built per Run, so no guest
  state crosses Runs; the cache is bounded (16 entries, LRU-ish) and opt-in (no key ⇒ no caching, so
  the dynamic in-browser-compiled-C path and the parity prover are unaffected). It touches no
  `static mut`/`CODEGEN_LOCK`/`PAR_RUN_GEN` state — no cross-Worker race surface.
  - **Measured in Chromium** (`browser-jit-cache-test.mjs`): re-Running the same module produces
    byte-identical stdout every Run and compiles exactly once (`{compiles:1, hits:2}`). Warm re-Runs:
    **hello_c 33 ms → 4 ms** (now *beats* the interpreter's ~8 ms — the "light script slower under
    JIT" footgun is fixed), **qjs_repl ~4.4 s → ~2.5 s (~1.8×)** — the reused Module keeps V8's
    tiered-up code warm across Runs, saving far more than the ~30–50 ms `WebAssembly.compile` alone.
- **Browser — step 2 (deferred, gated on need).** Have the cdylib itself skip *re-emit* across Runs
  (replace the per-Run `PAR_RUN_GEN` emit-dedup key with the `temen_encode::digest256` content key —
  the same one `temen_run::CompiledCache` uses). Higher-risk (it edits the I22 shared-stash lifetime).
  Step 1's warm numbers already clear the footgun gate, so this waits for a measured re-emit cost that
  step 1 doesn't cover.
- **Gate:** (native, met) `compiled_cache.rs` green + `run_powerbox` parity; (browser, met) second
  Run of a light script (hello_c 4 ms) now beats interpreter-only (~8 ms) — footgun closed, pinned by
  `browser-jit-cache-test.mjs`.

### Follow-on to slice 1 — warm-runtime snapshot (PROTOTYPED native; the do-nothing-program floor)

Slice 1 caches *code*; it does nothing about **cause 4** — the program-independent runtime init that
re-runs every Run. This prototype attacks that directly: run the guest's init **once**, snapshot the
post-init guest memory, and **restore the snapshot per Run**, evaluating only the user's code on top.

- **Shape.** Split the on-ramp driver into three exports (`crates/temen-run/demos/quickjs/qjs_snapshot.c`):
  `main` (the original cold read→init→eval→print, the baseline), `warmup` (init runtime+context+bindings
  into statics, then return — no stdin, no eval, so the produced memory is program-independent), and
  `eval_run` (read stdin, eval over the warm context, print). The host snapshots after `warmup` and
  restores before each `eval_run`.
- **Fresh-per-Run isolation is preserved (INVARIANT #6).** Every Run restores the *same* post-warmup
  image into a fresh zeroed window, so a `var` defined in one Run cannot leak into the next. Proven by
  byte-for-byte cold≡warm output parity (below); this is the same fresh-activation guarantee the
  code cache already holds, extended to a restored-not-rebuilt warm image.
- **The memory-model wrinkle (why it's not just a memcpy of the window).** These are ordinary exports
  (`params = [i64 sp]`), not the synthesized `_start` (func 0), so the harness reproduces what `_start`
  does for the on-ramp: grant the §3e powerbox + bind the module's manifest imports (deterministic
  handles, re-established per Run), seed the guest heap bump words (`POWERBOX_HEAP_BRK`/`_TOP`), and
  pass `sp = powerbox_entry_sp`. The on-ramp allocator grows the heap **above** the declared window
  (`heap_base = 1 << size_log2`), and that growth's mapped-width state lives in the `Mem`, *not* in the
  window bytes — so a naïve window memcpy restores the bytes but faults on the warm heap
  (`MemoryFault`). The native prototype maps a **larger window** (2^26) so the whole heap stays inside
  the mapped region: no `vm_map` growth, a contiguous guest image captured by a plain memcpy of the
  live prefix `[0, brk)`. **The browser session no longer needs that trick (#816):** the module keeps
  its declared window, the heap `vm_map`-grows into the 2^26 backing (the run's reservation is clamped
  to it, so over-growth fails probeably instead of silently dropping writes), and the warmup's
  contiguous committed extent is captured alongside the image and re-established — without re-zeroing —
  before every eval (`SharedProgram::run_over_grown`, `browser/tests/warm_grow.rs`). A warmup that
  leaves page state one bound can't represent (sparse/`Ro`/`Unmapped`) fails closed at open. The
  warm+JIT tier is unchanged: it opens only for `WasmDriven` (no-page-op) modules, so a growing guest
  evaluates on the interpreter warm path until the tier-up driver work (#809) reaches it.
- **Measured native** (`crates/temen-llvm/examples/qjs_snapshot.rs`, release, bytecode interpreter, the
  same QuickJS on-ramp module as the playground): warmup once ~23 ms; **live warm image ~4.1 MiB**;
  restore ~3.5 ms (memcpy the live prefix).

  | program | cold ms | warm ms (restore+eval) | speedup |
  | --- | --- | --- | --- |
  | `1;` (trivial) | 32 | 8 | **4.0×** |
  | fib/sort/JSON (user's) | ~150 | ~105 | ~1.4× |
  | 100k-iter loop | ~3900 | ~3150 | ~1.2× |

  Byte-identical output on all three (`fib 0 1 1 2 …`, sorted array, `JSON.stringify` incl. `Math.PI`,
  `0.1+0.2`, the loop sum `4999950000`). The fixed ~24 ms init is replaced by a ~3.5 ms restore — the
  win is largest for light scripts (where init dominated) and shrinks as the eval itself grows, exactly
  as cause 4 predicts.
- **Browser plumbing — PROTOTYPED (`temen_warm_open`/`temen_warm_eval`/`temen_warm_close`).** A stateful
  browser session (`browser/src/lib.rs`, the twin of the native prototype and of the `PgSession`
  reactor): `temen_warm_open` decodes the two-phase driver, enlarges the mapped window to `WARM_MAPPED_LOG2`
  (2^26 — the same keep-the-heap-inside trick), runs `warmup` **once** over an owned window, and keeps
  the live prefix `[0, brk)` as the warm image; `temen_warm_eval` restores that image (zeroing only the
  heap tail a prior eval grew, for byte-identical fresh state) and runs `eval_run`, staging stdout into
  the same capture slots `temen_run_onramp` uses; `temen_warm_close` frees it. Reuses `grant_onramp_caps` and
  `SharedProgram::run_over(seed_data=false)`. Keeping the whole ~4 MiB image (vs the reactor `Session`'s
  256 KiB `REACTOR_SNAP_CAP`) is what closes the gap for QuickJS.
- **Measured in Node/V8** (`browser/warm-snapshot-test.mjs`, the shipping engine FFI, the committed
  `web/assets/qjs_snapshot.temen`): warmup once ~430 ms — i.e. the QuickJS runtime rebuild **is** the
  ~380–430 ms warm floor — then:

  | program | cold ms | warm ms (restore+eval) | speedup |
  | --- | --- | --- | --- |
  | `1;` (trivial) | ~570 | **2** | **~250×** |
  | fib/sort/JSON (user's) | ~400 | ~70 | ~6× |
  | 100k-iter loop | ~4600 | ~2900 | ~1.6× |

  Byte-identical cold≡warm output on all three (`temen_run_onramp` `_start`/cold vs `temen_warm_eval`/warm).
  The "do-nothing program takes >1 s" case collapses to ~2 ms — the whole fixed init is gone. The win is
  far larger than native because the browser runs QuickJS init through the interpreter-in-wasm; it also
  composes with the slice-1 code cache (cache keeps V8's code warm, snapshot skips `JS_NewRuntime`).
- **Card wiring (next).** The exports exist and are proven; wiring the playground's qjs card to
  `open`-once-then-`eval`-per-Run (with a warm session cached across Runs, invalidated on module change)
  is the remaining UI step. Kept separate so the reactor lands with its own test first.
- **Gate (met):** cold≡warm output parity on the QuickJS on-ramp across trivial/heavy/loop inputs,
  native (`qjs_snapshot.rs`) and through the wasm FFI (`warm-snapshot-test.mjs`), with the fixed-init
  cost demonstrably removed from the warm path.

### Slice 2 — default the JIT tier on where eligible — LANDED

- **Demo cards — already default-on.** Every `ex.jit` card already builds its "wasm-JIT" checkbox
  `checked = true` (`play.js` `buildCard`), with the checkbox as an off-switch and the "prove it"
  button for the interp≡JIT parity run. Fail-closed is unchanged: a non-eligible or trapping module
  throws and the card falls back to the interpreter. `compile_tier_eligibility`/`analyze` stay the
  single routing predicate (INVARIANTS #9). Nothing to change here.
- **TEMEN-text compute recipe — now tiers up (the gap the plan named).** The TEMEN-text editor's
  `plain` ("none / compute only") recipe ran pure-interpreter across Workers with no JIT path.
  `runText` now passes `tierup: true` for it, so the interpreter drives and hot in-subset functions
  run on emitted wasm over the same live window (fail-closed per-function; the `§22-jit`/`§14-inst`
  recipes keep their own JIT, `io` stays on the interpreter for now). The done-line reports how many
  regions tiered up.
- **This also closes slice 0's browser residual.** That path is the `temen_par_run`/`PAR_TIERUP`
  **mainline tier-up over a live window** the JACL postmortem flagged — the one piece slice 0 could
  only pin natively. `browser-tierup-mainline-test.mjs` (new) now validates it in real Chromium: an
  TEMEN-text compute guest whose root loops calling a **window-round-tripping** leaf returns the
  identical value with tier-up on as all-interpreter (INVARIANT 9), with tier-up actually firing
  (50 000 regions). (Aside surfaced by the test: the TEMEN-text `run()` path does not materialize
  `data` segments into the window — a pre-existing property, not introduced here; both tiers read
  identically, so it's a differential no-op. A follow-on if a hand-written TEMEN-text guest ever needs
  a data segment.)
- **Gate (met):** `browser-test.mjs` (the full playground, tier-up now on for `plain`) stays green,
  and `browser-tierup-mainline-test.mjs` pins interp≡tier-up + non-vacuity.

### Slice 3 — redundant confinement-check elimination (the Lua/SQLite lever) — LANDED (provable-bound form)

The BROWSER.md-named lever, in its most-proven form: the wasm-JIT now elides a memory access's
bounds-**trap** branch when the address is *provably* in-window, reusing the native Cranelift JIT's
existing D63 "guard-when-bounded" analysis rather than inventing one.

- **Stage 1 (shared proof).** Lifted `UB_TOP`/`ub_at`/`ub_of`/`in_window` out of the native JIT's
  private copy into `temen_ir::bounds` — one audited definition of the confinement veto predicate for
  both JITs (INVARIANTS #9), behavior-preserving for `temen-jit` (jit_diff stays green).
- **Stage 2 (wasm elision).** `emit_block_body` threads a per-block upper-bound map (`ubs`) in
  lockstep with the emitter's own value numbering (reset per block; block params = `UB_TOP`), and at
  each Load/Store/atomic/v128 access `elide_access` consults `in_window`. When proven bounded, the
  `eff > mapped - width` trap branch is dropped. **The `& MASK` clamp is always emitted** — so this
  is a *strictly safer subset* of the native JIT's elision (which also drops the clamp): a wrong
  proof here is at worst a trap-parity divergence (caught by the differential), never an escape.
- **Fuzzed as its own unit** (AGENTS.md "…or proven bounded"): `differential.rs` gains bounded-index
  kernels that actually fire the elision (`(v0 & K)*W`, the top-byte boundary, the width-8
  no-elide edge, a mixed elided/non-elided block), a size proof that the branch is really removed
  (`elision_actually_removes_bytes`), and a 3000-iteration `elision_boundary_sweep` asserting
  trap+value parity with the interpreter oracle around the window edge. `temen_ir::bounds` carries unit
  tests for `in_window` overflow-safety and `ub_of`'s rules.
- **Scope.** Matches the native proof exactly (const / `& K` / `+`/`|`/`^` / `* W` / `ExtendI32U`) —
  notably it does **not** yet model `PtrAdd` (the C-frontend address op), so C-guest addresses stay
  conservatively checked, same as on `temen-jit`. Widening the proof (or the "same-base dominating
  check" redundancy form) is a follow-on if a measured target needs it.
- **Measured target (next):** Lua 5M-loop / SQLite REPL throughput on the playground path and the
  bench README's `chase`/`chase_rand` rows — to quantify the win on real programs. If it plateaus,
  **function splitting** is the named fallback (BROWSER.md slice-8); stackification and the relooper
  stay rejected on the recorded A/B evidence.

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
| #6 one world / fresh activation | cache holds code only, never window state; the warm-runtime snapshot restores an identical *program-independent* post-init image into a fresh zeroed window per Run (no guest state crosses Runs) — pinned by cold≡warm output parity |
| #9 oracle; decline, never diverge | routing predicate unchanged and single; parity gates run in both toggle-default states; slice 0 fixes a decline-path fault |
