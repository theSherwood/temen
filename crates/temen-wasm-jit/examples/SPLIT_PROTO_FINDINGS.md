# #1110 emit-split prototype — findings

**TL;DR: the emit-split premise is falsified.** Splitting the emitted wasm module does **not** speed up
the hot path's first-Run tier-up. The variable that governs first-Run latency is the **hot function's own
size** (its TurboFan compile time), which module-splitting does not change. The real lever is reducing
*function* size (function outlining), not *module* size.

## What was built

- **`compile_module_split`** (in `temen-wasm-jit`): emits one partition of a whole-program split. A
  `Call`/`ReturnCall` to a function emitted by a sibling partition lowers to a `call_indirect` through the
  shared reserved funcref table (`env.__indirect_function_table`) at the callee's index — the same
  host-populates-every-slot contract as Model B2. Confinement mask stays the compile-time constant
  `1<<table_log2` (invariant I2). Additive: every existing caller passes an empty `cross_module` slice and
  is byte-for-byte unchanged.
- **`tests/split.rs`**: correctness differential — the same IR split at different cut points (plus
  single-module, plus the interpreter oracle) must all agree. Covers cross-module `call_indirect` and
  `return_call_indirect`. Green.
- **`examples/split_proto.rs` + `examples/split_proto.mjs`**: the measurement vehicle. The `.rs` builds a
  synthetic guest (a hot loop `f0` calling helper `f1`, plus cold filler to inflate the module) and emits
  four configs sharing one reserved-table ABI; the `.mjs` instantiates each under Node/V8 — one shared
  memory + one shared table, every slot populated — and times `f0` per-run to expose the Liftoff→TurboFan
  tier-up run. `f0`'s body can be spread across N blocks to make `f0` itself a large function (the QuickJS
  `JS_CallInternal` shape).

## The experiment

V8's Liftoff→TurboFan tier-up is only observable under V8 (wasmi/Cranelift are single-tier), and the
#1068 data that motivated #1110 was itself collected under Node/V8 — so Node is the right instrument.

Configs (all the *same* IR, different module cuts):
- `single`     — whole-program: `f0` sits in the multi-MB module (status quo).
- `split_good` — `f0`,`f1` in a tiny module; cold filler in a second module (hot path intra-module).
- `split_xmod` — `f0` and `f1` each in their own tiny module; filler in a third (pure cross-module call).
- `split_bad`  — `f1` stranded in the cold module (a deliberately bad partition).

## Result 1 — module split does not change tier-up latency

With a QuickJS-`JS_CallInternal`-scale hot function (`f0` ≈ 1.9–2.3 MB emitted, ~1000 blocks), measured in
**separate Node processes** (no cross-config V8 code-cache sharing), `single`, `split_good` and
`split_xmod` tier up at the **same run** with identical Liftoff and steady-state timings:

```
single      | per-run ms:  310 274 308 287  81  83 ...   tier-up @ run 4   steady 80ms
split_good  | per-run ms:  302 258 251 280  88  88 ...   tier-up @ run 4   steady 80ms
split_xmod  | per-run ms:  304 250 270 282  81  82 ...   tier-up @ run 4   steady 84ms
```

Isolating the hot function in its own tiny module bought **nothing**: V8 tiers a function up on its own
dynamic-tiering budget, independent of the surrounding module's size, and does **not** gate tier-up on the
whole module's baseline (Liftoff) compilation finishing.

## Result 2 — tier-up latency is governed by hot-*function* size

A single large hot function stays on Liftoff for several runs and pays a large cold (run 0) cost. Measured
at **fixed** N (a caveat below), a 1.9 MB `f0`: run 0 ≈ 330–390 ms, full-speed only by run ~2–3, steady
61 ms. A trivial (tiny) `f0` tiers up at run 0 regardless of the 10 MB of filler around it.

> **Caveat (correction).** An earlier version of this doc reported a monotonic `size → tier-up run` curve
> (`0.44MB→run1 … 2.7MB→run11`). That sweep used a *different iteration count per size* (`N = 1.1M/blocks`);
> V8's tier-up trigger is iteration-budget-based, so fewer iterations ⇒ later observed tier-up. Re-measured
> at fixed N, the tier-up *run* of a single function is noisy (±2–3 runs) and not cleanly size-monotonic.
> The robust, controlled statements are Result 1 (identical N ⇒ split makes no difference) and Result 4
> (identical N + identical total code ⇒ outlining helps).

## Result 3 — cross-module call cost

With a large hot function the per-iteration helper call is a negligible fraction: `split_xmod`
(cross-module `call_indirect`) vs `split_good` (intra-module direct call) differ by ~0% at steady state.
The cross-module cost is only visible for a *trivial* hot loop where the call dominates (≈ +21% there) —
but that regime tiers up at run 0 anyway, so it never matters in practice. So cross-module dispatch is
cheap where it would be used; it is simply pointed at the wrong problem.

## Result 4 — function outlining *does* help (the constructive lever)

`--outline K` emits the hot path as a small dispatcher loop calling K handler functions (each ≈ body/K)
instead of one monolithic `f0`, **same total hot code**. Monolithic 1.9 MB `f0` vs 8×0.24 MB handlers,
fixed N=1100, 3 fresh Node processes each:

| | cold run 0 | full-speed by | steady |
|---|---|---|---|
| Monolithic 1.9 MB | 326–391 ms | run 2–3 | 61 ms |
| Outlined 8×0.24 MB | 122–141 ms | run 1 | 61 ms |

The smaller functions reach TurboFan almost immediately, so even run 0 is ~2.7× cheaper and the hot path
is full-speed by run 1 — with **no steady-state penalty** (V8 inlines the direct intra-module calls under
TurboFan). This is the same lever #1068/#1110 were after, applied to *function* size, not *module* size.

Caveat for the follow-up: the synthetic dispatcher uses **direct** calls to the handlers (best case for
inlining). Outlining a real `JS_CallInternal` br_table dispatcher would select handlers **indirectly** (by
opcode) and marshal interpreter state (PC/sp/stack) across the cut — both need their own measurement.

## Result 5 — the indirect-dispatch tax is ~0% (Slice 0, #1120)

The realistic form selects handlers **indirectly** (by opcode), which TurboFan cannot inline. The `indirect`
config emits the dispatcher and handlers as separate modules, so the dispatcher→handler calls become
cross-module `call_indirect`. 8 indirect calls/iteration (a conservative worst case vs one-per-opcode),
fixed N=1100, 4 fresh Node processes each:

| | cold run 0 | full-speed by | steady |
|---|---|---|---|
| Monolithic 1.9 MB | 335–390 ms | run 1–3 | 61–63 ms |
| Outlined **direct** | ~128 ms | run 1 | 61 ms |
| Outlined **indirect** | 123–137 ms | run 1 | 60–61 ms |

Making the dispatch indirect costs **nothing measurable** at steady state (60–61 ms = monolithic) while
keeping the full cold-start win (~2.8×, full-speed by run 1). The indirect call is amortized because each
handler does real work. Caveat: these handlers are uniformly medium (~0.22 MB); a real interpreter has a
mix including tiny hot opcodes, where a per-call indirect could matter — so the outlining **granularity**
(group tiny opcodes to keep each handler's work above the call cost) is the real design knob (Slice 2).

## Conclusion / recommended pivot

Module-splitting is the wrong lever for first-Run latency. The bottleneck is a single giant hot function
(`JS_CallInternal`) whose TurboFan compile takes several runs to finish, and you cannot split a function
across modules. **The lever that works is reducing hot-function size — function outlining** (Result 4):
breaking the giant dispatch function into smaller functions so each reaches TurboFan on run 1. It is a
different, more invasive transform (call boundaries inside the hot path, SSA-state marshaling across the
cut, indirect dispatch), scoped separately.

The `compile_module_split` capability + its differential test are correct and retained as the reproducible
evidence for the negative finding; they are not wired into `compile_jit`.

## Result 6 — the split does NOT engage on the real QuickJS card (Slice 2c, #1120)

Slices 1–2b built + differential-tested the intra-function split emitter (env-scratch trampolining:
inter-group edges `return_call` a uniform-signature group function, marshaling the target block's
live-in values through the fixed **64-slot** cross-tier scratch, `XCALL_MAX_SLOTS`). Slice 2c wired an
opt-in toggle into the warm+JIT path and measured the real cards with `browser/warm-jit-split-test.mjs`
(Node/V8, the shipping engine FFI + `primeWarmJit`/`runWarmJit`, over the committed snapshots).

**The toggle is a no-op on QuickJS.** `pick_split_target` correctly picks `JS_CallInternal` (func 27:
est 5.4 MiB, 1838 blocks, K=15), but `emit_module`'s `plan_fn_split` then **declines** it, so the emit is
byte-identical (14,313,159 bytes split-on == split-off). Reason: the split marshals each inter-group entry
block's params through the 64-slot scratch, and `JS_CallInternal`'s SSA liveness is enormous —

| | inter-group entry blocks | over the 64-slot cap | max param slots |
|---|---|---|---|
| QuickJS `JS_CallInternal` (K=15) | 246 | **241** | **216** |
| — all 1838 blocks | — | 1815 (98.7%) | 227 |
| Lua hot fn (func 372, K=3) | 69 | **0** | 35 (all blocks max 43) |

High boundary-liveness is **pervasive** in `JS_CallInternal` (1815/1838 blocks > 64 slots), so no choice of
cut points can get under the cap — the 64-slot env-scratch ABI simply cannot carry ~216 live values across a
group edge. `plan_fn_split` declines the whole function ⇒ monolithic fallback ⇒ no tier-up change.

**And it cannot be measured on Lua either.** Lua's hot function *is* outlinable by liveness (max 43 < 64;
`compile_module_reactor_split` emit grows +230,777 bytes, K=3), but Lua's warm+JIT tier itself **declines**:
`eval_run` reaches `lua_pcall`'s setjmp, so `compile_jit` routes it to `InterpDriven` (#1081) and
`temen_warm_jit_open` returns `STATUS_UNSUPPORTED`. Lua runs on warm-**coop**, never on the split emit.

The earlier "3.95× Run-1 speedup" from an in-process A/B was a **run-order artifact** (whichever variant ran
first paid V8/process cold-start); the emit being byte-identical proves the split contributed nothing. Use
`--variant=off`/`--variant=on` in separate processes to avoid the confound.

**Conclusion.** The split mechanism is correct and tested, but delivers no first-Run win on any current
warm+JIT card: QuickJS (the only WasmDriven warm card) exceeds the 64-slot marshalling ABI, and Lua (the
only other) declines to InterpDriven. To make outlining help QuickJS, the group-edge marshalling must carry
~256+ live values — a **wider group-edge scratch/spill region** (independent of the 64-slot cross-tier call
scratch), or a region-based partition that cuts at low-liveness boundaries. That is a follow-up ABI change,
not part of Slice 2c. The toggle (default off), the reactor/local split entries, and this harness are
retained as the reproducible evidence and the ready hook for that follow-up.

## Result 7 — outlining JS_CallInternal is CORRECT but a net steady-state loss (Slice 3, #1120)

Slice 3 gave inter-group edges a dedicated 256-slot scratch region (separate from the 64-slot cross-tier
call scratch), so `plan_fn_split` no longer declines `JS_CallInternal`'s high-liveness boundaries. It now
outlines. A/B on the real QuickJS card (Node/V8, `warm-jit-split-test.mjs --variant=off/on`, fresh process
each, fib(28), 6 runs; the shipping engine over `qjs_snapshot.temen`):

| | emit bytes | prime | Run 1 | steady (min last 3) | Run-1/steady | interp-parity |
|---|---|---|---|---|---|---|
| split-off (monolithic) | 14,313,162 | 3153 ms | 9984 ms | **2331 ms** | 4.28× | OK |
| split-on (K≈15) | 15,420,505 | 715 ms | 10268 ms | **8882 ms** | 1.16× | OK |

Two real effects, one good and one disqualifying:

- **Correct + the tier-up penalty is gone (good).** The +1.1 MB emit confirms the split engaged; the
  interpreter-oracle parity confirms the widened marshalling ABI is correct on a 5.4 MB function cut into
  ~15 groups with 200+ live values per group edge. And Run-1/steady drops from 4.28× to 1.16× — the split
  *does* reach its steady state on Run 1 (TurboFan finishes each small group immediately), while the
  monolith spends run 1 on Liftoff.
- **Steady state is 3.8× SLOWER (disqualifying).** The split's steady state (8882 ms) is 3.8× the monolith's
  (2331 ms), so it is slower on *every* run — the Run-1 tier-up win is dwarfed. `JS_CallInternal` is a giant
  hot **dispatch loop**; the contiguous block-index cuts (`i*k/nb`) fall *across* that loop, so every
  bytecode dispatch now crosses a group boundary and pays the marshal-to-memory + `return_call` + read-back
  cost — with up to 227 slots copied through memory on the heavy edges. The property that makes the function
  tier up slowly (one big hot loop) is exactly what makes it pathological to outline this way.

**Conclusion.** Block-group outlining is the wrong lever for `JS_CallInternal`. The mechanism is correct and
retained (default off — shipping cards unaffected, byte-identical), and it may still help a function whose
hot path does not straddle the cuts. But a *contiguous, liveness-and-hotness-blind* cut of a hot dispatch
loop trades a one-time tier-up latency for a permanent per-iteration trampoline tax. The only outlining that
could help QuickJS is a **hotness-aware partition** that outlines the *cold* blocks (rare opcodes, error
paths) while keeping the hot dispatch core monolithic — shrinking the function TurboFan must compile without
putting marshalling on the hot path. That needs per-block hotness (static heuristic or profile) and is a
distinct, larger design. Otherwise the first-Run lever for QuickJS lies elsewhere (shrinking the emitted
hot function, or making Liftoff→TurboFan of one big function faster), not in splitting it.
