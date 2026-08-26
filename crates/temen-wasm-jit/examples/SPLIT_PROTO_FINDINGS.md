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

Sweeping `f0`'s size (single config, fresh process each):

| `f0` emitted size | tier-up run |
|-------------------|-------------|
| 0.44 MB           | run 1       |
| 0.98 MB           | run 3       |
| 1.9 MB            | run 5       |
| ~2.7 MB           | run 11      |

Tier-up run grows (super-linearly) with the hot **function's** size — reproducing the real card's "run 6+
/ run 10" shape from a *single* large function, with no help from module structure. A trivial hot loop
(tiny `f0`) tiers up at run 0 regardless.

## Result 3 — cross-module call cost

With a large hot function the per-iteration helper call is a negligible fraction: `split_xmod`
(cross-module `call_indirect`) vs `split_good` (intra-module direct call) differ by ~0% at steady state.
The cross-module cost is only visible for a *trivial* hot loop where the call dominates (≈ +21% there) —
but that regime tiers up at run 0 anyway, so it never matters in practice. So cross-module dispatch is
cheap where it would be used; it is simply pointed at the wrong problem.

## Conclusion / recommended pivot

Module-splitting is the wrong lever for first-Run latency. The bottleneck is a single giant hot function
(`JS_CallInternal`) whose TurboFan compile does not finish until ~run 10, and you cannot split a function
across modules. **The lever that would actually move first-Run latency is reducing hot-function size** —
i.e. *function outlining*: breaking the giant dispatch function into smaller functions so each reaches
TurboFan far sooner (the size→tier-up curve above predicts run ~10 → run ~1–3). That is a different, more
invasive transform (it introduces call boundaries inside the hot path, whose cost must be measured), and
should be scoped separately.

The `compile_module_split` capability + its differential test are correct and retained as the reproducible
evidence for this finding; they are not wired into `compile_jit`.
