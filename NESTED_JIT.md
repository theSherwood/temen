# NESTED_JIT.md — widening §14 VM-in-VM coverage on the wasm-JIT tier

Decomposition of the work to make more of a **nesting parent** emit on the wasm-JIT tier
(`crates/svm-wasm-jit`), rather than falling back to the interpreter. Companion to `DESIGN.md`
§14 (composition & nesting), §22 (guest-driven `Jit`), §4/§13 (memory model), and `BROWSER.md`
"wasm-JIT tier".

## Framing (what is and isn't the problem)

VM-in-VM **already runs correctly on every tier today.** A §14 unit fed to the general front
door (`compile_jit`) classifies its `cap.call` functions as out-of-subset and runs
interpreter-driven; the interpreter services `instantiate`/`join` via `VcpuStop::Instantiate`
exactly as always. The dedicated nested front door (`compile_module_nested`) is a **performance**
path: it lowers the parent's `instantiate`/`join`/window-carve/thread ops to host bounces
(`env.instantiate` / `env.join` / `env.thread_spawn` / …) so the parent runs on emitted wasm.

So the goal is **acceleration coverage**, not correctness: which nesting parents get to run on
wasm instead of the interpreter. Two gaps hard-`Err` the nested front door today
(`compile_module_nested_with_eligibility`, `lib.rs:1318–1342`):

1. A reachable function that is not nested-emittable **and** has a `v128`/float in its
   signature → `Err` (can't be marshalled through the i64-slot cross-tier ABI).
2. The **entry** (func 0) is not itself nested-emittable — e.g. it uses a fiber (`cont.*`) —
   → `Err("nested entry outside the subset")`.

The browser works around (2) with an *external* fallback (`browser/src/lib.rs:1092–1116`): if
the nested emit `Err`s, it re-compiles with `compile_jit(Shape::Threaded)` and runs
interpreter-driven with tier-up. That proves the shape; this work folds it into the library and
widens the ABI so fewer parents need the fallback at all.

Tracks in increasing difficulty. **Track 3 is a genuine design question and is out of scope
here** — recorded at the end pending an owner decision.

---

## Track 1 — widen the cross-tier `call_interp` ABI to scalar floats

**Problem.** A non-emittable function is kept as a cross-tier interpreter leaf **iff** its
signature is all-integer (`int_sig`, `lib.rs:1131`), because `env.call_interp` marshals each
arg/result as one 8-byte slot at `env + ENV_SCRATCH_OFF + i*8`, widening `i32`→`i64` and
narrowing back. A `f32`/`f64`/`v128` in the signature can't ride that path, so the nested front
door hard-`Err`s (gap 1).

**Change.** Extend the slot encoding to **scalar floats**, which fit the existing 8-byte slot:

| type | store into slot | read back from slot |
|------|-----------------|---------------------|
| `i32` | `i64.extend_i32_u` (unchanged) | `i32.wrap_i64` (unchanged) |
| `i64` | as-is (unchanged) | as-is (unchanged) |
| `f32` | `f32.store` (low 4 bytes; high 4 ignored) | `f32.load` |
| `f64` | `f64.store` (all 8 bytes) | `f64.load` |

`int_sig` is replaced by a `marshallable_sig` predicate (`i32`/`i64`/`f32`/`f64`). The same
predicate gates the reactor / tier-up cross-tier paths, so they widen too — no behavior change
for existing all-integer modules (the `i32`/`i64` encodings are byte-identical).

**`v128` is explicitly deferred** (a separate, larger sub-increment): a `v128` needs **two**
8-byte slots, which changes the slot stride, `ENV_CELL_BYTES`, and every servicer's slot
arithmetic. Until then a `v128`-signature non-emittable function keeps a clear
`Unsupported("v128 signature not marshallable cross-tier")`.

**Surface (both tiers must agree — one encoding, several readers):**
- Emitter (`svm-wasm-jit/src/lib.rs`): the two direct-`Call` marshal sites (`~3300`, `~3727`)
  and `emit_trampoline` (`~2266`). Encode/decode per the table above.
- The three cdylib servicers (`browser/src/lib.rs`): `svm_onramp_jit_call_interp`,
  `svm_onramp_jit_run_call_interp`, `svm_wasmjit_call_interp` — today each does
  `_ => Value::I64(slot)` on args and `_ => STATUS_TRAP` on results; teach them `F32`/`F64`.
- Test harnesses that stand in for the servicer (`tierup.rs`, `cross_tier.rs`, `differential.rs`).

**Invariant check (INVARIANTS.md #2).** The scratch lives in the **`env` cell**, not the guest
window; float marshalling reuses the *same* already-emitted `env`-pointer store/load as the
integer path (`i64.store`→`f64.store` is the only shape change). No new emitted-code or
window-access surface, so the masking hinge is untouched. This is the JIT-internal cross-tier
transport, distinct from the service/dispatch reply ABI that INVARIANTS.md §"single-slot scalar
replies" protects.

**Test.** A differential test with an in-subset caller of a **non-subset, float-signature** leaf
(e.g. a helper doing scalar `f64.fma`, which has no core-wasm opcode) — the emitted caller must
match the interpreter oracle across the float round-trip.

---

## Track 2 — fibers in a nested unit (interpreter-driven fallback)

**Problem.** `cont.new`/`cont.resume`/`suspend` switch stacks; a wasm frame can't unwind for a
stack switch, so a fiber-using function can't be emitted on the wasm tier. When the **entry**
uses a fiber, the nested front door hard-`Err`s (gap 2). DESIGN.md §22 (renegotiated 2026-07-30)
already blesses a submitted `Jit` unit hosting fibers — it runs its own scheduler to completion
on the caller's thread and never parks across the synchronous `cap.call` — so the semantics are
settled; only the wasm-tier plumbing is missing.

**Change.** A new front door `compile_nested(m, shared_memory) -> Artifact` gives the nested path the
**two-mode shape** `compile_jit` already uses for the top level. The gate is **fibers specifically**,
not all of §12 — threads/futex already emit as host bounces and must stay on the fast path:

- **No reachable fiber** (`!reachable_fibers`, using `Func::uses_fibers`) → the existing nested emit
  (host-bounced `instantiate`/`join`/`thread`/`futex`), host calls `f0` — `DriveMode::WasmDriven`.
- **A reachable fiber** → a **`nested_caps`-aware tier-up** (`compile_module_tierup_caps`, the twin of
  `compile_module_tierup`): the interpreter owns the top frame (running the fibers and servicing
  `instantiate`/`join` natively), and hot in-subset compute — including instantiate/join/thread
  functions — still tiers up with its bounces intact — `DriveMode::InterpDriven`.

Both modes emit the **same** nested import set, so the artifact carries one uniform import layout
regardless of mode. `compile_module_nested_with_eligibility` additionally now **fails closed on a
reachable fiber** (a fiber function is no longer silently made a cross-tier leaf — a suspend across
the synchronous `env.call_interp` seam couldn't unwind the wasm frame), so `compile_nested` never
routes a fiber through the emit path and any direct caller falls back safely.

**Soundness.** Fibers are never emitted — they only ever run on the interpreter, which owns the frame
in the fallback (INVARIANTS.md #2 untouched: no new emitted-code or window-access surface). A fiber
that suspends past a synchronous boundary fails closed at runtime (§22: an invoked unit that parks
across the seam gets an inert `CapFault`); this work adds no new way to suspend across a wasm frame.
Threads/futex keep their host-bounce emit lowering — only fibers force the interpreter driver.

**Tests** (`tests/nested_vm.rs`): a pure instantiator and a threads/futex unit are both `WasmDriven`
(threads stay on the fast path); a fiber-entry unit is `InterpDriven` (not `Err`) with the fiber
entry unemitted and the pure helper + fiber body tiered up; and the interp-driven tier-up wasm is
instantiated under the full eight-import nested linker and its emitted `f1` runs at interpreter parity
(pinning the `nested_caps` import layout + emitted-function base offset).

**Deferred (not this PR):** rewiring the browser's hand-rolled fallback (`browser/src/lib.rs`
`1092–1116`) to call `compile_nested`. It's a clean consolidation but changes that path's emitted
import layout to the nested set (the host must then always provide `env.instantiate`/`join`/`thread_*`
even for a plain tier-up), so it wants its own change with the Worker-side import wiring.

---

## Track 3 — `map` / `unmap` / `protect` + intra-window page enforcement  [DEFERRED — owner decision]

Not in scope for this doc; recorded so the boundary is explicit.

`AddressSpace` ops 0/1/2 (`map`/`unmap`/`protect`) stay out-of-subset (`lib.rs:784–788`). The
whole confinement model on the native tier is **mask + `PROT_NONE` guard region**: unmapped or
wrongly-protected pages fault via the host MMU (DESIGN.md §4:1075–1078, §13). **Wasm linear
memory has no intra-memory page protection** — there is no `PROT_NONE` sub-region inside a wasm
memory — so an emitted load sails straight through a page the interpreter would trap on.

Two sub-cases differ in severity:
- **`unmap`/`protect`-RO then access** is a genuine *semantic divergence* (interp traps, JIT
  returns stale/zero) — the differential oracle flags it. Must fail closed.
- **D40 read-only const segments** (`protect` RO at instantiation, backing const globals / string
  literals) is only a *defense-in-depth* gap: a guest corrupting its own const data can't escape,
  it just loses the §5 self-corruption detect-and-kill. The wasm tier runs these guests
  *correctly*, only without the hardening.

Options (to discuss): (a) fail-closed — a unit using `unmap`/`protect` runs interpreter-driven
(correct, cheap, gives up accel); (b) per-access software page-check in emitted code (kills §14's
zero-overhead thesis — almost certainly not worth it); (c) scope it out — declare the wasm tier
does not enforce intra-window page protection and such units run interpreter-driven (a deliberate
renegotiation of the §4/§13 confinement invariant *on that tier*). The near-term recommendation
is (a) + (c); (b) is not recommended.
