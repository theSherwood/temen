# Simplification & Refactoring Audit — 2026-08-13

A whole-tree audit against one question: **is the system as lean, simple, correct, fast,
and flexible as we can make it without compromising the safety invariants?** Backwards
compatibility was explicitly out of scope — only the final form matters. Every
recommendation was checked against `INVARIANTS.md`; none touches the confinement regime's
semantics, and several *strengthen* it.

This is a point-in-time report, not a tracker. Per `ISSUE_TRACKING.md`, items the owner
accepts should become GitHub issues under the appropriate workstream epics; this file can
then be deleted or trimmed to a record.

**Method.** Nine parallel deep-audits over the subsystem partitions (IR/encoding, the
verify/mask/mem hinge, the tree-walk oracle, the bytecode interpreter, both JITs, the
runtime/durability layer, the compile pipeline & frontends, the consumer surface, and
cross-cutting engines×ops costs), followed by hand spot-verification of the headline
claims (all confirmed; file:line refs below are current as of `9b1c34a`).

---

## Verdict in one paragraph

The core design is delivering what it promises: the verifier really is a single linear
forward pass; the masking hinge (`svm-mask`) is a model TCB unit; invariant 9's "one
shared veto predicate" genuinely holds (`svm_ir::bounds`, `serve_qualifies` — one
definition each); fuel-safepoint parity claims check out against the code; dependency
hygiene of the escape-TCB crates holds; and the op set has far less fat than a 465-mnemonic
matrix suggests (only ~9 mnemonics are cuttable; everything else traced to a live
consumer). The debt is not architectural — it is **accreted duplication and monolith
files**: the same check or table hand-copied 2–25×, three files over 12k lines each with
no module structure, a decoy front-door crate, and a long tail of "next wire rev" cuts
that the no-back-compat ruling now makes free. One behavioral discrepancy was found and
verified (§7, routing fold). Nothing found violates an invariant outright.

---

## 1. Highest-value items (ranked)

The ten changes with the best simplification-per-risk, drawn from the sections below:

1. **Single-site `ubs` maintenance in the Cranelift JIT** (§2.1) — replaces 24 hand-synced
   copies of the most escape-critical bookkeeping in the tree. Small, and it strengthens
   invariant 2.
2. **Merge `svm-mem`'s `Shared`/`Mapped` twins** (§2.2) — deletes ~200 duplicated `unsafe`
   lines from the crate whose whole purpose is isolating auditable `unsafe`.
3. **Verifier: one definition of result arity/types + one call-arg check** (§2.3) —
   removes a documented "keep in sync" hazard and two `unreachable!()` panics from the TCB.
4. **Verify/pin the reactor routing-fold discrepancy** (§7) — `run_capture_on` omits a
   clause its comment claims to mirror; either a real gap or an undocumented narrowing.
5. **The final-form wire rev** (§3) — cut ~9 ops, the v9 compat window, the dead `align`
   byte, `CallSym`'s dead operand, and renumber the opcode fragments, in one coordinated rev.
6. **Lift capability iface ids into `svm-ir`** (§2.5) — today a renumbered iface would
   silently disable the wasm-JIT's page-ops gate: an *unsound emit*, currently prevented
   only by nothing changing.
7. **One errno table** (§4.1) — ~64 definitions across 8 files with two sign conventions,
   exactly the invariant-5 bug class.
8. **Split the three monolith files** (§5) — `svm-interp/lib.rs` (24.5k), `svm-run/lib.rs`
   (6.3k), `svm-llvm/lib.rs` (22.2k) into modules along seams that already exist. Pure code
   motion, big auditability win.
9. **Extract the ×10 carve/entry-admission check and the ×2 spawn-record parse** (§4.2,
   §4.3) — security-relevant admission logic with ten hand-synced copies.
10. **Make `svm` (or `svm-run`) the one obvious front door and stop `run_text` swallowing
    traps** (§6.1) — the consumer-facing fix with the best effort/benefit ratio.

---

## 2. The safety hinge — simplifications that strengthen it

These are TCB or TCB-adjacent edits; each is gated by the existing differential/fuzz
matrix, and each *reduces* what an auditor must hold in their head.

### 2.1 Cranelift JIT: 24 scattered `ubs.resize` calls maintain the mask-elision map
`crates/svm-jit/src/lib.rs:6104-6817`. The upper-bound (`ubs`) vector must stay in
lockstep with the SSA value vector; the file's own comment (:6106) says a misalignment
"could mis-elide" — i.e. **drop a confinement check**. Today that lockstep is maintained
by ~24 hand-placed `ubs.resize(vals.len(), UB_TOP)` calls, one per early-`continue` arm.
The wasm-JIT already has the correct shape: one maintenance point per loop iteration
(`crates/svm-wasm-jit/src/lib.rs:4150-4159`). Fix: one resize at the top of each
iteration; delete the 24 copies. A mistake in this refactor fails *toward* `UB_TOP`
(check emitted, never elided). Effort S, risk low.

### 2.2 svm-mem: `Shared` and `Mapped` are ~200 byte-identical lines of `unsafe`
`crates/svm-mem/src/lib.rs:315-494` vs `:500-760`. Every accessor body is identical; the
only real difference is mmap ownership. The crate currently *gates drift with a fuzz
test* (:314) — strictly worse than making drift impossible. Fix: `Mapped` becomes a thin
owner delegating to `Shared` (constructor + `Drop` only). One body to audit under
miri/tsan instead of two. Effort M, risk low.

### 2.3 svm-verify: three parallel encodings of result arity; six copies of the call check
`crates/svm-verify/src/lib.rs`. (a) `verify_func`'s inline appends (:397-842),
(b) `block_value_types`' mirror match (:864-912 — with an explicit "**keep the two in
sync**" comment at :854-856), and (c) `Inst::result_count`, which the
`GcRootsMaskUnsafe` host-pointer-leak check depends on (:922-935) — drift between (a) and
(c) can silently skip the static mask rejection. Additionally the arg-count/zip/extend
call check is hand-copied six times (:413-751) and two `unreachable!()` arms sit inside a
component whose fuzzed contract is "never panics" (:1059-1078). Fix: one
`inst_result_types` used by both walks and cross-checked with `result_count`; one
`check_args` helper mirroring the existing `check_edge`; flatten the 26-arm if-let chain
to a single `match`. The pass then *reads* as the single linear pass it already is.
Effort M, risk low (gated by `spec_verify`/`decode_verify` differential fuzz).

### 2.4 One fuzzed home for the span-confinement arithmetic
The bulk-span predicate exists three ways: the JIT's subtraction formula
(`svm-jit/src/lib.rs:8796-8807`), the interpreter's checked-add form
(`svm-interp/src/lib.rs:21808-21814`), and — worst — a **hand-transcribed copy inside the
fuzz target itself** (`fuzz/fuzz_targets/mask.rs:69-79`, "kept in sync with it"). If
`confine_span` changes and the transcription doesn't, the fuzzer green-lights the wrong
formula. Fix: add the one or two scalar reference functions to `svm-mask`, fuzz them
against the u128 oracle there, and have the interpreter call them; the JIT comment then
points at the fuzzed reference it must emit. Keep the API widening minimal (invariant 1).
Effort M, risk low.

### 2.5 Lift capability iface ids to `svm-ir`
Both JITs' veto predicates hardcode raw iface numbers (`type_id == 4/5/6`:
`svm-wasm-jit/src/lib.rs:770-806,1100-1156`; `svm-jit/src/lib.rs:6043-6058,6202`) whose
canonical definitions live in `svm_interp::cap_id` (`svm-interp/src/lib.rs:14636-14657`)
— which neither JIT can see. A renumbering would make `module_uses_page_ops` quietly stop
gating: an unsound emit. Fix: constants next to `CAP_SELF_TYPE_ID` in `svm-ir`, used by
all three. Effort S, risk minimal.

### 2.6 Deduplicate the Cranelift module setup
`svm-jit/src/lib.rs:2638-2670` vs `:4929-4944`: `compile` and `compile_child` each
hand-copy the ISA flag block — including `enable_probestack=false`, a load-bearing
soundness pin (:2648, :8016) — plus the arena provider and declare loop. Two copies that
can drift independently on a soundness-critical configuration. Fix: one
`new_jit_module()`. Effort S, risk minimal.

### 2.7 Also worth doing here
- **svm-mask legacy API**: delete `mask()` (zero callers; actively misleading — suggests a
  masking regime that no longer exists) and `size()` (one caller, trivially `reserved()`)
  — `crates/svm-mask/src/lib.rs:165-178`.
- **svm-mask header drift**: the crate still describes the superseded D38 AND-clamp as the
  production Spectre mechanism (`:31-38`); the shipped scalar lowering is the D63
  branchless `select_spectre_guard` redirect (`svm-jit:8601-8717`). The hinge's spec
  should match the code an auditor diffs it against. Doc-only.
- **wasm-JIT confinement fuzz gap**: the third confinement lowering
  (`svm-wasm-jit:3057-3252`) is tested with hand-written kernels but never fuzzed; the
  Cranelift lowering gets the generative irgen escape-oracle. Invariant 2 calls masking
  "the fuzzed hinge" — add an irgen-driven interp-vs-wasmi target. Effort M-L, test-only.

---

## 3. Cutting ops and the final-form wire rev

With back-compat off the table, the accumulated "next wire rev" items should land as
**one coordinated rev**. Verified cut list:

| Cut | Where | Why it's free |
|---|---|---|
| `CapSelfCount/Get/Resolve/Label/Attest` (5 variants, opcodes 0x7A/0x7B/0x7E/0x7F/0xBE) | `svm-ir:1869-1934` | Self-described sugar: every backend already lowers them to `cap.call CAP_SELF_TYPE_ID op 0-4` (`svm-jit:6355-6460`). Keep the mnemonics as **svm-text sugar** over the `CapCall` form — frontends keep emitting the same text. −5 ops × ~15 files ≈ 75 match arms. Keep `CapSelfTypeId`/`CapSelfCovers` (they carry a type-section index). |
| `PtrAdd`/`PtrCast` (opcodes 0x76-0x78) | `svm-ir:1720-1729` | No frontend emits them (verified: not svm-llvm, svm-wasm, chibicc, svm-leng). Kept for a CHERI backend that doesn't exist — invariant 1's textbook case. `SPEC.md:8-12` records the JIT shipped without lowerings for them and nobody noticed. A future CHERI port is a wire rev anyway. |
| `SimdWidthBytes` (simd sub-op 0x0E) | `svm-ir:2355-2359` | Constant 16 on every backend, no consumer, exists for hypothetical v256/v512. |
| `ContResumeBlock` (opcode 0xBF) → flag on `ContResume` | `svm-ir:1985-1999` | Documented as an advisory hint with identical semantics; every fast backend aliases it; no frontend emits it. |
| `align` byte on `Load`/`Store`/`V128Load`/`V128Store` | `svm-ir:1615-1635,2123-2135` | **No backend reads it** (JIT binds `..`; parity catalog says it changes nothing). Deletes the memarg text machinery and two verifier rules with it. |
| `Ordering` lattice → fence-only (or none) | `svm-ir:1469-1514` | Every backend executes everything seq-cst (documented :1464-1468); frontends emit non-SeqCst only on `AtomicFence`. Re-adding weakness later is one wire byte. Needs owner sign-off — the lattice is documented as deliberate. |
| v9 compatibility window | `svm-encode:244-251,1823-1831` | Kept only for 17 committed playground `.svmb` assets — regenerate them, restore the exact-version rule (the format's stated norm). |
| `CallSym.handle` dead operand | `svm-ir:1779-1787` | The in-source CONSOLIDATION §7 note already schedules its removal. |
| Opcode renumbering | `svm-encode:66-241` | `IMPORT_ATTACH` at 0x63 inside the conversions band, v7 ops squeezed into 0x0A-0x0E, 0xBE/0xBF parked — renumber into coherent family bands while the byte map is open. |

**Bigger, separately-staged representation change:** intern call signatures into the
type section. `CallIndirect`/`CapCall`/`CallImport`/`CallImportDyn`/`CallSym` each carry
a full inline `FuncType` (two heap `Vec`s, sizing `Inst` at ~96-104 bytes), a
self-described "Phase-1 simplification" from before `Module::types` existed
(`svm-ir:1738-1743` vs `:3110`). Referencing `types[t]` shrinks `Inst` toward ~40 bytes
(the data-oriented directive), stops repeating signatures on the wire, and *deletes* the
verifier's inline-vs-type-section cross-check rule. Effort L (touches every backend's
call lowering, mechanically), risk medium, fully pinned by the differential + spec_encode
suites.

**Explicitly not cuttable** (all traced to live consumers): the total SIMD lane-op
families (svm-llvm lowers `llvm.smin/…` generically over any shape; totality keeps the
verifier free of per-shape lane rules), `Eqz`, `FToITrap`/`FToISat`, narrow load/stores,
`Fma`/`VFma`, the shape-fixed singletons, and the 68-op bytecode set (no dead opcodes;
its `Op::Eval` delegation for the SIMD tail is the right anti-duplication shape).

---

## 4. One-definition consolidations (drift killers)

Each of these is the same logic hand-copied N times, where drift is either a security
hazard or a guest-visible inconsistency.

1. **Errno table** — ~64 definitions of ~20 errnos across 8 files, in **two sign
   conventions**: positive in `svm-fs:85-92`/`svm-exec:40-44`, negative in
   `svm-posix:224-233`/`svm-interp:14785-14800`, plus **ten function-local
   `const EINVAL: i64 = -22;`** in `svm-run` alone. Fix: one `pub mod errno` in `svm-ir`
   (dep-free; everything already depends on it), one sign convention. Effort S-M.
2. **§14 carve/entry admission check ×10** — the identical entry-signature + carve
   geometry block (`size_log2` range, pow2, alignment, `off+size ≤ isize`) is pasted 8×
   in `bytecode.rs` (:3383, :3468, :6018, :6145, :9724, :9949, :10851, :10990) and 2× in
   `lib.rs` (:10275, :10729). One `carve_fits` + `child_entry_ok` pair ends a world where
   a bound tweak needs ten coordinated edits. Effort M.
3. **Op-17 spawn-record layout ×2** — the 56-byte record is hand-decoded field-by-field
   in both `bytecode.rs:12449-12477` and `lib.rs:10078-10110`, with one already-documented
   error-order divergence. One `SpawnRec::parse` shared by both (and the Cranelift thunk).
   Effort S.
4. **Powerbox glue ×3 hosts** — the cap-name→`(type_id,op)` resolver, the powerbox grant
   sequence, the argv-blob layout, and the entry-shape predicate are each duplicated
   across `svm-run` (:3404, :4080, :3674), `browser/src/lib.rs` (:2476, :2542, :2961),
   and `svm-dap/src/backend.rs` (:44, :66) — self-described "twins". This is
   guest-ABI-shaped data; drift means the same guest behaves differently per host. The
   tree already hoisted `POWERBOX_CAP_NAMES` to `svm-ir` for exactly this reason — finish
   the job (keep the moved code std-free for the wasm build). Effort M.
5. **Backend op-classification from `effects()`** — `svm-wasm-jit` keeps three hand-listed
   memory-op subsets (:929, :1165, :1205-1225) that `Inst::effects()` (exhaustive,
   no-wildcard, already the optimizer's oracle) can answer. Also move
   `each_operand`/`map_operands` from `svm-opt` (:2205/:2473) into `svm-ir` beside
   `effects()` — svm-peval reaching into the optimizer for an IR-shape utility is a
   layering wart, and `svm-durable:285-333` maintains a partial third copy. Add
   `Inst::for_each_operand_mut` once. Effort M.
6. **`module_uses_*` via `Func::uses_*`** — `svm-jit:5989-6058` inline-copies predicates
   `svm-ir:2661-2762` already exports as "the single source of truth for backends that
   must agree". Effort S.
7. **`take_spawn_budget` over `Host::budget_for_spawn`** — `bytecode.rs:1085-1104`
   re-implements the drain-only-at-commit discipline that `lib.rs:19240-19256` exposes
   *specifically* for other tiers. Effort S.
8. **Waiter-store scans ×9** — `teardown_domain`/`teardown_run`/pipe-wakes
   (`svm-interp:5322-5530`, :4320-4352) repeat the same hand-rolled identity-scan loop
   over nine waiter stores, with the `Waiter → domain key` match duplicated 6×. One
   `Waiter::domain_key()` + one retain-style `drain_members` helper: ~230→~100 lines, and
   the highest-cyclomatic lifecycle block becomes linear. Preserve the drop-vs-reap
   subtlety for fiber waiters. Effort M, risk low-medium (loom/tsan gate).
9. **Durable layout constants ×2** — ~12 `STATE_*`/`SHADOW_*`/`ARM_*` constants defined
   in both `svm-durable:110-159` and `svm-interp:7156-7234` with "Must equal…" comments;
   the pin test covers only 7 of them. Both crates already depend on `svm-ir` — one
   `durable_abi` module there. Effort S.
10. **`SuspendKind` operand tables ×3** — `svm-durable:973-1275`: liveness marking,
    re-issue arg mapping, and the state-flip sequence are three hand-synced per-kind
    sites; a missed marking is silent under-spill (corrupted thaw). One
    `SuspendKind::operands()` makes that unrepresentable. Effort S-M.
11. **Snapshot codecs ×2** — the serve-trio and handle-record shapes each have two
    hand-copied encode/decode pairs (`svm-snapshot:447-519` vs :742-976); a format bump
    currently touches 4 sites. Effort S.
12. **svm-posix's second memfs** — ~400 lines re-implementing `svm-fs::MemFsState`, and
    already **drifted** (path normalization, dir-prefix rules, stat layouts differ between
    the fs cap and the personality a guest may hold simultaneously). Extract the store as
    the shared backing; svm-posix stays a thin ABI adapter. The normalization fix is a
    behavior change wanting the owner's nod (it is also the correct POSIX behavior).
    Effort M-L, risk medium.
13. **Debug-driver core ×2** — `ScheduledDebugRun::tick` vs `::drive`
    (`bytecode.rs:6904-7218`) duplicate the outcome-servicing match; a new scheduler op
    serviced in one and not the other desyncs reverse-`seek` from live runs. Effort S.

---

## 5. Monolith files → module splits (pure code motion)

No behavior change in any of these; consumers import narrow root surfaces, so the splits
are invisible outside the crate.

- **`svm-interp/src/lib.rs` (24,487 lines)** — interpreter semantics proper is ~7k;
  the rest is separable machinery: debug/time-travel (~1.5k), DPOR explorer (~770),
  M:N scheduler (~2.5k), powerbox types + the 216-method `Host` impl (~6.7k), memory
  model (~1.45k), SIMD helpers. Split to `debug.rs`, `explore.rs`, `sched.rs`, `host.rs`,
  `mem.rs`, `simd.rs`; a confinement reviewer then reads `mem.rs`, not a 24.5k grep.
  Within it, two targeted extractions:
  - **`run_inner` (4,445 lines)**: the instruction match is healthy; the fat is 13
    `CapCall` arms embedding whole subsystems inline (INSTANTIATOR ~856 lines, exec ~331,
    svc ~222) and four in-loop `macro_rules!` (one 373 lines) that exist only because
    ~30 destructured locals prevent helper fns. An `EvalCtx` struct + methods returning a
    small `Flow` enum takes it toward ~1,500 lines of visible dispatch. Do arm-by-arm
    behind the differential matrix. Effort L, staged.
  - **`Host` (76 fields)**: `fork_powerbox`/`regrant_into_child`/`bind_child_manifest`
    enumerate fields one-by-one — a missed field is an authority leak (invariant 3).
    Group into the sub-structs that half-exist, each with its own `fork()`; the
    grant-graph audit becomes local per subsystem. Effort M.
- **`svm-run/src/lib.rs` (6,273 lines, one section divider)** — ~10 products in one file
  (guest-JIT sessions, per-OS window backings, powerbox runners, `Instance`, reactor
  `Session`). Split to `powerbox.rs`, `instance.rs`, `reactor.rs`, `window.rs`,
  `guest_jit.rs`, `specialize.rs`. Effort M, risk very low.
- **`svm-llvm/src/lib.rs` (22,249 lines)** — ~42% is a hand-built runtime library:
  42 `synth_*` functions, a 3,415-line arbitrary-precision dtoa family, 1,762 lines of
  printf lowering, all as imperative `Bdr::push(Inst::…)` sequences (:3662-13083).
  Tier (a): move to `runtime.rs`/`printf.rs` (mechanical, −9k from the file).
  Tier (b): rewrite the pure-computation helpers as a committed `.ll` fixture the crate
  translates with its own parser and links via `svm_ir::link` — replacing thousands of
  lines of write-only builder code with source testable against native libc. Then split
  the remainder (`layout/i128/vector/eh` modules); `translate_inst` alone spans 1,300+
  lines. Effort (a) S, (b) M-L.
- **`svm-jit`**: split `lower_block` (1,729 lines) along its family seams (`lower_simd`,
  `lower_call_like`), keeping every masking call site adjacent to `mask_addr`. Same, less
  urgently, for `emit_block_body` (987 lines) in the wasm-JIT. Effort M.

---

## 6. Consumer surface

1. **The `svm` umbrella crate is a decoy front door** — its API has one external caller
   in the tree; every real consumer imports `svm_run` (whose own header says it is "the
   embedding runtime"). Worse, `svm::run_text` **silently swallows traps**
   (`crates/svm/src/lib.rs:62`, `unwrap_or_default()`). Fix: make `run_text` surface the
   trap (or delete it); give `svm` a feature-gated `pub use svm_run` so there is one
   documented import while the default feature set stays the dep-free auditable core;
   have `svm-run` re-export `parse_module`/`decode_module`. The minimal embedder is
   already genuinely lean (2 crates, ~7 types, 4 calls) — it just isn't findable.
2. **Collapse the telescoping runner families onto option structs**:
   - `svm-jit`'s 19-wrapper `compile_and_run*` family (:882-1670, ~789 lines) plus
     12-positional-arg `CompiledModule::compile` (34 call sites of unlabeled
     `None, None, None, None`) → one `RunOpts`. The family grew four names for
     `durable × mv × interruptible` alone. Keep thin named shims only where differential
     harnesses rely on name symmetry with the bytecode twins.
   - `svm-run`'s five `run_powerbox*` variants (callers: 74/7/2/2/3) → keep the base,
     add `run_powerbox_cfg(&RunConfig)` (the type that already unifies
     stdin/args/env/limits), delete the middle three.
   - wasm-JIT front doors: `compile_module_reactor` is provably the `_keep` variant with
     an empty keep-list; the four `compile_module*` prologues collapse to one helper
     (~100 lines).
   - `svm-interp`'s `drive` family: an 18-argument core with the config-derivation
     prelude duplicated across two wrappers → `RunConfig` + `ThawState` structs.
3. **One sentence of doc**: the ergonomic default `Instance::call` runs the
   three-backend differential — great safety culture, but a production embedder must
   learn to pass `Backend::Jit` to avoid 3× cost; say so at the crate front.
4. **Small**: a `lock_unpoisoned` helper for the 118 copies of
   `.lock().unwrap_or_else(|e| e.into_inner())` in svm-interp — the continue-through-
   poison policy deserves one documented home.

---

## 7. Verified behavioral discrepancy to resolve

`svm-run` composes the oracle-fold routing three times, and one copy differs:
`crates/svm-run/src/lib.rs:3831-3833` and `:5404-5406` fold on
`module_demand_spawns(m) || (module_serves(m) && !serve_qualifies(..) && !module_nests(m))`,
but `run_capture_on` (the reactor `Session` path, `:5808-5818`) **omits
`module_demand_spawns`** while its comment claims "the same routing as
`Instance::run_with_caps`". Verified against the source at `9b1c34a`. Either a real gap
(an op-17 demand-spawn module on a `Backend::Jit` session doesn't fold) or an
undocumented intentional narrowing — either way it needs a test that pins the answer and
one extracted `folds_to_oracle(m)` used at all three sites. The predicate *atoms* are
correctly single-definition; it is only this composition that drifted — exactly the class
invariant 9 bans.

---

## 8. Deletions (invariant 1 in the flesh)

| Item | Where | Evidence |
|---|---|---|
| `browser/threads-spike/` | own Cargo.toml | A completed spike ("step 1 of the plan"); the result shipped elsewhere; no CI reference. Also review unreferenced one-off `browser/*.mjs` drivers individually. |
| `arena-stacks` no-op feature + its CI rows | `svm-fiber/Cargo.toml`, ci.yml:925-970 | Arena is the default; the feature rows re-test the default config (~4 redundant CI runs). |
| `FuelMode::Memory` + `compile_module_fuel` | `svm-wasm-jit:134-138,1326-1347,3001-3016` | "Not a shipping path"; the A/B it existed for concluded for `Global`; keeps a second fuel-emission arm inside the shipping emitter. |
| Snapshot skip-unknown-tags | `svm-snapshot:596-608` | Dead flexibility: exact-version check (:589) means no unknown tag can ever reach it; today junk sections restore identically — make it `Malformed` (fail-closed, −5 lines). |
| svm-wasi crate | 231 lines, no production consumer | Fold its two proving tests into `crates/svm/tests/` and delete, or record it as the deliberate on-ramp demo. Owner-taste. |
| svm-webgpu **decision** | workspace-excluded, zero consumers, zero CI | `LLVM.md:2896-2898` documents a CI lane that doesn't exist — violating the repo's own "no built-but-unwired target" rule (ci.yml:206). Add the cheap lavapipe lane or delete the crate; fix the doc either way. |
| Paged tier-up mode **sunset** | `svm-wasm-jit:3107-3225` | Landed dark per DESIGN §14 ("no default flips until a hot consumer justifies it"); ~200 lines woven through the confinement-emission helpers. Not dangerous (inside the always-emitted clamp) but speculative surface in the most sensitive emitter. Set a date; pull if no consumer lands. |
| svm-text legacy spellings | `svm-text:1332-1345,1569,1612,1783` | The header claims "one grammar; dual spellings retired" — three legacy forms are still parsed, each **twice** (main parser + the `prescan_fn_results` duplicate section-walker). Delete them; the claim becomes true. |
| Optional crate merges | `svm-exec`→`svm-fs` | Identical shape/consumers; unifies one of the errno copies. Defensible to skip. |

---

## 9. Test, CI, and doc hygiene

**Tests** (~6-8k deletable lines, all test-tier):
- 26 `lua_futamura_*` files re-declare the Lua struct-offset table 20× (already defined
  in `tests/futamura/mod.rs`) — one layout change currently means 20 edits, a bug class
  that already bit (I71(b)). Consolidate; fewer test binaries also stop re-translating
  the multi-MB `lua_eval.ll` per binary.
- `to_slot`/`from_slot` copied in 9 files; equality helpers in 4; 11 near-identical
  `oracle()` helpers in `svm-wasm-jit/tests`. One `support/compare.rs` naming the two
  equality modes (bit-exact vs NaN-insensitive — invariant 9's core distinction).
- DAP test scaffolding duplicated across `dap.rs`/`dap_bytecode.rs` (~3.5k lines).
- Witness-module *builders* duplicated between `svm-parity/catalog.rs` (1,455 lines) and
  `svm-spec` — the independence rule fences expectations, not construction plumbing.

**CI**:
- The 19-target fuzz matrix in ci.yml is a hand-maintained "keep in lockstep" list —
  add the mechanical check (`ls fuzz/fuzz_targets` vs matrix), the same pattern as the
  existing `workflows-in-sync` job.
- No CI job compiles `svm-capi`'s `hello.c` with a real `cc` — the header↔ABI seam
  (~38 decls vs 30 exports) is hand-maintained. One linux step.
- OPT.md's own open checkboxes: wire the ablation/peval benches into the nightly lane
  (until then, the numbers justifying the 12 opt passes aren't watched).
- Admit `svm-llvm` to the workspace (tests behind a `clang-tests` gate) or at least drop
  its second lockfile — today an `svm-ir` API change breaks the biggest crate only in
  its own lane.
- Re-measure `BrIfCmp` post-fuel-unification (one `megabench loopc` A/B): the recorded
  ~2-3% was at the noise floor and partly an artifact of the old per-op fuel; if still
  noise, deleting it also deletes the whole fused/unfused dual-compile seam.

**Docs** (each small; together they misinform every new session):
- `SPEC.md:24,76` says "all 86 `Inst` variants" — actual: 98; `SPEC.md:62-64` still
  records `i64x2.{min,max}` as a JIT bail (now synthesized, `svm-jit:7059-7082`).
- `README.md` uses two-backend framing and bare "JIT" — both banned by DESIGN §3's
  naming standard / invariant 9.
- `INTERP_PERF.md:895-898` tracker says fuel unification "Not started"; the body records
  it complete.
- `INVARIANTS.md:43` cites the retired `ISSUES.md`. (`ISSUE_TRACKING.md` freezes old refs
  in *code comments*; the invariants file is the mandated first read — point it at the
  epic instead.)
- `WASM.md` and `OPT.md` are past their self-declared fold-into-DESIGN points.
- `svm-mask` header regime drift (§2.7); dangling `digest256` comment
  (`svm-snapshot:1197-1200`).

---

## 10. What NOT to do (checked and rejected)

Anti-recommendations, so future sessions don't re-litigate them:

- **No mega op-metadata table / proc-macro.** The hypothesized N-places-per-op crisis is
  mostly already solved: family-structured `Inst` (98 variants ≪ 465 mnemonics),
  `effects()`/`result_count` as central exhaustive tables, family-base opcode encoding
  (a new sub-op costs ~6 sites), and roundtrip fuzz + the independent spec crate fencing
  the mechanical pairs. A codegen table would put opacity into decode-adjacent TCB to
  save ~4 of ~18 per-op sites. The *targeted* moves (§4.5) capture the real win.
  The one honest cost driver is deliberate: spec/parity/effects redundancy is the
  forcing function.
- **No shared JIT lowering layer.** The two backends share a skeleton but disjoint
  emission substrates; a shared layer would abstract exactly the masking path behind a
  trait for ~zero deletable lines. Everything genuinely shareable already sits at the
  right seams (`svm_ir::bounds`, `Func::uses_*`, `serve_qualifies`).
- **Don't table-drive or macro-collapse the tree-walk oracle's op handling.** Its
  explicit match *is* the readable semantics definition; perf work is correctly routed
  to `bytecode.rs`.
- **Keep** the two-interpreter seam (owner decision 2026-06-18), the
  `svm-durable`/`svm-snapshot` split, the `svm-wasm`/`svm-wasm-jit` direction split
  (including the emitter's no-deps posture), `DetSched` separate from the threaded
  `Scheduler`, `ensure_supported` as a separate pre-pass, `svm-spec`'s deliberate
  independence, `svm-leng` as a separate frontend, and the thin `run*` entry families
  that pair 1:1 with differential harnesses.
- **No new shared IR builder** — four construction routes exist but no named consumer
  needs a fifth abstraction; §5's svm-llvm fixture work mostly dissolves the largest one.

---

## 11. Suggested sequencing

1. **Now, small and safe:** §2.1 (ubs), §2.5 (iface ids), §2.6 (module setup), §2.7
   (mask API + header), §7 (routing pin test), §4.1 (errno), the §8 deletions, and the
   doc fixes. Mostly S-effort, several strengthen the hinge.
2. **Next, the M-effort consolidations:** §2.2 (svm-mem), §2.3 (verifier), §2.4 (span
   reference), §4.2-4.13 in any order, §6 API collapses, test dedup.
3. **The wire rev (§3) as one slice:** op cuts + encoding retirements + asset regen,
   gated by spec_encode/parity/differential. FuncType interning as its own follow-up.
4. **The monolith splits (§5) as mechanical PRs**, then the staged `run_inner`/`Host`
   extractions behind the differential matrix.

Rough totals if all accepted: **~9 ops and 5+ opcode bytes cut, ~15-20k lines removed or
de-duplicated** (≈9k of it in svm-llvm's synthesized runtime, ~5k in tests), three
monolith files become navigable modules, ten copies of security-relevant admission logic
become one, and the wire format reaches its stated final form (exact-version, no dead
fields).
