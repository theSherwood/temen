//! The threads-tier gate (`BROWSER.md` § "wasm-JIT tier", per-Worker JIT). Unlike `differential.rs`
//! (which JITs a whole func-0-rooted kernel), a threaded guest keeps running on the resumable
//! interpreter and *tiers up* a direct `Call` to any emitted function. [`compile_module_tierup`]
//! decides that emit set: every in-subset function whose calls all route, **regardless** of func-0
//! reachability — so a pure compute leaf reachable only through `thread.spawn` still emits. These
//! tests pin the eligibility bitmap and differential-run each emitted `f{i}` against the bytecode
//! interpreter, the same MISCOMPILE-grade contract the whole-module path holds.

use temen_interp::{bytecode, Trap, Value};
use temen_wasm_jit::{
    compile_module_tierup, compile_module_tierup_b2, TRAP_MEMORY_FAULT, TRAP_OUT_OF_FUEL,
};
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const WIN_BASE: u32 = 0x1_0000;
const ENV_PTR: u32 = 1024;
const FUEL: u64 = 100_000_000;

#[derive(Debug, PartialEq)]
enum Outcome {
    Vals(Vec<Value>),
    Trap(TrapKind),
}
#[derive(Debug, PartialEq, Clone, Copy)]
enum TrapKind {
    DivByZero,
    OverflowOrConv,
    MemoryFault,
    Unreachable,
    OutOfFuel,
    Other,
}

/// Run Temen function `func` standalone on the bytecode interpreter (the oracle for what the emitted
/// `f{func}` region must compute).
fn oracle(m: &temen_ir::Module, func: u32, args: &[Value]) -> Outcome {
    let mut fuel = FUEL;
    match bytecode::compile_and_run(m, func, args, &mut fuel) {
        None => panic!("oracle: module unsupported by the bytecode engine"),
        Some(Ok(vals)) => Outcome::Vals(vals),
        Some(Err(t)) => Outcome::Trap(match t {
            Trap::DivByZero => TrapKind::DivByZero,
            Trap::IntOverflow | Trap::BadConversion => TrapKind::OverflowOrConv,
            Trap::MemoryFault => TrapKind::MemoryFault,
            Trap::Unreachable => TrapKind::Unreachable,
            Trap::OutOfFuel => TrapKind::OutOfFuel,
            _ => TrapKind::Other,
        }),
    }
}

/// Call the emitted `f{func}` under wasmi over the window at `WIN_BASE`. A cross-tier `call_interp`
/// runs the named leaf on the interpreter (so an emitted caller can reach an interp leaf), matching
/// the browser host's `temen_wasmjit_call_interp`.
fn wasm_run(m: &temen_ir::Module, wasm: &[u8], func: u32, args: &[Value]) -> Outcome {
    let engine = Engine::default();
    let module = WModule::new(&engine, wasm).expect("emitted wasm must validate");
    let mut store: Store<i32> = Store::new(&engine, 0);
    let pages = 2 + m.memory.map_or(0, |mc| (mc.size() >> 16) as u32);
    let memory = Memory::new(&mut store, MemoryType::new(pages, None)).unwrap();
    memory
        .write(&mut store, ENV_PTR as usize, &(FUEL as i64).to_le_bytes())
        .unwrap();
    let mut linker: Linker<i32> = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap("env", "trap", |mut caller: Caller<'_, i32>, code: i32| {
            *caller.data_mut() = code;
        })
        .unwrap();
    // These tier-up leaves are pure (no cross-tier calls in the differential guests), so the import
    // is present but a call would be a bug — assert it never fires.
    linker
        .func_wrap::<_, ()>(
            "env",
            "call_interp",
            |_: Caller<'_, i32>, _f: i32, _a: i32| {
                unreachable!("no cross-tier call expected in these tier-up leaves");
            },
        )
        .unwrap();
    let instance = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    let f = instance
        .get_func(&store, &format!("f{func}"))
        .unwrap_or_else(|| panic!("f{func} not exported"));

    let sig = &m.funcs[func as usize];
    let mut params = vec![Val::I32(WIN_BASE as i32), Val::I32(ENV_PTR as i32)];
    for (t, a) in sig.params.iter().zip(args) {
        params.push(match (t, a) {
            (temen_ir::ValType::I32, Value::I32(v)) => Val::I32(*v),
            (temen_ir::ValType::I64, Value::I64(v)) => Val::I64(*v),
            _ => panic!("arg/type mismatch"),
        });
    }
    let mut results: Vec<Val> = sig
        .results
        .iter()
        .map(|t| match t {
            temen_ir::ValType::I32 => Val::I32(0),
            _ => Val::I64(0),
        })
        .collect();

    match f.call(&mut store, &params, &mut results) {
        Ok(()) => Outcome::Vals(
            results
                .iter()
                .map(|v| match v {
                    Val::I32(x) => Value::I32(*x),
                    Val::I64(x) => Value::I64(*x),
                    _ => panic!("non-integer result"),
                })
                .collect(),
        ),
        Err(e) => {
            let host_code = *store.data();
            let kind = if host_code == TRAP_OUT_OF_FUEL {
                TrapKind::OutOfFuel
            } else if host_code == TRAP_MEMORY_FAULT {
                TrapKind::MemoryFault
            } else {
                use wasmi::core::TrapCode;
                match e.as_trap_code() {
                    Some(TrapCode::IntegerDivisionByZero) => TrapKind::DivByZero,
                    Some(TrapCode::IntegerOverflow | TrapCode::BadConversionToInteger) => {
                        TrapKind::OverflowOrConv
                    }
                    Some(TrapCode::UnreachableCodeReached) => TrapKind::Unreachable,
                    other => panic!("unexpected wasm trap: {other:?} ({e})"),
                }
            };
            Outcome::Trap(kind)
        }
    }
}

fn build(src: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// Differential-run every emitted `f{i}` against the interpreter across an i64 arg sweep.
fn diff_eligible(m: &temen_ir::Module, eligible: &[bool], wasm: &[u8], sweep: &[i64]) {
    let mut ran = 0;
    for (i, &e) in eligible.iter().enumerate() {
        if !e {
            continue;
        }
        let f = &m.funcs[i];
        // All-i64 leaves only (what the browser tier-up ABI marshals); skip any i32/float/v128 sigs.
        if !f
            .params
            .iter()
            .chain(&f.results)
            .all(|t| *t == temen_ir::ValType::I64)
        {
            continue;
        }
        let arity = f.params.len();
        // Fill every param with the sweep value (a spawned worker body is `(sp, arg)` — 2 i64
        // params); both engines get identical args, so the differential stays exact.
        let sweeps: Vec<Vec<Value>> = if arity == 0 {
            vec![vec![]]
        } else {
            sweep.iter().map(|a| vec![Value::I64(*a); arity]).collect()
        };
        for args in &sweeps {
            let want = oracle(m, i as u32, args);
            let got = wasm_run(m, wasm, i as u32, args);
            assert_eq!(want, got, "tier-up MISCOMPILE: f{i} args {args:?}");
            ran += 1;
        }
    }
    assert!(
        ran > 0,
        "vacuous: no eligible all-i64 function was differential-run"
    );
}

const SWEEP: &[i64] = &[0, 1, 2, 5, 64, 1000, -1, -1000, 100_000, i64::MIN, i64::MAX];

/// The flagship shape: func 0 spawns worker vCPUs (concurrency — never JITs), the worker func 1
/// increments a shared counter through an atomic (concurrency — never JITs) by an amount computed
/// in the pure leaf func 2. Only func 2 is emitted + eligible, though it is reachable **only**
/// through `thread.spawn` from func 0.
const SPAWN: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 500
  v1 = thread.spawn 1 v0 v0
  v2 = thread.join v1
  v3 = i64.const 0
  v4 = i64.atomic.load v3
  return v4
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, v0: i64) {
  v1 = call 2 (v0)
  v2 = i64.const 0
  v3 = i64.atomic.rmw.add v2 v1
  v4 = i64.const 0
  return v4
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 2
  v2 = i64.mul v0 v1
  v3 = i64.sub v2 v0
  return v3
  }
}
"#;

#[test]
fn spawn_only_leaf_is_eligible_and_matches() {
    let m = build(SPAWN);
    let (wasm, eligible) = compile_module_tierup(&m, false).expect("tier-up emit");
    // func 0 (thread.spawn) and func 1 (atomic.rmw) are concurrency → not in-subset → not emitted;
    // func 2 is a pure i64 leaf reachable only via spawn → emitted + eligible.
    assert_eq!(eligible, vec![false, false, true], "eligibility bitmap");
    diff_eligible(&m, &eligible, &wasm, SWEEP);
}

/// A pure leaf that itself calls a deeper pure leaf: the fixpoint must keep BOTH (an emitted caller's
/// direct call routes to an emitted target), so the tier-up module can run either as an entry.
const TRANSITIVE: &str = r#"
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = thread.spawn 1 v0 v0
  v2 = thread.join v1
  return v2
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, v0: i64) {
  v1 = call 2 (v0)
  return v1
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = call 3 (v0)
  v2 = i64.const 7
  v3 = i64.add v1 v2
  return v3
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 3
  v2 = i64.mul v0 v1
  return v2
  }
}
"#;

#[test]
fn transitive_pure_leaves_both_emitted() {
    let m = build(TRANSITIVE);
    let (wasm, eligible) = compile_module_tierup(&m, false).expect("tier-up emit");
    // func 0 (spawn) + func 1 (spawn worker via thread.spawn? no — func 1 is a plain worker body that
    // directly calls func 2). func 1 is in-subset (pure call), func 2 + func 3 pure. Only func 0 uses
    // concurrency. So funcs 1,2,3 are all emitted; func 0 is not.
    assert_eq!(
        eligible,
        vec![false, true, true, true],
        "eligibility bitmap"
    );
    diff_eligible(&m, &eligible, &wasm, SWEEP);
}

/// An in-subset function that calls a **non-leaf, non-subset** function (one using a concurrency op —
/// `memory.notify` — which forces the interpreter and is not a leaf) must be dropped: its emitted body
/// would carry an unroutable `Call`. The fixpoint removes it; the deeper pure leaf still emits.
/// (`memory.notify` — not an atomic — is the out-of-subset vehicle: atomics are now emitted in a
/// concurrency-free module, so a concurrency op is what a non-subset non-leaf needs here.)
const UNROUTABLE_CALLER: &str = r#"
memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = call 1 (v0)
  return v1
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  v2 = i32.const 1
  v3 = atomic.notify v1 v2
  v4 = call 2 (v0)
  return v4
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 11
  v2 = i64.add v0 v1
  return v2
  }
}
"#;

#[test]
fn caller_of_nonleaf_is_dropped() {
    let m = build(UNROUTABLE_CALLER);
    let (wasm, eligible) = compile_module_tierup(&m, false).expect("tier-up emit");
    // func 1 uses `memory.notify` (concurrency → not in-subset; and not a leaf — it has memory + a
    // call). func 0 is in-subset but calls func 1 (non-emitted, non-leaf) → dropped by the fixpoint.
    // func 2 is a pure leaf → emitted. Result: only func 2 eligible.
    assert_eq!(eligible, vec![false, false, true], "eligibility bitmap");
    diff_eligible(&m, &eligible, &wasm, SWEEP);
}

/// A fully in-subset module still tier-ups every function (the degenerate case — same emit set as
/// the whole-module path), so enabling tier-up never regresses a JITtable guest.
const ALL_SUBSET: &str = r#"
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = call 1 (v0)
  v2 = i64.const 1
  v3 = i64.add v1 v2
  return v3
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 10
  v2 = i64.mul v0 v1
  return v2
  }
}
"#;

#[test]
fn all_in_subset_all_eligible() {
    let m = build(ALL_SUBSET);
    let (wasm, eligible) = compile_module_tierup(&m, false).expect("tier-up emit");
    assert_eq!(eligible, vec![true, true], "eligibility bitmap");
    diff_eligible(&m, &eligible, &wasm, SWEEP);
}

/// #888 — the fixpoint cascade. `f1` is pure integer compute (in-subset, tier-up-eligible) whose
/// *only* disqualifier is a direct `Call` to `f2`, a `call.cap`-ing helper (out of subset). Under
/// the **local**-table emit (`compile_module_tierup`) `f2` is not a strict `interp_leaf`
/// (memory-free/call-free/cap-free), so the fixpoint drops `f1` — it cascades to the interpreter.
/// Under the **shared reserved table** (`compile_module_tierup_b2`, #888) `f2` is a marshallable
/// cross-tier callee (the reactor `cross` set — the host bounces it over the live window), so `f1`
/// stays emitted. This is the coverage lever measured in #887 (~30% → ~90% on the C-family cards),
/// proven here as one function flipping from interpreted to emitted purely from the widened set.
const CASCADE: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  vh = i32.const 0
  vp = i64.const 0
  vl = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vh (vp, vl)
  vx = i64.const 3
  vr = call 1 (vx)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vk = i64.const 100
  vsum = i64.add v0 vk
  vr = call 2 (vsum)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vh = i32.const 0
  vp = i64.const 0
  vl = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vh (vp, vl)
  return v0
  }
}
"#;

#[test]
fn widened_cross_tier_uncascades_the_caller() {
    let m = build(CASCADE);
    // f0 cap-calls (out of subset — interpreter-driven top frame); f2 cap-calls (out of subset);
    // f1 is pure integer compute that only calls f2.
    let (_, local) = compile_module_tierup(&m, false).expect("local tier-up emit");
    assert_eq!(
        local,
        vec![false, false, false],
        "local table: f1 cascades to the interpreter (f2 isn't a strict interp_leaf)"
    );
    let (_, widened) = compile_module_tierup_b2(&m, false, 10).expect("B2 tier-up emit");
    assert_eq!(
        widened,
        vec![false, true, false],
        "#888 widened cross-tier: f1 now emits (f2 is a marshallable cross-tier callee)"
    );
}

/// #1370 — the counterpart to [`widened_cross_tier_uncascades_the_caller`]: the #888 widening must
/// **not** reach a callee the bounce cannot complete.
///
/// The B2 cross-tier set admits any `marshallable_sig` non-subset function because its host services
/// `env.call_interp` over the run's live window/powerbox — which covers memory and cap calls, the
/// axes #887 widened for. It does not cover the **§12 scheduling** axis: a bounce is a synchronous
/// nested drive (`temen-interp`'s `drive_nested`), and that loop has no arm for `thread.*` or
/// `memory.wait`/`notify`, so the first such op traps `CapFault` and the whole run declines to the
/// interpreter — even though the identical call made *inline* runs fine. That breaks the tier-up
/// contract that eligibility is "a pure acceleration, never a correctness gate".
///
/// This is the JACL shape: `f1` is `__jacl_entry` (a small all-i64 body whose only work is a direct
/// call), `f2` is `jacl_sched_run_main` (the language runtime's scheduler, futex-bearing). Before the
/// fix `f2` was a widened leaf, so `f1` stayed emitted and every run of it trapped. The gate excludes
/// `f2` from the leaf set instead, and the fixpoint cascades `f1` off the emitted set — the region is
/// never emitted into a bounce that cannot complete.
const FUTEX_CALLEE: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  vh = i32.const 0
  vp = i64.const 0
  vl = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vh (vp, vl)
  vx = i64.const 3
  vr = call 1 (vx)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vk = i64.const 100
  vsum = i64.add v0 vk
  vr = call 2 (vsum)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vaddr = i64.const 16384
  vexp = i32.const 1
  vto = i64.const -1
  vst = i32.atomic.wait vaddr vexp vto
  vcnt = i32.const 1
  vwk = atomic.notify vaddr vcnt
  return v0
  }
}
"#;

#[test]
fn futex_callee_is_not_a_widened_cross_tier_leaf() {
    let m = build(FUTEX_CALLEE);
    // Sanity: f2 really is the futex-bearing callee this gate is about.
    assert!(
        m.funcs[2].uses_futex(),
        "f2 must carry the futex ops for this test to mean anything"
    );
    // The local table never admitted it (`interp_leaf` excludes `uses_concurrency`) — unchanged.
    let (_, local) = compile_module_tierup(&m, false).expect("local tier-up emit");
    assert_eq!(
        local,
        vec![false, false, false],
        "local table: f1 cascades off (f2 isn't a strict interp_leaf)"
    );
    // B2 must now agree: f2 is not bounce-serviceable, so f1 cascades off rather than being emitted
    // into a bounce that CapFaults. Before #1370 this was `[false, true, false]`.
    let (_, widened) = compile_module_tierup_b2(&m, false, 10).expect("B2 tier-up emit");
    assert_eq!(
        widened,
        vec![false, false, false],
        "#1370: a futex-bearing callee must not be a widened cross-tier leaf, so its caller cascades off"
    );
}

/// The gate is **surgical**, not a blanket retreat from #888: swapping the futex callee's body for a
/// `call.cap` (the memory/cap axis the widening was actually for) leaves `f1` emitted. Without this,
/// a regression that widened the gate to all of [`Func::uses_concurrency`] — or reverted #888
/// outright — would still satisfy the test above.
#[test]
fn the_futex_gate_does_not_narrow_the_888_widening() {
    let futex = build(FUTEX_CALLEE);
    let cap = build(CASCADE);
    // Identical shape (f0 interp-driven root, f1 pure compute calling f2); the ONLY difference is
    // what f2 does — futex ops vs a `call.cap`.
    let (_, futex_widened) = compile_module_tierup_b2(&futex, false, 10).expect("B2 emit");
    let (_, cap_widened) = compile_module_tierup_b2(&cap, false, 10).expect("B2 emit");
    assert_eq!(
        futex_widened,
        vec![false, false, false],
        "futex callee: f1 cascades off"
    );
    assert_eq!(
        cap_widened,
        vec![false, true, false],
        "cap callee: f1 still emits — the #888 widening is intact"
    );
}

/// #1370, transitivity: inside a bounce every callee runs on the same nested drive, so it is not
/// enough for the leaf's own body to be clean. Here `f2` (the candidate leaf) is pure arithmetic and
/// only its *callee* `f3` touches the futex — before the closure was taken, `f2` was admitted as a
/// leaf and `f1` stayed emitted, so the bounce still trapped one frame deeper.
const FUTEX_TRANSITIVE: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  vh = i32.const 0
  vp = i64.const 0
  vl = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vh (vp, vl)
  vx = i64.const 3
  vr = call 1 (vx)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vk = i64.const 100
  vsum = i64.add v0 vk
  vr = call 2 (vsum)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vk = i64.const 7
  vsum = i64.add v0 vk
  vr = call 3 (vsum)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vaddr = i64.const 16384
  vexp = i32.const 1
  vto = i64.const -1
  vst = i32.atomic.wait vaddr vexp vto
  return v0
  }
}
"#;

#[test]
fn a_futex_reaching_callee_closure_is_not_a_leaf() {
    let m = build(FUTEX_TRANSITIVE);
    // f2's own body is clean — only f3 carries the futex. The closure is what disqualifies it.
    assert!(!m.funcs[2].uses_futex(), "f2's own body must be clean");
    assert!(m.funcs[3].uses_futex(), "f3 carries the futex");
    let (_, widened) = compile_module_tierup_b2(&m, false, 10).expect("B2 tier-up emit");
    assert_eq!(
        widened,
        vec![false, false, false, false],
        "#1370: an unserviceable op anywhere in the callee closure disqualifies the leaf, and the \
         emit fixpoint cascades its callers off"
    );
}

/// #1370, the `suspend` seed. `drive_nested` services `cont.new`/`cont.resume` — whoever runs them
/// owns both sides — but its `FiberSuspend` arm needs a resumer on the chain, and a bounce entry has
/// none. So a `suspend`-bearing callee is unserviceable for the same reason a futex one is, while a
/// callee that merely drives its *own* fiber to completion stays a perfectly good leaf.
const SUSPEND_CALLEE: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  vh = i32.const 0
  vp = i64.const 0
  vl = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vh (vp, vl)
  vx = i64.const 3
  vr = call 1 (vx)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vk = i64.const 100
  vsum = i64.add v0 vk
  vr = call 2 (vsum)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vs = suspend v0
  return vs
  }
}
"#;

#[test]
fn a_suspending_callee_is_not_a_widened_cross_tier_leaf() {
    let m = build(SUSPEND_CALLEE);
    assert!(m.funcs[2].uses_suspend(), "f2 must carry the suspend");
    let (_, widened) = compile_module_tierup_b2(&m, false, 10).expect("B2 tier-up emit");
    assert_eq!(
        widened,
        vec![false, false, false],
        "#1370: a `suspend`-bearing callee CapFaults on the bounce's empty resume chain, so it must \
         not be a leaf"
    );
}

/// #1370, the indirect clause — the gate's most conservative rule, pinned in both directions.
///
/// A `call.dyn` inside a bounce resolves through the shared reserved table and can land on **any**
/// slot, so once the module contains an unserviceable op an indirect-calling candidate might reach
/// it. That is the same fail-closed posture `analyze_from` takes for indirect reachability. The
/// second half is what keeps the rule honest: with no unserviceable op anywhere, an indirect-calling
/// callee is still a perfectly good leaf, so the #888 widening is untouched on the C-family cards
/// (single-threaded guests carry no futex/thread/`suspend` op at all).
const INDIRECT_DISPATCH: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  vh = i32.const 0
  vp = i64.const 0
  vl = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vh (vp, vl)
  vx = i64.const 3
  vr = call 1 (vx)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vk = i64.const 100
  vsum = i64.add v0 vk
  vr = call 2 (vsum)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vh = i32.const 0
  vp = i64.const 0
  vl = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vh (vp, vl)
  vi = i32.wrap_i64 v0
  vr = call.dyn (i64) -> (i64) vi (v0)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vaddr = i64.const 16384
  vexp = i32.const 1
  vto = i64.const -1
  vst = i32.atomic.wait vaddr vexp vto
  return v0
  }
}
"#;

/// The same module with the futex body (f3) replaced by pure arithmetic — nothing unserviceable
/// anywhere, so the indirect clause must stay dormant.
const INDIRECT_DISPATCH_CLEAN: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  vh = i32.const 0
  vp = i64.const 0
  vl = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vh (vp, vl)
  vx = i64.const 3
  vr = call 1 (vx)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vk = i64.const 100
  vsum = i64.add v0 vk
  vr = call 2 (vsum)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vh = i32.const 0
  vp = i64.const 0
  vl = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vh (vp, vl)
  vi = i32.wrap_i64 v0
  vr = call.dyn (i64) -> (i64) vi (v0)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vk = i64.const 2
  vr = i64.mul v0 vk
  return vr
  }
}
"#;

#[test]
fn an_indirect_caller_is_gated_only_when_the_module_has_an_unserviceable_op() {
    // With a futex op in the module, the indirect-dispatching callee f2 could reach it → not a leaf,
    // so f1 cascades off.
    let dirty = build(INDIRECT_DISPATCH);
    let (_, widened) = compile_module_tierup_b2(&dirty, false, 10).expect("B2 tier-up emit");
    assert!(
        !widened[1],
        "an indirect callee may dispatch to the futex function, so its caller must cascade off: \
         {widened:?}"
    );

    // Same module, nothing unserviceable: the indirect clause is dormant and f1 still emits.
    let clean = build(INDIRECT_DISPATCH_CLEAN);
    let (_, widened_clean) = compile_module_tierup_b2(&clean, false, 10).expect("B2 tier-up emit");
    assert!(
        widened_clean[1],
        "with no unserviceable op anywhere the indirect clause must not fire — the #888 widening \
         stands: {widened_clean:?}"
    );
}

/// #1546 — the third seed shape, and the one that fails **open** rather than closed.
///
/// A bounce into a `gc.roots`-bearing callee *completes*. That is the problem. GC.md §3.1 requires
/// coverage of "every fiber not actively executing guest mutator code **+ the caller of
/// `gc.roots`**", and at a cross-tier bounce the caller chain runs down into the **emitted wasm
/// frame** that called `env.call_interp`. That frame's live values are wasm locals and operand-stack
/// entries, which neither the guest nor the host can enumerate (BROWSER.md: "on wasm even a thunk
/// can't see JITted locals"). So the scan returns a set with those roots missing, and the guest's
/// non-moving collector frees a live object — silent heap corruption, where the futex seed above
/// merely traps.
///
/// This is the JACL shape again: `f1` is guest code holding a heap pointer across an allocation,
/// `f2` is `jacl_gc_collect` → `jacl_gc_collect_stw`. Two sets had to change, so this pins both:
///
/// * the **local** table's strict `interp_leaf` — a `gc.roots` body is marshallable, call-free,
///   cap-free and `Inst::Store`-free, so it qualified as a leaf on the predicate's own terms (the
///   `gc.roots` write is not a `Store`, and `uses_concurrency` does not cover the op);
/// * the **B2** widened `cross` set, via [`bounce_serviceable`]'s fourth seed.
const GC_ROOTS_CALLEE: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  vh = i32.const 0
  vp = i64.const 0
  vl = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vh (vp, vl)
  vx = i64.const 3
  vr = call 1 (vx)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vk = i64.const 100
  vsum = i64.add v0 vk
  vr = call 2 (vsum)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vlo = i64.const 0
  vhi = i64.const 4096
  vmask = i64.const -1
  vbuf = i64.const 128
  vcap = i64.const 8
  vn = gc.roots vlo vhi vmask vbuf vcap
  return vn
  }
}
"#;

#[test]
fn gc_roots_callee_is_not_a_cross_tier_leaf_on_either_table() {
    let m = build(GC_ROOTS_CALLEE);
    // Sanity: f2 really is the scanning callee this gate is about, and it is NOT caught by the
    // concurrency predicate that already excluded the futex case above.
    assert!(
        m.funcs[2].uses_gc_roots(),
        "f2 must carry gc.roots for this test to mean anything"
    );
    assert!(
        !m.funcs[2].uses_concurrency(),
        "and uses_concurrency must NOT cover it — that is why an explicit exclusion was needed"
    );
    // Local table: f2 is not a leaf, so f1 cascades off. Before #1546 this was `[false, true,
    // false]` — the strict leaf set admitted a gc.roots body, which is the sharper half of the bug.
    let (_, local) = compile_module_tierup(&m, false).expect("local tier-up emit");
    assert_eq!(
        local,
        vec![false, false, false],
        "local table: a gc.roots body is not a strict interp_leaf, so f1 cascades off"
    );
    // B2 widened table: same verdict by the bounce-serviceable seed. Before #1546: `[false, true,
    // false]`.
    let (_, widened) = compile_module_tierup_b2(&m, false, 10).expect("B2 tier-up emit");
    assert_eq!(
        widened,
        vec![false, false, false],
        "B2 widened cross-tier: gc.roots is not bounce-serviceable, so f1 cascades off too"
    );
}
