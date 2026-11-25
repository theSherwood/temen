//! Differential interp↔JIT tests for §12 **fibers** (`cont.new`/`cont.resume`/`suspend`).
//!
//! Fibers are single-threaded and cooperative, so a run is deterministic — which means the same
//! interp-vs-JIT differential oracle that guards the scalar slice applies directly here. The JIT now
//! lowers the fiber ops to its host fiber runtime (native stack switching via `temen-fiber`); for every
//! program below the JIT must produce exactly what the reference interpreter does.
//!
//! Stack switching exists on x86-64 unix, aarch64 unix, and x86-64 Windows today
//! (`temen_fiber::supported()`); elsewhere the JIT bails `Unsupported`, so these tests are gated to it.
#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

use temen_interp::{run, Trap, Value};
use temen_jit::{compile_and_run, JitOutcome, TrapKind};
use temen_text::parse_module;
use temen_verify::verify_module;

fn to_slot(v: &Value) -> i64 {
    match v {
        Value::I32(x) => *x as i64,
        Value::I64(x) => *x,
        Value::F32(x) => x.to_bits() as i64,
        Value::F64(x) => x.to_bits() as i64,
        Value::V128(b) => i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]),
        Value::Ref(x) => *x as i64,
    }
}

fn trap_matches(t: &Trap, k: &TrapKind) -> bool {
    t.code() == k.code() // one wire code for both engines (#1735)
}

/// Run `src` on both backends and assert they agree (results bit-for-bit, or the same trap kind).
fn assert_jit_matches_interp(src: &str) {
    let m = parse_module(src).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
    verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}\n{src}"));
    let mut fuel = 10_000_000u64;
    let interp = run(&m, 0, &[], &mut fuel);
    let jit = compile_and_run(&m, 0, &[]).expect("jit compile/run");
    match (&interp, &jit) {
        (Ok(vals), JitOutcome::Returned(slots)) => {
            let want: Vec<i64> = vals.iter().map(to_slot).collect();
            assert_eq!(&want, slots, "interp vs jit results differ\n{src}");
        }
        (Err(t), JitOutcome::Trapped(k)) if trap_matches(t, k) => {}
        _ => panic!("interp {interp:?} vs jit {jit:?} disagree\n{src}"),
    }
    in_a_thread_on_the_kernel_route(src, &m);
}

/// #3078 slice 2 — a fiber-only module keeps its single-vCPU path (`fiber only`), so each case whose
/// entry is `() -> (i64)` also runs inside one `thread.spawn`ed vCPU, which the kernel route admits:
/// a wrapper entry spawns a thread that calls the case's entry, and joins it. That run, metered
/// (since #3080), answers as the oracle's run of it.
fn in_a_thread_on_the_kernel_route(src: &str, m: &temen_ir::Module) {
    use temen_run::jit_kernel;
    let entry = &m.funcs[0];
    if !entry.params.is_empty() || entry.results != [temen_ir::ValType::I64] {
        return;
    }
    let declined = jit_kernel::admit(m, &temen_interp::Host::new(), false).err();
    assert_eq!(declined, Some("fiber only"), "{src}");
    let n = m.funcs.len();
    let wrapped = format!(
        "{src}
func (i64, i64) -> (i64) {{
block 0 (va: i64, vb: i64) {{
  vr = call 0()
  return vr
  }}
}}
func () -> (i64) {{
block 0 () {{
  vz = i64.const 0
  vt = thread.spawn {n} vz vz
  vs, vv = thread.join vt
  return vv
  }}
}}
"
    );
    let w = parse_module(&wrapped).unwrap_or_else(|e| panic!("parse: {e:?}\n{wrapped}"));
    verify_module(&w).unwrap_or_else(|e| panic!("verify: {e:?}\n{wrapped}"));
    let entry = (n + 1) as u32;
    let mut fuel = 10_000_000u64;
    let interp = run(&w, entry, &[], &mut fuel);
    let kernel = jit_kernel::expect_kernel(
        &w,
        entry,
        &[],
        None,
        u64::MAX,
        jit_kernel::Expect::Kernel { vcpus: 2 },
    )
    .expect("the kernel route admits it");
    match (&interp, &kernel) {
        (Ok(vals), JitOutcome::Returned(slots)) => {
            let want: Vec<i64> = vals.iter().map(to_slot).collect();
            assert_eq!(&want, slots, "interp vs the kernel route\n{wrapped}");
        }
        (Err(t), JitOutcome::Trapped(k)) if trap_matches(t, k) => {}
        _ => panic!("interp {interp:?} vs the kernel route {kernel:?} disagree\n{wrapped}"),
    }
}

/// A fiber `(i64 sp, i64 arg)` that `suspend`s its arg, then on the next resume adds 100 and returns.
/// Root: resume(10) -> (SUSPENDED, 10); resume(7) -> (RETURNED, 107).
#[test]
fn fiber_suspend_then_resume() {
    let src = "func () -> (i32, i64, i32, i64) {\n\
        block 0 () {\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const 10\n\
        \x20 v4, v5 = cont.resume v2 v3\n\
        \x20 v6 = i64.const 7\n\
        \x20 v7, v8 = cont.resume v2 v6\n\
        \x20 return v4 v5 v7 v8\n\
          }\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block 0 (v0: i64, v1: i64) {\n\
        \x20 v2 = suspend v1\n\
        \x20 v3 = i64.const 100\n\
        \x20 v4 = i64.add v2 v3\n\
        \x20 return v4\n\
          }\n\
        }\n";
    assert_jit_matches_interp(src);
}

/// A generator fiber yields 1, 2, 3 then returns 4; the root loops resuming it and sums every
/// delivered value (10) — repeated resume/suspend with the handle threaded as a block param.
#[test]
fn fiber_generator_loop() {
    let src = "func () -> (i64) {\n\
        block 0 () {\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const 0\n\
        \x20 br 1(v2, v3)\n\
        }\n\
        block 1 (v4: i64, v5: i64) {\n\
        \x20 v6 = i64.const 0\n\
        \x20 v7, v8 = cont.resume v4 v6\n\
        \x20 v9 = i64.add v5 v8\n\
        \x20 br_if v7 2(v9) 1(v4, v9)\n\
        }\n\
        block 2 (v10: i64) {\n\
        \x20 return v10\n\
          }\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block 0 (v0: i64, v1: i64) {\n\
        \x20 v2 = i64.const 1\n\
        \x20 v3 = suspend v2\n\
        \x20 v4 = i64.const 2\n\
        \x20 v5 = suspend v4\n\
        \x20 v6 = i64.const 3\n\
        \x20 v7 = suspend v6\n\
        \x20 v8 = i64.const 4\n\
        \x20 return v8\n\
          }\n\
        }\n";
    assert_jit_matches_interp(src);
}

/// A three-level resume chain (root → A → B): B suspends back to A, A suspends back to root, then the
/// chain unwinds to completion. The decisive test for the yielder-stack nesting discipline — a
/// `suspend` must return to the *correct* resumer.
#[test]
fn fiber_nested_resume_chain() {
    let src = "func () -> (i64, i64) {\n\
        block 0 () {\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const 0\n\
        \x20 v4, v5 = cont.resume v2 v3\n\
        \x20 v6, v7 = cont.resume v2 v3\n\
        \x20 return v5 v7\n\
          }\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block 0 (v0: i64, v1: i64) {\n\
        \x20 v2 = ref.func 2\n\
        \x20 v3 = i64.const 8192\n\
        \x20 v4 = cont.new v2 v3\n\
        \x20 v5 = i64.const 0\n\
        \x20 v6, v7 = cont.resume v4 v5\n\
        \x20 v8 = suspend v7\n\
        \x20 v9, v10 = cont.resume v4 v5\n\
        \x20 return v10\n\
          }\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block 0 (v0: i64, v1: i64) {\n\
        \x20 v2 = i64.const 11\n\
        \x20 v3 = suspend v2\n\
        \x20 v4 = i64.const 22\n\
        \x20 return v4\n\
          }\n\
        }\n";
    assert_jit_matches_interp(src);
}

/// Resuming a fiber that already returned is inert on both backends → `FiberFault`.
#[test]
fn fiber_resume_after_return_traps() {
    let src = "func () -> (i64) {\n\
        block 0 () {\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const 1\n\
        \x20 v4, v5 = cont.resume v2 v3\n\
        \x20 v6, v7 = cont.resume v2 v3\n\
        \x20 return v7\n\
          }\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block 0 (v0: i64, v1: i64) {\n\
        \x20 return v1\n\
          }\n\
        }\n";
    assert_jit_matches_interp(src);
}

/// A `suspend` from the root computation (no running fiber) traps on both backends → `FiberFault`.
#[test]
fn fiber_root_suspend_traps() {
    let src = "func () -> (i64) {\n\
        block 0 () {\n\
        \x20 v0 = i64.const 5\n\
        \x20 v1 = suspend v0\n\
        \x20 return v1\n\
          }\n\
        }\n";
    assert_jit_matches_interp(src);
}

/// **Fiber handle *values* are identical across backends** (the D57 3b-i gate). Historically the
/// interp's table held the root as slot 0 (first `cont.new` → handle 1) while the JIT ran the root
/// off-table (first handle 0) — a documented safe divergence (DESIGN §3a) the fiber fuzzer had to
/// work around. The run-shared interp registry holds only `cont.new`-created fibers, unifying the
/// namespace: handles are 0, 1, … on **both** backends and may flow into observable output. Pins
/// both the cross-backend agreement and the absolute values (so the namespace can't drift again).
#[test]
fn fiber_handle_values_match_across_backends() {
    let src = "func () -> (i64, i64, i32, i64) {\n\
        block 0 () {\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = cont.new v0 v1\n\
        \x20 v4 = i64.const 3\n\
        \x20 v5, v6 = cont.resume v3 v4\n\
        \x20 return v2 v3 v5 v6\n\
          }\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block 0 (v0: i64, v1: i64) {\n\
        \x20 return v1\n\
          }\n\
        }\n";
    // Cross-backend agreement (the differential)…
    assert_jit_matches_interp(src);
    // …and the absolute namespace: first handle 0, second 1, resume ran fiber 1 to completion.
    let m = parse_module(src).expect("parse");
    let mut fuel = 1_000_000u64;
    let got = run(&m, 0, &[], &mut fuel).expect("interp");
    assert_eq!(
        got,
        vec![Value::I64(0), Value::I64(1), Value::I32(1), Value::I64(3)],
        "the unified handle namespace must start at 0 and number densely"
    );
}

/// `cont.new` over a **wrong-type funcref** (the funcref names a function whose signature is not the
/// fiber entry type `(i64, i64) -> i64`) faults on first resume on *both* backends → `FiberFault`. The
/// verifier only checks the funcref is an `i32` (§12 — the type is a runtime use-site check), so this
/// path is reachable; both backends type-check lazily at first resume, and the fault is a *fiber* fault
/// (the forged-handle / dead / bomb family), not a generic `IndirectCallType`.
#[test]
fn fiber_wrong_type_funcref_traps() {
    let src = "func () -> (i32, i64) {\n\
        block 0 () {\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const 1\n\
        \x20 v4, v5 = cont.resume v2 v3\n\
        \x20 return v4 v5\n\
          }\n\
        }\n\
        func (i32) -> (i32) {\n\
        block 0 (v0: i32) {\n\
        \x20 return v0\n\
          }\n\
        }\n";
    assert_jit_matches_interp(src);
}

/// A fiber whose body actually uses the **data stack** (its `sp`) and shared memory: it stores `arg`
/// to `mem[sp]`, suspends, then reloads and returns it — exercising the two-stack split end to end and
/// the `mem_base`/`sp` threading into the fiber entry.
#[test]
fn fiber_uses_data_stack_and_memory() {
    let src = "memory 16\n\
        func () -> (i64) {\n\
        block 0 () {\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 1024\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const 777\n\
        \x20 v4, v5 = cont.resume v2 v3\n\
        \x20 v6 = i64.const 0\n\
        \x20 v7, v8 = cont.resume v2 v6\n\
        \x20 return v8\n\
          }\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block 0 (v0: i64, v1: i64) {\n\
        \x20 i64.store v0 v1\n\
        \x20 v2 = i64.const 0\n\
        \x20 v3 = suspend v2\n\
        \x20 v4 = i64.load v0\n\
        \x20 return v4\n\
          }\n\
        }\n";
    assert_jit_matches_interp(src);
}

/// W5 JIT/DWARF Stage 4c — the fiber-rooted backtrace. A fiber entry `func 1` calls helper `func 2`,
/// which `suspend`s; the root resumes the fiber once and **returns without completing it**, leaving it
/// parked. The host then walks the suspended fiber's control stack and symbolizes its guest frames.
/// A `-g` module whose root resumes a fiber through a helper and returns its suspended value,
/// leaving it parked (see [`fiber_backtrace_walks_a_suspended_fibers_guest_stack`]).
const PARKS_A_FIBER_WITH_DEBUG_INFO: &str = r#"
func () -> (i64) {
block 0 () {
  v0 = ref.func 1
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v3 = i64.const 5
  v4, v5 = cont.resume v2 v3
  return v5
  }
}

func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = call 2(v0, v1)
  return v2
  }
}

func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = suspend v1
  return v2
  }
}

debug.file 0 "fib.c"
debug.loc 1 0 0 0 5 3
debug.loc 2 0 0 0 9 3
"#;

/// [`PARKS_A_FIBER_WITH_DEBUG_INFO`] with two fibers parked: handles 0 and 1.
const PARKS_TWO_FIBERS_WITH_DEBUG_INFO: &str = r#"
func () -> (i64) {
block 0 () {
  v0 = ref.func 1
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v3 = i64.const 5
  v4, v5 = cont.resume v2 v3
  v6 = cont.new v0 v1
  v7, v8 = cont.resume v6 v3
  return v5
  }
}

func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = call 2(v0, v1)
  return v2
  }
}

func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = suspend v1
  return v2
  }
}

debug.file 0 "fib.c"
debug.loc 1 0 0 0 5 3
debug.loc 2 0 0 0 9 3
"#;

#[test]
fn fiber_backtrace_walks_a_suspended_fibers_guest_stack() {
    use temen_ir::DEFAULT_RESERVED_LOG2;
    use temen_jit::{CompiledModule, JitOutcome, INERT_CAP_THUNK};

    // `func 1` (entry) calls `func 2` (helper) at fib.c:5; the helper `suspend`s at fib.c:9.
    let src = PARKS_A_FIBER_WITH_DEBUG_INFO;
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut cm = CompiledModule::compile(
        &m,
        0,
        INERT_CAP_THUNK,
        core::ptr::null_mut(),
        DEFAULT_RESERVED_LOG2,
        None,
        None,
        None, // fuel
        None,
        None,
        0,
        false,
    )
    .expect("compile");
    // A run takes its parked fibers' backtraces as it ends only when asked (#3020).
    cm.keep_parked_backtraces(true);

    // Before the run no fiber has been created, so there is nothing to walk.
    assert!(
        cm.fiber_backtrace(0).is_empty(),
        "no parked fiber before the run"
    );

    // The root resumes the fiber (it suspends) and returns its yielded value, leaving fiber 0 parked.
    let (outcome, _) = cm.run(&[], None, None).expect("run");
    let JitOutcome::Returned(ref v) = outcome else {
        panic!("expected the root to return, got {outcome:?}");
    };
    assert_eq!(
        v.as_slice(),
        &[5i64],
        "root returned the fiber's suspended value, leaving it parked"
    );

    // The fiber-rooted walk: the parked fiber's guest call stack, innermost frame first.
    let bt = cm.fiber_backtrace(0);
    let frames: Vec<(u32, &str, u32)> = bt
        .iter()
        .map(|f| (f.func, f.file.as_str(), f.line))
        .collect();
    assert_eq!(
        frames,
        vec![(2, "fib.c", 9), (1, "fib.c", 5)],
        "backtrace is [helper (suspended, innermost), entry (its caller)]"
    );
}

// ---- Recycling ABA guard, cross-backend (DURABILITY.md §12.8 steps 1/3) ----
//
// A fiber guest handle carries a generation in its high bits (`FIBER_GEN_SHIFT == 16`), and
// `cont.resume` rejects a handle whose generation doesn't match the slot's current one — the ABA
// guard that makes slot recycling safe. The interpreter pins this directly
// (`temen-durable/tests/fiber.rs::{forged_fiber_generation_is_rejected,
// recycling_reuses_a_freed_slot_with_a_bumped_generation}`) and the JIT registry has a unit test
// (`fiber_registry::claim_gen_rejects_a_stale_generation`); these pin the **wired** JIT path
// (`cont.resume` → `fiber_resume` → `resolve` + `claim_gen`) against the interpreter oracle, so the
// generation guard is enforced identically end-to-end on both backends.

/// A genuine handle (slot 0, generation 0) resumes; a forged generation-1 handle for the same slot
/// (`(1 << 24) | 0 == 16777216`, which the slot mask clamps back to slot 0) faults — on both backends.
#[test]
fn fiber_forged_generation_faults_identically() {
    // Genuine handle: the fiber runs and returns 99.
    assert_jit_matches_interp(
        "func () -> (i64) {\n\
        block 0 () {\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v4 = i64.const 0\n\
        \x20 v5, v6 = cont.resume v2 v4\n\
        \x20 return v6\n\
          }\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block 0 (v0: i64, v1: i64) {\n\
        \x20 v2 = i64.const 99\n\
        \x20 return v2\n\
          }\n\
        }\n",
    );
    // Forged handle `(1 << 24) | 0`: same slot 0, generation 1 ≠ 0 ⇒ FiberFault on both backends.
    assert_jit_matches_interp(
        "func () -> (i64) {\n\
        block 0 () {\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const 16777216\n\
        \x20 v4 = i64.const 0\n\
        \x20 v5, v6 = cont.resume v3 v4\n\
        \x20 return v6\n\
          }\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block 0 (v0: i64, v1: i64) {\n\
        \x20 v2 = i64.const 99\n\
        \x20 return v2\n\
          }\n\
        }\n",
    );
}

/// #3166 — a module of four functions whose last is a fiber entry: the root makes a fiber with
/// funcref `funcref` (an `i32` constant) and resumes it, after a spawned thread's join when `spawn`;
/// the fiber runs `body`, which defines `vr`, and returns it. A funcref is masked to the table, so
/// `-1` names the last function.
fn last_slot_fiber(funcref: i32, spawn: bool, root: &str, body: &str) -> String {
    let sp = match spawn {
        true => "  vt = thread.spawn 1 vz vz\n  vts, vtv = thread.join vt\n",
        false => "",
    };
    format!(
        "memory 16
func () -> (i64) {{
block 0 () {{
  vz = i64.const 0
{sp}  vf = i32.const {funcref}
  vk = cont.new vf vz
{root}
  return vr
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (va: i64, vb: i64) {{
  return va
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (va: i64, vb: i64) {{
  return va
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
{body}
  return vr
  }}
}}
"
    )
}

/// #3166 — `cont.new -1` names the table's last function, which is a fiber entry here, and makes an
/// ordinary guest fiber, on every engine: it suspends and returns as `cont.new 3` does. The JIT once
/// took it for a pool task's own slot (`is_platform()` was `func == -1`), so its `suspend` trapped
/// `FiberFault` as a task root's does. Here it suspends 7, then returns 9:
/// `s0 * 1000 + v0 * 100 + s1 * 10 + v1` is 719 everywhere.
#[test]
fn a_fiber_made_from_funcref_minus_one_is_a_guest_fiber() {
    let root = "  vs0, vv0 = cont.resume vk vz\n  vs1, vv1 = cont.resume vk vz\n  va = i64.extend_i32_s vs0\n  vb = i64.extend_i32_s vs1\n  vk1000 = i64.const 1000\n  vk100 = i64.const 100\n  vk10 = i64.const 10\n  vx = i64.mul va vk1000\n  vy = i64.mul vv0 vk100\n  vw = i64.mul vb vk10\n  vxy = i64.add vx vy\n  vxyw = i64.add vxy vw\n  vr = i64.add vxyw vv1";
    let body = "  v7 = i64.const 7\n  vx = suspend v7\n  vr = i64.const 9";
    for spawn in [false, true] {
        for funcref in [3, -1] {
            let src = last_slot_fiber(funcref, spawn, root, body);
            let m = parse_module(&src).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
            verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}\n{src}"));
            let mut fuel = 10_000_000u64;
            let bytecode = temen_interp::bytecode::compile_and_run(&m, 0, &[], &mut fuel);
            assert_eq!(
                bytecode,
                Some(Ok(vec![Value::I64(719)])),
                "the bytecode engine, funcref {funcref}"
            );
            let jit = compile_and_run(&m, 0, &[]).expect("jit compile/run");
            assert_eq!(
                jit,
                JitOutcome::Returned(vec![719]),
                "the JIT, funcref {funcref}"
            );
            if !spawn {
                // The tree-walker, and the kernel route inside a spawned vCPU.
                assert_jit_matches_interp(&src);
            }
        }
    }
}

/// #3166 — a fiber made from funcref `-1` scans its roots as any fiber does. The JIT took it for a
/// task's slot and, with no fiber nested in it, started the root computation's scan at its own stack
/// pointer, reading every word from the fiber's stack up to the root's OS stack: the process died
/// with `SIGSEGV`. Here it returns, as `cont.new 3` does, on the JIT: the fiber answers whether
/// the scan's count is a count (not negative), and the root `status * 1000` plus that.
#[test]
fn a_fiber_made_from_funcref_minus_one_scans_its_roots_as_any_fiber() {
    let root = "  vs0, vv0 = cont.resume vk vz\n  va = i64.extend_i32_s vs0\n  vk1000 = i64.const 1000\n  vx = i64.mul va vk1000\n  vr = i64.add vx vv0";
    let body = "  vlo = i64.const 0\n  vhi = i64.const 65536\n  vmask = i64.const -1\n  vbuf = i64.const 16384\n  vcap = i64.const 16\n  vn = gc.roots vlo vhi vmask vbuf vcap\n  vz0 = i64.const 0\n  vok = i64.ge_s vn vz0\n  vr = i64.extend_i32_u vok";
    for spawn in [false, true] {
        let run = |funcref| {
            let src = last_slot_fiber(funcref, spawn, root, body);
            let m = parse_module(&src).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
            verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}\n{src}"));
            compile_and_run(&m, 0, &[]).expect("jit compile/run")
        };
        // The roots are a conservative superset, which the stacks' other words can grow, so the
        // count itself is not asked for.
        for funcref in [3, -1] {
            let got = run(funcref);
            assert_eq!(
                got,
                JitOutcome::Returned(vec![1001]),
                "funcref {funcref}, spawn {spawn}"
            );
        }
    }
}

/// #3166 — a `thread.join` inside a fiber made from funcref `-1` blocks its vCPU in place, as in
/// any fiber. The root spawns a vCPU (func 1) that counts to 2,000,000 and returns 5, and makes a
/// fiber of func 3, the table's last slot, from `funcref`, handing it the handle; the fiber joins
/// it and returns its value, and the root `status * 1000 + value`: 1005 on the tree-walker and the
/// JIT, on its own route and the kernel's. The JIT took the `-1` fiber for a pool task's own slot,
/// so a join that waited event-parked as a task's does and surfaced `FIBER_PARKED` to the root.
/// (The count keeps the vCPU running when the fiber joins it, so the join waits; a join that found
/// it done would pass on either.)
#[test]
fn a_join_in_a_fiber_from_funcref_minus_one_blocks_its_vcpu() {
    for funcref in [3, -1] {
        let src = format!(
            "memory 16
func () -> (i64) {{
block 0 () {{
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  vt64 = i64.extend_i32_u vt
  vf = i32.const {funcref}
  vk = cont.new vf vt64
  vs0, vv0 = cont.resume vk vz
  va = i64.extend_i32_s vs0
  vk1000 = i64.const 1000
  vx = i64.mul va vk1000
  vr = i64.add vx vv0
  return vr
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (va: i64, vb: i64) {{
  vz = i64.const 0
  br 1(vz)
}}
block 1 (vi: i64) {{
  vn = i64.const 2000000
  vlt = i64.lt_s vi vn
  br_if vlt 2(vi) 3()
}}
block 2 (vj: i64) {{
  vone = i64.const 1
  vk = i64.add vj vone
  br 1(vk)
}}
block 3 () {{
  v5 = i64.const 5
  return v5
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (va: i64, vb: i64) {{
  return va
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  vh = i32.wrap_i64 vsp
  vs, vv = thread.join vh
  return vv
  }}
}}
"
        );
        let m = parse_module(&src).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
        verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}\n{src}"));
        let mut fuel = 10_000_000_000u64;
        let oracle = run(&m, 0, &[], &mut fuel);
        assert_eq!(
            oracle,
            Ok(vec![Value::I64(1005)]),
            "the tree-walker, funcref {funcref}"
        );
        let jit = compile_and_run(&m, 0, &[]).expect("jit compile/run");
        assert_eq!(
            jit,
            JitOutcome::Returned(vec![1005]),
            "the JIT, funcref {funcref}"
        );
        let kernel = temen_run::jit_kernel::expect_kernel(
            &m,
            0,
            &[],
            None,
            u64::MAX,
            temen_run::jit_kernel::Expect::Kernel { vcpus: 2 },
        )
        .expect("the kernel route admits it");
        assert_eq!(
            kernel,
            JitOutcome::Returned(vec![1005]),
            "the kernel route, funcref {funcref}"
        );
    }
}

/// A finished fiber's slot is recycled at a bumped generation: the next `cont.new` reuses slot 0 at
/// generation 1 (handle `(1 << 24) | 0 == 16777216`), and the *stale* generation-0 handle to the
/// former occupant then faults even though slot 0 is live. Both backends must agree on each.
#[test]
fn recycled_slot_generation_guard_agrees() {
    // Fiber A (slot 0, gen 0) finishes; the next cont.new reuses slot 0 at gen 1 — returning the i64
    // handle makes the reuse observable (16777216). Both backends must produce the same handle.
    assert_jit_matches_interp(
        "func () -> (i64) {\n\
        block 0 () {\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const 0\n\
        \x20 v4, v5 = cont.resume v2 v3\n\
        \x20 v6 = cont.new v0 v1\n\
        \x20 return v6\n\
          }\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block 0 (v0: i64, v1: i64) {\n\
        \x20 v2 = i64.const 7\n\
        \x20 return v2\n\
          }\n\
        }\n",
    );
    // After slot 0 is recycled (now gen 1), resuming A's stale gen-0 handle (i64 0) must fault on
    // both backends — even though slot 0 is live — because the generation no longer matches.
    assert_jit_matches_interp(
        "func () -> (i64) {\n\
        block 0 () {\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const 0\n\
        \x20 v4, v5 = cont.resume v2 v3\n\
        \x20 v6 = cont.new v0 v1\n\
        \x20 v9 = i64.const 0\n\
        \x20 v7, v8 = cont.resume v9 v3\n\
        \x20 return v8\n\
          }\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block 0 (v0: i64, v1: i64) {\n\
        \x20 v2 = i64.const 7\n\
        \x20 return v2\n\
          }\n\
        }\n",
    );
}

/// #3020 — a parked fiber's backtrace is its run's: a run takes its fibers' backtraces as it ends
/// only when asked, and the next run's start retires them, whatever that run does. A run that does
/// not ask leaves none, and a handle reads as the live table reads it (a guest tag above the
/// generation is ignored).
#[test]
fn fiber_backtrace_reaches_the_last_run_s_fibers_alone() {
    use temen_ir::DEFAULT_RESERVED_LOG2;
    use temen_jit::{CompiledModule, INERT_CAP_THUNK};
    let m = parse_module(PARKS_TWO_FIBERS_WITH_DEBUG_INFO).expect("parse");
    verify_module(&m).expect("verify");
    let mut cm = CompiledModule::compile(
        &m,
        0,
        INERT_CAP_THUNK,
        core::ptr::null_mut(),
        DEFAULT_RESERVED_LOG2,
        None,
        None,
        None,
        None,
        None,
        0,
        false,
    )
    .expect("compile");
    let _ = cm.run(&[], None, None).expect("run 1");
    assert!(cm.fiber_backtrace(0).is_empty(), "nothing taken unasked");

    cm.keep_parked_backtraces(true);
    let _ = cm.run(&[], None, None).expect("run 2");
    let bt = cm.fiber_backtrace(1);
    assert_eq!(bt.len(), 2, "run 2's second parked fiber: {bt:?}");
    // Handle 1 with a guest tag in the top byte: the slot is read masked to the table (two
    // slots), the generation masked, as a live table reads them.
    let tagged = (0x5a_i64 << 56) | 1;
    assert_eq!(
        cm.fiber_backtrace(tagged),
        bt,
        "a tag above the generation is ignored"
    );
    assert_eq!(
        cm.fiber_backtrace(0).len(),
        2,
        "and the first fiber is its own"
    );

    cm.keep_parked_backtraces(false);
    let _ = cm.run(&[], None, None).expect("run 3");
    assert!(
        cm.fiber_backtrace(0).is_empty(),
        "run 3 retired run 2's backtraces and took none"
    );
}
