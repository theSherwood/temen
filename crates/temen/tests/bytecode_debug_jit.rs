//! Debugging the §22 guest-driven **`Jit` capability** (iface 11: `compile`/`install`/`uninstall`/
//! `invoke`) on the interpreter debug engines. Before this slice the debug drivers declined every
//! §22 op — the former single-vCPU `DebugRun` trapped `Malformed`, the `ScheduledDebugRun` returned
//! `Declined` — so a guest-JIT program fell back to the tree-walker oracle instead of being
//! debuggable over bytecode. Now the ops are serviced **inline in `debug_advance_fiber`** (they mutate
//! only the stepping vCPU + the shared dispatch table, spawning no scheduler task), so the engine
//! steps a §22 program op-by-op, breakpoints fire around the ops, and the result stays bit-identical to
//! the tree-walker oracle + the production bytecode engine.
//!
//! `install` + `call.dyn` steps op-by-op like any module-≥1 frame; `invoke` **steps into** the
//! submitted unit (#1517 slice 3 — it was a seam-free leaf before).

use temen_encode::encode_module;
use temen_interp::bytecode::{SchedBreak, SchedStop, ScheduledDebugRun};
use temen_interp::moment::Refusal;
use temen_interp::{run_with_host, Host, IrPc, Trap, Value};
use temen_ir::Data;
use temen_run::grant_jit;
use temen_text::parse_module;
use temen_verify::verify_module;

/// Where the guest reads the submitted blob from (matches the `i64.const 20480` in the guests below).
/// Above the #1094 NULL guard — a `data`/blob segment in `[0, 16384)` now traps `MemoryFault`.
const BLOB_OFF: u64 = 20480;

/// Encode `src` as the binary unit blob a guest submits to `Jit.compile`.
fn blob(src: &str) -> Vec<u8> {
    let m = parse_module(src).expect("parse blob");
    verify_module(&m).expect("verify blob");
    encode_module(&m)
}

/// Parse `guest_src` (with `BLOBLEN` replaced by `b.len()`) and inject `b` as a data segment at
/// [`BLOB_OFF`] — the debug engines seed memory from the module's data segments (`Mem::root`), so this
/// is how the blob reaches the guest (the tree-walker seeds the same segment, keeping them identical).
fn guest_module(guest_src: &str, b: &[u8]) -> temen_ir::Module {
    let src = guest_src.replace("BLOBLEN", &b.len().to_string());
    let mut m = parse_module(&src).expect("parse guest");
    m.data.push(Data {
        offset: BLOB_OFF,
        readonly: false,
        bytes: b.to_vec(),
    });
    verify_module(&m).expect("verify guest");
    m
}

/// A fresh host with the `Jit` cap granted a `2^table_log2`-slot install table; returns `(host, jit)`.
/// Deterministic, so every backend mints the same handle.
fn jit_host(m: &temen_ir::Module, table_log2: u8) -> (Host, i32) {
    let mut host = Host::new();
    let jit = grant_jit(&mut host, m, table_log2);
    (host, jit)
}

/// The tree-walker oracle result (`run_with_host` seeds the injected data segment like the debug run).
fn oracle(m: &temen_ir::Module, args: &[Value]) -> Result<Vec<Value>, Trap> {
    let (mut host, _) = jit_host(m, 4);
    let mut fuel = 50_000_000u64;
    run_with_host(m, 0, args, &mut fuel, &mut host)
}

/// Drive a `ScheduledDebugRun` to completion (no breakpoints), returning the root result.
fn sched_to_end(run: &mut ScheduledDebugRun, fuel: &mut u64) -> Result<Vec<Value>, Trap> {
    loop {
        match run.run_until_stop(fuel) {
            SchedStop::Finished(r) => return r,
            SchedStop::Break { .. } => continue,
            // No blocking stdin in these runs: a stdin park would be as stuck as a deadlock.
            SchedStop::Blocked | SchedStop::StdinPark { .. } | SchedStop::CapPark { .. } => {
                return Err(Trap::Malformed)
            }
            // The whole point of this slice: a §22 op must NOT decline to the tree-walker.
            SchedStop::Declined => panic!("scheduled debug engine declined a §22 op"),
        }
    }
}

/// Arm `bps` and run to the next breakpoint (its pc) or completion (`None`).
fn run_to(run: &mut ScheduledDebugRun, bps: &[IrPc], fuel: &mut u64) -> Option<IrPc> {
    run.set_breakpoints(bps.to_vec());
    match run.run_until_stop(fuel) {
        SchedStop::Break { pc, .. } => Some(pc),
        SchedStop::Finished(_) => None,
        other => panic!("unexpected §22 debug stop {other:?}"),
    }
}

/// **old→new via `install`, under the debugger.** Guest `(jit, a, b)`: compile a unit, `install` it,
/// `call.dyn` the returned slot with `(a, b)`. The unit is `(a,b) -> a*b + 100`, so `(6,7) → 142`.
/// Both debug engines must run it to completion (no decline) and agree with the tree-walker oracle;
/// a breakpoint set *after* the `install` op must fire (proving the install was serviced, not declined).
#[test]
fn debug_install_then_call_indirect_agrees() {
    let b = blob(
        "memory 16\nfunc (i32, i32) -> (i32) {\nblock 0 (v0: i32, v1: i32) {\n  \
         v2 = i32.mul v0 v1\n  v3 = i32.const 100\n  v4 = i32.add v2 v3\n  return v4\n  }\n}\n",
    );
    // inst 0: const 20480 · 1: const BLOBLEN · 2: compile · 3: install · 4: wrap · 5: call.dyn.
    let guest_src =
        "memory 16\nfunc (i32, i32, i32) -> (i32) {\nblock 0 (v0: i32, v1: i32, v2: i32) {\n  \
         v3 = i64.const 20480\n  v4 = i64.const BLOBLEN\n  \
         v5 = call.cap 11 0 (i64, i64) -> (i64) v0 (v3, v4)\n  \
         v6 = call.cap 11 3 (i64) -> (i64) v0 (v5)\n  \
         v7 = i32.wrap_i64 v6\n  \
         v8 = call.dyn (i32, i32) -> (i32) v7 (v1, v2)\n  return v8\n  }\n}\n";
    let m = guest_module(guest_src, &b);
    // Reserve a 16-slot table (log2=4): the parent has 1 func, so the unit installs at slot 1.
    let (host, jit) = jit_host(&m, 4);
    let args = [Value::I32(jit), Value::I32(6), Value::I32(7)];

    let want = oracle(&m, &args);
    assert!(
        matches!(&want, Ok(v) if v.as_slice() == [Value::I32(142)]),
        "oracle: 6*7+100 = 142, got {want:?}"
    );

    // Driven to completion: the §22 ops are serviced inline, not declined to the tree-walker.
    let mut run =
        ScheduledDebugRun::new_with_host(&m, 0, &args, host).expect("debug engine drives §22");
    let mut fuel = 50_000_000u64;
    assert_eq!(
        sched_to_end(&mut run, &mut fuel),
        want,
        "debug-engine result must match the tree-walker oracle"
    );

    // A breakpoint at inst 4 (`i32.wrap_i64`, right after the `install` at inst 3) must fire — which
    // is only reachable if the install executed and did not decline to the tree-walker.
    let (host2, _) = jit_host(&m, 4);
    let mut run2 =
        ScheduledDebugRun::new_with_host(&m, 0, &args, host2).expect("debug engine drives §22");
    let after_install = IrPc {
        module: 0,
        func: 0,
        block: 0,
        inst: 4,
    };
    let mut fuel2 = 50_000_000u64;
    assert_eq!(
        run_to(&mut run2, &[after_install], &mut fuel2),
        Some(after_install),
        "breakpoint after the install op must fire (the install was serviced, not declined)"
    );
    // …and continuing from there still reaches the same result.
    run2.set_breakpoints(Vec::new());
    assert_eq!(sched_to_end(&mut run2, &mut fuel2), want.clone());
}

/// **`invoke` under the debugger — result agreement.** Guest `(jit, a, b)`: compile a unit and `invoke`
/// it with `(a, b)`. The unit is `(a,b) -> a+b`, so `(6,7) → 13`. Run to completion: both engines step
/// *into* the invoked unit (see `debug_invoke_step_into_breakpoint`) and the result matches the oracle
/// (the unit is a seam-free leaf over the caller's window either way).
#[test]
fn debug_invoke_agrees() {
    let b = blob(
        "memory 16\nfunc (i32, i32) -> (i32) {\nblock 0 (v0: i32, v1: i32) {\n  \
         v2 = i32.add v0 v1\n  return v2\n  }\n}\n",
    );
    let guest_src =
        "memory 16\nfunc (i32, i32, i32) -> (i32) {\nblock 0 (v0: i32, v1: i32, v2: i32) {\n  \
         v3 = i64.const 20480\n  v4 = i64.const BLOBLEN\n  \
         v5 = call.cap 11 0 (i64, i64) -> (i64) v0 (v3, v4)\n  \
         v6 = call.cap 11 1 (i64, i32, i32) -> (i32) v0 (v5, v1, v2)\n  return v6\n  }\n}\n";
    let m = guest_module(guest_src, &b);
    let (host, jit) = jit_host(&m, 0); // invoke needs no install table
    let args = [Value::I32(jit), Value::I32(6), Value::I32(7)];

    let want = oracle(&m, &args);
    assert!(
        matches!(&want, Ok(v) if v.as_slice() == [Value::I32(13)]),
        "oracle: 6+7 = 13, got {want:?}"
    );

    let mut run = ScheduledDebugRun::new_with_host(&m, 0, &args, host)
        .expect("debug engine drives §22 invoke");
    let mut fuel = 50_000_000u64;
    assert_eq!(
        sched_to_end(&mut run, &mut fuel),
        want,
        "invoked-unit result must match the oracle"
    );
}

/// **Step *into* an invoked unit** — on both engines (#1517 slice 3: the scheduled engine used to keep
/// invoke an opaque leaf). A breakpoint set at an op **inside** the invoked unit (module ≥ 1 — the unit
/// is `source.push`ed at invoke time) must fire, and the reported stop `IrPc` must be in the unit's
/// module, not the caller's — i.e. the debugger descends into `Jit.invoke`. The unit is
/// `(a,b) -> a + b + 100` (inst 0 `add`, inst 1 `const 100`, inst 2 `add`), so a breakpoint at inst 1
/// fires only if inst 0 executed *inside* the unit; continuing yields `6 + 7 + 100 = 113`, matching
/// the oracle. A `step_out` from inside the unit lands back in the caller (the cumulative depth).
#[test]
fn debug_invoke_step_into_breakpoint() {
    let b = blob(
        "memory 16\nfunc (i32, i32) -> (i32) {\nblock 0 (v0: i32, v1: i32) {\n  \
         v2 = i32.add v0 v1\n  v3 = i32.const 100\n  v4 = i32.add v2 v3\n  return v4\n  }\n}\n",
    );
    let guest_src =
        "memory 16\nfunc (i32, i32, i32) -> (i32) {\nblock 0 (v0: i32, v1: i32, v2: i32) {\n  \
         v3 = i64.const 20480\n  v4 = i64.const BLOBLEN\n  \
         v5 = call.cap 11 0 (i64, i64) -> (i64) v0 (v3, v4)\n  \
         v6 = call.cap 11 1 (i64, i32, i32) -> (i32) v0 (v5, v1, v2)\n  \
         v7 = i32.add v6 v2\n  return v7\n  }\n}\n";
    let m = guest_module(guest_src, &b);
    let (host, jit) = jit_host(&m, 0);
    let args = [Value::I32(jit), Value::I32(6), Value::I32(7)];

    let want = oracle(&m, &args);
    assert!(
        matches!(&want, Ok(v) if v.as_slice() == [Value::I32(120)]),
        "oracle: (6+7+100) + 7 = 120, got {want:?}"
    );

    // The invoked unit is pushed to module 1 (the guest is module 0, no installs). Break at inst 1
    // (`i32.const 100`) — reachable only by stepping into the unit past its first op.
    let inside_unit = IrPc {
        module: 1,
        func: 0,
        block: 0,
        inst: 1,
    };
    let mut run = ScheduledDebugRun::new_with_host(&m, 0, &args, host)
        .expect("debug engine drives §22 invoke");
    let mut fuel = 50_000_000u64;
    assert_eq!(
        run_to(&mut run, &[inside_unit], &mut fuel),
        Some(inside_unit),
        "breakpoint INSIDE the invoked unit (module 1) must fire — the debugger stepped into invoke"
    );
    // The backtrace at the stop is inside the unit: the top frame's module is the unit's, not module 0.
    assert_eq!(
        run.frame_pc(0).map(|pc| pc.module),
        Some(1),
        "the running frame at the breakpoint is the invoked unit's (module 1)"
    );
    // Step out of the unit: back in the caller (module 0), then to completion, matching the oracle.
    run.set_breakpoints(Vec::new());
    assert!(
        matches!(
            run.step_out(&mut fuel),
            SchedStop::Break { pc, reason: SchedBreak::Step } if pc.module == 0
        ),
        "step_out from inside the unit lands in the caller"
    );
    assert_eq!(sched_to_end(&mut run, &mut fuel), want.clone());
}

/// **Fail-closed under the debugger.** A `Jit.install` / `Jit.invoke` of a **forged** code handle
/// (never minted by `compile`) resolves no unit — `Host::resolve_jit_unit`, which every driver's
/// `Jit` service starts from, returns a trap, exactly as on the production `drive`. The debug engines must trap identically to the
/// tree-walker oracle (not decline, not diverge). Pins the error path the happy-path tests don't reach.
#[test]
fn debug_forged_handle_traps_identically() {
    // A staged blob is irrelevant here (the forged handle is never resolved), but keep the shape.
    let b = blob(
        "memory 16\nfunc () -> (i32) {\nblock 0 () {\n  v0 = i32.const 1\n  return v0\n  }\n}\n",
    );

    // `(jit) -> i32`: install a bogus code handle (7777) — the call.cap traps; the run never returns.
    let install_src = "memory 16\nfunc (i32) -> (i32) {\nblock 0 (v0: i32) {\n  \
         v1 = i64.const 7777\n  v2 = call.cap 11 3 (i64) -> (i64) v0 (v1)\n  \
         v3 = i32.wrap_i64 v2\n  return v3\n  }\n}\n";
    // …and the same for invoke (op 1).
    let invoke_src = "memory 16\nfunc (i32) -> (i32) {\nblock 0 (v0: i32) {\n  \
         v1 = i64.const 7777\n  v2 = call.cap 11 1 (i64) -> (i32) v0 (v1)\n  return v2\n  }\n}\n";

    for src in [install_src, invoke_src] {
        let m = guest_module(src, &b);
        let (host, jit) = jit_host(&m, 4);
        let args = [Value::I32(jit)];

        let want = oracle(&m, &args);
        assert!(
            want.is_err(),
            "forged handle must trap on the oracle, got {want:?}"
        );

        let mut run =
            ScheduledDebugRun::new_with_host(&m, 0, &args, host).expect("debug engine builds");
        let mut fuel = 50_000_000u64;
        assert_eq!(
            sched_to_end(&mut run, &mut fuel),
            want,
            "the debug engine must trap identically to the oracle (forged handle)"
        );
    }
}

/// Guest `(jit)`: a `compile` of 64 zero bytes (rejected, but its bytes are charged to the quota), a
/// 20-iteration loop, a `compile` of the real blob, and another 20-iteration loop. Returns the second
/// compile's result: a code handle, or `-ENOMEM` when the quota left after the first is too small.
const QUOTA_GUEST: &str = "memory 16\nfunc (i32) -> (i64) {\nblock 0 (v0: i32) {\n  \
     v1 = i64.const 24576\n  v2 = i64.const 64\n  \
     v3 = call.cap 11 0 (i64, i64) -> (i64) v0 (v1, v2)\n  \
     v4 = i32.const 20\n  br 1(v0, v4)\n}\n\
     block 1 (vj: i32, vk: i32) {\n  vm1 = i32.const -1\n  vn = i32.add vk vm1\n  \
     br_if vn 1(vj, vn) 2(vj)\n}\n\
     block 2 (vj2: i32) {\n  v5 = i64.const 20480\n  v6 = i64.const BLOBLEN\n  \
     v7 = call.cap 11 0 (i64, i64) -> (i64) vj2 (v5, v6)\n  v8 = i32.const 20\n  br 3(v7, v8)\n}\n\
     block 3 (vr: i64, vt: i32) {\n  vm2 = i32.const -1\n  vt2 = i32.add vt vm2\n  \
     br_if vt2 3(vr, vt2) 4(vr)\n}\n\
     block 4 (vr2: i64) {\n  return vr2\n  }\n}\n";

fn quota_guest() -> temen_ir::Module {
    guest_module(
        QUOTA_GUEST,
        &blob(
            "memory 16\nfunc (i32, i32) -> (i32) {\nblock 0 (v0: i32, v1: i32) {\n  \
             v2 = i32.add v0 v1\n  return v2\n  }\n}\n",
        ),
    )
}

/// A debug run of [`quota_guest`] whose quota holds one byte less than both compiles together, so
/// the second fails only if the first one's charge is still on the books.
fn quota_run(m: &temen_ir::Module) -> ScheduledDebugRun {
    let blob_len = m.data.last().expect("the blob segment").bytes.len() as u64;
    let (mut host, jit) = jit_host(m, 0);
    host.set_jit_quota(4, 64 + blob_len - 1);
    ScheduledDebugRun::new_with_host(m, 0, &[Value::I32(jit)], host).expect("in the debug subset")
}

/// **#2015 — a granted `Jit` table with no unit in it does not refuse a checkpoint**, and the compile
/// quota rides it. The guest's first `compile` is rejected but charges its bytes; a checkpoint taken
/// after it, restored into a freshly built run, must still leave the second `compile` short of quota,
/// as the uninterrupted run is. Before, the granted table alone refused every checkpoint.
#[test]
fn a_jit_table_without_units_checkpoints_and_keeps_its_quota() {
    let m = quota_guest();
    let mut fuel = 50_000_000u64;
    let want = sched_to_end(&mut quota_run(&m), &mut fuel);
    assert_eq!(
        want,
        Ok(vec![Value::I64(-12)]),
        "the rejected compile's charge leaves the second one short (-ENOMEM)"
    );

    let mut run = quota_run(&m);
    let mut fuel = 50_000_000u64;
    while run.op_turn() < 30 && run.tick(&mut fuel) {}
    let turn = run.op_turn();
    let snap = run
        .snapshot()
        .expect("a granted, unit-less Jit table is checkpointable");

    let mut warm = quota_run(&m);
    warm.restore(turn, &snap);
    let mut fuel = 50_000_000u64;
    assert_eq!(
        sched_to_end(&mut warm, &mut fuel),
        want,
        "restored at turn {turn}, the run ends as the uninterrupted one does"
    );
}

/// The other side of #2015: once a unit is compiled the table holds a `JitCode` handle a restore
/// cannot rebuild, so the run stops being checkpointable.
#[test]
fn a_compiled_unit_still_refuses_a_checkpoint() {
    let m = quota_guest();
    let (host, jit) = jit_host(&m, 0); // the default quota: both compiles fit
    let mut run =
        ScheduledDebugRun::new_with_host(&m, 0, &[Value::I32(jit)], host).expect("in the subset");
    let mut fuel = 50_000_000u64;
    while run.op_turn() < 30 && run.tick(&mut fuel) {}
    assert!(run.snapshot().is_ok(), "no unit yet");
    assert!(
        matches!(sched_to_end(&mut run, &mut fuel), Ok(v) if matches!(v[..], [Value::I64(h)] if h > 0)),
        "the second compile succeeds"
    );
    assert_eq!(
        run.snapshot().err(),
        Some(Refusal::JitUnit),
        "a unit is held: refused"
    );
}

/// **#2015, the journal's half.** The undo journal restores into the same run, so it cannot take back
/// a `compile`: the unit and its `JitCode` handle stay, and a replay would compile again under the
/// next handle (it returned 258 where the run had 257). An undo across a compile declines, and `seek`
/// serves it; one after the last compile still undoes and agrees.
#[test]
fn undo_declines_across_a_compile() {
    let m = quota_guest();
    let fresh = || {
        let (host, jit) = jit_host(&m, 0);
        ScheduledDebugRun::new_with_host(&m, 0, &[Value::I32(jit)], host).expect("in the subset")
    };
    let mut fuel = 50_000_000u64;
    let want = sched_to_end(&mut fresh(), &mut fuel);

    let mut run = fresh();
    run.set_journal_armed(true);
    run.set_journal_policy(temen_interp::journal::JournalPolicy {
        state_stride: 1,
        ..Default::default()
    });
    let mut fuel = 50_000_000u64;
    assert_eq!(sched_to_end(&mut run, &mut fuel), want);
    let end = run.op_turn();
    for t in [1u64, 30] {
        assert!(
            !run.can_undo_to(t),
            "turn {t} is before a compile: declined"
        );
        assert!(!run.undo_to(t), "and undo_to agrees");
    }
    let after = end - 1;
    assert!(
        run.can_undo_to(after),
        "turn {after} is after the last compile"
    );
    assert!(run.undo_to(after));
    assert_eq!(
        sched_to_end(&mut run, &mut fuel),
        want,
        "undo to {after}, then on"
    );
}

// #2192 — the debugger steps into an invoked unit through the engines' own nested drive
// (`drive_nested`, one op at a time), so a unit runs under the debugger what it runs on the engines.
// Before, the step-into was a second loop that handled a plain op, the unit's return and a punted cap
// call, and faulted on everything else.

/// A host holding the `Jit` cap with `units` compiled into it: `(host, jit, code handles)`.
fn compiled(m: &temen_ir::Module, units: &[&str]) -> (Host, i32, Vec<i32>) {
    let mut host = Host::new();
    let jit = grant_jit(&mut host, m, 0);
    let codes = units
        .iter()
        .map(|src| {
            host.jit_compile(jit, &blob(src))
                .expect("no trap")
                .expect("compile ok")
                .handle
        })
        .collect();
    (host, jit, codes)
}

/// `guest`'s function 0 on the tree-walker, the cooperative pump and the debugger, each over a host
/// with `units` compiled and the arguments `args(jit, codes)`.
fn three_ways(
    guest: &str,
    units: &[&str],
    args: impl Fn(i32, &[i32]) -> Vec<Value>,
) -> [Result<Vec<Value>, Trap>; 3] {
    let m = parse_module(guest).expect("parse guest");
    verify_module(&m).expect("verify guest");
    let (mut host, jit, codes) = compiled(&m, units);
    let mut fuel = 50_000_000u64;
    let tree = run_with_host(&m, 0, &args(jit, &codes), &mut fuel, &mut host);
    let (mut host, jit, codes) = compiled(&m, units);
    let mut fuel = 50_000_000u64;
    let pump = temen_interp::bytecode::compile_and_run_with_host(
        &m,
        0,
        &args(jit, &codes),
        &mut fuel,
        &mut host,
    )
    .expect("the bytecode engine runs the module");
    let (host, jit, codes) = compiled(&m, units);
    let mut run = ScheduledDebugRun::new_with_host(&m, 0, &args(jit, &codes), host)
        .expect("the debugger runs the module");
    let mut fuel = 50_000_000u64;
    let debug = sched_to_end(&mut run, &mut fuel);
    [tree, pump, debug]
}

/// A unit that invokes a second unit (#1334): the guest invokes unit 1 with `(jit, unit 2)`, and unit
/// 1 invokes unit 2 with `(6, 7)`, which adds them. The debugger steps over the inner invoke, as a
/// step over any call: its frames are the inner unit's, run to its end in one step.
#[test]
fn a_unit_the_debugger_steps_into_can_invoke_another() {
    let guest = "memory 16
func (i32, i32, i32) -> (i64) {
block 0 (vj: i32, vc1: i32, vc2: i32) {
  vj64 = i64.extend_i32_u vj
  vc164 = i64.extend_i32_u vc1
  vc264 = i64.extend_i32_u vc2
  vr = call.cap 11 1 (i64, i64, i64) -> (i64) vj (vc164, vj64, vc264)
  return vr
  }
}
";
    let outer = "memory 16
func (i64, i64) -> (i64) {
block 0 (vj: i64, vc2: i64) {
  vj32 = i32.wrap_i64 vj
  v6 = i64.const 6
  v7 = i64.const 7
  vr = call.cap 11 1 (i64, i64, i64) -> (i64) vj32 (vc2, v6, v7)
  return vr
  }
}
";
    let inner = "memory 16
func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  vr = i64.add va vb
  return vr
  }
}
";
    let ran = three_ways(guest, &[outer, inner], |jit, codes| {
        vec![Value::I32(jit), Value::I32(codes[0]), Value::I32(codes[1])]
    });
    assert_eq!(
        ran,
        [
            Ok(vec![Value::I64(13)]),
            Ok(vec![Value::I64(13)]),
            Ok(vec![Value::I64(13)])
        ]
    );
}

/// A unit that hosts a fiber (#845, the unit `invoke_fibers.rs` pins on the engines): it starts a
/// fiber over program function 1 by its natural-table slot and resumes it twice. The fiber suspends
/// `5 + 777` and then returns the second resume's `5`, so the unit returns 787.
#[test]
fn a_unit_the_debugger_steps_into_can_host_a_fiber() {
    let guest = "memory 16
func (i32, i32) -> (i64) {
block 0 (vj: i32, vc: i32) {
  vc64 = i64.extend_i32_u vc
  vx = i64.const 5
  vr = call.cap 11 1 (i64, i64) -> (i64) vj (vc64, vx)
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vk = i64.const 777
  vs = i64.add varg vk
  vv = suspend vs
  return vv
  }
}
";
    let unit = "memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vf = i32.const 1
  vsp = i64.const 0
  vk = cont.new vf vsp
  vs1, vv1 = cont.resume vk v0
  vs2, vv2 = cont.resume vk v0
  vr = i64.add vv1 vv2
  return vr
  }
}
";
    let ran = three_ways(guest, &[unit], |jit, codes| {
        vec![Value::I32(jit), Value::I32(codes[0])]
    });
    assert_eq!(
        ran,
        [
            Ok(vec![Value::I64(787)]),
            Ok(vec![Value::I64(787)]),
            Ok(vec![Value::I64(787)])
        ]
    );
}

/// A host-completed cap call inside an invoked unit: only the embedder completes it, and nothing can
/// while the unit runs, so the pump declines it with `CapFault` (#1954). The debugger used to wait on
/// it forever; it now declines as the pump does. The debug run happens on a thread with a bounded
/// wait, so a regression fails instead of hanging the suite.
#[test]
fn a_host_completed_call_in_a_unit_the_debugger_steps_into_declines() {
    let guest = "memory 16
func (i32, i32, i32) -> (i64) {
block 0 (vj: i32, vc: i32, vh: i32) {
  vc64 = i64.extend_i32_u vc
  vh64 = i64.extend_i32_u vh
  vr = call.cap 11 1 (i64, i64) -> (i64) vj (vc64, vh64)
  return vr
  }
}
";
    let unit = "memory 16
func (i64) -> (i64) {
block 0 (vh: i64) {
  vh32 = i32.wrap_i64 vh
  v7 = i64.const 7
  vr = call.cap 13 1 (i64) -> (i64) vh32 (v7)
  return vr
  }
}
";
    let m = parse_module(guest).expect("parse guest");
    verify_module(&m).expect("verify guest");
    // The `Jit` cap and an offloadable host procedure whose op 1 only the embedder completes.
    let setup = |m: &temen_ir::Module| {
        let (mut host, jit, codes) = compiled(m, &[unit]);
        let proc = host.grant_host_proc_offloadable(
            Box::new(|_, _| temen_interp::OffloadOutcome::Host(Box::new(|_| ()))),
            temen_interp::CapState::Stateless,
        );
        let args = vec![Value::I32(jit), Value::I32(codes[0]), Value::I32(proc)];
        (host, args)
    };
    let (mut host, args) = setup(&m);
    let mut fuel = 50_000_000u64;
    let pump =
        temen_interp::bytecode::compile_and_run_with_host(&m, 0, &args, &mut fuel, &mut host)
            .expect("the bytecode engine runs the module");
    assert_eq!(pump, Err(Trap::CapFault), "the pump declines");

    let (done, debugged) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let (host, args) = setup(&m);
        let mut run =
            ScheduledDebugRun::new_with_host(&m, 0, &args, host).expect("the debugger runs it");
        let mut fuel = 50_000_000u64;
        let _ = done.send(sched_to_end(&mut run, &mut fuel));
    });
    let debug = debugged
        .recv_timeout(std::time::Duration::from_secs(60))
        .expect("the debugger answers instead of waiting on a call nothing can complete");
    assert_eq!(
        debug,
        Err(Trap::CapFault),
        "the debugger declines as the pump does"
    );
}
