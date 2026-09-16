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
/// [`BLOB_OFF`] — the debug engines seed memory from the module's data segments (`build_mem`), so this
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
/// (never minted by `compile`) resolves no unit — the authority arm in `dbg_jit_install`/`dbg_jit_invoke`
/// returns a trap, exactly as the production `drive`. The debug engines must trap identically to the
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
