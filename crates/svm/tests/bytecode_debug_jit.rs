//! Debugging the §22 guest-driven **`Jit` capability** (iface 11: `compile`/`install`/`uninstall`/
//! `invoke`) on the interpreter debug engines. Before this slice the debug drivers declined every
//! §22 op — the single-vCPU `DebugRun` trapped `Malformed`, the `ScheduledDebugRun` returned
//! `Declined` — so a guest-JIT program fell back to the tree-walker oracle instead of being
//! debuggable over bytecode. Now the ops are serviced **inline in `debug_advance_fiber`** (they mutate
//! only the stepping vCPU + the shared dispatch table, spawning no scheduler task), so both engines
//! step a §22 program op-by-op, breakpoints fire around the ops, and the result stays bit-identical to
//! the tree-walker oracle + the production bytecode engine.
//!
//! `invoke` runs the submitted unit to completion as a **seam-free leaf** (stepping *over* it, matching
//! production `run_invoke`); `install` + `call_indirect` steps op-by-op like any module-≥1 frame.

use svm_encode::encode_module;
use svm_interp::bytecode::{DebugRun, SchedStop, ScheduledDebugRun};
use svm_interp::{run_with_host, Host, IrPc, Trap, Value};
use svm_ir::Data;
use svm_run::grant_jit;
use svm_text::parse_module;
use svm_verify::verify_module;

/// Where the guest reads the submitted blob from (matches the `i64.const 4096` in the guests below).
const BLOB_OFF: u64 = 4096;

/// Encode `src` as the binary unit blob a guest submits to `Jit.compile`.
fn blob(src: &str) -> Vec<u8> {
    let m = parse_module(src).expect("parse blob");
    verify_module(&m).expect("verify blob");
    encode_module(&m)
}

/// Parse `guest_src` (with `BLOBLEN` replaced by `b.len()`) and inject `b` as a data segment at
/// [`BLOB_OFF`] — the debug engines seed memory from the module's data segments (`build_mem`), so this
/// is how the blob reaches the guest (the tree-walker seeds the same segment, keeping them identical).
fn guest_module(guest_src: &str, b: &[u8]) -> svm_ir::Module {
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
fn jit_host(m: &svm_ir::Module, table_log2: u8) -> (Host, i32) {
    let mut host = Host::new();
    let jit = grant_jit(&mut host, m, table_log2);
    (host, jit)
}

/// The tree-walker oracle result (`run_with_host` seeds the injected data segment like the debug run).
fn oracle(m: &svm_ir::Module, args: &[Value]) -> Result<Vec<Value>, Trap> {
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
            SchedStop::Blocked => return Err(Trap::Malformed),
            // The whole point of this slice: a §22 op must NOT decline to the tree-walker.
            SchedStop::Declined => panic!("scheduled debug engine declined a §22 op"),
        }
    }
}

/// **old→new via `install`, under the debugger.** Guest `(jit, a, b)`: compile a unit, `install` it,
/// `call_indirect` the returned slot with `(a, b)`. The unit is `(a,b) -> a*b + 100`, so `(6,7) → 142`.
/// Both debug engines must run it to completion (no decline) and agree with the tree-walker oracle;
/// a breakpoint set *after* the `install` op must fire (proving the install was serviced, not declined).
#[test]
fn debug_install_then_call_indirect_agrees() {
    let b = blob(
        "memory 16\nfunc (i32, i32) -> (i32) {\nblock 0 (v0: i32, v1: i32) {\n  \
         v2 = i32.mul v0 v1\n  v3 = i32.const 100\n  v4 = i32.add v2 v3\n  return v4\n  }\n}\n",
    );
    // inst 0: const 4096 · 1: const BLOBLEN · 2: compile · 3: install · 4: wrap · 5: call_indirect.
    let guest_src =
        "memory 16\nfunc (i32, i32, i32) -> (i32) {\nblock 0 (v0: i32, v1: i32, v2: i32) {\n  \
         v3 = i64.const 4096\n  v4 = i64.const BLOBLEN\n  \
         v5 = cap.call 11 0 (i64, i64) -> (i64) v0 (v3, v4)\n  \
         v6 = cap.call 11 3 (i64) -> (i64) v0 (v5)\n  \
         v7 = i32.wrap_i64 v6\n  \
         v8 = call_indirect (i32, i32) -> (i32) v7 (v1, v2)\n  return v8\n  }\n}\n";
    let m = guest_module(guest_src, &b);
    // Reserve a 16-slot table (log2=4): the parent has 1 func, so the unit installs at slot 1.
    let (host, jit) = jit_host(&m, 4);
    let args = [Value::I32(jit), Value::I32(6), Value::I32(7)];

    let want = oracle(&m, &args);
    assert!(
        matches!(&want, Ok(v) if v.as_slice() == [Value::I32(142)]),
        "oracle: 6*7+100 = 142, got {want:?}"
    );

    // Single-vCPU DebugRun, driven to completion (a §22 program has no threads).
    let mut run = DebugRun::new_with_host(&m, 0, &args, host).expect("debug engine drives §22");
    let mut fuel = 50_000_000u64;
    assert_eq!(
        run.run_to(&[], &mut fuel),
        None,
        "run to completion (no breakpoints), not a mid-run decline"
    );
    assert_eq!(
        run.result().cloned(),
        Some(want.clone()),
        "single-vCPU DebugRun result must match the tree-walker oracle"
    );

    // A breakpoint at inst 4 (`i32.wrap_i64`, right after the `install` at inst 3) must fire — which
    // is only reachable if the install executed and did not decline to the tree-walker.
    let (host2, _) = jit_host(&m, 4);
    let mut run2 = DebugRun::new_with_host(&m, 0, &args, host2).expect("debug engine drives §22");
    let after_install = IrPc {
        module: 0,
        func: 0,
        block: 0,
        inst: 4,
    };
    let mut fuel2 = 50_000_000u64;
    assert_eq!(
        run2.run_to(&[after_install], &mut fuel2),
        Some(after_install),
        "breakpoint after the install op must fire (the install was serviced, not declined)"
    );
    // …and continuing from there still reaches the same result.
    assert_eq!(run2.run_to(&[], &mut fuel2), None);
    assert_eq!(run2.result().cloned(), Some(want.clone()));

    // The scheduled engine must service it too (same guest; no threads, but proves the twin path).
    let (host3, _) = jit_host(&m, 4);
    let mut sched = ScheduledDebugRun::new_with_host(&m, 0, &args, host3)
        .expect("scheduled debug engine drives §22");
    let mut fuel3 = 50_000_000u64;
    assert_eq!(
        sched_to_end(&mut sched, &mut fuel3),
        want,
        "ScheduledDebugRun result must match the oracle"
    );
}

/// **`invoke` under the debugger — seam-free leaf.** Guest `(jit, a, b)`: compile a unit and `invoke`
/// it with `(a, b)`. The unit is `(a,b) -> a+b`, so `(6,7) → 13`. Stepping *over* `Jit.invoke` runs the
/// unit atomically (no step-into); the result matches the oracle on both engines.
#[test]
fn debug_invoke_leaf_agrees() {
    let b = blob(
        "memory 16\nfunc (i32, i32) -> (i32) {\nblock 0 (v0: i32, v1: i32) {\n  \
         v2 = i32.add v0 v1\n  return v2\n  }\n}\n",
    );
    let guest_src =
        "memory 16\nfunc (i32, i32, i32) -> (i32) {\nblock 0 (v0: i32, v1: i32, v2: i32) {\n  \
         v3 = i64.const 4096\n  v4 = i64.const BLOBLEN\n  \
         v5 = cap.call 11 0 (i64, i64) -> (i64) v0 (v3, v4)\n  \
         v6 = cap.call 11 1 (i64, i32, i32) -> (i32) v0 (v5, v1, v2)\n  return v6\n  }\n}\n";
    let m = guest_module(guest_src, &b);
    let (host, jit) = jit_host(&m, 0); // invoke needs no install table
    let args = [Value::I32(jit), Value::I32(6), Value::I32(7)];

    let want = oracle(&m, &args);
    assert!(
        matches!(&want, Ok(v) if v.as_slice() == [Value::I32(13)]),
        "oracle: 6+7 = 13, got {want:?}"
    );

    let mut run =
        DebugRun::new_with_host(&m, 0, &args, host).expect("debug engine drives §22 invoke");
    let mut fuel = 50_000_000u64;
    assert_eq!(run.run_to(&[], &mut fuel), None, "invoke run to completion");
    assert_eq!(
        run.result().cloned(),
        Some(want.clone()),
        "invoked-unit result must match the oracle (seam-free leaf)"
    );

    let (host2, _) = jit_host(&m, 0);
    let mut sched = ScheduledDebugRun::new_with_host(&m, 0, &args, host2)
        .expect("scheduled debug engine drives §22 invoke");
    let mut fuel2 = 50_000_000u64;
    assert_eq!(sched_to_end(&mut sched, &mut fuel2), want);
}
