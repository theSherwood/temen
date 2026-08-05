//! §4 durability × §22 guest-JIT, **Slice 1 — instrument-on-submit** (DURABILITY.md §12.5). A
//! **durable** `Jit` grant (`grant_jit_durable`) installs a validator that runs the durable transform
//! (`svm_durable::transform_module`) on each submitted unit before verify, so a durable domain
//! **admits** the (instrumented) unit instead of failing closed — the "host runs the pass on submitted
//! IR" composition. Without the durable grant a durable domain still refuses `compile`; a unit outside
//! the strict transform's scope (a guest-memory op) fails closed.
//!
//! Slice 1 covers *compile-side* composition + a NORMAL-state end-to-end run.
//!
//! **Slice 2 — persist `JitCode`/`JitDomain` across a snapshot**
//! (`durable_jit_domain_survives_freeze_and_invokes`): a durable domain's compiled units ride the
//! artifact (svm-snapshot Section 5), and after restore the re-pinned handles invoke the
//! reconstructed (re-verified) unit on the interpreter.
//!
//! **Slice 3 — native reconstruct-on-thaw** (`…reconstructs_and_invokes_native`): `jit_cap_run`
//! re-compiles a restored unit into the fresh module, so the native tier invokes its own code ≡ 42.
//!
//! **Install-durability** (`durable_jit_install_slot_survives_freeze_thaw`(`_native`)): a unit's B2
//! `install` occupancy rides the artifact (Section 5 / v17) and is re-applied on thaw, so a
//! `call_indirect` through an installed slot resolves after a freeze/thaw on both backends.
//!
//! **In-flight continuation** (`durable_jit_install_call_indirect_freezes_in_flight_continuation`):
//! `invoke` is a seam-free atomic leaf (never interrupted mid-flight), so the freezable path for
//! suspendable unit code is `install` + `call_indirect` — the unit runs in the caller's durable
//! frames, so a continuation suspended inside it freezes/thaws with the caller's shadow stack.
//!
//! **Install fence** (`durable_jit_compile_fences_suspending_untainted_unit`): the by-signature-taint
//! gap (R8) — a durable domain rejects a *suspendable* unit whose entry signature the program does not
//! taint at `compile`, so it can never be installed and reached by an un-instrumented `call_indirect`.

use svm_durable::{
    begin_thaw, init_durable_window, transform_module, write_state, STATE_NORMAL, STATE_UNWINDING,
};
use svm_encode::encode_module;
use svm_interp::{run_capture_reserved_with_host, Host, Value};
use svm_jit::JitOutcome;
use svm_run::{grant_jit, grant_jit_durable, jit_cap_run};
use svm_snapshot::{freeze, restore};
use svm_text::parse_module;
use svm_verify::verify_module;

const SIZE_LOG2: u8 = 17; // 128 KiB ≥ the durable reserve (64 KiB)
const WINDOW: usize = 1 << SIZE_LOG2;
const BLOB_OFF: usize = 0x1_1000; // above DURABLE_RESERVE (64 KiB) — the guest usable region

/// Encode+verify a unit blob a guest submits to `Jit.compile`.
fn blob(src: &str) -> Vec<u8> {
    let m = parse_module(src).expect("parse blob");
    verify_module(&m).expect("verify blob");
    encode_module(&m)
}

/// A memory-declaring guest (so `grant_*` mints the domain against `size_log2 = 17`, matching the
/// units) with a trivial entry — used only to set up the domain for the host-API compile tests.
fn dummy_guest() -> svm_ir::Module {
    parse_module(
        "memory 17\nfunc () -> (i64) {\nblock 0 () {\n  v0 = i64.const 0\n  return v0\n  }\n}\n",
    )
    .expect("parse guest")
}

/// **The Slice-1 mechanism.** A durable domain admits a submitted unit *iff* the grant hosts durability
/// (which instruments it); a non-durable domain admits it unconditionally; a durable domain **without**
/// the durable grant refuses (the un-instrumented-unit fail-closed).
#[test]
fn durable_grant_admits_submitted_unit() {
    // An in-scope unit: declares memory 17 (memory-match), non-suspending, returns 42. The transform
    // returns a non-suspending function unchanged, so it is admitted and invoke-equivalent.
    let unit = blob(
        "memory 17\nfunc () -> (i64) {\nblock 0 () {\n  v0 = i64.const 42\n  return v0\n  }\n}\n",
    );
    let guest = dummy_guest();

    // (a) durable + durable grant → admitted (instrumented).
    let mut h = Host::new();
    h.set_durable(true);
    let jit = grant_jit_durable(&mut h, &guest, 0);
    assert!(
        matches!(h.jit_compile(jit, &unit), Ok(Ok(_))),
        "a durable domain with grant_jit_durable must admit an in-scope unit"
    );

    // (b) durable + plain grant → refused (the gate: durable && !jit_hosts_durable).
    let mut h2 = Host::new();
    h2.set_durable(true);
    let jit2 = grant_jit(&mut h2, &guest, 0);
    assert!(
        matches!(h2.jit_compile(jit2, &unit), Ok(Err(-22))),
        "a durable domain WITHOUT the durable grant must refuse compile (fail-closed)"
    );

    // (c) non-durable + plain grant → admitted (the pre-existing baseline, unchanged).
    let mut h3 = Host::new();
    let jit3 = grant_jit(&mut h3, &guest, 0);
    assert!(
        matches!(h3.jit_compile(jit3, &unit), Ok(Ok(_))),
        "a non-durable domain admits the unit as before"
    );
}

/// A unit outside the **strict** transform's scope fails closed on a durable grant, exactly like any
/// other rejected blob. Here a **may-suspend** function (it does a `cap.call`) that also touches guest
/// memory hits `GuestUsesMemory` — the strict path won't instrument a memory-using suspend point (it
/// could alias the reserved durable slice). (Admitting confined memory use via
/// `transform_module_assume_confined` is a later refinement.)
#[test]
fn durable_grant_rejects_memory_touching_unit() {
    let unit = blob(
        "memory 17\nfunc (i32) -> (i64) {\nblock 0 (v0: i32) {\n  \
         vaddr = i64.const 65536\n  v1 = i64.load vaddr\n  \
         v2 = cap.call 2 0 () -> (i64) v0 ()\n  v3 = i64.add v1 v2\n  return v3\n  }\n}\n",
    );
    let guest = dummy_guest();
    let mut h = Host::new();
    h.set_durable(true);
    let jit = grant_jit_durable(&mut h, &guest, 0);
    assert!(
        matches!(h.jit_compile(jit, &unit), Ok(Err(-22))),
        "a memory-touching unit is outside the strict transform scope → fail closed"
    );
}

/// **End-to-end, NORMAL state.** A durable guest `compile`s + `invoke`s a unit; the instrumented unit
/// runs over the durable window (state NORMAL) and returns 42 — the same value a non-durable run of the
/// same guest yields. The guest is itself durable-instrumented (a single block with two `cap.call`s,
/// the shape the durable tests use); the submitted unit is instrumented by the durable validator.
#[test]
fn durable_run_compiles_and_invokes_agrees() {
    // The submitted unit: `() -> i64` returning 42 (non-suspending; declares memory 17 to match).
    let unit = blob(
        "memory 17\nfunc () -> (i64) {\nblock 0 () {\n  v0 = i64.const 42\n  return v0\n  }\n}\n",
    );

    // Guest `(jit) -> i64`: compile the unit staged at BLOB_OFF, then invoke it. Single block, two
    // cap.calls, return — in the durable transform's shape. The blob ptr is above DURABLE_RESERVE.
    let guest_src = format!(
        "memory 17\nfunc (i32) -> (i64) {{\nblock 0 (v0: i32) {{\n  \
         v1 = i64.const {off}\n  v2 = i64.const {len}\n  \
         v3 = cap.call 11 0 (i64, i64) -> (i64) v0 (v1, v2)\n  \
         v4 = cap.call 11 1 (i64) -> (i64) v0 (v3)\n  return v4\n  }}\n}}\n",
        off = BLOB_OFF,
        len = unit.len(),
    );
    let guest = parse_module(&guest_src).expect("parse guest");
    verify_module(&guest).expect("verify guest");

    // --- Non-durable baseline: the same guest, plain grant, ordinary window → 42.
    let mut init_plain = vec![0u8; WINDOW];
    init_plain[BLOB_OFF..BLOB_OFF + unit.len()].copy_from_slice(&unit);
    let mut hp = Host::new();
    let jp = grant_jit(&mut hp, &guest, 0);
    let mut fuel = 5_000_000u64;
    let (rp, _) = run_capture_reserved_with_host(
        &guest,
        0,
        &[Value::I32(jp)],
        &mut fuel,
        &init_plain,
        SIZE_LOG2,
        &mut hp,
    );
    assert_eq!(
        rp.expect("non-durable run ok"),
        vec![Value::I64(42)],
        "baseline: compile+invoke → 42"
    );

    // --- Durable run: instrument the guest, durable window (NORMAL), durable grant, blob staged.
    let inst = transform_module(&guest).expect("guest must be in transform scope");
    verify_module(&inst).expect("instrumented guest verifies");
    let mut win = init_durable_window(WINDOW);
    write_state(&mut win, STATE_NORMAL);
    win[BLOB_OFF..BLOB_OFF + unit.len()].copy_from_slice(&unit);
    let mut hd = Host::new();
    hd.set_durable(true);
    let jd = grant_jit_durable(&mut hd, &guest, 0);
    let mut fuel2 = 5_000_000u64;
    let (rd, _) = run_capture_reserved_with_host(
        &inst,
        0,
        &[Value::I32(jd)],
        &mut fuel2,
        &win,
        SIZE_LOG2,
        &mut hd,
    );
    assert_eq!(
        rd.expect("durable run ok"),
        vec![Value::I64(42)],
        "durable compile+invoke of an instrumented unit → 42, matching the non-durable run"
    );
}

/// **Slice 2, end-to-end.** A durable domain `compile`s a unit, is **frozen and serialized**, and
/// restored into a fresh host; the re-pinned `JitCode` handle then `invoke`s the reconstructed unit
/// on the interpreter → 42. This is the freeze/thaw of the *units themselves* — the artifact carries
/// the domain's instrumented+verified IR (Section 5), and restore rebuilds + re-verifies it, so a
/// guest that held a compiled unit across a checkpoint keeps it (instead of draining it, as Slice 1
/// required). The invoker is the durable-instrumented gate module; the freeze is at NORMAL state
/// (no in-flight continuation), so the thaw is a plain re-entry.
#[test]
fn durable_jit_domain_survives_freeze_and_invokes() {
    // The unit a guest compiled at some earlier point: `() -> i64` returning 42.
    let unit = blob(
        "memory 17\nfunc () -> (i64) {\nblock 0 () {\n  v0 = i64.const 42\n  return v0\n  }\n}\n",
    );

    // The gate module holding the `Jit` cap — an **invoker** `(jit_domain, code_handle) -> i64` that
    // calls `Jit.invoke` (iface 11 op 1) on the domain handle, passing the compiled-code handle. Its
    // single-block/one-cap.call shape is in the durable transform scope.
    let invoker_src = "memory 17\nfunc (i32, i64) -> (i64) {\nblock 0 (v0: i32, v1: i64) {\n  \
         v2 = cap.call 11 1 (i64) -> (i64) v0 (v1)\n  return v2\n  }\n}\n";
    let invoker = parse_module(invoker_src).expect("parse invoker");
    verify_module(&invoker).expect("verify invoker");
    let inst = transform_module(&invoker).expect("invoker in transform scope");
    verify_module(&inst).expect("instrumented invoker verifies");

    // Durable domain: grant + compile the unit (host-API, the state a run would have left behind).
    let mut hd = Host::new();
    hd.set_durable(true);
    let jd = grant_jit_durable(&mut hd, &invoker, 0);
    let code = match hd.jit_compile(jd, &unit) {
        Ok(Ok(c)) => c,
        _ => panic!("durable compile must admit the instrumented unit"),
    };

    // Freeze the domain (NORMAL state — no continuation) and serialize the real artifact. Before
    // Slice 2 this refused (`JitCode`/`JitDomain` were non-durable); now the units ride Section 5.
    let win = {
        let mut w = init_durable_window(WINDOW);
        write_state(&mut w, STATE_NORMAL);
        w
    };
    let artifact = freeze(&inst, &win, &hd).expect("freeze admits the durable JIT domain");

    // Restore into a FRESH durable host: no validator installed — restore rebuilds + re-verifies the
    // unit from the artifact. The handle table re-pins `jd`/`code` at their slots/generations.
    let mut th = Host::new();
    th.set_durable(true);
    let window = restore(&artifact, &inst, &mut th).expect("restore");

    // Invoke the restored unit through the re-pinned handles → 42 (the unit survived the snapshot).
    let mut fuel = 5_000_000u64;
    let (r, _) = run_capture_reserved_with_host(
        &inst,
        0,
        &[Value::I32(jd), Value::I64(code.handle as i64)],
        &mut fuel,
        &window,
        SIZE_LOG2,
        &mut th,
    );
    assert_eq!(
        r.expect("thawed invoke ok"),
        vec![Value::I64(42)],
        "the restored, re-verified unit invokes to 42 through the re-pinned handle"
    );
}

/// **Slice 3, native reconstruct-on-thaw.** The same freeze/restore as above, but the thaw runs on
/// the **Cranelift** backend: `jit_cap_run` re-compiles the restored unit into the fresh
/// `CompiledModule` (`reconstruct_jit_units`) before re-entering the guest, so the invoke runs the
/// unit's *own native code*, not the interpreter reference. Proves the code pointers — dropped at
/// freeze because they are process-local — are re-established on thaw. The result matches the interp
/// thaw (42), the cross-backend durability contract (§12.6).
#[test]
fn durable_jit_domain_reconstructs_and_invokes_native() {
    let unit = blob(
        "memory 17\nfunc () -> (i64) {\nblock 0 () {\n  v0 = i64.const 42\n  return v0\n  }\n}\n",
    );
    let invoker_src = "memory 17\nfunc (i32, i64) -> (i64) {\nblock 0 (v0: i32, v1: i64) {\n  \
         v2 = cap.call 11 1 (i64) -> (i64) v0 (v1)\n  return v2\n  }\n}\n";
    let invoker = parse_module(invoker_src).expect("parse invoker");
    verify_module(&invoker).expect("verify invoker");

    // Durable domain: grant + compile the unit (host-API).
    let mut hd = Host::new();
    hd.set_durable(true);
    let jd = grant_jit_durable(&mut hd, &invoker, 0);
    let code = match hd.jit_compile(jd, &unit) {
        Ok(Ok(c)) => c,
        _ => panic!("durable compile must admit the instrumented unit"),
    };

    // Freeze (NORMAL) + serialize.
    let win = {
        let mut w = init_durable_window(WINDOW);
        write_state(&mut w, STATE_NORMAL);
        w
    };
    let artifact = freeze(&invoker, &win, &hd).expect("freeze admits the durable JIT domain");

    // Restore into a fresh host, then thaw on the **native** backend. `jit_cap_run` compiles the
    // invoker, reconstructs the restored unit into that module, and the guest invokes it natively.
    let mut th = Host::new();
    th.set_durable(true);
    let window = restore(&artifact, &invoker, &mut th).expect("restore");
    let (out, _mem) = jit_cap_run(
        &invoker,
        0,
        &[jd as i64, code.handle as i64],
        &window,
        SIZE_LOG2,
        0,
        &mut th,
    )
    .expect("native thaw run");
    match out {
        JitOutcome::Returned(slots) => assert_eq!(
            slots,
            vec![42],
            "native invoke of the reconstructed unit → 42 (matches the interp thaw)"
        ),
        other => panic!("expected Returned(42) from the native thaw, got {other:?}"),
    }
}

/// A durable program with two entries: **func 0** compiles + B2-`install`s a unit and returns its
/// table slot; **func 1** `call_indirect`s that slot. Shared by the interp + native install-slot
/// durability tests. The unit is `() -> i64` returning 42; the module declares memory 17 to match.
const INSTALLER_CALLER: &str = "memory 17\n\
    func (i32, i64, i64) -> (i64) {\nblock 0 (v0: i32, v1: i64, v2: i64) {\n  \
      v3 = cap.call 11 0 (i64, i64) -> (i64) v0 (v1, v2)\n  \
      v4 = cap.call 11 3 (i64) -> (i64) v0 (v3)\n  return v4\n  }\n}\n\
    func (i32) -> (i64) {\nblock 0 (v0: i32) {\n  \
      v1 = call_indirect () -> (i64) v0 ()\n  return v1\n  }\n}\n";

/// The unit blob the installer compiles + installs.
fn install_unit() -> Vec<u8> {
    blob("memory 17\nfunc () -> (i64) {\nblock 0 () {\n  v0 = i64.const 42\n  return v0\n  }\n}\n")
}

/// A `call_indirect` table reservation with room for the installer's padding slots.
const TABLE_LOG2: u8 = 4;

/// **Install-durability, interpreter.** A guest compiles + `install`s a unit into a `call_indirect`
/// slot, the domain is frozen + restored, and a second entry `call_indirect`s that slot → 42. The
/// dispatch table is a per-run transient (§12.4), so the occupancy is recorded on the domain, rides
/// the artifact (Section 5, v17), and is **re-applied** when the thaw run builds its table.
#[test]
fn durable_jit_install_slot_survives_freeze_thaw() {
    let unit = install_unit();
    let m = parse_module(INSTALLER_CALLER).expect("parse installer/caller");
    verify_module(&m).expect("verify");

    let mut hd = Host::new();
    let jd = grant_jit(&mut hd, &m, TABLE_LOG2);

    // Entry 0: compile + install the unit (staged at BLOB_OFF) → the table slot.
    let mut win = vec![0u8; WINDOW];
    win[BLOB_OFF..BLOB_OFF + unit.len()].copy_from_slice(&unit);
    let mut fuel = 5_000_000u64;
    let (r_inst, snap) = run_capture_reserved_with_host(
        &m,
        0,
        &[
            Value::I32(jd),
            Value::I64(BLOB_OFF as i64),
            Value::I64(unit.len() as i64),
        ],
        &mut fuel,
        &win,
        SIZE_LOG2,
        &mut hd,
    );
    let slot = match r_inst.expect("install ok").as_slice() {
        [Value::I64(s)] if *s >= 0 => *s as i32,
        other => panic!("install returned {other:?}"),
    };

    // Freeze + restore. The install occupancy is captured on the domain and re-granted.
    let artifact = freeze(&m, &snap, &hd).expect("freeze");
    let mut th = Host::new();
    let window = restore(&artifact, &m, &mut th).expect("restore");
    assert_eq!(
        th.jit_installs(0),
        vec![(slot as u32, 0)],
        "the install occupancy round-trips (slot → unit 0)"
    );
    assert_eq!(
        freeze(&m, &window, &th).expect("re-freeze"),
        artifact,
        "re-serialize of a restored install-bearing domain is byte-identical"
    );

    // Entry 1 over the thawed host: call_indirect through the restored slot resolves to the unit.
    let mut fuel2 = 5_000_000u64;
    let (r_call, _) = run_capture_reserved_with_host(
        &m,
        1,
        &[Value::I32(slot)],
        &mut fuel2,
        &window,
        SIZE_LOG2,
        &mut th,
    );
    assert_eq!(
        r_call.expect("call ok"),
        vec![Value::I64(42)],
        "call_indirect through the restored install slot → 42"
    );
}

/// **Install-durability, native.** The same install → freeze → restore, but both entries run on the
/// Cranelift backend: `jit_cap_run` reconstructs the unit and reproduces its `call_indirect` table
/// slot (`reconstruct_jit_units` → `CompiledModule::install_at`), so the thawed `call_indirect`
/// dispatches to the unit's own native code → 42.
#[test]
fn durable_jit_install_slot_survives_freeze_thaw_native() {
    let unit = install_unit();
    let m = parse_module(INSTALLER_CALLER).expect("parse installer/caller");
    verify_module(&m).expect("verify");

    let mut hd = Host::new();
    let jd = grant_jit(&mut hd, &m, TABLE_LOG2);

    // Entry 0 (native): compile + install the unit → the table slot, then snapshot the run window.
    let mut win = vec![0u8; WINDOW];
    win[BLOB_OFF..BLOB_OFF + unit.len()].copy_from_slice(&unit);
    let (out, snap) = jit_cap_run(
        &m,
        0,
        &[jd as i64, BLOB_OFF as i64, unit.len() as i64],
        &win,
        SIZE_LOG2,
        TABLE_LOG2,
        &mut hd,
    )
    .expect("native install run");
    let slot = match out {
        JitOutcome::Returned(ref s) if s.len() == 1 && s[0] >= 0 => s[0] as i32,
        other => panic!("install returned {other:?}"),
    };

    // Freeze + restore.
    let artifact = freeze(&m, &snap, &hd).expect("freeze");
    let mut th = Host::new();
    let window = restore(&artifact, &m, &mut th).expect("restore");
    assert_eq!(
        th.jit_installs(0),
        vec![(slot as u32, 0)],
        "install round-trips"
    );

    // Entry 1 (native) over the thawed host: reconstruct reproduces the slot; call_indirect → 42.
    let (out2, _) = jit_cap_run(
        &m,
        1,
        &[slot as i64],
        &window,
        SIZE_LOG2,
        TABLE_LOG2,
        &mut th,
    )
    .expect("native thaw call run");
    match out2 {
        JitOutcome::Returned(slots) => assert_eq!(
            slots,
            vec![42],
            "native call_indirect through the reproduced install slot → 42"
        ),
        other => panic!("expected Returned(42), got {other:?}"),
    }
}

/// **In-flight continuation across a freeze — the design's freezable path.** `Jit.invoke` is a
/// *seam-free atomic leaf* by design (DESIGN.md §22 / CONSOLIDATION.md §11 — anything but a plain
/// return is a CapFault, the fuzzed hinge), so it is never interrupted mid-flight: a freeze lands at
/// the parent's poll *after* the invoke returns. The **freezable** way to run suspendable guest-JIT
/// code is `install` + `call_indirect`, which executes the installed unit in the **caller's own
/// durable frames** — so an in-flight continuation *inside* an installed unit rides the caller's
/// shadow stack like any other frame and freezes/thaws with it.
///
/// This pins that end-to-end: an instrumented program installs an instrumented unit whose entry
/// suspends on a `cap.call` (Clock), then a run that `call_indirect`s the unit **under UNWINDING**
/// freezes the continuation *inside* the unit. After restore + REWINDING the caller re-issues the
/// `call_indirect` (the install slot survived — install-durability) and the unit rewinds, **reloading
/// the saved clock (42) rather than re-issuing it (0)** → 142, matching an uninterrupted run.
#[test]
fn durable_jit_install_call_indirect_freezes_in_flight_continuation() {
    // The submitted unit: `(clk) -> i64` = Clock.now + 100 — a `cap.call` suspend point.
    let unit = blob(
        "memory 17\nfunc (i32) -> (i64) {\nblock 0 (v0: i32) {\n  \
         v1 = i32.const 0\n  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)\n  \
         v3 = i64.const 100\n  v4 = i64.add v2 v3\n  return v4\n  }\n}\n",
    );
    // The program: func 0 = installer (compile+install → slot); func 1 = caller (call_indirect the
    // slot, passing the Clock handle); func 2 = a `(i32)->(i64)` may-suspend function so the taint
    // analysis marks the caller's `call_indirect` of that signature may-suspend (PropagatedIndirect),
    // the fork-critical case — else the site would be under-instrumented and the freeze miss it.
    let m_src = "memory 17\n\
        func (i32, i64, i64) -> (i64) {\nblock 0 (v0: i32, v1: i64, v2: i64) {\n  \
          v3 = cap.call 11 0 (i64, i64) -> (i64) v0 (v1, v2)\n  \
          v4 = cap.call 11 3 (i64) -> (i64) v0 (v3)\n  return v4\n  }\n}\n\
        func (i32, i32) -> (i64) {\nblock 0 (v0: i32, v1: i32) {\n  \
          v2 = call_indirect (i32) -> (i64) v0 (v1)\n  return v2\n  }\n}\n\
        func (i32) -> (i64) {\nblock 0 (v0: i32) {\n  \
          v1 = i32.const 0\n  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)\n  return v2\n  }\n}\n";
    let m = parse_module(m_src).expect("parse program");
    verify_module(&m).expect("verify program");
    let inst = transform_module(&m).expect("program in transform scope");
    verify_module(&inst).expect("instrumented program verifies");

    // Install the unit once (NORMAL), so the occupancy is recorded on the domain (it persists across
    // runs and rides the freeze). Grant Clock (durable) + a durable JIT domain with table room.
    let install = |h: &mut Host, jd: i32| -> i32 {
        let mut w = init_durable_window(WINDOW);
        write_state(&mut w, STATE_NORMAL);
        w[BLOB_OFF..BLOB_OFF + unit.len()].copy_from_slice(&unit);
        let mut fuel = 5_000_000u64;
        let (r, _) = run_capture_reserved_with_host(
            &inst,
            0,
            &[
                Value::I32(jd),
                Value::I64(BLOB_OFF as i64),
                Value::I64(unit.len() as i64),
            ],
            &mut fuel,
            &w,
            SIZE_LOG2,
            h,
        );
        match r.expect("install ok").as_slice() {
            [Value::I64(s)] if *s >= 0 => *s as i32,
            other => panic!("install returned {other:?}"),
        }
    };

    // --- Baseline: install, then a NORMAL call_indirect into the unit → Clock(42) + 100 = 142.
    let mut hb = Host::new();
    hb.clock_ns = 42;
    let clk_b = hb.grant_clock();
    let jd_b = grant_jit_durable(&mut hb, &m, TABLE_LOG2);
    let slot_b = install(&mut hb, jd_b);
    let mut wb = init_durable_window(WINDOW);
    write_state(&mut wb, STATE_NORMAL);
    let mut fuel = 5_000_000u64;
    let (rb, _) = run_capture_reserved_with_host(
        &inst,
        1,
        &[Value::I32(slot_b), Value::I32(clk_b)],
        &mut fuel,
        &wb,
        SIZE_LOG2,
        &mut hb,
    );
    assert_eq!(
        rb.expect("baseline ok"),
        vec![Value::I64(142)],
        "uninterrupted: Clock(42) + 100"
    );

    // --- Freeze run: install, then call_indirect the unit UNDER UNWINDING. The unit's `cap.call`
    // unwinds the continuation into the shadow stack (inside the installed unit's frame).
    let mut hf = Host::new();
    hf.clock_ns = 42;
    let clk = hf.grant_clock();
    let jd = grant_jit_durable(&mut hf, &m, TABLE_LOG2);
    let slot = install(&mut hf, jd);
    assert_eq!(slot, slot_b, "install is deterministic across hosts");
    let mut wf = init_durable_window(WINDOW);
    write_state(&mut wf, STATE_UNWINDING);
    let mut fuel = 5_000_000u64;
    let (rf, snap) = run_capture_reserved_with_host(
        &inst,
        1,
        &[Value::I32(slot), Value::I32(clk)],
        &mut fuel,
        &wf,
        SIZE_LOG2,
        &mut hf,
    );
    assert!(rf.is_ok(), "freeze returns a placeholder: {rf:?}");

    // Serialize + restore into a fresh host (Clock now 0 — a re-issue would give 100, not 142).
    let artifact = freeze(&inst, &snap, &hf).expect("freeze");
    let mut th = Host::new();
    th.clock_ns = 0;
    let mut window = restore(&artifact, &inst, &mut th).expect("restore");

    // Thaw: REWINDING → the caller re-issues the call_indirect (install slot survived), the unit
    // rewinds and reloads the saved clock (42), not the fresh 0.
    begin_thaw(&mut window, 0);
    let mut fuel = 5_000_000u64;
    let (rt, _) = run_capture_reserved_with_host(
        &inst,
        1,
        &[Value::I32(slot), Value::I32(clk)],
        &mut fuel,
        &window,
        SIZE_LOG2,
        &mut th,
    );
    assert_eq!(
        rt.expect("thaw ok"),
        vec![Value::I64(142)],
        "in-flight continuation inside the installed unit resumed: saved Clock(42) reloaded (not re-issued → 100)"
    );
}

/// **Durable install fence (DURABILITY.md §12.5, R8 fork-critical case).** A durable domain must not
/// admit a *suspendable* unit whose entry signature the program does not taint: a `call_indirect`
/// reaching such an installed unit would be at an un-instrumented site, silently losing the unit's
/// continuation on thaw (confirmed corruption). The gate fails it closed at `compile`. Three branches:
/// suspendable + untainted → **rejected**; suspendable + tainted → admitted; non-suspendable → admitted
/// (safe regardless of taint — no continuation to lose).
#[test]
fn durable_jit_compile_fences_suspending_untainted_unit() {
    // Program taints only `(i32)->(i64)` — a may-suspend function of that signature (never called;
    // it establishes the `call_indirect` seam the taint analysis keys on).
    let prog = parse_module(
        "memory 17\nfunc (i32) -> (i64) {\nblock 0 (v0: i32) {\n  \
         v1 = i32.const 0\n  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)\n  return v2\n  }\n}\n",
    )
    .expect("parse program");
    verify_module(&prog).expect("verify program");

    let mut h = Host::new();
    h.set_durable(true);
    let jit = grant_jit_durable(&mut h, &prog, TABLE_LOG2);

    // (A) suspendable, signature `(i32)->(i64)` — TAINTED by the program → admitted.
    let unit_a = blob(
        "memory 17\nfunc (i32) -> (i64) {\nblock 0 (v0: i32) {\n  \
         v1 = i32.const 0\n  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)\n  \
         v3 = i64.const 100\n  v4 = i64.add v2 v3\n  return v4\n  }\n}\n",
    );
    assert!(
        matches!(h.jit_compile(jit, &unit_a), Ok(Ok(_))),
        "suspendable unit of a TAINTED signature is admitted (the call_indirect seam exists)"
    );

    // (B) suspendable, signature `(i64)->(i64)` — NOT tainted → fenced closed.
    let unit_b = blob(
        "memory 17\nfunc (i64) -> (i64) {\nblock 0 (v0: i64) {\n  \
         vh = i32.const 0\n  v2 = cap.call 2 0 (i32) -> (i64) vh (vh)\n  return v2\n  }\n}\n",
    );
    assert!(
        matches!(h.jit_compile(jit, &unit_b), Ok(Err(-22))),
        "suspendable unit of an UNTAINTED signature is fenced (would lose its continuation on thaw)"
    );

    // (C) non-suspendable, signature `(i64)->(i64)` — untainted but SAFE → admitted.
    let unit_c = blob(
        "memory 17\nfunc (i64) -> (i64) {\nblock 0 (v0: i64) {\n  \
         vc = i64.const 7\n  return vc\n  }\n}\n",
    );
    assert!(
        matches!(h.jit_compile(jit, &unit_c), Ok(Ok(_))),
        "non-suspendable unit is admitted regardless of taint (no continuation to lose)"
    );
}
