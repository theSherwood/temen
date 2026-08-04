//! §4 durability × §22 guest-JIT, **Slice 1 — instrument-on-submit** (DURABILITY.md §12.5). A
//! **durable** `Jit` grant (`grant_jit_durable`) installs a validator that runs the durable transform
//! (`svm_durable::transform_module`) on each submitted unit before verify, so a durable domain
//! **admits** the (instrumented) unit instead of failing closed — the "host runs the pass on submitted
//! IR" composition. Without the durable grant a durable domain still refuses `compile`; a unit outside
//! the strict transform's scope (a guest-memory op) fails closed.
//!
//! Slice 1 covers *compile-side* composition + a NORMAL-state end-to-end run. Persisting `JitCode`/
//! `JitDomain` handles across a snapshot (freeze/thaw of the units themselves) is the Slice-2 follow-on;
//! here JIT handles still drain before a checkpoint (the existing `drain_non_durable` path).

use svm_durable::{init_durable_window, transform_module, write_state, STATE_NORMAL};
use svm_encode::encode_module;
use svm_interp::{run_capture_reserved_with_host, Host, Value};
use svm_run::{grant_jit, grant_jit_durable};
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
