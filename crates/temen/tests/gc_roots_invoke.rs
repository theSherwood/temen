//! #1660 — §GC `gc.roots` reached **inside a `Jit.invoke`** (DESIGN.md §22 new→old). An invoke
//! runs the unit inline over the invoker's window, so over the same heap: a collection inside it
//! must still see everything the invoker holds — its frames, its parked fibers, and, under a nested
//! invoke, every invoker below that (GC.md §3.1: the caller's whole live stack).
//!
//! Each case runs on the tree-walk oracle, the bytecode engine and Cranelift, and requires every
//! root on every one. Roots arrive as **arguments**, not constants: a constant is not a heap root,
//! and a compiler may rematerialize it after a call rather than keep it live.
//!
//! Before the fix the oracle missed the invoker's roots (unsound — an under-approximation) and the
//! bytecode engine trapped `CapFault` (its nested drive had no `gc.roots` arm).

use std::collections::BTreeSet;

use temen_interp::{bytecode, run_capture_reserved_with_host, Host, Value};
use temen_ir::DEFAULT_RESERVED_LOG2;
use temen_jit::JitOutcome;
use temen_run::{grant_jit, jit_cap_run};
use temen_text::parse_module;
use temen_verify::verify_module;

/// The collector every case reaches: scan `[4096, 8192)` into the buffer at 16384 and return the
/// total. `4096` (its own `vlo`) is always among the roots.
const COLLECTOR: &str = r#"func () -> (i64) {
block 0 () {
  vlo = i64.const 4096
  vhi = i64.const 8192
  vmask = i64.const -1
  vbuf = i64.const 16384
  vcap = i64.const 64
  vt = gc.roots vlo vhi vmask vbuf vcap
  return vt
  }
}
"#;

/// Compile `units` into a fresh host holding the `Jit` cap, returning `(host, jit, code handles)`.
/// Granting and compiling into a fresh host is deterministic, so every backend sees the same handles.
fn setup(guest: &temen_ir::Module, units: &[&str]) -> (Host, i32, Vec<i32>) {
    let mut host = Host::new();
    let jit = grant_jit(&mut host, guest, 0);
    let codes = units
        .iter()
        .map(|src| {
            let unit = parse_module(src).expect("parse unit");
            verify_module(&unit).expect("verify unit");
            host.jit_compile(jit, &temen_encode::encode_module(&unit))
                .expect("no trap")
                .expect("compile ok")
                .handle
        })
        .collect();
    (host, jit, codes)
}

fn roots_in(snap: &[u8]) -> BTreeSet<u64> {
    (0..64)
        .map(|i| u64::from_le_bytes(snap[16384 + i * 8..16384 + i * 8 + 8].try_into().unwrap()))
        .filter(|w| (4096..8192).contains(w))
        .collect()
}

/// Run `guest` (args: the jit handle, every code handle, then `roots`) on all three backends and
/// require each to complete and to report every root in `roots` plus the collector's own 4096.
fn check(guest: &str, units: &[&str], roots: &[i64]) {
    let m = parse_module(guest).expect("parse guest");
    verify_module(&m).expect("verify guest");
    let init = vec![0u8; 20480];
    let mut want: BTreeSet<u64> = roots.iter().map(|&r| r as u64).collect();
    want.insert(4096);
    let slots = |jit: i32, codes: &[i32]| -> Vec<i64> {
        let mut v = vec![jit as i64];
        v.extend(codes.iter().map(|&c| c as i64));
        v.extend_from_slice(roots);
        v
    };
    let values = |jit: i32, codes: &[i32]| -> Vec<Value> {
        let mut v = vec![Value::I32(jit)];
        v.extend(codes.iter().map(|&c| Value::I32(c)));
        v.extend(roots.iter().map(|&r| Value::I64(r)));
        v
    };

    let (mut h, jit, codes) = setup(&m, units);
    let mut fuel = 1_000_000u64;
    let (res, snap) = run_capture_reserved_with_host(
        &m,
        0,
        &values(jit, &codes),
        &mut fuel,
        &init,
        DEFAULT_RESERVED_LOG2,
        &mut h,
    );
    assert!(res.is_ok(), "tree-walker: {res:?}");
    let got = roots_in(&snap);
    assert!(
        want.is_subset(&got),
        "tree-walker missed a root: want {want:?}, got {got:?}"
    );

    let (mut h, jit, codes) = setup(&m, units);
    let mut fuel = 1_000_000u64;
    let (res, snap) = bytecode::compile_and_run_capture_reserved_with_host(
        &m,
        0,
        &values(jit, &codes),
        &mut fuel,
        &init,
        DEFAULT_RESERVED_LOG2,
        &mut h,
    )
    .expect("the bytecode engine accepts the module");
    assert!(res.is_ok(), "bytecode engine: {res:?}");
    let got = roots_in(&snap);
    assert!(
        want.is_subset(&got),
        "bytecode engine missed a root: want {want:?}, got {got:?}"
    );

    let (mut h, jit, codes) = setup(&m, units);
    let (out, snap) = jit_cap_run(
        &m,
        0,
        &slots(jit, &codes),
        &init,
        DEFAULT_RESERVED_LOG2,
        0,
        &mut h,
    )
    .expect("jit run");
    assert!(matches!(out, JitOutcome::Returned(_)), "cranelift: {out:?}");
    let got = roots_in(snap.bytes());
    assert!(
        want.is_subset(&got),
        "cranelift missed a root: want {want:?}, got {got:?}"
    );
}

/// The guest holds `groot` across `Jit.invoke(code, uroot)`; the unit holds `uroot` across a
/// `call.dyn` into the collector (natural-table slot 1).
#[test]
fn invoker_frame_is_scanned() {
    let guest = format!(
        r#"memory 16
func (i32, i32, i64, i64) -> (i64) {{
block 0 (vj: i32, vc: i32, vg: i64, vu: i64) {{
  v2 = i64.extend_i32_u vc
  v3 = call.cap 11 1 (i64, i64) -> (i64) vj (v2, vu)
  v4 = i64.add v3 vg
  return v4
  }}
}}
{COLLECTOR}"#
    );
    let unit = r#"memory 16
func (i64) -> (i64) {
block 0 (vu: i64) {
  v1 = i32.const 1
  v2 = call.dyn () -> (i64) v1 ()
  v3 = i64.add v2 vu
  return v3
  }
}
"#;
    check(&guest, &[unit], &[5000, 6000]);
}

/// As above, but the invoker first parks a fiber holding `froot` across its `suspend` — the
/// invoker's fiber registry must be scanned too, not only its frames.
#[test]
fn invokers_parked_fiber_is_scanned() {
    let guest = format!(
        r#"memory 16
func (i32, i32, i64, i64, i64) -> (i64) {{
block 0 (vj: i32, vc: i32, vg: i64, vu: i64, vfr: i64) {{
  vf = ref.func 2
  vsp = i64.const 0
  vk = cont.new vf vsp
  vst, vval = cont.resume vk vfr
  v2 = i64.extend_i32_u vc
  v3 = call.cap 11 1 (i64, i64) -> (i64) vj (v2, vu)
  v4 = i64.add v3 vg
  return v4
  }}
}}
{COLLECTOR}func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  vy = i64.const 1
  vr = suspend vy
  vsum = i64.add varg vr
  return vsum
  }}
}}
"#
    );
    let unit = r#"memory 16
func (i64) -> (i64) {
block 0 (vu: i64) {
  v1 = i32.const 1
  v2 = call.dyn () -> (i64) v1 ()
  v3 = i64.add v2 vu
  return v3
  }
}
"#;
    check(&guest, &[unit], &[5000, 6000, 7000]);
}

/// A **nested** invoke: the guest invokes unit 1 holding `groot`; unit 1 holds `r1` across a
/// `call.dyn` into guest func 1, which invokes unit 2; unit 2 holds `r2` across a `call.dyn` into the
/// collector (slot 2). Every level's root must survive — the view beneath is forwarded, not replaced.
#[test]
fn every_level_of_a_nested_invoke_is_scanned() {
    let guest = format!(
        r#"memory 16
func (i32, i32, i32, i64, i64, i64) -> (i64) {{
block 0 (vj: i32, vc1: i32, vc2: i32, vg: i64, vr1: i64, vr2: i64) {{
  vj64 = i64.extend_i32_u vj
  vc164 = i64.extend_i32_u vc1
  vc264 = i64.extend_i32_u vc2
  v3 = call.cap 11 1 (i64, i64, i64, i64, i64) -> (i64) vj (vc164, vj64, vc264, vr1, vr2)
  v4 = i64.add v3 vg
  return v4
  }}
}}
func (i64, i64, i64) -> (i64) {{
block 0 (vj: i64, vc2: i64, vr2: i64) {{
  vj32 = i32.wrap_i64 vj
  v3 = call.cap 11 1 (i64, i64) -> (i64) vj32 (vc2, vr2)
  return v3
  }}
}}
{COLLECTOR}"#
    );
    let unit1 = r#"memory 16
func (i64, i64, i64, i64) -> (i64) {
block 0 (vj: i64, vc2: i64, vr1: i64, vr2: i64) {
  v1 = i32.const 1
  v2 = call.dyn (i64, i64, i64) -> (i64) v1 (vj, vc2, vr2)
  v3 = i64.add v2 vr1
  return v3
  }
}
"#;
    let unit2 = r#"memory 16
func (i64) -> (i64) {
block 0 (vr2: i64) {
  v1 = i32.const 2
  v2 = call.dyn () -> (i64) v1 ()
  v3 = i64.add v2 vr2
  return v3
  }
}
"#;
    check(&guest, &[unit1, unit2], &[5000, 6000, 7000]);
}
