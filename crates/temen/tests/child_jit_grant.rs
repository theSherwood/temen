//! #1296 slice 1 — a §14 child holds a **`Jit` capability of its own**. A parent re-grants its `jit`
//! handle by name (op 11's grant list); the child gets a **fresh, empty unit table** — not an alias of
//! the parent's (units, installs and B2 dispatch slots are per-table artifacts) — with a compile quota
//! attenuated to at most the parent's remaining, and compiles + invokes a unit over its own window.
//! The memory-match precondition resolves against the child's *own* module at compile time (here the
//! same module as the parent, `memory 17`). Both interpreter engines must agree.
//!
//! Also pinned: the quota attenuates (a parent with one unit left hands the child one unit; the
//! child's second `compile` is `-ENOMEM`), and a child spawned **without** the grant has no `jit`
//! name to resolve (the resolve returns a negative errno, never a handle into the parent's table).
//!
//! #1296 slice 2 — the **native JIT** runs the same program (`run_jit`): a §14 child compiles to the
//! root's shape (`CompiledModule`), so its re-granted `Jit` table compiles units into the child's own
//! module and installs into the child's own dispatch table; the four outcomes must match the
//! interpreter's byte-for-byte.

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use std::sync::Arc;
use temen_encode::encode_module;
use temen_interp::{bytecode, run_with_host, Host, MemLayout, Trap, Value};
use temen_ir::{SpawnRec, DEFAULT_RESERVED_LOG2};
use temen_jit::JitOutcome;
use temen_run::grant_jit;
use temen_text::parse_module;
use temen_verify::verify_module;

/// Where the unit blob sits, a data segment above the NULL guard and the grant scratch: the child
/// runs the parent's module, so its own window holds the blob there too.
const BLOB_OFF: usize = 20480;

/// The unit the child compiles, declaring the module's memory (`memory 17`): `(a, b) -> a + b`, or for
/// the install probe a 0-arg `() -> 42` (an installed slot is `call.dyn`ed with no args).
fn blob(zero_arg: bool) -> Vec<u8> {
    let src = if zero_arg {
        "memory 17\nfunc () -> (i32) {\nblock 0 () {\n  v0 = i32.const 42\n  return v0\n  }\n}\n"
    } else {
        "memory 17\nfunc (i32, i32) -> (i32) {\nblock 0 (v0: i32, v1: i32) {\n  v2 = i32.add v0 v1\n  return v2\n  }\n}\n"
    };
    let m = parse_module(src).expect("parse blob");
    verify_module(&m).expect("verify blob");
    encode_module(&m)
}

/// func 0 (parent, `(Instantiator, Jit, Budget)`): one grant record at 16384 naming the `Jit` handle
/// `"jit"` (the name a data segment at 16484 holds), spawn the child detached through a v1 record at
/// 17536 (entry 1, the parent's module, paid from the `Budget`, grant list `(16384, grants_n)`),
/// join, return its result.
/// func 1 (child, `(Instantiator)`): resolve `"jit"` by name (the same data in its own window),
/// compile the blob at `BLOB_OFF` `times` times (the last compile's code handle is what it invokes),
/// invoke `(3, 4)`, return the sum. `times == 2` is the quota probe: it returns the **second compile's**
/// result instead (`-ENOMEM` when the child's table has one unit). Modes 5/6: the child `install`s
/// its unit and returns the slot; in mode 5 the parent `call.dyn`s that slot (in its own table), in
/// mode 6 it returns the slot as is.
fn src(grants_n: u32, blob: &[u8], mode: u32) -> String {
    let blob_len = blob.len();
    let blob_data: String = blob.iter().map(|b| format!("\\x{b:02x}")).collect();
    let spawn = SpawnRec {
        grants_ptr: 16384,
        grants_n: u64::from(grants_n),
        ..SpawnRec::v1(1)
    };
    let child_pre = format!(
        "  n0 = i64.const 16484\n  len3 = i64.const 3\n  hj = self.resolve n0 len3\n  vb = i64.const {}\n  vl = i64.const {}\n  vc = call.cap 11 0 (i64, i64) -> (i64) hj (vb, vl)\n",
        BLOB_OFF, blob_len
    );
    let child_tail = match mode {
        2 => "  vc2 = call.cap 11 0 (i64, i64) -> (i64) hj (vb, vl)\n  return vc2\n".to_string(),
        5 | 6 => "  vslot = call.cap 11 3 (i64) -> (i64) hj (vc)\n  return vslot\n".to_string(),
        _ => "  va = i32.const 3\n  vb4 = i32.const 4\n  vr = call.cap 11 1 (i64, i32, i32) -> (i32) hj (vc, va, vb4)\n  vr64 = i64.extend_i32_s vr\n  return vr64\n".to_string(),
    };
    // Mode 5: the parent `call.dyn`s the slot the child installed — in the PARENT's table, which the
    // child's install never touched, so it must trap (the child's units are its own).
    let parent_tail = if mode == 5 {
        "  rs = i32.wrap_i64 r\n  rd = call.dyn () -> (i32) rs ()\n  rd64 = i64.extend_i32_s rd\n  return rd64\n"
    } else {
        "  return r\n"
    };
    format!(
        r#"memory 17
func (i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vjit: i32, vbud: i32) {{
  a0 = i64.const 16384
  n100 = i32.const 16484
  i32.store a0 n100
  a4 = i64.const 16388
  n3 = i32.const 3
  i32.store a4 n3
  a8 = i64.const 16392
  i32.store a8 vjit
  a12 = i64.const 16396
  z0 = i32.const 0
  i32.store a12 z0
  qb = i64.const 17564
  i32.store qb vbud
  q0a0 = i64.const 17536
  vch = call.cap 6 17 (i64) -> (i32) vinst (q0a0)
  r = call.cap 6 1 (i32) -> (i64) vinst (vch)
{parent_tail}  }}
}}
func (i64) -> (i64) {{
block 0 (vci: i64) {{
{child_pre}{child_tail}  }}
}}
data 16484 "jit"
data {BLOB_OFF} "{blob_data}"
{rec}"#,
        rec = rec::segment(17536, &spawn),
    )
}

/// The parent's host for one run: an `Instantiator`, a `Jit` table (16 install slots for the install
/// probe) with a `units` compile quota, and a `Budget` that pays for the child. The host knows the
/// parent's module, which the child runs.
fn setup(m: &temen_ir::Module, mode: u32, units: u32) -> (Host, [i32; 3]) {
    let mut host = Host::new();
    host.set_self_module(&Arc::new(m.clone()));
    let ih = host.grant_instantiator(0, 128 << 10);
    let jh = grant_jit(&mut host, m, if mode >= 5 { 4 } else { 0 });
    host.set_jit_quota(units, 1 << 20);
    let bh = host.grant_budget(-1, 1 << 20, -1);
    (host, [ih, jh, bh])
}

/// Run on the tree-walker and on the bytecode engine with the same host setup (a root `Jit` grant of
/// `units` units); return both outcomes for the caller's assertions.
fn run_both(grants_n: u32, mode: u32, units: u32) -> [Result<Vec<Value>, Trap>; 2] {
    let b = blob(mode >= 5);
    let m = parse_module(&src(grants_n, &b, mode)).expect("parse");
    verify_module(&m).expect("verify");
    let mut out = Vec::new();
    for engine in 0..2 {
        let (mut host, handles) = setup(&m, mode, units);
        let args = handles.map(Value::I32);
        let mut fuel = 5_000_000u64;
        let r = if engine == 0 {
            run_with_host(&m, 0, &args, &mut fuel, &mut host)
        } else {
            match bytecode::compile_and_run_with_host(&m, 0, &args, &mut fuel, &mut host) {
                Some(r) => r,
                None => panic!(
                    "the bytecode engine declined; tree-walker said {:?}",
                    out[0]
                ),
            }
        };
        out.push(r);
    }
    [out.remove(0), out.remove(0)]
}

/// The same program on the native JIT, with temen-run's production child hooks (the child is built
/// host-side by the detached builder, which reports the `Jit` grant's table reservation).
fn run_jit(grants_n: u32, mode: u32, units: u32) -> JitOutcome {
    let b = blob(mode >= 5);
    let m = parse_module(&src(grants_n, &b, mode)).expect("parse");
    verify_module(&m).expect("verify");
    let (mut host, handles) = setup(&m, mode, units);
    temen_run::jit_cap_run(
        &m,
        0,
        &handles.map(i64::from),
        &MemLayout::image(Vec::new()),
        DEFAULT_RESERVED_LOG2,
        0,
        &mut host,
        None,
    )
    .expect("jit")
    .0
}

/// Interp `Ok([I64(x)])` ≡ JIT `Returned([x])`; any interp trap ≡ any JIT trap (the kind is not part
/// of the pinned contract here — the interp's `IndirectCallType` vs the JIT's `Trapped(..)`).
fn agrees(interp: &Result<Vec<Value>, Trap>, jit: &JitOutcome) -> bool {
    match (interp, jit) {
        (Ok(v), JitOutcome::Returned(r)) => {
            matches!((v.first(), r.first()), (Some(Value::I64(a)), Some(b)) if a == b)
        }
        (Err(_), JitOutcome::Trapped(_)) => true,
        _ => false,
    }
}

#[test]
fn a_child_compiles_and_invokes_a_unit_on_its_own_jit_table() {
    let [tw, bc] = run_both(1, 1, 4096);
    assert_eq!(
        tw,
        Ok(vec![Value::I64(7)]),
        "tree-walker: child invoked 3 + 4"
    );
    assert_eq!(bc, tw, "bytecode engine agrees");
    let jo = run_jit(1, 1, 4096);
    assert!(agrees(&tw, &jo), "native JIT agrees: {jo:?}");
}

#[test]
fn the_child_table_quota_is_at_most_the_parents_remaining() {
    // The parent has one unit left; the child inherits one, so its second compile is -ENOMEM.
    let [tw, bc] = run_both(1, 2, 1);
    assert_eq!(
        tw,
        Ok(vec![Value::I64(-12)]),
        "tree-walker: ENOMEM on the second compile"
    );
    assert_eq!(bc, tw, "bytecode engine agrees");
    let jo = run_jit(1, 2, 1);
    assert!(agrees(&tw, &jo), "native JIT agrees: {jo:?}");
}

#[test]
fn a_child_without_the_grant_cannot_reach_the_parents_table() {
    let [tw, bc] = run_both(0, 1, 4096);
    for (r, name) in [(&tw, "tree-walker"), (&bc, "bytecode")] {
        // A trap is an equally closed outcome; a value must be a negative errno.
        if let Ok(v) = r {
            assert!(
                matches!(v.first(), Some(Value::I64(x)) if *x < 0),
                "{name}: no `jit` name resolves, got {v:?}"
            );
        }
    }
    assert_eq!(tw, bc, "both engines refuse the same way");
    let jo = run_jit(0, 1, 4096);
    assert!(
        matches!(&jo, JitOutcome::Returned(r) if r.first().is_some_and(|x| *x < 0))
            || matches!(jo, JitOutcome::Trapped(_)),
        "native JIT: no `jit` name resolves, got {jo:?}"
    );
}

#[test]
fn a_childs_install_is_invisible_to_the_parents_dispatch_table() {
    // The child installs its unit and reports the slot; the parent's `call.dyn` on that slot finds
    // nothing (its own table was never written) and traps — on both engines.
    let [tw, bc] = run_both(1, 5, 4096);
    assert!(
        tw.is_err(),
        "tree-walker: the parent's call.dyn must trap, got {tw:?}"
    );
    assert!(
        bc.is_err(),
        "bytecode: the parent's call.dyn must trap, got {bc:?}"
    );
    let jo = run_jit(1, 5, 4096);
    assert!(
        matches!(jo, JitOutcome::Trapped(_)),
        "native JIT: the parent's call.dyn must trap, got {jo:?}"
    );
}

#[test]
fn a_child_installs_into_its_own_reserved_table() {
    // The `Jit` grant carries the parent's table reservation (16 slots) into the child, so the
    // child's `install` lands in a padding slot of the CHILD's dispatch table: a non-negative slot
    // (never `-ENOSPC` from a natural, padding-free table), identical on every engine.
    let [tw, bc] = run_both(1, 6, 4096);
    assert!(
        matches!(tw.as_deref(), Ok([Value::I64(s)]) if *s >= 0),
        "tree-walker: the child's install must return a slot, got {tw:?}"
    );
    assert_eq!(bc, tw, "bytecode engine agrees on the slot");
    let jo = run_jit(1, 6, 4096);
    assert!(agrees(&tw, &jo), "native JIT agrees: {jo:?}");
}
