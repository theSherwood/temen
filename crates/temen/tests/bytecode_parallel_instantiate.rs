//! THREADS.md 4c-domain — a §14 child under the **parallel** driver, where it runs on an OS thread of
//! its own. DESIGN.md §12 / I37: a trap is terminal for its **domain**, and a child is its own domain
//! — the owner observes the trap only if it joins (or polls). Here the root spawns a child that divides
//! by zero, never joins it, waits out a few 1 ms timeouts and returns 42: the child's trap must not
//! take the root's domain down with it, on the cooperative driver or the parallel one. (A trap in one
//! of the root's own `thread.spawn` threads does end the run — `bytecode_parallel.rs`'s
//! `a_thread_trap_ends_the_run_on_every_driver`.) The parallel driver's spawns are pinned against the
//! oracle by `temen-interp/tests/spawn_every_driver.rs`.

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use temen_interp::{bytecode, Host, Value};
use temen_ir::{Module, SpawnRec};
use temen_text::parse_module;

/// The root `(i32 inst, i32 child, i32 budget) -> i64`: spawns `child` (func 1's child image, #2219)
/// by the v1 record at 17408, paid from `budget`, then waits on a word nothing notifies, five times.
const TRAP_UNJOINED: &str = r#"memory 17
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, vc: i32, vb: i32) {
  vrm = i64.const 17432
  i32.store vrm vc
  vrb = i64.const 17436
  i32.store vrb vb
  vrp = i64.const 17408
  vh = call.cap 6 17 (i64) -> (i32) v0 (vrp)
  vi = i64.const 0
  br 1(vi)
}
block 1 (vi1: i64) {
  vn = i64.const 5
  vlt = i64.lt_u vi1 vn
  br_if vlt 2(vi1) 3()
}
block 2 (vi2: i64) {
  va = i64.const 16384
  vz = i32.const 0
  vw = i64.const 1000000
  vr = i32.atomic.wait va vz vw
  v1 = i64.const 1
  vn2 = i64.add vi2 v1
  br 1(vn2)
}
block 3 () {
  v42 = i64.const 42
  return v42
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 1
  vz = i64.const 0
  v2 = i64.div_s v1 vz
  return v2
  }
}
"#;

fn module() -> Module {
    let src = format!("{TRAP_UNJOINED}{}", rec::segment(17408, &SpawnRec::v1()));
    parse_module(&src).expect("parse")
}

/// The root's powerbox and args: its `Instantiator`, the child image, and a `Budget` for its window.
fn host(m: &Module) -> (Host, [Value; 3]) {
    let mut h = Host::new();
    let inst = h.grant_instantiator(0, 128 << 10);
    let child = h.grant_module(&temen_ir::child_image_at(m, 1).expect("child image"));
    let budget = h.grant_budget(-1, 1 << 20, -1);
    (h, [inst, child, budget].map(Value::I32))
}

#[test]
fn a_childs_trap_ends_only_its_own_domain() {
    let m = module();
    let want = Ok(vec![Value::I64(42)]);
    let (mut h, args) = host(&m);
    let mut fuel = 50_000_000u64;
    let coop = bytecode::compile_and_run_with_host(&m, 0, &args, &mut fuel, &mut h)
        .expect("the cooperative driver runs the root");
    assert_eq!(coop, want, "cooperative");
    for i in 0..if cfg!(miri) { 1 } else { 10 } {
        let (mut h, args) = host(&m);
        let mut fuel = 50_000_000u64;
        let (r, _) = bytecode::compile_and_run_capture_over_parallel_with_host(
            &m,
            0,
            &args,
            &mut fuel,
            &[],
            None,
            &mut h,
        )
        .expect("the parallel driver runs the root");
        assert_eq!(r, want, "parallel (run {i})");
    }
}
