//! #3078 phase 2 slice 1 — engine-level rows for the ways a fiber's futex wait leaves the run's wait
//! index (`Waits`) that `fiber_wait_order.rs` and `fiber_wait_counts_out.rs` do not reach: its
//! domain's end (`Tasks::retire_fibers`). A wait that stayed filed past its fiber's registry would be
//! found by the next rule that walks the index (a notify, or the clock reaching its deadline), which
//! checks that the entry it takes is the wait its record filed and fails the run if not.
//!
//! The other leaves: the freeze's flatten has its row in `bytecode_durable_fibers_freeze.rs`
//! (`a_wait_parked_fiber_freezes_and_thaws_on_the_bytecode_engine`, an unwoken fiber wait flattened
//! by the bytecode freeze driver), and a poll past a fiber's real deadline on executor 2's
//! registry has a unit test (`a_poll_past_a_fibers_real_deadline_takes_its_wait_out_of_the_index`).

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use temen_interp::{bytecode, run_with_host, Host, Value};
use temen_ir::{Module, SpawnRec};
use temen_text::parse_module;

/// The root `(i32 inst, i32 child, i32 budget) -> i64` spawns a §14 child (func 1's image), never
/// joins it, then waits 100 ms on a word nothing notifies and returns 42. The child makes a fiber
/// (func 2) that parks on a 50 ms timed wait in the child's own window, and returns at once: its
/// domain ends with its fiber still waiting. Were that wait left in the index, its 50 ms deadline
/// would be the run's next one, reached before the root's.
const CHILD_ENDS_WITH_A_WAITING_FIBER: &str = r#"memory 17
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, vc: i32, vb: i32) {
  vrm = i64.const 17416
  i32.store vrm vc
  vrb = i64.const 17420
  i32.store vrb vb
  vrp = i64.const 17408
  vh = call.cap 6 17 (i64) -> (i32) v0 (vrp)
  va = i64.const 16384
  vz = i32.const 0
  vw = i64.const 100000000
  vr = i32.atomic.wait va vz vw
  v42 = i64.const 42
  return v42
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vf = ref.func 2
  vz = i64.const 0
  vc = cont.new vf vz
  vs, vv = cont.resume vc vz
  return vz
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  va = i64.const 16384
  vz = i32.const 0
  vw = i64.const 50000000
  vr = i32.atomic.wait va vz vw
  vr64 = i64.extend_i32_s vr
  return vr64
  }
}
"#;

fn module() -> Module {
    let src = format!(
        "{CHILD_ENDS_WITH_A_WAITING_FIBER}{}",
        rec::segment(17408, &SpawnRec::default())
    );
    parse_module(&src).expect("parse")
}

/// The root's powerbox and args: its `Instantiator`, the child image, and a `Budget` for its window.
fn host(m: &Module) -> (Host, [Value; 3]) {
    let mut h = Host::new();
    let inst = h.grant_instantiator();
    let child = h.grant_module(&temen_ir::child_image_at(m, 1).expect("child image"));
    let budget = h.grant_budget(-1, 1 << 20, -1);
    (h, [inst, child, budget].map(Value::I32))
}

/// A child domain that ends with a fiber still in a timed wait takes the wait out of the index
/// with its registry: the run goes on to the root's own deadline and answers as the oracle does,
/// on the pump and on executor 2.
#[test]
fn a_domain_that_ends_with_a_waiting_fiber_takes_its_wait_out_of_the_index() {
    let m = module();
    temen_verify::verify_module(&m).expect("verify");
    let want = Ok(vec![Value::I64(42)]);
    let (mut h, args) = host(&m);
    let mut fuel = 50_000_000u64;
    assert_eq!(
        run_with_host(&m, 0, &args, &mut fuel, &mut h),
        want,
        "the oracle"
    );
    let (mut h, args) = host(&m);
    let mut fuel = 50_000_000u64;
    let pump = bytecode::compile_and_run_with_host(&m, 0, &args, &mut fuel, &mut h)
        .expect("the pump runs the root");
    assert_eq!(pump, want, "the pump");
    let (mut h, args) = host(&m);
    let mut fuel = 50_000_000u64;
    let (par, _) = bytecode::compile_and_run_capture_over_parallel_with_host(
        &m,
        0,
        &args,
        &mut fuel,
        &[],
        None,
        &mut h,
    )
    .expect("executor 2 runs the root");
    assert_eq!(par, want, "executor 2");
}
