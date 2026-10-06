//! #2112 — **a live fiber spends `Budget.mem`**, on every engine: each interpreter driver and the
//! Cranelift JIT.
//!
//! Each fiber `cont.new` makes is `FIBER_STACK` of its domain's `mem`, charged to the domain's node and
//! every ancestor, all or nothing: past a ceiling `cont.new` traps `FiberFault`. The charge goes back
//! when the fiber returns, and with the domain when it ends with the fiber unfinished.
//!
//! The root pays for a detached child from a budget that holds the child's window and some fibers.
//! The child reads the budget's `mem` room around its fibers: one it runs to its return, then one it
//! leaves unfinished. The parent reads the room again once the child has ended.

#[path = "../../temen-interp/tests/support/drivers.rs"]
mod drivers;
#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use std::sync::Arc;

use drivers::{agree_on_every_driver, Ran};
use temen_interp::{Host, MemLayout, Value};
use temen_ir::{Module, SpawnRec, FIBER_STACK};
use temen_jit::{JitError, JitOutcome};

/// The child's window: the module's declared memory, which a self-spawned detached child takes.
const WINDOW: i64 = 1 << 16;

/// A root that pays for func 1 from its `Budget` and returns `child * 10_000_000 + room`: the child's
/// result (with `wait`, the trap code it ended with) and the budget's `mem` room once it has ended.
/// The child returns a bit per expected room: 1 a new fiber spends `FIBER_STACK`, 2 its return hands
/// it back, 4 a second fiber spends it again.
fn src(collect: &str) -> String {
    format!(
        "memory 16
data 20000 \"budget\"
func (i32, i32) -> (i64) {{
block 0 (vinst: i32, vbud: i32) {{
  rb = i64.const 17436
  i32.store rb vbud
  rp = i64.const 17408
  vch = call.cap 6 17 (i64) -> (i32) vinst (rp)
  vr = call.cap 6 {collect} (i32) -> (i64) vinst (vch)
  one = i64.const 1
  vroom = call.cap 14 1 (i64) -> (i64) vbud (one)
  k = i64.const 10000000
  vhi = i64.mul vr k
  vsum = i64.add vhi vroom
  return vsum
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vinst: i64, vas: i64) {{
  l6 = i64.const 6
  np = i64.const 20000
  vb = self.resolve np l6
  one = i64.const 1
  r0 = call.cap 14 1 (i64) -> (i64) vb (one)
  vf = ref.func 2
  vz = i64.const 0
  k1 = cont.new vf vz
  r1 = call.cap 14 1 (i64) -> (i64) vb (one)
  vs, vv = cont.resume k1 vz
  r2 = call.cap 14 1 (i64) -> (i64) vb (one)
  k2 = cont.new vf vz
  r3 = call.cap 14 1 (i64) -> (i64) vb (one)
  fs = i64.const {FIBER_STACK}
  d1 = i64.sub r0 r1
  e1 = i64.eq d1 fs
  e2 = i64.eq r2 r0
  d3 = i64.sub r0 r3
  e3 = i64.eq d3 fs
  w1 = i64.extend_i32_u e1
  w2 = i64.extend_i32_u e2
  w3 = i64.extend_i32_u e3
  two = i64.const 2
  four = i64.const 4
  x2 = i64.mul w2 two
  x3 = i64.mul w3 four
  s12 = i64.add w1 x2
  s = i64.add s12 x3
  return s
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  return varg
  }}
}}
{rec}",
        rec = rec::segment(17408, &SpawnRec::v1(1)),
    )
}

fn module(collect: &str) -> Module {
    let m: Module = temen_text::parse_module(&src(collect)).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// A powerbox whose `Budget` holds the child's window and `fibers` fibers' worth of `mem`.
fn setup(m: &Module, fibers: i64) -> impl Fn() -> (Host, Vec<Value>) + '_ {
    move || {
        let mut h = Host::new();
        h.set_self_module(&Arc::new(m.clone()));
        let i = h.grant_instantiator(0, 1 << 16);
        let b = h.grant_budget(-1, WINDOW + fibers * FIBER_STACK as i64, -1);
        (h, vec![Value::I32(i), Value::I32(b)])
    }
}

/// Run `m` on every interpreter driver and on the Cranelift JIT, where this target has its child
/// executor, and assert each returns `want`.
fn every_engine(what: &str, m: &Module, setup: &dyn Fn() -> (Host, Vec<Value>), want: i64) {
    let ran = Ran {
        result: Ok(vec![Value::I64(want)]),
        stdout: Vec::new(),
        stderr: Vec::new(),
    };
    agree_on_every_driver(what, m, setup, &ran);
    let (mut host, args) = setup();
    let args: Vec<i64> = args
        .iter()
        .map(|v| match v {
            Value::I32(h) => i64::from(*h),
            other => panic!("a handle, not {other:?}"),
        })
        .collect();
    let init = MemLayout::image(vec![0u8; 1 << 16]);
    match temen_run::jit_cap_run(m, 0, &args, &init, 16, 0, &mut host, None) {
        Ok((o, _)) => assert_eq!(o, JitOutcome::Returned(vec![want]), "{what}: Cranelift"),
        Err(JitError::Unsupported(_)) => {} // a target without the child executor
        Err(e) => panic!("{what}: the JIT run failed: {e:?}"),
    }
}

/// With room for two fibers, each the child makes spends `FIBER_STACK`, the one that returns hands it
/// back, and the one left unfinished hands it back as the child ends: the whole ceiling is room again.
#[test]
fn a_fiber_spends_its_domains_mem_until_it_returns_or_the_domain_ends_on_every_engine() {
    let m = module("1"); // join: the child's result
    let ceiling = WINDOW + 2 * FIBER_STACK as i64;
    every_engine("fibers", &m, &setup(&m, 2), 7 * 10_000_000 + ceiling);
}

/// With no room past the child's window, its first `cont.new` traps `FiberFault`, which the parent's
/// `wait` reads; the refusal charged nothing, and the window goes back as the child ends.
#[test]
fn a_fiber_past_the_ceiling_traps_and_charges_nothing_on_every_engine() {
    let m = module("18"); // wait: how the child ended
    let fiber_fault = temen_ir::trap_code::FIBER_FAULT;
    every_engine(
        "a fiber past the ceiling",
        &m,
        &setup(&m, 0),
        fiber_fault * 10_000_000 + WINDOW,
    );
}
