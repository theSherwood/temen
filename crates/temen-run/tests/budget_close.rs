//! **#2384 — `Budget.close`** (op 3): a domain gives a `Budget` handle up. The slot closes as a
//! `Stream`'s does, so the handle answers `-EBADF` from then on, and the node stays in the run's
//! budget tree, where charges name it by id: a child spawned from it is capped by it still.
//!
//! A parent that splits a node per child, to cap each child's fuel at its endowment (theSherwood/unir's
//! binding does), would otherwise keep a handle per child it has ever spawned and fill its table.
//!
//! Every case runs on the tree-walk oracle, every bytecode driver ([`drivers::ALL`]) and the Cranelift
//! JIT, and each must give the oracle's answer. The op is one arm of the shared dispatch.

use std::sync::Arc;

use temen_interp::{Host, MemLayout, Trap, Value};
use temen_ir::errno::EBADF;
use temen_ir::trap_code::OUT_OF_FUEL;
use temen_ir::{Module, SpawnRec};
use temen_jit::JitOutcome;

#[path = "../../temen-interp/tests/support/drivers.rs"]
mod drivers;
#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use drivers::{agree_on_every_driver, Ran};

/// Splits and closes past the table's 1,024 slots (#2365).
const SPLITS: u64 = 1100;

/// The fuel ceiling of the node the child is spawned from: far less than its loop burns.
const CHILD_FUEL: i64 = 10_000;

fn module(src: &str) -> Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// A root that splits a node from its `Budget` and closes its handle [`SPLITS`] times, counting the
/// splits that fail; then splits a node with a fuel ceiling of [`CHILD_FUEL`], spawns func 1's child
/// image from it (the record at 17408), closes the node's handle, reads the closed handle, and joins
/// the child, which loops past that ceiling. It returns `failed * 10^9 + closes * 10^6 + (read +
/// 100) * 10^3 + status`, where `closes` sums what every close answered and `status` is how the join
/// answers the child's end (#2066).
fn src() -> String {
    format!(
        "memory 16
func (i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vbud: i32, vmod: i32) {{
  vma = i64.const {module_at}
  i32.store vma vmod
  vn = i64.const {SPLITS}
  vz = i64.const 0
  br 1(vn, vz, vz, vinst, vbud)
}}
block 1 (vleft: i64, vfailed: i64, vcloses: i64, vi: i32, vb: i32) {{
  vm1 = i64.const -1
  vnode = call.cap 14 0 (i64, i64, i64) -> (i64) vb (vm1, vm1, vm1)
  vzero = i64.const 0
  vneg = i64.lt_s vnode vzero
  vneg64 = i64.extend_i32_u vneg
  vfailed2 = i64.add vfailed vneg64
  vh = i32.wrap_i64 vnode
  vc = call.cap 14 3 () -> (i64) vh ()
  vcloses2 = i64.add vcloses vc
  vone = i64.const 1
  vleft2 = i64.sub vleft vone
  vmore = i64.ne vleft2 vzero
  br_if vmore 1(vleft2, vfailed2, vcloses2, vi, vb) 2(vfailed2, vcloses2, vi, vb)
}}
block 2 (vf: i64, vcs: i64, vi2: i32, vb2: i32) {{
  vfuel = i64.const {CHILD_FUEL}
  vu = i64.const -1
  vcn = call.cap 14 0 (i64, i64, i64) -> (i64) vb2 (vfuel, vu, vu)
  vch32 = i32.wrap_i64 vcn
  vba = i64.const {budget_at}
  i32.store vba vch32
  vrp = i64.const 17408
  vkid = call.cap 6 17 (i64) -> (i32) vi2 (vrp)
  vcl = call.cap 14 3 () -> (i64) vch32 ()
  vcs2 = i64.add vcs vcl
  vfield = i64.const 0
  vread = call.cap 14 1 (i64) -> (i64) vch32 (vfield)
  vwait, vjv = call.cap 6 1 (i32) -> (i64, i64) vi2 (vkid)
  vg = i64.const 1000000000
  vm = i64.const 1000000
  vk = i64.const 1000
  vhundred = i64.const 100
  va = i64.mul vf vg
  vb3 = i64.mul vcs2 vm
  vr2 = i64.add vread vhundred
  vc3 = i64.mul vr2 vk
  vs1 = i64.add va vb3
  vs2 = i64.add vs1 vc3
  vs3 = i64.add vs2 vwait
  return vs3
}}
}}
func (i64, i64) -> (i64) {{
block 0 (vinst: i64, vas: i64) {{
  vn = i32.const 100000000
  br 1(vn)
}}
block 1 (vi: i32) {{
  vone = i32.const 1
  vj = i32.sub vi vone
  br_if vj 1(vj) 2()
}}
block 2 () {{
  vseven = i64.const 7
  return vseven
}}
}}
{rec}",
        module_at = 17408 + SpawnRec::MODULE_AT,
        budget_at = 17408 + SpawnRec::BUDGET_AT,
        rec = rec::segment(17408, &SpawnRec::default()),
    )
}

/// The root's powerbox: an `Instantiator` over its window, an unbounded `Budget` its nodes are split
/// from, and the child image.
fn setup(m: &Module) -> impl Fn() -> (Host, Vec<Value>) + '_ {
    move || {
        let mut h = Host::new();
        let i = h.grant_instantiator();
        let b = h.grant_budget(-1, -1, -1);
        let c = h.grant_module(&temen_ir::child_image_at(m, 1).expect("child image"));
        (h, [i, b, c].map(Value::I32).to_vec())
    }
}

/// `m` on the Cranelift JIT, over the powerbox [`setup`] builds, on a thread of its own: a run that
/// hangs fails here instead of stalling the suite.
fn on_the_jit(m: &Module) -> Result<Vec<Value>, Trap> {
    let m = Arc::new(m.clone());
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let (mut host, args) = setup(&m)();
        let args: Vec<i64> = args
            .iter()
            .map(|v| match v {
                Value::I32(x) => i64::from(*x),
                other => panic!("unexpected arg {other:?}"),
            })
            .collect();
        let window = MemLayout::image(vec![0u8; 1 << 16]);
        let (jo, _) = temen_run::jit_cap_run(&m, 0, &args, &window, 0, 0, &mut host, None)
            .expect("the JIT compiles it");
        let _ = tx.send(jo);
    });
    match rx
        .recv_timeout(std::time::Duration::from_secs(60))
        .expect("the JIT run hung")
    {
        JitOutcome::Returned(v) => Ok(v.into_iter().map(Value::I64).collect()),
        JitOutcome::Trapped(k) => Err(Trap::from_code(k as i64).expect("a trap code")),
        other => panic!("the JIT run ended as {other:?}"),
    }
}

/// **A closed `Budget` handle frees its slot, and its node still caps a child.** The root splits and
/// closes [`SPLITS`] nodes, more than its table holds, and every split succeeds and every close
/// answers 0. Then it spawns a child from a node with a small fuel ceiling and closes that node's
/// handle at once: a read through it answers `-EBADF`, and the child still runs out of the node's
/// fuel (its `join` answers `OUT_OF_FUEL`).
#[test]
fn a_closed_budget_frees_its_slot_and_its_node_still_caps_a_child() {
    let m = module(&src());
    let want = (EBADF + 100) * 1000 + OUT_OF_FUEL;
    let want = Ran {
        result: Ok(vec![Value::I64(want)]),
        stdout: Vec::new(),
        stderr: Vec::new(),
    };
    agree_on_every_driver("Budget.close", &m, &setup(&m), &want);
    assert_eq!(
        on_the_jit(&m),
        want.result,
        "Budget.close: the Cranelift JIT"
    );
}
