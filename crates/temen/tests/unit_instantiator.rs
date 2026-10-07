//! A §22 unit's `Instantiator`, by route (#1726, #1578, #2143) — on the tree-walk oracle, every
//! bytecode driver and Cranelift alike.
//!
//! DESIGN §22 gives a unit two ways in. **Install** + `call.dyn` runs it in the caller's own frames,
//! where a spawn is an ordinary *module-aware* spawn: a same-module child (`module = -1`) runs the
//! spawning frame's module — the unit's program, not module 0's — exactly as `thread.spawn` does
//! (`bytecode_parallel_jit.rs::installed_unit_spawns_its_own_module`). **Invoke** runs it as a
//! seam-free leaf, where the whole `Instantiator` is unavailable: a child would outlive the synchronous
//! call over code nothing keeps (an invoked unit is never installed), and no one could join it.
//!
//! The unit spawns detached, through a v1 record the guest carries at 17408, paid from the guest's
//! `"budget"`. The base program *also* has a func 1, returning 7, while the unit's returns 42 — so a
//! child built from the wrong module still runs and answers 7, instead of hiding behind the same trap
//! as "no such function".
//!
//! Before #1726 the install route's carve spawn failed on all three engines, each differently (oracle
//! `Malformed`, bytecode `ThreadFault`, Cranelift `CapFault`); before #2143 its detached spawn ran the
//! decoy on all three. Before #1578 the invoke route spawned on the oracle and Cranelift and
//! `CapFault`ed on bytecode.

#[path = "../../temen-interp/tests/support/drivers.rs"]
mod drivers;
#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use temen_interp::{Host, MemLayout, Trap, Value};
use temen_ir::{SpawnRec, DEFAULT_RESERVED_LOG2};
use temen_jit::{JitOutcome, TrapKind};
use temen_run::{grant_jit, jit_cap_run};
use temen_text::parse_module;
use temen_verify::verify_module;

/// The decoy every guest carries as func 1 — what a child built from module 0 would run.
const DECOY: &str = r#"
func (i64) -> (i64) {
block 0 (v0: i64) {
  v = i64.const 7
  return v
  }
}
"#;

/// Install the unit and `call.dyn` it with the `Instantiator` handle.
const INSTALL: &str = r#"memory 17
func (i32, i32, i32) -> (i64) {
block 0 (vjit: i32, vcode: i32, vinst: i32) {
  vc = i64.extend_i32_u vcode
  vslot = call.cap 11 3 (i64) -> (i64) vjit (vc)
  vs32 = i32.wrap_i64 vslot
  vh = i64.extend_i32_u vinst
  vr = call.dyn (i64) -> (i64) vs32 (vh)
  return vr
  }
}
"#;

/// `Jit.invoke` the unit with the `Instantiator` handle.
const INVOKE: &str = r#"memory 17
func (i32, i32, i32) -> (i64) {
block 0 (vjit: i32, vcode: i32, vinst: i32) {
  vc = i64.extend_i32_u vcode
  vh = i64.extend_i32_u vinst
  vr = call.cap 11 1 (i64, i64) -> (i64) vjit (vc, vh)
  return vr
  }
}
"#;

/// The unit's entry: spawn the guest's record at 17408 — a same-module child at func 1, paid from the
/// `"budget"` it names into the record — join it, and return the child's result.
const UNIT_ENTRY: &str = r#"memory 17
func (i64) -> (i64) {
block 0 (vi64: i64) {
  vi = i32.wrap_i64 vi64
  vnp = i64.const 16400
  vnl = i64.const 6
  vb = self.resolve vnp vnl
  vbp = i64.const 17436
  i32.store vbp vb
  vrp = i64.const 17408
  vh = call.cap 6 17 (i64) -> (i32) vi (vrp)
  vj = call.cap 6 1 (i32) -> (i64) vi (vh)
  return vj
  }
}
"#;

/// The unit: its entry spawns its own func 1, which returns 42 plus the byte at 17408 of its window.
/// The unit's module has no data segments, so its child's window is zeros there; a window seeded
/// with the base program's image instead holds the record's version byte, `1`.
const UNIT: &str = r#"func (i64) -> (i64) {
block 0 (v0: i64) {
  va = i64.const 17408
  vb = i32.load8_u va
  vb64 = i64.extend_i32_u vb
  v = i64.const 42
  vr = i64.add v vb64
  return vr
  }
}
"#;

/// A unit whose child spawns in turn: func 1, in the child's fresh window, names its own `"budget"`,
/// writes a v1 record for func 2 with `module = -1`, spawns and joins it, and returns `100 +` its
/// result. Func 2 returns 42, so 142 means both generations ran the unit's program: the child's
/// `-1` names the module the child runs, which is the unit's.
const NESTED_UNIT: &str = r#"func (i64) -> (i64) {
block 0 (vi64: i64) {
  vi = i32.wrap_i64 vi64
  vnp = i64.const 16400
  vname = i64.const 127978875155810
  i64.store vnp vname
  vnl = i64.const 6
  vb = self.resolve vnp vnl
  vrp = i64.const 17408
  vone = i32.const 1
  i32.store vrp vone
  vep = i64.const 17412
  vtwo = i32.const 2
  i32.store vep vtwo
  vnone = i32.const -1
  vpp = i64.const 17428
  i32.store vpp vnone
  vmp = i64.const 17432
  i32.store vmp vnone
  vbp = i64.const 17436
  i32.store vbp vb
  vgp = i64.const 17480
  i32.store vgp vnone
  vh = call.cap 6 17 (i64) -> (i32) vi (vrp)
  vj = call.cap 6 1 (i32) -> (i64) vi (vh)
  vhundred = i64.const 100
  vr = i64.add vj vhundred
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v = i64.const 42
  return v
  }
}
"#;

/// What a run came to, in one vocabulary across the three engines.
#[derive(Debug, PartialEq)]
enum Outcome {
    Returned(i64),
    CapFault,
    Other(String),
}

/// The guest: `route`, the decoy, the `"budget"` name and the unit entry's spawn record.
fn guest(route: &str) -> temen_ir::Module {
    let rec = rec::segment(17408, &SpawnRec::v1(1));
    let m =
        parse_module(&format!("{route}{DECOY}data 16400 \"budget\"\n{rec}")).expect("parse guest");
    verify_module(&m).expect("verify guest");
    m
}

/// A fresh host with the `Jit`, an `Instantiator` over the whole window and a `"budget"` for the
/// unit's children, the guest registered as the running module, and `unit` compiled into it —
/// deterministic, so every engine sees the same handles `[jit, code, instantiator]`.
fn setup(guest: &temen_ir::Module, unit: &str) -> (Host, [i32; 3]) {
    let mut host = Host::new();
    host.set_self_module(&std::sync::Arc::new(guest.clone()));
    let jit = grant_jit(&mut host, guest, 4);
    let inst = host.grant_instantiator(0, 128 << 10);
    let budget = host.grant_budget(-1, 1 << 20, -1);
    host.register_cap_name("budget", budget);
    let unit = parse_module(&format!("{UNIT_ENTRY}{unit}")).expect("parse unit");
    verify_module(&unit).expect("verify unit");
    let code = host
        .jit_compile(jit, &temen_encode::encode_module(&unit))
        .expect("no trap")
        .expect("compile ok")
        .handle;
    (host, [jit, code, inst])
}

fn interp(r: Result<Vec<Value>, Trap>) -> Outcome {
    match r.as_deref() {
        Ok([Value::I64(x)]) => Outcome::Returned(*x),
        Err(Trap::CapFault) => Outcome::CapFault,
        other => Outcome::Other(format!("{other:?}")),
    }
}

fn cranelift(m: &temen_ir::Module, unit: &str) -> Outcome {
    let (mut host, h) = setup(m, unit);
    let slots: Vec<i64> = h.iter().map(|&x| x as i64).collect();
    let (out, _) = jit_cap_run(
        m,
        0,
        &slots,
        &MemLayout::image(Vec::new()),
        DEFAULT_RESERVED_LOG2,
        4,
        &mut host,
        None,
    )
    .expect("jit run");
    match out {
        JitOutcome::Returned(v) if v.len() == 1 => Outcome::Returned(v[0]),
        JitOutcome::Trapped(TrapKind::CapFault) => Outcome::CapFault,
        other => Outcome::Other(format!("{other:?}")),
    }
}

/// Assert that `route` reaching `unit` comes to `want` on every engine: the tree-walk oracle, the four
/// bytecode drivers ([`drivers::ALL`]) and Cranelift. A failure names every engine that disagreed.
fn on_every_engine(route: &str, unit: &str, want: Outcome) {
    let m = guest(route);
    let powerbox = || {
        let (host, h) = setup(&m, unit);
        (host, h.map(Value::I32).to_vec())
    };
    let mut got: Vec<(String, Outcome)> = drivers::ALL
        .iter()
        .map(|&d| {
            let o = drivers::run_on(d, &m, &powerbox)
                .map_or(Outcome::Other("declined".into()), |r| interp(r.result));
            (format!("{d:?}"), o)
        })
        .collect();
    got.push(("Cranelift".into(), cranelift(&m, unit)));
    let wrong: Vec<_> = got.iter().filter(|(_, o)| *o != want).collect();
    assert!(
        wrong.is_empty(),
        "want {want:?} on every engine, but {wrong:?}"
    );
}

/// #1726, #2143 — an installed unit's same-module child runs the unit's function, and joins.
#[test]
fn an_installed_unit_spawns_its_own_function() {
    on_every_engine(INSTALL, UNIT, Outcome::Returned(42));
}

/// #2143 — the child runs the unit's program as its own, so its `-1` names the unit's program too.
#[test]
fn an_installed_units_child_spawns_the_units_program() {
    on_every_engine(INSTALL, NESTED_UNIT, Outcome::Returned(142));
}

/// #1578 — the same unit, invoked, reaches no `Instantiator` at all: a `CapFault` at the spawn.
#[test]
fn an_invoked_unit_has_no_instantiator() {
    on_every_engine(INVOKE, UNIT, Outcome::CapFault);
}
