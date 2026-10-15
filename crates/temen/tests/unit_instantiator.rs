//! A §22 unit's `Instantiator`, by route (#1578, #2219) — on the tree-walk oracle, every bytecode
//! driver and Cranelift alike.
//!
//! DESIGN §22 gives a unit two ways in. **Install** + `call.dyn` runs it in the caller's own frames,
//! where a spawn is an ordinary spawn: its record names a `Module` the domain holds, and the child
//! starts at that module's function 0. `-1` names no module, from a unit's frame as from the
//! program's (#2219); it used to name the spawning frame's program, the unit's own (#1726, #2143).
//! **Invoke** runs a unit as a seam-free leaf, where the whole `Instantiator` is unavailable: a child
//! would outlive the synchronous call over code nothing keeps (an invoked unit is never installed),
//! and no one could join it.
//!
//! The unit spawns detached, through a v1 record the guest carries at 17408, paid from the guest's
//! `"budget"`, running the module the host granted as `"kid"`, whose function 0 returns 42.
//!
//! Before #1578 the invoke route spawned on the oracle and Cranelift and `CapFault`ed on bytecode.

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

/// The module the host grants as `"kid"`: its function 0 returns 42.
const KID: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  v = i64.const 42
  return v
  }
}
"#;

/// The unit's module operand: the `"kid"` grant, resolved by name.
const NAMES_KID: &str =
    "  vkp = i64.const 16416\n  vkl = i64.const 3\n  vm = self.resolve vkp vkl\n";

/// The unit's module operand: `-1`, which names no module.
const MINUS_ONE: &str = "  vm = i32.const -1\n";

/// The unit: put the `"budget"` and `module` (IR that defines `vm`) into the guest's record at
/// 17408, spawn it, join it, and return the child's result.
fn unit(module: &str) -> String {
    format!(
        "memory 17
func (i64) -> (i64) {{
block 0 (vi64: i64) {{
  vi = i32.wrap_i64 vi64
  vnp = i64.const 16400
  vnl = i64.const 6
  vb = self.resolve vnp vnl
  vbp = i64.const 17436
  i32.store vbp vb
{module}  vmp = i64.const 17432
  i32.store vmp vm
  vrp = i64.const 17408
  vh = call.cap 6 17 (i64) -> (i32) vi (vrp)
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vi (vh)
  return vj
  }}
}}
"
    )
}

/// What a run came to, in one vocabulary across the three engines.
#[derive(Debug, PartialEq)]
enum Outcome {
    Returned(i64),
    CapFault,
    Other(String),
}

/// The guest: `route`, the `"budget"` and `"kid"` names and the unit's spawn record.
fn guest(route: &str) -> temen_ir::Module {
    let rec = rec::segment(17408, &SpawnRec::v1());
    let m = parse_module(&format!(
        "{route}data 16400 \"budget\"\ndata 16416 \"kid\"\n{rec}"
    ))
    .expect("parse guest");
    verify_module(&m).expect("verify guest");
    m
}

/// A fresh host with the `Jit`, an `Instantiator` over the whole window, a `"budget"` for the unit's
/// children and the `"kid"` module, the guest registered as the running module, and the unit naming
/// `module` compiled into it — deterministic, so every engine sees the same handles
/// `[jit, code, instantiator]`.
fn setup(guest: &temen_ir::Module, module: &str) -> (Host, [i32; 3]) {
    let mut host = Host::new();
    host.set_self_module(&std::sync::Arc::new(guest.clone()));
    let jit = grant_jit(&mut host, guest, 4);
    let inst = host.grant_instantiator();
    let budget = host.grant_budget(-1, 1 << 20, -1);
    host.register_cap_name("budget", budget);
    let kid = host.grant_module(&parse_module(KID).expect("parse kid"));
    host.register_cap_name("kid", kid);
    let unit = parse_module(&unit(module)).expect("parse unit");
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

fn cranelift(m: &temen_ir::Module, module: &str) -> Outcome {
    let (mut host, h) = setup(m, module);
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

/// Assert that `route` reaching the unit that spawns `module` comes to `want` on every engine: the
/// tree-walk oracle, the four bytecode drivers ([`drivers::ALL`]) and Cranelift. A failure names
/// every engine that disagreed.
fn on_every_engine(route: &str, module: &str, want: Outcome) {
    let m = guest(route);
    let powerbox = || {
        let (host, h) = setup(&m, module);
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
    got.push(("Cranelift".into(), cranelift(&m, module)));
    let wrong: Vec<_> = got.iter().filter(|(_, o)| *o != want).collect();
    assert!(
        wrong.is_empty(),
        "want {want:?} on every engine, but {wrong:?}"
    );
}

/// An installed unit spawns a module the domain holds, and joins it.
#[test]
fn an_installed_unit_spawns_a_granted_module() {
    on_every_engine(INSTALL, NAMES_KID, Outcome::Returned(42));
}

/// #2219 — from an installed unit's frame, `-1` names no module: the spawn faults, as from the
/// program's.
#[test]
fn an_installed_units_minus_one_names_no_module() {
    on_every_engine(INSTALL, MINUS_ONE, Outcome::CapFault);
}

/// #1578 — the same unit, invoked, reaches no `Instantiator` at all: a `CapFault` at the spawn.
#[test]
fn an_invoked_unit_has_no_instantiator() {
    on_every_engine(INVOKE, NAMES_KID, Outcome::CapFault);
}
