//! #1867 — **the carve spawn is refused on every engine.** Ops 0 (`instantiate`), 5
//! (`instantiate_module`) and 13 (`instantiate_module_named`), and an op-17 record of version 0,
//! spawned a child into a carve of its spawner's window. That placement is retired (INVARIANTS #13):
//! each traps `CapFault` once the `Instantiator` resolves, the answer the oracle gives every op it
//! has no arm for, on the tree-walker, each bytecode driver and the Cranelift JIT alike. No engine
//! runs a spawn another refuses (INVARIANTS #9). The browser's nested path is pinned by
//! `gencorpus`'s `carve_*` entries, against its wasm build.
//!
//! Each form is otherwise well-formed — a granted module where one is named, a 4 KiB carve the
//! window holds, an entry that exists — so only the form is refused, and the same harness runs a
//! version-1 record of the same child to its join on every engine. A deleted coroutine op (op 2,
//! #2069) rides along: the bytecode engine lowers every op the oracle has no arm for to the same
//! trap, where it used to decline the whole module.

#[path = "../../temen-interp/tests/support/drivers.rs"]
mod drivers;
#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use temen_interp::{Host, MemLayout, Trap, Value};
use temen_ir::{Module, SpawnRec};
use temen_jit::{JitOutcome, TrapKind};

fn module(src: &str) -> Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// The module a form names: a 4 KiB window, its entry returning 42.
const CHILD: &str = "memory 12
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 42
  return v1
  }
}
";

/// Where every parent keeps its spawn record.
const REC: u64 = 17408;

/// A version-0 record at [`REC`]: entry 1 in a 4 KiB carve at 64 KiB, no pager, the spawner's own
/// module, no budget, quota or grants. The record's 32-byte tail reads as zeros.
fn v0_record() -> String {
    let mut b = Vec::new();
    for w in [0u32, 1] {
        b.extend_from_slice(&w.to_le_bytes());
    }
    b.extend_from_slice(&65536u64.to_le_bytes());
    for w in [12u32, u32::MAX, u32::MAX, 0] {
        b.extend_from_slice(&w.to_le_bytes());
    }
    let bytes: String = b.iter().map(|x| format!("\\x{x:02x}")).collect();
    format!("data {REC} \"{bytes}\"\n")
}

/// A parent `(inst, module, budget) -> i64` that runs `body` and returns its `vr`, then func 1, an
/// entry that returns 42. `body` sees `vi` (the `Instantiator`), `vmod` and `vb` (the module and the
/// `Budget`), `vm` (the module, as an `i64`), the carve's `ve` (entry 1), `vo` (64 KiB), `vs`
/// (`size_log2` 12) and `vz` (0), and `vp` (the record at [`REC`]).
fn parent(body: &str, data: &str) -> Module {
    module(&format!(
        "memory 17
func (i32, i32, i32) -> (i64) {{
block 0 (vi: i32, vmod: i32, vb: i32) {{
  vm = i64.extend_i32_s vmod
  ve = i64.const 1
  vo = i64.const 65536
  vs = i64.const 12
  vz = i64.const 0
  vp = i64.const {REC}
{body}
  return vr
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  v1 = i64.const 42
  return v1
  }}
}}
{data}"
    ))
}

/// A parent that makes the spawn `call` and returns its handle.
fn spawn(call: &str, data: &str) -> Module {
    parent(&format!("  vh = {call}\n  vr = i64.extend_i32_s vh"), data)
}

fn forms() -> Vec<(&'static str, Module)> {
    vec![
        (
            "op 0",
            spawn(
                "call.cap 6 0 (i64, i64, i64, i64) -> (i32) vi (ve, vo, vs, vz)",
                "",
            ),
        ),
        (
            "op 5",
            spawn(
                "call.cap 6 5 (i64, i64, i64, i64, i64) -> (i32) vi (vm, vz, vo, vs, vz)",
                "",
            ),
        ),
        (
            "op 13",
            spawn(
                "call.cap 6 13 (i64, i64, i64, i64, i64, i64, i64) -> (i32) vi (vm, vz, vz, vz, vo, vs, vz)",
                "",
            ),
        ),
        (
            "a version-0 record",
            spawn("call.cap 6 17 (i64) -> (i32) vi (vp)", &v0_record()),
        ),
        (
            "op 2",
            spawn(
                "call.cap 6 2 (i64, i64, i64, i64) -> (i32) vi (ve, vo, vs, vz)",
                "",
            ),
        ),
    ]
}

/// The parent's powerbox: an `Instantiator` over its window, the granted child module, and a
/// `Budget` a detached child's window spends, with their handles.
fn powerbox() -> (Host, [i32; 3]) {
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 1 << 17);
    let child = host.grant_module(&module(CHILD));
    let budget = host.grant_budget(-1, 1 << 20, -1);
    (host, [inst, child, budget])
}

fn setup() -> (Host, Vec<Value>) {
    let (host, h) = powerbox();
    (host, h.map(Value::I32).to_vec())
}

/// The Cranelift JIT as an embedder drives it (`temen_run::jit_cap_run`): the reference host and
/// the production hooks a spawn's admission needs, without which a valid spawn faults too.
fn cranelift(m: &Module) -> JitOutcome {
    let (mut host, h) = powerbox();
    temen_run::jit_cap_run(
        m,
        0,
        &h.map(i64::from),
        &MemLayout::image(Vec::new()),
        temen_ir::DEFAULT_RESERVED_LOG2,
        0,
        &mut host,
        None,
    )
    .expect("the JIT compiles the module")
    .0
}

#[test]
fn every_engine_traps_a_carve_spawn_with_capfault() {
    for (form, m) in forms() {
        for driver in drivers::ALL {
            let ran = drivers::run_on(driver, &m, &setup)
                .unwrap_or_else(|| panic!("{form}: {driver:?} declined the module"));
            assert_eq!(ran.result, Err(Trap::CapFault), "{form} on {driver:?}");
        }
        assert_eq!(
            cranelift(&m),
            JitOutcome::Trapped(TrapKind::CapFault),
            "{form} on the Cranelift JIT"
        );
    }
}

/// The control: a version-1 record naming the same child, paid from the `Budget`, spawns and joins
/// to 42 on every engine — so the harness can spawn, and each refusal above is the form's.
#[test]
fn the_same_harness_runs_a_version_1_spawn() {
    let m = parent(
        &format!(
            "  vma = i64.const {}
  i32.store vma vmod
  vba = i64.const {}
  i32.store vba vb
  vh = call.cap 6 17 (i64) -> (i32) vi (vp)
  vr_st, vr = call.cap 6 1 (i32) -> (i64, i64) vi (vh)",
            REC + rec::MODULE_AT,
            REC + rec::BUDGET_AT,
        ),
        &rec::segment(REC, &SpawnRec::v1(0)),
    );
    for driver in drivers::ALL {
        let ran = drivers::run_on(driver, &m, &setup)
            .unwrap_or_else(|| panic!("{driver:?} declined the module"));
        assert_eq!(ran.result, Ok(vec![Value::I64(42)]), "{driver:?}");
    }
    assert_eq!(
        cranelift(&m),
        JitOutcome::Returned(vec![42]),
        "the Cranelift JIT"
    );
}
