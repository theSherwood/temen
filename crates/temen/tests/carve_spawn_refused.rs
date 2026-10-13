//! #1867 — **the carve spawn is refused on every engine.** Ops 0 (`instantiate`), 5
//! (`instantiate_module`) and 13 (`instantiate_module_named`), and an op-17 record of version 0,
//! spawned a child into a carve of its spawner's window. That placement is retired (INVARIANTS #13):
//! each traps `CapFault` once the `Instantiator` resolves, the answer the oracle gives every op it
//! has no arm for, on the tree-walker, each bytecode driver and the Cranelift JIT alike. No engine
//! runs a spawn another refuses (INVARIANTS #9). The browser's nested path is pinned by
//! `gencorpus`'s `carve_*` entries, against its wasm build.
//!
//! Each form is otherwise well-formed — a granted module where one is named, a 4 KiB carve the
//! window holds, an entry that exists — so only the form is refused. A deleted coroutine op (op 2,
//! #2069) rides along: the bytecode engine lowers every op the oracle has no arm for to the same
//! trap, where it used to decline the whole module.

#[path = "../../temen-interp/tests/support/drivers.rs"]
mod drivers;

use temen_interp::{Host, Trap, Value};
use temen_ir::Module;
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

/// A version-0 record at 17408: entry 1 in a 4 KiB carve at 64 KiB, no pager, the spawner's own
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
    format!("data 17408 \"{bytes}\"\n")
}

/// A parent `(inst, module) -> i64` that makes `call` and returns its handle, then func 1, an entry
/// that returns 42. `call` sees `vi` (the `Instantiator`), `vm` (the module, as an `i64`), the
/// carve's `ve` (entry 1), `vo` (64 KiB), `vs` (`size_log2` 12) and `vz` (0), and `vp` (the record
/// at 17408).
fn parent(call: &str, data: &str) -> Module {
    module(&format!(
        "memory 17
func (i32, i32) -> (i64) {{
block 0 (vi: i32, vmod: i32) {{
  vm = i64.extend_i32_s vmod
  ve = i64.const 1
  vo = i64.const 65536
  vs = i64.const 12
  vz = i64.const 0
  vp = i64.const 17408
  vh = {call}
  vr = i64.extend_i32_s vh
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

fn forms() -> Vec<(&'static str, Module)> {
    vec![
        (
            "op 0",
            parent(
                "call.cap 6 0 (i64, i64, i64, i64) -> (i32) vi (ve, vo, vs, vz)",
                "",
            ),
        ),
        (
            "op 5",
            parent(
                "call.cap 6 5 (i64, i64, i64, i64, i64) -> (i32) vi (vm, vz, vo, vs, vz)",
                "",
            ),
        ),
        (
            "op 13",
            parent(
                "call.cap 6 13 (i64, i64, i64, i64, i64, i64, i64) -> (i32) vi (vm, vz, vz, vz, vo, vs, vz)",
                "",
            ),
        ),
        (
            "a version-0 record",
            parent(
                "call.cap 6 17 (i64) -> (i32) vi (vp)",
                &v0_record(),
            ),
        ),
        (
            "op 2",
            parent(
                "call.cap 6 2 (i64, i64, i64, i64) -> (i32) vi (ve, vo, vs, vz)",
                "",
            ),
        ),
    ]
}

/// The parent's powerbox: an `Instantiator` over its window and the granted child module, with
/// their handles.
fn powerbox() -> (Host, [i32; 2]) {
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 1 << 17);
    let child = host.grant_module(&module(CHILD));
    (host, [inst, child])
}

fn setup() -> (Host, Vec<Value>) {
    let (host, h) = powerbox();
    (host, h.map(Value::I32).to_vec())
}

#[test]
fn every_engine_traps_a_carve_spawn_with_capfault() {
    for (form, m) in forms() {
        for driver in drivers::ALL {
            let ran = drivers::run_on(driver, &m, &setup)
                .unwrap_or_else(|| panic!("{form}: {driver:?} declined the module"));
            assert_eq!(ran.result, Err(Trap::CapFault), "{form} on {driver:?}");
        }
        let (mut host, h) = powerbox();
        let jit = temen_jit::compile_and_run_with_host(
            &m,
            0,
            &h.map(i64::from),
            temen_run::cap_thunk,
            &mut host as *mut Host as *mut core::ffi::c_void,
        )
        .expect("the JIT compiles the module");
        assert_eq!(
            jit,
            JitOutcome::Trapped(TrapKind::CapFault),
            "{form} on the Cranelift JIT"
        );
    }
}
