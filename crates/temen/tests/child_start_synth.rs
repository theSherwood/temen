//! `synth_manifest_child_start`: a module a frontend linked itself (a runtime plus a program, no
//! translator-made `_start`) runs as a §14 child. Its synthesized `_start` takes the starter
//! capability, ignores it, calls the entry with the powerbox `sp`, and returns the entry's result
//! widened to the `i64` status `join` reads — spawned carved (op 13) and detached (op 15), on the
//! tree-walk oracle, the cooperative bytecode driver, and Cranelift.
//!
//! And `vm_region_create` in `CHILD_BINDABLE`: a child whose manifest imports it (a separately
//! compiled runtime that can mint §13 regions) binds it to its own `AddressSpace` instead of failing
//! the spawn closed, and mints a region.
#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

use temen_interp::{bytecode, run_capture_reserved_with_host, Host, Trap, Value};
use temen_ir::{Module, ValType};
use temen_jit::{JitError, JitOutcome};
use temen_text::parse_module;
use temen_verify::verify_module;

const PARENT_LOG2: u8 = 23;

/// A library-shaped entry `(i64 sp) -> (i32)`: stores 42 on its data stack and returns it, so the
/// synthesized `_start` must pass a real, writable `sp`.
const STORE: &str = "memory 16
func (i64) -> (i32) {
block 0 (sp: i64) {
  v1 = i32.const 42
  i32.store sp v1
  v2 = i32.load sp
  return v2
  }
}
";

/// Mints a 64 KiB region through the manifest import and returns 1 if it got a handle.
const MINT: &str = "memory 16
import 0 \"vm_region_create\" (i64) -> (i64)
func (i64) -> (i32) {
block 0 (sp: i64) {
  len = i64.const 65536
  h = call.import 0 (len)
  z = i64.const 0
  ok = i64.ge_s h z
  return ok
  }
}
";

/// The parent (args `(instantiator, child module, budget)`): spawn the child carved at 4 MiB (op 13)
/// or detached (op 15) with a `2^child_log2` window, join it, return its status.
fn parent(detached: bool, child_log2: u8) -> String {
    let spawn = if detached {
        "ch = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) vinst (vb, me, gz, gz, gz, sl, gz)"
    } else {
        "ch = call.cap 6 13 (i64, i64, i64, i64, i64, i64, i64) -> (i32) vinst (me, gz, gz, gz, off, sl, gz)"
    };
    format!(
        "memory {PARENT_LOG2}
func (i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vmod: i32, vbud: i32) {{
  me = i64.extend_i32_s vmod
  vb = i64.extend_i32_s vbud
  gz = i64.const 0
  off = i64.const 4194304
  sl = i64.const {child_log2}
  {spawn}
  vr = call.cap 6 1 (i32) -> (i64) vinst (ch)
  return vr
  }}
}}
"
    )
}

fn parse(src: &str) -> Module {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    m
}

/// `src`'s function 0 as the entry of a synthesized child `_start`.
fn child(src: &str) -> Module {
    let m = temen_ir::synth_manifest_child_start(parse(src), 0, false).expect("synth");
    verify_module(&m).expect("verify the child");
    m
}

fn powerbox(child: &Module) -> (Host, [i32; 3]) {
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 1 << PARENT_LOG2);
    let modh = host.grant_module(child);
    let budget = host.grant_budget(-1, -1, -1);
    (host, [inst, modh, budget])
}

fn log2(m: &Module) -> u8 {
    m.memory.expect("a window").size_log2
}

fn oracle(child: &Module, detached: bool) -> Result<Vec<Value>, Trap> {
    let parent = parse(&parent(detached, log2(child)));
    let (mut host, h) = powerbox(child);
    let mut fuel = 50_000_000u64;
    let init = vec![0u8; 1 << PARENT_LOG2];
    run_capture_reserved_with_host(
        &parent,
        0,
        &h.map(Value::I32),
        &mut fuel,
        &init,
        0,
        &mut host,
    )
    .0
}

fn bytecode(child: &Module, detached: bool) -> Option<Result<Vec<Value>, Trap>> {
    let parent = parse(&parent(detached, log2(child)));
    let (mut host, h) = powerbox(child);
    let mut fuel = 50_000_000u64;
    bytecode::compile_and_run_with_host(&parent, 0, &h.map(Value::I32), &mut fuel, &mut host)
}

/// `None` on a target without the child executor.
fn cranelift(child: &Module, detached: bool) -> Option<JitOutcome> {
    let parent = parse(&parent(detached, log2(child)));
    let (mut host, h) = powerbox(child);
    let init = vec![0u8; 1 << PARENT_LOG2];
    let args = h.map(i64::from);
    match temen_run::jit_cap_run(&parent, 0, &args, &init, PARENT_LOG2, 0, &mut host) {
        Ok((o, _)) => Some(o),
        Err(JitError::Unsupported(_)) => None,
        Err(e) => panic!("JIT run failed: {e:?}"),
    }
}

/// Runs `child` both ways on every engine and checks each returns `want`.
fn every_engine(child: &Module, want: i64) {
    for detached in [false, true] {
        let how = if detached { "detached" } else { "carved" };
        assert_eq!(
            oracle(child, detached),
            Ok(vec![Value::I64(want)]),
            "oracle, {how}"
        );
        let r = bytecode(child, detached).expect("the bytecode engine lowers the parent");
        assert_eq!(r, Ok(vec![Value::I64(want)]), "bytecode, {how}");
        let o = cranelift(child, detached).expect("Cranelift hosts §14 children here");
        assert_eq!(o, JitOutcome::Returned(vec![want]), "Cranelift, {how}");
    }
}

#[test]
fn the_child_start_has_the_instantiate_module_abi() {
    let m = child(STORE);
    assert_eq!(m.funcs[0].params, vec![ValType::I64], "one starter param");
    assert_eq!(m.funcs[0].results, vec![ValType::I64], "an i64 status");
    assert!(m.exports.iter().any(|e| e.name == "_start" && e.func == 0));
}

#[test]
fn a_linked_module_runs_as_a_child_with_a_real_data_stack() {
    every_engine(&child(STORE), 42);
}

#[test]
fn a_child_binds_vm_region_create_to_its_own_address_space() {
    every_engine(&child(MINT), 1);
}

#[test]
fn a_child_entry_with_no_result_returns_status_zero() {
    let src = "memory 16
func (i64) -> () {
block 0 (sp: i64) {
  return
  }
}
";
    every_engine(&child(src), 0);
}

#[test]
fn a_child_entry_result_that_cannot_widen_is_refused() {
    let src = "memory 16
func (i64) -> (f64) {
block 0 (sp: i64) {
  v = f64.const 1.0
  return v
  }
}
";
    let err = temen_ir::synth_manifest_child_start(parse(src), 0, false).unwrap_err();
    assert!(err.contains("i64 status"), "{err}");
}
