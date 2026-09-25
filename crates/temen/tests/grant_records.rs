//! The §14 by-name grant list is guest data: `grants_n` records of 16 bytes
//! `{name_off: u32, name_len: u32, handle: i32, flags: u32}` read out of the spawner's window. Every
//! engine parses it with one reader (`temen_interp::read_grant_records`, #1736), so a malformed list
//! gets one answer everywhere — a trap in the guest, never a host failure.
//!
//! The case that motivated pinning it: a huge `grants_n`. The records run off the end of the window
//! long before the count is reached, so the answer is the out-of-window read's `MemoryFault`. The
//! native reader used to pre-size its list on the untrusted count, so on Cranelift the same guest
//! aborted the host process on the allocation instead.

use temen_interp::{bytecode, Host, MemLayout, StreamRole, Trap, Value};
use temen_ir::DEFAULT_RESERVED_LOG2;
use temen_jit::{JitOutcome, TrapKind};
use temen_run::jit_cap_run;
use temen_text::parse_module;
use temen_verify::verify_module;

/// `instantiate_module_named` (op 13) of [`CHILD`] over `grants_n` records at `grants_ptr`, into a
/// 128 KiB carve at 128 KiB. Records 0 and 1 at 16640 (clear of the guarded low pages) are written to
/// grant the stream under an empty name, so a list there is well formed.
fn guest(grants_ptr: i64, grants_n: i64) -> temen_ir::Module {
    let src = format!(
        r#"memory 19
func (i32, i32, i32) -> (i64) {{
block 0 (vi: i32, vm: i32, vs: i32) {{
  h0 = i64.const 16648
  i32.store h0 vs
  h1 = i64.const 16664
  i32.store h1 vs
  mh = i64.extend_i32_u vm
  gp = i64.const {grants_ptr}
  gn = i64.const {grants_n}
  ent = i64.const 0
  off = i64.const 131072
  sl = i64.const 17
  qz = i64.const 0
  ch = call.cap 6 13 (i64, i64, i64, i64, i64, i64, i64) -> (i32) vi (mh, gp, gn, ent, off, sl, qz)
  r = i64.extend_i32_s ch
  return r
  }}
}}
"#
    );
    let m = parse_module(&src).expect("parse guest");
    verify_module(&m).expect("verify guest");
    m
}

/// The spawned program: a 128 KiB window, returns 42.
const CHILD: &str = r#"memory 17
func (i64) -> (i64) {
block 0 (v0: i64) {
  r = i64.const 42
  return r
  }
}
"#;

/// The entry's three arguments: the `Instantiator`, the child `Module`, and a copyable stream to grant.
fn host() -> (Host, [i32; 3]) {
    let child = parse_module(CHILD).expect("parse child");
    verify_module(&child).expect("verify child");
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 1 << 19);
    let module = host.grant_module(&child);
    let out = host.grant_stream(StreamRole::Out);
    (host, [inst, module, out])
}

fn oracle(m: &temen_ir::Module) -> Result<Vec<Value>, Trap> {
    let (mut host, h) = host();
    let mut fuel = 50_000_000u64;
    temen_interp::run_with_host(m, 0, &h.map(Value::I32), &mut fuel, &mut host)
}

fn bytecode_engine(m: &temen_ir::Module) -> Result<Vec<Value>, Trap> {
    let (mut host, h) = host();
    let mut fuel = 50_000_000u64;
    bytecode::compile_and_run_with_host(m, 0, &h.map(Value::I32), &mut fuel, &mut host)
        .expect("the bytecode engine accepts the guest")
}

fn cranelift(m: &temen_ir::Module) -> JitOutcome {
    let (mut host, h) = host();
    jit_cap_run(
        m,
        0,
        &h.map(i64::from),
        &MemLayout::image(Vec::new()),
        DEFAULT_RESERVED_LOG2,
        0,
        &mut host,
    )
    .expect("jit run")
    .0
}

/// The list starts past any window, so it fails at record 0 however long it says it is.
#[test]
fn a_huge_grant_count_is_a_memory_fault_on_every_engine() {
    let m = guest(-16, 1 << 60);
    assert_eq!(oracle(&m), Err(Trap::MemoryFault), "oracle");
    assert_eq!(bytecode_engine(&m), Err(Trap::MemoryFault), "bytecode");
    assert!(
        matches!(cranelift(&m), JitOutcome::Trapped(TrapKind::MemoryFault)),
        "cranelift"
    );
}

/// Control: two well-formed records inside the window grant the stream, and the child spawns.
#[test]
fn a_grant_list_inside_the_window_spawns() {
    let m = guest(16640, 2);
    assert_eq!(
        oracle(&m),
        Ok(vec![Value::I64(0)]),
        "oracle: the first child's handle"
    );
    assert_eq!(bytecode_engine(&m), Ok(vec![Value::I64(0)]), "bytecode");
    assert!(
        matches!(cranelift(&m), JitOutcome::Returned(ref v) if v == &[0]),
        "cranelift"
    );
}
