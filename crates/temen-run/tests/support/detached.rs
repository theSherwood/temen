//! The op-15 (`instantiate_detached`) fixtures more than one test binary drives: a parent that
//! spawns a child detached, the host that grants it, and its run on the tree-walker and the JIT. A binary that holds a
//! process-wide resource (the address space, #2305) needs its own process, so it includes these by
//! `#[path]` rather than copying them.
#![allow(dead_code)] // each including binary uses a different subset

use core::ffi::c_void;
use temen_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use temen_jit::{compile_and_run_capture_reserved_with_host_ex, GrantChildHooks, JitOutcome};

/// #1234 — the production table, derived from one [`temen_run::CapCtx`] so the hook family and
/// the parent pointer it decodes are chosen together (this used to hand-roll both, and nothing
/// checked that the pointer matched the ctx the run baked).
pub fn grant_hooks(host: *mut Host) -> GrantChildHooks {
    temen_run::production_grant_hooks(temen_run::CapCtx::Raw(host))
}

pub fn module(text: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(text).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// A host granting the parent `[Instantiator, child Module, Budget]` — a budget of `minter_quota`
/// bytes of `mem`.
pub fn host(child: &temen_ir::Module, minter_quota: u64) -> (Host, [i32; 3]) {
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 1u64 << 17);
    let modh = host.grant_module(child);
    let budget = host.grant_budget(-1, (minter_quota) as i64, -1);
    (host, [inst, modh, budget])
}

/// `v0` Instantiator, `v1` the child `Module`, `v2` the `Budget`: spawn the child detached (window
/// 2^16, grants `(gp, gn)`), and if the spawn is refused return the room left in `v2`, else `-1`.
pub fn parent_room_after_refusal(gp: u64, gn: u64) -> String {
    format!(
        r#"memory 17
func (i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32) {{
  vmh = i64.extend_i32_u v1
  vmin = i64.extend_i32_u v2
  vz = i64.const 0
  vgp = i64.const {gp}
  vgn = i64.const {gn}
  vlog = i64.const 16
  vh = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vgp, vgn, vz, vlog, vz, vz, vz)
  vz32 = i32.const 0
  vneg = i32.lt_s vh vz32
  br_if vneg 1(v2) 2()
}}
block 1 (vb: i32) {{
  vf = i64.const 1
  vr = call.cap 14 1 (i64) -> (i64) vb (vf)
  return vr
}}
block 2 () {{
  vm = i64.const -1
  return vm
  }}
}}
"#
    )
}

/// Run `parent` on the JIT over `host` and its `h` handles, its window reserving
/// `2^reserved_log2` bytes, with no fuel armed for the root.
pub fn jit_on_reserved(
    parent: &temen_ir::Module,
    host: &mut Host,
    h: [i32; 3],
    reserved_log2: u8,
) -> JitOutcome {
    let args = [h[0] as i64, h[1] as i64, h[2] as i64];
    let (jo, _) = compile_and_run_capture_reserved_with_host_ex(
        parent,
        0,
        &args,
        &[],
        reserved_log2,
        temen_run::cap_thunk,
        host as *mut Host as *mut c_void,
        Some(temen_run::module_resolver),
        Some(grant_hooks(host as *mut Host)),
    )
    .expect("jit run");
    jo
}

/// Run `parent` on the tree-walker over `host` and its `h` handles, its window reserving
/// `2^reserved_log2` bytes.
pub fn interp_on_reserved(
    parent: &temen_ir::Module,
    host: &mut Host,
    h: [i32; 3],
    reserved_log2: u8,
) -> Result<Vec<Value>, Trap> {
    let mut fuel = 50_000_000u64;
    let args = [Value::I32(h[0]), Value::I32(h[1]), Value::I32(h[2])];
    run_capture_reserved_with_host(parent, 0, &args, &mut fuel, &[], reserved_log2, host).0
}
