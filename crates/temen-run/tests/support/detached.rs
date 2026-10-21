//! The detached-spawn fixtures more than one test binary drives: a parent that spawns a child
//! detached from a spawn record (`Instantiator` op 17, [`temen_ir::SpawnRec`]), the host that grants
//! it, and its run on the tree-walker and the JIT. A binary that holds a process-wide resource (the
//! address space, #2305) needs its own process, so it includes these by `#[path]` rather than
//! copying them.
#![allow(dead_code)] // each including binary uses a different subset

#[path = "../../../temen-interp/tests/support/rec.rs"]
pub mod rec;

use core::ffi::c_void;
use temen_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use temen_ir::SpawnRec;
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
    let inst = host.grant_instantiator();
    let modh = host.grant_module(child);
    let budget = host.grant_budget(-1, (minter_quota) as i64, -1);
    (host, [inst, modh, budget])
}

/// `v0` Instantiator, `v1` the child `Module`, `v2` the `Budget`: spawn the child detached from the
/// record at 17408 (window 2^16, grants `(gp, gn)`), and if the spawn is refused return the room left
/// in `v2`, else `-1`.
pub fn parent_room_after_refusal(gp: u64, gn: u64) -> String {
    let child = SpawnRec {
        size_log2: 16,
        grants_ptr: gp,
        grants_n: gn,
        ..SpawnRec::default()
    };
    format!(
        r#"memory 17
func (i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32) {{
  vma = i64.const {module_at}
  i32.store vma v1
  vba = i64.const {budget_at}
  i32.store vba v2
  vrp = i64.const 17408
  vh = call.cap 6 17 (i64) -> (i32) v0 (vrp)
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
{rec}"#,
        module_at = 17408 + SpawnRec::MODULE_AT,
        budget_at = 17408 + SpawnRec::BUDGET_AT,
        rec = rec::segment(17408, &child),
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

/// #1956 — a detached child (`memory 16`) that reads a depth `d` from its payload. While `d > 0` it
/// spawns the module it was granted as `"nester"` (itself) detached from its own `"budget"`, by its
/// spawn record at 17408, with `d - 1` as the payload and `"nester"` granted on, joins it and returns
/// `10 * result + d`; at `d = 0` it returns 7. `orphan` makes the `d = 1` level return 5 without
/// joining, and the `d = 0` level count down 200 000 first, so it outlives its parent.
pub fn nester(orphan: bool) -> String {
    let (join, leaf) = if orphan {
        (
            "vfive = i64.const 5\n  return vfive",
            "vn = i64.const 200000\n  br 3(vn)",
        )
    } else {
        (
            "vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vinst (vh)\n  vten = i64.const 10\n  vm = i64.mul vj vten\n  vr = i64.add vm vd1\n  return vr",
            "v7 = i64.const 7\n  return v7",
        )
    };
    let child = SpawnRec {
        size_log2: 16,
        grants_ptr: 20032,
        grants_n: 1,
        args: (24576, 8),
        ..SpawnRec::default()
    };
    format!(
        r#"memory 16
data 20000 "budget"
data 20016 "nester"
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vab = i64.const {args}
  vd = i64.load vab
  vz = i64.const 0
  vleaf = i64.eq vd vz
  br_if vleaf 2() 1(v0, vd)
}}
block 1 (vi: i64, vd1: i64) {{
  vone = i64.const 1
  vnext = i64.sub vd1 vone
  vpp = i64.const 24576
  i64.store vpp vnext
  vnp = i64.const 20000
  vnl = i64.const 6
  vb = self.resolve vnp vnl
  vmp = i64.const 20016
  vml = i64.const 6
  vnest = self.resolve vmp vml
{grant}  vma = i64.const {module_at}
  i32.store vma vnest
  vba = i64.const {budget_at}
  i32.store vba vb
  vrp = i64.const 17408
  vinst = i32.wrap_i64 vi
  vh = call.cap 6 17 (i64) -> (i32) vinst (vrp)
  {join}
}}
block 2 () {{
  {leaf}
}}
block 3 (vc: i64) {{
  vdec = i64.const 1
  vc1 = i64.sub vc vdec
  vdone = i64.eqz vc1
  br_if vdone 4() 3(vc1)
}}
block 4 () {{
  v7l = i64.const 7
  return v7l
  }}
}}
{rec}"#,
        args = temen_ir::module_args_base(),
        grant = rec::grant("g", 20032, 20016, 6, "vnest"),
        module_at = 17408 + SpawnRec::MODULE_AT,
        budget_at = 17408 + SpawnRec::BUDGET_AT,
        rec = rec::segment(17408, &child),
    )
}

/// Run `f` on a thread of its own: a run that deadlocks fails here instead of hanging the suite.
pub fn within_a_minute<T: Send + 'static>(what: &str, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(std::time::Duration::from_secs(60))
        .unwrap_or_else(|_| panic!("{what}: no result within a minute (deadlocked?)"))
}
