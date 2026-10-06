//! PROCESS.md S2 — a parent re-grants one of its own coordinate-free capabilities
//! (`Stream`/`Exit`/`Clock`) into a §14 child's powerbox, so the child is **not born destitute** —
//! it can do I/O. This is the load-bearing "children can hold capabilities" primitive the process
//! substrate needs (a shell hands its child stdout/stderr/stdin).
//!
//! §3d: the spelling is the op-17 **record** with a one-entry named-grant list: the parent lays the
//! name + 16-byte grant record in its window, the child resolves the cap **by name**
//! (`self.resolve`) instead of receiving a third entry arg. A forged / non-copyable handle (an
//! index-carrying or window-coordinate cap) still fails the whole spawn closed (`CapFault`) at the
//! record's grant-list validation.
//!
//! The interpreter and the JIT run the same program. The JIT keeps the `Host` opaque, so the child
//! powerbox is built host-side by temen-run's production grant hooks, which `jit_cap_run` installs —
//! so both backends hand the child an identical set of handles and share the parent's stdout sink
//! (stdio inheritance).

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use std::sync::Arc;
use temen_interp::{run_capture_reserved_with_host, Host, MemLayout, StreamRole, Trap, Value};
use temen_ir::{Module, SpawnRec};
use temen_jit::{JitOutcome, TrapKind};
use temen_text::parse_module;
use temen_verify::verify_module;

/// func 0 (parent, `(Instantiator, grant_handle, Budget)`): spawn the child (func 1) detached through
/// the v1 record at 20544, paid from the `Budget` and re-granting the parent's `grant_handle` under
/// the name `"g"` (name at 20480, grant record at 20488), then `join` and return the child's result.
///
/// func 1 (child, `(Instantiator, AddressSpace)`): write the three bytes `"hi\n"` into its own
/// window above its NULL guard, resolve `"g"` by name, `Stream.write` them through it, then return 7.
const SRC: &str = "memory 17\n\
func (i32, i32, i32) -> (i64) {\n\
block 0 (vinst: i32, vstream: i32, vbud: i32) {\n\
  vg = i32.const 103\n\
  vnp1 = i64.const 20480\n\
  i32.store8 vnp1 vg\n\
  vgr0 = i64.const 20488\n\
  vno = i32.const 20480\n\
  i32.store vgr0 vno\n\
  vgr1 = i64.const 20492\n\
  vnl1 = i32.const 1\n\
  i32.store vgr1 vnl1\n\
  vgr2 = i64.const 20496\n\
  i32.store vgr2 vstream\n\
  vgr3 = i64.const 20500\n\
  vz32 = i32.const 0\n\
  i32.store vgr3 vz32\n\
  rrb = i64.const 20572\n\
  i32.store rrb vbud\n\
  rra0 = i64.const 20544\n\
  vch = call.cap 6 17 (i64) -> (i32) vinst (rra0)\n\
  vres = call.cap 6 1 (i32) -> (i64) vinst (vch)\n\
  return vres\n\
  }\n\
}\n\
func (i64, i64) -> (i64) {\n\
block 0 (vcinst: i64, vcas: i64) {\n\
  v0 = i64.const 16640\n\
  vhb = i32.const 104\n\
  i32.store8 v0 vhb\n\
  v1 = i64.const 16641\n\
  vib = i32.const 105\n\
  i32.store8 v1 vib\n\
  v2 = i64.const 16642\n\
  vnb = i32.const 10\n\
  i32.store8 v2 vnb\n\
  vg = i32.const 103\n\
  vnp = i64.const 16700\n\
  i32.store8 vnp vg\n\
  vnl = i64.const 1\n\
  vsh = self.resolve vnp vnl\n\
  vlen = i64.const 3\n\
  vw = call.cap 0 1 (i64, i64) -> (i64) vsh (v0, vlen)\n\
  v7 = i64.const 7\n\
  return v7\n\
  }\n\
}\n";

/// [`SRC`] with its spawn record.
fn module() -> Module {
    let spawn = SpawnRec {
        grants_ptr: 20488,
        grants_n: 1,
        ..SpawnRec::v1(1)
    };
    let m = parse_module(&format!("{SRC}{}", rec::segment(20544, &spawn))).expect("parse");
    verify_module(&m).expect("verify");
    m
}

/// A host for `m` and the parent's three args. `stream_grant` picks the second: the re-grantable
/// `Stream` (happy path) or the non-copyable `Instantiator` (negative path), to prove it is refused.
fn host(m: &Module, stream_grant: bool) -> (Host, [i32; 3]) {
    let mut host = Host::new();
    host.set_self_module(&Arc::new(m.clone()));
    let ih = host.grant_instantiator(0, 128 << 10);
    let sh = host.grant_stream(StreamRole::Out);
    let budget = host.grant_budget(-1, 1 << 20, -1);
    let grant = if stream_grant { sh } else { ih };
    (host, [ih, grant, budget])
}

/// Run [`SRC`] on the interpreter: the parent's result and the effective stdout bytes (the child's
/// output, shared into the parent's sink).
fn run_interp(stream_grant: bool) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    let m = module();
    let (mut host, args) = host(&m, stream_grant);
    let mut fuel = 5_000_000u64;
    let (res, _snap) = run_capture_reserved_with_host(
        &m,
        0,
        &args.map(Value::I32),
        &mut fuel,
        &[0u8; 128 << 10],
        0,
        &mut host,
    );
    (res, host.stdout_bytes())
}

/// Run [`SRC`] on the JIT. Same shape as [`run_interp`].
fn run_jit(stream_grant: bool) -> (JitOutcome, Vec<u8>) {
    let m = module();
    let (mut host, args) = host(&m, stream_grant);
    let (jo, _) = temen_run::jit_cap_run(
        &m,
        0,
        &args.map(i64::from),
        &MemLayout::image(vec![0u8; 128 << 10]),
        0,
        0,
        &mut host,
        None,
    )
    .expect("jit");
    (jo, host.stdout_bytes())
}

#[test]
fn granted_child_writes_stdout_on_both() {
    let (ir, iout) = run_interp(true);
    let (jo, jout) = run_jit(true);
    // Interpreter reference: the child ran, wrote through the re-granted stdout, joined with 7.
    assert_eq!(ir, Ok(vec![Value::I64(7)]), "interp: child ran and joined");
    assert_eq!(
        iout, b"hi\n",
        "interp: child output through re-granted stdout"
    );
    // JIT parity: same return value, same bytes into the same (shared) sink.
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[7]),
        "jit: granted child must join with 7, got {jo:?}"
    );
    assert_eq!(jout, iout, "jit: granted child's stdout must match interp");
}

#[test]
fn non_copyable_grant_capfaults_on_both() {
    // Granting the Instantiator handle (a window-coordinate cap) is refused by the spawn's grant-list
    // validation, so the spawn is a `CapFault` on both backends — never a silent success.
    let (ir, iout) = run_interp(false);
    let (jo, jout) = run_jit(false);
    assert_eq!(ir, Err(Trap::CapFault), "interp: non-copyable grant faults");
    assert!(
        matches!(jo, JitOutcome::Trapped(TrapKind::CapFault)),
        "jit: non-copyable grant must fault, got {jo:?}"
    );
    assert!(iout.is_empty(), "interp: nothing written");
    assert!(jout.is_empty(), "jit: nothing written");
}
