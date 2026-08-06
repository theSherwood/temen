//! PROCESS.md S2 — a parent re-grants one of its own coordinate-free capabilities
//! (`Stream`/`Exit`/`Clock`) into a §14 child's powerbox, so the child is **not born destitute** —
//! it can do I/O. This is the load-bearing "children can hold capabilities" primitive the process
//! substrate needs (a shell hands its child stdout/stderr/stdin).
//!
//! §3d: the spelling is the op-17 **record** with a one-entry named-grant list (the old positional
//! op 8 is deleted): the parent lays the name + 16-byte grant record in its window, the child
//! resolves the cap **by name** (`cap.self.resolve`) instead of receiving a third entry arg. A
//! forged / non-copyable handle (an index-carrying or window-coordinate cap) still fails the whole
//! spawn closed (`CapFault`) at the record's grant-list validation.

use svm_interp::{run_capture_reserved_with_host, Host, StreamRole, Trap, Value};
use svm_text::parse_module;
use svm_verify::verify_module;

/// func 0 (parent, `(Instantiator, grant_handle)`): spawn the child (func 1) via the record in a
/// 4 KiB carve at offset 0, re-granting the parent's `grant_handle` under the name `"g"` (name at
/// 4096, grant record at 4104, spawn record at 4160 — all above the carve), then `join` and return
/// the child's result.
///
/// func 1 (child, `(Instantiator, AddressSpace)`): write the three bytes `"hi\n"` into its own
/// window, resolve `"g"` by name, `Stream.write(0, 3)` through it, then return 7.
const SRC: &str = "memory 17\n\
func (i32, i32) -> (i64) {\n\
block 0 (vinst: i32, vstream: i32) {\n\
  vg = i32.const 103\n\
  vnp1 = i64.const 4096\n\
  i32.store8 vnp1 vg\n\
  vgr0 = i64.const 4104\n\
  vno = i32.const 4096\n\
  i32.store vgr0 vno\n\
  vgr1 = i64.const 4108\n\
  vnl1 = i32.const 1\n\
  i32.store vgr1 vnl1\n\
  vgr2 = i64.const 4112\n\
  i32.store vgr2 vstream\n\
  vgr3 = i64.const 4116\n\
  vz32 = i32.const 0\n\
  i32.store vgr3 vz32\n\
  rrv0 = i64.const 4294967296\n\
  rrvz = i64.const 0\n\
  rrv2 = i64.const -4294967284\n\
  rrv3 = i64.const 4294967295\n\
  rrgp = i64.const 4104\n\
  rrgn = i64.const 1\n\
  rra0 = i64.const 4160\n\
  i64.store rra0 rrv0\n\
  rra1 = i64.const 4168\n\
  i64.store rra1 rrvz\n\
  rra2 = i64.const 4176\n\
  i64.store rra2 rrv2\n\
  rra3 = i64.const 4184\n\
  i64.store rra3 rrv3\n\
  rra4 = i64.const 4192\n\
  i64.store rra4 rrvz\n\
  rra5 = i64.const 4200\n\
  i64.store rra5 rrgp\n\
  rra6 = i64.const 4208\n\
  i64.store rra6 rrgn\n\
  vch = cap.call 6 17 (i64) -> (i32) vinst (rra0)\n\
  vres = cap.call 6 1 (i32) -> (i64) vinst (vch)\n\
  return vres\n\
  }\n\
}\n\
func (i64, i64) -> (i64) {\n\
block 0 (vcinst: i64, vcas: i64) {\n\
  v0 = i64.const 0\n\
  vhb = i32.const 104\n\
  i32.store8 v0 vhb\n\
  v1 = i64.const 1\n\
  vib = i32.const 105\n\
  i32.store8 v1 vib\n\
  v2 = i64.const 2\n\
  vnb = i32.const 10\n\
  i32.store8 v2 vnb\n\
  vg = i32.const 103\n\
  vnp = i64.const 512\n\
  i32.store8 vnp vg\n\
  vnl = i64.const 1\n\
  vsh = cap.self.resolve vnp vnl\n\
  vptr = i64.const 0\n\
  vlen = i64.const 3\n\
  vw = cap.call 0 1 (i64, i64) -> (i64) vsh (vptr, vlen)\n\
  v7 = i64.const 7\n\
  return v7\n\
  }\n\
}\n";

fn run(inst_first: bool) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    let m = parse_module(SRC).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, 128 << 10);
    let sh = host.grant_stream(StreamRole::Out);
    // The parent entry is `(Instantiator, grant_handle)`. The happy path passes the Stream as the
    // grant; the negative path passes the Instantiator itself (a window-coordinate cap → not
    // copyable) to prove it is refused.
    let grant = if inst_first { ih } else { sh };
    let mut fuel = 5_000_000u64;
    let (res, _snap) = run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(ih), Value::I32(grant)],
        &mut fuel,
        &[0u8; 128 << 10],
        0,
        &mut host,
    );
    // The child's stdout was shared into the parent host's sink (stdio inheritance), so read the
    // effective bytes, not the now-promoted local `stdout` Vec.
    (res, host.stdout_bytes())
}

#[test]
fn child_writes_stdout_through_inherited_stream() {
    let (res, out) = run(false); // grant the Stream
    assert_eq!(res, Ok(vec![Value::I64(7)]), "child ran and joined");
    assert_eq!(
        out, b"hi\n",
        "the child produced output through the re-granted stdout Stream"
    );
}

#[test]
fn non_copyable_grant_is_capfault() {
    // Passing the Instantiator handle as the grant: a window-coordinate cap is not re-grantable, so
    // `resolve_copyable` refuses it and the `instantiate_granted` cap.call is a `CapFault`.
    let (res, out) = run(true);
    assert_eq!(
        res,
        Err(Trap::CapFault),
        "a non-copyable grant must fault, not silently succeed"
    );
    assert!(out.is_empty(), "nothing should have been written");
}
