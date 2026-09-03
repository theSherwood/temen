//! PROCESS.md S2 (JIT parity) — `Instantiator.instantiate_named` (op 11) on the **JIT**: a **multi-cap
//! grant list** re-granted into a §14 child **by name**, discovered with `self.resolve` (the
//! general form of op 8's single positional grant). The interpreter has done this since S2
//! (`instantiate_named.rs`); this pins the JIT to the same observable behavior.
//!
//! The JIT keeps the `Host` opaque, so the child powerbox is built host-side by
//! `temen_run::grant_named_child_build`: it reads the guest's `grants_n` 16-byte records
//! `{name_off, name_len, handle, flags}` from the parent window and re-grants each copyable handle
//! under its name (via the shared `Host::spawn_named_child`), so both backends register the same names
//! against the same shared sinks. The child then finds each cap by `self.resolve` — which lowers to
//! the run's `call.cap` thunk with the child host as ctx, so name resolution works unchanged.

#[path = "support/grant_hooks.rs"]
mod grant_hooks_mod;
use grant_hooks_mod::grant_hooks;

use temen_interp::{run_capture_reserved_with_host, Host, StreamRole, Value};
use temen_jit::{compile_and_run_capture_reserved_with_host_ex, JitOutcome};
use temen_text::parse_module;
use temen_verify::verify_module;

/// func 0 (parent, `(Instantiator, stdout_handle, stderr_handle)`): lay out two grant records at
/// window 0/16 with names `"stdout"`@100 and `"stderr"`@110, `instantiate_named` a 64 KiB child at
/// offset 64 KiB granting both, `join`, return the child's result.
///
/// func 1 (child, `(Instantiator)`): resolve `"stdout"` and `"stderr"` by name (each written into its
/// own window first), write `'O'` to the former and `'E'` to the latter, return 7. (Byte-identical to
/// the interpreter's `instantiate_named.rs` source, so the two backends run the same program.)
const SRC: &str = r#"memory 17
func (i32, i32, i32) -> (i64) {
block 0 (vinst: i32, vout: i32, verr: i32) {
  a0 = i64.const 16384
  n100 = i32.const 16484
  i32.store a0 n100
  a4 = i64.const 16388
  n6 = i32.const 6
  i32.store a4 n6
  a8 = i64.const 16392
  i32.store a8 vout
  a12 = i64.const 16396
  z0 = i32.const 0
  i32.store a12 z0
  a16 = i64.const 16400
  n110 = i32.const 16494
  i32.store a16 n110
  a20 = i64.const 16404
  i32.store a20 n6
  a24 = i64.const 16408
  i32.store a24 verr
  a28 = i64.const 16412
  i32.store a28 z0
  cs = i32.const 115
  ct = i32.const 116
  cd = i32.const 100
  co = i32.const 111
  cu = i32.const 117
  ce = i32.const 101
  cr = i32.const 114
  p100 = i64.const 16484
  i32.store8 p100 cs
  p101 = i64.const 16485
  i32.store8 p101 ct
  p102 = i64.const 16486
  i32.store8 p102 cd
  p103 = i64.const 16487
  i32.store8 p103 co
  p104 = i64.const 16488
  i32.store8 p104 cu
  p105 = i64.const 16489
  i32.store8 p105 ct
  p110 = i64.const 16494
  i32.store8 p110 cs
  p111 = i64.const 16495
  i32.store8 p111 ct
  p112 = i64.const 16496
  i32.store8 p112 cd
  p113 = i64.const 16497
  i32.store8 p113 ce
  p114 = i64.const 16498
  i32.store8 p114 cr
  p115 = i64.const 16499
  i32.store8 p115 cr
  ; spawn via record (op 17): entry=1 off=65536 sl=16 quota=0
  q0v0 = i64.const 4294967296
  q0v1 = i64.const 65536
  q0v2 = i64.const -4294967280
  q0v3 = i64.const 4294967295
  q0v4 = i64.const 0
  q0gp = i64.const 16384
  q0v5 = i64.const 2
  q0a0 = i64.const 17536
  i64.store q0a0 q0v0
  q0a1 = i64.const 17544
  i64.store q0a1 q0v1
  q0a2 = i64.const 17552
  i64.store q0a2 q0v2
  q0a3 = i64.const 17560
  i64.store q0a3 q0v3
  q0a4 = i64.const 17568
  i64.store q0a4 q0v4
  q0a5 = i64.const 17576
  i64.store q0a5 q0gp
  q0a6 = i64.const 17584
  i64.store q0a6 q0v5
  vch = call.cap 6 17 (i64) -> (i32) vinst (q0a0)
  r = call.cap 6 1 (i32) -> (i64) vinst (vch)
  return r
  }
}
func (i64) -> (i64) {
block 0 (vci: i64) {
  cs = i32.const 115
  ct = i32.const 116
  cd = i32.const 100
  co = i32.const 111
  cu = i32.const 117
  ce = i32.const 101
  cr = i32.const 114
  a0 = i64.const 16384
  i32.store8 a0 cs
  a1 = i64.const 16385
  i32.store8 a1 ct
  a2 = i64.const 16386
  i32.store8 a2 cd
  a3 = i64.const 16387
  i32.store8 a3 co
  a4 = i64.const 16388
  i32.store8 a4 cu
  a5 = i64.const 16389
  i32.store8 a5 ct
  len6 = i64.const 6
  hout = self.resolve a0 len6
  a16 = i64.const 16400
  cO = i32.const 79
  i32.store8 a16 cO
  one = i64.const 1
  wo = call.cap 0 1 (i64, i64) -> (i64) hout (a16, one)
  a32 = i64.const 16416
  i32.store8 a32 cs
  a33 = i64.const 16417
  i32.store8 a33 ct
  a34 = i64.const 16418
  i32.store8 a34 cd
  a35 = i64.const 16419
  i32.store8 a35 ce
  a36 = i64.const 16420
  i32.store8 a36 cr
  a37 = i64.const 16421
  i32.store8 a37 cr
  herr = self.resolve a32 len6
  a40 = i64.const 16424
  cE = i32.const 69
  i32.store8 a40 cE
  we = call.cap 0 1 (i64, i64) -> (i64) herr (a40, one)
  v7 = i64.const 7
  return v7
  }
}
"#;

/// (parent result, stdout bytes, stderr bytes) on the interpreter.
fn run_interp() -> (Result<Vec<Value>, temen_interp::Trap>, Vec<u8>, Vec<u8>) {
    let m = parse_module(SRC).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, 128 << 10);
    let oh = host.grant_stream(StreamRole::Out);
    let eh = host.grant_stream(StreamRole::Err);
    let mut fuel = 5_000_000u64;
    let (res, _snap) = run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(ih), Value::I32(oh), Value::I32(eh)],
        &mut fuel,
        &[0u8; 128 << 10],
        0,
        &mut host,
    );
    (res, host.stdout_bytes(), host.stderr_bytes())
}

/// (parent outcome, stdout bytes, stderr bytes) on the JIT with the named-grant hooks installed.
fn run_jit() -> (JitOutcome, Vec<u8>, Vec<u8>) {
    let m = parse_module(SRC).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, 128 << 10);
    let oh = host.grant_stream(StreamRole::Out);
    let eh = host.grant_stream(StreamRole::Err);
    let (jo, _jmem) = compile_and_run_capture_reserved_with_host_ex(
        &m,
        0,
        &[ih as i64, oh as i64, eh as i64],
        &[0u8; 128 << 10],
        0,
        temen_run::cap_thunk,
        &mut host as *mut Host as *mut core::ffi::c_void,
        None,
        Some(grant_hooks()),
    )
    .expect("jit");
    (jo, host.stdout_bytes(), host.stderr_bytes())
}

#[test]
fn named_child_resolves_two_grants_matches_interp() {
    let (ir, iout, ierr) = run_interp();
    let (jo, jout, jerr) = run_jit();
    // Interpreter reference: the child resolved both names and wrote one byte to each.
    assert_eq!(ir, Ok(vec![Value::I64(7)]), "interp: child ran and joined");
    assert_eq!(iout, b"O", "interp: name-resolved stdout");
    assert_eq!(ierr, b"E", "interp: name-resolved stderr");
    // JIT parity: same return value, same bytes into the same (shared) sinks.
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[7]),
        "jit: named child must join with 7, got {jo:?}"
    );
    assert_eq!(jout, iout, "jit: name-resolved stdout must match interp");
    assert_eq!(jerr, ierr, "jit: name-resolved stderr must match interp");
}
