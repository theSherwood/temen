//! **Process-parity slice 1 — target.** §14 `instantiate_module_named` (op 13) on the **bytecode
//! engine's cooperative single-thread driver** (`compile_and_run_with_host`, the browser's wasm-safe
//! entry). Op 13 = `instantiate_module` (op 5, already on bytecode — `bytecode_separate_module.rs`) +
//! `instantiate_named` (op 11, re-grant caps by name). The shell's external-command path uses op 13
//! (exec-with-inherited-stdout), so it is the first browser parity gap (STAGE1.md "Remaining work").
//!
//! This pins the target: an op-13 spawn + `join` of a separate-module child, with a re-granted
//! `stdout` in the child's powerbox, must return the **same** result on the bytecode engine as on the
//! tree-walk oracle — single-thread, no OS threads. The child here returns a fixed value (the grant is
//! re-granted but unused), so the test isolates **op-13 lowering + cooperative named-child spawn**;
//! a follow-up has the child actually `write` through the granted `stdout`.
//!
//! **Landed.** `compile_inst` now lowers op 13 (`(INSTANTIATOR, 13)` → `Op::InstantiateModule` with a
//! `grants: Some((grants_ptr, grants_n))` field; op 5 stays `None`, byte-identical). The cooperative
//! `drive` path reads the grant records from the parent window and builds the child's by-name powerbox
//! via the shared `Host::spawn_named_child` (`lib.rs` ~14982) — the same builder the tree-walker's
//! op-13 arm (~7734) uses. The other bytecode drivers (debugger replay, OS-thread parallel) fail op 13
//! closed, since only the cooperative single-thread driver is the browser's wasm-safe entry.
#![cfg(unix)]

use temen_interp::{bytecode, run_capture_reserved_with_host, Host, StreamRole, Trap, Value};
use temen_text::parse_module;
use temen_verify::verify_module;

const WIN: usize = 256 << 10;
const CARVE: u64 = 128 << 10; // the child's carve — its declared `memory 17` (128 KiB)

/// The child ("command"): a 128 KiB window, data `"VM"` at 100; child-entry `(i64 starter) -> (i64)`
/// returns its data byte + 1000. (It ignores its re-granted `stdout` — this test isolates the spawn.)
const CHILD: &str = "memory 17
data 100 \"VM\"
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 100
  v2 = i32.load8_u v1
  v5 = i64.extend_i32_u v2
  v6 = i64.const 1000
  v7 = i64.add v5 v6
  return v7
  }
}
";

/// Parent: lay a grant record at window 16384 (above the #1094 NULL guard) —
/// `{name_off=16484, name_len=6, handle=stdout, flags=0}` with `"stdout"` at 16484 — then
/// `instantiate_module_named(module, grants_ptr=16384, grants_n=1, entry=0, off=CARVE, size_log2=17,
/// quota=0)` (op 13) → `join` (op 1) → the child's result. Args:
/// `(instantiator, module handle, stdout stream)`.
const PARENT: &str = "memory 18
func (i32, i32, i32) -> (i64) {
block 0 (vinst: i32, vmod: i32, vout: i32) {
  r0 = i64.const 16384
  n100 = i32.const 16484
  i32.store r0 n100
  r4 = i64.const 16388
  n6 = i32.const 6
  i32.store r4 n6
  r8 = i64.const 16392
  i32.store r8 vout
  r12 = i64.const 16396
  zf = i32.const 0
  i32.store r12 zf
  cS = i32.const 115
  cT = i32.const 116
  cD = i32.const 100
  cO = i32.const 111
  cU = i32.const 117
  q100 = i64.const 16484
  i32.store8 q100 cS
  q101 = i64.const 16485
  i32.store8 q101 cT
  q102 = i64.const 16486
  i32.store8 q102 cD
  q103 = i64.const 16487
  i32.store8 q103 cO
  q104 = i64.const 16488
  i32.store8 q104 cU
  q105 = i64.const 16489
  i32.store8 q105 cT
  me = i64.extend_i32_s vmod
  gp = i64.const 16384
  gn = i64.const 1
  ent = i64.const 0
  off = i64.const 131072
  sl = i64.const 17
  qz = i64.const 0
  ch = call.cap 6 13 (i64, i64, i64, i64, i64, i64, i64) -> (i32) vinst (me, gp, gn, ent, off, sl, qz)
  r = call.cap 6 1 (i32) -> (i64) vinst (ch)
  return r
  }
}
";

fn grants(host: &mut Host, child: &temen_ir::Module) -> (i32, i32, i32) {
    let ih = host.grant_instantiator(0, WIN as u64);
    let mh = host.grant_module(child);
    let oh = host.grant_stream(StreamRole::Out);
    (ih, mh, oh)
}

fn oracle() -> Result<Vec<Value>, Trap> {
    let parent = parse_module(PARENT).expect("parse parent");
    verify_module(&parent).expect("verify parent");
    let child = parse_module(CHILD).expect("parse child");
    verify_module(&child).expect("verify child");
    let mut host = Host::new();
    let (ih, mh, oh) = grants(&mut host, &child);
    let mut fuel = 50_000_000u64;
    run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(ih), Value::I32(mh), Value::I32(oh)],
        &mut fuel,
        &vec![0u8; WIN],
        0,
        &mut host,
    )
    .0
}

fn bytecode_cooperative() -> Option<Result<Vec<Value>, Trap>> {
    let parent = parse_module(PARENT).expect("parse parent");
    verify_module(&parent).expect("verify parent");
    let child = parse_module(CHILD).expect("parse child");
    verify_module(&child).expect("verify child");
    let mut host = Host::new();
    let (ih, mh, oh) = grants(&mut host, &child);
    let mut fuel = 50_000_000u64;
    let _ = CARVE;
    bytecode::compile_and_run_with_host(
        &parent,
        0,
        &[Value::I32(ih), Value::I32(mh), Value::I32(oh)],
        &mut fuel,
        &mut host,
    )
}

#[test]
fn instantiate_module_named_join_matches_oracle_on_cooperative_bytecode() {
    let want = oracle().expect("oracle runs");
    assert_eq!(
        want,
        vec![Value::I64(1086)],
        "oracle: op-13 named-grant child returns byte+1000",
    );
    let got = bytecode_cooperative().expect(
        "bytecode `compile_and_run_with_host` must drive op 13 (currently None — the slice to close)",
    );
    assert_eq!(
        got.expect("bytecode run ok"),
        want,
        "op 13: bytecode cooperative driver must match the tree-walk oracle",
    );
}
