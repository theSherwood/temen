//! #1226 — a fiber whose entry is an **installed §22 unit** function must run on the **bytecode
//! engine** identically to the tree-walk oracle and the Cranelift JIT. This is the exact shape the
//! Forth kernel (#1214) hits with `' word task`: module 0 `cont.new`s a fiber over a `Jit.install`
//! slot that points into an installed unit (a module ≥ 1 entry). Before the fix the bytecode engine
//! resolved a fiber funcref only through module 0's natural table (`& primary.table_mask`) and
//! trapped `FiberFault` for any installed-unit entry; now it resolves **module-aware** through the
//! shared `call.dyn` dispatch table, exactly as `Op::CallIndirect` / the tree-walker's
//! `dispatch_indirect` / the JIT's shared `fn_table`. The tree-walk + Cranelift legs guard against a
//! regression the other way (they already ran this case); the bytecode leg is the one #1226 closes.
#![cfg(all(unix, target_arch = "x86_64"))]

use temen_interp::{bytecode, run_capture_reserved_with_host, Host, Value};
use temen_ir::DEFAULT_RESERVED_LOG2;
use temen_jit::JitOutcome;
use temen_run::{grant_jit_fibers, jit_cap_run};
use temen_text::parse_module;
use temen_verify::verify_module;

// #1094: one NULL guard (16384) above the legacy 4096 scratch — where the parent embeds the blob.
const BLOB_OFF: usize = 20480;
const TABLE_LOG2: u8 = 3; // 8 dispatch slots: room to `Jit.install` the unit past the parent's funcs

// The submitted unit: a generator fiber body `(i64 sp, i64 arg) -> i64` — `suspend arg+1`, then on
// the next resume `return arg+100`. resume(10) → (SUSPENDED, 11); resume(7) → (RETURNED, 107).
const UNIT: &str = "memory 20\n\
func (i64, i64) -> (i64) {\n\
block 0 (sp: i64, arg: i64) {\n\
  one = i64.const 1\n\
  s1 = i64.add arg one\n\
  got = suspend s1\n\
  h = i64.const 100\n\
  r = i64.add got h\n\
  return r\n\
  }\n\
}\n";

// Module 0: compile the embedded unit, **install** it (→ a `call.dyn` slot pointing at the unit's
// func 0), then `cont.new` a fiber over that installed slot and resume it twice — the installed-unit
// fiber entry #1226 fixes. Returns the fiber's final value (107).
const PARENT: &str = "memory 20\n\
func (i32) -> (i64) {\n\
block 0 (jit: i32) {\n\
  ptr = i64.const 20480\n\
  len = i64.const BLOBLEN\n\
  code = call.cap 11 0 (i64, i64) -> (i64) jit (ptr, len)\n\
  slot = call.cap 11 3 (i64) -> (i64) jit (code)\n\
  slot32 = i32.wrap_i64 slot\n\
  sp = i64.const 131072\n\
  gen = cont.new slot32 sp\n\
  a = i64.const 10\n\
  s0, v0 = cont.resume gen a\n\
  b = i64.const 7\n\
  s1, v1 = cont.resume gen b\n\
  return v1\n\
  }\n\
}\n";

/// The parsed+verified parent module and the init memory with the unit blob seeded at `BLOB_OFF`.
fn setup() -> (temen_ir::Module, Vec<u8>) {
    let unit = parse_module(UNIT).expect("parse unit");
    verify_module(&unit).expect("verify unit");
    let blob = temen_encode::encode_module(&unit);
    let src = PARENT.replace("BLOBLEN", &blob.len().to_string());
    let m = parse_module(&src).expect("parse parent");
    verify_module(&m).expect("verify parent");
    let mut init = vec![0u8; BLOB_OFF + blob.len()];
    init[BLOB_OFF..].copy_from_slice(&blob);
    (m, init)
}

const WANT: i64 = 107;

#[test]
fn installed_unit_fiber_entry_agrees_across_engines() {
    let (m, init) = setup();

    // Tree-walk oracle.
    let mut host_i = Host::new();
    let h_i = grant_jit_fibers(&mut host_i, &m, TABLE_LOG2);
    let mut fuel = 50_000_000u64;
    let (ires, imem) = run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(h_i)],
        &mut fuel,
        &init,
        DEFAULT_RESERVED_LOG2,
        &mut host_i,
    );
    let ivals = ires.expect("tree-walk: installed-unit fiber must run, not FiberFault");
    assert_eq!(ivals, vec![Value::I64(WANT)], "tree-walk value");

    // Bytecode engine — the leg #1226 fixes (was `FiberFault` before the module-aware resolve).
    let mut host_b = Host::new();
    let h_b = grant_jit_fibers(&mut host_b, &m, TABLE_LOG2);
    assert_eq!(h_i, h_b, "identical grant mints identical handles");
    let mut fuel_b = 50_000_000u64;
    let (bres, bmem) = bytecode::compile_and_run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(h_b)],
        &mut fuel_b,
        &init,
        DEFAULT_RESERVED_LOG2,
        &mut host_b,
    )
    .expect("bytecode engine is in-subset for this program");
    let bvals = bres.expect("bytecode: installed-unit fiber must run, not FiberFault (#1226)");
    assert_eq!(
        bvals, ivals,
        "bytecode result must equal the tree-walk oracle"
    );
    assert_eq!(bmem, imem, "bytecode final memory must be byte-identical");

    // Cranelift JIT.
    let mut host_j = Host::new();
    let h_j = grant_jit_fibers(&mut host_j, &m, TABLE_LOG2);
    let (jout, jmem) = jit_cap_run(
        &m,
        0,
        &[h_j as i64],
        &init,
        DEFAULT_RESERVED_LOG2,
        TABLE_LOG2,
        &mut host_j,
    )
    .expect("jit run");
    match jout {
        JitOutcome::Returned(slots) => {
            assert_eq!(slots, vec![WANT], "Cranelift JIT value");
        }
        other => panic!("Cranelift JIT diverged: {other:?}"),
    }
    assert_eq!(jmem, imem, "Cranelift final memory must be byte-identical");
}
