//! #1730 — a detached (op 15) child's `readonly` data segments are read-only in its own window on
//! Cranelift, as on the oracle and as in a root on either engine. A detached child is root-shaped, so
//! a write to its const data is a `MemoryFault`, not a successful store.
#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

use temen_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use temen_ir::Module;
use temen_jit::{JitError, JitOutcome, TrapKind};
use temen_text::parse_module;
use temen_verify::verify_module;

const PARENT_LOG2: u8 = 18;

/// The parent (256 KiB, args `(instantiator, child module, budget)`): spawn the child detached with a
/// 64 KiB window, join it, and return what it returns (a child trap propagates at the join).
const PARENT: &str = "memory 18
func (i32, i32, i32) -> (i64) {
block 0 (vinst: i32, vmod: i32, vbud: i32) {
  me = i64.extend_i32_s vmod
  vb = i64.extend_i32_s vbud
  gz = i64.const 0
  sl = i64.const 16
  ch = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) vinst (vb, me, gz, gz, gz, sl, gz)
  vr = call.cap 6 1 (i32) -> (i64) vinst (ch)
  return vr
  }
}
";

/// The child (64 KiB): a readonly segment at 16384 and a writable one at 32768; `body` runs, then
/// the child returns the i32 at `ret`. RO protection is host-page granular, so the two segments sit
/// in different 16 KiB pages (macOS arm64's page) — at 20480 the writable one shared the RO page.
fn child(body: &str, ret: u64) -> String {
    format!(
        "memory 16
data ro 16384 \"abcd\"
data 32768 \"wxyz\"
func (i64) -> (i64) {{
block 0 (v0: i64) {{
{body}  vra = i64.const {ret}
  vv = i32.load vra
  vr = i64.extend_i32_u vv
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

/// The parent's powerbox: `(instantiator, child module, budget)`.
fn powerbox(child: &Module) -> (Host, [i32; 3]) {
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 1 << PARENT_LOG2);
    let modh = host.grant_module(child);
    let budget = host.grant_budget(0, 1 << 20, 0);
    (host, [inst, modh, budget])
}

fn oracle(child: &Module) -> Result<i64, Trap> {
    let parent = parse(PARENT);
    let (mut host, h) = powerbox(child);
    let init = vec![0u8; 1 << PARENT_LOG2];
    let mut fuel = 50_000_000u64;
    let args = h.map(Value::I32);
    let r = run_capture_reserved_with_host(&parent, 0, &args, &mut fuel, &init, 0, &mut host).0;
    r.map(|v| match v[..] {
        [Value::I64(n)] => n,
        ref other => panic!("unexpected result {other:?}"),
    })
}

/// `None` on a target without the child executor.
fn cranelift(child: &Module) -> Option<JitOutcome> {
    let parent = parse(PARENT);
    let (mut host, h) = powerbox(child);
    let args = h.map(i64::from);
    let init = vec![0u8; 1 << PARENT_LOG2];
    match temen_run::jit_cap_run(&parent, 0, &args, &init, PARENT_LOG2, 0, &mut host) {
        Ok((o, _)) => Some(o),
        Err(JitError::Unsupported(_)) => None,
        Err(e) => panic!("JIT run failed: {e:?}"),
    }
}

/// `"abcd"` and `"wxyz"` as the little-endian i32 a load reads.
const ABCD: i64 = 0x6463_6261;
const WXYZ: i64 = 0x7a79_7877;

/// Control: both segments are seeded and readable.
#[test]
fn a_detached_child_reads_its_data_segments() {
    for (at, want) in [(16384, ABCD), (32768, WXYZ)] {
        let c = parse(&child("", at));
        assert_eq!(oracle(&c), Ok(want), "oracle reads {at}");
        if let Some(o) = cranelift(&c) {
            assert_eq!(o, JitOutcome::Returned(vec![want]), "Cranelift reads {at}");
        }
    }
}

/// Control: a store to the writable segment lands.
#[test]
fn a_detached_child_writes_its_writable_segment() {
    let c = parse(&child(
        "  va = i64.const 32768\n  vz = i32.const 7\n  i32.store va vz\n",
        32768,
    ));
    assert_eq!(oracle(&c), Ok(7));
    if let Some(o) = cranelift(&c) {
        assert_eq!(o, JitOutcome::Returned(vec![7]));
    }
}

/// A store to the readonly segment faults on both engines.
#[test]
fn a_detached_childs_store_to_its_readonly_segment_faults() {
    let c = parse(&child(
        "  va = i64.const 16384\n  vz = i32.const 7\n  i32.store va vz\n",
        16384,
    ));
    assert_eq!(oracle(&c), Err(Trap::MemoryFault), "oracle");
    if let Some(o) = cranelift(&c) {
        assert_eq!(o, JitOutcome::Trapped(TrapKind::MemoryFault), "Cranelift");
    }
}
