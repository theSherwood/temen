//! PROCESS.md §4 / S4 — a **cross-domain pipe**: a parent mints a pipe (`grant_pipe`), keeps the read
//! end, and re-grants the **write end** into a §14 child via the v1 record's named-grant list — the
//! child resolves it as `"g"`. The child writes bytes to its granted end; after `join` the parent reads
//! them from its read end. The FIFO is `Arc`-shared, so the two domains see the same queue — the
//! substrate half of `cmd1 | cmd2`.
//!
//! Both backends re-grant the pipe end through the **same** `Host::regrant_into_child` (the interp
//! grant-list path and the JIT's production grant hooks, which `jit_cap_run` installs, both route
//! through it), so this is a cross-backend differential.

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use std::sync::Arc;
use temen_interp::{run_capture_reserved_with_host, Host, MemLayout, Value};
use temen_ir::{Module, SpawnRec};
use temen_jit::JitOutcome;
use temen_text::parse_module;
use temen_verify::verify_module;

/// func 0 (parent, `(Instantiator, read_end, write_end, Budget)`): spawn a child (func 1) detached
/// through the v1 record at 20544, paid from the `Budget` and re-granting the **write end** as `"g"`;
/// `join`; then read 2 bytes from the **read end** into window offset 16400 and encode
/// `read_count * 65536 + byte0 * 256 + byte1`. The child writes `"hi"` (`'h'=104`, `'i'=105`), so the
/// parent reads count `2` and those bytes → `2*65536 + 104*256 + 105` = `157801` — proving the bytes
/// crossed the domain boundary through the shared FIFO.
///
/// func 1 (child, `(Instantiator, AddressSpace)`): write `"hi"` into its own window above its NULL
/// guard, resolve the granted write end as `"g"`, `Stream.write` the two bytes through it, then
/// return 7.
const SRC: &str = "memory 17
func (i32, i32, i32, i32) -> (i64) {
block 0 (vinst: i32, vread: i32, vwrite: i32, vbud: i32) {
  vg = i32.const 103
  vnp1 = i64.const 20480
  i32.store8 vnp1 vg
  vgr0 = i64.const 20488
  vno = i32.const 20480
  i32.store vgr0 vno
  vgr1 = i64.const 20492
  vnl1 = i32.const 1
  i32.store vgr1 vnl1
  vgr2 = i64.const 20496
  i32.store vgr2 vwrite
  rrb = i64.const 20572
  i32.store rrb vbud
  rra0 = i64.const 20544
  vch = call.cap 6 17 (i64) -> (i32) vinst (rra0)
  vcr = call.cap 6 1 (i32) -> (i64) vinst (vch)
  a16 = i64.const 16400
  vlen = i64.const 2
  vrd = call.cap 0 0 (i64, i64) -> (i64) vread (a16, vlen)
  vb0 = i32.load8_u a16
  a17 = i64.const 16401
  vb1 = i32.load8_u a17
  k256 = i32.const 256
  k65536 = i32.const 65536
  vrdi = i32.wrap_i64 vrd
  t0 = i32.mul vrdi k65536
  t1 = i32.mul vb0 k256
  t2 = i32.add t0 t1
  t3 = i32.add t2 vb1
  vresult = i64.extend_i32_u t3
  return vresult
  }
}
func (i64, i64) -> (i64) {
block 0 (vci: i64, vca: i64) {
  a0 = i64.const 16640
  ch = i32.const 104
  i32.store8 a0 ch
  a1 = i64.const 16641
  ci = i32.const 105
  i32.store8 a1 ci
  vg = i32.const 103
  vnp = i64.const 16700
  i32.store8 vnp vg
  vnl = i64.const 1
  vwh = self.resolve vnp vnl
  vlen = i64.const 2
  vw = call.cap 0 1 (i64, i64) -> (i64) vwh (a0, vlen)
  v7 = i64.const 7
  return v7
  }
}
";

/// [`SRC`] with its spawn record, and a host for it with the parent's four args.
fn setup() -> (Module, Host, [i32; 4]) {
    let spawn = SpawnRec {
        grants_ptr: 20488,
        grants_n: 1,
        ..SpawnRec::v1(1)
    };
    let m = parse_module(&format!("{SRC}{}", rec::segment(20544, &spawn))).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    host.set_self_module(&Arc::new(m.clone()));
    let ih = host.grant_instantiator(0, 128 << 10);
    let (w, r) = host.grant_pipe();
    let budget = host.grant_budget(-1, 1 << 20, -1);
    (m, host, [ih, r, w, budget])
}

fn run_interp() -> Result<Vec<Value>, temen_interp::Trap> {
    let (m, mut host, args) = setup();
    let mut fuel = 50_000_000u64;
    run_capture_reserved_with_host(
        &m,
        0,
        &args.map(Value::I32),
        &mut fuel,
        &[0u8; 128 << 10],
        0,
        &mut host,
    )
    .0
}

fn run_jit() -> JitOutcome {
    let (m, mut host, args) = setup();
    temen_run::jit_cap_run(
        &m,
        0,
        &args.map(i64::from),
        &MemLayout::image(vec![0u8; 128 << 10]),
        0,
        0,
        &mut host,
        None,
    )
    .expect("jit")
    .0
}

#[test]
fn child_writes_pipe_parent_reads_matches_interp() {
    let ir = run_interp();
    let jo = run_jit();
    assert_eq!(
        ir,
        Ok(vec![Value::I64(157_801)]),
        "interp: parent read 'hi' the child wrote through the granted pipe end"
    );
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[157_801]),
        "jit: cross-domain pipe must match interp, got {jo:?}"
    );
}
