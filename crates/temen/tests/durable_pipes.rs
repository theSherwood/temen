//! #1680 — **pipes inside the cut ride a freeze** (DURABILITY §4, "the cut and its boundary").
//!
//! A pipe whose every end is held inside the frozen domain tree is plain data: a byte FIFO and its
//! open-end counts. Until #1680 any live pipe end made a domain unfreezable, so a guest whose
//! children were piped only to each other could not be frozen at all.
//!
//! The guest is `pipe_cross_domain.rs`'s, made durable: a parent mints nothing itself (the host
//! grants it a pipe), re-grants the **write end** into a same-module §14 child, joins it, then reads
//! two bytes from its **read end**. The child writes `"hi"` and returns 7. The freeze lands with the
//! child live and the bytes already in the pipe; the pipe's two write ends are split across the
//! root and the child, so the artifact must carry both tables' ends and one shared FIFO.

use temen_durable::{
    begin_thaw, init_durable_window, read_state, transform_module_assume_confined, write_state,
    STATE_UNWINDING,
};
use temen_interp::{run_capture_reserved_with_host, DurableBinding, Host, Value};
use temen_ir::Module;

const SIZE_LOG2: u8 = 18;
const WINDOW: usize = 1 << SIZE_LOG2;
const TEST_ARENA: temen_ir::durable_abi::ShadowArena = temen_ir::durable_abi::ShadowArena {
    base: 16448,
    end: 65536,
};

/// func 0 (parent, `(Instantiator, read_end, write_end)`): spawn func 1 through an op-17 record in the
/// carve `[128 KiB, 256 KiB)`, re-granting the write end as `"g"`; join; read 2 bytes from the read
/// end into 70200 and return `n·65536 + byte0·256 + byte1`. Scratch sits above the shadow arena.
///
/// func 1 (child): write `"hi"` through `"g"` and return 7.
const SRC: &str = "memory 18 shadow 16448 65536
func (i32, i32, i32) -> (i64) {
block 0 (vinst: i32, vread: i32, vwrite: i32) {
  vg = i32.const 103
  vname = i64.const 70000
  i32.store8 vname vg
  vgr0 = i64.const 70008
  vno = i32.const 70000
  i32.store vgr0 vno
  vgr1 = i64.const 70012
  vnl1 = i32.const 1
  i32.store vgr1 vnl1
  vgr2 = i64.const 70016
  i32.store vgr2 vwrite
  vgr3 = i64.const 70020
  vz32 = i32.const 0
  i32.store vgr3 vz32
  rrv0 = i64.const 4294967296
  rrvz = i64.const 0
  rroff = i64.const 131072
  rrv2 = i64.const -4294967279
  rrv3 = i64.const 4294967295
  rrgp = i64.const 70008
  rrgn = i64.const 1
  rra0 = i64.const 70064
  i64.store rra0 rrv0
  rra1 = i64.const 70072
  i64.store rra1 rroff
  rra2 = i64.const 70080
  i64.store rra2 rrv2
  rra3 = i64.const 70088
  i64.store rra3 rrv3
  rra4 = i64.const 70096
  i64.store rra4 rrvz
  rra5 = i64.const 70104
  i64.store rra5 rrgp
  rra6 = i64.const 70112
  i64.store rra6 rrgn
  vch = call.cap 6 17 (i64) -> (i32) vinst (rra0)
  vcr = call.cap 6 1 (i32) -> (i64) vinst (vch)
  vbuf = i64.const 70200
  vlen = i64.const 2
  vrd = call.cap 0 0 (i64, i64) -> (i64) vread (vbuf, vlen)
  vb0 = i32.load8_u vbuf
  vbuf1 = i64.const 70201
  vb1 = i32.load8_u vbuf1
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
  a0 = i64.const 70000
  ch = i32.const 104
  i32.store8 a0 ch
  a1 = i64.const 70001
  ci = i32.const 105
  i32.store8 a1 ci
  vg = i32.const 103
  vnp = i64.const 70512
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

/// `2·65536 + 'h'·256 + 'i'`: the parent read both bytes its child wrote.
const READ_HI: i64 = 2 * 65536 + 104 * 256 + 105;

fn instrumented() -> Module {
    let m = temen_text::parse_module(SRC).expect("parse");
    let inst = transform_module_assume_confined(&m).expect("transform");
    temen_verify::verify_module(&inst).expect("verify");
    inst
}

fn run(
    m: &Module,
    host: &mut Host,
    args: &[Value],
    win: &[u8],
) -> (Result<Vec<Value>, temen_interp::Trap>, Vec<u8>) {
    let mut fuel = 50_000_000u64;
    run_capture_reserved_with_host(m, 0, args, &mut fuel, win, SIZE_LOG2, host)
}

/// **Freeze a parent and its live child piped together, through the codec.** The artifact carries
/// the pipe (the `"hi"` the child wrote before the cut) and an end in each domain's table; the
/// restore rebuilds one FIFO both ends share; a re-freeze is byte-identical (§12.6); and the thaw
/// reads the bytes, reproducing the uninterrupted run.
#[test]
fn a_pipe_between_a_parent_and_its_live_child_rides_the_codec() {
    let m = instrumented();

    let mut host = Host::new();
    host.set_durable(true);
    let ih = host.grant_instantiator(0, WINDOW as u64);
    let (w, r) = host.grant_pipe();
    let args = [Value::I32(ih), Value::I32(r), Value::I32(w)];
    let (base, _) = run(
        &m,
        &mut host,
        &args,
        &init_durable_window(WINDOW, TEST_ARENA),
    );
    assert_eq!(base, Ok(vec![Value::I64(READ_HI)]), "uninterrupted run");

    let mut fhost = Host::new();
    fhost.set_durable(true);
    let fih = fhost.grant_instantiator(0, WINDOW as u64);
    let (fw, fr) = fhost.grant_pipe();
    assert_eq!(
        (fih, fw, fr),
        (ih, w, r),
        "same handle values as the control"
    );
    let mut win = init_durable_window(WINDOW, TEST_ARENA);
    write_state(&mut win, STATE_UNWINDING);
    let (res, snap) = run(&m, &mut fhost, &args, &win);
    assert!(res.is_ok(), "a freeze, not a refusal: {res:?}");
    assert_eq!(read_state(&snap), STATE_UNWINDING, "frozen");
    assert_eq!(fhost.frozen_nested().len(), 1, "the child froze live");
    let child_ends = fhost.frozen_child_state()[0]
        .handles
        .iter()
        .filter(|h| matches!(h.binding, DurableBinding::PipeEnd { write: true, .. }))
        .count();
    assert_eq!(child_ends, 1, "the child's write end rides its own table");

    let artifact =
        temen_snapshot::freeze(&m, &snap, &fhost).expect("a pipe inside the cut serializes");

    let mut thost = Host::new();
    thost.set_durable(true);
    let window = temen_snapshot::restore(&artifact, &m, &mut thost).expect("restores");
    assert_eq!(
        temen_snapshot::freeze(&m, &window, &thost).expect("re-freeze"),
        artifact,
        "canonical re-freeze byte-identical"
    );

    let mut twin = window;
    begin_thaw(&mut twin, TEST_ARENA, 0);
    let (tres, _) = run(&m, &mut thost, &args, &twin);
    assert_eq!(
        tres,
        Ok(vec![Value::I64(READ_HI)]),
        "the thaw reads the bytes the child wrote before the cut"
    );
}
