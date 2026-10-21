//! #1680 — **pipes inside the cut ride a freeze** (DURABILITY §4, "the cut and its boundary").
//!
//! A pipe whose every end is held inside the frozen domain tree is plain data: a byte FIFO and its
//! open-end counts. Until #1680 any live pipe end made a domain unfreezable.
//!
//! The root holds both ends (the host grants it a pipe): it writes `"hi"` through the write end, runs
//! a loop the freeze lands in, then reads two bytes from the read end. The cut falls with the bytes
//! in the pipe, so the artifact must carry the FIFO and both ends, and the thaw reads the bytes the
//! frozen run wrote. A pipe with an end outside the cut refuses: the codec's
//! `a_pipe_with_an_end_outside_the_cut_refuses`.

use temen_durable::{
    arm_freeze_after_backedges, begin_thaw, init_durable_window, read_state,
    transform_module_assume_confined, STATE_UNWINDING,
};
use temen_interp::{run_capture_reserved_with_host, DurableBinding, Host, Value};
use temen_ir::Module;

const SIZE_LOG2: u8 = 18;
const WINDOW: usize = 1 << SIZE_LOG2;
const TEST_ARENA: temen_ir::durable_abi::ShadowArena =
    temen_ir::durable_abi::ShadowArena::new(16448, 65536);

/// The root `(read_end, write_end)`: write `"hi"` through the write end, count to 10 on a back-edge
/// the armed freeze lands on, then read 2 bytes from the read end into 70200 and return
/// `n·65536 + byte0·256 + byte1`. Scratch sits above the shadow arena.
const SRC: &str = "memory 18 shadow 16448 65536
func (i32, i32) -> (i64) {
block 0 (vread: i32, vwrite: i32) {
  a0 = i64.const 70000
  ch = i32.const 104
  i32.store8 a0 ch
  a1 = i64.const 70001
  ci = i32.const 105
  i32.store8 a1 ci
  vlen = i64.const 2
  vw = call.cap 0 1 (i64, i64) -> (i64) vwrite (a0, vlen)
  vi0 = i64.const 0
  br 1(vread, vi0)
}
block 1 (vr1: i32, vi: i64) {
  vone = i64.const 1
  vi2 = i64.add vi vone
  vlim = i64.const 10
  vmore = i64.lt_u vi2 vlim
  br_if vmore 1(vr1, vi2) 2(vr1)
}
block 2 (vr2: i32) {
  vbuf = i64.const 70200
  vlen2 = i64.const 2
  vrd = call.cap 0 0 (i64, i64) -> (i64) vr2 (vbuf, vlen2)
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
";

/// `2·65536 + 'h'·256 + 'i'`: the root read both bytes it wrote.
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

/// **Freeze a root holding a pipe with bytes in it, through the codec.** The artifact carries the
/// pipe (the `"hi"` written before the cut) and both its ends; a re-freeze is byte-identical
/// (§12.6); and the thaw reads the bytes, reproducing the uninterrupted run.
#[test]
fn a_pipe_with_bytes_in_it_rides_the_codec() {
    let m = instrumented();

    let mut host = Host::new();
    host.set_durable(true);
    let (w, r) = host.grant_pipe();
    let args = [Value::I32(r), Value::I32(w)];
    let (base, _) = run(
        &m,
        &mut host,
        &args,
        &init_durable_window(WINDOW, TEST_ARENA),
    );
    assert_eq!(base, Ok(vec![Value::I64(READ_HI)]), "uninterrupted run");

    let mut fhost = Host::new();
    fhost.set_durable(true);
    assert_eq!(
        fhost.grant_pipe(),
        (w, r),
        "same handle values as the control"
    );
    let mut win = init_durable_window(WINDOW, TEST_ARENA);
    arm_freeze_after_backedges(&mut win, TEST_ARENA, 3);
    let (res, snap) = run(&m, &mut fhost, &args, &win);
    assert!(res.is_ok(), "a freeze, not a refusal: {res:?}");
    assert_eq!(
        read_state(&snap, TEST_ARENA),
        STATE_UNWINDING,
        "frozen in the loop"
    );
    let ends = fhost
        .capture_durable_handles()
        .expect("both ends are inside the cut")
        .iter()
        .filter(|h| matches!(h.binding, DurableBinding::PipeEnd { .. }))
        .count();
    assert_eq!(ends, 2, "both ends ride the root's table");

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
        "the thaw reads the bytes written before the cut"
    );
}
