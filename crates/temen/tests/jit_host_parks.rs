//! #1826 — a host call that would block parks and runs again on **every** JIT run, not only in a
//! process tree, as on the oracle: a pipe read waits for a sibling's write, a pipe write for a
//! sibling's drain. Before, the JIT kept the op's placeholder answer, so an empty pipe read as EOF
//! and a full pipe wrote nothing. A park nothing could ever end is the oracle's deadlock verdict,
//! `ThreadFault`: a blocking stdin read, which nothing can feed during a JIT run, used to read as EOF.
#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

use temen_interp::{run_capture_reserved_with_host, Host, MemLayout, StreamRole, Value};
use temen_jit::JitOutcome;

const SIZE_LOG2: u8 = 17;

/// What a run answers, engine-neutral: the returned `i64`, or the trap's wire code.
type Answer = Result<i64, i64>;

/// Run `src` on the oracle and on the JIT, each over a fresh host that `grant` sets up, and return
/// both answers.
fn both(src: &str, grant: impl Fn(&mut Host) -> Vec<i32>) -> (Answer, Answer) {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let win = vec![0u8; 1 << SIZE_LOG2];

    let mut h = Host::new();
    let args: Vec<Value> = grant(&mut h).into_iter().map(Value::I32).collect();
    let mut fuel = 100_000_000u64;
    let (r, _) = run_capture_reserved_with_host(&m, 0, &args, &mut fuel, &win, SIZE_LOG2, &mut h);
    let oracle = match r {
        Ok(v) => match v[..] {
            [Value::I64(n)] => Ok(n),
            ref other => panic!("unexpected result {other:?}"),
        },
        Err(t) => Err(t.code()),
    };

    let mut h = Host::new();
    let args: Vec<i64> = grant(&mut h).into_iter().map(i64::from).collect();
    let jit = match temen_run::jit_cap_run(
        &m,
        0,
        &args,
        &MemLayout::image(win),
        SIZE_LOG2,
        0,
        &mut h,
        None,
    )
    .expect("the JIT runs the module")
    {
        (JitOutcome::Returned(v), _) => Ok(v[0]),
        (JitOutcome::Trapped(t), _) => Err(t.code()),
        (other, _) => panic!("unexpected outcome {other:?}"),
    };
    (oracle, jit)
}

fn pipe(h: &mut Host) -> Vec<i32> {
    let (w, r) = h.grant_pipe();
    vec![r, w]
}

/// The sibling every pipe case spawns: it spins long enough that the root reaches its park first,
/// then does the case's op with the pipe end it was handed, and returns 7.
fn with_sibling(root: &str, sibling_op: &str) -> String {
    format!(
        r#"
memory 17
func (i32, i32) -> (i64) {{
block 0 (vr: i32, vw: i32) {{
{root}
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  vi0 = i64.const 0
  br 1(varg, vi0)
}}
block 1 (va: i64, vi: i64) {{
  vone = i64.const 1
  vi2 = i64.add vi vone
  vlim = i64.const 2000000
  vmore = i64.ne vi2 vlim
  br_if vmore 1(va, vi2) 2(va)
}}
block 2 (vend: i64) {{
  vh = i32.wrap_i64 vend
{sibling_op}
  vr = i64.const 7
  return vr
  }}
}}
"#
    )
}

/// A pipe read of an empty pipe whose write end a spinning sibling holds: it waits for the write,
/// `1000·1 + 'x' + 7`.
#[test]
fn a_pipe_read_waits_for_a_siblings_write() {
    let src = with_sibling(
        "  vz = i64.const 0
  vw64 = i64.extend_i32_u vw
  vt = thread.spawn 1 vz vw64
  vbuf = i64.const 66100
  vlen = i64.const 1
  vn = call.cap 0 0 (i64, i64) -> (i64) vr (vbuf, vlen)
  vj = thread.join vt
  vk = i64.const 1000
  vnk = i64.mul vn vk
  vb = i32.load8_u vbuf
  vb64 = i64.extend_i32_u vb
  vs = i64.add vnk vb64
  vres = i64.add vs vj
  return vres",
        "  vbuf = i64.const 66200
  vx = i32.const 120
  i32.store8 vbuf vx
  vlen = i64.const 1
  vn = call.cap 0 1 (i64, i64) -> (i64) vh (vbuf, vlen)",
    );
    let want = Ok(1000 + i64::from(b'x') + 7);
    assert_eq!(both(&src, pipe), (want, want));
}

/// A pipe write to a full pipe (its capacity is the guest's whole 64 KiB) whose read end a spinning
/// sibling drains: it waits for the drain, `1000·1 + 7`, the second write's count.
#[test]
fn a_pipe_write_waits_for_a_siblings_drain() {
    let src = with_sibling(
        "  vz = i64.const 0
  vr64 = i64.extend_i32_u vr
  vt = thread.spawn 1 vz vr64
  vbuf = i64.const 65536
  vcap = i64.const 65536
  vfill = call.cap 0 1 (i64, i64) -> (i64) vw (vbuf, vcap)
  vone = i64.const 1
  vn = call.cap 0 1 (i64, i64) -> (i64) vw (vbuf, vone)
  vj = thread.join vt
  vk = i64.const 1000
  vnk = i64.mul vn vk
  vres = i64.add vnk vj
  return vres",
        "  vbuf = i64.const 65536
  vcap = i64.const 65536
  vn = call.cap 0 0 (i64, i64) -> (i64) vh (vbuf, vcap)",
    );
    assert_eq!(both(&src, pipe), (Ok(1007), Ok(1007)));
}

/// A blocking stdin read with no input: nothing can feed stdin during a JIT run (the embedder's
/// `push_stdin` needs the `Host` the run holds), and a JIT powerbox run delivers no signal that could
/// interrupt it (#1826), so the park can never end: the deadlock verdict. JIT-only: the oracle counts a
/// stdin park as externally wakeable (its host's signal source can interrupt it), so it waits.
#[test]
fn a_blocking_stdin_read_nothing_can_feed_is_a_deadlock_on_the_jit() {
    let m = temen_text::parse_module(
        r#"
memory 17
func (i32) -> (i64) {
block 0 (vin: i32) {
  vbuf = i64.const 66100
  vlen = i64.const 1
  vn = call.cap 0 0 (i64, i64) -> (i64) vin (vbuf, vlen)
  return vn
  }
}
"#,
    )
    .expect("parse");
    let mut h = Host::new();
    let vin = h.grant_stream(StreamRole::In);
    h.set_stdin_blocking(true);
    let (out, _) = temen_run::jit_cap_run(
        &m,
        0,
        &[i64::from(vin)],
        &MemLayout::image(vec![0u8; 1 << SIZE_LOG2]),
        SIZE_LOG2,
        0,
        &mut h,
        None,
    )
    .expect("the JIT runs the module");
    assert_eq!(out, JitOutcome::Trapped(temen_jit::TrapKind::ThreadFault));
}
