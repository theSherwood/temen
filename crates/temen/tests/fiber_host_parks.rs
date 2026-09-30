//! #1952 — a pipe or stdin op that must wait parks **the calling fiber**, on every engine: its resumer
//! gets `FIBER_PARKED` and runs on, and the wake re-admits the fiber, whose op re-executes. Each case is
//! one vCPU: the root resumes a fiber that parks, then does what ends the park (or not) itself — which
//! only works if the park took the fiber alone. Before, a fiber's pipe op read as EOF (the oracle and
//! the JIT), a fiber's stdin read read as EOF (the JIT), and the bytecode engine parked the whole vCPU.
#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

use std::time::Duration;
use temen_interp::{bytecode, run_capture_reserved_with_host, Host, MemLayout, StreamRole, Value};
use temen_jit::JitOutcome;

const SIZE_LOG2: u8 = 17;
/// `cont.resume`'s status for a fiber that parked, and for one that returned.
const PARKED: i64 = 3;
const RETURNED: i64 = 1;

#[derive(Clone, Copy, Debug)]
enum Engine {
    Interp,
    /// The bytecode engine's cooperative pump (also the browser session's, and the wasm-jit's
    /// interp-driven fold's).
    Bytecode,
    /// The bytecode engine's single-vCPU driver an embedder steps (the browser's per-Worker vCPU).
    BytecodeVcpu,
    /// The bytecode engine's OS-thread-parallel driver.
    BytecodeParallel,
    Jit,
}

/// A zeroed window the test leaks, for the drivers that run over a caller's backing.
fn backing() -> std::sync::Arc<temen_interp::Region> {
    let size = 1usize << SIZE_LOG2;
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: a non-zero, 8-aligned layout; leaked for the test's lifetime, never freed.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `size` valid, 8-aligned bytes owned here and never freed.
    std::sync::Arc::new(unsafe { temen_interp::Region::shared(base, size as u64) })
}

/// What a run answers, engine-neutral: the returned `i64`, or the trap's wire code.
type Answer = Result<i64, i64>;

/// Run `src` on `engine` over a fresh host `grant` sets up, failing (not hanging) after 20 s.
fn run(engine: Engine, src: &str, grant: fn(&mut Host) -> Vec<i32>) -> Answer {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut h = Host::new();
        let args = grant(&mut h);
        let win = vec![0u8; 1 << SIZE_LOG2];
        let vals: Vec<Value> = args.iter().map(|&a| Value::I32(a)).collect();
        let mut fuel = 100_000_000u64;
        let of = |r: Result<Vec<Value>, temen_interp::Trap>| match r {
            Ok(v) => match v[..] {
                [Value::I64(n)] => Ok(n),
                ref other => panic!("unexpected result {other:?}"),
            },
            Err(t) => Err(t.code()),
        };
        let answer = match engine {
            Engine::Interp => of(run_capture_reserved_with_host(
                &m, 0, &vals, &mut fuel, &win, SIZE_LOG2, &mut h,
            )
            .0),
            Engine::Bytecode => of(bytecode::compile_and_run_capture_reserved_with_host(
                &m, 0, &vals, &mut fuel, &win, SIZE_LOG2, &mut h,
            )
            .expect("the bytecode engine runs the module")
            .0),
            Engine::BytecodeVcpu => {
                let prog = bytecode::VcpuProgram::compile(&m).expect("compile");
                let mut vcpu =
                    bytecode::Vcpu::new_root_with_powerbox(&prog, 0, &vals, backing(), &win, h)
                        .expect("root");
                match vcpu.run() {
                    bytecode::VcpuEvent::Done(v) => of(Ok(v)),
                    bytecode::VcpuEvent::Trapped(t) => of(Err(t)),
                    _ => panic!("an event this guest cannot raise"),
                }
            }
            Engine::BytecodeParallel => {
                of(bytecode::compile_and_run_capture_over_parallel_with_host(
                    &m,
                    0,
                    &vals,
                    &mut fuel,
                    &win,
                    backing(),
                    &mut h,
                )
                .expect("the parallel driver runs the module")
                .0)
            }
            Engine::Jit => {
                let slots: Vec<i64> = args.iter().map(|&a| i64::from(a)).collect();
                match temen_run::jit_cap_run(
                    &m,
                    0,
                    &slots,
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
                }
            }
        };
        let _ = tx.send(answer);
    });
    rx.recv_timeout(Duration::from_secs(20))
        .unwrap_or_else(|_| panic!("{engine:?}: the run completes"))
}

/// Every engine answers `want`.
fn check(src: &str, grant: fn(&mut Host) -> Vec<i32>, want: i64) {
    let got: Vec<(Engine, Answer)> = [
        Engine::Interp,
        Engine::Bytecode,
        Engine::BytecodeVcpu,
        Engine::BytecodeParallel,
        Engine::Jit,
    ]
    .into_iter()
    .map(|e| (e, run(e, src, grant)))
    .collect();
    assert!(
        got.iter().all(|(_, a)| *a == Ok(want)),
        "want {want} on every engine; got {got:?}"
    );
}

fn pipe(h: &mut Host) -> Vec<i32> {
    let (w, r) = h.grant_pipe();
    vec![r, w]
}

/// The root's shape: resume the fiber (func 1) with `arg`, which parks it; do `between`; resume it
/// again and answer `first status · 100000 + second status · 10000 + the fiber's result`.
fn root_resumes_twice(arg: &str, between: &str, fiber: &str) -> String {
    format!(
        r#"
memory 17
func (i32, i32) -> (i64) {{
block 0 (vr: i32, vw: i32) {{
  vf = ref.func 1
  vsp = i64.const 4096
  vk = cont.new vf vsp
  va = i64.extend_i32_u {arg}
  vs1, vx1 = cont.resume vk va
{between}
  vz = i64.const 0
  vs2, vx2 = cont.resume vk vz
  vs164 = i64.extend_i32_u vs1
  vs264 = i64.extend_i32_u vs2
  vc1 = i64.const 100000
  vc2 = i64.const 10000
  vp1 = i64.mul vs164 vc1
  vp2 = i64.mul vs264 vc2
  vp = i64.add vp1 vp2
  vres = i64.add vp vx2
  return vres
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (v0: i64, v1: i64) {{
  vh = i32.wrap_i64 v1
{fiber}
  }}
}}
"#
    )
}

/// A fiber reads an empty pipe and parks; its root writes `x` and resumes it: the fiber's read
/// re-executes and returns the byte, `1000·1 + 'x'`.
#[test]
fn a_fibers_pipe_read_parks_the_fiber() {
    let src = root_resumes_twice(
        "vr",
        "  vbuf = i64.const 66200
  vx = i32.const 120
  i32.store8 vbuf vx
  vlen = i64.const 1
  vn = call.cap 0 1 (i64, i64) -> (i64) vw (vbuf, vlen)",
        "  vbuf = i64.const 66100
  vlen = i64.const 1
  vn = call.cap 0 0 (i64, i64) -> (i64) vh (vbuf, vlen)
  vk = i64.const 1000
  vnk = i64.mul vn vk
  vb = i32.load8_u vbuf
  vb64 = i64.extend_i32_u vb
  vr = i64.add vnk vb64
  return vr",
    );
    check(
        &src,
        pipe,
        PARKED * 100_000 + RETURNED * 10_000 + 1000 + i64::from(b'x'),
    );
}

/// A fiber fills the pipe (its capacity is the guest's whole 64 KiB), writes one more byte and
/// parks; its root drains the pipe and resumes it: the write re-executes, `1000·1`.
#[test]
fn a_fibers_pipe_write_parks_the_fiber() {
    let src = root_resumes_twice(
        "vw",
        "  vbuf = i64.const 65536
  vcap = i64.const 65536
  vn = call.cap 0 0 (i64, i64) -> (i64) vr (vbuf, vcap)",
        "  vbuf = i64.const 65536
  vcap = i64.const 65536
  vfill = call.cap 0 1 (i64, i64) -> (i64) vh (vbuf, vcap)
  vone = i64.const 1
  vn = call.cap 0 1 (i64, i64) -> (i64) vh (vbuf, vone)
  vk = i64.const 1000
  vr = i64.mul vn vk
  return vr",
    );
    check(&src, pipe, PARKED * 100_000 + RETURNED * 10_000 + 1000);
}

/// A fiber reads a blocking stdin with nothing to read and parks; its root, which nothing blocks,
/// finishes the run without it: `first status · 100 + 42`.
#[test]
fn a_fibers_stdin_read_parks_only_the_fiber() {
    let src = r#"
memory 17
func (i32, i32) -> (i64) {
block 0 (vin: i32, vunused: i32) {
  vf = ref.func 1
  vsp = i64.const 4096
  vk = cont.new vf vsp
  va = i64.extend_i32_u vin
  vs1, vx1 = cont.resume vk va
  vs164 = i64.extend_i32_u vs1
  vc = i64.const 100
  vp = i64.mul vs164 vc
  v42 = i64.const 42
  vres = i64.add vp v42
  return vres
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  vh = i32.wrap_i64 v1
  vbuf = i64.const 66100
  vlen = i64.const 1
  vn = call.cap 0 0 (i64, i64) -> (i64) vh (vbuf, vlen)
  return vn
  }
}
"#;
    fn stdin(h: &mut Host) -> Vec<i32> {
        let vin = h.grant_stream(StreamRole::In);
        h.set_stdin_blocking(true);
        vec![vin, 0]
    }
    check(src, stdin, PARKED * 100 + 42);
}

/// The same fiber pipe read on a **threaded** run (the JIT's locked thunk, which must release the
/// domain's `Host` while the fiber waits): the root resumes its fiber, which parks on the empty pipe,
/// then signals a sibling vCPU through a futex (so the write lands after the park, whatever the
/// interleaving), joins it once it has written `x`, and resumes the fiber, `… + 1000·1 + 'x' + 7`. The
/// bytecode engine's cooperative entry refuses `thread.*`, so its parallel driver answers it, with the
/// oracle and the JIT.
#[test]
fn a_fibers_pipe_read_parks_the_fiber_on_a_threaded_run() {
    let src = r#"
memory 17
func (i32, i32) -> (i64) {
block 0 (vr: i32, vw: i32) {
  vz = i64.const 0
  vw64 = i64.extend_i32_u vw
  vt = thread.spawn 2 vz vw64
  vf = ref.func 1
  vsp = i64.const 4096
  vk = cont.new vf vsp
  va = i64.extend_i32_u vr
  vs1, vx1 = cont.resume vk va
  vflag = i64.const 66300
  vone = i32.const 1
  i32.atomic.store vflag vone
  vwoke = atomic.notify vflag vone
  vj = thread.join vt
  vs2, vx2 = cont.resume vk vz
  vs164 = i64.extend_i32_u vs1
  vs264 = i64.extend_i32_u vs2
  vc1 = i64.const 100000
  vc2 = i64.const 10000
  vp1 = i64.mul vs164 vc1
  vp2 = i64.mul vs264 vc2
  vp = i64.add vp1 vp2
  vq = i64.add vp vx2
  vres = i64.add vq vj
  return vres
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  vh = i32.wrap_i64 v1
  vbuf = i64.const 66100
  vlen = i64.const 1
  vn = call.cap 0 0 (i64, i64) -> (i64) vh (vbuf, vlen)
  vk = i64.const 1000
  vnk = i64.mul vn vk
  vb = i32.load8_u vbuf
  vb64 = i64.extend_i32_u vb
  vr = i64.add vnk vb64
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vh = i32.wrap_i64 varg
  vflag = i64.const 66300
  vzero = i32.const 0
  vinf = i64.const -1
  vst = i32.atomic.wait vflag vzero vinf
  vbuf = i64.const 66200
  vx = i32.const 120
  i32.store8 vbuf vx
  vlen = i64.const 1
  vn = call.cap 0 1 (i64, i64) -> (i64) vh (vbuf, vlen)
  vr = i64.const 7
  return vr
  }
}
"#;
    let want = Ok(PARKED * 100_000 + RETURNED * 10_000 + 1000 + i64::from(b'x') + 7);
    for e in [Engine::Interp, Engine::BytecodeParallel, Engine::Jit] {
        assert_eq!(run(e, src, pipe), want, "{e:?}");
    }
}
