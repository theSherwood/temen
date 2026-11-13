//! #2996 — a host function the guest calls that touches the guest's window, where only the hardware
//! knows whether the page is there, answers the guest's `MemoryFault` on every engine, and the run
//! ends: on the Cranelift JIT through `try_access`, which returns to the host function instead of
//! unwinding past it, so nothing it holds is leaked.
//!
//! The page each kernel touches is in the reserved tail past the declared window (`memory 16`: the
//! window is 64 KiB, so `TAIL` is inside the reservation but backs nothing).
//!
//! The futex-in-a-fiber kernel is the one that fails on the code before #2996: the JIT's fiber wait
//! read the word under the hub lock, the fault unwound past the lock's guard, and the next wait or
//! notify in the domain blocked on it forever. A kernel that ends in a lock the handler leaked hangs
//! rather than fails, so every kernel runs under a bound and a run that outlives it fails the test.

use std::sync::mpsc;
use std::time::Duration;
use temen_run::{instantiate, Backend, RunConfig};

/// In the reservation, past the 64 KiB window: a page that is not there.
const TAIL: u64 = 1 << 20;

/// How a run of `src` on `backend` ended: `Ok` with its outcome, or `Err` with the error text.
/// Fails the test if it does not end within a minute.
fn run_bounded(src: String, backend: Backend) -> Result<String, String> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let module = temen_text::parse_module(&src).expect("parse");
        temen_verify::verify_module(&module).expect("verify");
        let instance = instantiate(module).expect("instantiate");
        let ran = instance
            .run(backend, &RunConfig::default())
            .map(|r| format!("{:?}", r.outcome));
        let _ = tx.send(ran);
    });
    rx.recv_timeout(Duration::from_secs(60))
        .unwrap_or_else(|_| panic!("{backend:?}: the run did not end (a leaked lock?)"))
}

/// `src` traps `MemoryFault` on every engine.
fn faults_everywhere(what: &str, src: String) {
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        match run_bounded(src.clone(), backend) {
            Err(e) if e.contains("MemoryFault") => {}
            other => panic!("{what} on {backend:?}: expected a MemoryFault, got {other:?}"),
        }
    }
}

/// A `func () -> (i64)` program whose body is `body`, then `rest`.
fn program(body: &str, rest: &str) -> String {
    format!(
        "memory 16
export 0 func \"_start\" 0
func () -> (i64) {{
block 0 () {{
{body}
  vret = i64.const 0
  return vret
  }}
}}
{rest}"
    )
}

/// `i32.atomic.wait` on a word that is not there: the wait thunk's first read.
#[test]
fn a_wait_on_an_absent_word_faults() {
    let body = format!(
        "  va = i64.const {TAIL}
  ve = i32.const 0
  vt = i64.const -1
  vw = i32.atomic.wait va ve vt"
    );
    faults_everywhere("a wait", program(&body, ""));
}

/// The same wait inside a fiber: on the JIT, the read under the hub lock (`fiber_futex_wait`). The
/// root then notifies, which takes that lock — on the old code, forever.
#[test]
fn a_fibers_wait_on_an_absent_word_faults_and_leaves_the_futex_lock_free() {
    let body = "  v0 = ref.func 1
  v1 = i64.const 0
  v2 = cont.new v0 v1
  vs, vv = cont.resume v2 v1
  vn = i64.const 16384
  vc = i32.const 1
  vk = atomic.notify vn vc";
    let fiber = format!(
        "func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  va = i64.const {TAIL}
  ve = i32.const 0
  vt = i64.const -1
  vw = i32.atomic.wait va ve vt
  vr = i64.extend_i32_s vw
  return vr
  }}
}}
"
    );
    faults_everywhere("a fiber's wait", program(body, &fiber));
}

/// `mem.fill` and `mem.copy` whose spans end on a page that is not there (#2534's helpers).
#[test]
fn bulk_ops_onto_an_absent_page_fault() {
    let near = TAIL - 8;
    let fill = format!(
        "  vd = i64.const {near}
  vv = i32.const 65
  vl = i64.const 16
  mem.fill vd vv vl"
    );
    faults_everywhere("a fill", program(&fill, ""));
    let copy = format!(
        "  vd = i64.const 16384
  vs = i64.const {near}
  vl = i64.const 16
  mem.copy vd vs vl"
    );
    faults_everywhere("a copy", program(&copy, ""));
}

/// `gc.roots` into a buffer that is not there.
#[test]
fn gc_roots_into_an_absent_buffer_faults() {
    let body = format!(
        "  vlo = i64.const 0
  vhi = i64.const -1
  vmask = i64.const -1
  vbuf = i64.const {TAIL}
  vcap = i64.const 8
  vn = gc.roots vlo vhi vmask vbuf vcap"
    );
    faults_everywhere("gc.roots", program(&body, ""));
}

// The guest's own faults, where its code runs on a stack the root did not start with. Each must
// still be the guest's `MemoryFault`: the code faulting is emitted code, in its module's arena.

/// A load from a page that is not there, as a fiber's body.
const LOADS_THE_TAIL: &str = "func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  va = i64.const 1048576
  vr = i64.load va
  return vr
  }
}
";

/// A guest load faulting inside a fiber.
#[test]
fn a_guest_fault_in_a_fiber_traps() {
    let body = "  v0 = ref.func 1
  v1 = i64.const 0
  v2 = cont.new v0 v1
  vs, vv = cont.resume v2 v1";
    faults_everywhere("a fiber's load", program(body, LOADS_THE_TAIL));
}

/// A guest load faulting inside a spawned vCPU.
#[test]
fn a_guest_fault_in_a_spawned_vcpu_traps() {
    let body = "  vsp = i64.const 32768
  va = i64.const 0
  vt = thread.spawn 1 vsp va
  vst, vr = thread.join vt";
    faults_everywhere("a spawned vCPU's load", program(body, LOADS_THE_TAIL));
}
