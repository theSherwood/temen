//! #1469 — a §14 child that uses `thread.spawn`/`join` runs on the **Cranelift JIT**, as on the
//! tree-walk oracle. A detached child (spawned from a record, op 17) runs as an executor task; a
//! task's spawned vCPUs run in a domain of the child's own — its thread handles number from 0, they
//! share the child's window and fiber registry — while waits, notifies and the run-wide vCPU count
//! stay the run's. Each program runs on both engines, and the child must have been JIT-compiled.
#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use core::ffi::c_void;
use temen_interp::{run_capture_reserved_with_host, Host, Value};
use temen_ir::{Module, SpawnRec};
use temen_jit::{compile_and_run_capture_reserved_with_host_ex, JitOutcome};

const PARENT_LOG2: u8 = 23;
const CHILD_LOG2: u8 = 16;

/// The parent (args `(instantiator, child module, budget)`): spawns the child detached from the
/// record at 17408, joins it, and returns `status * 1000 + value`: what the child returned, or its
/// trap's wire code times 1000 (#2066).
fn parent_src() -> String {
    let child = SpawnRec {
        size_log2: CHILD_LOG2.into(),
        ..SpawnRec::v1()
    };
    format!(
        "memory {PARENT_LOG2}
func (i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vmod: i32, vbud: i32) {{
  vma = i64.const {module_at}
  i32.store vma vmod
  vba = i64.const {budget_at}
  i32.store vba vbud
  vrp = i64.const 17408
  ch = call.cap 6 17 (i64) -> (i32) vinst (vrp)
  vr_st, vr = call.cap 6 1 (i32) -> (i64, i64) vinst (ch)
  vk = i64.const 1000
  vsk = i64.mul vr_st vk
  vt = i64.add vsk vr
  return vt
  }}
}}
{rec}",
        module_at = 17408 + rec::MODULE_AT,
        budget_at = 17408 + rec::BUDGET_AT,
        rec = rec::segment(17408, &child),
    )
}

fn parse(src: &str) -> Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

fn powerbox(child: &Module) -> (Host, [i32; 3]) {
    let mut host = Host::new();
    let inst = host.grant_instantiator();
    let module = host.grant_module(child);
    let budget = host.grant_budget(-1, -1, -1);
    (host, [inst, module, budget])
}

/// The tree-walk oracle's result, or its trap's name.
fn oracle(parent: &Module, child: &Module) -> Result<i64, String> {
    let (mut host, h) = powerbox(child);
    let mut fuel = 10_000_000u64;
    let init = vec![0u8; 1 << PARENT_LOG2];
    let args = h.map(Value::I32);
    match run_capture_reserved_with_host(parent, 0, &args, &mut fuel, &init, 0, &mut host).0 {
        Ok(v) => match v.as_slice() {
            [Value::I64(x)] => Ok(*x),
            other => panic!("unexpected oracle result {other:?}"),
        },
        Err(t) => Err(format!("{t:?}")),
    }
}

/// The JIT's result, or its trap's name.
fn jit(parent: &Module, child: &Module) -> Result<i64, String> {
    let (mut host, h) = powerbox(child);
    let args = h.map(i64::from);
    let hp = &mut host as *mut Host;
    let (jo, _) = compile_and_run_capture_reserved_with_host_ex(
        parent,
        0,
        &args,
        &[],
        PARENT_LOG2,
        temen_run::cap_thunk,
        hp as *mut c_void,
        Some(temen_run::module_resolver),
        Some(temen_run::production_grant_hooks(temen_run::CapCtx::Raw(
            hp,
        ))),
    )
    .expect("the JIT compiles the parent");
    match jo {
        JitOutcome::Returned(v) => Ok(v[0]),
        JitOutcome::Trapped(t) => Err(format!("{t:?}")),
        o => panic!("jit ended abnormally: {o:?}"),
    }
}

/// Run `child` under the parent on both engines; the JIT must match the oracle and must have
/// compiled the child. Returns the (common) outcome.
fn parity(child: &str) -> Result<i64, String> {
    let c = parse(child);
    let p = parse(&parent_src());
    let want = oracle(&p, &c);
    let before = temen_jit::child_compiles();
    assert_eq!(jit(&p, &c), want, "the JIT matches the oracle");
    assert!(
        temen_jit::child_compiles() > before,
        "the child was JIT-compiled"
    );
    want
}

/// A thread entry: returns `arg * 10`.
const TIMES_TEN: &str = "func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  ten = i64.const 10
  r = i64.mul varg ten
  return r
  }
}
";

/// The child's root spawns two vCPUs and joins both. Its handles are its own (0 and 1), however many
/// vCPUs the parent's domain has: `h0 * 1000 + h1 * 100 + 10 + 20`.
#[test]
fn a_child_spawns_and_joins_its_own_threads() {
    let child = format!(
        "memory {CHILD_LOG2}
func (i64) -> (i64) {{
block 0 (vs: i64) {{
  s0 = i64.const 40960
  a0 = i64.const 1
  h0 = thread.spawn 1 s0 a0
  s1 = i64.const 49152
  a1 = i64.const 2
  h1 = thread.spawn 1 s1 a1
  r0 = thread.join h0
  r1 = thread.join h1
  k = i64.const 1000
  c = i64.const 100
  x0 = i64.extend_i32_s h0
  x1 = i64.extend_i32_s h1
  y0 = i64.mul x0 k
  y1 = i64.mul x1 c
  y = i64.add y0 y1
  r = i64.add r0 r1
  out = i64.add y r
  return out
  }}
}}
{TIMES_TEN}"
    );
    assert_eq!(parity(&child), Ok(130));
}

/// A child's vCPU parks on a word of the child's window until the child's root notifies it, then
/// stores a value the root reads after the join: the wait status (0) + 7, plus 1000 × the stored 5.
#[test]
fn a_childs_thread_waits_for_its_root() {
    let child = format!(
        "memory {CHILD_LOG2}
func (i64) -> (i64) {{
block 0 (vs: i64) {{
  s0 = i64.const 40960
  a0 = i64.const 0
  h = thread.spawn 1 s0 a0
  br 1(h)
  }}
block 1 (hh: i32) {{
  addr = i64.const 32768
  one = i32.const 1
  n = atomic.notify addr one
  zz = i32.const 0
  more = i32.eq n zz
  br_if more 1(hh) 2(hh)
  }}
block 2 (h2: i32) {{
  r = thread.join h2
  pa = i64.const 32776
  v = i64.load pa
  k = i64.const 1000
  kv = i64.mul v k
  out = i64.add kv r
  return out
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  addr = i64.const 32768
  exp = i32.const 0
  inf = i64.const -1
  w = i32.atomic.wait addr exp inf
  pa = i64.const 32776
  five = i64.const 5
  i64.store pa five
  w64 = i64.extend_i32_u w
  seven = i64.const 7
  out = i64.add w64 seven
  return out
  }}
}}"
    );
    assert_eq!(parity(&child), Ok(5007));
}

/// A child's vCPUs share its fiber registry: the root creates a fiber, a spawned vCPU resumes it by
/// handle (the child's first fiber, handle 0) and returns what it returned, `arg + 100`.
#[test]
fn a_childs_threads_share_its_fiber_registry() {
    let child = format!(
        "memory {CHILD_LOG2}
func (i64) -> (i64) {{
block 0 (vs: i64) {{
  f = ref.func 2
  z = i64.const 0
  fh = cont.new f z
  s0 = i64.const 40960
  h = thread.spawn 1 s0 fh
  r = thread.join h
  k = i64.const 1000
  x = i64.mul fh k
  out = i64.add x r
  return out
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  a = i64.const 5
  s, r = cont.resume varg a
  return r
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  c = i64.const 100
  r = i64.add varg c
  return r
  }}
}}"
    );
    assert_eq!(parity(&child), Ok(105));
}

/// A trap in a child's vCPU ends the child's domain — its root, parked in the join, included — and
/// the parent's join of the child answers it, as on the oracle: the thread's `1 / 0`.
#[test]
fn a_trap_in_a_childs_thread_ends_the_child() {
    let child = format!(
        "memory {CHILD_LOG2}
func (i64) -> (i64) {{
block 0 (vs: i64) {{
  s0 = i64.const 40960
  a0 = i64.const 0
  h = thread.spawn 1 s0 a0
  r = thread.join h
  return r
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  one = i64.const 1
  r = i64.div_s one varg
  return r
  }}
}}"
    );
    assert_eq!(parity(&child), Ok(temen_ir::trap_code::DIV_BY_ZERO * 1000));
}

/// A child's root that returns while a vCPU it spawned still runs: the child's outcome is the
/// root's, published at once (the vCPU is still sleeping on a timed wait), and the run ends
/// cleanly with the vCPU outliving its root.
#[test]
fn a_childs_root_returns_while_its_thread_runs() {
    let child = format!(
        "memory {CHILD_LOG2}
func (i64) -> (i64) {{
block 0 (vs: i64) {{
  s0 = i64.const 40960
  a0 = i64.const 0
  h = thread.spawn 1 s0 a0
  r = i64.const 42
  return r
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  addr = i64.const 32768
  exp = i32.const 0
  t = i64.const 50000000
  w = i32.atomic.wait addr exp t
  w64 = i64.extend_i32_u w
  return w64
  }}
}}"
    );
    assert_eq!(parity(&child), Ok(42));
}

/// **A child task that joins a finished vCPU is not blocked.** The child's root spawns W, which
/// waits indefinitely for the store the root makes after joining C, a vCPU that naps 20 µs so the
/// root is parked in `thread.join` when it ends — 300 rounds. The root is a task, so its join parks
/// the *task*, counted by the executor until C's exit broadcast re-offers it — *after* C's exit has
/// dropped the run-wide `live`. W, woken by that exit, reads the predicate in between. Today the
/// parent's instantiator `join` is never counted parked (#1820), which keeps `live > parked` here;
/// this pins that counting it must not turn the window into a false `ThreadFault`. The task half of
/// `futex_deadlock_all_backends.rs`'s `JOIN_THEN_NOTIFY`.
#[test]
fn a_task_joining_a_finished_vcpu_is_not_counted_blocked() {
    let child = format!(
        "memory {CHILD_LOG2}
func (i64) -> (i64) {{
block 0 (vs: i64) {{
  vz = i64.const 0
  br 1(vz)
}}
block 1 (vi: i64) {{
  vz1 = i64.const 0
  vc = thread.spawn 1 vz1 vz1
  vw = thread.spawn 2 vz1 vi
  vrc = thread.join vc
  vk = i64.const 16392
  vone = i64.const 1
  vnext = i64.add vi vone
  vn32 = i32.wrap_i64 vnext
  i32.atomic.store vk vn32
  vcnt = i32.const 1
  vwoke = atomic.notify vk vcnt
  vrw = thread.join vw
  vrounds = i64.const 300
  vmore = i64.lt_u vnext vrounds
  br_if vmore 1(vnext) 2(vnext)
}}
block 2 (vr: i64) {{
  return vr
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  vaddr = i64.const 16384
  vexp = i32.const 0
  vto = i64.const 20000
  vst = i32.atomic.wait vaddr vexp vto
  vz = i64.const 0
  return vz
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  vk = i64.const 16392
  vexp = i32.wrap_i64 varg
  vinf = i64.const -1
  vst = i32.atomic.wait vk vexp vinf
  vst64 = i64.extend_i32_u vst
  return vst64
  }}
}}"
    );
    assert_eq!(parity(&child), Ok(300));
}
