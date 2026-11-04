//! #2722 — **an idle child pool does not spin.** A detached §14 child runs as a `ChildExec` task
//! (temen#1469); when its root parks with no deadline, the pool must sleep until something wakes it,
//! not re-offer it on a cadence. Before this fix an idle worker swept every parked task every 20 ms,
//! so a `unir` session (≈5 parked vats) burned ~1 CPU core sitting at its prompt — all of it in the
//! `temen-child` workers re-resuming fibers that could not advance.
//!
//! The observable is [`temen_jit::pool_resumes`]: one tick per [`ChildExec::run_once`] (a fiber
//! resume). This test parks a child root forever on word A, sleeps ~300 ms in a sibling vCPU, then
//! `atomic.notify`s A to wake it. A correct pool resumes the root exactly twice — once to reach the
//! park, once when the notify arrives — plus at most one more if the trailing join parks. The old
//! cadence added ~15 (300 ms / 20 ms). So the resume bound both proves the idle pool is quiet and,
//! because the run only completes if the notify actually wakes the forever-parked task, proves the
//! wake still lands without the cadence to fall back on.
//!
//! Its own test binary (one test, so the process-global counter is uncontended) — the pool tests in
//! `jit_child_threads.rs` run concurrently and would pollute a shared count.
#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use core::ffi::c_void;
use temen_ir::{Module, SpawnRec};
use temen_jit::{compile_and_run_capture_reserved_with_host_ex, JitOutcome};

const PARENT_LOG2: u8 = 23;
const CHILD_LOG2: u8 = 16;

/// Milliseconds the sibling vCPU naps before it wakes the forever-parked root. Long enough that the
/// old 20 ms sweep would have re-offered the root ~15 times — far past the bound below — so the test
/// fails on the pre-#2722 code and passes on the fixed pool.
const IDLE_MS: u64 = 300;

/// The most residencies a correct pool spends on the child root: reach the park (1), wake on the
/// notify (1), and at most one more if the trailing `thread.join` parks before the sibling exits.
/// The old cadence blew past this; a healthy idle pool sits well under it.
const MAX_RESUMES: u64 = 6;

/// The parent (args `(instantiator, child module, budget)`): spawns the child detached from the
/// record at 17408, joins it, and returns `status * 1000 + value` — the child's own result (#2066).
fn parent_src() -> String {
    let child = SpawnRec {
        size_log2: CHILD_LOG2.into(),
        ..SpawnRec::default()
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
        module_at = 17408 + SpawnRec::MODULE_AT,
        budget_at = 17408 + SpawnRec::BUDGET_AT,
        rec = rec::segment(17408, &child),
    )
}

/// The child: `func 0` (root) spawns `func 1` (a sibling vCPU), then parks **forever** on word A
/// (32768) — a no-deadline park of the child task. The sibling naps `IDLE_MS` on word B (16384,
/// never notified, so it times out), then `atomic.notify`s A to wake the root. The root joins the
/// sibling and returns `wait_status*1000 + join_status*100 + join_value + 42` = 42 (woken=0,
/// returned=0, value=0): a self-checking marker that the wake delivered and the join completed.
fn child_src() -> String {
    format!(
        "memory {CHILD_LOG2}
func (i64) -> (i64) {{
block 0 (vs: i64) {{
  vstk = i64.const 40960
  varg = i64.const 0
  vh = thread.spawn 1 vstk varg
  vA = i64.const 32768
  vexp = i32.const 0
  vinf = i64.const -1
  vws = i32.atomic.wait vA vexp vinf
  vrs, vr = thread.join vh
  vws64 = i64.extend_i32_s vws
  vk1 = i64.const 1000
  va = i64.mul vws64 vk1
  vk2 = i64.const 100
  vb = i64.mul vrs vk2
  vab = i64.add va vb
  vabr = i64.add vab vr
  v42 = i64.const 42
  vout = i64.add vabr v42
  return vout
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  vB = i64.const 16384
  vbexp = i32.const 0
  vto = i64.const {idle_ns}
  vbws = i32.atomic.wait vB vbexp vto
  vA = i64.const 32768
  vone = i32.const 1
  vn = atomic.notify vA vone
  vz = i64.const 0
  return vz
  }}
}}",
        idle_ns = IDLE_MS * 1_000_000,
    )
}

fn parse(src: &str) -> Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// Run `parent` (spawning `child`) on the JIT; return its single i64 result.
fn jit(parent: &Module, child: &Module) -> i64 {
    let mut host = temen_interp::Host::new();
    let inst = host.grant_instantiator();
    let module = host.grant_module(child);
    let budget = host.grant_budget(-1, -1, -1);
    let args = [inst, module, budget].map(i64::from);
    let hp = &mut host as *mut temen_interp::Host;
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
        JitOutcome::Returned(v) => v[0],
        other => panic!("jit ended abnormally: {other:?}"),
    }
}

/// A forever-parked child root is woken by a notify, not by a cadence sweep: the run completes with
/// the self-checking 42, and the pool resumes the root only a handful of times across a 300 ms idle
/// window (not once per 20 ms, as the pre-#2722 sweep did).
#[test]
fn an_idle_child_pool_does_not_spin() {
    let parent = parse(&parent_src());
    let child = parse(&child_src());

    let before = temen_jit::pool_resumes();
    assert_eq!(
        jit(&parent, &child),
        42,
        "the notify woke the forever-parked root"
    );
    let resumes = temen_jit::pool_resumes() - before;

    assert!(
        resumes <= MAX_RESUMES,
        "an idle child pool re-resumed the parked root {resumes} times across {IDLE_MS}ms \
         (bound {MAX_RESUMES}); the cadence sweep is back"
    );
}
