//! D66 — **the child-domain executor on the native JIT** (DESIGN.md §23; #1600 slice 2). Detached
//! children (op 15) are migrating tasks over a lane-bounded worker pool, not one OS thread each:
//!
//! * **the dispatch-time bound** — with the parent's lane cap at 1, two spinning children never
//!   overlap (a shared pre-mapped region carries the running counter they both bump);
//! * **a park frees the lane** — under that same cap of 1, a child parked in `atomic.wait` lets its
//!   sibling run and notify it, so the pair completes (a pinned worker would deadlock);
//! * **a timed wait fires on an idle worker** — no resumer polls a task, the executor's deadline does;
//! * **teardown unwinds a task parked forever** — a parent that never joins still returns.
//!
//! The interpreter (whose lanes landed in slice 4) is the oracle where its scheduling is
//! deterministic for the program; the executor's own guarantees are pinned directly.

use core::ffi::c_void;
use temen_interp::{run_with_host, Host, Value};
use temen_jit::{compile_and_run_capture_reserved_with_host_ex, GrantChildHooks, JitOutcome};

fn grant_hooks(host: *mut temen_interp::Host) -> GrantChildHooks {
    temen_run::production_grant_hooks(temen_run::CapCtx::Raw(host))
}

fn module(text: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(text).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// `v0` Instantiator, `v1` AddressSpace, `v2` child module A, `v3` child module B, `v4` Budget.
/// Mints a 64 KiB region, maps it at 65536 of its own window, spawns A then B detached with the
/// region pre-mapped at 65536 of each child's window (2^17), then runs `tail`.
fn parent(tail: &str) -> String {
    format!(
        r#"memory 17
func (i32, i32, i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32, v3: i32, v4: i32) {{
  vlen = i64.const 65536
  vrh64 = call.cap 5 5 (i64) -> (i64) v1 (vlen)
  vrh = i32.wrap_i64 vrh64
  vwo = i64.const 65536
  vro = i64.const 0
  vprot = i32.const 3
  vm0 = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vrh (vwo, vro, vlen, vprot)
  vma = i64.extend_i32_u v2
  vmb = i64.extend_i32_u v3
  vb = i64.extend_i32_u v4
  vz = i64.const 0
  vlog = i64.const 17
  vreg = i64.extend_i32_u vrh
  voff = i64.const 65536
  vca = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vb, vma, vz, vz, vz, vlog, vz, vz, vz, vreg, voff)
  vcb = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vb, vmb, vz, vz, vz, vlog, vz, vz, vz, vreg, voff)
  {tail}
  }}
}}
"#
    )
}

/// Join both, then return the region's **violation** word (region byte 8).
const TAIL_VIOLATIONS: &str = "vja = call.cap 6 1 (i32) -> (i64) v0 (vca)
  vjb = call.cap 6 1 (i32) -> (i64) v0 (vcb)
  vviol = i64.const 65544
  vv = i64.load vviol
  return vv";

/// Join both; return `1000·A + B`.
const TAIL_SUM: &str = "vja = call.cap 6 1 (i32) -> (i64) v0 (vca)
  vjb = call.cap 6 1 (i32) -> (i64) v0 (vcb)
  vk = i64.const 1000
  vm = i64.mul vja vk
  vr = i64.add vm vjb
  return vr";

/// Join neither: return at once, leaving the children to run teardown.
const TAIL_LEAVE: &str = "vr = i64.const 7
  return vr";

/// A spinner over the shared region: bump RUN (region byte 0), note a violation at VIOL (byte 8) if
/// another spinner was already running, spin, drop RUN. The interpreter's `lanes.rs` sibling, as a
/// detached child.
const SPINNER: &str = r#"memory 17
func (i64) -> (i64) {
block 0 (v0: i64) {
  vrun = i64.const 65536
  vone = i64.const 1
  vold = i64.atomic.rmw.add vrun vone
  vge = i64.ge_s vold vone
  br_if vge 1() 2()
}
block 1 () {
  vviol = i64.const 65544
  vone1 = i64.const 1
  vx = i64.atomic.rmw.add vviol vone1
  br 2()
}
block 2 () {
  vz = i64.const 0
  br 3(vz)
}
block 3 (vi: i64) {
  vlim = i64.const 20000000
  vlt = i64.lt_u vi vlim
  vone3 = i64.const 1
  vi1 = i64.add vi vone3
  br_if vlt 3(vi1) 4()
}
block 4 () {
  vrun2 = i64.const 65536
  vneg = i64.const -1
  vd = i64.atomic.rmw.add vrun2 vneg
  vz4 = i64.const 0
  return vz4
  }
}
"#;

/// Park on region word X (byte 16) while it is 0, with `timeout` ns (`-1` = forever); return
/// `100·status + X`.
fn waiter(timeout: i64) -> String {
    format!(
        r#"memory 17
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vx = i64.const 65552
  ve = i32.const 0
  vt = i64.const {timeout}
  vs = i32.atomic.wait vx ve vt
  vv = i32.load vx
  vs64 = i64.extend_i32_u vs
  vk = i64.const 100
  vm = i64.mul vs64 vk
  vv64 = i64.extend_i32_u vv
  vr = i64.add vm vv64
  return vr
  }}
}}
"#
    )
}

/// Store 1 to X and notify one waiter; return how many were woken.
const NOTIFIER: &str = r#"memory 17
func (i64) -> (i64) {
block 0 (v0: i64) {
  vx = i64.const 65552
  vone = i32.const 1
  i32.atomic.store vx vone
  vn = atomic.notify vx vone
  vr = i64.extend_i32_u vn
  return vr
  }
}
"#;

const TRIVIAL: &str = r#"memory 17
func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i64.const 0
  return vz
  }
}
"#;

fn host(a: &temen_ir::Module, b: &temen_ir::Module, lane_cap: i64) -> (Host, [i32; 5]) {
    let mut host = Host::new();
    host.set_region_factory(temen_run::new_shared_region);
    host.set_lane_cap(lane_cap);
    let inst = host.grant_instantiator(0, 1u64 << 17);
    let aspace = host.grant_address_space(0, 1u64 << 17);
    let ma = host.grant_module(a);
    let mb = host.grant_module(b);
    let budget = host.grant_budget(0, 2i64 << 17, 0);
    (host, [inst, aspace, ma, mb, budget])
}

fn run_jit(p: &temen_ir::Module, a: &temen_ir::Module, b: &temen_ir::Module, cap: i64) -> i64 {
    let (mut host, h) = host(a, b, cap);
    let args: Vec<i64> = h.iter().map(|&x| x as i64).collect();
    let (jo, _) = compile_and_run_capture_reserved_with_host_ex(
        p,
        0,
        &args,
        &[],
        temen_ir::DEFAULT_RESERVED_LOG2,
        temen_run::cap_thunk,
        &mut host as *mut Host as *mut c_void,
        Some(temen_run::module_resolver),
        Some(grant_hooks(&mut host as *mut Host)),
    )
    .expect("jit run");
    match jo {
        JitOutcome::Returned(ref v) => v.first().copied().unwrap_or(-1),
        ref o => panic!("jit ended abnormally: {o:?}"),
    }
}

fn run_interp(p: &temen_ir::Module, a: &temen_ir::Module, b: &temen_ir::Module, cap: i64) -> i64 {
    let (mut host, h) = host(a, b, cap);
    let args: Vec<Value> = h.iter().map(|&x| Value::I32(x)).collect();
    let mut fuel = u64::MAX;
    let r = run_with_host(p, 0, &args, &mut fuel, &mut host).expect("interp run");
    match r.first() {
        Some(Value::I64(x)) => *x,
        other => panic!("unexpected interp result {other:?}"),
    }
}

/// **The dispatch-time bound on the JIT.** Two spinning detached children under a parent lane cap
/// of 1 never share a worker instant: the lane is taken at each resume and every resume is gated
/// on the whole chain. (Without the cap this program reports violations on ≥2 cores — the mutation
/// the pin was checked against; deliberately not asserted, as in the interpreter's twin.)
#[test]
fn a_parent_lane_cap_of_one_serializes_two_detached_children_on_the_jit() {
    let p = module(&parent(TAIL_VIOLATIONS));
    let s = module(SPINNER);
    assert_eq!(run_interp(&p, &s, &s, 1), 0, "the oracle");
    assert_eq!(
        run_jit(&p, &s, &s, 1),
        0,
        "no two children were ever on a worker at once"
    );
}

/// **A park frees the lane.** Cap 1, child A parks in an infinite `atomic.wait` on the shared word,
/// child B (spawned second, so dispatched second) stores + notifies it. A holds no worker while
/// parked, so B runs on the one lane, its notify wakes A (status 0, X = 1 ⇒ A returns 1; B woke 1)
/// and the parent joins both: `1001`. A worker pinned to the parked child would never run B.
#[test]
fn a_parked_child_hands_its_lane_to_the_sibling_that_wakes_it() {
    let p = module(&parent(TAIL_SUM));
    let a = module(&waiter(-1));
    let b = module(NOTIFIER);
    // The oracle's worker pool does not order two freshly spawned children — B may store before A
    // parks (`101000`, the racy shape `futex_cross_domain.rs` documents) — so it pins the outcome
    // set, and the executor's FIFO dispatch pins the order.
    let interp = run_interp(&p, &a, &b, 1);
    assert!(
        interp == 1001 || interp == 101_000,
        "the oracle completes either way: {interp}"
    );
    assert_eq!(run_jit(&p, &a, &b, 1), 1001);
}

/// **A timed wait fires on an idle worker.** No resumer polls a task; the executor re-dispatches
/// it once its deadline is due. A 1 ms wait nobody notifies returns `TIMED_OUT` (2) with X still
/// 0: `200`, and the trivial sibling `0` ⇒ `200000`.
#[test]
fn a_tasks_timed_wait_times_out_without_a_poller() {
    let p = module(&parent(TAIL_SUM));
    let a = module(&waiter(1_000_000));
    let b = module(TRIVIAL);
    assert_eq!(run_interp(&p, &a, &b, 1), 200_000, "the oracle");
    assert_eq!(run_jit(&p, &a, &b, 1), 200_000);
}

/// **Teardown unwinds a task parked forever.** The parent spawns a child that waits on a word no
/// one stores, and returns without joining. Run teardown poisons the parked task (the domain is
/// over — `DOMAIN_DONE`, D37 death-is-revocation), it unwinds through its trailing guard, and the
/// run completes with the parent's `7` instead of hanging on the child.
#[test]
fn run_teardown_unwinds_a_detached_child_parked_forever() {
    let p = module(&parent(TAIL_LEAVE));
    let a = module(&waiter(-1));
    let b = module(TRIVIAL);
    assert_eq!(run_jit(&p, &a, &b, -1), 7);
    assert_eq!(run_jit(&p, &a, &b, 1), 7, "under a cap too");
}

/// Two **carve** children (Instantiator op 0, same module) under a parent lane cap of 1. A carve
/// child's window is a private image of its carve, written back at finish — so this pins the new
/// copy-back path *and* that the carve path goes through the same dispatch gate without wedging:
/// each child stores a marker in its own 4 KiB carve and returns a value the parent joins. Both
/// carves carry their marker afterwards, and the parent sees `42 + 43`.
const TWO_CARVE_CHILDREN: &str = r#"memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  vf1 = i64.const 1
  vf2 = i64.const 2
  voa = i64.const 65536
  vob = i64.const 69632
  vlog = i64.const 12
  vq = i64.const 0
  vca = call.cap 6 0 (i64, i64, i64, i64) -> (i32) v0 (vf1, voa, vlog, vq)
  vcb = call.cap 6 0 (i64, i64, i64, i64) -> (i32) v0 (vf2, vob, vlog, vq)
  vja = call.cap 6 1 (i32) -> (i64) v0 (vca)
  vjb = call.cap 6 1 (i32) -> (i64) v0 (vcb)
  vr = i64.add vja vjb
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  va = i64.const 2048
  vm = i64.const 171
  i64.store va vm
  vr = i64.const 42
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  va = i64.const 2048
  vm = i64.const 172
  i64.store va vm
  vr = i64.const 43
  return vr
  }
}
"#;

#[test]
fn two_carve_children_copy_back_under_a_lane_cap() {
    let m = module(TWO_CARVE_CHILDREN);
    let run = |cap: i64| -> (i64, Vec<u8>) {
        let mut host = Host::new();
        host.set_lane_cap(cap);
        let inst = host.grant_instantiator(0, 1u64 << 17);
        let (jo, mem) = compile_and_run_capture_reserved_with_host_ex(
            &m,
            0,
            &[inst as i64],
            &[],
            temen_ir::DEFAULT_RESERVED_LOG2,
            temen_run::cap_thunk,
            &mut host as *mut Host as *mut c_void,
            Some(temen_run::module_resolver),
            Some(grant_hooks(&mut host as *mut Host)),
        )
        .expect("jit run");
        match jo {
            JitOutcome::Returned(ref v) => (v.first().copied().unwrap_or(-1), mem),
            ref o => panic!("jit ended abnormally: {o:?}"),
        }
    };
    for cap in [1i64, 2, -1] {
        let (r, mem) = run(cap);
        assert_eq!(r, 85, "both carve children joined (cap {cap})");
        // The copy-back: each child's marker landed in *its own* carve, at carve offset 2048.
        assert_eq!(mem[65536 + 2048], 171, "child A's carve (cap {cap})");
        assert_eq!(mem[69632 + 2048], 172, "child B's carve (cap {cap})");
    }
}

/// A futex ping-pong side: wait until the turn word (region byte 0) reads `mine`, hand the turn to
/// `other` and notify, `rounds` times. Returns `rounds`.
fn pingpong(mine: i32, other: i32, rounds: i64) -> String {
    format!(
        r#"memory 17
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vz = i64.const 0
  br 1(vz)
}}
block 1 (vi: i64) {{
  vn = i64.const {rounds}
  vlt = i64.lt_u vi vn
  br_if vlt 2(vi) 5(vi)
}}
block 2 (vi2: i64) {{
  vt = i64.const 65536
  vcur = i32.atomic.load vt
  vmine = i32.const {mine}
  veq = i32.eq vcur vmine
  br_if veq 4(vi2) 3(vi2)
}}
block 3 (vi3: i64) {{
  vt3 = i64.const 65536
  vother3 = i32.const {other}
  vinf = i64.const -1
  vs = i32.atomic.wait vt3 vother3 vinf
  br 2(vi3)
}}
block 4 (vi4: i64) {{
  vt4 = i64.const 65536
  vother4 = i32.const {other}
  i32.atomic.store vt4 vother4
  vone = i32.const 1
  vw = atomic.notify vt4 vone
  vstep = i64.const 1
  vi5 = i64.add vi4 vstep
  br 1(vi5)
}}
block 5 (vi6: i64) {{
  return vi6
  }}
}}
"#
    )
}

/// **Child-to-child pipelining.** Two detached children hand a turn word back and forth over a
/// shared pre-mapped region, 200 rounds each, under a parent lane cap of 1 — so the pair can only
/// make progress if each `notify` promptly re-offers the parked sibling *and* the parked one's lane
/// is free for it. Both complete their rounds: `1000·200 + 200`.
///
/// The elapsed-time bound is the pin on the wake route itself (`thread_notify` →
/// `Domain::wake_child_tasks`). Without it the pair still finishes — an idle worker's cadence
/// sweep is the correctness backstop — but every handoff then waits out a sweep. Measured on this
/// box: 0.16s with the route, 4.17s with it mutated out. The threshold sits between, with a wide
/// margin on both sides.
#[test]
fn two_child_tasks_ping_pong_through_the_futex_without_waiting_on_the_sweep() {
    let p = module(&parent(TAIL_SUM));
    let a = module(&pingpong(0, 1, 200));
    let b = module(&pingpong(1, 0, 200));
    let start = std::time::Instant::now();
    assert_eq!(
        run_jit(&p, &a, &b, 1),
        200_200,
        "both sides completed 200 rounds"
    );
    let elapsed = start.elapsed();
    assert!(
        elapsed < std::time::Duration::from_millis(1200),
        "400 handoffs took {elapsed:?} — the notify wake route is not reaching parked tasks"
    );
}
