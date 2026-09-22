//! **`notify(key, n)` wakes at most `n`** — cross-engine (#1600 follow-on).
//!
//! The interpreter is the oracle and it is exact: `notify` pops up to `count` waiters off the
//! key's queue and leaves the rest parked (`temen-interp`'s `notify`). A guest that hands work to
//! one worker per notify reads that count as a promise.
//!
//! The JIT keeps two waiter representations behind one key. Fibers get a per-waiter status cell,
//! drained by count, so they match the oracle. OS-thread vCPUs share a *generation* counter: one
//! notify bumps it once and **every** OS waiter on the key sees the bump, so all of them return
//! `WAIT_WOKEN` while `futex_notify` reports only `min(waiters, count)`. Same program, same key,
//! two different answers depending on which engine ran it — an invariant-15 second path through
//! one behaviour, and a §5 lie: the count a guest is told is not the count it got.
//!
//! This test states the property in guest terms — *the number of waiters that report woken equals
//! the number `notify` said it woke* — so it is race-free: whatever the scheduler does to
//! registration timing, the two numbers must agree.

use core::ffi::c_void;
use temen_interp::{run_with_host, Host, Value};
use temen_jit::{compile_and_run_capture_reserved_with_host_ex, JitOutcome};

fn module(text: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(text).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// Run `m` on both engines and return `(interp, jit)`.
fn both(m: &temen_ir::Module) -> (i64, i64) {
    let mut host = Host::new();
    let mut fuel = u64::MAX;
    let interp = match run_with_host(m, 0, &[], &mut fuel, &mut host)
        .expect("interp run")
        .first()
    {
        Some(Value::I64(x)) => *x,
        other => panic!("unexpected interp result {other:?}"),
    };

    let mut host = Host::new();
    let (jo, _) = compile_and_run_capture_reserved_with_host_ex(
        m,
        0,
        &[],
        &[],
        temen_ir::DEFAULT_RESERVED_LOG2,
        temen_run::cap_thunk,
        &mut host as *mut Host as *mut c_void,
        Some(temen_run::module_resolver),
        None,
    )
    .expect("jit run");
    let jit = match jo {
        JitOutcome::Returned(ref v) => v.first().copied().unwrap_or(-1),
        ref o => panic!("jit ended abnormally: {o:?}"),
    };
    (interp, jit)
}

/// Three siblings park on one key with a finite timeout, each bumping PARKED on the way in and
/// WOKEN on the way out **only if it reports status 0** (a timeout reports 2). The root waits for
/// all three to be on their way in, settles, then notifies the key **once with a count of 1**,
/// retrying only while the notify reports zero waiters — so exactly one notify of count 1 is ever
/// delivered. It joins all three and returns `100·notified + woken`.
///
/// `101` is the only correct answer: one claimed, one woken. `103` is the generation bug — one
/// claimed, all three woken. Addresses are above the #1094 NULL guard: KEY 16392 (never stored to,
/// so `wait` never reports not-equal), PARKED 16400, WOKEN 16408, handles 16416+.
const NOTIFY_ONE_OF_THREE: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  br 1(vz)
}
block 1 (vi: i64) {
  vthree = i64.const 3
  vz1 = i64.const 0
  vlt = i64.lt_u vi vthree
  br_if vlt 2(vi) 3(vz1)
}
block 2 (vi2: i64) {
  vsp = i64.const 0
  vt = thread.spawn 1 vsp vi2
  vh = i64.const 16416
  v4 = i64.const 4
  vm = i64.mul vi2 v4
  va = i64.add vh vm
  i32.store va vt
  vone = i64.const 1
  vnx = i64.add vi2 vone
  br 1(vnx)
}
block 3 (vd: i64) {
  vpa = i64.const 16400
  vp = i64.atomic.load vpa
  vthree3 = i64.const 3
  vlt3 = i64.lt_u vp vthree3
  br_if vlt3 3(vd) 4(vd)
}
block 4 (vd2: i64) {
  vz4 = i64.const 0
  br 5(vz4)
}
block 5 (vk: i64) {
  vlim = i64.const 200000
  vlt5 = i64.lt_u vk vlim
  vone5 = i64.const 1
  vk1 = i64.add vk vone5
  vz5 = i64.const 0
  br_if vlt5 5(vk1) 6(vz5)
}
block 6 (vr: i64) {
  vka = i64.const 16392
  vc = i32.const 1
  vnn = atomic.notify vka vc
  vnn64 = i64.extend_i32_u vnn
  vz6 = i64.const 0
  vgt = i64.lt_u vz6 vnn64
  br_if vgt 8(vnn64) 7(vr)
}
block 7 (vr2: i64) {
  vrl = i64.const 100000
  vlt7 = i64.lt_u vr2 vrl
  vone7 = i64.const 1
  vr3 = i64.add vr2 vone7
  vz7 = i64.const 0
  br_if vlt7 6(vr3) 8(vz7)
}
block 8 (vnf: i64) {
  vz8 = i64.const 0
  br 9(vz8, vnf)
}
block 9 (vj: i64, vnf2: i64) {
  vthree9 = i64.const 3
  vlt9 = i64.lt_u vj vthree9
  br_if vlt9 10(vj, vnf2) 11(vnf2)
}
block 10 (vj2: i64, vnf3: i64) {
  vh10 = i64.const 16416
  v410 = i64.const 4
  vm10 = i64.mul vj2 v410
  va10 = i64.add vh10 vm10
  vt10 = i32.load va10
  vr10 = thread.join vt10
  vone10 = i64.const 1
  vj3 = i64.add vj2 vone10
  br 9(vj3, vnf3)
}
block 11 (vnf4: i64) {
  vwa = i64.const 16408
  vw = i64.atomic.load vwa
  vk11 = i64.const 100
  vmm = i64.mul vnf4 vk11
  vsum = i64.add vmm vw
  return vsum
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vpa = i64.const 16400
  vone = i64.const 1
  vold = i64.atomic.rmw.add vpa vone
  vka = i64.const 16392
  ve = i32.const 0
  vto = i64.const 2000000000
  vs = i32.atomic.wait vka ve vto
  vs64 = i64.extend_i32_u vs
  vz = i64.const 0
  vgt = i64.lt_u vz vs64
  br_if vgt 2() 1()
}
block 1 () {
  vwa = i64.const 16408
  vone1 = i64.const 1
  vx = i64.atomic.rmw.add vwa vone1
  br 2()
}
block 2 () {
  vz2 = i64.const 0
  return vz2
  }
}
"#;

/// **A notify of 1 wakes 1, on both engines.**
#[test]
fn a_notify_of_one_wakes_exactly_one_of_three_waiters_on_both_engines() {
    let m = module(NOTIFY_ONE_OF_THREE);
    let (interp, jit) = both(&m);
    assert_eq!(
        interp, 101,
        "the oracle: notify claimed 1 and exactly 1 waiter reported woken"
    );
    assert_eq!(
        jit, 101,
        "the JIT must agree — a notify's count is a promise, not an estimate"
    );
}

/// **One key, both waiter kinds — and which one the notify claims.** A guest fiber and a
/// `thread.spawn` vCPU park on the same address; the root notifies **once with a count of 1**,
/// polls the fiber to completion, then joins the sibling. Each waiter bumps a counter whose
/// address it was handed, so the result says *which kind* woke:
/// `100·notified + 10·fiber_woken + vcpu_woken`.
///
/// Registration order here is deterministic, not a race: `cont.resume` returns `FIBER_PARKED`
/// only after the fiber has registered, and the root spawns the sibling afterwards. So the fiber
/// is strictly the older waiter and the answer pins wake *order* across the two kinds, which a
/// three-of-a-kind test cannot.
///
/// It also pins the liveness half of the unified queue: a wake spent on the **fiber** is latched
/// in its cell and delivered at a poll, so the root's `cont.resume.block` loop terminates either
/// way — a claimed fiber returns `WAIT_WOKEN`, an unclaimed one fires its own deadline at a poll.
/// Func 1 is both the fiber body and the sibling's entry: same signature, same program, two
/// parks, told apart only by the counter address each is handed.
const NOTIFY_ONE_OF_A_FIBER_AND_A_VCPU: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vf = ref.func 1
  vz = i64.const 0
  vk = cont.new vf vz
  vfa = i64.const 16424
  vs0, vv0 = cont.resume vk vfa
  vva = i64.const 16432
  vt = thread.spawn 1 vz vva
  vha = i64.const 16416
  i32.store vha vt
  br 1(vk)
}
block 1 (vk1: i64) {
  vpa = i64.const 16400
  vp = i64.atomic.load vpa
  vtwo = i64.const 2
  vlt = i64.lt_u vp vtwo
  br_if vlt 1(vk1) 2(vk1)
}
block 2 (vk2: i64) {
  vz2 = i64.const 0
  br 3(vk2, vz2)
}
block 3 (vk3: i64, vi: i64) {
  vlim = i64.const 200000
  vlt3 = i64.lt_u vi vlim
  vone3 = i64.const 1
  vi1 = i64.add vi vone3
  vz3 = i64.const 0
  br_if vlt3 3(vk3, vi1) 4(vk3, vz3)
}
block 4 (vk4: i64, vr: i64) {
  vka = i64.const 16392
  vc = i32.const 1
  vn = atomic.notify vka vc
  vn64 = i64.extend_i32_u vn
  vz4 = i64.const 0
  vgt = i64.lt_u vz4 vn64
  br_if vgt 6(vk4, vn64) 5(vk4, vr)
}
block 5 (vk5: i64, vr2: i64) {
  vrl = i64.const 100000
  vlt5 = i64.lt_u vr2 vrl
  vone5 = i64.const 1
  vr3 = i64.add vr2 vone5
  vz5 = i64.const 0
  br_if vlt5 4(vk5, vr3) 6(vk5, vz5)
}
block 6 (vk6: i64, vcl: i64) {
  vz6 = i64.const 0
  vs, vv = cont.resume.block vk6 vz6
  vone6 = i32.const 1
  vdone = i32.eq vs vone6
  br_if vdone 7(vcl) 6(vk6, vcl)
}
block 7 (vcl2: i64) {
  vha7 = i64.const 16416
  vt7 = i32.load vha7
  vj = thread.join vt7
  vfa7 = i64.const 16424
  vfw = i64.atomic.load vfa7
  vva7 = i64.const 16432
  vvw = i64.atomic.load vva7
  vk100 = i64.const 100
  vkten = i64.const 10
  vm = i64.mul vcl2 vk100
  vmf = i64.mul vfw vkten
  vs1 = i64.add vm vmf
  vsum = i64.add vs1 vvw
  return vsum
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vpa = i64.const 16400
  vone = i64.const 1
  vold = i64.atomic.rmw.add vpa vone
  vka = i64.const 16392
  ve = i32.const 0
  vto = i64.const 2000000000
  vs = i32.atomic.wait vka ve vto
  vs64 = i64.extend_i32_u vs
  vz = i64.const 0
  vgt = i64.lt_u vz vs64
  br_if vgt 2() 1(varg)
}
block 1 (vw: i64) {
  vone1 = i64.const 1
  vx = i64.atomic.rmw.add vw vone1
  br 2()
}
block 2 () {
  vz2 = i64.const 0
  return vz2
  }
}
"#;

/// **A notify of 1 wakes 1, and both engines agree on which one.** `101` = one claimed, the
/// *newest* waiter (the vCPU) woken, the fiber timed out. The JIT answered `110` — the oldest,
/// the fiber — until `futex_notify` was made to drain the tail: the oracle's queue is a `Vec` it
/// `pop()`s, so it wakes last-in-first-out, and this shape is the race-free way to see that.
#[test]
fn a_notify_of_one_wakes_the_newest_of_a_fiber_and_a_vcpu_on_both_engines() {
    let m = module(NOTIFY_ONE_OF_A_FIBER_AND_A_VCPU);
    let (interp, jit) = both(&m);
    assert_eq!(
        interp, 101,
        "the oracle: one claimed, and it is the newer waiter — the vCPU"
    );
    assert_eq!(
        jit, 101,
        "the JIT agrees on the count *and* on which waiter the count bought"
    );
}

/// The two ends of the count contract that are not "wake exactly one". Two siblings park on one
/// key; the root delivers `notify(key, 0)` — which must claim **nobody**, whatever is parked —
/// then `notify(key, 10)`, a budget larger than the queue, which must claim **everything** and no
/// more. Returns `10000·n0 + 100·n_all + woken`.
///
/// The `notify(key, 0)` result needs no settling to be meaningful: a zero budget wakes nobody
/// whether or not the waiters have registered yet, so that digit is race-free on its own.
const NOTIFY_ZERO_THEN_ALL: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  br 1(vz)
}
block 1 (vi: i64) {
  vtwo = i64.const 2
  vz1 = i64.const 0
  vlt = i64.lt_u vi vtwo
  br_if vlt 2(vi) 3(vz1)
}
block 2 (vi2: i64) {
  vsp = i64.const 0
  vt = thread.spawn 1 vsp vi2
  vh = i64.const 16416
  v4 = i64.const 4
  vm = i64.mul vi2 v4
  va = i64.add vh vm
  i32.store va vt
  vone = i64.const 1
  vnx = i64.add vi2 vone
  br 1(vnx)
}
block 3 (vd: i64) {
  vpa = i64.const 16400
  vp = i64.atomic.load vpa
  vtwo3 = i64.const 2
  vlt3 = i64.lt_u vp vtwo3
  br_if vlt3 3(vd) 4(vd)
}
block 4 (vd2: i64) {
  vz4 = i64.const 0
  br 5(vz4)
}
block 5 (vk: i64) {
  vlim = i64.const 200000
  vlt5 = i64.lt_u vk vlim
  vone5 = i64.const 1
  vk1 = i64.add vk vone5
  vz5 = i64.const 0
  br_if vlt5 5(vk1) 6(vz5)
}
block 6 (vunused: i64) {
  vka = i64.const 16392
  vzc = i32.const 0
  vn0 = atomic.notify vka vzc
  vn064 = i64.extend_i32_u vn0
  vz6 = i64.const 0
  br 7(vn064, vz6)
}
block 7 (vn0a: i64, vr: i64) {
  vka7 = i64.const 16392
  vcten = i32.const 10
  vna = atomic.notify vka7 vcten
  vna64 = i64.extend_i32_u vna
  vz7 = i64.const 0
  vgt = i64.lt_u vz7 vna64
  br_if vgt 9(vn0a, vna64) 8(vn0a, vr)
}
block 8 (vn0b: i64, vr2: i64) {
  vrl = i64.const 100000
  vlt8 = i64.lt_u vr2 vrl
  vone8 = i64.const 1
  vr3 = i64.add vr2 vone8
  vz8 = i64.const 0
  br_if vlt8 7(vn0b, vr3) 9(vn0b, vz8)
}
block 9 (vn0c: i64, vnac: i64) {
  vz9 = i64.const 0
  br 10(vz9, vn0c, vnac)
}
block 10 (vj: i64, vn0d: i64, vnad: i64) {
  vtwo10 = i64.const 2
  vlt10 = i64.lt_u vj vtwo10
  br_if vlt10 11(vj, vn0d, vnad) 12(vn0d, vnad)
}
block 11 (vj2: i64, vn0e: i64, vnae: i64) {
  vh11 = i64.const 16416
  v411 = i64.const 4
  vm11 = i64.mul vj2 v411
  va11 = i64.add vh11 vm11
  vt11 = i32.load va11
  vr11 = thread.join vt11
  vone11 = i64.const 1
  vj3 = i64.add vj2 vone11
  br 10(vj3, vn0e, vnae)
}
block 12 (vn0f: i64, vnaf: i64) {
  vwa = i64.const 16408
  vw = i64.atomic.load vwa
  vkw = i64.const 10000
  vk100 = i64.const 100
  vm0 = i64.mul vn0f vkw
  vma = i64.mul vnaf vk100
  vsum0 = i64.add vm0 vma
  vsum = i64.add vsum0 vw
  return vsum
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vpa = i64.const 16400
  vone = i64.const 1
  vold = i64.atomic.rmw.add vpa vone
  vka = i64.const 16392
  ve = i32.const 0
  vto = i64.const 2000000000
  vs = i32.atomic.wait vka ve vto
  vs64 = i64.extend_i32_u vs
  vz = i64.const 0
  vgt = i64.lt_u vz vs64
  br_if vgt 2() 1()
}
block 1 () {
  vwa = i64.const 16408
  vone1 = i64.const 1
  vx = i64.atomic.rmw.add vwa vone1
  br 2()
}
block 2 () {
  vz2 = i64.const 0
  return vz2
  }
}
"#;

/// **A budget of zero claims nobody; a budget past the end claims the queue, not more.**
#[test]
fn a_notify_of_zero_wakes_none_and_a_notify_past_the_end_wakes_all_on_both_engines() {
    let m = module(NOTIFY_ZERO_THEN_ALL);
    let (interp, jit) = both(&m);
    assert_eq!(
        interp, 202,
        "the oracle: 0 claimed by the zero budget, then both claimed and both woken"
    );
    assert_eq!(jit, 202, "the JIT agrees at both ends of the contract");
}
