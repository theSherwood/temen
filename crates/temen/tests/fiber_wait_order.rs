//! #3078 phase 2 slice 1, #3086 — the order a `memory.notify` wakes its waiters in, task and fiber
//! alike, against the tree-walk oracle (INVARIANTS #9). The oracle keeps one queue per word in
//! **arrival order** (`notify_locked`; INVARIANTS #4's FIFO queues), so a notify of `n` wakes the `n`
//! waiters that waited first, whether each is a thread or a fiber. Each row is a program whose
//! arrival order its own program order forces: fibers parked one after another by one task, a task
//! that waits behind its own fiber, a fiber that leaves and waits again.
//!
//! Every row runs on the tree-walker, the bytecode engine's cooperative pump
//! ([`bytecode::compile_and_run`]) and executor 2 ([`bytecode::compile_and_run_capture_over_parallel`]),
//! asserts that each bytecode entry ran the program itself (it answers `None` for a module outside
//! its subset, and nothing falls back here), and compares the answers. No sleeps and no fuel as a
//! clock: every wait is timed at 10 s, which no row reaches, and every loop ends on a flag a waiter
//! sets once it is woken.

#[path = "support/kernel_lane.rs"]
mod kernel_lane;

use temen_interp::{bytecode, run, Trap, Value};
use temen_text::parse_module;

// The programs' words: every waiter waits on `W` (16384); a woken thread stores its status at
// `STATUS` (16392) and sets `FLAG` (16396).

/// A bound, not a clock: no row comes near it.
const FUEL: u64 = 10_000_000_000;

/// `src`'s function 0 on the three engines, each asserted to have run it: (tree-walker, pump,
/// executor 2).
fn engines(src: &str) -> [Result<Vec<Value>, Trap>; 3] {
    let m = parse_module(src).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}\n{src}"));
    let mut f = FUEL;
    let tw = run(&m, 0, &[], &mut f);
    let mut f = FUEL;
    let pump = bytecode::compile_and_run(&m, 0, &[], &mut f).expect("the pump ran it");
    let mut f = FUEL;
    let (par, _) = bytecode::compile_and_run_capture_over_parallel(&m, 0, &[], &mut f, &[], None)
        .expect("executor 2 ran it");
    [tw, pump, par]
}

/// Assert every engine answers `want`, the oracle's answer.
fn pin(src: &str, want: i64) {
    let [tw, pump, par] = engines(src);
    let i64_want = want;
    let want = Ok(vec![Value::I64(want)]);
    assert_eq!(tw, want, "the oracle\n{src}");
    assert_eq!(pump, want, "the pump against the oracle\n{src}");
    assert_eq!(par, want, "executor 2 against the oracle\n{src}");
    // #3078 slice 3 — a native fiber's waiter is an entry of the same index, in the same order.
    for fuel in kernel_lane::FUELS {
        assert_eq!(
            kernel_lane::i64_on_kernel(src, fuel),
            i64_want,
            "the kernel route against the oracle, fuel {fuel}\n{src}"
        );
    }
}

/// A fiber that waits on `W` once (`waits = 1`) or twice, and returns its last status.
fn fiber_waiter(waits: usize) -> String {
    let mut b = String::from(
        "func (i64, i64) -> (i64) {\nblock 0 (vsp: i64, varg: i64) {\n  vw = i64.const 16384\n  vz = i32.const 0\n  vto = i64.const 10000000000\n",
    );
    for i in 0..waits {
        b += &format!("  vst{i} = i32.atomic.wait vw vz vto\n");
    }
    b += &format!(
        "  vr = i64.extend_i32_s vst{}\n  return vr\n  }}\n}}\n",
        waits - 1
    );
    b
}

/// A thread that waits on `W`, then stores its status at `STATUS` and sets `FLAG`.
const THREAD_WAITER: &str = r#"func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vw = i64.const 16384
  vz = i32.const 0
  vto = i64.const 10000000000
  vst = i32.atomic.wait vw vz vto
  vsa = i64.const 16392
  i32.atomic.store vsa vst
  vfa = i64.const 16396
  vone = i32.const 1
  i32.atomic.store vfa vone
  vr = i64.const 0
  return vr
  }
}
"#;

/// A thread that waits for `READY` (16400) to be set, notifies `W` once, then notifies it one waiter
/// at a time until `FLAG` is set, and returns what its first notify woke.
const THREAD_NOTIFIER: &str = r#"func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  br 1()
  }
block 1 () {
  vra = i64.const 16400
  vrd = i32.atomic.load vra
  vz0 = i32.const 0
  vready = i32.ne vrd vz0
  br_if vready 2() 1()
  }
block 2 () {
  vw = i64.const 16384
  vone = i32.const 1
  vc = atomic.notify vw vone
  br 3(vc)
  }
block 3 (vc3: i32) {
  vfa = i64.const 16396
  vfl = i32.atomic.load vfa
  vz = i32.const 0
  vdone = i32.ne vfl vz
  br_if vdone 5(vc3) 4(vc3)
  }
block 4 (vc4: i32) {
  vw4 = i64.const 16384
  vone4 = i32.const 1
  vn = atomic.notify vw4 vone4
  br 3(vc4)
  }
block 5 (vc5: i32) {
  vr = i64.extend_i32_s vc5
  return vr
  }
}
"#;

/// Fold `cont.resume` statuses `s{i}` into a bitmask of the fibers that returned (status 1, woken),
/// `vmask{n}` its last value.
fn woken_mask(out: &mut String, statuses: &[String]) {
    *out += "  vmask0 = i64.const 0\n  vret = i32.const 1\n";
    for (i, s) in statuses.iter().enumerate() {
        *out += &format!(
            "  ve{i} = i32.eq {s} vret\n  ve64_{i} = i64.extend_i32_u ve{i}\n  vsh{i} = i64.const {i}\n  vbit{i} = i64.shl ve64_{i} vsh{i}\n  vmask{} = i64.or vmask{i} vbit{i}\n",
            i + 1
        );
    }
}

/// K fibers park on `W` in the order `order` (slot indices), then, with `thread`, a thread waits
/// on `W` behind them. The root notifies `n` and polls every fiber: the answer is
/// `count * 1000 + mask * 10 + thread_status`, where `mask` has bit `i` set if slot `i` was woken
/// and `thread_status` is the thread's `WAIT_*` status once the root has woken everyone (0, woken).
fn k_fibers(order: &[usize], n: u32, thread: bool) -> String {
    let k = order.len();
    let mut b = String::from(
        "memory 16\nfunc () -> (i64) {\nblock 0 () {\n  vf = ref.func 1\n  vz = i64.const 0\n",
    );
    for i in 0..k {
        b += &format!("  vc{i} = cont.new vf vz\n");
    }
    for &i in order {
        b += &format!("  vp{i}, vpv{i} = cont.resume vc{i} vz\n");
    }
    if thread {
        b += "  v500 = i64.const 500\n  vt = thread.spawn 2 v500 v500\n";
    } else {
        b += "  vt = i32.const 0\n";
    }
    b += &format!("  vw = i64.const 16384\n  vn = i32.const {n}\n  vcnt = atomic.notify vw vn\n");
    let statuses: Vec<String> = (0..k).map(|i| format!("vs{i}")).collect();
    for i in 0..k {
        b += &format!("  vs{i}, vsv{i} = cont.resume vc{i} vz\n");
    }
    woken_mask(&mut b, &statuses);
    b += &format!("  vall = i32.const {k}\n  vrest = atomic.notify vw vall\n");
    b += &format!("  br 1(vcnt, vmask{k}, vt)\n  }}\n");
    if thread {
        // Wake the thread, however far it got: it may not have parked yet.
        b += r#"block 1 (vcnt1: i32, vm1: i64, vt1: i32) {
  vfa = i64.const 16396
  vfl = i32.atomic.load vfa
  vz1 = i32.const 0
  vdone = i32.ne vfl vz1
  br_if vdone 3(vcnt1, vm1, vt1) 2(vcnt1, vm1, vt1)
  }
block 2 (vcnt2: i32, vm2: i64, vt2: i32) {
  vw2 = i64.const 16384
  vone = i32.const 1
  vnn = atomic.notify vw2 vone
  br 1(vcnt2, vm2, vt2)
  }
block 3 (vcnt3: i32, vm3: i64, vt3: i32) {
  vjs, vjv = thread.join vt3
  vsa = i64.const 16392
  vtst = i32.atomic.load vsa
  vtst64 = i64.extend_i32_s vtst
  br 4(vcnt3, vm3, vtst64)
  }
"#;
    } else {
        // No thread: blocks 1 to 3 only carry the answer on (labels run in order).
        b += r#"block 1 (vcnt1: i32, vm1: i64, vt1: i32) {
  br 2(vcnt1, vm1, vt1)
  }
block 2 (vcnt2: i32, vm2: i64, vt2: i32) {
  br 3(vcnt2, vm2, vt2)
  }
block 3 (vcnt3: i32, vm3: i64, vt3: i32) {
  vno = i64.const 0
  br 4(vcnt3, vm3, vno)
  }
"#;
    }
    b += r#"block 4 (vcnt4: i32, vm4: i64, vts: i64) {
  vcnt64 = i64.extend_i32_u vcnt4
  vk1000 = i64.const 1000
  va = i64.mul vcnt64 vk1000
  vk10 = i64.const 10
  vb = i64.mul vm4 vk10
  vab = i64.add va vb
  vr = i64.add vab vts
  return vr
  }
}
"#;
    b += &fiber_waiter(1);
    if thread {
        b += THREAD_WAITER;
    }
    b
}

/// The oracle's answer for [`k_fibers`]: the first `n` to arrive are woken.
fn first_n(order: &[usize], n: u32) -> i64 {
    let mask: i64 = order.iter().take(n as usize).map(|&i| 1 << i).sum();
    let count = (n as usize).min(order.len()) as i64;
    count * 1000 + mask * 10
}

/// #3086 — fibers of one task, parked in an order that is not their slot order: a notify of `n`
/// wakes the first `n` to park, not the lowest slots.
#[test]
fn a_notify_wakes_fibers_in_the_order_they_parked() {
    for n in [1, 2] {
        let order = [2, 0, 1];
        pin(&k_fibers(&order, n, false), first_n(&order, n));
    }
}

/// K fibers parked on one word and a thread waiting behind them: a notify of `n` (0 to K) wakes the
/// first `n` fibers to park and answers `n`; the thread, last to arrive, is woken once everyone ahead
/// of it is.
#[test]
fn k_fibers_and_a_thread_on_one_word_wake_in_arrival_order() {
    let order = [2, 0, 3, 1];
    for n in [0, 1, 3, 4] {
        pin(&k_fibers(&order, n, true), first_n(&order, n));
    }
}

/// #3086 — a task whose fiber parked on a word, then waits on the word itself, is behind its fiber:
/// another thread's notify of 1 wakes the fiber. The thread notifies once the task has said it is
/// about to wait (`READY`), so on an engine that ranked the task first the task would be parked by
/// then; under arrival order the fiber is first either way. The answer is `fiber_status * 10 + first_woke`:
/// the fiber returned (1), and the notifier's first notify woke one waiter (1). Were the task woken
/// first, its poll of the fiber would read it still parked (3).
#[test]
fn a_task_waiting_behind_its_own_fiber_is_woken_after_it() {
    let src = format!(
        r#"memory 16
func () -> (i64) {{
block 0 () {{
  vf = ref.func 1
  vz = i64.const 0
  vc = cont.new vf vz
  vp, vpv = cont.resume vc vz
  v500 = i64.const 500
  vt = thread.spawn 2 v500 v500
  vw = i64.const 16384
  vz32 = i32.const 0
  vto = i64.const 10000000000
  vra = i64.const 16400
  vone0 = i32.const 1
  i32.atomic.store vra vone0
  vst = i32.atomic.wait vw vz32 vto
  vfa = i64.const 16396
  vone = i32.const 1
  i32.atomic.store vfa vone
  vs, vsv = cont.resume vc vz
  vjs, vjv = thread.join vt
  vs64 = i64.extend_i32_s vs
  vk10 = i64.const 10
  va = i64.mul vs64 vk10
  vr = i64.add va vjv
  return vr
  }}
}}
{}{THREAD_NOTIFIER}"#,
        fiber_waiter(1)
    );
    pin(&src, 11);
}

/// A waiter that leaves and waits again arrives again, at the back (#3078 slice 1, condition 3).
/// Fiber A (slot 0) parks, then B (slot 1); a notify of 1 wakes A, whose resume runs it into its
/// second wait, behind B; the next notify of 1 wakes B. The answer is `status_b * 10 + status_a`:
/// B returned (1), A is parked again (3).
#[test]
fn a_waiter_that_waits_again_arrives_again() {
    let a = fiber_waiter(2);
    let b = fiber_waiter(1);
    let src = format!(
        r#"memory 16
func () -> (i64) {{
block 0 () {{
  vfa = ref.func 1
  vfb = ref.func 2
  vz = i64.const 0
  vca = cont.new vfa vz
  vcb = cont.new vfb vz
  vpa, vpav = cont.resume vca vz
  vpb, vpbv = cont.resume vcb vz
  vw = i64.const 16384
  vone = i32.const 1
  vn1 = atomic.notify vw vone
  vra, vrav = cont.resume vca vz
  vn2 = atomic.notify vw vone
  vsb, vsbv = cont.resume vcb vz
  vsa, vsav = cont.resume vca vz
  vsb64 = i64.extend_i32_s vsb
  vsa64 = i64.extend_i32_s vsa
  vk10 = i64.const 10
  vx = i64.mul vsb64 vk10
  vr = i64.add vx vsa64
  return vr
  }}
}}
{a}{b}"#
    );
    pin(&src, 13);
}

/// A fiber's timed wait is in the wait index's deadlines with the threads' (#3078 slice 1), so a run
/// where nothing else can move still times it out, though no thread waits: the root idles on its
/// fiber (`cont.resume.block`, no loop), and the pump's clock, finding nothing runnable, jumps to
/// the fiber's deadline. A poll loop (`cont.resume`) never idles; its poll times the wait out once its
/// real deadline has passed. Either way the fiber returns `WAIT_TIMED_OUT` (2): the answer is
/// `resume_status * 100 + 2` = 102. The pump's "a thread naps" checks (its quantum, its idle point)
/// ask only after threads' deadlines; this row is what shows that leaves a fiber's timed out.
#[test]
fn a_timed_fiber_wait_times_out_when_no_thread_waits() {
    let fiber = r#"func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vaddr = i64.const 16384
  vexp = i32.const 0
  vto = i64.const 10000000
  vst = i32.atomic.wait vaddr vexp vto
  vst64 = i64.extend_i32_s vst
  return vst64
  }
}
"#;
    let idles = format!(
        r#"memory 16
func () -> (i64) {{
block 0 () {{
  vf = ref.func 1
  vz = i64.const 0
  vc = cont.new vf vz
  vs, vv = cont.resume.block vc vz
  vk = i64.const 100
  vse = i64.extend_i32_s vs
  va = i64.mul vse vk
  vr = i64.add va vv
  return vr
  }}
}}
{fiber}"#
    );
    let polls = format!(
        r#"memory 16
func () -> (i64) {{
block 0 () {{
  vf = ref.func 1
  vz = i64.const 0
  vc = cont.new vf vz
  br 1(vc)
  }}
block 1 (vc1: i64) {{
  vz1 = i64.const 0
  vs, vv = cont.resume vc1 vz1
  vparked = i32.const 3
  vagain = i32.eq vs vparked
  br_if vagain 1(vc1) 2(vs, vv)
  }}
block 2 (vs2: i32, vv2: i64) {{
  vk = i64.const 100
  vse = i64.extend_i32_s vs2
  va = i64.mul vse vk
  vr = i64.add va vv2
  return vr
  }}
}}
{fiber}"#
    );
    pin(&idles, 102);
    pin(&polls, 102);
}

/// #3086 — a debugger restore files each futex wait back at its place in arrival order (its record
/// keeps it), not in task order. Thread 2 waits on `W` at once; thread 1 first naps 1 ms (the debug
/// scheduler's logical clock), then waits on `W` too, so the waits arrive in the reverse of task
/// order. The root naps 5 ms, notifies 1, and waits for the woken thread to write its index at
/// `STATUS` (16392) and notify it; then it wakes the other and joins both. The answer is the
/// index of the thread woken first: 2. A checkpoint taken at every turn, restored into a fresh
/// session and run on, must answer the same.
#[test]
fn a_debugger_restore_keeps_the_waits_in_arrival_order() {
    use temen_interp::bytecode::{BlockedOn, ScheduledDebugRun};
    let waiter = |id: u32, nap: bool| {
        let nap = if nap {
            "  vx = i64.const 16400\n  vxz = i32.const 0\n  vxt = i64.const 1000000\n  vxs = i32.atomic.wait vx vxz vxt\n"
        } else {
            ""
        };
        format!(
            r#"func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
{nap}  vw = i64.const 16384
  vz = i32.const 0
  vto = i64.const -1
  vst = i32.atomic.wait vw vz vto
  vf = i64.const 16392
  vid = i32.const {id}
  i32.atomic.store vf vid
  vone = i32.const 1
  vn = atomic.notify vf vone
  vr = i64.const 0
  return vr
  }}
}}
"#
        )
    };
    let src = format!(
        r#"memory 16
func () -> (i64) {{
block 0 () {{
  vz = i64.const 0
  vt1 = thread.spawn 1 vz vz
  vt2 = thread.spawn 2 vz vz
  vx = i64.const 16400
  vxz = i32.const 0
  vxt = i64.const 5000000
  vxs = i32.atomic.wait vx vxz vxt
  vw = i64.const 16384
  vone = i32.const 1
  vn1 = atomic.notify vw vone
  vf = i64.const 16392
  vinf = i64.const -1
  vfs = i32.atomic.wait vf vxz vinf
  vfirst = i32.atomic.load vf
  vn2 = atomic.notify vw vone
  vj1s, vj1 = thread.join vt1
  vj2s, vj2 = thread.join vt2
  vr = i64.extend_i32_u vfirst
  return vr
  }}
}}
{}{}"#,
        waiter(1, true),
        waiter(2, false)
    );
    let m = parse_module(&src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let session = || ScheduledDebugRun::new(&m, 0, &[]).expect("the debugger runs the module");
    let want = Some(Ok(vec![Value::I64(2)]));
    let mut reference = session();
    let mut fuel = FUEL;
    while reference.tick(&mut fuel) {}
    assert_eq!(
        reference.result().cloned(),
        want,
        "the thread that waited first"
    );
    let total = reference.op_turn();
    let mut both_parked = 0;
    for c in 0..=total {
        let mut at_c = session();
        let mut fuel = FUEL;
        while at_c.op_turn() < c && at_c.tick(&mut fuel) {}
        let Ok(snap) = at_c.snapshot() else {
            continue;
        };
        let on_w = (at_c.blocked_on().into_iter())
            .filter(|(_, on)| matches!(on, BlockedOn::Futex(16384)))
            .count();
        if on_w == 2 {
            both_parked += 1;
        }
        let mut warm = session();
        warm.restore(at_c.op_turn(), &snap);
        let mut fuel = FUEL;
        while warm.tick(&mut fuel) {}
        assert_eq!(warm.result().cloned(), want, "restored at turn {c}");
    }
    assert!(
        both_parked > 0,
        "a checkpoint caught both threads parked on the word"
    );
}
