//! #3078 phase 2 slice 1 — a fiber's futex wait counts out of its word's count as it leaves the wait
//! index, as a thread's does (#2958 slice 2's count protocol, `temen-interp/src/futex_count.rs`).
//! Before, a fiber's wait counted in and never out, so every word a fiber ever waited on kept
//! executor 2's notify off its no-wake fast path for the rest of the run.
//!
//! Each row parks a fiber on a word and lets it leave one way: a notify, its real deadline passed at
//! a `cont.resume` poll, or the park-time re-check (the word already changed). Then it notifies the
//! word `NOTIFIES` times with no waiter. On executor 2 each of those must be answered without the
//! run lock: [`bytecode::futex_no_wakes`] rises by at least `NOTIFIES`. The count is the process's
//! and kept only in a debug build, so this file holds one test, run in a debug build only, and the
//! answer is also compared with the tree-walk oracle's.

use temen_interp::{bytecode, run, Value};
use temen_text::parse_module;

/// How many notifies with no waiter each row makes once its fiber has left.
const NOTIFIES: u64 = 100;

/// A program whose fiber waits on 16384 for `expected` with a timeout of `timeout_ns`, after the root
/// stored `word` there. `wake` notifies it once it has parked. The root resumes the fiber until it
/// returns, then notifies 16384 [`NOTIFIES`] times; it returns the fiber's status * 1000 plus what
/// those notifies woke, which must be none.
fn program(word: i32, timeout_ns: i64, wake: bool) -> String {
    let notify = if wake {
        "  vone = i32.const 1\n  vwk = atomic.notify vw vone\n"
    } else {
        ""
    };
    format!(
        r#"memory 16
func () -> (i64) {{
block 0 () {{
  vw = i64.const 16384
  vword = i32.const {word}
  i32.atomic.store vw vword
  vf = ref.func 1
  vz = i64.const 0
  vc = cont.new vf vz
  vp, vpv = cont.resume vc vz
{notify}  br 1(vc)
  }}
block 1 (vc1: i64) {{
  vz1 = i64.const 0
  vs, vsv = cont.resume vc1 vz1
  vparked = i32.const 3
  vagain = i32.eq vs vparked
  br_if vagain 1(vc1) 2(vsv)
  }}
block 2 (vst: i64) {{
  vn = i64.const {NOTIFIES}
  vsum = i64.const 0
  br 3(vst, vn, vsum)
  }}
block 3 (vst3: i64, vn3: i64, vsum3: i64) {{
  vz3 = i64.const 0
  vmore = i64.ne vn3 vz3
  br_if vmore 4(vst3, vn3, vsum3) 5(vst3, vsum3)
  }}
block 4 (vst4: i64, vn4: i64, vsum4: i64) {{
  vw4 = i64.const 16384
  vone4 = i32.const 1
  vwoke = atomic.notify vw4 vone4
  vwoke64 = i64.extend_i32_u vwoke
  vsum5 = i64.add vsum4 vwoke64
  vdec = i64.const 1
  vn5 = i64.sub vn4 vdec
  br 3(vst4, vn5, vsum5)
  }}
block 5 (vst6: i64, vsum6: i64) {{
  vk = i64.const 1000
  vx = i64.mul vst6 vk
  vr = i64.add vx vsum6
  return vr
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  vw = i64.const 16384
  vexp = i32.const 0
  vto = i64.const {timeout_ns}
  vst = i32.atomic.wait vw vexp vto
  vr = i64.extend_i32_s vst
  return vr
  }}
}}
"#
    )
}

#[test]
#[cfg(debug_assertions)]
fn a_fiber_that_left_its_wait_leaves_its_word_on_the_fast_path() {
    // (how it leaves, the program, its `WAIT_*` status)
    let rows = [
        ("a notify", program(0, 10_000_000_000, true), 0),
        ("its real deadline, at a poll", program(0, 1, false), 2),
        (
            "the park-time re-check",
            program(1, 10_000_000_000, false),
            1,
        ),
    ];
    for (how, src, status) in rows {
        let m = parse_module(&src).expect("parse");
        temen_verify::verify_module(&m).expect("verify");
        let want = Ok(vec![Value::I64(status * 1000)]);
        let mut f = u64::MAX;
        assert_eq!(run(&m, 0, &[], &mut f), want, "{how}: the oracle");
        let before = bytecode::futex_no_wakes();
        let mut f = u64::MAX;
        let (got, _) =
            bytecode::compile_and_run_capture_over_parallel(&m, 0, &[], &mut f, &[], None)
                .expect("executor 2 ran it");
        let fast = bytecode::futex_no_wakes() - before;
        assert_eq!(got, want, "{how}: executor 2 against the oracle");
        assert!(
            fast >= NOTIFIES,
            "{how}: {fast} of {NOTIFIES} notifies with no waiter took the fast path"
        );
    }
}
