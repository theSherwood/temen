//! A [`CoopRun`] pumped in **slices** (`run_for`): it returns `Paused` after about the requested ops,
//! live and resumable, and resuming continues exactly where it stopped — so a slice-pumped run ends
//! with the same result as the one-shot run. This is what lets a browser Worker run a long program
//! at full engine speed while still streaming its output and honoring a Pause between slices.

use temen_interp::bytecode::{self, CoopEvent, CoopRun};
use temen_interp::{Host, Value};
use temen_text::parse_module;

/// An endless loop.
const SPIN: &str = r#"
func () -> (i32) {
block 0 () {
  v0 = i32.const 0
  br 1(v0)
  }
block 1 (v0: i32) {
  v1 = i32.const 1
  v2 = i32.add v0 v1
  br 1(v2)
  }
}
"#;

/// Sum 0..100_000, about half a million ops.
const SUM: &str = r#"
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  v1 = i64.const 0
  br 1(v0, v1)
  }
block 1 (v0: i64, v1: i64) {
  v2 = i64.const 100000
  v3 = i64.lt_s v0 v2
  br_if v3 2(v0, v1) 3(v1)
  }
block 2 (v0: i64, v1: i64) {
  v2 = i64.add v1 v0
  v3 = i64.const 1
  v4 = i64.add v0 v3
  br 1(v4, v2)
  }
block 3 (v0: i64) {
  return v0
  }
}
"#;

fn run(src: &str) -> CoopRun {
    let m = parse_module(src).expect("parse");
    CoopRun::new_seeded(&m, 0, &[], u64::MAX, Host::new(), &[])
        .expect("in the bytecode subset")
        .expect("builds")
}

#[test]
fn an_endless_loop_pauses_every_slice_and_stays_live() {
    let mut r = run(SPIN);
    for i in 0..5 {
        assert!(
            matches!(r.run_for(10_000), CoopEvent::Paused),
            "slice {i} pauses"
        );
    }
}

#[test]
fn a_slice_pumped_run_ends_as_the_one_shot_run_does() {
    let m = parse_module(SUM).expect("parse");
    let mut fuel = u64::MAX;
    let one_shot =
        bytecode::compile_and_run_seeded_with_host(&m, 0, &[], &mut fuel, &[], &mut Host::new())
            .expect("in subset")
            .expect("runs");
    assert_eq!(one_shot, vec![Value::I64(4_999_950_000)]);

    for slice in [1, 7, 1000, 1 << 20] {
        let mut r = run(SUM);
        let mut pauses = 0u64;
        let vals = loop {
            match r.run_for(slice) {
                CoopEvent::Paused => pauses += 1,
                CoopEvent::Done(v) => break v,
                _ => panic!("slice {slice}: the run neither paused nor finished"),
            }
        };
        assert_eq!(vals, one_shot, "slice {slice}");
        if slice < 1000 {
            assert!(pauses > 1000, "slice {slice} paused often ({pauses})");
        }
    }
}
