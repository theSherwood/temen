//! #1627 slice A — a `gc.roots` inside a **cross-tier bounce** on the cooperative driver.
//!
//! An emitted tier-up region is paused *between* two interpreted computations: the task that tiered
//! up (its frames are in the scheduler, paused at the `Call`), and the bounce it makes back into the
//! interpreter. Its own frames are wasm — invisible to any scan — so the region stores every integer
//! it holds live across a host-reaching call onto a **spill stack** first, and hands that stack to
//! [`CoopRun::bounce`](temen_interp::bytecode::CoopRun::bounce). A collection inside the bounce must
//! then see all three: the collector's own frame, the paused task's, and the spilled words.
//!
//! This stands in for the emitted region **without wasm** (like `coop_tierup.rs`): the host services
//! the tier-up event by bouncing straight into the collector with the words the region would have
//! spilled. Without a spill stack the bounce cannot see everything live below it, so the op must fail
//! closed rather than under-report (GC.md §3.2 licenses over-approximation only).

use temen_interp::bytecode::TierUpConfig;
use temen_interp::{bytecode, Host, Trap, Value};
use temen_text::parse_module;

const FUEL: u64 = 10_000_000;

/// func 0 holds its argument across the call to func 1 (the eligible, "emitted" region) and adds it
/// to the result. func 2 is the collector: scan `[4096, 8192)` and return the total — its own `vlo`
/// (4096) is the only in-range word in its frame.
const SRC: &str = r#"
memory 16
func (i64) -> (i64) {
block 0 (vroot: i64) {
  vres = call 1 (vroot)
  vs = i64.add vres vroot
  return vs
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  vres = call 2 ()
  return vres
  }
}
func () -> (i64) {
block 0 () {
  vlo = i64.const 4096
  vhi = i64.const 8192
  vmask = i64.const -1
  vbuf = i64.const 16384
  vcap = i64.const 64
  vt = gc.roots vlo vhi vmask vbuf vcap
  return vt
  }
}
"#;

/// Run the guest with `root`, servicing func 1's tier-up by bouncing into the collector with `spill`.
fn run_with_spill(root: i64, spill: Option<&[u64]>) -> Result<Vec<Value>, Trap> {
    let m = parse_module(SRC).unwrap();
    temen_verify::verify_module(&m).expect("verify");
    let tierup = TierUpConfig {
        eligible: std::sync::Arc::from(vec![false, true, false]),
        page_checked: false,
    };
    let mut run = bytecode::CoopRun::new(&m, 0, &[Value::I64(root)], FUEL, Host::new(), Some(tierup))
        .expect("supported")
        .expect("entry in range");
    loop {
        match run.run() {
            bytecode::CoopEvent::Done(vals) => return Ok(vals),
            bytecode::CoopEvent::Trapped(t) => return Err(t),
            bytecode::CoopEvent::TierUp { func, .. } => {
                assert_eq!(func, 1, "only func 1 is eligible");
                let mut io = vec![0i64; 8];
                match run.bounce(2, &mut io, spill) {
                    Ok(n) => run.deliver_tierup(&io[..n]),
                    Err(t) => return Err(t),
                }
            }
            _ => panic!("unexpected event (no fibers, threads or JIT units in this guest)"),
        }
    }
}

#[test]
fn bounce_scans_the_paused_task_and_the_spilled_words() {
    // Roots: the collector's 4096, the paused func 0's 5000, the spilled 6000 → total 3; + 5000.
    assert_eq!(run_with_spill(5000, Some(&[6000])), Ok(vec![Value::I64(5003)]));
}

#[test]
fn an_empty_spill_stack_still_scans_the_paused_task() {
    // Nothing spilled: the collector's 4096 and the paused func 0's 5000 → total 2; + 5000.
    assert_eq!(run_with_spill(5000, Some(&[])), Ok(vec![Value::I64(5002)]));
}

#[test]
fn spilled_words_are_range_filtered_like_any_candidate() {
    // Out-of-range spilled words are not roots: still {4096, 5000} → total 2; + 5000.
    assert_eq!(run_with_spill(5000, Some(&[1, 9000, u64::MAX])), Ok(vec![Value::I64(5002)]));
}

#[test]
fn without_a_spill_stack_the_bounce_fails_closed() {
    assert_eq!(run_with_spill(5000, None), Err(Trap::CapFault));
}
