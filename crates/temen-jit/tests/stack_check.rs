//! Functional test for the software stack-overflow guard (feature `stack-check`, STACK_GUARD.md).
//!
//! With the guard on, a fiber that recurses without bound must trap `StackOverflow` (the prologue
//! check fires ~`RED_ZONE` above the fiber's low bound) instead of running off its control stack —
//! and a normal, shallow fiber must still run to completion (the check doesn't false-trigger). The
//! recursion runs on the fiber's own 256 KiB control stack, not the host stack, so this is bounded.
#![cfg(all(
    feature = "stack-check",
    any(
        all(unix, target_arch = "x86_64"),
        all(unix, target_arch = "aarch64"),
        all(windows, target_arch = "x86_64")
    )
))]

use temen_jit::{compile_and_run, JitOutcome, TrapKind};
use temen_text::parse_module;

// Root creates a fiber and resumes it to completion. The fiber entry (func 1) calls func 2, which
// recurses into itself forever via a non-tail `call` (frames accumulate on the fiber's control stack).
const RECURSE: &str = "\
func () -> (i64) {
block 0 () {
  v0 = ref.func 1
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v3 = i64.const 0
  v4, v5 = cont.resume v2 v3
  return v5
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = call 2 (v0)
  return v2
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = call 2 (v0)
  return v1
  }
}
";

// Root creates a fiber that immediately returns 7 — no deep stack use, must run fine under the guard.
const SHALLOW: &str = "\
func () -> (i64) {
block 0 () {
  v0 = ref.func 1
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v3 = i64.const 0
  v4, v5 = cont.resume v2 v3
  return v5
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 7
  return v2
  }
}
";

#[test]
fn unbounded_fiber_recursion_traps_stack_overflow() {
    let m = parse_module(RECURSE).expect("parse");
    match compile_and_run(&m, 0, &[]).expect("jit compile/run") {
        JitOutcome::Trapped(TrapKind::StackOverflow) => {}
        other => panic!("expected StackOverflow trap, got {other:?}"),
    }
}

// Multi-vCPU: the root spawns a vCPU (its own OS thread) whose thread-entry creates + resumes a
// recursing fiber. The limit is per-vCPU by construction (each fiber entry supplies its own via an
// ABI param, no shared cell), so the spawned vCPU's fiber must trap StackOverflow on *its* stack.
const SPAWNED_RECURSE: &str = "\
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  v1 = thread.spawn 1 v0 v0
  v2 = thread.join v1
  return v2
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = ref.func 2
  v3 = i64.const 4096
  v4 = cont.new v2 v3
  v5 = i64.const 0
  v6, v7 = cont.resume v4 v5
  return v7
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = call 3 (v0)
  return v2
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = call 3 (v0)
  return v1
  }
}
";

#[test]
fn spawned_vcpu_fiber_recursion_traps_stack_overflow() {
    let m = parse_module(SPAWNED_RECURSE).expect("parse");
    // A spawned vCPU's fiber overflow is detect-and-killed and surfaces as a trap on the run.
    match compile_and_run(&m, 0, &[]).expect("jit compile/run") {
        JitOutcome::Trapped(TrapKind::StackOverflow) => {}
        other => panic!("expected StackOverflow from the spawned vCPU's fiber, got {other:?}"),
    }
}

#[test]
fn shallow_fiber_runs_under_the_guard() {
    let m = parse_module(SHALLOW).expect("parse");
    match compile_and_run(&m, 0, &[]).expect("jit compile/run") {
        JitOutcome::Returned(slots) => assert_eq!(slots, vec![7]),
        other => panic!("expected Returned([7]), got {other:?}"),
    }
}

// #1983 — recursion on a stack the JIT did not allocate: the root (the calling thread's own stack)
// and a spawned vCPU's top (its OS thread's stack). Each takes its thread's limit, so unbounded
// recursion traps `StackOverflow` rather than running into the OS guard page, where the process
// aborts.
const ROOT_RECURSE: &str = "\
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  v1 = call 1 (v0)
  return v1
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = call 1 (v0)
  return v1
  }
}
";

const SPAWNED_TOP_RECURSE: &str = "\
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  v1 = thread.spawn 1 v0 v0
  v2 = thread.join v1
  return v2
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = call 2 (v0)
  return v2
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = call 2 (v0)
  return v1
  }
}
";

// Recursion to depth `n` on the root, returning `n`: well short of the limit, it must run.
const ROOT_DEPTH: &str = "\
func (i64) -> (i64) {
block 0 (vn: i64) {
  vz = i64.const 0
  vdone = i64.eq vn vz
  br_if vdone 1(vz) 2(vn)
}
block 1 (vr: i64) {
  return vr
}
block 2 (vm: i64) {
  vone = i64.const 1
  vm1 = i64.sub vm vone
  vs = call 0(vm1)
  vt = i64.add vs vone
  return vt
  }
}
";

#[test]
fn unbounded_root_recursion_traps_stack_overflow() {
    let m = parse_module(ROOT_RECURSE).expect("parse");
    match compile_and_run(&m, 0, &[]).expect("jit compile/run") {
        JitOutcome::Trapped(TrapKind::StackOverflow) => {}
        other => panic!("expected StackOverflow on the root, got {other:?}"),
    }
}

#[test]
fn unbounded_spawned_vcpu_top_recursion_traps_stack_overflow() {
    let m = parse_module(SPAWNED_TOP_RECURSE).expect("parse");
    match compile_and_run(&m, 0, &[]).expect("jit compile/run") {
        JitOutcome::Trapped(TrapKind::StackOverflow) => {}
        other => panic!("expected StackOverflow on the spawned vCPU's top, got {other:?}"),
    }
}

#[test]
fn bounded_root_recursion_runs_under_the_limit() {
    let m = parse_module(ROOT_DEPTH).expect("parse");
    match compile_and_run(&m, 0, &[10_000]).expect("jit compile/run") {
        JitOutcome::Returned(slots) => assert_eq!(slots, vec![10_000]),
        other => panic!("expected Returned([10000]), got {other:?}"),
    }
}
