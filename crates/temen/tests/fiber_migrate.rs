//! §12/D57 **migratable fibers on the interpreter** (step 3b-i, DESIGN.md §23 "Integration
//! design"): the per-vCPU fiber tables are replaced by one **run-shared registry**, so a fiber
//! created (and even part-run) on one vCPU can be claimed and continued by another — a safe
//! `Vec<Frame>` hand-off, exactly like the vCPUs the scheduler already migrates. The registry's
//! claim is the **single-owner arbiter**: of any racing `cont.resume`s, exactly one wins and a
//! loser gets a clean `FiberFault`.
//!
//! These pin the reference semantics — the oracle the JIT's lock-free registry (3b-ii) and
//! cross-thread asm resume (3c) will be differentially tested against. Each behavior runs on the
//! real M:N pool (`run`), the seeded explorer (`run_scheduled`), and the exhaustive checker
//! (`explore_all`), and the exhaustive outcome sets are cross-checked against the **unreduced**
//! brute-force enumerator — proving the DPOR `MemAccess::Fiber` conflict rule (fiber ops don't
//! commute) loses no interleaving.

use temen_interp::{explore_all, explore_all_bruteforce, run, run_scheduled, run_with_host};
use temen_interp::{Host, Quota, Trap, Value};
use temen_text::parse_module;
use temen_verify::verify_module;

fn module(src: &str) -> temen_ir::Module {
    let m = parse_module(src).unwrap_or_else(|e| panic!("parse failed: {e:?}\n{src}"));
    verify_module(&m).unwrap_or_else(|e| panic!("verify failed: {e:?}\n{src}"));
    m
}

/// Run func 0 on the real M:N pool and return the single i64 result (or the trap).
fn run_i64(src: &str) -> Result<i64, Trap> {
    let m = module(src);
    let mut fuel = 10_000_000u64;
    match run(&m, 0, &[], &mut fuel) {
        Ok(vals) => match vals.as_slice() {
            [Value::I64(v)] => Ok(*v),
            other => panic!("expected one i64 result, got {other:?}"),
        },
        Err(t) => Err(t),
    }
}

/// Exhaustively explore `src` and assert the outcome set — and that the DPOR checker and the
/// unreduced brute-force enumerator (the reduction-soundness oracle) agree on it exactly.
fn assert_outcomes(src: &str, want: &[Result<i64, Trap>]) {
    let m = module(src);
    let to_set = |ex: temen_interp::Exhaustive| -> Vec<Result<i64, Trap>> {
        assert!(ex.complete, "exploration must complete");
        let mut got: Vec<Result<i64, Trap>> = ex
            .outcomes
            .into_iter()
            .map(|r| {
                r.map(|vals| match vals.as_slice() {
                    [Value::I64(v)] => *v,
                    other => panic!("expected one i64 result, got {other:?}"),
                })
            })
            .collect();
        got.sort_by_key(|r| format!("{r:?}"));
        got
    };
    let mut want: Vec<Result<i64, Trap>> = want.to_vec();
    want.sort_by_key(|r| format!("{r:?}"));
    let dpor = to_set(explore_all(&m, 0, &[], 1_000_000, 200_000));
    let brute = to_set(explore_all_bruteforce(&m, 0, &[], 1_000_000, 200_000));
    assert_eq!(dpor, want, "DPOR outcome set\n{src}");
    assert_eq!(
        brute, want,
        "brute-force outcome set (DPOR reduction oracle)\n{src}"
    );
}

/// **The headline: a mid-life fiber migrates across vCPUs.** The root creates fiber F and resumes
/// it once — F suspends, capturing its first-resume argument (5) in its parked stack. The root
/// then hands F's *handle* to a spawned vCPU, which resumes it there: F continues past its
/// `suspend` **on the other vCPU** and returns `10*x + arg = 10*7 + 5 = 75` — the `+ 5` proving
/// the stack state captured on the root survived the migration intact (a restart would lose it).
const MIGRATE: &str = r#"
func () -> (i64) {
block 0 () {
  v0 = ref.func 2
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v3 = i64.const 5
  v4, v5 = cont.resume v2 v3
  v6 = thread.spawn 1 v2 v2
  v7 = thread.join v6
  return v7
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  v0 = i64.const 7
  v1, v2 = cont.resume varg v0
  return v2
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  v0 = suspend varg
  v1 = i64.const 10
  v2 = i64.mul v0 v1
  v3 = i64.add v2 varg
  return v3
  }
}
"#;

#[test]
fn fiber_suspended_on_root_resumes_on_spawned_vcpu() {
    assert_eq!(run_i64(MIGRATE), Ok(75), "real M:N pool");
    for seed in 0..32 {
        let m = module(MIGRATE);
        assert_eq!(
            run_scheduled(&m, 0, &[], 10_000_000, seed),
            Ok(vec![Value::I64(75)]),
            "seeded explorer, seed {seed}"
        );
    }
    assert_outcomes(MIGRATE, &[Ok(75)]);
}

/// **Racing resumes: exactly one claimant wins.** The root creates one `Pending` fiber and spawns
/// two workers that both `cont.resume` it (the fiber returns `arg + 41 = 42` to whichever wins).
/// In *every* interleaving exactly one worker wins and the other's lost claim is a `FiberFault`
/// that propagates through the root's join — so the outcome set is exactly `{FiberFault}`.
/// Non-vacuous: if both claims could win, both joins would succeed and `Ok(84)` would appear; if
/// neither could, the fiber's 42 would never be computed (covered by the single-worker control).
const RACE: &str = r#"
func () -> (i64) {
block 0 () {
  v0 = ref.func 2
  v1 = i64.const 0
  v2 = cont.new v0 v1
  v3 = thread.spawn 1 v2 v2
  v4 = thread.spawn 1 v2 v2
  v5 = thread.join v3
  v6 = thread.join v4
  v7 = i64.add v5 v6
  return v7
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  v0 = i64.const 1
  v1, v2 = cont.resume varg v0
  return v2
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  v0 = i64.const 41
  v1 = i64.add varg v0
  return v1
  }
}
"#;

#[test]
fn racing_resumes_have_exactly_one_winner() {
    assert_outcomes(RACE, &[Err(Trap::FiberFault)]);
    // The real pool agrees (whichever worker loses, the join propagates its fault).
    assert_eq!(run_i64(RACE), Err(Trap::FiberFault));
}

/// The single-worker control for [`RACE`]: with no competitor, the foreign claim **wins** — a
/// fiber created on the root starts and completes on the spawned vCPU (`1 + 41 = 42`). This is
/// the non-vacuity half: a foreign resume is genuinely a successful claim, not an always-fault.
const NO_RACE: &str = r#"
func () -> (i64) {
block 0 () {
  v0 = ref.func 2
  v1 = i64.const 0
  v2 = cont.new v0 v1
  v3 = thread.spawn 1 v2 v2
  v4 = thread.join v3
  return v4
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  v0 = i64.const 1
  v1, v2 = cont.resume varg v0
  return v2
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  v0 = i64.const 41
  v1 = i64.add varg v0
  return v1
  }
}
"#;

#[test]
fn foreign_vcpu_claim_succeeds_without_a_race() {
    assert_eq!(run_i64(NO_RACE), Ok(42));
    assert_outcomes(NO_RACE, &[Ok(42)]);
}

/// **The fiber quota is per-run now** (the registry is run-shared; DESIGN.md §23 (per-run quota)): with
/// `max_fibers = 2` (the root computation + one creation), the root's `cont.new` fills the run's
/// budget, so a *spawned vCPU's* `cont.new` trips it — under the old per-vCPU tables the child's
/// fresh table would have admitted it (this is the non-vacuous pin of the semantic change).
#[test]
fn fiber_quota_spans_vcpus() {
    let src = r#"
func () -> (i64) {
block 0 () {
  v0 = ref.func 2
  v1 = i64.const 0
  v2 = cont.new v0 v1
  v3 = thread.spawn 1 v1 v1
  v4 = thread.join v3
  return v4
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  v0 = ref.func 2
  v1 = cont.new v0 varg
  return v1
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  return varg
  }
}
"#;
    let m = module(src);
    let run_quota = |max_fibers: usize| -> Result<Vec<Value>, Trap> {
        let mut host = Host::new();
        host.set_quota(Quota {
            max_fibers,
            max_vcpus: 1 << 16,
        });
        let mut fuel = 10_000_000u64;
        run_with_host(&m, 0, &[], &mut fuel, &mut host)
    };
    assert_eq!(
        run_quota(2),
        Err(Trap::FiberFault),
        "the child's cont.new must trip the run-wide quota the root already filled"
    );
    assert_eq!(
        run_quota(3),
        Ok(vec![Value::I64(1)]),
        "one more slot admits it — and the child's handle (1) continues the run's numbering"
    );
}

// ---- #1761: the bytecode engine's parallel driver shares the registry too ------------------------
// The bytecode engine's cooperative driver always had one registry per run (its vCPUs share a
// thread), but its **parallel** driver (one OS thread per vCPU — the native stand-in for the
// browser's Web Workers) gave each vCPU its own, so every fixture above faulted there. Each now runs
// on both bytecode drivers and must agree with the tree-walker's M:N pool.

/// A fiber that crosses vCPUs **both ways**: the root creates F and runs it to its first `suspend`
/// (5); a spawned vCPU resumes it (F's first `suspend` returns 7) to its second `suspend` (8); the
/// root, after the join, resumes it again — F last parked on the *child's* vCPU — and F returns
/// `7*100 + 9`. The root sums 5 + 8 + 709 = 722. A JACL pool job's life: created on one worker,
/// run and parked on its owner, resumed by whichever worker next owns it.
const MIGRATE_BACK: &str = r#"
func () -> (i64) {
block 0 () {
  v0 = ref.func 2
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v3 = i64.const 5
  v4, v5 = cont.resume v2 v3
  v6 = thread.spawn 1 v2 v2
  v7 = thread.join v6
  v8 = i64.const 9
  v9, v10 = cont.resume v2 v8
  v11 = i64.add v5 v7
  v12 = i64.add v11 v10
  return v12
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  v0 = i64.const 7
  v1, v2 = cont.resume varg v0
  return v2
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  v0 = suspend varg
  v1 = i64.const 1
  v2 = i64.add v0 v1
  v3 = suspend v2
  v4 = i64.const 100
  v5 = i64.mul v0 v4
  v6 = i64.add v5 v3
  return v6
  }
}
"#;

/// `src` with a window, as the bytecode drivers run it (the fixtures above need none of their own).
fn windowed(src: &str) -> temen_ir::Module {
    module(&format!("memory 16\n{src}"))
}

fn i64_result(r: Result<Vec<Value>, Trap>) -> Result<i64, Trap> {
    r.map(|vals| match vals.as_slice() {
        [Value::I64(v)] => *v,
        other => panic!("expected one i64 result, got {other:?}"),
    })
}

/// The bytecode engine's cooperative driver.
fn bytecode_coop(src: &str) -> Result<i64, Trap> {
    let mut f = 10_000_000u64;
    let (r, _) =
        temen_interp::bytecode::compile_and_run_capture(&windowed(src), 0, &[], &mut f, &[])
            .expect("bytecode compiles the fixture");
    i64_result(r)
}

/// The bytecode engine's parallel driver: one OS thread per vCPU over one shared window.
fn bytecode_parallel(src: &str) -> Result<i64, Trap> {
    let size = 1usize << 16;
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero layout; the buffer is this window's alone until freed below.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `base` is `size` valid 8-aligned bytes, freed only after the region is dropped.
    let back = std::sync::Arc::new(unsafe { temen_interp::Region::shared(base, size as u64) });
    let mut f = 10_000_000u64;
    let (r, _) = temen_interp::bytecode::compile_and_run_capture_over_parallel(
        &windowed(src),
        0,
        &[],
        &mut f,
        &[],
        std::sync::Arc::clone(&back),
    )
    .expect("bytecode compiles the fixture");
    drop(back);
    // SAFETY: same layout; the region and every borrow of `base` are gone (the run joined its vCPUs).
    unsafe { std::alloc::dealloc(base, layout) };
    i64_result(r)
}

#[test]
fn a_fiber_migrates_across_vcpus_on_every_driver() {
    assert_eq!(run_i64(MIGRATE_BACK), Ok(722), "tree-walker M:N pool");
    for (name, src, want) in [
        ("MIGRATE", MIGRATE, Ok(75)),
        ("MIGRATE_BACK", MIGRATE_BACK, Ok(722)),
        ("NO_RACE", NO_RACE, Ok(42)),
    ] {
        assert_eq!(bytecode_coop(src), want, "{name}: bytecode cooperative");
        // Real threads: repeat, so a registry race would show as a flaky result.
        for i in 0..if cfg!(miri) { 2 } else { 50 } {
            assert_eq!(
                bytecode_parallel(src),
                want,
                "{name}: bytecode parallel (run {i})"
            );
        }
    }
}

/// The claim is the arbiter across OS threads too: of two workers racing to resume one fiber, one
/// wins and the other's `FiberFault` reaches the root through its join — the tree-walker's outcome.
#[test]
fn racing_resumes_have_one_winner_on_the_parallel_driver() {
    for i in 0..if cfg!(miri) { 2 } else { 50 } {
        assert_eq!(bytecode_parallel(RACE), Err(Trap::FiberFault), "run {i}");
    }
}

/// `vcpu.tls` is the **executing vCPU's** word (its spec, `temen_ir::Inst::VcpuTlsGet`), seeded to a
/// dense id — root 0, children in spawn order. A fiber created and first run on the root reads 0;
/// resumed on the spawned vCPU (id 1) it reads 1; so it returns `1*10 + 3`, and the root adds its
/// own 0. The bytecode engine used to keep the word per `Vm` (a fiber read its own, never-seeded 0)
/// and its parallel driver never seeded children — a JACL pool worker then found worker 0's queue.
const TLS_FOLLOWS_THE_VCPU: &str = r#"
func () -> (i64) {
block 0 () {
  v0 = ref.func 2
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v3 = i64.const 0
  v4, v5 = cont.resume v2 v3
  v6 = thread.spawn 1 v2 v2
  v7 = thread.join v6
  v8 = vcpu.tls.get
  v9 = i64.add v5 v7
  v10 = i64.add v9 v8
  return v10
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  v0 = i64.const 3
  v1, v2 = cont.resume varg v0
  return v2
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  v0 = vcpu.tls.get
  v1 = suspend v0
  v2 = vcpu.tls.get
  v3 = i64.const 10
  v4 = i64.mul v2 v3
  v5 = i64.add v4 v1
  return v5
  }
}
"#;

#[test]
fn vcpu_tls_is_the_executing_vcpus_word_on_every_driver() {
    assert_eq!(
        run_i64(TLS_FOLLOWS_THE_VCPU),
        Ok(13),
        "tree-walker M:N pool"
    );
    assert_eq!(
        bytecode_coop(TLS_FOLLOWS_THE_VCPU),
        Ok(13),
        "bytecode cooperative"
    );
    for i in 0..if cfg!(miri) { 2 } else { 20 } {
        assert_eq!(
            bytecode_parallel(TLS_FOLLOWS_THE_VCPU),
            Ok(13),
            "bytecode parallel (run {i})"
        );
    }
}
