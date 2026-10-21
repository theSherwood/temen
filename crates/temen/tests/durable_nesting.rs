//! DURABILITY.md §4 — what a durable domain admits, and a freeze around its §14 children:
//!
//! * a durable domain's guest-driven `Jit.compile` (§22 — also a module installation) fails closed
//!   until a host-side instrumentation hook exists;
//! * a freeze landing after a detached child was joined thaws by reloading the join result, never
//!   re-running the child;
//! * a detached child never turns a freeze into a trap (INVARIANTS #5);
//! * the bytecode engine's durable-capture entry declines a module that spawns, leaving the nesting
//!   rules to the tree-walker.
//!
//! The freeze of a live child subtree and the admission of an unmarked grant are pinned on detached
//! children, the only kind left (#1867): `durable_detached_jit.rs` and temen-interp's
//! `durable_detached_parity.rs`. These parents spawn through a spawn record above their shadow arena,
//! storing the module and budget in it, so they take the confined transform
//! ([`instrument_parent`]): the strict one refuses any guest memory op (R9, `GuestUsesMemory`).

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use temen_durable::{
    arm_freeze_after, begin_thaw, init_durable_window, read_state, transform_module,
    transform_module_assume_confined, write_state, STATE_NORMAL, STATE_UNWINDING,
};
use temen_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use temen_ir::{Module, SpawnRec};
use temen_text::parse_module;
use temen_verify::verify_module;

const SIZE_LOG2: u8 = 18;
const WINDOW: usize = 1 << SIZE_LOG2;
use temen_ir::errno::EINVAL;

/// The arena every durable test module declares: the pre-#1503 fixed placement `[guard+64, 1<<16)`.
const TEST_ARENA: temen_ir::durable_abi::ShadowArena =
    temen_ir::durable_abi::ShadowArena::new(16448, 65536);

fn instrument(src: &str) -> Module {
    let m = parse_module(src).expect("parse");
    let inst = transform_module(&m).expect("transform");
    verify_module(&inst).expect("instrumented module verifies");
    inst
}

/// A durable parent from its source, with the spawn record it spawns through at 65536, above its
/// shadow arena: the child's 128 KiB window. The parent stores the module and budget in the record,
/// a guest memory op, so it takes the confined transform.
fn instrument_parent(src: &str) -> Module {
    let spawn = SpawnRec {
        size_log2: 17,
        ..SpawnRec::default()
    };
    let m = parse_module(&format!("{src}{}", rec::segment(65536, &spawn))).expect("parse");
    let inst = transform_module_assume_confined(&m).expect("transform");
    verify_module(&inst).expect("instrumented module verifies");
    inst
}

/// The child "plugin": 128 KiB window (its durable reserve + room), pure-compute entry
/// (`(i64) -> (i64)`, the starter-cap convention) returning 4321. Instrumented, so a durable
/// parent may admit it — *when the grant attests it*.
fn child() -> Module {
    instrument(
        "memory 17 shadow 16448 65536
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 4321
  return v1
  }
}
",
    )
}

/// §22 guest-driven JIT is a module installation too: a durable domain's `Jit.compile` fails
/// closed (`-EINVAL`) even with a working validator installed — an un-instrumented unit could
/// never drain-then-unwind. The identical non-durable host compiles the same bytes fine.
#[test]
fn durable_domain_refuses_guest_jit_compile() {
    fn trivial_validator(
        _bytes: &[u8],
        _mem: Option<u8>,
        _symtab: &[u8],
    ) -> Result<temen_interp::JitValidated, i64> {
        let m = parse_module(
            "func () -> (i64) {
block 0 () {
  v0 = i64.const 1
  return v0
  }
}
",
        )
        .expect("parse unit");
        Ok(temen_interp::JitValidated {
            funcs: m.funcs.into(),
            types: m.types.into(),
            data: None,
        })
    }

    // Control: the same validator + grant on a NON-durable host compiles.
    let mut host = Host::new();
    host.set_jit_validator(trivial_validator);
    let jh = host.grant_jit(None);
    assert!(
        host.jit_compile(jh, b"unit").expect("no trap").is_ok(),
        "non-durable: the unit compiles"
    );

    // Durable: same setup, compile fails closed before any unit is stored.
    let mut host = Host::new();
    host.set_durable(true);
    host.set_jit_validator(trivial_validator);
    let jh = host.grant_jit(None);
    assert!(
        matches!(host.jit_compile(jh, b"unit").expect("no trap"), Err(e) if e == EINVAL),
        "durable: guest JIT compile is refused fail-closed"
    );
}

// ---- Freeze × §14 children ----

/// Durable parent that spawns its granted child **detached** (through the spawn record at 65536 that
/// [`instrument_parent`] lays: no grants, `size_log2` 17; the module and budget stored there) and
/// joins it, then drives a fiber that suspends once (the armed freeze trigger ticks on
/// `cont.resume`, so `arm = 2` lands the freeze at the second resume — after the join, with the
/// fiber parked: the covered residue shape). Returns child result + fiber result = 4321 + 55 = 4376.
const PARENT_DETACHED_JOIN_THEN_FIBER: &str = "memory 18 shadow 16448 65536
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vrm = i64.const 65544
  i32.store vrm v1
  vrb = i64.const 65548
  i32.store vrb v2
  vrp = i64.const 65536
  v5 = call.cap 6 17 (i64) -> (i32) v0 (vrp)
  v6_st, v6 = call.cap 6 1 (i32) -> (i64, i64) v0 (v5)
  v7 = ref.func 1
  v8 = i64.const 4096
  v9 = cont.new v7 v8
  v10 = i64.const 0
  v11, v12 = cont.resume v9 v10
  v13, v14 = cont.resume v9 v12
  v15 = i64.add v6 v14
  return v15
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 5
  v3 = suspend v2
  v4 = i64.const 55
  return v4
  }
}
";

/// A durable powerbox for [`PARENT_DETACHED_JOIN_THEN_FIBER`]: the `Instantiator`, [`child`]'s
/// durable module grant, a `Budget` for its window, and the freeze authority over detached progeny
/// a durable domain needs to spawn detached at all — with their handles as the parent's entry args.
/// Every call grants in the same order, so a thawing host's handles are the frozen ones.
fn detached_powerbox() -> (Host, [Value; 3]) {
    let mut host = Host::new();
    host.set_durable(true);
    let inst = host.grant_instantiator();
    let modh = host.grant_durable_module(&child());
    let budget = host.grant_budget(-1, 1 << 20, -1);
    host.grant_freeze_authority();
    (
        host,
        [Value::I32(inst), Value::I32(modh), Value::I32(budget)],
    )
}

/// The refusal must not over-fire: a freeze landing **after** the §14 child was joined mints a
/// valid artifact, and the thaw **reloads** the child's join result (reload-not-reissue — the
/// child is never re-run) and reproduces the uninterrupted total.
#[test]
fn freeze_after_detached_child_joined_thaws_and_reloads_the_join_result() {
    let parent = instrument_parent(PARENT_DETACHED_JOIN_THEN_FIBER);

    // Control: uninterrupted total.
    let (mut host, args) = detached_powerbox();
    let mut fuel = 5_000_000u64;
    let (base, _) = run_capture_reserved_with_host(
        &parent,
        0,
        &args,
        &mut fuel,
        &init_durable_window(WINDOW, TEST_ARENA),
        SIZE_LOG2,
        &mut host,
    );
    assert_eq!(base, Ok(vec![Value::I64(4321 + 55)]), "uninterrupted total");

    // Armed freeze: tick 1 at the first resume (runs; the fiber parks), tick 2 at the second —
    // the freeze lands with the child already joined (its result checkpointed in the parent's
    // shadow frame) and the fiber parked (rides as Section-2 residue).
    let (mut fhost, fargs) = detached_powerbox();
    let mut win = init_durable_window(WINDOW, TEST_ARENA);
    arm_freeze_after(&mut win, 2);
    let mut fuel = 5_000_000u64;
    let (fr, fsnap) =
        run_capture_reserved_with_host(&parent, 0, &fargs, &mut fuel, &win, SIZE_LOG2, &mut fhost);
    assert!(
        fr.is_ok(),
        "freeze-after-join returns a placeholder: {fr:?}"
    );
    assert_eq!(
        read_state(&fsnap),
        STATE_UNWINDING,
        "the armed freeze landed (joined child does not refuse it)"
    );
    assert!(
        fhost.captured_detached().is_empty(),
        "a joined child is not captured"
    );

    // Thaw: the parent rewinds, reloading the spawn handle and the join result from its shadow
    // frame — the child never re-runs — and completes to the uninterrupted total.
    let mut twin = fsnap.clone();
    begin_thaw(&mut twin, TEST_ARENA, 0);
    let (mut thost, targs) = detached_powerbox();
    thost.set_frozen_fibers(fhost.frozen_fibers().to_vec());
    let mut fuel = 5_000_000u64;
    let (tr, tsnap) =
        run_capture_reserved_with_host(&parent, 0, &targs, &mut fuel, &twin, SIZE_LOG2, &mut thost);
    assert_eq!(
        tr,
        Ok(vec![Value::I64(4321 + 55)]),
        "thaw reproduces the total — the §14 join result reloaded, the child never re-ran"
    );
    assert_eq!(read_state(&tsnap), STATE_NORMAL, "thaw back to NORMAL");
}

/// The bytecode engine's durable-capture entry must **decline** a §14 module (fall back to the
/// tree-walker, which owns the durable nesting rules): its own instantiate arm has neither the
/// admission check nor the fail-closed, so driving a durable §14 module there would mint the exact
/// thaw-faulting artifact the tree-walker refuses. (temen-run's bytecode backend falls back on
/// `None`, so an embedder always lands on the enforced path.)
#[test]
fn bytecode_durable_capture_declines_a_nesting_module() {
    let parent = instrument_parent(PARENT_DETACHED_JOIN_THEN_FIBER);
    let (mut host, args) = detached_powerbox();
    let mut fuel = 5_000_000u64;
    let r = temen_interp::bytecode::compile_and_run_capture_reserved_with_host(
        &parent,
        0,
        &args,
        &mut fuel,
        &init_durable_window(WINDOW, TEST_ARENA),
        SIZE_LOG2,
        &mut host,
    );
    assert!(
        r.is_none(),
        "the bytecode durable entry must decline §14 modules (tree-walker owns the nesting rules)"
    );
}

/// #1412 (owner decision 2026-09-14) — **a freeze can no longer be killed by this path.**
///
/// Under #1289 R1 a durable domain could spawn a detached child, and freezing while that child was
/// live failed closed with `Trap::ThreadFault` — a *trap*, ending the run, on a platform lifecycle
/// action the guest cannot see coming. INVARIANTS #5 is explicit that "a lifecycle event is never a
/// domain-killing surprise", so that was the wrong failure shape even though the fail-closed instinct
/// was right.
///
/// With the interim spawn gate restored (#1412), a durable domain's detached spawn refuses probeably,
/// the detached child never comes into existence, and `detached_live_refused` is unreachable by
/// construction — so the freeze simply completes. This test pins that: **not** `ThreadFault`.
///
/// The parent is frozen from the start, so the run's value is the unwind's rather than the guest's
/// `-EINVAL` (the spawn's own refusal is pinned in `detached_windows.rs`); what matters here is the
/// absence of a trap.
///
/// **Rewritten again when the capture lands** (#1361): the spawn succeeds, and the freeze captures the
/// child into its own artifact rather than refusing anything.
const PARENT_DETACHED: &str = "memory 18 shadow 16448 65536
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vrm = i64.const 65544
  i32.store vrm v1
  vrb = i64.const 65548
  i32.store vrb v2
  vrp = i64.const 65536
  vs = call.cap 6 17 (i64) -> (i32) v0 (vrp)
  vr = i64.extend_i32_s vs
  return vr
  }
}
";

#[test]
fn a_durable_freeze_is_never_killed_by_a_detached_child() {
    let parent = instrument_parent(PARENT_DETACHED);
    let child_mod = child(); // memory 17 = the record's size_log2

    let mut host = Host::new();
    host.set_durable(true);
    let ih = host.grant_instantiator();
    let mh = host.grant_module(&child_mod);
    let bh = host.grant_budget(-1, (1u64 << 17) as i64, -1); // one detached window
    let mut win = init_durable_window(WINDOW, TEST_ARENA);
    write_state(&mut win, STATE_UNWINDING); // freeze from the start
    let mut fuel = 50_000_000u64;
    let (r, _snap) = run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(ih), Value::I32(mh), Value::I32(bh)],
        &mut fuel,
        &win,
        SIZE_LOG2,
        &mut host,
    );
    assert!(
        !matches!(r, Err(Trap::ThreadFault)),
        "a freeze must not end the domain in a trap: the detached spawn is refused as a value, so \
         there is no live detached child for the freeze to choke on (INVARIANTS #5) — got {r:?}"
    );
    assert!(
        r.is_ok(),
        "and the freeze completes rather than failing some other way: {r:?}"
    );
}
