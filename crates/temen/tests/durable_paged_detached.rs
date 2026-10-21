//! #2438 — a durable run freezes a live **demand-paged** detached child, and the thaw gives it back
//! with its pager. #2455 — **leaves first**: a freeze holds a pager at `NORMAL`, serving, until the
//! children it pages have unwound, so a child that faults after the freeze rang it is served and
//! unwinds, rather than stranding the freeze.
//!
//! The parents and children are `support/paged_detached.rs`'s.
#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

#[path = "support/detached_thaw.rs"]
mod detached_thaw;
#[path = "support/paged_detached.rs"]
mod paged_detached;

use detached_thaw::{child_module, run, Engine, Out, ARENA, PARENT_LOG2};
use paged_detached::*;
use temen_durable::{
    arm_freeze_after_backedges, init_durable_window, read_state, write_state, STATE_UNWINDING,
};

#[test]
fn a_paged_child_runs_to_its_answer_with_only_its_own_pages_served() {
    let child = child_module(&child_src(false));
    let parent = parent(PARENT_SRC, &[17]);
    let (out, ..) = run_from(&parent, &[&child], |_| {});
    assert_eq!(
        out,
        Out::Ret(EXPECT),
        "two guest pages served, the runtime's never asked for"
    );
}

/// The headline: cut mid-loop, after its first page was served, the child thaws with that page in
/// its image and its pager serving the next one.
#[test]
fn a_paged_child_cut_mid_run_thaws_with_its_pager() {
    let child = child_module(&child_src(false));
    let parent = parent(PARENT_SRC, &[17]);
    let (art, args) = artifact(&parent, &[&child], 1, |w| {
        arm_freeze_after_backedges(w, ARENA, 2000)
    });
    assert_eq!(
        thaw_on(Engine::Interp, &art, &parent, &[&child], &args),
        Some(Out::Ret(EXPECT)),
        "the thawed child resumes its loop and its pager serves the page it touches next"
    );
}

/// A child the freeze rang before it ran unwinds at its first safepoint, before it touches a page,
/// and thaws to the same answer. The parent was already unwinding when it spawned the child, so it
/// held for it (#2455), as it would have at a trigger.
#[test]
fn a_paged_child_frozen_before_it_runs_thaws_with_its_pager() {
    let child = child_module(&child_src(false));
    let parent = parent(PARENT_SRC, &[17]);
    let (art, args) = artifact(&parent, &[&child], 1, |w| {
        write_state(w, ARENA, STATE_UNWINDING)
    });
    assert_eq!(
        thaw_on(Engine::Interp, &art, &parent, &[&child], &args),
        Some(Out::Ret(EXPECT))
    );
}

/// #2455's repro: the freeze rings a paged child before it has run, and the child faults on its
/// first op, before its first safepoint. Its pager is held, still serving: the fault is served, the
/// child unwinds at its safepoint, then the pager does. Before leaves first, the pager had unwound
/// and served nothing, and the freeze was refused after the run had unwound.
#[test]
fn a_paged_child_that_faults_before_it_can_unwind_is_served_and_thaws() {
    let child = child_module(&child_src(true));
    let parent = parent(PARENT_SRC, &[17]);
    let (art, args) = artifact(&parent, &[&child], 1, |w| {
        write_state(w, ARENA, STATE_UNWINDING)
    });
    assert_eq!(
        thaw_on(Engine::Interp, &art, &parent, &[&child], &args),
        Some(Out::Ret(EXPECT))
    );
}

/// The same, from a countdown that fires mid-run: the parent's census passes, and the parent holds
/// for the live child it pages rather than unwinding.
#[test]
fn a_paged_child_rung_at_a_trigger_is_served_before_it_unwinds() {
    let child = child_module(&child_src(true));
    let parent = parent(PARENT_SRC, &[17]);
    // The countdown fires at the parent's first branch past the spawn (the transform's entry
    // dispatch is its first), long before its quantum ends and the child first runs.
    let (art, args) = artifact(&parent, &[&child], 1, |w| {
        arm_freeze_after_backedges(w, ARENA, 2)
    });
    assert_eq!(
        thaw_on(Engine::Interp, &art, &parent, &[&child], &args),
        Some(Out::Ret(EXPECT))
    );
}

/// #2455, the case the issue rates likely: a child streaming through memory its pager supplies a
/// page at a time touches a fresh page every few ops, so a freeze almost always rings it between a
/// fault and its next safepoint. Each cut captures it, and thaws to the fresh answer.
#[test]
fn a_paged_child_streaming_fresh_pages_freezes_mid_stream() {
    let child = child_module(&stream_src(28));
    let parent = parent(PARENT_SRC, &[22]);
    let (fresh, ..) = run_from(&parent, &[&child], |_| {});
    assert!(
        matches!(fresh, Out::Ret(v) if v >> 56 == 0 && (v >> 48) & 0xff == 224),
        "every page served once: {fresh:?}"
    );
    for arm in [20_000, 80_000] {
        let (art, args) = artifact(&parent, &[&child], 1, |w| {
            arm_freeze_after_backedges(w, ARENA, arm)
        });
        assert_eq!(
            thaw_on(Engine::Interp, &art, &parent, &[&child], &args),
            Some(fresh.clone()),
            "cut at back-edge {arm}"
        );
    }
}

/// #2455 — a child that unwinds while its pager is held is withheld from it: the pager runs on, and
/// must read the child as still running, since its own cut will record it so. Child A unwinds at once;
/// child B faults first, which keeps the pager held and running while it serves B. The parent waits
/// for A, joins it, then waits for B: had A read as ended, the parent would have joined it under the
/// hold and carried its placeholder past the cut.
#[test]
fn a_held_pager_reads_a_child_that_unwound_first_as_running() {
    let (a, b) = (
        child_module(&child_src(false)),
        child_module(&child_src(true)),
    );
    let parent = parent(TWO_PARENT_SRC, &[17, 17]);
    let (fresh, ..) = run_from(&parent, &[&a, &b], |_| {});
    assert_eq!(fresh, Out::Ret(EXPECT_TWO));
    let (art, args) = artifact(&parent, &[&a, &b], 2, |w| {
        arm_freeze_after_backedges(w, ARENA, 2)
    });
    assert_eq!(
        thaw_on(Engine::Interp, &art, &parent, &[&a, &b], &args),
        Some(Out::Ret(EXPECT_TWO))
    );
}

/// #1940, #2455 — the freeze's trigger lands while the child is parked on a fault
/// ([`fault_parked_at_the_trigger_thaws`]); the park matrix runs it as its tree-walker row.
#[test]
fn a_paged_child_parked_on_a_fault_at_the_trigger_unwinds_once_served() {
    fault_parked_at_the_trigger_thaws();
}

/// #2455 — a held pager kills a child whose unwind it is withheld from. The kill ends it as a kill
/// of a running child would: it is recorded as killed, and does not ride the artifact as running.
#[test]
fn a_held_pager_kills_a_child_that_unwound_first() {
    let (a, b) = (
        child_module(&child_src(false)),
        child_module(&child_src(true)),
    );
    let parent = parent(KILL_PARENT_SRC, &[17, 17]);
    let (fresh, ..) = run_from(&parent, &[&a, &b], |_| {});
    let Out::Ret(v) = fresh else {
        panic!("{fresh:?}")
    };
    assert_ne!(v >> 56, 0, "A ended by the kill");
    assert_eq!(
        v & ((1 << 52) - 1),
        EXPECT & ((1 << 48) - 1),
        "B ran to its end"
    );
    let (art, args) = artifact(&parent, &[&a, &b], 1, |w| {
        arm_freeze_after_backedges(w, ARENA, 2)
    });
    assert_eq!(
        thaw_on(Engine::Interp, &art, &parent, &[&a, &b], &args),
        Some(fresh)
    );
}

/// #2455 — a held pager kills the one child it is held for. The hold is over, and the release
/// reaches the pager while it runs: it unwinds at its next safepoint, the child recorded as killed.
#[test]
fn a_kill_of_the_last_child_a_pager_is_held_for_lets_it_unwind() {
    let child = child_module(&child_src(true));
    let parent = parent(KILL_AT_ONCE_PARENT_SRC, &[17]);
    let (fresh, ..) = run_from(&parent, &[&child], |_| {});
    assert!(matches!(fresh, Out::Ret(v) if v >> 56 != 0), "{fresh:?}");
    let (art, args) = artifact(&parent, &[&child], 0, |w| {
        arm_freeze_after_backedges(w, ARENA, 2)
    });
    assert_eq!(
        thaw_on(Engine::Interp, &art, &parent, &[&child], &args),
        Some(fresh)
    );
}

/// #2455 — a root that returns while it is held has completed the run: it answers as uninterrupted,
/// and nothing froze, though a child had unwound for the freeze.
#[test]
fn a_root_that_returns_while_held_completes_the_run() {
    let (a, b) = (
        child_module(&child_src(false)),
        child_module(&child_src(true)),
    );
    let parent = parent(RETURNS_PARENT_SRC, &[17, 17]);
    let (out, snap, mut fh, _) = run_from(&parent, &[&a, &b], |w| {
        arm_freeze_after_backedges(w, ARENA, 2)
    });
    assert_eq!(out, Out::Ret(42));
    assert_eq!(fh.take_freeze_declined(), None);
    assert_ne!(
        read_state(&snap, ARENA),
        STATE_UNWINDING,
        "the root did not freeze"
    );
    assert!(fh.frozen_root_sp().is_none() && fh.frozen_fibers().is_empty());
    assert!(fh.captured_detached().is_empty() && fh.unreached_detached().is_empty());
}

/// #2455 — a thawed run frozen again holds as a fresh one: the thaw's re-launched children are
/// withheld from their held pager too. The first cut has both children at their first safepoint. In
/// the thaw, A unwinds at its next one, while B, which streams through pages, faults again before
/// it reaches its own: the parent, held while it serves B, must read A as running.
#[test]
fn a_thawed_pager_holds_its_relaunched_children_again() {
    let (a, b) = (
        child_module(&child_src(false)),
        child_module(&stream_src(28)),
    );
    let parent = parent(TWO_PARENT_SRC, &[17, 22]);
    let (fresh, ..) = run_from(&parent, &[&a, &b], |_| {});
    let (art, args) = artifact(&parent, &[&a, &b], 2, |w| {
        arm_freeze_after_backedges(w, ARENA, 2)
    });
    let (mut h, mut win) = restored(&art, &parent, &[&a, &b]);
    // Past the parent's rewind, before its first quantum ends.
    arm_freeze_after_backedges(&mut win, ARENA, 10);
    let (_, snap) = run(Engine::Interp, &parent, &args, &win, &mut h).expect("runs");
    assert_eq!(h.take_freeze_declined(), None);
    assert_eq!(
        read_state(&snap, ARENA),
        STATE_UNWINDING,
        "the thaw froze again"
    );
    assert_eq!(h.captured_detached().len(), 2);
    let art = temen_snapshot::freeze(&parent, &snap, &h).expect("serialize");
    assert_eq!(
        thaw_on(Engine::Interp, &art, &parent, &[&a, &b], &args),
        Some(fresh)
    );
}

/// The Cranelift JIT pages a child only outside a durable run (#2489), so it refuses to thaw a paged
/// child rather than run it with no pager. A paged artifact thaws on the interpreter.
#[test]
fn the_jit_refuses_to_thaw_a_paged_child() {
    let child = child_module(&child_src(false));
    let parent = parent(PARENT_SRC, &[17]);
    let (art, args) = artifact(&parent, &[&child], 1, |w| {
        arm_freeze_after_backedges(w, ARENA, 2000)
    });
    assert_eq!(thaw_on(Engine::Jit, &art, &parent, &[&child], &args), None);
}

/// #2164 — a paged child's shadow arena is never demand-paged, in a run that is not durable too, on
/// the tree-walker and on the Cranelift JIT, which pages a child only outside a durable run. The child
/// reads a byte of its arena, which its window holds (0), then a page past it (100000), which its
/// pager supplies (24): one page served. Were the arena paged, the pager would refuse it, as below
/// 65536, and the child would fault.
#[test]
fn a_paged_childs_shadow_arena_is_its_own_outside_a_durable_run() {
    let child = child_module(
        "memory 17 shadow 16448 65536
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  vr = i64.const 20000
  vrb = i32.load8_u vr
  va = i64.const 100000
  vab = i32.load8_u va
  vr64 = i64.extend_i32_u vrb
  v8 = i64.const 8
  vsh = i64.shl vr64 v8
  va64 = i64.extend_i32_u vab
  vout = i64.or vsh va64
  return vout
  }
}
",
    );
    let parent = parent(PARENT_SRC, &[17]);
    for engine in [Engine::Interp, Engine::Jit] {
        let (mut h, args) = host(&parent, &[&child]);
        h.set_durable(false);
        let win = init_durable_window(1 << PARENT_LOG2, ARENA);
        let (out, _) =
            run(engine, &parent, &args, &win, &mut h).unwrap_or_else(|| panic!("{engine:?} runs"));
        assert_eq!(out, Out::Ret((1 << 48) | 24), "{engine:?}");
    }
}
