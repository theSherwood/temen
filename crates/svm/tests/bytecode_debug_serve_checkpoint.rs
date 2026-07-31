//! Reverse-debug **checkpointing of a §3.6 serve loop** (debugging slice 7). A serving domain drains
//! its embedder-seeded dispatch queue at a `svc.poll` service point, running each handler as a
//! rewind-linked activation over the one world (ISSUES.md I36 slice 1 — the native bytecode serve loop).
//! Across that drain the run holds **serve state** that is *not* in any `Vm`: the inbound queue, the
//! settled completion cells, and the next ticket ([`Host::svc_state`]). Before this slice the checkpoint
//! ladder captured only the `Vm` + window + I/O substate, so a checkpoint taken mid-drain — the queue
//! partly drained, or a handler activation on the stack — silently dropped the remaining queue and the
//! settled cells on restore, and a warm replay diverged. Now that serve state rides in the host replay
//! substate ([`Host::replay_substate`]/`restore_replay_substate`, DURABILITY.md §13.4 "serialize as-is"),
//! **every** clock round-trips — including while a handler is in flight (`serve_ticket = Some`, carried in
//! the `Vm` clone) and between handlers.
//!
//! This is a latent-divergence fix as much as a subset extension: the pre-slice engine returned `Some`
//! from `snapshot()` on a serving run (nothing gated it) but reconstructed the wrong state, so the
//! round-trip assertions below would have *failed*. Invariant #7: nothing captured here is a
//! waiter/scheduler record — the single-domain debug engine parks no cross-domain caller; the queue and
//! cells are plain data.

use std::sync::Arc;
use svm_interp::bytecode::{DebugRun, ScheduledDebugRun};
use svm_interp::{run_with_host, Host, Value};
use svm_text::parse_module;

// The §3.6 slice-2 serving domain, verbatim from `svm-interp/tests/bytecode_svc.rs`: offer "counter"
// op 0 = func 1 `bump(x) -> old + x` over the LIVE value at mem[0]; `main` (func 0) seeds mem[0] = 7,
// `svc.poll`s (`cap.call CAP_SELF(=u32::MAX) 9`) draining the embedder-seeded queue, and returns
// `served * 1000 + mem[0]`. Pure handler, no park-capable seam — so it compiles to the native serve
// loop the debug engine drives op-by-op (stepping *into* each handler body).
const SERVER: &str = r#"
memory 16
type 0 func (i64) -> (i64)
type 1 interface { bump: 0 }
export 0 interface "counter" 1 { bump: 1 }

func () -> (i64) {
block 0 () {
  va = i64.const 0
  vseed = i64.const 7
  i64.store va vseed
  vz = i32.const 0
  vn = cap.call 4294967295 9 () -> (i64) vz ()
  vafter = i64.load va
  vk = i64.const 1000
  vm = i64.mul vn vk
  vr = i64.add vm vafter
  return vr
  }
}

func (i64) -> (i64) {
block 0 (vx: i64) {
  va = i64.const 0
  vold = i64.load va
  vnew = i64.add vold vx
  i64.store va vnew
  return vold
  }
}
"#;

/// Three bumps — enough to exercise multiple handler activations and every between-handler boundary in
/// the drain. Chosen so each handler leaves a distinct live counter at mem[0]: 7 → 12 → 42 → 142.
fn dispatches() -> Vec<(u32, u32, Vec<i64>)> {
    vec![(0, 0, vec![5]), (0, 0, vec![30]), (0, 0, vec![100])]
}

fn module() -> Arc<svm_ir::Module> {
    let m = parse_module(SERVER).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    Arc::new(m)
}

/// A host seeded exactly as the oracle's: self-module registered, the queue pre-loaded with the
/// dispatches. Both the reference `DebugRun`, each warm `session()`, and the oracle build one of these,
/// so they serve the identical queue.
fn seeded_host(m: &Arc<svm_ir::Module>) -> Host {
    let mut host = Host::new();
    host.set_self_module(m);
    for (e, o, a) in dispatches() {
        host.svc_enqueue(e, o, a).expect("under queue cap");
    }
    host
}

/// A fresh single-vCPU `DebugRun` on the serving kernel, its host pre-seeded with the dispatch queue.
fn session() -> DebugRun {
    let m = module();
    let host = seeded_host(&m);
    DebugRun::new_with_host(&m, 0, &[], host)
        .expect("bytecode debug engine drives the native serve loop")
}

/// The expected result, pinned against the tree-walker oracle over the *same* seeded queue:
/// `served(3) * 1000 + mem[0](142) = 3142`.
fn oracle_result() -> Vec<Value> {
    let m = module();
    let mut host = seeded_host(&m);
    let mut fuel = 5_000_000u64;
    run_with_host(&m, 0, &[], &mut fuel, &mut host).expect("tree-walker serves")
}

/// A per-op observation: the op clock, the call-stack `IrPc`s (which enter the handler body when the
/// serve loop admits a dispatch — so a mid-handler stop is visible here), and the first 8 window bytes
/// holding the live counter mem[0] that each handler mutates.
fn obs(run: &DebugRun) -> (u64, String, Vec<u8>) {
    let clock = run.op_clock();
    let mut frames = Vec::new();
    for d in 0..run.depth() {
        if let Some(pc) = run.frame_pc(d) {
            frames.push(format!("f{}b{}i{}", pc.func, pc.block, pc.inst));
        }
    }
    let win = run.read_window(0, 8).unwrap_or_default();
    (clock, frames.join(","), win)
}

/// Warm≡cold oracle for **§7 serve-loop checkpointing** (DEBUGGING.md W1): at every clock where a
/// checkpoint can be taken — including while the serve loop is mid-drain and while a handler activation
/// is in flight — a `DebugRun::restore`d run replays forward identically to the trusted from-0 run, to
/// the same final result. This exercises the serve-state round-trip (`svc_state`/`set_svc_state` carried
/// in the host replay substate) and the preserved in-flight `serve_ticket`.
#[test]
fn serve_loop_checkpoint_snapshot_restore_round_trips() {
    const FUEL: u64 = 5_000_000;
    let want = oracle_result();

    let mut refr = session();
    let mut f = FUEL;
    let mut ref_obs = vec![obs(&refr)];
    while refr.tick(&mut f) {
        ref_obs.push(obs(&refr));
    }
    let total = ref_obs.len() - 1;
    assert_eq!(
        refr.result(),
        Some(&Ok(want.clone())),
        "reference serve run matches the oracle"
    );

    // No fibers/regions; the serving run's queue + cells now ride in the snapshot, so *every* clock is
    // checkpointable — including mid-drain (queue partly consumed) and mid-handler (`serve_ticket =
    // Some`). Before this slice `snapshot()` still returned `Some` here (nothing gated serving) but
    // `restore` rebuilt the wrong serve state, so the forward-replay asserts below would have diverged.
    let mut checkpoints = 0usize;
    for c in 0..=total {
        let mut at_c = session();
        let mut f = FUEL;
        while at_c.op_clock() < c as u64 && at_c.tick(&mut f) {}
        let Some(snap) = at_c.snapshot() else {
            continue;
        };
        checkpoints += 1;

        let mut warm = session();
        warm.restore(&snap);
        let mut i = c;
        assert_eq!(
            obs(&warm),
            ref_obs[i],
            "restore at C={c} lands at the reference state"
        );
        let mut f = FUEL;
        while warm.tick(&mut f) {
            i += 1;
            assert_eq!(
                obs(&warm),
                ref_obs[i],
                "forward serve replay diverged after restore at C={c}"
            );
        }
        assert_eq!(i, total, "warm serve run from C={c} reached the same end");
        assert_eq!(
            warm.result(),
            Some(&Ok(want.clone())),
            "warm serve run from C={c} matches the oracle"
        );
    }
    assert_eq!(
        checkpoints,
        total + 1,
        "every clock is checkpointable — including mid-drain and mid-handler, which a snapshot that \
         omitted the §3.6 serve state could not have round-tripped"
    );
}

/// Pin the headline serve semantics so the round-trip above is anchored to concrete values, not just
/// internal consistency: three bumps over the live counter starting at 7 give cells 7/12/42 and a final
/// `served(3)*1000 + mem[0](142) = 3142`.
#[test]
fn serve_run_result_is_pinned() {
    assert_eq!(oracle_result(), vec![Value::I64(3142)]);
}

/// A root-only serving run on the **multi-vCPU** scheduler. The pure serve module (no park seam / no
/// `instantiate`) is admitted natively on both debug engines, so the serve-state round-trip must hold
/// through [`ScheduledSnapshot`] too — the same `HostReplaySubstate` seam carries it. Mirrors the
/// single-vCPU round-trip over global turns.
fn sched_session() -> ScheduledDebugRun {
    let m = module();
    let host = seeded_host(&m);
    ScheduledDebugRun::new_with_host(&m, 0, &[], host)
        .expect("scheduled debug engine drives the native serve loop")
}

fn sched_obs(run: &ScheduledDebugRun) -> (u64, Vec<u8>) {
    (run.turn(), run.read_window(0, 8).unwrap_or_default())
}

#[test]
fn scheduled_serve_loop_checkpoint_snapshot_restore_round_trips() {
    const FUEL: u64 = 5_000_000;
    let want = oracle_result();

    let mut refr = sched_session();
    let mut f = FUEL;
    let mut ref_obs = vec![sched_obs(&refr)];
    while refr.tick(&mut f) {
        ref_obs.push(sched_obs(&refr));
    }
    let total = ref_obs.len() - 1;
    assert_eq!(
        refr.result(),
        Some(&Ok(want.clone())),
        "scheduled serve run matches the oracle"
    );

    let mut checkpoints = 0usize;
    for c in 0..=total {
        let mut at_c = sched_session();
        let mut f = FUEL;
        while at_c.op_turn() < c as u64 && at_c.tick(&mut f) {}
        let Some(snap) = at_c.snapshot() else {
            continue;
        };
        checkpoints += 1;

        let mut warm = sched_session();
        warm.restore(&snap);
        let mut i = c;
        assert_eq!(
            sched_obs(&warm),
            ref_obs[i],
            "scheduled restore at C={c} lands"
        );
        let mut f = FUEL;
        while warm.tick(&mut f) {
            i += 1;
            assert_eq!(
                sched_obs(&warm),
                ref_obs[i],
                "scheduled forward serve replay diverged after restore at C={c}"
            );
        }
        assert_eq!(
            i, total,
            "scheduled warm serve run from C={c} reached the same end"
        );
        assert_eq!(warm.result(), Some(&Ok(want.clone())));
    }
    assert_eq!(
        checkpoints,
        total + 1,
        "every scheduled turn is checkpointable — the serve state rides in ScheduledSnapshot's host substate"
    );
}
