//! **Checkpointing a guest that holds a host capability** (#1455, the ladder half).
//!
//! `Host::checkpoint_safe` used to require `host_procs.is_empty()`, so the time-travel checkpoint
//! ladder self-disabled for every guest that holds one — which is every *interesting* guest: a debugged
//! C program that does file I/O holds `vm_fs`, a playground reactor holds `display`/`keyboard`/`fs`.
//! Reverse debugging still worked (the `CapTape` records every `HOST_PROC` crossing, so a from-0 replay
//! reproduces them), but every `seek`/`step_back` paid O(t) instead of one stride.
//!
//! Two things make a host capability restorable, and they are the same two the §12 freeze path relies
//! on (`capture_durable_handles`):
//!
//! * a registered **name** — the reconstruction rule, since the powerbox mints by name in a fixed
//!   order, so a rebuilt run grants the same set in the same order and restored frames' handle values
//!   stay valid. An unnamed host-fn has no such rule and still disqualifies the run;
//! * its **own declared state** (`set_cap_state_capture`/`_restore`) riding the checkpoint, so a
//!   capability with guest-observable state of its own does not come back at its initial value under a
//!   guest resumed at logical time `c`. That is the identical pair a freeze writes into the artifact's
//!   named-capability section — one definition, read two ways.
//!
//! The `counter` capability below is the smallest guest-observable-state case there is: each call
//! returns the next integer. A checkpoint that dropped its state would restore a guest mid-run beside a
//! capability counting from zero again, and the forward replay would diverge on the very next call —
//! which is exactly what this asserts does *not* happen.

use std::sync::{Arc, Mutex};
use temen_interp::bytecode::ScheduledDebugRun;
use temen_interp::{run_with_host, GuestMem, Host, RegionMinter, Value};
use temen_text::parse_module;

/// A loop that calls the `cnt` capability once per iteration, accumulating what it returns into
/// mem[16384] (above the #1094 NULL guard) — so a restore has to reproduce both the continuation *and*
/// the capability's own state to keep agreeing. The guest is identical in both arms below; the only
/// thing that varies is whether the host registered a name for the grant, which is the axis under test.
const CAP_LOOP: &str = r#"
memory 16
func (i64) -> (i64) {
block 0 (vn: i64) {
  vacc0 = i64.const 0
  br 1(vn, vacc0)
}
block 1 (vi: i64, vacc: i64) {
  vz = i64.eqz vi
  br_if vz 2(vacc) 3(vi, vacc)
}
block 2 (vsum: i64) {
  return vsum
}
block 3 (vi2: i64, vacc2: i64) {
  vzero = i64.const 0
  vh = i32.const 0
  vt = call.sym "cnt" (i64) -> (i64) vh (vzero)
  vsum = i64.add vacc2 vt
  vcell = i64.const 16384
  i64.store vcell vsum
  vm1 = i64.const -1
  vnext = i64.add vi2 vm1
  br 1(vnext, vsum)
  }
}
"#;

const ITERS: i64 = 12;
const FUEL: u64 = 5_000_000;

fn module() -> Arc<temen_ir::Module> {
    let m = parse_module(CAP_LOOP).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    Arc::new(m)
}

/// Grant a **stateful** host capability on `host`: each call returns the next integer. Its state is the
/// counter, declared through the #1455 capture/restore pair, so a checkpoint carries it. `named`
/// selects whether the grant is registered under a name — the reconstruction rule the ladder requires.
fn grant_counter(host: &mut Host, named: bool) {
    let n = Arc::new(Mutex::new(0i64));
    let call = Arc::clone(&n);
    let h = host.grant_host_proc(Box::new(
        move |_op: u32,
              _args: &[i64],
              _mem: Option<&mut dyn GuestMem>,
              _minter: Option<&mut dyn RegionMinter>| {
            let mut c = call.lock().unwrap();
            *c += 1;
            Ok(vec![*c])
        },
    ));
    if named {
        host.register_cap_name("cnt", h);
    }
    // IMPORTS.md phase 4, exactly as the DAP session's `grant_io_powerbox` binds its `vm_fs` seam: the
    // module's one import (`call.sym "cnt"` interns it) maps to this flat host-proc.
    host.set_import_bindings(vec![temen_interp::BoundImport::required(
        temen_interp::cap_id::HOST_PROC,
        0,
        h,
    )]);
    let cap = Arc::clone(&n);
    host.set_cap_state_capture(
        h,
        Box::new(move || cap.lock().unwrap().to_le_bytes().to_vec()),
    );
    let put = Arc::clone(&n);
    host.set_cap_state_restore(
        h,
        Box::new(move |b| {
            let mut buf = [0u8; 8];
            let k = b.len().min(8);
            buf[..k].copy_from_slice(&b[..k]);
            *put.lock().unwrap() = i64::from_le_bytes(buf);
        }),
    );
}

fn session(named: bool) -> ScheduledDebugRun {
    let m = module();
    let mut host = Host::new();
    host.set_self_module(&m);
    grant_counter(&mut host, named);
    ScheduledDebugRun::new_with_host(&m, 0, &[Value::I64(ITERS)], host)
        .expect("the bytecode debug engine accepts the cap loop")
}

/// The result the tree-walk oracle produces over the same powerbox: 1+2+…+ITERS.
fn oracle_result() -> Vec<Value> {
    let m = module();
    let mut host = Host::new();
    host.set_self_module(&m);
    grant_counter(&mut host, true);
    let mut fuel = FUEL;
    run_with_host(&m, 0, &[Value::I64(ITERS)], &mut fuel, &mut host).expect("tree-walker runs")
}

/// A per-op observation: the turn, the call stack, and the accumulator the loop stores — enough that a
/// capability restored to the wrong count shows up on the next iteration.
fn obs(run: &ScheduledDebugRun) -> (u64, String, Vec<u8>) {
    let mut frames = Vec::new();
    for d in 0..run.depth() {
        if let Some(pc) = run.frame_pc(d) {
            frames.push(format!("f{}b{}i{}", pc.func, pc.block, pc.inst));
        }
    }
    (
        run.op_turn(),
        frames.join(","),
        run.read_window(16384, 8).unwrap_or_default(),
    )
}

/// The headline: a run holding a named host capability is checkpointable at every clock, and a restore
/// into a **freshly built** run — fresh powerbox, capability counting from zero again — replays forward
/// identically to the trusted from-0 run, to the same result.
#[test]
fn a_named_host_capability_checkpoints_and_restores() {
    let want = oracle_result();
    assert_eq!(
        want,
        vec![Value::I64((1..=ITERS).sum())],
        "the guest sums what the capability returns, so the capability's own state is in the answer"
    );

    let mut refr = session(true);
    let mut f = FUEL;
    let mut ref_obs = vec![obs(&refr)];
    while refr.tick(&mut f) {
        ref_obs.push(obs(&refr));
    }
    let total = ref_obs.len() - 1;
    assert_eq!(refr.result(), Some(&Ok(want.clone())));

    let mut checkpoints = 0usize;
    for c in 0..=total {
        let mut at_c = session(true);
        let mut f = FUEL;
        while at_c.op_turn() < c as u64 && at_c.tick(&mut f) {}
        let Some(snap) = at_c.snapshot() else {
            continue;
        };
        checkpoints += 1;

        let mut warm = session(true);
        warm.restore(at_c.op_turn(), &snap);
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
                "forward replay diverged after restore at C={c} — the capability came back at the \
                 wrong count"
            );
        }
        assert_eq!(i, total, "warm run from C={c} reached the same end");
        assert_eq!(
            warm.result(),
            Some(&Ok(want.clone())),
            "warm run from C={c} matches the oracle"
        );
    }
    assert_eq!(
        checkpoints,
        total + 1,
        "every clock of a named-capability run is checkpointable — before #1455 not one was"
    );
}

/// The negative: an **unnamed** host capability is an opaque closure with no reconstruction rule, so the
/// run stays outside the snapshottable subset and the backend falls back to replay-from-clock-0. This is
/// the same fail-closed rule `capture_durable_handles` applies to an unnamed `Binding::HostProc`.
#[test]
fn an_unnamed_host_capability_still_refuses_a_checkpoint() {
    let mut run = session(false);
    let mut f = FUEL;
    // Run a little way in, so this is "a live unnamed capability", not "nothing has happened yet".
    for _ in 0..40 {
        if !run.tick(&mut f) {
            break;
        }
    }
    assert!(
        run.snapshot().is_none(),
        "an unnamed host capability has nothing to re-grant it by — the ladder must refuse rather \
         than restore a guest beside a capability the rebuild could not reproduce"
    );
}

/// …and the refusal is about the *name*, not about holding a capability at all: the identical run with
/// the grant registered is checkpointable. Pins the two halves against each other so a future change
/// cannot quietly widen or narrow the rule without one of them failing.
#[test]
fn the_refusal_is_the_name_not_the_capability() {
    let mut f = FUEL;
    let mut named = session(true);
    let mut unnamed = session(false);
    for _ in 0..40 {
        named.tick(&mut f);
        unnamed.tick(&mut f);
    }
    assert!(named.snapshot().is_some());
    assert!(unnamed.snapshot().is_none());
}
