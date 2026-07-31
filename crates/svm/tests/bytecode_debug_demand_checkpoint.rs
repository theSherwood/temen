//! Reverse-debug **checkpointing of a §14 demand (fault-driven-yield) coroutine** (debugging slice 6b).
//! A demand child starts with its whole window `Unmapped` and faults into the parent on first touch of
//! each page; the parent supplies the page (places its bytes in the shared window, marks it `Rw`) and
//! resumes. That makes the child's `nested_view` layout **non-pristine** — so before this slice the
//! checkpoint ladder excluded any run holding a demand coroutine (`fault_yields`), falling back to
//! replay-from-clock-0. Now the child's own page-protection map rides in its `CoroSnapshot`
//! ([`Mem::prot_snapshot`]/`install_prot`) — reinstated on restore, with its bytes carried by the parent
//! window snapshot — and `fault_yields` is preserved, so a mid-demand coroutine round-trips.

use svm_interp::bytecode::DebugRun;
use svm_interp::{run_with_host, Host, Value};
use svm_text::parse_module;

// Same `DEMAND_SAME` fixture as `bytecode_demand_coroutine.rs`, driven through the single-vCPU debug
// engine. The parent spawns a demand coroutine into a 64 KiB carve at 64 KiB (op 4 = spawn_demand_
// coroutine), resumes it — the child's `load mem[0]` FAULTs at the child's (unmapped) page — supplies
// `123` at the fault address, and resumes again; the rewound load reads `123` and the child RETURNs.
// Result = 123 + FAULTED*1e6 + RETURNED*1e3.
const DEMAND_SAME: &str = r#"memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 1
  v2 = i64.const 65536
  v3 = i64.const 16
  v4 = i64.const 0
  v5 = cap.call 6 4 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = i64.const 0
  v7, v8 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v6)
  v9 = i32.const 123
  i32.store8 v8 v9
  v10 = i64.const 0
  v11, v12 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v10)
  v13 = i64.extend_i32_s v7
  v14 = i64.const 1000000
  v15 = i64.mul v13 v14
  v16 = i64.extend_i32_s v11
  v17 = i64.const 1000
  v18 = i64.mul v16 v17
  v19 = i64.add v12 v15
  v20 = i64.add v19 v18
  return v20
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  v2 = i32.load8_u v1
  v3 = i64.extend_i32_u v2
  return v3
  }
}
"#;

/// A fresh single-vCPU `DebugRun` on the demand-coroutine kernel carrying the `Instantiator` powerbox.
fn session() -> DebugRun {
    let m = parse_module(DEMAND_SAME).expect("parse");
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 128 << 10);
    DebugRun::new_with_host(&m, 0, &[Value::I32(inst)], host)
        .expect("bytecode debug engine must drive a demand coroutine")
}

/// The expected result, pinned against the tree-walker oracle: `123 + FAULTED(2)*1e6 + RETURNED(1)*1e3`.
fn oracle_result() -> Vec<Value> {
    let m = parse_module(DEMAND_SAME).expect("parse");
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 128 << 10);
    let mut fuel = 5_000_000u64;
    run_with_host(&m, 0, &[Value::I32(inst)], &mut fuel, &mut host).expect("tree-walker runs")
}

/// A per-op observation: the op clock, the call-stack `IrPc`s (which enter the coroutine body when the
/// debugger steps into it), and the window range `[0, 128 KiB)` — covering the child's carve at 64 KiB,
/// so the supplied byte `123` at the fault address is observed.
fn obs(run: &DebugRun) -> (u64, String, Vec<u8>) {
    let clock = run.op_clock();
    let mut frames = Vec::new();
    for d in 0..run.depth() {
        if let Some(pc) = run.frame_pc(d) {
            frames.push(format!("f{}b{}i{}", pc.func, pc.block, pc.inst));
        }
    }
    let win = run.read_window(0, 128 << 10).unwrap_or_default();
    (clock, frames.join(","), win)
}

/// Warm≡cold oracle for **§6b demand-coroutine checkpointing** (DEBUGGING.md W1): at every clock where a
/// checkpoint can be taken — including while the demand coroutine is suspended mid-fault, awaiting a
/// supplied page — a `DebugRun::restore`d run replays forward identically to the trusted from-0 run, to
/// the same final result. This exercises the coroutine's page-map round-trip (`prot_snapshot`/
/// `install_prot`) and the preserved `fault_yields`.
#[test]
fn demand_coroutine_checkpoint_snapshot_restore_round_trips() {
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
        "reference run matches the oracle"
    );

    // No fibers/regions, and the demand coroutine is now admitted, so *every* clock is checkpointable —
    // including the ones while the coroutine is live and mid-fault. Under the old subset (which excluded
    // `fault_yields`), those clocks returned `None`, so this count would be strictly smaller and the
    // mid-demand state would never round-trip.
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
                "forward replay diverged after restore at C={c}"
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
        "every clock is checkpointable — including while the demand coroutine is live/mid-fault, which \
         the old subset (excluding `fault_yields`) would have skipped"
    );
}
