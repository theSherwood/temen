//! Reverse-debug **checkpointing of a page-mapping `instantiate` child** on the multi-vCPU scheduler
//! (debugging slice 6b). An `instantiate` child holds its own attenuated `AddressSpace`, so it can
//! `map`/`unmap`/`protect` its own confined window — making that child's `nested_view` layout
//! non-pristine. Before this slice the checkpoint ladder excluded such a run; now each child env's own
//! page-protection map rides in its `EnvSnapshot` ([`Mem::prot_snapshot`]/`install_prot`), reinstated on
//! restore with its bytes carried by the shared window snapshot.
//!
//! The child unmaps a page and then, as its terminal op, loads it — a `MemoryFault`. As in the root
//! page-mapping oracle, that makes the child's `Unmapped` protection **observable** through forward
//! replay: a restore that dropped the entry would read zero instead of faulting, diverging.

use temen_interp::bytecode::ScheduledDebugRun;
use temen_interp::{Host, Value};
use temen_text::parse_module;

// Root `(i32 instantiator) -> (i64)`: instantiate the child (func 1) into a 64 KiB carve at 64 KiB with a
// two-arg entry (so it receives both an attenuated Instantiator and an AddressSpace), then join it.
// Child `(i64 inst, i64 as) -> (i64)`: write a marker at offset 0, `unmap` the 16 KiB page group at
// child-offset 16 KiB via its AddressSpace (cap 5 op 1), then load it → terminal MemoryFault.
const CHILD_PAGE_MAPPING: &str = r#"memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  ; spawn via record (op 17): entry=1 off=65536 sl=16 quota=0
  q0v0 = i64.const 4294967296
  q0v1 = i64.const 65536
  q0v2 = i64.const -4294967280
  q0v3 = i64.const 4294967295
  q0v4 = i64.const 0
  q0a0 = i64.const 1152
  i64.store q0a0 q0v0
  q0a1 = i64.const 1160
  i64.store q0a1 q0v1
  q0a2 = i64.const 1168
  i64.store q0a2 q0v2
  q0a3 = i64.const 1176
  i64.store q0a3 q0v3
  q0a4 = i64.const 1184
  i64.store q0a4 q0v4
  q0a5 = i64.const 1192
  i64.store q0a5 q0v4
  q0a6 = i64.const 1200
  i64.store q0a6 q0v4
  vch = call.cap 6 17 (i64) -> (i32) v0 (q0a0)
  vr = call.cap 6 1 (i32) -> (i64) v0 (vch)
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  va = i64.const 0
  vm = i32.const 11
  i32.store8 va vm
  vuo = i64.const 16384
  vul = i64.const 16384
  vur = call.cap 5 1 (i64, i64) -> (i64) v1 (vuo, vul)
  vlu = i32.load8_u vuo
  vrv = i64.extend_i32_u vlu
  return vrv
  }
}
"#;

fn sched_session() -> ScheduledDebugRun {
    let m = parse_module(CHILD_PAGE_MAPPING).expect("parse");
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 128 << 10);
    ScheduledDebugRun::new_with_host(&m, 0, &[Value::I32(inst)], host)
        .expect("scheduled debug engine must drive a page-mapping instantiate child")
}

/// A per-turn observation: the global turn, every live task's call stack (frames as
/// `m{module}f{func}b{block}i{inst}`), and the window range `[0, 128 KiB)` (covering the child carve at
/// 64 KiB, so the child's marker and unmapped-zeroed page are observed).
fn sched_obs(run: &mut ScheduledDebugRun) -> (u64, String, Vec<u8>) {
    let turn = run.op_turn();
    let mut stacks = Vec::new();
    for tid in run.threads() {
        run.select_task(tid);
        let mut frames = Vec::new();
        for d in 0..run.depth() {
            if let Some(pc) = run.frame_pc(d) {
                frames.push(format!(
                    "m{}f{}b{}i{}",
                    pc.module, pc.func, pc.block, pc.inst
                ));
            }
        }
        stacks.push(format!("{tid}=[{}]", frames.join(",")));
    }
    let win = run.read_window(0, 128 << 10).unwrap_or_default();
    (turn, stacks.join(";"), win)
}

/// Warm≡cold oracle for **§6b page-mapping-`instantiate`-child checkpointing**: at every checkpointable
/// turn — including after the child has unmapped a page — a `ScheduledDebugRun::restore`d run replays
/// forward identically to the from-0 run. The forward replay re-executes the child's terminal load of the
/// unmapped page (which would *not* fault, diverging, if the `Unmapped` entry were dropped from the
/// child's `EnvSnapshot`), so it pins the child env's protection-map fidelity.
#[test]
fn scheduled_child_page_mapping_checkpoint_snapshot_restore_round_trips() {
    const FUEL: u64 = 5_000_000;

    let mut refr = sched_session();
    let mut f = FUEL;
    let mut ref_obs = vec![sched_obs(&mut refr)];
    while refr.tick(&mut f) {
        ref_obs.push(sched_obs(&mut refr));
    }
    let total = ref_obs.len() - 1;
    assert!(
        ref_obs.iter().any(|(_, s, _)| s.contains("m0f1")),
        "the child (func 1) runs as its own scheduled vCPU"
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
            sched_obs(&mut warm),
            ref_obs[i],
            "restore at C={c} lands at the reference state"
        );
        let mut f = FUEL;
        while warm.tick(&mut f) {
            i += 1;
            assert_eq!(
                sched_obs(&mut warm),
                ref_obs[i],
                "forward replay diverged after restore at C={c}"
            );
        }
        assert_eq!(i, total, "warm run from C={c} reached the same terminal");
    }
    assert_eq!(
        checkpoints,
        total + 1,
        "every turn is checkpointable — including those after the child unmapped a page (non-pristine \
         env layout), which the old bytes-only env predicate would have excluded"
    );
}
