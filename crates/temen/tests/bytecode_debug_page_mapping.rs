//! Reverse-debug **checkpointing of a page-mapping root window** (debugging slice 6a). Before this
//! slice the checkpoint ladder fell back to replay-from-clock-0 whenever the root window's layout was
//! non-pristine — i.e. the guest had `map`ped (grown a reserved-tail page), `unmap`ped, or `protect`ed
//! (`Ro`) any page through its `AddressSpace` (cap 5). The single-vCPU `DebugRun` (and the multi-vCPU
//! scheduler, same `self.mem`) now capture the window's **page-protection map alongside its bytes**
//! ([`Mem::layout_snapshot`]/`restore_layout`), so such a run is bounded-reverse like any other. Only
//! §13 region aliasing (cross-domain shared bytes) still forces the fallback.
//!
//! The fixture drives all three non-pristine transitions and then, as its terminal op, loads the
//! `unmap`ped page — a `MemoryFault` that ends the run. The warm≡cold round-trip therefore proves both
//! halves of fidelity at once: the **bytes** round-trip (`read_window` over the whole touched range,
//! grown tail included, matches at every step), and the **protection map** round-trips (forward replay
//! after restore re-executes the load of the grown page — which would fault if its `Rw` entry were lost
//! — and the terminal load of the unmapped page — which would *not* fault, diverging, if its `Unmapped`
//! entry were lost).

use temen_interp::bytecode::{self, DebugRun, ScheduledDebugRun};
use temen_interp::{run_with_host, Host, Value};
use temen_text::parse_module;

// A `memory 17` (128 KiB mapped, 1 TiB reserved) root granted an `AddressSpace` over `[0, 1 MiB)`, so it
// can manage pages inside — including growing a page just past the 128 KiB prefix. All spans are 16 KiB
// (a whole multiple of every host page size), so each op lands on clean page boundaries on any host.
//
// The entry `(i32 as_handle) -> (i64)`:
//   - writes marker 11 at offset 0                 (a plain prefix page, stays `Rw`)
//   - map(131072, 16384, RW) then writes 22 there  (grows a reserved-tail page → `Rw` beyond `mapped`)
//   - writes 33 at 16384 then protect(16384, RO)   (a readable-but-not-writable `Ro` page group)
//   - writes 44 at 114688 then unmap(114688)       (an `Unmapped` page group — zeroed on decommit)
//   - loads offsets 0, 131072, 16384               (force the grown/Ro pages to be re-read on replay)
//   - loads offset 114688 → **MemoryFault**        (the terminal trap: the unmapped page)
const PAGE_MAPPING: &str = r#"memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 0
  v2 = i32.const 11
  i32.store8 v1 v2
  v3 = i64.const 131072
  v4 = i64.const 16384
  v5 = i64.const 3
  v6 = call.cap 5 0 (i64, i64, i64) -> (i64) v0 (v3, v4, v5)
  v7 = i32.const 22
  i32.store8 v3 v7
  v8 = i64.const 16384
  v9 = i32.const 33
  i32.store8 v8 v9
  v10 = i64.const 16384
  v11 = i64.const 1
  v12 = call.cap 5 2 (i64, i64, i64) -> (i64) v0 (v8, v10, v11)
  v13 = i64.const 114688
  v14 = i32.const 44
  i32.store8 v13 v14
  v15 = i64.const 16384
  v16 = call.cap 5 1 (i64, i64) -> (i64) v0 (v13, v15)
  v17 = i32.load8_u v1
  v18 = i32.load8_u v3
  v19 = i32.load8_u v8
  v20 = i32.load8_u v13
  v21 = i64.extend_i32_u v20
  return v21
  }
}
"#;

const GRANT: u64 = 1 << 20; // the AddressSpace sub-range: [0, 1 MiB)

/// A fresh single-vCPU `DebugRun` on the page-mapping kernel carrying the `AddressSpace` powerbox. The
/// grant is deterministic (a fresh `Host` mints the same handle), so every session is bit-identical.
fn session() -> DebugRun {
    let m = parse_module(PAGE_MAPPING).expect("parse");
    let mut host = Host::new();
    let h = host.grant_address_space(0, GRANT);
    DebugRun::new_with_host(&m, 0, &[Value::I32(h)], host)
        .expect("bytecode debug engine must drive a page-mapping root")
}

/// A per-op observation: the op clock, the call-stack `IrPc`s, and the whole touched window range
/// `[0, 160 KiB)` (covering the grown tail page at 128 KiB and the zeroed unmapped range). `read_window`
/// is a pure byte view (it bounds- but not protection-checks), so it reads uncommitted pages as zero.
fn obs(run: &DebugRun) -> (u64, String, Vec<u8>) {
    let clock = run.op_clock();
    let mut frames = Vec::new();
    for d in 0..run.depth() {
        if let Some(pc) = run.frame_pc(d) {
            frames.push(format!("f{}b{}i{}", pc.func, pc.block, pc.inst));
        }
    }
    let win = run.read_window(0, 160 << 10).unwrap_or_default();
    (clock, frames.join(","), win)
}

/// The page-mapping fixture reaches its terminal `MemoryFault` (the load of the unmapped page) on the
/// production bytecode engine **and** the tree-walker oracle — a shared reference point for the round-trip.
#[test]
fn page_mapping_fixture_faults_on_both_engines() {
    let m = parse_module(PAGE_MAPPING).expect("parse");

    let mut h_bc = Host::new();
    let bc_h = h_bc.grant_address_space(0, GRANT);
    let mut f_bc = 5_000_000u64;
    let bc = bytecode::compile_and_run_with_host(&m, 0, &[Value::I32(bc_h)], &mut f_bc, &mut h_bc)
        .expect("bytecode engine compiles the kernel");
    assert!(
        bc.is_err(),
        "bytecode: terminal load of the unmapped page traps"
    );

    let mut h_tw = Host::new();
    let tw_h = h_tw.grant_address_space(0, GRANT);
    let mut f_tw = 5_000_000u64;
    let tw = run_with_host(&m, 0, &[Value::I32(tw_h)], &mut f_tw, &mut h_tw);
    assert!(
        tw.is_err(),
        "tree-walker: terminal load of the unmapped page traps"
    );
}

/// Warm≡cold oracle for **§6a page-mapping-root checkpointing** (DEBUGGING.md W1): at every clock where
/// a checkpoint can be taken — including after the window has grown a page, protected one `Ro`, and
/// unmapped another — a `DebugRun::restore`d run replays forward **identically** to the trusted from-0
/// run, all the way to the shared terminal `MemoryFault`. This exercises the full
/// `layout_snapshot`/`restore_layout` round-trip: the committed byte range (grown tail included) and the
/// page-protection map.
#[test]
fn page_mapping_root_checkpoint_snapshot_restore_round_trips() {
    const FUEL: u64 = 5_000_000;

    let mut refr = session();
    let mut f = FUEL;
    let mut ref_obs = vec![obs(&refr)];
    while refr.tick(&mut f) {
        ref_obs.push(obs(&refr));
    }
    let total = ref_obs.len() - 1;
    assert!(
        matches!(refr.result(), Some(Err(_))),
        "the fixture ends in a MemoryFault (terminal load of the unmapped page)"
    );

    // The run has no regions/fibers/coroutines, so *every* clock is checkpointable — including the ones
    // after the window grew/protected/unmapped pages. That count (`total + 1`, asserted below) is the
    // adversarial teeth: under the old bytes-only `snapshot_safe`, those post-mapping clocks returned
    // `None` (non-pristine layout), so this would have taken strictly fewer checkpoints and never
    // round-tripped through the grown/`Ro`/`Unmapped` state.
    let mut checkpoints = 0usize;
    for c in 0..=total {
        let mut at_c = session();
        let mut f = FUEL;
        while at_c.op_clock() < c as u64 && at_c.tick(&mut f) {}
        let Some(snap) = at_c.snapshot() else {
            continue; // outside the checkpointable subset
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
                "forward replay diverged after restore at C={c} (step {i})"
            );
        }
        assert_eq!(i, total, "warm run from C={c} reached the same terminal op");
        assert!(
            matches!(warm.result(), Some(Err(_))),
            "warm run from C={c} reached the same MemoryFault"
        );
    }
    assert_eq!(
        checkpoints,
        total + 1,
        "every clock is checkpointable — including those after the window turned non-pristine \
         (grew/protected/unmapped pages), which the bytes-only predicate would have excluded"
    );
}

// The same page-mapping root, but preceded by a `thread.spawn`/`join` of two workers — so the run is
// driven by the multi-vCPU `ScheduledDebugRun` (all tasks share the one `self.mem`). The workers each
// `atomic.rmw.add 1` into mem[0] (determinate under the cooperative debug scheduler) and join before the
// root grows/protects/unmaps disjoint pages and takes the terminal `MemoryFault`. This exercises the
// scheduled engine's `layout_snapshot`/`restore_layout` of the shared window.
const THREADED_PAGE_MAPPING: &str = r#"memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  vsp = i64.const 0
  va = i64.const 1
  vh0 = thread.spawn 1 vsp va
  vh1 = thread.spawn 1 vsp va
  vj0 = thread.join vh0
  vj1 = thread.join vh1
  vg = i64.const 131072
  vgl = i64.const 16384
  vrw = i64.const 3
  vgr = call.cap 5 0 (i64, i64, i64) -> (i64) v0 (vg, vgl, vrw)
  vm22 = i32.const 22
  i32.store8 vg vm22
  vro = i64.const 16384
  vm33 = i32.const 33
  i32.store8 vro vm33
  vrol = i64.const 16384
  vrop = i64.const 1
  vpr = call.cap 5 2 (i64, i64, i64) -> (i64) v0 (vro, vrol, vrop)
  vuo = i64.const 114688
  vm44 = i32.const 44
  i32.store8 vuo vm44
  vul = i64.const 16384
  vur = call.cap 5 1 (i64, i64) -> (i64) v0 (vuo, vul)
  vl1 = i32.load8_u vg
  vl2 = i32.load8_u vro
  vl3 = i32.load8_u vuo
  vrv = i64.extend_i32_u vl3
  return vrv
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vaddr = i64.const 0
  vrmw = i64.atomic.rmw.add vaddr varg
  vz = i64.const 0
  return vz
  }
}
"#;

/// A fresh scheduled run on the threaded page-mapping kernel carrying the `AddressSpace` powerbox.
fn sched_session() -> ScheduledDebugRun {
    let m = parse_module(THREADED_PAGE_MAPPING).expect("parse");
    let mut host = Host::new();
    let h = host.grant_address_space(0, GRANT);
    ScheduledDebugRun::new_with_host(&m, 0, &[Value::I32(h)], host)
        .expect("scheduled debug engine must drive a threaded page-mapping root")
}

/// A per-turn observation: the global turn, every live task's call stack, and the whole touched window
/// range (shared across tasks; carries the workers' mem[0] counter and the root's grown/`Ro` pages).
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
    let win = run.read_window(0, 160 << 10).unwrap_or_default();
    (turn, stacks.join(";"), win)
}

/// Warm≡cold oracle for **§6a page-mapping-root checkpointing on the multi-vCPU engine**: at every
/// checkpointable turn — including after the shared window has grown/protected/unmapped pages — a
/// `ScheduledDebugRun::restore`d run replays forward identically to the from-0 run, to the shared
/// terminal `MemoryFault`. Proves the scheduled `snapshot`/`restore` carry the window's protection map
/// alongside its bytes.
#[test]
fn scheduled_page_mapping_checkpoint_snapshot_restore_round_trips() {
    const FUEL: u64 = 5_000_000;

    let mut refr = sched_session();
    let mut f = FUEL;
    let mut ref_obs = vec![sched_obs(&mut refr)];
    while refr.tick(&mut f) {
        ref_obs.push(sched_obs(&mut refr));
    }
    let total = ref_obs.len() - 1;
    assert!(
        matches!(refr.result(), Some(Err(_))),
        "the fixture ends in a MemoryFault (terminal load of the unmapped page)"
    );
    assert!(
        ref_obs.iter().any(|(_, s, _)| s.contains(';')),
        "two worker vCPUs are live at some point (the run is genuinely scheduled)"
    );

    let mut checkpoints = 0usize;
    let mut non_pristine_checkpoint = false;
    for c in 0..=total {
        let mut at_c = sched_session();
        let mut f = FUEL;
        while at_c.op_turn() < c as u64 && at_c.tick(&mut f) {}
        let Some(snap) = at_c.snapshot() else {
            continue;
        };
        checkpoints += 1;
        // A checkpoint within a few ops of the terminal fault is necessarily post-mapping (non-pristine).
        if c + 4 >= total {
            non_pristine_checkpoint = true;
        }

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
        assert_eq!(i, total, "warm run from C={c} reached the same terminal op");
        assert!(
            matches!(warm.result(), Some(Err(_))),
            "warm run from C={c} reached the same fault"
        );
    }
    assert!(checkpoints > 0, "at least one checkpoint was taken");
    assert!(
        non_pristine_checkpoint,
        "a checkpoint was taken (and restored) after the shared window grew/protected/unmapped pages"
    );
}
