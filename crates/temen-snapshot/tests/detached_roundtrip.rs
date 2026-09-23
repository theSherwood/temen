//! #1361 step 4 — a live detached child rides its parent's artifact (Section 8, v29) as its own
//! root-shaped artifact behind the spawner-held launch record, and restore hands it back as a
//! [`ThawedDetached`] in its own powerbox, all-or-nothing.
//!
//! The child here is frozen as a root (freeze-from-start) and attached to a parent by hand, since a
//! durable parent cannot yet spawn detached outside `temen-interp`'s own tests (the run-time capture
//! and re-launch are pinned there, `src/detached_freeze_tests.rs`). What this pins is the codec: every
//! field survives, the child's own residue restores into the child's powerbox and not the parent's,
//! a re-freeze of the restored tree is byte-identical, and a missing module grant refuses the restore.

use std::sync::{Arc, Mutex};
use temen_durable::{init_durable_window, transform_module, write_state, STATE_UNWINDING};
use temen_interp::{
    module_digest, run_capture_reserved_with_host, CapturedDetached, CapturedProt, DetachedLaunch,
    Host, MemLayout, ThawedDetached, Value,
};
use temen_ir::durable_abi::ShadowArena;
use temen_ir::Module;
use temen_snapshot::{freeze_with_prots, restore_with_prots, FreezeError, PageProt, RestoreError};

const ARENA: ShadowArena = ShadowArena {
    base: 16448,
    end: 65536,
};

fn instrument(src: &str) -> Module {
    let m = temen_text::parse_module(src).expect("parse");
    let inst = transform_module(&m).expect("transform");
    temen_verify::verify_module(&inst).expect("verifies");
    inst
}

fn parent() -> Module {
    instrument(
        "memory 17 shadow 16448 65536
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 7
  return v1
  }
}
",
    )
}

/// A looping child, so its freeze leaves a real continuation in its window.
fn child() -> Module {
    instrument(
        "memory 17 shadow 16448 65536
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  v2 = i64.const 0
  br 1(v1, v2)
}
block 1 (v3: i64, v4: i64) {
  v5 = i64.const 100
  v6 = i64.lt_s v3 v5
  br_if v6 2(v3, v4) 3(v4)
}
block 2 (v7: i64, v8: i64) {
  v9 = i64.add v8 v7
  v10 = i64.const 1
  v11 = i64.add v7 v10
  br 1(v11, v9)
}
block 3 (v12: i64) {
  return v12
  }
}
",
    )
}

/// Freeze `m` from the start as a root over a 128 KiB window: its unwound image and its powerbox.
fn frozen_root(m: &Module) -> (MemLayout, Host) {
    let mut host = Host::new();
    host.set_durable(true);
    let inst = host.grant_instantiator(0, 1 << 17);
    host.grant_address_space(0, 1 << 17);
    let mut win = init_durable_window(1 << 17, ARENA);
    write_state(&mut win, STATE_UNWINDING);
    let mut fuel = 1_000_000;
    let (r, snap) = run_capture_reserved_with_host(
        m,
        0,
        &[Value::I64(inst as i64)],
        &mut fuel,
        &win,
        17,
        &mut host,
    );
    assert!(r.is_ok(), "froze: {r:?}");
    let pages = snap.len() / 4096;
    let layout = MemLayout::from_dense(snap, &vec![CapturedProt::Rw; pages], 1 << 17);
    (layout, host)
}

fn launch(child: &Module) -> DetachedLaunch {
    DetachedLaunch {
        task: 3,
        entry: 0,
        digest: module_digest(child),
        fuel: 12_345,
        lane: 2,
        channel: -1,
        max_vcpus: 4,
        same_module: false,
        names: vec![("fs".to_string(), 0x0103)],
    }
}

/// The parent's frozen window and a powerbox that granted `child` durable and captured it live.
fn parent_with_child(c: &Module) -> (Vec<u8>, Host) {
    let (window, chost) = frozen_root(c);
    let mut host = Host::new();
    host.set_durable(true);
    host.grant_durable_module(c);
    host.set_captured_detached(vec![CapturedDetached {
        parent_task: 0,
        slot: 1,
        window,
        reserved_log2: 17,
        host: Arc::new(Mutex::new(chost)),
        module: Arc::new(c.clone()),
        launch: launch(c),
    }]);
    let mut win = init_durable_window(1 << 17, ARENA);
    write_state(&mut win, STATE_UNWINDING);
    (win, host)
}

fn freeze(p: &Module, win: &[u8], host: &Host) -> Result<Vec<u8>, FreezeError> {
    freeze_with_prots(p, win, &vec![PageProt::Rw; win.len() / 4096], 17, host)
}

/// The restored child re-captured, so the tree can be frozen again from what restore produced.
fn recaptured(t: ThawedDetached, c: &Module) -> CapturedDetached {
    CapturedDetached {
        parent_task: t.parent_task,
        slot: t.slot,
        window: t.window,
        reserved_log2: t.reserved_log2,
        host: Arc::new(Mutex::new(t.host)),
        module: Arc::new(c.clone()),
        launch: t.launch,
    }
}

#[test]
fn a_live_detached_child_round_trips_through_its_parents_artifact() {
    let (p, c) = (parent(), child());
    let (win, host) = parent_with_child(&c);
    let art = freeze(&p, &win, &host).expect("freeze the tree");

    let mut rhost = Host::new();
    rhost.set_durable(true);
    rhost.grant_durable_module(&c); // the restoring host re-grants the child's program (D-scope)
    let (rwin, _, reserved) = restore_with_prots(&art, &p, &mut rhost).expect("restore");
    assert_eq!(
        rwin, win,
        "the parent's own window is unchanged by carrying a child"
    );
    assert_eq!(reserved, 17);

    let thawed = rhost.take_thawed_detached();
    assert_eq!(thawed.len(), 1, "one child came back");
    let t = &thawed[0];
    assert_eq!((t.parent_task, t.slot), (0, 1));
    assert_eq!(t.launch, launch(&c), "the launch record survives exactly");
    assert_eq!(t.reserved_log2, 17);
    let (orig_window, orig_host) = frozen_root(&c);
    assert_eq!(
        t.window.bytes(),
        orig_window.bytes(),
        "the child's image survives"
    );
    assert_eq!(
        t.host.capture_durable_handles(),
        orig_host.capture_durable_handles(),
        "the child's handle table restored into the child's powerbox"
    );
    assert_eq!(
        rhost.capture_durable_handles(),
        host.capture_durable_handles(),
        "and the parent's table is its own (just the module grant), not the child's"
    );

    // §12.6 canonicality: freezing the restored tree reproduces the artifact byte for byte.
    let mut again = Host::new();
    again.set_durable(true);
    again.grant_durable_module(&c);
    again.set_captured_detached(thawed.into_iter().map(|t| recaptured(t, &c)).collect());
    assert_eq!(freeze(&p, &win, &again).expect("re-freeze"), art);
}

/// Restore is all-or-nothing across the tree: a restoring host that no longer grants the child's
/// module refuses the whole artifact, with the same answer a handle would get.
#[test]
fn a_child_whose_module_is_not_re_granted_refuses_the_restore() {
    let (p, c) = (parent(), child());
    let (win, host) = parent_with_child(&c);
    let art = freeze(&p, &win, &host).expect("freeze the tree");
    let mut rhost = Host::new();
    rhost.set_durable(true);
    assert_eq!(
        restore_with_prots(&art, &p, &mut rhost).map(|_| ()),
        Err(RestoreError::ModuleUnresolved(module_digest(&c)))
    );
}
