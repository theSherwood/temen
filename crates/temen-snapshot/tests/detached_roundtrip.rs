//! #1361 step 4 — a live detached child rides its parent's artifact (Section 8, v29) as its own
//! root-shaped artifact behind the spawner-held launch record, and restore hands it back as a
//! [`ThawedDetached`] in its own powerbox, all-or-nothing.
//!
//! The child here is frozen as a root (freeze-from-start) and attached to a parent by hand, so the codec
//! is pinned apart from the run-time capture and re-launch (`temen-interp`'s
//! `src/detached_freeze_tests.rs`, and `temen`'s `durable_detached_jit.rs`). What this pins: every
//! field survives, the child's own residue restores into the child's powerbox and not the parent's,
//! a re-freeze of the restored tree is byte-identical, and a missing module grant refuses the restore.

use std::sync::{Arc, Mutex};
use temen_durable::{init_durable_window, transform_module, write_state, STATE_UNWINDING};
use temen_interp::{
    detached_launch, module_digest, run_capture_reserved_with_host, CapturedDetached, CapturedProt,
    DetachedLaunch, DurableBinding, Host, Lease, LiveDetached, MemLayout, Residue, StreamRole,
    Value, WINDOW_RUNS,
};
use temen_ir::durable_abi::ShadowArena;
use temen_ir::Module;
use temen_snapshot::{freeze_with_prots, restore_with_prots, FreezeError, PageProt, RestoreError};

const ARENA: ShadowArena = ShadowArena::new(16448, 65536);

/// A detached child's lease on `budget`, for a fresh window of its 128 KiB.
fn lease(budget: i32) -> Lease {
    Lease {
        budget,
        bytes: 1 << 17,
        runs: WINDOW_RUNS,
    }
}

fn instrument(src: &str) -> Module {
    let m = temen_text::parse_module(src).expect("parse");
    let inst = transform_module(&m).expect("transform");
    temen_verify::verify_module(&inst).expect("verifies");
    inst
}

/// Returns 7; on a path it never takes, it spawns a detached child (#3034: the cuts below carry
/// detached children, which only a module with an `Instantiator` call makes).
fn parent() -> Module {
    instrument(
        "memory 17 shadow 16448 65536
func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i64.const 0
  vc = i64.eq v0 vz
  br_if vc 1() 2(v0)
}
block 1 () {
  v1 = i64.const 7
  return v1
}
block 2 (vi: i64) {
  vh = i32.wrap_i64 vi
  vrp = i64.const 0
  vs = call.cap 6 17 (i64) -> (i32) vh (vrp)
  vs64 = i64.extend_i32_u vs
  return vs64
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
    let inst = host.grant_instantiator();
    host.grant_address_space(0, 1 << 17);
    let mut win = init_durable_window(1 << 17, ARENA);
    write_state(&mut win, ARENA, STATE_UNWINDING);
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
        digest: module_digest(child),
        lane: 2,
        names: vec![("fs".to_string(), 0x0103)],
        pager: None,
        joinable: true,
    }
}

/// The parent's frozen window and a powerbox that granted `child` durable and captured it live.
fn parent_with_child(c: &Module) -> (Vec<u8>, Host) {
    parent_with(c, frozen_root(c), launch(c))
}

/// [`parent_with_child`] over a given frozen child and launch record.
fn parent_with(
    c: &Module,
    (window, chost): (MemLayout, Host),
    launch: DetachedLaunch,
) -> (Vec<u8>, Host) {
    let mut host = Host::new();
    host.set_durable(true);
    host.grant_durable_module(c);
    // #2043 — its child table: slot 0's child spent, slot 1 the captured one's.
    children(&mut host, vec![1, 1]);
    host.set_captured_detached(vec![CapturedDetached {
        parent_task: 0,
        slot: 1,
        window,
        reserved_log2: 17,
        host: Arc::new(Mutex::new(chost)),
        module: Arc::new(c.clone()),
        launch,
    }]);
    let mut win = init_durable_window(1 << 17, ARENA);
    write_state(&mut win, ARENA, STATE_UNWINDING);
    (win, host)
}

fn freeze(p: &Module, win: &[u8], host: &Host) -> Result<Vec<u8>, FreezeError> {
    freeze_with_prots(p, win, &vec![PageProt::Rw; win.len() / 4096], 17, host)
}

/// The restored child re-captured, so the tree can be frozen again from what restore produced.
fn recaptured(t: LiveDetached, c: &Module) -> CapturedDetached {
    let launch = detached_launch(&t);
    CapturedDetached {
        parent_task: t.parent_task,
        slot: t.slot,
        window: t.payload.window,
        reserved_log2: t.payload.reserved_log2,
        host: Arc::new(Mutex::new(t.payload.host)),
        module: Arc::new(c.clone()),
        launch,
    }
}

/// #2043 — the parent's child table, as a hand-assembled cut gives it (#3034: through the thaw pass).
fn children(host: &mut Host, generations: Vec<u32>) {
    host.admit_thaw(
        &parent(),
        Residue {
            children: generations,
            ..Residue::default()
        },
        temen_interp::Serve::default(),
    )
    .expect("a child table a run makes");
}

/// The live detached children a restore left on `host`, and its child table.
fn thawed(host: &mut Host) -> (Vec<LiveDetached>, Vec<u32>) {
    let r = host.take_residue();
    (r.live_detached, r.children)
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

    let (thawed, generations) = thawed(&mut rhost);
    assert_eq!(thawed.len(), 1, "one child came back");
    let t = &thawed[0];
    assert_eq!((t.parent_task, t.slot), (0, 1));
    assert_eq!(
        detached_launch(t),
        launch(&c),
        "the launch record survives exactly"
    );
    assert_eq!(t.payload.reserved_log2, 17);
    let (orig_window, orig_host) = frozen_root(&c);
    assert_eq!(
        t.payload.window.bytes(),
        orig_window.bytes(),
        "the child's image survives"
    );
    assert_eq!(
        t.payload.host.capture_durable_handles(),
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
    children(&mut again, generations);
    again.set_captured_detached(thawed.into_iter().map(|t| recaptured(t, &c)).collect());
    assert_eq!(freeze(&p, &win, &again).expect("re-freeze"), art);
}

/// #2054 — a stream the spawner re-granted its child (§7c) rides the child's artifact as inherited,
/// so the child's thaw can alias it to its spawner's stream again. Restored and not yet re-launched,
/// it is still inherited, so a re-freeze of the restored tree is byte-identical.
#[test]
fn a_childs_inherited_stdout_rides_its_artifact_as_inherited() {
    let (p, c) = (parent(), child());
    let mut spawner = Host::new();
    spawner.set_durable(true);
    let out = spawner.grant_stream(StreamRole::Out);
    let (chost, _, _) = spawner
        .spawn_detached_child(&[("stdout".to_string(), out)], 1 << 17, lease(-1), 2, false)
        .expect("the child's powerbox");
    let inherited = DurableBinding::Stream {
        role: StreamRole::Out,
        inherited: true,
    };
    let holds = |h: &Host| {
        h.capture_durable_handles()
            .expect("durable")
            .iter()
            .any(|d| d.binding == inherited)
    };
    assert!(holds(&chost), "the child's stdout is its spawner's");
    let (window, _) = frozen_root(&c);
    let (win, host) = parent_with(&c, (window, chost), launch(&c));
    let art = freeze(&p, &win, &host).expect("freeze the tree");

    let mut rhost = Host::new();
    rhost.set_durable(true);
    rhost.grant_durable_module(&c);
    restore_with_prots(&art, &p, &mut rhost).expect("restore");
    let (thawed, generations) = thawed(&mut rhost);
    assert!(
        holds(&thawed[0].payload.host),
        "restored, the stream is still inherited"
    );

    let mut again = Host::new();
    again.set_durable(true);
    again.grant_durable_module(&c);
    children(&mut again, generations);
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

/// A parent declaring a `{ page: (i64) -> (i64) }` export, the one shape a pager takes (#744).
fn pager_parent() -> Module {
    instrument(
        "memory 17 shadow 16448 65536
type 0 func (i64) -> (i64)
type 1 interface { page: 0 }
export 0 interface \"pager\" 1 { page: 1 }
func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i64.const 0
  vc = i64.eq v0 vz
  br_if vc 1() 2(v0)
}
block 1 () {
  v1 = i64.const 7
  return v1
}
block 2 (vi: i64) {
  vh = i32.wrap_i64 vi
  vrp = i64.const 0
  vs = call.cap 6 17 (i64) -> (i32) vh (vrp)
  vs64 = i64.extend_i32_u vs
  return vs64
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  return v0
  }
}
",
    )
}

/// #2438 — a demand-paged child's pager rides its launch record (v47), so the thaw re-links its faults
/// to it; and it must name a pager export of the spawner's program, as op 17 requires, or the restore
/// is refused.
#[test]
fn a_paged_childs_pager_rides_its_launch_record() {
    let (p, c) = (pager_parent(), child());
    let paged = |pager| {
        let (win, host) = parent_with(
            &c,
            frozen_root(&c),
            DetachedLaunch {
                pager: Some(pager),
                ..launch(&c)
            },
        );
        freeze(&p, &win, &host).expect("freeze the tree")
    };
    let restore = |art: &[u8]| {
        let mut rhost = Host::new();
        rhost.set_durable(true);
        rhost.grant_durable_module(&c);
        restore_with_prots(art, &p, &mut rhost).map(|_| thawed(&mut rhost).0)
    };
    let thawed = restore(&paged(0)).expect("restore");
    assert_eq!(thawed[0].payload.pager, Some(0), "the pager survives");
    assert_eq!(
        restore(&paged(1)).map(|_| ()),
        Err(RestoreError::Malformed),
        "export 1 is no pager"
    );
}

/// #2465 — whether the spawner still holds its join claim rides the launch record (v51): a child it
/// detached (op 10) thaws unjoinable. The claim is one byte, and any other value is malformed.
#[test]
fn a_childs_join_claim_rides_its_launch_record() {
    let (p, c) = (parent(), child());
    let art = |joinable| {
        let (win, host) = parent_with(
            &c,
            frozen_root(&c),
            DetachedLaunch {
                joinable,
                ..launch(&c)
            },
        );
        freeze(&p, &win, &host).expect("freeze the tree")
    };
    let restore = |art: &[u8]| {
        let mut rhost = Host::new();
        rhost.set_durable(true);
        rhost.grant_durable_module(&c);
        restore_with_prots(art, &p, &mut rhost).map(|_| thawed(&mut rhost).0)
    };
    for joinable in [true, false] {
        let thawed = restore(&art(joinable)).expect("restore");
        assert_eq!(thawed[0].joinable, joinable, "the claim survives");
    }
    let (claimed, detached) = (art(true), art(false));
    let at: Vec<usize> = (0..claimed.len())
        .filter(|&i| claimed[i] != detached[i])
        .collect();
    assert_eq!(
        (claimed.len(), at.len()),
        (detached.len(), 1),
        "one byte holds the claim"
    );
    let mut bad = claimed;
    bad[at[0]] = 2;
    assert_eq!(
        restore(&bad).map(|_| ()),
        Err(RestoreError::Malformed),
        "a claim byte other than 0 or 1"
    );
}

/// A child's own restore failing fails the parent's (#1698): never swallowed into a child that thaws
/// short of what it froze with. Here the child holds a named `vm_fs`, so its restore needs the
/// registrar the restoring host lends it.
#[test]
fn a_child_whose_restore_fails_refuses_the_parents() {
    let (p, c) = (parent(), child());
    let (window, mut chost) = frozen_root(&c);
    temen_fs::grant_vm_fs(&mut chost, None);
    let (win, host) = parent_with(&c, (window, chost), launch(&c));
    let art = freeze(&p, &win, &host).expect("freeze the tree");

    let restoring = || {
        let mut h = Host::new();
        h.set_durable(true);
        h.grant_durable_module(&c);
        h
    };
    assert_eq!(
        restore_with_prots(&art, &p, &mut restoring()).map(|_| ()),
        Err(RestoreError::NamedCapRefused("vm_fs".to_string())),
        "no registrar serves the child's capability"
    );
    let mut rhost = restoring();
    rhost.set_named_cap_registrar(Box::new(temen_fs::regrant_vm_fs));
    restore_with_prots(&art, &p, &mut rhost)
        .expect("a registrar serving `vm_fs` restores the tree");
    assert_eq!(thawed(&mut rhost).0.len(), 1);
}

/// #1944 slice 2: a detached child's `"budget"` is its spawner's node, so parent and child share a
/// tree. The parent's artifact numbers the whole cut's nodes — its own and those only the child names
/// (here a node the child split) — and carries them once; the thaw rebuilds them once, so a charge
/// the thawed child makes reaches the thawed parent's node, and a re-freeze is byte-identical.
#[test]
fn a_budget_node_the_parent_and_child_share_is_one_node_after_a_thaw() {
    use temen_interp::cap_id;
    let read = |h: &mut Host, b: i32| h.cap_call(cap_id::BUDGET, 1, b, &[1], 1, None);
    let (p, c) = (parent(), child());
    let (window, mut chost) = frozen_root(&c);
    let mut host = Host::new();
    host.set_durable(true);
    host.grant_durable_module(&c);
    let root = host.grant_budget(-1, 1 << 20, -1);
    let paid = host
        .cap_call(cap_id::BUDGET, 0, root, &[-1, 1 << 18, -1], 1, None)
        .unwrap()[0] as i32;
    assert!(
        host.budget_mem_take(paid, 1 << 17),
        "the child's window, charged"
    );
    host.give_child_budget(lease(paid), &mut chost);
    let cb = chost
        .resolve_cap_name("budget")
        .expect("the child's budget");
    let mine = chost
        .cap_call(cap_id::BUDGET, 0, cb, &[-1, 1 << 16, -1], 1, None)
        .unwrap()[0] as i32;
    assert!(mine >= 0, "the child splits a node of its own");
    children(&mut host, vec![1, 1]);
    host.set_captured_detached(vec![CapturedDetached {
        parent_task: 0,
        slot: 1,
        window,
        reserved_log2: 17,
        host: Arc::new(Mutex::new(chost)),
        module: Arc::new(c.clone()),
        launch: launch(&c),
    }]);
    let mut win = init_durable_window(1 << 17, ARENA);
    write_state(&mut win, ARENA, STATE_UNWINDING);
    let art = freeze(&p, &win, &host).expect("freeze the tree");

    let mut rhost = Host::new();
    rhost.set_durable(true);
    rhost.grant_durable_module(&c);
    restore_with_prots(&art, &p, &mut rhost).expect("restore");
    let (mut thawed, generations) = thawed(&mut rhost);
    let t = &mut thawed[0].payload;
    assert_eq!(
        read(&mut rhost, paid),
        Ok(vec![1 << 17]),
        "the window's charge stayed"
    );
    assert_eq!(
        read(&mut t.host, cb),
        Ok(vec![1 << 17]),
        "the child's budget is that node"
    );
    assert!(
        t.host.budget_mem_take(mine, 1 << 16),
        "the child charges its own node"
    );
    assert_eq!(
        read(&mut rhost, paid),
        Ok(vec![1 << 16]),
        "and the charge reaches the parent's node"
    );
    assert_eq!(
        read(&mut rhost, root),
        Ok(vec![(1 << 20) - (1 << 17) - (1 << 16)])
    );
    t.host.budget_mem_give(mine, 1 << 16);

    // §12.6 canonicality: the restored tree re-freezes to the same bytes.
    children(&mut rhost, generations);
    rhost.set_captured_detached(thawed.into_iter().map(|t| recaptured(t, &c)).collect());
    assert_eq!(freeze(&p, &win, &rhost).expect("re-freeze"), art);
}

/// #2941 — a nested domain's registrar refusing, after the root's has answered, leaves the restoring
/// host as it found it: the root's grant not installed, its registrar back in place (the nested
/// powerbox borrowed it, `Lent`). The same host then restores the tree whose child asks for nothing
/// the registrar refuses, and the restored tree re-freezes byte-identically.
#[test]
fn a_nested_registrars_refusal_leaves_the_host_and_its_registrar_as_they_were() {
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    let (p, c) = (parent(), child());
    let tree = |child_fs: bool| {
        let (window, mut chost) = frozen_root(&c);
        if child_fs {
            temen_fs::grant_vm_fs(&mut chost, None);
        }
        let (win, mut host) = parent_with(&c, (window, chost), launch(&c));
        let fs = host.grant_host_proc(temen_interp::HostProcGrant::new(
            temen_interp::ProcHandler::Sync(Box::new(|_op, _args, out, _mem, _| {
                temen_interp::put_results(out, &[0])
            })),
            temen_interp::CapState::Stateless,
        ));
        host.register_cap_name("fs", fs);
        (freeze(&p, &win, &host).expect("freeze the tree"), win)
    };
    let (bad, win) = tree(true);
    let (good, _) = tree(false);
    let asked = std::sync::Arc::new(AtomicUsize::new(0));
    let restoring = || {
        let mut h = Host::new();
        h.set_durable(true);
        h.grant_durable_module(&c);
        let a = std::sync::Arc::clone(&asked);
        h.set_named_cap_registrar(Box::new(move |name, _state, _payer| {
            a.fetch_add(1, SeqCst);
            (name == "fs").then(|| temen_interp::NamedCapGrant {
                grant: temen_interp::HostProcGrant::new(
                    temen_interp::ProcHandler::Sync(Box::new(|_op, _args, out, _mem, _| {
                        temen_interp::put_results(out, &[1])
                    })),
                    temen_interp::CapState::Stateless,
                ),
                process: None,
            })
        }));
        h
    };
    let fresh = restoring();
    let mut h = restoring();
    assert_eq!(
        restore_with_prots(&bad, &p, &mut h).map(|_| ()),
        Err(RestoreError::NamedCapRefused("vm_fs".to_string()))
    );
    assert_eq!(
        asked.load(SeqCst),
        2,
        "the root's `fs`, then the child's `vm_fs`"
    );
    assert_eq!(h.restore_footprint(), fresh.restore_footprint());
    assert!(h.residue().is_empty());
    assert_eq!(
        freeze(&p, &win, &h).unwrap(),
        freeze(&p, &win, &fresh).unwrap()
    );
    let (rwin, rprots, _) =
        restore_with_prots(&good, &p, &mut h).expect("the same host, its registrar back");
    let (live, generations) = thawed(&mut h);
    let recapt: Vec<_> = live.into_iter().map(|t| recaptured(t, &c)).collect();
    h.admit_thaw(
        &p,
        Residue {
            children: generations,
            ..Residue::default()
        },
        h.serve(),
    )
    .expect("the child table the restore admitted");
    h.set_captured_detached(recapt);
    assert_eq!(
        freeze_with_prots(&p, &rwin, &rprots, 17, &h).expect("re-freeze"),
        good,
        "the restored tree re-freezes byte-identically"
    );
}

/// #2941 — the canonical section order holds in a nested artifact too: a child's artifact with two of
/// its sections swapped refuses its parent's restore. The identity control re-encodes it unchanged.
#[test]
fn a_nested_artifacts_sections_are_in_the_one_order_too() {
    let (p, c) = (parent(), child());
    let (win, host) = parent_with_child(&c);
    let art = freeze(&p, &win, &host).expect("freeze the tree");
    let restores = |a: &[u8]| {
        let mut h = Host::new();
        h.set_durable(true);
        h.grant_durable_module(&c);
        restore_with_prots(a, &p, &mut h).map(|_| ())
    };
    let bent = |swap: bool| {
        let mut edit = temen_snapshot::decode_residue(&art).expect("decode");
        let child = &mut edit.live_detached[0].payload.artifact;
        *child = swap_first_two_sections(child, swap);
        temen_snapshot::encode_residue(&art, &edit).expect("encode")
    };
    assert_eq!(restores(&bent(false)), Ok(()), "the identity control");
    assert_eq!(restores(&bent(true)), Err(RestoreError::Malformed));
}

/// `art` with its first two sections swapped (`swap`), or re-encoded as it is.
fn swap_first_two_sections(art: &[u8], swap: bool) -> Vec<u8> {
    fn uleb(b: &[u8], at: &mut usize) -> u64 {
        let (mut v, mut shift) = (0u64, 0);
        loop {
            let byte = b[*at];
            *at += 1;
            v |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return v;
            }
            shift += 7;
        }
    }
    fn put(out: &mut Vec<u8>, mut v: u64) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            out.push(if v == 0 { b } else { b | 0x80 });
            if v == 0 {
                return;
            }
        }
    }
    let (hdr, payload) = temen_encode::wire::read_header(art).expect("a header");
    let mut secs = Vec::new();
    let mut at = 0;
    while at < payload.len() {
        let tag = uleb(payload, &mut at);
        let len = uleb(payload, &mut at) as usize;
        secs.push((tag, payload[at..at + len].to_vec()));
        at += len;
    }
    if swap {
        secs.swap(0, 1);
    }
    let mut out = Vec::new();
    temen_encode::wire::write_header(&mut out, hdr.kind, hdr.version, hdr.flags);
    for (tag, body) in secs {
        put(&mut out, tag);
        put(&mut out, body.len() as u64);
        out.extend_from_slice(&body);
    }
    out
}
