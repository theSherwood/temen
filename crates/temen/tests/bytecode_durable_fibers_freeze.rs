//! Multi-fiber durability, commit 2 (INTERP_PERF.md; DURABILITY.md §12.8): the bytecode engine's
//! **freeze driver** + **thaw seeding** for a durable run with a live (parked) fiber, checked
//! byte-for-byte against the tree-walker oracle and the `temen-snapshot` §12 codec — the bytecode mirror
//! of the JIT's `durable_fibers_jit.rs` cross-backend checks.
//!
//! Freeze/thaw is IR-driven (the transform spills/rebuilds each continuation into its in-window shadow
//! region), but the *idle fibers* a freeze finds parked are not on any native stack — so the runtime
//! must drive each one under `UNWINDING` to flatten it into its own region (the **freeze driver**), and
//! re-create it from the artifact residue on thaw (**thaw seeding**). These checks freeze the same
//! instrumented fiber module on both backends and require:
//!   1. a NORMAL durable run agrees with the tree-walker;
//!   2. an UNWINDING freeze produces a **byte-identical** window snapshot *and* §12 artifact (the
//!      flattened fiber's region bytes + its `FrozenFiber` residue match the tree-walker);
//!   3. restore + re-freeze is byte-identical (the §12.6 canonical invariant);
//!   4. thawing the bytecode artifact (REWINDING, fibers re-seeded) reproduces the tree-walker's thaw
//!      result and ends NORMAL.

/// The arena every durable test module declares: the pre-#1503 fixed placement `[guard+64, 1<<16)`.
const TEST_ARENA: temen_ir::durable_abi::ShadowArena = temen_ir::durable_abi::ShadowArena {
    base: 16448,
    end: 65536,
};
use temen_durable::{
    begin_thaw, init_durable_window, read_state, transform_module, write_state, STATE_NORMAL,
    STATE_UNWINDING,
};
use temen_interp::{bytecode, run_capture_reserved_with_host, FrozenFiber, Host, Trap, Value};
use temen_snapshot::{freeze, restore};
use temen_text::parse_module;
use temen_verify::verify_module;

const SIZE_LOG2: u8 = 17; // 128 KiB ≥ the durable reserve
const WINDOW: usize = 1 << SIZE_LOG2;

fn window_with(state: i32) -> Vec<u8> {
    let mut w = init_durable_window(WINDOW, TEST_ARENA);
    write_state(&mut w, state);
    w
}

/// A durable host seeded with any `frozen` fibers a thaw must re-create.
fn durable_host(frozen: Vec<FrozenFiber>) -> Host {
    let mut h = Host::new();
    h.set_durable(true);
    h.set_frozen_fibers(frozen);
    h
}

/// Run the transformed entry on the **tree-walker** over `window`, with `frozen` fibers re-seeded.
fn tw_run(
    inst: &temen_ir::Module,
    window: &[u8],
    frozen: Vec<FrozenFiber>,
) -> (Result<Vec<Value>, Trap>, Vec<u8>, Host) {
    let mut h = durable_host(frozen);
    let mut fuel = 1_000_000u64;
    let (r, win) =
        run_capture_reserved_with_host(inst, 0, &[], &mut fuel, window, SIZE_LOG2, &mut h);
    (r, win, h)
}

/// Run the transformed entry on the **bytecode engine** over `window`, with `frozen` fibers re-seeded.
fn bc_run(
    inst: &temen_ir::Module,
    window: &[u8],
    frozen: Vec<FrozenFiber>,
) -> (Result<Vec<Value>, Trap>, Vec<u8>, Host) {
    let mut h = durable_host(frozen);
    let mut fuel = 1_000_000u64;
    let (r, win) = bytecode::compile_and_run_capture_reserved_with_host(
        inst,
        0,
        &[],
        &mut fuel,
        window,
        SIZE_LOG2,
        &mut h,
    )
    .expect("bytecode engine must drive a single-vCPU multi-fiber durable module (commit 2)");
    (r, win, h)
}

fn check(src: &str) {
    let m = parse_module(src).expect("parse");
    let inst = transform_module(&m).expect("module must be in transform scope");
    verify_module(&inst).expect("instrumented IR verifies");

    // (1) NORMAL: the bytecode durable run agrees with the tree-walker (the baseline result).
    let (base_tw, _, _) = tw_run(&inst, &window_with(STATE_NORMAL), vec![]);
    let (base_bc, _, _) = bc_run(&inst, &window_with(STATE_NORMAL), vec![]);
    let base = base_tw.expect("baseline is trap-free");
    assert_eq!(
        base_bc,
        Ok(base.clone()),
        "NORMAL durable run: tree-walker != bytecode\n{src}"
    );

    // (2) freeze: UNWINDING flattens the root *and* the parked fiber; the snapshot + artifact match.
    let (fr_tw, snap_tw, host_tw) = tw_run(&inst, &window_with(STATE_UNWINDING), vec![]);
    let (fr_bc, snap_bc, host_bc) = bc_run(&inst, &window_with(STATE_UNWINDING), vec![]);
    assert!(
        fr_tw.is_ok() && fr_bc.is_ok(),
        "freeze returns a placeholder\n{src}"
    );
    assert_eq!(
        read_state(&snap_bc),
        STATE_UNWINDING,
        "bytecode leaves the artifact UNWINDING\n{src}"
    );
    assert!(
        !host_bc.frozen_fibers().is_empty(),
        "the freeze driver flattened at least one parked fiber\n{src}"
    );
    assert_eq!(
        host_tw.frozen_fibers(),
        host_bc.frozen_fibers(),
        "frozen-fiber residue: tree-walker != bytecode\n{src}"
    );
    assert_eq!(
        snap_tw, snap_bc,
        "freeze snapshot (incl. each fiber's flattened region): tree-walker != bytecode\n{src}"
    );

    let art_tw = freeze(&inst, &snap_tw, &host_tw).expect("tree-walker freeze serializes");
    let art_bc = freeze(&inst, &snap_bc, &host_bc).expect("bytecode freeze serializes");
    assert_eq!(
        art_tw, art_bc,
        "§12 artifact (incl. fiber residue): tree-walker != bytecode\n{src}"
    );

    // (3) restore + re-freeze is byte-identical (the §12.6 canonical invariant).
    let mut rhost = Host::new();
    let rwin = restore(&art_bc, &inst, &mut rhost).expect("artifact restores");
    assert!(
        !rhost.frozen_fibers().is_empty(),
        "restore re-seeds the frozen-fiber residue\n{src}"
    );
    assert_eq!(
        freeze(&inst, &rwin, &rhost).expect("re-freeze"),
        art_bc,
        "re-serialize of a restored domain is byte-identical\n{src}"
    );

    // (4) thaw: re-seed the frozen fibers and rewind on both backends; the results must agree and
    // end NORMAL. The frozen point's results are reloaded, so the run reproduces the baseline.
    let thaw_fibers = rhost.frozen_fibers().to_vec();
    let mut thaw_win = rwin;
    begin_thaw(&mut thaw_win, TEST_ARENA, 0);
    let (thaw_tw, final_tw, _) = tw_run(&inst, &thaw_win, thaw_fibers.clone());
    let (thaw_bc, final_bc, _) = bc_run(&inst, &thaw_win, thaw_fibers);
    assert_eq!(
        thaw_bc, thaw_tw,
        "thaw result: tree-walker != bytecode\n{src}"
    );
    assert_eq!(
        thaw_bc,
        Ok(base),
        "bytecode thaw must reproduce the uninterrupted result\n{src}"
    );
    assert_eq!(
        read_state(&final_tw),
        STATE_NORMAL,
        "tree-walker thaw flips the state word back to NORMAL\n{src}"
    );
    assert_eq!(
        read_state(&final_bc),
        STATE_NORMAL,
        "bytecode thaw flips the state word back to NORMAL\n{src}"
    );
}

/// A pure-arithmetic fiber module (no caps ⇒ deterministic), shared with the JIT cross-backend tests:
/// the root resumes a fiber that suspends once (yielding 42) and then returns 7 + 100. Freezing under
/// `UNWINDING` parks the fiber after its suspend, so the freeze driver must flatten it into its region.
/// `base = 107`.
const FIBER_SRC: &str = "memory 17 shadow 16448 65536\n\
    func () -> (i64) {\n\
    block 0 () {\n\
    \x20 v0 = ref.func 1\n\
    \x20 v1 = i64.const 4096\n\
    \x20 v2 = cont.new v0 v1\n\
    \x20 v3 = i64.const 0\n\
    \x20 v4, v5 = cont.resume v2 v3\n\
    \x20 v6 = i64.const 7\n\
    \x20 v7, v8 = cont.resume v2 v6\n\
    \x20 return v8\n\
      }\n\
    }\n\
    func (i64, i64) -> (i64) {\n\
    block 0 (v0: i64, v1: i64) {\n\
    \x20 v2 = i64.const 42\n\
    \x20 v3 = suspend v2\n\
    \x20 v4 = i64.const 100\n\
    \x20 v5 = i64.add v3 v4\n\
    \x20 return v5\n\
      }\n\
    }\n";

#[test]
fn parked_fiber_freeze_thaw_round_trip() {
    check(FIBER_SRC);
}

/// Two fibers parked concurrently at freeze (each in its own shadow region), exercising the freeze
/// driver's ascending-slot flatten over more than one fiber and the dense thaw re-seed. Each fiber
/// suspends once (yielding 42) then returns its resume arg + 100; the root resumes A then B to
/// completion. `base = (7 + 100) + (8 + 100)`.
const TWO_FIBERS_SRC: &str = "memory 17 shadow 16448 65536\n\
    func () -> (i64) {\n\
    block 0 () {\n\
    \x20 v0 = ref.func 1\n\
    \x20 v1 = i64.const 4096\n\
    \x20 v2 = cont.new v0 v1\n\
    \x20 v3 = i64.const 0\n\
    \x20 v4, v5 = cont.resume v2 v3\n\
    \x20 v6 = i64.const 8192\n\
    \x20 v7 = cont.new v0 v6\n\
    \x20 v8 = i64.const 0\n\
    \x20 v9, v10 = cont.resume v7 v8\n\
    \x20 v11 = i64.const 7\n\
    \x20 v12, v13 = cont.resume v2 v11\n\
    \x20 v14 = i64.const 8\n\
    \x20 v15, v16 = cont.resume v7 v14\n\
    \x20 v17 = i64.add v13 v16\n\
    \x20 return v17\n\
      }\n\
    }\n\
    func (i64, i64) -> (i64) {\n\
    block 0 (v0: i64, v1: i64) {\n\
    \x20 v2 = i64.const 42\n\
    \x20 v3 = suspend v2\n\
    \x20 v4 = i64.const 100\n\
    \x20 v5 = i64.add v3 v4\n\
    \x20 return v5\n\
      }\n\
    }\n";

#[test]
fn two_fibers_freeze_thaw_round_trip() {
    check(TWO_FIBERS_SRC);
}

/// #1538 on the bytecode engine — a fiber whose park was already **consumed** (resume #1 took its `1`
/// under `NORMAL` and ran on) is flattened idle when the freeze lands at the root's `call.cap` (the
/// host proc flips the state word to `UNWINDING`). Thawed on either engine, the root reloads the
/// call's result and runs *forward* into resume #2, whose plain claim must deliver `10` at the fiber's
/// rewound `suspend` — `(1, 11)`, not the stale re-park `(1, 1)`. The bytecode residue records the
/// park as consumed and matches the tree-walker's.
#[test]
fn a_consumed_park_redelivers_after_thaw_on_the_bytecode_engine() {
    const SRC: &str = "memory 17 shadow 16448 65536\n\
        func (i32) -> (i64, i64) {\n\
        block 0 (v0: i32) {\n\
        \x20 v1 = ref.func 1\n\
        \x20 v2 = i64.const 4096\n\
        \x20 v3 = cont.new v1 v2\n\
        \x20 v5 = i64.const 0\n\
        \x20 v6, v7 = cont.resume v3 v5\n\
        \x20 v8 = call.cap 13 0 () -> (i64) v0 ()\n\
        \x20 v9 = i64.const 10\n\
        \x20 v10, v11 = cont.resume v3 v9\n\
        \x20 return v7 v11\n\
          }\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block 0 (v0: i64, v1: i64) {\n\
        \x20 v2 = i64.const 1\n\
        \x20 v3 = suspend v2\n\
        \x20 v4 = i64.add v3 v2\n\
        \x20 v5 = suspend v4\n\
        \x20 v6 = i64.const 0\n\
        \x20 return v6\n\
          }\n\
        }\n";
    let m = parse_module(SRC).expect("parse");
    let inst = transform_module(&m).expect("transform");
    verify_module(&inst).expect("verify");

    // A durable host whose host proc writes `UNWINDING` when `freeze`, seeded with `frozen`.
    let host = |freeze: bool, frozen: Vec<FrozenFiber>| -> (Host, i32) {
        let mut h = durable_host(frozen);
        let hf = h.grant_host_proc(Box::new(move |_op, _args, mem, _| {
            if freeze {
                if let Some(m) = mem {
                    m.write_bytes(temen_durable::STATE_OFF, &STATE_UNWINDING.to_le_bytes())
                        .expect("the state word is in the window");
                }
            }
            Ok(vec![0])
        }));
        (h, hf)
    };
    let want = Ok(vec![Value::I64(1), Value::I64(11)]);

    // Freeze on both engines: identical residue, the park recorded as consumed.
    let (mut hb, hfb) = host(true, vec![]);
    let mut fuel = 1_000_000u64;
    let (rb, snap_bc) = bytecode::compile_and_run_capture_reserved_with_host(
        &inst,
        0,
        &[Value::I32(hfb)],
        &mut fuel,
        &window_with(STATE_NORMAL),
        SIZE_LOG2,
        &mut hb,
    )
    .expect("bytecode drives the fiber module");
    assert!(rb.is_ok(), "bytecode freeze returns a placeholder: {rb:?}");
    let (mut ht, hft) = host(true, vec![]);
    let mut fuel = 1_000_000u64;
    let (rt, _snap_tw) = run_capture_reserved_with_host(
        &inst,
        0,
        &[Value::I32(hft)],
        &mut fuel,
        &window_with(STATE_NORMAL),
        SIZE_LOG2,
        &mut ht,
    );
    assert!(
        rt.is_ok(),
        "tree-walker freeze returns a placeholder: {rt:?}"
    );
    assert_eq!(
        hb.frozen_fibers(),
        ht.frozen_fibers(),
        "residue: bytecode != tree-walker"
    );
    assert!(
        hb.frozen_fibers()[0].consumed,
        "the idle fiber's park (taken by resume #1) is recorded as consumed"
    );
    let frozen = hb.frozen_fibers().to_vec();

    // Thaw the bytecode snapshot on both engines: the forward resume #2 delivers.
    let mut thaw_win = snap_bc;
    begin_thaw(&mut thaw_win, TEST_ARENA, 0);
    let (mut h1, hf1) = host(false, frozen.clone());
    let mut fuel = 1_000_000u64;
    let (r_bc, _) = bytecode::compile_and_run_capture_reserved_with_host(
        &inst,
        0,
        &[Value::I32(hf1)],
        &mut fuel,
        &thaw_win,
        SIZE_LOG2,
        &mut h1,
    )
    .expect("bytecode drives the thaw");
    assert_eq!(
        r_bc, want,
        "bytecode thaw delivers 10 into the consumed park"
    );
    let (mut h2, hf2) = host(false, frozen);
    let mut fuel = 1_000_000u64;
    let (r_tw, _) = run_capture_reserved_with_host(
        &inst,
        0,
        &[Value::I32(hf2)],
        &mut fuel,
        &thaw_win,
        SIZE_LOG2,
        &mut h2,
    );
    assert_eq!(
        r_tw, want,
        "tree-walker thaw of the bytecode snapshot delivers too"
    );
}
