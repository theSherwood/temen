//! **#1534 — a C guest on the LLVM on-ramp is a durable domain.**
//!
//! #1503 made the durable shadow arena a module-declared, verified parameter (`memory N shadow BASE
//! END`, INVARIANTS.md #16: no default placement — a module that declares none is not freezable).
//! Every hand-written and Forth-emitted module declares one; the translator did not, so
//! `transform_module*` failed closed with `NoShadowArena` for every C guest. `--shadow-arena
//! <contexts>` (`TranslateOptions::shadow_contexts`) reserves the arena as BSS above the other
//! low-window reserves and below the heap — allocator-neutral by construction, and clear of every
//! data segment, which is R9's "guest bytes never alias the arena" as a static check.
//!
//! The fixture (`fixtures/durable_probe.c`) is a loop that writes one line per iteration through the
//! powerbox `Stream` cap. Armed with `arm_freeze_after_backedges`, a run unwinds mid-loop into that
//! arena, freezes through `temen-snapshot`, restores into a **fresh** host and thaws to finish the
//! remaining lines: the two halves' stdout concatenated is the uninterrupted run's.

#![cfg(all(unix, target_arch = "x86_64"))]

use temen_durable::{
    arm_freeze_after_backedges, begin_thaw, init_durable_window, read_state, STATE_UNWINDING,
};
use temen_interp::{cap_id, run_capture_reserved_with_host, BoundImport, Host, StreamRole};
use temen_ir::durable_abi::{ShadowArena, DURABLE_CONTROL_END, MAX_SHADOW_CONTEXTS, SHADOW_STRIDE};
use temen_ir::Module;

/// The probe's five lines, the uninterrupted run's stdout.
const FULL_OUT: &str = "tick\ntick\ntick\ntick\ntick\n";

const PROBE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/durable_probe.ll"
);

/// A guest that *does* use the data stack (frames) and the float scratch (`printf`), so its window
/// already reserves the regions the arena has to sit above.
const PIPELINE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/pipeline.ll");

/// Translate `ll`, reserving `contexts` shadow regions (`None` = no arena, today's default).
fn translate(ll: &str, contexts: Option<u32>) -> Result<Module, temen_llvm::Error> {
    let opts = temen_llvm::TranslateOptions {
        shadow_contexts: contexts,
        ..Default::default()
    };
    let m = temen_llvm::translate_ll_path_with_options(ll, opts)?.module;
    temen_verify::verify_module(&m).expect("a translated module verifies");
    Ok(m)
}

/// The probe, which every durable check below runs.
fn translated(contexts: Option<u32>) -> Result<Module, temen_llvm::Error> {
    translate(PROBE, contexts)
}

fn arena_of(m: &Module) -> ShadowArena {
    m.memory
        .and_then(|x| x.shadow)
        .expect("the module declares a shadow arena")
}

/// The gap #1534 closes: without the flag the translator emits no arena — which is right for a
/// non-durable guest, and is exactly why the durable transform fails closed on it.
#[test]
fn without_the_flag_a_c_guest_is_not_freezable() {
    let m = translated(None).expect("translate");
    assert!(
        m.memory
            .expect("the probe declares a window")
            .shadow
            .is_none(),
        "a guest that asked for no arena must reserve none"
    );
    assert_eq!(
        temen_durable::transform_module_assume_confined(&m),
        Err(temen_durable::TransformError::NoShadowArena),
        "the transform must fail closed on a module that declares no arena (INVARIANTS.md #16)"
    );
}

/// With the flag the arena is where the verifier and every backend need it: above the always-live
/// control words, 8-aligned, `contexts` whole regions, inside the window — and clear of the guest's
/// own data (R9). `verify_module` in `translated()` has already rejected a violation of any of
/// these; the assertions name them so a regression says which one moved.
#[test]
fn the_flag_declares_a_verified_arena_clear_of_the_guest_image() {
    let m = translated(Some(4)).expect("translate");
    let a = arena_of(&m);
    let win = m.memory.expect("window").size();
    assert!(
        a.base >= DURABLE_CONTROL_END,
        "arena under the control words"
    );
    assert_eq!(a.base % 8, 0, "arena base must be 8-aligned");
    assert_eq!(a.contexts(), 4, "four regions were asked for");
    assert_eq!(a.size(), 4 * SHADOW_STRIDE);
    assert!(
        a.end <= win,
        "arena [{:#x}, {:#x}) outside the window {win:#x}",
        a.base,
        a.end
    );
    for (i, d) in m.data.iter().enumerate() {
        let end = d.offset + d.bytes.len() as u64;
        assert!(
            end <= a.base || d.offset >= a.end,
            "data segment {i} [{:#x}, {end:#x}) overlaps the arena [{:#x}, {:#x})",
            d.offset,
            a.base,
            a.end
        );
    }
    // …and not merely clear of them *byte*-wise: the arena is written during freeze/thaw while D40
    // protects read-only segments page-granularly, so sharing a page with the last read-only global
    // faults on the first shadow push (it did, before the base was page-aligned).
    assert_eq!(
        a.base % temen_ir::POWERBOX_STACK_PAGE,
        0,
        "the arena base must be host-page-aligned, not just 8-aligned"
    );
}

/// The arena sits on top of what the window already holds, not at a fixed offset — so a guest whose
/// window already reserves a data stack and float scratch pays **nothing** for it, and one that
/// reserves neither does not pay for a data stack it never uses.
#[test]
fn the_arena_is_placed_above_what_the_window_already_holds() {
    let bare = translate(PIPELINE, None).expect("translate");
    let durable = translate(PIPELINE, Some(8)).expect("translate");
    let (bare, durable) = (
        bare.memory.expect("window"),
        durable.memory.expect("window"),
    );
    assert_eq!(
        bare.size_log2, durable.size_log2,
        "a guest with frames + float scratch has room for the arena in the window it already had"
    );
    let a = durable.shadow.expect("arena");
    assert!(
        a.end <= durable.size(),
        "arena [{:#x}, {:#x}) inside the window {:#x}",
        a.base,
        a.end,
        durable.size()
    );
    // The frameless probe is the other side of the rule: its window grows to hold the arena, but
    // only by the arena — not by the 1 MiB data-stack reserve it never uses.
    let probe = translated(Some(4))
        .expect("translate")
        .memory
        .expect("window");
    let a = probe.shadow.expect("arena");
    assert!(
        probe.size() < a.base + 4 * temen_ir::POWERBOX_STACK_RESERVE,
        "a frameless guest must not be sized as if it reserved a data stack (window {:#x}, arena at {:#x})",
        probe.size(),
        a.base
    );
}

/// The durable transform accepts the translated guest on the **strict** path — the one for an
/// untrusted module — and the instrumented module verifies. The probe passes its string constant by
/// address but never loads or stores through it, so R9's "no guest memory ops" rule holds; a C guest
/// that does touch memory (most of them) needs `transform_module_assume_confined`, the cooperating-
/// toolchain path the Forth kernel takes (#1236). Both are checked here, since the arena is what
/// either one needs.
#[test]
fn the_durable_transform_accepts_a_translated_c_guest() {
    let m = translated(Some(4)).expect("translate");
    for (path, t) in [
        ("strict", temen_durable::transform_module(&m)),
        (
            "confined",
            temen_durable::transform_module_assume_confined(&m),
        ),
    ] {
        let inst = t.unwrap_or_else(|e| panic!("the {path} transform accepts the guest: {e:?}"));
        assert_eq!(
            inst.funcs.len(),
            m.funcs.len(),
            "{path}: the transform instruments in place; it must not add or drop functions"
        );
        temen_verify::verify_module(&inst)
            .unwrap_or_else(|e| panic!("the {path}-instrumented guest verifies: {e:?}"));
    }
}

/// The count is the module's, so it is range-checked where it is declared rather than deferred to
/// the verifier's `ShadowArenaInvalid`.
#[test]
fn the_context_count_is_range_checked() {
    for n in [0, MAX_SHADOW_CONTEXTS as u32 + 1] {
        assert!(
            matches!(translated(Some(n)), Err(temen_llvm::Error::Unsupported(_))),
            "{n} contexts must be refused at translate time"
        );
    }
    translated(Some(MAX_SHADOW_CONTEXTS as u32)).expect("the cap itself is accepted");
}

/// The engines a durable translated guest runs on. `jit_cap_run` is the Cranelift tier; the
/// bytecode engine's scope is pinned separately below.
#[derive(Clone, Copy, Debug)]
enum Engine {
    TreeWalk,
    Jit,
}

const JIT_TABLE_LOG2: u8 = 10;

/// Run the guest's `_start` over `window` on `engine`, returning the captured window.
fn run_on(engine: Engine, inst: &Module, window: &[u8], size_log2: u8, host: &mut Host) -> Vec<u8> {
    match engine {
        Engine::TreeWalk => {
            let mut fuel = 5_000_000_000u64;
            let (r, snap) =
                run_capture_reserved_with_host(inst, 0, &[], &mut fuel, window, size_log2, host);
            r.expect("the guest runs to a return");
            snap
        }
        Engine::Jit => {
            let (out, snap) =
                temen_run::jit_cap_run(inst, 0, &[], window, size_log2, JIT_TABLE_LOG2, host)
                    .expect("the guest compiles and runs on the JIT");
            assert!(
                matches!(out, temen_jit::JitOutcome::Returned(_)),
                "JIT run: {out:?}"
            );
            snap
        }
    }
}

/// A durable host with the probe's one import (`write`) bound to a fresh stdout.
fn durable_host() -> (Host, i32) {
    let mut h = Host::new();
    h.set_durable(true);
    let out = h.grant_stream(StreamRole::Out);
    h.set_import_bindings(vec![BoundImport::required(cap_id::STREAM, 1, out)]);
    (h, out)
}

/// The whole point: a translated C guest **freezes mid-loop and thaws to finish**. The run unwinds
/// into its declared arena with the loop counter live in a shadow frame; the artifact restores into a
/// host that never ran the guest, and the thawed half prints exactly the lines the frozen half did
/// not. The artifact is engine-neutral (DURABILITY.md §4 "JIT parity"), so the legs freeze and thaw
/// on the tree-walker, on the JIT, and across them.
fn freeze_and_thaw(freeze_on: Engine, thaw_on: Engine) {
    let m = translated(Some(4)).expect("translate");
    let inst = temen_durable::transform_module(&m).expect("transform");
    temen_verify::verify_module(&inst).expect("instrumented verifies");
    let arena = arena_of(&m);
    let size_log2 = m.memory.expect("window").size_log2;

    // Freeze. Three branch terminators per iteration, so the 7th promotes the state word partway
    // through the third: two lines are out, the loop counter is live, and the next poll unwinds.
    let (mut fh, out) = durable_host();
    let mut win = init_durable_window(1usize << size_log2, arena);
    arm_freeze_after_backedges(&mut win, 7);
    let snap = run_on(freeze_on, &inst, &win, size_log2, &mut fh);
    assert_eq!(
        read_state(&snap),
        STATE_UNWINDING,
        "the armed run on {freeze_on:?} must have unwound (frozen), not run to completion"
    );
    let first = String::from_utf8(fh.stdout_bytes()).expect("utf8");
    assert!(
        FULL_OUT.starts_with(&first) && !first.is_empty() && first != FULL_OUT,
        "the frozen half is a proper prefix of the uninterrupted output, got {first:?}"
    );
    let artifact = temen_snapshot::freeze(&inst, &snap, &fh).expect("freeze admits the guest");

    // Thaw into a host that never ran the guest: the artifact's handle table re-pins stdout at the
    // same slot, and the rewind picks the loop up where the freeze left it.
    let mut th = Host::new();
    th.set_durable(true);
    let mut window = temen_snapshot::restore(&artifact, &inst, &mut th).expect("restore");
    th.set_import_bindings(vec![BoundImport::required(cap_id::STREAM, 1, out)]);
    begin_thaw(&mut window, arena, 0);
    let second = {
        run_on(thaw_on, &inst, &window, size_log2, &mut th);
        String::from_utf8(th.stdout_bytes()).expect("utf8")
    };
    assert_eq!(
        format!("{first}{second}"),
        FULL_OUT,
        "frozen half {first:?} ({freeze_on:?}) + thawed half {second:?} ({thaw_on:?}) must equal the uninterrupted run"
    );
}

#[test]
fn a_translated_c_guest_freezes_mid_loop_and_thaws() {
    freeze_and_thaw(Engine::TreeWalk, Engine::TreeWalk);
}

#[test]
fn a_guest_frozen_on_the_tree_walker_thaws_on_the_jit() {
    freeze_and_thaw(Engine::TreeWalk, Engine::Jit);
}

/// Why there is no *freeze*-on-the-JIT leg: the back-edge **countdown** is the interpreter's
/// deterministic test oracle (its window is private and synchronous), while the JIT's trigger for
/// the same poll is the async `FreezeController` — so an armed window runs to completion on the JIT
/// with the arm never promoting. That is the tree's existing split (`temen/tests/durable_backedge_jit.rs`
/// freezes on the interpreter and thaws on the JIT, as the leg above does), not something this
/// guest's arena changes. If the JIT ever ticks the countdown, this pin fails and the
/// `Jit → Jit` leg becomes free.
#[test]
fn the_backedge_countdown_oracle_is_interpreter_only() {
    let m = translated(Some(4)).expect("translate");
    let inst = temen_durable::transform_module(&m).expect("transform");
    let arena = arena_of(&m);
    let size_log2 = m.memory.expect("window").size_log2;
    let (mut h, _) = durable_host();
    let mut win = init_durable_window(1usize << size_log2, arena);
    arm_freeze_after_backedges(&mut win, 7);
    let snap = run_on(Engine::Jit, &inst, &win, size_log2, &mut h);
    assert_eq!(
        read_state(&snap),
        temen_durable::STATE_ARMED,
        "the JIT does not tick the deterministic countdown; the arm stays pending"
    );
    assert_eq!(
        String::from_utf8(h.stdout_bytes()).expect("utf8"),
        FULL_OUT,
        "so the run finishes normally"
    );
}
