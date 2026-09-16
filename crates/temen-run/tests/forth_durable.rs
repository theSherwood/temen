//! #1236 — **freeze/thaw a live Forth session.**
//!
//! Step 1: *is the Forth kernel a durable domain?* Yes, on both halves. Its **code** is durable-ready: the durable transform accepts all 77 functions
//! through `transform_module_assume_confined`, and the instrumented module verifies. Its **memory map**
//! declares where the durable runtime may keep its per-context shadow regions — `memory 20 shadow
//! 475136 524288`, i.e. `[0x74000, 0x80000)`, inside the `sandbox` spawn-scratch page — and the verifier
//! holds that declaration to the geometry every backend assumes, including that no data segment
//! overlaps it (the R9 "guest bytes never alias the arena" contract, checked statically).
//!
//! History: before #1503 the arena was a substrate constant under `0x10000`, and the kernel's 24 data
//! segments at `0x8000` sat inside it — a live collision once a session had a root plus a few `task`
//! fibers. The fix was never to move the data: placement is the guest's (INVARIANTS.md #16), so the
//! kernel now says where its arena goes and the collision cannot exist. `transform_module` (the strict
//! path for an *untrusted* module) still fails closed with `GuestUsesMemory` because the kernel does
//! guest loads/stores at all; a cooperating toolchain's module — this one — uses `assume_confined`.
//!
//! Steps 2–3 (below the step-1 checks): the kernel's *words* are durable too — under a durable `Jit`
//! grant every submitted unit is instrumented before verify (the confined durable validator, since a
//! REPL line-unit touches the REPL stack), the R8 install fence admits the suspending fiber words —
//! and a live session freezes mid-transcript with a `task` parked at its `yield`, serializes through
//! `temen-snapshot`, restores into a fresh host, and thaws to the uninterrupted output, on the
//! tree-walker, on the JIT, and across them.
#![cfg(all(unix, target_arch = "x86_64"))]

fn kernel() -> temen_ir::Module {
    let m = temen_text::parse_module(include_str!("../demos/forth/forth.temt"))
        .expect("forth.temt parses");
    temen_verify::verify_module(&m).expect("forth.temt verifies");
    m
}

/// The kernel's **code** is durable-ready: every function passes the durable transform's shape rules
/// through the confined path, and the instrumented module verifies. This is the half of #1236 that
/// needed no work — if it ever regresses, the memory-map work below is pointless.
#[test]
fn the_durable_transform_accepts_the_whole_kernel() {
    let m = kernel();
    let t = temen_durable::transform_module_assume_confined(&m)
        .expect("the durable transform accepts the kernel through the confined path");
    assert_eq!(
        t.funcs.len(),
        m.funcs.len(),
        "the transform instruments in place; it must not add or drop functions"
    );
    temen_verify::verify_module(&t).expect("the instrumented kernel verifies");
}

/// …and the **strict** path refuses it, because the kernel does guest memory ops that could alias
/// the reserved region. That refusal is correct and is what `assume_confined` exists to let a
/// cooperating guest opt out of — once it genuinely reserves the region (see the module docs).
#[test]
fn the_strict_transform_refuses_the_kernel_for_aliasing_the_durable_reserve() {
    let m = kernel();
    assert_eq!(
        temen_durable::transform_module(&m),
        Err(temen_durable::TransformError::GuestUsesMemory),
        "the strict path must fail closed for a guest whose memory ops could alias the durable control words or its arena"
    );
}

/// The kernel **declares** its shadow arena, the verifier accepts it, and it is clear of every data
/// segment — the precondition #1236's freeze/thaw needs, now a property of the module rather than a
/// gap to work around. It also holds enough contexts for a REPL with a root and a handful of `task`
/// fibers.
#[test]
fn the_kernel_declares_an_arena_clear_of_its_data() {
    use temen_ir::durable_abi::{ShadowArena, SHADOW_STRIDE};
    let m = kernel(); // `kernel()` already ran the verifier, which rejects a data/arena overlap
    let arena = m
        .memory
        .and_then(|x| x.shadow)
        .expect("forth.temt declares a shadow arena");
    assert_eq!(
        arena,
        ShadowArena {
            base: 0x74000,
            end: 0x80000
        },
        "the arena lives in the sandbox spawn-scratch page, below the child carve"
    );
    for (i, d) in m.data.iter().enumerate() {
        let end = d.offset + d.bytes.len() as u64;
        assert!(
            end <= arena.base || d.offset >= arena.end,
            "data segment {i} [{:#x}, {end:#x}) overlaps the arena [{:#x}, {:#x})",
            d.offset,
            arena.base,
            arena.end
        );
    }
    assert!(
        arena.contexts() >= 8,
        "a REPL with a root and a few task fibers needs several contexts; got {} (stride {SHADOW_STRIDE})",
        arena.contexts()
    );
}

// ---------------------------------------------------------------------------------------------
// Steps 2 + 3 — the kernel as a live durable domain: words compile under a durable `Jit` grant,
// and a session freezes mid-transcript (a `task` suspended) and thaws to the same output.

use temen_durable::{arm_freeze_after, begin_thaw, init_durable_window, read_state, write_state};
use temen_interp::{
    bytecode, cap_id, run_capture_reserved_with_host, BoundImport, Host, StreamRole,
};
use temen_jit::JitOutcome;
use temen_snapshot::{freeze, restore};

const SIZE_LOG2: u8 = 20;
const WINDOW: usize = 1 << SIZE_LOG2;
const JIT_TABLE_LOG2: u8 = 10;

/// The transcript every freeze/thaw check runs: a definition, a `task` resumed twice (so it is
/// **suspended** with state when the freeze lands between its resumes), a second definition and a
/// third resume after the freeze point, and memory words whose line-units load/store the REPL stack.
const SESSION: &str = ": counter ( x -- y ) begin 1+ dup yield drop again ;\n\
    ' counter task\n\
    dup 0 resume . . cr\n\
    dup 10 resume . . cr\n\
    : sq ( n -- n ) dup * ;\n\
    7 sq . cr\n\
    dup 100 resume . . cr\n\
    variable x  42 x !  x @ . cr\n\
    drop\n";
const SESSION_OUT: &str = "1 0 \n2 0 \n49 \n3 0 \n42 \n";

fn instrumented() -> temen_ir::Module {
    let inst = temen_durable::transform_module_assume_confined(&kernel())
        .expect("the durable transform accepts the kernel through the confined path");
    temen_verify::verify_module(&inst).expect("instrumented kernel verifies");
    inst
}

/// The kernel's seven imports (`write read vm_jit_compile vm_jit_install vm_jit_invoke2
/// vm_jit_release vm_jit_uninstall`, in declaration order) bound to stdout, stdin and the `Jit`
/// domain — what `forth_session.rs` binds, and what the CLI's powerbox resolves by name.
fn bind_imports(host: &mut Host, out: i32, inp: i32, jit: i32) {
    host.set_import_bindings(vec![
        BoundImport::required(cap_id::STREAM, 1, out),
        BoundImport::required(cap_id::STREAM, 0, inp),
        BoundImport::required(cap_id::JIT, 0, jit),
        BoundImport::required(cap_id::JIT, 3, jit),
        BoundImport::required(cap_id::JIT, 1, jit),
        BoundImport::required(cap_id::JIT, 2, jit),
        BoundImport::required(cap_id::JIT, 4, jit),
    ]);
}

/// A durable powerbox for the kernel: stdout/stdin, and a `Jit` domain that hosts fibers and
/// durability through the **confined** durable validator (the kernel is a cooperating toolchain —
/// its line-units load/store the REPL stack, which the strict validator rightly refuses for an
/// untrusted unit). Returns the host and the three handles (the thaw host re-pins the same ones).
fn durable_host(m: &temen_ir::Module, stdin: &str) -> (Host, [i32; 3]) {
    let mut host = Host::new();
    host.set_durable(true);
    host.set_jit_hosts_fibers(true);
    host.stdin = stdin.as_bytes().to_vec();
    let out = host.grant_stream(StreamRole::Out);
    let inp = host.grant_stream(StreamRole::In);
    let jit = temen_run::grant_jit_durable_confined(&mut host, m, JIT_TABLE_LOG2);
    bind_imports(&mut host, out, inp, jit);
    (host, [out, inp, jit])
}

/// The engines that drive a durable Forth session. The bytecode engine is not one: it declines the
/// whole kernel under durability (`thread.*` for `spawn`/`join`, the `Instantiator` cap for
/// `sandbox` — DURABILITY.md §4) and falls back to the tree-walker, pinned below.
#[derive(Clone, Copy, Debug)]
enum Engine {
    TreeWalk,
    Jit,
}

/// Run the kernel's `_start` over `window` on `engine` and return the captured window.
fn run_on(engine: Engine, inst: &temen_ir::Module, window: &[u8], host: &mut Host) -> Vec<u8> {
    match engine {
        Engine::TreeWalk => {
            let mut fuel = 50_000_000_000u64;
            let (r, snap) =
                run_capture_reserved_with_host(inst, 0, &[], &mut fuel, window, SIZE_LOG2, host);
            r.expect("the kernel runs to a return");
            snap
        }
        Engine::Jit => {
            let (out, snap) =
                temen_run::jit_cap_run(inst, 0, &[], window, SIZE_LOG2, JIT_TABLE_LOG2, host)
                    .expect("the kernel compiles and runs on the JIT");
            assert!(matches!(out, JitOutcome::Returned(_)), "JIT run: {out:?}");
            snap
        }
    }
}

/// **Step 3 — the install fence and the durable gate admit the kernel's words.** Under a durable
/// `Jit` grant every unit the kernel submits is instrumented before verify: pure words, memory
/// words, and the fiber words (`task`/`resume`/`yield` compile to `cont.*` in the unit — the
/// `(i64) -> (i64)` word shape is a signature the kernel's own fiber trampoline taints, so the R8
/// install fence admits a suspending word). The transcript's output is the plain run's, byte for
/// byte. The one refusal that surfaced — a REPL line-unit that resumes a fiber *and* loads/stores the
/// REPL data stack fails the strict validator's R9 check — is what the confined variant exists for.
#[test]
fn the_kernels_words_compile_under_a_durable_jit_grant() {
    let inst = instrumented();
    for engine in [Engine::TreeWalk, Engine::Jit] {
        let (mut host, _) = durable_host(&kernel(), SESSION);
        let mut win = init_durable_window(WINDOW, arena());
        write_state(&mut win, temen_durable::STATE_NORMAL);
        run_on(engine, &inst, &win, &mut host);
        assert_eq!(
            String::from_utf8(host.stdout_bytes()).unwrap(),
            SESSION_OUT,
            "a durable session on {engine:?} must print exactly what a plain one does"
        );
    }
}

/// The bytecode engine's scope pin: it declines the kernel under durability (the `thread.*` /
/// `Instantiator` ops of `spawn`/`join`/`sandbox` put the module outside its durable subset), so
/// the tree-walker and the JIT are the two engines a durable Forth session runs on. If this ever
/// returns `Some`, the engine widened its scope and the freeze/thaw checks below want a third leg.
#[test]
fn the_bytecode_engine_declines_the_kernel_under_durability() {
    let inst = instrumented();
    let (mut host, _) = durable_host(&kernel(), SESSION);
    let win = init_durable_window(WINDOW, arena());
    let mut fuel = 50_000_000_000u64;
    assert!(
        bytecode::compile_and_run_capture_reserved_with_host(
            &inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut host
        )
        .is_none(),
        "the bytecode engine falls back to the tree-walker for a durable kernel with thread/nesting ops"
    );
}

/// The gap the confined validator closes, pinned: the **strict** durable grant refuses the first
/// line-unit that resumes a fiber (it also loads/stores the REPL stack), so the transcript degrades
/// to `compile error` there. If this ever passes, the strict path started admitting memory ops and
/// R9 needs a second look.
#[test]
fn the_strict_durable_grant_refuses_a_line_unit_that_resumes_a_fiber() {
    let inst = instrumented();
    let mut host = Host::new();
    host.set_durable(true);
    host.set_jit_hosts_fibers(true);
    host.stdin = SESSION.as_bytes().to_vec();
    let out = host.grant_stream(StreamRole::Out);
    let inp = host.grant_stream(StreamRole::In);
    let jit = temen_run::grant_jit_durable(&mut host, &kernel(), JIT_TABLE_LOG2);
    bind_imports(&mut host, out, inp, jit);
    let mut win = init_durable_window(WINDOW, arena());
    write_state(&mut win, temen_durable::STATE_NORMAL);
    run_on(Engine::TreeWalk, &inst, &win, &mut host);
    let out = String::from_utf8(host.stdout_bytes()).unwrap();
    assert!(
        out.starts_with("compile error -22\n"),
        "the strict validator must refuse the resuming line-unit (R9: it touches guest memory), got {out:?}"
    );
}

fn arena() -> temen_ir::durable_abi::ShadowArena {
    kernel()
        .memory
        .and_then(|x| x.shadow)
        .expect("the kernel declares its arena")
}

/// **Step 2 — freeze a live session, thaw it, continue.** Armed to freeze at a fiber safepoint
/// after `counter` has been resumed once (its first `yield` parks it holding `1`), the run unwinds:
/// the root REPL frames and the parked task flatten into the kernel's declared arena, the compiled
/// words ride the artifact (Section 5), and the artifact restores into a **fresh** host that never
/// saw the transcript. Thawed, the session resumes the same `task` (its state intact — `2 0`, then
/// `3 0` after the second definition), defines and calls new words through the re-pinned `Jit`
/// domain, and its stdout concatenated with the frozen half equals the uninterrupted run.
/// The artifact is engine-neutral (DURABILITY.md §4 "JIT parity"): the three legs freeze and thaw
/// on the tree-walker, on the JIT, and across them.
fn freeze_and_thaw(freeze_on: Engine, thaw_on: Engine) {
    let inst = instrumented();
    let arena = arena();

    // Freeze: arm the window so the state word flips to UNWINDING at the 4th fiber safepoint —
    // `task` (cont.new) is not one; resume #1, `counter`'s first `yield`, resume #2 and its second
    // `yield` are #1–#4. The countdown ticks *before* the op, so #4 promotes as `yield` #2 runs: the
    // fiber parks holding `2`, resume #2's trailing poll unwinds the REPL line-unit before it prints,
    // and the freeze driver flattens the idle task. (Arming at #3 — the resume of an already-parked
    // fiber — hits the engine gap pinned in `temen-durable/tests/freeze_trigger.rs`: the fiber
    // unwinds *after* the resume delivered its value, and the thaw re-parks it instead.)
    let (mut fh, handles) = durable_host(&kernel(), SESSION);
    let mut win = init_durable_window(WINDOW, arena);
    write_state(&mut win, temen_durable::STATE_NORMAL);
    arm_freeze_after(&mut win, 4);
    let snap = run_on(freeze_on, &inst, &win, &mut fh);
    assert_eq!(
        read_state(&snap),
        temen_durable::STATE_UNWINDING,
        "the armed run on {freeze_on:?} must have unwound (frozen), not run to completion"
    );
    let first = String::from_utf8(fh.stdout_bytes()).unwrap();
    assert!(
        SESSION_OUT.starts_with(&first) && !first.is_empty() && first != SESSION_OUT,
        "the frozen half is a proper prefix of the uninterrupted output, got {first:?}"
    );
    assert!(
        !fh.frozen_fibers().is_empty(),
        "the freeze driver flattened the suspended `task`"
    );
    let artifact = freeze(&inst, &snap, &fh).expect("freeze admits the session");

    // Thaw into a host that never ran the transcript: the same durable powerbox *policy* (fibers,
    // the confined durable validator, the install fence), but no grants — the artifact's handle
    // table re-pins stdout/stdin/the `Jit` domain (its compiled words re-verified) at their slots.
    let mut th = Host::new();
    th.set_durable(true);
    th.set_jit_hosts_fibers(true);
    temen_run::set_jit_durable_policy_confined(&mut th, &kernel());
    let mut window = restore(&artifact, &inst, &mut th).expect("restore");
    bind_imports(&mut th, handles[0], handles[1], handles[2]);
    assert!(
        !th.frozen_fibers().is_empty(),
        "the restore re-seeded the suspended `task` from the artifact"
    );
    begin_thaw(&mut window, arena, 0);
    run_on(thaw_on, &inst, &window, &mut th);
    let second = String::from_utf8(th.stdout_bytes()).unwrap();
    assert_eq!(
        format!("{first}{second}"),
        SESSION_OUT,
        "frozen half {first:?} ({freeze_on:?}) + thawed half {second:?} ({thaw_on:?}) must equal the uninterrupted transcript"
    );
}

#[test]
fn a_live_session_freezes_with_a_suspended_task_and_thaws() {
    freeze_and_thaw(Engine::TreeWalk, Engine::TreeWalk);
}

#[test]
fn a_live_session_freezes_and_thaws_on_the_jit() {
    freeze_and_thaw(Engine::Jit, Engine::Jit);
}

#[test]
fn a_session_frozen_on_the_tree_walker_thaws_on_the_jit() {
    freeze_and_thaw(Engine::TreeWalk, Engine::Jit);
}
