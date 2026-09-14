//! #1235 — **the Forth REPL under `JitSession` compaction.**
//!
//! Every top-level line compiles a fresh §22 unit (compile → install → `call.dyn` → uninstall →
//! release) and every `: … ;` redefinition leaves the old word's code behind. `cranelift-jit` has no
//! per-function free, so a long session walks the code arena up to the compile quota and then prints
//! `compile error -12` (`-ENOMEM`). Running the kernel through `temen_run::JitSession` — re-entering
//! its `prompt` entry per line and auto-compacting past a byte watermark — is what bounds it.
//!
//! The kernel's `prompt(ptr, len)` entry (func 74) carries the dictionary, REPL stack and heap across
//! prompts in the session's window snapshot. That is why the kernel's memory map splits at `0x40000`:
//! a session carries only the low 256 KiB (`SESSION_SNAP`), so everything that must survive a prompt
//! lives below the line and the per-unit scratch above it.
#![cfg(all(unix, target_arch = "x86_64"))]

use temen_interp::{cap_id, BoundImport, Host, StreamRole};
use temen_ir::DEFAULT_RESERVED_LOG2;
use temen_jit::JitOutcome;
use temen_run::JitSession;

/// The kernel's `call.dyn` install table, matching what the CLI reserves for a `Jit.install` guest.
const JIT_TABLE_LOG2: u8 = 10;
/// The `prompt` entry's index (`export 1 func "prompt" 74`).
const PROMPT_ENTRY: u32 = 74;
/// Where the driver stages each line — the kernel's prompt-staging region. It must sit **below** the
/// session's snapshot line: `JitSession::seed_window` clamps to the carried 256 KiB, so staging above
/// it seeds nothing at all (silently — the guest then tokenizes whatever is there).
const LINE_OFF: usize = 0x3C000;

fn kernel() -> temen_ir::Module {
    let m = temen_text::parse_module(include_str!("../demos/forth/forth.temt"))
        .expect("forth.temt parses");
    temen_verify::verify_module(&m).expect("forth.temt verifies");
    m
}

/// A REPL session over the kernel: the powerbox the CLI would build (stdout/stdin + a fiber-hosting
/// `Jit` with an install table), the kernel's seven imports bound in declaration order, and a
/// `watermark`-auto-compacting session entered at `prompt`.
fn session(m: &temen_ir::Module, watermark: usize) -> (JitSession, i64) {
    let mut host = Host::new();
    host.set_jit_validator(temen_run::jit_blob_validator);
    // The kernel's words are §12 fiber hosts (`task`/`yield`/`resume` compile to `cont.*` inside the
    // submitted unit), so the grant must admit them — the CLI's `set_jit_hosts_fibers(true)`.
    host.set_jit_hosts_fibers(true);
    let out = host.grant_stream(StreamRole::Out);
    let stdin = host.grant_stream(StreamRole::In);
    let jit = temen_run::grant_jit_fibers(&mut host, m, JIT_TABLE_LOG2);
    // `import 0 "write" … 6 "vm_jit_uninstall"`, in the module's declaration order.
    host.set_import_bindings(vec![
        BoundImport::required(cap_id::STREAM, 1, out),
        BoundImport::required(cap_id::STREAM, 0, stdin),
        BoundImport::required(cap_id::JIT, 0, jit),
        BoundImport::required(cap_id::JIT, 3, jit),
        BoundImport::required(cap_id::JIT, 1, jit),
        BoundImport::required(cap_id::JIT, 2, jit),
        BoundImport::required(cap_id::JIT, 4, jit),
    ]);
    let domain = host.resolve_jit_domain(jit).expect("jit domain");
    let s = JitSession::new(
        m,
        PROMPT_ENTRY,
        DEFAULT_RESERVED_LOG2,
        JIT_TABLE_LOG2,
        domain,
        watermark,
        host,
    )
    .expect("session");
    (s, jit as i64)
}

/// Run one prompt: stage `line` in the window above the snapshot line, then re-enter `prompt`.
fn prompt(s: &mut JitSession, line: &str) {
    s.seed_window(LINE_OFF, line.as_bytes());
    match s
        .run_prompt(&[LINE_OFF as i64, line.len() as i64])
        .expect("prompt runs")
    {
        JitOutcome::Returned(_) => {}
        other => panic!("prompt {line:?} returned {other:?}"),
    }
}

/// **The REPL carries its world across prompts.** Words defined on one prompt are callable on the
/// next, the data stack persists, and `variable`/`!`/`@` survive — i.e. the dictionary, REPL stack
/// and heap all live below the session's snapshot line.
#[test]
fn forth_session_carries_the_repl_across_prompts() {
    let m = kernel();
    let (mut s, _) = session(&m, 0);
    prompt(&mut s, ": sq ( n -- n ) dup * ;\n");
    prompt(&mut s, "7 sq . cr\n");
    prompt(&mut s, "variable x  42 x !\n");
    prompt(&mut s, "x @ . cr\n");
    // The data stack persists between prompts, the way it does between lines under `_start`.
    prompt(&mut s, "1 2\n");
    prompt(&mut s, "+ . cr\n");
    let out = String::from_utf8(s.into_host().stdout_bytes()).expect("utf-8");
    assert_eq!(out, "49 \n42 \n3 \n", "the world must carry across prompts");
}

/// **Slice 1 — auto-compaction bounds the arena, transparently.** Every prompt compiles a fresh unit
/// and every redefinition abandons the old word's code; `cranelift-jit` has no per-function free, so
/// without compaction a long session's occupancy climbs monotonically toward the compile quota. With
/// a watermark the session reclaims between prompts, and the guest cannot tell: the transcript is
/// byte-identical either way.
#[test]
fn forth_session_auto_compacts_without_the_guest_noticing() {
    let m = kernel();
    let transcript: Vec<String> = (0..24)
        .map(|i| format!(": w ( n -- n ) {} + ;\n{} w . cr\n", i, i))
        .collect();

    let run = |watermark: usize| -> (String, usize, usize) {
        let (mut s, _) = session(&m, watermark);
        for line in &transcript {
            prompt(&mut s, line);
        }
        let (occ, comp) = (s.occupancy(), s.compactions());
        let out = String::from_utf8(s.into_host().stdout_bytes()).expect("utf-8");
        (out, occ, comp)
    };

    let (out_off, occ_off, comp_off) = run(0);
    assert_eq!(comp_off, 0, "watermark 0 disables auto-compaction");
    assert!(occ_off > 0, "the session must have compiled something");

    // A watermark of ~a quarter of the uncompacted run, so it fires several times.
    let (out_on, occ_on, comp_on) = run(occ_off / 4);
    assert_eq!(
        out_on, out_off,
        "the guest must not observe the reclaim — compacted transcript must match"
    );
    assert!(
        comp_on > 0,
        "the session must have compacted at least once (occupancy off={occ_off})"
    );
    assert!(
        occ_on < occ_off,
        "compaction must bound the arena: on={occ_on} vs off={occ_off}"
    );
}

/// **Slice 2 — the #1214 limitation-2 question: does compaction relocate code under a *suspended
/// fiber*?** A Forth `task` is a §12 fiber whose entry is an installed word's code; suspended at a
/// `yield`, it holds a return address into that code. Whole-module recompaction moves code, so if
/// the compactor does not treat a live fiber's frames as roots, resuming after a compaction lands in
/// freed or moved code.
///
/// Create the task on one prompt, force a compaction, resume it on a later one. The resumed
/// generator must keep counting — `1`, then `2` — exactly as it does without the compaction.
#[test]
fn forth_session_resumes_a_fiber_across_a_compaction() {
    let m = kernel();
    let drive = |compact: bool| -> String {
        let (mut s, _) = session(&m, 0); // manual: compact exactly where we choose
        prompt(
            &mut s,
            ": counter ( x -- y ) begin 1+ dup yield drop again ;\n",
        );
        prompt(&mut s, "' counter task\n");
        prompt(&mut s, "dup 0 resume . . cr\n");
        if compact {
            // Quiescent point between prompts — the only place a session may compact.
            let before = s.occupancy();
            s.compact().expect("compaction between prompts");
            assert!(
                s.compactions() == 1 && before > 0,
                "non-vacuity: the compaction must have run over a non-empty arena \
                 (before={before}, compactions={})",
                s.compactions()
            );
        }
        prompt(&mut s, "dup 0 resume . . cr\n");
        prompt(&mut s, "drop\n");
        String::from_utf8(s.into_host().stdout_bytes()).expect("utf-8")
    };
    let plain = drive(false);
    assert_eq!(
        drive(true),
        plain,
        "a suspended fiber must survive a compaction — if this diverges, the compactor is not \
         treating live fiber frames as roots (a JitSession bug, not a Forth one: file under Backends \
         and keep compaction off for fiber-hosting sessions)"
    );
    assert!(
        plain.contains('1') && plain.contains('2'),
        "non-vacuity: the generator must actually have advanced, got {plain:?}"
    );
}

/// **Slice 3 — install slots are reused, so a redefinition-heavy session does not exhaust the
/// table.** Each `: … ;` installs its unit at a `call.dyn` slot; the 1024-slot reservation would run
/// out in ~1000 redefinitions if `uninstall` never freed one. Redefine the same word well past the
/// table size and check the session is still computing correctly at the end — an exhausted table
/// would surface as an install error long before.
#[test]
fn forth_session_reuses_install_slots_under_redefinition() {
    let m = kernel();
    let (mut s, _) = session(&m, 0);
    // 1 << JIT_TABLE_LOG2 = 1024 slots; go well past it.
    for i in 0..1400 {
        prompt(&mut s, &format!(": w ( n -- n ) {i} + ;\n"));
    }
    prompt(&mut s, "5 w . cr\n");
    let out = String::from_utf8(s.into_host().stdout_bytes()).expect("utf-8");
    assert_eq!(
        out, "1404 \n",
        "after 1400 redefinitions the latest `w` must still install and run (got {out:?}) — an \
         exhausted 1024-slot table would have failed to install long before"
    );
}
