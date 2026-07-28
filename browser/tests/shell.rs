//! The **`svm-posix` shell** running in the browser (`posix_shell_exec` / the wasm `svm_run_shell`
//! export) — STAGE1.md "real posix shell in the playground". The committed fixture
//! `fixtures/shell.svmb` is the **full** shell: the same `shim.c + ring.c + shell_main.c` the
//! differential `crates/svm/tests/c_shell.rs` compiles, *including* external-command spawn (op 13) and
//! concurrent ring pipelines (op 11 + `SharedRegion` + futex). Those cap.calls now run on the browser's
//! bytecode cooperative engine (the slices that lowered ops 13/11 + region + futex, plus op-13
//! child-manifest binding). Regenerate both fixtures with
//! `cargo test -p svm --test c_shell -- --ignored --exact gen_browser_shell_fixture`.
//!
//! The shell reads its script from the personality's stdin and loops to EOF; its `write(1, …)` lands
//! in the personality's captured stdout. Run on the **bytecode** engine — the single-threaded,
//! wasm-safe interpreter tier (the tree-walk/JIT engines the *native* differential runs use OS threads
//! + a wall clock, absent under wasm). The whole surface (builtins, redirects, memfs + **ring**
//! pipelines, external commands, if, vars, globbing) is byte-checked against the tree-walk oracle by
//! `c_shell`'s three-way interp==JIT==bytecode differential.

use svm_browser::{posix_shell_exec, posix_shell_exec_with, STATUS_OK};

fn run(script: &str) -> String {
    let bytes = include_bytes!("fixtures/shell.svmb");
    let m = svm_encode::decode_module(bytes).expect("decode shell.svmb");
    let out = posix_shell_exec(&m, script.as_bytes());
    assert_eq!(
        out.status, STATUS_OK,
        "the shell should run to EOF cleanly (script: {script:?})",
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// As [`run`], with the full playground PATH registered: the `__stage` ring-filter runner (so
/// pipelines take the concurrent ring path — op 11 + `SharedRegion` + futex — instead of sequential
/// memfs staging) and the `primes` external command (an op-13 §14 child). Exactly what the browser's
/// `svm_run_shell` grants, so these tests mirror the real playground registry.
fn run_with_stage(script: &str) -> String {
    let bytes = include_bytes!("fixtures/shell.svmb");
    let m = svm_encode::decode_module(bytes).expect("decode shell.svmb");
    let rbytes = include_bytes!("fixtures/stage_runner.svmb");
    let runner = svm_encode::decode_module(rbytes).expect("decode stage_runner.svmb");
    let pbytes = include_bytes!("fixtures/primes.svmb");
    let primes = svm_encode::decode_module(pbytes).expect("decode primes.svmb");
    let out = posix_shell_exec_with(
        &m,
        script.as_bytes(),
        &[("__stage", &runner), ("primes", &primes)],
    );
    assert_eq!(
        out.status, STATUS_OK,
        "the shell should run to EOF cleanly (script: {script:?})",
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn shell_echo_and_variables() {
    // `echo` with a literal and a `$VAR` expansion, then a second line — the read-eval loop.
    let out = run("echo hello world\nX=42\necho val $X\n");
    assert_eq!(out, "hello world\nval 42\n");
}

#[test]
fn shell_pwd_after_cd() {
    // `cd` mutates the personality cwd; `pwd` reads it back — personality state across commands.
    let out = run("cd /tmp\npwd\n");
    assert_eq!(out, "/tmp\n");
}

#[test]
fn shell_redirect_and_cat() {
    // Redirection into the in-window memfs, then read it back — the fs surface (`open`/`write`/`read`).
    let out = run("echo persisted > f\ncat f\n");
    assert_eq!(out, "persisted\n");
}

#[test]
fn shell_pipeline_through_memfs_staging() {
    // With no `__stage` runner registered a pipeline falls back to sequential memfs-temp staging — real
    // `cat | grep` semantics in-process, no rings. (The ring path is exercised below.)
    let out = run("echo a > f\necho b >> f\necho c >> f\ncat f | grep b\n");
    assert_eq!(out, "b\n");
}

#[test]
fn shell_ring_pipeline_sort_uniq() {
    // The headline browser capability: with `__stage` on PATH, `cat f | sort | uniq` runs as three
    // **concurrent** stages over shared-memory rings (op 11 + `SharedRegion` + futex) on the bytecode
    // cooperative engine — the child stages' output unifies into the shell's captured stdout.
    let out = run_with_stage("echo b > f\necho a >> f\necho b >> f\ncat f | sort | uniq\n");
    assert_eq!(out, "a\nb\n");
}

#[test]
fn shell_ring_pipeline_status_and_early_exit() {
    // Status threads back through a ring child (`grep` no-match → 1) and `head`'s early exit closes its
    // input ring (SIGPIPE-lite) so the producer stops instead of wedging — matching the native `c_shell`
    // ring differential, now on the browser engine.
    let out = run_with_stage(
        "echo one > f\necho two >> f\necho one >> f\n\
         cat f | grep zzz\necho rc $?\ncat f | head -n 1\necho rc $?\n",
    );
    assert_eq!(out, "rc 1\none\nrc 0\n");
}

#[test]
fn shell_external_command_primes() {
    // An **external command** (op-13 §14 child): `primes` is a separate compiled-C program on PATH, not
    // a builtin. The shell spawns it, delivers `argv`, and its stdout streams into the shell's sink.
    let out = run_with_stage("primes 20\n");
    assert_eq!(out, "2\n3\n5\n7\n11\n13\n17\n19\n");
}

#[test]
fn shell_command_list_short_circuits() {
    // `;` / `&&` / `||` with `$?` short-circuiting (true/false builtins).
    let out = run("true && echo yes\nfalse && echo no\nfalse || echo fallback\n");
    assert_eq!(out, "yes\nfallback\n");
}

#[test]
fn shell_unknown_command_reports_not_found() {
    // No PATH registry in the browser this slice, so an unknown command is `not found` (status path
    // intact); the shell keeps looping.
    let out = run("nosuchcmd\necho after\n");
    assert!(
        out.contains("not found") && out.ends_with("after\n"),
        "unknown command reported, loop continues: {out:?}",
    );
}
