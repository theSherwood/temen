//! The **`svm-posix` shell** running in the browser (`posix_shell_exec` / the wasm `svm_run_shell`
//! export) — STAGE1.md "real posix shell in the playground". The committed fixture
//! `fixtures/shell.svmb` is the shell's **sequential subset** (`SVM_SHELL_SEQUENTIAL`): the same
//! `shim.c + shell_main.c` the differential `crates/svm/tests/c_shell.rs` compiles, minus the
//! external-command spawn and concurrent ring pipelines — so it carries no `Instantiator`/
//! `SharedRegion` cap.calls and compiles on the browser's bytecode engine. Regenerate with
//! `cargo test -p svm --test c_shell -- --ignored --exact gen_browser_shell_fixture`.
//!
//! The shell reads its script from the personality's stdin and loops to EOF; its `write(1, …)` lands
//! in the personality's captured stdout. Run on the **bytecode** engine — the single-threaded,
//! wasm-safe interpreter tier (the tree-walk/JIT engines that run the full shell use OS threads + a
//! wall clock, absent under wasm). The Stage-0 surface (builtins, redirects, memfs pipelines, if,
//! vars, globbing) is bytecode-compilable and byte-checked against the tree-walk oracle by `c_shell`.

use svm_browser::{posix_shell_exec, STATUS_OK};

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
    // A pipeline with no `__stage` runner registered falls back to sequential memfs-temp staging (the
    // browser grants no external commands this slice) — real `cat | grep | wc` semantics in-process.
    let out = run("echo a > f\necho b >> f\necho c >> f\ncat f | grep b\n");
    assert_eq!(out, "b\n");
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
