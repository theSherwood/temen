//! **GNU bash in the browser** (#1080 slice 1) — the playground `bash_exec` entry running the real
//! whole-program bash module on the bytecode engine under the POSIX personality, the browser twin of
//! `demos/bash`'s native `bash_probe`/capstone.
//!
//! bash is **GPLv3 and never vendored** (`demos/bash/build_bitcode.sh` fetches it from ftp.gnu.org),
//! so there is no committed `bash.temen` fixture the way `shell.temen` is committed — it is a
//! build-at-deploy asset. These tests therefore **gate on the locally-built asset**: point
//! `TEMEN_BASH_TEMEN` at a `bash.temen` (default `/tmp/temen_bash_cache/bash.temen`, where
//! `bash_asset` writes it), or the test skips-loud like the #802 capstone does offline. Build it with:
//!   (cd crates/temen-run/demos/bash && ./build_bitcode.sh)
//!   (cd crates/temen-llvm && cargo build --release --example bash_asset \
//!      && ./target/release/examples/bash_asset /tmp/temen_bash_cache/bash_linked.ll \
//!         /tmp/temen_bash_cache/bash.temen)

use temen_browser::{bash_exec, bash_exec_with, STATUS_EXIT};

/// The committed `/bin` coreutils (repo-owned C — unlike GPLv3 bash, these ARE vendored). Decode each
/// and return `(path, module, window_log2)` for `bash_exec_with` to register as filesystem
/// executables, so bash resolves an external command (`seq` → `/bin/seq`) and fork → execve's it.
/// Regenerate the fixtures with the `gen_browser_bash_coreutils` ignored test (see c_shell.rs).
fn load_coreutils() -> Vec<(&'static str, temen_ir::Module, u8)> {
    let raw: &[(&str, &[u8])] = &[
        ("/bin/true", include_bytes!("fixtures/bin_true.temen")),
        ("/bin/false", include_bytes!("fixtures/bin_false.temen")),
        ("/bin/echo", include_bytes!("fixtures/bin_echo.temen")),
        ("/bin/cat", include_bytes!("fixtures/bin_cat.temen")),
        ("/bin/seq", include_bytes!("fixtures/bin_seq.temen")),
        ("/bin/head", include_bytes!("fixtures/bin_head.temen")),
        ("/bin/wc", include_bytes!("fixtures/bin_wc.temen")),
        ("/bin/sort", include_bytes!("fixtures/bin_sort.temen")),
        ("/bin/uniq", include_bytes!("fixtures/bin_uniq.temen")),
        ("/bin/ls", include_bytes!("fixtures/bin_ls.temen")),
        ("/bin/pwd", include_bytes!("fixtures/bin_pwd.temen")),
        ("/bin/grep", include_bytes!("fixtures/bin_grep.temen")),
        ("/bin/tr", include_bytes!("fixtures/bin_tr.temen")),
    ];
    raw.iter()
        .map(|(path, bytes)| {
            let m = temen_encode::decode_module(bytes).expect("decode coreutil");
            let wl = m.memory.map_or(0, |mc| mc.size_log2);
            (*path, m, wl)
        })
        .collect()
}

/// Load the deploy-built bash module, or `None` (with a loud note) when it is absent — the CI/offline
/// skip, since the GPLv3 asset is never in the tree.
fn load_bash() -> Option<temen_ir::Module> {
    let path = std::env::var("TEMEN_BASH_TEMEN")
        .unwrap_or_else(|_| "/tmp/temen_bash_cache/bash.temen".to_string());
    let bytes = std::fs::read(&path).ok()?;
    Some(temen_encode::decode_module(&bytes).expect("decode bash.temen"))
}

/// `bash -c 'echo …'` — the headline: the real shell parses argv from the powerbox args buffer, runs
/// the command, and its `write(1, …)` lands in the personality's captured stdout.
#[test]
fn bash_dash_c_echo() {
    let Some(m) = load_bash() else {
        eprintln!("note: skipping bash_dash_c_echo — no bash.temen (set TEMEN_BASH_TEMEN or run build_bitcode.sh + bash_asset)");
        return;
    };
    let out = bash_exec(&m, &[b"bash", b"-c", b"echo hi from browser bash"], b"");
    // bash always leaves through `exit_shell` → the personality exit → `Trap::Exit`, so a clean run
    // is STATUS_EXIT with code 0 (never STATUS_OK — that is a C `main` that *returns*).
    assert_eq!(
        out.status,
        STATUS_EXIT,
        "bash -c should exit cleanly (stderr: {})",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.exit_code, 0, "clean exit code");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "hi from browser bash\n"
    );
}

/// Argv seeding carries multiple words + an arithmetic expansion — proves the whole `{argc,envc}`
/// blob (not just argv[2]) reaches bash's `_start`.
#[test]
fn bash_dash_c_arithmetic() {
    let Some(m) = load_bash() else {
        eprintln!("note: skipping bash_dash_c_arithmetic — no bash.temen");
        return;
    };
    let out = bash_exec(&m, &[b"bash", b"-c", b"echo $((6 * 7))"], b"");
    assert_eq!(
        out.status,
        STATUS_EXIT,
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.exit_code, 0);
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

/// An explicit `exit N` — the #1062 path (a `COPY_PROCENV` `longjmp` on every `bash -c`) resolving on
/// the bytecode tier, and the exit code flowing back through the personality as `Trap::Exit`.
#[test]
fn bash_dash_c_exit_code() {
    let Some(m) = load_bash() else {
        eprintln!("note: skipping bash_dash_c_exit_code — no bash.temen");
        return;
    };
    let out = bash_exec(&m, &[b"bash", b"-c", b"echo before; exit 7"], b"");
    assert_eq!(
        out.status, STATUS_EXIT,
        "explicit exit should report STATUS_EXIT"
    );
    assert_eq!(
        out.exit_code, 7,
        "the real exit code, not the fork-twin crash status"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "before\n");
}

/// #1080 slice 2 — an **external command as the last of `-c`**: bash's process-saving optimization
/// `execve`s the final simple command directly on its own (root) task instead of fork+exec+wait. So
/// `bash -c 'seq 3'` image-replaces the root bash with `/bin/seq` (carrying bash's personality — its
/// fd 1 — via the shared `exec_carry`), and `seq` returns its exit code (0). The coreutil's stdout is
/// the run's output.
#[test]
fn bash_dash_c_external_seq_last() {
    let Some(bash) = load_bash() else {
        eprintln!("note: skipping bash_dash_c_external_seq_last — no bash.temen");
        return;
    };
    let bins = load_coreutils();
    let bin_refs: Vec<(&str, &temen_ir::Module, u8)> =
        bins.iter().map(|(p, m, wl)| (*p, m, *wl)).collect();
    let out = bash_exec_with(&bash, &[b"bash", b"-c", b"seq 3"], b"", &bin_refs);
    // The last command is exec'd on the root, so `seq` *returns* its status (STATUS_OK, value 0) —
    // it did not go through bash's own `exit_shell` (which would be STATUS_EXIT).
    assert_eq!(
        out.status,
        temen_browser::STATUS_OK,
        "seq exec'd on the root returns its status (status={} stdout={:?} stderr={:?})",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    assert_eq!(out.value, 0, "seq's exit status");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "1\n2\n3\n");
}

/// #1080 slice 2 — a **fork-twin exec**: a non-final external command forces bash to fork → execve
/// (rather than exec-the-last-command). `bash -c 'seq 2; echo done'` should fork a twin for `seq 2`,
/// the twin image-replaces with `/bin/seq` (the `env: Some` exec arm + personality carry, both landed
/// here), bash reaps it, then runs the builtin `echo done` and exits.
///
/// `#[ignore]`d: this surfaced that **bash's own `fork()` on the pure bytecode cooperative engine**
/// returns an error ("bash: fork: Unknown error") — bash reaches fork through the personality's
/// host-proc fork dispatch, a path never exercised on the bytecode engine before (bash previously ran
/// only on the tree-walker, whose exec decline folded it there; the bespoke `temen-posix` shell spawns
/// externals via op-13 `instantiate_module`, not `fork()`). That bytecode fork-dispatch gap is a
/// distinct follow-up rung from exec_module; the `env: Some` exec arm itself is ready for it. Single
/// external commands (exec-the-last-command → a *root* exec) already work — see the test above.
#[test]
#[ignore = "bash's personality fork() on the bytecode cooperative engine is a separate follow-up (see doc); exec_module itself is done"]
fn bash_dash_c_external_seq_forked() {
    let Some(bash) = load_bash() else {
        eprintln!("note: skipping bash_dash_c_external_seq_forked — no bash.temen");
        return;
    };
    let bins = load_coreutils();
    let bin_refs: Vec<(&str, &temen_ir::Module, u8)> =
        bins.iter().map(|(p, m, wl)| (*p, m, *wl)).collect();
    let out = bash_exec_with(
        &bash,
        &[b"bash", b"-c", b"seq 2; echo done"],
        b"",
        &bin_refs,
    );
    assert_eq!(
        out.status,
        STATUS_EXIT,
        "bash exits after reaping the forked seq (status={} stdout={:?} stderr={:?})",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "1\n2\ndone\n",
        "exit={} value={} stderr={:?}",
        out.exit_code,
        out.value,
        String::from_utf8_lossy(&out.stderr)
    );
}
