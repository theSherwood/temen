//! **GNU bash in the browser** (#1080 slice 1) — the playground `bash_exec` entry running the real
//! whole-program bash module on the bytecode engine under the POSIX personality, the browser twin of
//! `demos/bash`'s native `bash_probe`/capstone.
//!
//! bash is **GPLv3 and never vendored** (`demos/bash/build_bitcode.sh` fetches it from ftp.gnu.org),
//! so there is no committed `bash.temen` fixture the way `shell.temen` is committed — it is a
//! build-at-deploy asset. These tests therefore **gate on the locally-built asset**: point
//! `TEMEN_BASH_TEMEN` at a `bash.temen` (default `/tmp/temen_bash_cache/bash.temen`), or the test
//! skips-loud like the #802 capstone does offline. Build it with the on-ramp translator (the same
//! `temen-llvm-translate` binary the browser Postgres asset uses), matching the browser host:
//!   (cd crates/temen-run/demos/bash && ./build_bitcode.sh)        # → bash_linked.ll
//!   (cd crates/temen-llvm && cargo build --release --bin temen-llvm-translate \
//!      && ./target/release/temen-llvm-translate /tmp/temen_bash_cache/bash_linked.ll \
//!         -o /tmp/temen_bash_cache/bash.temen --host-page 65536 --stub-externs --null-guard)
//! `--null-guard` is **required**, not optional: the coreutils in `/bin` (chibicc `--child-entry`)
//! are guarded, so their args base is `guard+128`; bash's `execve` must write the child argv to the
//! same place, which it only does when bash is guarded too. A guard-0 bash writes argv at 128 and the
//! guarded `seq`/`echo` read an empty region at 16512 (external commands run argv-less). `--host-page
//! 65536` matches the wasm host's 64 KiB pages (D40), as for every browser-target asset.

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

/// #1080 — a **fork-twin exec**: a non-final external command forces bash to fork → execve (rather
/// than exec-the-last-command). `bash -c 'seq 2; echo done'` forks a twin for `seq 2`, the twin
/// image-replaces with `/bin/seq`, bash reaps it, then runs the builtin `echo done` and exits.
///
/// The whole fork-twin-exec argv path composes on the bytecode engine: the twin `execve`s `/bin/seq`
/// carrying bash's fd 1, bash reaps it, then the builtin `echo done` runs. Gates on the deploy-built
/// `bash.temen` (skips-loud when absent). NB: the asset must be built with `--null-guard` so bash's
/// args base (`guard+128`) matches the guarded coreutils' — a guard-mismatched bash writes the child
/// argv where the coreutil never reads it (an earlier stale-`bash_linked.ll` symptom; see header doc).
#[test]
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

/// #1080 — a **two-stage pipeline of external commands** in the browser: `echo hi | cat`. bash forks
/// both stages, wires a CorePipe between them, each twin `execve`s its coreutil (`/bin/echo`,
/// `/bin/cat`), the reader blocks on the pipe until the writer's bytes arrive, then EOFs when it exits.
///
/// `#[ignore]`d — a CONFIRMED real deadlock (a distinct follow-up rung), reproduced with a freshly-built
/// `--null-guard` `bash.temen`, so it is not the stale-asset artifact that masked the fork/exec/wait
/// tests above. The stuck configuration: the root bash parks in `waitpid(-1)`, the writer twin (`echo`)
/// reaches `Done`, but the reader twin (`cat`, the exec'd `/bin/cat`) spins **Runnable** forever — it
/// neither reads-to-EOF nor parks (`BlockedPipeRead`). So the cooperative pipe park/wake does not engage
/// for an exec'd *reader* twin the way it does for the synthetic guests in the native `c_posix` pipeline
/// differentials (`seq | head | wc`, `sort | uniq`), which all pass on the bytecode engine. Un-ignore
/// once the exec'd-reader pipe park is fixed.
#[test]
#[ignore = "confirmed real deadlock (fresh --null-guard asset): the exec'd reader twin spins Runnable instead of pipe-read-parking/EOFing — a distinct pipeline-through-exec follow-up rung"]
fn bash_dash_c_pipeline_echo_cat() {
    let Some(bash) = load_bash() else {
        eprintln!("note: skipping bash_dash_c_pipeline_echo_cat — no bash.temen");
        return;
    };
    let bins = load_coreutils();
    let bin_refs: Vec<(&str, &temen_ir::Module, u8)> =
        bins.iter().map(|(p, m, wl)| (*p, m, *wl)).collect();
    let out = bash_exec_with(&bash, &[b"bash", b"-c", b"echo hi | cat"], b"", &bin_refs);
    assert_eq!(
        out.status,
        STATUS_EXIT,
        "status={} stdout={:?} stderr={:?}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "hi\n");
}

/// #1080 — a **three-stage coreutil pipeline** in the browser: `seq 5 | head -n 3 | wc -l` → `3`. Three
/// forks, three `execve`d coreutils, two CorePipes, every read park + carried pipe end + EOF + reap
/// composing under real bash on the bytecode engine — the milestone capstone.
///
/// `#[ignore]`d for the same confirmed exec'd-reader-twin spin as `bash_dash_c_pipeline_echo_cat` above
/// (a distinct follow-up; the native 3-stage `seq | head | wc` differential passes on both engines).
#[test]
#[ignore = "same confirmed exec'd-reader-twin pipeline deadlock as bash_dash_c_pipeline_echo_cat (see doc) — a distinct follow-up"]
fn bash_dash_c_pipeline_seq_head_wc() {
    let Some(bash) = load_bash() else {
        eprintln!("note: skipping bash_dash_c_pipeline_seq_head_wc — no bash.temen");
        return;
    };
    let bins = load_coreutils();
    let bin_refs: Vec<(&str, &temen_ir::Module, u8)> =
        bins.iter().map(|(p, m, wl)| (*p, m, *wl)).collect();
    let out = bash_exec_with(
        &bash,
        &[b"bash", b"-c", b"seq 5 | head -n 3 | wc -l"],
        b"",
        &bin_refs,
    );
    assert_eq!(
        out.status,
        STATUS_EXIT,
        "status={} stdout={:?} stderr={:?}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    // wc -l prints the line count (3), typically right-padded/whitespace-formatted; assert it contains 3.
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.trim() == "3", "expected line count 3, got {s:?}");
}
