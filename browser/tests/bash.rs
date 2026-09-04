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
        ("/bin/cut", include_bytes!("fixtures/bin_cut.temen")),
        ("/bin/tail", include_bytes!("fixtures/bin_tail.temen")),
        ("/bin/tac", include_bytes!("fixtures/bin_tac.temen")),
        ("/bin/rev", include_bytes!("fixtures/bin_rev.temen")),
        ("/bin/nl", include_bytes!("fixtures/bin_nl.temen")),
        ("/bin/fold", include_bytes!("fixtures/bin_fold.temen")),
        (
            "/bin/basename",
            include_bytes!("fixtures/bin_basename.temen"),
        ),
        ("/bin/dirname", include_bytes!("fixtures/bin_dirname.temen")),
        ("/bin/tee", include_bytes!("fixtures/bin_tee.temen")),
        ("/bin/touch", include_bytes!("fixtures/bin_touch.temen")),
        ("/bin/mkdir", include_bytes!("fixtures/bin_mkdir.temen")),
        ("/bin/rmdir", include_bytes!("fixtures/bin_rmdir.temen")),
        ("/bin/rm", include_bytes!("fixtures/bin_rm.temen")),
        ("/bin/cp", include_bytes!("fixtures/bin_cp.temen")),
        ("/bin/mv", include_bytes!("fixtures/bin_mv.temen")),
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
/// Root-caused and fixed: the wedge was an engine LIVELOCK, not a pipe bug — the root bash's
/// `waitpid(-1)` park (`BlockedReapPersonality(None)`) was re-woken forever by the already-reaped
/// `echo` twin (a stale `Done` entry in the any-child wake set), and the endlessly re-woken root
/// (lowest task index) starved the `cat` twin, which never got scheduled at all. The fix prunes
/// consumed Done twins from the wake set when an any-child parker re-parks (bytecode.rs ReapWait arm).
/// Diagnosed natively via `bash_probe` (`BASH_PROBE_BACKEND=bytecode`) — same asset, same personality,
/// reliable builds — where the pick counts showed root:~2M, echo:1, cat:0.
#[test]
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

/// #801 — the **`cut` coreutil** in real pipelines: field mode (`-d`/`-f` with a comma list and an
/// open `N-` range), char mode (`-c` range), and the GNU no-delimiter passthrough — each `echo`/`printf`
/// piped into `/bin/cut`, exec'd on the bytecode engine. Outputs are byte-verified against native
/// `cut` (`b:d` / `bcd` / `r s` / the passthrough+field / multi-line field). Adds `cut` to bash's `/bin`
/// so real pipelines like `ls | grep .txt | cut -d. -f1` run in the card.
#[test]
fn bash_dash_c_cut_pipeline() {
    let Some(bash) = load_bash() else {
        eprintln!("note: skipping bash_dash_c_cut_pipeline — no bash.temen");
        return;
    };
    let bins = load_coreutils();
    let bin_refs: Vec<(&str, &temen_ir::Module, u8)> =
        bins.iter().map(|(p, m, wl)| (*p, m, *wl)).collect();
    let script = "echo 'a:b:c:d' | cut -d: -f2,4; \
                  echo abcdef | cut -c2-4; \
                  echo 'p q r s' | cut -d' ' -f3-; \
                  printf 'nodlim\\nx:y\\n' | cut -d: -f1";
    let out = bash_exec_with(&bash, &[b"bash", b"-c", script.as_bytes()], b"", &bin_refs);
    assert_eq!(
        out.status,
        STATUS_EXIT,
        "status={} stdout={:?} stderr={:?}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    // Byte-for-byte the native `cut` output for the same four pipelines.
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "b:d\nbcd\nr s\nnodlim\nx\n"
    );
}

/// #801 — the **tier-1 + tier-2 coreutil batch** in real `bash -c` pipelines: `tail -n`, `rev`,
/// `tac`, `nl`, `fold -w`, and the args-only `basename`/`dirname` path staples, each exec'd on the
/// bytecode engine and byte-verified against the GNU shapes. Rounds out bash's `/bin` so scripts can
/// filter and re-shape streams (and munge paths) the way a real shell does.
#[test]
fn bash_dash_c_tier1_tier2_coreutils() {
    let Some(bash) = load_bash() else {
        eprintln!("note: skipping bash_dash_c_tier1_tier2_coreutils — no bash.temen");
        return;
    };
    let bins = load_coreutils();
    let bin_refs: Vec<(&str, &temen_ir::Module, u8)> =
        bins.iter().map(|(p, m, wl)| (*p, m, *wl)).collect();
    let script = "seq 1 5 | tail -n 2; \
                  echo 'x y z' | rev; \
                  printf 'a\\nb\\nc\\n' | tac; \
                  printf 'p\\nq\\n' | nl; \
                  echo abcdef | fold -w2; \
                  basename /usr/local/bin.txt .txt; \
                  dirname /usr/local/bin";
    let out = bash_exec_with(&bash, &[b"bash", b"-c", script.as_bytes()], b"", &bin_refs);
    // The final simple command (`dirname`) is exec'd on the root via bash's process-saving
    // optimization, so it *returns* its status (STATUS_OK, 0) rather than going through
    // bash's own `exit_shell` — the same contract as `bash_dash_c_external_seq_last`.
    assert_eq!(
        out.status,
        temen_browser::STATUS_OK,
        "status={} stdout={:?} stderr={:?}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "4\n5\nz y x\nc\nb\na\n     1\tp\n     2\tq\nab\ncd\nef\nbin\n/usr/local\n"
    );
}

/// #801 — the **tier-3 filesystem writers** through real bash: `tee` writes a file (and echoes),
/// `cp`/`mv` move its bytes, `cat` reads each result back, `touch` makes an empty file, `mkdir`/`rmdir`
/// create and drop a directory — all exec'd coreutils mutating bash's shared memfs. Proves the write
/// surface (`open(O_CREAT)` + `file_write`, `unlink`, `rename`, `mkdir`, `rmdir`) is reachable from a
/// real shell in the browser, not just filter pipelines.
#[test]
fn bash_dash_c_fs_writers() {
    let Some(bash) = load_bash() else {
        eprintln!("note: skipping bash_dash_c_fs_writers — no bash.temen");
        return;
    };
    let bins = load_coreutils();
    let bin_refs: Vec<(&str, &temen_ir::Module, u8)> =
        bins.iter().map(|(p, m, wl)| (*p, m, *wl)).collect();
    // tee echoes "one" and writes it to `a`; cp→b, mv b→c, cat each; touch an empty `t` (cat prints
    // nothing); rm/mkdir/rmdir mutate silently; the builtin `echo done` ends the run via exit_shell.
    let script = "echo one | tee a; cp a b; cat b; mv b c; cat c; \
                  touch t; cat t; rm a; mkdir d; rmdir d; echo done";
    let out = bash_exec_with(&bash, &[b"bash", b"-c", script.as_bytes()], b"", &bin_refs);
    assert_eq!(
        out.status,
        STATUS_EXIT,
        "status={} stdout={:?} stderr={:?}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "one\none\none\ndone\n"
    );
}

/// #1080 — a **three-stage coreutil pipeline** in the browser: `seq 5 | head -n 3 | wc -l` → `3`. Three
/// forks, three `execve`d coreutils, two CorePipes, every read park + carried pipe end + EOF + reap
/// composing under real bash on the bytecode engine — the milestone capstone.
///
/// Fixed by the same any-child-wake prune as `bash_dash_c_pipeline_echo_cat` above (see its doc) —
/// three forks, three exec'd coreutils, two CorePipes, read parks + EOF + reaps composing under real
/// bash on the bytecode engine.
#[test]
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
