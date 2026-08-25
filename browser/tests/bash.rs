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

use temen_browser::{bash_exec, STATUS_EXIT};

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
