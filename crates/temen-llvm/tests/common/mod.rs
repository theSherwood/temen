//! Shared harness for the `peval_*` on-ramp tests (`peval_in_sandbox.rs`, `peval_jit.rs`,
//! `peval_futamura.rs`). Each runs the manual probe — `rustc +1.81 --emit=llvm-ir` → `llvm-link-18 -S`
//! → `opt-18 -S internalize,globaldce` → translate → verify → run — on an in-repo fixture crate under
//! `tests/fixtures/<name>`. The build half is identical across them, so it lives here.
//!
//! As a `tests/common/mod.rs` submodule it is **not** compiled as its own test binary; each test does
//! `mod common;` and calls [`build_fixture_bc`].

#![allow(dead_code)] // each test binary uses only the part it needs

use std::path::{Path, PathBuf};
use std::process::Command;

/// Is `cmd <args>` runnable and successful? Used to auto-skip when a pipeline tool is absent.
fn tool_ok(cmd: &str, args: &[&str]) -> bool {
    Command::new(cmd)
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// True when the on-ramp toolchain (`rustc +1.81.0`, `llvm-link-18`, `opt-18`) is available.
pub fn toolchain_present() -> bool {
    tool_ok("rustc", &["+1.81.0", "--version"])
        && tool_ok("llvm-link-18", &["--version"])
        && tool_ok("opt-18", &["--version"])
}

/// [`toolchain_present`] plus the `rust-src` component `-Z build-std` needs to compile std from
/// source (the standard library `Cargo.toml` under the 1.81 sysroot). Gates [`build_fixture_bc_std`].
pub fn build_std_toolchain_present() -> bool {
    if !toolchain_present() {
        return false;
    }
    let Some(sysroot) = Command::new("rustc")
        .args(["+1.81.0", "--print", "sysroot"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    else {
        return false;
    };
    Path::new(&sysroot)
        .join("lib/rustlib/src/rust/library/std/Cargo.toml")
        .exists()
}

/// Build the in-repo fixture crate `tests/fixtures/<fixture>` to a single legalized textual LLVM `.ll`
/// module, ready for [`temen_llvm::translate_ll_path`]. Returns `None` (skip) if the toolchain is absent
/// or no IR is emitted.
///
/// Mirrors the manual probe exactly: emit per-crate textual IR for the whole dependency closure
/// (`RUSTFLAGS=--emit=llvm-ir cargo +1.81.0 build --release`), `llvm-link-18 -S` them, then
/// `opt-18 internalize,globaldce` down to the closure reachable from the powerbox `main`/`malloc`/
/// `free`. Building the fixture as a `lib` means no final executable link, so cargo exits cleanly even
/// though `malloc`/`free`/`write`/`__vm_jit_*` are undefined (the on-ramp synthesizes/lowers them); we
/// still tolerate a non-zero status and check for the `.bc`.
pub fn build_fixture_bc(fixture: &str) -> Option<PathBuf> {
    if !toolchain_present() {
        eprintln!("note: skipping {fixture} (need `rustc +1.81.0`, `llvm-link-18`, `opt-18`)");
        return None;
    }

    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(fixture);
    // A dedicated, out-of-tree target dir keeps the repo clean and isolates the bitcode artifacts.
    let work = std::env::temp_dir().join(format!("{fixture}_{}", std::process::id()));
    let target = work.join("target");
    std::fs::create_dir_all(&target).expect("create target dir");

    let status = Command::new("cargo")
        .current_dir(&fixture_dir)
        .env("RUSTFLAGS", "--emit=llvm-ir")
        .env("CARGO_TARGET_DIR", &target)
        .args(["+1.81.0", "build", "--release", "--ignore-rust-version"])
        .status()
        .unwrap_or_else(|e| panic!("run cargo build for the {fixture} fixture: {e}"));
    if !status.success() {
        eprintln!("note: {fixture} `cargo build` returned {status} (tolerated if .ll emitted)");
    }

    let deps = target.join("release/deps");
    let mut lls: Vec<PathBuf> = std::fs::read_dir(&deps)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "ll").unwrap_or(false))
        .collect();
    lls.sort();
    if lls.is_empty() {
        eprintln!("note: skipping {fixture} (no .ll emitted — build failed before codegen)");
        return None;
    }

    // Merge + prune as textual `.ll` (`-S`), so translation goes through the version-tolerant textual
    // reader (`translate_ll_path`) rather than the `llvm-dis` bitcode shim.
    let linked = work.join("linked.ll");
    assert!(
        Command::new("llvm-link-18")
            .arg("-S")
            .args(&lls)
            .arg("-o")
            .arg(&linked)
            .status()
            .expect("run llvm-link-18")
            .success(),
        "llvm-link-18 failed"
    );

    let legalized = work.join("legalized.ll");
    assert!(
        Command::new("opt-18")
            .args([
                "-S",
                "-passes=internalize,globaldce",
                "-internalize-public-api-list=main,malloc,free",
            ])
            .arg(&linked)
            .arg("-o")
            .arg(&legalized)
            .status()
            .expect("run opt-18")
            .success(),
        "opt-18 failed"
    );
    Some(legalized)
}

/// [`build_fixture_bc`], but with **`std` compiled from source** via `-Z build-std` (W5 §3e). The
/// plain [`build_fixture_bc`] leaves libcore/liballoc symbols (`fmt`/`FromStr`/panic) *external* — fine
/// for a translate-only test, but a real *run* would trap in a stub. This variant makes them real:
/// `RUSTC_BOOTSTRAP=1` unlocks `-Z build-std` on **stable 1.81** (whose LLVM 18 matches the
/// `llvm-*-18` tools), and `build-std-features=panic_immediate_abort` collapses every panic to a bare
/// `abort` so no float-formatting (`flt2dec`/dragon4) is pulled in for panic messages. The result links
/// to a module whose only externals are the allocator + `bcmp` — all synthesized/recognized by the
/// on-ramp — so `translate_to_text` runs to a correct result in the sandbox.
///
/// Returns `None` (skip) if the `-Z build-std` toolchain (incl. the `rust-src` component) is absent.
pub fn build_fixture_bc_std(fixture: &str) -> Option<PathBuf> {
    if !build_std_toolchain_present() {
        eprintln!(
            "note: skipping {fixture} (need `rustc +1.81.0` + `rust-src`, `llvm-link-18`, `opt-18`)"
        );
        return None;
    }

    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(fixture);
    let work = std::env::temp_dir().join(format!("{fixture}_std_{}", std::process::id()));
    let target = work.join("target");
    std::fs::create_dir_all(&target).expect("create target dir");

    // `RUSTC_BOOTSTRAP=1` = `-Z` on stable; `panic_immediate_abort` keeps the float formatter out of
    // the closure. `--target` (even the host triple) is required for `build-std` to take effect.
    const TRIPLE: &str = "x86_64-unknown-linux-gnu";
    let status = Command::new("cargo")
        .current_dir(&fixture_dir)
        .env("RUSTFLAGS", "--emit=llvm-ir")
        .env("CARGO_TARGET_DIR", &target)
        .env("RUSTC_BOOTSTRAP", "1")
        .args([
            "+1.81.0",
            "build",
            "--release",
            "-Zbuild-std=std,panic_abort",
            "-Zbuild-std-features=panic_immediate_abort",
            "--target",
            TRIPLE,
            "--ignore-rust-version",
        ])
        .status()
        .unwrap_or_else(|e| panic!("run build-std cargo build for the {fixture} fixture: {e}"));
    if !status.success() {
        eprintln!(
            "note: {fixture} build-std `cargo build` returned {status} (tolerated if .ll emitted)"
        );
    }

    // build-std artifacts land under `<triple>/release/deps` (std/core/alloc + the crate closure).
    let deps = target.join(TRIPLE).join("release/deps");
    let mut lls: Vec<PathBuf> = std::fs::read_dir(&deps)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "ll").unwrap_or(false))
        // `panic_unwind` and `panic_abort` both define `__rust_panic_cleanup`; with `panic=abort` the
        // program only needs `panic_abort`, so drop `panic_unwind` to avoid a duplicate-symbol link error.
        .filter(|p| {
            !p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("panic_unwind"))
        })
        .collect();
    lls.sort();
    if lls.is_empty() {
        eprintln!("note: skipping {fixture} (no .ll emitted — build-std failed before codegen)");
        return None;
    }

    let linked = work.join("linked.ll");
    assert!(
        Command::new("llvm-link-18")
            .arg("-S")
            .args(&lls)
            .arg("-o")
            .arg(&linked)
            .status()
            .expect("run llvm-link-18")
            .success(),
        "llvm-link-18 failed"
    );

    let legalized = work.join("legalized.ll");
    assert!(
        Command::new("opt-18")
            .args([
                "-S",
                "-passes=internalize,globaldce",
                "-internalize-public-api-list=main,malloc,free",
            ])
            .arg(&linked)
            .arg("-o")
            .arg(&legalized)
            .status()
            .expect("run opt-18")
            .success(),
        "opt-18 failed"
    );
    Some(legalized)
}
