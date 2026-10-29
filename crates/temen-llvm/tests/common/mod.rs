//! Shared harness for the Rust on-ramp probes (`peval_in_sandbox.rs`, `peval_jit.rs`,
//! `peval_futamura.rs`, `w5_leng_*.rs`). Each runs the manual probe — `rustc --emit=llvm-ir` under
//! `-Z build-std` → `llvm-link -S` → `opt -S internalize,globaldce` → translate → verify → run — on
//! an in-repo fixture crate under `tests/fixtures/<name>`. The build half is identical across them,
//! so it lives here.
//!
//! As a `tests/common/mod.rs` submodule it is **not** compiled as its own test binary; each test does
//! `mod common;` and calls [`build_fixture_bc`].

#![allow(dead_code)] // each test binary uses only the part it needs

use std::path::{Path, PathBuf};
use std::process::Command;
use temen_dap::Json;

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

/// True when the on-ramp toolchain (`rustc`, `llvm-link`, `opt`) is available. The default `rustc`'s
/// LLVM must not be newer than `llvm-link`/`opt` (they ingest IR of their own version or older) — CI
/// pins the two majors equal (`scripts/ci/install-llvm.sh`, checked by `ci_tool_canary`).
pub fn toolchain_present() -> bool {
    tool_ok("rustc", &["--version"])
        && tool_ok("llvm-link", &["--version"])
        && tool_ok("opt", &["--version"])
}

/// [`toolchain_present`] plus the `rust-src` component `-Z build-std` needs to compile std from
/// source (the standard library `Cargo.toml` under the toolchain's sysroot). Gates [`build_fixture_bc`].
pub fn build_std_toolchain_present() -> bool {
    if !toolchain_present() {
        return false;
    }
    let Some(sysroot) = Command::new("rustc")
        .args(["--print", "sysroot"])
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
/// module, ready for [`temen_llvm::translate_ll_path`]. Returns `None` (skip) if the toolchain is
/// absent; with the toolchain present, a build that fails is a test failure, never a skip (#2696).
///
/// Mirrors the manual probe exactly, with **`std` compiled from source** via `-Z build-std` (W5 §3e):
/// emit per-crate textual IR for the whole closure — the fixture, its temen crates, *and*
/// core/alloc/std (`RUSTFLAGS=--emit=llvm-ir cargo build -Zbuild-std`), `llvm-link -S` them, then
/// `opt internalize,globaldce` down to the closure reachable from the powerbox `main`/`malloc`/`free`.
/// Building std from source is what makes the closure complete: `--emit=llvm-ir` covers only the crates
/// cargo compiles, and a modern rustc leaves real code in the precompiled sysroot (since 1.83 `Vec`
/// growth is the non-generic `RawVecInner::grow_one` inside `liballoc`; `String::from_utf8_lossy`
/// reaches `libcore`'s `Utf8Chunks`; hashbrown, …), which no stub can stand in for — and the sysroot
/// rlibs' embedded bitcode is `panic=unwind` code. `-Cpanic=immediate-abort` keeps the float formatter
/// out of the closure. Building the fixture as a `lib` means no final executable link, so cargo exits
/// cleanly even though `malloc`/`free`/`write`/`__vm_jit_*` are undefined (the on-ramp
/// synthesizes/lowers them).
///
/// The std build lands in one **shared** target dir (`$TMPDIR/temen_llvm_buildstd_target`), so
/// core/alloc/std compile once per machine/CI run and every fixture reuses them (cargo fingerprints
/// them like any dependency; its target-dir lock serializes concurrent tests). That dir holds every
/// fixture's crates, so the link takes only the ones this build reports, never the whole dir.
///
/// Needs the `rust-src` component (`build_std_toolchain_present`; the CI `temen-llvm` job installs it).
pub fn build_fixture_bc(fixture: &str) -> Option<PathBuf> {
    if !build_std_toolchain_present() {
        eprintln!("note: skipping {fixture} (need `rustc` + `rust-src`, `llvm-link`, `opt`)");
        return None;
    }

    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(fixture);
    let work = std::env::temp_dir().join(format!("{fixture}_std_{}", std::process::id()));
    let target = std::env::temp_dir().join("temen_llvm_buildstd_target");
    std::fs::create_dir_all(&work).expect("create work dir");
    std::fs::create_dir_all(&target).expect("create target dir");

    // `RUSTC_BOOTSTRAP=1` = `-Z` on stable; `-Cpanic=immediate-abort` keeps the float formatter out of
    // the closure. It is not the `panic_immediate_abort` std feature: rustc promoted that to a real
    // panic strategy, and the old spelling is now a hard `compile_error!` in `core`. `--target` (even
    // the host triple) is required for `build-std` to take effect.
    const TRIPLE: &str = "x86_64-unknown-linux-gnu";
    let out = Command::new("cargo")
        .current_dir(&fixture_dir)
        .env(
            "RUSTFLAGS",
            "--emit=llvm-ir -Zunstable-options -Cpanic=immediate-abort",
        )
        .env("CARGO_TARGET_DIR", &target)
        .env("RUSTC_BOOTSTRAP", "1")
        .args([
            "build",
            "--release",
            "-Zbuild-std=std,panic_abort",
            "--target",
            TRIPLE,
            "--ignore-rust-version",
            "--message-format=json-render-diagnostics",
        ])
        .stderr(std::process::Stdio::inherit())
        .output()
        .unwrap_or_else(|e| panic!("run build-std cargo build for the {fixture} fixture: {e}"));
    assert!(
        out.status.success(),
        "{fixture}: build-std `cargo build` failed ({})",
        out.status
    );

    // The crates to link: each one this build compiled or found fresh, by cargo's `compiler-artifact`
    // messages. The fixture's own crate goes last, as `--override`.
    let deps = target.join(TRIPLE).join("release/deps");
    let manifest = fixture_dir.join("Cargo.toml");
    let mut own = None;
    let mut lls: Vec<PathBuf> = Vec::new();
    for msg in String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(temen_dap::parse)
        .filter(|m| m.get("reason").and_then(Json::as_str) == Some("compiler-artifact"))
    {
        let rlibs = msg
            .get("filenames")
            .and_then(Json::as_array)
            .unwrap_or_default()
            .iter()
            .filter_map(Json::as_str)
            .filter(|f| f.ends_with(".rlib"));
        for rlib in rlibs {
            let ll = ll_of(Path::new(rlib), &deps);
            if msg
                .get("manifest_path")
                .and_then(Json::as_str)
                .map(Path::new)
                == Some(&*manifest)
            {
                own = Some(ll);
            } else {
                lls.push(ll);
            }
        }
    }
    let own = own.unwrap_or_else(|| panic!("{fixture}: cargo reported no crate for {manifest:?}"));
    // `panic_unwind` and `panic_abort` both define `__rust_panic_cleanup`; with `panic=abort` the
    // program only needs `panic_abort`, so drop `panic_unwind` to avoid a duplicate-symbol link error.
    lls.retain(|p| {
        !p.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("panic_unwind"))
    });
    lls.sort();
    for ll in lls.iter().chain([&own]) {
        assert!(ll.exists(), "{fixture}: rustc emitted no {ll:?}");
    }

    // The fixture's own module overrides what the rest of the closure also defines: a `no_std`
    // fixture's `#[panic_handler]` replaces std's, which `-Zbuild-std=std` builds for every fixture.
    let linked = work.join("linked.ll");
    assert!(
        Command::new("llvm-link")
            .arg("-S")
            .args(&lls)
            .arg("--override")
            .arg(&own)
            .arg("-o")
            .arg(&linked)
            .status()
            .expect("run llvm-link")
            .success(),
        "llvm-link failed (is it older than `rustc -vV`'s LLVM? CI pins the two equal)"
    );

    let legalized = work.join("legalized.ll");
    assert!(
        Command::new("opt")
            .args([
                "-S",
                "-passes=internalize,globaldce",
                "-internalize-public-api-list=main,malloc,free",
            ])
            .arg(&linked)
            .arg("-o")
            .arg(&legalized)
            .status()
            .expect("run opt")
            .success(),
        "opt failed"
    );
    Some(legalized)
}

/// The `.ll` rustc emitted for the crate whose rlib cargo reported as `rlib`. That is `deps`'s
/// `lib<name>-<hash>.rlib`, whose IR is `<name>-<hash>.ll` beside it, except for the fixture's own
/// crate: cargo reports it by its uplifted copy, `lib<name>.rlib` outside `deps`, and its twin in
/// `deps` is the `lib<name>-<hash>.rlib` with the same bytes.
fn ll_of(rlib: &Path, deps: &Path) -> PathBuf {
    let rlib = if rlib.parent() == Some(deps) {
        rlib.to_path_buf()
    } else {
        let bytes = std::fs::read(rlib).unwrap_or_else(|e| panic!("read {rlib:?}: {e}"));
        let prefix = format!(
            "{}-",
            rlib.file_stem()
                .and_then(|s| s.to_str())
                .expect("rlib name")
        );
        std::fs::read_dir(deps)
            .expect("read deps")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .find(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(&prefix) && n.ends_with(".rlib"))
                    && std::fs::read(p).is_ok_and(|b| b == bytes)
            })
            .unwrap_or_else(|| panic!("no rlib in {deps:?} has the bytes of {rlib:?}"))
    };
    let stem = rlib
        .file_stem()
        .and_then(|s| s.to_str())
        .expect("rlib name");
    deps.join(format!("{}.ll", stem.strip_prefix("lib").unwrap_or(stem)))
}
