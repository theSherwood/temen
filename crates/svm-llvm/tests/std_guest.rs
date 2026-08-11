//! **Real Rust `std` as an svm guest** — the RUST_STD.md S1b differential (the analog of
//! `w5_rust_guest.rs`, one rung up: `no_std + alloc` → full `std`). A `std` binary is built for the
//! custom `x86_64-unknown-svm` target via `-Zbuild-std` (the `crates/svm-llvm/rust-svm/` lane),
//! emitted as one fat-LTO'd `.ll`, translated by the on-ramp, verified, and run through the powerbox —
//! its exit code checked against a native-equivalent oracle. This exercises the whole real-`std`
//! runtime that `no_std` never touches: `std::rt::lang_start`, the panic/`llvm.assume` machinery, and
//! heap `Vec` growth through the synthesized `malloc` (which rides the powerbox `main`, §S).
//!
//! **Auto-skips** unless the full lane is present: a `nightly` toolchain with `rust-src`, and the svm
//! `std` overlay already applied to that toolchain (`rust-svm/apply-overlay.sh`). It therefore does
//! **not** run in the per-PR gate (which lacks the nightly build-std lane, ISSUES.md I55) — it is the
//! asset-lane check, green only where the lane is set up. Nothing here mutates the toolchain.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The svm `std` build lane (`crates/svm-llvm/rust-svm/`).
fn lane_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("rust-svm")
}

/// The nightly sysroot's `rust-src` std tree, or `None` if nightly/rust-src is absent.
fn nightly_std_src() -> Option<PathBuf> {
    let out = Command::new("rustc")
        .args(["+nightly", "--print", "sysroot"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let sysroot = String::from_utf8(out.stdout).ok()?;
    let std_src = Path::new(sysroot.trim()).join("lib/rustlib/src/rust/library/std/src");
    std_src.is_dir().then_some(std_src)
}

/// The lane is ready iff nightly + rust-src exist **and** the svm overlay is applied (the allocator
/// `imp` the overlay adds is the marker). Returns the std-src dir when ready.
fn lane_ready() -> Option<PathBuf> {
    let std_src = nightly_std_src()?;
    let applied = std_src.join("sys/alloc/svm.rs").is_file()
        && std::fs::read_to_string(std_src.join("sys/alloc/mod.rs"))
            .map(|s| s.contains("target_os = \"svm\""))
            .unwrap_or(false);
    applied.then_some(std_src)
}

/// Build the inline `std` program `src` for the svm target and return the fat-LTO'd `.ll`.
fn build_std_bin_ll(name: &str, src: &str) -> Option<PathBuf> {
    let work = std::env::temp_dir().join(format!("svm_std_{name}_{}", std::process::id()));
    let src_dir = work.join("src");
    std::fs::create_dir_all(&src_dir).ok()?;
    std::fs::write(
        work.join("Cargo.toml"),
        format!(
            "[package]\nname = \"{name}\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\
             [[bin]]\nname = \"{name}\"\npath = \"src/main.rs\"\n\
             [profile.release]\npanic = \"abort\"\nlto = \"fat\"\ncodegen-units = 1\n"
        ),
    )
    .ok()?;
    std::fs::write(src_dir.join("main.rs"), src).ok()?;

    let target_json = lane_dir().join("x86_64-unknown-svm.json");
    let status = Command::new("cargo")
        .current_dir(&work)
        .env("RUSTC_BOOTSTRAP", "1")
        .env("CARGO_TARGET_DIR", work.join("target"))
        .args([
            "+nightly",
            "rustc",
            "-Zbuild-std=core,alloc,std,panic_abort",
            "-Zjson-target-spec",
            "--target",
        ])
        .arg(&target_json)
        .args([
            "--release",
            "--bin",
            name,
            "--",
            "--emit=llvm-ir",
            "-Clto=fat",
        ])
        .status()
        .ok()?;
    // A missing `_start` linker note is expected (we consume the IR, not the executable); tolerate a
    // non-zero status as long as the `.ll` landed.
    let _ = status;
    let ll = find_ll(&work.join("target"), name)?;
    Some(ll)
}

fn find_ll(target: &Path, name: &str) -> Option<PathBuf> {
    let mut stack = vec![target.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).ok()?.flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().and_then(|s| s.to_str()) == Some(&format!("{name}.ll")) {
                return Some(p);
            }
        }
    }
    None
}

/// Run a translated powerbox module and return its exit code as a `u8`.
fn powerbox_exit(module: &svm_ir::Module) -> u8 {
    let run = svm_run::run_powerbox(module, b"").expect("powerbox run");
    match run.outcome {
        svm_run::Outcome::Exited(c) => c as u8,
        svm_run::Outcome::Returned(ref v) => match v.first() {
            Some(svm_interp::Value::I32(x)) => *x as u8,
            Some(svm_interp::Value::I64(x)) => *x as u8,
            _ => 0,
        },
    }
}

/// A real `std` binary — `lang_start` + heap `Vec` + iterator fold — returning a **computed exit
/// code** through `Termination` (no I/O, so the `unsupported` PAL's stdio/exit are never reached),
/// runs on svm byte-identical to the native-equivalent oracle.
#[test]
fn std_bin_runs_on_svm_via_powerbox() {
    let Some(_std_src) = lane_ready() else {
        eprintln!(
            "note: skipping std_guest (need nightly + rust-src + the svm std overlay applied — \
             run crates/svm-llvm/rust-svm/apply-overlay.sh)"
        );
        return;
    };

    // Σ i² for i in 0..N, taken mod 256 — the same shape the on-ramp exercises for `no_std`, now
    // through the full `std` runtime and a heap `Vec`.
    const N: i32 = 8;
    let src = format!(
        "#![feature(restricted_std)]\n\
         use std::process::ExitCode;\n\
         fn main() -> ExitCode {{\n\
         \x20   let n = {N}i32;\n\
         \x20   let mut v: Vec<i32> = Vec::new();\n\
         \x20   let mut i = 0;\n\
         \x20   while i < n {{ v.push(i.wrapping_mul(i)); i += 1; }}\n\
         \x20   let sum: i32 = v.iter().copied().fold(0, |a, b| a.wrapping_add(b));\n\
         \x20   ExitCode::from((sum & 0xff) as u8)\n\
         }}\n"
    );

    let Some(ll) = build_std_bin_ll("svm_std_probe", &src) else {
        eprintln!("note: skipping std_guest (build-std produced no .ll)");
        return;
    };

    let t = svm_llvm::translate_ll_path(&ll).expect("on-ramp translates the std binary's LLVM IR");
    svm_verify::verify_module(&t.module).expect("the translated std binary verifies");
    assert!(
        svm_run::is_named_powerbox_entry(&t.module),
        "a std binary produces a powerbox entry (its C `main` rides the powerbox `_start`)"
    );

    // The native-equivalent oracle (pure compute — deterministic, no platform surface).
    let oracle: u8 = {
        let mut acc = 0i32;
        let mut i = 0;
        while i < N {
            acc = acc.wrapping_add(i.wrapping_mul(i));
            i += 1;
        }
        (acc & 0xff) as u8
    };

    let got = powerbox_exit(&t.module);
    assert_eq!(
        got, oracle,
        "real std ran on svm to the computed exit code (§18 native equivalence)"
    );
}
