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
    build_std_bin_ll_target(name, src, "x86_64-unknown-svm.json")
}

/// Build `src` for a specific svm target spec (the lean `x86_64-unknown-svm.json` or the threaded
/// `x86_64-unknown-svm-threads.json`) and return the fat-LTO'd `.ll`. The work dir is keyed on the
/// target stem so a lean and a threaded build of the same program don't clobber each other.
fn build_std_bin_ll_target(name: &str, src: &str, target_file: &str) -> Option<PathBuf> {
    let stem = Path::new(target_file)
        .file_stem()
        .and_then(|s| s.to_str())?;
    let work = std::env::temp_dir().join(format!("svm_std_{name}_{stem}_{}", std::process::id()));
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

    let target_json = lane_dir().join(target_file);
    let mut child = Command::new("cargo")
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
        .spawn()
        .ok()?;
    // #788 wedge guard: a `rustc`/`cargo` that hangs mid-`build-std` must not stall the whole (serial)
    // job to the 6-hour ceiling. Bound each build and kill a wedged child, returning `None` (skip)
    // instead of blocking forever. Override the budget with `SVM_STD_BUILD_TIMEOUT_SECS`.
    let budget = std::env::var("SVM_STD_BUILD_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(600);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(budget);
    loop {
        match child.try_wait() {
            // A missing `_start` linker note is expected (we consume the IR, not the executable); a
            // non-zero status is tolerated as long as the `.ll` landed (checked below).
            Ok(Some(_)) => break,
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    eprintln!(
                        "note: skipping std_guest ({target_file}): build-std exceeded {budget}s \
                         (possible #788 rustc wedge)"
                    );
                    return None;
                }
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
            Err(_) => return None,
        }
    }
    let ll = find_ll(&work.join("target"), name)?;
    // Free the per-test build tree now. Each test does a full `build-std` (~hundreds of MB of target
    // artifacts); run serially across the whole suite that exhausts the runner's disk allowance —
    // ENOSPC mid-`build-std` on the first full CI run, which the harness would then silently skip.
    // The fat-LTO'd `.ll` is self-contained (the on-ramp reads only this one file), so copy it out to
    // a small standalone path and drop the tree, bounding peak disk to a single build at a time.
    let out_ll =
        std::env::temp_dir().join(format!("svm_std_{name}_{stem}_{}.ll", std::process::id()));
    std::fs::copy(&ll, &out_ll).ok()?;
    let _ = std::fs::remove_dir_all(&work);
    Some(out_ll)
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

/// Translate + verify + run `src` on the powerbox **with a granted `posix` cap** (`run_with_caps`),
/// letting `seed` stage the personality (env, a pinned clock, a seeded memfs, …) → `(stdout, exit)`.
/// This is the richer-`std::sys` path (`time`/`env`/`fs`) that reaches the host via `__vm_host_call`.
fn svm_run_std_posix(
    name: &str,
    src: &str,
    seed: impl FnOnce(&svm_posix::Posix),
) -> Option<(Vec<u8>, u8)> {
    let ll = build_std_bin_ll(name, src)?;
    let t = svm_llvm::translate_ll_path(&ll).expect("on-ramp translates the std binary's LLVM IR");
    svm_verify::verify_module(&t.module).expect("the translated std binary verifies");
    let (cap, posix) = svm_run::posix::posix_cap(0, 0, Vec::new());
    seed(&posix);
    let out = svm_run::instantiate(t.module)
        .expect("instantiate")
        .run_with_caps(
            svm_run::Backend::Jit,
            &svm_run::RunConfig::default(),
            &[("posix", cap)],
        )
        .expect("run_with_caps");
    let exit = match out.outcome {
        svm_run::Outcome::Exited(c) => c as u8,
        svm_run::Outcome::Returned(ref v) => match v.first() {
            Some(svm_interp::Value::I32(x)) => *x as u8,
            Some(svm_interp::Value::I64(x)) => *x as u8,
            _ => 0,
        },
    };
    Some((out.stdout, exit))
}

/// [`svm_run_std_posix`] with the **`net` capability** granted alongside `posix` (POSIX.md §5a) —
/// the `std::net` path: loopback rides the in-personality memnet; `seed` may wire a `NetDelegate`
/// (`set_net`) for scripted egress.
fn svm_run_std_net(
    name: &str,
    src: &str,
    seed: impl FnOnce(&svm_posix::Posix),
) -> Option<(Vec<u8>, u8)> {
    let ll = build_std_bin_ll(name, src)?;
    let t = svm_llvm::translate_ll_path(&ll).expect("on-ramp translates the std binary's LLVM IR");
    svm_verify::verify_module(&t.module).expect("the translated std binary verifies");
    let (cap, posix) = svm_run::posix::posix_cap(0, 0, Vec::new());
    let net = svm_run::posix::net_cap(&posix);
    seed(&posix);
    let out = svm_run::instantiate(t.module)
        .expect("instantiate")
        .run_with_caps(
            svm_run::Backend::Jit,
            &svm_run::RunConfig::default(),
            &[("posix", cap), ("net", net)],
        )
        .expect("run_with_caps");
    let exit = match out.outcome {
        svm_run::Outcome::Exited(c) => c as u8,
        svm_run::Outcome::Returned(ref v) => match v.first() {
            Some(svm_interp::Value::I32(x)) => *x as u8,
            Some(svm_interp::Value::I64(x)) => *x as u8,
            _ => 0,
        },
    };
    Some((out.stdout, exit))
}

/// Translate + verify + run `src` as a std guest on the powerbox → `(stdout, exit code)`.
fn svm_run_std(name: &str, src: &str) -> Option<(Vec<u8>, u8)> {
    let ll = build_std_bin_ll(name, src)?;
    let t = svm_llvm::translate_ll_path(&ll).expect("on-ramp translates the std binary's LLVM IR");
    svm_verify::verify_module(&t.module).expect("the translated std binary verifies");
    assert!(
        svm_run::is_named_powerbox_entry(&t.module),
        "a std binary produces a powerbox entry (its C `main` rides the powerbox `_start`)"
    );
    let run = svm_run::run_powerbox(&t.module, b"").expect("powerbox run");
    let exit = match run.outcome {
        svm_run::Outcome::Exited(c) => c as u8,
        svm_run::Outcome::Returned(ref v) => match v.first() {
            Some(svm_interp::Value::I32(x)) => *x as u8,
            Some(svm_interp::Value::I64(x)) => *x as u8,
            _ => 0,
        },
    };
    Some((run.stdout, exit))
}

/// Translate + verify + run `src` built for the **threaded** target (`x86_64-unknown-svm-threads`,
/// `singlethread=false` → real atomic-ordering codegen, futex-backed `sys/sync`, native TLS) →
/// `(stdout, exit code)`. The program runs on a single vCPU (no `thread::spawn` yet — that is the
/// #823 follow-up), so this proves the threaded-`std` codegen builds, translates, verifies, and runs
/// identically to the lean spec's oracle.
fn svm_run_std_threads(name: &str, src: &str) -> Option<(Vec<u8>, u8)> {
    let ll = build_std_bin_ll_target(name, src, "x86_64-unknown-svm-threads.json")?;
    let t = svm_llvm::translate_ll_path(&ll)
        .expect("on-ramp translates the threaded std binary's LLVM IR");
    svm_verify::verify_module(&t.module).expect("the translated threaded std binary verifies");
    assert!(
        svm_run::is_named_powerbox_entry(&t.module),
        "a threaded std binary produces a powerbox entry (its C `main` rides the powerbox `_start`)"
    );
    let run = svm_run::run_powerbox(&t.module, b"").expect("powerbox run");
    let exit = match run.outcome {
        svm_run::Outcome::Exited(c) => c as u8,
        svm_run::Outcome::Returned(ref v) => match v.first() {
            Some(svm_interp::Value::I32(x)) => *x as u8,
            Some(svm_interp::Value::I64(x)) => *x as u8,
            _ => 0,
        },
    };
    Some((run.stdout, exit))
}

/// Translate + verify + run `src` built for the **threaded** target with the **`posix` + `net`** caps
/// granted (`run_with_caps`) → `(stdout, exit code)`. This is the seam for genuinely-concurrent I/O:
/// several vCPUs share one personality (its `Inner` is behind a host-side `Mutex`), so a second thread
/// can drive the memnet peer of a socket the first thread blocks (retries) on.
fn svm_run_std_threads_net(name: &str, src: &str) -> Option<(Vec<u8>, u8)> {
    let ll = build_std_bin_ll_target(name, src, "x86_64-unknown-svm-threads.json")?;
    let t = svm_llvm::translate_ll_path(&ll)
        .expect("on-ramp translates the threaded std binary's LLVM IR");
    svm_verify::verify_module(&t.module).expect("the translated threaded std binary verifies");
    let (cap, posix) = svm_run::posix::posix_cap(0, 0, Vec::new());
    let net = svm_run::posix::net_cap(&posix);
    let out = svm_run::instantiate(t.module)
        .expect("instantiate")
        .run_with_caps(
            svm_run::Backend::Jit,
            &svm_run::RunConfig::default(),
            &[("posix", cap), ("net", net)],
        )
        .expect("run_with_caps");
    let exit = match out.outcome {
        svm_run::Outcome::Exited(c) => c as u8,
        svm_run::Outcome::Returned(ref v) => match v.first() {
            Some(svm_interp::Value::I32(x)) => *x as u8,
            Some(svm_interp::Value::I64(x)) => *x as u8,
            _ => 0,
        },
    };
    Some((out.stdout, exit))
}

/// Translate + verify + run `src` (built for the **threaded** target) under the **parallel** driver
/// (`run_with_caps_parallel`, THREADS.md 4c) → `(stdout, exit code)`. Unlike `svm_run_std_threads`
/// (the deterministic cooperative oracle), each vCPU runs on its own OS thread over one shared window,
/// so the *schedule* is nondeterministic — this is **smoke** coverage: the observable outcome must
/// still match the oracle even though the interleaving is real. Used to prove the same threaded `std`
/// programs the cooperative entries pin exactly also run correctly on real OS threads.
fn svm_run_std_threads_parallel(name: &str, src: &str) -> Option<(Vec<u8>, u8)> {
    let ll = build_std_bin_ll_target(name, src, "x86_64-unknown-svm-threads.json")?;
    let t = svm_llvm::translate_ll_path(&ll)
        .expect("on-ramp translates the threaded std binary's LLVM IR");
    svm_verify::verify_module(&t.module).expect("the translated threaded std binary verifies");
    assert!(
        svm_run::is_named_powerbox_entry(&t.module),
        "a threaded std binary produces a powerbox entry (its C `main` rides the powerbox `_start`)"
    );
    let out = svm_run::instantiate(t.module)
        .expect("instantiate")
        .run_with_caps_parallel(&svm_run::RunConfig::default(), &[])
        .expect("run_with_caps_parallel");
    let exit = match out.outcome {
        svm_run::Outcome::Exited(c) => c as u8,
        svm_run::Outcome::Returned(ref v) => match v.first() {
            Some(svm_interp::Value::I32(x)) => *x as u8,
            Some(svm_interp::Value::I64(x)) => *x as u8,
            _ => 0,
        },
    };
    Some((out.stdout, exit))
}

/// Translate + verify + run `src` on the powerbox **with argv** → `(stdout, exit code)`.
/// `argv[0]` is the program name, as in a native run.
fn svm_run_std_with_args(name: &str, src: &str, argv: &[&[u8]]) -> Option<(Vec<u8>, u8)> {
    let ll = build_std_bin_ll(name, src)?;
    let t = svm_llvm::translate_ll_path(&ll).expect("on-ramp translates the std binary's LLVM IR");
    svm_verify::verify_module(&t.module).expect("the translated std binary verifies");
    let run = svm_run::run_powerbox_with_args(&t.module, b"", argv, &[]).expect("powerbox run");
    let exit = match run.outcome {
        svm_run::Outcome::Exited(c) => c as u8,
        svm_run::Outcome::Returned(ref v) => match v.first() {
            Some(svm_interp::Value::I32(x)) => *x as u8,
            Some(svm_interp::Value::I64(x)) => *x as u8,
            _ => 0,
        },
    };
    Some((run.stdout, exit))
}

/// Build + run the **native-equivalent oracle** with the given extra args (`argv[0]` is supplied by
/// the OS, so `extra_args` are `argv[1..]`). Returns `(stdout, exit code)`, or `None` if host `rustc`
/// is absent.
fn native_oracle_args(name: &str, src: &str, extra_args: &[&str]) -> Option<(Vec<u8>, u8)> {
    let host_src = src.replace("#![feature(restricted_std)]\n", "");
    let dir = std::env::temp_dir().join(format!("svm_std_native_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok()?;
    let rs = dir.join("main.rs");
    let bin = dir.join("oracle");
    std::fs::write(&rs, host_src).ok()?;
    let built = Command::new("rustc")
        .args(["--edition", "2021", "-O"])
        .arg(&rs)
        .arg("-o")
        .arg(&bin)
        .status()
        .ok()?
        .success();
    if !built {
        return None;
    }
    let out = Command::new(&bin).args(extra_args).output().ok()?;
    Some((out.stdout, out.status.code().unwrap_or(-1) as u8))
}

/// The no-args oracle (`argv` = just the program name).
fn native_oracle(name: &str, src: &str) -> Option<(Vec<u8>, u8)> {
    native_oracle_args(name, src, &[])
}

/// Run `src` on the powerbox → `(stdout, stderr, exit)` — the stderr-separating variant.
fn svm_run_std_streams(name: &str, src: &str) -> Option<(Vec<u8>, Vec<u8>, u8)> {
    let ll = build_std_bin_ll(name, src)?;
    let t = svm_llvm::translate_ll_path(&ll).expect("on-ramp translates the std binary's LLVM IR");
    svm_verify::verify_module(&t.module).expect("the translated std binary verifies");
    let run = svm_run::run_powerbox(&t.module, b"").expect("powerbox run");
    let exit = match run.outcome {
        svm_run::Outcome::Exited(c) => c as u8,
        svm_run::Outcome::Returned(ref v) => match v.first() {
            Some(svm_interp::Value::I32(x)) => *x as u8,
            Some(svm_interp::Value::I64(x)) => *x as u8,
            _ => 0,
        },
    };
    Some((run.stdout, run.stderr, exit))
}

/// Native oracle returning stdout and stderr **separately**.
fn native_oracle_streams(name: &str, src: &str) -> Option<(Vec<u8>, Vec<u8>, u8)> {
    let host_src = src.replace("#![feature(restricted_std)]\n", "");
    let dir = std::env::temp_dir().join(format!("svm_std_natst_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok()?;
    let rs = dir.join("main.rs");
    let bin = dir.join("oracle");
    std::fs::write(&rs, host_src).ok()?;
    let built = Command::new("rustc")
        .args(["--edition", "2021", "-O"])
        .arg(&rs)
        .arg("-o")
        .arg(&bin)
        .status()
        .ok()?
        .success();
    if !built {
        return None;
    }
    let out = Command::new(&bin).output().ok()?;
    Some((
        out.stdout,
        out.stderr,
        out.status.code().unwrap_or(-1) as u8,
    ))
}

/// S1b — a real `std` binary (`lang_start` + heap `Vec` + iterator fold) returning a **computed exit
/// code** through `Termination` runs on svm byte-identical to the native-equivalent oracle. This is
/// pure compute (no I/O), so it passes even with only the `unsupported` PAL for stdio.
#[test]
fn std_bin_runs_on_svm_via_powerbox() {
    if lane_ready().is_none() {
        eprintln!(
            "note: skipping std_guest (need nightly + rust-src + the svm std overlay applied — \
             run crates/svm-llvm/rust-svm/apply-overlay.sh)"
        );
        return;
    }

    // Σ i² for i in 0..8, mod 256 — the on-ramp's `no_std` shape, now through the full `std` runtime.
    let src = "#![feature(restricted_std)]\n\
         use std::process::ExitCode;\n\
         fn main() -> ExitCode {\n\
         \x20   let mut v: Vec<i32> = Vec::new();\n\
         \x20   let mut i = 0i32;\n\
         \x20   while i < 8 { v.push(i.wrapping_mul(i)); i += 1; }\n\
         \x20   let sum: i32 = v.iter().copied().fold(0, |a, b| a.wrapping_add(b));\n\
         \x20   ExitCode::from((sum & 0xff) as u8)\n\
         }\n";

    let Some((stdout, exit)) = svm_run_std("svm_std_compute", src) else {
        eprintln!("note: skipping std_guest (build-std produced no .ll)");
        return;
    };
    assert!(stdout.is_empty(), "pure-compute program writes nothing");
    assert_eq!(
        exit, 140,
        "Σ i² for i<8 = 140; ran on svm to the computed exit code"
    );
}

/// S1c — the svm `std::sys::svm` PAL: real `println!` reaches the host through the powerbox `write`
/// binding, and `std::process::exit` through the `Exit` binding. A `std` program's **stdout and exit
/// code** match a real native run byte-for-byte (the `powerbox_diff` analog, one language up).
#[test]
fn std_stdout_and_exit_match_native() {
    if lane_ready().is_none() {
        eprintln!("note: skipping std_guest stdout (need the svm std overlay — see rust-svm/)");
        return;
    }

    let src = "#![feature(restricted_std)]\n\
         fn main() {\n\
         \x20   println!(\"hello from std on svm\");\n\
         \x20   let v: Vec<u32> = (1..=5).collect();\n\
         \x20   println!(\"sum(1..=5) = {}\", v.iter().sum::<u32>());\n\
         \x20   std::process::exit(7);\n\
         }\n";

    let Some((svm_stdout, svm_exit)) = svm_run_std("svm_std_hello", src) else {
        eprintln!("note: skipping std_guest stdout (build-std produced no .ll)");
        return;
    };
    let Some((native_stdout, native_exit)) = native_oracle("svm_std_hello", src) else {
        // No host rustc — still assert the svm side against the known-good bytes.
        assert_eq!(svm_stdout, b"hello from std on svm\nsum(1..=5) = 15\n");
        assert_eq!(svm_exit, 7);
        return;
    };
    assert_eq!(
        svm_stdout, native_stdout,
        "real std `println!` on svm matches native stdout byte-for-byte"
    );
    assert_eq!(
        svm_exit, native_exit,
        "std `process::exit` code matches native"
    );
}

/// S1e — distinct `stderr`: the svm PAL's `Stderr` routes to the powerbox stderr `Stream` (the
/// `__vm_write_stderr` builtin → the appended `"stderr"` handle), so interleaved `println!`/
/// `eprintln!` land in **separate** streams, each matching native.
#[test]
fn std_stderr_is_distinct_from_stdout() {
    if lane_ready().is_none() {
        eprintln!("note: skipping std_guest stderr (need the svm std overlay — see rust-svm/)");
        return;
    }

    let src = "#![feature(restricted_std)]\n\
         fn main() {\n\
         \x20   println!(\"out line 1\");\n\
         \x20   eprintln!(\"err line 1\");\n\
         \x20   println!(\"out line 2\");\n\
         \x20   eprintln!(\"err line 2\");\n\
         }\n";

    let Some((svm_out, svm_err, _)) = svm_run_std_streams("svm_std_stderr", src) else {
        eprintln!("note: skipping std_guest stderr (build-std produced no .ll)");
        return;
    };
    // The svm side is fully determined: stdout has only the `println!` lines, stderr only `eprintln!`.
    assert_eq!(
        String::from_utf8_lossy(&svm_out),
        "out line 1\nout line 2\n"
    );
    assert_eq!(
        String::from_utf8_lossy(&svm_err),
        "err line 1\nerr line 2\n"
    );

    // Cross-check both streams against a real native run.
    if let Some((n_out, n_err, _)) = native_oracle_streams("svm_std_stderr", src) {
        assert_eq!(svm_out, n_out, "svm stdout matches native stdout");
        assert_eq!(svm_err, n_err, "svm stderr matches native stderr");
    }
}

/// S1d — `std::env::args`: the svm PAL's `init` captures the powerbox-threaded `argv`, and the svm
/// `args` module walks it. A program echoing its arguments matches a native run with the **same
/// argv** byte-for-byte (`argv[0]` = program name, then the passed args, including one with spaces).
#[test]
fn std_env_args_match_native() {
    if lane_ready().is_none() {
        eprintln!("note: skipping std_guest args (need the svm std overlay — see rust-svm/)");
        return;
    }

    let src = "#![feature(restricted_std)]\n\
         fn main() {\n\
         \x20   let args: Vec<String> = std::env::args().collect();\n\
         \x20   println!(\"argc = {}\", args.len());\n\
         \x20   for (i, a) in args.iter().enumerate() {\n\
         \x20       println!(\"argv[{i}] = {a}\");\n\
         \x20   }\n\
         }\n";

    // argv[0] is the program name; the on-ramp/powerbox and the native OS each supply their own, so
    // use a matching program name and compare the whole vector.
    let extra = ["alpha", "beta gamma"];
    let svm_argv: &[&[u8]] = &[b"prog", b"alpha", b"beta gamma"];

    let Some((svm_stdout, _)) = svm_run_std_with_args("svm_std_args", src, svm_argv) else {
        eprintln!("note: skipping std_guest args (build-std produced no .ll)");
        return;
    };
    // The svm side is fully determined; assert it directly (argv[0] = "prog").
    let expected = "argc = 3\nargv[0] = prog\nargv[1] = alpha\nargv[2] = beta gamma\n";
    assert_eq!(
        String::from_utf8_lossy(&svm_stdout),
        expected,
        "std::env::args on svm echoes the powerbox-threaded argv"
    );

    // Cross-check the shape against a native run (its argv[0] is the binary path, so compare from
    // argv[1] onward — i.e. everything after the first line's count and the argv[0] line).
    if let Some((native_stdout, _)) = native_oracle_args("svm_std_args", src, &extra) {
        let svm_tail: Vec<&str> = std::str::from_utf8(&svm_stdout)
            .unwrap()
            .lines()
            .skip(2) // "argc = 3", "argv[0] = prog"
            .collect();
        let native_tail: Vec<&str> = std::str::from_utf8(&native_stdout)
            .unwrap()
            .lines()
            .skip(2) // "argc = 3", "argv[0] = <binary path>"
            .collect();
        assert_eq!(
            svm_tail, native_tail,
            "argv[1..] on svm matches native (argc and the passed args agree)"
        );
        assert!(
            native_stdout.starts_with(b"argc = 3\n"),
            "native also sees argc = 3"
        );
    }
}

/// S2 — `std::time` via the **posix-cap path**: `SystemTime`/`Instant` reach the host clock through
/// the svm PAL's `__vm_host_call` bridge to the granted `posix` personality (`OP_CLOCK`). Run with a
/// **pinned clock** so the output is deterministic — the first exercise of `run_with_caps` + a posix
/// cap, and of the §9 constant-`op` requirement (`op` is the literal `33` at the call site).
#[test]
fn std_time_reads_the_posix_clock() {
    if lane_ready().is_none() {
        eprintln!("note: skipping std_guest time (need the svm std overlay — see rust-svm/)");
        return;
    }

    let src = "#![feature(restricted_std)]\n\
         use std::time::{SystemTime, UNIX_EPOCH, Instant};\n\
         fn main() {\n\
         \x20   let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();\n\
         \x20   println!(\"realtime_secs = {secs}\");\n\
         \x20   let a = Instant::now();\n\
         \x20   let b = Instant::now();\n\
         \x20   println!(\"monotonic_nondecreasing = {}\", b >= a);\n\
         }\n";

    // Pin the clock to 1.7e18 ns = 1_700_000_000 s, so the output is fully determined.
    let seeded_nanos: i64 = 1_700_000_000_000_000_000;
    let Some((stdout, _)) = svm_run_std_posix("svm_std_time", src, |p| p.set_clock(seeded_nanos))
    else {
        eprintln!("note: skipping std_guest time (build-std produced no .ll)");
        return;
    };
    assert_eq!(
        String::from_utf8_lossy(&stdout),
        "realtime_secs = 1700000000\nmonotonic_nondecreasing = true\n",
        "std::time reads the seeded posix clock, and Instant is non-decreasing"
    );
}

/// S2 (env) — `std::env::{var_os, set_var, remove_var}` via the posix-cap path: the svm `env` module
/// reaches the personality's env map through the PAL `host` bridge (buffer-writing `OP_GETENV_R` /
/// `OP_SETENV` / `OP_UNSETENV`, so no personality arena contends with the guest heap). A program
/// reads a seeded var, sets one, and removes one — fully deterministic output.
///
/// (`std::env::var` / `vars` — the `OsString`→`String` paths through `core::str::lossy::Debug::fmt` —
/// now translate too, once the on-ramp's entry-block slot-numbering fix for named params landed; see
/// `std_fs_round_trips` and the `ll/parse.rs` fix in this change.)
#[test]
fn std_env_var_os_round_trips() {
    if lane_ready().is_none() {
        eprintln!("note: skipping std_guest env (need the svm std overlay — see rust-svm/)");
        return;
    }

    let src = "#![feature(restricted_std)]\n\
         fn main() {\n\
         \x20   let seeded = std::env::var_os(\"SEEDED\").and_then(|v| v.to_str().map(|s| s.to_owned()));\n\
         \x20   println!(\"SEEDED={}\", seeded.as_deref().unwrap_or(\"<unset>\"));\n\
         \x20   std::env::set_var(\"MADE\", \"here\");\n\
         \x20   let made = std::env::var_os(\"MADE\").and_then(|v| v.to_str().map(|s| s.to_owned()));\n\
         \x20   println!(\"MADE={}\", made.as_deref().unwrap_or(\"<unset>\"));\n\
         \x20   std::env::remove_var(\"SEEDED\");\n\
         \x20   println!(\"after_remove_present={}\", std::env::var_os(\"SEEDED\").is_some());\n\
         }\n";

    let Some((stdout, _)) =
        svm_run_std_posix("svm_std_env", src, |p| p.set_env("SEEDED", "from_host"))
    else {
        eprintln!("note: skipping std_guest env (build-std produced no .ll)");
        return;
    };
    assert_eq!(
        String::from_utf8_lossy(&stdout),
        "SEEDED=from_host\nMADE=here\nafter_remove_present=false\n",
        "env var_os/set_var/remove_var round-trip through the posix cap"
    );
}

/// S2 (fs) — `std::fs` via the posix-cap path: the svm `fs` module reaches the personality's in-memory
/// filesystem through the PAL `host` bridge's file ops (`OP_OPEN`/`OP_READ`/`OP_WRITE`/`OP_LSEEK`/
/// `OP_CLOSE`/`OP_UNLINK`/`OP_STAT`/`OP_OPENDIR`/`OP_READDIR`/`OP_CLOSEDIR`). One program exercises the
/// whole surface: `File::create`+`write_all`, `read_to_string`, `metadata`, `Seek`+`read`, `read_dir`,
/// an `ErrorKind::NotFound` on a missing path, and `remove_file`+`exists` — fully deterministic output.
/// The memfs is seeded with one extra file so `read_dir` returns a stable, sorted two-entry listing.
#[test]
fn std_fs_round_trips() {
    if lane_ready().is_none() {
        eprintln!("note: skipping std_guest fs (need the svm std overlay — see rust-svm/)");
        return;
    }

    let src = "#![feature(restricted_std)]\n\
         use std::fs;\n\
         use std::io::{Read, Seek, SeekFrom, Write};\n\
         fn main() {\n\
         \x20   let mut f = fs::File::create(\"/data/hello.txt\").expect(\"create\");\n\
         \x20   f.write_all(b\"hello svm fs\").expect(\"write\");\n\
         \x20   drop(f);\n\
         \x20   println!(\"read_to_string={:?}\", fs::read_to_string(\"/data/hello.txt\").expect(\"read\"));\n\
         \x20   let md = fs::metadata(\"/data/hello.txt\").expect(\"metadata\");\n\
         \x20   println!(\"is_file={} len={}\", md.is_file(), md.len());\n\
         \x20   let mut g = fs::File::open(\"/data/hello.txt\").expect(\"open\");\n\
         \x20   g.seek(SeekFrom::Start(6)).expect(\"seek\");\n\
         \x20   let mut s = String::new();\n\
         \x20   g.read_to_string(&mut s).expect(\"read\");\n\
         \x20   println!(\"after_seek={s:?}\");\n\
         \x20   let mut names: Vec<String> = fs::read_dir(\"/data\").expect(\"read_dir\")\n\
         \x20       .map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();\n\
         \x20   names.sort();\n\
         \x20   println!(\"read_dir={names:?}\");\n\
         \x20   println!(\"missing_kind={:?}\", fs::metadata(\"/data/nope\").unwrap_err().kind());\n\
         \x20   fs::remove_file(\"/data/hello.txt\").expect(\"remove\");\n\
         \x20   println!(\"exists_after_remove={}\", fs::exists(\"/data/hello.txt\").unwrap());\n\
         }\n";

    let Some((stdout, _)) = svm_run_std_posix("svm_std_fs", src, |p| {
        p.write_file("/data/seed.txt", b"seed")
    }) else {
        eprintln!("note: skipping std_guest fs (build-std produced no .ll)");
        return;
    };
    assert_eq!(
        String::from_utf8_lossy(&stdout),
        "read_to_string=\"hello svm fs\"\n\
         is_file=true len=12\n\
         after_seek=\"svm fs\"\n\
         read_dir=[\"hello.txt\", \"seed.txt\"]\n\
         missing_kind=NotFound\n\
         exists_after_remove=false\n",
        "fs File/metadata/seek/read_dir/remove round-trip through the posix cap"
    );
}

/// S2 (process) — `std::process::Command` via the posix-cap path: the svm `process` module reaches the
/// personality's **fork-free** spawn (`OP_SPAWN`/`OP_WAITPID`) through the PAL `host` bridge. A spawn
/// runs the named command to completion synchronously; `output` captures its stdout by bracketing the
/// spawn with an `OP_PIPE`+`OP_DUP2` fd-1 redirect (restored afterwards, so captured output does not
/// leak to the parent's stdout), and `status`/`wait` reap the exit code. The embedder wires a scripted
/// spawn delegate (`set_spawn`) — `echo` echoes its args, `true`/`false` set the exit code.
#[test]
fn std_process_round_trips() {
    if lane_ready().is_none() {
        eprintln!("note: skipping std_guest process (need the svm std overlay — see rust-svm/)");
        return;
    }

    let src = "#![feature(restricted_std)]\n\
         use std::process::{Command, Stdio};\n\
         fn main() {\n\
         \x20   let out = Command::new(\"echo\").arg(\"hello\").arg(\"world\").output().expect(\"output\");\n\
         \x20   println!(\"echo_code={:?}\", out.status.code());\n\
         \x20   println!(\"echo_stdout={:?}\", String::from_utf8_lossy(&out.stdout).trim_end());\n\
         \x20   let st = Command::new(\"false\").status().expect(\"status\");\n\
         \x20   println!(\"false_code={:?}\", st.code());\n\
         \x20   let mut child = Command::new(\"true\").stdout(Stdio::null()).spawn().expect(\"spawn\");\n\
         \x20   println!(\"true_ok={}\", child.wait().expect(\"wait\").success());\n\
         \x20   let noisy = Command::new(\"noisy\").output().expect(\"output\");\n\
         \x20   println!(\"noisy_out={:?}\", String::from_utf8_lossy(&noisy.stdout).trim_end());\n\
         \x20   println!(\"noisy_err={:?}\", String::from_utf8_lossy(&noisy.stderr).trim_end());\n\
         \x20   println!(\"pid={}\", std::process::id());\n\
         }\n";

    let Some((stdout, _)) = svm_run_std_posix("svm_std_process", src, |p| {
        p.set_spawn(|name: &str, argv: &[String], _stdin: &[u8]| {
            let (stdout, stderr, status) = match name {
                "echo" => (
                    format!("{}\n", argv[1..].join(" ")).into_bytes(),
                    Vec::new(),
                    0,
                ),
                "true" => (Vec::new(), Vec::new(), 0),
                "false" => (Vec::new(), Vec::new(), 1),
                // Emits on *both* streams, so `output()` capturing stderr separately is exercised.
                "noisy" => (b"on stdout\n".to_vec(), b"on stderr\n".to_vec(), 0),
                _ => (Vec::new(), Vec::new(), 127),
            };
            svm_posix::SpawnResult {
                stdout,
                stderr,
                status,
            }
        })
    }) else {
        eprintln!("note: skipping std_guest process (build-std produced no .ll)");
        return;
    };
    assert_eq!(
        String::from_utf8_lossy(&stdout),
        "echo_code=Some(0)\n\
         echo_stdout=\"hello world\"\n\
         false_code=Some(1)\n\
         true_ok=true\n\
         noisy_out=\"on stdout\"\n\
         noisy_err=\"on stderr\"\n\
         pid=1\n",
        "Command output (stdout+stderr)/status/spawn+wait round-trip through the fork-free spawn"
    );
}

/// S2 (fs — directory mutation) — the `std::fs` dir ops the memfs now backs: `create_dir` /
/// `create_dir_all` (explicit-dir op `OP_MKDIR`, with `AlreadyExists` on a repeat), `rename` (file
/// move), `remove_dir` (`OP_RMDIR`, `DirectoryNotEmpty` on a populated dir), and `File::try_clone`
/// (a second fd via `OP_DUP`). Seeded with one file so `/data` exists as a parent directory.
#[test]
fn std_fs_dir_ops() {
    if lane_ready().is_none() {
        eprintln!("note: skipping std_guest fs-dir (need the svm std overlay — see rust-svm/)");
        return;
    }

    let src = "#![feature(restricted_std)]\n\
         use std::fs;\n\
         use std::io::Read;\n\
         fn main() {\n\
         \x20   fs::create_dir(\"/data/new\").expect(\"create_dir\");\n\
         \x20   println!(\"is_dir={}\", fs::metadata(\"/data/new\").unwrap().is_dir());\n\
         \x20   println!(\"exists_err={:?}\", fs::create_dir(\"/data/new\").unwrap_err().kind());\n\
         \x20   fs::create_dir_all(\"/data/a/b/c\").expect(\"create_dir_all\");\n\
         \x20   println!(\"deep_is_dir={}\", fs::metadata(\"/data/a/b/c\").unwrap().is_dir());\n\
         \x20   fs::write(\"/data/f.txt\", b\"hi there\").expect(\"write\");\n\
         \x20   fs::rename(\"/data/f.txt\", \"/data/g.txt\").expect(\"rename\");\n\
         \x20   println!(\"renamed={:?}\", fs::read_to_string(\"/data/g.txt\").unwrap());\n\
         \x20   println!(\"old_gone={}\", !fs::exists(\"/data/f.txt\").unwrap());\n\
         \x20   fs::remove_dir(\"/data/new\").expect(\"remove_dir\");\n\
         \x20   println!(\"removed_gone={}\", !fs::exists(\"/data/new\").unwrap());\n\
         \x20   println!(\"rmdir_nonempty_err={:?}\", fs::remove_dir(\"/data/a\").unwrap_err().kind());\n\
         \x20   let mut clone = fs::File::open(\"/data/g.txt\").expect(\"open\").try_clone().expect(\"clone\");\n\
         \x20   let mut s = String::new();\n\
         \x20   clone.read_to_string(&mut s).expect(\"read clone\");\n\
         \x20   println!(\"clone_read={s:?}\");\n\
         }\n";

    let Some((stdout, _)) =
        svm_run_std_posix("svm_std_fs_dir", src, |p| p.write_file("/data/seed", b"x"))
    else {
        eprintln!("note: skipping std_guest fs-dir (build-std produced no .ll)");
        return;
    };
    assert_eq!(
        String::from_utf8_lossy(&stdout),
        "is_dir=true\n\
         exists_err=AlreadyExists\n\
         deep_is_dir=true\n\
         renamed=\"hi there\"\n\
         old_gone=true\n\
         removed_gone=true\n\
         rmdir_nonempty_err=DirectoryNotEmpty\n\
         clone_read=\"hi there\"\n",
        "create_dir/create_dir_all/rename/remove_dir/try_clone over the memfs dir ops"
    );
}

/// S2 (net) — `std::net` via the **`net` capability** (POSIX.md §5a): a lockstep self-connect over
/// the loopback **memnet** (bind `:0` → ephemeral port, connect, accept, bytes both ways, EOF on
/// drop, `ConnectionRefused` with no listener, `localhost` resolution — all deterministic, no
/// delegate), then scripted **egress through a `NetDelegate`** (`set_net`): the guest resolves
/// `example.com` and connects to it purely through the embedder's script — the personality itself
/// holds no network authority. Data rides the ordinary posix fd `read`/`write` ops.
#[test]
fn std_net_round_trips() {
    if lane_ready().is_none() {
        eprintln!("note: skipping std_guest net (need the svm std overlay — see rust-svm/)");
        return;
    }

    struct Canned;
    impl svm_posix::NetStream for Canned {
        fn send(&mut self, buf: &[u8]) -> i64 {
            assert_eq!(buf, b"GET /");
            buf.len() as i64
        }
        fn recv(&mut self, buf: &mut [u8]) -> i64 {
            let msg = b"HTTP/1.1 200 OK";
            buf[..msg.len()].copy_from_slice(msg);
            msg.len() as i64
        }
    }
    struct Scripted;
    impl svm_posix::NetDelegate for Scripted {
        fn connect(
            &mut self,
            addr: &svm_posix::NetAddr,
        ) -> Result<Box<dyn svm_posix::NetStream>, i64> {
            assert_eq!(addr.port, 80);
            assert_eq!(&addr.addr[..4], &[93, 184, 216, 34]);
            Ok(Box::new(Canned))
        }
        fn resolve(&mut self, host: &str) -> Result<Vec<svm_posix::NetAddr>, i64> {
            assert_eq!(host, "example.com");
            let mut addr = [0u8; 16];
            addr[..4].copy_from_slice(&[93, 184, 216, 34]);
            Ok(vec![svm_posix::NetAddr {
                v6: false,
                port: 0,
                addr,
            }])
        }
    }

    let src = "#![feature(restricted_std)]\n\
         use std::io::{Read, Write};\n\
         use std::net::{TcpListener, TcpStream};\n\
         fn main() {\n\
         \x20   let listener = TcpListener::bind(\"127.0.0.1:0\").expect(\"bind\");\n\
         \x20   let addr = listener.local_addr().expect(\"local_addr\");\n\
         \x20   println!(\"bound_port_ephemeral={}\", addr.port() >= 49152);\n\
         \x20   let mut client = TcpStream::connect(addr).expect(\"connect\");\n\
         \x20   let (mut server, peer) = listener.accept().expect(\"accept\");\n\
         \x20   println!(\"peer_is_loopback={}\", peer.ip().is_loopback());\n\
         \x20   client.write_all(b\"ping\").expect(\"cwrite\");\n\
         \x20   let mut buf = [0u8; 32];\n\
         \x20   let n = server.read(&mut buf).expect(\"sread\");\n\
         \x20   println!(\"server_got={:?}\", core::str::from_utf8(&buf[..n]).unwrap());\n\
         \x20   server.write_all(b\"pong!\").expect(\"swrite\");\n\
         \x20   let n = client.read(&mut buf).expect(\"cread\");\n\
         \x20   println!(\"client_got={:?}\", core::str::from_utf8(&buf[..n]).unwrap());\n\
         \x20   drop(client);\n\
         \x20   println!(\"eof={}\", server.read(&mut buf).expect(\"eof read\") == 0);\n\
         \x20   println!(\"refused={:?}\", TcpStream::connect(\"127.0.0.1:12345\").unwrap_err().kind());\n\
         \x20   let mut egress = TcpStream::connect(\"example.com:80\").expect(\"delegate connect\");\n\
         \x20   egress.write_all(b\"GET /\").expect(\"ewrite\");\n\
         \x20   let n = egress.read(&mut buf).expect(\"eread\");\n\
         \x20   println!(\"egress={:?}\", core::str::from_utf8(&buf[..n]).unwrap());\n\
         }\n";

    let Some((stdout, _)) = svm_run_std_net("svm_std_net", src, |p| p.set_net(Scripted)) else {
        eprintln!("note: skipping std_guest net (build-std produced no .ll)");
        return;
    };
    assert_eq!(
        String::from_utf8_lossy(&stdout),
        "bound_port_ephemeral=true\n\
         peer_is_loopback=true\n\
         server_got=\"ping\"\n\
         client_got=\"pong!\"\n\
         eof=true\n\
         refused=ConnectionRefused\n\
         egress=\"HTTP/1.1 200 OK\"\n",
        "TcpListener/TcpStream over the memnet + scripted delegate egress"
    );
}

// ---------------------------------------------------------------------------------------------
// #821 — the threaded target spec (`x86_64-unknown-svm-threads`, `singlethread=false`). These build
// the same programs for the threaded spec (real atomic-ordering codegen, futex-backed `sys/sync`,
// native TLS) and run them on a single vCPU — no `thread::spawn` yet (#823). They prove the threaded
// `std` build translates, verifies, and runs byte-identically to the native oracle.
// ---------------------------------------------------------------------------------------------

/// The pure-compute Σ i² program (the lean spec's `std_bin_runs_on_svm_via_powerbox`) built for the
/// **threaded** spec. Exercises `singlethread=false` codegen through the whole on-ramp → verify → run
/// path; the result must match the lean spec exactly (140).
#[test]
fn std_threads_spec_bin_runs_on_svm() {
    if lane_ready().is_none() {
        eprintln!("note: skipping std_guest threads (need the svm std overlay — see rust-svm/)");
        return;
    }

    let src = "#![feature(restricted_std)]\n\
         use std::process::ExitCode;\n\
         fn main() -> ExitCode {\n\
         \x20   let mut v: Vec<i32> = Vec::new();\n\
         \x20   let mut i = 0i32;\n\
         \x20   while i < 8 { v.push(i.wrapping_mul(i)); i += 1; }\n\
         \x20   let sum: i32 = v.iter().copied().fold(0, |a, b| a.wrapping_add(b));\n\
         \x20   ExitCode::from((sum & 0xff) as u8)\n\
         }\n";

    let Some((stdout, exit)) = svm_run_std_threads("svm_stdt_compute", src) else {
        eprintln!("note: skipping std_guest threads (build-std produced no .ll)");
        return;
    };
    assert!(stdout.is_empty(), "pure-compute program writes nothing");
    assert_eq!(
        exit, 140,
        "Σ i² for i<8 = 140 under the threaded spec, byte-identical to the lean spec"
    );
}

/// Futex-backed `sys/sync` (uncontended, single vCPU): a `Mutex` guards an accumulator and a `Once`
/// runs its initializer exactly once. With `singlethread=false` these route to std's `futex`
/// implementations over `__vm_wait32`/`__vm_notify` (never actually blocking here, since there is no
/// contention). Differential against the native oracle.
#[test]
fn std_threads_spec_futex_sync_round_trips() {
    if lane_ready().is_none() {
        eprintln!("note: skipping std_guest threads (need the svm std overlay — see rust-svm/)");
        return;
    }

    let src = "#![feature(restricted_std)]\n\
         use std::process::ExitCode;\n\
         use std::sync::{Mutex, Once};\n\
         static INIT: Once = Once::new();\n\
         fn main() -> ExitCode {\n\
         \x20   let m = Mutex::new(0u32);\n\
         \x20   {\n\
         \x20       let mut g = m.lock().unwrap();\n\
         \x20       for i in 0..10u32 { *g = g.wrapping_add(i); }\n\
         \x20   }\n\
         \x20   let mut once_ran = 0u32;\n\
         \x20   INIT.call_once(|| { once_ran = 7; });\n\
         \x20   let total = m.lock().unwrap().wrapping_add(once_ran);\n\
         \x20   ExitCode::from((total & 0xff) as u8)\n\
         }\n";

    let Some((_, exit)) = svm_run_std_threads("svm_stdt_sync", src) else {
        eprintln!("note: skipping std_guest threads (build-std produced no .ll)");
        return;
    };
    // Σ 0..10 = 45, + the Once's 7 = 52.
    assert_eq!(
        exit, 52,
        "futex-backed Mutex + Once compute the expected total"
    );
    if let Some((_, oracle)) = native_oracle("svm_stdt_sync_oracle", src) {
        assert_eq!(exit, oracle, "threaded-spec sync matches the native oracle");
    }
}

/// Native TLS (`thread_local!`) on the threaded spec: `has-thread-local=true` routes std to the
/// `native` TLS impl, which emits `#[thread_local]` globals + `llvm.threadlocal.address`. On a single
/// vCPU the on-ramp lowers those as the NIM.md §3d **Tier-1** collapse (thread-local → plain global),
/// so a read-modify-write of a `thread_local` cell yields the same result as native.
#[test]
fn std_threads_spec_thread_local_round_trips() {
    if lane_ready().is_none() {
        eprintln!("note: skipping std_guest threads (need the svm std overlay — see rust-svm/)");
        return;
    }

    let src = "#![feature(restricted_std)]\n\
         use std::cell::Cell;\n\
         use std::process::ExitCode;\n\
         thread_local! { static COUNTER: Cell<u32> = Cell::new(100); }\n\
         fn main() -> ExitCode {\n\
         \x20   COUNTER.with(|c| c.set(c.get().wrapping_add(23)));\n\
         \x20   let v = COUNTER.with(|c| c.get());\n\
         \x20   ExitCode::from((v & 0xff) as u8)\n\
         }\n";

    let Some((_, exit)) = svm_run_std_threads("svm_stdt_tls", src) else {
        eprintln!("note: skipping std_guest threads (build-std produced no .ll)");
        return;
    };
    // 100 + 23 = 123.
    assert_eq!(
        exit, 123,
        "native thread_local read-modify-write (Tier-2 block, root base set in _start)"
    );
    if let Some((_, oracle)) = native_oracle("svm_stdt_tls_oracle", src) {
        assert_eq!(exit, oracle, "threaded-spec TLS matches the native oracle");
    }
}

/// `std::thread::spawn` + `JoinHandle::join`: a spawned vCPU runs a `move` closure (capturing an
/// outer value) and its return value comes back through `join`. Exercises the whole `sys/thread/svm`
/// path — the fixed `__rust_thread_start` trampoline over `__vm_thread_spawn`, the per-thread TLS
/// block (`__vm_tls_size`/`template`), a guest-carved stack, and `__vm_thread_join`. Differential
/// against the native oracle; deterministic on the cooperative driver.
#[test]
fn std_threads_spec_spawn_join_returns_value() {
    if lane_ready().is_none() {
        eprintln!("note: skipping std_guest threads (need the svm std overlay — see rust-svm/)");
        return;
    }

    let src = "#![feature(restricted_std)]\n\
         use std::process::ExitCode;\n\
         fn main() -> ExitCode {\n\
         \x20   let base = 30u32;\n\
         \x20   let h = std::thread::spawn(move || base.wrapping_add(12));\n\
         \x20   let v = h.join().unwrap();\n\
         \x20   ExitCode::from((v & 0xff) as u8)\n\
         }\n";

    let Some((_, exit)) = svm_run_std_threads("svm_stdt_spawn", src) else {
        eprintln!("note: skipping std_guest threads (build-std produced no .ll)");
        return;
    };
    assert_eq!(exit, 42, "spawn(move || 30+12).join() == 42");
    if let Some((_, oracle)) = native_oracle("svm_stdt_spawn_oracle", src) {
        assert_eq!(exit, oracle, "spawn/join matches the native oracle");
    }
}

/// Several spawned threads sharing an `Arc<AtomicU32>`, each `fetch_add`-ing, joined back — the
/// Rust analog of the THREADS.md counter kernel. Exercises real cross-thread state: `Arc` (whose
/// drop uses the acquire `fence` the on-ramp now lowers), `AtomicU32` RMW, per-thread TLS isolation
/// (each thread's `thread::current` handle is its own), and N spawn/joins. On the cooperative
/// deterministic driver the total is exact (4 × 10 = 40).
#[test]
fn std_threads_spec_shared_atomic_counter() {
    if lane_ready().is_none() {
        eprintln!("note: skipping std_guest threads (need the svm std overlay — see rust-svm/)");
        return;
    }

    let src = "#![feature(restricted_std)]\n\
         use std::process::ExitCode;\n\
         use std::sync::Arc;\n\
         use std::sync::atomic::{AtomicU32, Ordering};\n\
         fn main() -> ExitCode {\n\
         \x20   let counter = Arc::new(AtomicU32::new(0));\n\
         \x20   let mut handles = Vec::new();\n\
         \x20   for _ in 0..4 {\n\
         \x20       let c = Arc::clone(&counter);\n\
         \x20       handles.push(std::thread::spawn(move || {\n\
         \x20           for _ in 0..10 { c.fetch_add(1, Ordering::SeqCst); }\n\
         \x20       }));\n\
         \x20   }\n\
         \x20   for h in handles { h.join().unwrap(); }\n\
         \x20   ExitCode::from((counter.load(Ordering::SeqCst) & 0xff) as u8)\n\
         }\n";

    let Some((_, exit)) = svm_run_std_threads("svm_stdt_counter", src) else {
        eprintln!("note: skipping std_guest threads (build-std produced no .ll)");
        return;
    };
    assert_eq!(exit, 40, "4 threads × 10 fetch_add = 40");
    if let Some((_, oracle)) = native_oracle("svm_stdt_counter_oracle", src) {
        assert_eq!(
            exit, oracle,
            "shared atomic counter matches the native oracle"
        );
    }
}

/// `thread::yield_now` and `thread::sleep` are real (futex-park) rather than inert: a spawned worker
/// increments a shared `AtomicU32` with a `yield_now` between steps while `main` `sleep`s, then both
/// join at the sum. This exercises the park/resume path (the worker must make progress across the
/// main thread's sleep, and the yields must not hang or trap) and pins the result against native.
#[test]
fn std_threads_spec_yield_and_sleep() {
    if lane_ready().is_none() {
        eprintln!("note: skipping std_guest threads (need the svm std overlay — see rust-svm/)");
        return;
    }

    let src = "#![feature(restricted_std)]\n\
         use std::process::ExitCode;\n\
         use std::sync::Arc;\n\
         use std::sync::atomic::{AtomicU32, Ordering};\n\
         use std::time::Duration;\n\
         fn main() -> ExitCode {\n\
         \x20   let c = Arc::new(AtomicU32::new(0));\n\
         \x20   let cw = Arc::clone(&c);\n\
         \x20   let h = std::thread::spawn(move || {\n\
         \x20       for _ in 0..15 {\n\
         \x20           cw.fetch_add(1, Ordering::SeqCst);\n\
         \x20           std::thread::yield_now();\n\
         \x20       }\n\
         \x20   });\n\
         \x20   std::thread::sleep(Duration::from_millis(1));\n\
         \x20   std::thread::yield_now();\n\
         \x20   h.join().unwrap();\n\
         \x20   ExitCode::from((c.load(Ordering::SeqCst) & 0xff) as u8)\n\
         }\n";

    let Some((_, exit)) = svm_run_std_threads("svm_stdt_yieldsleep", src) else {
        eprintln!("note: skipping std_guest threads (build-std produced no .ll)");
        return;
    };
    assert_eq!(
        exit, 15,
        "worker completes 15 yield-separated increments across main's sleep"
    );
    if let Some((_, oracle)) = native_oracle("svm_stdt_yieldsleep_oracle", src) {
        assert_eq!(
            exit, oracle,
            "yield_now/sleep progress matches the native oracle"
        );
    }
}

/// #825 — **genuinely concurrent I/O across two vCPUs** over the loopback memnet. `main` binds a
/// listener and moves it to a server thread, then acts as the client: connect, send, and read the
/// echo. The memnet is nonblocking (an empty accept/read is `WouldBlock`, never a deadlocking block —
/// "a cooperative guest cannot block on itself"), so the server's `accept`/`read` **retry with
/// `thread::yield_now()`** until the client's op lands — the second vCPU drains the peer the first is
/// spinning on. This proves the EAGAIN + retry-yield contract composes with real threads (the personality's
/// `Inner` is `Mutex`-guarded host-side) and exercises the real `yield_now`. Deterministic on the
/// cooperative driver, so it differentials against native.
#[test]
fn std_threads_spec_two_thread_memnet_echo() {
    if lane_ready().is_none() {
        eprintln!("note: skipping std_guest threads (need the svm std overlay — see rust-svm/)");
        return;
    }

    let src = "#![feature(restricted_std)]\n\
         use std::io::{Read, Write};\n\
         use std::net::{TcpListener, TcpStream};\n\
         use std::process::ExitCode;\n\
         // Block on a would-block memnet op by retrying with a yield, so the peer vCPU can run.\n\
         fn until<T>(mut op: impl FnMut() -> std::io::Result<T>) -> T {\n\
         \x20   loop {\n\
         \x20       match op() {\n\
         \x20           Ok(v) => return v,\n\
         \x20           Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => std::thread::yield_now(),\n\
         \x20           Err(e) => panic!(\"io: {e}\"),\n\
         \x20       }\n\
         \x20   }\n\
         }\n\
         fn main() -> ExitCode {\n\
         \x20   let listener = TcpListener::bind(\"127.0.0.1:0\").expect(\"bind\");\n\
         \x20   let port = listener.local_addr().expect(\"local_addr\").port();\n\
         \x20   let server = std::thread::spawn(move || {\n\
         \x20       let (mut s, _) = until(|| listener.accept());\n\
         \x20       let mut buf = [0u8; 16];\n\
         \x20       let n = until(|| {\n\
         \x20           let n = s.read(&mut buf)?;\n\
         \x20           if n == 0 { Err(std::io::ErrorKind::WouldBlock.into()) } else { Ok(n) }\n\
         \x20       });\n\
         \x20       let up: Vec<u8> = buf[..n].iter().map(|b| b.to_ascii_uppercase()).collect();\n\
         \x20       s.write_all(&up).expect(\"server echo\");\n\
         \x20   });\n\
         \x20   let mut client = until(|| TcpStream::connect((\"127.0.0.1\", port)));\n\
         \x20   client.write_all(b\"ping\").expect(\"client write\");\n\
         \x20   let mut buf = [0u8; 16];\n\
         \x20   let n = until(|| {\n\
         \x20       let n = client.read(&mut buf)?;\n\
         \x20       if n == 0 { Err(std::io::ErrorKind::WouldBlock.into()) } else { Ok(n) }\n\
         \x20   });\n\
         \x20   server.join().expect(\"join\");\n\
         \x20   // Echo is the upper-cased request: \"ping\" -> \"PING\".\n\
         \x20   let ok = &buf[..n] == b\"PING\";\n\
         \x20   ExitCode::from(if ok { n as u8 } else { 0 })\n\
         }\n";

    let Some((_, exit)) = svm_run_std_threads_net("svm_stdt_netecho", src) else {
        eprintln!("note: skipping std_guest threads (build-std produced no .ll)");
        return;
    };
    assert_eq!(
        exit, 4,
        "server thread echoed \"PING\" (4 bytes) back to the client thread"
    );
    if let Some((_, oracle)) = native_oracle("svm_stdt_netecho_oracle", src) {
        assert_eq!(
            exit, oracle,
            "two-thread loopback echo matches the native oracle"
        );
    }
}

/// #824 follow-through — `std::sync::mpsc` across two vCPUs (no per-target code; it rides the
/// futex-backed `sys/sync`). A producer thread sends 1..=9 down a channel and drops the sender; the
/// main thread's `for v in rx` blocks on `recv` until the channel closes, summing to 45. Unlike the
/// lock-free atomic-counter test, this exercises the futex **notify/wake** path (an empty `recv`
/// parks the consumer until the producer's `send` wakes it) end to end. Differential vs native.
#[test]
fn std_threads_spec_mpsc_channel() {
    if lane_ready().is_none() {
        eprintln!("note: skipping std_guest threads (need the svm std overlay — see rust-svm/)");
        return;
    }

    let src = "#![feature(restricted_std)]\n\
         use std::process::ExitCode;\n\
         use std::sync::mpsc;\n\
         fn main() -> ExitCode {\n\
         \x20   let (tx, rx) = mpsc::channel();\n\
         \x20   let producer = std::thread::spawn(move || {\n\
         \x20       for i in 1..=9u32 { tx.send(i).expect(\"send\"); }\n\
         \x20       // tx drops here → the channel closes, ending the consumer's loop.\n\
         \x20   });\n\
         \x20   let mut sum = 0u32;\n\
         \x20   for v in rx { sum = sum.wrapping_add(v); }\n\
         \x20   producer.join().expect(\"join\");\n\
         \x20   ExitCode::from((sum & 0xff) as u8)\n\
         }\n";

    let Some((_, exit)) = svm_run_std_threads("svm_stdt_mpsc", src) else {
        eprintln!("note: skipping std_guest threads (build-std produced no .ll)");
        return;
    };
    assert_eq!(
        exit, 45,
        "mpsc producer/consumer summed 1..=9 = 45 across two threads"
    );
    if let Some((_, oracle)) = native_oracle("svm_stdt_mpsc_oracle", src) {
        assert_eq!(exit, oracle, "mpsc channel matches the native oracle");
    }
}

// === #848 — the parallel-driver smoke lane =======================================================
// The `std_threads_spec_*` tests above pin their programs on the **cooperative** driver (`drive`), the
// deterministic oracle: one OS thread, a fixed round-robin schedule, an exact result every run. These
// smoke tests re-run a representative subset under the **parallel** driver (`drive_parallel`, THREADS.md
// 4c): one real OS thread per vCPU over a single shared window, with genuine races. The schedule is
// nondeterministic, so we assert only the *outcome* (which the guests' own synchronization makes
// schedule-independent) — proving the threaded `std` codegen + the futex/atomics primitives are correct
// under real concurrency, not just the cooperative interleaving. See `run_with_caps_parallel`.

/// Smoke: the shared-atomic-counter kernel under real OS threads. `fetch_add` is commutative, so the
/// total (4 × 10 = 40) is schedule-independent even though the interleaving is now genuinely parallel —
/// this is the cleanest outcome-stable exercise of cross-thread atomics + N spawn/joins on the parallel
/// driver. Skipped (like the cooperative specs) when the std overlay / build-std lane is unavailable.
#[test]
fn std_threads_parallel_smoke_atomic_counter() {
    if lane_ready().is_none() {
        eprintln!("note: skipping std_guest threads (need the svm std overlay — see rust-svm/)");
        return;
    }

    let src = "#![feature(restricted_std)]\n\
         use std::process::ExitCode;\n\
         use std::sync::Arc;\n\
         use std::sync::atomic::{AtomicU32, Ordering};\n\
         fn main() -> ExitCode {\n\
         \x20   let counter = Arc::new(AtomicU32::new(0));\n\
         \x20   let mut handles = Vec::new();\n\
         \x20   for _ in 0..4 {\n\
         \x20       let c = Arc::clone(&counter);\n\
         \x20       handles.push(std::thread::spawn(move || {\n\
         \x20           for _ in 0..10 { c.fetch_add(1, Ordering::SeqCst); }\n\
         \x20       }));\n\
         \x20   }\n\
         \x20   for h in handles { h.join().unwrap(); }\n\
         \x20   ExitCode::from((counter.load(Ordering::SeqCst) & 0xff) as u8)\n\
         }\n";

    let Some((_, exit)) = svm_run_std_threads_parallel("svm_stdt_counter_par", src) else {
        eprintln!("note: skipping std_guest threads (build-std produced no .ll)");
        return;
    };
    assert_eq!(
        exit, 40,
        "4 threads × 10 fetch_add = 40 under the parallel driver (commutative → schedule-independent)"
    );
}

/// Smoke: `spawn(move || …).join()` value passing under real OS threads. The child computes a value and
/// `join` hands it back to the parent — the join barrier makes the result (42) schedule-independent.
/// Exercises the spawn/join handshake + the boxed-closure trampoline on the parallel driver.
#[test]
fn std_threads_parallel_smoke_spawn_join() {
    if lane_ready().is_none() {
        eprintln!("note: skipping std_guest threads (need the svm std overlay — see rust-svm/)");
        return;
    }

    let src = "#![feature(restricted_std)]\n\
         use std::process::ExitCode;\n\
         fn main() -> ExitCode {\n\
         \x20   let base = 30u32;\n\
         \x20   let h = std::thread::spawn(move || base.wrapping_add(12));\n\
         \x20   let v = h.join().unwrap();\n\
         \x20   ExitCode::from((v & 0xff) as u8)\n\
         }\n";

    let Some((_, exit)) = svm_run_std_threads_parallel("svm_stdt_spawn_par", src) else {
        eprintln!("note: skipping std_guest threads (build-std produced no .ll)");
        return;
    };
    assert_eq!(
        exit, 42,
        "spawn(move || 30+12).join() == 42 under the parallel driver"
    );
}

/// Smoke: the `std::sync::mpsc` producer/consumer under real OS threads — the futex **notify/wake** path
/// end to end. An empty `recv` parks the consumer until the producer's `send` wakes it; the channel
/// close ends the loop. The sum (1..=9 = 45) is schedule-independent (every value is delivered exactly
/// once), so it holds under genuine parallelism. This is the parallel-driver exercise of `memory.wait`/
/// `memory.notify` — the one primitive the lock-free counter smoke test does not touch.
#[test]
fn std_threads_parallel_smoke_mpsc_channel() {
    if lane_ready().is_none() {
        eprintln!("note: skipping std_guest threads (need the svm std overlay — see rust-svm/)");
        return;
    }

    let src = "#![feature(restricted_std)]\n\
         use std::process::ExitCode;\n\
         use std::sync::mpsc;\n\
         fn main() -> ExitCode {\n\
         \x20   let (tx, rx) = mpsc::channel();\n\
         \x20   let producer = std::thread::spawn(move || {\n\
         \x20       for i in 1..=9u32 { tx.send(i).expect(\"send\"); }\n\
         \x20   });\n\
         \x20   let mut sum = 0u32;\n\
         \x20   for v in rx { sum = sum.wrapping_add(v); }\n\
         \x20   producer.join().expect(\"join\");\n\
         \x20   ExitCode::from((sum & 0xff) as u8)\n\
         }\n";

    let Some((_, exit)) = svm_run_std_threads_parallel("svm_stdt_mpsc_par", src) else {
        eprintln!("note: skipping std_guest threads (build-std produced no .ll)");
        return;
    };
    assert_eq!(
        exit, 45,
        "mpsc producer/consumer summed 1..=9 = 45 across two OS threads (futex notify/wake)"
    );
}
