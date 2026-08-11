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

/// Translate + verify + run `src` on the powerbox **with a granted `posix` cap** (`run_with_caps`),
/// letting `seed` stage the personality (env, a pinned clock, …) → `(stdout, exit)`. This is the
/// richer-`std::sys` path (`time` here, `fs`/`env` later) that reaches the host via `__vm_host_call`.
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
