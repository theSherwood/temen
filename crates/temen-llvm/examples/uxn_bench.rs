//! Uxn benchmark — the bunnymark-style stress ROM (`demos/uxn/bench.tal`: 512 sprites bouncing on a
//! 320×240 screen) run headless for N frames by `uxn_bench.c` on **native `cc`**, the **bytecode
//! interpreter** and the **Cranelift JIT** (the same C through the LLVM on-ramp), reporting frames per
//! second and the ratio to native. Every run prints an FNV-1a hash of its last frame, asserted equal
//! across engines, so a faster engine is never a wrong one. The wasm tiers are the playground's
//! business: the Uxn card shows live fps on the interpreter and the wasm-JIT toggle.
//!
//!   cd crates/temen-llvm && cargo run --release --example uxn_bench        # UXN_BENCH_FRAMES=600
//!
//! Needs `cc` + `clang` (the on-ramp's own tools); skips with a note otherwise.

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;
use temen_run::{Backend, Limits, Outcome, RunConfig};

fn main() {
    let frames: usize = std::env::var("UXN_BENCH_FRAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(600);
    let demo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../temen-run/demos/uxn");
    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    let ok =
        |st: std::io::Result<std::process::ExitStatus>| st.map(|s| s.success()).unwrap_or(false);

    // The ROM: bench.tal through the native uxnasm.
    let asm = tmp.join(format!("temen_uxnasm_{pid}"));
    let rom_path = tmp.join(format!("temen_uxn_bench_{pid}.rom"));
    if !ok(Command::new("cc")
        .args(["-O2", "-o"])
        .arg(&asm)
        .arg(demo.join("uxnasm.c"))
        .status())
    {
        eprintln!("note: cc unavailable; skipping uxn_bench");
        return;
    }
    assert!(
        ok(Command::new(&asm)
            .arg(demo.join("bench.tal"))
            .arg(&rom_path)
            .status()),
        "assemble bench.tal"
    );
    let rom = std::fs::read(&rom_path).unwrap();

    // Native: uxn_bench.c with cc -O2, timed as a process (its startup is microseconds).
    let frames_def = format!("-DUXN_BENCH_FRAMES={frames}");
    let exe = tmp.join(format!("temen_uxn_bench_{pid}"));
    assert!(ok(Command::new("cc")
        .args(["-O2", &frames_def, "-o"])
        .arg(&exe)
        .arg(demo.join("uxn_bench.c"))
        .status()));
    let t0 = Instant::now();
    let mut child = Command::new(&exe)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    std::io::Write::write_all(child.stdin.as_mut().unwrap(), &rom).unwrap();
    drop(child.stdin.take());
    let native_out = String::from_utf8(child.wait_with_output().unwrap().stdout).unwrap();
    let native_secs = t0.elapsed().as_secs_f64();

    // Guest: clang → on-ramp → instantiate once; each backend runs `_start` with the ROM on stdin.
    let ll = tmp.join(format!("temen_uxn_bench_{pid}.ll"));
    if !ok(Command::new("clang")
        .args([
            "-O2",
            "-emit-llvm",
            "-S",
            "-fno-vectorize",
            "-fno-slp-vectorize",
            &frames_def,
            "-o",
        ])
        .arg(&ll)
        .arg(demo.join("uxn_bench.c"))
        .status())
    {
        eprintln!("note: clang unavailable; skipping the guest rows");
        return;
    }
    let t = temen_llvm::translate_ll_path(&ll).expect("translate uxn_bench.c");
    let inst = temen_run::instantiate(t.module).expect("instantiate");
    let config = RunConfig {
        limits: Limits {
            fuel: None,
            deadline: None,
            max_fibers: 0,
            max_vcpus: 0,
        },
        stdin: rom,
        memory_size_log2: None,
        args: vec![],
        env: vec![],
        ..RunConfig::default()
    };
    println!("uxn bench — bench.tal, {frames} frames, 512 sprites at 320x240 (frames/s; ratio = native/this)");
    println!(
        "{:<20} {:>10} {:>9} {:>8}   {}",
        "engine", "seconds", "fps", "ratio", "last-frame hash"
    );
    let native_fps = frames as f64 / native_secs;
    println!(
        "{:<20} {:>10.3} {:>9.0} {:>8.2}   {}",
        "native (cc -O2)",
        native_secs,
        native_fps,
        1.0,
        native_out.trim()
    );
    for (name, backend) in [
        ("temen-bytecode", Backend::Bytecode),
        ("temen-jit (cranelift)", Backend::Jit),
    ] {
        let t0 = Instant::now();
        let run = inst
            .run(backend, &config)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let secs = t0.elapsed().as_secs_f64();
        assert!(
            matches!(run.outcome, Outcome::Returned(_) | Outcome::Exited(0)),
            "{name}: {:?}",
            run.outcome
        );
        let out = String::from_utf8(run.stdout).unwrap();
        assert_eq!(
            out.trim(),
            native_out.trim(),
            "{name}: last-frame hash differs from native"
        );
        println!(
            "{:<20} {:>10.3} {:>9.0} {:>8.2}   {}",
            name,
            secs,
            frames as f64 / secs,
            secs / native_secs,
            out.trim()
        );
    }
}
