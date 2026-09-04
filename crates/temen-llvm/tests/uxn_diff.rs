//! Uxn **frame-hash differential** (the §18 oracle for `crates/temen-run/demos/uxn`): the same
//! `uxn_diff.c` — the Uxn CPU + Varvara devices + a headless driver that runs the demo ROM for
//! `UXN_DIFF_FRAMES` frames under a fixed key script and prints an FNV-1a hash of every composed
//! frame — built once with native `cc` and once through the on-ramp (`clang -O2 -emit-llvm` →
//! `translate_ll_path`), fed the committed demo ROM on stdin. The two hash streams must be
//! byte-identical on every backend: the composed framebuffer of the sandboxed Uxn is exactly the
//! native one, frame for frame.
//!
//! Also pins the committed ROM: `demo.tal` assembled by the in-tree `uxnasm.c` (built with `cc`) must
//! equal `browser/web/assets/uxn_demo.rom`, so a demo edit without `ONLY=uxn bash
//! scripts/rebuild-assets.sh` fails here rather than shipping stale.
//!
//! Gated on `clang` + `cc` (the on-ramp's own dependencies); skipped if absent.

use std::path::PathBuf;
use std::process::Command;
use temen_run::{Backend, Limits, Outcome, RunConfig};

const FRAMES: u32 = 120;

fn demo_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../temen-run/demos/uxn")
}

fn rom() -> Vec<u8> {
    std::fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../browser/web/assets/uxn_demo.rom"),
    )
    .expect("the committed demo ROM")
}

/// `cc` the file to `exe`; `None` if `cc` is unavailable.
fn cc(args: &[&str], exe: &PathBuf) -> Option<()> {
    match Command::new("cc").args(args).arg("-o").arg(exe).status() {
        Ok(s) if s.success() => Some(()),
        _ => None,
    }
}

#[test]
fn committed_rom_is_fresh() {
    let dir = std::env::temp_dir();
    let asm = dir.join(format!("temen_uxnasm_{}", std::process::id()));
    if cc(
        &["-O2", demo_dir().join("uxnasm.c").to_str().unwrap()],
        &asm,
    )
    .is_none()
    {
        eprintln!("note: skipping committed_rom_is_fresh (cc unavailable)");
        return;
    }
    let out = dir.join(format!("temen_uxn_{}.rom", std::process::id()));
    let st = Command::new(&asm)
        .arg(demo_dir().join("demo.tal"))
        .arg(&out)
        .status()
        .expect("run uxnasm");
    assert!(st.success(), "uxnasm assembles demo.tal");
    assert_eq!(
        std::fs::read(&out).unwrap(),
        rom(),
        "browser/web/assets/uxn_demo.rom is stale — run `ONLY=uxn bash scripts/rebuild-assets.sh`"
    );
}

#[test]
fn guest_frame_hashes_match_native() {
    let dir = std::env::temp_dir();
    let src = demo_dir().join("uxn_diff.c");
    let frames = format!("-DUXN_DIFF_FRAMES={FRAMES}");
    let ll = dir.join(format!("temen_uxn_diff_{}.ll", std::process::id()));
    let exe = dir.join(format!("temen_uxn_diff_{}", std::process::id()));
    let clang = Command::new("clang")
        .args([
            "-O2",
            "-emit-llvm",
            "-S",
            "-fno-vectorize",
            "-fno-slp-vectorize",
            &frames,
        ])
        .arg(&src)
        .arg("-o")
        .arg(&ll)
        .status();
    if !matches!(clang, Ok(s) if s.success()) {
        eprintln!("note: skipping uxn_diff (clang unavailable)");
        return;
    }
    if cc(&["-O2", &frames, src.to_str().unwrap()], &exe).is_none() {
        eprintln!("note: skipping uxn_diff (cc unavailable)");
        return;
    }
    let rom = rom();

    // Native oracle: the same binary over the same ROM.
    let mut native = Command::new(&exe)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("run native uxn_diff");
    std::io::Write::write_all(native.stdin.as_mut().unwrap(), &rom).unwrap();
    drop(native.stdin.take());
    let native = native.wait_with_output().expect("native output");
    assert!(native.status.success());
    let native = String::from_utf8(native.stdout).unwrap();
    assert_eq!(
        native.lines().count(),
        FRAMES as usize,
        "one hash line per frame"
    );
    let unique: std::collections::HashSet<&str> = native.lines().collect();
    assert!(
        unique.len() > FRAMES as usize / 2,
        "the swarm animates (frames differ)"
    );

    // Guest: translate, then run `_start` on each backend with the ROM on stdin.
    let t = temen_llvm::translate_ll_path(&ll).expect("translate uxn_diff.c");
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
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let run = inst
            .run(backend, &config)
            .unwrap_or_else(|e| panic!("{backend:?} run failed: {e}"));
        match run.outcome {
            Outcome::Returned(_) | Outcome::Exited(0) => {}
            other => panic!("{backend:?}: unexpected outcome {other:?}"),
        }
        let guest = String::from_utf8(run.stdout).unwrap();
        assert_eq!(
            guest, native,
            "{backend:?}: frame hashes differ from native"
        );
    }
    eprintln!("uxn_diff: {FRAMES} frames byte-identical to native on TreeWalk/Bytecode/Jit");
}

/// The CPU against the **golden opcode corpus** (`demos/uxn/corpus/opcodes.corpus`): 303 programs
/// whose end states were recorded from uxn5's spec-compliant core — every non-control-flow opcode in
/// every mode over random operands (stack and memory wrap-around included), every jump form, lambdas,
/// subroutines, and a primes program. `uxn_corpus.c` runs them all on `uxn.c` and exits non-zero on
/// any divergence, so the sandboxed Uxn's CPU is pinned to the reference with no dependency at test
/// time beyond `cc`.
#[test]
fn cpu_matches_golden_corpus() {
    let dir = std::env::temp_dir();
    let exe = dir.join(format!("temen_uxn_corpus_{}", std::process::id()));
    if cc(
        &["-O2", demo_dir().join("uxn_corpus.c").to_str().unwrap()],
        &exe,
    )
    .is_none()
    {
        eprintln!("note: skipping cpu_matches_golden_corpus (cc unavailable)");
        return;
    }
    let out = Command::new(&exe)
        .arg(demo_dir().join("corpus/opcodes.corpus"))
        .output()
        .expect("run uxn_corpus");
    assert!(
        out.status.success(),
        "uxn.c diverges from the golden corpus:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    eprintln!("{}", String::from_utf8_lossy(&out.stdout).trim());
}
