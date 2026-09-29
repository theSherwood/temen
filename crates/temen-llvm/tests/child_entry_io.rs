//! **#1011 slice 3c (2b) — a §14 child-entry phase does real I/O through a re-granted cap.** A compiler
//! phase spawned as a child must reach its `stdout`/`fs` — which arrive as **manifest imports** bound at
//! spawn to the child's re-granted named caps (`Host::bind_child_manifest`). This proves that binding
//! for an on-ramp child-entry module: a real Rust guest compiled `--child-entry` that calls `write(1,
//! …)` is spawned (`temen_run::conductor`, its own window) with `stdout` re-granted, and its bytes land in the
//! shared sink — the exact hand-off a JIT'd `nifler` uses to reach its `fs`. (The cooperative engine
//! binds the child manifest inline; wiring the same into the resumable/tier-up path is a follow-up.)

#![cfg(target_os = "linux")]

use temen_interp::{run_with_host, Host, StreamRole, Value};

// A Rust guest that writes "hi" to fd 1. `write` lowers to a `Stream` manifest import, which (a) forces
// a synthesized powerbox `_start` and (b) must bind to the re-granted `stdout` at spawn. Compiled
// `--child-entry`, its `_start` is the §14 child ABI.
const GUEST: &str = r##"
#![no_std]
#![allow(internal_features)]
#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! { loop {} }
#[no_mangle]
pub extern "C" fn rust_eh_personality() {}
extern "C" {
    fn write(fd: i64, buf: *const u8, n: i64) -> i64;
}
static MSG: [u8; 2] = *b"hi";
#[no_mangle]
pub extern "C" fn main() -> i32 {
    unsafe { write(1, MSG.as_ptr(), 2); }
    0
}
"##;

fn emit_ll(src: &std::path::Path, ll: &std::path::Path) -> bool {
    std::process::Command::new("rustc")
        .args([
            "--edition",
            "2021",
            "-O",
            "-Cpanic=abort",
            "--emit=llvm-ir",
            "--crate-type=cdylib",
        ])
        .arg(src)
        .arg("-o")
        .arg(ll)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[test]
fn child_entry_writes_through_a_regranted_stdout() {
    let dir = std::env::temp_dir().join(format!("ce_io_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("g.rs");
    let ll = dir.join("g.ll");
    std::fs::write(&src, GUEST).unwrap();
    if !emit_ll(&src, &ll) {
        eprintln!("note: skipping (rustc unavailable)");
        return;
    }

    let opts = temen_llvm::TranslateOptions {
        child_entry: true,
        ..Default::default()
    };
    let child = temen_llvm::translate_ll_path_with_options(&ll, opts)
        .expect("translate child-entry")
        .module;
    temen_verify::verify_module(&child).expect("child verifies");

    // The parent spawns the child re-granting the `stdout` handle under that name.
    let parent = temen_run::conductor(&["stdout"], &[]);

    let mut host = Host::new();
    let sink = host.shared_stdout(); // the shared Out sink we read after the run
    let out_h = host.grant_stream(StreamRole::Out);
    let (inst, modh, budget) = temen_run::grant_conductor(&mut host, &child);

    let mut fuel = 200_000_000u64;
    let r = run_with_host(
        &parent,
        0,
        &[
            Value::I32(inst),
            Value::I32(modh),
            Value::I32(budget),
            Value::I32(out_h),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("parent run");

    // The child's `main` returned 0, joined back.
    assert!(
        matches!(r.as_slice(), [Value::I64(0)] | [Value::I32(0)]),
        "child status 0 joined back: {r:?}"
    );
    assert_eq!(
        &*sink.lock().unwrap(),
        b"hi",
        "the child-entry guest's write bound to the re-granted stdout and reached the shared sink"
    );
}
