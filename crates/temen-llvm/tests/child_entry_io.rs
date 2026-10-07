//! **#1011 slice 3c (2b) — a §14 child-entry phase does real I/O through a re-granted cap.** A compiler
//! phase spawned as a child must reach its `stdout`/`fs` — which arrive as **manifest imports** bound at
//! spawn to the child's re-granted named caps (`Host::bind_child_manifest`). This proves that binding
//! for an on-ramp child-entry module: a real Rust guest compiled `--child-entry` that calls `write(1,
//! …)` is spawned (`temen_run::conductor`, its own window) with `stdout` re-granted, and its bytes land in the
//! parent's stdout — the exact hand-off a JIT'd `nifler` uses to reach its `fs`. It runs on the
//! tree-walk oracle and every bytecode driver ([`drivers::ALL`]): each binds the child manifest at the
//! one admission they share, the resumable `Vcpu` (the engine a phase tiers up on) among them.

#![cfg(target_os = "linux")]

use temen_interp::{Host, StreamRole, Value};

#[path = "../../temen-interp/tests/support/drivers.rs"]
mod drivers;

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
    let powerbox = || {
        let mut host = Host::new();
        let out_h = host.grant_stream(StreamRole::Out);
        let (inst, modh, budget) = temen_run::grant_conductor(&mut host, &child);
        (host, [inst, modh, budget, out_h].map(Value::I32).to_vec())
    };
    for driver in drivers::ALL {
        let ran = drivers::run_on(driver, &parent, &powerbox)
            .unwrap_or_else(|| panic!("{driver:?} declined the conductor"));
        // The child's `main` returned 0, joined back.
        assert_eq!(
            ran.result,
            Ok(vec![Value::I64(0)]),
            "{driver:?}: child status 0 joined back"
        );
        assert_eq!(
            ran.stdout, b"hi",
            "{driver:?}: the child-entry guest's write bound to the re-granted stdout and reached the \
             parent's stdout"
        );
    }
}
