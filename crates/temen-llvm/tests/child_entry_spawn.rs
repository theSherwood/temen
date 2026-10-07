//! **#1011 slice 3c — the on-ramp emits §14 child-entry phases.** For a guest-orchestrated nim driver
//! to spawn a compiler phase as a §14 child, the phase module's entry must be a §14
//! **child entry** — `(i64 starter) -> (i64 status)` — not the paramless top-level powerbox `_start`.
//! `TranslateOptions::child_entry` (this slice) synthesizes exactly that: the same powerbox prologue
//! (heap seed, ctors), but taking the starter capability (ignored) and returning `main`'s result widened
//! to the `i64` status the parent reads back via `join`.
//!
//! This proves it end-to-end: a real Rust program compiled in child-entry mode is spawned by
//! [`temen_run::conductor`] into a window of its own, runs (it uses `snprintf`, which forces a
//! synthesized powerbox `_start` — the exact case that was previously un-spawnable), and its `main`
//! status flows back through `join`, on the tree-walk oracle and every bytecode driver
//! ([`drivers::ALL`]). Window confinement (§2) is unchanged: the child is masked to its own window like
//! any §14 child.

#![cfg(target_os = "linux")]

use temen_interp::{Host, Value};

#[path = "../../temen-interp/tests/support/drivers.rs"]
mod drivers;

// A Rust guest that forces a synthesized powerbox `_start` (`snprintf` needs the powerbox window
// layout) but needs no runtime capability — isolating the child-entry ENTRY ABI. `main` returns 42.
const GUEST: &str = r##"
#![no_std]
#![allow(internal_features)]
#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! { loop {} }
#[no_mangle]
pub extern "C" fn rust_eh_personality() {}
extern "C" {
    fn snprintf(buf: *mut u8, n: usize, fmt: *const u8, ...) -> i32;
}
#[no_mangle]
pub extern "C" fn main() -> i32 {
    let mut buf = [0u8; 16];
    unsafe { snprintf(buf.as_mut_ptr(), 16, b"%d\0".as_ptr(), 42i32); }
    // Return the first two ASCII digits folded back to a number, so the result actually depends on
    // snprintf having run: '4','2' -> 42. (If snprintf were a no-op the buffer stays 0 and this is 0.)
    let d0 = (buf[0].wrapping_sub(b'0')) as i32;
    let d1 = (buf[1].wrapping_sub(b'0')) as i32;
    d0 * 10 + d1
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
fn child_entry_module_is_instantiable() {
    let dir = std::env::temp_dir().join(format!("ce_spawn_{}", std::process::id()));
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
    assert_eq!(
        child.funcs[0].params,
        vec![temen_ir::ValType::I64],
        "func 0 is a §14 child entry"
    );
    assert_eq!(child.funcs[0].results, vec![temen_ir::ValType::I64]);

    let parent = temen_run::conductor(&[], &[]);
    let powerbox = || {
        let mut host = Host::new();
        let (inst, modh, budget) = temen_run::grant_conductor(&mut host, &child);
        (host, [inst, modh, budget].map(Value::I32).to_vec())
    };
    for driver in drivers::ALL {
        let ran = drivers::run_on(driver, &parent, &powerbox)
            .unwrap_or_else(|| panic!("{driver:?} declined the conductor"));
        assert_eq!(
            ran.result,
            Ok(vec![Value::I64(42)]),
            "{driver:?}: the child-entry Rust module was spawned and its main status (42) joined back"
        );
    }
}
