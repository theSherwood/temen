//! **#1011 slice 3c — an allocating §14 child grows its heap inside its own window.** A real nim phase
//! (nifler) `malloc`s, so a child-entry module must be able to grow its heap: `bind_child_manifest`
//! binds the `vm_map` family to the child's auto-granted `AddressSpace` cap (whose range is exactly
//! `[0, child_size)`), so the allocator's page-commit does not `CapFault` — all still masked to the
//! child's window (§2 unchanged).
//!
//! The guest allocates a growing `Vec` (forcing the synthesized `malloc` → `vm_map`) and returns its
//! checksum; [`temen_run::conductor`] spawns it into a window of its own, and its result joins back
//! correctly on the tree-walk oracle and every bytecode driver ([`drivers::ALL`]) — proving the heap
//! grew inside the confined child.

#![cfg(target_os = "linux")]

use temen_interp::{Host, Value};

#[path = "../../temen-interp/tests/support/drivers.rs"]
mod drivers;

// A child-entry Rust guest that allocates. `malloc`/`free` externs + a `GlobalAlloc` over them make the
// on-ramp link its `vm_map`-backed heap; the `Vec` growth exercises it. Returns
// `sum(0..100) = 4950`.
const GUEST: &str = r##"
#![no_std]
#![allow(internal_features)]
extern crate alloc;
use alloc::vec::Vec;
use core::alloc::{GlobalAlloc, Layout};
extern "C" {
    fn malloc(n: usize) -> *mut u8;
    fn free(p: *mut u8);
}
struct A;
unsafe impl GlobalAlloc for A {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 { malloc(l.size()) }
    unsafe fn dealloc(&self, p: *mut u8, _l: Layout) { free(p) }
}
#[global_allocator]
static GA: A = A;
#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! { loop {} }
#[no_mangle]
pub extern "C" fn rust_eh_personality() {}
#[no_mangle]
pub extern "C" fn main() -> i32 {
    let mut v: Vec<i32> = Vec::new();
    let mut i = 0i32;
    while i < 100 { v.push(i); i += 1; }
    let mut s = 0i32;
    for &x in v.iter() { s += x; }
    s
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
fn child_entry_malloc_grows_heap_in_its_window() {
    let dir = std::env::temp_dir().join(format!("ce_malloc_{}", std::process::id()));
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
        .expect("translate child-entry malloc")
        .module;
    temen_verify::verify_module(&child).expect("child verifies");

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
            Ok(vec![Value::I64(4950)]),
            "{driver:?}: the allocating child grew its heap (sum 0..100) and joined its result"
        );
    }
}
