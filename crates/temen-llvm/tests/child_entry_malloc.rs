//! **#1011 slice 3c — an allocating §14 child grows its heap inside its carve.** A real nim phase
//! (nifler) `malloc`s, so a child-entry module must be able to grow its heap. Two enabling changes are
//! exercised here: (1) `bind_child_manifest` now binds the `vm_map` family to the child's auto-granted
//! `AddressSpace` cap (whose range is exactly `[0, child_size)`), so the allocator's page-commit no
//! longer `CapFault`s; (2) the resumable engine's `mod_ok` accepts a carve **larger** than the module's
//! declared memory, giving heap room above `heap_base` — all still masked to the carve (§2 unchanged).
//!
//! The guest allocates a growing `Vec` (forcing the synthesized `malloc` → `vm_map`) and returns its
//! checksum; it is `instantiate_module`'d into a carve one power-of-two above its declared window, and
//! its result joins back correctly — proving the heap grew inside the confined child.

#![cfg(target_os = "linux")]

use std::sync::Arc;
use temen_interp::{bytecode, Host, Region, Trap, Value};

// A child-entry Rust guest that allocates. `malloc`/`free` externs + a `GlobalAlloc` over them force the
// on-ramp to synthesize the `vm_map`-backed bump allocator; the `Vec` growth exercises it. Returns
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

#[derive(Clone, Copy)]
struct WinPtr(*mut u8);

fn drive(
    prog: &bytecode::VcpuProgram,
    base: WinPtr,
    mut vcpu: bytecode::Vcpu<'_>,
) -> Result<Vec<Value>, Trap> {
    let mut children: Vec<Result<Vec<Value>, Trap>> = Vec::new();
    loop {
        match vcpu.run() {
            bytecode::VcpuEvent::Done(v) => return Ok(v),
            bytecode::VcpuEvent::Trapped(t) => return Err(t),
            bytecode::VcpuEvent::Instantiate {
                module,
                entry,
                carve,
                size_log2,
                fuel,
            } => {
                let granted = vcpu.take_granted_host();
                // SAFETY: the engine validated the carve within this vCPU's window (which outlives the
                // child); the child's region aliases that sub-window.
                let child_base = WinPtr(unsafe { base.0.add(carve as usize) });
                // SAFETY: `2^size_log2` valid bytes at the validated carve.
                let back = Arc::new(unsafe { Region::shared(child_base.0, 1u64 << size_log2) });
                let child = match granted {
                    Some(host) => bytecode::Vcpu::new_confined_child_over_host(
                        prog, module, entry, back, size_log2, fuel, host,
                    ),
                    None => bytecode::Vcpu::new_confined_child(
                        prog, module, entry, back, size_log2, fuel,
                    ),
                }
                .expect("confined child builds");
                let r = drive(prog, child_base, child);
                let handle = children.len() as i32;
                children.push(r);
                vcpu.deliver_handle(handle);
            }
            bytecode::VcpuEvent::Join { handle } => {
                vcpu.deliver_join(children[handle as usize].clone());
            }
            _ => panic!("unexpected event"),
        }
    }
}

#[test]
fn child_entry_malloc_grows_heap_in_carve() {
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
    let declared = child.memory.expect("child window").size_log2;
    // Spawn into a carve one power-of-two ABOVE the declared memory — the extra span is heap room the
    // allocator grows into via `vm_map` (the relaxed `mod_ok` admits carve > declared).
    let sl = declared + 1;

    let carve_off = 1u64 << sl;
    let parent_src = format!(
        r#"memory {psl}
func (i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32) {{
  vmh = i64.extend_i32_u v1
  ventry = i64.const 0
  voff = i64.const {carve_off}
  vsl = i64.const {sl}
  vq = i64.const 0
  vh = cap.call 6 5 (i64, i64, i64, i64, i64) -> (i32) v0 (vmh, ventry, voff, vsl, vq)
  vr = cap.call 6 1 (i32) -> (i64) v0 (vh)
  return vr
  }}
}}
"#,
        psl = sl + 1,
    );
    let parent = temen_text::parse_module(&parent_src).expect("parse parent");
    temen_verify::verify_module(&parent).expect("verify parent");
    let prog = bytecode::VcpuProgram::compile(&parent).expect("compile parent");

    let win = 1u64 << (sl + 1);
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, win);
    let modh = host.grant_module(&child);

    let size = win as usize;
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero layout; `size` valid 8-aligned bytes owned here until freed below.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `base` addresses `size` valid bytes, exclusively this run's, freed only after the vCPUs.
    let back = Arc::new(unsafe { Region::shared(base, win) });

    let root = bytecode::Vcpu::new_root_with_powerbox(
        &prog,
        0,
        &[Value::I32(inst), Value::I32(modh)],
        Arc::clone(&back),
        &[],
        host,
    )
    .expect("root vcpu");
    let r = drive(&prog, WinPtr(base), root);

    drop(back);
    // SAFETY: same layout; every vCPU and region view is dropped, so no borrow outlives this.
    unsafe { std::alloc::dealloc(base, layout) };

    assert_eq!(
        r,
        Ok(vec![Value::I64(4950)]),
        "the allocating child grew its heap inside the carve (sum 0..100) and joined its result"
    );
}
