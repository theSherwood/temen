//! **#1011 slice 3c — the on-ramp emits §14 child-entry phases.** For a guest-orchestrated nim driver
//! to spawn a compiler phase via `instantiate_module` (op 13/5), the phase module's entry must be a §14
//! **child entry** — `(i64 starter) -> (i64 status)` — not the paramless top-level powerbox `_start`.
//! `TranslateOptions::child_entry` (this slice) synthesizes exactly that: the same powerbox prologue
//! (heap seed, ctors), but taking the starter capability (ignored) and returning `main`'s result widened
//! to the `i64` status the parent reads back via `join`.
//!
//! This proves it end-to-end: a real Rust program compiled in child-entry mode is `instantiate_module`d
//! as a confined child by an SVM parent, runs (it uses `snprintf`, which forces a synthesized powerbox
//! `_start` — the exact case that was previously un-spawnable), and its `main` status flows back through
//! `join`. Window confinement (§2) is unchanged: the child is masked to its carve like any §14 child.

#![cfg(target_os = "linux")]

use std::sync::Arc;
use svm_interp::{bytecode, Host, Region, Trap, Value};

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

/// A raw window base that keeps derived provenance (offset into the one live allocation).
#[derive(Clone, Copy)]
struct WinPtr(*mut u8);

/// Drive one vCPU of the run to completion, servicing §14 instantiate events (op-5 here: no grant list,
/// so `take_granted_host` is `None` and the child runs with the plain confined constructor).
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
                // child); the child's region aliases that sub-window — the §14 shared data plane.
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

    let opts = svm_llvm::TranslateOptions {
        child_entry: true,
        ..Default::default()
    };
    let child = svm_llvm::translate_ll_path_with_options(&ll, opts)
        .expect("translate child-entry")
        .module;
    svm_verify::verify_module(&child).expect("child verifies");
    assert_eq!(
        child.funcs[0].params,
        vec![svm_ir::ValType::I64],
        "func 0 is a §14 child entry"
    );
    assert_eq!(child.funcs[0].results, vec![svm_ir::ValType::I64]);
    let sl = child.memory.expect("child window").size_log2; // e.g. 21

    // An SVM parent that `instantiate_module`s the child (op 5) into a carve one child-window up, then
    // joins its status. `v0` = Instantiator, `v1` = the granted child Module handle.
    let carve_off = 1u64 << sl;
    let parent_src = format!(
        r#"memory {psl}
func (i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32) {{
  vmh = i64.extend_i32_u v1
  ventry = i64.const 0
  voff = i64.const {off}
  vsl = i64.const {sl}
  vq = i64.const 0
  vh = cap.call 6 5 (i64, i64, i64, i64, i64) -> (i32) v0 (vmh, ventry, voff, vsl, vq)
  vr = cap.call 6 1 (i32) -> (i64) v0 (vh)
  return vr
  }}
}}
"#,
        psl = sl + 1,
        off = carve_off,
        sl = sl,
    );
    let parent = svm_text::parse_module(&parent_src).expect("parse parent");
    svm_verify::verify_module(&parent).expect("verify parent");
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
        Ok(vec![Value::I64(42)]),
        "the child-entry Rust module was instantiate_module'd and its main status (42) joined back"
    );
}
