//! **#1011 slice 3c — a §14 child-entry phase binds its imports on the *resumable* (tier-up) engine.**
//! `child_entry_io.rs` proved a child-entry guest's `write` binds to a re-granted `stdout` on the
//! **cooperative** engine (which binds the child manifest inline). But a nim phase child JITs only on
//! the **resumable** engine (`new_confined_child` / `new_confined_child_over_host`), which previously
//! did *not* bind the child manifest — so a phase child with a `write`/`fs` import would `CapFault`.
//! This proves the wiring: `new_confined_child_core` now calls `bind_child_manifest`, so the same
//! child-entry `write` guest, `instantiate_module_named` (op 13)'d over the **resumable drive loop**
//! with `stdout` re-granted, reaches the shared sink. Window confinement (§2) is untouched: the grant
//! is authority (§3), a cross-tier `call.cap`, not a window access.

#![cfg(target_os = "linux")]

use std::sync::Arc;
use temen_interp::{bytecode, Host, Region, StreamRole, Trap, Value};

// A Rust guest compiled `--child-entry` that writes "hi" to fd 1. `write` is a `Stream` manifest import
// that must bind to the re-granted `stdout` at spawn.
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

/// A raw window base that keeps derived provenance (offset into the one live allocation).
#[derive(Clone, Copy)]
struct WinPtr(*mut u8);

/// The resumable-engine drive loop: on `Instantiate`, take the op-13 re-granted powerbox
/// (`take_granted_host`) and run the child over it (`new_confined_child_over_host`, which binds the
/// child manifest against that powerbox); a grant-less child would use the plain constructor.
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
fn child_entry_binds_imports_on_the_resumable_engine() {
    let dir = std::env::temp_dir().join(format!("ce_io_res_{}", std::process::id()));
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
    let sl = child.memory.expect("child window").size_log2;

    // The `"stdout"` name packed into one i64, and the grant record's first word
    // (`name_off:u32=2048 | name_len:u32=6`). The parent lays the record, then op-13-spawns the child
    // re-granting the `stdout` handle (arg `v2`) under that name.
    let name_i64: u64 = b"stdout"
        .iter()
        .enumerate()
        .fold(0u64, |acc, (i, &b)| acc | ((b as u64) << (8 * i)));
    let word0: u64 = 2048 | (6u64 << 32);
    let carve_off: u64 = 1u64 << sl;
    let parent_src = format!(
        r#"memory {psl}
func (i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32) {{
  vname = i64.const {name_i64}
  vnoff = i64.const 2048
  i64.store vnoff vname
  vrec0 = i64.const {word0}
  vrecoff = i64.const 1024
  i64.store vrecoff vrec0
  vsh = i64.extend_i32_u v2
  vrec1off = i64.const 1032
  i64.store vrec1off vsh
  vmh = i64.extend_i32_u v1
  vgptr = i64.const 1024
  vgn = i64.const 1
  ventry = i64.const 0
  voff = i64.const {carve_off}
  vsl = i64.const {sl}
  vq = i64.const 0
  vh = call.cap 6 13 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmh, vgptr, vgn, ventry, voff, vsl, vq)
  vr = call.cap 6 1 (i32) -> (i64) v0 (vh)
  return vr
  }}
}}
"#,
        psl = sl + 1,
    );
    let parent = temen_text::parse_module(&parent_src).expect("parse parent");
    temen_verify::verify_module(&parent).expect("verify parent");
    let prog = bytecode::VcpuProgram::compile(&parent).expect("compile parent");

    let mut host = Host::new();
    let sink = host.shared_stdout();
    let out_h = host.grant_stream(StreamRole::Out);
    let win = 1u64 << (sl + 1);
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
        &[Value::I32(inst), Value::I32(modh), Value::I32(out_h)],
        Arc::clone(&back),
        &[],
        host,
    )
    .expect("root vcpu");
    let r = drive(&prog, WinPtr(base), root);

    drop(back);
    // SAFETY: same layout; every vCPU and region view is dropped, so no borrow outlives this.
    unsafe { std::alloc::dealloc(base, layout) };

    assert!(
        matches!(r, Ok(ref v) if matches!(v.as_slice(), [Value::I64(0)] | [Value::I32(0)])),
        "child status 0 joined back on the resumable engine: {r:?}"
    );
    assert_eq!(
        &*sink.lock().unwrap(),
        b"hi",
        "the child-entry write bound to the re-granted stdout on the resumable engine (manifest bound in new_confined_child)"
    );
}
