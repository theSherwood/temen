//! **Minimal repro / bring-up: a manifest-import child on the JIT.** `child_entry_io` proves a
//! child-entry `write("hi")` guest binds its `write` import to a re-granted `stdout` on the tree-walk
//! oracle and every bytecode driver. This is the **JIT** case — the smallest real phase child (one
//! `Stream` manifest import, no `malloc`, no `fs`), spawned by `temen_run::conductor` and run through
//! the granted-spawn hooks — isolating the `call.import`-in-a-JIT-child dispatch that the full `nifler`
//! child of the since-retired LLVM route tripped a `CapFault` on. Gated to Linux + `rustc`.

#![cfg(target_os = "linux")]

use core::ffi::c_void;
use temen_interp::{Host, StreamRole, Value};
use temen_jit::{compile_and_run_capture_reserved_with_host_ex, GrantChildHooks, JitOutcome};

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

// A child that `malloc`s (forcing the synthesized allocator's `vm_map` against the child's auto-granted
// AddressSpace) — the feature the write-only guest lacks and `nifler` has. Writes through the pointer,
// returns the byte, so a correct run returns 42.
const GUEST_MALLOC: &str = r##"
#![no_std]
#![allow(internal_features)]
#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! { loop {} }
#[no_mangle]
pub extern "C" fn rust_eh_personality() {}
extern "C" {
    fn malloc(n: usize) -> *mut u8;
}
#[no_mangle]
pub extern "C" fn main() -> i32 {
    let p = unsafe { malloc(4096) };
    if p.is_null() { return -1; }
    unsafe {
        *p = 42;
        core::ptr::read_volatile(p) as i32
    }
}
"##;

/// #1234 — the production table, derived from one [`temen_run::CapCtx`] so the hook family and
/// the parent pointer it decodes are chosen together (this used to hand-roll both, and nothing
/// checked that the pointer matched the ctx the run baked).
fn grant_hooks(host: *mut temen_interp::Host) -> GrantChildHooks {
    temen_run::production_grant_hooks(temen_run::CapCtx::Raw(host))
}

#[test]
fn child_entry_write_binds_stdout_on_the_jit() {
    let dir = std::env::temp_dir().join(format!("ce_io_jit_{}", std::process::id()));
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
    let parent = temen_run::conductor(&["stdout"], &[]);

    let mut host = Host::new();
    let sink = host.shared_stdout();
    let out_h = host.grant_stream(StreamRole::Out);
    let (inst, modh, budget) = temen_run::grant_conductor(&mut host, &child);

    let args = [inst as i64, modh as i64, budget as i64, out_h as i64];
    let (jo, _) = compile_and_run_capture_reserved_with_host_ex(
        &parent,
        0,
        &args,
        &[],
        temen_ir::DEFAULT_RESERVED_LOG2,
        temen_run::cap_thunk,
        &mut host as *mut Host as *mut c_void,
        Some(temen_run::module_resolver),
        Some(grant_hooks(&mut host as *mut Host)),
    )
    .expect("jit run");
    // The conductor answers `(status, value)`, as the join did (#2066).
    let status = match jo {
        JitOutcome::Returned(ref v) if v.len() == 2 && v[0] == 0 => v[1],
        ref o => panic!("the child did not return on the jit: {o:?}"),
    };
    assert!(
        matches!(status, 0),
        "child status 0 joined back on the JIT: {status:?}"
    );
    let _ = Value::I64(0);
    assert_eq!(
        &sink.lock().unwrap()[..],
        b"hi",
        "the child-entry write bound to the re-granted stdout on the JIT"
    );
}

#[test]
fn child_entry_malloc_binds_vm_map_on_the_jit() {
    let dir = std::env::temp_dir().join(format!("ce_malloc_jit_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("g.rs");
    let ll = dir.join("g.ll");
    std::fs::write(&src, GUEST_MALLOC).unwrap();
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
    // The child's heap grows past its declared window into the window's reserved tail through
    // `vm_map`, bound to its auto-granted AddressSpace — so the grant list is empty.
    let parent = temen_run::conductor(&[], &[]);

    let mut host = Host::new();
    let (inst, modh, budget) = temen_run::grant_conductor(&mut host, &child);

    let args = [inst as i64, modh as i64, budget as i64];
    let (jo, _) = compile_and_run_capture_reserved_with_host_ex(
        &parent,
        0,
        &args,
        &[],
        temen_ir::DEFAULT_RESERVED_LOG2,
        temen_run::cap_thunk,
        &mut host as *mut Host as *mut c_void,
        Some(temen_run::module_resolver),
        Some(grant_hooks(&mut host as *mut Host)),
    )
    .expect("jit run");
    // The conductor answers `(status, value)`, as the join did (#2066).
    let status = match jo {
        JitOutcome::Returned(ref v) if v.len() == 2 && v[0] == 0 => v[1],
        ref o => panic!("the child did not return on the jit: {o:?}"),
    };
    assert_eq!(
        status, 42,
        "the child-entry malloc bound vm_map to its AddressSpace on the JIT"
    );
}
