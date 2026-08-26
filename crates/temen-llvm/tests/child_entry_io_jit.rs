//! **Minimal repro / bring-up: a manifest-import child on the JIT via op-13.** `child_entry_io` proves a
//! child-entry `write("hi")` guest binds its `write` import to a re-granted `stdout` on the cooperative
//! engine; `child_entry_io_resumable` does it on the resumable engine. This is the **JIT** case — the
//! smallest real op-13 phase-child (one `Stream` manifest import, no `malloc`, no `fs`) run through the
//! granted-spawn hooks — isolating the `call.import`-in-a-JIT-child dispatch that the full `nifler`
//! child (`nifler_child_jit`) trips a `CapFault` on. If this passes, nifler's fault is `malloc`/`fs`;
//! if it faults, the Stream import dispatch is the culprit. Gated to Linux + `rustc`.

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

fn grant_hooks() -> GrantChildHooks {
    GrantChildHooks {
        build: temen_run::grant_child_build,
        build_named: temen_run::grant_named_child_build,
        bind_imports: temen_run::child_bind_imports,
        release: temen_run::grant_child_release,
        mint: temen_run::child_offer_mint,
        thunk: temen_run::cap_thunk_locked,
        register_serve: temen_run::child_register_serve,
    }
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
    let sl = child.memory.expect("child window").size_log2;

    // A one-entry grant record `{"stdout" -> v2}` at 1024, name at 2048; op-13 into a carve at `1<<sl`.
    let word0: u64 = 2048 | (6u64 << 32);
    let carve_off: u64 = 1u64 << sl;
    let parent_src = format!(
        r#"memory {psl}
data 2048 "stdout"
func (i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32) {{
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

    let mut host = Host::new();
    let sink = host.shared_stdout();
    let out_h = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, 1u64 << (sl + 1));
    let modh = host.grant_module(&child);

    let args = [inst as i64, modh as i64, out_h as i64];
    let (jo, _) = compile_and_run_capture_reserved_with_host_ex(
        &parent,
        0,
        &args,
        &[],
        temen_ir::DEFAULT_RESERVED_LOG2,
        temen_run::cap_thunk,
        &mut host as *mut Host as *mut c_void,
        Some(temen_run::module_resolver),
        Some(grant_hooks()),
    )
    .expect("jit run");
    let status = match jo {
        JitOutcome::Returned(ref v) => v.first().copied().unwrap_or(-1),
        JitOutcome::Exited(c) => c as i64,
        ref o => panic!("jit ended abnormally: {o:?}"),
    };
    assert!(
        matches!(status, 0),
        "child status 0 joined back on the JIT: {status:?}"
    );
    let _ = Value::I64(0);
    assert_eq!(
        &*sink.lock().unwrap(),
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
    // A malloc child needs heap room above its declared window: the synthesized bump allocator's
    // `heap_base` is `1<<declared`, and it grows the heap up into `[1<<declared, carve)`. So the
    // carve must be **larger** than the declared window (FORK.md §8.6 / #773: `declared <= carve` —
    // a generous window is a safe superset, confinement still masks to the carve). Carve one
    // power-of-two above the declared window.
    let decl = child.memory.expect("child window").size_log2;
    let sl = decl + 1;
    let carve_off: u64 = 1u64 << sl;
    // No re-granted caps: `vm_map` binds to the child's *auto-granted* AddressSpace, so the grant list
    // is empty (grants_n = 0, grants_ptr unused).
    let parent_src = format!(
        r#"memory {psl}
func (i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32) {{
  vmh = i64.extend_i32_u v1
  vgptr = i64.const 0
  vgn = i64.const 0
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

    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 1u64 << (sl + 1));
    let modh = host.grant_module(&child);

    let args = [inst as i64, modh as i64];
    let (jo, _) = compile_and_run_capture_reserved_with_host_ex(
        &parent,
        0,
        &args,
        &[],
        temen_ir::DEFAULT_RESERVED_LOG2,
        temen_run::cap_thunk,
        &mut host as *mut Host as *mut c_void,
        Some(temen_run::module_resolver),
        Some(grant_hooks()),
    )
    .expect("jit run");
    let status = match jo {
        JitOutcome::Returned(ref v) => v.first().copied().unwrap_or(-1),
        JitOutcome::Exited(c) => c as i64,
        ref o => panic!("jit ended abnormally: {o:?}"),
    };
    assert_eq!(
        status, 42,
        "the child-entry malloc bound vm_map to its AddressSpace on the JIT"
    );
}

#[test]
fn child_entry_malloc_via_op5_binds_vm_map_on_the_jit() {
    // The op-13 malloc test proves a **named-grant** separate-module child (`call.cap 6 13`, empty
    // grant list) mallocs on the JIT. This proves the plain **op-5** spawn (`call.cap 6 5`, a granted
    // `Module` with no grant list at all) does too — the JIT's op-5 delegates to the op-13 powerbox
    // builder for a separate-module child, so it gets the same Instantiator + AddressSpace + bound
    // manifest and its `vm_map` heap growth works, matching the interpreter's op-5. (Before the
    // delegation, op-5 handed the child an *empty* powerbox and this `vm_map` CapFaulted.)
    let dir = std::env::temp_dir().join(format!("ce_malloc_op5_jit_{}", std::process::id()));
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
    // Carve one power-of-two above the declared window for heap room (`declared <= carve`).
    let decl = child.memory.expect("child window").size_log2;
    let sl = decl + 1;
    let carve_off: u64 = 1u64 << sl;
    // op-5: `call.cap 6 5 (module, entry, off, size_log2, quota)` — no grant list at all. `vm_map`
    // binds to the child's auto-granted AddressSpace via its manifest.
    let parent_src = format!(
        r#"memory {psl}
func (i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32) {{
  vmh = i64.extend_i32_u v1
  ventry = i64.const 0
  voff = i64.const {carve_off}
  vsl = i64.const {sl}
  vq = i64.const 0
  vh = call.cap 6 5 (i64, i64, i64, i64, i64) -> (i32) v0 (vmh, ventry, voff, vsl, vq)
  vr = call.cap 6 1 (i32) -> (i64) v0 (vh)
  return vr
  }}
}}
"#,
        psl = sl + 1,
    );
    let parent = temen_text::parse_module(&parent_src).expect("parse parent");
    temen_verify::verify_module(&parent).expect("verify parent");

    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 1u64 << (sl + 1));
    let modh = host.grant_module(&child);

    let args = [inst as i64, modh as i64];
    let (jo, _) = compile_and_run_capture_reserved_with_host_ex(
        &parent,
        0,
        &args,
        &[],
        temen_ir::DEFAULT_RESERVED_LOG2,
        temen_run::cap_thunk,
        &mut host as *mut Host as *mut c_void,
        Some(temen_run::module_resolver),
        Some(grant_hooks()),
    )
    .expect("jit run");
    let status = match jo {
        JitOutcome::Returned(ref v) => v.first().copied().unwrap_or(-1),
        JitOutcome::Exited(c) => c as i64,
        ref o => panic!("jit ended abnormally: {o:?}"),
    };
    assert_eq!(
        status, 42,
        "the op-5 child-entry malloc bound vm_map to its auto-granted AddressSpace on the JIT"
    );
}
