//! **#1011 slice 3c — an argv-taking §14 child entry.** A real nim phase runs as `nifler p <in> <out>`,
//! so the child-entry mode must handle a `main(argc, argv)`, not just `main(void)`. The argv `_start`
//! (`synth_start_argv`) now honors child-entry: it takes the starter capability (ignored), parses the
//! parent-seeded args buffer at `POWERBOX_ARGS_BASE` into `argv[]` exactly as a top-level run does, and
//! returns `main`'s result widened to the i64 status the parent joins. This drives that `_start`
//! directly with a seeded args buffer (the same buffer an op-13 parent seeds into the child's carve).

#![cfg(target_os = "linux")]

use temen_interp::{run_capture_reserved_with_host, Host, Value};

// A child-entry Rust guest `main(argc, argv)`. It calls `snprintf` (forcing a synthesized powerbox
// `_start` — no runtime cap needed) and returns `argc*100 + argv[1][0]`, so the result depends on both
// the argc count and following `argv[1]`'s pointer to its bytes.
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
pub extern "C" fn main(argc: i32, argv: *const *const u8) -> i32 {
    let mut b = [0u8; 8];
    // Format a *runtime* value so the optimizer can't const-fold + eliminate the call (which would drop
    // the powerbox `_start`); a volatile read then keeps it alive but contributes 0 to the result.
    unsafe { snprintf(b.as_mut_ptr(), 8, b"%d\0".as_ptr(), argc); }
    let keep = unsafe { core::ptr::read_volatile(b.as_ptr()) } as i32;
    unsafe {
        let a1 = *argv.add(1);   // argv[1]
        let c = *a1;             // argv[1][0]
        argc * 100 + c as i32 + (keep & 0)
    }
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
fn child_entry_parses_argv() {
    let dir = std::env::temp_dir().join(format!("ce_argv_{}", std::process::id()));
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
    let t =
        temen_llvm::translate_ll_path_with_options(&ll, opts).expect("translate child-entry argv");
    temen_verify::verify_module(&t.module).expect("child verifies");
    // func 0 is the child entry: a starter cap in, an i64 status out.
    assert_eq!(
        t.module.funcs[0].params,
        vec![temen_ir::ValType::I64],
        "child entry takes the starter"
    );
    assert_eq!(
        t.module.funcs[0].results,
        vec![temen_ir::ValType::I64],
        "child entry returns i64 status"
    );

    // Seed the §3e args buffer at the child's args base: `{argc=3, envc=0}` + packed "p\0Z\0q\0".
    // This is exactly what an op-13 parent seeds into the child's carve. #964/#1094: a guarded child
    // reads argv one guard up, so key off `module_args_base` (== POWERBOX_ARGS_BASE for a legacy child).
    let base = temen_ir::module_args_base(&t.module) as usize;
    let mut init = vec![0u8; base + 32];
    init[base..base + 4].copy_from_slice(&3u32.to_le_bytes());
    init[base + 4..base + 8].copy_from_slice(&0u32.to_le_bytes());
    init[base + 8..base + 8 + 6].copy_from_slice(b"p\0Z\0q\0");

    let mut host = Host::new();
    let mut fuel = 50_000_000u64;
    // Run func 0 (the child-entry `_start`) with a dummy starter capability (ignored).
    let (r, _) = run_capture_reserved_with_host(
        &t.module,
        0,
        &[Value::I64(0)],
        &mut fuel,
        &init,
        0,
        &mut host,
    );
    let out = match r.expect("run").as_slice() {
        [Value::I64(x)] => *x,
        [Value::I32(x)] => *x as i64,
        other => panic!("result: {other:?}"),
    };
    // 3 (argc) * 100 + 'Z' (90) = 390 — proves argc AND the argv[1] pointer indirection.
    assert_eq!(
        out, 390,
        "the child-entry argv `_start` parsed argc and argv[1] and joined the status"
    );
}
