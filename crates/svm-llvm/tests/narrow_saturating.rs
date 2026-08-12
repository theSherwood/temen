//! Narrow-width saturating intrinsics (`llvm.{u,s}{add,sub}.sat.i{8,16}`), run on svm.
//!
//! Rust's `u8`/`i8`/`u16`/`i16` `saturating_add`/`saturating_sub` emit these — and so does `std`'s
//! float parser (`dec2flt`, `usub.sat.i8`), which is why translating the real `svm-leng` as an svm
//! guest hit them (W5 §3e). The on-ramp previously supported only i32/i64 saturating; this pins the
//! i8/i16 extension: a Rust guest doing all four ops at both narrow widths runs on svm and matches a
//! native oracle, interp == JIT (§9) == native (§18). Linux + `rustc` gated.

#![cfg(target_os = "linux")]

use svm_interp::Value;

const GUEST_SRC: &str = r#"
#![no_std]
#[panic_handler] fn ph(_:&core::panic::PanicInfo)->!{loop{}}
#[no_mangle] pub extern "C" fn rust_eh_personality(){}
#[no_mangle]
pub extern "C" fn run(n: i64) -> i64 {
    let a = (n & 0xff) as u8;  let b = ((n >> 8) & 0xff) as u8;
    let ia = (n & 0xff) as i8; let ib = ((n >> 8) & 0xff) as i8;
    let ua = (n & 0xffff) as u16; let ub = ((n >> 16) & 0xffff) as u16;
    let mut s = 0i64;
    s = s.wrapping_add(a.saturating_sub(b) as i64);
    s = s.wrapping_add(a.saturating_add(b) as i64);
    s = s.wrapping_add(ia.saturating_sub(ib) as i64);
    s = s.wrapping_add(ia.saturating_add(ib) as i64);
    s = s.wrapping_add(ua.saturating_sub(ub) as i64);
    s = s.wrapping_add(ua.saturating_add(ub) as i64);
    s
}
"#;

/// The native oracle — the identical computation.
fn oracle(n: i64) -> i64 {
    let a = (n & 0xff) as u8;
    let b = ((n >> 8) & 0xff) as u8;
    let ia = (n & 0xff) as i8;
    let ib = ((n >> 8) & 0xff) as i8;
    let ua = (n & 0xffff) as u16;
    let ub = ((n >> 16) & 0xffff) as u16;
    let mut s = 0i64;
    s = s.wrapping_add(a.saturating_sub(b) as i64);
    s = s.wrapping_add(a.saturating_add(b) as i64);
    s = s.wrapping_add(ia.saturating_sub(ib) as i64);
    s = s.wrapping_add(ia.saturating_add(ib) as i64);
    s = s.wrapping_add(ua.saturating_sub(ub) as i64);
    s = s.wrapping_add(ua.saturating_add(ub) as i64);
    s
}

fn rustc_emit_ll(src: &std::path::Path, ll: &std::path::Path) -> bool {
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
fn narrow_saturating_ops_run_on_svm() {
    let dir = std::env::temp_dir().join(format!("narrow_sat_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("g.rs");
    let ll = dir.join("g.ll");
    std::fs::write(&src, GUEST_SRC).unwrap();
    if !rustc_emit_ll(&src, &ll) {
        eprintln!("note: skipping (rustc --emit=llvm-ir unavailable)");
        return;
    }
    // Guard: the guest really does emit the narrow intrinsics this test exists to cover.
    let emitted = std::fs::read_to_string(&ll).unwrap();
    assert!(
        emitted.contains("usub.sat.i8") && emitted.contains("uadd.sat.i16"),
        "guest should emit narrow saturating intrinsics"
    );

    let t = svm_llvm::translate_ll_path(&ll).expect("translate narrow-saturating guest");
    svm_verify::verify_module(&t.module).expect("verify");
    let entry = t.exports.iter().find(|(n, _)| n == "run").unwrap().1;
    let sp = t.entry_sp as i64;

    // n values chosen to drive saturation both ways: borrow/carry, signed over/underflow, no-op.
    for n in [
        0i64,
        0x0102,
        0x0201,
        0x00ff_00ff,
        0x7f80,
        0x8080,
        0xffff_ffff,
        0x1234_5678,
        -1,
    ] {
        let mut fuel = u64::MAX;
        let interp = match svm_interp::run(
            &t.module,
            entry,
            &[Value::I64(sp), Value::I64(n)],
            &mut fuel,
        )
        .expect("interp")
        .as_slice()
        {
            [Value::I64(v)] => *v,
            o => panic!("interp: {o:?}"),
        };
        let jit = match svm_jit::compile_and_run(&t.module, entry, &[sp, n]).expect("jit") {
            svm_jit::JitOutcome::Returned(v) => v[0],
            o => panic!("jit: {o:?}"),
        };
        assert_eq!(interp, jit, "§9 interp/JIT parity for n={n:#x}");
        assert_eq!(interp, oracle(n), "§18 svm == native for n={n:#x}");
    }
}
