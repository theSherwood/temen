//! Linking against the **real `system` module** (NIM.md W2, Path A). The stand-in system objects in
//! `whole.rs`/`strings.rs` are no-op stubs; here a user program binds a `sysvq0asl` edge to a proc
//! **translated from the actual nimony `system` module** — real compiled stdlib code running on SVM.
//!
//! The keystone is `=wasMoved`, string's ARC "moved-from" reset: `s.bytes = 0` through a `ptr string`
//! (verbatim from `system/stringimpl.nim`, exportc `nimStrWasMoved`). A driver calls it across the
//! module boundary; the linker binds the import to the real proc. Both engines, §9 parity.
//!
//! Most of the ARC set (`=destroy`/`=dup`/`=copy`) and the system `ini` chain need translator work
//! still ahead — sub-word (`u8`) loads, `deref` of a computed address, parameter spill, exception
//! globals — so the wider `system` link stays future work; this pins the first real binding.

use svm_interp::Value;
use svm_leng::LengModule;

#[test]
fn real_system_wasmoved_runs() {
    const SYSTEM: &str = include_str!("fixtures/real_system_arc.leng.nif");
    // A driver whose `clr(p)` calls the real `=wasMoved` across the module boundary — hexer would
    // emit exactly this `=wasMoved.2.sysvq0asl` reference. The linker binds it to the system proc.
    let driver = "\
(stmts
 (proc :clr.0. (params (param :p.0 . (i +64))) . .
  (stmts . (call =wasMoved.2.sysvq0asl p.0))))";
    let linked = svm_leng::link_units(&[
        LengModule {
            stem: "drv",
            src: driver,
            names: &["clr.0."],
        },
        LengModule {
            stem: "sysvq0asl",
            src: SYSTEM,
            names: &["=wasMoved.2."],
        },
    ])
    .unwrap_or_else(|e| panic!("link against real system: {e}"));
    svm_verify::verify_module(&linked).unwrap_or_else(|e| panic!("verify: {e:?}"));

    // Seed a `string {bytes: 0xdeadbeef, more: 0}` at offset 256; `clr` must zero its `bytes` word.
    let s = 256usize;
    let mut seed = vec![0u8; 4096];
    seed[s..s + 8].copy_from_slice(&0xdeadbeefu64.to_le_bytes());
    seed[s + 8..s + 16].copy_from_slice(&7u64.to_le_bytes()); // `more` — must be left untouched

    // clr (func 0): (p) -> ().
    let ivals = [Value::I64(s as i64)];
    let mut fuel = u64::MAX;
    let (ir, imem) = svm_interp::run_capture(&linked, 0, &ivals, &mut fuel, &seed);
    ir.expect("interp clr");
    let (_j, jmem) = svm_jit::compile_and_run_capture(&linked, 0, &[s as i64], &seed).expect("jit");
    let n = imem.len().min(jmem.len());
    assert_eq!(imem[..n], jmem[..n], "§9 interp/JIT window parity");

    let bytes = u64::from_le_bytes(imem[s..s + 8].try_into().unwrap());
    let more = u64::from_le_bytes(imem[s + 8..s + 16].try_into().unwrap());
    assert_eq!(bytes, 0, "real =wasMoved zeroed string.bytes");
    assert_eq!(more, 7, "=wasMoved touches only `bytes`, not `more`");
}
