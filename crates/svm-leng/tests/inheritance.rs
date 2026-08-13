//! Object inheritance tests (NIM.md Phase 2, W1 Leng totality — the last W1 translation gap). An
//! `object of RootObj` is *inheritable*: it carries a leading vtable/type-header pointer (the
//! positional slot an `(oconstr T <vtable> …)` fills), then the base's fields, then its own — so a
//! derived object inlines its base's layout at the front. This is nimony's general OOP object model.
//!
//! The vtable pointer itself is stored but **opaque**: only *dynamic dispatch* would read through it,
//! and that fail-closes, so a `Type.vt` const gets a zeroed placeholder address. (Value-object
//! *exception payloads* — an object stored into the error tuple's scalar `ErrorCode` slot — are a
//! murky nimony-internal pun and stay fail-closed; the clean `distinct`-int raiser is in
//! `exceptions.rs`.) Both engines.

use svm_interp::Value;

/// Run a void (sret/frame) proc with a seeded window; return the final bytes (both engines agree).
fn run_void_capture(m: &svm_ir::Module, idx: u32, args: &[i64], win: usize) -> Vec<u8> {
    svm_verify::verify_module(m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let seed = vec![0u8; win];
    let ivals: Vec<Value> = args.iter().map(|&n| Value::I64(n)).collect();
    let mut fuel = u64::MAX;
    let (ir, imem) = svm_interp::run_capture(m, idx, &ivals, &mut fuel, &seed);
    ir.expect("interp");
    let (_j, jmem) = svm_jit::compile_and_run_capture(m, idx, args, &seed).expect("jit");
    let n = imem.len().min(jmem.len());
    assert_eq!(imem[..n], jmem[..n], "§9 interp/JIT window parity");
    imem
}

fn read_i64(mem: &[u8], off: usize) -> i64 {
    i64::from_le_bytes(mem[off..off + 8].try_into().unwrap())
}

#[test]
fn derived_layout_inlines_base_after_vtable() {
    // Derived (of Base of RootObj) constructs {vtable@0, value@8, extra@16}; mk builds one into its
    // sret pointer and read-back confirms the offsets (base field before derived field, both after
    // the vtable header).
    let leng = "\
(stmts
 (type :Base.0. . (object RootObj.0.ext (fld :value.0 . (i +64))))
 (type :Derived.0. . (object Base.0. (fld :extra.0 . (i +64))))
 (proc :mk.0 (params (param :a.0 . (i +64)) (param :b.0 . (i +64))) Derived.0. .
  (stmts .
   (ret (oconstr Derived.0. (kv value.0 a.0) (kv extra.0 b.0))))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    let dst = 512usize;
    let mem = run_void_capture(&m, 0, &[dst as i64, 5, 7], 4096);
    assert_eq!(read_i64(&mem, dst), 0, "vtable slot (unset) at 0");
    assert_eq!(read_i64(&mem, dst + 8), 5, "base field `value` at 8");
    assert_eq!(read_i64(&mem, dst + 16), 7, "derived field `extra` at 16");
}

#[test]
fn rtti_vtable_field_reads_by_name() {
    // nimony accesses the `RootObj` RTTI type-header by name — `(dot (deref x) vt.00 <level>)` — in
    // `=dup`/`=destroy`/`of`-checks/method dispatch. The synthesized inheritable-root header slot is
    // named `vt.00` (not an anonymous placeholder), so those accesses resolve (#859). Read the header
    // pointer of a Derived through a pointer: it lives at offset 0, ahead of the inlined base fields.
    let leng = "\
(stmts
 (type :Base.0. . (object RootObj.0.ext (fld :value.0 . (i +64))))
 (type :Derived.0. . (object Base.0. (fld :extra.0 . (i +64))))
 (proc :getVt.0 (params (param :d.0 . (ptr Derived.0.))) (i +64) .
  (stmts .
   (ret (dot (deref d.0) vt.00 2)))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    let obj = 256usize;
    let mut seed = vec![0u8; 4096];
    seed[obj..obj + 8].copy_from_slice(&0xABCDi64.to_le_bytes()); // vt header at offset 0
    seed[obj + 8..obj + 16].copy_from_slice(&42i64.to_le_bytes());
    svm_verify::verify_module(&m).unwrap();
    let mut fuel = u64::MAX;
    let (ir, _) = svm_interp::run_capture(&m, 0, &[Value::I64(obj as i64)], &mut fuel, &seed);
    assert_eq!(ir.expect("interp").as_slice(), &[Value::I64(0xABCD)]);
    let (jout, _) = svm_jit::compile_and_run_capture(&m, 0, &[obj as i64], &seed).expect("jit");
    match jout {
        svm_jit::JitOutcome::Returned(v) => assert_eq!(v, vec![0xABCD], "§9 parity"),
        o => panic!("jit: {o:?}"),
    }
}

#[test]
fn base_field_read_through_pointer() {
    // getVal(d: ptr Derived): int = (deref d).value — reads the *base* field at offset 8 through a
    // pointer to a derived object. The caller builds the object; runs with no runtime.
    let leng = "\
(stmts
 (type :Base.0. . (object RootObj.0.ext (fld :value.0 . (i +64))))
 (type :Derived.0. . (object Base.0. (fld :extra.0 . (i +64))))
 (proc :getVal.0 (params (param :d.0 . (ptr Derived.0.))) (i +64) .
  (stmts .
   (ret (dot (deref d.0) value.0 0)))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    // Build a Derived {vtable:0, value:42, extra:99} at offset 256 and read its base `value`.
    let obj = 256usize;
    let mut seed = vec![0u8; 4096];
    seed[obj + 8..obj + 16].copy_from_slice(&42i64.to_le_bytes());
    seed[obj + 16..obj + 24].copy_from_slice(&99i64.to_le_bytes());
    svm_verify::verify_module(&m).unwrap();
    let mut fuel = u64::MAX;
    let (ir, _) = svm_interp::run_capture(&m, 0, &[Value::I64(obj as i64)], &mut fuel, &seed);
    assert_eq!(ir.expect("interp").as_slice(), &[Value::I64(42)]);
    let (jout, _) = svm_jit::compile_and_run_capture(&m, 0, &[obj as i64], &seed).expect("jit");
    match jout {
        svm_jit::JitOutcome::Returned(v) => assert_eq!(v, vec![42], "§9 parity"),
        o => panic!("jit: {o:?}"),
    }
}

/// Real nimony `hexer` output for `kindOf(e: BaseError): int = e.value` and `mkDerived(a,b):
/// DerivedError` — genuine `object of RootObj`/`object of BaseError` inheritance with `Rtti` vtables.
/// `kindOf` reads the base field through a pointer and **runs** (no runtime needed); `mkDerived`
/// constructs the inherited object (with its vtable) and **translates + verifies** — running it needs
/// the ARC destructor imports bound (the W3 runtime edge), like seq.
#[test]
fn real_nimony_inheritance() {
    const REAL: &str = include_str!("fixtures/real_inherit.leng.nif");
    // kindOf: `(ptr BaseError) -> i64`, reads `e.value` (offset 8, past the vtable). Runs.
    let m = svm_leng::translate_proc(REAL, "kindOf.0.")
        .unwrap_or_else(|e| panic!("translate real kindOf: {e}"));
    let base = 256usize;
    let mut seed = vec![0u8; 4096];
    seed[base + 8..base + 16].copy_from_slice(&123i64.to_le_bytes());
    svm_verify::verify_module(&m).unwrap();
    let mut fuel = u64::MAX;
    let (ir, _) = svm_interp::run_capture(&m, 0, &[Value::I64(base as i64)], &mut fuel, &seed);
    assert_eq!(
        ir.expect("interp").as_slice(),
        &[Value::I64(123)],
        "e.value"
    );
    let (jout, _) = svm_jit::compile_and_run_capture(&m, 0, &[base as i64], &seed).expect("jit");
    assert!(matches!(jout, svm_jit::JitOutcome::Returned(ref v) if v == &[123]));

    // mkDerived constructs an inherited object (vtable + base + derived fields) — translate + verify.
    let m2 = svm_leng::translate_proc(REAL, "mkDerived.0.")
        .unwrap_or_else(|e| panic!("translate real mkDerived: {e}"));
    svm_verify::verify_module(&m2).unwrap_or_else(|e| panic!("verify mkDerived: {e:?}"));
}
