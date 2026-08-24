//! Aggregate-return (sret) tests (NIM.md Phase 2, W1 Leng totality). A proc whose return type is a
//! named aggregate returns `void` and takes a hidden `$sret` pointer param (after `$sp`, before the
//! Leng params); `(ret aggval)` constructs/copies the result into that pointer. A caller assigning
//! the call to an aggregate destination hands the destination's address down as `$sret` — no
//! temporary. Validated on both engines (§9 parity), including the memory the callee wrote, plus
//! genuine nimony `mk`/`mkSum` bytes.

use temen_interp::Value;

/// Run func `idx` with i64 `args` on interp + JIT; assert both agree on a single i64 result.
fn run(module: &temen_ir::Module, idx: u32, args: &[i64]) -> i64 {
    temen_verify::verify_module(module).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let ivals: Vec<Value> = args.iter().map(|&n| Value::I64(n)).collect();
    let mut fuel = u64::MAX;
    let interp = temen_interp::run(module, idx, &ivals, &mut fuel).expect("interp run");
    let interp_n = match interp.as_slice() {
        [Value::I64(n)] => *n,
        other => panic!("expected i64, got {other:?}"),
    };
    let jit = match temen_jit::compile_and_run(module, idx, args).expect("jit compile") {
        temen_jit::JitOutcome::Returned(v) => v,
        other => panic!("jit: {other:?}"),
    };
    assert_eq!(jit.as_slice(), &[interp_n], "§9 interp/JIT parity");
    interp_n
}

/// Run a **void** (sret) func on both engines and return the final window bytes each wrote, asserting
/// the two snapshots agree (§9 parity extends to the memory the sret proc stored). The window is
/// seeded with `win` zero bytes so the snapshot captures that whole prefix (the snapshot spans the
/// seeded length).
fn run_void_capture(module: &temen_ir::Module, idx: u32, args: &[i64], win: usize) -> Vec<u8> {
    temen_verify::verify_module(module).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let seed = vec![0u8; win];
    let ivals: Vec<Value> = args.iter().map(|&n| Value::I64(n)).collect();
    let mut fuel = u64::MAX;
    let (ir, imem) = temen_interp::run_capture(module, idx, &ivals, &mut fuel, &seed);
    ir.expect("interp run");
    let (jout, jmem) =
        temen_jit::compile_and_run_capture(module, idx, args, &seed).expect("jit compile");
    assert!(
        matches!(jout, temen_jit::JitOutcome::Returned(ref v) if v.is_empty()),
        "sret proc returns void, got {jout:?}"
    );
    let n = imem.len().min(jmem.len());
    assert_eq!(imem[..n], jmem[..n], "§9 interp/JIT window parity");
    imem
}

/// Read a little-endian i64 at byte offset `off` in a window snapshot.
fn read_i64(mem: &[u8], off: usize) -> i64 {
    i64::from_le_bytes(mem[off..off + 8].try_into().unwrap())
}

#[test]
fn sret_proc_writes_into_destination() {
    // mk(a,b): Pt = Pt(x:a, y:b)  — an aggregate return. The proc is void with a leading $sret ptr;
    // `(ret (oconstr …))` constructs straight into it. Call with a window offset and read x/y back.
    let leng = "\
(stmts
 (type :Pt.0. . (object . (fld :x.0 . (i +64)) (fld :y.0 . (i +64))))
 (proc :mk.0 (params (param :a.0 . (i +64)) (param :b.0 . (i +64))) Pt.0. .
  (stmts .
   (ret (oconstr Pt.0. (kv x.0 a.0) (kv y.0 b.0))))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    let text = temen_leng::translate_to_text(leng).unwrap();
    // Signature is `(i64 sret, i64 a, i64 b) -> ()` — void, one leading pointer.
    assert!(
        text.contains("func (i64, i64, i64) -> ()"),
        "sret signature:\n{text}"
    );
    let dst = 4096usize;
    let mem = run_void_capture(&m, 0, &[dst as i64, 7, 9], 8192);
    assert_eq!(read_i64(&mem, dst), 7, "x");
    assert_eq!(read_i64(&mem, dst + 8), 9, "y");
}

#[test]
fn caller_threads_destination_as_sret() {
    // mk(a,b): Pt = Pt(x:a,y:b);  mkSum(a,b): int = (var p = mk(a,b); p.x + p.y).
    // mkSum's aggregate `var p` is frame-resident; its address is handed to mk as the $sret pointer,
    // so mk writes p in place — then mkSum reads the fields and returns an int. Full sret round-trip.
    let leng = "\
(stmts
 (type :Pt.0. . (object . (fld :x.0 . (i +64)) (fld :y.0 . (i +64))))
 (proc :mk.0 (params (param :a.0 . (i +64)) (param :b.0 . (i +64))) Pt.0. .
  (stmts .
   (ret (oconstr Pt.0. (kv x.0 a.0) (kv y.0 b.0)))))
 (proc :mkSum.0 (params (param :a.0 . (i +64)) (param :b.0 . (i +64))) (i +64) .
  (stmts .
   (var :p.0 . Pt.0. (call mk.0 a.0 b.0))
   (ret (add (i +64) (dot p.0 x.0 0) (dot p.0 y.0 0))))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    let sp = 8192;
    assert_eq!(run(&m, 1, &[sp, 3, 4]), 7);
    assert_eq!(run(&m, 1, &[sp, 100, -25]), 75);
}

#[test]
fn whole_aggregate_return_by_copy() {
    // copyPt(p: Pt): Pt = p  — the returned aggregate is a whole-copy of the by-address param into
    // the sret pointer (`mem.copy`). sumUse(a,b) builds a Pt, copies it via copyPt, sums the copy.
    let leng = "\
(stmts
 (type :Pt.0. . (object . (fld :x.0 . (i +64)) (fld :y.0 . (i +64))))
 (proc :copyPt.0 (params (param :p.0 . Pt.0.)) Pt.0. .
  (stmts .
   (ret p.0)))
 (proc :sumUse.0 (params (param :a.0 . (i +64)) (param :b.0 . (i +64))) (i +64) .
  (stmts .
   (var :src.0 . Pt.0. (oconstr Pt.0. (kv x.0 a.0) (kv y.0 b.0)))
   (var :dst.0 . Pt.0. (call copyPt.0 src.0))
   (ret (add (i +64) (dot dst.0 x.0 0) (dot dst.0 y.0 0))))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    let text = temen_leng::translate_to_text(leng).unwrap();
    assert!(text.contains("mem.copy"), "return-by-copy:\n{text}");
    let sp = 8192;
    assert_eq!(run(&m, 1, &[sp, 5, 6]), 11);
    assert_eq!(run(&m, 1, &[sp, -3, 30]), 27);
}

#[test]
fn scalar_call_of_sret_proc_is_fail_closed() {
    // Using an aggregate-returning call as a scalar value (not assigned to an aggregate) fail-closes
    // rather than emitting a call with the wrong arity.
    let leng = "\
(stmts
 (type :Pt.0. . (object . (fld :x.0 . (i +64)) (fld :y.0 . (i +64))))
 (proc :mk.0 (params (param :a.0 . (i +64)) (param :b.0 . (i +64))) Pt.0. .
  (stmts .
   (ret (oconstr Pt.0. (kv x.0 a.0) (kv y.0 b.0)))))
 (proc :bad.0 (params (param :a.0 . (i +64)) (param :b.0 . (i +64))) (i +64) .
  (stmts .
   (ret (add (i +64) (call mk.0 a.0 b.0) 1)))))";
    match temen_leng::translate(leng) {
        Err(temen_leng::LengError::Unsupported(_)) | Err(temen_leng::LengError::Malformed(_)) => {}
        other => panic!("expected fail-closed, got {other:?}"),
    }
}

/// Real nimony `hexer` output for `mk(a,b): Pt = Pt(x:a,y:b)` and its `mkSum` caller — genuine sret
/// bytes: `(proc :mk.0. … Pt.0. …)` returning an aggregate, and `(var :p.0 . Pt.0. (call mk.0. …))`.
#[test]
fn real_nimony_sret_direct() {
    const REAL: &str = include_str!("fixtures/real_sret.leng.nif");
    // Real `mk` emits `var result: Pt; result = Pt(x,y); ret result` — the `result` aggregate is
    // frame-resident, so mk needs *both* a frame and the sret pointer: `(i64 $sp, i64 $sret, i64 a,
    // i64 b) -> ()`. It builds `result` in the frame, then copies it into the sret pointer. Give it a
    // frame at 2048 and a destination at 4096, then read the copied Pt back.
    let m = temen_leng::translate_proc(REAL, "mk.0.")
        .unwrap_or_else(|e| panic!("translate real mk: {e}"));
    let frame = 2048usize;
    let dst = 4096usize;
    let mem = run_void_capture(&m, 0, &[frame as i64, dst as i64, 11, 22], 8192);
    assert_eq!(read_i64(&mem, dst), 11, "x");
    assert_eq!(read_i64(&mem, dst + 8), 22, "y");
}

#[test]
fn real_nimony_sret_caller() {
    const REAL: &str = include_str!("fixtures/real_sret.leng.nif");
    // `mkSum` calls the sret `mk` and returns `p.x + p.y` — the real caller→callee sret path, both
    // procs lifted out of the module together (func 0 = mk, func 1 = mkSum).
    let m = temen_leng::translate_procs(REAL, &["mk.0.", "mkSum.0."])
        .unwrap_or_else(|e| panic!("translate real mk/mkSum: {e}"));
    let sp = 8192;
    assert_eq!(run(&m, 1, &[sp, 3, 4]), 7);
    assert_eq!(run(&m, 1, &[sp, 100, 200]), 300);
}
