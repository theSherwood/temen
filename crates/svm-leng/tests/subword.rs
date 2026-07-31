//! Sub-word memory access (NIM.md W2 — ARC totality). nimony reads and writes single bytes through
//! **pointer casts** — e.g. `=destroy` checks a string's low length byte via
//! `(deref (cast (ptr (u 8)) …))`. That needs two things the skeleton gained here: a `deref` of a
//! *computed* pointer (a `(cast (ptr T) e)`, not just a symbol), and `u8`/`u16` loads/stores that
//! touch exactly 1 or 2 bytes (`i32.load8_u`/`store8`, …) instead of a full 4/8-byte access. Both
//! engines, §9 parity, verified against the raw window so a too-wide access would be caught.

use svm_interp::Value;

fn run_capture_both(
    m: &svm_ir::Module,
    idx: u32,
    args: &[i64],
    seed: &[u8],
) -> (Vec<i64>, Vec<u8>) {
    svm_verify::verify_module(m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let ivals: Vec<Value> = args.iter().map(|&n| Value::I64(n)).collect();
    let mut fuel = u64::MAX;
    let (ir, imem) = svm_interp::run_capture(m, idx, &ivals, &mut fuel, seed);
    let ivals_out = ir.expect("interp");
    let (jout, jmem) = svm_jit::compile_and_run_capture(m, idx, args, seed).expect("jit");
    let jvals = match jout {
        svm_jit::JitOutcome::Returned(v) => v,
        o => panic!("jit: {o:?}"),
    };
    let iwords: Vec<i64> = ivals_out
        .iter()
        .map(|v| match v {
            Value::I64(n) => *n,
            Value::I32(n) => *n as i64,
            o => panic!("unexpected {o:?}"),
        })
        .collect();
    assert_eq!(iwords, jvals, "§9 interp/JIT return parity");
    let n = imem.len().min(jmem.len());
    assert_eq!(imem[..n], jmem[..n], "§9 interp/JIT window parity");
    (iwords, imem)
}

#[test]
fn u8_load_reads_one_byte_zero_extended() {
    // `lowbyte(p) = u8 at p, widened to int`. Only the byte at p is read — the neighbours are 0xFF,
    // and the result is 0x000000AB, proving it's a 1-byte zero-extended load, not a wide one.
    let leng = "\
(stmts
 (proc :lowbyte.0. (params (param :p.0 . (i +64))) (i +64) .
  (stmts . (ret (conv (i +64) (deref (cast (ptr (u 8)) p.0)))))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    let mut seed = vec![0xFFu8; 4096];
    seed[256] = 0xAB;
    let (out, _) = run_capture_both(&m, 0, &[256], &seed);
    assert_eq!(out, vec![0xAB], "one byte, zero-extended");
}

#[test]
fn i8_load_sign_extends() {
    // A signed byte load sign-extends: 0xFF at p → -1.
    let leng = "\
(stmts
 (proc :sbyte.0. (params (param :p.0 . (i +64))) (i +64) .
  (stmts . (ret (conv (i +64) (deref (cast (ptr (i 8)) p.0)))))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    let mut seed = vec![0u8; 4096];
    seed[256] = 0xFF;
    let (out, _) = run_capture_both(&m, 0, &[256], &seed);
    assert_eq!(out, vec![-1], "0xFF sign-extends to -1");
}

#[test]
fn u8_store_writes_exactly_one_byte() {
    // `setbyte(p, v)` stores v's low byte at p. Starting from all-0xFF, writing 0x00 must change
    // *only* the byte at p — the next byte stays 0xFF (a wide store would clobber it).
    let leng = "\
(stmts
 (proc :setbyte.0. (params (param :p.0 . (i +64)) (param :v.0 . (i +64))) . .
  (stmts . (asgn (deref (cast (ptr (u 8)) p.0)) v.0))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    let seed = vec![0xFFu8; 4096];
    let (_, mem) = run_capture_both(&m, 0, &[256, 0], &seed);
    assert_eq!(mem[256], 0x00, "the target byte is written");
    assert_eq!(mem[257], 0xFF, "the neighbour byte is untouched");
    assert_eq!(mem[255], 0xFF, "the previous byte is untouched");
}
