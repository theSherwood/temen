//! Enum / distinct / opaque named-type tests (NIM.md Phase 2, W1 Leng totality). A named type is an
//! aggregate only when it is a locally-declared `(object …)`/`(array …)`; every *other* named type —
//! an `(enum …)`, a `distinct` int, a `proctype`, or a type external to the module — is an integer
//! scalar (its values are plain integers, `Red`/`Green`/`Blue` = `0`/`1`/`2`). This is the piece
//! nimony's exception ABI needs (its `ErrorCode` field is an enum). Both engines, hand-written and
//! against genuine nimony `toNum`/`roundtrip` bytes.

use svm_interp::Value;

fn run(module: &svm_ir::Module, idx: u32, args: &[i64]) -> i64 {
    svm_verify::verify_module(module).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let ivals: Vec<Value> = args.iter().map(|&n| Value::I64(n)).collect();
    let mut fuel = u64::MAX;
    let interp = svm_interp::run(module, idx, &ivals, &mut fuel).expect("interp run");
    let interp_n = match interp.as_slice() {
        [Value::I64(n)] => *n,
        other => panic!("expected i64, got {other:?}"),
    };
    let jit = match svm_jit::compile_and_run(module, idx, args).expect("jit compile") {
        svm_jit::JitOutcome::Returned(v) => v,
        other => panic!("jit: {other:?}"),
    };
    assert_eq!(jit.as_slice(), &[interp_n], "§9 interp/JIT parity");
    interp_n
}

#[test]
fn enum_param_is_scalar() {
    // classify(c: Color): int — Color is an enum (scalar); comparing/branching on its integer value.
    // A proc taking a bare-symbol enum type must lower to a plain `(i64)` param, not a by-address
    // aggregate.
    let leng = "\
(stmts
 (type :Color.0. . (enum (u 8) (efld :Red.0. 0) (efld :Green.0. 1) (efld :Blue.0. 2)))
 (proc :classify.0 (params (param :c.0 . Color.0.)) (i +64) .
  (stmts .
   (var :r.0 . (i +64) .)
   (if
    (elif (eq c.0 1) (stmts . (asgn r.0 10)))
    (else (stmts . (asgn r.0 (add (i +64) c.0 100)))))
   (ret r.0))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    let text = svm_leng::translate_to_text(leng).unwrap();
    assert!(
        text.contains("func (i64) -> (i64)"),
        "enum param is a scalar i64:\n{text}"
    );
    assert_eq!(run(&m, 0, &[1]), 10, "Green → 10");
    assert_eq!(run(&m, 0, &[0]), 100, "Red → 0+100");
    assert_eq!(run(&m, 0, &[2]), 102, "Blue → 2+100");
}

#[test]
fn distinct_int_type_is_scalar() {
    // A `distinct` int type (`Id`) is a scalar too — used directly as an integer.
    let leng = "\
(stmts
 (type :Id.0. . (distinct (i +64)))
 (proc :bump.0 (params (param :x.0 . Id.0.)) (i +64) .
  (stmts .
   (ret (add (i +64) x.0 1)))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    assert_eq!(run(&m, 0, &[41]), 42);
}

/// Real nimony `hexer` output for `toNum(c: Color): int` — an enum param compared against its integer
/// values (`Green` = 1, `Blue` = 2) in an if/elif chain.
#[test]
fn real_nimony_enum_compare() {
    const REAL: &str = include_str!("fixtures/real_enum.leng.nif");
    let m = svm_leng::translate_proc(REAL, "toNum.0.")
        .unwrap_or_else(|e| panic!("translate real toNum: {e}"));
    assert_eq!(run(&m, 0, &[1]), 1, "Green");
    assert_eq!(run(&m, 0, &[2]), 2, "Blue");
    assert_eq!(run(&m, 0, &[0]), 0, "Red → else");
}

/// Real nimony `hexer` output for `roundtrip(c: Color): Color` — an enum-in, enum-out passthrough
/// (`var d = c; result = d`). Because `Color` is a scalar, this is a plain `(i64) -> (i64)`, *not*
/// an sret aggregate return.
#[test]
fn real_nimony_enum_passthrough() {
    const REAL: &str = include_str!("fixtures/real_enum.leng.nif");
    let m = svm_leng::translate_proc(REAL, "roundtrip.0.")
        .unwrap_or_else(|e| panic!("translate real roundtrip: {e}"));
    let text = svm_leng::translate_proc_to_text(REAL, "roundtrip.0.").unwrap();
    assert!(
        text.contains("func (i64) -> (i64)"),
        "enum return is scalar, not sret:\n{text}"
    );
    assert_eq!(run(&m, 0, &[2]), 2);
    assert_eq!(run(&m, 0, &[0]), 0);
}
