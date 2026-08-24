//! "Go deep" test (NIM.md Phase 2): consume a **real** Leng file emitted by nimony's own `hexer`
//! (`hexer c --isMain <module>.s.nif`) and translate a proc out of it — real NIF bytes, real
//! line-info, real mangled symbols — not a hand-written fixture.
//!
//! The fixture `fixtures/real_module.leng.nif` is verbatim hexer output for:
//! ```nim
//! proc addTwo(a: int; b: int): int = result = a + b
//! proc main() = (let x = addTwo(20, 22); discard x)
//! main()
//! ```
//! The module also contains `gvar`/`type`/`main`/`ini` constructs outside the current subset, so we
//! target `addTwo.0.` specifically via `translate_proc` and run it on both engines.

use temen_interp::Value;

const REAL: &str = include_str!("fixtures/real_module.leng.nif");

#[test]
fn translates_real_hexer_proc_addtwo() {
    // Sanity: the fixture really is raw hexer output (directives + line-info the reader must strip).
    assert!(REAL.contains("(.nif27)"), "fixture should be raw NIF");
    assert!(
        REAL.contains("addTwo.0.@"),
        "fixture should carry line-info on symbols"
    );

    let module = temen_leng::translate_proc(REAL, "addTwo.0.")
        .unwrap_or_else(|e| panic!("translate real addTwo: {e}"));
    temen_verify::verify_module(&module).unwrap_or_else(|e| panic!("verify: {e:?}"));

    // addTwo(a, b) = a + b, as func 0 with two i64 params → one i64 result. Run on both engines.
    for (a, b) in [(20i64, 22i64), (-5, 8), (1_000_000, 337)] {
        let args = [Value::I64(a), Value::I64(b)];
        let mut fuel = u64::MAX;
        let interp = temen_interp::run(&module, 0, &args, &mut fuel).expect("interp run");
        let interp_n = match interp.as_slice() {
            [Value::I64(n)] => *n,
            other => panic!("expected i64, got {other:?}"),
        };
        let jit = match temen_jit::compile_and_run(&module, 0, &[a, b]).expect("jit compile") {
            temen_jit::JitOutcome::Returned(v) => v,
            other => panic!("jit: {other:?}"),
        };
        assert_eq!(interp_n, a + b, "addTwo({a},{b})");
        assert_eq!(jit.as_slice(), &[interp_n], "§9 interp/JIT parity");
    }
}

#[test]
fn missing_proc_is_reported() {
    match temen_leng::translate_proc(REAL, "nope.0.") {
        Err(temen_leng::LengError::Malformed(_)) => {}
        other => panic!("expected Malformed for missing proc, got {other:?}"),
    }
}
