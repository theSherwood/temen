//! nim's `{.emit: "…".}` — raw C spliced into the generated source (#1443).
//!
//! There is no C front-end on this path, so the general case must fail closed. But one emit body —
//! `std/atomics`' `cpuRelax()` spin-wait hint — was blocking six stdlib modules at link, and a
//! `PAUSE`/`YIELD` is a hint with no architectural effect, so it lowers to nothing. This pins both
//! halves: the hint is accepted and emits no code, and anything else is still refused, *naming the
//! body* so the next one is triageable.

use temen_leng::LengError;

/// A proc whose body is one `emit` of `body`, plus a `ret` so the shape is a real function.
fn proc_with_emit(body: &str) -> String {
    format!(
        "(stmts (proc :f.0. (params) (i 64) (pragmas) (stmts (emit {body}) (ret (suf 7 \"i64\")))))"
    )
}

#[test]
fn cpu_relax_hint_lowers_to_nothing() {
    // `asm volatile("pause");` — NIF-escapes `(`, `"`, `)` as `\28`, `\22`, `\29`.
    for body in [
        r#""asm volatile\28\22pause\22\29;""#,
        r#""asm volatile\28\22yield\22\29;""#,
    ] {
        let text = temen_leng::translate_to_text(&proc_with_emit(body))
            .unwrap_or_else(|e| panic!("the spin hint must lower: {e:?}"));
        // It contributes no instructions: the only thing in the body is the return.
        let ops = text
            .lines()
            .filter(|l| l.trim_start().starts_with('v') || l.trim_start().starts_with("return"))
            .count();
        assert_eq!(
            ops, 2,
            "expected only `v0 = i64.const 7` + `return`, got:\n{text}"
        );
    }
}

#[test]
fn whitespace_variation_still_matches() {
    // The match is whitespace-normalized, so hexer reflowing the body doesn't reopen the gap.
    let body = r#""asm   volatile\28\22pause\22\29;""#;
    temen_leng::translate_to_text(&proc_with_emit(body)).expect("normalized match");
}

#[test]
fn other_emits_fail_closed_and_name_the_body() {
    // Anything with real semantics is still refused — silently dropping arbitrary inline asm would
    // be a correctness hole, and the *next* emit to appear might compute something.
    let body = r#""x = __atomic_fetch_add\28p, 1, 5\29;""#;
    let err = temen_leng::translate_to_text(&proc_with_emit(body))
        .expect_err("an emit with semantics must fail closed");
    let LengError::Unsupported(msg) = err else {
        panic!("expected Unsupported, got {err:?}");
    };
    assert!(
        msg.contains("__atomic_fetch_add"),
        "the message must quote the body it refused — the old bare `statement `emit`` said nothing \
         about which emit. got: {msg}"
    );
}
