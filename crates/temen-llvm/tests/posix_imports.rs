//! #1609 — a declaration-only `__px_<op>` function is the POSIX personality's import, typed from its
//! own vocabulary (`temen_posix_abi`): the import nim programs and chibicc guests already bind (#1668),
//! so an on-ramp guest — `temen-link`, the self-hosted lane's in-guest linker — reaches the
//! personality the way they do, and binds in an `execve`'d powerbox like them.

use temen_ir::{FuncType, ImportShape, TypeEntry, ValType::I64};

/// `write(1, "hi\n", 3)` through the personality, then `main` returns 0.
const WRITES: &str = r#"
@s = private constant [3 x i8] c"hi\0A"
declare i64 @__px_write(i64, i64, i64)
define i32 @main() {
  %r = call i64 @__px_write(i64 1, i64 ptrtoint (ptr @s to i64), i64 3)
  ret i32 0
}
"#;

#[test]
fn a_px_declaration_is_the_personality_import_with_its_vocabulary_signature() {
    let t = temen_llvm::translate_ll_str(WRITES).expect("translate");
    temen_verify::verify_module(&t.module).expect("verify");
    let imp = t
        .module
        .imports
        .iter()
        .find(|i| i.name == "__px_write")
        .expect("a `__px_write` import");
    let ImportShape::Func(ty) = imp.shape else {
        panic!("a function import")
    };
    assert_eq!(
        t.module.types[ty as usize],
        TypeEntry::Func(FuncType {
            params: vec![I64; 3],
            results: vec![I64],
        }),
        "typed from the vocabulary: `write(fd, buf, len) -> i64`"
    );
    assert!(
        t.module.exports.iter().any(|e| e.name == "_start"),
        "an import makes the module a powerbox entry"
    );
}

/// A declaration that disagrees with the op's arity is the guest's mistake: refused at translation,
/// never bound wrong.
#[test]
fn a_px_call_with_the_wrong_arity_is_refused() {
    let bad = WRITES
        .replace(
            "declare i64 @__px_write(i64, i64, i64)",
            "declare i64 @__px_write(i64, i64)",
        )
        .replace(", i64 3)", ")");
    let err = temen_llvm::translate_ll_str(&bad)
        .map(|_| ())
        .expect_err("the wrong arity must be refused");
    assert!(
        format!("{err:?}").contains("POSIX vocabulary"),
        "names the vocabulary: {err:?}"
    );
}
