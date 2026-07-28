//! `svm-leng` — a **Leng-NIF → SVM-IR** translator (NIM.md Phase 2).
//!
//! Leng is nimony's C-like midlevel IR (`doc/leng-spec.md`): the seam its C/C++/LLVM/arkham
//! backends already consume, sitting just after hexer's lowering (ARC, exceptions, monomorph).
//! This is the **fourth SVM frontend**, alongside the chibicc C fork, `svm-wasm`, and `svm-llvm`.
//! Like them it is an untrusted producer (DESIGN.md §2a): the verifier re-checks every module it
//! emits, so a bug here is a clean error, never an escape.
//!
//! ## Scope — a walking skeleton
//!
//! This first slice handles the **integer / arithmetic / local / direct-call** subset with
//! straight-line bodies and `ret`. Everything outside it — control flow (`if`/`while`/`case`/
//! `jmp`), aggregates (`object`/`array`/`union`), pointers (`ptr`/`aptr`/`deref`/`addr`), floats,
//! `onerr`/`try`, `emit` — is a fail-closed [`LengError::Unsupported`], never a silent
//! mistranslation (the `svm-wasm`/`svm-llvm` `unsup(...)` discipline). Growing the frontend means
//! adding grammar arms below, not rearchitecting.
//!
//! Like chibicc's `codegen_ir.c`, it emits **SVM text** and hands it to [`svm_text::parse_module`];
//! [`translate`] returns the parsed (but not-yet-verified) [`Module`].
//!
//! ```
//! let leng = "(stmts (proc :main.0 . (i +64) . (stmts . (ret (add (i +64) 3 (mul (i +64) 4 2))))))";
//! let m = svm_leng::translate(leng).unwrap();      // one func: () -> (i64) returning 11
//! assert_eq!(m.funcs.len(), 1);
//! ```

use svm_ir::{Module, ValType};

mod nif;
mod translate;

pub use nif::{parse as parse_nif, Node};

/// A translation failure. `Unsupported` is the fail-closed catch-all for any Leng construct the
/// skeleton does not yet lower — the frontend never emits IR it cannot stand behind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LengError {
    /// The NIF text did not parse (unbalanced parens, unterminated string, …).
    Parse(String),
    /// A well-formed Leng construct the skeleton does not lower yet. Extend the translator.
    Unsupported(String),
    /// The Leng module violates the grammar the translator expects (e.g. a `proc` without a body
    /// where one is required, an arithmetic node with the wrong child count).
    Malformed(String),
}

impl std::fmt::Display for LengError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LengError::Parse(m) => write!(f, "leng parse error: {m}"),
            LengError::Unsupported(m) => write!(f, "leng: unsupported construct: {m}"),
            LengError::Malformed(m) => write!(f, "leng: malformed: {m}"),
        }
    }
}
impl std::error::Error for LengError {}

/// Translate a Leng-NIF module to **SVM text**. The seam a caller can inspect/debug (the emitted
/// IR is human-readable and rides `svm_text::parse_module`).
pub fn translate_to_text(src: &str) -> Result<String, LengError> {
    let root = nif::parse(src).map_err(LengError::Parse)?;
    translate::Translator::new().module(&root)
}

/// Translate a Leng-NIF module to an SVM-IR [`Module`]. The module is **not** verified here —
/// callers run `svm_verify::verify_module` (the frontend is untrusted; DESIGN.md §2a).
pub fn translate(src: &str) -> Result<Module, LengError> {
    let text = translate_to_text(src)?;
    svm_text::parse_module(&text).map_err(|e| {
        LengError::Malformed(format!(
            "emitted IR failed to parse: {e:?}\n--- IR ---\n{text}"
        ))
    })
}

/// Translate a **single named proc** out of a full Leng module to SVM text — the "go deep" entry
/// for real nimony output, where the enclosing module still carries constructs the skeleton does
/// not lower (`gvar`/`type`/`if`/pointers). The named proc becomes func 0; any call it makes to a
/// proc *other than itself* fail-closes (only the target is emitted). `name` is the mangled Leng
/// symbol, e.g. `addTwo.0.`.
pub fn translate_proc_to_text(src: &str, name: &str) -> Result<String, LengError> {
    let root = nif::parse(src).map_err(LengError::Parse)?;
    translate::Translator::new().one_proc(&root, name)
}

/// As [`translate_proc_to_text`], returning the parsed (unverified) [`Module`].
pub fn translate_proc(src: &str, name: &str) -> Result<Module, LengError> {
    let text = translate_proc_to_text(src, name)?;
    svm_text::parse_module(&text).map_err(|e| {
        LengError::Malformed(format!(
            "emitted IR failed to parse: {e:?}\n--- IR ---\n{text}"
        ))
    })
}

/// Translate a **named subset** of a module's procs together — the multi-proc generalization of
/// [`translate_proc`], so a real caller→callee pair (e.g. an sret `mk` and its `mkSum` caller) lifts
/// out of a module whose other top-levels the skeleton can't lower yet. Procs are func-indexed in
/// `names` order. Returns the parsed (unverified) [`Module`].
pub fn translate_procs(src: &str, names: &[&str]) -> Result<Module, LengError> {
    let root = nif::parse(src).map_err(LengError::Parse)?;
    let text = translate::Translator::new().some_procs(&root, names)?;
    svm_text::parse_module(&text).map_err(|e| {
        LengError::Malformed(format!(
            "emitted IR failed to parse: {e:?}\n--- IR ---\n{text}"
        ))
    })
}

/// A translated SVM value: its SSA id and type. The unit the expression translator threads.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Val {
    pub id: u32,
    pub ty: ValType,
}
