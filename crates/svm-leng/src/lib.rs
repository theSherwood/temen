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
//! The frontend now lowers integers/floats, arithmetic, locals and direct/indirect calls, control
//! flow (`if`/`while`/`case`, `break`/`continue`, and the low-level `jmp`/`lab` jump family), memory
//! (`ptr`/`aptr`/`deref`/`addr` + window frames), aggregates (`object`/`array` with constructors,
//! copy, and sret return), enum/distinct/opaque named types (integer scalars), globals, and — falling
//! straight out of those — nimony's **exception ABI** (a `.raises` proc returns an `(ErrorCode,
//! result)` tuple by sret; `try`/`except` is an error-code check plus a `jmp` to a handler label).
//! `seq`/`string` are `{len, data*}` objects: their value layout and element access lower here, and
//! their stdlib operations (`add`/`[]`/`len`/`toOpenArray`) lower to **imports** — valid IR that runs
//! once those imports are bound to a real seq runtime (the W3 runtime edge). Multiple modules **link**
//! into one ([`link_units`], NIM.md W2): nimony's cross-module calls resolve against the defining
//! module's exports. `object of RootObj` **inheritance** lowers too: a derived object inlines its
//! base's layout after a leading vtable header, and the vtable pointer is stored but opaque (dynamic
//! dispatch fail-closes). What remains
//! outside the subset — `union`, `emit`, dynamic method dispatch, value-object exception payloads
//! (an object punned into the error tuple's scalar slot) — is a fail-closed [`LengError::Unsupported`],
//! never a silent mistranslation (the `svm-wasm`/`svm-llvm` `unsup(...)` discipline). (The
//! `jtrue`/`mflag`/`vflag` cfvar jump forms never reach us — hexer's `xelim` lowers them away before
//! the final IR.) Growing the frontend means adding grammar arms below, not rearchitecting.
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

/// One nimony module in a multi-module link (NIM.md W2 — the linker). `src` is the module's `hexer`
/// Leng; `stem` is the file id that qualifies its symbols globally — a proc `P.` defined here is
/// referenced from *other* modules as `P.<stem>` (nimony's cross-module mangling); `names` are the
/// local proc names to translate out of it.
pub struct LengModule<'a> {
    pub stem: &'a str,
    pub src: &'a str,
    pub names: &'a [&'a str],
}

/// **Link several nimony modules into one svm-ir [`Module`]** (NIM.md W2). Each module's `names` are
/// translated and exported under their **global** (stem-suffixed) names; every unit's cross-module
/// calls — which nimony emits as named imports `callee.<defining-stem>` — resolve against those
/// exports via [`svm_ir::link`]. Units keep the given order, so the first module's first proc is
/// func 0 (a natural entry). The linked module is **not** verified here (untrusted frontend — the
/// caller runs `svm_verify::verify_module`, which re-checks the whole linked result).
///
/// This is the same mechanism the end-to-end runtime shim uses, generalized: a real dependency
/// module takes the place the hand-written shim held, so `helper` in module B resolves a `call`
/// from module A to *compiled Nim*, not a stand-in.
pub fn link_units(units: &[LengModule]) -> Result<Module, LengError> {
    let mut link_units = Vec::with_capacity(units.len());
    for u in units {
        // Translate in **link-unit mode** — globals via `data.self`, so `link` relocates each unit's
        // data into one shared window (an absolute-offset unit would silently alias).
        let root = nif::parse(u.src).map_err(LengError::Parse)?;
        let text = translate::Translator::new_for_link().some_procs(&root, u.names)?;
        let module = svm_text::parse_module(&text).map_err(|e| {
            LengError::Malformed(format!(
                "emitted IR failed to parse: {e:?}\n--- IR ---\n{text}"
            ))
        })?;
        let exports = u
            .names
            .iter()
            .enumerate()
            .map(|(i, local)| (format!("{local}{}", u.stem), i as u32))
            .collect();
        link_units.push(svm_ir::LinkUnit {
            module,
            exports,
            ..Default::default()
        });
    }
    svm_ir::link(&link_units).map_err(|e| LengError::Malformed(format!("link failed: {e:?}")))
}

/// A translated SVM value: its SSA id and type. The unit the expression translator threads.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Val {
    pub id: u32,
    pub ty: ValType,
}
