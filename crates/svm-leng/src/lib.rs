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
//! once those imports are bound to a real seq runtime (the W3 runtime edge). A module compiles to a
//! binary **`.svmo` link object** ([`compile_object`], exports in-band) that composes with other
//! producers' objects through the shared `svm_ir::link` — [`link_units`] does exactly that across
//! several nimony modules (NIM.md W2). `object of RootObj` **inheritance** lowers too: a derived object inlines its
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

use svm_ir::{Export, Module, ValType};

mod dethash;
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

/// Translate a Leng-NIF module to SVM text with **Tier-2 TLS lowering** (NIM.md §3d): a `tvar`
/// (thread-var) becomes a per-vCPU TLS-block access (`vcpu.tls.get() + off`) instead of a plain
/// window global. The generated module assumes the runtime has established this thread's TLS base
/// (`vcpu.tls.set`) before any `tvar` access — the block layout is per-`tvar` offsets from that base.
pub fn translate_tls_to_text(src: &str) -> Result<String, LengError> {
    let root = nif::parse(src).map_err(LengError::Parse)?;
    translate::Translator::new().with_tls().module(&root)
}

/// [`translate_tls_to_text`] parsed to an SVM-IR [`Module`] (unverified; the caller verifies).
pub fn translate_tls(src: &str) -> Result<Module, LengError> {
    let text = translate_tls_to_text(src)?;
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

/// Merge a fragment of external **type declarations** into a module before translation. `types` is
/// a `(stmts (type …)*)` fragment (e.g. the `string`/`LongString` defs from the compiled `system`
/// module, named with the global names the module references, like `string.0.sysvq0asl`); its type
/// defs are prepended so `collect_types` registers their layouts. This is the manual precursor to
/// automatic cross-module type resolution (NIM.md W2, Path A) — a module can't lower a value of an
/// external aggregate type (a string literal's `string`) without that type's layout.
fn merge_type_prelude(types: &str, src: &str) -> Result<Node, LengError> {
    let pre = nif::parse(types).map_err(LengError::Parse)?;
    let root = nif::parse(src).map_err(LengError::Parse)?;
    let mut merged = vec![Node::Atom("stmts".into())];
    for r in [pre, root] {
        if let Node::List(items) = r {
            merged.extend(items.into_iter().skip(1)); // drop the leading `stmts` atom
        }
    }
    Ok(Node::List(merged))
}

/// As [`translate_proc`], but with a fragment of external **type declarations** ([`merge_type_prelude`])
/// available — so a proc that constructs/returns a value of a type another module defines (a `string`
/// literal) can lower it. Returns the parsed (unverified) [`Module`].
pub fn translate_proc_with_types(src: &str, name: &str, types: &str) -> Result<Module, LengError> {
    let root = merge_type_prelude(types, src)?;
    let text = translate::Translator::new().one_proc(&root, name)?;
    svm_text::parse_module(&text).map_err(|e| {
        LengError::Malformed(format!(
            "emitted IR failed to parse: {e:?}\n--- IR ---\n{text}"
        ))
    })
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
/// local proc names to translate out of it. This is the **selective** ("go deep") shape: it lifts a
/// caller→callee pair out of a module whose *other* top-levels the skeleton can't lower yet. To
/// compile a module in full — every proc, its `ini`/C-`main` scaffolding and all — use
/// [`WholeModule`]/[`link_whole_units`] (NIM.md W2, Path A).
pub struct LengModule<'a> {
    pub stem: &'a str,
    pub src: &'a str,
    pub names: &'a [&'a str],
}

/// One nimony module linked **in full** (NIM.md W2, Path A) — every proc it defines, together with
/// the module scaffolding nimony emits (`ini` module-initializer, C `main`, exportc gvars). This is
/// what a real compiled module (e.g. the `system` module) is: it contributes *all* its procs, and
/// each is exported under its global (stem-suffixed) name so other modules' cross-module calls
/// resolve. The whole-module counterpart of [`LengModule`], with no proc selection to make.
pub struct WholeModule<'a> {
    pub stem: &'a str,
    pub src: &'a str,
}

/// How much of a module to translate into its link object: a named subset, or the whole thing.
#[derive(Clone, Copy)]
enum Select<'a> {
    Names(&'a [&'a str]),
    Whole,
}

/// Translate one nimony module as a **relocatable link unit** and stamp its in-band export table.
/// Link-unit mode: globals via `data.self` (so `link` relocates each unit's data into a disjoint
/// window region — an absolute-offset unit would silently alias), and each proc exported under its
/// **global** (stem-suffixed) name, the form nimony's cross-module calls reference (`callee.<stem>`).
/// `sel` is the named subset or the whole module; `ext_types` are external aggregate layouts
/// (sibling units' pooled type defs) available while translating — see [`link_units`].
fn translate_object_module(
    stem: &str,
    src: &str,
    sel: Select,
    ext_types: &[(String, translate::Layout)],
    ext_funcrefs: &[(String, translate::FnPtrSig)],
    ext_frame_procs: &[String],
    tls_layout: Option<&crate::dethash::HashMap<String, u64>>,
) -> Result<Module, LengError> {
    let root = nif::parse(src).map_err(LengError::Parse)?;
    let mut t = translate::Translator::new_for_link();
    t.import_types(ext_types);
    t.import_funcrefs(ext_funcrefs);
    t.import_proc_frames(ext_frame_procs);
    // Tier-2 TLS link (NIM.md §3d): inject the whole-program shared TLS layout so this unit's
    // `tvar` accesses — its own and any cross-module references — bake the agreed block offsets.
    if let Some(layout) = tls_layout {
        t.import_tls_layout(layout, stem);
    }
    // Whole module → translate every proc, exporting the exact local names the translator emitted
    // in func order; a named subset → exactly those, in list order.
    let (text, export_names) = match sel {
        Select::Whole => t.module_with_names(&root)?,
        Select::Names(names) => {
            let text = t.some_procs(&root, names)?;
            (text, names.iter().map(|s| s.to_string()).collect())
        }
    };
    let mut module = svm_text::parse_module(&text).map_err(|e| {
        LengError::Malformed(format!(
            "emitted IR failed to parse: {e:?}\n--- IR ---\n{text}"
        ))
    })?;
    // Procs export in-band under their global (stem-suffixed) names; this unit's `gvar`s export as
    // cross-module data symbols so another unit's `data.sym` can bind to them.
    module.exports = export_names
        .iter()
        .enumerate()
        .map(|(i, local)| Export {
            name: format!("{local}{stem}"),
            func: i as u32,
        })
        .collect();
    module.data_exports = t.global_exports(stem);
    // Funcref gvars with a static proc initializer (`var oomHandler = continueAfterOutOfMem`) ride a
    // `data.funcref` reloc the linker resolves to the merged funcidx — the value the gvar holds.
    module.data_funcrefs = t.funcref_relocs(stem);
    // Also expose `exportc` symbols under their C names (the conventional entry points — the C
    // `main`, `cmdCount`, …), so a host / `svm-run --link` can bind to them by name (Path A).
    let (ec_procs, ec_data) = t.exportc_exports(&root)?;
    module.exports.extend(ec_procs);
    module.data_exports.extend(ec_data);
    Ok(module)
}

/// Compile one nimony module to a binary **`.svmo` link object** (NIM.md W2, the object dialect) —
/// the narrow-waist artifact any linker consumer (`svm-run --link`, another frontend's build, a
/// cache) can take, the counterpart of `svm-llvm-translate -o out.svmo`. It's a relocatable link
/// unit: globals via `data.self`, cross-module callees as named imports, and its procs exported
/// **in-band** under their global names. Untrusted like any frontend output — the bytes pass the
/// hardened `decode_unit` firewall on the way back in, and the linked result is re-verified.
///
/// Compiled **standalone**, without sibling units: a value of an aggregate type *another* module
/// defines (a `string` literal's `string.0.<stem>`) fail-closes here, because its layout is needed
/// at translate time. [`link_units`] pools the linked units' type defs so those resolve.
pub fn compile_object(unit: &LengModule) -> Result<Vec<u8>, LengError> {
    Ok(svm_encode::encode_unit(&translate_object_module(
        unit.stem,
        unit.src,
        Select::Names(unit.names),
        &[],
        &[],
        &[],
        None,
    )?))
}

/// Compile a nimony module **in full** to a binary `.svmo` link object (NIM.md W2, Path A) — every
/// proc plus its scaffolding, each exported under its global name. The whole-module counterpart of
/// [`compile_object`]; the same untrusted-producer discipline (the bytes re-enter through the
/// hardened `decode_unit` firewall, and the linked result is re-verified).
pub fn compile_whole_object(unit: &WholeModule) -> Result<Vec<u8>, LengError> {
    Ok(svm_encode::encode_unit(&translate_object_module(
        unit.stem,
        unit.src,
        Select::Whole,
        &[],
        &[],
        &[],
        None,
    )?))
}

/// The shared link engine (NIM.md W2), *through the `.svmo` narrow waist*: each `(stem, src, sel)` is
/// compiled to a binary object, decoded back through the hardened `decode_unit` firewall, paired into
/// a [`svm_ir::LinkUnit`] from its in-band export tables (the same conversion `svm-run --link` does),
/// and statically linked into one svm-ir [`Module`]. Units keep the given order, so the first
/// module's first proc is func 0 (a natural entry). Not verified here (untrusted frontend — the
/// caller runs `svm_verify::verify_module` on the result).
///
/// **Cross-module type resolution**: proc and data symbols resolve at *link* time, but an aggregate
/// **type**'s layout is needed at *translate* time (field offsets are baked into loads/stores). So
/// before translating any unit, every unit's `(type …)` defs are pooled under their stem-suffixed
/// global names ([`translate::Translator::export_types`]) and made available to all — a module
/// constructing a `string.0.sysvq0asl` gets the system module's layout automatically, with no
/// hand-supplied prelude.
fn link_selected(units: &[(&str, &str, Select)]) -> Result<Module, LengError> {
    link_selected_with_extra(units, Vec::new(), false, false)
}

/// [`link_selected`] with pre-built **runtime** link units appended (e.g. the W3 bottom-edge shim
/// binding `mmap`/`memcpy`/atomics). The nimony units pool their aggregate types and compile as
/// usual; the extra units link in as-is, so their exports resolve the nimony modules' unbound
/// bottom-edge imports. This is how a real program links: the compiled stdlib plus a host runtime.
fn link_selected_with_extra(
    units: &[(&str, &str, Select)],
    extra: Vec<svm_ir::LinkUnit>,
    tls: bool,
    manifest: bool,
) -> Result<Module, LengError> {
    let mut pooled = Vec::new();
    let mut pooled_funcrefs = Vec::new();
    // Frame-graph nodes across all units: (global_name, own_needs_frame, global_callees).
    let mut frame_nodes: Vec<(String, bool, Vec<String>)> = Vec::new();
    // Tier-2 TLS (NIM.md §3d): pooled `(stem-suffixed tvar name, size)` across all units, in unit
    // order, to lay out the one shared per-vCPU block below.
    let mut pooled_tls: Vec<(String, u64)> = Vec::new();
    for (stem, src, _) in units {
        let root = nif::parse(src).map_err(LengError::Parse)?;
        pooled.extend(translate::Translator::export_types(&root, stem)?);
        pooled_funcrefs.extend(translate::Translator::export_funcrefs(&root, stem)?);
        frame_nodes.extend(translate::Translator::proc_frame_nodes(&root, stem)?);
        if tls {
            pooled_tls.extend(translate::Translator::export_tls_vars(&root, stem)?);
        }
    }
    // The shared TLS layout: each thread-var gets a disjoint offset in the per-vCPU block. Every
    // unit is handed this map, so a `tvar` defined in one unit and referenced from another lower to
    // the same `vcpu.tls.get()+off` — the offset-agreement a cross-module global gets from `data.sym`.
    let tls_layout: Option<crate::dethash::HashMap<String, u64>> = tls.then(|| {
        let mut layout = crate::dethash::HashMap::default();
        let mut off = 0u64;
        for (name, size) in &pooled_tls {
            layout.insert(name.clone(), off);
            off += size;
        }
        layout
    });
    // Whole-program frame fixpoint: a proc needs a frame if it does itself, or if it calls one that
    // does — transitively, across module boundaries (`program → seq → alloc → alloc.0.`). Each unit
    // then translates knowing the *final* frame-need of every callee, so a cross-module call to a
    // frame-needing proc passes `$sp` and never mismatches the resolved signature.
    let mut pooled_frame_procs: crate::dethash::HashSet<String> = frame_nodes
        .iter()
        .filter(|(_, own, _)| *own)
        .map(|(name, _, _)| name.clone())
        .collect();
    loop {
        let mut added = false;
        for (name, _, callees) in &frame_nodes {
            if !pooled_frame_procs.contains(name)
                && callees.iter().any(|c| pooled_frame_procs.contains(c))
            {
                pooled_frame_procs.insert(name.clone());
                added = true;
            }
        }
        if !added {
            break;
        }
    }
    let pooled_frame_procs: Vec<String> = pooled_frame_procs.into_iter().collect();
    let objects: Vec<Vec<u8>> = units
        .iter()
        .map(|(stem, src, sel)| {
            Ok(svm_encode::encode_unit(&translate_object_module(
                stem,
                src,
                *sel,
                &pooled,
                &pooled_funcrefs,
                &pooled_frame_procs,
                tls_layout.as_ref(),
            )?))
        })
        .collect::<Result<_, LengError>>()?;
    let mut link_units = Vec::with_capacity(objects.len() + extra.len());
    for bytes in &objects {
        let module = svm_encode::decode_unit(bytes)
            .map_err(|e| LengError::Malformed(format!("decode `.svmo` object: {e:?}")))?;
        let exports = module
            .exports
            .iter()
            .map(|e| (e.name.clone(), e.func))
            .collect();
        let data_exports = module
            .data_exports
            .iter()
            .map(|e| (e.name.clone(), e.offset))
            .collect();
        link_units.push(svm_ir::LinkUnit {
            module,
            exports,
            data_exports,
        });
    }
    link_units.extend(extra);
    // `manifest`: retain any import **no unit exports** (the raw-syscall leaves — `write`/`read`/
    // `_exit`, spelled by their nim symbol) in the merged manifest instead of failing the link, so
    // the host binds them at instantiation (NIM.md W3: "raw syscalls → unresolved named imports the
    // POSIX personality resolves at load"). The fail-closed `link` is the default (a whole-program
    // link whose every leaf resolves to the pure-IR runtime shim).
    let linked = if manifest {
        svm_ir::link_with_manifest(&link_units)
    } else {
        svm_ir::link(&link_units)
    };
    linked.map_err(|e| LengError::Malformed(format!("link failed: {e:?}")))
}

/// **Link several nimony modules into one svm-ir [`Module`]** (NIM.md W2), each contributing a
/// selected subset of its procs ([`LengModule`]). See [`link_selected`] for the mechanism. Use
/// [`link_whole_units`] to link modules *in full* (Path A).
pub fn link_units(units: &[LengModule]) -> Result<Module, LengError> {
    let sel: Vec<(&str, &str, Select)> = units
        .iter()
        .map(|u| (u.stem, u.src, Select::Names(u.names)))
        .collect();
    link_selected(&sel)
}

/// **Link several nimony modules in full** into one svm-ir [`Module`] (NIM.md W2, Path A): every
/// module contributes *all* its procs and scaffolding ([`WholeModule`]) — the shape a real program
/// links, where each `.svmo` is a whole compiled module (the analog of `lld` over object files).
/// See [`link_selected`] for the mechanism.
pub fn link_whole_units(units: &[WholeModule]) -> Result<Module, LengError> {
    let sel: Vec<(&str, &str, Select)> = units
        .iter()
        .map(|u| (u.stem, u.src, Select::Whole))
        .collect();
    link_selected(&sel)
}

/// Link whole nimony modules **together with a host runtime** (NIM.md W3): the program and the
/// compiled `system` module link as in [`link_whole_units`], and `runtime` supplies pre-built link
/// units whose exports bind the stdlib's bottom-edge C imports (`mmap`, `memcpy`, the atomics, …).
/// This is the full shape of a running program — real stdlib over a host runtime — as one linked
/// [`Module`] (still unverified; the caller runs `svm_verify::verify_module`).
pub fn link_whole_with_runtime(
    units: &[WholeModule],
    runtime: Vec<svm_ir::LinkUnit>,
) -> Result<Module, LengError> {
    let sel: Vec<(&str, &str, Select)> = units
        .iter()
        .map(|u| (u.stem, u.src, Select::Whole))
        .collect();
    link_selected_with_extra(&sel, runtime, false, false)
}

/// [`link_whole_with_runtime`], but **retaining the raw-syscall leaves** (`write`/`read`/`_exit`,
/// spelled by their nim symbol) as manifest imports the host binds at instantiation (NIM.md W3, the
/// POSIX-personality seam) — instead of fail-closing on them. The compute bottom edge (`memcpy`,
/// atomics, …) still resolves against `runtime`; only the imports **no unit exports** survive, so a
/// program that does real I/O links (the `write`/`read`/`_exit` a pure-IR shim can't provide) and
/// runs once its host grants those slots. Feed the result to a powerbox runner (`run_powerbox`) or
/// bind each retained slot to a host proc; re-verify like any linked output.
pub fn link_whole_with_runtime_manifest(
    units: &[WholeModule],
    runtime: Vec<svm_ir::LinkUnit>,
) -> Result<Module, LengError> {
    let sel: Vec<(&str, &str, Select)> = units
        .iter()
        .map(|u| (u.stem, u.src, Select::Whole))
        .collect();
    link_selected_with_extra(&sel, runtime, false, true)
}

/// **Link several nimony modules in Tier-2 TLS mode** (NIM.md §3d) together with a runtime that
/// establishes each thread's TLS base. Like [`link_units`], but `tvar`s lower to per-vCPU TLS-block
/// accesses over one **shared** block layout (so a thread-var defined in one unit — e.g. the
/// allocator state in `system` — and referenced from another agree on its offset). `runtime` supplies
/// the pre-built link units that `vcpu.tls.set` a real block base at thread entry (the analog of the
/// W3 allocator shim); without such a unit the linked module has no valid TLS base to run against.
pub fn link_units_tls_with_runtime(
    units: &[LengModule],
    runtime: Vec<svm_ir::LinkUnit>,
) -> Result<Module, LengError> {
    let sel: Vec<(&str, &str, Select)> = units
        .iter()
        .map(|u| (u.stem, u.src, Select::Names(u.names)))
        .collect();
    link_selected_with_extra(&sel, runtime, true, false)
}

/// The **pure-IR runtime for the C bottom edge** (NIM.md §3b, W3 — issue #761). The compiled Nim
/// `system` module bottoms out at ~15 `{.importc.}`/magic leaves; the *compute* ones (no OS authority
/// needed) lower to plain SVM ops, so they bind as ordinary linked SVM functions — **not** host
/// capabilities — keeping the runtime inside the pure-IR / both-engines model (the seam
/// [`link_whole_with_runtime`] exists for). The allocator (`mmap`/`munmap`) and syscalls
/// (`write`/`_exit`/…) still need the Memory cap / POSIX personality (Phase-1 seam); those are *not*
/// here — every leaf provided here is pure compute.
///
/// Function indices ([`bottom_edge_index`] maps a C leaf name to one):
/// `0` `memcpy(dst,src,n)->dst` · `1` `memset(dst,c,n)->dst` · `2` `bswap64(x)->x` ·
/// `3` `clzll(x)` · `4` `ctzll(x)` · `5` `popcountll(x)` · `6` `__atomic_add_fetch(p,v)->new` ·
/// `7` `__atomic_sub_fetch(p,v)->new` · `8` `__atomic_fetch_add(p,v)->old` ·
/// `9` `__atomic_exchange(p,v)->old` · `10` `__atomic_load(p)->v` · `11` `__atomic_store(p,v)` ·
/// `12` `memcmp(a,b,n)` (first differing unsigned byte's `a[i]-b[i]`, or `0`).
/// Atomics are single-threaded lowerings (plain load/modify/store) — correct for a single-vCPU guest
/// (NIM.md §3d); a threaded ARC guest would bind the `rmw.*` ops instead.
const BOTTOM_EDGE_RUNTIME: &str = r#"
func (i64, i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64, v2: i64) {
  mem.copy v0 v1 v2
  return v0
  }
}
func (i64, i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64, v2: i64) {
  v3 = i32.wrap_i64 v1
  mem.fill v0 v3 v2
  return v0
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 56
  v2 = i64.shr_u v0 v1
  v3 = i64.const 40
  v4 = i64.shr_u v0 v3
  v5 = i64.const 65280
  v6 = i64.and v4 v5
  v7 = i64.or v2 v6
  v8 = i64.const 24
  v9 = i64.shr_u v0 v8
  v10 = i64.const 16711680
  v11 = i64.and v9 v10
  v12 = i64.or v7 v11
  v13 = i64.const 8
  v14 = i64.shr_u v0 v13
  v15 = i64.const 4278190080
  v16 = i64.and v14 v15
  v17 = i64.or v12 v16
  v18 = i64.shl v0 v13
  v19 = i64.const 1095216660480
  v20 = i64.and v18 v19
  v21 = i64.or v17 v20
  v22 = i64.shl v0 v8
  v23 = i64.const 280375465082880
  v24 = i64.and v22 v23
  v25 = i64.or v21 v24
  v26 = i64.shl v0 v3
  v27 = i64.const 71776119061217280
  v28 = i64.and v26 v27
  v29 = i64.or v25 v28
  v30 = i64.shl v0 v1
  v31 = i64.or v29 v30
  return v31
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.clz v0
  return v1
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.ctz v0
  return v1
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.popcnt v0
  return v1
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.load v0
  v3 = i64.add v2 v1
  i64.store v0 v3
  return v3
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.load v0
  v3 = i64.sub v2 v1
  i64.store v0 v3
  return v3
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.load v0
  v3 = i64.add v2 v1
  i64.store v0 v3
  return v2
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.load v0
  i64.store v0 v1
  return v2
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.load v0
  return v1
  }
}
func (i64, i64) -> () {
block 0 (v0: i64, v1: i64) {
  i64.store v0 v1
  return
  }
}
func (i64, i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64, v2: i64) {
  v3 = i64.const 0
  br 1(v0, v1, v2, v3)
  }
block 1 (v0: i64, v1: i64, v2: i64, v3: i64) {
  v4 = i64.lt_u v3 v2
  br_if v4 2(v0, v1, v2, v3) 4()
  }
block 2 (v0: i64, v1: i64, v2: i64, v3: i64) {
  v4 = i64.add v0 v3
  v5 = i64.load8_u v4
  v6 = i64.add v1 v3
  v7 = i64.load8_u v6
  v8 = i64.eq v5 v7
  br_if v8 3(v0, v1, v2, v3) 5(v5, v7)
  }
block 3 (v0: i64, v1: i64, v2: i64, v3: i64) {
  v4 = i64.const 1
  v5 = i64.add v3 v4
  br 1(v0, v1, v2, v5)
  }
block 4 () {
  v0 = i64.const 0
  return v0
  }
block 5 (v0: i64, v1: i64) {
  v2 = i64.sub v0 v1
  return v2
  }
}
"#;

/// The **pure-IR C-bottom-edge runtime** module (see [`BOTTOM_EDGE_RUNTIME`]): the compute leaves of
/// the Nim stdlib's `{.importc.}` surface (NIM.md §3b, W3 — #761) as ordinary SVM functions, ready to
/// [`svm_ir::link`] against a translated module whose matching imports are bound via
/// [`bottom_edge_index`]. Panics only on an internal svm-text regression (the const is a fixed literal).
pub fn bottom_edge_runtime() -> Module {
    svm_text::parse_module(BOTTOM_EDGE_RUNTIME).expect("bottom-edge runtime svm-text parses")
}

/// Map a C bottom-edge leaf name (or any import name containing it — nimony spells `copyMem`'s import
/// as the C `memcpy`) to its [`bottom_edge_runtime`] function index, or `None` if this runtime does
/// not provide it (the allocator/syscall leaves). Substring match, so a stem-qualified
/// or namespaced spelling still binds.
pub fn bottom_edge_index(name: &str) -> Option<u32> {
    Some(match name {
        n if n.contains("memcpy") => 0,
        n if n.contains("memset") => 1,
        n if n.contains("bswap") => 2,
        n if n.contains("clz") => 3,
        n if n.contains("ctz") => 4,
        n if n.contains("popcount") || n.contains("popcnt") => 5,
        n if n.contains("add_fetch") => 6,
        n if n.contains("sub_fetch") => 7,
        n if n.contains("fetch_add") => 8,
        n if n.contains("exchange") => 9,
        n if n.contains("atomic_load") => 10,
        n if n.contains("atomic_store") => 11,
        n if n.contains("memcmp") => 12,
        _ => return None,
    })
}

/// A translated SVM value: its SSA id and type. The unit the expression translator threads.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Val {
    pub id: u32,
    pub ty: ValType,
}
