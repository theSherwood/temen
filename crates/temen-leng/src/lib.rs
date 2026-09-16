//! `temen-leng` — a **Leng-NIF → TEMEN-IR** translator (NIM.md Phase 2).
//!
//! Leng is nimony's C-like midlevel IR (`doc/leng-spec.md`): the seam its C/C++/LLVM/arkham
//! backends already consume, sitting just after hexer's lowering (ARC, exceptions, monomorph).
//! This is the **fourth Temen frontend**, alongside the chibicc C fork, `temen-wasm`, and `temen-llvm`.
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
//! binary **`.temeno` link object** ([`compile_object`], exports in-band) that composes with other
//! producers' objects through the shared `temen_ir::link` — [`link_units`] does exactly that across
//! several nimony modules (NIM.md W2). `object of RootObj` **inheritance** lowers too: a derived object inlines its
//! base's layout after a leading vtable header, and the vtable pointer is stored but opaque (dynamic
//! dispatch fail-closes). What remains
//! outside the subset — `union`, `emit`, dynamic method dispatch, value-object exception payloads
//! (an object punned into the error tuple's scalar slot) — is a fail-closed [`LengError::Unsupported`],
//! never a silent mistranslation (the `temen-wasm`/`temen-llvm` `unsup(...)` discipline). (The
//! `jtrue`/`mflag`/`vflag` cfvar jump forms never reach us — hexer's `xelim` lowers them away before
//! the final IR.) Growing the frontend means adding grammar arms below, not rearchitecting.
//!
//! Like chibicc's `codegen_ir.c`, it emits **Temen text** and hands it to [`temen_text::parse_module`];
//! [`translate`] returns the parsed (but not-yet-verified) [`Module`].
//!
//! ```
//! let leng = "(stmts (proc :main.0 . (i +64) . (stmts . (ret (add (i +64) 3 (mul (i +64) 4 2))))))";
//! let m = temen_leng::translate(leng).unwrap();      // one func: () -> (i64) returning 11
//! assert_eq!(m.funcs.len(), 1);
//! ```

use temen_ir::{Export, Module, ValType};

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

/// Translate a Leng-NIF module to **Temen text**. The seam a caller can inspect/debug (the emitted
/// IR is human-readable and rides `temen_text::parse_module`).
pub fn translate_to_text(src: &str) -> Result<String, LengError> {
    let root = nif::parse(src).map_err(LengError::Parse)?;
    translate::Translator::new().module(&root)
}

/// Translate a Leng-NIF module to an TEMEN-IR [`Module`]. The module is **not** verified here —
/// callers run `temen_verify::verify_module` (the frontend is untrusted; DESIGN.md §2a).
pub fn translate(src: &str) -> Result<Module, LengError> {
    let text = translate_to_text(src)?;
    temen_text::parse_module(&text).map_err(|e| {
        LengError::Malformed(format!(
            "emitted IR failed to parse: {e:?}\n--- IR ---\n{text}"
        ))
    })
}

/// Translate a Leng-NIF module to Temen text with **Tier-2 TLS lowering** (NIM.md §3d): a `tvar`
/// (thread-var) becomes a per-vCPU TLS-block access (`vcpu.tls.get() + off`) instead of a plain
/// window global. The generated module assumes the runtime has established this thread's TLS base
/// (`vcpu.tls.set`) before any `tvar` access — the block layout is per-`tvar` offsets from that base.
pub fn translate_tls_to_text(src: &str) -> Result<String, LengError> {
    let root = nif::parse(src).map_err(LengError::Parse)?;
    translate::Translator::new().with_tls().module(&root)
}

/// [`translate_tls_to_text`] parsed to an TEMEN-IR [`Module`] (unverified; the caller verifies).
pub fn translate_tls(src: &str) -> Result<Module, LengError> {
    let text = translate_tls_to_text(src)?;
    temen_text::parse_module(&text).map_err(|e| {
        LengError::Malformed(format!(
            "emitted IR failed to parse: {e:?}\n--- IR ---\n{text}"
        ))
    })
}

/// Translate a **single named proc** out of a full Leng module to Temen text — the "go deep" entry
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
    temen_text::parse_module(&text).map_err(|e| {
        LengError::Malformed(format!(
            "emitted IR failed to parse: {e:?}\n--- IR ---\n{text}"
        ))
    })
}

/// As [`translate_proc_to_text`], returning the parsed (unverified) [`Module`].
pub fn translate_proc(src: &str, name: &str) -> Result<Module, LengError> {
    let text = translate_proc_to_text(src, name)?;
    temen_text::parse_module(&text).map_err(|e| {
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
    temen_text::parse_module(&text).map_err(|e| {
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
// The pooled cross-module inputs (types, funcrefs, frame procs, sret procs, TLS) are each a distinct
// list the linker threads in; grouping them into a struct would obscure more than it clarifies here.
#[allow(clippy::too_many_arguments)]
fn translate_object_module(
    stem: &str,
    src: &str,
    sel: Select,
    ext_types: &[(String, translate::Layout)],
    ext_funcrefs: &[(String, translate::FnPtrSig)],
    ext_frame_procs: &[String],
    ext_sret: &[(String, translate::TyDesc)],
    ext_proc_params: &[translate::ProcParamSig],
    ext_consts: &[(String, i64)],
    tls_layout: Option<&crate::dethash::HashMap<String, u64>>,
) -> Result<Module, LengError> {
    let root = nif::parse(src).map_err(LengError::Parse)?;
    let mut t = translate::Translator::new_for_link();
    t.import_types(ext_types);
    t.import_funcrefs(ext_funcrefs);
    t.import_proc_frames(ext_frame_procs);
    t.import_sret_procs(ext_sret);
    t.import_proc_params(ext_proc_params);
    t.import_consts(ext_consts);
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
    let mut module = temen_text::parse_module(&text).map_err(|e| {
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
    // `main`, `cmdCount`, …), so a host / `temen-run --link` can bind to them by name (Path A).
    let (ec_procs, ec_data) = t.exportc_exports(&root)?;
    module.exports.extend(ec_procs);
    module.data_exports.extend(ec_data);
    Ok(module)
}

/// Compile one nimony module to a binary **`.temeno` link object** (NIM.md W2, the object dialect) —
/// the narrow-waist artifact any linker consumer (`temen-run --link`, another frontend's build, a
/// cache) can take, the counterpart of `temen-llvm-translate -o out.temeno`. It's a relocatable link
/// unit: globals via `data.self`, cross-module callees as named imports, and its procs exported
/// **in-band** under their global names. Untrusted like any frontend output — the bytes pass the
/// hardened `decode_unit` firewall on the way back in, and the linked result is re-verified.
///
/// Compiled **standalone**, without sibling units: a value of an aggregate type *another* module
/// defines (a `string` literal's `string.0.<stem>`) fail-closes here, because its layout is needed
/// at translate time. [`link_units`] pools the linked units' type defs so those resolve.
pub fn compile_object(unit: &LengModule) -> Result<Vec<u8>, LengError> {
    Ok(temen_encode::encode_unit(&translate_object_module(
        unit.stem,
        unit.src,
        Select::Names(unit.names),
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        None,
    )?))
}

/// Compile a nimony module **in full** to a binary `.temeno` link object (NIM.md W2, Path A) — every
/// proc plus its scaffolding, each exported under its global name. The whole-module counterpart of
/// [`compile_object`]; the same untrusted-producer discipline (the bytes re-enter through the
/// hardened `decode_unit` firewall, and the linked result is re-verified).
pub fn compile_whole_object(unit: &WholeModule) -> Result<Vec<u8>, LengError> {
    Ok(temen_encode::encode_unit(&translate_object_module(
        unit.stem,
        unit.src,
        Select::Whole,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        None,
    )?))
}

/// The shared link engine (NIM.md W2), *through the `.temeno` narrow waist*: each `(stem, src, sel)` is
/// compiled to a binary object, decoded back through the hardened `decode_unit` firewall, paired into
/// a [`temen_ir::LinkUnit`] from its in-band export tables (the same conversion `temen-run --link` does),
/// and statically linked into one temen-ir [`Module`]. Units keep the given order, so the first
/// module's first proc is func 0 (a natural entry). Not verified here (untrusted frontend — the
/// caller runs `temen_verify::verify_module` on the result).
///
/// **Cross-module type resolution**: proc and data symbols resolve at *link* time, but an aggregate
/// **type**'s layout is needed at *translate* time (field offsets are baked into loads/stores). So
/// before translating any unit, every unit's `(type …)` defs are pooled under their stem-suffixed
/// global names ([`translate::Translator::export_types_pooled`]) and made available to all — a module
/// constructing a `string.0.sysvq0asl` gets the system module's layout automatically, with no
/// hand-supplied prelude.
fn link_selected(units: &[(&str, &str, Select)]) -> Result<Module, LengError> {
    link_selected_with_extra(units, Vec::new(), false, false, false)
}

/// Build the **powerbox `_start` link unit** (function 0): a paramless entry that reads the
/// post-link data-stack base (`data.top`, which the linker resolves to `powerbox_entry_sp` and
/// reserves stack above) and tail-calls the C-shaped `main($sp, argc, argv, envp)` with
/// `argc/argv/envp = 0`, returning its `cint`. `main` is a cross-unit symbol resolved at link
/// (`call.import "main"` → a direct `call` once merged). Injecting the entry **as a unit** (linked
/// first, so it is function 0) — rather than [`temen_ir::synth_manifest_start`]-prepending it after
/// the link — is what keeps the program's `data.funcref` initializers valid: the linker numbers
/// `_start` first and bakes every funcref at its final merged index in one pass, so nothing needs a
/// post-hoc +1 shift (which the discarded relocation metadata could no longer drive).
///
/// The guest heap bump-pointer words ([`temen_ir::POWERBOX_HEAP_BRK`]/[`POWERBOX_HEAP_TOP`]) are
/// **not** seeded here — this unit is built before the merged window size is known, and the heap
/// ceiling *is* that window top. [`seed_powerbox_heap`] bakes both words into the linked module's
/// data image (post-link, where the window is known); see it for the #1051/#1054/#1060 rationale.
fn synth_start_unit(entry: &str) -> Result<temen_ir::LinkUnit, LengError> {
    // `argc = 0`, and `argv`/`envp` point at **one-entry NULL-terminated vectors** rather than being
    // NULL themselves (#1422): `_start` writes the terminator into the reserved page-0 scratch at
    // [`temen_ir::POWERBOX_EMPTY_ARGV`]/[`POWERBOX_EMPTY_ENVP`] and hands `main` their addresses.
    // Passing 0 is what a C `main` is never given, and nim's `getEnvVarsC` walks `nimEnviron` until
    // it reads NULL — so a null `envp` faulted on the very first load against the #1094 guard,
    // taking every module that reaches `std/envvars` (`os`, `paths`, `strtabs`, `appdirs`, …) down
    // with it. One store each, paid once per run.
    let argv = temen_ir::POWERBOX_NULL_GUARD + temen_ir::POWERBOX_EMPTY_ARGV;
    let envp = temen_ir::POWERBOX_NULL_GUARD + temen_ir::POWERBOX_EMPTY_ENVP;
    let text = format!(
        "import 0 \"{entry}\" (i64, i32, i64, i64) -> (i32)\n\
         func () -> (i32) {{\n\
         block 0 () {{\n\
         \x20 v0 = data.top\n\
         \x20 v1 = i32.const 0\n\
         \x20 vz = i64.const 0\n\
         \x20 v2 = i64.const {argv}\n\
         \x20 i64.store v2 vz\n\
         \x20 v3 = i64.const {envp}\n\
         \x20 i64.store v3 vz\n\
         \x20 v4 = call.import 0 (v0, v1, v2, v3)\n\
         \x20 return v4\n\
         \x20 }}\n\
         }}\n"
    );
    let module = temen_text::parse_module(&text)
        .map_err(|e| LengError::Malformed(format!("synth `_start` unit: {e:?}")))?;
    Ok(temen_ir::LinkUnit {
        module,
        // #964/#1094 guarded layout: `[0, POWERBOX_NULL_GUARD)` is kept empty and a host seeds it
        // `Unmapped` unconditionally (the one canonical layout), so a NULL deref traps. Leng bases its
        // globals + scratch one guard up (`translate::globals_base`, `seed_powerbox_heap`) to keep that
        // region clear. The `__null_guard` marker export is retired (#1094) — the guard no longer needs
        // an opt-in signal.
        exports: vec![("_start".to_string(), 0)],
        ..Default::default()
    })
}

/// [`link_selected`] with pre-built **runtime** link units appended (e.g. the W3 bottom-edge shim
/// binding `mmap`/`memcpy`/atomics). The nimony units pool their aggregate types and compile as
/// usual; the extra units link in as-is, so their exports resolve the nimony modules' unbound
/// bottom-edge imports. This is how a real program links: the compiled stdlib plus a host runtime.
///
/// When `synth_start` is set, a powerbox `_start` unit ([`synth_start_unit`]) is linked **first**,
/// so the merged module is a runnable powerbox entry (function 0 = `_start`, calling `main`).
fn link_selected_with_extra(
    units: &[(&str, &str, Select)],
    extra: Vec<temen_ir::LinkUnit>,
    tls: bool,
    manifest: bool,
    synth_start: bool,
) -> Result<Module, LengError> {
    // Cross-module aggregate type layouts. Unlike the other pools, an object type can *inherit* a
    // base defined in another unit (`JsonParser = object of BaseLexer`, `BaseLexer` in a sibling
    // module), and the base is inlined into the derived layout at translate time — so a single
    // per-module pass exports a lossy layout (missing every inherited field) whenever the base is
    // cross-module. Pool to a **fixpoint** instead: each round re-exports with the prior round's
    // pool visible, so a chain of any depth is fully inlined. Inlining only *adds* fields, so the
    // summed field count is monotonic and converges; the unit count bounds the max chain depth.
    let roots: Vec<_> = units
        .iter()
        .map(|(_, src, _)| nif::parse(src).map_err(LengError::Parse))
        .collect::<Result<_, _>>()?;
    let mut pooled: Vec<(String, translate::Layout)> = Vec::new();
    let mut prev_fields = usize::MAX;
    for _ in 0..=units.len() {
        let mut next: Vec<(String, translate::Layout)> = Vec::new();
        for ((stem, _, _), root) in units.iter().zip(&roots) {
            next.extend(translate::Translator::export_types_pooled(
                root, stem, &pooled,
            )?);
        }
        let fields: usize = next.iter().map(|(_, l)| l.field_count()).sum();
        pooled = next;
        if fields == prev_fields {
            break;
        }
        prev_fields = fields;
    }
    let mut pooled_funcrefs = Vec::new();
    // Frame-graph nodes across all units: (global_name, own_needs_frame, global_callees).
    let mut frame_nodes: Vec<(String, bool, Vec<String>)> = Vec::new();
    // Tier-2 TLS (NIM.md §3d): pooled `(stem-suffixed tvar name, size)` across all units, in unit
    // order, to lay out the one shared per-vCPU block below.
    let mut pooled_tls: Vec<(String, u64)> = Vec::new();
    // Pooled **sret procs** across all units (stem-suffixed name → returned aggregate). A caller of
    // an aggregate-returning proc materializes a result temp and passes it as `$sret` — and a proc
    // that does so becomes frame-needing — so the sret set must be known **before** the frame
    // fixpoint (`proc_frame_nodes`) runs, hence a first pass over the units to build it.
    let mut pooled_sret: Vec<(String, translate::TyDesc)> = Vec::new();
    // Pooled **proc param types and return type** across all units (stem-suffixed name → declared
    // param ValTypes + return ValType): a cross-module call coerces each scalar arg to the callee's
    // real param type and its result from the callee's real return type, so a narrow value reaches a
    // wider param widened (#1400) and a narrow return lands widened (#1404) — the arg/return-width
    // twin of `pooled_sret`. Without it the import signature is call-site-derived and a width mismatch
    // fails verification post-link.
    let mut pooled_proc_params: Vec<translate::ProcParamSig> = Vec::new();
    // Pooled **scalar-int consts** across all units (stem-suffixed name → value): a scalar `const` is
    // inlined at use and never exported as data, so a cross-module reference to one (`replRune.0.<uni>`,
    // which `fastRuneAt`'s template expansion plants in every consumer) has no data symbol to bind.
    // Pooling lets the referencing unit inline the same value the defining unit does.
    let mut pooled_consts: Vec<(String, i64)> = Vec::new();
    for (stem, src, _) in units {
        let root = nif::parse(src).map_err(LengError::Parse)?;
        pooled_funcrefs.extend(translate::Translator::export_funcrefs(&root, stem)?);
        pooled_sret.extend(translate::Translator::export_sret_procs(&root, stem)?);
        pooled_proc_params.extend(translate::Translator::export_proc_params(&root, stem)?);
        pooled_consts.extend(translate::Translator::export_consts(&root, stem)?);
        if tls {
            pooled_tls.extend(translate::Translator::export_tls_vars(&root, stem)?);
        }
    }
    // Frame fixpoint input — computed now that every unit's sret-ness is pooled, so a proc that calls
    // an sret proc is correctly seen as frame-needing (its result temp lives in its own frame).
    for (stem, src, _) in units {
        let root = nif::parse(src).map_err(LengError::Parse)?;
        frame_nodes.extend(translate::Translator::proc_frame_nodes(
            &root,
            stem,
            &pooled,
            &pooled_sret,
        )?);
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
    // **Guest-libc leaves are frame-needing** ([`LIBC_SERVED`]): every chibicc-compiled function takes
    // a leading `$sp`, so a nim call site must pass one. Seeding the leaf symbols here makes
    // `call_import` prepend `sp + frame_size` (its `ext_frame_procs` check is on the *bare* import
    // name) and, through the fixpoint below, makes every caller frame-needing so it owns an `$sp` to
    // hand down. A callee edge is recorded in its **globalized** form (`proc_frame_nodes` suffixes a
    // trailing-`.` name with the importer's stem), so seed both spellings.
    for ((stem, _, _), root) in units.iter().zip(&roots) {
        for (sym, c) in translate::Translator::importc_procs(root)? {
            if libc_serves(&c) {
                pooled_frame_procs.insert(format!("{sym}{stem}"));
                pooled_frame_procs.insert(sym);
            }
        }
    }
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
            Ok(temen_encode::encode_unit(&translate_object_module(
                stem,
                src,
                *sel,
                &pooled,
                &pooled_funcrefs,
                &pooled_frame_procs,
                &pooled_sret,
                &pooled_proc_params,
                &pooled_consts,
                tls_layout.as_ref(),
            )?))
        })
        .collect::<Result<_, LengError>>()?;
    let mut link_units = Vec::with_capacity(objects.len() + extra.len() + 1);
    // The powerbox `_start` links **first** so it is function 0 (the manifest entry shape), and so
    // the linker's single-pass funcref baking numbers the program's `data.funcref` initializers
    // above it — no fragile post-link index shift. See [`synth_start_unit`].
    if synth_start {
        link_units.push(synth_start_unit("main")?);
    }
    for bytes in &objects {
        let module = temen_encode::decode_unit(bytes)
            .map_err(|e| LengError::Malformed(format!("decode `.temeno` object: {e:?}")))?;
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
        link_units.push(temen_ir::LinkUnit {
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
        temen_ir::link_with_manifest(&link_units)
    } else {
        temen_ir::link(&link_units)
    };
    let mut linked = linked.map_err(|e| LengError::Malformed(format!("link failed: {e:?}")))?;
    // A synth-`_start` module is a powerbox entry whose guest allocator bumps in-window; seed its
    // heap bump-pointer words now that the merged window is known (see [`seed_powerbox_heap`]).
    if synth_start {
        seed_powerbox_heap(&mut linked);
    }
    Ok(linked)
}

/// Seed the guest **heap bump-pointer words** into a linked powerbox module's data image:
/// [`temen_ir::POWERBOX_HEAP_BRK`] ← the heap base (just above the data stack), and
/// [`temen_ir::POWERBOX_HEAP_TOP`] ← the mapped-window top (`1 << size_log2`), the heap ceiling.
///
/// **Why (#1051 / #1054 / #1060).** The nim compute-shim `mmap` ([`POWERBOX_COMPUTE_SHIM`]) serves
/// the allocator by handing out `[brk, brk+len)` and advancing `POWERBOX_HEAP_BRK`. If that word is
/// left 0 the arena starts at address 0 and overlaps the placed static data — a heap allocation can
/// reuse a program's `LongString` const and `add`/realloc scribbles it (the #1051 corruption, whose
/// order-sensitivity was #1054). Seeding the base to `data.top + POWERBOX_STACK_RESERVE` puts the
/// heap above **all** placed data for any link order. Seeding the ceiling to the real window top
/// makes the heap use the whole remaining window (no capacity cliff) and lets the shim `mmap`
/// fail closed when the guest would bump past it (#1060) — the confinement mask already prevents an
/// out-of-window access from escaping, so this turns a self-corrupting wrap into a clean allocation
/// failure. The window itself is sized to hold `data + stack + heap` reserves by
/// [`temen_ir::link`] ([`temen_ir::POWERBOX_HEAP_RESERVE`]), independent of any runtime unit's
/// `memory N` declaration.
///
/// Baked as a 16-byte writable data segment (not `_start` stores) because it must run **before**
/// `main`, and because [`synth_start_unit`] is built before the window size is known. The two words
/// live in the guard's **scratch page** at `guard + POWERBOX_HEAP_BRK`/`TOP` (#1091): the #964 NULL
/// guard reserves `[0, POWERBOX_NULL_GUARD)` empty, so the pre-guard offsets 32/40 would be seeded
/// `Unmapped` and the compute-shim's `mmap` would fault reading them. The compute shim reads/advances
/// them at the same shifted offsets, and the DAP heap view already keys off `scratch + POWERBOX_HEAP_BRK`
/// (`scratch == module_null_guard`). No unit's globals cover this page (they base at
/// `guard + POWERBOX_STACK_PAGE`, one page above — see `translate::globals_base`).
fn seed_powerbox_heap(m: &mut Module) {
    let Some(mem) = m.memory else { return };
    let win = 1u64 << mem.size_log2;
    let brk = temen_ir::powerbox_entry_sp(m) + temen_ir::POWERBOX_STACK_RESERVE;
    // `POWERBOX_HEAP_TOP` sits 8 bytes above `POWERBOX_HEAP_BRK`; write both in one contiguous segment.
    debug_assert_eq!(temen_ir::POWERBOX_HEAP_TOP, temen_ir::POWERBOX_HEAP_BRK + 8);
    let mut bytes = brk.to_le_bytes().to_vec();
    bytes.extend_from_slice(&win.to_le_bytes());
    m.data.push(temen_ir::Data {
        offset: temen_ir::POWERBOX_NULL_GUARD + temen_ir::POWERBOX_HEAP_BRK,
        bytes,
        readonly: false,
    });
}

/// **Link several nimony modules into one temen-ir [`Module`]** (NIM.md W2), each contributing a
/// selected subset of its procs ([`LengModule`]). See [`link_selected`] for the mechanism. Use
/// [`link_whole_units`] to link modules *in full* (Path A).
pub fn link_units(units: &[LengModule]) -> Result<Module, LengError> {
    let sel: Vec<(&str, &str, Select)> = units
        .iter()
        .map(|u| (u.stem, u.src, Select::Names(u.names)))
        .collect();
    link_selected(&sel)
}

/// **Link several nimony modules in full** into one temen-ir [`Module`] (NIM.md W2, Path A): every
/// module contributes *all* its procs and scaffolding ([`WholeModule`]) — the shape a real program
/// links, where each `.temeno` is a whole compiled module (the analog of `lld` over object files).
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
/// [`Module`] (still unverified; the caller runs `temen_verify::verify_module`).
pub fn link_whole_with_runtime(
    units: &[WholeModule],
    runtime: Vec<temen_ir::LinkUnit>,
) -> Result<Module, LengError> {
    let sel: Vec<(&str, &str, Select)> = units
        .iter()
        .map(|u| (u.stem, u.src, Select::Whole))
        .collect();
    link_selected_with_extra(&sel, runtime, false, false, false)
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
    runtime: Vec<temen_ir::LinkUnit>,
) -> Result<Module, LengError> {
    let sel: Vec<(&str, &str, Select)> = units
        .iter()
        .map(|u| (u.stem, u.src, Select::Whole))
        .collect();
    link_selected_with_extra(&sel, runtime, false, true, false)
}

/// [`link_whole_with_runtime_manifest`] **plus a synthesized powerbox `_start`** (function 0): the
/// merged module is a runnable **powerbox entry** — `_start` reads the post-link data-stack base and
/// calls the program's C `main($sp, argc, argv, envp)` (with empty argv), returning its `cint`. The
/// raw-syscall leaves are still retained as manifest imports the host binds at instantiation. This is
/// the full Path-B I/O shape: hand the result to `run_powerbox`-style caps or bind each retained slot
/// to a host proc and run function 0 with no args — no hand-provided `$sp`. Re-verify like any linked
/// output. See [`synth_start_unit`] for why the entry is injected as a first link unit rather than
/// prepended after the fact.
pub fn link_whole_powerbox_manifest(
    units: &[WholeModule],
    runtime: Vec<temen_ir::LinkUnit>,
) -> Result<Module, LengError> {
    let sel: Vec<(&str, &str, Select)> = units
        .iter()
        .map(|u| (u.stem, u.src, Select::Whole))
        .collect();
    link_selected_with_extra(&sel, runtime, false, true, true)
}

/// The **pure-compute C bottom edge** as TEMEN-text funcs — nimony's `memcpy`/`memcmp`/`memset`,
/// the `atomic*` family, `bswap64`/`ctz64`/`clz64`, and a **bump `mmap`** (serves from the powerbox
/// heap-brk word) — plus stubbed `exit`/`getpid`/`kill`/`cWriteErr`/`dl*`. Linked into a nim program
/// so those leaves resolve to compiled code; only the true syscalls are left for the host.
const POWERBOX_COMPUTE_SHIM: &str = include_str!("powerbox_compute_shim.temt.txt");

/// The compute-shim func index serving each pure-compute leaf (longest-prefix match — so
/// `atomicCompareExchangeN` isn't shadowed by a shorter atomic). The func order is
/// [`POWERBOX_COMPUTE_SHIM`]'s.
/// A pure-compute leaf binding: the symbol's **name prefix**, the signature it must match, and the
/// [`POWERBOX_COMPUTE_SHIM`] func index serving it.
///
/// Most leaves are a unique C symbol, so [`ANY`] matches them on name alone (longest prefix wins —
/// `atomicCompareExchangeN` isn't shadowed by a shorter atomic). The generic `builtin*` atomics
/// (#1499) instead pin an exact signature, because one name covers several monomorphized widths.
type ComputeLeaf = (&'static str, Option<ComputeSig>, u32);

/// `(params, results)` a signature-qualified leaf must match exactly.
type ComputeSig = (&'static [ValType], &'static [ValType]);

/// Match on the name alone — the signature is not consulted.
const ANY: Option<ComputeSig> = None;

const I32: ValType = ValType::I32;
const I64: ValType = ValType::I64;

/// Pin a leaf to one exact signature (const-fn so the table stays a `const`).
const fn sig(params: &'static [ValType], results: &'static [ValType]) -> Option<ComputeSig> {
    Some((params, results))
}

const COMPUTE_LEAVES: &[ComputeLeaf] = &[
    ("cExitSys", ANY, 0),
    ("cGetpid", ANY, 1),
    ("cKill", ANY, 2),
    ("c_memcpy", ANY, 3),
    ("c_memcmp", ANY, 4),
    ("c_memset", ANY, 5),
    ("mmap", ANY, 6),
    ("atomicLoadN", ANY, 7),
    ("atomicStoreN", ANY, 8),
    ("atomicCompareExchangeN", ANY, 9),
    ("atomicExchangeN", ANY, 10),
    ("atomicAddFetch", ANY, 11),
    ("atomicSubFetch", ANY, 12),
    ("bswap64", ANY, 13),
    ("ctz64", ANY, 14),
    ("clz64", ANY, 15),
    ("cWriteErr", ANY, 16),
    ("dlopen", ANY, 17),
    ("dlclose", ANY, 18),
    ("dlsym", ANY, 19),
    // **POSIX leaves nim's `std/posix` declares** (pulled in by `std/times`/`std/monotimes`). A
    // sandboxed program is granted no ambient filesystem, process table, or clock, so these are
    // fail-closed stubs rather than real syscalls: `open`/`getdents64`/`wait4`/`execve` report
    // failure, `close` succeeds harmlessly, and `clock_gettime` writes a **zero timespec** and
    // succeeds — a deterministic epoch, so a `times` program runs and reads a well-defined time
    // instead of being handed ambient authority (#1422). Granting a real clock is a capability
    // decision for the host, not a default of the nim bottom edge.
    ("open", ANY, 20),
    ("close", ANY, 21),
    ("getdents64", ANY, 22),
    ("wait4", ANY, 23),
    ("execve", ANY, 24),
    ("clock_gettime", ANY, 25),
    // **The rest of the posix bottom edge** `std/os`/`paths`/`dirs`/`envvars`/`strtabs`/`appdirs`/
    // `memfiles`/`osproc`/`terminal`/`rawthreads` declare (#1422). Same posture as the six above and
    // for the same reason: a playground guest is granted no ambient filesystem, environment, process
    // table, or OS threads, so every one of these is a **fail-closed stub** — the metadata and
    // mutation calls report failure, `getcwd`/`c_getenv` report "absent" (a null pointer, i.e. an
    // empty environment and no current directory), and `nanosleep` succeeds immediately.
    //
    // The point is *linkability*, not emulation: a program that only uses the pure half of these
    // modules — `paths`/`pathnorm` string manipulation, a `strtabs` table, `os`'s path helpers — now
    // links and runs, where before the whole module failed to link and nothing in it was reachable.
    // A program that genuinely touches the filesystem gets nim's ordinary error path (an `OSError`,
    // an empty result) instead of a link failure. Serving any of these for real is a capability the
    // host grants, not a default of the nim bottom edge.
    ("stat", ANY, 26),
    ("lstat", ANY, 27),
    ("fstat", ANY, 28),
    ("mkdir", ANY, 29),
    ("rmdir", ANY, 30),
    ("unlink", ANY, 31),
    ("chdir", ANY, 32),
    ("getcwd", ANY, 33),
    ("readlink", ANY, 34),
    ("ftruncate", ANY, 35),
    ("munmap", ANY, 36),
    ("c_rename", ANY, 37),
    ("c_getenv", ANY, 38),
    ("c_setenv", ANY, 39),
    ("c_unsetenv", ANY, 40),
    ("fork", ANY, 41),
    ("exitnow", ANY, 42),
    ("pipe", ANY, 43),
    ("dup2", ANY, 44),
    ("setpgid", ANY, 45),
    ("kill", ANY, 46),
    ("nanosleep", ANY, 47),
    ("sysconf", ANY, 48),
    ("nativeIoctl", ANY, 49),
    ("pthread_attr_init", ANY, 50),
    ("pthread_attr_setstacksize", ANY, 51),
    ("pthread_attr_destroy", ANY, 52),
    ("pthread_create", ANY, 53),
    ("pthread_join", ANY, 54),
    ("cpusetZero", ANY, 55),
    ("cpusetIncl", ANY, 56),
    ("setAffinity", ANY, 57),
    ("syscall", ANY, 58),
    // **The rest of `std/atomics`' builtin family** (#1443). The other six atomics were already here
    // under nim's `atomic*` spelling; these four are what `std/atomics` itself calls, and they were
    // the only thing left once the `cpuRelax` `{.emit.}` stopped failing the link. Same
    // single-vCPU-guest posture as their neighbours (§3d): with one vCPU an atomic is just the
    // load/modify/store, and a fence has nothing to order against.
    //
    // `testAndSet`/`clear` operate on C's `bool` flag object — one byte, hence `load8_u`/`store8`,
    // *not* the word-width the other atomics use.
    ("builtinTestAndSet", ANY, 59),
    ("builtinClear", ANY, 60),
    ("builtinThreadFence", ANY, 61),
    ("builtinSignalFence", ANY, 62), // **The generic `builtin*` atomics** (#1499). Unlike every row above, these are matched on
    // `(name, signature)`: `std/atomics` declares them `[T: SomeInteger]`, so nimony monomorphizes
    // one `importc` instance per width with an opaque instance hash (`builtinLoadN.0.I99n2w21.`)
    // and the name alone cannot say which width an instance is. Binding by name is not merely
    // imprecise, it is wrong — it silently gives an `i32` atomic the `i64` shim, which is how the
    // probe recorded in #1499 produced `TypeMismatch { expected: I64, found: I32 }`.
    //
    // Only the two widths nimony actually instantiates are here. A `[T]` at `i8`/`i16` would ride
    // `i32` slots and present this same `i32` signature, so the machine type cannot distinguish it
    // and the `i32` shim's 4-byte access would over-read a 1-byte cell — but nimony does not
    // instantiate those (the byte-wide atomics are `testAndSet`/`clear`, which are non-generic and
    // served at rows 59/60 with `load8_u`/`store8`). An instance whose signature matches no row
    // here stays **unbound** and fails the link loudly, which is the point: a missing width is a
    // visible gap, never a silent wrong-width bind.
    //
    // The 64-bit `load`/`store`/`cmpxchg`/`exchange` reuse the `atomic*` funcs above — same
    // operation, same signature, so a second copy would be a second thing to keep in step. The
    // `fetch_*` pair cannot: `__atomic_fetch_add` returns the value **before** the add, while
    // nim's `atomicAddFetch` (row 11) returns the value after, so rows 69/70 are their own funcs.
    ("builtinLoadN", sig(&[I64, I32], &[I32]), 63),
    ("builtinLoadN", sig(&[I64, I32], &[I64]), 7),
    ("builtinStoreN", sig(&[I64, I32, I32], &[]), 64),
    ("builtinStoreN", sig(&[I64, I64, I32], &[]), 8),
    (
        "builtinCompareExchangeN",
        sig(&[I64, I64, I32, I32, I32, I32], &[I32]),
        65,
    ),
    (
        "builtinCompareExchangeN",
        sig(&[I64, I64, I64, I32, I32, I32], &[I32]),
        9,
    ),
    ("builtinExchangeN", sig(&[I64, I32, I32], &[I32]), 66),
    ("builtinExchangeN", sig(&[I64, I64, I32], &[I64]), 10),
    ("builtinFetchAdd", sig(&[I64, I32, I32], &[I32]), 67),
    ("builtinFetchSub", sig(&[I64, I32, I32], &[I32]), 68),
    ("builtinFetchAdd", sig(&[I64, I64, I32], &[I64]), 69),
    ("builtinFetchSub", sig(&[I64, I64, I32], &[I64]), 70),
];

/// The C symbols the **prebuilt guest libc** ([`nim_libc_units`]) serves for a nim program — the
/// bottom-edge leaves whose real implementation is far too large to hand-write as Temen text the way
/// [`POWERBOX_COMPUTE_SHIM`] does: C's `snprintf` (nim's `formatBiggestFloat` needs `%#.*g`/`%#.*e`/
/// `%#.*f` with correct rounding), `strtod`, and the libm transcendentals. They come from the same
/// guest-C libc the chibicc playground card uses, compiled once into a linkable unit.
///
/// Matched on the leaf's **`importc` C name**, not its nim symbol: nim's `float32`/`float64`
/// overloads share one name (`sin`) and differ only by their `importc` (`sinf` / `sin`).
///
/// **ABI note.** Every chibicc-compiled function takes a leading **`$sp`** (its frame pointer), so a
/// nim call site must pass one. [`link_selected_with_extra`] therefore seeds these leaves into the
/// whole-program frame set, which makes `call_import` prepend `sp + frame_size` exactly as it does
/// for a frame-needing nim proc — and makes every *caller* frame-needing through the same fixpoint,
/// so it has an `$sp` to hand down. C varargs are already clang-wasm-style (one pointer to a slot
/// buffer), which is precisely how leng marshals a `{.varargs.}` import, so `snprintf` binds with no
/// shim. This table is the single authority: it decides both what is frame-marked and what is bound,
/// so the two can never disagree (a name the libc turns out not to export is simply left unbound).
const LIBC_SERVED: &[&str] = &[
    "snprintf",
    "strtod",
    // **libm.** This list must match what `browser/playground-include/__pg_math_impl.h` actually
    // defines — a name missing here is not a compile error, it is a leaf left unbound at link, and the
    // program traps at *run* with nothing to say. That is how `$sqrt(4.0)` shipped broken: the first
    // version of this table held only the trigonometric family, because the test written alongside it
    // happened to call only those. Anything added to the guest libc belongs here in the same commit.
    //
    // Powers, roots, logs, rounding — the half of `std/math` a program actually reaches for.
    "sqrt",
    "exp",
    "pow",
    "log",
    "log10",
    "log2",
    "cbrt",
    "hypot",
    "fmod",
    "floor",
    "ceil",
    "round",
    "trunc",
    "fabs",
    "fmax",
    "fmin",
    "copysign",
    "frexp",
    "ldexp",
    "modf",
    // Trigonometric and hyperbolic, float64.
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    // The float32 (`…f`) overloads nim declares alongside each of the above.
    "sqrtf",
    "expf",
    "powf",
    "logf",
    "log10f",
    "log2f",
    "cbrtf",
    "hypotf",
    "fmodf",
    "floorf",
    "ceilf",
    "roundf",
    "truncf",
    "fabsf",
    "copysignf",
    "atan2f",
    "sinf",
    "cosf",
    "tanf",
    "asinf",
    "acosf",
    "atanf",
    "sinhf",
    "coshf",
    "tanhf",
    "asinhf",
    "acoshf",
    "atanhf",
];

/// True if the prebuilt guest libc serves the bottom-edge leaf whose `importc` C name is `c_name`.
fn libc_serves(c_name: &str) -> bool {
    LIBC_SERVED.contains(&c_name)
}

/// The **guest-libc leaves** a set of nim units imports, as `(import symbol, C name)` — the `importc`
/// procs whose C name is in [`LIBC_SERVED`]. Drives both the frame seeding and [`nim_libc_units`].
///
/// Each leaf is yielded under **both spellings the linker may see**: the bare leng symbol
/// (`c_snprintf.0.`, how a module calls a leaf it declares itself) and the stem-suffixed global
/// (`sin.2.mat7cnfv21`, how a *sibling* module calls a leaf `std/math` declares). A cross-module call
/// resolves the callee to its global name, so binding only the bare form would leave every libm leaf
/// an unbound manifest import.
pub fn libc_leaves_of(units: &[WholeModule]) -> Result<Vec<(String, String)>, LengError> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut push = |name: String, c: &str| {
        if !out.iter().any(|(s, _)| *s == name) {
            out.push((name, c.to_string()));
        }
    };
    for u in units {
        let root = nif::parse(u.src).map_err(LengError::Parse)?;
        for (sym, c) in translate::Translator::importc_procs(&root)? {
            if libc_serves(&c) {
                push(format!("{sym}{}", u.stem), &c);
                push(sym, &c);
            }
        }
    }
    out.sort();
    Ok(out)
}

/// Stubs for the guest libc's **non-`write` host caps**. The prebuilt libc is one translation unit,
/// so linking it for `snprintf`/`strtod` also pulls in its file/heap layer (`fopen`, `malloc`, …) —
/// dead for a nim program, which has its own allocator and I/O, but their capability imports would
/// still have to be bound at instantiation or the run refuses to start. Resolving them here to
/// fail-closed stubs keeps the nim program's manifest to the single `write` STREAM cap it actually
/// uses (pg_libc's `write` has the same `(buf, len) -> n` shape, so it unifies with the syscall
/// adapter's). A nim program never reaches these; if one ever did, it sees a clean failure (`-1` /
/// EOF / a null mapping), never a silent wrong answer. Func order is fixed — see
/// [`LIBC_CAP_STUB_NAMES`].
const LIBC_CAP_STUBS: &str = "\
import 0 \"write\" (i64, i64) -> (i64)

func (i64, i64, i64, i64, i64) -> (i64) { block 0 (v0: i64, v1: i64, v2: i64, v3: i64, v4: i64) { v5 = i64.const -1 return v5 } }
func (i64, i64) -> (i64) { block 0 (v0: i64, v1: i64) { v2 = i64.const 0 return v2 } }
func (i32) -> () { block 0 (v0: i32) { return } }
func (i64, i64, i32) -> (i64) { block 0 (v0: i64, v1: i64, v2: i32) { v3 = i64.const 0 return v3 } }
func () -> (i64) { block 0 () { v0 = i64.const 65536 return v0 } }
func (i64, i64) -> (i64) { block 0 (v0: i64, v1: i64) { v2 = call.import 0 (v0, v1) return v2 } }";

/// The cap names [`LIBC_CAP_STUBS`] serves, in its func order.
///
/// **Keyed by the guest libc's actual import names, so it moves when that edge is renamed.** The libc
/// used to reach stdin/stdout through the frontend's fd-less `write`/`read` builtins; once `<unistd.h>`
/// gained real fd-dispatching definitions those became `__vm_stream_write`/`__vm_stream_read`, i.e.
/// `stream_write`/`stream_read`. A name here that the libc no longer imports is not an error — the
/// export simply goes unused, the libc's real import stays unbound, and it surfaces in the *program's*
/// capability manifest. `libc_cap_edge_is_fully_served` pins that, because the nim end-to-end test
/// that caught it needs the real nimony toolchain and skips without one.
///
/// `stream_write` is the one that is **aliased rather than stubbed**: it resolves to a body that
/// tail-calls the powerbox `write` cap, the same import [`SYSCALL_ADAPTER`] carries, so the two
/// coalesce into the single manifest import the program already had. Stubbing it would have kept the
/// manifest just as narrow while silently swallowing anything the libc ever writes.
const LIBC_CAP_STUB_NAMES: &[&str] = &[
    "vm_fs",
    "stream_read",
    "exit",
    "vm_map",
    "vm_page_size",
    "stream_write",
];

/// Build the **prebuilt guest-libc link units** for a nim program: the libc itself (its functions
/// exported under the *nim* leaf symbols that import them, so the linker resolves them directly) plus
/// [`LIBC_CAP_STUBS`]. `libc` is an encoded `.temeno` — the committed unit `genlibc` builds by
/// compiling the playground C headers with the committed `chibicc.temen`. A leaf the libc turns out
/// not to export is skipped: it stays an unbound manifest import rather than failing the link.
pub fn nim_libc_units(
    libc: &[u8],
    units: &[WholeModule],
) -> Result<Vec<temen_ir::LinkUnit>, LengError> {
    let module = temen_encode::decode_unit(libc)
        .map_err(|e| LengError::Malformed(format!("decode guest libc unit: {e:?}")))?;
    let mut exports: Vec<(String, u32)> = Vec::new();
    for (sym, c) in libc_leaves_of(units)? {
        if let Some(e) = module.exports.iter().find(|e| e.name == c) {
            exports.push((sym, e.func));
        }
    }
    let libc_unit = temen_ir::LinkUnit {
        module,
        exports,
        ..Default::default()
    };
    let stubs = temen_text::parse_module(LIBC_CAP_STUBS)
        .map_err(|e| LengError::Malformed(format!("libc cap stubs parse: {e:?}")))?;
    let stub_unit = temen_ir::LinkUnit {
        module: stubs,
        exports: LIBC_CAP_STUB_NAMES
            .iter()
            .enumerate()
            .map(|(i, n)| ((*n).to_string(), i as u32))
            .collect(),
        ..Default::default()
    };
    Ok(vec![libc_unit, stub_unit])
}

/// The **syscall adapter** unit. nimony spells its bottom-edge syscalls `sysWrite(fd, buf, len)` (etc.,
/// the C `write` ABI), but the §3e powerbox's STREAM `write` cap is `(buf, len) -> n` — no `fd`. This
/// unit reconciles the two: `nimWrite` (func 0) drops `fd` and tail-calls the powerbox `write` (its
/// one import), so the cap sees the shape it expects; the file-op syscalls `read`/`close`/`open`/
/// `lseek` (funcs 1–4) — imported by `syncio` but never reached by a stdout-only program — resolve to
/// harmless stubs (EOF / success / -1). The powerbox binds the retained `write` STREAM cap at run, so
/// the program writes to the real host stdout. Func order is fixed; [`link_nim_powerbox`] maps the
/// retained `sysWrite`/`sysRead`/`sysClose`/`sysOpen`/`sysLseek` names onto it.
const SYSCALL_ADAPTER: &str = "\
import 0 \"write\" (i64, i64) -> (i64)

func (i32, i64, i64) -> (i64) { block 0 (v0: i32, v1: i64, v2: i64) { v3 = call.import 0 (v1, v2) return v3 } }
func (i32, i64, i64) -> (i64) { block 0 (v0: i32, v1: i64, v2: i64) { v3 = i64.const 0 return v3 } }
func (i32) -> (i32) { block 0 (v0: i32) { v1 = i32.const 0 return v1 } }
func (i64, i32, i64) -> (i32) { block 0 (v0: i64, v1: i32, v2: i64) { v3 = i32.const -1 return v3 } }
func (i32, i64, i32) -> (i64) { block 0 (v0: i32, v1: i64, v2: i32) { v3 = i64.const -1 return v3 } }";

/// The compute-shim func index for a bottom-edge leaf import `name`, or `None` for a name the shim
/// doesn't serve (the true syscalls — those go to the adapter / powerbox).
/// The shim func serving a leaf, or `None` to leave it unbound.
///
/// `want` is the import's own signature. A row that pins a signature ([`ComputeLeaf`]) is eligible
/// only when it matches exactly; an [`ANY`] row ignores it. Among eligible rows the longest name
/// prefix wins, and a signature-pinned row outranks an `ANY` row of the same length so a generic
/// family can never fall back to a name-only bind.
///
/// No eligible row means **unbound** — the leaf survives to fail the link by name. That is the
/// fail-closed half of #1499: binding a generic atomic instance to a shim of the wrong width would
/// produce a module that verifies only by accident, or reads past the end of the cell it was handed.
fn compute_leaf_index(name: &str, want: Option<(&[ValType], &[ValType])>) -> Option<u32> {
    COMPUTE_LEAVES
        .iter()
        .filter(|(p, s, _)| {
            name.starts_with(p)
                && match (s, want) {
                    (None, _) => true,
                    (Some((wp, wr)), Some((gp, gr))) => *wp == gp && *wr == gr,
                    // A pinned row with no signature to check against must not bind.
                    (Some(_), None) => false,
                }
        })
        .max_by_key(|(p, s, _)| (p.len(), s.is_some()))
        .map(|(_, _, i)| *i)
}

/// An import's `(params, results)`, resolved through the module's type section — the `want` side of
/// [`compute_leaf_index`]. `None` when the import isn't a flat func (an interface import) or its
/// type index doesn't resolve, which leaves any signature-pinned leaf unbound.
fn import_sig<'a>(m: &'a Module, imp: &temen_ir::Import) -> Option<(&'a [ValType], &'a [ValType])> {
    let temen_ir::ImportShape::Func(t) = imp.shape else {
        return None;
    };
    match m.types.get(t as usize)? {
        temen_ir::TypeEntry::Func(f) => Some((&f.params, &f.results)),
        _ => None,
    }
}

/// Link whole nimony modules into a **powerbox-runnable module** — the shape `temen_run::run_powerbox`
/// (and the browser's `temen_run_onramp`) executes with the host granting stdout, so a real Nim I/O
/// program *prints for real* with no custom personality.
///
/// [`link_whole_with_runtime`] binds *every* bottom-edge leaf to a pure-IR shim (a self-contained
/// module for the offline runner). This instead splits the bottom edge the way the powerbox wants it:
/// the **pure-compute leaves** (memcpy/atomics/bswap/bump-`mmap`/…) resolve to [`POWERBOX_COMPUTE_SHIM`],
/// while the **syscalls** route through [`SYSCALL_ADAPTER`], which reconciles nimony's
/// `sysWrite(fd,buf,len)` with the §3e powerbox STREAM `write(buf,len)` cap and leaves that cap a
/// bound-at-run manifest import (`write`). The merged module is a powerbox entry (`_start` at func 0).
///
/// Two link passes: the first (compute shim only) surfaces the retained syscall imports — their nim
/// names (`sysWrite.0.` …) aren't known until link — then the adapter is bound onto them and the whole
/// thing re-linked. Re-verify the result like any linked output (the caller runs `run_powerbox`, which
/// verifies).
pub fn link_nim_powerbox(units: &[WholeModule], libc: Option<&[u8]>) -> Result<Module, LengError> {
    // #1051/#1054: link the `system` unit **first**, as belt-and-suspenders. The real fix for #1051
    // is the heap seed in [`synth_start_unit`] — without it the guest heap arena started at 0 and
    // overlapped placed static data, so a heap allocation could reuse a program's `LongString` const
    // and `add`/realloc would scribble its length (`write` then dumps stray bytes). That corruption
    // surfaced only for some link orders (whichever placed a referenced const where an allocation
    // landed), which looked like `temen_ir::link` was order-sensitive — it is not; its address
    // arithmetic is order-independent (#1054). With the heap seed the output is correct for **every**
    // order; this reorder is kept as defense in depth (and to pin a deterministic layout). The
    // `_start` entry is func 0 via the synthesized start unit, not unit position, so reordering is
    // safe. Stable-partition so `sysv…` units come first and the rest keep their given order.
    let mut reordered: Vec<WholeModule> = units
        .iter()
        .map(|u| WholeModule {
            stem: u.stem,
            src: u.src,
        })
        .collect();
    reordered.sort_by_key(|u| !u.stem.starts_with("sysv"));
    let units: &[WholeModule] = &reordered;
    let mut runtime = nim_powerbox_runtime(units)?;
    // The **prebuilt guest libc** ([`LIBC_SERVED`]), when the caller supplies it: `snprintf`/`strtod`/
    // libm, which no hand-written shim could reasonably carry. Without it those leaves stay unbound
    // manifest imports and a program that formats a float (or calls `sin`) cannot run.
    if let Some(libc) = libc {
        runtime.extend(nim_libc_units(libc, units)?);
    }
    // Pass 2: link with the compute shim + the adapter. Only the powerbox `write` cap is left.
    link_whole_powerbox_manifest(units, runtime)
}

/// Build the **nim→powerbox runtime link units** ([compute shim, syscall adapter]) that
/// [`link_nim_powerbox`] links a program against — split out so a caller (or the #1054
/// order-independence test) can link whole units against the same runtime via
/// [`link_whole_powerbox_manifest`] directly. The result is independent of the units' order (it
/// depends only on which bottom-edge leaves and raw syscalls the program references), so one build
/// serves every permutation.
pub fn nim_powerbox_runtime(units: &[WholeModule]) -> Result<Vec<temen_ir::LinkUnit>, LengError> {
    // The compute shim must know which leaf names to export; discover them from the `system` unit's
    // own compiled imports (every pure-compute leaf originates there — a self-contained module that
    // compiles standalone, unlike a program unit that references a sibling's aggregate type).
    let sys = units
        .iter()
        .find(|u| u.stem.starts_with("sysv"))
        .ok_or_else(|| LengError::Malformed("no `system` unit (stem `sysv…`) to link".into()))?;
    let sys_obj = temen_encode::decode_unit(&compile_whole_object(sys)?)
        .map_err(|e| LengError::Malformed(format!("decode system object: {e:?}")))?;
    let mut compute_exports: Vec<(String, u32)> = Vec::new();
    for imp in &sys_obj.imports {
        if let Some(i) = compute_leaf_index(&imp.name, import_sig(&sys_obj, imp)) {
            if compute_exports.iter().all(|(n, _)| n != &imp.name) {
                compute_exports.push((imp.name.clone(), i));
            }
        }
    }
    let compute_unit = |exports: Vec<(String, u32)>| -> Result<temen_ir::LinkUnit, LengError> {
        let module = temen_text::parse_module(POWERBOX_COMPUTE_SHIM)
            .map_err(|e| LengError::Malformed(format!("compute shim parse: {e:?}")))?;
        Ok(temen_ir::LinkUnit {
            module,
            exports,
            ..Default::default()
        })
    };

    // Pass 1: link with only the compute shim, so the true syscalls survive as retained imports.
    let m1 = link_whole_powerbox_manifest(units, vec![compute_unit(compute_exports.clone())?])?;

    // Widen the compute set with any leaf the shim serves that survived pass 1. The `system` scan
    // above finds every leaf that module declares, but a leaf declared by *another* stdlib module
    // (`std/posix`'s `clock_gettime`, reached through `std/times`) only shows up once the whole
    // program is linked. Both passes feed one export list, so the final compute unit serves both.
    for imp in &m1.imports {
        if let Some(i) = compute_leaf_index(&imp.name, import_sig(&m1, imp)) {
            if compute_exports.iter().all(|(n, _)| n != &imp.name) {
                compute_exports.push((imp.name.clone(), i));
            }
        }
    }

    // Map each retained syscall onto the adapter's fixed func order.
    let mut adapter_exports: Vec<(String, u32)> = Vec::new();
    for imp in &m1.imports {
        let n = &imp.name;
        let f = if n.starts_with("sysWrite") {
            0
        } else if n.starts_with("sysRead") {
            1
        } else if n.starts_with("sysClose") {
            2
        } else if n.starts_with("sysOpen") {
            3
        } else if n.starts_with("sysLseek") {
            4
        } else {
            continue; // anything else stays a manifest import the powerbox binds (write) or rejects
        };
        adapter_exports.push((n.clone(), f));
    }
    let adapter = temen_ir::LinkUnit {
        module: temen_text::parse_module(SYSCALL_ADAPTER)
            .map_err(|e| LengError::Malformed(format!("syscall adapter parse: {e:?}")))?,
        exports: adapter_exports,
        ..Default::default()
    };

    Ok(vec![compute_unit(compute_exports)?, adapter])
}

/// **Link several nimony modules in Tier-2 TLS mode** (NIM.md §3d) together with a runtime that
/// establishes each thread's TLS base. Like [`link_units`], but `tvar`s lower to per-vCPU TLS-block
/// accesses over one **shared** block layout (so a thread-var defined in one unit — e.g. the
/// allocator state in `system` — and referenced from another agree on its offset). `runtime` supplies
/// the pre-built link units that `vcpu.tls.set` a real block base at thread entry (the analog of the
/// W3 allocator shim); without such a unit the linked module has no valid TLS base to run against.
pub fn link_units_tls_with_runtime(
    units: &[LengModule],
    runtime: Vec<temen_ir::LinkUnit>,
) -> Result<Module, LengError> {
    let sel: Vec<(&str, &str, Select)> = units
        .iter()
        .map(|u| (u.stem, u.src, Select::Names(u.names)))
        .collect();
    link_selected_with_extra(&sel, runtime, true, false, false)
}

/// The **pure-IR runtime for the C bottom edge** (NIM.md §3b, W3 — issue #761). The compiled Nim
/// `system` module bottoms out at ~15 `{.importc.}`/magic leaves; the *compute* ones (no OS authority
/// needed) lower to plain Temen ops, so they bind as ordinary linked Temen functions — **not** host
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
/// the Nim stdlib's `{.importc.}` surface (NIM.md §3b, W3 — #761) as ordinary Temen functions, ready to
/// [`temen_ir::link`] against a translated module whose matching imports are bound via
/// [`bottom_edge_index`]. Panics only on an internal temen-text regression (the const is a fixed literal).
pub fn bottom_edge_runtime() -> Module {
    temen_text::parse_module(BOTTOM_EDGE_RUNTIME).expect("bottom-edge runtime temen-text parses")
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

/// A translated Temen value: its SSA id and type. The unit the expression translator threads.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Val {
    pub id: u32,
    pub ty: ValType,
}
