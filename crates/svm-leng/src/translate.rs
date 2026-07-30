//! Leng-NIF → SVM-text translation (NIM.md Phase 2).
//!
//! Emits SVM text (chibicc's `codegen_ir.c` model), let `svm_text::parse_module` build the IR.
//! Anything outside the supported subset is a fail-closed [`LengError::Unsupported`] — never a
//! silent mistranslation.
//!
//! **Locals as block parameters (the chibicc/on-ramp φ model).** Every non-address-taken local
//! (params + `var`s) is a *slot*, threaded as a parameter of **every** block. Within a block a slot
//! holds a current SSA value; a branch passes the current values as block args, so a control-flow
//! merge is just the successor's block parameter — no separate φ/dominance analysis. Value numbers
//! reset per block (svm-text convention): a block's params are `v0..v(nslots-1)`, instructions
//! continue from `nslots`.
//!
//! **Address-taken locals live in a window frame.** A local whose address is taken (`(addr x)`) is
//! demoted from an SSA slot to a byte offset in a per-call data-stack frame; the proc gains a
//! leading `$sp` stack-pointer param (slot 0) and reads/writes it via `load`/`store` at `sp+off`,
//! and a call to a frame-needing proc passes `sp + frame_size` as the callee's frame.
//!
//! **Aggregates via `lvalue_addr` (the `gen_addr` model).** Named `(type … (object …))`/`(array …)`
//! layouts are registered up front; `dot`/`at`/`pat`/`deref` and a frame/aggregate symbol all
//! resolve to `(address, type descriptor)`, then a scalar leaf loads/stores. Object params are
//! passed by address; aggregate `var`s are frame-resident. Whole-aggregate copy/constructors and
//! C-ABI field offsets remain later refinements.

use std::cell::RefCell;
use std::collections::HashMap;

use svm_ir::ValType;

use crate::{LengError, Node, Val};

/// A cross-module callee lowered to an SVM `import` — its declared signature and slot. The runtime
/// binds it by name at instantiation (like `write`), exactly as the embedder binds the powerbox.
struct ImportDecl {
    name: String,
    params: Vec<ValType>,
    ret: Option<ValType>,
}

/// Imports discovered while emitting a module. Interior-mutable because they're found during
/// (immutable-`&self`) proc emission and materialized into the module header afterward.
#[derive(Default)]
struct ImportTable {
    decls: Vec<ImportDecl>, // slot = index
    slot_of: HashMap<String, u32>,
}

/// Escape raw bytes as an svm-text string body: every byte as `\xHH` (round-trips through the
/// lexer's `\xHH` escape; simple and unambiguous for the short scalar-initializer payloads here).
fn escape_bytes(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 4);
    for b in bytes {
        s.push_str(&format!("\\x{b:02x}"));
    }
    s
}

/// The C name in a `(pragmas … (exportc "name") …)` node, if present. `pragmas` is the optional
/// `(pragmas …)` at a proc/gvar's pragma slot (or `.`/absent — then `None`).
fn exportc_name(pragmas: Option<&Node>) -> Option<String> {
    let p = pragmas?;
    if p.tag() != Some("pragmas") {
        return None;
    }
    for prag in p.args() {
        if prag.tag() == Some("exportc") {
            if let Some(name) = prag.args().first().and_then(|n| n.as_atom()) {
                return Some(name.trim_matches('"').to_string());
            }
        }
    }
    None
}

/// True if a node is a NIF string literal (a quote-delimited atom, e.g. a `LongString`'s `data`).
fn is_string_literal(node: &Node) -> bool {
    matches!(node.as_atom(), Some(a) if a.starts_with('"') && a.ends_with('"') && a.len() >= 2)
}

/// Decode a NIF string-literal atom (quotes included) to its raw bytes. NIF escapes a byte as `\HH`
/// (two hex digits); a `\` before any other byte is that literal byte (`\\`, `\"`). Everything else
/// is copied verbatim. (The tokenizer keeps the escapes intact in the atom text.)
fn nif_str_bytes(atom: &str) -> Result<Vec<u8>, LengError> {
    let s = atom
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .ok_or_else(|| LengError::Malformed("expected a string literal".into()))?;
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'\\'
            && i + 2 < b.len()
            && b[i + 1].is_ascii_hexdigit()
            && b[i + 2].is_ascii_hexdigit()
        {
            let hi = (b[i + 1] as char).to_digit(16).unwrap();
            let lo = (b[i + 2] as char).to_digit(16).unwrap();
            out.push((hi * 16 + lo) as u8);
            i += 3;
        } else if b[i] == b'\\' && i + 1 < b.len() {
            out.push(b[i + 1]);
            i += 2;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    Ok(out)
}

/// Escape a symbol for an svm-text string literal (e.g. an `import` name). Printable ASCII passes
/// through; `\` and `"` are backslash-escaped; anything else is `\xHH`. Necessary because mangled
/// Leng names carry NIF hex escapes — the `[]` operator is literally `\5B\5D…`, whose bare backslash
/// the lexer would reject as an unknown escape.
fn escape_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        match b {
            b'\\' => out.push_str("\\\\"),
            b'"' => out.push_str("\\\""),
            0x20..=0x7e => out.push(b as char),
            _ => out.push_str(&format!("\\x{b:02x}")),
        }
    }
    out
}

/// The svm-text prefix for a value type (`i32`/`i64`). This subset only produces integer types.
fn prefix(ty: ValType) -> &'static str {
    match ty {
        ValType::I32 => "i32",
        ValType::I64 => "i64",
        ValType::F32 => "f32",
        ValType::F64 => "f64",
        _ => "i64",
    }
}

/// A proc's SVM-visible signature, collected so calls can resolve names → indices.
struct Sig {
    index: u32,
    params: Vec<ValType>,
    ret: Option<ValType>, // None = void
    /// True if the proc takes the address of a local, so its emitted signature has a leading
    /// stack-pointer param and callers must pass a fresh frame.
    needs_frame: bool,
    /// `Some(desc)` if the proc returns an aggregate by value: it returns `void` and instead takes a
    /// hidden `$sret` pointer param (after `$sp`, before the Leng params) into which `ret` copies the
    /// result. Callers pass the destination's address as that pointer.
    sret: Option<TyDesc>,
}

/// What a computed lvalue address points at — a scalar (with load/store width) or a named
/// aggregate (whose fields/elements are reached by further `dot`/`at`).
#[derive(Clone, Debug)]
pub(crate) enum TyDesc {
    /// A full-width scalar: `i32`/`i64`/`f32`/`f64`, loaded/stored with the plain `iN.load`/`store`.
    Scalar(ValType),
    /// A **sub-word integer** — 1 or 2 bytes (`u8`/`i8`/`u16`/`i16`, e.g. `char`) — loaded and
    /// stored with `iN.load8/16`/`store8/16`. Its SSA value type is `i32` (the load zero-/sign-
    /// extends per `signed`). nimony reads these through pointer casts (`=destroy` checks a string's
    /// low length byte via `(deref (cast (ptr (u 8)) …))`).
    Narrow {
        bytes: u8,
        signed: bool,
    },
    /// A **typed pointer** — value type `i64`, but carrying what it points at, so a `deref` of a
    /// *computed* pointer (a pointer-valued field, `(deref (dot s more))`) recovers the pointee's
    /// layout. nimony's ARC ops walk `string.more → LongString.rc` this way.
    Ptr(Box<TyDesc>),
    Agg(String),
}

impl TyDesc {
    /// The machine type of a value held in one SSA slot / scalar memory cell (a pointer is `i64`);
    /// `None` for a `Narrow` char cell or an aggregate (handled on their own paths).
    fn scalar_ty(&self) -> Option<ValType> {
        match self {
            TyDesc::Scalar(t) => Some(*t),
            TyDesc::Ptr(_) => Some(ValType::I64),
            _ => None,
        }
    }
}

/// The in-memory layout of a named aggregate type (`(type :Name … Body)`).
#[derive(Clone, Debug)]
pub(crate) enum Layout {
    /// `(object … (fld :f … T)*)` — fields packed at natural size (consistent within svm-leng;
    /// C-ABI/SysV offsets are a later refinement for host interop).
    Object {
        fields: Vec<(String, u64, TyDesc)>, // (name, byte offset, type)
        size: u64,
    },
    /// `(array Elem Count)` — `Count` elements of `Elem`.
    Array {
        elem: TyDesc,
        elem_size: u64,
        size: u64,
    },
}

pub(crate) struct Translator {
    procs: HashMap<String, Sig>,
    types: HashMap<String, Layout>,
    /// Names of locally-declared `object`/`array` aggregate types. A bare-symbol type is treated as
    /// an aggregate (held by address, indexed by `dot`/`at`) iff it is in this set; every other named
    /// type (`enum`, `distinct` int, `proctype`, opaque/external) is an integer scalar.
    agg_names: std::collections::HashSet<String>,
    /// Mutable module globals (`gvar`) → (fixed window offset, type). Zero-initialized (the window
    /// starts zeroed); non-zero initializers are a later slice.
    globals: HashMap<String, (u64, TyDesc)>,
    /// Scalar integer `const`s, inlined at use.
    consts: HashMap<String, i64>,
    /// Non-zero scalar global initializers → `(unit-local offset, little-endian bytes)`, folded into
    /// the globals `data` segment. Zero-init globals stay zero (the segment is zero-filled).
    data_inits: Vec<(u64, Vec<u8>)>,
    /// End of the globals region (unit-local). The globals occupy `[16, globals_top)`.
    globals_top: u64,
    /// **Link-unit mode.** When true, the module is emitted as a relocatable link unit (`.svmo`
    /// shape): a global's address is a `data.self <off>` the linker rewrites, and one zero-filled
    /// `data` segment covers `[16, globals_top)` so the linker reserves and relocates the whole
    /// region. When false (the default, runnable `.svmb` shape), globals sit at fixed absolute
    /// window offsets (`i64.const`) and only non-zero initializers emit `data` — so the module runs
    /// directly, without a link step.
    link_mode: bool,
    /// Cross-module callees lowered to SVM imports (discovered during emission).
    imports: RefCell<ImportTable>,
}

impl Translator {
    pub fn new() -> Self {
        Translator {
            procs: HashMap::new(),
            types: HashMap::new(),
            agg_names: std::collections::HashSet::new(),
            data_inits: Vec::new(),
            globals_top: 16,
            link_mode: false,
            globals: HashMap::new(),
            consts: HashMap::new(),
            imports: RefCell::new(ImportTable::default()),
        }
    }

    /// A translator that emits a **relocatable link unit** (`.svmo` shape) instead of a directly
    /// runnable module — globals via `data.self`, so several units link into one (NIM.md W2).
    pub fn new_for_link() -> Self {
        Translator {
            link_mode: true,
            ..Translator::new()
        }
    }

    /// Register (or reuse) an import slot for a cross-module callee. First use fixes its signature;
    /// a later call with a different arg count fail-closes (rather than emit a mismatched
    /// `call.import` the verifier would reject).
    fn register_import(
        &self,
        name: &str,
        argtys: &[ValType],
        ret: Option<ValType>,
    ) -> Result<u32, LengError> {
        let mut t = self.imports.borrow_mut();
        if let Some(&slot) = t.slot_of.get(name) {
            if t.decls[slot as usize].params.len() != argtys.len() {
                return Err(LengError::Unsupported(format!(
                    "import `{name}` called with inconsistent arity"
                )));
            }
            return Ok(slot);
        }
        let slot = t.decls.len() as u32;
        t.decls.push(ImportDecl {
            name: name.to_string(),
            params: argtys.to_vec(),
            ret,
        });
        t.slot_of.insert(name.to_string(), slot);
        Ok(slot)
    }

    /// Assemble the final module text: `memory` (if used) + `import` declarations + the globals
    /// `data` segment (relocatable; addressed via `data.self`) + funcs. The emitted module is a
    /// **link unit** whenever it has globals — its `data.self`/imports resolve at `link` time (a
    /// single-unit link makes a globals-only module runnable).
    fn assemble(&self, used_memory: bool, funcs: String) -> String {
        let has_globals = self.globals_top > 16;
        let mut out = String::new();
        // Data segments (globals) and loads/stores write the window, so declare `memory`.
        if used_memory || has_globals {
            out.push_str("memory 16\n\n");
        }
        let t = self.imports.borrow();
        for (slot, d) in t.decls.iter().enumerate() {
            let ps = d
                .params
                .iter()
                .map(|t| prefix(*t))
                .collect::<Vec<_>>()
                .join(", ");
            let rs = d.ret.map(prefix).unwrap_or("");
            out.push_str(&format!(
                "import {slot} \"{}\" ({ps}) -> ({rs})\n",
                escape_str(&d.name)
            ));
        }
        if !t.decls.is_empty() {
            out.push('\n');
        }
        if self.link_mode && has_globals {
            // Link unit: one `data` segment over the whole globals region `[16, globals_top)` —
            // zero-filled, non-zero initializers patched in — referenced by `data.self`, so the
            // linker relocates every global when it places this unit's data.
            let mut image = vec![0u8; self.globals_top as usize];
            for (off, bytes) in &self.data_inits {
                let o = *off as usize;
                image[o..o + bytes.len()].copy_from_slice(bytes);
            }
            out.push_str(&format!("data 16 \"{}\"\n\n", escape_bytes(&image[16..])));
        } else {
            // Runnable module: globals are fixed absolute offsets in the zeroed window, so only
            // non-zero initializers need a `data` segment.
            for (off, bytes) in &self.data_inits {
                out.push_str(&format!("data {off} \"{}\"\n", escape_bytes(bytes)));
            }
            if !self.data_inits.is_empty() {
                out.push('\n');
            }
        }
        out.push_str(&funcs);
        out
    }

    /// Collect `gvar`/`const` top-levels: globals get fixed window offsets (below the stack the
    /// caller passes in); scalar int consts are recorded for inlining.
    fn collect_globals(&mut self, root: &Node) -> Result<(), LengError> {
        let mut off = 16u64; // leave the low bytes as a null sentinel region
        for item in root.args() {
            match item.tag() {
                Some("gvar" | "tvar") => {
                    let a = item.args();
                    if a.len() < 3 {
                        return Err(LengError::Malformed("gvar needs :name pragmas type".into()));
                    }
                    let name = sym_def(&a[0])?;
                    let desc = self.tydesc(&a[2])?;
                    // A non-zero scalar initializer becomes a `data` segment at the global's offset
                    // (the window is otherwise zero).
                    if let Some(init) = a.get(3) {
                        if !init.is_empty_marker() && init.tag() != Some("nil") {
                            if let Some(0) = int_literal(init) {
                                // zero — the window is already zero-filled.
                            } else if let (Some(v), TyDesc::Scalar(t)) = (int_literal(init), &desc)
                            {
                                let w = match t {
                                    ValType::I32 | ValType::F32 => 4,
                                    _ => 8,
                                };
                                self.data_inits
                                    .push((off, (v as u64).to_le_bytes()[..w].to_vec()));
                            } else if init.as_atom().is_some() {
                                // A **symbol** initializer — a proc pointer (`gExitFlush =
                                // nimNoopFlush`) or another global's address. Its pointer value is
                                // opaque in this model: we don't materialize usable proc addresses,
                                // and an indirect call through it fail-closes. So reserve the scalar
                                // slot zero-initialized; nimony's `ini`/setup writes the real pointer
                                // before any use. (Relocating such an initializer to the referenced
                                // symbol is a later refinement.)
                            } else {
                                return Err(LengError::Unsupported(format!(
                                    "non-scalar-int global initializer for `{name}`"
                                )));
                            }
                        }
                    }
                    let sz = self.sizeof(&desc);
                    self.globals.insert(name, (off, desc));
                    off += sz.max(8);
                }
                Some("const") => {
                    let a = item.args();
                    if a.len() >= 4 {
                        let name = sym_def(&a[0])?;
                        if let Some(v) = int_literal(&a[3]) {
                            self.consts.insert(name, v);
                        } else if let Some((bytes, desc)) = self.const_aggregate_bytes(&a[3])? {
                            // A constant aggregate with data (a `LongString` string-literal blob):
                            // materialize its exact bytes into a data segment and register it as an
                            // addressable global, so `(addr strlit…)` — the `string` value's `more`
                            // pointer — resolves to it.
                            let sz = bytes.len() as u64;
                            self.data_inits.push((off, bytes));
                            self.globals.insert(name, (off, desc));
                            off += sz.max(8);
                        } else {
                            // A non-scalar const we can't materialize — e.g. an `Rtti` vtable
                            // (`Type.vt`) or its RTTI internals. Reserve an addressable, opaque
                            // placeholder global so `(addr Type.vt)` yields a valid window address.
                            // Its bytes are never read by translatable code (an object stores the
                            // vtable pointer, but only *dynamic dispatch* reads through it, and that
                            // fail-closes), so a zeroed fixed-size placeholder is sound.
                            const VT_PLACEHOLDER: u64 = 64;
                            self.globals
                                .insert(name, (off, TyDesc::Scalar(ValType::I64)));
                            off += VT_PLACEHOLDER;
                        }
                    }
                }
                _ => {}
            }
        }
        self.globals_top = off;
        Ok(())
    }

    /// This unit's globals as **cross-module data exports** (NIM.md W2): each `gvar`'s global
    /// (stem-suffixed) name → its unit-local data offset, the counterpart of a proc export. Another
    /// unit's `data.sym "<name>"` binds here. A local `gvar` name ends in `.`, so `name + stem`
    /// yields the same `<name>.<stem>` a referencing module emits.
    pub fn global_exports(&self, stem: &str) -> Vec<svm_ir::DataExport> {
        let mut out: Vec<svm_ir::DataExport> = self
            .globals
            .iter()
            .map(|(name, (off, _))| svm_ir::DataExport {
                name: format!("{name}{stem}"),
                offset: *off,
            })
            .collect();
        out.sort_by(|a, b| a.name.cmp(&b.name)); // deterministic order
        out
    }

    /// The module's **`exportc` symbols** as exports under their *C* names — the conventional entry
    /// points a host or `svm-run` binds to (the C `main`, `cmdCount`, `cmdLine`, `nimEnviron`), in
    /// addition to the mangled Leng names. A proc `(proc … (pragmas (exportc "cname") …) …)` exports
    /// as a func under `cname`; a `gvar` with `(exportc "cname")` as a data symbol. Call after
    /// translation (uses the populated proc/global tables). Path A: makes a whole-module object's
    /// C-ABI surface findable (`svm-run --link` can enter at `main`).
    pub fn exportc_exports(
        &self,
        root: &Node,
    ) -> Result<(Vec<svm_ir::Export>, Vec<svm_ir::DataExport>), LengError> {
        let mut procs = Vec::new();
        let mut data = Vec::new();
        for item in root.args() {
            match item.tag() {
                Some("proc") => {
                    // `(proc :name params ret pragmas body)` — pragmas at index 3.
                    if let Some(cname) = exportc_name(item.args().get(3)) {
                        let local = sym_def(&item.args()[0])?;
                        if let Some(sig) = self.procs.get(&local) {
                            procs.push(svm_ir::Export {
                                name: cname,
                                func: sig.index,
                            });
                        }
                    }
                }
                Some("gvar" | "tvar") => {
                    // `(gvar :name pragmas type init)` — pragmas at index 1.
                    if let Some(cname) = exportc_name(item.args().get(1)) {
                        let local = sym_def(&item.args()[0])?;
                        if let Some((off, _)) = self.globals.get(&local) {
                            data.push(svm_ir::DataExport {
                                name: cname,
                                offset: *off,
                            });
                        }
                    }
                }
                _ => {}
            }
        }
        Ok((procs, data))
    }

    /// Materialize a **constant aggregate** value — `(oconstr AggType (kv field val)*)` — into its
    /// little-endian window bytes, given the aggregate's (already resolved) layout. Scalar-int fields
    /// write width-appropriate bytes at their offset; a **string literal** (a `LongString`'s `data`
    /// flexible tail) writes its raw bytes there, growing the blob past the fixed size. Returns
    /// `None` for anything outside that shape (a non-`oconstr`, an unresolved type, an unsupported
    /// field value) so the caller falls back to an opaque placeholder. This is how a **long string
    /// literal** lowers: nimony emits it as a `const` `LongString` blob that the `string` value's
    /// `more` field points at (`(addr strlit…)`), NIM.md W2.
    fn const_aggregate_bytes(&self, val: &Node) -> Result<Option<(Vec<u8>, TyDesc)>, LengError> {
        if val.tag() != Some("oconstr") {
            return Ok(None);
        }
        let a = val.args();
        let tyname = match a.first().and_then(|n| n.as_atom()) {
            Some(n) => n,
            None => return Ok(None),
        };
        let (fields, base_size) = match self.types.get(tyname) {
            Some(Layout::Object { fields, size }) => (fields.clone(), *size),
            _ => return Ok(None),
        };
        let mut bytes = vec![0u8; base_size as usize];
        for kv in &a[1..] {
            if kv.tag() != Some("kv") {
                continue;
            }
            let ka = kv.args();
            if ka.len() < 2 {
                continue;
            }
            let fname = ka[0]
                .as_atom()
                .ok_or_else(|| LengError::Malformed("const kv needs a field name".into()))?;
            let (_, off, fdesc) = fields.iter().find(|(n, _, _)| n == fname).ok_or_else(|| {
                LengError::Unsupported(format!("const field `{fname}` not in `{tyname}`"))
            })?;
            let off = *off as usize;
            if let Some(v) = int_literal(&ka[1]) {
                let w = match fdesc {
                    TyDesc::Scalar(ValType::I32 | ValType::F32) => 4,
                    _ => 8,
                };
                if bytes.len() < off + w {
                    bytes.resize(off + w, 0);
                }
                bytes[off..off + w].copy_from_slice(&(v as u64).to_le_bytes()[..w]);
            } else if is_string_literal(&ka[1]) {
                let data = nif_str_bytes(ka[1].as_atom().unwrap())?;
                if bytes.len() < off + data.len() {
                    bytes.resize(off + data.len(), 0);
                }
                bytes[off..off + data.len()].copy_from_slice(&data);
            } else {
                return Ok(None); // an unsupported const field value — fall back to a placeholder
            }
        }
        Ok(Some((bytes, TyDesc::Agg(tyname.to_string()))))
    }

    /// Byte size of a type descriptor.
    fn sizeof(&self, d: &TyDesc) -> u64 {
        match d {
            TyDesc::Scalar(ValType::I32 | ValType::F32) => 4,
            TyDesc::Scalar(_) => 8,
            TyDesc::Ptr(_) => 8,
            TyDesc::Narrow { bytes, .. } => *bytes as u64,
            TyDesc::Agg(name) => match self.types.get(name) {
                Some(Layout::Object { size, .. }) | Some(Layout::Array { size, .. }) => *size,
                None => 8,
            },
        }
    }

    /// The type descriptor of a Leng type node: a named symbol is an aggregate; `ptr`/`aptr` are
    /// scalar `i64` (the pointer itself); otherwise an integer scalar.
    fn tydesc(&self, node: &Node) -> Result<TyDesc, LengError> {
        match node.tag() {
            // A **typed pointer**: value type `i64`, carrying the pointee for `deref`. A pointer to
            // an opaque/void pointee (`(ptr (void))`, `(ptr)`) stays a bare `i64` scalar.
            Some("ptr") | Some("aptr") => match node.args().first() {
                Some(t) if t.tag() != Some("void") => Ok(TyDesc::Ptr(Box::new(self.tydesc(t)?))),
                _ => Ok(TyDesc::Scalar(ValType::I64)),
            },
            Some("f") => Ok(TyDesc::Scalar(float_ty(node)?)),
            // A sub-word integer (`u8`/`i8`/`u16`/`i16`, `char`) is a `Narrow` scalar — loaded and
            // stored at its true width, not widened to a 4/8-byte access.
            Some("i" | "u" | "c") => {
                let (vt, signed) = int_ty_signed(node)?;
                match vt {
                    ValType::I32 => {
                        let bytes = int_bits(
                            node.args()
                                .first()
                                .and_then(|n| n.as_atom())
                                .unwrap_or("+32"),
                        )?;
                        if bytes <= 8 {
                            Ok(TyDesc::Narrow { bytes: 1, signed })
                        } else if bytes <= 16 {
                            Ok(TyDesc::Narrow { bytes: 2, signed })
                        } else {
                            Ok(TyDesc::Scalar(ValType::I32))
                        }
                    }
                    _ => Ok(TyDesc::Scalar(vt)),
                }
            }
            Some(_) => Ok(TyDesc::Scalar(int_ty(node)?)),
            None => match node.as_atom() {
                // A bare-symbol type is an aggregate only if it's a declared object/array; any other
                // named type (enum, distinct int, proctype, opaque external) is an integer scalar.
                Some(name) if self.agg_names.contains(name) => Ok(TyDesc::Agg(name.to_string())),
                Some(_) => Ok(TyDesc::Scalar(ValType::I64)),
                None => Err(LengError::Malformed("expected a type".into())),
            },
        }
    }

    /// Collect all `(type :Name … Body)` top-level declarations into the layout registry, resolving
    /// forward references (nimony emits array types *after* their use).
    fn collect_types(&mut self, root: &Node) -> Result<(), LengError> {
        let mut raw: HashMap<String, &Node> = HashMap::new();
        for item in root.args() {
            if item.tag() == Some("type") {
                let a = item.args();
                if a.len() >= 3 {
                    raw.insert(sym_def(&a[0])?, &a[2]);
                }
            }
        }
        // Record which declared types are aggregates (object/array) *before* resolving, so
        // `tydesc` classifies every other named type (enum/distinct/…) as a scalar as it resolves
        // fields. Extend (don't overwrite): external types registered by `import_types` stay.
        self.agg_names.extend(
            raw.iter()
                .filter(|(_, body)| matches!(body.tag(), Some("object") | Some("array")))
                .map(|(n, _)| n.clone()),
        );
        let names: Vec<String> = raw.keys().cloned().collect();
        for name in names {
            self.resolve_type(&name, &raw)?;
        }
        Ok(())
    }

    fn resolve_type(&mut self, name: &str, raw: &HashMap<String, &Node>) -> Result<(), LengError> {
        if self.types.contains_key(name) {
            return Ok(());
        }
        let body = match raw.get(name) {
            Some(b) => *b,
            None => return Ok(()), // an external/opaque type; leave unresolved (sizeof falls back)
        };
        let layout = match body.tag() {
            Some("object") => {
                // `(object [Empty|Base] (fld :f pragmas Type)*)` — packed at natural size.
                let mut fields = Vec::new();
                let mut off = 0u64;
                // The first son is the base: `.` = no base; a symbol = inheritance. An inheritable
                // object carries a leading vtable/type-header pointer (the positional slot an
                // `(oconstr T <vtable> …)` fills), then the base's fields, then its own — so the base
                // layout is inlined at the front.
                if let Some(base) = body.args().first().and_then(|n| n.as_atom()) {
                    if base != "." {
                        self.resolve_type(base, raw)?;
                        match self.types.get(base).cloned() {
                            Some(Layout::Object {
                                fields: bf,
                                size: bsize,
                            }) => {
                                fields.extend(bf);
                                off = bsize;
                            }
                            // External inheritable root (`RootObj`/`Exception`): a single 8-byte
                            // vtable/type-header pointer at offset 0.
                            _ => {
                                fields.push((
                                    "$vtable".to_string(),
                                    0,
                                    TyDesc::Scalar(ValType::I64),
                                ));
                                off = 8;
                            }
                        }
                    }
                }
                for fld in body.args() {
                    if fld.tag() != Some("fld") {
                        continue; // skip the base/Empty slot
                    }
                    let fa = fld.args();
                    if fa.len() < 3 {
                        return Err(LengError::Malformed("fld needs :name pragmas type".into()));
                    }
                    let fname = sym_def(&fa[0])?;
                    // A flexible-array tail (`uarray`/`flexarray` — an `UncheckedArray`, e.g.
                    // `LongString.data`): unsized inline data. It occupies no *fixed* size (the
                    // object's size stops here); a `const` materializes its bytes past the fixed
                    // part. Record it at the current offset as an opaque pointer-width slot and stop
                    // advancing.
                    if matches!(fa[2].tag(), Some("uarray") | Some("flexarray")) {
                        fields.push((fname, off, TyDesc::Scalar(ValType::I64)));
                        continue;
                    }
                    let fdesc = self.tydesc(&fa[2])?;
                    if let TyDesc::Agg(n) = &fdesc {
                        self.resolve_type(n, raw)?;
                    }
                    let fsize = self.sizeof(&fdesc);
                    fields.push((fname, off, fdesc));
                    off += fsize;
                }
                Layout::Object { fields, size: off }
            }
            Some("array") => {
                // `(array Elem Count)`.
                let a = body.args();
                if a.len() < 2 {
                    return Err(LengError::Malformed("array needs elem and count".into()));
                }
                let elem = self.tydesc(&a[0])?;
                if let TyDesc::Agg(n) = &elem {
                    self.resolve_type(n, raw)?;
                }
                let elem_size = self.sizeof(&elem);
                let count = a[1]
                    .as_atom()
                    .and_then(|s| s.parse::<u64>().ok())
                    .ok_or_else(|| LengError::Unsupported("non-constant array length".into()))?;
                Layout::Array {
                    elem,
                    elem_size,
                    size: elem_size * count,
                }
            }
            // proctype / enum / other named types aren't aggregates we index into — skip.
            _ => return Ok(()),
        };
        self.types.insert(name.to_string(), layout);
        Ok(())
    }

    /// Pre-register **external** aggregate type layouts (another module's types, under the
    /// stem-suffixed global names this module references them by — [`export_types`]). Must run
    /// before `collect_types`, whose `resolve_type` skips already-registered names.
    pub fn import_types(&mut self, ext: &[(String, Layout)]) {
        for (name, layout) in ext {
            self.agg_names.insert(name.clone());
            self.types.insert(name.clone(), layout.clone());
        }
    }

    /// Collect a module's aggregate type layouts under their **stem-suffixed global names** — the
    /// form *other* modules reference them by (a type `T.0.` defined in module `stem` is
    /// `T.0.<stem>` elsewhere, the same mangling as procs/gvars). Field/element types that name a
    /// sibling type from the same module are rewritten to their suffixed form too, so nested
    /// aggregates resolve in the importing translator. This is the cross-module *type* half of
    /// linking (NIM.md W2): layouts are baked at translate time, unlike proc/data symbols, which
    /// resolve at link time — so `link_units` pools these across its units before translating any.
    pub fn export_types(root: &Node, stem: &str) -> Result<Vec<(String, Layout)>, LengError> {
        let mut t = Translator::new();
        t.collect_types(root)?;
        let local: std::collections::HashSet<&String> = t.types.keys().collect();
        let suffix = |n: &String| {
            if local.contains(n) {
                format!("{n}{stem}")
            } else {
                n.clone()
            }
        };
        let rewrite = |d: &TyDesc| match d {
            TyDesc::Agg(n) => TyDesc::Agg(suffix(n)),
            s => s.clone(),
        };
        let mut out: Vec<(String, Layout)> = t
            .types
            .iter()
            .map(|(name, layout)| {
                let l = match layout {
                    Layout::Object { fields, size } => Layout::Object {
                        fields: fields
                            .iter()
                            .map(|(f, off, d)| (f.clone(), *off, rewrite(d)))
                            .collect(),
                        size: *size,
                    },
                    Layout::Array {
                        elem,
                        elem_size,
                        size,
                    } => Layout::Array {
                        elem: rewrite(elem),
                        elem_size: *elem_size,
                        size: *size,
                    },
                };
                (format!("{name}{stem}"), l)
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0)); // HashMap order → deterministic output
        Ok(out)
    }

    /// Translate a `(stmts TopLevelConstruct*)` module to SVM text (all procs).
    pub fn module(&mut self, root: &Node) -> Result<String, LengError> {
        Ok(self.module_with_names(root)?.0)
    }

    /// Translate an **entire** module (every proc, in source order) to SVM text, also returning each
    /// proc's **local name** in func-index order — the whole-module counterpart of [`some_procs`],
    /// for compiling a real nimony module (all its scaffolding: `ini`, C `main`, the exportc gvars)
    /// as a link object. The names feed the object's export table so *other* modules' cross-module
    /// calls (`callee.<stem>`) resolve against it (NIM.md W2, Path A).
    pub fn module_with_names(&mut self, root: &Node) -> Result<(String, Vec<String>), LengError> {
        if root.tag() != Some("stmts") {
            return Err(LengError::Malformed(format!(
                "module root must be (stmts …), got {:?}",
                root.tag()
            )));
        }
        self.collect_types(root)?;
        self.collect_globals(root)?;
        let mut proc_nodes = Vec::new();
        let mut names = Vec::new();
        for item in root.args() {
            match item.tag() {
                Some("proc") => {
                    let (name, params, ret0) = self.proc_sig(item)?;
                    let sret = self.ret_sret(&item.args()[2])?;
                    let ret = if sret.is_some() { None } else { ret0 };
                    let index = proc_nodes.len() as u32;
                    let needs_frame = proc_needs_frame(item);
                    self.procs.insert(
                        name.clone(),
                        Sig {
                            index,
                            params,
                            ret,
                            needs_frame,
                            sret,
                        },
                    );
                    names.push(name);
                    proc_nodes.push(item);
                }
                Some("type" | "gvar" | "tvar" | "const") => {} // handled by collect_types/globals
                Some(other) => {
                    return Err(LengError::Unsupported(format!(
                        "top-level construct `{other}` (proc/type/gvar/const supported)"
                    )))
                }
                None => {
                    if !item.is_empty_marker() {
                        return Err(LengError::Malformed("headless top-level node".into()));
                    }
                }
            }
        }
        let mut out = String::new();
        let mut used_memory = false;
        for p in &proc_nodes {
            used_memory |= self.proc_body(p, &mut out)?;
        }
        Ok((self.assemble(used_memory, out), names))
    }

    /// Translate a **single named proc** as func 0 (NIM.md Phase 2 "go deep"): the real hexer output
    /// carries `gvar`/`type`/other procs the skeleton can't lower yet, so we locate just the target.
    /// Only its own signature is registered, so a call to any other proc fail-closes.
    pub fn one_proc(&mut self, root: &Node, name: &str) -> Result<String, LengError> {
        if root.tag() != Some("stmts") {
            return Err(LengError::Malformed("module root must be (stmts …)".into()));
        }
        self.collect_types(root)?;
        self.collect_globals(root)?;
        for item in root.args() {
            if item.tag() != Some("proc") {
                continue;
            }
            let pname = item
                .args()
                .first()
                .and_then(|n| n.as_atom())
                .map(|a| a.strip_prefix(':').unwrap_or(a).to_string());
            if pname.as_deref() != Some(name) {
                continue;
            }
            let (n, params, ret0) = self.proc_sig(item)?;
            let sret = self.ret_sret(&item.args()[2])?;
            let ret = if sret.is_some() { None } else { ret0 };
            let needs_frame = proc_needs_frame(item);
            self.procs.insert(
                n,
                Sig {
                    index: 0,
                    params,
                    ret,
                    needs_frame,
                    sret,
                },
            );
            let mut out = String::new();
            let used_memory = self.proc_body(item, &mut out)?;
            return Ok(self.assemble(used_memory, out));
        }
        Err(LengError::Malformed(format!(
            "proc `{name}` not found in module"
        )))
    }

    /// Translate a **named subset** of a module's procs together (in the order `names` lists them),
    /// so intra-subset calls resolve by index — the multi-proc generalization of [`one_proc`], for
    /// running a real caller→callee pair (e.g. `mkSum` calling the sret `mk`) lifted out of a module
    /// whose `ini`/`main`/`gvar` top-levels the skeleton can't lower wholesale. A call to any proc
    /// outside the subset (and not a defined import) fail-closes as usual.
    pub fn some_procs(&mut self, root: &Node, names: &[&str]) -> Result<String, LengError> {
        if root.tag() != Some("stmts") {
            return Err(LengError::Malformed("module root must be (stmts …)".into()));
        }
        self.collect_types(root)?;
        self.collect_globals(root)?;
        // Register the requested procs first (indices in `names` order), so calls between them bind.
        let mut selected: Vec<&Node> = Vec::new();
        for (index, want) in names.iter().enumerate() {
            let node = root.args().iter().find(|item| {
                item.tag() == Some("proc")
                    && item
                        .args()
                        .first()
                        .and_then(|n| n.as_atom())
                        .map(|a| a.strip_prefix(':').unwrap_or(a))
                        == Some(want)
            });
            let node =
                node.ok_or_else(|| LengError::Malformed(format!("proc `{want}` not found")))?;
            let (n, params, ret0) = self.proc_sig(node)?;
            let sret = self.ret_sret(&node.args()[2])?;
            let ret = if sret.is_some() { None } else { ret0 };
            self.procs.insert(
                n,
                Sig {
                    index: index as u32,
                    params,
                    ret,
                    needs_frame: proc_needs_frame(node),
                    sret,
                },
            );
            selected.push(node);
        }
        let mut out = String::new();
        let mut used_memory = false;
        for node in &selected {
            used_memory |= self.proc_body(node, &mut out)?;
        }
        Ok(self.assemble(used_memory, out))
    }

    /// Extract `(proc :Sym Params RetType Pragmas [Body])` → (name, param types, ret).
    fn proc_sig(&self, p: &Node) -> Result<(String, Vec<ValType>, Option<ValType>), LengError> {
        let a = p.args();
        if a.len() < 4 {
            return Err(LengError::Malformed(
                "proc needs :name params ret pragmas".into(),
            ));
        }
        let name = sym_def(&a[0])?;
        let params = self.params(&a[1])?;
        let ret = self.ret_ty(&a[2])?;
        Ok((name, params.into_iter().map(|(_, t)| t).collect(), ret))
    }

    /// `Params ::= Empty | (params (param :Sym Pragmas Type)*)` → ordered (name, ValType).
    fn params(&self, node: &Node) -> Result<Vec<(String, ValType)>, LengError> {
        if node.is_empty_marker() {
            return Ok(vec![]);
        }
        if node.tag() != Some("params") {
            return Err(LengError::Malformed("expected (params …) or `.`".into()));
        }
        let mut out = Vec::new();
        for prm in node.args() {
            if prm.tag() != Some("param") {
                return Err(LengError::Malformed("expected (param …)".into()));
            }
            let pa = prm.args();
            if pa.len() < 3 {
                return Err(LengError::Malformed(
                    "param needs :name pragmas type".into(),
                ));
            }
            out.push((sym_def(&pa[0])?, val_ty(&pa[2])?));
        }
        Ok(out)
    }

    /// Param name + its (borrowed) type node, for descriptor computation.
    fn param_types<'n>(&self, node: &'n Node) -> Result<Vec<(String, &'n Node)>, LengError> {
        if node.is_empty_marker() {
            return Ok(vec![]);
        }
        if node.tag() != Some("params") {
            return Err(LengError::Malformed("expected (params …) or `.`".into()));
        }
        let mut out = Vec::new();
        for prm in node.args() {
            if prm.tag() != Some("param") {
                return Err(LengError::Malformed("expected (param …)".into()));
            }
            let pa = prm.args();
            if pa.len() < 3 {
                return Err(LengError::Malformed(
                    "param needs :name pragmas type".into(),
                ));
            }
            out.push((sym_def(&pa[0])?, &pa[2]));
        }
        Ok(out)
    }

    /// If `node` is `(ptr T)`/`(aptr T)`, the descriptor `T` points at. A `void`/opaque pointee has
    /// no typed target (a `deref` of it fail-closes), so it yields `None`.
    fn pointee_desc(&self, node: &Node) -> Result<Option<TyDesc>, LengError> {
        match node.tag() {
            Some("ptr") | Some("aptr") => match node.args().first() {
                Some(t) if t.tag() != Some("void") => Ok(Some(self.tydesc(t)?)),
                _ => Ok(None),
            },
            _ => Ok(None),
        }
    }

    /// Collect every `(var :name … Type …)` in a body → (name, descriptor), and record pointer vars'
    /// pointees.
    fn collect_var_descs(
        &self,
        node: &Node,
        out: &mut Vec<(String, TyDesc)>,
        pointee: &mut HashMap<String, TyDesc>,
    ) -> Result<(), LengError> {
        if let Node::List(_) = node {
            if node.tag() == Some("var") {
                let a = node.args();
                if a.len() >= 3 {
                    let name = sym_def(&a[0])?;
                    let d = self.tydesc(&a[2])?;
                    if !out.iter().any(|(n, _)| n == &name) {
                        out.push((name.clone(), d));
                    }
                    if let Some(pt) = self.pointee_desc(&a[2])? {
                        pointee.insert(name, pt);
                    }
                }
            }
            for c in node.args() {
                self.collect_var_descs(c, out, pointee)?;
            }
        }
        Ok(())
    }

    /// Return type: `(void)`/`.` → None, otherwise the scalar value type (int, float, or pointer).
    fn ret_ty(&self, node: &Node) -> Result<Option<ValType>, LengError> {
        if node.tag() == Some("void") || node.is_empty_marker() {
            return Ok(None);
        }
        Ok(Some(val_ty(node)?))
    }

    /// `Some(desc)` if the return type is a named aggregate (object/array) — returned by sret rather
    /// than as a scalar. A named non-aggregate type (enum/distinct-int/proctype) is still scalar.
    fn ret_sret(&self, node: &Node) -> Result<Option<TyDesc>, LengError> {
        if node.tag() == Some("void") || node.is_empty_marker() {
            return Ok(None);
        }
        let d = self.tydesc(node)?;
        if let TyDesc::Agg(n) = &d {
            if self.types.contains_key(n) {
                return Ok(Some(d));
            }
        }
        Ok(None)
    }

    /// Emit one proc's `func {…}` into `out`; returns whether it touched the window (so the caller
    /// can emit the module `memory` declaration).
    fn proc_body(&self, p: &Node, out: &mut String) -> Result<bool, LengError> {
        let a = p.args();
        let params = self.params(&a[1])?;
        // An aggregate return is by sret: the proc returns `void` and gets a hidden `$sret` pointer.
        let sret_desc = self.ret_sret(&a[2])?;
        let ret = if sret_desc.is_some() {
            None
        } else {
            self.ret_ty(&a[2])?
        };
        let body = a.get(4);

        // Type descriptor of every local (aggregate params are by-address), and the pointee of every
        // pointer local — for `lvalue_addr` / `deref`.
        let mut local_desc: HashMap<String, TyDesc> = HashMap::new();
        let mut pointee: HashMap<String, TyDesc> = HashMap::new();
        for (name, tnode) in self.param_types(&a[1])? {
            local_desc.insert(name.clone(), self.tydesc(tnode)?);
            if let Some(pt) = self.pointee_desc(tnode)? {
                pointee.insert(name, pt);
            }
        }
        // Vars, with their type descriptors.
        let mut var_descs: Vec<(String, TyDesc)> = Vec::new();
        if let Some(b) = body {
            self.collect_var_descs(b, &mut var_descs, &mut pointee)?;
        }

        // Which locals are address-taken (`(addr x)`).
        let mut addr_taken = std::collections::HashSet::new();
        if let Some(b) = body {
            collect_addr_taken(b, &mut addr_taken);
        }
        // An address-taken **aggregate** param is already passed by-address — its slot value *is*
        // the address, so `(addr p)` works with no spill. A scalar param has no address (it lives in
        // an SSA slot); taking its address still fail-closes (scalar spill is a later refinement).
        for (pn, _) in &params {
            if addr_taken.contains(pn) && !matches!(local_desc.get(pn), Some(TyDesc::Agg(_))) {
                return Err(LengError::Unsupported(format!(
                    "address of scalar parameter `{pn}` (param spill is a later refinement)"
                )));
            }
        }

        // A var is frame-resident if it is an aggregate (indexed by `at`/`dot`) or address-taken;
        // the rest are SSA slots. Frame locals get natural-size byte offsets.
        let mut mem: HashMap<String, (u64, TyDesc)> = HashMap::new();
        let mut frame_size = 0u64;
        let mut ssa_vars: Vec<(String, ValType)> = Vec::new();
        for (vn, desc) in &var_descs {
            local_desc.insert(vn.clone(), desc.clone());
            let framed = matches!(desc, TyDesc::Agg(_)) || addr_taken.contains(vn);
            if framed {
                if !mem.contains_key(vn) {
                    let sz = self.sizeof(desc);
                    mem.insert(vn.clone(), (frame_size, desc.clone()));
                    frame_size += sz;
                }
            } else if let Some(vt) = desc.scalar_ty() {
                if !ssa_vars.iter().any(|(n, _)| n == vn) {
                    ssa_vars.push((vn.clone(), vt));
                }
            }
        }
        let needs_frame = !mem.is_empty();
        let has_sret = sret_desc.is_some();

        // SSA slots (block-parameter set): [$sp] ++ [$sret] ++ params ++ non-framed scalar vars.
        // `$sret` (the aggregate-return destination pointer) sits right after `$sp` so `$sp` stays
        // slot 0 and callers thread the frame the same way.
        let mut slots: Vec<(String, ValType)> = Vec::new();
        if needs_frame {
            slots.push(("$sp".into(), ValType::I64));
        }
        if has_sret {
            slots.push(("$sret".into(), ValType::I64));
        }
        slots.extend(params.clone());
        slots.extend(ssa_vars);
        let nparams = usize::from(needs_frame) + usize::from(has_sret) + params.len();
        let sret = sret_desc.map(|d| (usize::from(needs_frame), d));

        // Signature: `i64` stack pointer (if frame-needing), `i64` sret pointer (if agg-returning),
        // then the Leng params. The return type is `void` when sret.
        let mut ptys: Vec<String> = Vec::new();
        if needs_frame {
            ptys.push("i64".into());
        }
        if has_sret {
            ptys.push("i64".into());
        }
        ptys.extend(params.iter().map(|(_, t)| prefix(*t).to_string()));
        let rty = ret.map(|t| prefix(t).to_string()).unwrap_or_default();
        out.push_str(&format!("func ({}) -> ({}) {{\n", ptys.join(", "), rty));

        let mut f = FuncGen::new(
            self,
            ret,
            nparams,
            slots,
            pointee,
            local_desc,
            needs_frame,
            mem,
            frame_size,
            sret,
        );
        // Entry block: params default to their block-param value (slot i = v i); a var not yet
        // assigned reads 0 until its declaration binds it (matches Leng default-init semantics).
        f.open_entry();
        // Pre-allocate a block id for every `(lab :L)` so forward `(jmp L)` can target it.
        if let Some(b) = body {
            f.declare_labels(b);
        }
        match body {
            Some(b) if !matches!(b, Node::Atom(_)) => {
                f.stmt_list(b)?;
            }
            _ => {}
        }
        f.close_fallthrough()?;
        let used_memory = f.used_memory;
        for blk in f.blocks {
            out.push_str(&blk);
        }
        out.push_str("}\n");
        Ok(used_memory)
    }
}

/// Per-function emission over multiple blocks. Slots (params + vars) are the block-parameter set;
/// `cur` maps slot → current SSA value id in the block under construction.
struct FuncGen<'a> {
    t: &'a Translator,
    ret: Option<ValType>,
    slots: Vec<(String, ValType)>,
    slot_of: HashMap<String, usize>,
    /// The first `nparams` slots are the function parameters; the entry block (id 0) must carry
    /// **only** those (the ABI: entry params == function params). Var slots follow and are carried
    /// as parameters of every *successor* block only.
    nparams: usize,
    /// Pointer-typed locals → the type descriptor they point at (`deref`/`at`/`pat` element).
    pointee: HashMap<String, TyDesc>,
    /// Every local → its type descriptor (aggregate params are by-address; used by `lvalue_addr`).
    local_desc: HashMap<String, TyDesc>,
    /// This proc takes the address of a local (or frames an aggregate), so slot 0 is `$sp`.
    has_sp: bool,
    /// Frame-resident locals → (byte offset, type). Scalars are load/stored; aggregates are bases
    /// for `at`/`dot`. Address is `sp + off`.
    mem: HashMap<String, (u64, TyDesc)>,
    /// Total frame bytes; a call to a frame-needing proc passes `sp + frame_size` as its frame.
    frame_size: u64,
    /// `Some((slot, desc))` when this proc returns an aggregate by sret: `cur[slot]` is the caller's
    /// destination pointer, into which `ret` copies/constructs the aggregate before a void return.
    sret: Option<(usize, TyDesc)>,
    /// Leng label name (`(lab :L)`) → its pre-allocated block id, so a `(jmp L)` can branch to a
    /// label defined later (forward jumps: `break`, `block`-`break`). All labels are declared before
    /// any body statement is emitted.
    label_block: HashMap<String, u32>,
    /// Stack of enclosing `while` loops as `(header, exit)` block ids, so a bare `(break)` branches to
    /// the innermost exit and `(continue)` to the innermost header (nimony's `for` lowers to
    /// `while (true) { … else break }`).
    loop_stack: Vec<(u32, u32)>,
    /// Set once the function emits a load/store (so the module declares a window).
    used_memory: bool,
    /// Rendered blocks (header + body + terminator), indexed by block id.
    blocks: Vec<String>,
    next_block: u32,
    // Current block being built:
    cur_id: u32,
    cur_buf: String,
    cur_next: u32, // value counter within the block
    cur: Vec<u32>, // slot → current value id
    terminated: bool,
}

impl<'a> FuncGen<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        t: &'a Translator,
        ret: Option<ValType>,
        nparams: usize,
        slots: Vec<(String, ValType)>,
        pointee: HashMap<String, TyDesc>,
        local_desc: HashMap<String, TyDesc>,
        has_sp: bool,
        mem: HashMap<String, (u64, TyDesc)>,
        frame_size: u64,
        sret: Option<(usize, TyDesc)>,
    ) -> Self {
        let slot_of = slots
            .iter()
            .enumerate()
            .map(|(i, (n, _))| (n.clone(), i))
            .collect();
        FuncGen {
            t,
            ret,
            slots,
            slot_of,
            nparams,
            pointee,
            local_desc,
            has_sp,
            mem,
            frame_size,
            sret,
            label_block: HashMap::new(),
            loop_stack: Vec::new(),
            used_memory: false,
            blocks: Vec::new(),
            next_block: 0,
            cur_id: 0,
            cur_buf: String::new(),
            cur_next: 0,
            cur: Vec::new(),
            terminated: false,
        }
    }

    /// The block-parameter header. The entry block carries only the function params; every other
    /// block carries all slots (params + vars) — the merge points where a var's value flows in.
    fn block_params(&self, id: u32) -> String {
        let upto = if id == 0 {
            self.nparams
        } else {
            self.slots.len()
        };
        let ps: Vec<String> = self.slots[..upto]
            .iter()
            .enumerate()
            .map(|(i, (_, t))| format!("v{i}: {}", prefix(*t)))
            .collect();
        format!("({})", ps.join(", "))
    }

    /// Branch-argument list carrying the current slot values: `(vX, vY, …)`.
    fn branch_args(&self) -> String {
        let a: Vec<String> = self.cur.iter().map(|id| format!("v{id}")).collect();
        format!("({})", a.join(", "))
    }

    fn open_entry(&mut self) {
        self.next_block = 1;
        self.cur_id = 0;
        self.cur_buf.clear();
        self.terminated = false;
        // Entry params are v0..v(nparams-1). Var slots get a default `const 0` (Leng default-init),
        // so every slot has a value before any branch passes the full slot set as block args.
        self.cur = (0..self.nparams as u32).collect();
        self.cur_next = self.nparams as u32;
        for i in self.nparams..self.slots.len() {
            let ty = self.slots[i].1;
            let v = self.emit_const(ty, 0);
            self.cur.push(v.id);
        }
    }

    /// Assign a fresh block id to each `(lab :L)` in the body (document order), so a `(jmp L)` to a
    /// forward label resolves. Structured control flow allocates its blocks after these.
    fn declare_labels(&mut self, body: &Node) {
        let mut names = Vec::new();
        collect_labels(body, &mut names);
        for name in names {
            if !self.label_block.contains_key(&name) {
                let id = self.new_block_id();
                self.label_block.insert(name, id);
            }
        }
    }

    fn reset_cur_state(&mut self) {
        // Successor blocks carry every slot as a parameter, so slot i starts as block param v i.
        let n = self.slots.len() as u32;
        self.cur_next = n;
        self.cur = (0..n).collect();
        self.cur_buf.clear();
        self.terminated = false;
    }

    fn new_block_id(&mut self) -> u32 {
        let id = self.next_block;
        self.next_block += 1;
        id
    }

    /// Finish the current block with `terminator` and store it; then begin building block `id`.
    fn finish_block(&mut self, terminator: String, next_id: u32) {
        let header = format!(
            "block {} {} {{\n",
            self.cur_id,
            self.block_params(self.cur_id)
        );
        let mut rendered = header;
        rendered.push_str(&self.cur_buf);
        rendered.push_str(&format!("  {terminator}\n  }}\n"));
        // Store at index cur_id (blocks may be created out of order; pad as needed).
        let idx = self.cur_id as usize;
        if self.blocks.len() <= idx {
            self.blocks.resize(idx + 1, String::new());
        }
        self.blocks[idx] = rendered;
        self.cur_id = next_id;
        self.reset_cur_state();
    }

    /// Terminate the (final) current block by falling off the end: `return` for void, else an error.
    fn close_fallthrough(&mut self) -> Result<(), LengError> {
        if self.terminated {
            // Already terminated (e.g. a trailing `ret`); still must store the block.
            return Ok(());
        }
        match self.ret {
            None => {
                self.finish_block("return".into(), self.next_block);
                Ok(())
            }
            Some(_) => Err(LengError::Malformed(
                "non-void proc falls off the end without `ret`".into(),
            )),
        }
    }

    fn fresh(&mut self) -> u32 {
        let id = self.cur_next;
        self.cur_next += 1;
        id
    }

    fn set_slot(&mut self, name: &str, v: Val) -> Result<(), LengError> {
        let i = *self.slot_of.get(name).ok_or_else(|| {
            LengError::Unsupported(format!("assignment to unknown local `{name}`"))
        })?;
        let want = self.slots[i].1;
        let vid = if v.ty != want {
            self.convert(v, want).id
        } else {
            v.id
        };
        self.cur[i] = vid;
        Ok(())
    }

    fn lookup(&self, name: &str) -> Option<Val> {
        self.slot_of.get(name).map(|&i| Val {
            id: self.cur[i],
            ty: self.slots[i].1,
        })
    }

    /// Read a scalar local by name: a frame scalar emits a `load` at `sp+off`; an SSA slot returns
    /// its current value. Aggregate frame locals return `None` (not a scalar rvalue).
    fn read_local(&mut self, name: &str) -> Option<Val> {
        if let Some((off, desc)) = self.mem.get(name).cloned() {
            if let Some(ty) = desc.scalar_ty() {
                let sp = self.cur[0];
                let id = self.fresh();
                self.used_memory = true;
                self.cur_buf.push_str(&format!(
                    "  v{id} = {}.load v{sp} offset={off}\n",
                    prefix(ty)
                ));
                return Some(Val { id, ty });
            }
        }
        self.lookup(name)
    }

    /// Write a scalar local by name: a frame scalar emits a `store`; an SSA slot rebinds.
    fn write_local(&mut self, name: &str, v: Val) -> Result<(), LengError> {
        if let Some((off, ty)) = self
            .mem
            .get(name)
            .cloned()
            .and_then(|(off, d)| d.scalar_ty().map(|t| (off, t)))
        {
            let val = if v.ty != ty { self.convert(v, ty) } else { v };
            let sp = self.cur[0];
            self.used_memory = true;
            self.cur_buf.push_str(&format!(
                "  {}.store v{sp} v{} offset={off}\n",
                prefix(ty),
                val.id
            ));
            return Ok(());
        }
        self.set_slot(name, v)
    }

    /// The **address** of an lvalue and what it points at. Unifies `deref`/`dot`/`at`/`pat` and a
    /// frame/aggregate symbol into `(addr value id, type descriptor)` — the `codegen_ir.c` `gen_addr`.
    fn lvalue_addr(&mut self, node: &Node) -> Result<(u32, TyDesc), LengError> {
        match node {
            Node::Atom(name) => {
                if let Some((off, desc)) = self.mem.get(name).cloned() {
                    let sp = self.cur[0];
                    return Ok((self.add_const_off(sp, off), desc));
                }
                // A module global's address: a fixed absolute offset in a runnable module, or a
                // relocatable `data.self <off>` in a link unit (the linker rewrites it on placement).
                if let Some((off, desc)) = self.t.globals.get(name).cloned() {
                    let base = if self.t.link_mode {
                        self.emit_data_self(off)
                    } else {
                        self.emit_const(ValType::I64, off as i64).id
                    };
                    return Ok((base, desc));
                }
                // An aggregate passed by address: its slot value *is* the address.
                if let Some(desc @ TyDesc::Agg(_)) = self.local_desc.get(name).cloned() {
                    if let Some(v) = self.lookup(name) {
                        return Ok((v.id, desc));
                    }
                }
                // In a link unit, an atom resolved by none of the above is a **cross-module data
                // symbol** (a `gvar` another unit defines) → a relocatable `data.sym`. Assumed i64
                // scalar; the linker binds it, and an unresolved name is a fail-closed link error.
                // (A runnable module has nothing to bind to, so it stays the error below.)
                if self.t.link_mode {
                    let addr = self.emit_data_sym(name, 0);
                    return Ok((addr, TyDesc::Scalar(ValType::I64)));
                }
                Err(LengError::Unsupported(format!(
                    "`{name}` is not an addressable lvalue"
                )))
            }
            Node::List(_) => match node.tag() {
                Some("deref") => {
                    let operand = node
                        .args()
                        .first()
                        .ok_or_else(|| LengError::Malformed("deref needs an operand".into()))?;
                    // A **symbol pointer**: a local whose pointee type we tracked.
                    if let Some(pname) = operand.as_atom() {
                        let desc = self.pointee.get(pname).cloned().ok_or_else(|| {
                            LengError::Unsupported(format!("`{pname}` is not a known pointer"))
                        })?;
                        let pv = self.lookup(pname).ok_or_else(|| {
                            LengError::Unsupported(format!("unknown pointer `{pname}`"))
                        })?;
                        return Ok((pv.id, desc));
                    }
                    // A **computed pointer** with an explicit pointee type: `(deref (cast (ptr T)
                    // e))` — evaluate `e` as the address, take the pointee `T` from the cast. nimony
                    // emits this to read through a re-typed address (`=destroy` reads a string's low
                    // length byte via `(ptr (u 8))`).
                    if operand.tag() == Some("cast") {
                        let ca = operand.args();
                        if ca.len() >= 2 {
                            if let Some(pointee) = ptr_pointee(&ca[0]) {
                                let desc = self.t.tydesc(pointee)?;
                                let addr = self.expr_typed(&ca[1], ValType::I64)?;
                                return Ok((addr.id, desc));
                            }
                        }
                    }
                    // A **pointer-valued lvalue**: `(deref (dot s more))` — a typed-pointer field.
                    // Load the pointer value at the field's address; its pointee is the address of
                    // the target aggregate. Walks `string.more → LongString` for the ARC ops.
                    let (laddr, ldesc) = self.lvalue_addr(operand)?;
                    if let TyDesc::Ptr(pointee) = ldesc {
                        let pv = self.fresh();
                        self.used_memory = true;
                        self.cur_buf
                            .push_str(&format!("  v{pv} = i64.load v{laddr}\n"));
                        return Ok((pv, *pointee));
                    }
                    Err(LengError::Unsupported(
                        "deref of non-pointer expression".into(),
                    ))
                }
                Some("dot") => {
                    let a = node.args();
                    let (baddr, bdesc) = self.lvalue_addr(&a[0])?;
                    let fname = a
                        .get(1)
                        .and_then(|n| n.as_atom())
                        .ok_or_else(|| LengError::Malformed("dot needs a field name".into()))?;
                    let (foff, fdesc) = self.field_of(&bdesc, fname)?;
                    Ok((self.add_const_off(baddr, foff), fdesc))
                }
                Some("at") => {
                    let a = node.args();
                    let (baddr, bdesc) = self.lvalue_addr(&a[0])?;
                    let (esize, edesc) = self.array_of(&bdesc)?;
                    let idx = self.expr_typed(&a[1], ValType::I64)?;
                    Ok((self.add_scaled(baddr, idx.id, esize), edesc))
                }
                Some("pat") => {
                    let a = node.args();
                    let pname = a.first().and_then(|n| n.as_atom()).ok_or_else(|| {
                        LengError::Unsupported("pat on non-symbol pointer".into())
                    })?;
                    let pdesc = self.pointee.get(pname).cloned().ok_or_else(|| {
                        LengError::Unsupported(format!("`{pname}` is not a known pointer"))
                    })?;
                    let p = self.expr(&a[0])?;
                    let esize = self.t.sizeof(&pdesc);
                    let idx = self.expr_typed(&a[1], ValType::I64)?;
                    Ok((self.add_scaled(p.id, idx.id, esize), pdesc))
                }
                other => Err(LengError::Unsupported(format!(
                    "lvalue `{}`",
                    other.unwrap_or("<headless>")
                ))),
            },
        }
    }

    /// Load the scalar value of an lvalue.
    fn load_lvalue(&mut self, node: &Node) -> Result<Val, LengError> {
        let (addr, desc) = self.lvalue_addr(node)?;
        match desc {
            TyDesc::Scalar(ty) => {
                let id = self.fresh();
                self.used_memory = true;
                self.cur_buf
                    .push_str(&format!("  v{id} = {}.load v{addr}\n", prefix(ty)));
                Ok(Val { id, ty })
            }
            TyDesc::Ptr(_) => {
                let id = self.fresh();
                self.used_memory = true;
                self.cur_buf
                    .push_str(&format!("  v{id} = i64.load v{addr}\n"));
                Ok(Val {
                    id,
                    ty: ValType::I64,
                })
            }
            TyDesc::Narrow { bytes, signed } => {
                // A sub-word load, widened to `i32` (`i32.load8_u` / `load16_s` / …).
                let id = self.fresh();
                self.used_memory = true;
                let ext = if signed { 's' } else { 'u' };
                self.cur_buf
                    .push_str(&format!("  v{id} = i32.load{}_{ext} v{addr}\n", bytes * 8));
                Ok(Val {
                    id,
                    ty: ValType::I32,
                })
            }
            TyDesc::Agg(n) => Err(LengError::Unsupported(format!(
                "reading aggregate `{n}` as a value (whole-aggregate ops are a later slice)"
            ))),
        }
    }

    /// Store `rhs` through an lvalue.
    fn store_lvalue(&mut self, lhs: &Node, rhs: &Node) -> Result<(), LengError> {
        let (addr, desc) = self.lvalue_addr(lhs)?;
        if let TyDesc::Narrow { bytes, .. } = desc {
            // A sub-word store (`i32.store8` / `store16`), truncating the `i32` value.
            let v = self.expr_typed(rhs, ValType::I32)?;
            self.used_memory = true;
            self.cur_buf
                .push_str(&format!("  i32.store{} v{addr} v{}\n", bytes * 8, v.id));
            return Ok(());
        }
        let ty = match desc {
            TyDesc::Scalar(t) => t,
            TyDesc::Ptr(_) => ValType::I64,
            TyDesc::Narrow { .. } => unreachable!("handled above"),
            TyDesc::Agg(n) => {
                return Err(LengError::Unsupported(format!(
                    "assigning to aggregate lvalue `{n}`"
                )))
            }
        };
        let v = self.expr_typed(rhs, ty)?;
        self.used_memory = true;
        self.cur_buf
            .push_str(&format!("  {}.store v{addr} v{}\n", prefix(ty), v.id));
        Ok(())
    }

    /// `base + off` (an unchanged base when `off == 0`).
    fn add_const_off(&mut self, base: u32, off: u64) -> u32 {
        if off == 0 {
            return base;
        }
        let ko = self.emit_const(ValType::I64, off as i64);
        let id = self.fresh();
        self.cur_buf
            .push_str(&format!("  v{id} = i64.add v{base} v{}\n", ko.id));
        id
    }

    /// `base + idx * size` (element/aptr addressing).
    fn add_scaled(&mut self, base: u32, idx: u32, size: u64) -> u32 {
        let ks = self.emit_const(ValType::I64, size as i64);
        let m = self.fresh();
        self.cur_buf
            .push_str(&format!("  v{m} = i64.mul v{idx} v{}\n", ks.id));
        let id = self.fresh();
        self.cur_buf
            .push_str(&format!("  v{id} = i64.add v{base} v{m}\n"));
        id
    }

    /// Field `(offset, type)` in an object descriptor.
    fn field_of(&self, bdesc: &TyDesc, fname: &str) -> Result<(u64, TyDesc), LengError> {
        if let TyDesc::Agg(n) = bdesc {
            if let Some(Layout::Object { fields, .. }) = self.t.types.get(n) {
                for (fnm, off, fd) in fields {
                    if fnm == fname {
                        return Ok((*off, fd.clone()));
                    }
                }
                return Err(LengError::Unsupported(format!(
                    "no field `{fname}` in `{n}`"
                )));
            }
        }
        Err(LengError::Unsupported("`dot` on a non-object".into()))
    }

    /// Element `(size, type)` of an array descriptor.
    fn array_of(&self, bdesc: &TyDesc) -> Result<(u64, TyDesc), LengError> {
        if let TyDesc::Agg(n) = bdesc {
            if let Some(Layout::Array {
                elem, elem_size, ..
            }) = self.t.types.get(n)
            {
                return Ok((*elem_size, elem.clone()));
            }
        }
        Err(LengError::Unsupported("`at` on a non-array".into()))
    }

    /// The type descriptor of an lvalue, computed **without emitting** (for dispatching aggregate
    /// vs scalar assignment). `None` for a plain SSA-scalar local (which has no address).
    fn lvalue_type(&self, node: &Node) -> Option<TyDesc> {
        match node {
            Node::Atom(name) => {
                if let Some((_, d)) = self.mem.get(name) {
                    return Some(d.clone());
                }
                if let Some((_, d)) = self.t.globals.get(name) {
                    return Some(d.clone());
                }
                self.local_desc.get(name).cloned()
            }
            Node::List(_) => match node.tag() {
                Some("deref" | "pat") => node
                    .args()
                    .first()
                    .and_then(|n| n.as_atom())
                    .and_then(|p| self.pointee.get(p).cloned()),
                Some("dot") => {
                    let a = node.args();
                    let bd = self.lvalue_type(a.first()?)?;
                    let f = a.get(1)?.as_atom()?;
                    self.field_of(&bd, f).ok().map(|(_, d)| d)
                }
                Some("at") => {
                    let bd = self.lvalue_type(node.args().first()?)?;
                    self.array_of(&bd).ok().map(|(_, d)| d)
                }
                _ => None,
            },
        }
    }

    /// `(asgn Lvalue Expr)`. An **aggregate** destination copies/constructs whole; a scalar symbol
    /// rebinds its slot; a scalar `deref`/`dot`/`at`/`pat`/global stores through its address.
    fn assign(&mut self, lhs: &Node, rhs: &Node) -> Result<(), LengError> {
        if let Some(ddesc @ TyDesc::Agg(_)) = self.lvalue_type(lhs) {
            let (daddr, _) = self.lvalue_addr(lhs)?;
            return self.assign_aggregate(daddr, &ddesc, rhs);
        }
        if matches!(lhs.tag(), Some("deref" | "dot" | "at" | "pat")) {
            return self.store_lvalue(lhs, rhs);
        }
        let name = lhs.as_atom().ok_or_else(|| {
            LengError::Unsupported(format!("assignment to lvalue `{:?}`", lhs.tag()))
        })?;
        // A global is stored through its window address; a cross-module data symbol (link unit,
        // not a local) stores through its `data.sym` address; a local rebinds/stores its slot.
        if self.t.globals.contains_key(name) || (self.t.link_mode && !self.is_local(name)) {
            return self.store_lvalue(lhs, rhs);
        }
        let v = self.expr(rhs)?;
        self.write_local(name, v)
    }

    /// Assign into an aggregate at address `daddr`: an `oconstr`/`aconstr` constructs field-by-field
    /// in place; any other rhs is a whole-aggregate copy (`mem.copy` of the source's bytes).
    fn assign_aggregate(
        &mut self,
        daddr: u32,
        ddesc: &TyDesc,
        rhs: &Node,
    ) -> Result<(), LengError> {
        match rhs.tag() {
            Some("oconstr") => {
                // `(oconstr T [Vtable] (kv Field Expr)*)`. For an inheritable object the first
                // positional element is the vtable/type-header pointer → the synthetic `$vtable`
                // field at offset 0; the rest are keyed fields.
                for elem in &rhs.args()[1..] {
                    if elem.tag() == Some("kv") {
                        let ka = elem.args();
                        let fname = ka
                            .first()
                            .and_then(|n| n.as_atom())
                            .ok_or_else(|| LengError::Malformed("kv needs a field name".into()))?;
                        let (foff, fdesc) = self.field_of(ddesc, fname)?;
                        let faddr = self.add_const_off(daddr, foff);
                        self.store_member(faddr, &fdesc, &ka[1])?;
                    } else if !elem.is_empty_marker() {
                        // A positional element: the vtable pointer of an inheritable object.
                        let (voff, vdesc) = self.field_of(ddesc, "$vtable")?;
                        let vaddr = self.add_const_off(daddr, voff);
                        self.store_member(vaddr, &vdesc, elem)?;
                    }
                }
                Ok(())
            }
            Some("aconstr") => {
                // `(aconstr T Elem*)`.
                let (esize, edesc) = self.array_of(ddesc)?;
                for (i, ee) in rhs.args()[1..].iter().enumerate() {
                    let eaddr = self.add_const_off(daddr, i as u64 * esize);
                    self.store_member(eaddr, &edesc, ee)?;
                }
                Ok(())
            }
            Some("call") => {
                // An aggregate-returning call: hand `daddr` to the callee as its `$sret` pointer, so
                // it writes the result straight into the destination (no temporary).
                self.call_sret(rhs, daddr)
            }
            _ => {
                // Whole-aggregate copy: the rhs is an aggregate lvalue; copy its bytes.
                let (saddr, _) = self.lvalue_addr(rhs)?;
                let size = self.t.sizeof(ddesc);
                let szc = self.emit_const(ValType::I64, size as i64);
                self.used_memory = true;
                self.cur_buf
                    .push_str(&format!("  mem.copy v{daddr} v{saddr} v{}\n", szc.id));
                Ok(())
            }
        }
    }

    /// Store one member (field/element) at `addr`: a scalar stores directly; a nested aggregate
    /// constructs/copies via [`assign_aggregate`].
    fn store_member(&mut self, addr: u32, desc: &TyDesc, expr: &Node) -> Result<(), LengError> {
        match desc {
            TyDesc::Scalar(ty) => {
                let v = self.expr_typed(expr, *ty)?;
                self.used_memory = true;
                self.cur_buf
                    .push_str(&format!("  {}.store v{addr} v{}\n", prefix(*ty), v.id));
                Ok(())
            }
            TyDesc::Narrow { bytes, .. } => {
                let v = self.expr_typed(expr, ValType::I32)?;
                self.used_memory = true;
                self.cur_buf
                    .push_str(&format!("  i32.store{} v{addr} v{}\n", bytes * 8, v.id));
                Ok(())
            }
            TyDesc::Ptr(_) => {
                let v = self.expr_typed(expr, ValType::I64)?;
                self.used_memory = true;
                self.cur_buf
                    .push_str(&format!("  i64.store v{addr} v{}\n", v.id));
                Ok(())
            }
            TyDesc::Agg(_) => self.assign_aggregate(addr, desc, expr),
        }
    }

    /// `StmtList ::= (stmts SCOPE? Stmt*)` — real hexer often omits the leading SCOPE atom.
    fn stmt_list(&mut self, node: &Node) -> Result<(), LengError> {
        if node.is_empty_marker() {
            return Ok(());
        }
        if node.tag() != Some("stmts") {
            return Err(LengError::Malformed("expected (stmts …)".into()));
        }
        let children = node.args();
        let start = usize::from(matches!(children.first(), Some(Node::Atom(_))));
        for stmt in &children[start..] {
            // After a terminator, statements are dead — except a `(lab L)`, which reopens a block
            // reachable via `(jmp L)` from elsewhere.
            if self.terminated && stmt.tag() != Some("lab") {
                continue;
            }
            self.stmt(stmt)?;
        }
        Ok(())
    }

    fn stmt(&mut self, s: &Node) -> Result<(), LengError> {
        match s.tag() {
            // Nested block / scope: recurse (hexer emits `(stmts (stmts …))` and `(scope (stmts …))`).
            Some("stmts") => self.stmt_list(s),
            Some("scope") => {
                for child in s.args() {
                    if child.tag() == Some("stmts") {
                        self.stmt_list(child)?;
                    }
                }
                Ok(())
            }
            Some("ret") => {
                let a = s.args();
                // Aggregate return by sret: construct/copy the result into the caller's destination
                // pointer (`cur[slot]`), then a plain void return.
                if let Some((slot, desc)) = self.sret.clone() {
                    if !(a.is_empty() || a[0].is_empty_marker()) {
                        let dst = self.cur[slot];
                        self.assign_aggregate(dst, &desc, &a[0])?;
                    }
                    self.finish_block("return".into(), self.next_block);
                    self.terminated = true;
                    return Ok(());
                }
                let term = if a.is_empty() || a[0].is_empty_marker() {
                    "return".to_string()
                } else {
                    // Coerce the returned value to the function's result type (a bare literal
                    // defaults to i64, but the proc may return i32).
                    let v = match self.ret {
                        Some(t) => self.expr_typed(&a[0], t)?,
                        None => self.expr(&a[0])?,
                    };
                    format!("return v{}", v.id)
                };
                self.finish_block(term, self.next_block);
                self.terminated = true;
                Ok(())
            }
            Some("var") => {
                let a = s.args();
                if a.len() < 3 {
                    return Err(LengError::Malformed("var needs :name pragmas type".into()));
                }
                let name = sym_def(&a[0])?;
                // An aggregate frame local's storage is the (zeroed) frame: a default-init is a
                // no-op; an initializer (`oconstr`/`aconstr`/another aggregate) constructs in place.
                if let Some((off, ddesc)) = self.mem.get(&name).cloned() {
                    if matches!(ddesc, TyDesc::Agg(_)) {
                        return match a.get(3) {
                            Some(init) if !init.is_empty_marker() => {
                                let sp = self.cur[0];
                                let daddr = self.add_const_off(sp, off);
                                self.assign_aggregate(daddr, &ddesc, init)
                            }
                            _ => Ok(()),
                        };
                    }
                }
                let ty = val_ty(&a[2])?;
                let v = match a.get(3) {
                    Some(init) if !init.is_empty_marker() => self.expr_typed(init, ty)?,
                    _ => self.emit_const(ty, 0),
                };
                self.write_local(&name, v)
            }
            Some("asgn") => {
                // `(asgn Lvalue Expr)`.
                let a = s.args();
                if a.len() != 2 {
                    return Err(LengError::Malformed("asgn needs lvalue and expr".into()));
                }
                self.assign(&a[0], &a[1])
            }
            Some("store") => {
                // `(store Expr Lvalue)` — asgn with reversed operands (Leng StoreStmt).
                let a = s.args();
                if a.len() != 2 {
                    return Err(LengError::Malformed("store needs value and lvalue".into()));
                }
                self.assign(&a[1], &a[0])
            }
            Some("discard") => {
                if let Some(e) = s.args().first() {
                    if !e.is_empty_marker() {
                        self.expr(e)?;
                    }
                }
                Ok(())
            }
            Some("call") => {
                self.call(s, None)?; // statement position: result (if any) discarded
                Ok(())
            }
            Some("if") => self.if_stmt(s),
            Some("while") => self.while_stmt(s),
            Some("case") => self.case_stmt(s),
            Some("jmp") => self.jmp_stmt(s),
            Some("lab") => self.lab_stmt(s),
            Some("break") => self.loop_jump(false, "break"),
            Some("continue") => self.loop_jump(true, "continue"),
            other => Err(LengError::Unsupported(format!(
                "statement `{}`",
                other.unwrap_or("<headless>")
            ))),
        }
    }

    /// `(if (elif Cond Body)+ (else Body)?)` → a chain of `br_if`s over blocks.
    fn if_stmt(&mut self, s: &Node) -> Result<(), LengError> {
        let mut elifs = Vec::new();
        let mut else_body = None;
        for clause in s.args() {
            match clause.tag() {
                Some("elif") => {
                    let c = clause.args();
                    if c.len() < 2 {
                        return Err(LengError::Malformed("elif needs cond and body".into()));
                    }
                    elifs.push((&c[0], &c[1]));
                }
                Some("else") => {
                    else_body = clause.args().first();
                }
                _ => return Err(LengError::Malformed("if expects elif/else clauses".into())),
            }
        }
        if elifs.is_empty() {
            return Err(LengError::Malformed("if with no elif".into()));
        }
        let cont = self.new_block_id();
        for (cond, body) in elifs {
            let then_id = self.new_block_id();
            let next_id = self.new_block_id(); // next elif's test, or the else/cont
            let vc = self.expr(cond)?;
            let vc = self.as_i32_cond(vc);
            let args = self.branch_args();
            self.finish_block(
                format!("br_if v{vc} {then_id}{args} {next_id}{args}"),
                then_id,
            );
            // then block
            self.stmt_list_or_single(body)?;
            if !self.terminated {
                let a = self.branch_args();
                self.finish_block(format!("br {cont}{a}"), next_id);
            } else {
                // then already returned; still must open next_id.
                self.cur_id = next_id;
                self.reset_cur_state();
            }
        }
        // `cur` is now the last `next_id` block: the else arm (or a fallthrough to cont).
        if let Some(eb) = else_body {
            self.stmt_list_or_single(eb)?;
        }
        if !self.terminated {
            let a = self.branch_args();
            self.finish_block(format!("br {cont}{a}"), cont);
        } else {
            self.cur_id = cont;
            self.reset_cur_state();
        }
        Ok(())
    }

    /// `(while Cond Body)` → header/body/exit blocks.
    fn while_stmt(&mut self, s: &Node) -> Result<(), LengError> {
        let a = s.args();
        if a.len() < 2 {
            return Err(LengError::Malformed("while needs cond and body".into()));
        }
        let header = self.new_block_id();
        let body = self.new_block_id();
        let exit = self.new_block_id();
        // Enter the loop header.
        let args = self.branch_args();
        self.finish_block(format!("br {header}{args}"), header);
        // Header: test.
        let vc = self.expr(&a[0])?;
        let vc = self.as_i32_cond(vc);
        let hargs = self.branch_args();
        self.finish_block(format!("br_if v{vc} {body}{hargs} {exit}{hargs}"), body);
        // Body: run (with this loop's break/continue targets in scope), then back to header.
        self.loop_stack.push((header, exit));
        let r = self.stmt_list_or_single(&a[1]);
        self.loop_stack.pop();
        r?;
        if !self.terminated {
            let bargs = self.branch_args();
            self.finish_block(format!("br {header}{bargs}"), exit);
        } else {
            self.cur_id = exit;
            self.reset_cur_state();
        }
        Ok(())
    }

    /// `(break)` / `(continue)` → branch to the innermost enclosing loop's `exit` / `header` block,
    /// threading the current slots. The block is terminated (following code is dead until a `lab`).
    fn loop_jump(&mut self, to_header: bool, kw: &str) -> Result<(), LengError> {
        let &(header, exit) = self
            .loop_stack
            .last()
            .ok_or_else(|| LengError::Malformed(format!("`{kw}` outside a loop")))?;
        let target = if to_header { header } else { exit };
        let args = self.branch_args();
        self.finish_block(format!("br {target}{args}"), self.next_block);
        self.terminated = true;
        Ok(())
    }

    /// `(jmp L)` → an unconditional branch to label `L`'s block, threading the current slot values.
    /// Forward (`break`, `block`-`break`) and backward jumps both work: every block carries all
    /// slots, so any edge just passes the live slot set. The block is terminated; following code is
    /// dead until the next `(lab …)` reopens a reachable block.
    fn jmp_stmt(&mut self, s: &Node) -> Result<(), LengError> {
        let name = s
            .args()
            .first()
            .and_then(|n| n.as_atom())
            .ok_or_else(|| LengError::Malformed("jmp needs a label".into()))?;
        let lb = *self
            .label_block
            .get(name)
            .ok_or_else(|| LengError::Unsupported(format!("jmp to unknown label `{name}`")))?;
        let args = self.branch_args();
        self.finish_block(format!("br {lb}{args}"), self.next_block);
        self.terminated = true;
        Ok(())
    }

    /// `(lab :L)` → open label `L`'s pre-allocated block. If the current block is still live it falls
    /// through into `L` (threading slots); otherwise `L` is reached only by `(jmp L)` edges and we
    /// just switch to emitting there.
    fn lab_stmt(&mut self, s: &Node) -> Result<(), LengError> {
        let name = s
            .args()
            .first()
            .map(sym_def)
            .transpose()?
            .ok_or_else(|| LengError::Malformed("lab needs a name".into()))?;
        let lb = *self
            .label_block
            .get(&name)
            .ok_or_else(|| LengError::Unsupported(format!("undeclared label `{name}`")))?;
        if self.terminated {
            self.cur_id = lb;
            self.reset_cur_state();
        } else {
            let args = self.branch_args();
            self.finish_block(format!("br {lb}{args}"), lb);
        }
        Ok(())
    }

    /// `(case Disc (of (ranges V+) Body)* (else Body)?)` → a `br_table` over a dense integer span
    /// (normalized index; out-of-range → default). Sparse/large-span or non-integer discriminants
    /// fail-closed (a comparison-chain lowering for those is a later refinement).
    fn case_stmt(&mut self, s: &Node) -> Result<(), LengError> {
        let a = s.args();
        if a.is_empty() {
            return Err(LengError::Malformed("case needs a discriminant".into()));
        }
        // Parse the `of`/`else` clauses into integer value sets.
        struct Branch<'n> {
            vals: Vec<i64>,
            ranges: Vec<(i64, i64)>,
            body: &'n Node,
        }
        let mut branches: Vec<Branch> = Vec::new();
        let mut else_body: Option<&Node> = None;
        for clause in &a[1..] {
            match clause.tag() {
                Some("of") => {
                    let c = clause.args();
                    if c.len() < 2 || c[0].tag() != Some("ranges") {
                        return Err(LengError::Malformed(
                            "of needs (ranges …) and a body".into(),
                        ));
                    }
                    let mut vals = Vec::new();
                    let mut ranges = Vec::new();
                    for v in c[0].args() {
                        if let Some(n) = int_literal(v) {
                            vals.push(n);
                        } else if v.tag() == Some("range") {
                            let ra = v.args();
                            let lo = ra.first().and_then(int_literal);
                            let hi = ra.get(1).and_then(int_literal);
                            match (lo, hi) {
                                (Some(l), Some(h)) => ranges.push((l, h)),
                                _ => {
                                    return Err(LengError::Unsupported(
                                        "non-integer case range".into(),
                                    ))
                                }
                            }
                        } else {
                            return Err(LengError::Unsupported(
                                "non-integer case branch value (symbol/char) — later slice".into(),
                            ));
                        }
                    }
                    branches.push(Branch {
                        vals,
                        ranges,
                        body: &c[1],
                    });
                }
                Some("else") => else_body = clause.args().first(),
                _ => return Err(LengError::Malformed("case expects of/else clauses".into())),
            }
        }
        if branches.is_empty() {
            return Err(LengError::Malformed("case with no `of`".into()));
        }

        // Dense span across all branch values; bail (fail-closed) if sparse/huge.
        let mut lo = i64::MAX;
        let mut hi = i64::MIN;
        for b in &branches {
            for &v in &b.vals {
                lo = lo.min(v);
                hi = hi.max(v);
            }
            for &(l, h) in &b.ranges {
                lo = lo.min(l);
                hi = hi.max(h);
            }
        }
        let span = (hi as i128) - (lo as i128) + 1;
        if !(1..=256).contains(&span) {
            return Err(LengError::Unsupported(
                "case span too large/sparse for a br_table (comparison-chain lowering is a later slice)"
                    .into(),
            ));
        }
        let span = span as usize;

        // Discriminant → i32 index normalized to `lo`. Out-of-range → br_table default (a negative
        // or over-large index selects `default`).
        let disc = self.expr(&a[0])?;
        let disc32 = if disc.ty == ValType::I32 {
            disc
        } else {
            self.convert(disc, ValType::I32)
        };
        let loc = self.emit_const(ValType::I32, lo);
        let nidx = self.fresh();
        self.cur_buf
            .push_str(&format!("  v{nidx} = i32.sub v{} v{}\n", disc32.id, loc.id));

        // Blocks: one per branch body, an optional else, and the continuation.
        let cont = self.new_block_id();
        let bblocks: Vec<u32> = (0..branches.len()).map(|_| self.new_block_id()).collect();
        let else_block = else_body.map(|_| self.new_block_id());
        let default_target = else_block.unwrap_or(cont);

        // One table entry per value in [lo, hi]; each maps to its covering branch, else default.
        let args = self.branch_args();
        let mut targets = Vec::with_capacity(span);
        for i in 0..span {
            let v = lo + i as i64;
            let tgt = branches
                .iter()
                .position(|b| {
                    b.vals.contains(&v) || b.ranges.iter().any(|&(l, h)| v >= l && v <= h)
                })
                .map(|bi| bblocks[bi])
                .unwrap_or(default_target);
            targets.push(format!("{tgt}{args}"));
        }
        self.finish_block(
            format!(
                "br_table v{nidx} [{}] {default_target}{args}",
                targets.join(", ")
            ),
            bblocks[0],
        );

        // Branch bodies, each falling through to `cont`.
        let nbr = branches.len();
        for bi in 0..nbr {
            let body = branches[bi].body;
            let next = if bi + 1 < nbr {
                bblocks[bi + 1]
            } else {
                else_block.unwrap_or(cont)
            };
            self.stmt_list_or_single(body)?;
            if !self.terminated {
                let a2 = self.branch_args();
                self.finish_block(format!("br {cont}{a2}"), next);
            } else {
                self.cur_id = next;
                self.reset_cur_state();
            }
        }
        // else body (cur is `else_block` if present).
        if let Some(eb) = else_body {
            self.stmt_list_or_single(eb)?;
            if !self.terminated {
                let a2 = self.branch_args();
                self.finish_block(format!("br {cont}{a2}"), cont);
            } else {
                self.cur_id = cont;
                self.reset_cur_state();
            }
        }
        Ok(())
    }

    /// A control-flow arm body: a `(stmts …)`, or a single statement.
    fn stmt_list_or_single(&mut self, node: &Node) -> Result<(), LengError> {
        if node.tag() == Some("stmts") {
            self.stmt_list(node)
        } else {
            self.stmt(node)
        }
    }

    /// Ensure a value is an `i32` condition (comparisons already produce i32). A wider integer
    /// condition — e.g. an `i64` enum error code from `if canRaise.fld.0` — is reduced to a correct
    /// 0/1 via `!= 0` (an i64→i32 *wrap* would drop the high word and misjudge truthiness).
    fn as_i32_cond(&mut self, v: Val) -> u32 {
        match v.ty {
            ValType::I32 => v.id,
            ValType::I64 => {
                let z = self.emit_const(ValType::I64, 0);
                let id = self.fresh();
                self.cur_buf
                    .push_str(&format!("  v{id} = i64.ne v{} v{}\n", v.id, z.id));
                id
            }
            _ => self.convert(v, ValType::I32).id,
        }
    }

    fn expr(&mut self, e: &Node) -> Result<Val, LengError> {
        match e {
            Node::Atom(a) => {
                if let Some(v) = self.read_local(a) {
                    return Ok(v);
                }
                if let Some(&c) = self.t.consts.get(a) {
                    return Ok(self.emit_const(ValType::I64, c));
                }
                if self.t.globals.contains_key(a) {
                    return self.load_lvalue(e); // load the scalar global
                }
                if let Ok(n) = parse_int(a) {
                    return Ok(self.emit_const(ValType::I64, n));
                }
                if let Ok(f) = a.parse::<f64>() {
                    return Ok(self.emit_fconst(ValType::F64, f)); // float literal (2.0, 1.5, 1e3)
                }
                // A link unit: a leftover atom is a cross-module data symbol — load through `data.sym`.
                if self.t.link_mode {
                    return self.load_lvalue(e);
                }
                Err(LengError::Unsupported(format!("atom expression `{a}`")))
            }
            Node::List(_) => match e.tag() {
                Some(op @ ("add" | "sub" | "mul" | "div" | "mod")) => self.arith(op, e),
                Some(op @ ("bitand" | "bitor" | "bitxor" | "shl" | "shr" | "ashr")) => {
                    self.bitwise(op, e)
                }
                Some(op @ ("eq" | "neq" | "lt" | "le")) => self.compare(op, e),
                Some("bitnot") => {
                    // Bitwise complement `(bitnot T x)` — `x xor -1` at the type's width.
                    let a = e.args();
                    if a.len() != 2 {
                        return Err(LengError::Malformed(
                            "bitnot needs Type and one operand".into(),
                        ));
                    }
                    let (ty, _) = int_ty_signed(&a[0])?;
                    let x = self.expr_typed(&a[1], ty)?;
                    let ones = self.emit_const(ty, -1);
                    Ok(self.emit_bin("xor", ty, x, ones))
                }
                Some("not") => {
                    // Logical `not` on a bool (i32 0/1): `x == 0`, an i32 0/1.
                    let x = self.expr(&e.args()[0])?;
                    let zero = self.emit_const(x.ty, 0);
                    let id = self.fresh();
                    self.cur_buf.push_str(&format!(
                        "  v{id} = {}.eq v{} v{}\n",
                        prefix(x.ty),
                        x.id,
                        zero.id
                    ));
                    Ok(Val {
                        id,
                        ty: ValType::I32,
                    })
                }
                Some("neg") => {
                    let a = e.args();
                    let ty = val_ty(&a[0])?;
                    let x = self.expr_typed(&a[1], ty)?;
                    if is_float(ty) {
                        // `fN.neg` (a proper negate — handles signed zero).
                        let id = self.fresh();
                        self.cur_buf
                            .push_str(&format!("  v{id} = {}.neg v{}\n", prefix(ty), x.id));
                        Ok(Val { id, ty })
                    } else {
                        let zero = self.emit_const(ty, 0);
                        Ok(self.emit_bin("sub", ty, zero, x))
                    }
                }
                Some("conv") => {
                    // Value-preserving numeric conversion (int width, int↔float, f32↔f64).
                    let a = e.args();
                    let ty = val_ty(&a[0])?;
                    let x = self.expr(&a[1])?;
                    Ok(self.convert(x, ty))
                }
                Some("cast") => {
                    // A C-style cast — for the scalar/pointer subset, a width reinterpretation
                    // (pointer↔pointer and same-width are no-ops; i32↔i64 extend/wrap).
                    let a = e.args();
                    let ty = val_ty(&a[0])?;
                    let x = self.expr(&a[1])?;
                    Ok(self.convert(x, ty))
                }
                Some("par") => self.expr(&e.args()[0]),
                // Literals: booleans are `i32` 0/1; `nil` is a null `i64` pointer.
                Some("true") => Ok(self.emit_const(ValType::I32, 1)),
                Some("false") => Ok(self.emit_const(ValType::I32, 0)),
                Some("nil") => Ok(self.emit_const(ValType::I64, 0)),
                // Float specials.
                Some("inf") => Ok(self.emit_fconst(ValType::F64, f64::INFINITY)),
                Some("neginf") => Ok(self.emit_fconst(ValType::F64, f64::NEG_INFINITY)),
                Some("nan") => Ok(self.emit_fconst(ValType::F64, f64::NAN)),
                Some("addr") => {
                    // The address of an lvalue (a frame/aggregate local, or a `dot`/`at`/`pat`).
                    let (id, _desc) = self.lvalue_addr(&e.args()[0])?;
                    Ok(Val {
                        id,
                        ty: ValType::I64,
                    })
                }
                Some("sizeof") => {
                    // `(sizeof T)` — a compile-time constant, the byte size of the type (nimony emits
                    // it for `copyMem` lengths). Resolved from the layout registry.
                    let a = e.args();
                    let t = a
                        .first()
                        .ok_or_else(|| LengError::Malformed("sizeof needs a type".into()))?;
                    let desc = self.t.tydesc(t)?;
                    Ok(self.emit_const(ValType::I64, self.t.sizeof(&desc) as i64))
                }
                Some("suf") => {
                    // A **suffixed literal** — `255'i64`, `1.5'f32` — value then a `"type"` tag.
                    let a = e.args();
                    if a.len() < 2 {
                        return Err(LengError::Malformed(
                            "suf needs a literal and a type".into(),
                        ));
                    }
                    let ty = suf_val_ty(a[1].as_atom().unwrap_or(""));
                    if is_float(ty) {
                        let f = a[0]
                            .as_atom()
                            .and_then(|s| s.trim_matches('"').parse::<f64>().ok())
                            .ok_or_else(|| {
                                LengError::Unsupported("non-float suf literal".into())
                            })?;
                        Ok(self.emit_fconst(ty, f))
                    } else {
                        let n = int_literal(&a[0])
                            .ok_or_else(|| LengError::Unsupported("non-int suf literal".into()))?;
                        Ok(self.emit_const(ty, n))
                    }
                }
                // Reading through an lvalue: load the scalar it addresses.
                Some("deref" | "dot" | "at" | "pat") => self.load_lvalue(e),
                Some("call") => self.call(e, Some(ValType::I64)), // value wanted (default i64 hint)
                other => Err(LengError::Unsupported(format!(
                    "expression `{}`",
                    other.unwrap_or("<headless>")
                ))),
            },
        }
    }

    fn expr_typed(&mut self, e: &Node, want: ValType) -> Result<Val, LengError> {
        if let Node::Atom(a) = e {
            if self.lookup(a).is_none() {
                if let Ok(n) = parse_int(a) {
                    return Ok(self.emit_const(want, n));
                }
            }
        }
        // A call in typed position hands `want` down as the return hint, so a cross-module callee's
        // import is declared returning exactly this type (not a guessed `i64`).
        let v = if e.tag() == Some("call") {
            self.call(e, Some(want))?
        } else {
            self.expr(e)?
        };
        Ok(if v.ty != want {
            self.convert(v, want)
        } else {
            v
        })
    }

    /// `(add|sub|mul|div|mod Type Expr Expr)` — integer or floating-point per the carried `Type`.
    fn arith(&mut self, op: &str, e: &Node) -> Result<Val, LengError> {
        let a = e.args();
        if a.len() != 3 {
            return Err(LengError::Malformed(format!(
                "`{op}` needs Type and two operands"
            )));
        }
        // Floating-point arithmetic: `fN.{add,sub,mul,div}` (no signedness, no `mod`).
        if a[0].tag() == Some("f") {
            let ty = float_ty(&a[0])?;
            let l = self.expr_typed(&a[1], ty)?;
            let r = self.expr_typed(&a[2], ty)?;
            let name = match op {
                "add" => "add",
                "sub" => "sub",
                "mul" => "mul",
                "div" => "div",
                _ => {
                    return Err(LengError::Unsupported(format!(
                        "float `{op}` (no `mod` on floats)"
                    )))
                }
            };
            return Ok(self.emit_bin(name, ty, l, r));
        }
        let (ty, signed) = int_ty_signed(&a[0])?;
        let l = self.expr_typed(&a[1], ty)?;
        let r = self.expr_typed(&a[2], ty)?;
        let name = match op {
            "add" => "add",
            "sub" => "sub",
            "mul" => "mul",
            "div" => {
                if signed {
                    "div_s"
                } else {
                    "div_u"
                }
            }
            "mod" => {
                if signed {
                    "rem_s"
                } else {
                    "rem_u"
                }
            }
            _ => unreachable!(),
        };
        Ok(self.emit_bin(name, ty, l, r))
    }

    /// `(bitand|bitor|bitxor|shl|shr|ashr Type Expr Expr)` — bitwise/shift, same `(Type a b)` shape
    /// as [`arith`]. `shr` is logical for unsigned types, arithmetic for signed; `ashr` is always
    /// arithmetic.
    fn bitwise(&mut self, op: &str, e: &Node) -> Result<Val, LengError> {
        let a = e.args();
        if a.len() != 3 {
            return Err(LengError::Malformed(format!(
                "`{op}` needs Type and two operands"
            )));
        }
        let (ty, signed) = int_ty_signed(&a[0])?;
        let l = self.expr_typed(&a[1], ty)?;
        let r = self.expr_typed(&a[2], ty)?;
        let name = match op {
            "bitand" => "and",
            "bitor" => "or",
            "bitxor" => "xor",
            "shl" => "shl",
            "shr" if signed => "shr_s",
            "shr" => "shr_u",
            "ashr" => "shr_s",
            _ => unreachable!(),
        };
        Ok(self.emit_bin(name, ty, l, r))
    }

    /// `(eq|neq|lt|le Expr Expr)` — no explicit type (Leng grammar); infer from the left operand.
    /// Leng ints are signed. Result is `i32` (0/1).
    fn compare(&mut self, op: &str, e: &Node) -> Result<Val, LengError> {
        let a = e.args();
        if a.len() != 2 {
            return Err(LengError::Malformed(format!("`{op}` needs two operands")));
        }
        let l = self.expr(&a[0])?;
        let r = self.expr_typed(&a[1], l.ty)?;
        // Float compares use unsigned-free names (`lt`/`le`); int compares are signed here.
        let name = if is_float(l.ty) {
            match op {
                "eq" => "eq",
                "neq" => "ne",
                "lt" => "lt",
                "le" => "le",
                _ => unreachable!(),
            }
        } else {
            match op {
                "eq" => "eq",
                "neq" => "ne",
                "lt" => "lt_s",
                "le" => "le_s",
                _ => unreachable!(),
            }
        };
        let id = self.fresh();
        self.cur_buf.push_str(&format!(
            "  v{id} = {}.{name} v{} v{}\n",
            prefix(l.ty),
            l.id,
            r.id
        ));
        Ok(Val {
            id,
            ty: ValType::I32,
        })
    }

    /// `(call Callee Expr*)` — a direct call to a module proc, or an SVM `import` for a cross-module
    /// callee. `ret_hint` is the type the result is wanted as (from the enclosing `expr_typed`), or
    /// `None` in statement position (result discarded / void) — it fixes an **import**'s declared
    /// return type, so a `bool`-returning cross-module callee (`arcDec`) is `(…) -> (i32)`, not a
    /// mis-guessed `i64` the linker would reject against the real proc.
    fn call(&mut self, e: &Node, ret_hint: Option<ValType>) -> Result<Val, LengError> {
        let a = e.args();
        if a.is_empty() {
            return Err(LengError::Malformed("call needs a callee".into()));
        }
        let callee = a[0].as_atom().ok_or_else(|| {
            LengError::Unsupported("indirect call (callee is not a symbol)".into())
        })?;
        // Cross-module callee (not defined in this module) → an SVM import.
        if !self.t.procs.contains_key(callee) {
            return self.call_import(callee, &a[1..], ret_hint);
        }
        // An aggregate-returning call must flow to an aggregate destination (`asgn`/`var`/`ret`),
        // which routes through `call_sret`. Reaching plain `call` means it was used as a scalar
        // value or a bare statement — fail closed rather than emit a mismatched call.
        if self
            .t
            .procs
            .get(callee)
            .and_then(|s| s.sret.as_ref())
            .is_some()
        {
            return Err(LengError::Unsupported(format!(
                "aggregate-returning call to `{callee}` must be directly assigned to an aggregate destination"
            )));
        }
        let sig = self.t.procs.get(callee).expect("checked above");
        let index = sig.index;
        let ptys = sig.params.clone();
        let ret = sig.ret;
        let callee_needs_frame = sig.needs_frame;
        if a.len() - 1 != ptys.len() {
            return Err(LengError::Malformed(format!(
                "call to `{callee}`: {} args for {} params",
                a.len() - 1,
                ptys.len()
            )));
        }
        let mut argvals = Vec::new();
        // A frame-needing callee gets a fresh frame beyond ours: `sp_callee = sp + frame_size`.
        if callee_needs_frame {
            if !self.has_sp {
                return Err(LengError::Unsupported(format!(
                    "frameless proc calls frame-needing `{callee}` (no stack pointer to hand down)"
                )));
            }
            let sp = self.cur[0];
            let fs = self.emit_const(ValType::I64, self.frame_size as i64);
            let spid = self.fresh();
            self.cur_buf
                .push_str(&format!("  v{spid} = i64.add v{sp} v{}\n", fs.id));
            argvals.push(spid);
        }
        for (arg, want) in a[1..].iter().zip(ptys) {
            argvals.push(self.call_arg(arg, want)?);
        }
        let arglist = argvals
            .iter()
            .map(|id| format!("v{id}"))
            .collect::<Vec<_>>()
            .join(", ");
        match ret {
            Some(ty) => {
                let id = self.fresh();
                self.cur_buf
                    .push_str(&format!("  v{id} = call {index} ({arglist})\n"));
                Ok(Val { id, ty })
            }
            None => {
                self.cur_buf
                    .push_str(&format!("  call {index} ({arglist})\n"));
                Ok(Val {
                    id: u32::MAX,
                    ty: ValType::I32,
                })
            }
        }
    }

    /// `(call P Expr*)` where `P` returns an aggregate by sret. Emits a void call passing `dest_addr`
    /// as `P`'s hidden `$sret` pointer, so the callee writes its result straight into the
    /// destination. Arg order matches `P`'s signature: `[$sp?] [dest_addr] [params…]`.
    fn call_sret(&mut self, e: &Node, dest_addr: u32) -> Result<(), LengError> {
        let a = e.args();
        let callee = a[0]
            .as_atom()
            .ok_or_else(|| LengError::Unsupported("indirect aggregate-returning call".into()))?;
        // A cross-module aggregate-returning callee (e.g. `toOpenArray`) becomes an sret import.
        if !self.t.procs.contains_key(callee) {
            return self.call_import_sret(callee, &a[1..], dest_addr);
        }
        let sig = self.t.procs.get(callee).expect("checked above");
        if sig.sret.is_none() {
            return Err(LengError::Unsupported(format!(
                "`{callee}` does not return an aggregate, but is used in aggregate position"
            )));
        }
        let index = sig.index;
        let ptys = sig.params.clone();
        let callee_needs_frame = sig.needs_frame;
        if a.len() - 1 != ptys.len() {
            return Err(LengError::Malformed(format!(
                "call to `{callee}`: {} args for {} params",
                a.len() - 1,
                ptys.len()
            )));
        }
        let mut argvals = Vec::new();
        if callee_needs_frame {
            if !self.has_sp {
                return Err(LengError::Unsupported(format!(
                    "frameless proc calls frame-needing `{callee}` (no stack pointer to hand down)"
                )));
            }
            let sp = self.cur[0];
            let fs = self.emit_const(ValType::I64, self.frame_size as i64);
            let spid = self.fresh();
            self.cur_buf
                .push_str(&format!("  v{spid} = i64.add v{sp} v{}\n", fs.id));
            argvals.push(spid);
        }
        argvals.push(dest_addr); // the `$sret` destination pointer
        for (arg, want) in a[1..].iter().zip(ptys) {
            argvals.push(self.call_arg(arg, want)?);
        }
        let arglist = argvals
            .iter()
            .map(|id| format!("v{id}"))
            .collect::<Vec<_>>()
            .join(", ");
        self.cur_buf
            .push_str(&format!("  call {index} ({arglist})\n"));
        Ok(())
    }

    /// One call argument → its SVM value id. An **aggregate** argument is passed by address (the
    /// callee's param is a by-address pointer); a scalar is evaluated and coerced to the param type.
    fn call_arg(&mut self, arg: &Node, want: ValType) -> Result<u32, LengError> {
        if let Some(TyDesc::Agg(_)) = self.lvalue_type(arg) {
            let (addr, _) = self.lvalue_addr(arg)?;
            return Ok(addr);
        }
        Ok(self.expr_typed(arg, want)?.id)
    }

    /// Lower a cross-module call to a declared SVM `import` + `call.import`. Param types come from
    /// the args; the return arity from the call position (a stmt-call is treated as void). The
    /// runtime binds the import by name at instantiation — the frontend only makes it well-typed.
    fn call_import(
        &mut self,
        name: &str,
        args: &[Node],
        ret: Option<ValType>,
    ) -> Result<Val, LengError> {
        let mut argvals = Vec::new();
        let mut argtys = Vec::new();
        for arg in args {
            // Aggregate args pass by address (matching by-address params); scalars by value.
            if let Some(TyDesc::Agg(_)) = self.lvalue_type(arg) {
                let (addr, _) = self.lvalue_addr(arg)?;
                argvals.push(addr);
                argtys.push(ValType::I64);
            } else {
                let v = self.expr(arg)?;
                argvals.push(v.id);
                argtys.push(v.ty);
            }
        }
        let slot = self.t.register_import(name, &argtys, ret)?;
        let arglist = argvals
            .iter()
            .map(|id| format!("v{id}"))
            .collect::<Vec<_>>()
            .join(", ");
        match ret {
            Some(ty) => {
                let id = self.fresh();
                self.cur_buf
                    .push_str(&format!("  v{id} = call.import {slot} ({arglist})\n"));
                Ok(Val { id, ty })
            }
            None => {
                self.cur_buf
                    .push_str(&format!("  call.import {slot} ({arglist})\n"));
                Ok(Val {
                    id: u32::MAX,
                    ty: ValType::I32,
                })
            }
        }
    }

    /// An aggregate-returning cross-module call (e.g. `toOpenArray`): the import takes a leading
    /// `$sret` pointer (`dest_addr`) and returns void, so the runtime writes the aggregate straight
    /// into the destination. Aggregate args pass by address, scalars by value.
    fn call_import_sret(
        &mut self,
        name: &str,
        args: &[Node],
        dest_addr: u32,
    ) -> Result<(), LengError> {
        let mut argvals = vec![dest_addr];
        let mut argtys = vec![ValType::I64]; // the sret pointer
        for arg in args {
            if let Some(TyDesc::Agg(_)) = self.lvalue_type(arg) {
                let (addr, _) = self.lvalue_addr(arg)?;
                argvals.push(addr);
                argtys.push(ValType::I64);
            } else {
                let v = self.expr(arg)?;
                argvals.push(v.id);
                argtys.push(v.ty);
            }
        }
        let slot = self.t.register_import(name, &argtys, None)?; // void: writes via the sret pointer
        let arglist = argvals
            .iter()
            .map(|id| format!("v{id}"))
            .collect::<Vec<_>>()
            .join(", ");
        self.cur_buf
            .push_str(&format!("  call.import {slot} ({arglist})\n"));
        Ok(())
    }

    fn emit_const(&mut self, ty: ValType, n: i64) -> Val {
        let id = self.fresh();
        self.cur_buf
            .push_str(&format!("  v{id} = {}.const {n}\n", prefix(ty)));
        Val { id, ty }
    }

    /// A relocatable address of this unit's own data at `off` (`data.self`). Resolved to a concrete
    /// window address by `link`; touching the window, so it forces the `memory` declaration.
    fn emit_data_self(&mut self, off: u64) -> u32 {
        let id = self.fresh();
        self.used_memory = true;
        self.cur_buf
            .push_str(&format!("  v{id} = data.self {off}\n"));
        id
    }

    /// A relocatable address of a **cross-module data symbol** (`data.sym`) — a `gvar` defined and
    /// exported by another unit. `link` binds it to that unit's `data_export`; an unresolved name is
    /// a fail-closed link error.
    fn emit_data_sym(&mut self, name: &str, addend: i64) -> u32 {
        let id = self.fresh();
        self.used_memory = true;
        self.cur_buf.push_str(&format!(
            "  v{id} = data.sym \"{}\" {addend}\n",
            escape_str(name)
        ));
        id
    }

    /// Whether `name` is a function-local (an SSA slot or a frame slot) — as opposed to a module
    /// global, a const, or a cross-module data symbol.
    fn is_local(&self, name: &str) -> bool {
        self.slot_of.contains_key(name) || self.mem.contains_key(name)
    }

    /// A float constant (`{:?}` round-trips through svm-text's `parse_float`).
    fn emit_fconst(&mut self, ty: ValType, f: f64) -> Val {
        let id = self.fresh();
        self.cur_buf
            .push_str(&format!("  v{id} = {}.const {f:?}\n", prefix(ty)));
        Val { id, ty }
    }

    fn emit_bin(&mut self, op: &str, ty: ValType, l: Val, r: Val) -> Val {
        let id = self.fresh();
        self.cur_buf.push_str(&format!(
            "  v{id} = {}.{op} v{} v{}\n",
            prefix(ty),
            l.id,
            r.id
        ));
        Val { id, ty }
    }

    /// Scalar numeric conversion: integer widths, int↔float (signed trunc/convert), and f32↔f64.
    fn convert(&mut self, v: Val, to: ValType) -> Val {
        use ValType::{F32, F64, I32, I64};
        if v.ty == to {
            return v;
        }
        let id = self.fresh();
        let insn = match (v.ty, to) {
            (I32, I64) => format!("i64.extend_i32_s v{}", v.id),
            (I64, I32) => format!("i32.wrap_i64 v{}", v.id),
            // int → float (value-preserving, signed).
            (I32, F32) => format!("f32.convert_i32_s v{}", v.id),
            (I32, F64) => format!("f64.convert_i32_s v{}", v.id),
            (I64, F32) => format!("f32.convert_i64_s v{}", v.id),
            (I64, F64) => format!("f64.convert_i64_s v{}", v.id),
            // float → int (truncate toward zero, signed — matches Nim `int(x)`).
            (F32, I32) => format!("i32.trunc_f32_s v{}", v.id),
            (F32, I64) => format!("i64.trunc_f32_s v{}", v.id),
            (F64, I32) => format!("i32.trunc_f64_s v{}", v.id),
            (F64, I64) => format!("i64.trunc_f64_s v{}", v.id),
            // float ↔ float.
            (F32, F64) => format!("f64.promote_f32 v{}", v.id),
            (F64, F32) => format!("f32.demote_f64 v{}", v.id),
            _ => format!("i64.extend_i32_s v{}", v.id), // unreachable in this subset
        };
        self.cur_buf.push_str(&format!("  v{id} = {insn}\n"));
        Val { id, ty: to }
    }
}

// ---------------------------------------------------------------------------
// Pre-scan + type/atom helpers.
// ---------------------------------------------------------------------------

/// SVM value type of a Leng type: pointers (`ptr`/`aptr`) and named aggregates (held by address)
/// are `i64`; floats are `f32`/`f64`; else an integer scalar.
fn val_ty(node: &Node) -> Result<ValType, LengError> {
    match node.tag() {
        Some("ptr") | Some("aptr") => Ok(ValType::I64),
        Some("f") => float_ty(node),
        Some(_) => int_ty(node),
        None => match node.as_atom() {
            Some(_) => Ok(ValType::I64),
            None => Err(LengError::Malformed("expected a type".into())),
        },
    }
}

/// `(f N)` → `f32` (N ≤ 32) or `f64`.
fn float_ty(node: &Node) -> Result<ValType, LengError> {
    let bits = node
        .args()
        .first()
        .and_then(|n| n.as_atom())
        .ok_or_else(|| LengError::Malformed("`f` type needs a bit width".into()))?;
    Ok(if int_bits(bits)? <= 32 {
        ValType::F32
    } else {
        ValType::F64
    })
}

fn is_float(t: ValType) -> bool {
    matches!(t, ValType::F32 | ValType::F64)
}

/// A proc needs a frame iff it takes the address of a local, or declares an aggregate-typed local
/// (indexed by `at`/`dot`, hence frame-resident). Kept in sync with `proc_body`'s frame decision.
fn proc_needs_frame(p: &Node) -> bool {
    let body = match p.args().get(4) {
        Some(b) => b,
        None => return false,
    };
    let mut addr = std::collections::HashSet::new();
    collect_addr_taken(body, &mut addr);
    !addr.is_empty() || has_aggregate_var(body)
}

/// True if any `(var …)` in the tree has a bare-symbol (named aggregate) type.
fn has_aggregate_var(node: &Node) -> bool {
    if let Node::List(_) = node {
        if node.tag() == Some("var") {
            let a = node.args();
            if a.len() >= 3 && matches!(&a[2], Node::Atom(s) if s != ".") {
                return true;
            }
        }
        return node.args().iter().any(has_aggregate_var);
    }
    false
}

/// Collect `(lab :L)` label names in document order (recursing through scopes, loops, and `if`s).
fn collect_labels(node: &Node, out: &mut Vec<String>) {
    if let Node::List(_) = node {
        if node.tag() == Some("lab") {
            if let Some(name) = node.args().first().and_then(|n| n.as_atom()) {
                out.push(name.strip_prefix(':').unwrap_or(name).to_string());
            }
        }
        for child in node.args() {
            collect_labels(child, out);
        }
    }
}

/// Collect the names of locals whose address is taken (`(addr name)`).
fn collect_addr_taken(node: &Node, out: &mut std::collections::HashSet<String>) {
    if let Node::List(_) = node {
        if node.tag() == Some("addr") {
            if let Some(name) = node.args().first().and_then(|n| n.as_atom()) {
                out.insert(name.to_string());
            }
        }
        for child in node.args() {
            collect_addr_taken(child, out);
        }
    }
}

/// Parse an integer Leng type `(i N)`/`(u N)`/`(c N)`/`(bool)` to a ValType; error on non-int.
fn int_ty(node: &Node) -> Result<ValType, LengError> {
    int_ty_signed(node).map(|(t, _)| t)
}

/// As [`int_ty`], also returning whether the type is signed (`i`/`c`) vs unsigned (`u`/`bool`).
fn int_ty_signed(node: &Node) -> Result<(ValType, bool), LengError> {
    match node.tag() {
        Some(k @ ("i" | "u" | "c")) => {
            let bits = node
                .args()
                .first()
                .and_then(|n| n.as_atom())
                .ok_or_else(|| LengError::Malformed(format!("`{k}` type needs a bit width")))?;
            let width = int_bits(bits)?;
            let vt = if width > 32 {
                ValType::I64
            } else {
                ValType::I32
            };
            Ok((vt, k != "u"))
        }
        Some("bool") => Ok((ValType::I32, false)),
        Some(other) => Err(LengError::Unsupported(format!(
            "type `{other}` (only integer/bool types are supported)"
        ))),
        None => match node.as_atom() {
            Some(sym) => Err(LengError::Unsupported(format!("named type `{sym}`"))),
            None => Err(LengError::Malformed("expected a type".into())),
        },
    }
}

/// `IntBits ::= ('+'|'-') [0-9]+`, where `-1` means machine word size (→ 64). Real hexer emits a
/// concrete width (`64`) with no sign.
fn int_bits(s: &str) -> Result<u32, LengError> {
    let stripped = s.strip_prefix('+').unwrap_or(s);
    if stripped == "-1" {
        return Ok(64);
    }
    stripped
        .parse::<u32>()
        .map_err(|_| LengError::Malformed(format!("bad IntBits `{s}`")))
}

/// A `:symbol-definition` atom → the bare mangled name (leading `:` stripped).
fn sym_def(node: &Node) -> Result<String, LengError> {
    match node.as_atom() {
        Some(a) => Ok(a.strip_prefix(':').unwrap_or(a).to_string()),
        None => Err(LengError::Malformed("expected a symbol definition".into())),
    }
}

/// Parse a Leng integer literal (decimal, optional sign).
fn parse_int(s: &str) -> Result<i64, ()> {
    // A trailing `u` marks an unsigned literal (nimony emits e.g. an SSO string's packed bytes as
    // `122511465736197u`); keep the bit pattern, and allow values above i64::MAX via u64.
    let t = s.strip_suffix('u').unwrap_or(s);
    if let Ok(n) = t.parse::<i64>() {
        return Ok(n);
    }
    t.parse::<u64>().map(|n| n as i64).map_err(|_| ())
}

/// The integer value of an atom literal node, if it is one.
fn int_literal(node: &Node) -> Option<i64> {
    if node.tag() == Some("suf") {
        // A suffixed integer literal `N'iXX` in constant position — the value is the first son.
        return node.args().first().and_then(int_literal);
    }
    node.as_atom().and_then(|s| parse_int(s).ok())
}

/// The pointee type node of a `(ptr T)` / `(aptr T)` type, if `node` is one (else `None` — e.g. a
/// non-pointer cast target). Used to type a `(deref (cast (ptr T) …))`.
fn ptr_pointee(node: &Node) -> Option<&Node> {
    match node.tag() {
        Some("ptr") | Some("aptr") => node.args().first(),
        _ => None,
    }
}

/// The value type of a NIF literal **suffix** (`"i64"`, `"u8"`, `"f32"`, …): floats map to
/// `f32`/`f64`; a `64`-wide integer to `i64`; everything else (narrower ints, `char`) to `i32`.
fn suf_val_ty(s: &str) -> ValType {
    match s.trim_matches('"') {
        "f32" => ValType::F32,
        "f64" => ValType::F64,
        t if t.ends_with("64") => ValType::I64,
        _ => ValType::I32,
    }
}
