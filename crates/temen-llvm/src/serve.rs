//! #2961 — **interfaces from C and Rust**: a module serves an interface its children call by name,
//! or imports one, through the substrate's own ops. Nothing here is a second serving mechanism: a
//! served interface is an [`temen_ir::ImplExport`], an imported one a grouped manifest import that a
//! `call.import` dispatches through, and a handle among an op's arguments or results is a `cap`
//! slot, which every engine translates between the two domains (INVARIANTS #8).
//!
//! The source names an interface with a **descriptor**: a constant byte string under the symbol
//! `__temen_iface__<tag>`, holding the interface's name and its ops in order,
//!
//! ```text
//! exec.host { run: (i64, i64) -> (i64); pass: (cap) -> (cap); ping: () -> () }
//! ```
//!
//! each op's shape in the text IR's own function-type syntax ([`temen_text::parse_func_type`]), its
//! slots drawn from `i32 i64 f32 f64 cap`. The name is what the manifest and the grant say, dots and
//! all; the tag is any identifier, only tying the ops to their descriptor. Each op is a function named
//! `__temen_op__<tag>__<op>`. A module that **defines** every op serves the interface; one that only
//! **declares** them imports it, and a call of a declared op is a `call.import` of that op. A mix is
//! refused. A `cap` is an `int` / `i32` in the source; two or more results are a small struct
//! returned by value (the §3a multi-result convention).
//!
//! A translated function takes a data-SP first (§3d), and the serve loop that dispatches a call is
//! suspended mid-frame, so a served op runs on a stack of its own, through a trampoline. When no op
//! can park, handlers never interleave, and one **serve stack** serves them all. When one can, a
//! parked handler lets its serve loop admit the next call, so each dispatch takes a stack from the
//! program's heap (#2964): an op may allocate, do I/O and block, like any service. Which applies is
//! decided by the predicate every engine asks of a serve handler ([`temen_ir::seams`]).

use std::cell::RefCell;
use std::collections::HashMap;

use temen_ir::seams::handler_never_parks;
use temen_ir::{Block, Func, FuncType, IfaceOp, ImplExport, Inst, Terminator, TypeEntry, ValType};

use crate::ll::ast::{Module as LModule, Parameter, Type};
use crate::{param_vtypes, result_types, unsup, Error, Types};

/// The descriptor symbol prefix: `__temen_iface__<tag>`.
pub(crate) const IFACE_PREFIX: &str = "__temen_iface__";
/// The op symbol prefix: `__temen_op__<tag>__<op>`.
pub(crate) const OP_PREFIX: &str = "__temen_op__";
/// A served op's data stack, the serve stack or a dispatch's own: a [`temen_ir::FIBER_STACK`],
/// what a handler fiber is charged on the oracle.
pub(crate) const SERVE_STACK_SIZE: u64 = 256 * 1024;

/// One interface a module names.
pub(crate) struct Iface {
    /// The interface's name, as the descriptor states it: the impl-export's or the import's.
    pub name: String,
    /// The descriptor symbol's tag, which names the op functions ([`op_symbol`]).
    pub tag: String,
    /// The ops in declaration order: their names and signatures, a handle slot typed `cap`.
    pub ops: Vec<(String, FuncType)>,
    /// Each op's signature in the source, where a `cap` slot is the `i32` (or, widened by the
    /// frontend's struct-return coercion, the `i64`) holding the handle: what the op's function
    /// takes and answers ([`check_sig`]). An imported op never declared is the `i32` form.
    pub storage: Vec<FuncType>,
    /// Whether this module serves it (defines its ops) rather than importing it.
    pub served: bool,
}

/// Every interface a module names, sorted by descriptor symbol, so a served interface's export index (the
/// handle a grant names it by) does not depend on symbol order.
#[derive(Default)]
pub(crate) struct Ifaces {
    pub list: Vec<Iface>,
    /// An imported op's symbol → `(interface, op)`.
    pub imported_ops: HashMap<String, (usize, u32)>,
}

impl Ifaces {
    pub fn any_served(&self) -> bool {
        self.list.iter().any(|i| i.served)
    }
}

/// The op function's symbol.
pub(crate) fn op_symbol(tag: &str, op: &str) -> String {
    format!("{OP_PREFIX}{tag}__{op}")
}

/// Collect the module's interfaces from its descriptors (`gbytes`: each constant global's bytes).
pub(crate) fn collect(
    m: &LModule,
    gbytes: &HashMap<String, Vec<u8>>,
    defined: &HashMap<String, u32>,
) -> Result<Ifaces, Error> {
    let mut list = Vec::new();
    let mut names: Vec<(&String, &Vec<u8>)> = gbytes
        .iter()
        .filter(|(n, _)| n.starts_with(IFACE_PREFIX))
        .collect();
    names.sort();
    for (sym, bytes) in names {
        let tag = &sym[IFACE_PREFIX.len()..];
        if !is_ident(tag) {
            return unsup(format!(
                "interface descriptor `{sym}`: `{tag}` is not a tag"
            ));
        }
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        let text = std::str::from_utf8(&bytes[..end]).map_err(|_| {
            Error::Unsupported(format!("interface descriptor `{sym}` is not UTF-8"))
        })?;
        let (name, ops) = parse_descriptor(text)
            .map_err(|e| Error::Unsupported(format!("interface descriptor `{sym}`: {e}")))?;
        if list.iter().any(|i: &Iface| i.name == name) {
            return unsup(format!("interface `{name}` has two descriptors"));
        }
        let defs = ops
            .iter()
            .filter(|(op, _)| defined.contains_key(&op_symbol(tag, op)))
            .count();
        if defs != 0 && defs != ops.len() {
            return unsup(format!(
                "interface `{name}`: {defs} of its {} ops are defined here; a module serves an \
                 interface (defines every op) or imports it (defines none)",
                ops.len()
            ));
        }
        let mut storage = Vec::with_capacity(ops.len());
        for (op, sig) in &ops {
            let sym = op_symbol(tag, op);
            let src = match m.functions.iter().find(|f| f.name == sym) {
                Some(f) => Some((&f.parameters, f.return_type.as_ref())),
                None => m
                    .func_declarations
                    .iter()
                    .find(|d| d.name == sym)
                    .map(|d| (&d.parameters, d.return_type.as_ref())),
            };
            storage.push(match src {
                Some((ps, ret)) => check_sig(&sym, ps, ret, &m.types, sig)?,
                None => FuncType {
                    params: narrow(&sig.params),
                    results: narrow(&sig.results),
                },
            });
        }
        list.push(Iface {
            name,
            tag: tag.to_string(),
            ops,
            storage,
            served: defs != 0,
        });
    }
    // An op symbol with no descriptor is a typo or a missing header: refused, never a plain call.
    let mut known: HashMap<String, (usize, u32)> = HashMap::new();
    for (i, iface) in list.iter().enumerate() {
        for (k, (op, _)) in iface.ops.iter().enumerate() {
            known.insert(op_symbol(&iface.tag, op), (i, k as u32));
        }
    }
    let fn_names = m
        .functions
        .iter()
        .map(|f| f.name.as_str())
        .chain(m.func_declarations.iter().map(|d| d.name.as_str()));
    for f in fn_names {
        if f.starts_with(OP_PREFIX) && !known.contains_key(f) {
            return unsup(format!("`{f}` names no op of any interface descriptor"));
        }
    }
    let imported_ops = known
        .into_iter()
        .filter(|(_, (i, _))| !list[*i].served)
        .collect();
    Ok(Ifaces { list, imported_ops })
}

/// What the call lowering needs of a module's interfaces.
#[derive(Default)]
pub(crate) struct Lowering {
    /// A served interface's descriptor symbol → its impl-export index.
    pub grants: HashMap<String, u32>,
    /// An imported op's symbol → `(interface, op, signature, source signature)`.
    pub ops: HashMap<String, (String, u32, FuncType, FuncType)>,
}

impl Lowering {
    pub fn new(ifaces: &Ifaces) -> Lowering {
        let grants = ifaces
            .list
            .iter()
            .filter(|i| i.served)
            .enumerate()
            .map(|(k, i)| (format!("{IFACE_PREFIX}{}", i.tag), k as u32))
            .collect();
        let ops = ifaces
            .imported_ops
            .iter()
            .map(|(sym, &(i, k))| {
                let iface = &ifaces.list[i];
                let sig = iface.ops[k as usize].1.clone();
                let storage = iface.storage[k as usize].clone();
                (sym.clone(), (iface.name.clone(), k, sig, storage))
            })
            .collect();
        Lowering { grants, ops }
    }
}

/// Intern an interface's op list into the module's type section: each op's signature as a `Func`
/// entry, then the `Interface` entry naming them (deduped, like [`crate::intern_sig`]).
pub(crate) fn intern_iface(types: &RefCell<Vec<TypeEntry>>, ops: &[(String, FuncType)]) -> u32 {
    let elems: Vec<IfaceOp> = ops
        .iter()
        .map(|(name, sig)| IfaceOp {
            name: name.clone(),
            ty: crate::intern_sig(types, sig.clone()),
        })
        .collect();
    let mut t = types.borrow_mut();
    if let Some(i) = t
        .iter()
        .position(|e| matches!(e, TypeEntry::Interface(x) if *x == elems))
    {
        return i as u32;
    }
    t.push(TypeEntry::Interface(elems));
    (t.len() - 1) as u32
}

/// Whether some served op can park: then a dispatch may start while another's handler is parked,
/// and each needs a data stack of its own (#2964). Decided by the predicate every engine asks of a
/// serve handler ([`handler_never_parks`]), over each op's own function.
pub(crate) fn parks(ifaces: &Ifaces, name2idx: &HashMap<String, u32>, funcs: &[Func]) -> bool {
    ifaces
        .list
        .iter()
        .filter(|i| i.served)
        .flat_map(|i| i.ops.iter().map(|(op, _)| name2idx[&op_symbol(&i.tag, op)]))
        .any(|f| !handler_never_parks(funcs, f))
}

/// Each served interface's impl-export. Its ops are trampolines appended to `funcs`: each takes the
/// op's own signature, `cap` slots and all, and calls the op's source function with a data-SP (§3d).
/// That is the serve stack (`serve_sp`) when no op can park; otherwise (`heap` = the program's
/// `(malloc, free)`) a [`SERVE_STACK_SIZE`] stack of the dispatch's own, allocated on the serve
/// stack, which a call that never parks only borrows.
pub(crate) fn impl_exports(
    ifaces: &Ifaces,
    name2idx: &HashMap<String, u32>,
    serve_sp: u64,
    heap: Option<(u32, u32)>,
    types: &RefCell<Vec<TypeEntry>>,
    funcs: &mut Vec<Func>,
) -> Vec<ImplExport> {
    let mut out = Vec::new();
    for iface in ifaces.list.iter().filter(|i| i.served) {
        let mut ops = Vec::with_capacity(iface.ops.len());
        for ((op, sig), storage) in iface.ops.iter().zip(&iface.storage) {
            let target = name2idx[&op_symbol(&iface.tag, op)];
            ops.push(funcs.len() as u32);
            funcs.push(match heap {
                None => trampoline(target, sig, storage, serve_sp),
                Some((malloc, free)) => {
                    heap_trampoline(target, sig, storage, serve_sp, malloc, free)
                }
            });
        }
        out.push(ImplExport {
            name: iface.name.clone(),
            interface: intern_iface(types, &iface.ops),
            ops,
            threaded: false,
        });
    }
    out
}

/// One block: call the op on the serve stack.
fn trampoline(target: u32, sig: &FuncType, storage: &FuncType, serve_sp: u64) -> Func {
    // v0..n are the op's arguments; `next` is the next value an instruction defines.
    let n = sig.params.len() as u32;
    let mut b = Body {
        insts: vec![Inst::ConstI64(serve_sp as i64)],
        next: n + 1,
    };
    let results = b.call_op(target, n, 0, sig, storage);
    Func {
        params: sig.params.clone(),
        results: sig.results.clone(),
        blocks: vec![Block {
            parent: 0,
            captures: vec![],
            params: sig.params.clone(),
            insts: b.insts,
            term: Terminator::Return(results),
        }],
    }
}

/// Three blocks: allocate the dispatch's stack (block 0), call the op on it and free it (block 1),
/// or trap when the heap has no room for it (block 2), as a fiber past its budget does.
fn heap_trampoline(
    target: u32,
    sig: &FuncType,
    storage: &FuncType,
    serve_sp: u64,
    malloc: u32,
    free: u32,
) -> Func {
    let n = sig.params.len() as u32;
    // Block 0: v0..n the arguments; vn the serve stack, vn+1 the size, vn+2 the stack, vn+3 null?
    let entry = Block {
        parent: 0,
        captures: vec![],
        params: sig.params.clone(),
        insts: vec![
            Inst::ConstI64(serve_sp as i64),
            Inst::ConstI64(SERVE_STACK_SIZE as i64),
            Inst::Call {
                func: malloc,
                args: vec![n, n + 1].into(),
            },
            Inst::Eqz {
                ty: temen_ir::IntTy::I64,
                a: n + 2,
            },
        ],
        term: Terminator::BrIf {
            cond: n + 3,
            then_blk: 2,
            then_args: vec![],
            else_blk: 1,
            else_args: std::iter::once(n + 2).chain(0..n).collect(),
        },
    };
    // Block 1: v0 the stack, v1..=n the arguments.
    let mut params = vec![ValType::I64];
    params.extend_from_slice(&sig.params);
    let mut b = Body {
        insts: Vec::new(),
        next: n + 1,
    };
    let results = b.call_op(target, 0, 1, sig, storage);
    let sp = b.push(Inst::ConstI64(serve_sp as i64));
    b.insts.push(Inst::Call {
        func: free,
        args: vec![sp, 0].into(),
    });
    let body = Block {
        parent: 0,
        captures: vec![],
        params,
        insts: b.insts,
        term: Terminator::Return(results),
    };
    let fail = Block {
        parent: 0,
        captures: vec![],
        params: vec![],
        insts: vec![],
        term: Terminator::Unreachable,
    };
    Func {
        params: sig.params.clone(),
        results: sig.results.clone(),
        blocks: vec![entry, body, fail],
    }
}

/// A one-block body under construction: its instructions and the next value one defines.
struct Body {
    insts: Vec<Inst>,
    next: u32,
}

impl Body {
    /// Append `inst`, which defines one value, and return it.
    fn push(&mut self, inst: Inst) -> u32 {
        self.insts.push(inst);
        self.next += 1;
        self.next - 1
    }

    /// Call op function `target` with data-SP `stack` and the op's arguments from value `args`
    /// on, each converted to its source storage; the op's results, converted back.
    fn call_op(
        &mut self,
        target: u32,
        stack: u32,
        args: u32,
        sig: &FuncType,
        storage: &FuncType,
    ) -> Vec<u32> {
        let mut call = vec![stack];
        for (k, (&want, &have)) in storage.params.iter().zip(&sig.params).enumerate() {
            call.push(self.convert(args + k as u32, have, want));
        }
        self.insts.push(Inst::Call {
            func: target,
            args: call.into(),
        });
        let first = self.next;
        self.next += sig.results.len() as u32;
        storage
            .results
            .iter()
            .zip(&sig.results)
            .enumerate()
            .map(|(k, (&have, &want))| self.convert(first + k as u32, have, want))
            .collect()
    }

    /// Value `v` of type `have` as a value of type `want` ([`conversion`]).
    fn convert(&mut self, v: u32, have: ValType, want: ValType) -> u32 {
        match conversion(have, want) {
            Some(op) => self.push(Inst::Convert { op, a: v }),
            None => v,
        }
    }
}

/// The conversion from a value of type `have` to one of type `want`, where one is a `cap` slot and
/// the other its `i64` source storage; `None` when the two hold a handle alike. A handle is an
/// `i32`, and an `i64` holding one is its sign extension.
pub(crate) fn conversion(have: ValType, want: ValType) -> Option<temen_ir::ConvOp> {
    match (have, want) {
        (ValType::Cap | ValType::I32, ValType::I64) => Some(temen_ir::ConvOp::ExtendI32S),
        (ValType::I64, ValType::Cap | ValType::I32) => Some(temen_ir::ConvOp::WrapI64),
        _ => None,
    }
}

/// Check that a source function's signature is the op's, a `cap` slot being an `i32`.
pub(crate) fn check_sig(
    what: &str,
    params: &[Parameter],
    ret: &Type,
    types: &Types,
    want: &FuncType,
) -> Result<FuncType, Error> {
    let mut ps = Vec::new();
    for p in params {
        ps.extend(param_vtypes(p.ty.as_ref())?);
    }
    let rs = result_types(ret, types)?;
    let fits = |have: &[ValType], want: &[ValType]| {
        have.len() == want.len()
            && have.iter().zip(want).all(|(&h, &w)| {
                h == w || (w == ValType::Cap && matches!(h, ValType::I32 | ValType::I64))
            })
    };
    if !fits(&ps, &want.params) || !fits(&rs, &want.results) {
        return unsup(format!(
            "`{what}` is declared ({ps:?}) -> ({rs:?}) but its interface says ({:?}) -> ({:?}); a \
             `cap` is an `int` / `i32`",
            want.params, want.results
        ));
    }
    Ok(FuncType {
        params: ps,
        results: rs,
    })
}

/// A signature's source form when nothing declares it: each `cap` an `i32`.
fn narrow(ts: &[ValType]) -> Vec<ValType> {
    ts.iter()
        .map(|&t| if t == ValType::Cap { ValType::I32 } else { t })
        .collect()
}

fn is_ident(s: &str) -> bool {
    !s.is_empty()
        && !s.contains("__")
        && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !s.starts_with(|c: char| c.is_ascii_digit())
}

/// Parse `name { op: (t, …) -> (t, …); … }`: the interface's name, and its ops in order.
pub(crate) fn parse_descriptor(text: &str) -> Result<(String, Vec<(String, FuncType)>), String> {
    let (name, rest) = text
        .split_once('{')
        .ok_or("expected `name { op: type; … }`")?;
    let name = name.trim();
    let body = rest
        .trim_end()
        .strip_suffix('}')
        .ok_or("expected `}` closing the op list")?;
    if name.is_empty() || name.contains(|c: char| c.is_whitespace() || "{};:".contains(c)) {
        return Err(format!("`{name}` is not an interface name"));
    }
    let mut ops: Vec<(String, FuncType)> = Vec::new();
    for item in body.split(';').map(str::trim).filter(|s| !s.is_empty()) {
        let (op, ty) = item
            .split_once(':')
            .ok_or(format!("`{item}`: expected `op: type`"))?;
        let op = op.trim();
        if !is_ident(op) {
            return Err(format!("`{op}` is not an op name"));
        }
        if ops.iter().any(|(n, _)| n == op) {
            return Err(format!("op `{op}` is declared twice"));
        }
        let sig = temen_text::parse_func_type(ty).map_err(|e| format!("op `{op}`: {}", e.0))?;
        if let Some(t) = sig.params.iter().chain(&sig.results).find(|t| {
            !matches!(
                t,
                ValType::I32 | ValType::I64 | ValType::F32 | ValType::F64 | ValType::Cap
            )
        }) {
            return Err(format!("op `{op}`: a `{t:?}` slot (i32 i64 f32 f64 cap)"));
        }
        ops.push((op.to_string(), sig));
    }
    if ops.is_empty() {
        return Err("an interface has at least one op".into());
    }
    Ok((name.to_string(), ops))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_descriptor_parses_its_name_and_its_ops_in_order() {
        let (name, ops) = parse_descriptor(
            " exec.host { run: (i64, i64) -> (i64); pass: (cap)->(cap) ;ping: () -> () } ",
        )
        .unwrap();
        assert_eq!(name, "exec.host");
        let names: Vec<&str> = ops.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["run", "pass", "ping"]);
        assert_eq!(ops[1].1.params, [ValType::Cap]);
        assert_eq!(ops[1].1.results, [ValType::Cap]);
        assert!(ops[2].1.results.is_empty());
    }

    #[test]
    fn a_malformed_descriptor_is_refused() {
        for bad in [
            "",
            "x { }",
            "{ run: () -> () }",
            "x { run }",
            "x { run: (i8) -> () }",
            "x { run: () -> i64 }",
            "x { run: (v128) -> () }",
            "x { a: () -> (); a: () -> () }",
            "x { y__z: () -> () }",
            "x { run: () -> () junk }",
            "a b { run: () -> () }",
        ] {
            assert!(parse_descriptor(bad).is_err(), "{bad:?}");
        }
    }
}
