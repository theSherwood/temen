//! #2961 — **interfaces from C and Rust**: a module serves an interface its children call by name,
//! or imports one, through the substrate's own ops. Nothing here is a second serving mechanism: a
//! served interface is an [`temen_ir::ImplExport`], an imported one a grouped manifest import that a
//! `call.import` dispatches through, and a handle among an op's arguments or results is a `cap`
//! slot, which every engine translates between the two domains (INVARIANTS #8).
//!
//! The source names an interface with a **descriptor**: a constant byte string under the symbol
//! `__temen_iface__<name>`, holding its ops in order,
//!
//! ```text
//! run(i64, i64) -> (i64); pass(cap) -> (cap); ping()
//! ```
//!
//! each op's parameter and result types drawn from `i32 i64 f32 f64 cap`, a missing `-> (…)` meaning
//! no results. Each op is a function named `__temen_op__<name>__<op>`. A module that **defines** every
//! op serves the interface; one that only **declares** them imports it, and a call of a declared op
//! is a `call.import` of that op. A mix is refused. A `cap` is an `int` / `i32` in the source; two or
//! more results are a small struct returned by value (the §3a multi-result convention).
//!
//! A served op runs on the module's **serve stack**, a data stack of its own: the serve loop that
//! dispatches it is suspended mid-frame, so the op cannot share the loop's stack. One serve stack is
//! enough only while no two handlers are live at once, which holds exactly when no handler can park
//! (a parked handler lets its serve loop admit the next call). So an op that can reach a park seam
//! is refused at translation, through the seam scan every engine shares ([`temen_ir::seams`]).

use std::cell::RefCell;
use std::collections::HashMap;

use temen_ir::seams::handler_never_parks;
use temen_ir::{
    Block, Func, FuncType, IfaceOp, ImplExport, Inst, Module, Terminator, TypeEntry, ValType,
};

use crate::ll::ast::{Module as LModule, Parameter, Type};
use crate::{param_vtypes, result_types, unsup, Error, Types};

/// The descriptor symbol prefix: `__temen_iface__<name>`.
pub(crate) const IFACE_PREFIX: &str = "__temen_iface__";
/// The op symbol prefix: `__temen_op__<iface>__<op>`.
pub(crate) const OP_PREFIX: &str = "__temen_op__";
/// The serve stack's size: a [`temen_ir::FIBER_STACK`], what a handler fiber is charged on the
/// oracle.
pub(crate) const SERVE_STACK_SIZE: u64 = 256 * 1024;

/// One interface a module names.
pub(crate) struct Iface {
    pub name: String,
    /// The ops in declaration order: their names and signatures, a handle slot typed `cap`.
    pub ops: Vec<(String, FuncType)>,
    /// Each op's signature in the source, where a `cap` slot is the `i32` (or, widened by the
    /// frontend's struct-return coercion, the `i64`) holding the handle: what the op's function
    /// takes and answers ([`check_sig`]). An imported op never declared is the `i32` form.
    pub storage: Vec<FuncType>,
    /// Whether this module serves it (defines its ops) rather than importing it.
    pub served: bool,
}

/// Every interface a module names, sorted by name, so a served interface's export index (the
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
pub(crate) fn op_symbol(iface: &str, op: &str) -> String {
    format!("{OP_PREFIX}{iface}__{op}")
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
        let name = &sym[IFACE_PREFIX.len()..];
        if !is_ident(name) {
            return unsup(format!(
                "interface descriptor `{sym}`: `{name}` is not a name"
            ));
        }
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        let text = std::str::from_utf8(&bytes[..end]).map_err(|_| {
            Error::Unsupported(format!("interface `{name}`: descriptor is not UTF-8"))
        })?;
        let ops = parse_descriptor(text)
            .map_err(|e| Error::Unsupported(format!("interface `{name}`: {e}")))?;
        let defs = ops
            .iter()
            .filter(|(op, _)| defined.contains_key(&op_symbol(name, op)))
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
            let sym = op_symbol(name, op);
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
            name: name.to_string(),
            ops,
            storage,
            served: defs != 0,
        });
    }
    // An op symbol with no descriptor is a typo or a missing header: refused, never a plain call.
    let mut known: HashMap<String, (usize, u32)> = HashMap::new();
    for (i, iface) in list.iter().enumerate() {
        for (k, (op, _)) in iface.ops.iter().enumerate() {
            known.insert(op_symbol(&iface.name, op), (i, k as u32));
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
            .map(|(k, i)| (format!("{IFACE_PREFIX}{}", i.name), k as u32))
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

/// Each served interface's impl-export. Its ops are trampolines appended to `funcs`: each takes the
/// op's own signature, `cap` slots and all, and calls the op's source function on the serve stack
/// (`serve_sp`), the data-SP every translated function takes first (§3d).
pub(crate) fn impl_exports(
    ifaces: &Ifaces,
    name2idx: &HashMap<String, u32>,
    serve_sp: u64,
    types: &RefCell<Vec<TypeEntry>>,
    funcs: &mut Vec<Func>,
) -> Vec<ImplExport> {
    let mut out = Vec::new();
    for iface in ifaces.list.iter().filter(|i| i.served) {
        let mut ops = Vec::with_capacity(iface.ops.len());
        for ((op, sig), storage) in iface.ops.iter().zip(&iface.storage) {
            let target = name2idx[&op_symbol(&iface.name, op)];
            ops.push(funcs.len() as u32);
            funcs.push(trampoline(target, sig, storage, serve_sp));
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

fn trampoline(target: u32, sig: &FuncType, storage: &FuncType, serve_sp: u64) -> Func {
    // v0..n are the op's arguments; `next` is the next value an instruction defines.
    let n = sig.params.len() as u32;
    let mut b = Body {
        insts: vec![Inst::ConstI64(serve_sp as i64)],
        next: n + 1,
    };
    let mut args = vec![n];
    for (k, (&want, &have)) in storage.params.iter().zip(&sig.params).enumerate() {
        args.push(b.convert(k as u32, have, want));
    }
    b.insts.push(Inst::Call {
        func: target,
        args: args.into(),
    });
    let first = b.next;
    b.next += sig.results.len() as u32;
    let mut results = Vec::with_capacity(sig.results.len());
    for (k, (&have, &want)) in storage.results.iter().zip(&sig.results).enumerate() {
        results.push(b.convert(first + k as u32, have, want));
    }
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

/// A one-block body under construction: its instructions and the next value one defines.
struct Body {
    insts: Vec<Inst>,
    next: u32,
}

impl Body {
    /// Value `v` of type `have` as a value of type `want` ([`conversion`]).
    fn convert(&mut self, v: u32, have: ValType, want: ValType) -> u32 {
        match conversion(have, want) {
            Some(op) => {
                self.insts.push(Inst::Convert { op, a: v });
                self.next += 1;
                self.next - 1
            }
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

/// Refuse a module whose served op can park ([`SERVE_STACK_SIZE`]'s one stack): decided by the
/// predicate every engine asks of a serve handler ([`temen_ir::seams::handler_never_parks`]).
pub(crate) fn check_handlers(m: &Module) -> Result<(), Error> {
    for e in &m.impl_exports {
        let Some(k) = e
            .ops
            .iter()
            .position(|&f| !handler_never_parks(&m.funcs, f))
        else {
            continue;
        };
        let op = match m.types.get(e.interface as usize) {
            Some(TypeEntry::Interface(ops)) => ops.get(k).map_or("?", |o| o.name.as_str()),
            _ => "?",
        };
        return unsup(format!(
            "interface `{}`: op `{op}` can park (it reaches a blocking call, an import, a spawn or \
             join, an indirect call, or a fiber or thread op); a served op must run to completion, \
             since every op of a module runs on its one serve stack",
            e.name
        ));
    }
    Ok(())
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

/// Parse `op(t, …) -> (t, …); …`.
pub(crate) fn parse_descriptor(text: &str) -> Result<Vec<(String, FuncType)>, String> {
    let mut ops: Vec<(String, FuncType)> = Vec::new();
    for item in text.split(';').map(str::trim).filter(|s| !s.is_empty()) {
        let open = item
            .find('(')
            .ok_or(format!("`{item}`: expected `op(…)`"))?;
        let name = item[..open].trim();
        if !is_ident(name) {
            return Err(format!("`{name}` is not an op name"));
        }
        if ops.iter().any(|(n, _)| n == name) {
            return Err(format!("op `{name}` is declared twice"));
        }
        let rest = &item[open..];
        let close = rest.find(')').ok_or(format!("`{item}`: unclosed `(`"))?;
        let params = types_of(&rest[1..close])?;
        let tail = rest[close + 1..].trim();
        let results = if tail.is_empty() {
            Vec::new()
        } else {
            let r = tail
                .strip_prefix("->")
                .map(str::trim)
                .and_then(|r| r.strip_prefix('('))
                .and_then(|r| r.strip_suffix(')'))
                .ok_or(format!("`{item}`: expected `-> (…)` after the parameters"))?;
            types_of(r)?
        };
        ops.push((name.to_string(), FuncType { params, results }));
    }
    if ops.is_empty() {
        return Err("an interface has at least one op".into());
    }
    Ok(ops)
}

fn types_of(list: &str) -> Result<Vec<ValType>, String> {
    list.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|t| match t {
            "i32" => Ok(ValType::I32),
            "i64" => Ok(ValType::I64),
            "f32" => Ok(ValType::F32),
            "f64" => Ok(ValType::F64),
            "cap" => Ok(ValType::Cap),
            other => Err(format!(
                "`{other}` is not a slot type (i32 i64 f32 f64 cap)"
            )),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_descriptor_parses_in_order_with_caps_and_no_results() {
        let ops = parse_descriptor(" run(i64, i64) -> (i64); pass(cap)->(cap) ;ping() ").unwrap();
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
            "run",
            "run(i8)",
            "run() -> i64",
            "a(); a()",
            "x__y()",
            "run() junk",
        ] {
            assert!(parse_descriptor(bad).is_err(), "{bad:?}");
        }
    }
}
