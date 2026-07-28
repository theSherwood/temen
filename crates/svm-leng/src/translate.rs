//! Leng-NIF → SVM-text translation for the integer/arithmetic/call/local subset (NIM.md Phase 2).
//!
//! Straight-line bodies with `ret`; anything else is a fail-closed [`LengError::Unsupported`]. The
//! model mirrors chibicc's `codegen_ir.c`: emit SVM text, let `svm_text::parse_module` build the IR.
//! Non-address-taken scalar locals map directly to SSA values (no data-stack frame needed for this
//! subset — that arrives with `addr`/aggregates), so a `var`/`asgn` just (re)binds a name to a value.

use std::collections::HashMap;

use svm_ir::ValType;

use crate::{LengError, Node, Val};

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

/// A proc's SVM-visible signature, collected in a first pass so calls can resolve names → indices.
struct Sig {
    index: u32,
    params: Vec<ValType>,
    ret: Option<ValType>, // None = void
}

pub(crate) struct Translator {
    procs: HashMap<String, Sig>,
}

impl Translator {
    pub fn new() -> Self {
        Translator {
            procs: HashMap::new(),
        }
    }

    /// Translate a `(stmts TopLevelConstruct*)` module to SVM text.
    pub fn module(&mut self, root: &Node) -> Result<String, LengError> {
        if root.tag() != Some("stmts") {
            return Err(LengError::Malformed(format!(
                "module root must be (stmts …), got {:?}",
                root.tag()
            )));
        }
        // Pass 1: collect proc signatures (index = order of appearance among procs).
        let mut proc_nodes = Vec::new();
        for item in root.args() {
            match item.tag() {
                Some("proc") => {
                    let (name, params, ret) = self.proc_sig(item)?;
                    let index = proc_nodes.len() as u32;
                    self.procs.insert(name, Sig { index, params, ret });
                    proc_nodes.push(item);
                }
                // Top-level constructs other than procs (globals, types, bare stmts) are out of
                // scope for the skeleton — fail closed rather than silently drop them.
                Some(other) => {
                    return Err(LengError::Unsupported(format!(
                        "top-level construct `{other}` (only `proc` is supported)"
                    )))
                }
                None => {
                    if !item.is_empty_marker() {
                        return Err(LengError::Malformed("headless top-level node".into()));
                    }
                }
            }
        }

        // Pass 2: emit each proc.
        let mut out = String::new();
        for p in &proc_nodes {
            self.proc_body(p, &mut out)?;
        }
        Ok(out)
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
            out.push((sym_def(&pa[0])?, int_ty(&pa[2])?));
        }
        Ok(out)
    }

    /// Return type: `(void)` → None, otherwise an integer ValType.
    fn ret_ty(&self, node: &Node) -> Result<Option<ValType>, LengError> {
        if node.tag() == Some("void") || node.is_empty_marker() {
            return Ok(None);
        }
        Ok(Some(int_ty(node)?))
    }

    fn proc_body(&self, p: &Node, out: &mut String) -> Result<(), LengError> {
        let a = p.args();
        let name = sym_def(&a[0])?;
        let params = self.params(&a[1])?;
        let ret = self.ret_ty(&a[2])?;
        let body = a.get(4);

        // Signature line.
        let ptys: Vec<&str> = params.iter().map(|(_, t)| prefix(*t)).collect();
        let rty = ret.map(|t| prefix(t).to_string()).unwrap_or_default();
        out.push_str(&format!("func ({}) -> ({}) {{\n", ptys.join(", "), rty));

        let mut f = FuncGen::new(self, ret);
        // Block 0 params = the proc params, bound to v0..v(n-1).
        let block_params: Vec<String> = params
            .iter()
            .enumerate()
            .map(|(i, (nm, t))| {
                f.bind(
                    nm,
                    Val {
                        id: i as u32,
                        ty: *t,
                    },
                );
                format!("v{i}: {}", prefix(*t))
            })
            .collect();
        f.next = params.len() as u32;
        out.push_str(&format!("block 0 ({}) {{\n", block_params.join(", ")));

        match body {
            None | Some(Node::Atom(_)) => {
                // No body / `.` — a forward decl. The skeleton has nothing to import against, so an
                // empty proc just returns (only valid for void).
                f.terminate_fallthrough(out)?;
            }
            Some(b) => {
                f.stmt_list(b, out)?;
                if !f.terminated {
                    f.terminate_fallthrough(out)?;
                }
            }
        }
        out.push_str("  }\n}\n");
        let _ = name;
        Ok(())
    }
}

/// Per-function emission state: the value counter, the name→Val environment, and whether the
/// current (single, straight-line) block already has a terminator.
struct FuncGen<'a> {
    t: &'a Translator,
    ret: Option<ValType>,
    next: u32,
    env: Vec<(String, Val)>,
    terminated: bool,
}

impl<'a> FuncGen<'a> {
    fn new(t: &'a Translator, ret: Option<ValType>) -> Self {
        FuncGen {
            t,
            ret,
            next: 0,
            env: Vec::new(),
            terminated: false,
        }
    }

    fn fresh(&mut self) -> u32 {
        let id = self.next;
        self.next += 1;
        id
    }
    fn bind(&mut self, name: &str, v: Val) {
        // Rebind (SSA update) if the name exists, else push. Last binding wins on lookup.
        if let Some(slot) = self.env.iter_mut().rev().find(|(n, _)| n == name) {
            slot.1 = v;
        } else {
            self.env.push((name.to_string(), v));
        }
    }
    fn lookup(&self, name: &str) -> Option<Val> {
        self.env
            .iter()
            .rev()
            .find(|(n, _)| n == name)
            .map(|(_, v)| *v)
    }

    fn terminate_fallthrough(&mut self, out: &mut String) -> Result<(), LengError> {
        match self.ret {
            None => {
                out.push_str("  return\n");
                self.terminated = true;
                Ok(())
            }
            Some(_) => Err(LengError::Malformed(
                "non-void proc falls off the end without `ret`".into(),
            )),
        }
    }

    /// `StmtList ::= (stmts SCOPE Stmt*)` — the SCOPE child (first arg) is a scope marker we ignore.
    fn stmt_list(&mut self, node: &Node, out: &mut String) -> Result<(), LengError> {
        if node.is_empty_marker() {
            return Ok(());
        }
        if node.tag() != Some("stmts") {
            return Err(LengError::Malformed("expected (stmts …) body".into()));
        }
        // args()[0] is the scope symbol/`.`; statements follow.
        for stmt in node.args().iter().skip(1) {
            if self.terminated {
                // Dead code after a terminator — Leng shouldn't emit it; ignore defensively.
                break;
            }
            self.stmt(stmt, out)?;
        }
        Ok(())
    }

    fn stmt(&mut self, s: &Node, out: &mut String) -> Result<(), LengError> {
        match s.tag() {
            Some("ret") => {
                let a = s.args();
                if a.is_empty() || a[0].is_empty_marker() {
                    out.push_str("  return\n");
                } else {
                    let v = self.expr(&a[0], out)?;
                    out.push_str(&format!("  return v{}\n", v.id));
                }
                self.terminated = true;
                Ok(())
            }
            Some("var") => {
                // (var :Sym Pragmas Type [Empty | Expr])
                let a = s.args();
                if a.len() < 3 {
                    return Err(LengError::Malformed("var needs :name pragmas type".into()));
                }
                let name = sym_def(&a[0])?;
                let ty = int_ty(&a[2])?;
                let v = match a.get(3) {
                    Some(init) if !init.is_empty_marker() => self.expr(init, out)?,
                    _ => self.emit_const(ty, 0, out), // default-init to 0
                };
                if v.ty != ty {
                    // Width mismatch between the declared type and the initializer: convert.
                    let cv = self.convert(v, ty, out);
                    self.bind(&name, cv);
                } else {
                    self.bind(&name, v);
                }
                Ok(())
            }
            Some("asgn") => {
                // (asgn Lvalue Expr) — skeleton lvalue = a plain local symbol.
                let a = s.args();
                if a.len() != 2 {
                    return Err(LengError::Malformed("asgn needs lvalue and expr".into()));
                }
                let name = a[0]
                    .as_atom()
                    .ok_or_else(|| LengError::Unsupported("asgn to non-symbol lvalue".into()))?;
                if self.lookup(name).is_none() {
                    return Err(LengError::Unsupported(format!(
                        "asgn to unknown local `{name}`"
                    )));
                }
                let v = self.expr(&a[1], out)?;
                self.bind(name, v);
                Ok(())
            }
            Some("discard") => {
                // Evaluate for side effects, drop the value.
                if let Some(e) = s.args().first() {
                    if !e.is_empty_marker() {
                        self.expr(e, out)?;
                    }
                }
                Ok(())
            }
            Some("call") => {
                // Call in statement position (return value, if any, discarded).
                self.call(s, out)?;
                Ok(())
            }
            other => Err(LengError::Unsupported(format!(
                "statement `{}`",
                other.unwrap_or("<headless>")
            ))),
        }
    }

    fn expr(&mut self, e: &Node, out: &mut String) -> Result<Val, LengError> {
        match e {
            Node::Atom(a) => {
                if let Some(v) = self.lookup(a) {
                    return Ok(v);
                }
                if let Ok(n) = parse_int(a) {
                    // Bare integer literal — default to i64 (arithmetic nodes carry the real width;
                    // a literal used directly is rare). Callers with a type context override.
                    return Ok(self.emit_const(ValType::I64, n, out));
                }
                Err(LengError::Unsupported(format!("atom expression `{a}`")))
            }
            Node::List(_) => match e.tag() {
                Some(op @ ("add" | "sub" | "mul" | "div" | "mod")) => self.arith(op, e, out),
                Some("neg") => {
                    // (neg Type Expr) → 0 - x
                    let a = e.args();
                    let ty = int_ty(&a[0])?;
                    let x = self.expr_typed(&a[1], ty, out)?;
                    let zero = self.emit_const(ty, 0, out);
                    Ok(self.emit_bin("sub", ty, zero, x, out))
                }
                Some("conv") => {
                    // (conv Type Expr) — value-preserving numeric conversion (int widths here).
                    let a = e.args();
                    let ty = int_ty(&a[0])?;
                    let x = self.expr(&a[1], out)?;
                    Ok(self.convert(x, ty, out))
                }
                Some("par") => self.expr(&e.args()[0], out),
                Some("call") => self.call(e, out),
                other => Err(LengError::Unsupported(format!(
                    "expression `{}`",
                    other.unwrap_or("<headless>")
                ))),
            },
        }
    }

    /// Evaluate an expression, coercing an untyped integer literal to `want`'s width.
    fn expr_typed(&mut self, e: &Node, want: ValType, out: &mut String) -> Result<Val, LengError> {
        if let Node::Atom(a) = e {
            if self.lookup(a).is_none() {
                if let Ok(n) = parse_int(a) {
                    return Ok(self.emit_const(want, n, out));
                }
            }
        }
        let v = self.expr(e, out)?;
        Ok(if v.ty != want {
            self.convert(v, want, out)
        } else {
            v
        })
    }

    /// `(add|sub|mul|div|mod Type Expr Expr)` → the matching `iN.*` op.
    fn arith(&mut self, op: &str, e: &Node, out: &mut String) -> Result<Val, LengError> {
        let a = e.args();
        if a.len() != 3 {
            return Err(LengError::Malformed(format!(
                "`{op}` needs Type and two operands"
            )));
        }
        let (ty, signed) = int_ty_signed(&a[0])?;
        let l = self.expr_typed(&a[1], ty, out)?;
        let r = self.expr_typed(&a[2], ty, out)?;
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
        Ok(self.emit_bin(name, ty, l, r, out))
    }

    /// `(call Callee Expr*)` — direct call to a named proc.
    fn call(&mut self, e: &Node, out: &mut String) -> Result<Val, LengError> {
        let a = e.args();
        if a.is_empty() {
            return Err(LengError::Malformed("call needs a callee".into()));
        }
        let callee = a[0].as_atom().ok_or_else(|| {
            LengError::Unsupported("indirect call (callee is not a symbol)".into())
        })?;
        let sig = self.t.procs.get(callee).ok_or_else(|| {
            LengError::Unsupported(format!("call to unknown/external proc `{callee}`"))
        })?;
        let index = sig.index;
        let ptys = sig.params.clone();
        let ret = sig.ret;
        if a.len() - 1 != ptys.len() {
            return Err(LengError::Malformed(format!(
                "call to `{callee}`: {} args for {} params",
                a.len() - 1,
                ptys.len()
            )));
        }
        let mut argvals = Vec::new();
        for (arg, want) in a[1..].iter().zip(ptys) {
            argvals.push(self.expr_typed(arg, want, out)?.id);
        }
        let arglist = argvals
            .iter()
            .map(|id| format!("v{id}"))
            .collect::<Vec<_>>()
            .join(", ");
        match ret {
            Some(ty) => {
                let id = self.fresh();
                out.push_str(&format!("  v{id} = call {index} ({arglist})\n"));
                Ok(Val { id, ty })
            }
            None => {
                out.push_str(&format!("  call {index} ({arglist})\n"));
                // A void call yields no value; callers in expr position shouldn't reach here.
                Ok(Val {
                    id: u32::MAX,
                    ty: ValType::I32,
                })
            }
        }
    }

    fn emit_const(&mut self, ty: ValType, n: i64, out: &mut String) -> Val {
        let id = self.fresh();
        out.push_str(&format!("  v{id} = {}.const {n}\n", prefix(ty)));
        Val { id, ty }
    }

    fn emit_bin(&mut self, op: &str, ty: ValType, l: Val, r: Val, out: &mut String) -> Val {
        let id = self.fresh();
        out.push_str(&format!(
            "  v{id} = {}.{op} v{} v{}\n",
            prefix(ty),
            l.id,
            r.id
        ));
        Val { id, ty }
    }

    /// Integer width conversion between i32/i64 (value-preserving, signed).
    fn convert(&mut self, v: Val, to: ValType, out: &mut String) -> Val {
        if v.ty == to {
            return v;
        }
        let id = self.fresh();
        let insn = match (v.ty, to) {
            (ValType::I32, ValType::I64) => format!("i64.extend_i32_s v{}", v.id),
            (ValType::I64, ValType::I32) => format!("i32.wrap_i64 v{}", v.id),
            // Non-integer conversions aren't produced by this subset.
            _ => format!("i64.extend_i32_s v{}", v.id),
        };
        out.push_str(&format!("  v{id} = {insn}\n"));
        Val { id, ty: to }
    }
}

// ---------------------------------------------------------------------------
// Type + atom helpers.
// ---------------------------------------------------------------------------

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
            // A bare symbol type (named/nominal) — aggregates/pointers, out of scope.
            Some(sym) => Err(LengError::Unsupported(format!("named type `{sym}`"))),
            None => Err(LengError::Malformed("expected a type".into())),
        },
    }
}

/// `IntBits ::= ('+'|'-') [0-9]+`, where `-1` means machine word size (→ 64).
fn int_bits(s: &str) -> Result<u32, LengError> {
    let stripped = s.strip_prefix('+').unwrap_or(s);
    if stripped == "-1" {
        return Ok(64); // machine word
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

/// Parse a Leng integer literal (decimal, optional sign). Hex/suffix forms aren't in this subset.
fn parse_int(s: &str) -> Result<i64, ()> {
    s.parse::<i64>().map_err(|_| ())
}
