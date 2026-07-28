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
//! and a call to a frame-needing proc passes `sp + frame_size` as the callee's frame. Aggregates
//! (`at`/`dot` over arrays/objects) build on this and are a later slice.

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

/// A proc's SVM-visible signature, collected so calls can resolve names → indices.
struct Sig {
    index: u32,
    params: Vec<ValType>,
    ret: Option<ValType>, // None = void
    /// True if the proc takes the address of a local, so its emitted signature has a leading
    /// stack-pointer param and callers must pass a fresh frame.
    needs_frame: bool,
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

    /// Translate a `(stmts TopLevelConstruct*)` module to SVM text (all procs).
    pub fn module(&mut self, root: &Node) -> Result<String, LengError> {
        if root.tag() != Some("stmts") {
            return Err(LengError::Malformed(format!(
                "module root must be (stmts …), got {:?}",
                root.tag()
            )));
        }
        let mut proc_nodes = Vec::new();
        for item in root.args() {
            match item.tag() {
                Some("proc") => {
                    let (name, params, ret) = self.proc_sig(item)?;
                    let index = proc_nodes.len() as u32;
                    let needs_frame = proc_needs_frame(item);
                    self.procs.insert(
                        name,
                        Sig {
                            index,
                            params,
                            ret,
                            needs_frame,
                        },
                    );
                    proc_nodes.push(item);
                }
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
        let mut out = String::new();
        let mut used_memory = false;
        for p in &proc_nodes {
            used_memory |= self.proc_body(p, &mut out)?;
        }
        Ok(with_memory(used_memory, out))
    }

    /// Translate a **single named proc** as func 0 (NIM.md Phase 2 "go deep"): the real hexer output
    /// carries `gvar`/`type`/other procs the skeleton can't lower yet, so we locate just the target.
    /// Only its own signature is registered, so a call to any other proc fail-closes.
    pub fn one_proc(&mut self, root: &Node, name: &str) -> Result<String, LengError> {
        if root.tag() != Some("stmts") {
            return Err(LengError::Malformed("module root must be (stmts …)".into()));
        }
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
            let (n, params, ret) = self.proc_sig(item)?;
            let needs_frame = proc_needs_frame(item);
            self.procs.insert(
                n,
                Sig {
                    index: 0,
                    params,
                    ret,
                    needs_frame,
                },
            );
            let mut out = String::new();
            let used_memory = self.proc_body(item, &mut out)?;
            return Ok(with_memory(used_memory, out));
        }
        Err(LengError::Malformed(format!(
            "proc `{name}` not found in module"
        )))
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

    /// Return type: `(void)`/`.` → None, otherwise an integer ValType.
    fn ret_ty(&self, node: &Node) -> Result<Option<ValType>, LengError> {
        if node.tag() == Some("void") || node.is_empty_marker() {
            return Ok(None);
        }
        Ok(Some(int_ty(node)?))
    }

    /// Emit one proc's `func {…}` into `out`; returns whether it touched the window (so the caller
    /// can emit the module `memory` declaration).
    fn proc_body(&self, p: &Node, out: &mut String) -> Result<bool, LengError> {
        let a = p.args();
        let params = self.params(&a[1])?;
        let ret = self.ret_ty(&a[2])?;
        let body = a.get(4);

        // Address-taken locals become window frame slots; the proc gains a threaded stack pointer.
        let mut addr_taken = std::collections::HashSet::new();
        if let Some(b) = body {
            collect_addr_taken(b, &mut addr_taken);
        }
        for (pn, _) in &params {
            if addr_taken.contains(pn) {
                // Spilling an address-taken *param* to the frame is a later refinement.
                return Err(LengError::Unsupported(format!(
                    "address of parameter `{pn}`"
                )));
            }
        }
        let needs_frame = !addr_taken.is_empty();

        // Every `var` in the body: address-taken ones go to the frame, the rest are SSA slots.
        let mut all_vars: Vec<(String, ValType)> = Vec::new();
        if let Some(b) = body {
            collect_vars(b, &mut all_vars)?;
        }
        // SSA slots (block-parameter set): [$sp] ++ params ++ non-address-taken vars.
        let mut slots: Vec<(String, ValType)> = Vec::new();
        if needs_frame {
            slots.push(("$sp".into(), ValType::I64));
        }
        slots.extend(params.clone());
        let mut mem: HashMap<String, (u64, ValType)> = HashMap::new();
        let mut frame_size = 0u64;
        for (vn, vt) in all_vars {
            if addr_taken.contains(&vn) {
                mem.entry(vn).or_insert_with(|| {
                    let off = frame_size;
                    frame_size += 8; // one 8-byte slot per address-taken local
                    (off, vt)
                });
            } else if !slots.iter().any(|(n, _)| n == &vn) {
                slots.push((vn, vt));
            }
        }
        let nparams = usize::from(needs_frame) + params.len();

        // Pointee types of pointer-typed locals (params + vars), for load/store width on `deref`.
        let mut pointee = HashMap::new();
        collect_pointees(&a[1], &mut pointee);
        if let Some(b) = body {
            collect_pointees(b, &mut pointee);
        }

        // Signature: a leading `i64` stack pointer when frame-needing, then the Leng params.
        let mut ptys: Vec<String> = Vec::new();
        if needs_frame {
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
            needs_frame,
            mem,
            frame_size,
        );
        // Entry block: params default to their block-param value (slot i = v i); a var not yet
        // assigned reads 0 until its declaration binds it (matches Leng default-init semantics).
        f.open_entry();
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
    /// Pointer-typed locals → their pointee value type, for `deref`/store width.
    pointee: HashMap<String, ValType>,
    /// This proc takes the address of a local, so slot 0 is a threaded stack pointer (`$sp`).
    has_sp: bool,
    /// Address-taken locals → (frame byte offset, value type). Accessed via load/store at `sp+off`.
    mem: HashMap<String, (u64, ValType)>,
    /// Total frame bytes; a call to a frame-needing proc passes `sp + frame_size` as its frame.
    frame_size: u64,
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
        pointee: HashMap<String, ValType>,
        has_sp: bool,
        mem: HashMap<String, (u64, ValType)>,
        frame_size: u64,
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
            has_sp,
            mem,
            frame_size,
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

    /// Read a local: an address-taken (frame) local emits a `load` at `sp+off`; an SSA slot returns
    /// its current value.
    fn read_local(&mut self, name: &str) -> Option<Val> {
        if let Some(&(off, ty)) = self.mem.get(name) {
            let sp = self.cur[0]; // slot 0 is $sp for a frame-needing proc
            let id = self.fresh();
            self.used_memory = true;
            self.cur_buf.push_str(&format!(
                "  v{id} = {}.load v{sp} offset={off}\n",
                prefix(ty)
            ));
            return Some(Val { id, ty });
        }
        self.lookup(name)
    }

    /// Write a local: an address-taken (frame) local emits a `store` at `sp+off`; an SSA slot rebinds.
    fn write_local(&mut self, name: &str, v: Val) -> Result<(), LengError> {
        if let Some(&(off, ty)) = self.mem.get(name) {
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

    /// `(addr x)` — the window address of a frame local: `sp + off`.
    fn addr_of(&mut self, name: &str) -> Result<Val, LengError> {
        let off = match self.mem.get(name) {
            Some(&(off, _)) => off,
            None => {
                return Err(LengError::Unsupported(format!(
                    "address of non-frame local `{name}` (only address-taken locals are framed)"
                )))
            }
        };
        let sp = self.cur[0];
        let ko = self.emit_const(ValType::I64, off as i64);
        let id = self.fresh();
        self.cur_buf
            .push_str(&format!("  v{id} = i64.add v{sp} v{}\n", ko.id));
        Ok(Val {
            id,
            ty: ValType::I64,
        })
    }

    /// The pointee value type of a pointer local `p`, plus its current value — errors if `p` isn't a
    /// known pointer local (unknown load/store width would be a silent-wrongness hazard).
    fn ptr_local(&self, name: &str) -> Result<(Val, ValType), LengError> {
        let p = self
            .lookup(name)
            .ok_or_else(|| LengError::Unsupported(format!("deref of unknown pointer `{name}`")))?;
        let pty = *self.pointee.get(name).ok_or_else(|| {
            LengError::Unsupported(format!("deref of `{name}` with unknown pointee type"))
        })?;
        Ok((p, pty))
    }

    /// `(asgn Lvalue Expr)`: a symbol lvalue rebinds a slot; a `(deref P)` lvalue stores through P.
    fn assign(&mut self, lhs: &Node, rhs: &Node) -> Result<(), LengError> {
        if lhs.tag() == Some("deref") {
            let pname = lhs
                .args()
                .first()
                .and_then(|n| n.as_atom())
                .ok_or_else(|| LengError::Unsupported("store through non-symbol pointer".into()))?;
            let (p, pty) = self.ptr_local(pname)?;
            let v = self.expr_typed(rhs, pty)?;
            self.used_memory = true;
            self.cur_buf
                .push_str(&format!("  {}.store v{} v{}\n", prefix(pty), p.id, v.id));
            return Ok(());
        }
        let name = lhs.as_atom().ok_or_else(|| {
            LengError::Unsupported(format!("assignment to lvalue `{:?}`", lhs.tag()))
        })?;
        let v = self.expr(rhs)?;
        self.write_local(name, v)
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
            if self.terminated {
                break; // dead code after a terminator
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
                let term = if a.is_empty() || a[0].is_empty_marker() {
                    "return".to_string()
                } else {
                    let v = self.expr(&a[0])?;
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
                let ty = val_ty(&a[2])?;
                let v = match a.get(3) {
                    Some(init) if !init.is_empty_marker() => self.expr(init)?,
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
                self.call(s)?;
                Ok(())
            }
            Some("if") => self.if_stmt(s),
            Some("while") => self.while_stmt(s),
            // A bare label with no live jmp target (e.g. hexer's trailing `whileStmtLabel`) is inert
            // for our structured lowering; `jmp` itself is not yet supported.
            Some("lab") => Ok(()),
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
        // Body: run, then back to header.
        self.stmt_list_or_single(&a[1])?;
        if !self.terminated {
            let bargs = self.branch_args();
            self.finish_block(format!("br {header}{bargs}"), exit);
        } else {
            self.cur_id = exit;
            self.reset_cur_state();
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

    /// Ensure a value is an `i32` condition (comparisons already produce i32; an integer coerces).
    fn as_i32_cond(&mut self, v: Val) -> u32 {
        if v.ty == ValType::I32 {
            v.id
        } else {
            // Reduce i64 to an i32 truthiness via wrap (nonzero-preserving for our comparison
            // results, which are already 0/1; general i64 conditions aren't produced by this subset).
            self.convert(v, ValType::I32).id
        }
    }

    fn expr(&mut self, e: &Node) -> Result<Val, LengError> {
        match e {
            Node::Atom(a) => {
                if let Some(v) = self.read_local(a) {
                    return Ok(v);
                }
                if let Ok(n) = parse_int(a) {
                    return Ok(self.emit_const(ValType::I64, n));
                }
                Err(LengError::Unsupported(format!("atom expression `{a}`")))
            }
            Node::List(_) => match e.tag() {
                Some(op @ ("add" | "sub" | "mul" | "div" | "mod")) => self.arith(op, e),
                Some(op @ ("eq" | "neq" | "lt" | "le")) => self.compare(op, e),
                Some("neg") => {
                    let a = e.args();
                    let ty = int_ty(&a[0])?;
                    let x = self.expr_typed(&a[1], ty)?;
                    let zero = self.emit_const(ty, 0);
                    Ok(self.emit_bin("sub", ty, zero, x))
                }
                Some("conv") => {
                    let a = e.args();
                    let ty = int_ty(&a[0])?;
                    let x = self.expr(&a[1])?;
                    Ok(self.convert(x, ty))
                }
                Some("par") => self.expr(&e.args()[0]),
                Some("addr") => {
                    let name =
                        e.args().first().and_then(|n| n.as_atom()).ok_or_else(|| {
                            LengError::Unsupported("address of a non-symbol".into())
                        })?;
                    self.addr_of(name)
                }
                Some("deref") => {
                    // `(deref P)` — load through a pointer local (width = its pointee type).
                    let a = e.args();
                    let pname = a.first().and_then(|n| n.as_atom()).ok_or_else(|| {
                        LengError::Unsupported("deref of non-symbol pointer".into())
                    })?;
                    let (p, pty) = self.ptr_local(pname)?;
                    let id = self.fresh();
                    self.used_memory = true;
                    self.cur_buf
                        .push_str(&format!("  v{id} = {}.load v{}\n", prefix(pty), p.id));
                    Ok(Val { id, ty: pty })
                }
                Some("call") => self.call(e),
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
        let v = self.expr(e)?;
        Ok(if v.ty != want {
            self.convert(v, want)
        } else {
            v
        })
    }

    /// `(add|sub|mul|div|mod Type Expr Expr)`.
    fn arith(&mut self, op: &str, e: &Node) -> Result<Val, LengError> {
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

    /// `(eq|neq|lt|le Expr Expr)` — no explicit type (Leng grammar); infer from the left operand.
    /// Leng ints are signed. Result is `i32` (0/1).
    fn compare(&mut self, op: &str, e: &Node) -> Result<Val, LengError> {
        let a = e.args();
        if a.len() != 2 {
            return Err(LengError::Malformed(format!("`{op}` needs two operands")));
        }
        let l = self.expr(&a[0])?;
        let r = self.expr_typed(&a[1], l.ty)?;
        let name = match op {
            "eq" => "eq",
            "neq" => "ne",
            "lt" => "lt_s",
            "le" => "le_s",
            _ => unreachable!(),
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

    /// `(call Callee Expr*)` — direct call to a named proc.
    fn call(&mut self, e: &Node) -> Result<Val, LengError> {
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
            argvals.push(self.expr_typed(arg, want)?.id);
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

    fn emit_const(&mut self, ty: ValType, n: i64) -> Val {
        let id = self.fresh();
        self.cur_buf
            .push_str(&format!("  v{id} = {}.const {n}\n", prefix(ty)));
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

    /// Integer width conversion between i32/i64 (value-preserving, signed).
    fn convert(&mut self, v: Val, to: ValType) -> Val {
        if v.ty == to {
            return v;
        }
        let id = self.fresh();
        let insn = match (v.ty, to) {
            (ValType::I32, ValType::I64) => format!("i64.extend_i32_s v{}", v.id),
            (ValType::I64, ValType::I32) => format!("i32.wrap_i64 v{}", v.id),
            _ => format!("i64.extend_i32_s v{}", v.id),
        };
        self.cur_buf.push_str(&format!("  v{id} = {insn}\n"));
        Val { id, ty: to }
    }
}

// ---------------------------------------------------------------------------
// Pre-scan + type/atom helpers.
// ---------------------------------------------------------------------------

/// Collect every `(var :name . Type …)` in a statement tree into `slots` (params are already there).
/// Recurses into nested `stmts`/`scope`/`if`/`while` bodies so a var declared in a branch still gets
/// a block-parameter slot (default 0 before its declaration executes).
fn collect_vars(node: &Node, slots: &mut Vec<(String, ValType)>) -> Result<(), LengError> {
    if let Node::List(_) = node {
        if node.tag() == Some("var") {
            let a = node.args();
            if a.len() >= 3 {
                if let (Ok(name), Ok(ty)) = (sym_def(&a[0]), val_ty(&a[2])) {
                    if !slots.iter().any(|(n, _)| n == &name) {
                        slots.push((name, ty));
                    }
                }
            }
        }
        for child in node.args() {
            collect_vars(child, slots)?;
        }
    }
    Ok(())
}

/// Prepend the module `memory` declaration when any proc used the window. `memory 16` = 2^16 = 64
/// KiB, the small-program default; pointer offsets in the current subset stay well within it.
fn with_memory(used: bool, funcs: String) -> String {
    if used {
        format!("memory 16\n\n{funcs}")
    } else {
        funcs
    }
}

/// Value type of a Leng type: pointers (`ptr`/`aptr`) are `i64` window offsets; else an integer.
fn val_ty(node: &Node) -> Result<ValType, LengError> {
    match node.tag() {
        Some("ptr") | Some("aptr") => Ok(ValType::I64),
        _ => int_ty(node),
    }
}

/// If `node` is `(ptr T)`/`(aptr T)`, the pointee value type (load/store width through `deref`).
fn pointee_ty(node: &Node) -> Option<ValType> {
    match node.tag() {
        Some("ptr") | Some("aptr") => node.args().first().and_then(|t| val_ty(t).ok()),
        _ => None,
    }
}

/// A proc needs a frame iff it takes the address of a local somewhere in its body.
fn proc_needs_frame(p: &Node) -> bool {
    let mut s = std::collections::HashSet::new();
    if let Some(b) = p.args().get(4) {
        collect_addr_taken(b, &mut s);
    }
    !s.is_empty()
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

/// Record every pointer-typed `(param …)`/`(var …)` in a tree → its pointee value type.
fn collect_pointees(node: &Node, map: &mut HashMap<String, ValType>) {
    if let Node::List(_) = node {
        if matches!(node.tag(), Some("param") | Some("var")) {
            let a = node.args();
            if a.len() >= 3 {
                if let (Ok(name), Some(pt)) = (sym_def(&a[0]), pointee_ty(&a[2])) {
                    map.insert(name, pt);
                }
            }
        }
        for child in node.args() {
            collect_pointees(child, map);
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
    s.parse::<i64>().map_err(|_| ())
}
