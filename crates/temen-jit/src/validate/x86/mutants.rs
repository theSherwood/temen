//! #2799 layer 2 — by-construction mutants of real Cranelift output: the machine-code validator's
//! accept-direction test (the claim it carries is *accepted ⇒ the function cannot escape*, DESIGN
//! §2a).
//!
//! Each operator rewrites one instruction, one jump-table entry or one constant of a function the
//! validator accepted, in place: every other byte stays, so every offset, branch and relocation
//! still holds. It applies only where its precondition says the rewrite breaks what the acceptance
//! rested on, so the mutant can escape and the validator must reject it. A mutant it accepts is a
//! soundness bug, unless the precondition is wrong.
//!
//! The preconditions read the facts the validator proved of the unmutated function: which register
//! holds `mem_base`, which index a D63 `cmov` confined, which slot holds a saved register. So an
//! operator tests that a rule holds wherever the facts say the code relies on it, not that the facts
//! are right; the pins (`tests/pins.rs`) and #2799's layer 3, which runs what the validator accepts,
//! test those. Each rewritten instruction is decoded again, and the mutant kept only if the decoder
//! reads it as the instruction the operator meant.

use std::collections::BTreeMap;

use super::*;
use crate::validate::CapturedFn;

/// What the mutation test made of one module's functions ([`crate::validator_mutants`]).
#[derive(Debug, Default)]
pub struct Report {
    /// How many mutants each operator made.
    pub made: BTreeMap<&'static str, usize>,
    /// Each mutant the validator accepted: a soundness bug, unless the operator's precondition is
    /// wrong.
    pub survivors: Vec<String>,
    /// Each finding the validator raised on unmutated code. Such a function gets no mutants.
    pub unclean: Vec<String>,
}

/// Run every operator over every captured function.
pub(crate) fn run(caps: &[CapturedFn]) -> Report {
    let mut report = Report::default();
    for op in OPS {
        report.made.insert(op.name, 0);
    }
    for (n, cap) in caps.iter().enumerate() {
        let f = match Func::new(cap) {
            Ok(f) => f,
            Err(findings) => {
                for x in findings {
                    report
                        .unclean
                        .push(format!("fn {n}: {} at +0x{:x}", x.why, x.at));
                }
                continue;
            }
        };
        for op in OPS {
            let mutants = (op.make)(&f);
            *report.made.entry(op.name).or_default() += mutants.len();
            for m in mutants {
                // A mutant the validator does not reject is a soundness hole: it rewrote the code
                // where the acceptance relied on a rule, so the mutant can escape. The original
                // bytes and what the rewrite made of them pin it.
                if f.check(&m.bytes).is_empty() {
                    report.survivors.push(format!(
                        "fn {n}: {} at +0x{:x} ({}): {:02x?} -> {:02x?}",
                        op.name,
                        m.at,
                        m.what,
                        &f.bytes[m.at..m.at + m.len],
                        &m.bytes[m.at..m.at + m.len],
                    ));
                }
            }
        }
    }
    report
}

/// One operator: its name, and the mutants it makes of a function.
struct Op {
    name: &'static str,
    make: fn(&Func) -> Vec<Mutant>,
}

/// The operators, each with the hole it would reopen (#2603's numbering) or the rule it targets.
const OPS: &[Op] = &[
    // Holes 2–4: D63's `cmp v, k; cmov` is the confinement.
    Op {
        name: "d63-invert",
        make: d63_invert,
    },
    Op {
        name: "d63-delete",
        make: d63_delete,
    },
    Op {
        name: "d63-subtract",
        make: d63_subtract,
    },
    Op {
        name: "d63-bound",
        make: d63_bound,
    },
    Op {
        name: "d63-guard",
        make: d63_guard,
    },
    // The access itself (hole 10's window half, CVE-2023-26489's class).
    Op {
        name: "base-or-index",
        make: base_or_index,
    },
    Op {
        name: "scale",
        make: scale,
    },
    Op {
        name: "displacement",
        make: displacement,
    },
    // Hole 5: a reload is as wide as its spill (CVE-2021-32629's class).
    Op {
        name: "reload-narrow",
        make: reload_narrow,
    },
    Op {
        name: "reload-sign-extend",
        make: reload_sign_extend,
    },
    // Hole 16 (I5) and hole 14 (#2534): the stack check and the call boundary.
    Op {
        name: "stack-check",
        make: stack_check,
    },
    Op {
        name: "call-argument",
        make: call_argument,
    },
    // I2: every transfer lands on a decoded boundary.
    Op {
        name: "mid-instruction-jump",
        make: mid_instruction_jump,
    },
    Op {
        name: "jump-table-entry",
        make: jump_table_entry,
    },
    // Claim item 4: every exit restores the callee-saved registers.
    Op {
        name: "callee-saved-restore",
        make: callee_saved_restore,
    },
];

/// A mutant: the function's bytes with `len` of them at `at` rewritten.
struct Mutant {
    at: usize,
    len: usize,
    bytes: Vec<u8>,
    what: String,
}

/// A function the validator accepted, with the state it proved of each block.
struct Func<'a> {
    bytes: &'a [u8],
    relocs: &'a [Reloc],
    guest: Guest<'a>,
    frame: Frame,
    cfg: Cfg,
    states: Vec<Option<State>>,
    reentry: Vec<Slot>,
}

impl<'a> Func<'a> {
    /// `cap`, if the validator accepts it; else what it found.
    fn new(cap: &'a CapturedFn) -> Result<Func<'a>, Vec<Finding>> {
        let guest = cap.guest();
        let frame = Frame {
            stack_args: cap.contract.and_then(|c| c.pops),
            sret_results: cap.sret_results,
        };
        let findings = check(&cap.code, &cap.relocs, Entry::Guest(&guest), frame);
        if !findings.is_empty() {
            return Err(findings);
        }
        let ctx = Ctx {
            bytes: &cap.code,
            relocs: &cap.relocs,
            guest,
            frame,
            reentry: None,
        };
        let cfg = Cfg::build(ctx, &mut Vec::new());
        let (states, reentry) = propose_reentry(&cfg, ctx);
        Ok(Func {
            bytes: &cap.code,
            relocs: &cap.relocs,
            guest,
            frame,
            cfg,
            states,
            reentry,
        })
    }

    /// What the transfer reads, as `verify` runs it.
    fn ctx(&self) -> Ctx<'_> {
        Ctx {
            bytes: self.bytes,
            relocs: self.relocs,
            guest: self.guest,
            frame: self.frame,
            reentry: Some(&self.reentry),
        }
    }

    /// The validator's verdict on `bytes` as this function's code.
    fn check(&self, bytes: &[u8]) -> Vec<Finding> {
        check(bytes, self.relocs, Entry::Guest(&self.guest), self.frame)
    }

    /// Each instruction of a block the entry reaches: its block, its index there, its offset, and it.
    fn reached(&self) -> impl Iterator<Item = (usize, usize, usize, Decoded)> + '_ {
        self.cfg
            .blocks
            .iter()
            .enumerate()
            .filter(|&(b, _)| self.states[b].is_some())
            .flat_map(|(b, insns)| {
                insns
                    .iter()
                    .enumerate()
                    .map(move |(k, &(pos, d))| (b, k, pos, d))
            })
    }

    /// The instructions an operator may rewrite: those [`Func::reached`] yields that no relocation
    /// patches.
    fn insns(&self) -> impl Iterator<Item = (usize, usize, usize, Decoded)> + '_ {
        self.reached()
            .filter(|&(_, _, pos, d)| !self.ctx().patched(pos, d.len))
    }

    /// The state before instruction `k` of block `b`, which the entry reaches.
    fn before(&self, b: usize, k: usize) -> State {
        let s = self.states[b].clone().unwrap_or_else(State::top);
        transfer_block(&self.cfg.blocks[b][..k], s, None, self.ctx(), false, None).fall
    }

    /// The mutant with the instruction at `pos` rewritten to `insn`, if the decoder reads the
    /// rewrite as an instruction of the same length that `meant` says is the one intended.
    fn rewrite(
        &self,
        pos: usize,
        insn: &[u8],
        meant: impl Fn(&Decoded) -> bool,
        what: String,
    ) -> Option<Mutant> {
        let mut bytes = self.bytes.to_vec();
        bytes.get_mut(pos..pos + insn.len())?.copy_from_slice(insn);
        let n = decode(&bytes, pos);
        (n.len == insn.len() && meant(&n)).then_some(Mutant {
            at: pos,
            len: insn.len(),
            bytes,
            what,
        })
    }

    /// The mutant with the `len` bytes at `at` (data: a constant, a jump-table entry) replaced by
    /// `value`'s low bytes, if no relocation patches them.
    fn poke(&self, at: usize, len: usize, value: u64, what: String) -> Option<Mutant> {
        if self.ctx().patched(at, len) {
            return None;
        }
        let mut bytes = self.bytes.to_vec();
        bytes
            .get_mut(at..at + len)?
            .copy_from_slice(&value.to_le_bytes()[..len]);
        Some(Mutant {
            at,
            len,
            bytes,
            what,
        })
    }

    /// The bytes of the instruction at `pos`.
    fn insn(&self, pos: usize, d: &Decoded) -> &[u8] {
        &self.bytes[pos..pos + d.len]
    }
}

/// The mutant that replaces the instruction at `pos` with `nop`s.
fn nops(f: &Func, pos: usize, d: &Decoded, what: String) -> Mutant {
    let mut bytes = f.bytes.to_vec();
    bytes[pos..pos + d.len].fill(0x90);
    Mutant {
        at: pos,
        len: d.len,
        bytes,
        what,
    }
}

/// Whether, after instruction `k` of block `b`, the block accesses memory through `r`, as a base
/// or an index, before anything writes `r`.
fn addresses(f: &Func, b: usize, k: usize, r: u8) -> bool {
    for (_, d) in &f.cfg.blocks[b][k + 1..] {
        if let Some(m) = d.operands.mem {
            if !matches!(d.eff, Eff::Lea { .. }) && (m.base == Some(r) || m.index == Some(r)) {
                return true;
            }
        }
        if writes_of(d) & bit(r) != 0 {
            return false;
        }
    }
    false
}

/// A register other than `rsp` and `not`, about which the validator knows nothing in `st`, that
/// the instruction can name without a REX prefix it lacks: what an operator puts where the code
/// relied on a fact.
fn unknown_reg(st: &State, not: &[Option<u8>], rex: bool) -> Option<u8> {
    (0..if rex { 16 } else { 8 })
        .find(|&r| r != reg::RSP && !not.contains(&Some(r)) && st.regs[r as usize] == Fact::Unknown)
}

/// Where an instruction's encoding fields sit, as offsets from its start: the register-extension
/// prefix (REX, or the VEX prefix that carries its bits inverted), the opcode, the ModRM and SIB
/// bytes, and the displacement with its width. Legacy, REX and VEX encodings with a ModRM byte, which
/// covers every form an operator rewrites (EVEX's are not rewritten). A rewrite is decoded again
/// ([`Func::rewrite`]), so a field read wrongly here makes no mutant rather than a wrong one.
#[derive(Clone, Copy)]
struct Enc {
    rex: Option<Ext>,
    op: usize,
    modrm: usize,
    sib: Option<usize>,
    disp: Option<(usize, usize)>,
}

/// Where an instruction's register-extension bits are: a REX prefix, a three-byte VEX prefix's
/// inverted `R`, `X` and `B`, or a two-byte one's inverted `R` alone.
#[derive(Clone, Copy)]
enum Ext {
    Rex(usize),
    Vex3,
    Vex2,
}

impl Enc {
    fn of(b: &[u8]) -> Option<Enc> {
        let (rex, op) = match *b.first()? {
            0xC4 => (Some(Ext::Vex3), 3),
            0xC5 => (Some(Ext::Vex2), 2),
            0x62 => return None,
            _ => {
                let mut i = 0;
                while matches!(b.get(i)?, 0x66 | 0xF0 | 0xF2 | 0xF3) {
                    i += 1;
                }
                let rex = matches!(b.get(i)?, 0x40..=0x4F).then_some(Ext::Rex(i));
                (rex, i + usize::from(rex.is_some()))
            }
        };
        let mut i = op;
        if matches!(rex, Some(Ext::Rex(_)) | None) {
            match *b.get(i)? {
                0x0F if matches!(b.get(i + 1)?, 0x38 | 0x3A) => i += 2,
                0x0F => i += 1,
                _ => {}
            }
        }
        let modrm = i + 1;
        let m = *b.get(modrm)?;
        let (md, rm) = (m >> 6, m & 7);
        let sib = (md != 3 && rm == 4).then_some(modrm + 1);
        let next = modrm + 1 + usize::from(sib.is_some());
        let base5 = sib.map_or(rm == 5, |s| b.get(s).is_some_and(|&x| x & 7 == 5));
        let w = match md {
            0 if base5 => 4,
            1 => 1,
            2 => 4,
            _ => 0,
        };
        Some(Enc {
            rex,
            op,
            modrm,
            sib,
            disp: (w > 0).then_some((next, w)),
        })
    }

    /// Set (`on`) or clear the register-extension bit `bit` (REX's `W R X B` numbering) of `v`:
    /// `None` when it must be set and the encoding has nowhere to set it.
    fn rex_bit(&self, v: &mut [u8], bit: u8, on: bool) -> Option<()> {
        // VEX keeps `R X B` inverted in bits 7–5 of its second byte; the two-byte form has `R` only.
        let vex = |v: &mut [u8], b: u8| {
            if on {
                v[1] &= !b
            } else {
                v[1] |= b
            }
        };
        match (self.rex, bit) {
            (Some(Ext::Rex(at)), _) if on => v[at] |= bit,
            (Some(Ext::Rex(at)), _) => v[at] &= !bit,
            (Some(Ext::Vex3), 0b0100 | 0b0010 | 0b0001) => vex(v, bit << 5),
            (Some(Ext::Vex2), 0b0100) => vex(v, 0x80),
            _ if on => return None,
            _ => {}
        }
        Some(())
    }

    /// `insn` with its memory operand's base register `r`.
    fn base(&self, insn: &[u8], r: u8) -> Option<Vec<u8>> {
        let mut v = insn.to_vec();
        let at = self.sib.unwrap_or(self.modrm);
        v[at] = (v[at] & !7) | (r & 7);
        self.rex_bit(&mut v, 0b0001, r >= 8)?;
        Some(v)
    }

    /// `insn` with its memory operand's index register `r`.
    fn index(&self, insn: &[u8], r: u8) -> Option<Vec<u8>> {
        let s = self.sib?;
        let mut v = insn.to_vec();
        v[s] = (v[s] & !0x38) | ((r & 7) << 3);
        self.rex_bit(&mut v, 0b0010, r >= 8)?;
        Some(v)
    }

    /// `insn` with its ModRM `reg` field naming `r`.
    fn reg(&self, insn: &[u8], r: u8) -> Option<Vec<u8>> {
        let mut v = insn.to_vec();
        v[self.modrm] = (v[self.modrm] & !0x38) | ((r & 7) << 3);
        self.rex_bit(&mut v, 0b0100, r >= 8)?;
        Some(v)
    }

    /// `insn` with its register r/m operand `r`.
    fn rm(&self, insn: &[u8], r: u8) -> Option<Vec<u8>> {
        let mut v = insn.to_vec();
        v[self.modrm] = (v[self.modrm] & !7) | (r & 7);
        self.rex_bit(&mut v, 0b0001, r >= 8)?;
        Some(v)
    }
}

/// Whether `n` is `d` with its memory operand `m` instead.
fn mem_is(d: Decoded, m: Mem) -> impl Fn(&Decoded) -> bool {
    move |n| {
        *n == Decoded {
            operands: Operands {
                mem: Some(m),
                ..d.operands
            },
            ..d
        }
    }
}

/// A D63 confinement a later access in its block relies on: the `cmov` at instruction `k` of
/// block `b`, the register `dst` that holds the confined index after it, the register `index` that
/// holds the unconfined one before it, and the compare at `cmp` whose flags it tests.
struct D63 {
    b: usize,
    k: usize,
    pos: usize,
    d: Decoded,
    dst: u8,
    cc: u8,
    w: u8,
    index: u8,
    guard_reg: u8,
    guard: u64,
    cmp: (usize, usize, Decoded),
}

/// Every D63 confinement in `f` whose index is unbounded without it, so that the access it guards
/// can escape when it doesn't confine. Both shapes: a `cmov` that redirects the index to the guard
/// (`cmovae idx, guard`), and one that selects the index into the guard (`cmovb guard, idx`, as the
/// br_table clamp does).
fn d63s(f: &Func) -> Vec<D63> {
    let mut out = Vec::new();
    for (b, k, pos, d) in f.insns() {
        let Eff::Cmov { dst, cc, w } = d.eff else {
            continue;
        };
        let Some(src) = d.operands.rm_reg else {
            continue;
        };
        let Fact::Confined { guard, .. } = f.before(b, k + 1).regs[dst as usize] else {
            continue;
        };
        if !addresses(f, b, k, dst) {
            continue;
        }
        let st = f.before(b, k);
        let is_guard = |r: u8| st.regs[r as usize] == Fact::Const(guard as i64);
        let (index, guard_reg) = match (is_guard(src), is_guard(dst)) {
            (true, _) => (dst, src),
            (_, true) => (src, dst),
            _ => continue,
        };
        // Bounded without it, the confinement is redundant: removing it is not an escape.
        if max_value(st.regs[index as usize]).is_some() {
            continue;
        }
        // The compare: the last instruction before the `cmov` to write the flags.
        let insns = &f.cfg.blocks[b];
        let Some(j) = (0..k).rev().find(|&j| !insns[j].1.keeps_flags) else {
            continue;
        };
        if !matches!(insns[j].1.eff, Eff::Cmp { .. }) {
            continue;
        }
        out.push(D63 {
            b,
            k,
            pos,
            d,
            dst,
            cc,
            w,
            index,
            guard_reg,
            guard,
            cmp: (j, insns[j].0, insns[j].1),
        });
    }
    out
}

/// Hole 2: the `cmov` tests the opposite condition, so it keeps the index exactly when it is out
/// of range.
fn d63_invert(f: &Func) -> Vec<Mutant> {
    d63s(f)
        .into_iter()
        .filter_map(|s| {
            let e = Enc::of(f.insn(s.pos, &s.d))?;
            let mut v = f.insn(s.pos, &s.d).to_vec();
            v[e.op + 1] ^= 1;
            let want = Decoded {
                eff: Eff::Cmov {
                    dst: s.dst,
                    cc: s.cc ^ 1,
                    w: s.w,
                },
                ..s.d
            };
            f.rewrite(s.pos, &v, |n| *n == want, format!("cmov cc {:x}", s.cc ^ 1))
        })
        .collect()
}

/// D63 without its `cmov`, where the `cmov` redirects the index (deleting a select would leave the
/// guard, which only faults).
fn d63_delete(f: &Func) -> Vec<Mutant> {
    d63s(f)
        .into_iter()
        .filter(|s| s.index == s.dst)
        .map(|s| nops(f, s.pos, &s.d, "cmov deleted".into()))
        .collect()
}

/// Hole 3: the compare becomes a subtract, which sets the same flags but leaves `v − k` in the
/// register the access then uses, wrapped past the reservation on exactly the in-range path. Only
/// where the confined index is the subtracted register, or a copy of it made after the compare.
fn d63_subtract(f: &Func) -> Vec<Mutant> {
    let mut out = Vec::new();
    for s in d63s(f) {
        let (j, pos, d) = s.cmp;
        let insn = f.insn(pos, &d);
        let Some(e) = Enc::of(insn) else { continue };
        let md = insn[e.modrm] >> 6;
        // The register the subtract writes, and its encoding: `cmp r, r/m` → `sub r, r/m`,
        // `cmp r/m, r` and `cmp r/m, imm` (register forms) → `sub r/m, …`.
        let (lhs, v) = match (insn[e.op], d.operands) {
            (0x3B, Operands { reg: Some(r), .. }) => (r, [(e.op, 0x2B)]),
            (
                0x39,
                Operands {
                    rm_reg: Some(r), ..
                },
            ) if md == 3 => (r, [(e.op, 0x29)]),
            (
                0x81 | 0x83,
                Operands {
                    rm_reg: Some(r), ..
                },
            ) if md == 3 => (r, [(e.modrm, insn[e.modrm] & !0x38 | 0x28)]),
            _ => continue,
        };
        let copied = f.cfg.blocks[s.b][j + 1..s.k].iter().any(|(_, x)| {
            x.eff
                == Eff::MovReg {
                    dst: s.index,
                    src: lhs,
                }
        });
        if s.index != lhs && !copied {
            continue;
        }
        let mut sub = insn.to_vec();
        for (at, byte) in v {
            sub[at] = byte;
        }
        let mut bytes = f.bytes.to_vec();
        bytes[pos..pos + d.len].copy_from_slice(&sub);
        let n = decode(&bytes, pos);
        let is_sub = matches!(n.eff, Eff::Alu { dst, alu: Alu::Sub, .. } if dst == lhs)
            || n.eff == Eff::Wr32 { dst: lhs };
        if n.len == d.len && is_sub {
            out.push(Mutant {
                at: pos,
                len: d.len,
                bytes,
                what: "cmp as sub".into(),
            });
        }
    }
    out
}

/// Hole 4: the compare's bound, its constant-pool entry or its immediate, becomes all ones, so the
/// `cmov` confines nothing. A 64-bit compare only: a 32-bit one bounds below 2^32, inside the
/// reservation whatever the bound.
fn d63_bound(f: &Func) -> Vec<Mutant> {
    let mut out = Vec::new();
    for s in d63s(f) {
        let (_, pos, d) = s.cmp;
        let Eff::Cmp { w: 8, rhs, .. } = d.eff else {
            continue;
        };
        let m = match rhs {
            Opnd::Mem => f
                .ctx()
                .rip_target(&d, pos + d.len)
                .and_then(|t| usize::try_from(t).ok())
                .and_then(|t| f.poke(t, 8, u64::MAX, "bound in the pool".into())),
            Opnd::Imm(_) if d.imm_len == 4 => {
                f.poke(pos + d.len - 4, 4, u64::MAX, "bound immediate".into())
            }
            _ => None,
        };
        out.extend(m);
    }
    out
}

/// D63 redirects an out-of-range index past the reservation instead of to its guard page: the
/// guard constant (`movabs r, guard_offset`) moves two pages up.
fn d63_guard(f: &Func) -> Vec<Mutant> {
    let page = crate::mem::trailing_guard() as u64;
    let mut out = Vec::new();
    for s in d63s(f) {
        let insns = &f.cfg.blocks[s.b];
        // The guard's materialization: the last write of its register before the `cmov`.
        let Some(j) = (0..s.k)
            .rev()
            .find(|&j| writes_of(&insns[j].1) & bit(s.guard_reg) != 0)
        else {
            continue;
        };
        let (pos, d) = insns[j];
        if d.eff
            != (Eff::MovImm {
                dst: s.guard_reg,
                imm: s.guard as i64,
            })
            || d.imm_len != 8
        {
            continue;
        }
        out.extend(f.poke(
            pos + d.len - 8,
            8,
            s.guard + 2 * page,
            "guard two pages up".into(),
        ));
    }
    out
}

/// Each window access in `f` (its base, or a scale-1 index, `mem_base` or a pointer into the
/// window): the instruction, the state before it, and its encoding.
fn window_accesses(f: &Func) -> Vec<(usize, Decoded, State, Mem, Enc)> {
    let into = |x: Fact| matches!(x, Fact::MemBase | Fact::WinPtr(_));
    let mut out = Vec::new();
    for (b, k, pos, d) in f.insns() {
        let Some(m) = d.operands.mem.filter(|m| !m.rip) else {
            continue;
        };
        if matches!(d.eff, Eff::Lea { .. }) {
            continue;
        }
        let st = f.before(b, k);
        let fact = |r: Option<u8>| r.map(|r| st.regs[r as usize]);
        let window = fact(m.base).is_some_and(into) || fact(m.index).is_some_and(into);
        if let (true, Some(e)) = (window, Enc::of(f.insn(pos, &d))) {
            out.push((pos, d, st, m, e));
        }
    }
    out
}

/// A guest-controlled register where the window access relied on `mem_base` (the base) or on a
/// confined or bounded index.
fn base_or_index(f: &Func) -> Vec<Mutant> {
    let mut out = Vec::new();
    for (pos, d, st, m, e) in window_accesses(f) {
        let insn = f.insn(pos, &d);
        let not = [m.base, m.index];
        let Some(r) = unknown_reg(&st, &not, e.rex.is_some()) else {
            continue;
        };
        // Whichever operand holds the window pointer and whichever the index, each is one the
        // access relies on.
        if m.base.is_some() {
            let meant = mem_is(d, Mem { base: Some(r), ..m });
            out.extend(
                e.base(insn, r)
                    .and_then(|v| f.rewrite(pos, &v, meant, format!("base r{r}"))),
            );
        }
        if m.index.is_some() {
            let meant = mem_is(
                d,
                Mem {
                    index: Some(r),
                    ..m
                },
            );
            out.extend(
                e.index(insn, r)
                    .and_then(|v| f.rewrite(pos, &v, meant, format!("index r{r}"))),
            );
        }
    }
    out
}

/// CVE-2023-26489's class: a shift folded into the SIB scale. A window access's scale-1 index,
/// confined (whose largest value is the guard offset) or the window pointer itself, scaled by 8.
fn scale(f: &Func) -> Vec<Mutant> {
    let mut out = Vec::new();
    for (pos, d, st, m, e) in window_accesses(f) {
        let (Some(i), Some(s), 1) = (m.index, e.sib, m.scale) else {
            continue;
        };
        if !matches!(
            st.regs[i as usize],
            Fact::Confined { .. } | Fact::MemBase | Fact::WinPtr(_)
        ) {
            continue;
        }
        let mut v = f.insn(pos, &d).to_vec();
        v[s] |= 0xC0;
        let meant = mem_is(d, Mem { scale: 8, ..m });
        out.extend(f.rewrite(pos, &v, meant, "scale 8".into()));
    }
    out
}

/// A window access's 32-bit displacement grown so that its largest offset, as the validator bounds
/// it, ends past the reservation. Only where that fits a displacement: an index the D63 `cmov`
/// confined reaches the guard offset, a page below the top.
fn displacement(f: &Func) -> Vec<Mutant> {
    let page = crate::mem::trailing_guard() as u64;
    let Some(top) = f.guest.window.map(|l| l.guard_offset.saturating_add(page)) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (pos, d, st, m, e) in window_accesses(f) {
        let (Some((at, 4)), Ok(Some(off))) = (e.disp, window_offset(&m, &st, f.guest.window))
        else {
            continue;
        };
        let disp = top
            .checked_sub(off)
            .and_then(|up| i64::try_from(up).ok())
            .and_then(|up| up.checked_add(m.disp))
            .filter(|&x| (0..=i64::from(i32::MAX)).contains(&x));
        let Some(disp) = disp else { continue };
        let mut v = f.insn(pos, &d).to_vec();
        v[at..at + 4].copy_from_slice(&(disp as i32).to_le_bytes());
        let meant = mem_is(d, Mem { disp, ..m });
        out.extend(f.rewrite(pos, &v, meant, format!("disp 0x{disp:x}")));
    }
    out
}

/// Each 8-byte reload of a frame slot (`REX.W 8B`, `mov r64, [rsp + d]`) in `f` whose register a
/// later access in its block reads before anything writes it: the instruction, the register, the
/// fact it reloads, and its REX prefix's offset and opcode's.
fn reloads(f: &Func) -> Vec<(usize, Decoded, u8, Fact, usize, usize)> {
    let mut out = Vec::new();
    for (b, k, pos, d) in f.insns() {
        let (Eff::LoadReg { dst }, Some(m)) = (d.eff, d.operands.mem) else {
            continue;
        };
        if !matches!(frame_addr(&f.before(b, k), &m), Some(Some(_))) {
            continue;
        }
        let Some(e) = Enc::of(f.insn(pos, &d)) else {
            continue;
        };
        let (Some(Ext::Rex(rex)), 0x8B) = (e.rex, f.bytes[pos + e.op]) else {
            continue;
        };
        if f.bytes[pos + rex] & 8 == 0 || !addresses(f, b, k, dst) {
            continue;
        }
        let fact = f.before(b, k + 1).regs[dst as usize];
        out.push((pos, d, dst, fact, rex, e.op));
    }
    out
}

/// Hole 5: a pointer reloaded at 32 bits (`REX.W` cleared), which zero-extends its low half into a
/// base an access then uses.
fn reload_narrow(f: &Func) -> Vec<Mutant> {
    let mut out = Vec::new();
    for (pos, d, dst, fact, rex, _) in reloads(f) {
        if fact == Fact::Unknown || max_value(fact).is_some() || matches!(fact, Fact::Saved(_)) {
            continue;
        }
        let mut v = f.insn(pos, &d).to_vec();
        v[rex] &= !8;
        let meant = |n: &Decoded| {
            n.eff == Eff::LoadZx32 { dst } && n.operands.mem == d.operands.mem && n.width == 4
        };
        out.extend(f.rewrite(pos, &v, meant, format!("{fact:?} reloaded at 32 bits")));
    }
    out
}

/// Hole 5 as CVE-2021-32629 shaped it: an index reloaded sign-extending its low half (`movsxd`),
/// where it can be 2^31 or more, so an access gets an index near 2^64.
fn reload_sign_extend(f: &Func) -> Vec<Mutant> {
    let mut out = Vec::new();
    for (pos, d, dst, fact, _, op) in reloads(f) {
        if !max_value(fact).is_some_and(|m| m >= 1 << 31) {
            continue;
        }
        let mut v = f.insn(pos, &d).to_vec();
        v[op] = 0x63;
        let meant = |n: &Decoded| {
            n.eff == Eff::LoadSx32 { dst } && n.operands.mem == d.operands.mem && n.width == 4
        };
        out.extend(f.rewrite(pos, &v, meant, format!("{fact:?} reloaded sign-extending")));
    }
    out
}

/// The contract of the call at instruction `k` of block `b`, as the validator identifies it.
fn callee(f: &Func, b: usize, k: usize, pos: usize, d: &Decoded) -> Option<Contract> {
    match d.kind {
        Kind::DirectCall => f.ctx().call_reloc(pos, d).and_then(|r| r.callee),
        Kind::IndirectCall => indirect_callee(&f.before(b, k), d, f.ctx()),
        _ => None,
    }
}

/// Hole 16 (I5): the prologue's stack-overflow check without its `jb`, in a function that makes a
/// call it must dominate.
fn stack_check(f: &Func) -> Vec<Mutant> {
    let calls = f
        .reached()
        .any(|(b, k, pos, d)| callee(f, b, k, pos, &d).is_some_and(|c| !c.safepoint));
    if !calls {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (b, insns) in f.cfg.blocks.iter().enumerate() {
        let (Some(s), Some(&(pos, d))) = (&f.states[b], insns.last()) else {
            continue;
        };
        let checks = !s.checked
            && matches!(d.kind, Kind::DirectJcc { cc: 0x2, .. })
            && f.cfg.succs[b].iter().any(|&j| {
                f.cfg.leaders[j] == pos + d.len && f.states[j].as_ref().is_some_and(|t| t.checked)
            });
        if checks {
            out.push(nops(f, pos, &d, "stack check's jb deleted".into()));
        }
    }
    out
}

/// Hole 14 (#2534): a tagged call argument (`vmctx`, a frame buffer, the `jmp_buf`, the fuel cell,
/// the frame pointer) taken from a register the validator knows nothing about. The argument's last
/// write before the call, in its block, is a register move or a `lea`; its source becomes that
/// register.
fn call_argument(f: &Func) -> Vec<Mutant> {
    let mut out = Vec::new();
    for (b, k, pos, d) in f.reached() {
        let Some(c) = callee(f, b, k, pos, &d) else {
            continue;
        };
        for (i, &tag) in c.args.iter().enumerate() {
            let Some(&r) = reg::ARG_REGS.get(i).filter(|_| tag != ArgTag::Any) else {
                continue;
            };
            let insns = &f.cfg.blocks[b];
            let Some(j) = (0..k).rev().find(|&j| writes_of(&insns[j].1) & bit(r) != 0) else {
                continue;
            };
            let (wpos, wd) = insns[j];
            if f.ctx().patched(wpos, wd.len) {
                continue;
            }
            let insn = f.insn(wpos, &wd);
            let Some(e) = Enc::of(insn) else { continue };
            let st = f.before(b, j);
            let what = |x: u8| format!("argument {i} ({tag:?}) from r{x}");
            let m = match wd.eff {
                Eff::MovReg { dst, src } if dst == r => {
                    let Some(x) = unknown_reg(&st, &[Some(src), Some(dst)], e.rex.is_some()) else {
                        continue;
                    };
                    let v = match insn[e.op] {
                        0x89 => e.reg(insn, x),
                        0x8B => e.rm(insn, x),
                        _ => None,
                    };
                    let meant = |n: &Decoded| n.eff == Eff::MovReg { dst, src: x };
                    v.and_then(|v| f.rewrite(wpos, &v, meant, what(x)))
                }
                Eff::Lea { dst, full: true } if dst == r => {
                    let Some(m) = wd.operands.mem.filter(|m| m.base.is_some() && !m.rip) else {
                        continue;
                    };
                    let Some(x) = unknown_reg(&st, &[m.base, m.index], e.rex.is_some()) else {
                        continue;
                    };
                    let meant = mem_is(wd, Mem { base: Some(x), ..m });
                    e.base(insn, x)
                        .and_then(|v| f.rewrite(wpos, &v, meant, what(x)))
                }
                _ => None,
            };
            out.extend(m);
        }
    }
    out
}

/// An offset one byte into the first instruction of two or more bytes in block `b`. Aimed at by a
/// branch from `b`, which the entry reaches, so `b`'s own decode still runs that instruction too:
/// a target in the middle of an instruction the code executes, which I2 forbids. (Aimed into a block
/// nothing else reaches, it would only start a new decode, which the check covers.)
fn mid_instruction(f: &Func, b: usize) -> Option<usize> {
    f.cfg.blocks[b]
        .iter()
        .find(|(_, d)| d.len >= 2)
        .map(|&(pos, _)| pos + 1)
}

/// I2: a direct branch aimed one byte into an instruction of its own block.
fn mid_instruction_jump(f: &Func) -> Vec<Mutant> {
    let mut out = Vec::new();
    for (b, _, pos, d) in f.insns() {
        let kind = |t| match d.kind {
            Kind::DirectJcc { cc, .. } => Some(Kind::DirectJcc { target: t, cc }),
            Kind::DirectJmp { .. } => Some(Kind::DirectJmp { target: t }),
            _ => None,
        };
        let Some(t) = mid_instruction(f, b) else {
            continue;
        };
        let Some(kind) = kind(t) else { continue };
        let end = pos + d.len;
        let rel = t as i64 - end as i64;
        let width = match f.bytes[pos] {
            0xEB | 0x70..=0x7F => 1,
            _ => 4,
        };
        let fits = if width == 1 {
            i8::try_from(rel).is_ok()
        } else {
            i32::try_from(rel).is_ok()
        };
        if !fits {
            continue;
        }
        let mut v = f.insn(pos, &d).to_vec();
        let n = v.len();
        v[n - width..].copy_from_slice(&rel.to_le_bytes()[..width]);
        let want = Decoded { kind, ..d };
        out.extend(f.rewrite(pos, &v, |n| *n == want, format!("target +0x{t:x}")));
    }
    out
}

/// I2, hole 11: a jump table's first entry aimed one byte into an instruction of the block that
/// jumps through it.
fn jump_table_entry(f: &Func) -> Vec<Mutant> {
    let mut out = Vec::new();
    for (b, insns) in f.cfg.blocks.iter().enumerate() {
        let Some(&(_, d)) = insns.last().filter(|(_, d)| d.kind == Kind::IndirectJmp) else {
            continue;
        };
        if f.states[b].is_none() {
            continue;
        }
        let Some(r) = d.operands.rm_reg else { continue };
        let out_st = transfer_block(insns, State::top(), None, f.ctx(), false, None).fall;
        let Fact::TableTarget { at, .. } = out_st.regs[r as usize & 15] else {
            continue;
        };
        let at = at as usize;
        let Some(t) = mid_instruction(f, b) else {
            continue;
        };
        let entry = (t as i64 - at as i64) as u64;
        out.extend(f.poke(at, 4, entry, format!("entry 0 targets +0x{t:x}")));
    }
    out
}

/// Claim item 4: an exit block's reload of a callee-saved register from its save slot deleted,
/// where the register holds something else before it.
fn callee_saved_restore(f: &Func) -> Vec<Mutant> {
    let mut out = Vec::new();
    for (b, k, pos, d) in f.insns() {
        let Eff::LoadReg { dst } = d.eff else {
            continue;
        };
        let exit = f.cfg.blocks[b].last().is_some_and(|&(at, x)| {
            x.kind == Kind::Ret
                || matches!(x.kind, Kind::DirectJmp { .. }) && tail_call(&x, at, f.ctx())
        });
        if !exit || !reg::CALLEE_SAVED.contains(&dst) {
            continue;
        }
        let saved = |s: &State| s.regs[dst as usize] == Fact::Saved(dst);
        if saved(&f.before(b, k + 1)) && !saved(&f.before(b, k)) {
            out.push(nops(f, pos, &d, format!("r{dst}'s restore deleted")));
        }
    }
    out
}
