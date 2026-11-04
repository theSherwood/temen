//! The decoder's completeness oracle (#2603 PR 4, hole 17).
//!
//! Cranelift's x64 encoder is generated from one table, the DSL in `cranelift-assembler-x64-meta`.
//! [`every_cranelift_form_is_rejected_or_modeled_soundly`] walks every form in it, synthesizes the
//! bytes that encoder emits for it, and decodes them with the validator's decoder and with iced-x86.
//! The decoder must reject the form, or agree with iced wherever the validator relies on it: the
//! length, the memory operand and whether it is written, the GP registers written, any implicit
//! memory access, and where control goes next. A Cranelift upgrade that adds an encoding therefore
//! fails here, not in a user's compile.

use super::*;
use cranelift_assembler_x64_meta::dsl::{
    self, Customization, Encoding, Feature, Features, Group1Prefix, Group2Prefix, OperandKind as K,
    VexEscape, VexPrefix,
};
use cranelift_assembler_x64_meta::instructions;
use iced_x86::FlowControl;
use std::collections::BTreeMap;

/// The registers a synthesized form names in each encoding field. They are distinct, so a write the
/// decoder misses can't hide behind another operand naming the same register.
#[derive(Clone, Copy)]
struct Regs {
    reg: u8,
    vvvv: u8,
    rm: u8,
    is4: u8,
}

/// Registers 4–7, which need no extension bit (a byte operand among them makes Cranelift emit the
/// bare REX that selects spl–dil), and 9–12, which need REX/VEX/EVEX extension bits.
const LOW: Regs = Regs {
    reg: 5,
    vvvv: 6,
    rm: 7,
    is4: 4,
};
const HIGH: Regs = Regs {
    reg: 9,
    vvvv: 10,
    rm: 11,
    is4: 12,
};

/// What a form's ModRM r/m field names.
#[derive(Clone, Copy, PartialEq)]
enum Rm {
    /// The form has no ModRM byte.
    None,
    Reg,
    /// A register or memory.
    Any,
    Mem,
}

/// Where a form's operands go, as the DSL's generator (`generate/format.rs`) assigns them.
struct Layout {
    /// The ModRM.reg field holds this opcode extension (`/digit`), else a register operand.
    digit: Option<u8>,
    rm: Rm,
    /// VEX/EVEX.vvvv names an operand; otherwise it holds 1111b, no register.
    vvvv: bool,
    /// The opcode's low three bits name a register (`+r`).
    opcode_reg: bool,
    /// The top nibble of a trailing byte names a register (`/is4`).
    is4: bool,
}

fn rm_of(k: &K) -> Rm {
    match k {
        K::RegMem(_) => Rm::Any,
        K::Mem(_) => Rm::Mem,
        _ => Rm::Reg,
    }
}

/// A REX-encoded form's layout: `generate_rex_prefix`'s operand patterns.
fn rex_layout(kinds: &[K], digit: Option<u8>) -> Layout {
    use K::{FixedReg as F, Imm as I, Mem as M, Reg as R, RegMem as RM};
    let l = |digit, rm, opcode_reg| Layout {
        digit,
        rm,
        vvvv: false,
        opcode_reg,
        is4: false,
    };
    match kinds {
        [] | [I(_)] | [F(_)] | [F(_), F(_)] | [F(_), I(_)] => l(None, Rm::None, false),
        [R(_)] => l(None, Rm::None, true),
        [R(_), I(_)] if digit.is_some() => l(digit, Rm::Reg, false),
        [R(_), I(_)] => l(None, Rm::None, true),
        [F(_), m @ RM(_)]
        | [F(_), F(_), m @ RM(_)]
        | [m @ RM(_), F(_)]
        | [m @ M(_), I(_)]
        | [m @ RM(_), I(_)]
        | [m @ RM(_)]
        | [F(_), F(_), F(_), F(_), m @ M(_)] => l(digit, rm_of(m), false),
        [R(_), m @ (RM(_) | M(_))]
        | [R(_), m @ RM(_), I(_) | F(_)]
        | [m @ (RM(_) | M(_)), R(_)]
        | [m @ (RM(_) | M(_)), R(_), I(_) | F(_)] => l(None, rm_of(m), false),
        [R(_), R(_)] | [R(_), R(_), I(_)] => l(None, Rm::Reg, false),
        other => panic!("REX operand pattern the generator doesn't have: {other:?}"),
    }
}

/// A VEX- or EVEX-encoded form's layout: `generate_vex_or_evex_prefix`'s operand patterns.
fn vex_layout(kinds: &[K], digit: Option<u8>) -> Layout {
    use K::{FixedReg as F, Imm as I, Mem as M, Reg as R, RegMem as RM};
    let l = |digit, rm, vvvv, is4| Layout {
        digit,
        rm,
        vvvv,
        opcode_reg: false,
        is4,
    };
    match kinds {
        [R(_), R(_), R(_)] => l(None, Rm::Reg, true, false),
        [R(_), R(_), m @ (RM(_) | M(_))]
        | [R(_), R(_), m @ RM(_), I(_) | F(_)]
        | [R(_), m @ RM(_), R(_)] => l(None, rm_of(m), true, false),
        [R(_), R(_), m @ RM(_), R(_)] => l(None, rm_of(m), true, true),
        // With a digit, the register operand is in vvvv and ModRM.reg holds the digit.
        [R(_), m @ RM(_)] | [m @ RM(_), R(_)] | [R(_), m @ RM(_), I(_)] => {
            l(digit, rm_of(m), digit.is_some(), false)
        }
        [R(_), R(_)] | [R(_), R(_), I(_)] => l(digit, Rm::Reg, digit.is_some(), false),
        [R(_), m @ M(_)] | [m @ M(_), R(_)] | [m @ RM(_), R(_), I(_)] => {
            l(None, rm_of(m), false, false)
        }
        other => panic!("VEX/EVEX operand pattern the generator doesn't have: {other:?}"),
    }
}

/// Whether Cranelift may emit a form in 64-bit mode, any ISA extension assumed present.
fn in_64bit_mode(f: &Features) -> bool {
    match f {
        Features::And(a, b) => in_64bit_mode(a) && in_64bit_mode(b),
        Features::Or(a, b) => in_64bit_mode(a) || in_64bit_mode(b),
        Features::Feature(f) => *f != Feature::compat,
    }
}

/// The ModRM byte (and displacement) for the reg-field value `reg` and the r/m: register `rm` (when
/// `!mem`), or the memory operand `[rm + disp]` — `disp8` when `disp8`, else `disp32`. (An EVEX
/// `disp8` is scaled by the access size, which the decoder cannot know, so the decoder rejects that
/// form; exercising both widths checks both the rejection and the `disp32` decode.)
fn modrm(out: &mut Vec<u8>, reg: u8, rm: u8, mem: bool, disp8: bool) {
    let md: u8 = match (mem, disp8) {
        (false, _) => 0b11,
        (true, true) => 0b01,
        (true, false) => 0b10,
    };
    out.push(md << 6 | (reg & 7) << 3 | (rm & 7));
    if mem {
        out.extend(if disp8 { vec![2] } else { vec![2, 0, 0, 0] });
    }
}

/// The bytes Cranelift's encoder emits for `inst`, with the registers `r` in its fields and, when
/// `mem`, its r/m operand in memory; `None` when the form has no such variant.
fn synth(inst: &dsl::Inst, r: Regs, mem: bool, disp8: bool) -> Option<Vec<u8>> {
    if inst.custom.contains(Customization::Encode) {
        // Only the multi-byte `nop`s have a hand-written encoder: Intel's recommended sequences.
        let nop: &[u8] = match inst.format.name.as_str() {
            "1B" => &[0x90],
            "2B" => &[0x66, 0x90],
            "3B" => &[0x0F, 0x1F, 0x00],
            "4B" => &[0x0F, 0x1F, 0x40, 0x00],
            "5B" => &[0x0F, 0x1F, 0x44, 0x00, 0x00],
            "6B" => &[0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00],
            "7B" => &[0x0F, 0x1F, 0x80, 0, 0, 0, 0],
            "8B" => &[0x0F, 0x1F, 0x84, 0x00, 0, 0, 0, 0],
            "9B" => &[0x66, 0x0F, 0x1F, 0x84, 0x00, 0, 0, 0, 0],
            other => panic!(
                "a custom encoder this oracle doesn't know: {} {other}",
                inst.name()
            ),
        };
        return (!mem && disp8).then(|| nop.to_vec());
    }
    let kinds = inst.format.operands_by_kind();
    let imm = match kinds.last() {
        Some(K::Imm(loc)) => usize::from(loc.bytes()),
        _ => 0,
    };
    let (layout, w, mut out) = match &inst.encoding {
        Encoding::Rex(rex) => {
            let l = rex_layout(&kinds, rex.modrm.and_then(|m| m.digit()));
            (l, rex.w.to_string() == "W1", Vec::new())
        }
        Encoding::Vex(vex) => {
            let l = vex_layout(&kinds, vex.modrm.and_then(|m| m.digit()));
            (l, vex.w.to_string() == "W1", Vec::new())
        }
        Encoding::Evex(evex) => {
            let l = vex_layout(&kinds, evex.modrm.and_then(|m| m.digit()));
            (l, evex.w.to_string() == "W1", Vec::new())
        }
    };
    match (layout.rm, mem) {
        (Rm::Any, _) | (Rm::Reg | Rm::None, false) | (Rm::Mem, true) => {}
        _ => return None,
    }
    // The ModRM.reg field's value, and the register (or base) the r/m field names.
    let reg = layout.digit.unwrap_or(r.reg);
    let has_modrm = layout.rm != Rm::None;
    let rm = if has_modrm || layout.opcode_reg {
        r.rm
    } else {
        0
    };
    let ext = |n: u8| (n >> 3) & 1;
    match &inst.encoding {
        Encoding::Rex(rex) => {
            let p = &rex.opcodes.prefixes;
            out.extend(p.group1.as_ref().map(|g| match g {
                Group1Prefix::Lock => 0xF0,
                Group1Prefix::REPNorBND => 0xF2,
                Group1Prefix::REP_ => 0xF3,
            }));
            out.extend(p.group2.as_ref().map(|g| match g {
                Group2Prefix::CSorBNT => 0x2E,
                Group2Prefix::SS => 0x36,
                Group2Prefix::DSorBT => 0x3E,
                Group2Prefix::ES => 0x26,
                Group2Prefix::FS => 0x64,
                Group2Prefix::GS => 0x65,
            }));
            out.extend(p.group3.as_ref().map(|_| 0x66));
            out.extend(p.group4.as_ref().map(|_| 0x67));
            // REX: emitted when it sets a bit, or to select spl–dil for a byte operand in 4–7.
            if !kinds.is_empty() && !matches!(kinds.as_slice(), [K::Imm(_)]) {
                let byte_op = inst
                    .format
                    .locations()
                    .any(|l| l.bits() == 8 && matches!(l.kind(), K::Reg(_) | K::RegMem(_)));
                let special = |n: u8| (4..=7).contains(&n);
                let reg_field = if layout.opcode_reg { rm } else { reg };
                let names_reg = has_modrm || layout.opcode_reg;
                let must = byte_op
                    && names_reg
                    && (special(reg_field) || (!mem && has_modrm && special(rm)));
                let (rx, bx) = if layout.opcode_reg {
                    (0, ext(rm))
                } else if has_modrm {
                    (ext(reg), ext(rm))
                } else {
                    (0, 0)
                };
                let byte = 0x40 | u8::from(w) << 3 | rx << 2 | bx;
                if byte != 0x40 || must {
                    out.push(byte);
                }
            }
            if rex.opcodes.escape {
                out.push(0x0F);
            }
            let low = if layout.opcode_reg { rm & 7 } else { 0 };
            out.push(rex.opcodes.primary | low);
            out.extend(rex.opcodes.secondary);
        }
        Encoding::Vex(vex) => {
            let (r_, b_) = (1 - ext(reg), 1 - ext(rm));
            let vvvv = if layout.vvvv { r.vvvv } else { 0 };
            let pp = vex.pp.map_or(0, pp_bits);
            let mmmmm = map_bits(vex.mmmmm.expect("a VEX form names its map"));
            let l = vex.length.vex_bits();
            let tail = (!vvvv & 0xF) << 3 | l << 2 | pp;
            if b_ == 1 && !w && mmmmm == 1 {
                out.extend([0xC5, r_ << 7 | tail]);
            } else {
                out.extend([
                    0xC4,
                    r_ << 7 | 1 << 6 | b_ << 5 | mmmmm,
                    u8::from(w) << 7 | tail,
                ]);
            }
            out.push(vex.opcode);
        }
        Encoding::Evex(evex) => {
            let (r_, b_) = (1 - ext(reg), 1 - ext(rm));
            let vvvv = if layout.vvvv { r.vvvv } else { 0 };
            let pp = evex.pp.map_or(0, pp_bits);
            let mmm = map_bits(evex.mmm.expect("an EVEX form names its map"));
            let ll = evex.length.evex_bits();
            // R' and V' (inverted) stay 1: registers 0–15. No masking, no broadcast.
            out.extend([
                0x62,
                r_ << 7 | 1 << 6 | b_ << 5 | 1 << 4 | mmm,
                u8::from(w) << 7 | (!vvvv & 0xF) << 3 | 0b100 | pp,
                ll << 5 | 1 << 3,
            ]);
            out.push(evex.opcode);
        }
    }
    if has_modrm {
        modrm(&mut out, reg, rm, mem, disp8);
    }
    out.extend(std::iter::repeat_n(0x01, imm));
    if layout.is4 {
        out.push(r.is4 << 4);
    }
    Some(out)
}

fn pp_bits(pp: VexPrefix) -> u8 {
    match pp {
        VexPrefix::_66 => 0b01,
        VexPrefix::_F3 => 0b10,
        VexPrefix::_F2 => 0b11,
    }
}

fn map_bits(m: VexEscape) -> u8 {
    match m {
        VexEscape::_0F => 0b01,
        VexEscape::_0F38 => 0b10,
        VexEscape::_0F3A => 0b11,
    }
}

/// Where iced says control goes after `insn`, as the decoder's [`Kind`] must classify it.
fn flow_agrees(d: &Decoded, insn: &Instruction) -> bool {
    let target = insn.near_branch_target() as usize;
    match insn.flow_control() {
        FlowControl::Next => d.kind == Kind::Normal,
        // `int3` traps to the debugger and, resumed, falls through.
        FlowControl::Interrupt => d.kind == Kind::Normal && insn.mnemonic() == Mnemonic::Int3,
        FlowControl::UnconditionalBranch => d.kind == Kind::DirectJmp { target },
        FlowControl::ConditionalBranch => {
            matches!(d.kind, Kind::DirectJcc { target: t, cc } if t == target
                && insn.condition_code() as u8 == cc + 1)
        }
        FlowControl::Call => d.kind == Kind::DirectCall,
        FlowControl::IndirectCall => d.kind == Kind::IndirectCall,
        FlowControl::IndirectBranch => d.kind == Kind::IndirectJmp,
        FlowControl::Return => d.kind == Kind::Ret,
        FlowControl::Exception => d.kind == Kind::Ud2 && insn.mnemonic() == Mnemonic::Ud2,
        _ => false,
    }
}

/// The decoder against iced on one synthesized form: `Ok(false)` when the decoder rejects it,
/// `Ok(true)` when it models it soundly, and otherwise the first disagreement.
fn check_form(bytes: &[u8], insn: &Instruction) -> Result<bool, String> {
    let d = decode(bytes, 0);
    if matches!(d.kind, Kind::Undecodable | Kind::Forbidden(_)) {
        return Ok(false);
    }
    if d.len != insn.len() {
        return Err(format!("length: ours {}, iced {}", d.len, insn.len()));
    }
    let ours = d
        .operands
        .mem
        .map(|m| if m.rip { Mem { disp: 0, ..m } } else { m });
    if ours != iced_mem(insn) {
        return Err(format!(
            "memory operand: ours {ours:?}, iced {:?}",
            iced_mem(insn)
        ));
    }
    let (ours_w, iced_w) = (writes_of(&d), iced_gp_writes(insn));
    if iced_w & !ours_w != 0 {
        return Err(format!("GP writes: iced {iced_w:#06x}, ours {ours_w:#06x}"));
    }
    let mut f = InstructionInfoFactory::new();
    for m in f.info(insn).used_memory() {
        // Synthesized memory operands never use rsp, so an rsp-based access is an implicit one:
        // the push, pop, call or return the decoder must know moves rsp.
        if m.base() == Register::RSP {
            let stack = matches!(d.eff, Eff::Push { .. } | Eff::Pop { .. })
                || matches!(d.kind, Kind::DirectCall | Kind::IndirectCall | Kind::Ret);
            if !stack {
                return Err("an implicit stack access the decoder doesn't model".into());
            }
            continue;
        }
        if d.operands.mem.is_none() {
            return Err(format!("an implicit memory access: {m:?}"));
        }
        let reads = matches!(
            m.access(),
            OpAccess::Read | OpAccess::CondRead | OpAccess::NoMemAccess
        );
        if d.reads_only && !reads {
            return Err("writes the memory it claims only to read".into());
        }
    }
    // A reported width must not be *smaller* than the real access (that would let an access run
    // past a region's end unnoticed); a larger one is a sound over-approximation — a stricter bound.
    // `0` means "unknown, assume the widest", also sound.
    if d.operands.mem.is_some() && d.width != 0 && usize::from(d.width) < insn.memory_size().size()
    {
        return Err(format!(
            "access width under-reported: ours {}, iced {}",
            d.width,
            insn.memory_size().size()
        ));
    }
    if d.keeps_flags && insn.rflags_modified() != 0 {
        return Err("modifies the flags it claims to keep".into());
    }
    if !flow_agrees(&d, insn) {
        return Err(format!(
            "control flow: ours {:?}, iced {:?}",
            d.kind,
            insn.flow_control()
        ));
    }
    Ok(true)
}

/// Every form Cranelift's x64 encoder can emit in 64-bit mode, in each register/memory variant it
/// has, either is rejected by the decoder or decodes soundly (`check_form`).
#[test]
fn every_cranelift_form_is_rejected_or_modeled_soundly() {
    let mut failures = Vec::new();
    // Each variant's outcome: modeled or rejected, keyed by the form's name and whether its r/m
    // operand is in memory.
    let mut outcomes: BTreeMap<(String, bool), Vec<bool>> = BTreeMap::new();
    for inst in instructions::list() {
        if !in_64bit_mode(&inst.features) {
            continue;
        }
        // Variants: the register form, and the memory form at both a `disp8` and a `disp32` (an EVEX
        // `disp8` is rejected, its `disp32` modeled — both are exercised). `disp8` is irrelevant to
        // the register form, so it runs once there.
        let variants = [(false, true), (true, true), (true, false)];
        for (mem, disp8) in variants {
            for regs in [LOW, HIGH] {
                let Some(bytes) = synth(&inst, regs, mem, disp8) else {
                    continue;
                };
                let name = inst.name();
                let Some(insn) = iced_at(&bytes, 0).filter(|i| i.len() == bytes.len()) else {
                    failures.push(format!(
                        "{name}: iced doesn't decode the synthesized {bytes:02x?}"
                    ));
                    continue;
                };
                match check_form(&bytes, &insn) {
                    Ok(modeled) => outcomes.entry((name, mem)).or_default().push(modeled),
                    Err(why) => failures.push(format!("{name} {bytes:02x?}: {why}")),
                }
            }
        }
    }
    // A coverage report (shown under `--nocapture`): which forms the decoder rejects outright, and
    // how many variants were checked in all. Every emitted form either is rejected here or decodes
    // soundly; the rejections are the forms Cranelift never emits in guest code (`hlt`, the far
    // branches, `bt` with a memory bit-offset) plus the EVEX compressed-`disp8` operand.
    let rejected: Vec<String> = outcomes
        .iter()
        .filter(|(_, v)| v.iter().any(|m| !m))
        .map(|((n, mem), v)| {
            let variants = if v.iter().all(|m| !m) {
                ""
            } else {
                " (some variants)"
            };
            format!("{n}{}{variants}", if *mem { " [mem]" } else { "" })
        })
        .collect();
    eprintln!("rejected ({}):\n{}", rejected.len(), rejected.join("\n"));
    eprintln!(
        "{} variants checked",
        outcomes.values().map(Vec::len).sum::<usize>()
    );
    assert!(
        failures.is_empty(),
        "{} unsound decodes:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
