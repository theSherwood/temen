//! #2481 — a validator over the **machine code** the Cranelift backend emits, the layer nothing else
//! checks (`temen-verify` checks the IR; Cranelift's verifier checks CLIF; neither sees instruction
//! selection, register allocation, or encoding — where Cranelift's published sandbox escapes came
//! from). It decodes the bytes itself and never trusts Cranelift's view of them.
//!
//! **Slice 1: x86-64, shadow mode.** A hand-rolled decoder (owned in-tree so a codegen surprise is
//! patched here, not upstream) walks each emitted function and the validator checks two things:
//! 1. every byte decodes and nothing is outside an allowlist (no `syscall`, no privileged or
//!    segment/control-register instructions, …) — the start of control-flow integrity (I2);
//! 2. a direct near/short `jmp`/`jcc` lands on a basic-block start Cranelift reported
//!    (`machine_code_cfg_info` → `bb_starts`), not mid-instruction or outside the function.
//!
//! It is **shadow mode**: a disagreement is logged and counted, never fatal. The memory-access
//! policy (#2535), host-call argument contracts (#2534), the replay/trampoline hooks (#2536),
//! fail-closed, and aarch64 are later slices. The decoder is cross-checked against iced-x86 (a
//! dev-dependency, never in the product) on real compiled modules — see the `x86::tests` below.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// Whether the machine-code validator runs. False on a production compile (so it costs nothing, and
/// `jit_isa` does not ask Cranelift for block-layout info), and false on targets slice 1 does not
/// cover yet (only x86-64). True under the fuzzers and when `TEMEN_JIT_VALIDATE` is set. Cached.
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        cfg!(target_arch = "x86_64")
            && (cfg!(fuzzing) || std::env::var_os("TEMEN_JIT_VALIDATE").is_some())
    })
}

/// How many shadow-mode disagreements have been seen this process (so a fuzz run can assert the
/// count stayed at the baseline the corpus is known clean against, and a human sees a total).
static DISAGREEMENTS: AtomicU64 = AtomicU64::new(0);

/// The running disagreement count (#2481 shadow mode).
pub(crate) fn disagreement_count() -> u64 {
    DISAGREEMENTS.load(Ordering::Relaxed)
}

/// Validate one emitted function's `bytes` against `bb_starts` (its basic-block start offsets, from
/// `CompiledCode::bb_starts`). Shadow mode: every problem is logged and counted, never returned.
pub(crate) fn shadow_check(bytes: &[u8], bb_starts: &[u32]) {
    #[cfg(target_arch = "x86_64")]
    for f in x86::check(bytes, bb_starts) {
        // Count every disagreement (a fuzz run asserts the total), but cap the logging so a bad
        // decode on every module can't swamp a fuzzer's stderr.
        let n = DISAGREEMENTS.fetch_add(1, Ordering::Relaxed);
        if n < 64 {
            eprintln!("jit-validate(x86-64): {} at +0x{:x}", f.why, f.at);
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (bytes, bb_starts);
}

// --- test-only capture of real compiled code, cross-checked against iced-x86 ---
//
// The whole subsystem exists only for the x86-64 decoder cross-check (`x86::tests`), and its one
// caller in `lib.rs` is gated to match, so it is `all(test, target_arch = "x86_64")` — on another
// target (the aarch64 clippy lane) it would be dead code under `-D warnings`.

/// One captured function's machine code and its basic-block start offsets.
#[cfg(all(test, target_arch = "x86_64"))]
type CapturedFn = (Vec<u8>, Vec<u32>);

#[cfg(all(test, target_arch = "x86_64"))]
thread_local! {
    /// When armed, each function the compile defines appends its `(code, bb_starts)` here, so a test
    /// can decode real Cranelift output and compare our decoder to iced-x86.
    static CAPTURE: std::cell::RefCell<Option<Vec<CapturedFn>>> =
        const { std::cell::RefCell::new(None) };
}

/// Record one defined function's code + block starts, if a test armed the capture.
#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) fn capture_code(code: &[u8], bb_starts: &[u32]) {
    CAPTURE.with(|c| {
        if let Some(v) = c.borrow_mut().as_mut() {
            v.push((code.to_vec(), bb_starts.to_vec()));
        }
    });
}

/// Arm capture, clearing anything held.
#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) fn arm_capture() {
    CAPTURE.with(|c| *c.borrow_mut() = Some(Vec::new()));
}

/// Take the captured functions.
#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) fn take_capture() -> Vec<CapturedFn> {
    CAPTURE.with(|c| c.borrow_mut().take().unwrap_or_default())
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86 {
    //! The x86-64 instruction decoder (length + control-transfer + allowlist classification) and the
    //! slice-1 policy check. Only enough operand decoding to compute each instruction's length and
    //! find its control-flow destination; full operand decoding (the memory-access policy) is #2535.

    /// A machine-code policy disagreement: `why`, at byte offset `at` in the function.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub(crate) struct Finding {
        pub at: usize,
        pub why: String,
    }

    /// Validate `bytes` against `bb_starts`, returning every finding (slice-1 policy). Pure, so tests
    /// assert on the findings directly.
    ///
    /// Walk is **per basic block**: from each `bb_start`, decode forward until the block's
    /// non-fall-through terminator (`jmp`/`ret`/`ud2`/indirect `jmp`) or the next block start. The
    /// bytes between a terminator and the next block start are a constant island or a jump table that
    /// Cranelift placed there — **data, not code** — so the walk stops at the terminator and never
    /// decodes them. This is how the validator avoids mis-decoding embedded rodata without Cranelift
    /// exposing the text/data split (`code_info` gives only `total_size`).
    pub(crate) fn check(bytes: &[u8], bb_starts: &[u32]) -> Vec<Finding> {
        let mut findings = Vec::new();
        let mut starts: Vec<usize> = bb_starts.iter().map(|&o| o as usize).collect();
        starts.sort_unstable();
        starts.dedup();
        if starts.first() != Some(&0) {
            starts.insert(0, 0); // the entry is always a block start
        }
        let start_set = starts.clone();
        let is_block_start = |o: usize| start_set.binary_search(&o).is_ok();

        for (i, &block) in starts.iter().enumerate() {
            // This block's instructions occupy [block, limit): the next block start, or end of code.
            let limit = starts.get(i + 1).copied().unwrap_or(bytes.len());
            let mut pos = block;
            while pos < limit {
                let d = decode(bytes, pos);
                if d.len == 0 {
                    findings.push(Finding {
                        at: pos,
                        why: "undecodable instruction (decoder does not model this encoding)"
                            .into(),
                    });
                    break; // can't size it, so can't step past it within this block
                }
                match d.kind {
                    Kind::Forbidden(name) => findings.push(Finding {
                        at: pos,
                        why: format!("forbidden instruction: {name}"),
                    }),
                    Kind::DirectJmp { target } | Kind::DirectJcc { target } => {
                        if target >= bytes.len() {
                            findings.push(Finding {
                                at: pos,
                                why: format!("direct branch to +0x{target:x} is outside the code"),
                            });
                        } else if !is_block_start(target) {
                            findings.push(Finding {
                                at: pos,
                                why: format!("direct branch to +0x{target:x} is not a block start"),
                            });
                        }
                    }
                    // A direct `call rel32` to an external symbol is emitted with a relocation and a
                    // placeholder displacement, so its decoded target is meaningless until linked;
                    // checking call targets needs the relocation table (a later slice).
                    _ => {}
                }
                let next = pos + d.len;
                if next > limit {
                    findings.push(Finding {
                        at: pos,
                        why: format!(
                            "instruction overruns its block (ends +0x{next:x} > +0x{limit:x})"
                        ),
                    });
                    break;
                }
                pos = next;
                if !d.kind.falls_through() {
                    break; // block's terminator reached; any bytes to `limit` are island/table data
                }
            }
        }
        findings
    }

    /// What one decoded instruction is, for the slice-1 policy.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum Kind {
        /// Allowed, not a control transfer.
        Normal,
        /// `jmp rel8`/`rel32`: unconditional direct branch to `target` (absolute function offset).
        DirectJmp { target: usize },
        /// `jcc rel8`/`rel32`: conditional direct branch to `target`.
        DirectJcc { target: usize },
        /// `call rel32`: direct call (usually to an external symbol via relocation — its target is
        /// not checked in slice 1, so it carries none). Falls through.
        DirectCall,
        /// Indirect `call` (through a register or memory). Allowed; falls through. Its CFI is the §3c
        /// table dispatch, checked in a later slice.
        IndirectCall,
        /// Indirect `jmp` (through a register or memory, e.g. a `br_table` dispatch). Allowed; does
        /// not fall through, and is typically followed by the jump-table data.
        IndirectJmp,
        /// `ret`/`ret imm16`. Does not fall through.
        Ret,
        /// `ud2` — the trap Cranelift emits for `unreachable`. Allowed; does not fall through.
        Ud2,
        /// Outside the allowlist (`syscall`, `int`, privileged, segment/control-register, I/O, …).
        Forbidden(&'static str),
        /// The decoder does not model this encoding yet.
        Undecodable,
    }

    impl Kind {
        /// Whether control can continue to the next instruction after this one. `false` for the
        /// terminators that end a basic block without fall-through (`jmp`, `ret`, `ud2`, indirect
        /// `jmp`) — after which the next bytes may be a constant island or jump table, not code.
        fn falls_through(self) -> bool {
            !matches!(
                self,
                Kind::DirectJmp { .. } | Kind::IndirectJmp | Kind::Ret | Kind::Ud2
            )
        }
    }

    /// One decoded instruction: its length, its classification, and its operands (empty for
    /// instructions without a ModRM byte — the control-flow and no-operand forms).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct Decoded {
        pub len: usize,
        pub kind: Kind,
        // `operands` is consumed by the cross-check now and by the slice-2b policy next; see `Mem`.
        #[allow(dead_code)]
        pub operands: Operands,
    }

    const NO_OPERANDS: Operands = Operands {
        reg: None,
        rm_reg: None,
        mem: None,
    };

    const UNDECODABLE: Decoded = Decoded {
        len: 0,
        kind: Kind::Undecodable,
        operands: NO_OPERANDS,
    };

    /// Decode the instruction at `bytes[start..]`. Returns [`UNDECODABLE`] (length 0) for an encoding
    /// not modelled yet or a truncated instruction — a logged disagreement in shadow mode.
    pub(crate) fn decode(bytes: &[u8], start: usize) -> Decoded {
        let mut c = Cursor { bytes, pos: start };
        decode_inner(&mut c, start).unwrap_or(UNDECODABLE)
    }

    /// A bounds-checked forward reader; every read returns `None` past the end, so a truncated
    /// instruction decodes as `Undecodable` rather than panicking.
    struct Cursor<'a> {
        bytes: &'a [u8],
        pos: usize,
    }

    impl Cursor<'_> {
        fn peek(&self) -> Option<u8> {
            self.bytes.get(self.pos).copied()
        }
        fn next(&mut self) -> Option<u8> {
            let b = self.bytes.get(self.pos).copied()?;
            self.pos += 1;
            Some(b)
        }
        fn skip(&mut self, n: usize) -> Option<()> {
            self.pos = self.pos.checked_add(n)?;
            (self.pos <= self.bytes.len()).then_some(())
        }
        /// A signed `rel8`/`rel32` at the cursor, consuming it, as an `i64`.
        fn rel(&mut self, width: usize) -> Option<i64> {
            let v = match width {
                1 => i64::from(*self.bytes.get(self.pos)? as i8),
                4 => {
                    let b = self.bytes.get(self.pos..self.pos + 4)?;
                    i64::from(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                }
                _ => return None,
            };
            self.pos += width;
            Some(v)
        }
    }

    /// Operand-size inputs gathered from the prefixes and REX.
    #[derive(Default, Clone, Copy)]
    struct Pfx {
        op66: bool,  // 0x66 operand-size override
        rex_w: bool, // 64-bit operand size
        rex_r: bool, // ModRM.reg extension (bit 3)
        rex_x: bool, // SIB.index extension
        rex_b: bool, // ModRM.rm / SIB.base extension
    }

    fn decode_inner(c: &mut Cursor, start: usize) -> Option<Decoded> {
        let mut p = Pfx::default();
        // Legacy prefixes (any order/number in valid code; Cranelift emits at most a couple).
        loop {
            match c.peek()? {
                0x66 => {
                    p.op66 = true;
                    c.pos += 1;
                }
                // rep/repne (also SSE mandatory prefixes), address-size, lock, segment overrides:
                // consumed; none change our length model for the 64-bit encodings Cranelift emits.
                0xF2 | 0xF3 | 0x67 | 0xF0 | 0x2E | 0x36 | 0x3E | 0x26 | 0x64 | 0x65 => {
                    c.pos += 1;
                }
                _ => break,
            }
        }
        // VEX / EVEX (AVX): their prefix encodes the opcode map + ModRM-implied operands.
        match c.peek()? {
            0xC5 => return decode_vex2(c, start),
            0xC4 => return decode_vex3(c, start),
            0x62 => return decode_evex(c, start),
            _ => {}
        }
        // REX (0x40..=0x4F) — the last prefix before the opcode in 64-bit mode.
        if let Some(b @ 0x40..=0x4F) = c.peek() {
            p.rex_w = b & 0x08 != 0;
            p.rex_r = b & 0x04 != 0;
            p.rex_x = b & 0x02 != 0;
            p.rex_b = b & 0x01 != 0;
            c.pos += 1;
        }
        let op = c.next()?;
        if op == 0x0F {
            let op2 = c.next()?;
            match op2 {
                0x38 => decode_0f38(c, start, &p),
                0x3A => decode_0f3a(c, start, &p),
                _ => decode_0f(c, start, &p, op2),
            }
        } else {
            decode_1byte(c, start, &p, op)
        }
    }

    /// A memory operand `[base + index*scale + disp]`, or RIP-relative. Register numbers are the full
    /// 0..15 (REX-extended). `base`/`index` are `None` when absent.
    ///
    /// `#[allow(dead_code)]`: the decoder exposes these now and the iced-x86 cross-check consumes
    /// them, but the production reader — the slice-2b dataflow + memory-access policy — lands next.
    /// Scaffolding with a scheduled removal of the allow when that policy arrives (INVARIANTS #13).
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    #[allow(dead_code)]
    pub(crate) struct Mem {
        pub base: Option<u8>,
        pub index: Option<u8>,
        pub scale: u8,
        pub disp: i64,
        pub rip: bool,
    }

    /// The operands a ModRM instruction carries, as the policy/dataflow needs them: the `reg`-field
    /// register, and the r/m operand — either a register or a memory reference. (Operation class and
    /// read/write direction come in slice 2b, where the dataflow needs them.)
    ///
    /// `#[allow(dead_code)]`: see [`Mem`] — consumed by the cross-check now, by the policy in 2b.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    #[allow(dead_code)]
    pub(crate) struct Operands {
        pub reg: Option<u8>,
        pub rm_reg: Option<u8>,
        pub mem: Option<Mem>,
    }

    /// Consume a ModRM byte and whatever addressing bytes it implies (SIB, displacement) in 64-bit
    /// mode, then `imm` immediate bytes; classify as `kind` and attach the decoded operands.
    fn finish_modrm(
        c: &mut Cursor,
        start: usize,
        p: &Pfx,
        imm: usize,
        kind: Kind,
    ) -> Option<Decoded> {
        let operands = read_modrm(c, p)?;
        c.skip(imm)?;
        Some(Decoded {
            len: c.pos - start,
            kind,
            operands,
        })
    }

    /// Read a ModRM byte and its SIB/displacement in 64-bit mode, returning the operands and leaving
    /// the cursor just past them. The byte count consumed is identical to the length-only form this
    /// replaces, so instruction lengths are unchanged.
    fn read_modrm(c: &mut Cursor, p: &Pfx) -> Option<Operands> {
        let modrm = c.next()?;
        let md = modrm >> 6;
        let rm = modrm & 0x07;
        let reg = Some((u8::from(p.rex_r) << 3) | ((modrm >> 3) & 0x07));
        if md == 0b11 {
            // Register r/m: no SIB, no displacement.
            let rm_reg = Some((u8::from(p.rex_b) << 3) | rm);
            return Some(Operands {
                reg,
                rm_reg,
                mem: None,
            });
        }
        // Scale is reported as 1 when there is no index (iced's normalization), so default it to 1
        // and only derive it from the SIB scale field when an index is actually present.
        let mut m = Mem {
            scale: 1,
            ..Mem::default()
        };
        let mut disp_bytes = 0usize;
        if rm == 0b100 {
            // SIB byte.
            let sib = c.next()?;
            let ss = sib >> 6;
            let idx = (u8::from(p.rex_x) << 3) | ((sib >> 3) & 0x07);
            let base = (u8::from(p.rex_b) << 3) | (sib & 0x07);
            // index == 100 (no REX.X) encodes "no index".
            if idx != 0b100 {
                m.index = Some(idx);
                m.scale = 1 << ss;
            }
            if md == 0b00 && (sib & 0x07) == 0b101 {
                disp_bytes = 4; // base absent, disp32
            } else {
                m.base = Some(base);
            }
        } else if md == 0b00 && rm == 0b101 {
            m.rip = true; // RIP-relative disp32
            disp_bytes = 4;
        } else {
            m.base = Some((u8::from(p.rex_b) << 3) | rm);
        }
        match md {
            0b01 => disp_bytes = 1,
            0b10 => disp_bytes = 4,
            _ => {}
        }
        m.disp = read_disp(c, disp_bytes)?;
        Some(Operands {
            reg,
            rm_reg: None,
            mem: Some(m),
        })
    }

    /// Read a sign-extended displacement of `bytes` (0, 1, or 4) and advance the cursor.
    fn read_disp(c: &mut Cursor, bytes: usize) -> Option<i64> {
        let v = match bytes {
            0 => 0,
            1 => i64::from(*c.bytes.get(c.pos)? as i8),
            4 => {
                let b = c.bytes.get(c.pos..c.pos + 4)?;
                i64::from(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            }
            _ => return None,
        };
        c.pos += bytes;
        Some(v)
    }

    /// Size of an operand-size-dependent "immediate word" (`Iz`): 2 bytes under 0x66, else 4 (a
    /// REX.W instruction still takes a 4-byte sign-extended immediate).
    fn imm_z(p: &Pfx) -> usize {
        if p.op66 {
            2
        } else {
            4
        }
    }

    fn decode_1byte(c: &mut Cursor, start: usize, p: &Pfx, op: u8) -> Option<Decoded> {
        use Kind::*;
        match op {
            // --- forbidden / privileged / system ---
            0xF4 => return Some(one(c, start, Forbidden("hlt"))),
            0xFA => return Some(one(c, start, Forbidden("cli"))),
            0xFB => return Some(one(c, start, Forbidden("sti"))),
            0xCC => return Some(one(c, start, Normal)), // int3 — debugtrap/padding, allowed
            0xCD => return finish_imm(c, start, 1, Forbidden("int imm8")),
            0xCE => return Some(one(c, start, Forbidden("into"))),
            0xCF => return Some(one(c, start, Forbidden("iret"))),
            0xE4 | 0xE5 => return finish_imm(c, start, 1, Forbidden("in imm8")),
            0xE6 | 0xE7 => return finish_imm(c, start, 1, Forbidden("out imm8")),
            0xEC..=0xEF => return Some(one(c, start, Forbidden("in/out dx"))),
            0x6C..=0x6F => return Some(one(c, start, Forbidden("ins/outs"))),
            0xCB | 0xCA => return Some(one(c, start, Forbidden("far ret"))),
            0xEA => return Some(one(c, start, Forbidden("far jmp"))),
            0x9A => return Some(one(c, start, Forbidden("far call"))),
            // --- control transfers ---
            0xE9 => {
                let rel = c.rel(4)?; // jmp rel32
                return Some(rel_insn(start, c.pos, rel, Branch::Jmp));
            }
            0xEB => {
                let rel = c.rel(1)?; // jmp rel8
                return Some(rel_insn(start, c.pos, rel, Branch::Jmp));
            }
            0xE8 => {
                c.rel(4)?; // call rel32 — displacement consumed; target not checked in slice 1
                return Some(Decoded {
                    len: c.pos - start,
                    kind: DirectCall,
                    operands: NO_OPERANDS,
                });
            }
            0x70..=0x7F => {
                let rel = c.rel(1)?; // jcc rel8
                return Some(rel_insn(start, c.pos, rel, Branch::Jcc));
            }
            0xC3 => return Some(one(c, start, Ret)),
            0xC2 => return finish_imm(c, start, 2, Ret), // ret imm16
            _ => {}
        }
        match op {
            // ModRM, no immediate.
            0x00..=0x03
            | 0x08..=0x0B
            | 0x10..=0x13
            | 0x18..=0x1B
            | 0x20..=0x23
            | 0x28..=0x2B
            | 0x30..=0x33
            | 0x38..=0x3B
            | 0x84..=0x8B
            | 0x62
            | 0x63
            | 0x8D
            | 0x8F
            | 0xD0..=0xD3
            | 0xFE => finish_modrm(c, start, p, 0, Normal),
            // ModRM + imm8.
            0x80 | 0x82 | 0x83 | 0x6B | 0xC0 | 0xC1 | 0xC6 => finish_modrm(c, start, p, 1, Normal),
            // ModRM + immZ.
            0x81 | 0x69 | 0xC7 => finish_modrm(c, start, p, imm_z(p), Normal),
            // groups 3/5: immediate (if any) depends on the /reg field — peek at the ModRM.
            0xF6 => group3(c, start, p, 1),
            0xF7 => group3(c, start, p, imm_z(p)),
            0xFF => group5(c, start, p),
            // No ModRM, no immediate.
            0x06
            | 0x07
            | 0x0E
            | 0x16
            | 0x17
            | 0x1E
            | 0x1F
            | 0x27
            | 0x2F
            | 0x37
            | 0x3F
            | 0x40..=0x5F
            | 0x90..=0x99
            | 0x9B..=0x9F
            | 0xD7
            | 0xF5
            | 0xF8
            | 0xF9
            | 0xFC
            | 0xFD => Some(one(c, start, Normal)),
            // Accumulator + immediate.
            0x04 | 0x0C | 0x14 | 0x1C | 0x24 | 0x2C | 0x34 | 0x3C | 0xA8 | 0x6A => {
                finish_imm(c, start, 1, Normal)
            }
            0x05 | 0x0D | 0x15 | 0x1D | 0x25 | 0x2D | 0x35 | 0x3D | 0xA9 | 0x68 => {
                finish_imm(c, start, imm_z(p), Normal)
            }
            // mov reg, imm.
            0xB0..=0xB7 => finish_imm(c, start, 1, Normal),
            0xB8..=0xBF => finish_imm(c, start, if p.rex_w { 8 } else { imm_z(p) }, Normal),
            _ => Some(UNDECODABLE),
        }
    }

    /// Group 3 (`F6`/`F7`): `/0`,`/1` (test) carry an immediate; `/2..` (not/neg/mul/imul/div/idiv)
    /// do not.
    fn group3(c: &mut Cursor, start: usize, p: &Pfx, imm: usize) -> Option<Decoded> {
        let reg = (c.peek()? >> 3) & 0x07;
        finish_modrm(c, start, p, if reg <= 1 { imm } else { 0 }, Kind::Normal)
    }

    /// Group 5 (`FF`): `/2`,`/3` indirect call, `/4`,`/5` indirect jmp, others inc/dec/push r/m.
    fn group5(c: &mut Cursor, start: usize, p: &Pfx) -> Option<Decoded> {
        let reg = (c.peek()? >> 3) & 0x07;
        let kind = match reg {
            2 | 3 => Kind::IndirectCall,
            4 | 5 => Kind::IndirectJmp,
            _ => Kind::Normal,
        };
        finish_modrm(c, start, p, 0, kind)
    }

    fn decode_0f(c: &mut Cursor, start: usize, p: &Pfx, op2: u8) -> Option<Decoded> {
        use Kind::*;
        match op2 {
            0x05 => return Some(one(c, start, Forbidden("syscall"))),
            0x07 => return Some(one(c, start, Forbidden("sysret"))),
            0x34 => return Some(one(c, start, Forbidden("sysenter"))),
            0x35 => return Some(one(c, start, Forbidden("sysexit"))),
            0x0B => return Some(one(c, start, Ud2)), // the trap Cranelift emits for `unreachable`
            0x20..=0x23 => return finish_modrm(c, start, p, 0, Forbidden("mov cr/dr")),
            0x80..=0x8F => {
                let rel = c.rel(4)?; // jcc rel32
                return Some(rel_insn(start, c.pos, rel, Branch::Jcc));
            }
            _ => {}
        }
        match op2 {
            // ModRM + imm8 (pshuf, cmpps/sd, shufps, pinsrw/pextrw, the shift groups, shld/shrd imm8).
            0x70..=0x73 | 0xC2 | 0xC4 | 0xC5 | 0xC6 | 0xA4 | 0xAC => {
                finish_modrm(c, start, p, 1, Normal)
            }
            // The 0F opcodes with no ModRM byte Cranelift emits: emms, cpuid, rdtsc.
            0x77 | 0xA2 | 0x31 => Some(one(c, start, Normal)),
            // 0F AE (fence / clflush) and everything else in the 0F map: ModRM, no immediate.
            _ => finish_modrm(c, start, p, 0, Normal),
        }
    }

    fn decode_0f38(c: &mut Cursor, start: usize, p: &Pfx) -> Option<Decoded> {
        c.next()?; // opcode in the 0F38 map — all have ModRM, no immediate
        finish_modrm(c, start, p, 0, Kind::Normal)
    }

    fn decode_0f3a(c: &mut Cursor, start: usize, p: &Pfx) -> Option<Decoded> {
        c.next()?; // opcode in the 0F3A map — all have ModRM + imm8
        finish_modrm(c, start, p, 1, Kind::Normal)
    }

    fn decode_vex2(c: &mut Cursor, start: usize) -> Option<Decoded> {
        c.next()?; // 0xC5
        let b1 = c.next()?; // R vvvv L pp (R inverted; no X/B/W in 2-byte VEX)
        let op = c.next()?;
        let p = Pfx {
            rex_r: (b1 & 0x80) == 0,
            ..Pfx::default()
        };
        vex_body(c, start, &p, Map::Of, op)
    }

    fn decode_vex3(c: &mut Cursor, start: usize) -> Option<Decoded> {
        c.next()?; // 0xC4
        let b1 = c.next()?; // R X B mmmmm (R/X/B inverted)
        let b2 = c.next()?; // W vvvv L pp
        let op = c.next()?;
        let p = Pfx {
            rex_r: (b1 & 0x80) == 0,
            rex_x: (b1 & 0x40) == 0,
            rex_b: (b1 & 0x20) == 0,
            rex_w: (b2 & 0x80) != 0,
            ..Pfx::default()
        };
        let map = match b1 & 0x1f {
            1 => Map::Of,
            2 => Map::Of38,
            3 => Map::Of3a,
            _ => return Some(UNDECODABLE),
        };
        vex_body(c, start, &p, map, op)
    }

    fn decode_evex(c: &mut Cursor, start: usize) -> Option<Decoded> {
        c.next()?; // 0x62
        let b0 = c.next()?; // R X B R' 0 0 m m (R/X/B inverted)
        let b1 = c.next()?; // W v v v v 1 p p
        c.next()?; // p2
        let op = c.next()?;
        let p = Pfx {
            rex_r: (b0 & 0x80) == 0,
            rex_x: (b0 & 0x40) == 0,
            rex_b: (b0 & 0x20) == 0,
            rex_w: (b1 & 0x80) != 0,
            ..Pfx::default()
        };
        let map = match b0 & 0x07 {
            1 => Map::Of,
            2 => Map::Of38,
            3 => Map::Of3a,
            _ => return Some(UNDECODABLE),
        };
        vex_body(c, start, &p, map, op)
    }

    #[derive(Clone, Copy)]
    enum Map {
        Of,
        Of38,
        Of3a,
    }

    /// The ModRM (+imm8 for the 0F3A map and the 0F imm8 ops) body shared by VEX/EVEX; the opcode
    /// byte was already consumed.
    fn vex_body(c: &mut Cursor, start: usize, p: &Pfx, map: Map, op: u8) -> Option<Decoded> {
        let imm = match map {
            Map::Of3a => 1,
            Map::Of38 => 0,
            Map::Of => usize::from(matches!(op, 0x70..=0x73 | 0xC2 | 0xC4 | 0xC5 | 0xC6)),
        };
        finish_modrm(c, start, p, imm, Kind::Normal)
    }

    fn one(c: &Cursor, start: usize, kind: Kind) -> Decoded {
        Decoded {
            len: c.pos - start,
            kind,
            operands: NO_OPERANDS,
        }
    }

    fn finish_imm(c: &mut Cursor, start: usize, imm: usize, kind: Kind) -> Option<Decoded> {
        c.skip(imm)?;
        Some(Decoded {
            len: c.pos - start,
            kind,
            operands: NO_OPERANDS,
        })
    }

    enum Branch {
        Jmp,
        Jcc,
    }

    /// Classify a decoded `rel8`/`rel32` branch whose bytes end at `end` (= start+len), signed `rel`
    /// from the next instruction.
    fn rel_insn(start: usize, end: usize, rel: i64, br: Branch) -> Decoded {
        let target = (end as i64 + rel) as usize;
        let kind = match br {
            Branch::Jmp => Kind::DirectJmp { target },
            Branch::Jcc => Kind::DirectJcc { target },
        };
        Decoded {
            len: end - start,
            kind,
            operands: NO_OPERANDS,
        }
    }

    /// Visit each decoded instruction in basic-block order (the same traversal [`check`] uses):
    /// from each block start to its terminator, skipping inter-block islands/tables. Stops a block on
    /// a zero-length decode. Test-only, for the iced-x86 cross-check.
    #[cfg(test)]
    pub(crate) fn walk(bytes: &[u8], bb_starts: &[u32]) -> Vec<(usize, Decoded)> {
        let mut starts: Vec<usize> = bb_starts.iter().map(|&o| o as usize).collect();
        starts.sort_unstable();
        starts.dedup();
        if starts.first() != Some(&0) {
            starts.insert(0, 0);
        }
        let mut out = Vec::new();
        for (i, &block) in starts.iter().enumerate() {
            let limit = starts.get(i + 1).copied().unwrap_or(bytes.len());
            let mut pos = block;
            while pos < limit {
                let d = decode(bytes, pos);
                out.push((pos, d));
                if d.len == 0 || pos + d.len > limit {
                    break;
                }
                pos += d.len;
                if !d.kind.falls_through() {
                    break;
                }
            }
        }
        out
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::{compile, validate};
        use iced_x86::{Decoder, DecoderOptions, Instruction, OpKind, Register};
        use temen_text::parse_module;

        /// Compile `src` through the real JIT path and return each function's `(code, bb_starts)`.
        fn compiled(src: &str) -> Vec<(Vec<u8>, Vec<u32>)> {
            let m = parse_module(src).expect("parse");
            validate::arm_capture();
            let _ = compile(&m, 0).expect("compile");
            validate::take_capture()
        }

        /// iced-x86's decode of the one instruction at `bytes[at..]`, or `None` if iced rejects it.
        fn iced_at(bytes: &[u8], at: usize) -> Option<Instruction> {
            let mut dec = Decoder::with_ip(64, &bytes[at..], at as u64, DecoderOptions::NONE);
            let insn = dec.decode();
            (!insn.is_invalid()).then_some(insn)
        }

        /// A GP register's 0..15 encoding number, or `None` for a non-GP / absent register.
        fn gp(r: Register) -> Option<u8> {
            r.is_gpr64().then(|| r.number() as u8)
        }

        /// iced's single explicit memory operand as `(base, index, scale, disp, rip)`, in our shape,
        /// or `None` if the instruction has no explicit memory operand.
        fn iced_mem(insn: &Instruction) -> Option<Mem> {
            let has_mem = (0..insn.op_count()).any(|i| insn.op_kind(i) == OpKind::Memory);
            if !has_mem {
                return None;
            }
            let rip = insn.is_ip_rel_memory_operand();
            Some(Mem {
                base: if rip { None } else { gp(insn.memory_base()) },
                index: gp(insn.memory_index()),
                scale: insn.memory_index_scale() as u8,
                // iced reports the RIP-relative displacement as the resolved absolute target, not the
                // raw field, so only compare disp for non-rip operands.
                disp: if rip {
                    0
                } else {
                    insn.memory_displacement64() as i64
                },
                rip,
            })
        }

        /// The decoder's boundaries, validity, and memory-operand decode must match iced-x86 on real
        /// output, and `check` must find nothing on clean Cranelift code.
        fn cross_check(src: &str) {
            for (bytes, bb_starts) in compiled(src) {
                for (pos, d) in walk(&bytes, &bb_starts) {
                    assert_ne!(d.len, 0, "ours: undecodable at +0x{pos:x}\n{:02x?}", bytes);
                    let insn = iced_at(&bytes, pos).unwrap_or_else(|| {
                        panic!("iced rejects what we decoded at +0x{pos:x}\n{:02x?}", bytes)
                    });
                    assert_eq!(
                        d.len,
                        insn.len(),
                        "length mismatch at +0x{pos:x}: ours {} iced {}\n{:02x?}",
                        d.len,
                        insn.len(),
                        bytes
                    );
                    // When our decoder reports a memory operand, iced must see the same one.
                    if let Some(mem) = d.operands.mem {
                        let want = iced_mem(&insn);
                        // Compare the fields iced exposes identically (disp only when not rip-rel).
                        let ours = if mem.rip { Mem { disp: 0, ..mem } } else { mem };
                        assert_eq!(
                            Some(ours),
                            want,
                            "memory-operand mismatch at +0x{pos:x}: ours {:?} iced {:?}\n{:02x?}",
                            mem,
                            want,
                            bytes
                        );
                    }
                }
                let findings = check(&bytes, &bb_starts);
                assert!(
                    findings.is_empty(),
                    "clean Cranelift output flagged: {findings:?}\n{:02x?}",
                    bytes
                );
            }
        }

        #[test]
        fn integer_arithmetic_and_calls() {
            cross_check(
                "func (i64, i64) -> (i64) {\nblock 0 (v0: i64, v1: i64) {\n  \
                 v2 = i64.add v0 v1\n  v3 = i64.mul v2 v0\n  v4 = i64.sub v3 v1\n  \
                 return v4\n  }\n}\n",
            );
        }

        #[test]
        fn memory_access_mask_lowering() {
            // Loads/stores exercise the confinement lowering: cmp, select_spectre_guard (cmov),
            // branches — the bulk of the integer instruction set the decoder must size.
            cross_check(
                "memory 16\nfunc (i64) -> (i64) {\nblock 0 (v0: i64) {\n  \
                 v1 = i64.load v0\n  v2 = i64.const 8\n  v3 = i64.add v0 v2\n  \
                 i64.store v3 v1\n  return v1\n  }\n}\n",
            );
        }

        #[test]
        fn loops_and_conditional_branches() {
            // A back-edge loop: jcc/jmp with rel8 and rel32, and block starts the branch targets
            // must land on.
            cross_check(
                "func (i64) -> (i64) {\nblock 0 (v0: i64) {\n  v1 = i64.const 0\n  \
                 br 1(v0, v1)\n  }\nblock 1 (v2: i64, v3: i64) {\n  v4 = i64.const 0\n  \
                 v5 = i64.eq v2 v4\n  br_if v5 3(v3) 2(v2, v3)\n  }\n\
                 block 2 (v6: i64, v7: i64) {\n  v8 = i64.const 1\n  v9 = i64.sub v6 v8\n  \
                 v10 = i64.add v7 v6\n  br 1(v9, v10)\n  }\nblock 3 (v11: i64) {\n  \
                 return v11\n  }\n}\n",
            );
        }

        #[test]
        fn float_and_constants() {
            // Floats pull in SSE (movsd/addsd/mulsd) and a constant pool after the code — the island
            // the block-driven walk must not decode as instructions.
            cross_check(
                "func (f64, f64) -> (f64) {\nblock 0 (v0: f64, v1: f64) {\n  \
                 v2 = f64.add v0 v1\n  v3 = f64.mul v2 v0\n  v4 = f64.const 2.5\n  \
                 v5 = f64.div v3 v4\n  return v5\n  }\n}\n",
            );
        }

        #[test]
        fn select_and_extends() {
            // `select` lowers to cmov/setcc, and the i32->i64 extends to movsx/movzx — common opcodes
            // the other cases don't reach.
            cross_check(
                "func (i32, i32) -> (i64) {\nblock 0 (v0: i32, v1: i32) {\n  \
                 v2 = i32.eq v0 v1\n  v3 = i32.const 5\n  v4 = i32.const 9\n  \
                 v5 = select v2 v3 v4\n  v6 = i64.extend_i32_s v5\n  \
                 v7 = i64.extend_i32_u v0\n  v8 = i64.add v6 v7\n  return v8\n  }\n}\n",
            );
        }

        #[test]
        fn br_table_jump_table_island() {
            // `br_table` lowers to a bounds check + an indirect `jmp` through a jump table that
            // Cranelift embeds in the code buffer. The block-driven walk must stop at the indirect
            // jmp and not decode the table bytes as instructions.
            cross_check(
                "func (i32) -> (i32) {\nblock 0 (v0: i32) {\n  \
                 br_table v0 [1(), 2(), 3()] 4()\n  }\n\
                 block 1 () {\n  v1 = i32.const 10\n  return v1\n  }\n\
                 block 2 () {\n  v2 = i32.const 20\n  return v2\n  }\n\
                 block 3 () {\n  v3 = i32.const 30\n  return v3\n  }\n\
                 block 4 () {\n  v4 = i32.const 99\n  return v4\n  }\n}\n",
            );
        }

        #[test]
        fn decoder_rejects_a_syscall() {
            // Sanity on the allowlist: a hand-assembled `syscall` (0F 05) is Forbidden, and a `ret`
            // (C3) is Ret, with a `mov eax, 1` (B8 01 00 00 00) decoded at length 5 between them.
            let bytes = [0xB8, 0x01, 0x00, 0x00, 0x00, 0x0F, 0x05, 0xC3];
            assert_eq!(decode(&bytes, 0).len, 5);
            assert!(matches!(decode(&bytes, 5).kind, Kind::Forbidden("syscall")));
            assert!(matches!(decode(&bytes, 7).kind, Kind::Ret));
        }
    }
}
