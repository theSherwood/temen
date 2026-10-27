//! #2481 — a validator over the **machine code** the Cranelift backend emits, the layer nothing else
//! checks (`temen-verify` checks the IR; Cranelift's verifier checks CLIF; neither sees instruction
//! selection, register allocation, or encoding — where Cranelift's published sandbox escapes came
//! from). It decodes the bytes itself and never trusts Cranelift's view of them.
//!
//! **x86-64, shadow mode.** A hand-rolled decoder (owned in-tree so a codegen surprise is patched
//! here, not upstream) walks each emitted function. The validator rejects anything outside an
//! allowlist, recovers the machine CFG, and runs an abstract interpretation over it that re-proves
//! every memory access lands in a known region — a window access through a D63-confined or bounded
//! index stays within the reservation. Only a linear check of the state proposed for each block is
//! trusted (`x86::check`); the claim it carries, and why it holds, are in DESIGN §2a.
//!
//! It is **shadow mode**: a disagreement is logged and counted, never fatal. #2603 is the plan to
//! fail closed (the remaining regions and control-flow integrity, host-call contracts #2534, one gate
//! for every code origin #2536); aarch64 is #2642. The decoder is cross-checked against iced-x86 (a
//! dev-dependency, never in the product) on real compiled modules — see the `x86::tests` below.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// Whether the machine-code validator runs. False on a production compile (so it costs nothing) and
/// on targets it does not cover yet (only x86-64; aarch64 is #2642). True under the fuzzers and when
/// `TEMEN_JIT_VALIDATE` is set. Cached.
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

/// The module's §4 window geometry (a per-compile hint, #2603): the confinement mask
/// (`reserved − 1`) and the guard-page offset (`round_up(reserved, page)`, the D63 redirect target).
/// `None` for a memoryless module. The window-access policy re-proves every access's physical offset
/// stays within the reservation `[0, guard_offset + page)` against these — everything there is mapped
/// or `PROT_NONE` guard, so an access that stays inside it can fault but never escape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WindowLayout {
    pub mask: u64,
    pub guard_offset: u64,
}

/// The GP registers (x86-64 numbering, a bit each) a SysV-family call clobbers: Cranelift's
/// `SYSV_CLOBBERS`, shared by a Tail-CC callee, a host function and a libcall — rax, rcx, rdx, rsi,
/// rdi and r8–r11. rbx, rbp and r12–r15 are callee-saved (#2603 hole 1).
pub(crate) const SYSV_CLOBBERS: u16 = 0b0000_1111_1100_0111;

/// What a call does to its caller, as far as the validator models it (#2603): the GP registers the
/// callee may leave changed, and the bytes of stack arguments it pops on return (`None` when that
/// isn't known, so the caller's `rsp` is unknown after the call). PR 3 adds the argument tags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Contract {
    pub clobbers: u16,
    pub pops: Option<u32>,
}

/// The contract of a callee with signature `sig`. Every convention temen calls clobbers the SysV
/// caller-saved registers, except `PreserveAll`, which clobbers only the registers it returns in. A
/// `Tail` callee pops its stack arguments; the others leave them to the caller. `None` for a
/// convention temen never calls.
pub(crate) fn contract(sig: &cranelift_codegen::ir::Signature) -> Option<Contract> {
    use cranelift_codegen::isa::CallConv;
    let (clobbers, pops) = match sig.call_conv {
        CallConv::Tail => (SYSV_CLOBBERS, tail_stack_args(sig)),
        CallConv::SystemV | CallConv::WindowsFastcall => (SYSV_CLOBBERS, Some(0)),
        CallConv::PreserveAll if sig.returns.is_empty() => (0, Some(0)),
        CallConv::PreserveAll => (0b0101, Some(0)), // rax, rdx
        _ => return None,
    };
    Some(Contract { clobbers, pops })
}

/// The bytes of stack arguments a `Tail` callee with signature `sig` receives, and so pops: Cranelift
/// x64's argument layout, which passes the first six integer arguments in registers and the first
/// eight float and vector ones in xmm registers, the rest in stack slots of at least 8 bytes aligned
/// to their size, the area rounded up to 16. `None` for a parameter this rule doesn't cover (an
/// `i128`, a struct argument), so the caller's `rsp` is left unknown rather than guessed.
fn tail_stack_args(sig: &cranelift_codegen::ir::Signature) -> Option<u32> {
    use cranelift_codegen::ir::ArgumentPurpose;
    let (mut ints, mut vecs, mut stack) = (0u32, 0u32, 0u32);
    for p in &sig.params {
        let ty = p.value_type;
        if matches!(p.purpose, ArgumentPurpose::StructArgument(_)) || ty.bits() > 128 {
            return None;
        }
        let (used, max) = if ty.is_vector() || ty.is_float() {
            (&mut vecs, 8)
        } else if ty.bits() <= 64 {
            (&mut ints, 6)
        } else {
            return None; // i128
        };
        if *used < max {
            *used += 1;
        } else {
            let size = ty.bytes().max(8);
            stack = stack.next_multiple_of(size) + size;
        }
    }
    Some(stack.next_multiple_of(16))
}

/// One relocation in a function's code: the bytes the linker rewrites at `at` (their width is
/// `kind`'s) and the contract of the callee it names, `None` for anything the module doesn't allow
/// code to name. What runs at patched bytes is not what the validator reads, so no fact is ever
/// taken from them (#2603).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Reloc {
    pub at: u32,
    pub kind: cranelift_codegen::binemit::Reloc,
    pub addend: i64,
    pub callee: Option<Contract>,
}

impl Reloc {
    /// How many bytes it patches. A kind this table doesn't size is taken as 8, the widest x86-64
    /// relocation field: over-covering is sound.
    pub(crate) fn len(&self) -> usize {
        use cranelift_codegen::binemit::Reloc as R;
        match self.kind {
            R::Abs4
            | R::X86PCRel4
            | R::X86CallPCRel4
            | R::X86CallPLTRel4
            | R::X86GOTPCRel4
            | R::X86SecRel => 4,
            _ => 8,
        }
    }
}

/// The validator's view of a function's relocations: each one's patched bytes, and the contract of
/// what it names — `callee(id)` for a function the module declares, a libcall's from its signature
/// under `cc`, and `None` for anything else (data, a linker symbol, an offset in a function).
pub(crate) fn relocs(
    relocs: &[cranelift_module::ModuleReloc],
    callee: &dyn Fn(u32) -> Option<Contract>,
    cc: cranelift_codegen::isa::CallConv,
) -> Vec<Reloc> {
    use cranelift_module::ModuleRelocTarget as T;
    relocs
        .iter()
        .map(|r| Reloc {
            at: r.offset,
            kind: r.kind,
            addend: r.addend,
            callee: match &r.name {
                T::User {
                    namespace: 0,
                    index,
                } => callee(*index),
                T::LibCall(lc) => contract(&lc.signature(cc, cranelift_codegen::ir::types::I64)),
                _ => None,
            },
        })
        .collect()
}

/// What the validator knows of a function's entry (#2603). A guest function enters with its context
/// params in `sig_from`'s registers and accesses its module's window (`None` for a memoryless
/// module). A trampoline is called by the host, under the host's ABI with its own arguments; until
/// #2603 PR 3 gives each one a contract, it gets the decode and control-flow layers but not the
/// memory policy, which would otherwise read its arguments as a guest's context params.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Entry {
    // Only the x86-64 validator reads the layout (aarch64's is #2642).
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    Guest(Option<WindowLayout>),
    Host,
}

/// Validate one emitted function's `bytes` against its relocations and what is known of its
/// `entry`. Shadow mode: every problem is logged and counted, never returned.
pub(crate) fn shadow_check(bytes: &[u8], relocs: &[Reloc], entry: Entry) {
    #[cfg(target_arch = "x86_64")]
    for f in x86::check(bytes, relocs, entry) {
        // Count every disagreement (a fuzz run asserts the total), but cap the logging so a bad
        // decode on every module can't swamp a fuzzer's stderr.
        let n = DISAGREEMENTS.fetch_add(1, Ordering::Relaxed);
        if n < 64 {
            eprintln!("jit-validate(x86-64): {} at +0x{:x}", f.why, f.at);
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (bytes, relocs, entry);
}

// --- test-only capture of real compiled code, cross-checked against iced-x86 ---
//
// The whole subsystem exists only for the x86-64 decoder cross-check (`x86::tests`), and its one
// caller in `lib.rs` is gated to match, so it is `all(test, target_arch = "x86_64")` — on another
// target (the aarch64 clippy lane) it would be dead code under `-D warnings`.

/// One captured function: its machine code, relocations, the module's window layout, and its own
/// contract (what a caller expects its `ret` to pop).
#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) struct CapturedFn {
    pub code: Vec<u8>,
    pub relocs: Vec<Reloc>,
    pub layout: Option<WindowLayout>,
    pub contract: Option<Contract>,
}

#[cfg(all(test, target_arch = "x86_64"))]
thread_local! {
    /// When armed, each function the compile defines appends its capture here, so a test can decode
    /// real Cranelift output and compare our decoder to iced-x86 / run the policy.
    static CAPTURE: std::cell::RefCell<Option<Vec<CapturedFn>>> =
        const { std::cell::RefCell::new(None) };
}

/// Record one defined function, if a test armed the capture.
#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) fn capture_code(f: CapturedFn) {
    CAPTURE.with(|c| {
        if let Some(v) = c.borrow_mut().as_mut() {
            v.push(f);
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
    //! The x86-64 instruction decoder (length + operands + control-transfer + allowlist
    //! classification), the machine-CFG recovery and abstract dataflow (#2535), and the
    //! memory-access confinement policy (#2535/#2603) they feed.

    use super::{Contract, Entry, Reloc, WindowLayout, SYSV_CLOBBERS};

    /// A machine-code policy disagreement: `why`, at byte offset `at` in the function.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub(crate) struct Finding {
        pub at: usize,
        pub why: String,
    }

    /// x86-64 GP register numbers the entry ABI and the policy refer to by name (Tail CC, §3b).
    mod reg {
        pub const RCX: u8 = 1; // stack_limit (arg 3)
        pub const RDX: u8 = 2; // vmctx / trap_out (arg 2)
        pub const RSP: u8 = 4; // stack pointer
        pub const RSI: u8 = 6; // fn_table_base (arg 1)
        pub const RDI: u8 = 7; // mem_base (arg 0)
    }

    /// The largest machine access width (AVX-512 zmm) — used as a sound over-estimate when checking
    /// that a confined access's displacement rides within the guard slack. (Decoding the exact width
    /// is a precision refinement; over-estimating only ever rejects, never admits, so it is sound.)
    const MAX_ACCESS_WIDTH: u64 = 64;

    /// What a GP register holds, for the #2535 window-access policy. A finite lattice: [`Fact::meet`]
    /// is the fact true on *both* predecessors (`Unknown` when they disagree). `Bounded`'s upper
    /// bound widens under meet and is dropped to `Unknown` at a loop header, bounding the fixpoint.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Fact {
        /// The window base `mem_base` (rdi at entry) — the base of a confined window access.
        MemBase,
        /// `fn_table_base` (rsi) — the base of a `call.dyn` table access.
        FnTableBase,
        /// The `vmctx`/trap-out pointer (rdx).
        Vmctx,
        /// `stack_limit` (rcx).
        StackLimit,
        /// A known constant (e.g. the guard-offset materialization `mov r, imm`).
        Const(i64),
        /// In `[0, bound) ∪ {guard}` (unsigned): what a `cmov` leaves when it keeps a compared value
        /// only while it is below `bound` and otherwise takes the constant `guard`. D63's redirect
        /// (`cmp idx, reserved; cmovae idx, guard_offset`) and br_table's index clamp
        /// (`cmp idx, n; cmovb n, idx`) are both this shape (#2603).
        Confined { bound: u64, guard: u64 },
        /// Proven in `[0, ub)` (e.g. `and r, (2^k − 1)`).
        Bounded(u64),
        /// A pointer into the stack, at entry-relative offset `Some(off)` (`rsp` is `Stack(Some(0))`
        /// at entry, pointing at the return address), or somewhere in it at an unknown offset (`None`).
        Stack(Option<i64>),
        /// The address of function offset `.0`, computed by `lea r, [rip + disp]` (its constant pool
        /// or a jump table).
        RipRel(i64),
        /// One entry of the `n`-entry jump table at function offset `at`, sign-extended: what
        /// `movsxd e, [t + idx*4]` reads with `t` the table's address and `idx < n`.
        TableEntry { at: u32, n: u32 },
        /// The jump table's address plus one of its entries: the `jmp` target Cranelift's
        /// `JmpTableSeq` computes, one of the `n` targets the table at `at` lists.
        TableTarget { at: u32, n: u32 },
        /// The address of a callee a relocation names (`movabs r, sym`), which a call through the
        /// register calls under the callee's contract.
        FnAddr(Contract),
        /// No information.
        Unknown,
    }

    impl Fact {
        /// The fact true on both incoming paths. It is also the order the checker relies on: `a`
        /// implies `b` exactly when `a.meet(b) == b`.
        fn meet(self, other: Fact) -> Fact {
            match (self, other) {
                (a, b) if a == b => a,
                (Fact::Bounded(a), Fact::Bounded(b)) => Fact::Bounded(a.max(b)),
                (Fact::Confined { bound: a, guard: g }, Fact::Confined { bound: b, guard: h })
                    if g == h =>
                {
                    Fact::Confined {
                        bound: a.max(b),
                        guard: g,
                    }
                }
                (Fact::Stack(_), Fact::Stack(_)) => Fact::Stack(None),
                _ => Fact::Unknown,
            }
        }
    }

    /// How many spill slots the state tracks. Register allocation spills a context param (mem_base
    /// and friends) to the stack in a high-pressure function; a handful of slots covers that.
    /// Overflow simply stops tracking new slots — sound, since an untracked reload reads `Unknown`.
    const SLOT_CAP: usize = 8;

    /// One tracked spill slot: the 8-byte value at entry-relative stack offset `off`. Keying by the
    /// offset from the entry `rsp` (not by the instruction's `(base, disp)`) gives a slot one identity
    /// however `rsp` moves and whichever stack pointer addresses it (#2603 hole 6).
    #[derive(Clone, Copy, PartialEq, Eq)]
    struct Slot {
        off: i64,
        fact: Fact,
    }

    /// The abstract state at a program point: a fact per GP register, plus the tracked spill slots.
    #[derive(Clone, Copy, PartialEq, Eq)]
    struct State {
        regs: [Fact; 16],
        slots: [Option<Slot>; SLOT_CAP],
    }

    impl State {
        /// Nothing known: the state a block-local derivation starts from, which therefore holds of
        /// every execution that reaches the block.
        fn top() -> State {
            State {
                regs: [Fact::Unknown; 16],
                slots: [None; SLOT_CAP],
            }
        }
        /// The entry ABI: the four context params in their argument registers, `rsp` at the return
        /// address, everything else unknown, no slots.
        fn entry() -> State {
            let mut regs = [Fact::Unknown; 16];
            regs[reg::RDI as usize] = Fact::MemBase;
            regs[reg::RSI as usize] = Fact::FnTableBase;
            regs[reg::RDX as usize] = Fact::Vmctx;
            regs[reg::RCX as usize] = Fact::StackLimit;
            regs[reg::RSP as usize] = Fact::Stack(Some(0));
            State {
                regs,
                slots: [None; SLOT_CAP],
            }
        }
        fn meet(&self, other: &State) -> State {
            let mut regs = self.regs;
            for (r, &o) in regs.iter_mut().zip(other.regs.iter()) {
                *r = r.meet(o);
            }
            // A slot survives the meet only with a fact both paths imply.
            let mut slots = [None; SLOT_CAP];
            let mut n = 0;
            for s in self.slots.iter().flatten() {
                let fact = s.fact.meet(other.slot_get(s.off));
                if fact != Fact::Unknown && n < SLOT_CAP {
                    slots[n] = Some(Slot { off: s.off, fact });
                    n += 1;
                }
            }
            State { regs, slots }
        }
        /// The fact spilled to the 8-byte slot at `off`, or `Unknown` if no slot is tracked there.
        fn slot_get(&self, off: i64) -> Fact {
            self.slots
                .iter()
                .flatten()
                .find(|s| s.off == off)
                .map_or(Fact::Unknown, |s| s.fact)
        }
        /// Record `fact` in the 8-byte slot at `off`, whose bytes the caller has already clobbered.
        fn slot_set(&mut self, off: i64, fact: Fact) {
            if fact == Fact::Unknown {
                return;
            }
            if let Some(e) = self.slots.iter_mut().find(|s| s.is_none()) {
                *e = Some(Slot { off, fact });
            }
            // full ⇒ drop (sound: the reload reads Unknown)
        }
        /// Forget every slot that overlaps the bytes `[lo, hi)`.
        fn clobber(&mut self, lo: i64, hi: i64) {
            for s in self.slots.iter_mut() {
                if matches!(s, Some(e) if e.off < hi && lo < e.off.saturating_add(8)) {
                    *s = None;
                }
            }
        }
        fn clear_slots(&mut self) {
            self.slots = [None; SLOT_CAP];
        }
        /// Loop-header widening (#2535): a `Bounded` carried into a header could grow each iteration,
        /// so drop it to `Unknown` (in registers and slots). The context-param and `Confined` facts
        /// are loop-invariant and survive, so nearly every loop still settles in one round.
        fn widen(&mut self) {
            for f in &mut self.regs {
                if matches!(f, Fact::Bounded(_)) {
                    *f = Fact::Unknown;
                }
            }
            for s in self.slots.iter_mut() {
                if matches!(s, Some(e) if matches!(e.fact, Fact::Bounded(_))) {
                    *s = None;
                }
            }
        }
    }

    /// The edge condition the checker requires of the proposed states (`out ⊑ inn`): every fact
    /// `inn` asserts is implied by `out`, in the registers and the slots alike.
    fn implies(out: &State, inn: &State) -> bool {
        out.regs
            .iter()
            .zip(&inn.regs)
            .all(|(&o, &i)| o.meet(i) == i)
            && inn
                .slots
                .iter()
                .flatten()
                .all(|s| out.slot_get(s.off).meet(s.fact) == s.fact)
    }

    /// What the transfer reads besides the instruction: the function's bytes (a compare bound sits
    /// in its own constant pool), its relocations, and the module's window layout.
    #[derive(Clone, Copy)]
    struct Ctx<'a> {
        bytes: &'a [u8],
        relocs: &'a [Reloc],
        layout: Option<WindowLayout>,
    }

    impl Ctx<'_> {
        /// Whether a relocation patches any of the bytes `[at, at + len)`.
        fn patched(&self, at: usize, len: usize) -> bool {
            len != 0
                && self.relocs.iter().any(|r| {
                    let start = r.at as usize;
                    start < at + len && at < start + r.len()
                })
        }
        /// The relocation on the `rel32` of the 5-byte `call`/`jmp` `d` at `pos`, if it is the one
        /// a call or tail call carries (PC-relative to the field's end, addend −4).
        fn call_reloc(&self, pos: usize, d: &Decoded) -> Option<&Reloc> {
            use cranelift_codegen::binemit::Reloc as R;
            (d.len == 5).then_some(())?;
            self.relocs.iter().find(|r| {
                r.at as usize == pos + 1
                    && matches!(r.kind, R::X86CallPCRel4 | R::X86CallPLTRel4)
                    && r.addend == -4
            })
        }
        /// The relocation on the `imm64` of the `movabs` `d` at `pos`, if it is the absolute address
        /// of a symbol (`Abs8`, addend 0).
        fn abs_reloc(&self, pos: usize, d: &Decoded) -> Option<&Reloc> {
            use cranelift_codegen::binemit::Reloc as R;
            (matches!(d.eff, Eff::MovImm { .. }) && d.imm_len == 8).then_some(())?;
            let at = pos + d.len - 8;
            self.relocs
                .iter()
                .find(|r| r.at as usize == at && r.kind == R::Abs8 && r.addend == 0)
        }
        /// The relocation on an instruction's operand field the validator recognizes: a call's or tail
        /// call's `rel32`, or a `movabs`'s `imm64`.
        fn field_reloc(&self, pos: usize, d: &Decoded) -> Option<&Reloc> {
            match d.kind {
                Kind::DirectCall | Kind::DirectJmp { .. } => self.call_reloc(pos, d),
                _ => self.abs_reloc(pos, d),
            }
        }
        /// The `w`-byte little-endian constant at function offset `at`, if the function holds all of
        /// it and no relocation patches any of it.
        fn constant(&self, at: i64, w: usize) -> Option<u64> {
            let at = usize::try_from(at).ok()?;
            let b = self.bytes.get(at..at.checked_add(w)?)?;
            if self.patched(at, w) {
                return None;
            }
            let mut v = [0u8; 8];
            v[..w].copy_from_slice(b);
            Some(u64::from_le_bytes(v))
        }
        /// The function offset an instruction's rip-relative operand points at — unless the linker
        /// patches its displacement, which then names a symbol rather than the function's own bytes.
        fn rip_target(&self, d: &Decoded, end: usize) -> Option<i64> {
            let m = d.operands.mem.filter(|m| m.rip)?;
            let disp_at = end.checked_sub(usize::from(d.imm_len) + 4)?;
            (!self.patched(disp_at, 4)).then(|| end as i64 + m.disp)
        }
    }

    /// Whether the direct `jmp` `d` at `pos` is a tail call: a `jmp rel32` that carries a call
    /// relocation leaves the function for the symbol it names, so it has no successor here.
    fn tail_call(d: &Decoded, pos: usize, ctx: Ctx) -> bool {
        matches!(d.kind, Kind::DirectJmp { .. }) && ctx.call_reloc(pos, d).is_some()
    }

    /// A function's machine CFG: the block leaders, each block's decoded instructions, and each
    /// block's successors (indices into `leaders`).
    struct Cfg {
        leaders: Vec<usize>,
        blocks: Vec<Vec<(usize, Decoded)>>,
        succs: Vec<Vec<usize>>,
    }

    impl Cfg {
        /// Layers 1–2: recover the machine basic blocks, decode each (pushing the layer-1 findings),
        /// and collect their successors, an indirect `jmp`'s read from its jump table.
        fn build(ctx: Ctx, findings: &mut Vec<Finding>) -> Cfg {
            let bytes = ctx.bytes;
            let leaders = recover_leaders(ctx);
            let nb = leaders.len();
            let index_of = |o: usize| leaders.binary_search(&o).ok();
            let limit_of = |i: usize| leaders.get(i + 1).copied().unwrap_or(bytes.len());
            let mut blocks = Vec::with_capacity(nb);
            let mut succs: Vec<Vec<usize>> = vec![Vec::new(); nb];
            for i in 0..nb {
                let (insns, direct) = decode_block(ctx, leaders[i], limit_of(i), findings);
                for s in direct {
                    if let Some(si) = index_of(s) {
                        if !succs[i].contains(&si) {
                            succs[i].push(si);
                        }
                    }
                }
                if let Some(&(at, _)) = insns.last().filter(|(_, d)| d.kind == Kind::IndirectJmp) {
                    match table_targets(&insns, ctx) {
                        Ok(targets) => {
                            for t in targets {
                                match index_of(t) {
                                    Some(si) if !succs[i].contains(&si) => succs[i].push(si),
                                    Some(_) => {}
                                    None => findings.push(Finding {
                                        at,
                                        why: format!(
                                            "jump-table target +0x{t:x} is not a block start"
                                        ),
                                    }),
                                }
                            }
                        }
                        Err(why) => findings.push(Finding { at, why }),
                    }
                }
                blocks.push(insns);
            }
            classify_relocs(ctx, &blocks, findings);
            Cfg {
                leaders,
                blocks,
                succs,
            }
        }
    }

    /// Layer 1's relocation rule (#2603): every relocation must sit on an operand field the validator
    /// recognizes — a call's or tail call's `rel32`, or a `movabs`'s `imm64` — of an instruction it
    /// decodes, and name a callee the module allows. A `call rel32` must carry one: without it, the
    /// call lands at an offset of the code's own choosing.
    fn classify_relocs(ctx: Ctx, blocks: &[Vec<(usize, Decoded)>], findings: &mut Vec<Finding>) {
        let mut claimed = vec![false; ctx.relocs.len()];
        for &(pos, d) in blocks.iter().flatten() {
            let field = ctx.field_reloc(pos, &d).map(|r| r.at);
            for (i, r) in ctx.relocs.iter().enumerate() {
                let start = r.at as usize;
                if !(start < pos + d.len && pos < start + r.len()) {
                    continue;
                }
                claimed[i] = true;
                let why = if field != Some(r.at) {
                    "patches this instruction outside a call target or a movabs immediate"
                } else if r.callee.is_none() {
                    "names a target the module does not allow code to call"
                } else {
                    continue;
                };
                findings.push(Finding {
                    at: pos,
                    why: format!("relocation at +0x{start:x} {why}"),
                });
            }
            if d.kind == Kind::DirectCall && field.is_none() {
                findings.push(Finding {
                    at: pos,
                    why: "direct call without a relocation".into(),
                });
            }
        }
        for (r, _) in ctx.relocs.iter().zip(&claimed).filter(|(_, &c)| !c) {
            findings.push(Finding {
                at: r.at as usize,
                why: format!(
                    "relocation at +0x{:x} patches no instruction the code runs",
                    r.at
                ),
            });
        }
    }

    /// The targets of the indirect `jmp` that ends `insns`: the entries of the jump table its register
    /// was computed from — Cranelift's `JmpTableSeq` (`lea t, [rip + table]; movsxd e, [t + idx*4];
    /// add t, e; jmp t`) behind br_table's index clamp, which the `cmov` rule confines (#2603 hole
    /// 11). Derived in the block alone, from no assumption about its entry, so the targets hold on
    /// every execution that reaches the `jmp`. `Err` says why it is not such a jump.
    fn table_targets(insns: &[(usize, Decoded)], ctx: Ctx) -> Result<Vec<usize>, String> {
        let Some(&(_, d)) = insns.last() else {
            return Err("empty block".into());
        };
        let Some(r) = d.operands.rm_reg else {
            return Err("indirect jmp through memory".into());
        };
        let out = transfer_block(insns, State::top(), None, ctx);
        let Fact::TableTarget { at, n } = out.regs[r as usize & 15] else {
            return Err(format!(
                "indirect jmp through r{r}, which holds no jump-table target"
            ));
        };
        (0..i64::from(n))
            .map(|i| {
                let entry = ctx.constant(i64::from(at) + 4 * i, 4)? as u32 as i32;
                let t = i64::from(at) + i64::from(entry);
                usize::try_from(t).ok().filter(|&t| t < ctx.bytes.len())
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| "a jump-table entry targets outside the function".into())
    }

    /// The fact `movsxd e, dword [m]` leaves: an entry of the function's own jump table when `m` is
    /// `[t + idx*4]`, `t` a rip-relative address and `idx` bounded, and every entry the index can
    /// select lies in the function's bytes, none of them patched by a relocation.
    fn table_entry(state: &State, m: &Mem, ctx: Ctx) -> Fact {
        let fact = |r: u8| state.regs[r as usize & 15];
        let (Some(b), Some(i)) = (m.base, m.index) else {
            return Fact::Unknown;
        };
        let (Fact::RipRel(at), Some(max)) = (fact(b), max_value(fact(i))) else {
            return Fact::Unknown;
        };
        if m.scale != 4 || m.disp != 0 {
            return Fact::Unknown;
        }
        let entries = max.checked_add(1).and_then(|n| u32::try_from(n).ok());
        let (Some(n), Ok(at)) = (entries, u32::try_from(at)) else {
            return Fact::Unknown;
        };
        let (start, len) = (at as usize, 4 * n as usize);
        let inside = start
            .checked_add(len)
            .is_some_and(|end| end <= ctx.bytes.len());
        if inside && !ctx.patched(start, len) {
            Fact::TableEntry { at, n }
        } else {
            Fact::Unknown
        }
    }

    /// The largest value a register holding `f` can have, when `f` bounds it.
    fn max_value(f: Fact) -> Option<u64> {
        match f {
            Fact::Confined { bound, guard } => Some(bound.saturating_sub(1).max(guard)),
            Fact::Bounded(ub) => ub.checked_sub(1),
            Fact::Const(c) => u64::try_from(c).ok(),
            _ => None,
        }
    }

    /// Validate one emitted function's `bytes` against its relocations and what is known of its
    /// entry, returning every policy finding. Pure, so tests assert on the findings
    /// directly. Nothing Cranelift says about the code is an input (#2603 D3).
    ///
    /// Three layers (#2481 slice 1 + #2535):
    /// 1. **Decode + allowlist**, per machine basic block, flagging any instruction outside the
    ///    allowlist, any undecodable encoding, any block overrun, any direct branch out of code, and
    ///    control falling off the end. The bytes between a non-fall-through terminator and the next
    ///    block are a constant island or jump table Cranelift placed there (data, not code), so the
    ///    walk stops at the terminator.
    /// 2. **CFG recovery** ([`recover_leaders`]): the machine basic blocks are recovered by decoding
    ///    forward from the entry, every direct-branch target, and every jump-table target, which
    ///    [`table_targets`] reads from the table the bytes compute the `jmp`'s target from.
    /// 3. **Memory-access confinement** (#2535): an abstract interpretation over the recovered CFG
    ///    tracks which register holds `mem_base` and which holds a D63-confined index across blocks
    ///    and loops, then checks every memory access goes to a known region (the window through a
    ///    confined/bounded index, the stack, vmctx, the fn-table, or rip-relative rodata).
    ///
    /// Layer 3 is split so that only a check is trusted (#2603): [`propose`] runs the fixpoint and
    /// guesses a state per block — a heuristic, free to widen or give up — and [`verify`] checks the
    /// guess in one linear pass. A wrong or unconverged guess fails the check; it never admits code.
    /// It needs the guest entry contract, so a trampoline ([`Entry::Host`]) gets layers 1–2 only.
    pub(crate) fn check(bytes: &[u8], relocs: &[Reloc], entry: Entry) -> Vec<Finding> {
        let mut findings = Vec::new();
        if bytes.is_empty() {
            return findings;
        }
        let layout = match entry {
            Entry::Guest(layout) => layout,
            Entry::Host => None,
        };
        let ctx = Ctx {
            bytes,
            relocs,
            layout,
        };
        let cfg = Cfg::build(ctx, &mut findings);
        if let Entry::Host = entry {
            return findings;
        }
        let states = propose(&cfg, ctx);
        verify(&cfg, &states, ctx, &mut findings);
        findings
    }

    /// The untrusted half of layer 3: forward dataflow toward a fixpoint, proposing an in-state for
    /// every block the entry reaches. `in[b]` is the meet of its computed predecessors' out-states
    /// (the entry also meets the contract); `out[b]` is the block's transfer of `in[b]`. Loop headers
    /// widen `Bounded`, so this terminates; in practice the loop-carried facts are invariant and it
    /// settles in one or two rounds. The round cap only bounds the cost: [`verify`] rejects a
    /// proposal the cap cut short.
    fn propose(cfg: &Cfg, ctx: Ctx) -> Vec<Option<State>> {
        let nb = cfg.leaders.len();
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); nb];
        for (fi, ss) in cfg.succs.iter().enumerate() {
            for &ti in ss {
                preds[ti].push(fi);
            }
        }
        // A loop header is the target of a back edge. Blocks are sorted by offset and Cranelift lays
        // a loop out so its back edge comes from a higher offset, so `pred index >= own index`
        // identifies headers (a rare forward edge from a later block only over-widens — still sound).
        let is_header: Vec<bool> = (0..nb)
            .map(|ti| preds[ti].iter().any(|&fi| fi >= ti))
            .collect();
        let mut in_st: Vec<Option<State>> = vec![None; nb];
        let mut out_st: Vec<Option<State>> = vec![None; nb];
        for _ in 0..nb * 4 + 8 {
            let mut changed = false;
            for i in 0..nb {
                let mut cur = (i == 0).then(State::entry);
                for &p in &preds[i] {
                    if let Some(po) = out_st[p] {
                        cur = Some(cur.map_or(po, |s| s.meet(&po)));
                    }
                }
                if let Some(mut s) = cur {
                    if is_header[i] {
                        s.widen();
                    }
                    if in_st[i] != Some(s) {
                        in_st[i] = Some(s);
                        changed = true;
                    }
                    let o = transfer_block(&cfg.blocks[i], s, None, ctx);
                    if out_st[i] != Some(o) {
                        out_st[i] = Some(o);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        in_st
    }

    /// The trusted half of layer 3: one linear pass over the proposed `states`. It runs each
    /// proposed block's transfer once, applying the policy at every instruction, and requires the
    /// proposal to be **inductive**: the entry contract implies the entry block's state, and every
    /// edge's out-state implies its target's state. By induction every reachable block is then
    /// covered by a state that over-approximates every execution reaching it, so the policy findings
    /// hold for every run. A block whose successor has no state, or an edge the proposal doesn't
    /// hold on, is a finding.
    fn verify(cfg: &Cfg, states: &[Option<State>], ctx: Ctx, findings: &mut Vec<Finding>) {
        if !states
            .first()
            .copied()
            .flatten()
            .is_some_and(|s| implies(&State::entry(), &s))
        {
            findings.push(Finding {
                at: 0,
                why: "the entry block's state is not implied by the entry contract".into(),
            });
            return;
        }
        for (i, insns) in cfg.blocks.iter().enumerate() {
            let Some(s) = states[i] else { continue };
            let out = transfer_block(insns, s, Some(findings), ctx);
            for &j in &cfg.succs[i] {
                if !states[j].is_some_and(|sj| implies(&out, &sj)) {
                    findings.push(Finding {
                        at: cfg.leaders[j],
                        why: format!(
                            "the state proposed for this block does not hold on the edge from \
                             +0x{:x}",
                            cfg.leaders[i]
                        ),
                    });
                }
            }
        }
    }

    /// Recover the machine basic-block leaders: the entry, every direct-branch target, the
    /// fall-through of every conditional branch, and every jump-table target. Decoding from the
    /// current leader set discovers more targets; iterate to a fixpoint (the leader set only grows,
    /// bounded by the code size, so it terminates). The walk stops at a non-fall-through terminator,
    /// so inter-block island/table data is never decoded.
    fn recover_leaders(ctx: Ctx) -> Vec<usize> {
        let bytes = ctx.bytes;
        let len = bytes.len();
        let mut leaders: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        leaders.insert(0);
        loop {
            let before = leaders.len();
            let cur: Vec<usize> = leaders.iter().copied().collect();
            for (i, &ld) in cur.iter().enumerate() {
                let limit = cur.get(i + 1).copied().unwrap_or(len);
                let mut pos = ld;
                let mut run = Vec::new();
                while pos < limit {
                    let d = decode(bytes, pos);
                    if d.len == 0 {
                        break; // undecodable (flagged in decode_block)
                    }
                    let next = pos + d.len;
                    run.push((pos, d));
                    match d.kind {
                        Kind::DirectJmp { target } => {
                            if target < len && !tail_call(&d, pos, ctx) {
                                leaders.insert(target);
                            }
                            break;
                        }
                        Kind::DirectJcc { target } => {
                            if target < len {
                                leaders.insert(target);
                            }
                            if next < len {
                                leaders.insert(next); // the fall-through is its own block
                            }
                            break;
                        }
                        // A table this run computes adds its targets; one that doesn't is flagged in
                        // `Cfg::build`, from the block the leaders end up delimiting.
                        Kind::IndirectJmp => {
                            leaders.extend(table_targets(&run, ctx).unwrap_or_default());
                            break;
                        }
                        Kind::Ret | Kind::Ud2 => break,
                        _ => {}
                    }
                    if next > limit {
                        break; // overrun (flagged in decode_block)
                    }
                    pos = next;
                }
            }
            if leaders.len() == before {
                break;
            }
        }
        leaders.into_iter().collect()
    }

    /// Decode block `[block, limit)` into `(offset, Decoded)` pairs (layer 1), stopping at the
    /// block's non-fall-through terminator, and return the block's direct successor offsets (the
    /// branch target(s) and, for a conditional branch or a fall-through into the next block, the
    /// next offset). Pushes layer-1 findings: undecodable encodings, forbidden instructions, block
    /// overruns, direct branches out of the code, and control falling off the end of the function.
    fn decode_block(
        ctx: Ctx,
        block: usize,
        limit: usize,
        findings: &mut Vec<Finding>,
    ) -> (Vec<(usize, Decoded)>, Vec<usize>) {
        let bytes = ctx.bytes;
        let off_the_end = |at: usize| Finding {
            at,
            why: "control falls off the end of the function".into(),
        };
        let mut insns = Vec::new();
        let mut pos = block;
        while pos < limit {
            let d = decode(bytes, pos);
            if d.len == 0 {
                findings.push(Finding {
                    at: pos,
                    why: "undecodable instruction (decoder does not model this encoding)".into(),
                });
                return (insns, Vec::new());
            }
            match d.kind {
                Kind::Forbidden(name) => findings.push(Finding {
                    at: pos,
                    why: format!("forbidden instruction: {name}"),
                }),
                Kind::DirectJmp { target } | Kind::DirectJcc { target }
                    if target >= bytes.len() && !tail_call(&d, pos, ctx) =>
                {
                    findings.push(Finding {
                        at: pos,
                        why: format!("direct branch to +0x{target:x} is outside the code"),
                    });
                }
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
                return (insns, Vec::new());
            }
            insns.push((pos, d));
            match d.kind {
                Kind::DirectJmp { .. } if tail_call(&d, pos, ctx) => return (insns, Vec::new()),
                Kind::DirectJmp { target } => return (insns, vec![target]),
                Kind::DirectJcc { target } => {
                    if next >= bytes.len() {
                        findings.push(off_the_end(pos));
                    }
                    return (insns, vec![target, next]);
                }
                // `ret`/`ud2` leave the function; an indirect `jmp`'s successors are its jump table's.
                Kind::Ret | Kind::Ud2 | Kind::IndirectJmp => return (insns, Vec::new()),
                _ => {}
            }
            pos = next;
        }
        if limit >= bytes.len() {
            findings.push(off_the_end(insns.last().map_or(block, |&(p, _)| p)));
        }
        (insns, vec![limit]) // fell through to the next leader
    }

    /// A block-local symbolic value: what a register holds, as the value some instruction produced
    /// (`id`, unique within the block) plus a constant `off`. A 64-bit copy keeps it, a constant
    /// `lea`/`add`/`sub` moves `off`, and any other write makes a fresh value — so two registers with
    /// the same `id` hold values a known constant apart (#2603).
    #[derive(Clone, Copy, PartialEq, Eq)]
    struct Sym {
        id: u32,
        off: i64,
    }

    /// The flags a `cmp lhs, rhs` of width `w` leaves, for a `cmov` that tests them: each side as the
    /// value it compared (a register operand's [`Sym`]) and its constant (`w` bytes wide), when known.
    /// Block-local: a block starts with no compare known, and any instruction that writes the flags —
    /// or a call — forgets it (#2603 hole 3).
    #[derive(Clone, Copy)]
    struct Cmp {
        w: u8,
        lhs: Side,
        rhs: Side,
    }

    /// One side of a [`Cmp`].
    #[derive(Clone, Copy)]
    struct Side {
        sym: Option<Sym>,
        konst: Option<u64>,
    }

    /// The fact `cmovcc dst, src` (width `w`, condition `cc`) leaves in `dst`, given the compare whose
    /// flags it tests (#2603 holes 2–4). The result is `dst`'s old value or `src`'s. It is `Confined`
    /// when one of the two is a constant (the guard) and the other is the compared value `v` plus a
    /// constant `c ≥ 0` — D63 compares an address and selects the address plus the access's offset —
    /// and `cc` lets it through only while `v` is below the compare's constant side `k`: `v < k` keeps
    /// it under `k + c`, `v ≤ k` under `k + c + 1`. With `cmp v, k` those conditions are `b` and `be`;
    /// with `cmp k, v`, `a` and `ae`; a `cmov` that keeps its old value tests their negations (the
    /// low bit flipped). A 32-bit `cmov` zero-extends what it selects, so the same holds of the 64-bit
    /// result while `k + c` fits in 32 bits. Anything else — another condition, an unknown bound, a
    /// side that isn't the compared value — is `Unknown`.
    fn cmov_fact(
        flags: Option<Cmp>,
        w: u8,
        cc: u8,
        (dst, old): (Sym, Fact),
        (src, src_fact): (Option<Sym>, Fact),
    ) -> Fact {
        let Some(f) = flags.filter(|f| f.w == w) else {
            return Fact::Unknown;
        };
        let mask = if w == 8 {
            u64::MAX
        } else {
            u64::from(u32::MAX)
        };
        let konst = |x: Fact| match x {
            Fact::Const(c) => Some(c as u64 & mask),
            _ => None,
        };
        // How far `s` sits above the compared value `v`: `Some(c)` when it is `v + c`, `c ≥ 0`.
        let above = |s: Sym, v: Sym| {
            (s.id == v.id)
                .then(|| s.off.checked_sub(v.off))
                .flatten()
                .and_then(|c| u64::try_from(c).ok())
        };
        // The bound of `[0, k + c)` (`[0, k + c]` when `incl`), if it fits in `w` bits.
        let limit = |k: u64, c: u64, incl: bool| {
            let b = k.checked_add(c)?.checked_add(u64::from(incl))?;
            (w == 8 || b <= 1 << 32).then_some(b)
        };
        // (the compared side, the constant side, the conditions meaning `v < k` and `v ≤ k`)
        for (var, bound, lt, le) in [(f.lhs, f.rhs, 0x2u8, 0x6u8), (f.rhs, f.lhs, 0x7, 0x3)] {
            let (Some(v), Some(k)) = (var.sym, bound.konst) else {
                continue;
            };
            // Takes `src = v + c` while the condition holds, else keeps the constant old value.
            let taken = [(lt, false), (le, true)];
            if let (Some(c), Some(guard)) = (src.and_then(|s| above(s, v)), konst(old)) {
                let incl = taken.iter().find(|&&(x, _)| x == cc).map(|&(_, i)| i);
                if let Some(bound) = incl.and_then(|i| limit(k, c, i)) {
                    return Fact::Confined { bound, guard };
                }
            }
            // Keeps `dst = v + c` unless the negated condition holds, then takes the constant `src`.
            if let (Some(c), Some(guard)) = (above(dst, v), konst(src_fact)) {
                let incl = taken.iter().find(|&&(x, _)| x ^ 1 == cc).map(|&(_, i)| i);
                if let Some(bound) = incl.and_then(|i| limit(k, c, i)) {
                    return Fact::Confined { bound, guard };
                }
            }
        }
        Fact::Unknown
    }

    /// Run block `insns` from `state`, returning the out-state. With `findings = Some`, also apply
    /// the memory-access policy (layer 3) at each access. A compare's flags and the symbolic values
    /// are block-local, but the `Confined` fact a `cmov` derives from them flows across blocks through
    /// the lattice, so a guard Cranelift hoists out of a loop is still seen at the access.
    fn transfer_block(
        insns: &[(usize, Decoded)],
        mut state: State,
        mut findings: Option<&mut Vec<Finding>>,
        ctx: Ctx,
    ) -> State {
        let mut sym: [Sym; 16] = std::array::from_fn(|r| Sym {
            id: r as u32,
            off: 0,
        });
        let mut next_id = 16u32;
        let mut flags: Option<Cmp> = None;
        for &(pos, d) in insns {
            if let Some(f) = findings.as_deref_mut() {
                mem_policy(pos, &d, &state, ctx.layout, f);
            }
            let end = pos + d.len;
            // An immediate the linker patches is not the value that runs.
            let imm_ok = !ctx.patched(end - usize::from(d.imm_len), usize::from(d.imm_len));
            let written = writes_of(&d);
            // The registers the semantic below sets precisely; every other register the instruction
            // writes becomes `Unknown`.
            let mut defined = 0u16;
            slot_effect(&mut state, &d);
            let fact = |state: &State, r: u8| state.regs[r as usize & 15];
            let sym_of = |r: u8| sym[r as usize & 15];
            let mut set = |state: &mut State, r: u8, f: Fact| {
                state.regs[r as usize & 15] = f;
                defined |= bit(r);
            };
            match d.eff {
                Eff::Cmp { w, lhs, rhs } => {
                    let mask = if w == 8 {
                        u64::MAX
                    } else {
                        u64::from(u32::MAX)
                    };
                    let side = |o: Opnd| match o {
                        Opnd::Reg(r) => Side {
                            sym: Some(sym_of(r)),
                            konst: match fact(&state, r) {
                                Fact::Const(c) => Some(c as u64 & mask),
                                _ => None,
                            },
                        },
                        Opnd::Imm(v) => Side {
                            sym: None,
                            konst: imm_ok.then_some(v & mask),
                        },
                        Opnd::Mem => Side {
                            sym: None,
                            konst: ctx
                                .rip_target(&d, end)
                                .and_then(|t| ctx.constant(t, usize::from(w))),
                        },
                    };
                    flags = Some(Cmp {
                        w,
                        lhs: side(lhs),
                        rhs: side(rhs),
                    });
                }
                Eff::MovReg { dst, src } => {
                    let f = fact(&state, src);
                    set(&mut state, dst, f);
                }
                Eff::MovImm { dst, imm } => {
                    let f = match ctx.abs_reloc(pos, &d) {
                        Some(r) => r.callee.map_or(Fact::Unknown, Fact::FnAddr),
                        None if imm_ok => Fact::Const(imm),
                        None => Fact::Unknown,
                    };
                    set(&mut state, dst, f);
                }
                Eff::AluImm { dst, alu, imm } => {
                    let f = if imm_ok {
                        alu_fact(fact(&state, dst), alu, imm)
                    } else {
                        Fact::Unknown
                    };
                    set(&mut state, dst, f);
                }
                Eff::Cmov { dst, cc, w } => {
                    let src = d.operands.rm_reg;
                    let f = cmov_fact(
                        flags,
                        w,
                        cc,
                        (sym_of(dst), fact(&state, dst)),
                        (
                            src.map(sym_of),
                            src.map_or(Fact::Unknown, |s| fact(&state, s)),
                        ),
                    );
                    set(&mut state, dst, f);
                }
                Eff::Lea { dst, full: false } => set(&mut state, dst, Fact::Unknown),
                Eff::Lea { dst, full: true } => {
                    let f = match d.operands.mem {
                        Some(m) if m.rip => match ctx.rip_target(&d, end) {
                            Some(t) => Fact::RipRel(t),
                            None => Fact::Unknown, // a symbol address, not the function's rodata
                        },
                        Some(m) => match frame_addr(&state, &m) {
                            Some(Some(off)) => Fact::Stack(Some(off)),
                            Some(None) if m.index.is_none() => Fact::Stack(None),
                            _ => Fact::Unknown,
                        },
                        None => Fact::Unknown,
                    };
                    set(&mut state, dst, f);
                }
                Eff::LoadReg { dst } => {
                    // A full-width reload from a tracked slot restores its fact.
                    let f = match d.operands.mem.and_then(|m| frame_addr(&state, &m)) {
                        Some(Some(off)) => state.slot_get(off),
                        _ => Fact::Unknown,
                    };
                    set(&mut state, dst, f);
                }
                Eff::LoadSx32 { dst } => {
                    let f = d
                        .operands
                        .mem
                        .map_or(Fact::Unknown, |m| table_entry(&state, &m, ctx));
                    set(&mut state, dst, f);
                }
                Eff::AddReg { dst, src } => {
                    let f = match (fact(&state, dst), fact(&state, src)) {
                        (Fact::RipRel(t), Fact::TableEntry { at, n })
                        | (Fact::TableEntry { at, n }, Fact::RipRel(t))
                            if t == i64::from(at) =>
                        {
                            Fact::TableTarget { at, n }
                        }
                        _ => Fact::Unknown,
                    };
                    set(&mut state, dst, f);
                }
                Eff::Push { src } => {
                    // `[rsp − 8] := src; rsp −= 8` — the pushed fact is read before rsp moves.
                    let f = src.map_or(Fact::Unknown, |s| fact(&state, s));
                    let rsp = match fact(&state, reg::RSP) {
                        Fact::Stack(Some(o)) => {
                            state.clobber(o - 8, o);
                            state.slot_set(o - 8, f);
                            Fact::Stack(Some(o - 8))
                        }
                        Fact::Stack(None) => {
                            state.clear_slots();
                            Fact::Stack(None)
                        }
                        _ => Fact::Unknown,
                    };
                    set(&mut state, reg::RSP, rsp);
                }
                Eff::Pop { dst } => {
                    // `dst := [rsp]; rsp += 8` (a `pop rsp` keeps the loaded value).
                    let (f, rsp) = match fact(&state, reg::RSP) {
                        Fact::Stack(Some(o)) => (state.slot_get(o), Fact::Stack(Some(o + 8))),
                        Fact::Stack(None) => (Fact::Unknown, Fact::Stack(None)),
                        _ => (Fact::Unknown, Fact::Unknown),
                    };
                    set(&mut state, reg::RSP, rsp);
                    match dst {
                        Some(r) => set(&mut state, r, f),
                        None => state.clear_slots(), // `pop [mem]`: a frame store, address after the pop
                    }
                }
                Eff::StoreReg { .. } | Eff::WrReg | Eff::WrRm | Eff::Wr(_) | Eff::NoGp => {}
            }
            if matches!(d.kind, Kind::DirectCall | Kind::IndirectCall) {
                // A call runs its callee's contract: the registers it clobbers become unknown (below),
                // the rest keep their facts, the flags are forgotten, and rsp moves by what the
                // callee pops. An unidentified callee is a finding, and is taken to clobber the SysV
                // set and pop an unknown amount. No slot survives a call until #2603 PR 3 checks that
                // every callee writes only its own frame.
                let callee = match d.kind {
                    Kind::DirectCall => ctx.call_reloc(pos, &d).and_then(|r| r.callee),
                    _ => indirect_callee(&state, &d),
                };
                if callee.is_none() && d.kind == Kind::IndirectCall {
                    if let Some(f) = findings.as_deref_mut() {
                        f.push(Finding {
                            at: pos,
                            why: "indirect call to an unidentified callee".into(),
                        });
                    }
                }
                let c = callee.unwrap_or(Contract {
                    clobbers: SYSV_CLOBBERS,
                    pops: None,
                });
                state.clear_slots();
                let rsp = match (fact(&state, reg::RSP), c.pops) {
                    (Fact::Stack(Some(h)), Some(p)) => Fact::Stack(h.checked_add(i64::from(p))),
                    (Fact::Stack(_), _) => Fact::Stack(None),
                    _ => Fact::Unknown,
                };
                set(&mut state, reg::RSP, rsp);
                defined |= !c.clobbers;
            }
            for r in 0..16u8 {
                if written & !defined & bit(r) != 0 {
                    state.regs[r as usize] = Fact::Unknown;
                }
            }
            // The symbolic values: a 64-bit copy keeps the source's, a constant offset moves it, and
            // any other write makes a fresh one.
            let offset = |r: u8, by: Option<i64>| {
                let s = sym_of(r);
                by.and_then(|b| s.off.checked_add(b))
                    .map(|off| Sym { id: s.id, off })
            };
            let kept = match d.eff {
                Eff::MovReg { dst, src } => Some((dst, Some(sym_of(src)))),
                Eff::Lea { dst, full: true } => match d.operands.mem {
                    Some(Mem {
                        base: Some(b),
                        index: None,
                        disp,
                        rip: false,
                        ..
                    }) => Some((dst, offset(b, Some(disp)))),
                    _ => None,
                },
                Eff::AluImm { dst, alu, imm } if imm_ok => match alu {
                    Alu::Add => Some((dst, offset(dst, Some(imm)))),
                    Alu::Sub => Some((dst, offset(dst, imm.checked_neg()))),
                    Alu::And => None,
                },
                _ => None,
            };
            for r in 0..16u8 {
                if written & bit(r) != 0 {
                    sym[r as usize] = Sym {
                        id: next_id,
                        off: 0,
                    };
                    next_id += 1;
                }
            }
            if let Some((dst, Some(s))) = kept {
                sym[dst as usize & 15] = s;
            }
            if !matches!(d.eff, Eff::Cmp { .. }) && !d.keeps_flags {
                flags = None;
            }
        }
        state
    }

    /// The contract of what an indirect `call` reaches: a callee whose address a relocation names
    /// (`movabs r, sym; call r`). Anything else is unidentified.
    fn indirect_callee(state: &State, d: &Decoded) -> Option<Contract> {
        if d.operands.mem.is_some() {
            return None;
        }
        match state.regs[d.operands.rm_reg? as usize & 15] {
            Fact::FnAddr(c) => Some(c),
            _ => None,
        }
    }

    /// Where an explicit memory operand points in the stack frame: `Some(Some(off))` for a plain
    /// `[s + disp]` through a stack pointer `s` at a known entry-relative offset, `Some(None)` for any
    /// other access the frame might hold (a stack pointer at an unknown offset, or indexed by one),
    /// `None` for an access that cannot reach the stack.
    fn frame_addr(state: &State, m: &Mem) -> Option<Option<i64>> {
        if m.rip {
            return None;
        }
        let fact = |r: Option<u8>| r.map(|x| state.regs[x as usize & 15]);
        match (fact(m.base), fact(m.index)) {
            (Some(Fact::Stack(Some(off))), None) => Some(off.checked_add(m.disp)),
            (Some(Fact::Stack(_)), _) | (_, Some(Fact::Stack(_))) => Some(None),
            _ => None,
        }
    }

    /// Whether an instruction writes its explicit memory operand. Only the forms known to just read
    /// it say no; every other form — each SSE/AVX one among them — is taken to write it, which can
    /// only forget slots.
    fn writes_memory(d: &Decoded) -> bool {
        !matches!(
            d.eff,
            Eff::LoadReg { .. }
                | Eff::LoadSx32 { .. }
                | Eff::WrReg
                | Eff::Cmp { .. }
                | Eff::Cmov { .. }
                | Eff::Lea { .. }
                | Eff::Push { .. }
        ) && !matches!(d.kind, Kind::IndirectCall | Kind::IndirectJmp)
    }

    /// An explicit memory operand's effect on the tracked slots (#2603 holes 5–6): a 64-bit store to
    /// a known frame offset records the stored register's fact there, after forgetting every slot it
    /// overlaps; any other write to the frame only forgets — the slots it may overlap, or all of them
    /// when its offset is unknown. An access that cannot reach the stack leaves the slots alone (one
    /// through a wild base is itself a policy finding).
    fn slot_effect(state: &mut State, d: &Decoded) {
        let Some(m) = d.operands.mem else { return };
        if !writes_memory(d) {
            return;
        }
        let Some(at) = frame_addr(state, &m) else {
            return;
        };
        let Some(off) = at else {
            state.clear_slots();
            return;
        };
        let (w, fact) = match d.eff {
            Eff::StoreReg { src, w } => (
                i64::from(w),
                if w == 8 {
                    state.regs[src as usize & 15]
                } else {
                    Fact::Unknown
                },
            ),
            _ => (MAX_ACCESS_WIDTH as i64, Fact::Unknown),
        };
        state.clobber(off, off.saturating_add(w));
        state.slot_set(off, fact);
    }

    /// The GP registers an instruction writes: its effect's destinations plus a call's clobbers and
    /// the `rsp` that `push`/`pop`/`call`/`ret` move. A sound superset of what it actually writes
    /// (clearing extra facts is safe; missing a real write is not) — cross-checked against iced-x86 in
    /// the tests.
    fn writes_of(d: &Decoded) -> u16 {
        let eff = match d.eff {
            Eff::NoGp | Eff::Cmp { .. } | Eff::StoreReg { .. } => 0,
            Eff::MovImm { dst, .. }
            | Eff::MovReg { dst, .. }
            | Eff::LoadReg { dst }
            | Eff::LoadSx32 { dst }
            | Eff::AddReg { dst, .. }
            | Eff::Lea { dst, .. }
            | Eff::AluImm { dst, .. }
            | Eff::Cmov { dst, .. } => bit(dst),
            Eff::Push { .. } => bit(reg::RSP),
            Eff::Pop { dst } => bit(reg::RSP) | dst.map_or(0, bit),
            Eff::WrReg => d.operands.reg.map_or(0, bit),
            Eff::WrRm => d.operands.rm_reg.map_or(0, bit),
            Eff::Wr(m) => m,
        };
        let implicit = match d.kind {
            Kind::DirectCall | Kind::IndirectCall => SYSV_CLOBBERS | bit(reg::RSP),
            Kind::Ret => bit(reg::RSP),
            _ => 0,
        };
        eff | implicit
    }

    /// A 64-bit `and`/`add`/`sub r, imm` on a tracked value. `and` with a low-bits mask bounds the
    /// result; `add`/`sub` move a constant or a stack pointer (the prologue's `sub rsp, frame`).
    /// Anything else stops being tracked.
    fn alu_fact(prev: Fact, alu: Alu, imm: i64) -> Fact {
        match (alu, prev) {
            (Alu::And, _) if imm >= 0 && (imm as u64).wrapping_add(1).is_power_of_two() => {
                Fact::Bounded(imm as u64 + 1) // r & (2^k - 1) ∈ [0, 2^k)
            }
            (Alu::Add, Fact::Const(c)) => Fact::Const(c.wrapping_add(imm)),
            (Alu::Sub, Fact::Const(c)) => Fact::Const(c.wrapping_sub(imm)),
            (Alu::Add, Fact::Stack(off)) => Fact::Stack(off.and_then(|o| o.checked_add(imm))),
            (Alu::Sub, Fact::Stack(off)) => Fact::Stack(off.and_then(|o| o.checked_sub(imm))),
            _ => Fact::Unknown,
        }
    }

    /// The memory-access confinement policy (#2535/#2603): classify one access's base and index by
    /// their abstract facts and flag an access that does not land in a known region. For a window
    /// access, re-prove the physical offset from `mem_base` stays within the reservation
    /// `[0, guard_offset + page)` — everything there is mapped or `PROT_NONE` guard, so it can fault
    /// but never escape. The stack's, vmctx's, the fn-table's and rodata's extents are #2603 PR 3.
    fn mem_policy(
        pos: usize,
        d: &Decoded,
        state: &State,
        layout: Option<WindowLayout>,
        findings: &mut Vec<Finding>,
    ) {
        let mut flag = |why: String| findings.push(Finding { at: pos, why });
        // `push`/`pop`/`call`/`ret` read or write the stack at `rsp` implicitly.
        let implicit_stack = matches!(d.eff, Eff::Push { .. } | Eff::Pop { .. })
            || matches!(d.kind, Kind::DirectCall | Kind::IndirectCall | Kind::Ret);
        if implicit_stack && !matches!(state.regs[reg::RSP as usize], Fact::Stack(_)) {
            flag("stack access through an rsp that is not a stack pointer".into());
        }
        let Some(mem) = d.operands.mem else { return };
        // `lea` computes an address but does not dereference it.
        if matches!(d.eff, Eff::Lea { .. }) {
            return;
        }
        // rip-relative: the function's own constant pool / jump tables (rodata).
        if mem.rip {
            return;
        }
        let fact = |r: Option<u8>| r.map(|x| state.regs[x as usize & 15]);
        let (base_f, index_f) = (fact(mem.base), fact(mem.index));

        // Stack: based on a stack pointer.
        if matches!(base_f, Some(Fact::Stack(_))) {
            return;
        }
        // Window: base (or index) is `mem_base`; the other operand is the index/displacement.
        let is_membase = |f: Option<Fact>| matches!(f, Some(Fact::MemBase));
        if is_membase(base_f) || is_membase(index_f) {
            // A module with no window has no window access to re-prove: one is a violation, never
            // something to take the code's own word for.
            let Some(l) = layout else {
                flag("window access in a module with no window".into());
                return;
            };
            let offset_f = if is_membase(base_f) { index_f } else { base_f };
            let page = crate::mem::trailing_guard() as u64;
            let scale = u64::from(mem.scale);

            // A negative displacement on a window access could underflow below `mem_base` — the
            // lowering never emits one (guest offsets are non-negative), so treat it as a violation.
            if mem.disp < 0 {
                flag(format!(
                    "window access with a negative displacement -0x{:x}",
                    mem.disp.unsigned_abs()
                ));
                return;
            }
            let disp = mem.disp as u64;

            // The largest byte offset the index can contribute, and (for a confined index) the guard
            // the D63 `cmov` redirected to. `None` index_max ⇒ not provably bounded.
            let (index_max, confined_guard): (Option<u64>, Option<u64>) = match offset_f {
                None => (Some(0), None), // [mem_base + disp]
                Some(Fact::Confined { bound, guard }) => {
                    (Some(bound.saturating_sub(1).max(guard)), Some(guard))
                }
                Some(Fact::Bounded(ub)) => (Some(ub.saturating_sub(1).saturating_mul(scale)), None),
                Some(Fact::Const(c)) if c >= 0 => (Some((c as u64).saturating_mul(scale)), None),
                Some(_) => (None, None),
            };
            let Some(imax) = index_max else {
                flag(
                    "window access [mem_base + <unconfined index>]: the index is not confined by \
                      the D63 sequence nor proven bounded"
                        .into(),
                );
                return;
            };
            // A confined index is a byte offset (scale 1), and the guard it redirects to must be the
            // module's real guard offset (#2603).
            if let Some(guard) = confined_guard {
                if mem.scale != 1 {
                    flag(format!(
                        "confined window index with scale {} (expected a byte index, scale 1)",
                        mem.scale
                    ));
                    return;
                }
                if guard != l.guard_offset {
                    flag(format!(
                        "confined index redirects to +0x{guard:x}, not the module guard offset \
                         +0x{:x}",
                        l.guard_offset
                    ));
                    return;
                }
            }
            // Re-prove: the largest physical offset the access can reach stays within the window
            // reservation, whose top is `guard_offset + page`.
            let reach = l.guard_offset.saturating_add(page);
            let top = imax.saturating_add(disp).saturating_add(MAX_ACCESS_WIDTH);
            if top > reach {
                flag(format!(
                    "window access reaches +0x{top:x}, past the reservation +0x{reach:x}"
                ));
            }
            return;
        }
        // Other known bases: the fn-table (call.dyn dispatch), vmctx (the trap-out cell and
        // per-instance fields), and a lea'd pointer into the function's own rodata / jump tables (the
        // br_table dispatch's table read). Their extents are #2603 PR 3.
        if matches!(
            base_f,
            Some(Fact::FnTableBase | Fact::Vmctx | Fact::RipRel(_))
        ) {
            return;
        }
        // Anything else: an access through a base of unknown or non-pointer provenance.
        match mem.base {
            Some(b) => flag(format!(
                "memory access through r{b} of unrecognized provenance ({:?})",
                base_f.unwrap_or(Fact::Unknown)
            )),
            None => flag("absolute memory access (no base register)".into()),
        }
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

    /// A small-integer ALU op the transfer models for bound/const tracking (#2535). Everything else
    /// is an opaque GP write (the value becomes `Fact::Unknown`).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum Alu {
        Add,
        Sub,
        And,
    }

    /// The data-flow **effect** of an instruction, for the abstract interpreter (#2535): which GP
    /// register it writes and, for the handful of operations the window-access policy models, how.
    /// The decoder — the one place that parses opcodes (invariant #15) — sets this; the transfer
    /// derives the GP write-set from it (cross-checked against iced-x86 in the tests, so a missed
    /// write is caught). `dst`/`src`/`reg` are GP register numbers (0..15).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum Eff {
        /// Writes no GP register the state tracks: flags-only (`test`, a narrower `cmp`), control
        /// flow, SSE/vector ops (their operands are XMM, not GP), fences, nops.
        NoGp,
        /// A 32- or 64-bit `mov dst, imm` — `dst := Const(imm)`, `imm` the value the full register
        /// ends up holding (a 32-bit write zero-extends).
        MovImm { dst: u8, imm: i64 },
        /// A 64-bit `mov dst, src` between GP registers — `dst := fact(src)`.
        MovReg { dst: u8, src: u8 },
        /// A 64-bit GP load `mov dst, [mem]` (mem in `operands.mem`) — `dst := loaded value`, or the
        /// spill slot's fact when `mem` is a tracked frame slot. A narrower load is a `WrReg`.
        LoadReg { dst: u8 },
        /// A GP store `mov [mem], src` of `w` bytes (mem in `operands.mem`). A 64-bit store records
        /// `src`'s fact in the spill slot it writes. Writes no GP register.
        StoreReg { src: u8, w: u8 },
        /// `movsxd dst, dword [mem]` (REX.W) — a sign-extending 32-bit load, the read of a jump-table
        /// entry.
        LoadSx32 { dst: u8 },
        /// A 64-bit `add dst, src` between GP registers — the jump-table target `table + entry`.
        AddReg { dst: u8, src: u8 },
        /// `lea dst, [mem]` — computes an address into `dst` (does **not** dereference). Only a 64-bit
        /// one (`full`) yields a pointer; a narrower one truncates, so `dst` holds an unknown value.
        Lea { dst: u8, full: bool },
        /// A 64-bit `add`/`sub`/`and dst, imm`, `dst` the r/m register — `dst := alu(fact(dst), imm)`.
        AluImm { dst: u8, alu: Alu, imm: i64 },
        /// A 32- or 64-bit `cmp lhs, rhs` (it computes `lhs − rhs`). Writes only the flags.
        Cmp { w: u8, lhs: Opnd, rhs: Opnd },
        /// A 32- or 64-bit `cmovcc dst, src` (`src` in `operands`), `cc` its condition (`0F 40+cc`).
        /// Conditional, so always a write of `dst`.
        Cmov { dst: u8, cc: u8, w: u8 },
        /// `push src` — `[rsp − 8] := src; rsp −= 8` (`None` for an immediate or memory source).
        Push { src: Option<u8> },
        /// `pop dst` — `dst := [rsp]; rsp += 8` (`None` for `pop [mem]`).
        Pop { dst: Option<u8> },
        /// Writes `operands.reg` (a GP op whose destination is its reg field: `op reg, r/m`).
        WrReg,
        /// Writes `operands.rm_reg` when the r/m is a register (`op r/m, reg`); the memory form writes
        /// memory, no GP register.
        WrRm,
        /// Writes a fixed implicit GP mask (e.g. `mul`/`div` → rax,rdx; `cpuid`; accumulator forms).
        Wr(u16),
    }

    /// A `cmp` operand: a GP register, an immediate (sign-extended to 64 bits), or the memory
    /// operand in `operands.mem`.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum Opnd {
        Reg(u8),
        Imm(u64),
        Mem,
    }

    /// One decoded instruction: its length, its control-flow classification, its operands, its
    /// data-flow effect ([`Eff`], consumed by the #2535 abstract interpreter), the length of the
    /// immediate it ends with (a relocation may patch it), and whether it leaves the flags as they
    /// were. That is `false` unless its decode arm says so, so a `cmp`'s flags never outlive an
    /// instruction this decoder doesn't know to keep them (#2603).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct Decoded {
        pub len: usize,
        pub kind: Kind,
        pub operands: Operands,
        pub eff: Eff,
        pub imm_len: u8,
        pub keeps_flags: bool,
    }

    impl Decoded {
        /// Mark an instruction that leaves the flags unchanged (cross-checked against iced-x86).
        fn keep_flags(mut self) -> Decoded {
            self.keeps_flags = true;
            self
        }
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
        eff: Eff::NoGp,
        imm_len: 0,
        keeps_flags: false,
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

    impl Pfx {
        /// The operand size of a non-byte instruction: 8 under REX.W, else 2 under `0x66`, else 4.
        fn osz(&self) -> u8 {
            if self.rex_w {
                8
            } else if self.op66 {
                2
            } else {
                4
            }
        }
    }

    fn decode_inner(c: &mut Cursor, start: usize) -> Option<Decoded> {
        let mut p = Pfx::default();
        // fs/gs add a segment base and `0x67` truncates the address to 32 bits: either makes a memory
        // operand mean something other than `[base + index*scale + disp]`, the address the policy
        // checks, and Cranelift emits neither for guest code (#2603 hole 8).
        let mut addressing = None;
        // Legacy prefixes (any order/number in valid code; Cranelift emits at most a couple).
        loop {
            match c.peek()? {
                0x66 => p.op66 = true,
                0x64 | 0x65 => addressing = Some("fs/gs segment override"),
                0x67 => addressing = Some("address-size override"),
                // rep/repne (also SSE mandatory prefixes), lock, and the cs/ss/ds/es overrides (a
                // zero base in 64-bit mode): consumed; none changes the length or the address.
                0xF2 | 0xF3 | 0xF0 | 0x2E | 0x36 | 0x3E | 0x26 => {}
                _ => break,
            }
            c.pos += 1;
        }
        let d = match c.peek()? {
            // VEX / EVEX (AVX): their prefix encodes the opcode map + ModRM-implied operands.
            0xC5 => decode_vex2(c, start),
            0xC4 => decode_vex3(c, start),
            0x62 => decode_evex(c, start),
            _ => {
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
        }?;
        Some(match addressing {
            Some(why) => Decoded {
                kind: Kind::Forbidden(why),
                ..d
            },
            None => d,
        })
    }

    /// A memory operand `[base + index*scale + disp]`, or RIP-relative. Register numbers are the full
    /// 0..15 (REX-extended). `base`/`index` are `None` when absent. Consumed by the window-access
    /// policy ([`mem_policy`]) and cross-checked against iced-x86.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub(crate) struct Mem {
        pub base: Option<u8>,
        pub index: Option<u8>,
        pub scale: u8,
        pub disp: i64,
        pub rip: bool,
    }

    /// The operands a ModRM instruction carries, as the policy/dataflow needs them: the `reg`-field
    /// register, and the r/m operand — either a register or a memory reference.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub(crate) struct Operands {
        pub reg: Option<u8>,
        pub rm_reg: Option<u8>,
        pub mem: Option<Mem>,
    }

    /// How the decoder tells [`resolve_eff`] what an instruction does, before the operands and
    /// immediate are read. `resolve_eff` turns it into the final [`Eff`] using the parsed operands,
    /// the operand size (only a 64-bit operation preserves a pointer or bound fact — a narrower write
    /// truncates, zero-extends or merges, so it clobbers; `cmp`/`cmov` are modeled at 32 bits too),
    /// and the immediate.
    #[derive(Clone, Copy)]
    enum EffIntent {
        /// Writes no GP register (flags-only, control flow, SSE/vector, fences, nops).
        NoGp,
        /// Writes `operands.reg` with an unmodeled value (`op reg, r/m`).
        WrReg,
        /// Writes `operands.rm_reg` (a register r/m) with an unmodeled value; a memory r/m writes
        /// memory, no GP register.
        WrRm,
        /// Writes a fixed implicit GP mask (`mul`/`div` → rax,rdx; `pop r`/`xchg eax,r`; …).
        Wr(u16),
        /// `mov reg, r/m` — a GP-GP move (fact copy) or a load.
        MovToReg,
        /// `mov r/m, reg` — a GP-GP move or a store.
        MovToRm,
        /// `mov r/m, imm` — a const into the r/m register, or a store of an immediate.
        MovImmToRm,
        /// `lea reg, [mem]`.
        Lea,
        /// `cmp reg, r/m` (`3B`).
        CmpRegRm,
        /// `cmp r/m, reg` (`39`).
        CmpRmReg,
        /// `add r/m, reg` (`01`).
        AddRmReg,
        /// `add reg, r/m` (`03`).
        AddRegRm,
        /// `movsxd reg, r/m32` (`63`).
        Movsxd,
        /// `0x81/0x83 /digit` — the ModRM.reg field (`digit`) picks add/or/.../cmp with `imm`.
        AluImmGroup(u8),
        /// `0x80/0x82 /digit` — the byte forms, 8-bit whatever the prefixes say (REX.W doesn't widen
        /// a byte operation), so never a tracked value.
        ByteAluImm,
        /// `0xF6/0xF7 /digit` — test (writes flags), not/neg (writes r/m), or mul/div (writes rax,rdx).
        Group3(u8),
        /// `0xFF /digit` — inc/dec (writes r/m), push r/m, or call/jmp (no GP write of their own).
        Group5(u8),
        /// `cmovcc reg, r/m`, `cc` its condition.
        Cmov(u8),
        /// `xchg reg, r/m` — writes both.
        XchgRegRm,
        /// `pop r/m` (`8F /0`).
        PopRm,
    }

    /// A one-register GP write mask bit.
    fn bit(r: u8) -> u16 {
        1u16 << (r & 15)
    }

    /// Turn an [`EffIntent`] into the final [`Eff`] now that the operands, the operand size `osz`
    /// (8/4/2 bytes), and the immediate are known. The guiding rule for soundness: a fact (`MemBase`,
    /// `Confined`, …) is a property of a full 64-bit register value, so only a **64-bit** operation
    /// may carry one; a narrower write truncates, zero-extends or merges, and therefore *clobbers* the
    /// destination. Two exceptions, each exact: a 32-bit `mov r, imm` zero-extends to a known value,
    /// and a 32-bit `cmp`/`cmov` pair confines what its zero-extending `cmov` selects (#2603).
    fn resolve_eff(intent: EffIntent, ops: &Operands, osz: u8, immv: i64) -> Eff {
        use EffIntent as I;
        let (reg, rm, mem) = (ops.reg, ops.rm_reg, ops.mem.is_some());
        let rm_opnd = if mem {
            Some(Opnd::Mem)
        } else {
            rm.map(Opnd::Reg)
        };
        let modeled = matches!(osz, 4 | 8); // the widths a `cmp`/`cmov` is modeled at
        match intent {
            I::NoGp => Eff::NoGp,
            I::WrReg => Eff::WrReg,
            I::WrRm => Eff::WrRm,
            I::Wr(m) => Eff::Wr(m),
            // A narrower `lea` truncates the address, so it is no pointer (#2603 hole 7).
            I::Lea => reg.map_or(Eff::NoGp, |dst| Eff::Lea {
                dst,
                full: osz == 8,
            }),
            // Only a full-width load can restore a tracked fact (#2603 hole 5).
            I::MovToReg => match (reg, rm, mem) {
                (Some(dst), _, true) if osz == 8 => Eff::LoadReg { dst },
                (Some(dst), Some(src), false) if osz == 8 => Eff::MovReg { dst, src },
                (Some(_), ..) => Eff::WrReg, // narrower: truncates or zero-extends, so clobber
                _ => Eff::NoGp,
            },
            I::MovToRm => match (reg, rm, mem) {
                (Some(src), _, true) => Eff::StoreReg { src, w: osz },
                (Some(src), Some(dst), false) if osz == 8 => Eff::MovReg { dst, src },
                (_, Some(_), false) => Eff::WrRm,
                _ => Eff::NoGp,
            },
            // A 16-bit write keeps the upper 48 bits, so its result is not known (#2603 hole 7).
            I::MovImmToRm => match (rm, mem) {
                (_, true) => Eff::NoGp, // store of an immediate
                (Some(dst), false) if osz == 8 => Eff::MovImm { dst, imm: immv },
                (Some(dst), false) if osz == 4 => Eff::MovImm {
                    dst,
                    imm: immv & 0xFFFF_FFFF,
                },
                (Some(_), false) => Eff::WrRm,
                _ => Eff::NoGp,
            },
            I::AddRmReg => match (rm, reg) {
                (Some(dst), Some(src)) if osz == 8 => Eff::AddReg { dst, src },
                _ => Eff::WrRm, // narrower, or `add [mem], reg`
            },
            I::AddRegRm => match (reg, rm) {
                (Some(dst), Some(src)) if osz == 8 => Eff::AddReg { dst, src },
                _ => Eff::WrReg, // narrower, or `add reg, [mem]`
            },
            I::Movsxd => match (reg, mem) {
                (Some(dst), true) if osz == 8 => Eff::LoadSx32 { dst },
                _ => Eff::WrReg, // the register form, or without REX.W a plain 32-bit move
            },
            I::CmpRegRm => match (reg, rm_opnd) {
                (Some(r), Some(rhs)) if modeled => Eff::Cmp {
                    w: osz,
                    lhs: Opnd::Reg(r),
                    rhs,
                },
                _ => Eff::NoGp,
            },
            I::CmpRmReg => match (rm_opnd, reg) {
                (Some(lhs), Some(r)) if modeled => Eff::Cmp {
                    w: osz,
                    lhs,
                    rhs: Opnd::Reg(r),
                },
                _ => Eff::NoGp,
            },
            I::AluImmGroup(digit) => {
                if digit == 7 {
                    // cmp r/m, imm
                    return match rm_opnd {
                        Some(lhs) if modeled => Eff::Cmp {
                            w: osz,
                            lhs,
                            rhs: Opnd::Imm(immv as u64),
                        },
                        _ => Eff::NoGp,
                    };
                }
                let alu = match digit {
                    0 => Some(Alu::Add),
                    4 => Some(Alu::And),
                    5 => Some(Alu::Sub),
                    _ => None, // or/adc/sbb/xor — not modeled for bounds
                };
                match (mem, osz, alu, rm) {
                    (true, ..) => Eff::NoGp, // RMW on memory (access checked); no GP write
                    (false, 8, Some(alu), Some(dst)) => Eff::AluImm {
                        dst,
                        alu,
                        imm: immv,
                    },
                    (false, ..) => Eff::WrRm, // narrower, or or/adc/sbb/xor: clobber r/m
                }
            }
            I::ByteAluImm if mem => Eff::NoGp,
            I::ByteAluImm => Eff::WrRm,
            I::Group3(digit) => match digit {
                0 | 1 => Eff::NoGp,                // test r/m, imm — flags only
                4..=7 => Eff::Wr(bit(0) | bit(2)), // mul/imul/div/idiv → rax, rdx
                _ if mem => Eff::NoGp,             // not/neg [mem]
                _ => Eff::WrRm,                    // not/neg r/m
            },
            I::Group5(digit) => match digit {
                0 | 1 if !mem => Eff::WrRm, // inc/dec r/m register
                6 => Eff::Push {
                    src: rm.filter(|_| !mem),
                }, // push r/m
                _ => Eff::NoGp,             // inc/dec [mem]; call/jmp (kind already set)
            },
            I::Cmov(cc) => match reg {
                Some(dst) if modeled => Eff::Cmov { dst, cc, w: osz },
                Some(_) => Eff::WrReg, // 16-bit cmov: merges, so clobber
                None => Eff::NoGp,
            },
            I::XchgRegRm => Eff::Wr(reg.map_or(0, bit) | rm.map_or(0, bit)),
            I::PopRm => Eff::Pop {
                dst: rm.filter(|_| !mem),
            },
        }
    }

    /// Consume a ModRM byte and whatever addressing bytes it implies (SIB, displacement) in 64-bit
    /// mode, then `imm_bytes` of immediate; classify control flow as `kind`, attach the decoded
    /// operands, and resolve the data-flow effect from `intent`.
    fn finish_modrm(
        c: &mut Cursor,
        start: usize,
        p: &Pfx,
        imm_bytes: usize,
        kind: Kind,
        intent: EffIntent,
    ) -> Option<Decoded> {
        let operands = read_modrm(c, p)?;
        let immv = read_imm(c, imm_bytes)?;
        let eff = resolve_eff(intent, &operands, p.osz(), immv);
        Some(Decoded {
            len: c.pos - start,
            kind,
            operands,
            eff,
            imm_len: imm_bytes as u8,
            keeps_flags: false,
        })
    }

    /// Read a sign-extended immediate of `bytes` (0, 1, 2, 4, or 8) and advance the cursor.
    fn read_imm(c: &mut Cursor, bytes: usize) -> Option<i64> {
        let s = c.bytes.get(c.pos..c.pos + bytes)?;
        let v = match bytes {
            0 => 0,
            1 => i64::from(s[0] as i8),
            2 => i64::from(i16::from_le_bytes([s[0], s[1]])),
            4 => i64::from(i32::from_le_bytes([s[0], s[1], s[2], s[3]])),
            8 => i64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]),
            _ => return None,
        };
        c.pos += bytes;
        Some(v)
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
        use EffIntent as E;
        use Kind::*;
        match op {
            // --- forbidden / privileged / system ---
            0xF4 => return Some(one(c, start, Forbidden("hlt"), Eff::NoGp)),
            0xFA => return Some(one(c, start, Forbidden("cli"), Eff::NoGp)),
            0xFB => return Some(one(c, start, Forbidden("sti"), Eff::NoGp)),
            0xCC => return Some(one(c, start, Normal, Eff::NoGp)), // int3 — debugtrap/padding, allowed
            0xCD => return finish_imm(c, start, 1, Forbidden("int imm8"), Eff::NoGp),
            0xCE => return Some(one(c, start, Forbidden("into"), Eff::NoGp)),
            0xCF => return Some(one(c, start, Forbidden("iret"), Eff::NoGp)),
            0xE4 | 0xE5 => return finish_imm(c, start, 1, Forbidden("in imm8"), Eff::NoGp),
            0xE6 | 0xE7 => return finish_imm(c, start, 1, Forbidden("out imm8"), Eff::NoGp),
            0xEC..=0xEF => return Some(one(c, start, Forbidden("in/out dx"), Eff::NoGp)),
            0x6C..=0x6F => return Some(one(c, start, Forbidden("ins/outs"), Eff::NoGp)),
            0xCB | 0xCA => return Some(one(c, start, Forbidden("far ret"), Eff::NoGp)),
            0xEA => return Some(one(c, start, Forbidden("far jmp"), Eff::NoGp)),
            0x9A => return Some(one(c, start, Forbidden("far call"), Eff::NoGp)),
            // `xlat` reads `[rbx + al]`, a memory operand the ModRM doesn't name (#2603 hole 9).
            0xD7 => {
                return Some(one(
                    c,
                    start,
                    Forbidden("xlat (implicit memory operand)"),
                    Eff::NoGp,
                ))
            }
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
                    eff: Eff::NoGp,
                    imm_len: 0,
                    keeps_flags: false,
                });
            }
            0x70..=0x7F => {
                let rel = c.rel(1)?; // jcc rel8
                return Some(rel_insn(start, c.pos, rel, Branch::Jcc));
            }
            0xC3 => return Some(one(c, start, Ret, Eff::NoGp)),
            0xC2 => return finish_imm(c, start, 2, Ret, Eff::NoGp), // ret imm16
            _ => {}
        }
        // ALU r/m,reg forms (add/or/adc/sbb/and/sub/xor/cmp): columns rm8,r8 / rm,r / r8,rm8 / r,rm.
        // Group 7 is `cmp` (flags only — the dataflow models it); the rest write their destination.
        if matches!(op,
            0x00..=0x03 | 0x08..=0x0B | 0x10..=0x13 | 0x18..=0x1B
            | 0x20..=0x23 | 0x28..=0x2B | 0x30..=0x33 | 0x38..=0x3B)
        {
            let is_cmp = (op >> 3) & 0x07 == 0x07;
            let byte = op & 0x01 == 0; // the r/m8, r8 / r8, r/m8 columns: 8-bit whatever REX.W says
            let intent = match (is_cmp, op & 0x07) {
                (true, _) if byte => E::NoGp, // an 8-bit compare: flags only, not modeled
                (true, 0 | 1) => E::CmpRmReg,
                (true, _) => E::CmpRegRm,
                _ if op == 0x01 => E::AddRmReg,
                _ if op == 0x03 => E::AddRegRm,
                (false, 0 | 1) => E::WrRm,
                (false, _) => E::WrReg,
            };
            return finish_modrm(c, start, p, 0, Normal, intent);
        }
        match op {
            // ModRM, no immediate.
            0x84 | 0x85 => finish_modrm(c, start, p, 0, Normal, E::NoGp), // test
            // The moves, `lea`, `xchg` and `pop` leave the flags (a register allocator's moves may sit
            // between a `cmp` and the `cmov` that tests it).
            0x86 | 0x87 => keep(finish_modrm(c, start, p, 0, Normal, E::XchgRegRm)), // xchg
            0x88 => keep(finish_modrm(c, start, p, 0, Normal, E::WrRm)),             // mov rm8, r8
            0x89 => keep(finish_modrm(c, start, p, 0, Normal, E::MovToRm)),          // mov rm, r
            0x8A => keep(finish_modrm(c, start, p, 0, Normal, E::WrReg)),            // mov r8, rm8
            0x8B => keep(finish_modrm(c, start, p, 0, Normal, E::MovToReg)),         // mov r, rm
            0x63 => keep(finish_modrm(c, start, p, 0, Normal, E::Movsxd)), // movsxd r64, rm32
            0x8D => keep(finish_modrm(c, start, p, 0, Normal, E::Lea)),
            0x8F => keep(finish_modrm(c, start, p, 0, Normal, E::PopRm)), // pop r/m
            0xD0..=0xD3 => finish_modrm(c, start, p, 0, Normal, E::WrRm), // shift r/m, 1|cl
            0xFE => finish_modrm(c, start, p, 0, Normal, E::WrRm),        // inc/dec r/m8
            // ModRM + imm8.
            0x80 | 0x82 => finish_modrm(c, start, p, 1, Normal, E::ByteAluImm),
            0x83 => {
                let digit = (c.peek()? >> 3) & 0x07;
                finish_modrm(c, start, p, 1, Normal, E::AluImmGroup(digit))
            }
            0x6B => finish_modrm(c, start, p, 1, Normal, E::WrReg), // imul r, r/m, imm8
            0xC0 | 0xC1 => finish_modrm(c, start, p, 1, Normal, E::WrRm), // shift r/m, imm8
            0xC6 => keep(finish_modrm(c, start, p, 1, Normal, E::WrRm)), // mov r/m8, imm8
            // ModRM + immZ.
            0x81 => {
                let digit = (c.peek()? >> 3) & 0x07;
                finish_modrm(c, start, p, imm_z(p), Normal, E::AluImmGroup(digit))
            }
            0x69 => finish_modrm(c, start, p, imm_z(p), Normal, E::WrReg), // imul r, r/m, immZ
            0xC7 => keep(finish_modrm(c, start, p, imm_z(p), Normal, E::MovImmToRm)),
            // groups 3/5: immediate (if any) depends on the /reg field — peek at the ModRM.
            0xF6 => group3(c, start, p, 1),
            0xF7 => group3(c, start, p, imm_z(p)),
            0xFF => group5(c, start, p),
            // push r / pop r — the opcode-encoded register.
            0x50..=0x57 => {
                let src = Some((op & 0x07) | (u8::from(p.rex_b) << 3));
                Some(one(c, start, Normal, Eff::Push { src }).keep_flags())
            }
            0x58..=0x5F => {
                let dst = Some((op & 0x07) | (u8::from(p.rex_b) << 3));
                Some(one(c, start, Normal, Eff::Pop { dst }).keep_flags())
            }
            0x68 => keep(finish_imm(
                c,
                start,
                imm_z(p),
                Normal,
                Eff::Push { src: None },
            )),
            0x6A => keep(finish_imm(c, start, 1, Normal, Eff::Push { src: None })),
            // xchg eax, r — writes rax and r.
            0x91..=0x97 => {
                let r = (op & 0x07) | (u8::from(p.rex_b) << 3);
                Some(one(c, start, Normal, Eff::Wr(bit(0) | bit(r))))
            }
            0x98 => Some(one(c, start, Normal, Eff::Wr(bit(0)))), // cwde/cdqe → rax
            0x99 => Some(one(c, start, Normal, Eff::Wr(bit(2)))), // cwd/cdq/cqo → rdx
            // nops, flag ops, invalid-in-64 legacy: no tracked GP write.
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
            | 0x90
            | 0x9B..=0x9F
            | 0xF5
            | 0xF8
            | 0xF9
            | 0xFC
            | 0xFD => Some(one(c, start, Normal, Eff::NoGp)),
            // Accumulator ALU + immediate (writes rax); cmp/test/push imm write no GP register.
            0x04 | 0x0C | 0x14 | 0x1C | 0x24 | 0x2C | 0x34 => {
                finish_imm(c, start, 1, Normal, Eff::Wr(bit(0)))
            }
            0x3C | 0xA8 => finish_imm(c, start, 1, Normal, Eff::NoGp),
            0x05 | 0x0D | 0x15 | 0x1D | 0x25 | 0x2D | 0x35 => {
                finish_imm(c, start, imm_z(p), Normal, Eff::Wr(bit(0)))
            }
            0x3D | 0xA9 => finish_imm(c, start, imm_z(p), Normal, Eff::NoGp),
            // mov r8, imm8 — writes the opcode-encoded register (byte write → clobber).
            0xB0..=0xB7 => {
                let dst = (op & 0x07) | (u8::from(p.rex_b) << 3);
                keep(finish_imm(c, start, 1, Normal, Eff::Wr(bit(dst))))
            }
            // mov r32/r64, imm — writes a known constant into the opcode-encoded register; `mov r16,
            // imm16` keeps the upper 48 bits, so it only clobbers (#2603 hole 7).
            0xB8..=0xBF => {
                let dst = (op & 0x07) | (u8::from(p.rex_b) << 3);
                let size = if p.rex_w { 8 } else { imm_z(p) };
                let immv = read_imm(c, size)?;
                let eff = match size {
                    8 => Eff::MovImm { dst, imm: immv },
                    4 => Eff::MovImm {
                        dst,
                        imm: immv & 0xFFFF_FFFF,
                    },
                    _ => Eff::Wr(bit(dst)),
                };
                Some(Decoded {
                    len: c.pos - start,
                    kind: Normal,
                    operands: NO_OPERANDS,
                    eff,
                    imm_len: size as u8,
                    keeps_flags: true,
                })
            }
            _ => Some(UNDECODABLE),
        }
    }

    /// Group 3 (`F6`/`F7`): `/0`,`/1` (test) carry an immediate; `/2..` (not/neg/mul/imul/div/idiv)
    /// do not.
    fn group3(c: &mut Cursor, start: usize, p: &Pfx, imm: usize) -> Option<Decoded> {
        let digit = (c.peek()? >> 3) & 0x07;
        let imm_bytes = if digit <= 1 { imm } else { 0 };
        finish_modrm(
            c,
            start,
            p,
            imm_bytes,
            Kind::Normal,
            EffIntent::Group3(digit),
        )
    }

    /// Group 5 (`FF`): `/2` indirect call, `/4` indirect jmp, `/6` push r/m, `/0`,`/1` inc/dec. The far
    /// forms `/3`,`/5` load a new `cs:rip` from memory, so they are forbidden like `9A`/`EA`.
    fn group5(c: &mut Cursor, start: usize, p: &Pfx) -> Option<Decoded> {
        let digit = (c.peek()? >> 3) & 0x07;
        let kind = match digit {
            2 => Kind::IndirectCall,
            4 => Kind::IndirectJmp,
            3 => Kind::Forbidden("far call"),
            5 => Kind::Forbidden("far jmp"),
            _ => Kind::Normal,
        };
        let d = finish_modrm(c, start, p, 0, kind, EffIntent::Group5(digit))?;
        // `jmp`/`push` leave the flags; `inc`/`dec` write them, and a call's callee clobbers them.
        Some(if matches!(digit, 4 | 6) {
            d.keep_flags()
        } else {
            d
        })
    }

    fn decode_0f(c: &mut Cursor, start: usize, p: &Pfx, op2: u8) -> Option<Decoded> {
        use EffIntent as E;
        use Kind::*;
        match op2 {
            0x05 => return Some(one(c, start, Forbidden("syscall"), Eff::NoGp)),
            0x07 => return Some(one(c, start, Forbidden("sysret"), Eff::NoGp)),
            0x34 => return Some(one(c, start, Forbidden("sysenter"), Eff::NoGp)),
            0x35 => return Some(one(c, start, Forbidden("sysexit"), Eff::NoGp)),
            0x0B => return Some(one(c, start, Ud2, Eff::NoGp)), // trap Cranelift emits for `unreachable`
            0x20..=0x23 => return finish_modrm(c, start, p, 0, Forbidden("mov cr/dr"), E::NoGp),
            0x80..=0x8F => {
                let rel = c.rel(4)?; // jcc rel32
                return Some(rel_insn(start, c.pos, rel, Branch::Jcc));
            }
            // cmovcc r, r/m — the D63 confinement redirect.
            0x40..=0x4F => return keep(finish_modrm(c, start, p, 0, Normal, E::Cmov(op2 & 0x0F))),
            // setcc r/m8 (byte write → clobber the r/m register).
            0x90..=0x9F => return keep(finish_modrm(c, start, p, 0, Normal, E::WrRm)),
            // movzx/movsx r, r/m — zero/sign-extended, not a tracked pointer value → clobber reg.
            0xB6 | 0xB7 | 0xBE | 0xBF => {
                return keep(finish_modrm(c, start, p, 0, Normal, E::WrReg))
            }
            0xAF => return finish_modrm(c, start, p, 0, Normal, E::WrReg), // imul r, r/m
            0xBC | 0xBD => return finish_modrm(c, start, p, 0, Normal, E::WrReg), // bsf/bsr r, r/m
            // shld/shrd r/m, r, (imm8|cl) — write the r/m.
            0xA4 | 0xAC => return finish_modrm(c, start, p, 1, Normal, E::WrRm),
            0xA5 | 0xAD => return finish_modrm(c, start, p, 0, Normal, E::WrRm),
            // bt/bts/btr/btc r/m, reg: with a memory operand the register is a *signed bit offset*
            // into memory, reaching `[mem + (reg >> 3)]` far past the operand the policy checks
            // (#2603 hole 9). Cranelift emits only the register forms.
            0xA3 | 0xAB | 0xB3 | 0xBB => {
                let intent = if op2 == 0xA3 { E::NoGp } else { E::WrRm };
                let d = finish_modrm(c, start, p, 0, Normal, intent)?;
                return Some(if d.operands.mem.is_some() {
                    Decoded {
                        kind: Forbidden("bt* with a register bit offset into memory"),
                        ..d
                    }
                } else {
                    d
                });
            }
            0xC0 | 0xC1 => return finish_modrm(c, start, p, 0, Normal, E::XchgRegRm), // xadd — writes both
            0xB0 | 0xB1 => return finish_modrm(c, start, p, 0, Normal, E::Wr(bit(0))), // cmpxchg → rax (+ r/m)
            0xA2 => {
                return Some(one(
                    c,
                    start,
                    Normal,
                    Eff::Wr(bit(0) | bit(1) | bit(2) | bit(3)),
                ))
            } // cpuid
            0x31 => return Some(one(c, start, Normal, Eff::Wr(bit(0) | bit(2)))), // rdtsc → rax, rdx
            0x77 => return Some(one(c, start, Normal, Eff::NoGp)),                // emms
            _ => {}
        }
        match op2 {
            // SSE/vector ModRM + imm8 (operands are XMM — no GP write) …
            0x70..=0x73 | 0xC2 | 0xC4 | 0xC6 => finish_modrm(c, start, p, 1, Normal, E::NoGp),
            // … except pextrw r32, xmm, imm8, which writes a GP register.
            0xC5 => finish_modrm(c, start, p, 1, Normal, E::WrReg),
            // The SSE moves (movups/movss/movsd, movaps, movd/movq, movdqa/movdqu, movq) leave the
            // flags: a register allocator's spills and moves.
            0x10 | 0x11 | 0x28 | 0x29 | 0x6E | 0x6F | 0x7E | 0x7F | 0xD6 => {
                keep(finish_modrm(c, start, p, 0, Normal, E::NoGp))
            }
            // Everything else in the 0F map (SSE/SSE2 float+int vector ops, fences): ModRM, no imm.
            // A handful move a value into a GP register; those are cross-checked against iced and the
            // ones the corpus exercises are classified above. The rest are XMM-only (no GP write).
            _ => finish_modrm(c, start, p, 0, Normal, E::NoGp),
        }
    }

    fn decode_0f38(c: &mut Cursor, start: usize, p: &Pfx) -> Option<Decoded> {
        c.next()?; // opcode in the 0F38 map — all have ModRM, no immediate; XMM-only (no GP write)
        finish_modrm(c, start, p, 0, Kind::Normal, EffIntent::NoGp)
    }

    fn decode_0f3a(c: &mut Cursor, start: usize, p: &Pfx) -> Option<Decoded> {
        c.next()?; // opcode in the 0F3A map — all have ModRM + imm8; XMM-only (no GP write)
        finish_modrm(c, start, p, 1, Kind::Normal, EffIntent::NoGp)
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
        // VEX/EVEX encode AVX vector ops: their operands are XMM/YMM/ZMM, so no GP register is
        // written. (The BMI GP-writing VEX ops are not in the corpus; the iced write-set cross-check
        // would flag one if it appeared, to be classified then.)
        let d = finish_modrm(c, start, p, imm, Kind::Normal, EffIntent::NoGp)?;
        // The AVX moves (vmovups/vmovaps/vmovdqu/vmovq/…) leave the flags.
        let moves = matches!(map, Map::Of)
            && matches!(
                op,
                0x10 | 0x11 | 0x28 | 0x29 | 0x6E | 0x6F | 0x7E | 0x7F | 0xD6
            );
        Some(if moves { d.keep_flags() } else { d })
    }

    fn one(c: &Cursor, start: usize, kind: Kind, eff: Eff) -> Decoded {
        Decoded {
            len: c.pos - start,
            kind,
            operands: NO_OPERANDS,
            eff,
            imm_len: 0,
            keeps_flags: false,
        }
    }

    /// Mark a decode arm's result as leaving the flags unchanged.
    fn keep(d: Option<Decoded>) -> Option<Decoded> {
        d.map(Decoded::keep_flags)
    }

    fn finish_imm(
        c: &mut Cursor,
        start: usize,
        imm: usize,
        kind: Kind,
        eff: Eff,
    ) -> Option<Decoded> {
        c.skip(imm)?;
        Some(Decoded {
            len: c.pos - start,
            kind,
            operands: NO_OPERANDS,
            eff,
            imm_len: imm as u8,
            keeps_flags: false,
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
            eff: Eff::NoGp,
            imm_len: 0,
            keeps_flags: true, // a jmp/jcc leaves the flags
        }
    }

    /// Every instruction the validator decodes, block by block, as [`check`] visits them. Test-only,
    /// for the iced-x86 cross-check.
    #[cfg(test)]
    pub(crate) fn walk(bytes: &[u8], relocs: &[Reloc]) -> Vec<(usize, Decoded)> {
        let ctx = Ctx {
            bytes,
            relocs,
            layout: None,
        };
        let mut findings = Vec::new();
        Cfg::build(ctx, &mut findings)
            .blocks
            .into_iter()
            .flatten()
            .collect()
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::{compile, validate};
        use cranelift_codegen::binemit::Reloc as R;
        use iced_x86::{
            Decoder, DecoderOptions, Instruction, InstructionInfoFactory, OpAccess, OpKind,
            Register,
        };
        use temen_text::parse_module;

        /// Compile `src` through the real JIT path and return each function's capture.
        fn compiled(src: &str) -> Vec<crate::validate::CapturedFn> {
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

        /// iced's GP-register write-set for one instruction, as our 16-bit mask: every register it
        /// writes (unconditionally or conditionally), mapped to its 64-bit container's number.
        fn iced_gp_writes(insn: &Instruction) -> u16 {
            let mut f = InstructionInfoFactory::new();
            let mut w = 0u16;
            for ur in f.info(insn).used_registers() {
                let writes = matches!(
                    ur.access(),
                    OpAccess::Write
                        | OpAccess::ReadWrite
                        | OpAccess::CondWrite
                        | OpAccess::ReadCondWrite
                );
                if writes {
                    if let Some(n) = gp(ur.register().full_register()) {
                        w |= 1 << u16::from(n);
                    }
                }
            }
            w
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

        /// The decoder's boundaries, validity, memory-operand decode, and data-flow effect must match
        /// iced-x86 on real output, and `check` must find nothing on clean Cranelift code. The effect
        /// is compared wherever the transfer relies on it: the GP write-set (rsp/rbp included — they
        /// carry stack facts), the flags an instruction claims to keep, a `cmov`'s condition, the value
        /// a `mov r, imm` leaves, a `cmp`'s immediate, and a load's or store's width (#2603).
        fn cross_check(src: &str) {
            for f in compiled(src) {
                let bytes = &f.code;
                for (pos, d) in walk(bytes, &f.relocs) {
                    assert_ne!(d.len, 0, "ours: undecodable at +0x{pos:x}\n{:02x?}", bytes);
                    let insn = iced_at(bytes, pos).unwrap_or_else(|| {
                        panic!("iced rejects what we decoded at +0x{pos:x}\n{:02x?}", bytes)
                    });
                    let at = |what: &str| format!("{what} at +0x{pos:x} ({:?})\n{:02x?}", d, bytes);
                    assert_eq!(d.len, insn.len(), "{}", at("length mismatch"));
                    // When our decoder reports a memory operand, iced must see the same one.
                    if let Some(mem) = d.operands.mem {
                        // Compare the fields iced exposes identically (disp only when not rip-rel).
                        let ours = if mem.rip { Mem { disp: 0, ..mem } } else { mem };
                        assert_eq!(
                            Some(ours),
                            iced_mem(&insn),
                            "{}",
                            at("memory-operand mismatch")
                        );
                    }
                    // A GP register iced says the instruction writes but we don't would let a stale
                    // fact (mem_base, a confined index, a stack offset) survive a real overwrite.
                    let (ours, iced_w) = (writes_of(&d), iced_gp_writes(&insn));
                    assert_eq!(
                        iced_w & !ours,
                        0,
                        "{}",
                        at(&format!(
                            "GP write-set gap: iced writes {iced_w:#06x}, ours {ours:#06x}"
                        ))
                    );
                    // An instruction that claims to keep the flags must not modify any.
                    if d.keeps_flags {
                        assert_eq!(insn.rflags_modified(), 0, "{}", at("modifies the flags"));
                    }
                    // A `ret` pops what the function's own signature says its callers pass on the
                    // stack, the stack effect each call to it is checked under (#2603). (No function
                    // here tail-calls with more stack arguments than it takes, which would grow its
                    // argument area and its `ret`.)
                    if d.kind == Kind::Ret {
                        let pops = if d.len == 3 {
                            u32::from(u16::from_le_bytes([bytes[pos + 1], bytes[pos + 2]]))
                        } else {
                            0
                        };
                        let want = f.contract.and_then(|c| c.pops);
                        assert_eq!(Some(pops), want, "{}", at("ret pops"));
                    }
                    match d.eff {
                        Eff::Cmov { cc, .. } => {
                            // iced numbers the conditions from 1 (0 is "none").
                            assert_eq!(insn.condition_code() as u8, cc + 1, "{}", at("cmov cc"));
                        }
                        Eff::MovImm { imm, .. } => {
                            assert_eq!(imm as u64, insn.immediate(1), "{}", at("mov immediate"));
                        }
                        Eff::Cmp {
                            w,
                            rhs: Opnd::Imm(v),
                            ..
                        } => {
                            let mask = if w == 8 {
                                u64::MAX
                            } else {
                                u64::from(u32::MAX)
                            };
                            assert_eq!(v & mask, insn.immediate(1) & mask, "{}", at("cmp imm"));
                        }
                        Eff::LoadReg { .. } => {
                            assert_eq!(insn.memory_size().size(), 8, "{}", at("load width"));
                        }
                        Eff::StoreReg { w, .. } => {
                            let want = usize::from(w);
                            assert_eq!(insn.memory_size().size(), want, "{}", at("store width"));
                        }
                        _ => {}
                    }
                }
                let entry = Entry::Guest(f.layout);
                let findings = check(bytes, &f.relocs, entry);
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
        fn high_register_pressure_spills_membase() {
            // #2603 item 3: enough live values that register allocation spills `mem_base` to the
            // stack (`mov [rsp], rdi`) and reloads it (`mov rdi, [rsp]`) before a window access. The
            // spill-slot tracking must carry the `MemBase` fact across that round-trip, so the policy
            // sees `[mem_base + confined]` and finds nothing. (Before slot tracking the reload read
            // `Unknown` and the access was flagged.) The pair-sum keeps the args live so the pressure
            // is real; the load is late so `mem_base` is live under it.
            use std::fmt::Write;
            let n = 14usize;
            let mut src = String::from("memory 16\nfunc (");
            for i in 0..n {
                src.push_str(if i > 0 { ", i64" } else { "i64" });
            }
            src.push_str(") -> (i64) {\nblock 0 (");
            for i in 0..n {
                if i > 0 {
                    src.push_str(", ");
                }
                write!(src, "v{i}: i64").unwrap();
            }
            src.push_str(") {\n");
            let mut next = n;
            let mut partials = Vec::new();
            for j in 0..n / 2 {
                writeln!(src, "  v{next} = i64.add v{} v{}", 2 * j, 2 * j + 1).unwrap();
                partials.push(next);
                next += 1;
            }
            let load = next;
            writeln!(src, "  v{load} = i64.load v0").unwrap();
            next += 1;
            let mut acc = partials[0];
            for &p in &partials[1..] {
                writeln!(src, "  v{next} = i64.add v{acc} v{p}").unwrap();
                acc = next;
                next += 1;
            }
            writeln!(src, "  v{next} = i64.add v{acc} v{load}").unwrap();
            writeln!(src, "  return v{next}\n  }}\n}}").unwrap();
            cross_check(&src);
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
        fn call_with_stack_arguments_then_a_window_access() {
            // A call with five guest arguments (three on the stack: Tail CC passes only six integer
            // arguments in registers, four of them the context params), then a window access. The
            // caller parks mem_base in a callee-saved register across the call, and the D63 bound
            // comes from the constant pool (`cmp idx, [rip + k]`) — the shapes #2603 PR 1 models.
            cross_check(
                "memory 16\nfunc (i64, i64, i64, i64, i64) -> (i64) {\n\
                 block 0 (v0: i64, v1: i64, v2: i64, v3: i64, v4: i64) {\n  \
                 v5 = i64.add v0 v4\n  return v5\n  }\n}\n\
                 func (i64) -> (i64) {\nblock 0 (v0: i64) {\n  v1 = i64.const 1\n  \
                 v2 = call 0 (v0, v1, v1, v1, v0)\n  v3 = i64.load v0\n  v4 = i64.add v2 v3\n  \
                 return v4\n  }\n}\n",
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
        fn memory_loop_carries_membase_across_blocks() {
            // A load inside a loop body: `mem_base` (a context param) stays live across every block
            // while the loop-carried index is confined per iteration — the cross-block dataflow the
            // #2535 engine must get right (and find nothing to flag).
            cross_check(
                "memory 16\nfunc (i64) -> (i64) {\nblock 0 (v0: i64) {\n  v1 = i64.const 0\n  \
                 br 1(v0, v1)\n  }\nblock 1 (v2: i64, v3: i64) {\n  v4 = i64.const 0\n  \
                 v5 = i64.eq v2 v4\n  br_if v5 3(v3) 2(v2, v3)\n  }\n\
                 block 2 (v6: i64, v7: i64) {\n  v8 = i64.load v6\n  v9 = i64.add v7 v8\n  \
                 v10 = i64.const 8\n  v11 = i64.sub v6 v10\n  br 1(v11, v9)\n  }\n\
                 block 3 (v12: i64) {\n  return v12\n  }\n}\n",
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

        /// The abstract lattice's `meet` (the fact true on both paths, and the order the checker
        /// relies on) and the loop-header widening that bounds the fixpoint (#2535).
        #[test]
        fn lattice_meet_and_widen() {
            use Fact::*;
            let conf = Confined {
                bound: 0x20_0000_0000,
                guard: 0x20_0000_0000,
            };
            assert_eq!(MemBase.meet(MemBase), MemBase);
            assert_eq!(MemBase.meet(Vmctx), Unknown);
            assert_eq!(conf.meet(conf), conf);
            let other_guard = Confined {
                bound: 0x20_0000_0000,
                guard: 4096,
            };
            assert_eq!(conf.meet(other_guard), Unknown); // different guard ⇒ no fact
                                                         // The same guard with a looser bound: the meet keeps the looser one.
            let looser = Confined {
                bound: 0x30_0000_0000,
                guard: 0x20_0000_0000,
            };
            assert_eq!(conf.meet(looser), looser);
            assert_eq!(conf.meet(Unknown), Unknown);
            assert_eq!(Const(5).meet(Const(5)), Const(5));
            assert_eq!(Const(5).meet(Const(6)), Unknown);
            // A bound true on both paths is the looser (larger) of the two.
            assert_eq!(Bounded(4).meet(Bounded(8)), Bounded(8));
            assert_eq!(Bounded(4).meet(MemBase), Unknown);
            // Two stack pointers at different offsets still point into the stack.
            assert_eq!(Stack(Some(-8)).meet(Stack(Some(-16))), Stack(None));
            assert_eq!(Stack(Some(-8)).meet(MemBase), Unknown);

            // Widening at a loop header drops `Bounded` (which could grow each iteration) but keeps
            // the loop-invariant context-param and `Confined` facts, so the fixpoint terminates.
            let mut s = State::entry();
            s.regs[8] = Bounded(16);
            s.regs[9] = conf;
            s.widen();
            assert_eq!(s.regs[8], Unknown);
            assert_eq!(s.regs[9], conf);
            assert_eq!(s.regs[reg::RDI as usize], MemBase);
        }

        // --- the hand-assembled policy tests ---

        /// The 128 GiB reservation of a real module (`guard_offset` 0x20_0000_0000), for the window
        /// accesses the tests below assemble by hand.
        const WIN: Option<WindowLayout> = Some(WindowLayout {
            mask: 0x1F_FFFF_FFFF,
            guard_offset: 0x20_0000_0000,
        });

        /// `mov rax, 0x20_0000_0000` — the guard offset, D63's redirect target (10 bytes).
        const GUARD: [u8; 10] = [0x48, 0xb8, 0, 0, 0, 0, 0x20, 0, 0, 0];

        fn findings(bytes: &[u8]) -> Vec<Finding> {
            check(bytes, &[], Entry::Guest(WIN))
        }

        fn has(f: &[Finding], needle: &str) -> bool {
            f.iter().any(|f| f.why.contains(needle))
        }

        /// What the hand-assembled calls below name: a Tail-CC callee that pops nothing.
        const TAIL0: Contract = Contract {
            clobbers: SYSV_CLOBBERS,
            pops: Some(0),
        };

        /// The relocation a call or tail call to [`TAIL0`] carries on the `rel32` at `at`.
        fn call_reloc(at: u32) -> Reloc {
            Reloc {
                at,
                kind: R::X86CallPCRel4,
                addend: -4,
                callee: Some(TAIL0),
            }
        }

        /// A relocation of kind `kind` on data at `at`, naming nothing a call may reach.
        fn data_reloc(at: u32, kind: R) -> Reloc {
            Reloc {
                at,
                kind,
                addend: 0,
                callee: None,
            }
        }

        /// `bytes` = `GUARD` followed by `rest`.
        fn after_guard(rest: &[u8]) -> Vec<u8> {
            [&GUARD[..], rest].concat()
        }

        /// Run `bytes`, one block, from the entry state.
        fn run(bytes: &[u8]) -> State {
            run_with(bytes, &[])
        }

        /// [`run`], with `relocs`.
        fn run_with(bytes: &[u8], relocs: &[Reloc]) -> State {
            let mut fs = Vec::new();
            let ctx = Ctx {
                bytes,
                relocs,
                layout: WIN,
            };
            let (insns, _) = decode_block(ctx, 0, bytes.len(), &mut fs);
            transfer_block(&insns, State::entry(), None, ctx)
        }

        /// A window access through an index that was **not** confined by the D63 sequence is the
        /// escape-relevant case the policy must flag: `mov rax, [rdi + r8]` with `r8` a guest
        /// argument (unknown, unconfined) and `rdi` = `mem_base`.
        #[test]
        fn policy_flags_unconfined_window_index() {
            let f = findings(&[0x4a, 0x8b, 0x04, 0x07, 0xc3]); // mov rax,[rdi+r8*1]; ret
            assert!(has(&f, "unconfined index"), "{f:?}");
        }

        /// A memory access whose base register has no recognized provenance (here `r9`, a guest
        /// argument) is flagged — it is neither the window, the stack, vmctx, the fn-table, nor rodata.
        #[test]
        fn policy_flags_unknown_base() {
            let f = findings(&[0x49, 0x8b, 0x01, 0xc3]); // mov rax,[r9]; ret
            assert!(has(&f, "unrecognized provenance"), "{f:?}");
        }

        /// A module without a window has no window access to re-prove: one is flagged rather than
        /// checked against the guard the code itself chose.
        #[test]
        fn policy_flags_a_window_access_without_a_window() {
            let bytes = after_guard(&[
                0x4c, 0x3b, 0xc0, // cmp r8, rax
                0x4c, 0x0f, 0x43, 0xc0, // cmovae r8, rax
                0x4a, 0x8b, 0x04, 0x07, // mov rax, [rdi + r8*1]
                0xc3,
            ]);
            let f = check(&bytes, &[], Entry::Guest(None));
            assert!(has(&f, "no window"), "{f:?}");
            // A trampoline's arguments are the host's, so the memory policy waits for its contract.
            assert!(check(&bytes, &[], Entry::Host).is_empty());
        }

        /// The full D63 confinement sequence is recognized and produces no finding: materialize the
        /// guard, compare the index, `cmovae` it to the guard, then access `[mem_base + confined]`.
        #[test]
        fn policy_accepts_the_confinement_sequence() {
            let bytes = after_guard(&[
                0x4c, 0x3b, 0xc0, // cmp r8, rax
                0x4c, 0x0f, 0x43, 0xc0, // cmovae r8, rax
                0x4a, 0x8b, 0x04, 0x07, // mov rax, [rdi + r8*1]
                0xc3, // ret
            ]);
            let f = findings(&bytes);
            assert!(f.is_empty(), "the confinement sequence was flagged: {f:?}");
        }

        /// Real D63 code compares against a bound in the function's own constant pool, since the
        /// reservation doesn't fit an `imm32` (`cmp idx, [rip + k]`). The bound is read from the bytes
        /// — and only when no relocation patches them, so a patched constant confines nothing.
        #[test]
        fn policy_reads_the_d63_bound_from_the_constant_pool() {
            let bytes = [
                0x48, 0xb9, 0, 0, 0, 0, 0x20, 0, 0, 0, // [0]  mov rcx, guard
                0x4c, 0x3b, 0x05, 0x09, 0, 0, 0, // [10] cmp r8, [rip + 9] → [26]
                0x4c, 0x0f, 0x43, 0xc1, // [17] cmovae r8, rcx
                0x4a, 0x8b, 0x04, 0x07, // [21] mov rax, [rdi + r8*1]
                0xc3, // [25] ret
                0, 0, 0, 0, 0x20, 0, 0, 0, // [26] the bound, 0x20_0000_0000
            ];
            let f = findings(&bytes);
            assert!(
                f.is_empty(),
                "a constant-pool bound was not recognized: {f:?}"
            );
            let patched = [data_reloc(26, R::Abs8)];
            let f = check(&bytes, &patched, Entry::Guest(WIN));
            assert!(
                has(&f, "unconfined index"),
                "a patched bound confined: {f:?}"
            );
        }

        /// br_table's index clamp is the mirrored, 32-bit form of the same `cmov` shape: `mov eax, n;
        /// cmp ecx, eax; cmovb eax, ecx` leaves `rax` in `[0, n]`.
        #[test]
        fn the_br_table_clamp_is_confined() {
            let s = run(&[
                0xb8, 0x02, 0, 0, 0, // mov eax, 2
                0x44, 0x89, 0xc1, // mov ecx, r8d
                0x39, 0xc1, // cmp ecx, eax
                0x0f, 0x42, 0xc1, // cmovb eax, ecx
            ]);
            assert_eq!(s.regs[0], Fact::Confined { bound: 2, guard: 2 });
        }

        /// D63 with an access offset compares the address and selects the address *plus the
        /// offset*: `lea r10, [rcx + 0x14]; cmp rcx, k; cmova r10, guard`. `r10` is the compared value
        /// plus 0x14, so it is confined to `[0, k + 0x15) ∪ {guard}`. A negative offset could wrap below
        /// zero, so it confines nothing.
        #[test]
        fn policy_d63_with_an_access_offset_is_confined() {
            let form = |lea_disp: u8| {
                let mut b = vec![0x4c, 0x8d, 0x51, lea_disp]; // lea r10, [rcx + disp8]
                b.extend([0x48, 0xba, 0, 0, 0, 0, 0x20, 0, 0, 0]); // mov rdx, 0x20_0000_0000
                b.extend([0x48, 0x81, 0xf9, 0xe8, 0xff, 0xff, 0x7f]); // cmp rcx, 0x7fff_ffe8
                b.extend([0x4c, 0x0f, 0x47, 0xd2]); // cmova r10, rdx
                b.extend([0x42, 0x89, 0x04, 0x17]); // mov [rdi + r10*1], eax
                b.push(0xc3);
                b
            };
            let f = findings(&form(0x14));
            assert!(f.is_empty(), "D63 with an offset was flagged: {f:?}");
            assert!(has(&findings(&form(0xf8)), "unconfined index")); // lea r10, [rcx − 8]
        }

        /// Hole 7's `lea` form: a 32-bit `lea` truncates the address, so `lea eax, [rsp + 8]` is no
        /// stack pointer.
        #[test]
        fn policy_a_narrow_lea_is_no_pointer() {
            let bytes = [
                0x8d, 0x44, 0x24, 0x08, // lea eax, [rsp + 8]
                0x48, 0x8b, 0x08, // mov rcx, [rax]
                0xc3,
            ];
            let f = findings(&bytes);
            assert!(has(&f, "unrecognized provenance"), "{f:?}");
            // ... and the `lea` itself, which computes an address, is no access.
            assert!(f.iter().all(|f| f.at != 0), "{f:?}");
        }

        /// A `jmp rel32` whose displacement a relocation patches is a tail call (Cranelift's
        /// `return_call`): it leaves the function, so it is no edge into the bytes after it, which
        /// would carry the torn-down frame into a live block.
        #[test]
        fn cfg_a_patched_jmp_is_a_tail_call() {
            let bytes = [
                0x55, // [00] push rbp
                0x48, 0x89, 0xe5, // [01] mov rbp, rsp
                0x45, 0x85, 0xc0, // [04] test r8d, r8d
                0x0f, 0x84, 0x09, 0, 0, 0, // [07] je +9  (→ 0x16)
                0x48, 0x89, 0xec, // [0d] mov rsp, rbp
                0x5d, // [10] pop rbp
                0xe9, 0, 0, 0, 0, // [11] jmp <callee>  (rel32 patched at 0x12)
                0x48, 0x89, 0xec, // [16] mov rsp, rbp
                0x5d, // [19] pop rbp
                0xc3, // [1a] ret
            ];
            let f = check(&bytes, &[call_reloc(0x12)], Entry::Guest(WIN));
            assert!(
                f.is_empty(),
                "a tail call fell through into the next block: {f:?}"
            );
        }

        /// A callee's stack effect follows Cranelift x64's `Tail` layout: six integer and eight
        /// float or vector arguments in registers, the rest in 8-byte stack slots, the area rounded
        /// up to 16. SysV-family callees pop nothing, and `PreserveAll` clobbers only its results.
        #[test]
        fn contracts_follow_the_calling_convention() {
            use cranelift_codegen::ir::{types, AbiParam, Signature};
            use cranelift_codegen::isa::CallConv;
            let sig = |cc, params: &[types::Type], returns: &[types::Type]| {
                let mut s = Signature::new(cc);
                s.params = params.iter().map(|&t| AbiParam::new(t)).collect();
                s.returns = returns.iter().map(|&t| AbiParam::new(t)).collect();
                s
            };
            let pops = |params: &[types::Type]| {
                super::super::contract(&sig(CallConv::Tail, params, &[])).and_then(|c| c.pops)
            };
            use types::{F64, I128, I32, I64, I8X16};
            assert_eq!(pops(&[I64; 6]), Some(0));
            assert_eq!(pops(&[I64; 7]), Some(16)); // one 8-byte slot, rounded up to 16
            assert_eq!(pops(&[I64; 9]), Some(32)); // three slots
            assert_eq!(pops(&[I32; 8]), Some(16)); // narrow integers take whole slots
            assert_eq!(pops(&[F64; 9]), Some(16)); // the ninth float spills
            assert_eq!(pops(&[I8X16; 9]), Some(16)); // a 16-byte vector slot
            assert_eq!(pops(&[I64, F64, I64, F64]), Some(0));
            assert_eq!(pops(&[I128]), None); // not modeled: the caller's rsp is left unknown
            let sysv = super::super::contract(&sig(CallConv::SystemV, &[I64; 9], &[I64]));
            assert_eq!(
                sysv,
                Some(Contract {
                    clobbers: SYSV_CLOBBERS,
                    pops: Some(0)
                })
            );
            let preserve = |r: &[types::Type]| {
                super::super::contract(&sig(CallConv::PreserveAll, &[I64], r)).map(|c| c.clobbers)
            };
            assert_eq!(preserve(&[]), Some(0));
            assert_eq!(preserve(&[I64]), Some(0b0101));
            assert_eq!(
                super::super::contract(&sig(CallConv::Winch, &[], &[])),
                None
            );
        }

        /// Hole 14's relocation rule: every relocation sits on a call's `rel32` or a `movabs`'s
        /// `imm64` and names an allowed callee, and every `call rel32` carries one.
        #[test]
        fn relocs_must_sit_on_a_recognized_field() {
            let call = [0xe8, 0, 0, 0, 0, 0xc3]; // call <rel32>; ret
            assert!(has(&findings(&call), "direct call without a relocation"));
            let f = check(&call, &[call_reloc(1)], Entry::Guest(WIN));
            assert!(f.is_empty(), "a relocated call was flagged: {f:?}");
            let disallowed = Reloc {
                callee: None,
                ..call_reloc(1)
            };
            let f = check(&call, &[disallowed], Entry::Guest(WIN));
            assert!(has(&f, "does not allow code to call"), "{f:?}");
            let shifted = Reloc {
                addend: 0,
                ..call_reloc(1)
            };
            let f = check(&call, &[shifted], Entry::Guest(WIN));
            assert!(has(&f, "outside a call target"), "{f:?}");
            let misplaced = call_reloc(2);
            let f = check(&call, &[misplaced], Entry::Guest(WIN));
            assert!(has(&f, "outside a call target"), "{f:?}");
            let data = [0xc3, 0, 0, 0, 0, 0, 0, 0, 0]; // ret; then 8 bytes no code reaches
            let f = check(&data, &[data_reloc(1, R::Abs8)], Entry::Guest(WIN));
            assert!(has(&f, "patches no instruction"), "{f:?}");
        }

        /// A call runs its callee's contract. Through a `movabs` of a relocated symbol, `call rax`
        /// reaches that callee: a SysV one clobbers mem_base in rdi and pops nothing, a `Tail` one
        /// pops its stack arguments, a `PreserveAll` one clobbers nothing. A call through a register
        /// that names no callee is a finding.
        #[test]
        fn calls_run_their_callees_contract() {
            let bytes = [
                0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, // [0] movabs rax, <sym>   (Abs8 at 2)
                0xff, 0xd0, // [10] call rax
            ];
            let with = |c: Contract| {
                let r = Reloc {
                    at: 2,
                    kind: R::Abs8,
                    addend: 0,
                    callee: Some(c),
                };
                run_with(&bytes, &[r])
            };
            let sysv = with(Contract {
                clobbers: SYSV_CLOBBERS,
                pops: Some(0),
            });
            assert_eq!(sysv.regs[reg::RDI as usize], Fact::Unknown);
            assert_eq!(sysv.regs[reg::RSP as usize], Fact::Stack(Some(0)));
            let tail = with(Contract {
                clobbers: SYSV_CLOBBERS,
                pops: Some(16),
            });
            assert_eq!(tail.regs[reg::RSP as usize], Fact::Stack(Some(16)));
            let preserve = with(Contract {
                clobbers: 0,
                pops: Some(0),
            });
            assert_eq!(preserve.regs[reg::RDI as usize], Fact::MemBase);
            let unknown = [0x48, 0x89, 0xf0, 0xff, 0xd0, 0xc3]; // mov rax, rsi; call rax; ret
            assert!(has(&findings(&unknown), "unidentified callee"));
        }

        /// br_table's dispatch as Cranelift emits it (`JmpTableSeq` behind the index clamp), with
        /// the table's four entries pointing at four `ret`s after it. `entries` overrides them and
        /// `n` the clamp's bound, for the rejections below.
        fn jump_table(n: u8, entries: [i32; 4]) -> Vec<u8> {
            let mut bytes = vec![
                0xb8, n, 0, 0, 0, // [00] mov eax, n
                0x41, 0x8b, 0xc8, // [05] mov ecx, r8d
                0x3b, 0xc8, // [08] cmp ecx, eax
                0x0f, 0x42, 0xc1, // [0a] cmovb eax, ecx      (eax ∈ [0, n])
                0x48, 0x8d, 0x0d, 0x09, 0, 0, 0, // [0d] lea rcx, [rip + 9]  (→ +1d)
                0x48, 0x63, 0x04, 0x81, // [14] movsxd rax, [rcx + rax*4]
                0x48, 0x03, 0xc8, // [18] add rcx, rax
                0xff, 0xe1, // [1b] jmp rcx
            ];
            for e in entries {
                bytes.extend_from_slice(&e.to_le_bytes()); // [1d] the table
            }
            bytes.extend_from_slice(&[0xc3; 4]); // [2d] four `ret`s
            bytes
        }

        /// Hole 11: a jump table's targets are read from the bytes. The clamp confines the index
        /// to `[0, 3]`, so the `jmp` reaches exactly the four blocks the table lists.
        #[test]
        fn cfg_reads_the_jump_table_from_the_bytes() {
            let bytes = jump_table(3, [0x10, 0x11, 0x12, 0x13]);
            let f = findings(&bytes);
            assert!(f.is_empty(), "a well-formed jump table was flagged: {f:?}");
            let ctx = Ctx {
                bytes: &bytes,
                relocs: &[],
                layout: WIN,
            };
            let mut fs = Vec::new();
            let cfg = Cfg::build(ctx, &mut fs);
            let targets: Vec<usize> = cfg.succs[0].iter().map(|&s| cfg.leaders[s]).collect();
            assert_eq!(targets, [0x2d, 0x2e, 0x2f, 0x30]);
        }

        /// Hole 11's rejections: a `jmp` through a register that holds no table target, one whose
        /// index the clamp doesn't bound, a table that runs past the function, an entry that leaves
        /// the function or splits an instruction, and an entry a relocation patches.
        #[test]
        fn cfg_flags_a_jump_it_cannot_read_a_table_for() {
            let not_a_table = [0x48, 0x89, 0xf8, 0xff, 0xe0]; // mov rax, rdi; jmp rax
            assert!(has(&findings(&not_a_table), "holds no jump-table target"));
            let through_memory = [0xff, 0x20]; // jmp [rax]
            assert!(has(
                &findings(&through_memory),
                "indirect jmp through memory"
            ));
            let mut unclamped = jump_table(3, [0x10, 0x11, 0x12, 0x13]);
            unclamped[10..13].copy_from_slice(&[0x8b, 0xc1, 0x90]); // mov eax, ecx; nop
            assert!(has(&findings(&unclamped), "holds no jump-table target"));
            let past_the_end = jump_table(9, [0x10, 0x11, 0x12, 0x13]); // ten entries
            assert!(has(&findings(&past_the_end), "holds no jump-table target"));
            let outside = jump_table(3, [0x10, 0x11, 0x12, 0x400]);
            assert!(has(&findings(&outside), "targets outside the function"));
            let split = jump_table(3, [0x10, 0x11, 0x12, -0x0e]); // → +0f, inside the `lea`
            assert!(has(&findings(&split), "overruns its block"));
            let bytes = jump_table(3, [0x10, 0x11, 0x12, 0x13]);
            let patched = [data_reloc(0x21, R::Abs4)];
            let f = check(&bytes, &patched, Entry::Guest(WIN));
            assert!(has(&f, "holds no jump-table target"), "{f:?}");
        }

        /// Hole 1: a call clobbers the caller-saved registers. Here both `rdi` (mem_base) and the
        /// confined `r8` are dead after it, so the access through them is flagged.
        #[test]
        fn policy_a_call_clobbers_the_caller_saved_registers() {
            let bytes = after_guard(&[
                0x4c, 0x3b, 0xc0, // cmp r8, rax
                0x4c, 0x0f, 0x43, 0xc0, // cmovae r8, rax
                0xe8, 0, 0, 0, 0, // call
                0x4a, 0x8b, 0x04, 0x07, // mov rax, [rdi + r8*1]
                0xc3,
            ]);
            assert!(has(&findings(&bytes), "unrecognized provenance"));
            // Callee-saved registers survive: mem_base parked in rbx is still mem_base.
            let s = run(&[0x48, 0x89, 0xfb, 0xe8, 0, 0, 0, 0]); // mov rbx, rdi; call
            assert_eq!(s.regs[3], Fact::MemBase);
            assert_eq!(s.regs[reg::RDI as usize], Fact::Unknown);
        }

        /// Hole 2: the `cmov`'s condition matters. `cmovb` takes the guard when the index is in
        /// bounds and keeps it when it is not — the opposite of a confinement.
        #[test]
        fn policy_an_inverted_cmov_does_not_confine() {
            let bytes = after_guard(&[
                0x4c, 0x3b, 0xc0, // cmp r8, rax
                0x4c, 0x0f, 0x42, 0xc0, // cmovb r8, rax
                0x4a, 0x8b, 0x04, 0x07, // mov rax, [rdi + r8*1]
                0xc3,
            ]);
            assert!(has(&findings(&bytes), "unconfined index"));
        }

        /// Hole 3: a flag write between the `cmp` and the `cmov` means the `cmov` tests other flags.
        #[test]
        fn policy_a_flag_write_between_cmp_and_cmov_breaks_the_confinement() {
            let bytes = after_guard(&[
                0x4c, 0x3b, 0xc0, // cmp r8, rax
                0x48, 0x83, 0xc1, 0x01, // add rcx, 1   (writes CF)
                0x4c, 0x0f, 0x43, 0xc0, // cmovae r8, rax
                0x4a, 0x8b, 0x04, 0x07, // mov rax, [rdi + r8*1]
                0xc3,
            ]);
            assert!(has(&findings(&bytes), "unconfined index"));
        }

        /// Hole 4: the compare's bound is what bounds the index. An unknown bound confines nothing,
        /// and a bound past the reservation confines to past the reservation.
        #[test]
        fn policy_the_cmp_bound_must_be_known_and_inside_the_reservation() {
            let unknown = after_guard(&[
                0x4d, 0x3b, 0xc1, // cmp r8, r9   (r9: a guest argument)
                0x4c, 0x0f, 0x43, 0xc0, // cmovae r8, rax
                0x4a, 0x8b, 0x04, 0x07, // mov rax, [rdi + r8*1]
                0xc3,
            ]);
            assert!(has(&findings(&unknown), "unconfined index"));
            let huge = after_guard(&[
                0x49, 0xb9, 0, 0, 0, 0, 0, 0, 0x10, 0, // mov r9, 2^52
                0x4d, 0x3b, 0xc1, // cmp r8, r9
                0x4c, 0x0f, 0x43, 0xc0, // cmovae r8, rax
                0x4a, 0x8b, 0x04, 0x07, // mov rax, [rdi + r8*1]
                0xc3,
            ]);
            assert!(has(&findings(&huge), "past the reservation"));
        }

        /// Spill-slot tracking carries `mem_base` across a stack round-trip: spill rdi to `[rsp]`,
        /// confine r8, reload rdi from `[rsp]`, then access `[mem_base + confined]` — clean.
        #[test]
        fn policy_tracks_membase_through_a_spill_slot() {
            let mut bytes = vec![
                0x48, 0x83, 0xec, 0x10, // sub rsp, 16
                0x48, 0x89, 0x3c, 0x24, // mov [rsp], rdi  (spill mem_base)
            ];
            bytes.extend(GUARD);
            bytes.extend([
                0x4c, 0x3b, 0xc0, // cmp r8, rax
                0x4c, 0x0f, 0x43, 0xc0, // cmovae r8, rax
                0x48, 0x8b, 0x3c, 0x24, // mov rdi, [rsp]  (reload mem_base)
                0x4a, 0x8b, 0x04, 0x07, // mov rax, [rdi + r8*1]
                0x48, 0x83, 0xc4, 0x10, // add rsp, 16
                0xc3, // ret
            ]);
            let f = findings(&bytes);
            assert!(
                f.is_empty(),
                "spilled-then-reloaded mem_base was flagged: {f:?}"
            );
        }

        /// Soundness of slot tracking: a store that overwrites the slot must drop the stale fact, so
        /// the reload reads `Unknown` and the access is flagged — not silently admitted as the window.
        #[test]
        fn policy_slot_overwrite_drops_the_stale_fact() {
            let bytes = [
                0x48, 0x89, 0x3c, 0x24, // mov [rsp], rdi            (slot := mem_base)
                0x48, 0xc7, 0x04, 0x24, 0x00, 0x00, 0x00,
                0x00, // mov qword [rsp], 0  (overwrites it)
                0x48, 0x8b, 0x3c, 0x24, // mov rdi, [rsp]            (reload ⇒ Unknown)
                0x4a, 0x8b, 0x04, 0x07, // mov rax, [rdi + r8*1]
                0xc3, // ret
            ];
            assert!(has(&findings(&bytes), "unrecognized provenance"));
        }

        /// Hole 5 — the CVE-2021-32629 class: a slot holds a 64-bit fact only if a 64-bit store wrote
        /// it, and only a 64-bit load restores one.
        #[test]
        fn policy_a_narrow_spill_or_reload_restores_nothing() {
            let narrow_store = [
                0x48, 0x83, 0xec, 0x10, // sub rsp, 16
                0x89, 0x3c, 0x24, // mov [rsp], edi   (the low half of mem_base)
                0x48, 0x8b, 0x3c, 0x24, // mov rdi, [rsp]   (a 64-bit reload)
                0x48, 0x8b, 0x07, // mov rax, [rdi]
                0xc3,
            ];
            assert!(has(&findings(&narrow_store), "unrecognized provenance"));
            let narrow_load = [
                0x48, 0x83, 0xec, 0x10, // sub rsp, 16
                0x48, 0x89, 0x3c, 0x24, // mov [rsp], rdi
                0x8b, 0x3c, 0x24, // mov edi, [rsp]   (a 32-bit reload: zero-extends)
                0x48, 0x8b, 0x07, // mov rax, [rdi]
                0xc3,
            ];
            assert!(has(&findings(&narrow_load), "unrecognized provenance"));
        }

        /// Hole 6: slots are keyed by the offset from the entry rsp, so a `push` that moves rsp does
        /// not make `[rsp]` read the old slot, and a store that overlaps half a slot forgets it.
        #[test]
        fn policy_slots_follow_rsp_and_forget_overlapping_stores() {
            let push = [
                0x48, 0x83, 0xec, 0x10, // sub rsp, 16
                0x48, 0x89, 0x3c, 0x24, // mov [rsp], rdi   (slot −16 := mem_base)
                0x50, // push rax          (rsp → −24)
                0x48, 0x8b, 0x0c, 0x24, // mov rcx, [rsp]   (slot −24: rax, unknown)
                0x48, 0x8b, 0x01, // mov rax, [rcx]
                0xc3,
            ];
            assert!(has(&findings(&push), "unrecognized provenance"));
            let overlap = [
                0x48, 0x83, 0xec, 0x10, // sub rsp, 16
                0x48, 0x89, 0x3c, 0x24, // mov [rsp], rdi            (slot −16 := mem_base)
                0xc7, 0x44, 0x24, 0x04, 0, 0, 0, 0, // mov dword [rsp + 4], 0
                0x48, 0x8b, 0x3c, 0x24, // mov rdi, [rsp]
                0x48, 0x8b, 0x07, // mov rax, [rdi]
                0xc3,
            ];
            assert!(has(&findings(&overlap), "unrecognized provenance"));
            // `push rdi; pop rcx` moves mem_base through the stack.
            let s = run(&[0x57, 0x59]);
            assert_eq!(s.regs[1], Fact::MemBase);
            assert_eq!(s.regs[reg::RSP as usize], Fact::Stack(Some(0)));
        }

        /// Hole 7: `mov ax, 5` keeps rax's upper 48 bits, so it is no constant.
        #[test]
        fn policy_a_16_bit_immediate_is_not_a_constant() {
            let bytes = [
                0x48, 0xc7, 0xc0, 0xff, 0xff, 0xff, 0xff, // mov rax, -1
                0x66, 0xb8, 0x05, 0x00, // mov ax, 5   (rax = 0xffff_ffff_ffff_0005)
                0x48, 0x8b, 0x04, 0x07, // mov rax, [rdi + rax*1]
                0xc3,
            ];
            assert!(has(&findings(&bytes), "unconfined index"));
        }

        /// Holes 8–9: forms whose address is not the `[base + index*scale + disp]` the policy checks,
        /// and far transfers, are forbidden outright.
        #[test]
        fn decoder_forbids_forms_the_policy_cannot_check() {
            let cases: [(&[u8], &str); 5] = [
                (&[0x65, 0x48, 0x8b, 0x04, 0x07, 0xc3], "fs/gs"), // mov rax, gs:[rdi + rax]
                (&[0x67, 0x48, 0x8b, 0x04, 0x07, 0xc3], "address-size"), // mov rax, [edi + eax]
                (&[0xd7, 0xc3], "xlat"),                          // xlat: [rbx + al]
                (&[0x48, 0x0f, 0xab, 0x07, 0xc3], "bit offset"),  // bts [rdi], rax
                (&[0xff, 0x1f, 0xc3], "far call"),                // call far [rdi]
            ];
            for (bytes, why) in cases {
                assert!(
                    has(&findings(bytes), why),
                    "{bytes:02x?} not forbidden as {why}"
                );
            }
        }

        /// Hole 13: control may not run past the function's last byte.
        #[test]
        fn policy_flags_falling_off_the_end() {
            assert!(has(&findings(&[0x48, 0x8b, 0x07]), "off the end")); // mov rax, [rdi]
        }

        /// Hole 12 — the split: `verify` trusts no proposal. A state claiming `rdi` still holds
        /// mem_base after `xor rdi, rdi` is rejected at the edge, whatever produced it; and an entry
        /// state the contract doesn't imply is rejected at the entry.
        #[test]
        fn verify_rejects_a_proposal_that_is_not_inductive() {
            let bytes = [
                0x48, 0x31, 0xff, // xor rdi, rdi
                0xeb, 0x00, // jmp +0   (to the next block)
                0x48, 0x8b, 0x07, // mov rax, [rdi]
                0xc3, // ret
            ];
            let ctx = Ctx {
                bytes: &bytes,
                relocs: &[],
                layout: WIN,
            };
            let mut fs = Vec::new();
            let cfg = Cfg::build(ctx, &mut fs);
            assert!(fs.is_empty());
            let honest = propose(&cfg, ctx);
            let mut f = Vec::new();
            verify(&cfg, &honest, ctx, &mut f);
            assert!(has(&f, "unrecognized provenance"), "{f:?}");

            let mut lie = honest.clone();
            lie[1] = Some(State::entry());
            let mut f = Vec::new();
            verify(&cfg, &lie, ctx, &mut f);
            assert!(has(&f, "does not hold on the edge"), "{f:?}");

            let mut bad_entry = honest;
            let mut s = State::entry();
            s.regs[0] = Fact::MemBase;
            bad_entry[0] = Some(s);
            let mut f = Vec::new();
            verify(&cfg, &bad_entry, ctx, &mut f);
            assert!(has(&f, "entry contract"), "{f:?}");
        }

        /// An elided constant-displacement window access `[mem_base + 40000]` is re-proven against
        /// the module reservation from the layout hint (#2603): clean when it fits, flagged when the
        /// displacement reaches past the reservation.
        #[test]
        fn policy_reproves_elided_access_against_reservation() {
            let bytes = [
                0x48, 0x8b, 0x87, 0x40, 0x9c, 0x00, 0x00, // mov rax, [rdi + 0x9C40 (=40000)]
                0xc3, // ret
            ];
            // A large reservation easily holds offset 40000 → no finding.
            let big = Some(WindowLayout {
                mask: (1u64 << 37) - 1,
                guard_offset: 1u64 << 37,
            });
            assert!(check(&bytes, &[], Entry::Guest(big)).is_empty());
            // A 32 KiB reservation does not → the access reaches past it.
            let small = Some(WindowLayout {
                mask: (1u64 << 15) - 1,
                guard_offset: 1u64 << 15,
            });
            let f = check(&bytes, &[], Entry::Guest(small));
            assert!(has(&f, "past the reservation"), "{f:?}");
        }
    }
}
