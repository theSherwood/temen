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

/// Validate one emitted function's `bytes` against its basic-block hints from
/// `CompiledCode`: `bb_starts` (block start offsets) and `bb_edges` (block-to-block control-flow
/// edges, each `(from_start, to_start)`), with the module's `layout` (`None` if memoryless). Shadow
/// mode: every problem is logged and counted, never returned.
pub(crate) fn shadow_check(
    bytes: &[u8],
    bb_starts: &[u32],
    bb_edges: &[(u32, u32)],
    layout: Option<WindowLayout>,
) {
    #[cfg(target_arch = "x86_64")]
    for f in x86::check(bytes, bb_starts, bb_edges, layout) {
        // Count every disagreement (a fuzz run asserts the total), but cap the logging so a bad
        // decode on every module can't swamp a fuzzer's stderr.
        let n = DISAGREEMENTS.fetch_add(1, Ordering::Relaxed);
        if n < 64 {
            eprintln!("jit-validate(x86-64): {} at +0x{:x}", f.why, f.at);
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (bytes, bb_starts, bb_edges, layout);
}

// --- test-only capture of real compiled code, cross-checked against iced-x86 ---
//
// The whole subsystem exists only for the x86-64 decoder cross-check (`x86::tests`), and its one
// caller in `lib.rs` is gated to match, so it is `all(test, target_arch = "x86_64")` — on another
// target (the aarch64 clippy lane) it would be dead code under `-D warnings`.

/// One captured function's machine code, its basic-block hints (`bb_starts`, `bb_edges`), and the
/// module's window layout.
#[cfg(all(test, target_arch = "x86_64"))]
type CapturedFn = (Vec<u8>, Vec<u32>, Vec<(u32, u32)>, Option<WindowLayout>);

#[cfg(all(test, target_arch = "x86_64"))]
thread_local! {
    /// When armed, each function the compile defines appends its capture here, so a test can decode
    /// real Cranelift output and compare our decoder to iced-x86 / run the policy.
    static CAPTURE: std::cell::RefCell<Option<Vec<CapturedFn>>> =
        const { std::cell::RefCell::new(None) };
}

/// Record one defined function's code + block hints + layout, if a test armed the capture.
#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) fn capture_code(
    code: &[u8],
    bb_starts: &[u32],
    bb_edges: &[(u32, u32)],
    layout: Option<WindowLayout>,
) {
    CAPTURE.with(|c| {
        if let Some(v) = c.borrow_mut().as_mut() {
            v.push((code.to_vec(), bb_starts.to_vec(), bb_edges.to_vec(), layout));
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

    use super::WindowLayout;

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
        pub const RBP: u8 = 5; // frame pointer
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
        /// Confined by the D63 sequence (`cmp v, bound; cmovae v', guard`): either `< bound` or the
        /// guard offset, so safe as a window byte index for a displacement within the trailing-guard
        /// slack. Carries the guard constant the `cmov` redirected to, cross-checked against the
        /// module's `guard_offset` when the layout hint is present (#2603).
        Confined { guard: u64 },
        /// Proven in `[0, ub)` (e.g. `and r, (2^k − 1)`).
        Bounded(u64),
        /// A pointer into the stack frame, computed by `lea r, [rsp/rbp + disp]`.
        StackPtr,
        /// A pointer into the function's own rodata / jump tables, computed by `lea r, [rip + disp]`.
        RipRel,
        /// No information.
        Unknown,
    }

    impl Fact {
        /// The fact true on both incoming paths.
        fn meet(self, other: Fact) -> Fact {
            match (self, other) {
                (a, b) if a == b => a,
                (Fact::Bounded(a), Fact::Bounded(b)) => Fact::Bounded(a.max(b)),
                _ => Fact::Unknown,
            }
        }
    }

    /// The abstract state at a program point: a fact per GP register. (Spill-slot tracking — a
    /// context param reloaded from `[rbp-k]` — is a tracked follow-up; until then a reload reads
    /// `Unknown`, which is sound and, for the small corpus here where the params stay in registers,
    /// precise.)
    #[derive(Clone, Copy, PartialEq, Eq)]
    struct State {
        regs: [Fact; 16],
    }

    impl State {
        /// The entry ABI: the four context params in their argument registers, everything else
        /// unknown. rsp/rbp are the stack and are recognized structurally, not tracked as facts.
        fn entry() -> State {
            let mut regs = [Fact::Unknown; 16];
            regs[reg::RDI as usize] = Fact::MemBase;
            regs[reg::RSI as usize] = Fact::FnTableBase;
            regs[reg::RDX as usize] = Fact::Vmctx;
            regs[reg::RCX as usize] = Fact::StackLimit;
            State { regs }
        }
        fn meet(&self, other: &State) -> State {
            let mut regs = self.regs;
            for (r, &o) in regs.iter_mut().zip(other.regs.iter()) {
                *r = r.meet(o);
            }
            State { regs }
        }
        /// Loop-header widening (#2535): a `Bounded` carried into a header could grow each iteration,
        /// so drop it to `Unknown`. The context-param and `Confined` facts are loop-invariant and
        /// survive, so nearly every loop still settles in one round.
        fn widen(&mut self) {
            for f in &mut self.regs {
                if matches!(f, Fact::Bounded(_)) {
                    *f = Fact::Unknown;
                }
            }
        }
    }

    /// Validate one emitted function's `bytes` against its basic-block hints, returning every policy
    /// finding. Pure, so tests assert on the findings directly.
    ///
    /// Three layers (#2481 slice 1 + #2535):
    /// 1. **Decode + allowlist**, per machine basic block, flagging any instruction outside the
    ///    allowlist, any undecodable encoding, any block overrun, and any direct branch out of code.
    ///    The bytes between a non-fall-through terminator and the next block are a constant island or
    ///    jump table Cranelift placed there (data, not code), so the walk stops at the terminator.
    /// 2. **CFG recovery** ([`recover_leaders`]): the machine basic blocks are recovered by decoding
    ///    forward from the entry and every direct-branch target. `bb_starts`/`bb_edges` are *hints*,
    ///    not the machine CFG — they are Cranelift's CLIF-block CFG, so machine-lowered branches (the
    ///    stack-limit check, fuel/epoch polls, a `select`'s local `jcc`) target offsets no hint lists.
    ///    The one edge that cannot be decoded — an indirect `jmp`'s jump-table targets — is taken
    ///    from `bb_edges` (this is what the hint saves: the expensive indirect-jump recovery, #2481).
    /// 3. **Memory-access confinement** (#2535): an abstract interpretation over the recovered CFG
    ///    tracks which register holds `mem_base` and which holds a D63-confined index across blocks
    ///    and loops, then checks every memory access goes to a known region (the window through a
    ///    confined/bounded index, the stack, vmctx, the fn-table, or rip-relative rodata).
    pub(crate) fn check(
        bytes: &[u8],
        bb_starts: &[u32],
        bb_edges: &[(u32, u32)],
        layout: Option<WindowLayout>,
    ) -> Vec<Finding> {
        let mut findings = Vec::new();
        if bytes.is_empty() {
            return findings;
        }

        // Layers 1–2: recover the machine basic blocks, decode each, and collect direct successors.
        let leaders = recover_leaders(bytes, bb_starts, bb_edges);
        let nb = leaders.len();
        let index_of = |o: usize| leaders.binary_search(&o).ok();
        let limit_of = |i: usize| leaders.get(i + 1).copied().unwrap_or(bytes.len());

        let mut blocks: Vec<Vec<(usize, Decoded)>> = Vec::with_capacity(nb);
        let mut succs: Vec<Vec<usize>> = vec![Vec::new(); nb];
        for i in 0..nb {
            let (insns, direct) = decode_block(bytes, leaders[i], limit_of(i), &mut findings);
            for s in direct {
                if let Some(si) = index_of(s) {
                    if !succs[i].contains(&si) {
                        succs[i].push(si);
                    }
                }
            }
            blocks.push(insns);
        }
        // The one edge the decoder cannot read — an indirect `jmp`'s jump-table targets — comes from
        // the `bb_edges` hint: attach each edge's target to the machine block containing its source.
        for &(from, to) in bb_edges {
            if let (Some(ti), Some(fi)) = (index_of(to as usize), block_of(from as usize, &leaders))
            {
                if !succs[fi].contains(&ti) {
                    succs[fi].push(ti);
                }
            }
        }

        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); nb];
        for (fi, ss) in succs.iter().enumerate() {
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

        // Layer 3: forward dataflow to a fixpoint. `in[b]` is the meet of its computed predecessors'
        // out-states (the entry keeps its seed); `out[b]` is the block's transfer of `in[b]`. Meet
        // only moves up the lattice and headers widen `Bounded`, so this terminates; in practice the
        // loop-carried facts are invariant and it settles in one or two rounds.
        let entry = index_of(0).unwrap_or(0);
        let mut in_st: Vec<Option<State>> = vec![None; nb];
        let mut out_st: Vec<Option<State>> = vec![None; nb];
        let cap = nb * 4 + 8;
        for _ in 0..cap {
            let mut changed = false;
            for i in 0..nb {
                let mut cur = (i == entry).then(State::entry);
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
                    let o = transfer_block(&blocks[i], s, None, layout);
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

        // Final pass: replay each reachable block's transfer, collecting the memory-access findings.
        for i in 0..nb {
            if let Some(s) = in_st[i] {
                transfer_block(&blocks[i], s, Some(&mut findings), layout);
            }
        }
        findings
    }

    /// Recover the machine basic-block leaders: the entry, the hinted `bb_starts`/`bb_edge` targets,
    /// and — the part Cranelift's CLIF-block hints miss — every direct-branch target and the
    /// fall-through of every conditional branch. Decoding from the current leader set discovers more
    /// targets; iterate to a fixpoint (the leader set only grows, bounded by the code size, so it
    /// terminates). The walk stops at a non-fall-through terminator, so inter-block island/table
    /// data is never decoded.
    fn recover_leaders(bytes: &[u8], bb_starts: &[u32], bb_edges: &[(u32, u32)]) -> Vec<usize> {
        let len = bytes.len();
        let mut leaders: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        leaders.insert(0);
        for &s in bb_starts {
            if (s as usize) < len {
                leaders.insert(s as usize);
            }
        }
        for &(_, to) in bb_edges {
            if (to as usize) < len {
                leaders.insert(to as usize);
            }
        }
        loop {
            let before = leaders.len();
            let cur: Vec<usize> = leaders.iter().copied().collect();
            for (i, &ld) in cur.iter().enumerate() {
                let limit = cur.get(i + 1).copied().unwrap_or(len);
                let mut pos = ld;
                while pos < limit {
                    let d = decode(bytes, pos);
                    if d.len == 0 {
                        break; // undecodable (flagged in decode_block)
                    }
                    let next = pos + d.len;
                    match d.kind {
                        Kind::DirectJmp { target } => {
                            if target < len {
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
                        Kind::Ret | Kind::Ud2 | Kind::IndirectJmp => break,
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

    /// The index of the machine block containing byte offset `off` (the largest leader `<= off`).
    fn block_of(off: usize, leaders: &[usize]) -> Option<usize> {
        leaders.partition_point(|&l| l <= off).checked_sub(1)
    }

    /// Decode block `[block, limit)` into `(offset, Decoded)` pairs (layer 1), stopping at the
    /// block's non-fall-through terminator, and return the block's direct successor offsets (the
    /// branch target(s) and, for a conditional branch or a fall-through into the next block, the
    /// next offset). Pushes layer-1 findings: undecodable encodings, forbidden instructions, block
    /// overruns, and direct branches out of the code.
    fn decode_block(
        bytes: &[u8],
        block: usize,
        limit: usize,
        findings: &mut Vec<Finding>,
    ) -> (Vec<(usize, Decoded)>, Vec<usize>) {
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
                    if target >= bytes.len() =>
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
                Kind::DirectJmp { target } => return (insns, vec![target]),
                Kind::DirectJcc { target } => return (insns, vec![target, next]),
                // `ret`/`ud2` leave the function; an indirect `jmp`'s successors come from `bb_edges`.
                Kind::Ret | Kind::Ud2 | Kind::IndirectJmp => return (insns, Vec::new()),
                _ => {}
            }
            pos = next;
        }
        (insns, vec![limit]) // fell through to the next leader
    }

    /// Run block `insns` from `state`, returning the out-state. With `findings = Some`, also apply
    /// the memory-access policy (layer 3) at each access. The guard→cmp→cmov recognition that
    /// produces a [`Fact::Confined`] is block-local (`confinable`): the compared value, tracked
    /// through a `mov` copy, becomes `Confined` at a `cmov` whose source is a constant (the guard
    /// offset). The resulting fact flows across blocks through the lattice, so a guard Cranelift
    /// hoists out of a loop is still seen at the access.
    fn transfer_block(
        insns: &[(usize, Decoded)],
        mut state: State,
        mut findings: Option<&mut Vec<Finding>>,
        layout: Option<WindowLayout>,
    ) -> State {
        // Registers that currently hold the value most recently compared (`cmp`), tracked through
        // `mov` copies so a `cmovcc` on a copy is still recognized. Block-local.
        let mut confinable: u16 = 0;
        for &(pos, d) in insns {
            if let Some(f) = findings.as_deref_mut() {
                mem_policy(pos, &d, &state, layout, f);
            }
            let w = writes_of(d.eff, &d.operands);
            match d.eff {
                Eff::Cmp { reg, .. } => {
                    confinable = reg.map_or(0, bit); // the compared value lives in `reg`
                    continue; // writes no register
                }
                Eff::MovReg { dst, src } => {
                    state.regs[dst as usize & 15] = state.regs[src as usize & 15];
                    // The compared value, if copied, stays confinable in its new home.
                    if confinable & bit(src) != 0 {
                        confinable |= bit(dst);
                    } else {
                        confinable &= !bit(dst);
                    }
                    continue;
                }
                Eff::MovImm { dst, imm } => {
                    state.regs[dst as usize & 15] = Fact::Const(imm);
                }
                Eff::AluImm { dst, alu, imm } => {
                    state.regs[dst as usize & 15] =
                        alu_fact(state.regs[dst as usize & 15], alu, imm);
                }
                Eff::Cmov { dst } => {
                    // `cmovcc dst, src`: dst becomes either its old value or `src`. It is the D63
                    // confined value iff dst held the compared index and src is the guard constant.
                    let src_guard = match d.operands.rm_reg.map(|s| state.regs[s as usize & 15]) {
                        Some(Fact::Const(g)) => Some(g as u64),
                        _ => None,
                    };
                    let was_compared = confinable & bit(dst) != 0;
                    state.regs[dst as usize & 15] = match src_guard {
                        Some(guard) if was_compared => Fact::Confined { guard },
                        _ => Fact::Unknown,
                    };
                }
                Eff::Lea { dst } => {
                    // `lea` computes an address. A rip-relative one points into the function's own
                    // rodata/jump tables; an rsp/rbp-relative one into the stack frame. Both are known
                    // regions a later access may legitimately go through.
                    state.regs[dst as usize & 15] = match d.operands.mem {
                        Some(m) if m.rip => Fact::RipRel,
                        Some(Mem {
                            base: Some(reg::RSP | reg::RBP),
                            ..
                        }) => Fact::StackPtr,
                        _ => Fact::Unknown,
                    };
                }
                // Loads and the opaque GP writes clobber their destinations (the value is not a
                // tracked fact); the clobber mask `w` covers them below.
                Eff::LoadReg { .. } | Eff::WrReg | Eff::WrRm | Eff::Wr(_) | Eff::NoGp => {}
            }
            // Clobber every GP register this instruction writes that the semantic above didn't set,
            // and drop them from the confinable set (their value changed).
            for r in 0..16u8 {
                if w & bit(r) != 0 {
                    if !matches!(
                        d.eff,
                        Eff::MovImm { dst, .. }
                            | Eff::AluImm { dst, .. }
                            | Eff::Cmov { dst }
                            | Eff::Lea { dst }
                            if dst == r
                    ) {
                        state.regs[r as usize] = Fact::Unknown;
                    }
                    confinable &= !bit(r);
                }
            }
        }
        state
    }

    /// The GP register write mask of an instruction, from its effect and operands. A sound superset
    /// of what the instruction actually writes (clearing extra facts is safe; missing a real write
    /// is not) — cross-checked against iced-x86 in the tests.
    fn writes_of(eff: Eff, ops: &Operands) -> u16 {
        match eff {
            Eff::NoGp | Eff::Cmp { .. } => 0,
            Eff::MovImm { dst, .. }
            | Eff::MovReg { dst, .. }
            | Eff::LoadReg { dst }
            | Eff::Lea { dst }
            | Eff::AluImm { dst, .. }
            | Eff::Cmov { dst } => bit(dst),
            Eff::WrReg => ops.reg.map_or(0, bit),
            Eff::WrRm => ops.rm_reg.map_or(0, bit),
            Eff::Wr(m) => m,
        }
    }

    /// `and`/`add`/`sub r, imm` on a tracked value. `and` with a low-bits mask bounds the result;
    /// otherwise the value stops being tracked (bound/const arithmetic beyond this is deferred).
    fn alu_fact(prev: Fact, alu: Alu, imm: i64) -> Fact {
        match (alu, prev) {
            (Alu::And, _) if imm >= 0 && (imm as u64).wrapping_add(1).is_power_of_two() => {
                Fact::Bounded(imm as u64 + 1) // r & (2^k - 1) ∈ [0, 2^k)
            }
            (Alu::Add, Fact::Const(c)) => Fact::Const(c.wrapping_add(imm)),
            (Alu::Sub, Fact::Const(c)) => Fact::Const(c.wrapping_sub(imm)),
            _ => Fact::Unknown,
        }
    }

    /// The memory-access confinement policy (#2535/#2603): classify one access's base and index by
    /// their abstract facts and flag an access that does not land in a known region. For a window
    /// access, re-prove the physical offset from `mem_base` stays within the reservation
    /// `[0, guard_offset + page)` — everything there is mapped or `PROT_NONE` guard, so it can fault
    /// but never escape — using the module `layout` hint (when present) for the elided/bounded cases.
    fn mem_policy(
        pos: usize,
        d: &Decoded,
        state: &State,
        layout: Option<WindowLayout>,
        findings: &mut Vec<Finding>,
    ) {
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
        let mut flag = |why: String| findings.push(Finding { at: pos, why });

        // Stack: rsp/rbp-relative (frame-bound refinement is deferred).
        if matches!(mem.base, Some(reg::RSP | reg::RBP)) {
            return;
        }
        // Window: base (or index) is `mem_base`; the other operand is the index/displacement.
        let is_membase = |f: Option<Fact>| matches!(f, Some(Fact::MemBase));
        if is_membase(base_f) || is_membase(index_f) {
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
                Some(Fact::Confined { guard }) => (Some(guard), Some(guard)),
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
            // module's real guard offset (when the layout hint is present — #2603).
            if let Some(guard) = confined_guard {
                if mem.scale != 1 {
                    flag(format!(
                        "confined window index with scale {} (expected a byte index, scale 1)",
                        mem.scale
                    ));
                    return;
                }
                if let Some(l) = layout.filter(|l| l.guard_offset != guard) {
                    flag(format!(
                        "confined index redirects to +0x{guard:x}, not the module guard offset \
                         +0x{:x}",
                        l.guard_offset
                    ));
                    return;
                }
            }
            // Re-prove: the largest physical offset the access can reach stays within the window
            // reservation. The reservation top is `guard_offset + page`; a confined index supplies
            // its own guard (cross-checked above), the elided/bounded cases need the layout hint —
            // without it (a memoryless module) there is no reservation to prove against, so defer.
            let reach = match (confined_guard, layout) {
                (Some(guard), _) => guard.saturating_add(page),
                (None, Some(l)) => l.guard_offset.saturating_add(page),
                (None, None) => return,
            };
            let top = imax.saturating_add(disp).saturating_add(MAX_ACCESS_WIDTH);
            if top > reach {
                flag(format!(
                    "window access reaches +0x{top:x}, past the reservation +0x{reach:x}"
                ));
            }
            return;
        }
        // Other known bases: the fn-table (call.dyn dispatch; its bounds are a later slice), vmctx
        // (the trap-out cell and per-instance fields), a lea'd stack pointer, and a lea'd pointer
        // into the function's own rodata / jump tables (the br_table dispatch's table read; its
        // index-bounds re-proof is the control-flow-integrity slice).
        if matches!(
            base_f,
            Some(Fact::FnTableBase | Fact::Vmctx | Fact::StackPtr | Fact::RipRel)
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

    impl Kind {
        /// Whether control can continue to the next instruction after this one. `false` for the
        /// terminators that end a basic block without fall-through (`jmp`, `ret`, `ud2`, indirect
        /// `jmp`) — after which the next bytes may be a constant island or jump table, not code.
        /// Used by the test-only [`walk`] (the policy decodes blocks via [`decode_block`], which
        /// inspects `kind` directly).
        #[cfg(test)]
        fn falls_through(self) -> bool {
            !matches!(
                self,
                Kind::DirectJmp { .. } | Kind::IndirectJmp | Kind::Ret | Kind::Ud2
            )
        }
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
        /// Writes no GP register the state tracks: flags-only (`cmp`/`test`), control flow, SSE/vector
        /// ops (their operands are XMM, not GP), fences, nops, pushes/pops of the frame.
        NoGp,
        /// `mov dst, imm` — `dst := Const(imm)`.
        MovImm { dst: u8, imm: i64 },
        /// `mov dst, src` between GP registers — `dst := fact(src)`.
        MovReg { dst: u8, src: u8 },
        /// A GP load `mov dst, [mem]` (mem in `operands.mem`) — `dst := loaded value`.
        LoadReg { dst: u8 },
        /// `lea dst, [mem]` — computes an address into `dst` (does **not** dereference).
        Lea { dst: u8 },
        /// `add`/`sub`/`and dst, imm`, `dst` the r/m register — `dst := alu(fact(dst), imm)`.
        AluImm { dst: u8, alu: Alu, imm: i64 },
        /// The modeled compare before a confine `cmov`: `cmp reg, bound` (bound an immediate, or
        /// `None` for a register/memory right-hand side). Writes only flags.
        Cmp { reg: Option<u8>, bound: Option<u64> },
        /// `cmovcc dst, src` — `dst := Confined` when `src` is the guard const and `dst` was just
        /// compared; else `dst := Unknown`. (Conditional, so always a write of `dst`.)
        Cmov { dst: u8 },
        /// Writes `operands.reg` (a GP op whose destination is its reg field: `op reg, r/m`).
        WrReg,
        /// Writes `operands.rm_reg` when the r/m is a register (`op r/m, reg`); the memory form writes
        /// memory, no GP register.
        WrRm,
        /// Writes a fixed implicit GP mask (e.g. `mul`/`div` → rax,rdx; `cpuid`; accumulator forms).
        Wr(u16),
    }

    /// One decoded instruction: its length, its control-flow classification, its operands, and its
    /// data-flow effect ([`Eff`], consumed by the #2535 abstract interpreter).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct Decoded {
        pub len: usize,
        pub kind: Kind,
        pub operands: Operands,
        pub eff: Eff,
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
    /// the operand width (only 64-bit operations preserve a pointer/bound fact — a narrower write
    /// truncates or zero-extends, so it clobbers), and the immediate.
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
        /// `cmp reg, r/m` — the compared value is `reg`.
        CmpRegRm,
        /// `cmp r/m, reg` — the compared value is the r/m register.
        CmpRmReg,
        /// `0x80/0x81/0x83 /digit` — the ModRM.reg field (`digit`) picks add/or/.../cmp with `imm`.
        AluImmGroup(u8),
        /// `0xF6/0xF7 /digit` — test (writes flags), not/neg (writes r/m), or mul/div (writes rax,rdx).
        Group3(u8),
        /// `0xFF /digit` — inc/dec (writes r/m), or call/jmp/push (no tracked GP write).
        Group5(u8),
        /// `cmovcc reg, r/m`.
        Cmov,
        /// `xchg reg, r/m` — writes both.
        XchgRegRm,
    }

    /// A one-register GP write mask bit.
    fn bit(r: u8) -> u16 {
        1u16 << (r & 15)
    }

    /// Turn an [`EffIntent`] into the final [`Eff`] now that the operands, operand width (`w64`), and
    /// immediate are known. The guiding rule for soundness: a fact (`MemBase`, `Confined`, …) is a
    /// property of a full 64-bit register value, so only a **64-bit** operation may carry one; a
    /// narrower write truncates/zero-extends and therefore *clobbers* the destination. (`MovImm` is
    /// the exception — a narrower `mov` to a const is still a known, if zero-extended, value.)
    fn resolve_eff(intent: EffIntent, ops: &Operands, w64: bool, immv: i64) -> Eff {
        use EffIntent as I;
        let (reg, rm, mem) = (ops.reg, ops.rm_reg, ops.mem.is_some());
        match intent {
            I::NoGp => Eff::NoGp,
            I::WrReg => Eff::WrReg,
            I::WrRm => Eff::WrRm,
            I::Wr(m) => Eff::Wr(m),
            I::Lea => reg.map_or(Eff::NoGp, |dst| Eff::Lea { dst }),
            I::MovToReg => match (reg, rm, mem) {
                (Some(dst), _, true) => Eff::LoadReg { dst },
                (Some(dst), Some(src), false) if w64 => Eff::MovReg { dst, src },
                (Some(_), _, false) => Eff::WrReg, // narrower reg-reg: truncates, so clobber
                _ => Eff::NoGp,
            },
            I::MovToRm => match (reg, rm, mem) {
                (_, _, true) => Eff::NoGp, // store to memory (the access is checked separately)
                (Some(src), Some(dst), false) if w64 => Eff::MovReg { dst, src },
                (_, Some(_), false) => Eff::WrRm,
                _ => Eff::NoGp,
            },
            I::MovImmToRm => match (rm, mem) {
                (_, true) => Eff::NoGp, // store of an immediate
                (Some(dst), false) => Eff::MovImm {
                    dst,
                    imm: if w64 { immv } else { immv & 0xFFFF_FFFF },
                },
                _ => Eff::NoGp,
            },
            I::CmpRegRm if w64 => Eff::Cmp { reg, bound: None },
            I::CmpRmReg if w64 => Eff::Cmp {
                reg: rm,
                bound: None,
            },
            I::CmpRegRm | I::CmpRmReg => Eff::NoGp,
            I::AluImmGroup(digit) => {
                if digit == 7 {
                    // cmp r/m, imm
                    return if w64 {
                        Eff::Cmp {
                            reg: rm,
                            bound: Some(immv as u64),
                        }
                    } else {
                        Eff::NoGp
                    };
                }
                let alu = match digit {
                    0 => Some(Alu::Add),
                    4 => Some(Alu::And),
                    5 => Some(Alu::Sub),
                    _ => None, // or/adc/sbb/xor — not modeled for bounds
                };
                match (mem, w64, alu, rm) {
                    (true, ..) => Eff::NoGp, // RMW on memory (access checked); no GP write
                    (false, true, Some(alu), Some(dst)) => Eff::AluImm {
                        dst,
                        alu,
                        imm: immv,
                    },
                    (false, ..) => Eff::WrRm, // narrower, or or/adc/sbb/xor: clobber r/m
                }
            }
            I::Group3(digit) => match digit {
                0 | 1 => Eff::NoGp,                // test r/m, imm — flags only
                4..=7 => Eff::Wr(bit(0) | bit(2)), // mul/imul/div/idiv → rax, rdx
                _ if mem => Eff::NoGp,             // not/neg [mem]
                _ => Eff::WrRm,                    // not/neg r/m
            },
            I::Group5(digit) => match digit {
                0 | 1 if !mem => Eff::WrRm, // inc/dec r/m register
                _ => Eff::NoGp,             // inc/dec [mem]; call/jmp/push (kind already set)
            },
            I::Cmov => match reg {
                Some(dst) if w64 => Eff::Cmov { dst },
                Some(_) => Eff::WrReg, // narrower cmov: clobber
                None => Eff::NoGp,
            },
            I::XchgRegRm => Eff::Wr(reg.map_or(0, bit) | rm.map_or(0, bit)),
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
        let eff = resolve_eff(intent, &operands, p.rex_w, immv);
        Some(Decoded {
            len: c.pos - start,
            kind,
            operands,
            eff,
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
            let intent = match (is_cmp, op & 0x07) {
                (true, 0 | 1) => E::CmpRmReg,
                (true, _) => E::CmpRegRm,
                (false, 0 | 1) => E::WrRm,
                (false, _) => E::WrReg,
            };
            return finish_modrm(c, start, p, 0, Normal, intent);
        }
        match op {
            // ModRM, no immediate.
            0x84 | 0x85 => finish_modrm(c, start, p, 0, Normal, E::NoGp), // test
            0x86 | 0x87 => finish_modrm(c, start, p, 0, Normal, E::XchgRegRm), // xchg
            0x88 => finish_modrm(c, start, p, 0, Normal, E::WrRm),        // mov rm8, r8
            0x89 => finish_modrm(c, start, p, 0, Normal, E::MovToRm),     // mov rm, r
            0x8A => finish_modrm(c, start, p, 0, Normal, E::WrReg),       // mov r8, rm8
            0x8B => finish_modrm(c, start, p, 0, Normal, E::MovToReg),    // mov r, rm
            0x63 => finish_modrm(c, start, p, 0, Normal, E::WrReg),       // movsxd r64, rm32
            0x8D => finish_modrm(c, start, p, 0, Normal, E::Lea),
            0x8F => finish_modrm(c, start, p, 0, Normal, E::WrRm), // pop r/m
            0xD0..=0xD3 => finish_modrm(c, start, p, 0, Normal, E::WrRm), // shift r/m, 1|cl
            0xFE => finish_modrm(c, start, p, 0, Normal, E::WrRm), // inc/dec r/m8
            // ModRM + imm8.
            0x80 | 0x82 | 0x83 => {
                let digit = (c.peek()? >> 3) & 0x07;
                finish_modrm(c, start, p, 1, Normal, E::AluImmGroup(digit))
            }
            0x6B => finish_modrm(c, start, p, 1, Normal, E::WrReg), // imul r, r/m, imm8
            0xC0 | 0xC1 => finish_modrm(c, start, p, 1, Normal, E::WrRm), // shift r/m, imm8
            0xC6 => finish_modrm(c, start, p, 1, Normal, E::WrRm),  // mov r/m8, imm8
            // ModRM + immZ.
            0x81 => {
                let digit = (c.peek()? >> 3) & 0x07;
                finish_modrm(c, start, p, imm_z(p), Normal, E::AluImmGroup(digit))
            }
            0x69 => finish_modrm(c, start, p, imm_z(p), Normal, E::WrReg), // imul r, r/m, immZ
            0xC7 => finish_modrm(c, start, p, imm_z(p), Normal, E::MovImmToRm),
            // groups 3/5: immediate (if any) depends on the /reg field — peek at the ModRM.
            0xF6 => group3(c, start, p, 1),
            0xF7 => group3(c, start, p, imm_z(p)),
            0xFF => group5(c, start, p),
            // pop r — writes the opcode-encoded register.
            0x58..=0x5F => {
                let dst = (op & 0x07) | (u8::from(p.rex_b) << 3);
                Some(one(c, start, Normal, Eff::Wr(bit(dst))))
            }
            // xchg eax, r — writes rax and r.
            0x91..=0x97 => {
                let r = (op & 0x07) | (u8::from(p.rex_b) << 3);
                Some(one(c, start, Normal, Eff::Wr(bit(0) | bit(r))))
            }
            0x98 => Some(one(c, start, Normal, Eff::Wr(bit(0)))), // cwde/cdqe → rax
            0x99 => Some(one(c, start, Normal, Eff::Wr(bit(2)))), // cwd/cdq/cqo → rdx
            0xD7 => Some(one(c, start, Normal, Eff::Wr(bit(0)))), // xlat → al
            // push r, nops, flag ops, invalid-in-64 legacy: no tracked GP write.
            0x50..=0x57
            | 0x06
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
            0x3C | 0xA8 | 0x6A => finish_imm(c, start, 1, Normal, Eff::NoGp),
            0x05 | 0x0D | 0x15 | 0x1D | 0x25 | 0x2D | 0x35 => {
                finish_imm(c, start, imm_z(p), Normal, Eff::Wr(bit(0)))
            }
            0x3D | 0xA9 | 0x68 => finish_imm(c, start, imm_z(p), Normal, Eff::NoGp),
            // mov r8, imm8 — writes the opcode-encoded register (byte write → clobber).
            0xB0..=0xB7 => {
                let dst = (op & 0x07) | (u8::from(p.rex_b) << 3);
                finish_imm(c, start, 1, Normal, Eff::Wr(bit(dst)))
            }
            // mov r32/r64, imm — writes a known constant into the opcode-encoded register.
            0xB8..=0xBF => {
                let dst = (op & 0x07) | (u8::from(p.rex_b) << 3);
                let size = if p.rex_w { 8 } else { imm_z(p) };
                let immv = read_imm(c, size)?;
                Some(Decoded {
                    len: c.pos - start,
                    kind: Normal,
                    operands: NO_OPERANDS,
                    eff: Eff::MovImm {
                        dst,
                        imm: if p.rex_w { immv } else { immv & 0xFFFF_FFFF },
                    },
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

    /// Group 5 (`FF`): `/2`,`/3` indirect call, `/4`,`/5` indirect jmp, others inc/dec/push r/m.
    fn group5(c: &mut Cursor, start: usize, p: &Pfx) -> Option<Decoded> {
        let digit = (c.peek()? >> 3) & 0x07;
        let kind = match digit {
            2 | 3 => Kind::IndirectCall,
            4 | 5 => Kind::IndirectJmp,
            _ => Kind::Normal,
        };
        finish_modrm(c, start, p, 0, kind, EffIntent::Group5(digit))
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
            0x40..=0x4F => return finish_modrm(c, start, p, 0, Normal, E::Cmov),
            // setcc r/m8 (byte write → clobber the r/m register).
            0x90..=0x9F => return finish_modrm(c, start, p, 0, Normal, E::WrRm),
            // movzx/movsx r, r/m — zero/sign-extended, not a tracked pointer value → clobber reg.
            0xB6 | 0xB7 | 0xBE | 0xBF => return finish_modrm(c, start, p, 0, Normal, E::WrReg),
            0xAF => return finish_modrm(c, start, p, 0, Normal, E::WrReg), // imul r, r/m
            0xBC | 0xBD => return finish_modrm(c, start, p, 0, Normal, E::WrReg), // bsf/bsr r, r/m
            // shld/shrd r/m, r, (imm8|cl) — write the r/m.
            0xA4 | 0xAC => return finish_modrm(c, start, p, 1, Normal, E::WrRm),
            0xA5 | 0xAD => return finish_modrm(c, start, p, 0, Normal, E::WrRm),
            0xAB | 0xB3 | 0xBB => return finish_modrm(c, start, p, 0, Normal, E::WrRm), // bts/btr/btc r/m
            0xA3 => return finish_modrm(c, start, p, 0, Normal, E::NoGp), // bt — flags only
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
        finish_modrm(c, start, p, imm, Kind::Normal, EffIntent::NoGp)
    }

    fn one(c: &Cursor, start: usize, kind: Kind, eff: Eff) -> Decoded {
        Decoded {
            len: c.pos - start,
            kind,
            operands: NO_OPERANDS,
            eff,
        }
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
        use iced_x86::{
            Decoder, DecoderOptions, Instruction, InstructionInfoFactory, OpAccess, OpKind,
            Register,
        };
        use temen_text::parse_module;

        /// Compile `src` through the real JIT path and return each function's
        /// `(code, bb_starts, bb_edges)`.
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

        /// The decoder's boundaries, validity, and memory-operand decode must match iced-x86 on real
        /// output, and `check` must find nothing on clean Cranelift code.
        fn cross_check(src: &str) {
            for (bytes, bb_starts, bb_edges, layout) in compiled(src) {
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
                    // Soundness of the dataflow transfer: our GP write-set must cover iced's. A GP
                    // register iced says the instruction writes but we don't would let a stale fact
                    // (mem_base, a confined index) survive a real overwrite — the escape the #2535
                    // clobber exists to prevent. rsp/rbp are excluded: they are the stack pointer and
                    // frame pointer, recognized structurally, never carriers of a tracked fact.
                    const SP_BP: u16 = (1 << reg::RSP) | (1 << reg::RBP);
                    let ours = writes_of(d.eff, &d.operands);
                    let iced_w = iced_gp_writes(&insn);
                    assert_eq!(
                        iced_w & !ours & !SP_BP,
                        0,
                        "GP write-set gap at +0x{pos:x}: iced writes {iced_w:#06x}, ours {ours:#06x}, \
                         eff {:?}\n{:02x?}",
                        d.eff,
                        bytes
                    );
                }
                let findings = check(&bytes, &bb_starts, &bb_edges, layout);
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

        /// The abstract lattice's `meet` (the fact true on both paths) and the loop-header widening
        /// that bounds the fixpoint (#2535).
        #[test]
        fn lattice_meet_and_widen() {
            use Fact::*;
            let conf = Confined {
                guard: 0x2000000000,
            };
            assert_eq!(MemBase.meet(MemBase), MemBase);
            assert_eq!(MemBase.meet(Vmctx), Unknown);
            assert_eq!(conf.meet(conf), conf);
            assert_eq!(conf.meet(Confined { guard: 4096 }), Unknown); // different guard ⇒ no fact
            assert_eq!(conf.meet(Unknown), Unknown);
            assert_eq!(Const(5).meet(Const(5)), Const(5));
            assert_eq!(Const(5).meet(Const(6)), Unknown);
            // A bound true on both paths is the looser (larger) of the two.
            assert_eq!(Bounded(4).meet(Bounded(8)), Bounded(8));
            assert_eq!(Bounded(4).meet(MemBase), Unknown);

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

        /// A window access through an index that was **not** confined by the D63 sequence is the
        /// escape-relevant case the policy must flag: `mov rax, [rdi + r8]` with `r8` a guest
        /// argument (unknown, unconfined) and `rdi` = `mem_base`.
        #[test]
        fn policy_flags_unconfined_window_index() {
            let bytes = [0x4a, 0x8b, 0x04, 0x07, 0xc3]; // mov rax,[rdi+r8*1]; ret
            let f = check(&bytes, &[], &[], None);
            assert!(
                f.iter().any(|f| f.why.contains("unconfined index")),
                "expected an unconfined-window-index finding, got {f:?}"
            );
        }

        /// A memory access whose base register has no recognized provenance (here `r9`, a guest
        /// argument) is flagged — it is neither the window, the stack, vmctx, the fn-table, nor rodata.
        #[test]
        fn policy_flags_unknown_base() {
            let bytes = [0x49, 0x8b, 0x01, 0xc3]; // mov rax,[r9]; ret
            let f = check(&bytes, &[], &[], None);
            assert!(
                f.iter().any(|f| f.why.contains("unrecognized provenance")),
                "expected an unknown-base finding, got {f:?}"
            );
        }

        /// The full D63 confinement sequence is recognized and produces no finding: materialize the
        /// guard, compare the index, `cmovae` it to the guard, then access `[mem_base + confined]`.
        #[test]
        fn policy_accepts_the_confinement_sequence() {
            let bytes = [
                0x48, 0xb8, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00,
                0x00, // mov rax, 0x2000000000
                0x4c, 0x3b, 0xc0, // cmp r8, rax
                0x4c, 0x0f, 0x43, 0xc0, // cmovae r8, rax
                0x4a, 0x8b, 0x04, 0x07, // mov rax, [rdi + r8*1]
                0xc3, // ret
            ];
            let f = check(&bytes, &[], &[], None);
            assert!(f.is_empty(), "the confinement sequence was flagged: {f:?}");
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
            assert!(check(&bytes, &[], &[], big).is_empty());
            // A 32 KiB reservation does not → the access reaches past it.
            let small = Some(WindowLayout {
                mask: (1u64 << 15) - 1,
                guard_offset: 1u64 << 15,
            });
            let f = check(&bytes, &[], &[], small);
            assert!(
                f.iter().any(|f| f.why.contains("past the reservation")),
                "expected a reservation-overrun finding, got {f:?}"
            );
        }
    }
}
