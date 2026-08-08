//! **Interpreter-agnostic loop-cell discovery — the shared capture driver.**
//!
//! `lua_futamura_auto_rolled.rs` and `peval_second_interp.rs` used to carry near-identical discovery
//! code, one per interpreter. This is that code, extracted once and parameterized by a [`TargetDesc`]
//! — so "same technique, parallel code" becomes "same code, two interpreters". Everything Lua-specific
//! (or regVM-specific) is now a field of `TargetDesc`/`Located`, nothing is hardcoded in the driver.
//!
//! The driver observes the interpreter's dispatch block and derives, with no source-level knowledge:
//! - the **safepoint** (a clean resume point): the in-memory program-counter value with the longest
//!   run of consecutive dispatch hits — when the target keeps its pc in memory (Lua). A target whose
//!   pc lives in a machine register (the regVM) has no such signal and analyses from the first hit.
//! - the **loop-carried cells**: registers whose value takes ≥3 distinct values over the loop region
//!   (a one-time prologue write yields 2; an invariant yields 1), optionally tag-filtered.
//! - the **trip counter**: the carried cell whose region series is monotone non-increasing from its
//!   peak and net-decreases (robust to a prologue ramp-up that sets the initial count).

#![allow(dead_code)]

use svm_interp::{Inspector, IrPc, Stop, StopReason};
use svm_ir::{Module, Terminator};

/// The **static** shape of an interpreter's state, the part that does not depend on a particular run.
pub struct TargetDesc {
    /// Stride between adjacent registers, in bytes (Lua: 16 = one `TValue`; regVM: 8 = one `long`).
    pub reg_stride: u64,
    /// How many registers to scan from the register base.
    pub n_regs: u64,
    /// A value-tag filter `(offset_from_cell, tag_byte)`: keep only cells whose byte at `cell+offset`
    /// equals `tag_byte` (Lua: `(8, 0x03)` VNUMINT). `None` for an untagged register file (regVM).
    pub tag: Option<(u64, u8)>,
    /// Bytes of the guest window to snapshot at the safepoint.
    pub capture_len: usize,
    /// How many dispatch hits to observe.
    pub observe_hits: usize,
}

/// The **located** (per-run) addresses of an interpreter's state, computed from the entry image.
pub struct Located {
    /// Address of register 0 (Lua: `ci->func + 1 TValue`; regVM: the `reg` symbol's address).
    pub reg_base: u64,
    /// Address of the in-memory program counter (Lua: `ci->u.l.savedpc`), or `None` if the pc lives
    /// in a machine register (regVM) — then the driver cannot detect a resume safepoint by pc.
    pub pc_addr: Option<u64>,
}

/// What discovery found.
pub struct Discovered {
    /// Loop-carried cell addresses (ascending) — the residual's dynamic inputs.
    pub varying: Vec<u64>,
    /// The trip-counter cell (the monotone-decreasing one), or 0 if none was identified.
    pub counter: u64,
    /// The full guest window at the safepoint hit (empty region snapshot for pc-less targets: it is
    /// the window at the first dispatch hit, still a valid image for entry-rooted use).
    pub window: Vec<u8>,
    /// The dispatch-hit index chosen as the safepoint.
    pub safepoint_hit: usize,
}

pub fn rd_u64(w: &[u8], a: u64) -> u64 {
    u64::from_le_bytes(w[a as usize..a as usize + 8].try_into().unwrap())
}

/// The dispatch block of `func`: its widest `BrTable` (the opcode switch).
pub fn dispatch_block(m: &Module, func: u32) -> u32 {
    let mut best = (0u32, 0usize);
    for (bi, b) in m.funcs[func as usize].blocks.iter().enumerate() {
        if let Terminator::BrTable { targets, .. } = &b.term {
            if targets.len() > best.1 {
                best = (bi as u32, targets.len());
            }
        }
    }
    best.0
}

/// Observe `func`'s dispatch loop and discover its safepoint + loop-carried cells, parameterized only
/// by `desc` (static shape) and `loc` (located addresses). `make_insp` returns a fresh inspector
/// positioned at `func`'s entry, ready to run — called once per pass (analysis, then capture); it
/// encapsulates every interpreter-specific attach detail (Lua's powerbox `debug_attach` vs the
/// regVM's `Inspector::attach`).
pub fn discover(
    make_insp: &dyn Fn() -> Inspector,
    func: u32,
    dispatch: u32,
    loc: &Located,
    desc: &TargetDesc,
) -> Discovered {
    let disp_pc = IrPc {
        module: 0,
        func,
        block: dispatch as usize,
        inst: 0,
    };

    // ---- Pass 1: observe the dispatch stream (pc value + register file per hit). ----
    let mut insp = make_insp();
    insp.set_breakpoint(disp_pc);
    let mut pcs: Vec<u64> = Vec::new();
    let mut series: Vec<Vec<u64>> = vec![Vec::new(); desc.n_regs as usize];
    for _ in 0..desc.observe_hits {
        match insp.run_until_stop() {
            Stop::Break {
                reason: StopReason::Breakpoint,
                ..
            } => {
                let w = insp.read_window(0, desc.capture_len).expect("window");
                if let Some(pc) = loc.pc_addr {
                    pcs.push(rd_u64(&w, pc));
                }
                for (r, s) in series.iter_mut().enumerate() {
                    s.push(rd_u64(&w, loc.reg_base + r as u64 * desc.reg_stride));
                }
            }
            Stop::Finished { .. } => break,
            o => panic!("unexpected during observe: {o:?}"),
        }
    }
    insp.clear_breakpoint(disp_pc);
    let n_hits = series.first().map(|s| s.len()).unwrap_or(0);
    assert!(
        n_hits >= 4,
        "too few dispatch hits to find a loop ({n_hits})"
    );

    // Safepoint: with an in-memory pc, the loop body is the pc value with the longest run of
    // consecutive equal hits — its first hit is a clean resume image. Without one, analyse from hit 0.
    let safepoint_hit = if loc.pc_addr.is_some() {
        longest_run_start(&pcs)
    } else {
        0
    };

    // Carried cells over the loop region `[safepoint_hit..]`: a register that takes ≥3 distinct values
    // (a one-time prologue write yields 2; an invariant yields 1). This one predicate reproduces both
    // the Lua "varies in the loop region" and the regVM "distinct ≥ 3 over the whole stream" rules.
    let mut varying: Vec<u64> = Vec::new();
    for (r, full) in series.iter().enumerate() {
        let region = &full[safepoint_hit..];
        let mut d = region.to_vec();
        d.sort_unstable();
        d.dedup();
        if d.len() >= 3 {
            varying.push(loc.reg_base + r as u64 * desc.reg_stride);
        }
    }

    // ---- Pass 2: capture the full window at the safepoint hit. ----
    let mut insp2 = make_insp();
    insp2.set_breakpoint(disp_pc);
    let mut window = Vec::new();
    for hit in 0..=safepoint_hit {
        match insp2.run_until_stop() {
            Stop::Break {
                reason: StopReason::Breakpoint,
                ..
            } => {
                if hit == safepoint_hit {
                    window = insp2.read_window(0, desc.capture_len).expect("window");
                }
            }
            o => panic!("unexpected during capture: {o:?}"),
        }
    }
    insp2.clear_breakpoint(disp_pc);
    assert!(!window.is_empty(), "safepoint window not captured");

    // Optional tag filter against the safepoint window (drops non-VNUMINT scratch for Lua).
    if let Some((off, val)) = desc.tag {
        varying.retain(|&a| window.get((a + off) as usize).copied() == Some(val));
    }

    // Counter = the first carried cell whose region series is monotone non-increasing from its peak
    // and net-decreases. Analysing from the peak (not raw) skips any prologue ramp that sets the count.
    let reg_of = |addr: u64| ((addr - loc.reg_base) / desc.reg_stride) as usize;
    let mut counter = 0u64;
    for &addr in &varying {
        let region = &series[reg_of(addr)][safepoint_hit..];
        let argmax = region
            .iter()
            .enumerate()
            .max_by_key(|(_, v)| **v)
            .map(|(i, _)| i)
            .unwrap();
        let tail = &region[argmax..];
        let monotone_down = tail.windows(2).all(|w| w[1] <= w[0]) && tail.first() > tail.last();
        if monotone_down {
            counter = addr;
            break;
        }
    }

    Discovered {
        varying,
        counter,
        window,
        safepoint_hit,
    }
}

/// The start index of the longest run of consecutive equal values in `xs`.
fn longest_run_start(xs: &[u64]) -> usize {
    let (mut best_start, mut best_len) = (0usize, 0usize);
    let (mut run_start, mut run_len) = (0usize, 1usize);
    for i in 1..xs.len() {
        if xs[i] == xs[i - 1] {
            run_len += 1;
        } else {
            run_start = i;
            run_len = 1;
        }
        if run_len > best_len {
            best_len = run_len;
            best_start = run_start;
        }
    }
    best_start
}
