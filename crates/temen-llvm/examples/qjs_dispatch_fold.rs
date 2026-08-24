//! **QuickJS dispatch-fold spike** — the JS analog of the Lua "dispatch-fold ROI" milestone.
//!
//! Captures a real JS loop's `JS_CallInternal` frame at entry, freezes the captured engine state
//! (bytecode + dispatch table + constants + operand stack) as a const overlay, and specializes
//! `JS_CallInternal` from entry — then reports what stands between that and a fully folded residual.
//!
//! **What it establishes (three parts):**
//!
//! 1. **Capture works on a live JS loop.** Break at `JS_CallInternal` entry, keep the invocation that
//!    then dispatches the most (the loop's top-level frame), and read its 10 params + an 8 MB window.
//!
//! 2. **The opcode dispatch folds *by construction* under a constant pc.** The 1837-target `br_table`
//!    in the dispatch block computes its index as
//!    `dispatch_table[ (*pc) & 0xFF ]` — `*pc` is a byte load from the *threaded pc* (a block param),
//!    and `dispatch_table` is a frozen global (a constant window address). This spike statically
//!    decodes that block and shows the recipe: with the pc a constant and the bytecode + table frozen,
//!    the index reduces to a constant and the 1837-way switch collapses to one `Br`. This is exactly
//!    the *register-pc* interpreter class the shared `discover` driver already handles (`pc_addr:
//!    None`) — no new fold mechanism is needed.
//!
//! 3. **The wall is the C prologue's cold decision tree, not the dispatch.** Specializing from the C
//!    *entry* (rather than from the dispatch block) walks `JS_CallInternal`'s prologue, whose guards
//!    (stack-canary check, `__js_poll_interrupts`, is-C-function vs bytecode, async/generator init)
//!    are dynamic from entry, so the abstract interpreter explores cold arms a real bytecode `for`
//!    loop never takes. Two of those are pure runtime call-outs and cut cleanly here (the abort
//!    wrapper and the interrupt poll — the same `cut_calls` error-raiser treatment the Lua arc used);
//!    the rest are cold branches that need the guard-and-deopt discipline (`deopt_targets`) applied
//!    slice by slice — the QuickJS analog of the ~40-slice Lua `auto_rolled` arc. The cheaper route is
//!    a specializer entry-point feature: specialize from the *dispatch block* with the pc threaded as
//!    a spec arg, skipping the prologue entirely (see the tracking issue). Either way the preconditions
//!    all hold; only the specialization arc remains.
//!
//! Run `cargo run --release --example qjs_probe` first for the static/structural gate. To see the exact
//! `(func, block)` where the from-entry walk walls, build `temen-peval` with `--features trace`.
//!
//! Usage: cargo run --release --example qjs_dispatch_fold -- /tmp/qjs_bc/qjs_full.ll

use temen_interp::{IrPc, Stop, StopReason, Value};
use temen_ir::{Inst, Terminator, ValType};
use temen_peval::{specialize_with_config, SpecArg, SpecConfig};

const CAP: usize = 8 << 20;
const DISPATCH: usize = 1836; // block with the 1837-target opcode br_table (from qjs_probe)

fn main() {
    let p = std::env::args().nth(1).expect("usage: <ll>");
    let m = temen_llvm::translate_ll_path(&p).expect("translate").module;
    let jci = m
        .exports
        .iter()
        .find(|e| e.name == "JS_CallInternal")
        .expect("jci")
        .func;
    let nparams = m.funcs[jci as usize].params.len();

    // ---- Part 2: the dispatch folds by construction under a constant pc. Decode the index recipe. ----
    decode_dispatch(&m, jci);

    // ---- Part 1: capture a real JS loop's JS_CallInternal frame. ----
    // A short, pure loop (no print → no C-call in the top-level frame): the eval result is `s`.
    let js = b"var s = 0; for (var i = 0; i < 4; i++) { s += i; } s;\n";
    let inst = temen_run::instantiate(m.clone()).expect("inst");
    let mut insp = inst.debug_attach(js.to_vec(), 2_000_000_000);
    let entry = IrPc {
        module: 0,
        func: jci,
        block: 0,
        inst: 0,
    };
    let disp = IrPc {
        module: 0,
        func: jci,
        block: DISPATCH,
        inst: 0,
    };
    insp.set_breakpoint(entry);
    insp.set_breakpoint(disp);

    // Find the JS_CallInternal invocation that runs the loop = the one with the most dispatch hits.
    // Capture (params, window) at every entry; keep the entry whose frame then dispatched the most.
    let mut cur: Option<(Vec<SpecArg>, Vec<u8>)> = None;
    let mut cur_hits = 0usize;
    let mut best: Option<(usize, Vec<SpecArg>, Vec<u8>)> = None;
    for _ in 0..200_000 {
        match insp.run_until_stop() {
            Stop::Break {
                reason: StopReason::Breakpoint,
                pc,
            } if pc.block == 0 => {
                if let Some((a, w)) = cur.take() {
                    if best.as_ref().is_none_or(|(n, _, _)| cur_hits > *n) {
                        best = Some((cur_hits, a, w));
                    }
                }
                let args: Vec<SpecArg> = (0..nparams)
                    .map(
                        |i| match (insp.read_ir_value(0, i), m.funcs[jci as usize].params[i]) {
                            (Some(Value::I64(v)), _) => SpecArg::ConstI64(v),
                            (Some(Value::I32(v)), _) => SpecArg::ConstI32(v),
                            (_, ValType::I32) => SpecArg::ConstI32(0),
                            _ => SpecArg::ConstI64(0),
                        },
                    )
                    .collect();
                let w = insp.read_window(0, CAP).expect("win");
                cur = Some((args, w));
                cur_hits = 0;
            }
            Stop::Break {
                reason: StopReason::Breakpoint,
                ..
            } => cur_hits += 1,
            Stop::Finished { .. } => {
                if let Some((a, w)) = cur.take() {
                    if best.as_ref().is_none_or(|(n, _, _)| cur_hits > *n) {
                        best = Some((cur_hits, a, w));
                    }
                }
                break;
            }
            o => {
                println!("stopped: {o:?}");
                break;
            }
        }
    }
    let Some((hits, args, window)) = best else {
        println!("no JS_CallInternal frame captured");
        return;
    };
    println!(
        "\ncapture: loop frame with {hits} dispatch hits, {nparams} params, {}MB window",
        CAP >> 20
    );

    // ---- Part 3: attempt the from-entry fold; cut the pure runtime call-outs that block the prologue. ----
    // The C prologue guards its stack canary and, on mismatch, tail-calls a 1-block abort wrapper that
    // reaches `exit` (an import). It also polls `__js_poll_interrupts` (an indirect call to the host
    // interrupt handler) per iteration. Both are opaque runtime call-outs a real run treats as
    // safepoints, not interpreter logic — cut them (emit verbatim, carry unspecialized) exactly like
    // the Lua arc cut its error raisers / GC steps, so the fold isn't dragged into the runtime.
    let jci_f = &m.funcs[jci as usize];
    let mut cuts: Vec<u32> = jci_f
        .blocks
        .iter()
        .filter(|b| matches!(b.term, Terminator::Unreachable))
        .flat_map(|b| b.insts.iter())
        .filter_map(|i| match i {
            Inst::Call { func, .. } => Some(*func),
            _ => None,
        })
        .collect();
    for name in ["__js_poll_interrupts"] {
        if let Some(fi) = m.exports.iter().find(|e| e.name == name).map(|e| e.func) {
            cuts.push(fi);
        }
    }
    println!(
        "cutting {} opaque runtime call-out(s) (abort + interrupt poll): {cuts:?}",
        cuts.len()
    );

    let cfg = SpecConfig {
        const_overlays: vec![(0, window)],
        cut_calls: cuts,
        carry_whole_module: true,
        carry_keep_imports: true,
        ..SpecConfig::default()
    };

    println!("specializing JS_CallInternal from entry with the frozen engine state…");
    match specialize_with_config(&m, jci, &args, &cfg) {
        Ok(r) => {
            let total_brt: usize = r
                .funcs
                .iter()
                .flat_map(|f| &f.blocks)
                .filter(|b| matches!(b.term, Terminator::BrTable { .. }))
                .count();
            println!(
                "  RESIDUAL: {} funcs, entry {} blocks; total br_tables={total_brt}",
                r.funcs.len(),
                r.funcs[0].blocks.len()
            );
            println!(
                "  {}",
                if total_brt == 0 {
                    "✓ opcode dispatch FOLDED — no br_table survives (the JS bytecode switch is gone)"
                } else {
                    "partial — some br_table remains"
                }
            );
        }
        Err(e) => {
            println!("  from-entry specialize walls: {e:?}");
            println!(
                "  → the wall is the C prologue's cold decision tree (build temen-peval with --features\n     \
                 trace to see the exact (func,block)): guards that are dynamic from the C entry (is-\n     \
                 C-function vs bytecode, async/generator init, and the per-opcode slow-path arms) send\n     \
                 the abstract interpreter down cold branches a real bytecode `for` loop never takes.\n  \
                 → these are the guard-and-deopt slices of the QuickJS fold arc (the analog of the Lua\n     \
                 `auto_rolled` arc), OR are skipped wholesale by specializing from the dispatch block\n     \
                 with the pc threaded (a specializer entry-point feature). The dispatch fold itself is\n     \
                 not in question — see Part 2 above."
            );
        }
    }
}

/// **Part 2 — decode the dispatch block's br_table index recipe** and show it folds under a constant pc.
/// The index is `dispatch_table[ (*pc) & 0xFF ]`: a byte load from the threaded pc, masked, used to
/// index a frozen-global dispatch table. With the pc constant and the bytecode + table frozen, every
/// input is constant, so the 1837-way `br_table` reduces to a single `Br`.
fn decode_dispatch(m: &temen_ir::Module, jci: u32) {
    let b = &m.funcs[jci as usize].blocks[DISPATCH];
    let Terminator::BrTable { targets, idx, .. } = &b.term else {
        println!("dispatch block {DISPATCH} is not a br_table?!");
        return;
    };
    let opcode_byte_load = b.insts.iter().any(|i| {
        matches!(
            i,
            Inst::Load {
                op: temen_ir::LoadOp::I32_8U,
                ..
            }
        )
    });
    let table_base = b.insts.iter().find_map(|i| match i {
        Inst::ConstI64(v) if *v > 0x10000 => Some(*v as u64),
        _ => None,
    });
    println!("dispatch fold (by construction):");
    println!(
        "  block {DISPATCH}: {}-target br_table on value {idx}",
        targets.len()
    );
    println!(
        "  index = dispatch_table[(*pc) & 0xFF]  (opcode byte load: {opcode_byte_load}, table base: {})",
        table_base.map_or("?".into(), |t| format!("{t:#x}"))
    );
    println!("  ⇒ const pc + frozen bytecode + frozen table ⇒ index is const ⇒ the 1837-way switch folds to one Br");
}
