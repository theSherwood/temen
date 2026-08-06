//! **A call-bearing loop rolls** (slice 3 — selective poscall reload).
//!
//! `lua_futamura_call.rs` projects a single Lua call, in-and-out. `lua_futamura_rolled.rs` rolls a
//! *pure* loop (`x = x + 3`). This test is the join: a loop whose body **calls a Lua function each
//! iteration** — `for i=1,5 do x = add(x, 3) end` — rolls to a bounded, dispatch-free residual instead
//! of unrolling. The enabling piece is [`svm_peval::PoscallModel::selective`]: on the modeled return,
//! only the result **value** cell is reloaded dynamic while its **tag** stays folded (integer), so the
//! caller's register tags survive the call and the loop-header memo converges. (A full reload would
//! turn every tag dynamic and the loop would unroll.)
//!
//! Capture is at the loop-body safepoint (like the rolled test) so the loop-carried cells (`x`, `i`,
//! the FORLOOP counter) can be marked [`SpecConfig::dynamic_cells`] and the loop rolls. The `add` frame
//! overlays come from a second profiling pass to the first callee dispatch (deterministic arena).
//!
//! The residual keeps its cut call-outs (the opaque `precall`/`poscall`, the allocator/GC), so unlike
//! the pure rolled loop it is *not* self-contained and isn't executed here — the milestone is
//! **convergence**: the dispatch folds (`br_table=0`) and the loop rolls to a bounded block count with
//! the call inlined. End-to-end execution correctness of the precall/poscall/selective mechanism is
//! covered by `lua_futamura_call.rs` (which runs a projected call and diffs stdout).
//!
//! Run: `cargo test -p svm-llvm --test lua_futamura_call_loop -- --nocapture`

use svm_interp::{IrPc, Stop, StopReason, Value};
use svm_ir::{Inst, Module, Terminator};
use svm_peval::{specialize_with_config, LuaSite, PoscallModel, PrecallModel, SpecArg, SpecConfig};

const SCRIPT: &str = "local function add(a, b) return a + b end\n\
                      local x = 0\n\
                      for i = 1, 5 do x = add(x, 3) end\n\
                      return x\n";
const CAP: usize = 8 << 20;

const L_CI: u64 = 32;
const L_STACK_LAST: u64 = 40;
const L_STACK: u64 = 48;
const LUA_STATE_SIZE: usize = 200;
const CI_SIZE: usize = 104;
const CI_FUNC: u64 = 0;
const CI_PREVIOUS: u64 = 16;
const LCLOSURE_P: u64 = 24;
const PROTO_SIZECODE: u64 = 24;
const PROTO_SIZEP: u64 = 32;
const PROTO_CODE: u64 = 64;
const PROTO_P: u64 = 72;
const SV: u64 = 16; // sizeof(TValue)
const TAG_OFF: u64 = 8; // TValue tag byte
const VNUMINT: u8 = 0x03;

fn lua_module() -> Module {
    let p = format!(
        "{}/tests/fixtures/lua/lua_eval.ll",
        env!("CARGO_MANIFEST_DIR")
    );
    svm_llvm::translate_ll_path(&p).expect("translate").module
}
fn byname(m: &Module, n: &str) -> Option<u32> {
    m.exports.iter().find(|e| e.name == n).map(|e| e.func)
}
fn luav_execute(m: &Module) -> u32 {
    byname(m, "luaV_execute").expect("luaV_execute")
}
fn rd_u64(w: &[u8], a: u64) -> u64 {
    u64::from_le_bytes(w[a as usize..a as usize + 8].try_into().unwrap())
}
fn rd_i32(w: &[u8], a: u64) -> i32 {
    i32::from_le_bytes(w[a as usize..a as usize + 4].try_into().unwrap())
}
fn dispatch_block(m: &Module, luav: u32) -> u32 {
    let mut best = (0u32, 0usize);
    for (bi, b) in m.funcs[luav as usize].blocks.iter().enumerate() {
        if let Terminator::BrTable { targets, .. } = &b.term {
            if targets.len() > best.1 {
                best = (bi as u32, targets.len());
            }
        }
    }
    best.0
}
fn precall_block(m: &Module, luav: u32, precall: u32) -> u32 {
    m.funcs[luav as usize]
        .blocks
        .iter()
        .position(|b| {
            b.insts
                .iter()
                .any(|i| matches!(i, Inst::Call { func, .. } if *func == precall))
                && matches!(b.term, Terminator::BrIf { .. })
        })
        .expect("precall branch block") as u32
}
fn precall_ra_val(m: &Module, luav: u32, pblock: u32, precall: u32) -> u32 {
    m.funcs[luav as usize].blocks[pblock as usize]
        .insts
        .iter()
        .find_map(|i| match i {
            Inst::Call { func, args } if *func == precall => Some(args[2]),
            _ => None,
        })
        .expect("precall ra arg")
}

/// Capture at `luaV_execute` entry: `(sp, L, ci)`.
fn entry_regs(insp: &mut svm_interp::Inspector, luav: u32) -> (i64, u64, u64) {
    let pc = IrPc {
        module: 0,
        func: luav,
        block: 0,
        inst: 0,
    };
    insp.set_breakpoint(pc);
    let r = match insp.run_until_stop() {
        Stop::Break { .. } => {
            let g = |i| match insp.read_ir_value(0, i) {
                Some(Value::I64(v)) => v,
                o => panic!("v{i}: {o:?}"),
            };
            (g(0), g(1) as u64, g(2) as u64)
        }
        o => panic!("no entry break: {o:?}"),
    };
    insp.clear_breakpoint(pc);
    r
}

#[test]
fn a_call_bearing_loop_rolls() {
    let m = lua_module();
    let luav = luav_execute(&m);
    let precall = byname(&m, "luaD_precall").expect("precall");
    let poscall = byname(&m, "luaD_poscall").expect("poscall");
    let dispatch = dispatch_block(&m, luav);
    let pblock = precall_block(&m, luav, precall);
    let ra_val = precall_ra_val(&m, luav, pblock, precall);
    println!(
        "luav=f{luav} precall=f{precall} poscall=f{poscall} dispatch=b{dispatch} pblock=b{pblock}"
    );

    // ---- Pass 1: loop-body safepoint (break at the precall — the add call — and read the register
    // file + ra). The precall block is entered once per iteration; the first hit is the first call, at
    // which point x=0, i=1, counter set up. ----
    let inst = svm_run::instantiate(m.clone()).expect("inst");
    let mut insp = inst.debug_attach(SCRIPT.as_bytes().to_vec(), u64::MAX);
    let (sp, l, ci) = entry_regs(&mut insp, luav);
    let base = {
        let w0 = insp.read_window(0, CAP).expect("w");
        rd_u64(&w0, ci + CI_FUNC) + SV
    };
    let pbp = IrPc {
        module: 0,
        func: luav,
        block: pblock as usize,
        inst: 0,
    };
    insp.set_breakpoint(pbp);
    let (w, ra_add) = match insp.run_until_stop() {
        Stop::Break {
            reason: StopReason::Breakpoint,
            ..
        } => {
            let ra = match insp.read_ir_value(0, ra_val as usize) {
                Some(Value::I64(v)) => v as u64,
                o => panic!("ra: {o:?}"),
            };
            (insp.read_window(0, CAP).expect("w"), ra)
        }
        o => panic!("no precall break: {o:?}"),
    };
    insp.clear_breakpoint(pbp);
    assert_eq!(rd_u64(&w, l + L_CI), ci, "L->ci");
    let stack_lo = rd_u64(&w, l + L_STACK);
    let stack_hi = rd_u64(&w, l + L_STACK_LAST);
    let stack_len = (stack_hi - stack_lo) as usize;
    // Outer (main-chunk) frame.
    let outer_func = rd_u64(&w, ci + CI_FUNC);
    let outer_cl = rd_u64(&w, outer_func);
    let outer_proto = rd_u64(&w, outer_cl + LCLOSURE_P);
    let outer_code = rd_u64(&w, outer_proto + PROTO_CODE);
    let outer_sizecode = rd_i32(&w, outer_proto + PROTO_SIZECODE) as usize;
    let outer_pp = rd_u64(&w, outer_proto + PROTO_P);
    let outer_sizep = rd_i32(&w, outer_proto + PROTO_SIZEP) as usize;
    // Loop-carried cells (from the register diff): x=R2, i=R3, counter=R4, i-copy=R6.
    let x = base + 2 * SV;
    let i_reg = base + 3 * SV;
    let counter = base + 4 * SV;
    let i_copy = base + 6 * SV;
    println!(
        "base={base:#x} ra_add={ra_add:#x} x@{x:#x}={} i@{i_reg:#x}={} counter@{counter:#x}={}",
        rd_u64(&w, x),
        rd_u64(&w, i_reg),
        rd_u64(&w, counter)
    );
    assert!(
        stack_lo <= counter && counter < stack_hi,
        "counter on stack"
    );

    // ---- Pass 2: the callee (add) frame, captured at its first dispatch (deterministic arena). ----
    let (callee_ci, wb) = {
        let inst2 = svm_run::instantiate(m.clone()).expect("inst2");
        let mut in2 = inst2.debug_attach(SCRIPT.as_bytes().to_vec(), u64::MAX);
        let (_sp, l2, _ci2) = entry_regs(&mut in2, luav);
        let dbp = IrPc {
            module: 0,
            func: luav,
            block: dispatch as usize,
            inst: 0,
        };
        in2.set_breakpoint(dbp);
        let mut found = None;
        for _ in 0..4000 {
            match in2.run_until_stop() {
                Stop::Break {
                    reason: StopReason::Breakpoint,
                    ..
                } => {
                    let wv = in2.read_window(0, CAP).expect("w");
                    let cci = rd_u64(&wv, l2 + L_CI);
                    let cfunc = rd_u64(&wv, cci + CI_FUNC);
                    let ccl = rd_u64(&wv, cfunc);
                    let cproto = rd_u64(&wv, ccl + LCLOSURE_P);
                    if cproto != outer_proto {
                        found = Some((cci, wv));
                        break;
                    }
                }
                o => panic!("callee frame not reached: {o:?}"),
            }
        }
        found.expect("callee frame")
    };
    let callee_func = rd_u64(&wb, callee_ci + CI_FUNC);
    let callee_cl = rd_u64(&wb, callee_func);
    let callee_proto = rd_u64(&wb, callee_cl + LCLOSURE_P);
    let callee_code = rd_u64(&wb, callee_proto + PROTO_CODE);
    let callee_sizecode = rd_i32(&wb, callee_proto + PROTO_SIZECODE) as usize;
    println!(
        "callee: ci={callee_ci:#x} func={callee_func:#x} cl={callee_cl:#x} proto={callee_proto:#x}"
    );
    assert_eq!(callee_func, ra_add, "callee ci->func == the add call ra");

    let slice = |src: &[u8], a: u64, n: usize| src[a as usize..a as usize + n].to_vec();
    let read: Vec<u32> = [
        "luaH_get",
        "luaH_getshortstr",
        "luaH_getstr",
        "luaH_getint",
        "luaV_finishget",
        "luaC_step",
        "luaC_newobj",
        "luaH_new",
        "luaH_resize",
        "luaF_newLclosure",
        "luaF_newCclosure",
        "luaC_barrier_",
        "luaC_barrierback_",
    ]
    .iter()
    .filter_map(|n| byname(&m, n))
    .collect();
    let touch: Vec<u32> = [
        "luaD_precall",
        "precallC",
        "luaD_call",
        "luaD_callnoyield",
        "luaD_poscall",
        "luaD_growstack",
        "luaD_reallocstack",
    ]
    .iter()
    .filter_map(|n| byname(&m, n))
    .collect();

    let cfg = SpecConfig {
        const_overlays: vec![
            (l, slice(&w, l, LUA_STATE_SIZE)),
            (stack_lo, slice(&w, stack_lo, stack_len)),
            (ci, slice(&w, ci, CI_SIZE)),
            (outer_code, slice(&w, outer_code, 4 * outer_sizecode)),
            (outer_cl, slice(&w, outer_cl, 48)),
            (outer_proto, slice(&w, outer_proto, 128)),
            (outer_pp, slice(&w, outer_pp, 8 * outer_sizep)),
            (callee_ci, slice(&wb, callee_ci, CI_SIZE)),
            (callee_cl, slice(&wb, callee_cl, 48)),
            (callee_proto, slice(&wb, callee_proto, 128)),
            (callee_code, slice(&wb, callee_code, 4 * callee_sizecode)),
        ],
        rename: Some((l, l + LUA_STATE_SIZE as u64)),
        rename_extra: vec![
            (stack_lo, stack_hi),
            (ci, ci + CI_SIZE as u64),
            (callee_ci, callee_ci + CI_SIZE as u64),
        ],
        rename_is_private: true,
        rename_seed_from_image: true,
        // The loop-carried cells must be dynamic for the memo to converge (else each folds to a
        // different constant per iteration → unroll): x, i, the FORLOOP counter, and the loop-var copy.
        dynamic_cells: vec![(x, 8), (i_reg, 8), (counter, 8), (i_copy, 8)],
        cut_calls_read_state: read,
        cut_calls_touch_state: touch,
        carry_whole_module: true,
        carry_keep_imports: true,
        indirect_targets_cap: Some(16),
        precall_model: Some(PrecallModel {
            precall,
            ra_arg: 2,
            l_ci_addr: l + L_CI,
            lua_sites: vec![LuaSite {
                ra: ra_add,
                callee_ci,
                pins: vec![(callee_func, callee_cl)],
            }],
            c_sites: vec![],
            poscall: Some(PoscallModel {
                poscall,
                ci_previous_off: CI_PREVIOUS,
                ci_func_off: CI_FUNC,
                tag_off: TAG_OFF,
                // The add result is an integer: reload its value dynamic, keep its tag folded, so the
                // caller's `x` stays integer-tagged and the loop rolls.
                selective: vec![(callee_ci, VNUMINT)],
            }),
        }),
        ..SpecConfig::default()
    };
    let args = [
        SpecArg::ConstI64(sp),
        SpecArg::ConstI64(l as i64),
        SpecArg::ConstI64(ci as i64),
    ];
    let mut residual = specialize_with_config(&m, luav, &args, &cfg)
        .expect("a call-bearing loop rolls (run with svm-peval/trace to see any wall)");
    // `carry_keep_imports` kept the I/O layer bodies but left the manifest empty; re-attach it (and the
    // types the import sigs reference) so the standalone residual verifies.
    residual.imports = m.imports.clone();
    residual.types = m.types.clone();
    svm_verify::verify_module(&residual).expect("residual re-verifies");

    let entry = (residual.funcs.len() - 1) as u32;
    let f = &residual.funcs[entry as usize];
    let brt = f
        .blocks
        .iter()
        .filter(|b| matches!(b.term, Terminator::BrTable { .. }))
        .count();
    let calls: usize = f
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|i| matches!(i, Inst::Call { .. }))
        .count();
    let has_backedge = f.blocks.iter().enumerate().any(|(bi, b)| match &b.term {
        Terminator::Br { target, .. } => (*target as usize) <= bi,
        Terminator::BrIf {
            then_blk, else_blk, ..
        } => (*then_blk as usize) <= bi || (*else_blk as usize) <= bi,
        _ => false,
    });
    println!(
        "ROLLED call loop: {} blocks, br_table={brt}, calls={calls}, params={}, backedge={has_backedge}",
        f.blocks.len(),
        f.params.len()
    );
    // Rolled, not unrolled: dispatch folded, a bounded residual with a loop back-edge, taking the
    // loop-carried cells as dynamic params, and the call surviving as opaque cut call-outs.
    assert_eq!(brt, 0, "the dispatch folded through the in-loop call");
    assert!(
        has_backedge,
        "the loop rolled (a residual back-edge closes it)"
    );
    // Rolled, not unrolled: the residual holds a *single* iteration's worth of cut call-outs (precall,
    // poscall, and the allocator/GC), reached through the back-edge — a 5×-unrolled loop would have ~5×
    // as many. The counter is dynamic, so the residual is independent of the (captured) trip count.
    assert!(
        calls > 0,
        "the in-loop call survives as opaque cut call-outs"
    );
    assert!(
        calls <= 10,
        "one iteration's worth of cut calls (got {calls}) — not a 5×-unrolled body"
    );
    assert!(
        f.blocks.len() <= 200,
        "the loop rolled to a bounded block count (got {})",
        f.blocks.len()
    );
    assert!(
        f.params.len() >= 3,
        "residual takes the loop-carried cells as dynamic params"
    );
    println!(
        "  ✓ slice 3: a loop that calls a Lua function each iteration rolls (br_table=0, bounded)"
    );
}
