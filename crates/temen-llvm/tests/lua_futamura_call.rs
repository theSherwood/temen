//! **Project *through* a Lua-to-Lua call** (the precall post-state model — slice 1).
//!
//! `lua_futamura_print.rs` folds a whole `print`-only chunk but *deopts* the one edge a real script
//! leans on: the `OP_CALL` "callee was a Lua function" re-dispatch. This test exercises
//! [`temen_peval::PrecallModel`], which instead binds the (cut) `luaD_precall`'s result statically per
//! call site so that branch **folds** — a Lua call site projects into the callee's own dispatch, a C
//! call site (`print`) folds to the continue edge with the C function running inside the opaque call.
//!
//! Script: `local function add(a,b) return a+b end  print(add(40,2))` → `42\n`. Two calls go through
//! the shared `OP_CALL` handler: `add` (Lua — projected) and `print` (C — folded in place). The call
//! site is keyed by `ra` (the callee's stack slot, constant in the fold); the callee closure itself is
//! allocated by `OP_CLOSURE` at runtime (opaque) so it can't be read statically — instead the site
//! *pins* the closure pointer (captured, deterministic) so the callee-dispatch setup `cl→proto→code`
//! folds.
//!
//! Slice 1 projects the **call-in** and the callee's dispatch; the callee's *return* (`poscall`) is not
//! yet modeled (slice 2) — it bails to the carried baseline. Correctness is the gate: the embedded
//! residual must print byte-identically to the interpreter.
//!
//! Run: `cargo test -p temen-llvm --test lua_futamura_call -- --nocapture`

use temen_interp::{IrPc, Stop, StopReason, Value};
use temen_ir::{Block, Func, Inst, Module, Terminator, ValType};
use temen_peval::{
    specialize_with_config, LuaSite, PoscallModel, PrecallModel, SpecArg, SpecConfig,
};

const SCRIPT: &str = "local function add(a, b) return a + b end\nprint(add(40, 2))\n";
const CAPTURE_LEN: usize = 8 << 20;

const L_STACK_LAST: u64 = 40;
const L_STACK: u64 = 48;
const L_CI: u64 = 32;
const LUA_STATE_SIZE: usize = 200;
// The real CallInfo allocation stride (adjacent ci nodes are 64 apart). An over-slice makes one
// frame's ci overlay swallow the next node with stale bytes — the I71(b) root cause.
const CI_SIZE: usize = 64;
const CI_FUNC: u64 = 0;
const LCLOSURE_P: u64 = 24;
const PROTO_SIZECODE: u64 = 24;
const PROTO_SIZEP: u64 = 32;
const PROTO_CODE: u64 = 64;
const PROTO_P: u64 = 72;

fn lua_module() -> Module {
    let path = format!(
        "{}/tests/fixtures/lua/lua_eval.ll",
        env!("CARGO_MANIFEST_DIR")
    );
    temen_llvm::translate_ll_path(&path)
        .expect("translate")
        .module
}
fn byname(m: &Module, n: &str) -> Option<u32> {
    m.exports.iter().find(|e| e.name == n).map(|e| e.func)
}
fn luav_execute(m: &Module) -> u32 {
    byname(m, "luaV_execute").expect("luaV_execute export")
}
fn rd_u64(w: &[u8], a: u64) -> u64 {
    u64::from_le_bytes(w[a as usize..a as usize + 8].try_into().unwrap())
}
fn rd_i32(w: &[u8], a: u64) -> i32 {
    i32::from_le_bytes(w[a as usize..a as usize + 4].try_into().unwrap())
}

/// The dispatch header: the block whose terminator is the widest `br_table`.
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

/// The `OP_CALL` handler block: it calls `luaD_precall` and branches on the result (`newci != NULL`)
/// — the Lua-function edge re-enters the dispatch loop (via block 1, `startfunc`). Returns
/// `(block, ra_value_index)` — the SSA value index of the precall's `ra` argument (arg index 2:
/// `precall(sp, L, ra, nres)`).
fn precall_site(m: &Module, luav: u32, precall: u32) -> (u32, u32) {
    for (bi, b) in m.funcs[luav as usize].blocks.iter().enumerate() {
        let ra = b.insts.iter().find_map(|i| match i {
            Inst::Call { func, args } if *func == precall => Some(args[2]),
            _ => None,
        });
        let Some(ra) = ra else { continue };
        if matches!(b.term, Terminator::BrIf { .. }) {
            return (bi as u32, ra);
        }
    }
    panic!("no precall branch site found");
}

/// Capture `(sp, L, ci, window)` at `luaV_execute` entry.
fn capture_entry(m: &Module, luav: u32) -> (i64, u64, u64, Vec<u8>) {
    let inst = temen_run::instantiate(m.clone()).expect("instantiate");
    let mut insp = inst.debug_attach(SCRIPT.as_bytes().to_vec(), u64::MAX);
    insp.set_breakpoint(IrPc {
        module: 0,
        func: luav,
        block: 0,
        inst: 0,
    });
    match insp.run_until_stop() {
        Stop::Break { .. } => {
            let g = |i| match insp.read_ir_value(0, i) {
                Some(Value::I64(v)) => v,
                o => panic!("read v{i}: {o:?}"),
            };
            let w = insp.read_window(0, CAPTURE_LEN).expect("window");
            (g(0), g(1) as u64, g(2) as u64, w)
        }
        o => panic!("no entry break: {o:?}"),
    }
}

/// The `ra` (callee stack slot) at each of the first two `precall` hits — `add` then `print`.
fn capture_ra_values(m: &Module, luav: u32, pblock: u32, ra_val: u32) -> (u64, u64) {
    let inst = temen_run::instantiate(m.clone()).expect("instantiate");
    let mut insp = inst.debug_attach(SCRIPT.as_bytes().to_vec(), u64::MAX);
    insp.set_breakpoint(IrPc {
        module: 0,
        func: luav,
        block: pblock as usize,
        inst: 0,
    });
    let mut ras = Vec::new();
    while ras.len() < 2 {
        match insp.run_until_stop() {
            Stop::Break {
                reason: StopReason::Breakpoint,
                ..
            } => match insp.read_ir_value(0, ra_val as usize) {
                Some(Value::I64(v)) => ras.push(v as u64),
                o => panic!("read ra v{ra_val}: {o:?}"),
            },
            o => panic!("expected precall break, got {o:?}"),
        }
    }
    (ras[0], ras[1])
}

/// Run to the `add` callee's first dispatch and capture `(callee_ci, window)`.
fn capture_callee_frame(m: &Module, luav: u32, dispatch: u32, outer_proto: u64) -> (u64, Vec<u8>) {
    let inst = temen_run::instantiate(m.clone()).expect("instantiate");
    let mut insp = inst.debug_attach(SCRIPT.as_bytes().to_vec(), u64::MAX);
    // Grab L once (stable) from the entry break.
    insp.set_breakpoint(IrPc {
        module: 0,
        func: luav,
        block: 0,
        inst: 0,
    });
    let l = match insp.run_until_stop() {
        Stop::Break { .. } => match insp.read_ir_value(0, 1) {
            Some(Value::I64(v)) => v as u64,
            o => panic!("L: {o:?}"),
        },
        o => panic!("no entry break: {o:?}"),
    };
    insp.clear_breakpoint(IrPc {
        module: 0,
        func: luav,
        block: 0,
        inst: 0,
    });
    insp.set_breakpoint(IrPc {
        module: 0,
        func: luav,
        block: dispatch as usize,
        inst: 0,
    });
    for _ in 0..2000 {
        match insp.run_until_stop() {
            Stop::Break {
                reason: StopReason::Breakpoint,
                ..
            } => {
                let w = insp.read_window(0, CAPTURE_LEN).expect("window");
                let ci = rd_u64(&w, l + L_CI);
                let func = rd_u64(&w, ci + CI_FUNC);
                let cl = rd_u64(&w, func);
                let proto = rd_u64(&w, cl + LCLOSURE_P);
                if proto != outer_proto {
                    return (ci, w);
                }
            }
            o => panic!("dispatch never reached callee frame: {o:?}"),
        }
    }
    panic!("callee frame not reached in 2000 dispatch hits");
}

/// Everything the two projections share: the module (with a pristine baseline copy of `luaV_execute`
/// appended), the discovered blocks/indices, and the captured overlays + call-site addresses.
struct Cap {
    m: Module,
    luav: u32,
    baseline: u32,
    precall: u32,
    poscall: u32,
    post_return: u32,
    sp: i64,
    l: u64,
    outer_ci: u64,
    ra_add: u64,
    ra_print: u64,
    callee_ci: u64,
    callee_func: u64,
    callee_cl: u64,
    ci_previous_off: u64,
    /// The full `const_overlays` list, `rename`, and `rename_extra` — identical for both projections.
    overlays: Vec<(u64, Vec<u8>)>,
    rename: (u64, u64),
    rename_extra: Vec<(u64, u64)>,
    read: Vec<u32>,
    touch: Vec<u32>,
}

fn capture_all() -> Cap {
    let mut m = lua_module();
    let luav = luav_execute(&m);
    // Append a pristine copy of `luaV_execute`: the residual replaces index `luav`, so a deopt handler
    // (slice-1 return) must resume *this* copy, not the residual (which would recurse forever).
    let baseline = m.funcs.len() as u32;
    m.funcs.push(m.funcs[luav as usize].clone());
    let precall = byname(&m, "luaD_precall").expect("luaD_precall");
    let poscall = byname(&m, "luaD_poscall").expect("luaD_poscall");
    let dispatch = dispatch_block(&m, luav);
    let (pblock, ra_val) = precall_site(&m, luav, precall);
    // The shared post-return block: where every `luaD_poscall` block branches (the "return to caller or
    // exit luaV_execute" logic).
    let post_return = {
        let mut t = None;
        for b in &m.funcs[luav as usize].blocks {
            if b.insts
                .iter()
                .any(|i| matches!(i, Inst::Call { func, .. } if *func == poscall))
            {
                if let Terminator::Br { target, .. } = b.term {
                    t = Some(target);
                    break;
                }
            }
        }
        t.expect("a poscall block branching to the post-return block")
    };
    println!("luaV_execute=f{luav} precall=f{precall} poscall=f{poscall} dispatch=block{dispatch} precall_site=block{pblock} ra=v{ra_val} post_return=block{post_return}");

    let (sp, l, outer_ci, w) = capture_entry(&m, luav);
    let stack_lo = rd_u64(&w, l + L_STACK);
    let stack_hi = rd_u64(&w, l + L_STACK_LAST);
    let stack_len = (stack_hi - stack_lo) as usize;
    let outer_func = rd_u64(&w, outer_ci + CI_FUNC);
    let outer_cl = rd_u64(&w, outer_func);
    let outer_proto = rd_u64(&w, outer_cl + LCLOSURE_P);
    let outer_code = rd_u64(&w, outer_proto + PROTO_CODE);
    let outer_sizecode = rd_i32(&w, outer_proto + PROTO_SIZECODE) as usize;
    // The sub-proto pointer array (`outer_proto->p`): `OP_CLOSURE` reads `p = proto->p[bx]` and folds its
    // `sizeupvalues` to bound the upvalue-init loop; without this overlay that loop unrolls forever.
    let outer_pp = rd_u64(&w, outer_proto + PROTO_P);
    let outer_sizep = rd_i32(&w, outer_proto + PROTO_SIZEP) as usize;

    let (ra_add, ra_print) = capture_ra_values(&m, luav, pblock, ra_val);
    println!("ra_add={ra_add:#x} ra_print={ra_print:#x}");

    let (callee_ci, wb) = capture_callee_frame(&m, luav, dispatch, outer_proto);
    let callee_func = rd_u64(&wb, callee_ci + CI_FUNC);
    let callee_cl = rd_u64(&wb, callee_func);
    let callee_proto = rd_u64(&wb, callee_cl + LCLOSURE_P);
    let callee_code = rd_u64(&wb, callee_proto + PROTO_CODE);
    let callee_sizecode = rd_i32(&wb, callee_proto + PROTO_SIZECODE) as usize;
    assert_eq!(
        callee_func, ra_add,
        "the callee ci->func should be the add call's ra slot"
    );
    // Discover the CallInfo `previous` offset: the callee frame's `previous` points at the outer ci.
    let ci_previous_off = (0..CI_SIZE as u64)
        .step_by(8)
        .find(|&off| rd_u64(&wb, callee_ci + off) == outer_ci)
        .expect("callee ci->previous should equal the outer ci");
    println!(
        "callee: ci={callee_ci:#x} func_slot={callee_func:#x} cl={callee_cl:#x} proto={callee_proto:#x} code={callee_code:#x} sizecode={callee_sizecode} ci_previous_off={ci_previous_off}"
    );

    let slice = |src: &[u8], a: u64, n: usize| src[a as usize..a as usize + n].to_vec();
    let overlays = vec![
        (l, slice(&w, l, LUA_STATE_SIZE)),
        (stack_lo, slice(&w, stack_lo, stack_len)),
        (outer_ci, slice(&w, outer_ci, CI_SIZE)),
        (outer_code, slice(&w, outer_code, 4 * outer_sizecode)),
        (outer_cl, slice(&w, outer_cl, 48)),
        (outer_proto, slice(&w, outer_proto, 128)),
        (outer_pp, slice(&w, outer_pp, 8 * outer_sizep)),
        // Callee (add) frame — captured post-precall from the profiling run (deterministic arena).
        (callee_ci, slice(&wb, callee_ci, CI_SIZE)),
        (callee_cl, slice(&wb, callee_cl, 48)),
        (callee_proto, slice(&wb, callee_proto, 128)),
        (callee_code, slice(&wb, callee_code, 4 * callee_sizecode)),
    ];
    let read_names = [
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
        // GC write barriers — fired when the new `add` closure is stored; their `reallymarkobject`
        // marker is recursive and must stay opaque (a non-moving mark: spill for the scan, no reload).
        "luaC_barrier_",
        "luaC_barrierback_",
    ];
    let touch_names = [
        "luaD_precall",
        "precallC",
        "luaD_call",
        "luaD_callnoyield",
        "luaD_poscall",
        "luaD_growstack",
        "luaD_reallocstack",
    ];
    let read = read_names.iter().filter_map(|n| byname(&m, n)).collect();
    let touch = touch_names.iter().filter_map(|n| byname(&m, n)).collect();

    Cap {
        m,
        luav,
        baseline,
        precall,
        poscall,
        post_return,
        sp,
        l,
        outer_ci,
        ra_add,
        ra_print,
        callee_ci,
        callee_func,
        callee_cl,
        ci_previous_off,
        overlays,
        rename: (l, l + LUA_STATE_SIZE as u64),
        rename_extra: vec![
            (stack_lo, stack_hi),
            (outer_ci, outer_ci + CI_SIZE as u64),
            (callee_ci, callee_ci + CI_SIZE as u64),
        ],
        read,
        touch,
    }
}

/// Embed `residual`'s specialized `entry` in place of `luaV_execute` and diff stdout against baseline.
fn embed_and_diff(mut residual: Module, src: &Module, luav: u32, entry: u32) {
    residual.imports = src.imports.clone();
    residual.types = src.types.clone();
    residual.funcs[luav as usize] = Func {
        params: vec![ValType::I64, ValType::I64, ValType::I64],
        results: vec![],
        blocks: vec![Block {
            params: vec![ValType::I64, ValType::I64, ValType::I64],
            insts: vec![Inst::Call {
                func: entry,
                args: vec![],
            }],
            term: Terminator::Return(vec![]),
        }],
    };
    temen_verify::verify_module(&residual).expect("embedded residual verifies");
    let baseline = temen_run::run_powerbox(src, SCRIPT.as_bytes()).expect("baseline run");
    let embedded =
        temen_run::run_powerbox(&residual, SCRIPT.as_bytes()).expect("embedded residual run");
    println!(
        "baseline stdout={:?}  embedded stdout={:?}",
        String::from_utf8_lossy(&baseline.stdout),
        String::from_utf8_lossy(&embedded.stdout)
    );
    assert_eq!(baseline.stdout, b"42\n", "baseline prints 42");
    assert_eq!(
        embedded.stdout, baseline.stdout,
        "the embedded residual must print byte-identically to the interpreter"
    );
}

fn brtable_count(f: &Func) -> usize {
    f.blocks
        .iter()
        .filter(|b| matches!(b.term, Terminator::BrTable { .. }))
        .count()
}

/// Slice 1: the call-in projects (dispatch folds through the `add` call into the callee), the callee's
/// *return* bails to the pristine baseline copy.
#[test]
fn project_through_a_lua_call() {
    let c = capture_all();
    let cfg = SpecConfig {
        const_overlays: c.overlays.clone(),
        rename: Some(c.rename),
        rename_extra: c.rename_extra.clone(),
        rename_is_private: true,
        rename_seed_from_image: true,
        cut_calls_read_state: c.read.clone(),
        cut_calls_touch_state: c.touch.clone(),
        // Slice 1 does not model the callee's return (`poscall`): the post-return block bails to the
        // carried baseline, which resumes from the written-back window state and finishes the run.
        deopt_targets: vec![(c.luav, c.post_return)],
        deopt_handler: Some(c.baseline),
        carry_whole_module: true,
        carry_keep_imports: true,
        indirect_targets_cap: Some(16),
        precall_model: Some(PrecallModel {
            precall: c.precall,
            ra_arg: 2,
            l_ci_addr: c.l + L_CI,
            lua_sites: vec![LuaSite {
                ra: c.ra_add,
                callee_ci: c.callee_ci,
                pins: vec![(c.callee_func, c.callee_cl)],
            }],
            c_sites: vec![c.ra_print],
            poscall: None,
        }),
        ..SpecConfig::default()
    };
    let args = [
        SpecArg::ConstI64(c.sp),
        SpecArg::ConstI64(c.l as i64),
        SpecArg::ConstI64(c.outer_ci as i64),
    ];
    let residual =
        specialize_with_config(&c.m, c.luav, &args, &cfg).expect("project through the lua call");
    let entry = (residual.funcs.len() - 1) as u32;
    let f = &residual.funcs[entry as usize];
    let calls_precall = f
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .any(|i| matches!(i, Inst::Call { func, .. } if *func == c.precall));
    println!(
        "SLICE 1 PROJECTED: entry {} blocks, br_table={}",
        f.blocks.len(),
        brtable_count(f)
    );
    assert_eq!(
        brtable_count(f),
        0,
        "the dispatch folded through the Lua call (no br_table left)"
    );
    assert!(
        calls_precall,
        "the opaque precall cut survives as a residual call"
    );
    embed_and_diff(residual, &c.m, c.luav, entry);
    println!("  ✓ slice 1: projected through the Lua call-in — stdout byte-identical");
}

/// Slice 2: model `poscall` so the callee *return* composes into the caller — no deopt, no baseline
/// copy needed. The whole `print(add(40,2))` folds to a single dispatch-free residual.
#[test]
fn project_through_a_lua_call_and_return() {
    let c = capture_all();
    let cfg = SpecConfig {
        const_overlays: c.overlays.clone(),
        rename: Some(c.rename),
        rename_extra: c.rename_extra.clone(),
        rename_is_private: true,
        rename_seed_from_image: true,
        cut_calls_read_state: c.read.clone(),
        cut_calls_touch_state: c.touch.clone(),
        // No deopt — the return is modeled (poscall pops the frame in the abstract state).
        carry_whole_module: true,
        carry_keep_imports: true,
        indirect_targets_cap: Some(16),
        precall_model: Some(PrecallModel {
            precall: c.precall,
            ra_arg: 2,
            l_ci_addr: c.l + L_CI,
            lua_sites: vec![LuaSite {
                ra: c.ra_add,
                callee_ci: c.callee_ci,
                pins: vec![(c.callee_func, c.callee_cl)],
            }],
            c_sites: vec![c.ra_print],
            poscall: Some(PoscallModel {
                poscall: c.poscall,
                ci_previous_off: c.ci_previous_off,
                ci_func_off: CI_FUNC,
                tag_off: 8,
                selective: vec![],
            }),
        }),
        ..SpecConfig::default()
    };
    let args = [
        SpecArg::ConstI64(c.sp),
        SpecArg::ConstI64(c.l as i64),
        SpecArg::ConstI64(c.outer_ci as i64),
    ];
    let residual = specialize_with_config(&c.m, c.luav, &args, &cfg)
        .expect("project through the lua call and return");
    let entry = (residual.funcs.len() - 1) as u32;
    let f = &residual.funcs[entry as usize];
    let returns = f
        .blocks
        .iter()
        .filter(|b| matches!(b.term, Terminator::Return(_)))
        .count();
    println!(
        "SLICE 2 PROJECTED: entry {} blocks, br_table={}, returns={returns}",
        f.blocks.len(),
        brtable_count(f)
    );
    assert_eq!(
        brtable_count(f),
        0,
        "the dispatch folded through the call and the return"
    );
    assert!(
        returns > 0,
        "the residual returns from luaV_execute (the modeled return reached the entry frame)"
    );
    embed_and_diff(residual, &c.m, c.luav, entry);
    println!("  ✓ slice 2: projected through the Lua call-in AND return — stdout byte-identical");
}
