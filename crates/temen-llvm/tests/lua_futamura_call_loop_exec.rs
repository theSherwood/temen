//! **A call-bearing loop, executed** — the execution companion to `lua_futamura_call_loop`.
//!
//! ```lua
//! local function add(a, b) return a + b end
//! local x = 0
//! for i = 1, 5 do x = add(x, 3) end
//! print(x)                                 -- 15
//! ```
//!
//! `lua_futamura_call_loop` proves the *rolling* milestone structurally (safepoint-rooted,
//! `dynamic_cells`, bounded residual with a back-edge) but its residual assumes mid-loop entry state,
//! so it can't be embedded wholesale. This test takes the complementary shape: **entry-rooted**, so
//! the constant trip count unrolls (five bodies, each folding through the same `OP_CALL` site into
//! `add` and back via the selective return), and the whole program embeds and runs. Five sequential
//! calls through the **same** call site exercise the shared-`CallInfo` reuse (I71 facet c's
//! same-callee variant): one `LuaSite`, one frame node, five occupancies — the loop-carried `x` is
//! dynamic after the first selective reload, its tag pinned integer, so every iteration's fast-path
//! arithmetic folds while the values flow at runtime.
//!
//! The embedded residual prints `15\n` byte-identically to the interpreter — calls in a loop,
//! execution-correct. Together with `lua_futamura_call_loop` (rolls) this closes slice 3's
//! "structural-only" gap: the same mechanism both rolls and runs.
//!
//! Run: `cargo test -p temen-llvm --test lua_futamura_call_loop_exec -- --nocapture`

use temen_interp::{IrPc, Stop, StopReason, Value};
use temen_ir::{Block, Func, Inst, Module, Terminator, ValType};
use temen_peval::{
    specialize_with_config, LuaSite, PoscallModel, PrecallModel, SpecArg, SpecConfig,
};

const SCRIPT: &str = "local function add(a, b) return a + b end\n\
                      local x = 0\n\
                      for i = 1, 5 do x = add(x, 3) end\n\
                      print(x)\n";
const CAP: usize = 8 << 20;

const L_CI: u64 = 32;
const L_STACK_LAST: u64 = 40;
const L_STACK: u64 = 48;
const LUA_STATE_SIZE: usize = 200;
// The real CallInfo allocation stride (see I71 facet b).
const CI_SIZE: usize = 64;
const CI_FUNC: u64 = 0;
const LCLOSURE_P: u64 = 24;
const PROTO_SIZEK: u64 = 20;
const PROTO_SIZECODE: u64 = 24;
const PROTO_SIZEP: u64 = 32;
const PROTO_K: u64 = 56;
const PROTO_CODE: u64 = 64;
const PROTO_P: u64 = 72;
const VNUMINT: u8 = 0x03;

fn lua_module() -> Module {
    let p = format!(
        "{}/tests/fixtures/lua/lua_eval.ll",
        env!("CARGO_MANIFEST_DIR")
    );
    temen_llvm::translate_ll_path(&p).expect("translate").module
}
fn byname(m: &Module, n: &str) -> Option<u32> {
    m.exports.iter().find(|e| e.name == n).map(|e| e.func)
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
fn precall_site(m: &Module, luav: u32, precall: u32) -> (u32, u32) {
    for (bi, b) in m.funcs[luav as usize].blocks.iter().enumerate() {
        let ra = b.insts.iter().find_map(|i| match i {
            Inst::Call { func, args } if *func == precall => Some(args[2]),
            _ => None,
        });
        if let Some(ra) = ra {
            if matches!(b.term, Terminator::BrIf { .. }) {
                return (bi as u32, ra);
            }
        }
    }
    panic!("no precall branch site");
}
fn entry_capture(insp: &mut temen_interp::Inspector, luav: u32) -> (i64, u64, u64) {
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
fn call_bearing_loop_executes() {
    let m = lua_module();
    let luav = byname(&m, "luaV_execute").expect("luaV_execute");
    let precall = byname(&m, "luaD_precall").expect("precall");
    let poscall = byname(&m, "luaD_poscall").expect("poscall");
    let dispatch = dispatch_block(&m, luav);
    let (pblock, ra_val) = precall_site(&m, luav, precall);

    // ---- Pass 1: entry state + the call-site ras. The loop's five `add` calls all come from ONE
    // site (same ra); the first distinct ra after it is print's. ----
    let inst = temen_run::instantiate(m.clone()).expect("inst");
    let mut insp = inst.debug_attach(SCRIPT.as_bytes().to_vec(), u64::MAX);
    let (sp, l, main_ci) = entry_capture(&mut insp, luav);
    let w = insp.read_window(0, CAP).expect("w");
    let stack_lo = rd_u64(&w, l + L_STACK);
    let stack_hi = rd_u64(&w, l + L_STACK_LAST);
    let stack_len = (stack_hi - stack_lo) as usize;
    let main_func = rd_u64(&w, main_ci + CI_FUNC);
    let main_cl = rd_u64(&w, main_func);
    let main_proto = rd_u64(&w, main_cl + LCLOSURE_P);
    let main_code = rd_u64(&w, main_proto + PROTO_CODE);
    let main_sizecode = rd_i32(&w, main_proto + PROTO_SIZECODE) as usize;
    let main_pp = rd_u64(&w, main_proto + PROTO_P);
    let main_sizep = rd_i32(&w, main_proto + PROTO_SIZEP) as usize;
    let main_k = rd_u64(&w, main_proto + PROTO_K);
    let main_sizek = rd_i32(&w, main_proto + PROTO_SIZEK) as usize;

    let pbp = IrPc {
        module: 0,
        func: luav,
        block: pblock as usize,
        inst: 0,
    };
    insp.set_breakpoint(pbp);
    let mut ras: Vec<u64> = Vec::new();
    while ras.len() < 2 {
        match insp.run_until_stop() {
            Stop::Break {
                reason: StopReason::Breakpoint,
                ..
            } => match insp.read_ir_value(0, ra_val as usize) {
                Some(Value::I64(v)) => {
                    let ra = v as u64;
                    if !ras.contains(&ra) {
                        ras.push(ra);
                    }
                }
                o => panic!("ra: {o:?}"),
            },
            o => panic!("expected precall break: {o:?}"),
        }
    }
    let (ra_add, ra_print) = (ras[0], ras[1]);
    println!("ra_add={ra_add:#x} ra_print={ra_print:#x}");

    // ---- Pass 2: the add frame at its first dispatch. ----
    let (cci, wb) = {
        let inst2 = temen_run::instantiate(m.clone()).expect("inst2");
        let mut in2 = inst2.debug_attach(SCRIPT.as_bytes().to_vec(), u64::MAX);
        let (_s, l2, _c) = entry_capture(&mut in2, luav);
        let dbp = IrPc {
            module: 0,
            func: luav,
            block: dispatch as usize,
            inst: 0,
        };
        in2.set_breakpoint(dbp);
        loop {
            match in2.run_until_stop() {
                Stop::Break {
                    reason: StopReason::Breakpoint,
                    ..
                } => {
                    let wv = in2.read_window(0, CAP).expect("w");
                    let ci = rd_u64(&wv, l2 + L_CI);
                    let f = rd_u64(&wv, ci + CI_FUNC);
                    let cl = rd_u64(&wv, f);
                    let p = rd_u64(&wv, cl + LCLOSURE_P);
                    if p != main_proto {
                        break (ci, wv);
                    }
                }
                o => panic!("callee frame not reached: {o:?}"),
            }
        }
    };
    let cf = rd_u64(&wb, cci + CI_FUNC);
    let ccl = rd_u64(&wb, cf);
    let cp = rd_u64(&wb, ccl + LCLOSURE_P);
    let cc = rd_u64(&wb, cp + PROTO_CODE);
    let csc = rd_i32(&wb, cp + PROTO_SIZECODE) as usize;
    let ck = rd_u64(&wb, cp + PROTO_K);
    let csk = rd_i32(&wb, cp + PROTO_SIZEK) as usize;
    assert_eq!(cf, ra_add, "callee ci->func == the add call ra");
    let ci_previous_off = (0..CI_SIZE as u64)
        .step_by(8)
        .find(|&off| rd_u64(&wb, cci + off) == main_ci)
        .expect("ci->previous");

    let slice = |src: &[u8], a: u64, n: usize| src[a as usize..a as usize + n].to_vec();
    let mut overlays = vec![
        (l, slice(&w, l, LUA_STATE_SIZE)),
        (stack_lo, slice(&w, stack_lo, stack_len)),
        (main_ci, slice(&w, main_ci, CI_SIZE)),
        (main_code, slice(&w, main_code, 4 * main_sizecode)),
        (main_cl, slice(&w, main_cl, 48)),
        (main_proto, slice(&w, main_proto, 128)),
        (main_pp, slice(&w, main_pp, 8 * main_sizep)),
        (cci, slice(&wb, cci, CI_SIZE)),
        (ccl, slice(&wb, ccl, 48)),
        (cp, slice(&wb, cp, 128)),
        (cc, slice(&wb, cc, 4 * csc)),
    ];
    if main_sizek > 0 {
        overlays.push((main_k, slice(&w, main_k, 16 * main_sizek)));
    }
    if csk > 0 && ck != main_k {
        overlays.push((ck, slice(&wb, ck, 16 * csk)));
    }
    let rename_extra = vec![
        (stack_lo, stack_hi),
        (main_ci, main_ci + CI_SIZE as u64),
        (cci, cci + CI_SIZE as u64),
    ];

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
        "luaM_realloc_",
        "luaM_saferealloc_",
        "luaM_malloc_",
        "luaM_growaux_",
        "luaS_resize",
        "luaS_newlstr",
        "luaC_fullgc",
        "luaF_findupval",
        "luaF_closeupval",
        "luaF_close",
        "luaF_initupvals",
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
        const_overlays: overlays,
        rename: Some((l, l + LUA_STATE_SIZE as u64)),
        rename_extra,
        rename_is_private: true,
        rename_seed_from_image: true,
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
                callee_ci: cci,
                pins: vec![(cf, ccl)],
            }],
            c_sites: vec![ra_print],
            poscall: Some(PoscallModel {
                poscall,
                ci_previous_off,
                ci_func_off: CI_FUNC,
                tag_off: 8,
                selective: vec![(cci, VNUMINT)],
            }),
        }),
        ..SpecConfig::default()
    };
    let args = [
        SpecArg::ConstI64(sp),
        SpecArg::ConstI64(l as i64),
        SpecArg::ConstI64(main_ci as i64),
    ];
    let mut r = specialize_with_config(&m, luav, &args, &cfg)
        .expect("project the call-bearing loop from the entry (unrolled)");
    let entry = (r.funcs.len() - 1) as u32;
    let f = &r.funcs[entry as usize];
    let brt = f
        .blocks
        .iter()
        .filter(|b| matches!(b.term, Terminator::BrTable { .. }))
        .count();
    println!(
        "PROJECTED loop-exec: {} blocks, br_table={brt}",
        f.blocks.len()
    );
    assert_eq!(brt, 0, "the dispatch folded through all five in-loop calls");

    // ---- Embed + diff stdout. ----
    r.imports = m.imports.clone();
    r.types = m.types.clone();
    r.funcs[luav as usize] = Func {
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
    temen_verify::verify_module(&r).expect("embedded residual verifies");
    let base = temen_run::run_powerbox(&m, SCRIPT.as_bytes()).expect("baseline");
    let emb = temen_run::run_powerbox(&r, SCRIPT.as_bytes()).expect("embedded");
    println!(
        "baseline={:?} embedded={:?}",
        String::from_utf8_lossy(&base.stdout),
        String::from_utf8_lossy(&emb.stdout)
    );
    assert_eq!(base.stdout, b"15\n", "baseline prints 15");
    assert_eq!(emb.stdout, base.stdout, "residual byte-identical");
    println!("  ✓ a call-bearing loop executes correctly through the projection (15)");
}
