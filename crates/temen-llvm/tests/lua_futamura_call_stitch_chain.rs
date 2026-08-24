//! **Stitching slice 2: a three-residual chain — continuations that stitch onward.**
//!
//! `lua_futamura_call_stitch` composes two residuals at one boundary. The per-function execution
//! model's real claim is that stitches **chain**: a continuation residual is itself an ordinary
//! projection that can stitch to the *next* continuation at *its* next call boundary. Three residuals,
//! two boundaries, one program:
//!
//! ```lua
//! local function add(a, b) return a + b end
//! local function mul(a, b) return a * b end
//! print(add(2, 3) * mul(4, 5))             -- 5 * 20 = 100
//! ```
//!
//! - **Residual B** (entry-rooted): main's prologue → `add`'s `startfunc` → spill + tail-call A1.
//! - **Residual A1** (`add`-rooted): `add`'s body, return into main, main's middle (the `mul` call
//!   setup) → `mul`'s `startfunc` → spill + tail-call A2. A continuation that stitches onward.
//! - **Residual A2** (`mul`-rooted): `mul`'s body, return into main, the register-register `OP_MUL`
//!   (`add`'s folded result `5` × `mul`'s reloaded result), the `print`, and `luaV_execute`'s exit.
//!
//! The two Lua callees are **sequential**, so they reuse ONE cached `CallInfo` (I71 facet c): A1's
//! `mul` site pins the shared node's per-callee fields, and A1/A2 root at the *same* `ci` constant
//! with different dispatch-moment windows. The module is padded with **two** `() -> ()` placeholders
//! (one per continuation); all passes carry identity indices, so each continuation's stitch
//! `ReturnCall` lands on the right final slot when the placeholders are replaced.
//!
//! The chained triple prints `100\n` byte-identically to the interpreter.
//!
//! Run: `cargo test -p temen-llvm --test lua_futamura_call_stitch_chain -- --nocapture`

use temen_interp::{IrPc, Stop, StopReason, Value};
use temen_ir::{Block, Func, Inst, Module, Terminator, ValType};
use temen_peval::{
    specialize_with_config, LuaSite, PoscallModel, PrecallModel, SpecArg, SpecConfig,
};

const SCRIPT: &str = "local function add(a, b) return a + b end\n\
                      local function mul(a, b) return a * b end\n\
                      print(add(2, 3) * mul(4, 5))\n";
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
/// The precall branch block, its `ra` SSA index, and the startfunc arm (see the stitch test).
fn precall_site(m: &Module, luav: u32, precall: u32) -> (u32, u32, u32) {
    for (bi, b) in m.funcs[luav as usize].blocks.iter().enumerate() {
        let ra = b.insts.iter().find_map(|i| match i {
            Inst::Call { func, args } if *func == precall => Some(args[2]),
            _ => None,
        });
        if let Some(ra) = ra {
            if let Terminator::BrIf {
                then_blk, else_blk, ..
            } = b.term
            {
                let is_startfunc = |t: u32| {
                    matches!(
                        m.funcs[luav as usize].blocks[t as usize].term,
                        Terminator::Br { .. }
                    ) && m.funcs[luav as usize].blocks[t as usize].insts.is_empty()
                };
                let sf = if is_startfunc(then_blk) {
                    then_blk
                } else if is_startfunc(else_blk) {
                    else_blk
                } else {
                    panic!("neither precall arm looks like startfunc");
                };
                return (bi as u32, ra, sf);
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
fn cut_lists(m: &Module) -> (Vec<u32>, Vec<u32>) {
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
    .filter_map(|n| byname(m, n))
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
    .filter_map(|n| byname(m, n))
    .collect();
    (read, touch)
}

/// One callee's occupancy of the shared frame node at its own first dispatch.
struct Callee {
    func: u64,
    cl: u64,
    proto: u64,
    code: u64,
    sizecode: usize,
    k: u64,
    sizek: usize,
    w: Vec<u8>,
}

#[test]
fn three_residuals_chain() {
    let m = lua_module();
    let luav = byname(&m, "luaV_execute").expect("luaV_execute");
    let precall = byname(&m, "luaD_precall").expect("precall");
    let poscall = byname(&m, "luaD_poscall").expect("poscall");
    let dispatch = dispatch_block(&m, luav);
    let (pblock, ra_val, startfunc) = precall_site(&m, luav, precall);

    // ---- Capture: entry state + the three ras + both callee occupancies (shared ci). ----
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
    let mut ras = Vec::new();
    while ras.len() < 3 {
        match insp.run_until_stop() {
            Stop::Break {
                reason: StopReason::Breakpoint,
                ..
            } => match insp.read_ir_value(0, ra_val as usize) {
                Some(Value::I64(v)) => ras.push(v as u64),
                o => panic!("ra: {o:?}"),
            },
            o => panic!("expected precall break: {o:?}"),
        }
    }
    let (ra_add, ra_mul, ra_print) = (ras[0], ras[1], ras[2]);

    let (cci, callees) = {
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
        let mut ci_shared = 0u64;
        let mut seen: Vec<Callee> = Vec::new();
        for _ in 0..8000 {
            if seen.len() == 2 {
                break;
            }
            match in2.run_until_stop() {
                Stop::Break {
                    reason: StopReason::Breakpoint,
                    ..
                } => {
                    let wv = in2.read_window(0, CAP).expect("w");
                    let ci = rd_u64(&wv, l2 + L_CI);
                    let func = rd_u64(&wv, ci + CI_FUNC);
                    let cl = rd_u64(&wv, func);
                    let proto = rd_u64(&wv, cl + LCLOSURE_P);
                    if proto != main_proto && !seen.iter().any(|c| c.proto == proto) {
                        ci_shared = ci;
                        seen.push(Callee {
                            func,
                            cl,
                            proto,
                            code: rd_u64(&wv, proto + PROTO_CODE),
                            sizecode: rd_i32(&wv, proto + PROTO_SIZECODE) as usize,
                            k: rd_u64(&wv, proto + PROTO_K),
                            sizek: rd_i32(&wv, proto + PROTO_SIZEK) as usize,
                            w: wv,
                        });
                    }
                }
                o => panic!("callee frames not reached: {o:?}"),
            }
        }
        assert_eq!(seen.len(), 2, "both callee occupancies captured");
        (ci_shared, seen)
    };
    let find = |ra: u64| callees.iter().find(|c| c.func == ra).expect("callee");
    let (add, mul) = (find(ra_add), find(ra_mul));
    let savedpc_off = (0..CI_SIZE as u64)
        .step_by(8)
        .find(|&off| rd_u64(&add.w, cci + off) == add.code)
        .expect("ci->savedpc");
    let ci_previous_off = (0..CI_SIZE as u64)
        .step_by(8)
        .find(|&off| rd_u64(&add.w, cci + off) == main_ci)
        .expect("ci->previous");

    let slice = |src: &[u8], a: u64, n: usize| src[a as usize..a as usize + n].to_vec();
    let (read, touch) = cut_lists(&m);
    // Static program data shared by every pass.
    let statics = |v: &mut Vec<(u64, Vec<u8>)>| {
        v.push((main_code, slice(&w, main_code, 4 * main_sizecode)));
        v.push((main_cl, slice(&w, main_cl, 48)));
        v.push((main_proto, slice(&w, main_proto, 128)));
        v.push((main_pp, slice(&w, main_pp, 8 * main_sizep)));
        if main_sizek > 0 {
            v.push((main_k, slice(&w, main_k, 16 * main_sizek)));
        }
        for c in [add, mul] {
            v.push((c.cl, slice(&c.w, c.cl, 48)));
            v.push((c.proto, slice(&c.w, c.proto, 128)));
            v.push((c.code, slice(&c.w, c.code, 4 * c.sizecode)));
            if c.sizek > 0 && c.k != main_k {
                v.push((c.k, slice(&c.w, c.k, 16 * c.sizek)));
            }
        }
    };
    // Mutable state (L, stack, main ci, the shared callee ci node) from a given moment's window.
    let mutables = |v: &mut Vec<(u64, Vec<u8>)>, win: &[u8], with_cci: bool| {
        v.push((l, slice(win, l, LUA_STATE_SIZE)));
        v.push((stack_lo, slice(win, stack_lo, stack_len)));
        v.push((main_ci, slice(win, main_ci, CI_SIZE)));
        if with_cci {
            v.push((cci, slice(win, cci, CI_SIZE)));
        }
    };
    let rename_extra_full = vec![
        (stack_lo, stack_hi),
        (main_ci, main_ci + CI_SIZE as u64),
        (cci, cci + CI_SIZE as u64),
    ];
    // Per-site pins for the shared node (I71 facet c): the closure slot + the two per-callee fields.
    let site = |ra: u64, c: &Callee| LuaSite {
        ra,
        callee_ci: cci,
        pins: vec![
            (c.func, c.cl),
            (cci + CI_FUNC, ra),
            (cci + savedpc_off, c.code),
        ],
    };
    let pm = |lua_sites: Vec<LuaSite>, with_poscall: bool| PrecallModel {
        precall,
        ra_arg: 2,
        l_ci_addr: l + L_CI,
        lua_sites,
        c_sites: vec![ra_print],
        poscall: with_poscall.then(|| PoscallModel {
            poscall,
            ci_previous_off,
            ci_func_off: CI_FUNC,
            tag_off: 8,
            selective: vec![(cci, VNUMINT)],
        }),
    };

    // Two placeholder handlers: final slot N = A1 (add-rooted), N+1 = A2 (mul-rooted).
    let n = m.funcs.len() as u32;
    let (h_a1, h_a2) = (n, n + 1);
    let mut mp = m.clone();
    for _ in 0..2 {
        mp.funcs.push(Func {
            params: vec![],
            results: vec![],
            blocks: vec![Block {
                params: vec![],
                insts: vec![],
                term: Terminator::Return(vec![]),
            }],
        });
    }
    let base_cfg = |overlays: Vec<(u64, Vec<u8>)>,
                    rename_extra: Vec<(u64, u64)>,
                    model: PrecallModel,
                    deopt: Option<u32>| SpecConfig {
        const_overlays: overlays,
        rename: Some((l, l + LUA_STATE_SIZE as u64)),
        rename_extra,
        rename_is_private: true,
        rename_seed_from_image: true,
        cut_calls_read_state: read.clone(),
        cut_calls_touch_state: touch.clone(),
        carry_whole_module: true,
        carry_keep_imports: true,
        indirect_targets_cap: Some(16),
        deopt_targets: deopt.map(|_| vec![(luav, startfunc)]).unwrap_or_default(),
        deopt_handler: deopt,
        precall_model: Some(model),
        ..SpecConfig::default()
    };
    let root = |ci: u64| {
        [
            SpecArg::ConstI64(sp),
            SpecArg::ConstI64(l as i64),
            SpecArg::ConstI64(ci as i64),
        ]
    };

    // ---- Residual A2 (mul-rooted, runs to program exit). ----
    let mut ov = Vec::new();
    mutables(&mut ov, &mul.w, true);
    statics(&mut ov);
    let r_a2 = specialize_with_config(
        &mp,
        luav,
        &root(cci),
        &base_cfg(ov, rename_extra_full.clone(), pm(vec![], true), None),
    )
    .expect("residual A2: mul-rooted tail");
    let entry_a2 = r_a2.funcs.last().expect("entry A2").clone();

    // ---- Residual A1 (add-rooted, stitches onward at mul's startfunc). ----
    let mut ov = Vec::new();
    mutables(&mut ov, &add.w, true);
    statics(&mut ov);
    let r_a1 = specialize_with_config(
        &mp,
        luav,
        &root(cci),
        &base_cfg(
            ov,
            rename_extra_full.clone(),
            pm(vec![site(ra_mul, mul)], true),
            Some(h_a2),
        ),
    )
    .expect("residual A1: add-rooted continuation that stitches onward");
    let entry_a1 = r_a1.funcs.last().expect("entry A1").clone();

    // ---- Residual B (entry-rooted, stitches at add's startfunc). ----
    let mut ov = Vec::new();
    mutables(&mut ov, &w, false);
    // add's proto for main's OP_CLOSURE upvalue-loop bound (see the stitch test) — `statics` has it.
    statics(&mut ov);
    let r_b = specialize_with_config(
        &mp,
        luav,
        &root(main_ci),
        &base_cfg(
            ov,
            vec![(stack_lo, stack_hi), (main_ci, main_ci + CI_SIZE as u64)],
            pm(vec![site(ra_add, add)], false),
            Some(h_a1),
        ),
    )
    .expect("residual B: entry-rooted caller");
    let mut r = r_b;
    let entry_b = (r.funcs.len() - 1) as u32;

    for (nm, f, h) in [
        ("A1", &entry_a1, h_a2),
        ("A2", &entry_a2, u32::MAX),
        ("B", &r.funcs[entry_b as usize].clone(), h_a1),
    ] {
        assert!(
            f.params.is_empty() && f.results.is_empty(),
            "residual {nm} is () -> ()"
        );
        let stitches = f
            .blocks
            .iter()
            .filter(|b| matches!(b.term, Terminator::ReturnCall { func, .. } if func == h))
            .count();
        println!(
            "residual {nm}: {} blocks, {} stitch tail-call(s)",
            f.blocks.len(),
            stitches
        );
        if h != u32::MAX {
            assert!(
                stitches >= 1,
                "residual {nm} must stitch to its continuation"
            );
        }
    }

    // ---- Compose: placeholders → continuations; embed B; run the chain. ----
    r.funcs[h_a1 as usize] = entry_a1;
    r.funcs[h_a2 as usize] = entry_a2;
    r.imports = m.imports.clone();
    r.types = m.types.clone();
    r.funcs[luav as usize] = Func {
        params: vec![ValType::I64, ValType::I64, ValType::I64],
        results: vec![],
        blocks: vec![Block {
            params: vec![ValType::I64, ValType::I64, ValType::I64],
            insts: vec![Inst::Call {
                func: entry_b,
                args: vec![],
            }],
            term: Terminator::Return(vec![]),
        }],
    };
    temen_verify::verify_module(&r).expect("chained module verifies");
    let base = temen_run::run_powerbox(&m, SCRIPT.as_bytes()).expect("baseline");
    let emb = temen_run::run_powerbox(&r, SCRIPT.as_bytes()).expect("chained");
    println!(
        "baseline={:?} chained={:?}",
        String::from_utf8_lossy(&base.stdout),
        String::from_utf8_lossy(&emb.stdout)
    );
    assert_eq!(base.stdout, b"100\n", "baseline prints 100");
    assert_eq!(emb.stdout, base.stdout, "chained triple byte-identical");
    println!("  ✓ three residuals, two stitches, one program: B → A1 → A2 → 100");
}
