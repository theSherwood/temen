//! **Tail calls: `return f(x)` through `luaD_pretailcall`** — the frame-reuse call shape.
//!
//! ```lua
//! local function inner(x) return x + 1 end
//! local function outer(y) return inner(y) end
//! print(outer(10))                          -- inner(10) = 11
//! ```
//!
//! `outer`'s `return inner(y)` compiles to `OP_TAILCALL` → `luaD_pretailcall`, which does NOT push a
//! frame: it moves the callee closure and arguments down onto the **current** frame (`ci` unchanged,
//! `ci->func`'s slot now holds `inner`'s closure, `savedpc` re-aimed at `inner`'s code) and the
//! eventual return pops straight to the caller's caller. The new
//! [`svm_peval::PretailcallModel`] mirrors the precall model for this shape: at a matching `ra` the
//! cut's `int` result binds negative (the "Lua callee, frame moved" arm folds to the re-dispatch
//! edge), `L->ci` stays on the reused node, and the site's pins install the moved closure slot and
//! the callee `savedpc` — the same per-occupancy pinning as a shared-`CallInfo` sequential site
//! (I71 facet c). The tail-callee's result lands directly in the original call slot, so `main`
//! resumes exactly as if `outer` had returned it.
//!
//! The embedded residual prints `11` byte-identically to the interpreter.
//!
//! Run: `cargo test -p svm-llvm --test lua_futamura_call_tail -- --nocapture`

use svm_interp::{IrPc, Stop, StopReason, Value};
use svm_ir::{Block, Func, Inst, Module, Terminator, ValType};
use svm_peval::{
    specialize_with_config, LuaSite, PoscallModel, PrecallModel, PretailcallModel, SpecArg,
    SpecConfig, TailSite,
};

const SCRIPT: &str = "local function inner(x) return x + 1 end\n\
                      local function outer(y) return inner(y) end\n\
                      print(outer(10))\n";
const CAP: usize = 8 << 20;

const L_CI: u64 = 32;
const L_STACK_LAST: u64 = 40;
const L_STACK: u64 = 48;
const LUA_STATE_SIZE: usize = 200;
// The real CallInfo allocation stride (adjacent ci nodes are 64 apart). An over-slice makes one
// frame's ci overlay swallow the next node with stale bytes — the I71(b) root cause.
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
const TAG_OFF: u64 = 8;

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
/// The `luaD_pretailcall` branch block + the SSA value index of its `ra` (arg 3 of
/// `pretailcall(sp, L, ci, ra, narg1, delta)`).
fn pretail_site(m: &Module, luav: u32, pretail: u32) -> (u32, u32) {
    for (bi, b) in m.funcs[luav as usize].blocks.iter().enumerate() {
        let ra = b.insts.iter().find_map(|i| match i {
            Inst::Call { func, args } if *func == pretail => Some(args[3]),
            _ => None,
        });
        if let Some(ra) = ra {
            if matches!(b.term, Terminator::BrIf { .. }) {
                return (bi as u32, ra);
            }
        }
    }
    panic!("no pretailcall branch site");
}
fn entry_capture(insp: &mut svm_interp::Inspector, luav: u32) -> (i64, u64, u64) {
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

/// One callee's occupancy of the shared frame node, snapshotted at its own first dispatch.
struct Callee {
    ci: u64,
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
fn tail_call_reuses_the_frame() {
    let m = lua_module();
    let luav = byname(&m, "luaV_execute").expect("luaV_execute");
    let precall = byname(&m, "luaD_precall").expect("precall");
    let poscall = byname(&m, "luaD_poscall").expect("poscall");
    let pretail = byname(&m, "luaD_pretailcall").expect("pretailcall");
    let dispatch = dispatch_block(&m, luav);
    let (pblock, ra_val) = precall_site(&m, luav, precall);
    let (tblock, tra_val) = pretail_site(&m, luav, pretail);

    // ---- Pass 1: entry state + the ra of each call site (outer, the tail site, print). ----
    let inst = svm_run::instantiate(m.clone()).expect("inst");
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

    // Break order: outer's precall, then the tail pretailcall, then print's precall — read each ra
    // with single-breakpoint phases.
    let mut grab = |block: u32, val: u32| -> u64 {
        let bp = IrPc {
            module: 0,
            func: luav,
            block: block as usize,
            inst: 0,
        };
        insp.set_breakpoint(bp);
        let r = match insp.run_until_stop() {
            Stop::Break {
                reason: StopReason::Breakpoint,
                ..
            } => match insp.read_ir_value(0, val as usize) {
                Some(Value::I64(v)) => v as u64,
                o => panic!("ra: {o:?}"),
            },
            o => panic!("expected break: {o:?}"),
        };
        insp.clear_breakpoint(bp);
        r
    };
    let ra_outer = grab(pblock, ra_val);
    let ra_tail = grab(tblock, tra_val);
    let ra_print = grab(pblock, ra_val);
    println!("ra_outer={ra_outer:#x} ra_tail={ra_tail:#x} ra_print={ra_print:#x}");
    assert!(
        (stack_lo..stack_hi).contains(&ra_tail),
        "the tail ra is a stack slot"
    );

    // ---- Pass 2: capture each callee's occupancy of the frame node at its own dispatch. ----
    let callees = {
        let inst2 = svm_run::instantiate(m.clone()).expect("inst2");
        let mut in2 = inst2.debug_attach(SCRIPT.as_bytes().to_vec(), u64::MAX);
        let (_s, l2, _c) = entry_capture(&mut in2, luav);
        let dbp = IrPc {
            module: 0,
            func: luav,
            block: dispatch as usize,
            inst: 0,
        };
        in2.set_breakpoint(dbp);
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
                        seen.push(Callee {
                            ci,
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
        assert_eq!(seen.len(), 2, "must capture both callee occupancies");
        seen
    };
    // Dispatch order: outer first, then the tail-callee inner. Both occupy the SAME node — and,
    // unlike sequential calls, the tail even reuses the same `ci->func` slot (the frame is replaced
    // in place), so routing is by order, not by func.
    let (outer, inner) = (&callees[0], &callees[1]);
    assert_eq!(outer.ci, inner.ci, "a tail call reuses the frame node");
    assert_eq!(
        outer.func, inner.func,
        "a tail call reuses the frame's func slot (the closure is moved down into it)"
    );
    assert_eq!(outer.func, ra_outer, "the reused slot is outer's call slot");
    let cci = outer.ci;
    println!(
        "reused ci={cci:#x} func={:#x}   outer: proto={:#x} code={:#x}   inner: proto={:#x} code={:#x}",
        outer.func, outer.proto, outer.code, inner.proto, inner.code
    );
    // Discover the savedpc offset inside CallInfo: at a callee's first dispatch its savedpc is the
    // callee's code start. Cross-check on both occupancies.
    let savedpc_off = (0..CI_SIZE as u64)
        .step_by(8)
        .find(|&off| rd_u64(&outer.w, cci + off) == outer.code)
        .expect("ci->savedpc");
    assert_eq!(
        rd_u64(&inner.w, cci + savedpc_off),
        inner.code,
        "savedpc offset consistent across occupancies"
    );
    let ci_previous_off = (0..CI_SIZE as u64)
        .step_by(8)
        .find(|&off| rd_u64(&outer.w, cci + off) == main_ci)
        .expect("ci->previous");

    let slice = |src: &[u8], a: u64, n: usize| src[a as usize..a as usize + n].to_vec();
    // ONE overlay for the shared node (add's image). The fields that differ per callee — func and
    // savedpc — are pinned per call site below; everything the fold reads besides those (previous,
    // callstatus, trap) is identical across the two occupancies.
    let mut overlays = vec![
        (l, slice(&w, l, LUA_STATE_SIZE)),
        (stack_lo, slice(&w, stack_lo, stack_len)),
        (main_ci, slice(&w, main_ci, CI_SIZE)),
        (main_code, slice(&w, main_code, 4 * main_sizecode)),
        (main_cl, slice(&w, main_cl, 48)),
        (main_proto, slice(&w, main_proto, 128)),
        (main_pp, slice(&w, main_pp, 8 * main_sizep)),
        (cci, slice(&outer.w, cci, CI_SIZE)),
    ];
    if main_sizek > 0 {
        overlays.push((main_k, slice(&w, main_k, 16 * main_sizek)));
    }
    for c in [outer, inner] {
        overlays.push((c.cl, slice(&c.w, c.cl, 48)));
        overlays.push((c.proto, slice(&c.w, c.proto, 128)));
        overlays.push((c.code, slice(&c.w, c.code, 4 * c.sizecode)));
        if c.sizek > 0 && c.k != main_k {
            overlays.push((c.k, slice(&c.w, c.k, 16 * c.sizek)));
        }
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
        "luaD_pretailcall",
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

    // outer's entry: an ordinary Lua site (main -> outer). The tail site pins the REUSED node's
    // per-occupancy fields for inner: the moved closure slot (same address!), the unchanged
    // `ci->func` field, and inner's `savedpc`.
    let outer_site = LuaSite {
        ra: ra_outer,
        callee_ci: cci,
        pins: vec![
            (outer.func, outer.cl),
            (cci + CI_FUNC, ra_outer),
            (cci + savedpc_off, outer.code),
        ],
    };
    // The moved argument: inner's x lands at the reused frame's R0; its captured tag pins, its
    // value reloads dynamic (the move happens inside the opaque cut).
    let arg0 = inner.func + 16;
    let tail_site = TailSite {
        ra: ra_tail,
        callee_ci: cci,
        pins: vec![
            (inner.func, inner.cl),
            (cci + CI_FUNC, inner.func),
            (cci + savedpc_off, inner.code),
        ],
        args: vec![(arg0, inner.w[(arg0 + 8) as usize])],
    };
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
            lua_sites: vec![outer_site],
            c_sites: vec![ra_print],
            poscall: Some(PoscallModel {
                poscall,
                ci_previous_off,
                ci_func_off: CI_FUNC,
                tag_off: 8,
                // One reused node, one selective entry — the tail-callee's integer return lands in
                // the original call slot (`ci->func` = outer's slot), where main reads it.
                selective: vec![(cci, VNUMINT)],
            }),
        }),
        pretailcall_model: Some(PretailcallModel {
            pretailcall: pretail,
            ra_arg: 3,
            l_ci_addr: l + L_CI,
            tag_off: TAG_OFF,
            sites: vec![tail_site],
        }),
        ..SpecConfig::default()
    };
    let args = [
        SpecArg::ConstI64(sp),
        SpecArg::ConstI64(l as i64),
        SpecArg::ConstI64(main_ci as i64),
    ];
    let mut r = specialize_with_config(&m, luav, &args, &cfg)
        .expect("project through two sequential distinct callees sharing one CallInfo");
    let entry = (r.funcs.len() - 1) as u32;
    let f = &r.funcs[entry as usize];
    let brt = f
        .blocks
        .iter()
        .filter(|b| matches!(b.term, Terminator::BrTable { .. }))
        .count();
    println!("PROJECTED seq: {} blocks, br_table={brt}", f.blocks.len());
    assert_eq!(brt, 0, "the dispatch folded through both sequential calls");

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
    svm_verify::verify_module(&r).expect("embedded residual verifies");
    let base = svm_run::run_powerbox(&m, SCRIPT.as_bytes()).expect("baseline");
    let emb = svm_run::run_powerbox(&r, SCRIPT.as_bytes()).expect("embedded");
    println!(
        "baseline={:?} embedded={:?}",
        String::from_utf8_lossy(&base.stdout),
        String::from_utf8_lossy(&emb.stdout)
    );
    assert_eq!(base.stdout, b"11\n", "baseline prints 11");
    assert_eq!(emb.stdout, base.stdout, "residual byte-identical");
    println!("  ✓ a tail call (return inner(y)) projects and executes, frame reused (11)");
}
