//! **A nested Lua call stack — two distinct callees, one calling the other.**
//!
//! `lua_futamura_call.rs` projects a *single* Lua callee. This projects a **2-deep call stack**: the
//! main chunk calls `outer`, which calls `inner` — so at the inner call two Lua frames are live at
//! once. It checks the soundness question everything else rests on: the `OP_CALL` handler is shared, so
//! the projection distinguishes call sites **only** by their `ra` ([`temen_peval::LuaSite::ra`]) and each
//! frame by its own `CallInfo`. (Two *sequential* calls would reuse one cached `CallInfo` — Lua caches
//! `ci->next` — which one const overlay can't represent; nesting keeps the frames genuinely distinct.)
//!
//! ```lua
//! local function inner(x) return x + 1 end
//! local function outer(y) return inner(y) * 2 end
//! print(outer(10))                    -- inner(10)=11, *2 = 22
//! ```
//!
//! **RESOLVED — I71 facet (b).** Three ingredients, all config (no engine change):
//!
//! 1. Facet (a)'s per-frame `proto->k` overlays + the alloc/GC/string cut-set
//!    (`lua_futamura_call_arith`).
//! 2. The **upvalue machinery cut** (`outer` captures `inner` as an upvalue, so `OP_CLOSURE` calls
//!    `luaF_findupval`, which walks/mutates the open-upvalue list — opaque stateful machinery).
//! 3. **`CI_SIZE = 64`, the real `CallInfo` stride.** This was the long-hidden root cause: adjacent
//!    frames' ci nodes sit 64 bytes apart, so the old 104-byte over-slice made `outer`'s ci overlay
//!    swallow `inner`'s ci header with stale pre-call bytes — `inner.ci->func`/`savedpc` seeded as
//!    zero, its frame dispatched garbage, and the fold walked into the metamethod/GC cold subtree
//!    (`Budget`). That masqueraded as an engine-level "callee base is dynamic" blocker; it was an
//!    overlay collision in the capture config all along (found by `lua_futamura_call_reroot`).
//!
//! With those, the full 2-deep inline projection folds (`br_table == 0`) and the embedded residual
//! prints `22\n` byte-identically to the interpreter — the `ra` keying routes each site to the right
//! frame and the nested return chain composes, execution-correct.
//!
//! Run: `cargo test -p temen-llvm --test lua_futamura_call_nested -- --nocapture`

use temen_interp::{IrPc, Stop, StopReason, Value};
use temen_ir::{Block, Func, Inst, Module, Terminator, ValType};
use temen_peval::{
    specialize_with_config, LuaSite, PoscallModel, PrecallModel, SpecArg, SpecConfig,
};

const SCRIPT: &str = "local function inner(x) return x + 1 end\n\
                      local function outer(y) return inner(y) * 2 end\n\
                      print(outer(10))\n";
const CAP: usize = 8 << 20;

const L_CI: u64 = 32;
const L_STACK_LAST: u64 = 40;
const L_STACK: u64 = 48;
const LUA_STATE_SIZE: usize = 200;
/// One `CallInfo`'s overlay span — the real allocation stride (adjacent ci nodes are 64 apart), NOT a
/// generous over-slice: a larger span makes one frame's overlay swallow the next frame's header with
/// stale bytes, which was I71 facet (b)'s root cause. Every field the fold reads (func +0, previous
/// +16, savedpc, trap, callstatus +62) lies below 64.
const CI_SIZE: usize = 64;
const CI_FUNC: u64 = 0;
const LCLOSURE_P: u64 = 24;
const PROTO_SIZECODE: u64 = 24;
const PROTO_SIZEP: u64 = 32;
const PROTO_CODE: u64 = 64;
const PROTO_P: u64 = 72;
const PROTO_SIZEK: u64 = 20;
const PROTO_K: u64 = 56;
const TAG_OFF: u64 = 8;
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
/// The precall branch block + the SSA value index of its `ra` (arg 2 of `precall(sp, L, ra, nres)`).
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

/// One captured Lua callee frame, snapshotted at *its own* first dispatch (so `ci->savedpc` points at
/// this callee's code start, not a caller's advanced resume pc).
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
fn nested_two_frame_call_stack() {
    let m = lua_module();
    let luav = luav_execute(&m);
    let precall = byname(&m, "luaD_precall").expect("precall");
    let poscall = byname(&m, "luaD_poscall").expect("poscall");
    let dispatch = dispatch_block(&m, luav);
    let (pblock, ra_val) = precall_site(&m, luav, precall);
    println!(
        "luav=f{luav} precall=f{precall} poscall=f{poscall} dispatch=b{dispatch} pblock=b{pblock}"
    );

    // ---- Pass 1: entry + outer frame + the ra of each of the three call sites (add, mul, print). ----
    let inst = temen_run::instantiate(m.clone()).expect("inst");
    let mut insp = inst.debug_attach(SCRIPT.as_bytes().to_vec(), u64::MAX);
    let (sp, l, outer_ci) = entry_capture(&mut insp, luav);
    let w = insp.read_window(0, CAP).expect("w");
    let stack_lo = rd_u64(&w, l + L_STACK);
    let stack_hi = rd_u64(&w, l + L_STACK_LAST);
    let stack_len = (stack_hi - stack_lo) as usize;
    let outer_func = rd_u64(&w, outer_ci + CI_FUNC);
    let outer_cl = rd_u64(&w, outer_func);
    let outer_proto = rd_u64(&w, outer_cl + LCLOSURE_P);
    let outer_code = rd_u64(&w, outer_proto + PROTO_CODE);
    let outer_sizecode = rd_i32(&w, outer_proto + PROTO_SIZECODE) as usize;
    let outer_pp = rd_u64(&w, outer_proto + PROTO_P);
    let outer_sizep = rd_i32(&w, outer_proto + PROTO_SIZEP) as usize;
    let outer_k = rd_u64(&w, outer_proto + PROTO_K);
    let outer_sizek = rd_i32(&w, outer_proto + PROTO_SIZEK) as usize;

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
    // Call order: main→outer (first precall), outer→inner (second), print (third).
    let (ra_outer, ra_inner, ra_print) = (ras[0], ras[1], ras[2]);
    println!("ra_outer={ra_outer:#x} ra_inner={ra_inner:#x} ra_print={ra_print:#x}");
    assert_ne!(
        ra_outer, ra_inner,
        "distinct call sites must have distinct ra"
    );

    // ---- Pass 2: capture each callee frame at its OWN first dispatch (distinct non-outer protos). ----
    let callees = {
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
                    if proto != outer_proto && !seen.iter().any(|c| c.proto == proto) {
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
        assert_eq!(seen.len(), 2, "must capture both callee frames");
        seen
    };
    // Route each callee frame to its call site by ci->func == that site's ra.
    let find = |ra: u64| {
        callees
            .iter()
            .find(|c| c.func == ra)
            .expect("callee for ra")
    };
    let (outer, inner) = (find(ra_outer), find(ra_inner));
    println!(
        "outer: ci={:#x} proto={:#x}   inner: ci={:#x} proto={:#x}",
        outer.ci, outer.proto, inner.ci, inner.proto
    );
    assert_ne!(
        outer.ci, inner.ci,
        "the nested callees must have distinct live CallInfos"
    );
    // The `previous` offset is structural (same for every CallInfo); the outer frame's `previous` is the
    // main chunk's ci, so find the offset there, then it resolves inner->outer and outer->main alike.
    let ci_previous_off = (0..CI_SIZE as u64)
        .step_by(8)
        .find(|&off| rd_u64(&outer.w, outer.ci + off) == outer_ci)
        .expect("ci->previous");

    let slice = |src: &[u8], a: u64, n: usize| src[a as usize..a as usize + n].to_vec();
    let mut overlays = vec![
        (l, slice(&w, l, LUA_STATE_SIZE)),
        (stack_lo, slice(&w, stack_lo, stack_len)),
        (outer_ci, slice(&w, outer_ci, CI_SIZE)),
        (outer_code, slice(&w, outer_code, 4 * outer_sizecode)),
        (outer_cl, slice(&w, outer_cl, 48)),
        (outer_proto, slice(&w, outer_proto, 128)),
        (outer_pp, slice(&w, outer_pp, 8 * outer_sizep)),
    ];
    // The proto constant pool (`proto->k`) is static program data: an arith-K opcode reads its constant
    // operand + tag from here (see I71 / `lua_futamura_call_arith`); `outer`'s `inner(y) * 2` needs it.
    if outer_sizek > 0 {
        overlays.push((outer_k, slice(&w, outer_k, 16 * outer_sizek)));
    }
    let mut rename_extra = vec![(stack_lo, stack_hi), (outer_ci, outer_ci + CI_SIZE as u64)];
    for c in [outer, inner] {
        // Each frame's overlays come from its own window `c.w` (correct `ci->savedpc` for this callee).
        overlays.push((c.ci, slice(&c.w, c.ci, CI_SIZE)));
        overlays.push((c.cl, slice(&c.w, c.cl, 48)));
        overlays.push((c.proto, slice(&c.w, c.proto, 128)));
        overlays.push((c.code, slice(&c.w, c.code, 4 * c.sizecode)));
        if c.sizek > 0 && c.k != outer_k {
            overlays.push((c.k, slice(&c.w, c.k, 16 * c.sizek)));
        }
        rename_extra.push((c.ci, c.ci + CI_SIZE as u64));
    }

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
        // The allocation / GC / string-interning primitive family — opaque stateful machinery a dynamic
        // result would otherwise drag the fold through (I71). Same class as the GC barriers above.
        "luaM_realloc_",
        "luaM_saferealloc_",
        "luaM_malloc_",
        "luaM_growaux_",
        "luaS_resize",
        "luaS_newlstr",
        "luaC_fullgc",
        // Upvalue machinery: `outer` captures `inner` as an upvalue, so `OP_CLOSURE` calls
        // `luaF_findupval`, which walks/mutates the open-upvalue list and allocates — cut it opaque.
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
            lua_sites: vec![
                LuaSite {
                    ra: ra_outer,
                    callee_ci: outer.ci,
                    pins: vec![(outer.func, outer.cl)],
                },
                LuaSite {
                    ra: ra_inner,
                    callee_ci: inner.ci,
                    pins: vec![(inner.func, inner.cl)],
                },
            ],
            c_sites: vec![ra_print],
            poscall: Some(PoscallModel {
                poscall,
                ci_previous_off,
                ci_func_off: CI_FUNC,
                tag_off: TAG_OFF,
                selective: vec![(outer.ci, VNUMINT), (inner.ci, VNUMINT)],
            }),
        }),
        ..SpecConfig::default()
    };
    let args = [
        SpecArg::ConstI64(sp),
        SpecArg::ConstI64(l as i64),
        SpecArg::ConstI64(outer_ci as i64),
    ];
    let mut residual = specialize_with_config(&m, luav, &args, &cfg)
        .expect("project through the nested Lua calls");
    let entry = (residual.funcs.len() - 1) as u32;
    let f = &residual.funcs[entry as usize];
    let brt = f
        .blocks
        .iter()
        .filter(|b| matches!(b.term, Terminator::BrTable { .. }))
        .count();
    println!(
        "PROJECTED multi-callee: {} blocks, br_table={brt}",
        f.blocks.len()
    );
    assert_eq!(brt, 0, "the dispatch folded through both nested Lua calls");

    // ---- Embed + diff stdout ----
    residual.imports = m.imports.clone();
    residual.types = m.types.clone();
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
    let baseline = temen_run::run_powerbox(&m, SCRIPT.as_bytes()).expect("baseline");
    let embedded = temen_run::run_powerbox(&residual, SCRIPT.as_bytes()).expect("embedded");
    println!(
        "baseline={:?} embedded={:?}",
        String::from_utf8_lossy(&baseline.stdout),
        String::from_utf8_lossy(&embedded.stdout)
    );
    assert_eq!(baseline.stdout, b"22\n", "baseline prints 22");
    assert_eq!(
        embedded.stdout, baseline.stdout,
        "residual byte-identical to interpreter"
    );
    println!("  ✓ nested Lua call stack (outer→inner) projected and correct (22)");
}
