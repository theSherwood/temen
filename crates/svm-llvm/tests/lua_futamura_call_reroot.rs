//! **Shift-the-root: any frame of a Lua call tree can be a projection root.**
//!
//! Same nested program as `lua_futamura_call_nested`, projected from a different root: instead of
//! starting at the main chunk, capture at **`outer`'s first dispatch** and pass `outer`'s `ci` as the
//! constant entry argument. `outer` is then the *entry* frame — its base a specialization constant —
//! and `inner` an ordinary 1-deep callee (the slice-1/2 machinery). The projection folds `outer`'s
//! body, through the `inner` call, back into `main`'s post-call resume (the `print` site) and out
//! `luaV_execute`'s fresh-frame exit: `br_table == 0`, **no metamethod calls**, and — because Lua's
//! inline `RETURN1` fast path folds as plain const stores — the whole computation `(10 + 1) * 2`
//! reduces to a **literal `ConstI64(22)`** in the residual.
//!
//! This validates the composition claim behind **per-function residuals**: every frame in a call tree
//! can be a projection *root* handling its own body plus one call boundary, with boundaries composing
//! through the precall/poscall cut model. Depth-N inlining (`lua_futamura_call_nested`) is then an
//! *optimization*, not a prerequisite — the architecture that scales to real Lua call trees.
//!
//! (Historical note: this experiment is what exposed the `CI_SIZE` overlay collision — one frame's
//! 104-byte ci over-slice swallowing the next frame's 64-byte-strided header — that had masqueraded
//! as I71(b)'s "engine-level" blocker. See the `CI_SIZE` doc below and ISSUES.md I71.)
//!
//! What this test deliberately does NOT do: embed and execute the residual. A rerooted residual
//! assumes entry at `outer`'s live frame, but `run_powerbox` enters `luaV_execute` at `main`'s frame —
//! faithful execution needs per-function *stitching* (guarded call-site dispatch to callee residuals),
//! the follow-up this experiment de-risks. Mutable state (L, the stack, main's `CallInfo` with its
//! advanced `savedpc`) is therefore overlaid from `outer`'s dispatch-moment window, not the entry's.
//!
//! Run: `cargo test -p svm-llvm --test lua_futamura_call_reroot -- --nocapture`

use svm_interp::{IrPc, Stop, StopReason, Value};
use svm_ir::{Inst, Module, Terminator};
use svm_peval::{specialize_with_config, LuaSite, PoscallModel, PrecallModel, SpecArg, SpecConfig};

const SCRIPT: &str = "local function inner(x) return x + 1 end\n\
                      local function outer(y) return inner(y) * 2 end\n\
                      print(outer(10))\n";
const CAP: usize = 8 << 20;

const L_CI: u64 = 32;
const L_STACK_LAST: u64 = 40;
const L_STACK: u64 = 48;
const LUA_STATE_SIZE: usize = 200;
/// One `CallInfo`'s overlay span. This must not exceed the real allocation stride: adjacent frames'
/// ci nodes sit 64 bytes apart here (outer 0x..b0, inner 0x..f0), and every field the fold reads
/// (func +0, previous +16, savedpc, trap, callstatus +62) lies below 64. The old 104-byte over-slice
/// made one frame's ci overlay swallow the next frame's header with stale pre-call bytes — the root
/// cause of the multi-frame "callee reads garbage" divergence (I71 facet b).
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

/// One captured Lua callee frame, snapshotted at *its own* first dispatch (so `ci->savedpc` points at
/// this callee's code start and every mutated cell reflects that moment).
struct Frame {
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
fn reroot_at_outer_folds_the_nested_call() {
    let m = lua_module();
    let luav = byname(&m, "luaV_execute").expect("luaV_execute");
    let precall = byname(&m, "luaD_precall").expect("precall");
    let poscall = byname(&m, "luaD_poscall").expect("poscall");
    let dispatch = dispatch_block(&m, luav);
    let (pblock, ra_val) = precall_site(&m, luav, precall);

    // ---- Pass 1: main-chunk entry state + the ra of each call site (outer, inner, print). ----
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

    // ---- Pass 2: capture the outer and inner frames, each at its OWN first dispatch. ----
    let frames = {
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
        let mut seen: Vec<Frame> = Vec::new();
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
                        seen.push(Frame {
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
        assert_eq!(seen.len(), 2, "must capture both outer and inner frames");
        seen
    };
    let find = |ra: u64| frames.iter().find(|c| c.func == ra).expect("frame for ra");
    let (outer, inner) = (find(ra_outer), find(ra_inner));
    println!(
        "outer: ci={:#x} proto={:#x}   inner: ci={:#x} proto={:#x}",
        outer.ci, outer.proto, inner.ci, inner.proto
    );
    // The structural `previous` offset: outer's frame links back to main's ci.
    let ci_previous_off = (0..CI_SIZE as u64)
        .step_by(8)
        .find(|&off| rd_u64(&outer.w, outer.ci + off) == main_ci)
        .expect("ci->previous");
    // The overlay-collision guard (the I71(b) root cause): each ci overlay must cover ONLY its own
    // frame — an overlap would seed one frame's header from another frame's stale window.
    let mut cis = [main_ci, outer.ci, inner.ci];
    cis.sort();
    for p in cis.windows(2) {
        assert!(
            p[0] + CI_SIZE as u64 <= p[1],
            "ci overlays must not overlap: {:#x}+{CI_SIZE} > {:#x}",
            p[0],
            p[1]
        );
    }

    let slice = |src: &[u8], a: u64, n: usize| src[a as usize..a as usize + n].to_vec();
    // The projection starts at OUTER's dispatch moment, so all mutated state — L (its ci field points
    // at outer), the stack (main's registers hold the closures, the fetched `print`, and outer's arg
    // y=10), and main's CallInfo (savedpc advanced past the OP_CALL, the correct resume point) — is
    // overlaid from outer's window. Static program data (code, protos, closures, k pools) can come
    // from any window.
    let mut overlays = vec![
        (l, slice(&outer.w, l, LUA_STATE_SIZE)),
        (stack_lo, slice(&outer.w, stack_lo, stack_len)),
        (main_ci, slice(&outer.w, main_ci, CI_SIZE)),
        (main_code, slice(&w, main_code, 4 * main_sizecode)),
        (main_cl, slice(&w, main_cl, 48)),
        (main_proto, slice(&w, main_proto, 128)),
        (main_pp, slice(&w, main_pp, 8 * main_sizep)),
    ];
    if main_sizek > 0 {
        overlays.push((main_k, slice(&w, main_k, 16 * main_sizek)));
    }
    for c in [outer, inner] {
        overlays.push((c.ci, slice(&c.w, c.ci, CI_SIZE)));
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
        (outer.ci, outer.ci + CI_SIZE as u64),
        (inner.ci, inner.ci + CI_SIZE as u64),
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
        // The allocation / GC / string / upvalue primitive families (see I71): opaque stateful
        // machinery a dynamic value must not drag the fold through.
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
            // Rooted at outer, the only Lua call *made* is outer→inner; main's print resume is a
            // C site. (main→outer never occurs inside this projection — we start past it.)
            lua_sites: vec![LuaSite {
                ra: ra_inner,
                callee_ci: inner.ci,
                pins: vec![(inner.func, inner.cl)],
            }],
            c_sites: vec![ra_print],
            poscall: Some(PoscallModel {
                poscall,
                ci_previous_off,
                ci_func_off: CI_FUNC,
                tag_off: 8,
                // Both Lua returns on the path are integer-valued: inner→outer (11) and
                // outer→main (22). main's own return exits the fresh frame.
                selective: vec![(inner.ci, VNUMINT), (outer.ci, VNUMINT)],
            }),
        }),
        ..SpecConfig::default()
    };
    // The root shift itself: outer's ci is the constant entry argument.
    let args = [
        SpecArg::ConstI64(sp),
        SpecArg::ConstI64(l as i64),
        SpecArg::ConstI64(outer.ci as i64),
    ];
    let r = specialize_with_config(&m, luav, &args, &cfg)
        .expect("reroot projection: outer as entry frame, inner as its 1-deep callee");

    let entry = r.funcs.last().expect("entry fn");
    let brt = entry
        .blocks
        .iter()
        .filter(|b| matches!(b.term, Terminator::BrTable { .. }))
        .count();
    let count_calls = |targets: &[Option<u32>]| {
        let set: Vec<u32> = targets.iter().filter_map(|t| *t).collect();
        entry
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter(|i| matches!(i, Inst::Call { func, .. } if set.contains(func)))
            .count()
    };
    let n_precall = count_calls(&[Some(precall)]);
    let n_poscall = count_calls(&[Some(poscall)]);
    let n_metamethod = count_calls(&[
        byname(&m, "luaT_trybinTM"),
        byname(&m, "luaT_trybinassocTM"),
        byname(&m, "luaT_trybiniTM"),
        byname(&m, "callbinTM"),
        byname(&m, "luaT_gettmbyobj"),
    ]);
    println!(
        "REROOTED: {} blocks, br_table={brt}, precall_cuts={n_precall}, poscall_cuts={n_poscall}, metamethod_calls={n_metamethod}",
        entry.blocks.len()
    );
    let n_c22 = entry
        .blocks
        .iter()
        .flat_map(|b| b.insts.iter())
        .filter(|i| matches!(i, Inst::ConstI64(22)))
        .count();
    println!("folded ConstI64(22) occurrences: {n_c22}");
    assert_eq!(
        brt, 0,
        "the dispatch folded through outer's body, the inner call, and main's resume"
    );
    assert_eq!(
        n_metamethod, 0,
        "every arithmetic fast-path guard folded (no metamethod descent)"
    );
    assert!(
        n_precall >= 2,
        "the inner call and the print call are both present as cut boundaries"
    );
    // Not every return needs a poscall cut: `RETURN0`/`RETURN1` have inline fast paths that pop the
    // frame with plain stores, and those fold entirely (L->ci = previous is a const-to-const store).
    // At least one return on the path goes through the general `luaD_poscall`.
    assert!(
        n_poscall >= 1,
        "at least one return goes through the modeled luaD_poscall cut"
    );
    // The semantic money shot: inner's inline `RETURN1` folded const, so `(10 + 1) * 2` reduced to a
    // literal 22 at projection time — the program's answer is IN the residual as a constant.
    assert!(
        n_c22 >= 1,
        "the nested computation folded to the constant 22 in the residual"
    );
    println!("  ✓ rerooted at outer: the nested program folds, and (10+1)*2 reduced to const 22");
}
