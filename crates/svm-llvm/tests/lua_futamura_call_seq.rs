//! **I71 facet (c): two sequential, distinct Lua callees share one cached `CallInfo`.**
//!
//! ```lua
//! local function add(a, b) return a + b end
//! local function mul(a, b) return a * b end
//! print(add(2, 3) + mul(4, 5))            -- 5 + 20 = 25
//! ```
//!
//! Lua caches the callee frame node: both calls run in `main_ci->next` — the **same** `CallInfo`
//! address — with its per-callee fields (`func`, `savedpc`) overwritten between the calls. A single
//! const overlay can only hold one callee's image, which is what kept this facet open.
//!
//! The fix needs **no engine change**: [`svm_peval::LuaSite::pins`] is a generic list of
//! `(address, value)` cells pinned at the call's cut, and mem cells shadow the overlay seed. So each
//! call site pins the shared node's *per-callee* fields itself — `ci->func` (this site's `ra`) and
//! `ci->savedpc` (this callee's code start) — alongside the usual closure-slot pin. The overlay
//! carries one callee's image for the fields that are the same for both (previous, callstatus, trap);
//! the pins overwrite the two that differ. Both returns are integer-valued through the one shared
//! `ci`, so a single [`svm_peval::PoscallModel::selective`] entry covers them.
//!
//! The embedded residual prints `25\n` byte-identically to the interpreter — sequential distinct
//! callees through one shared `CallInfo`, execution-correct. See ISSUES.md I71.
//!
//! Run: `cargo test -p svm-llvm --test lua_futamura_call_seq -- --nocapture`

use svm_interp::{IrPc, Stop, StopReason, Value};
use svm_ir::{Block, Func, Inst, Module, Terminator, ValType};
use svm_peval::{specialize_with_config, LuaSite, PoscallModel, PrecallModel, SpecArg, SpecConfig};

const SCRIPT: &str = "local function add(a, b) return a + b end\n\
                      local function mul(a, b) return a * b end\n\
                      print(add(2, 3) + mul(4, 5))\n";
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
fn sequential_distinct_callees_share_one_callinfo() {
    let m = lua_module();
    let luav = byname(&m, "luaV_execute").expect("luaV_execute");
    let precall = byname(&m, "luaD_precall").expect("precall");
    let poscall = byname(&m, "luaD_poscall").expect("poscall");
    let dispatch = dispatch_block(&m, luav);
    let (pblock, ra_val) = precall_site(&m, luav, precall);

    // ---- Pass 1: entry state + the ra of each call site (add, mul, print). ----
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
    // Sequential call order: add, mul, print.
    let (ra_add, ra_mul, ra_print) = (ras[0], ras[1], ras[2]);
    println!("ra_add={ra_add:#x} ra_mul={ra_mul:#x} ra_print={ra_print:#x}");
    assert_ne!(ra_add, ra_mul, "distinct call sites have distinct ra");

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
    let find = |ra: u64| callees.iter().find(|c| c.func == ra).expect("callee");
    let (add, mul) = (find(ra_add), find(ra_mul));
    // The facet itself: one shared frame node, two occupancies.
    assert_eq!(
        add.ci, mul.ci,
        "sequential callees reuse the SAME cached CallInfo (else this test isn't testing facet c)"
    );
    let cci = add.ci;
    println!(
        "shared ci={cci:#x}   add: proto={:#x} code={:#x}   mul: proto={:#x} code={:#x}",
        add.proto, add.code, mul.proto, mul.code
    );
    // Discover the savedpc offset inside CallInfo: at a callee's first dispatch its savedpc is the
    // callee's code start. Cross-check on both occupancies.
    let savedpc_off = (0..CI_SIZE as u64)
        .step_by(8)
        .find(|&off| rd_u64(&add.w, cci + off) == add.code)
        .expect("ci->savedpc");
    assert_eq!(
        rd_u64(&mul.w, cci + savedpc_off),
        mul.code,
        "savedpc offset consistent across occupancies"
    );
    let ci_previous_off = (0..CI_SIZE as u64)
        .step_by(8)
        .find(|&off| rd_u64(&add.w, cci + off) == main_ci)
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
        (cci, slice(&add.w, cci, CI_SIZE)),
    ];
    if main_sizek > 0 {
        overlays.push((main_k, slice(&w, main_k, 16 * main_sizek)));
    }
    for c in [add, mul] {
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

    // Per-site pins carry each callee's occupancy of the shared node: the closure slot, plus the two
    // ci fields that differ between the calls.
    let site = |ra: u64, c: &Callee| LuaSite {
        ra,
        callee_ci: cci,
        pins: vec![
            (c.func, c.cl),
            (cci + CI_FUNC, ra),
            (cci + savedpc_off, c.code),
        ],
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
            lua_sites: vec![site(ra_add, add), site(ra_mul, mul)],
            c_sites: vec![ra_print],
            poscall: Some(PoscallModel {
                poscall,
                ci_previous_off,
                ci_func_off: CI_FUNC,
                tag_off: 8,
                // One shared node, one selective entry — both returns are integers.
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
    assert_eq!(base.stdout, b"25\n", "baseline prints 25");
    assert_eq!(emb.stdout, base.stdout, "residual byte-identical");
    println!("  ✓ sequential distinct callees through ONE shared CallInfo, execution-correct (25)");
}
