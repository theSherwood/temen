//! **A Lua call whose result feeds arithmetic in the caller** — `print(add(2,3) * 10)` → `50`.
//!
//! `lua_futamura_call.rs` returns a call result into a C function (`print`, opaque to tags). This
//! projects a result into a **register-arithmetic** opcode (`OP_MULK`), which is what real call-bearing
//! loops do (`x = add(x, k)`), and executes it byte-identically to the interpreter. Two things had to be
//! right, both "static program data / stateful machinery the specializer must see":
//!
//! 1. **Overlay each proto's constant pool (`proto->k`).** `OP_MULK` reads its constant operand (here
//!    `10`) and its *tag* from `k[C]`. Without the `k` overlay that tag stays dynamic, the
//!    `ttisinteger(v1) && ttisinteger(v2)` fast-path guard can't fold, and the fold falls into the
//!    `OP_MMBINK` metamethod slow path → `luaT_trybinassocTM` → `luaD_throw` (an unsupported `longjmp`).
//!    `add(40,2)` in the sibling test uses `LOADI` *immediates* (baked into the instruction word), which
//!    is why it never needed this.
//! 2. **Cut the allocation/GC/string primitive family** (`luaM_realloc_`/`luaM_saferealloc_`/
//!    `luaM_malloc_`/`luaM_growaux_`/`luaS_resize`/`luaS_newlstr`/`luaC_fullgc`). The selective-reload
//!    result is dynamic, so any residual arith slow path would otherwise drag the fold through the
//!    allocator's cold subtree (the `frealloc` grow path calls the host `vm_map`; the alloc-failure arm
//!    reaches emergency GC) — all stateful host-backed machinery that belongs opaque, like the GC
//!    barriers already cut in slice 1.
//!
//! See ISSUES.md I71. Run: `cargo test -p temen-llvm --test lua_futamura_call_arith -- --nocapture`

use temen_interp::{IrPc, Stop, StopReason, Value};
use temen_ir::{Block, Func, Inst, Module, Terminator, ValType};
use temen_peval::{
    specialize_with_config, LuaSite, PoscallModel, PrecallModel, SpecArg, SpecConfig,
};

const SCRIPT: &str = "local function add(a, b) return a + b end\nprint(add(2, 3) * 10)\n";
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
const PROTO_SIZECODE: u64 = 24;
const PROTO_SIZEP: u64 = 32;
const PROTO_CODE: u64 = 64;
const PROTO_P: u64 = 72;
const PROTO_SIZEK: u64 = 20;
const PROTO_K: u64 = 56;

fn lm() -> Module {
    let p = format!(
        "{}/tests/fixtures/lua/lua_eval.ll",
        env!("CARGO_MANIFEST_DIR")
    );
    temen_llvm::translate_ll_path(&p).expect("x").module
}
fn byname(m: &Module, n: &str) -> Option<u32> {
    m.exports.iter().find(|e| e.name == n).map(|e| e.func)
}
fn rd(w: &[u8], a: u64) -> u64 {
    u64::from_le_bytes(w[a as usize..a as usize + 8].try_into().unwrap())
}
fn rdi(w: &[u8], a: u64) -> i32 {
    i32::from_le_bytes(w[a as usize..a as usize + 4].try_into().unwrap())
}
fn disp(m: &Module, luav: u32) -> u32 {
    let mut b = (0u32, 0usize);
    for (i, bl) in m.funcs[luav as usize].blocks.iter().enumerate() {
        if let Terminator::BrTable { targets, .. } = &bl.term {
            if targets.len() > b.1 {
                b = (i as u32, targets.len());
            }
        }
    }
    b.0
}
fn pbl(m: &Module, luav: u32, pre: u32) -> (u32, u32) {
    for (i, b) in m.funcs[luav as usize].blocks.iter().enumerate() {
        if let Some(ra) = b.insts.iter().find_map(|x| match x {
            Inst::Call { func, args } if *func == pre => Some(args[2]),
            _ => None,
        }) {
            if matches!(b.term, Terminator::BrIf { .. }) {
                return (i as u32, ra);
            }
        }
    }
    panic!();
}
fn entry(insp: &mut temen_interp::Inspector, luav: u32) -> (i64, u64, u64) {
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
                o => panic!("{o:?}"),
            };
            (g(0), g(1) as u64, g(2) as u64)
        }
        o => panic!("{o:?}"),
    };
    insp.clear_breakpoint(pc);
    r
}

#[test]
fn call_result_feeds_arithmetic() {
    let m = lm();
    let luav = byname(&m, "luaV_execute").unwrap();
    let pre = byname(&m, "luaD_precall").unwrap();
    let pos = byname(&m, "luaD_poscall").unwrap();
    let d = disp(&m, luav);
    let (pb, ra_val) = pbl(&m, luav, pre);
    let inst = temen_run::instantiate(m.clone()).unwrap();
    let mut insp = inst.debug_attach(SCRIPT.as_bytes().to_vec(), u64::MAX);
    let (sp, l, oci) = entry(&mut insp, luav);
    let w = insp.read_window(0, CAP).unwrap();
    let slo = rd(&w, l + L_STACK);
    let shi = rd(&w, l + L_STACK_LAST);
    let of = rd(&w, oci + CI_FUNC);
    let ocl = rd(&w, of);
    let op = rd(&w, ocl + LCLOSURE_P);
    let oc = rd(&w, op + PROTO_CODE);
    let osc = rdi(&w, op + PROTO_SIZECODE) as usize;
    let opp = rd(&w, op + PROTO_P);
    let osp = rdi(&w, op + PROTO_SIZEP) as usize;
    let ok = rd(&w, op + PROTO_K);
    let osk = rdi(&w, op + PROTO_SIZEK) as usize;
    let pbp = IrPc {
        module: 0,
        func: luav,
        block: pb as usize,
        inst: 0,
    };
    insp.set_breakpoint(pbp);
    let ra_add = match insp.run_until_stop() {
        Stop::Break { .. } => match insp.read_ir_value(0, ra_val as usize) {
            Some(Value::I64(v)) => v as u64,
            o => panic!("{o:?}"),
        },
        o => panic!("{o:?}"),
    };
    let ra_print = match insp.run_until_stop() {
        Stop::Break { .. } => match insp.read_ir_value(0, ra_val as usize) {
            Some(Value::I64(v)) => v as u64,
            o => panic!("{o:?}"),
        },
        o => panic!("{o:?}"),
    };
    insp.clear_breakpoint(pbp);
    let inst2 = temen_run::instantiate(m.clone()).unwrap();
    let mut in2 = inst2.debug_attach(SCRIPT.as_bytes().to_vec(), u64::MAX);
    let (_s, l2, _c) = entry(&mut in2, luav);
    let dbp = IrPc {
        module: 0,
        func: luav,
        block: d as usize,
        inst: 0,
    };
    in2.set_breakpoint(dbp);
    let (cci, wb) = loop {
        match in2.run_until_stop() {
            Stop::Break {
                reason: StopReason::Breakpoint,
                ..
            } => {
                let wv = in2.read_window(0, CAP).unwrap();
                let ci = rd(&wv, l2 + L_CI);
                let f = rd(&wv, ci + CI_FUNC);
                let cl = rd(&wv, f);
                let p = rd(&wv, cl + LCLOSURE_P);
                if p != op {
                    break (ci, wv);
                }
            }
            o => panic!("{o:?}"),
        }
    };
    let cf = rd(&wb, cci + CI_FUNC);
    let ccl = rd(&wb, cf);
    let cp = rd(&wb, ccl + LCLOSURE_P);
    let cc = rd(&wb, cp + PROTO_CODE);
    let csc = rdi(&wb, cp + PROTO_SIZECODE) as usize;
    let ck = rd(&wb, cp + PROTO_K);
    let csk = rdi(&wb, cp + PROTO_SIZEK) as usize;
    let prev_off = (0..CI_SIZE as u64)
        .step_by(8)
        .find(|&o| rd(&wb, cci + o) == oci)
        .unwrap();
    let sl = |src: &[u8], a: u64, n: usize| src[a as usize..a as usize + n].to_vec();
    // The proto constant pools (k arrays) are static program data — a `MULK`/`ADDK` reads its constant
    // operand (and tag) from here, so without the overlay the arith fast-path guard can't fold.
    let mut kover: Vec<(u64, Vec<u8>)> = Vec::new();
    if osk > 0 {
        kover.push((ok, sl(&w, ok, 16 * osk)));
    }
    if csk > 0 && ck != ok {
        kover.push((ck, sl(&wb, ck, 16 * csk)));
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
        "luaM_realloc_",
        "luaM_saferealloc_",
        "luaM_malloc_",
        "luaM_growaux_",
        "luaS_resize",
        "luaS_newlstr",
        "luaC_fullgc",
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
        const_overlays: {
            let mut v = vec![
                (l, sl(&w, l, LUA_STATE_SIZE)),
                (slo, sl(&w, slo, (shi - slo) as usize)),
                (oci, sl(&w, oci, CI_SIZE)),
                (oc, sl(&w, oc, 4 * osc)),
                (ocl, sl(&w, ocl, 48)),
                (op, sl(&w, op, 128)),
                (opp, sl(&w, opp, 8 * osp)),
                (cci, sl(&wb, cci, CI_SIZE)),
                (ccl, sl(&wb, ccl, 48)),
                (cp, sl(&wb, cp, 128)),
                (cc, sl(&wb, cc, 4 * csc)),
            ];
            v.extend(kover);
            v
        },
        rename: Some((l, l + LUA_STATE_SIZE as u64)),
        rename_extra: vec![
            (slo, shi),
            (oci, oci + CI_SIZE as u64),
            (cci, cci + CI_SIZE as u64),
        ],
        rename_is_private: true,
        rename_seed_from_image: true,
        cut_calls_read_state: read,
        cut_calls_touch_state: touch,
        carry_whole_module: true,
        carry_keep_imports: true,
        indirect_targets_cap: Some(16),
        precall_model: Some(PrecallModel {
            precall: pre,
            ra_arg: 2,
            l_ci_addr: l + L_CI,
            lua_sites: vec![LuaSite {
                ra: ra_add,
                callee_ci: cci,
                pins: vec![(cf, ccl)],
            }],
            c_sites: vec![ra_print],
            poscall: Some(PoscallModel {
                poscall: pos,
                ci_previous_off: prev_off,
                ci_func_off: CI_FUNC,
                tag_off: 8,
                selective: vec![(cci, 0x03)],
            }),
        }),
        ..SpecConfig::default()
    };
    let args = [
        SpecArg::ConstI64(sp),
        SpecArg::ConstI64(l as i64),
        SpecArg::ConstI64(oci as i64),
    ];
    let mut r =
        specialize_with_config(&m, luav, &args, &cfg).expect("project a call result into OP_MULK");
    let e = (r.funcs.len() - 1) as u32;
    println!("PROJECTED: {} blocks", r.funcs[e as usize].blocks.len());
    r.imports = m.imports.clone();
    r.types = m.types.clone();
    r.funcs[luav as usize] = Func {
        params: vec![ValType::I64; 3],
        results: vec![],
        blocks: vec![Block {
            params: vec![ValType::I64; 3],
            insts: vec![Inst::Call {
                func: e,
                args: vec![],
            }],
            term: Terminator::Return(vec![]),
        }],
    };
    temen_verify::verify_module(&r).expect("verifies");
    let base = temen_run::run_powerbox(&m, SCRIPT.as_bytes()).unwrap();
    let emb = temen_run::run_powerbox(&r, SCRIPT.as_bytes()).unwrap();
    println!(
        "base={:?} emb={:?}",
        String::from_utf8_lossy(&base.stdout),
        String::from_utf8_lossy(&emb.stdout)
    );
    assert_eq!(base.stdout, b"50\n");
    assert_eq!(emb.stdout, base.stdout);
}
