//! **The capture-automation driver: Lua script in → verified residual out.**
//!
//! Everything the `lua_futamura_call*` tests hand-build — call-site discovery, per-frame windows,
//! overlay lists, per-site pins, return tags — derived automatically from ONE profiling run:
//!
//! 1. **Profile.** Break at `luaV_execute`'s entry (roots), then run with breakpoints on the
//!    `luaD_precall` branch block, the `luaD_pretailcall` branch block, and the dispatch `br_table`
//!    block. Every stop is an *event*; the stream is a complete trace of call activity.
//! 2. **Classify.** A precall event followed immediately by a dispatch whose frame `func` equals the
//!    event's `ra` is a **Lua site** (the dispatch window is that callee's occupancy snapshot — the
//!    I71(b) per-frame-window discipline); followed by anything else it is a **C site**. A
//!    pretailcall event followed by a new-proto dispatch is a **tail site** (frame reused). A repeat
//!    of a known site (a loop) dedupes.
//! 3. **Discover tags.** A site's return tag is read from the result slot (`frame.func + 8` — the
//!    same slot for normal and tail calls) at the first later event whose `L->ci` **depth** (its
//!    `previous`-chain length) is *smaller* than the callee's — i.e. after the callee returned but
//!    before anything overwrites the slot. Correct across nesting; falls back to integer when the
//!    call is the program's last activity.
//! 4. **Assemble.** Uniform per-site pins (`(func → cl)`, `(ci->func → func)`, `(ci->savedpc →
//!    code)` — the facet-(c)/tail-call pinning applied everywhere), per-occupancy `CallInfo`
//!    overlays at the real 64-byte stride with a collision assert, proto/code/`k`/sub-proto
//!    overlays, the standard cut lists, and a `PretailcallModel` iff tail sites exist.
//! 5. **Specialize, embed, verify, run.** The residual replaces `luaV_execute` wholesale and the
//!    pair of stdouts is returned for the caller to diff.
//!
//! Layout constants are properties of the vendored Lua build (discovered once in the feasibility
//! session; the `CallInfo` stride story is ISSUES I71 facet b). Not automated yet: rolling
//! safepoint roots (`dynamic_cells`), stitching, multi-value returns, C tail calls.

#![allow(dead_code)] // shared test-support module; each test crate uses a subset

use std::collections::BTreeMap;

use svm_interp::{IrPc, Stop, StopReason, Value};
use svm_ir::{Block, Func, Inst, Module, Terminator, ValType};
use svm_peval::{
    specialize_with_config, LuaSite, PoscallModel, PrecallModel, PretailcallModel, SpecArg,
    SpecConfig, TailSite,
};

pub const CAP: usize = 8 << 20;

// lua_State field offsets and object layouts for the vendored build.
const L_CI: u64 = 32;
const L_STACK_LAST: u64 = 40;
const L_STACK: u64 = 48;
const LUA_STATE_SIZE: usize = 200;
/// The real `CallInfo` allocation stride — NOT a generous over-slice (I71 facet b).
const CI_SIZE: usize = 64;
const CI_FUNC: u64 = 0;
const LCLOSURE_P: u64 = 24;
const PROTO_SIZEK: u64 = 20;
const PROTO_SIZECODE: u64 = 24;
const PROTO_SIZEP: u64 = 32;
const PROTO_K: u64 = 56;
const PROTO_CODE: u64 = 64;
const PROTO_P: u64 = 72;
const TAG_OFF: u64 = 8;
const VNUMINT: u8 = 0x03;
/// Runaway guard on the profiling event loop.
const MAX_EVENTS: usize = 100_000;

fn rd_u64(w: &[u8], a: u64) -> u64 {
    u64::from_le_bytes(w[a as usize..a as usize + 8].try_into().unwrap())
}
fn rd_i32(w: &[u8], a: u64) -> i32 {
    i32::from_le_bytes(w[a as usize..a as usize + 4].try_into().unwrap())
}

pub fn lua_module() -> Module {
    let p = format!(
        "{}/tests/fixtures/lua/lua_eval.ll",
        env!("CARGO_MANIFEST_DIR")
    );
    svm_llvm::translate_ll_path(&p).expect("translate").module
}
fn byname(m: &Module, n: &str) -> Option<u32> {
    m.exports.iter().find(|e| e.name == n).map(|e| e.func)
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
/// A cut-call branch block: the block calling `callee` that ends in the Lua-vs-continue `BrIf`.
/// Returns the block plus the SSA value indices of `args[ra_arg]` and `args[extra_arg]`.
fn call_site(
    m: &Module,
    luav: u32,
    callee: u32,
    ra_arg: usize,
    extra_arg: usize,
) -> (u32, u32, u32) {
    for (bi, b) in m.funcs[luav as usize].blocks.iter().enumerate() {
        let hit = b.insts.iter().find_map(|i| match i {
            Inst::Call { func, args } if *func == callee => Some((args[ra_arg], args[extra_arg])),
            _ => None,
        });
        if let Some((ra, extra)) = hit {
            if matches!(b.term, Terminator::BrIf { .. }) {
                return (bi as u32, ra, extra);
            }
        }
    }
    panic!("no branch site for the cut callee");
}

/// One callee occupancy of a frame node, snapshotted at its own first dispatch.
struct Occupancy {
    ci: u64,
    func: u64,
    cl: u64,
    proto: u64,
    code: u64,
    sizecode: usize,
    k: u64,
    sizek: usize,
    pp: u64,
    sizep: usize,
    depth: usize,
    w: Vec<u8>,
}

/// One discovered call site (Lua or tail), with its occupancy index and discovered return tag.
struct Site {
    ra: u64,
    occ: usize,
    tail: bool,
    /// Moved-argument count (tail sites only).
    nargs: usize,
    tag: Option<u8>,
}

pub struct AutoResult {
    pub baseline_stdout: Vec<u8>,
    pub residual_stdout: Vec<u8>,
    pub residual_blocks: usize,
    pub br_tables: usize,
    pub lua_sites: usize,
    pub c_sites: usize,
    pub tail_sites: usize,
}

/// The build half of the pipeline: the baseline module, the embedded residual module, and stats —
/// so a caller (e.g. the benchmark) can time the phases and the runs separately.
pub struct AutoBuild {
    pub baseline: Module,
    pub residual: Module,
    pub residual_blocks: usize,
    pub br_tables: usize,
    pub lua_sites: usize,
    pub c_sites: usize,
    pub tail_sites: usize,
    /// Wall time of the profiling run (breakpointed interpreter pass + windows).
    pub profile_time: std::time::Duration,
    /// Wall time of `specialize_with_config` alone.
    pub specialize_time: std::time::Duration,
}

/// The pipeline: profile `script`, derive the whole `SpecConfig`, specialize, embed, run both.
pub fn auto(script: &str) -> AutoResult {
    let b = auto_build(script);
    let baseline = svm_run::run_powerbox(&b.baseline, script.as_bytes()).expect("baseline run");
    let residual = svm_run::run_powerbox(&b.residual, script.as_bytes()).expect("residual run");
    AutoResult {
        baseline_stdout: baseline.stdout,
        residual_stdout: residual.stdout,
        residual_blocks: b.residual_blocks,
        br_tables: b.br_tables,
        lua_sites: b.lua_sites,
        c_sites: b.c_sites,
        tail_sites: b.tail_sites,
    }
}

/// Profile + assemble + specialize + embed (no runs).
pub fn auto_build(script: &str) -> AutoBuild {
    let t_profile = std::time::Instant::now();
    let m = lua_module();
    let luav = byname(&m, "luaV_execute").expect("luaV_execute");
    let precall = byname(&m, "luaD_precall").expect("precall");
    let poscall = byname(&m, "luaD_poscall").expect("poscall");
    let pretail = byname(&m, "luaD_pretailcall").expect("pretailcall");
    let dispatch = dispatch_block(&m, luav);
    let (pblock, ra_val, _) = call_site(&m, luav, precall, 2, 2);
    // `pretailcall(sp, L, ci, ra, narg1, delta)`: ra at 3, narg1 at 4 (the moved-arg count + 1).
    let (tblock, tra_val, tnarg_val) = call_site(&m, luav, pretail, 3, 4);

    // ---- Profile: roots + main-frame statics, then the event loop. ----
    let inst = svm_run::instantiate(m.clone()).expect("instantiate");
    let mut insp = inst.debug_attach(script.as_bytes().to_vec(), u64::MAX);
    let entry_bp = IrPc {
        module: 0,
        func: luav,
        block: 0,
        inst: 0,
    };
    insp.set_breakpoint(entry_bp);
    let (sp, l, main_ci) = match insp.run_until_stop() {
        Stop::Break { .. } => {
            let g = |i: usize| match insp.read_ir_value(0, i) {
                Some(Value::I64(v)) => v,
                o => panic!("entry v{i}: {o:?}"),
            };
            (g(0), g(1) as u64, g(2) as u64)
        }
        o => panic!("no entry break: {o:?}"),
    };
    insp.clear_breakpoint(entry_bp);
    let w = insp.read_window(0, CAP).expect("entry window");
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

    for b in [pblock, tblock, dispatch] {
        insp.set_breakpoint(IrPc {
            module: 0,
            func: luav,
            block: b as usize,
            inst: 0,
        });
    }

    let mut occs: Vec<Occupancy> = Vec::new();
    let mut sites: Vec<Site> = Vec::new();
    let mut c_sites: Vec<u64> = Vec::new();
    // A pending precall/pretailcall event, resolved by the VERY NEXT event.
    enum Pending {
        Pre(u64),
        Tail(u64, usize),
    }
    let mut pending: Option<Pending> = None;
    let mut ci_previous_off: Option<u64> = None;
    // The `previous`-chain length of `ci` in window `win` (main = 0). Bounded walk.
    let depth_of = |win: &[u8], ci: u64, prev_off: u64, main_ci: u64| -> usize {
        let mut c = ci;
        for d in 0..32 {
            if c == main_ci {
                return d;
            }
            c = rd_u64(win, c + prev_off);
        }
        panic!("unterminated previous chain");
    };

    for _ in 0..MAX_EVENTS {
        let stop = insp.run_until_stop();
        let pc = match &stop {
            Stop::Break {
                reason: StopReason::Breakpoint,
                pc,
            } => *pc,
            Stop::Finished(_) => break,
            o => panic!("unexpected stop: {o:?}"),
        };
        let win = insp.read_window(0, CAP).expect("event window");
        let cur_ci = rd_u64(&win, l + L_CI);
        // Tag discovery: any site still awaiting its tag gets it at the first event whose frame
        // depth is SMALLER than the callee's (the callee has returned, its result slot fresh).
        if let Some(prev_off) = ci_previous_off {
            let d = depth_of(&win, cur_ci, prev_off, main_ci);
            for s in sites.iter_mut().filter(|s| s.tag.is_none()) {
                if d < occs[s.occ].depth {
                    s.tag = Some(win[(occs[s.occ].func + TAG_OFF) as usize]);
                }
            }
        }
        if pc.block == pblock as usize || pc.block == tblock as usize {
            // An unresolved previous pending means it was a C call (no Lua dispatch followed).
            if let Some(Pending::Pre(ra)) = pending.take() {
                if !c_sites.contains(&ra) && !sites.iter().any(|s| s.ra == ra && !s.tail) {
                    c_sites.push(ra);
                }
            }
            let rd_val = |insp: &svm_interp::Inspector, idx: u32| match insp
                .read_ir_value(0, idx as usize)
            {
                Some(Value::I64(v)) => v,
                Some(Value::I32(v)) => v as i64,
                o => panic!("event value: {o:?}"),
            };
            pending = Some(if pc.block == pblock as usize {
                Pending::Pre(rd_val(&insp, ra_val) as u64)
            } else {
                let ra = rd_val(&insp, tra_val) as u64;
                let narg1 = rd_val(&insp, tnarg_val) as usize;
                Pending::Tail(ra, narg1.saturating_sub(1))
            });
            continue;
        }
        // A dispatch event. Identify the current frame.
        let func = rd_u64(&win, cur_ci + CI_FUNC);
        let cl = rd_u64(&win, func);
        let proto = rd_u64(&win, cl + LCLOSURE_P);
        if proto == main_proto {
            // Back in (or still in) main: a pending precall that got here was a C call.
            if let Some(Pending::Pre(ra)) = pending.take() {
                if !c_sites.contains(&ra) && !sites.iter().any(|s| s.ra == ra && !s.tail) {
                    c_sites.push(ra);
                }
            }
            continue;
        }
        let known = occs
            .iter()
            .position(|o| o.ci == cur_ci && o.proto == proto && o.func == func);
        let occ = match known {
            Some(i) => i, // a repeat (loop iteration) — site already recorded
            None => {
                if ci_previous_off.is_none() {
                    ci_previous_off = Some(
                        (0..CI_SIZE as u64)
                            .step_by(8)
                            .find(|&off| rd_u64(&win, cur_ci + off) == main_ci)
                            .expect("ci->previous"),
                    );
                }
                let prev_off = ci_previous_off.unwrap();
                let p = proto;
                occs.push(Occupancy {
                    ci: cur_ci,
                    func,
                    cl,
                    proto,
                    code: rd_u64(&win, p + PROTO_CODE),
                    sizecode: rd_i32(&win, p + PROTO_SIZECODE) as usize,
                    k: rd_u64(&win, p + PROTO_K),
                    sizek: rd_i32(&win, p + PROTO_SIZEK) as usize,
                    pp: rd_u64(&win, p + PROTO_P),
                    sizep: rd_i32(&win, p + PROTO_SIZEP) as usize,
                    depth: depth_of(&win, cur_ci, prev_off, main_ci),
                    w: win,
                });
                occs.len() - 1
            }
        };
        match pending.take() {
            Some(Pending::Pre(ra)) => {
                assert_eq!(
                    occs[occ].func, ra,
                    "a Lua dispatch right after precall must be the callee at that ra"
                );
                if !sites.iter().any(|s| s.ra == ra && !s.tail) {
                    sites.push(Site {
                        ra,
                        occ,
                        tail: false,
                        nargs: 0,
                        tag: None,
                    });
                }
            }
            Some(Pending::Tail(ra, nargs)) if !sites.iter().any(|s| s.ra == ra && s.tail) => {
                sites.push(Site {
                    ra,
                    occ,
                    tail: true,
                    nargs,
                    tag: None,
                });
            }
            Some(Pending::Tail(..)) => {} // a repeat of a known tail site
            None => {}                    // an in-callee dispatch (its own instructions)
        }
    }
    let profile_time = t_profile.elapsed();
    let ci_previous_off = ci_previous_off.expect("at least one Lua call");
    // The savedpc field: at a callee's first dispatch it points at that callee's code start.
    let savedpc_off = {
        let o = &occs[0];
        (0..CI_SIZE as u64)
            .step_by(8)
            .find(|&off| rd_u64(&o.w, o.ci + off) == o.code)
            .expect("ci->savedpc")
    };

    // ---- Assemble the config. ----
    let slice = |src: &[u8], a: u64, n: usize| src[a as usize..a as usize + n].to_vec();
    let mut overlays = vec![
        (l, slice(&w, l, LUA_STATE_SIZE)),
        (stack_lo, slice(&w, stack_lo, stack_len)),
        (main_ci, slice(&w, main_ci, CI_SIZE)),
        (main_code, slice(&w, main_code, 4 * main_sizecode)),
        (main_cl, slice(&w, main_cl, 48)),
        (main_proto, slice(&w, main_proto, 128)),
    ];
    if main_sizek > 0 {
        overlays.push((main_k, slice(&w, main_k, 16 * main_sizek)));
    }
    // Every DEFINED function's proto struct (and sub-proto array), transitively: `OP_CLOSURE` reads
    // `proto->p[bx]->sizeupvalues` to bound its upvalue-init loop, so even a never-called function's
    // proto must fold. Code/k pools are added per OCCUPANCY below (only executed bodies need them).
    let mut proto_work = vec![(main_proto, main_pp, main_sizep)];
    let mut seen_protos = vec![main_proto];
    while let Some((_, pp, sizep)) = proto_work.pop() {
        if sizep == 0 {
            continue;
        }
        overlays.push((pp, slice(&w, pp, 8 * sizep)));
        for i in 0..sizep {
            let sub = rd_u64(&w, pp + 8 * i as u64);
            if !seen_protos.contains(&sub) {
                seen_protos.push(sub);
                overlays.push((sub, slice(&w, sub, 128)));
                proto_work.push((
                    sub,
                    rd_u64(&w, sub + PROTO_P),
                    rd_i32(&w, sub + PROTO_SIZEP) as usize,
                ));
            }
        }
    }
    let mut rename_extra = vec![(stack_lo, stack_hi), (main_ci, main_ci + CI_SIZE as u64)];
    // One ci overlay per distinct frame node, from its FIRST occupancy's window; per-site pins fix
    // the per-occupancy fields. Non-overlap assert: the I71(b) collision guard.
    let mut seen_ci: Vec<u64> = vec![main_ci];
    let mut seen_static: Vec<u64> = Vec::new();
    for o in &occs {
        if !seen_ci.contains(&o.ci) {
            seen_ci.push(o.ci);
            overlays.push((o.ci, slice(&o.w, o.ci, CI_SIZE)));
            rename_extra.push((o.ci, o.ci + CI_SIZE as u64));
        }
        if !seen_static.contains(&o.proto) {
            seen_static.push(o.proto);
            overlays.push((o.cl, slice(&o.w, o.cl, 48)));
            if !seen_protos.contains(&o.proto) {
                overlays.push((o.proto, slice(&o.w, o.proto, 128)));
            }
            overlays.push((o.code, slice(&o.w, o.code, 4 * o.sizecode)));
            if o.sizek > 0 && o.k != main_k {
                overlays.push((o.k, slice(&o.w, o.k, 16 * o.sizek)));
            }
            if o.sizep > 0 {
                overlays.push((o.pp, slice(&o.w, o.pp, 8 * o.sizep)));
            }
        }
    }
    let mut cis = seen_ci.clone();
    cis.sort();
    for p in cis.windows(2) {
        assert!(
            p[0] + CI_SIZE as u64 <= p[1],
            "CallInfo overlays must not overlap (I71 facet b): {:#x}+{CI_SIZE} > {:#x}",
            p[0],
            p[1]
        );
    }

    let pins_of = |s: &Site| {
        let o = &occs[s.occ];
        vec![
            (o.func, o.cl),
            (o.ci + CI_FUNC, o.func),
            (o.ci + savedpc_off, o.code),
        ]
    };
    let lua_sites: Vec<LuaSite> = sites
        .iter()
        .filter(|s| !s.tail)
        .map(|s| LuaSite {
            ra: s.ra,
            callee_ci: occs[s.occ].ci,
            pins: pins_of(s),
        })
        .collect();
    // A tail site's moved arguments: value cells reload dynamic, tags pin to their captured bytes
    // (read from the occupancy window, where the move has already happened).
    let tail_sites: Vec<TailSite> = sites
        .iter()
        .filter(|s| s.tail)
        .map(|s| {
            let o = &occs[s.occ];
            TailSite {
                ra: s.ra,
                callee_ci: o.ci,
                pins: pins_of(s),
                args: (0..s.nargs)
                    .map(|j| {
                        let slot = o.func + 16 * (j as u64 + 1);
                        (slot, o.w[(slot + TAG_OFF) as usize])
                    })
                    .collect(),
            }
        })
        .collect();
    // One selective entry per distinct frame node; the discovered tag (integer fallback for a call
    // whose return was the program's last activity).
    let mut selective: BTreeMap<u64, u8> = BTreeMap::new();
    for s in &sites {
        selective
            .entry(occs[s.occ].ci)
            .or_insert_with(|| s.tag.unwrap_or(VNUMINT));
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

    let n_lua = lua_sites.len();
    let n_c = c_sites.len();
    let n_tail = tail_sites.len();
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
            lua_sites,
            c_sites,
            poscall: Some(PoscallModel {
                poscall,
                ci_previous_off,
                ci_func_off: CI_FUNC,
                tag_off: TAG_OFF,
                selective: selective.into_iter().collect(),
            }),
        }),
        pretailcall_model: (!tail_sites.is_empty()).then_some(PretailcallModel {
            pretailcall: pretail,
            ra_arg: 3,
            l_ci_addr: l + L_CI,
            tag_off: TAG_OFF,
            sites: tail_sites,
        }),
        ..SpecConfig::default()
    };

    // ---- Specialize, embed, verify, run both. ----
    let args = [
        SpecArg::ConstI64(sp),
        SpecArg::ConstI64(l as i64),
        SpecArg::ConstI64(main_ci as i64),
    ];
    let t_spec = std::time::Instant::now();
    let mut r = specialize_with_config(&m, luav, &args, &cfg).expect("auto projection");
    let specialize_time = t_spec.elapsed();
    let entry = (r.funcs.len() - 1) as u32;
    let f = &r.funcs[entry as usize];
    let residual_blocks = f.blocks.len();
    let br_tables = f
        .blocks
        .iter()
        .filter(|b| matches!(b.term, Terminator::BrTable { .. }))
        .count();
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
    AutoBuild {
        baseline: m,
        residual: r,
        residual_blocks,
        br_tables,
        lua_sites: n_lua,
        c_sites: n_c,
        tail_sites: n_tail,
        profile_time,
        specialize_time,
    }
}
