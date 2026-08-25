//! **Zero-config auto-rolled Lua residual — the driver-level `auto_rolled`.**
//!
//! `lua_futamura_auto_rolled.rs` first showed the rolled residual could be built with no hardcoded
//! offsets; this promotes that to a reusable driver function on top of the shared
//! [`peval_capture::discover`], so a caller hands it a Lua chunk and gets back the rolled residual +
//! the metadata to run/verify it (dynamic cells, the trip counter, the accumulator). Both the focused
//! correctness test and the end-to-end benchmark call this — one code path, script in → residual out.
//!
//! Scope: a single top-level numeric-`for` accumulator chunk (`local acc = …; for i = a,b do acc = … end`).
//! The result cell is identified generically by decoding the chunk's `RETURN`/`RETURN1` bytecode (its
//! `A` operand is the returned register), so multi-accumulator chunks like fibonacci work too. Integer
//! `%` / `//` are in scope: their divide-by-zero cold arm deopts to the carried baseline (the divisor
//! must be a register operand — a local, not a bare constant literal; see `auto_rolled`).

#![allow(dead_code)]

use crate::capture::{self as peval_capture, Located, TargetDesc};
use temen_interp::{IrPc, Stop, StopReason, Value};
use temen_ir::{Block, Func, Inst, LoadOp, Module, Terminator, ValType};
use temen_peval::{specialize_with_config, SpecArg, SpecConfig};

const CAPTURE_LEN: usize = 8 << 20;
// Lua 5.4.7 offsets (call-free loop ⇒ the 104-byte ci over-slice is safe: no adjacent-frame collision).
const L_STACK_LAST: u64 = 40;
const L_STACK: u64 = 48;
const LUA_STATE_SIZE: usize = 200;
const CI_SIZE: usize = 104;
const CI_FUNC: u64 = 0;
const CI_SAVEDPC: u64 = 32;
const LCLOSURE_P: u64 = 24;
const PROTO_SIZECODE: u64 = 24;
const PROTO_CODE: u64 = 64;
const STACKVALUE_SIZE: u64 = 16;
const VNUMINT_TAG: u8 = 0x03;

fn rd_u64(w: &[u8], a: u64) -> u64 {
    u64::from_le_bytes(w[a as usize..a as usize + 8].try_into().unwrap())
}
fn rd_i32(w: &[u8], a: u64) -> i32 {
    i32::from_le_bytes(w[a as usize..a as usize + 4].try_into().unwrap())
}

pub fn lua_module() -> Module {
    // From the out-of-workspace bench crate, the fixture lives under the temen-llvm crate's tests.
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../crates/temen-llvm/tests/fixtures/lua/lua_eval.ll"
    );
    temen_llvm::translate_ll_path(p).expect("translate").module
}
pub fn luav_execute(m: &Module) -> u32 {
    m.exports
        .iter()
        .find(|e| e.name == "luaV_execute")
        .expect("export")
        .func
}

fn read_entry(insp: &mut temen_interp::Inspector, luav: u32) -> (i64, u64, u64) {
    let entry = IrPc {
        module: 0,
        func: luav,
        block: 0,
        inst: 0,
    };
    insp.set_breakpoint(entry);
    let r = match insp.run_until_stop() {
        Stop::Break {
            reason: StopReason::Breakpoint,
            ..
        } => {
            let g = |i| match insp.read_ir_value(0, i) {
                Some(Value::I64(v)) => v,
                o => panic!("entry v: {o:?}"),
            };
            (g(0), g(1) as u64, g(2) as u64)
        }
        o => panic!("no entry break: {o:?}"),
    };
    insp.clear_breakpoint(entry);
    r
}

/// A zero-config rolled residual + everything needed to run and verify it.
pub struct AutoRolled {
    /// The rolled residual; params are the dynamic cells (ascending).
    pub residual: Module,
    /// Entry function of the residual. `0` on the fast path; on the whole-module-carry deopt fallback
    /// (used to deopt the div/mod cold arm for `%` / `//`) the entry lands at the last function, so
    /// callers must dispatch through this rather than assuming func 0.
    pub entry: u32,
    /// Dynamic-cell addresses (ascending) — the residual's parameters, positionally.
    pub dyn_cells: Vec<u64>,
    /// Index (in `dyn_cells` / residual params) of the trip counter — sweep this to vary the loop.
    pub counter_ix: usize,
    /// Index of the accumulator cell (the residual writes it back; read it for the result).
    pub acc_ix: usize,
    /// Address of the accumulator cell.
    pub acc_addr: u64,
    /// Captured seed value of each dynamic cell (positional) — reproduces the original run.
    pub captured: Vec<i64>,
    /// Residual and baseline `luaV_execute` block counts (for reporting).
    pub residual_blocks: usize,
    pub base_blocks: usize,
    /// Register base (`ci->func + 1 TValue`) — so callers can report cells as `R[n]`.
    pub reg_base: u64,
    /// Register stride in bytes.
    pub reg_stride: u64,
}

/// Build the rolled residual for a single numeric-`for` accumulator chunk, fully automatically.
pub fn auto_rolled(m: &Module, script: &str) -> AutoRolled {
    let luav = luav_execute(m);
    let dispatch = peval_capture::dispatch_block(m, luav);

    // Locate the register base + entry scalars.
    let inst = temen_run::instantiate(m.clone()).expect("instantiate");
    let mut insp0 = inst.debug_attach(script.as_bytes().to_vec(), u64::MAX);
    let (sp, l, ci) = read_entry(&mut insp0, luav);
    let w0 = insp0.read_window(0, CAPTURE_LEN).expect("window");
    drop(insp0);
    let base = rd_u64(&w0, ci + CI_FUNC) + STACKVALUE_SIZE;

    let loc = Located {
        reg_base: base,
        pc_addr: Some(ci + CI_SAVEDPC),
    };
    let desc = TargetDesc {
        reg_stride: STACKVALUE_SIZE,
        n_regs: 32,
        tag: Some((8, VNUMINT_TAG)),
        capture_len: CAPTURE_LEN,
        observe_hits: 48,
    };
    let script_owned = script.to_string();
    let make_insp = move || {
        let inst = temen_run::instantiate(m.clone()).expect("instantiate");
        let mut insp = inst.debug_attach(script_owned.as_bytes().to_vec(), u64::MAX);
        read_entry(&mut insp, luav);
        insp
    };
    let mut d = peval_capture::discover(&make_insp, luav, dispatch, &loc, &desc);
    assert!(d.counter != 0, "no trip counter discovered");
    assert!(
        d.varying.contains(&d.counter),
        "counter must be a dynamic cell"
    );

    let w = &d.window;
    let stack_lo = rd_u64(w, l + L_STACK);
    let stack_hi = rd_u64(w, l + L_STACK_LAST);
    let stack_len = (stack_hi - stack_lo) as usize;
    let func = rd_u64(w, ci + CI_FUNC);
    let closure = rd_u64(w, func);
    let proto = rd_u64(w, closure + LCLOSURE_P);
    let code = rd_u64(w, proto + PROTO_CODE);
    let sizecode = rd_i32(w, proto + PROTO_SIZECODE) as usize;
    let slice = |a: u64, n: usize| w[a as usize..a as usize + n].to_vec();

    // Integer `//` (OP_IDIV) and `%` (OP_MOD) are the only arithmetic ops that *raise* on a bad
    // operand — a zero divisor, or `INT_MIN // -1` overflow — and the register-register forms inline
    // that guard (and its `luaG_runerror` cold arm) directly in `luaV_execute`. The proven recipe
    // (`lua_futamura_deopt.rs`) is to make the **divisor a dynamic cell** so the `divisor == 0` guard is
    // a *dynamic* branch, then deopt its cold arm: the division folds on the hot path and the error case
    // deopts to the baseline. So decode every OP_IDIV / OP_MOD and mark its divisor register (operand C)
    // dynamic. A loop-invariant divisor (`local d = 2`) would otherwise sit in the renamed mutable stack
    // and load as an untracked `Dyn`, leaving the guard un-deoptable; forcing it a declared dynamic cell
    // is what threads it to the deopt edge. Lua 5.4 iABC: op=bits0..6, A=7..14, B=16..23, C=24..31; the
    // K-forms (OP_IDIVK / OP_MODK) instead *call* `luaV_idiv` / `luaV_mod`, whose error is inside the
    // helper (not inline, so not deoptable here) — those stay a scope wall, so a constant divisor must be
    // written as a local to become a register operand. Straight-line chunks contain no such op: inert.
    const OP_MOD: u32 = 37;
    const OP_IDIV: u32 = 40;
    for pc in 0..sizecode {
        let word = u32::from_le_bytes(
            w[(code + pc as u64 * 4) as usize..(code + pc as u64 * 4) as usize + 4]
                .try_into()
                .unwrap(),
        );
        let op = word & 0x7F;
        if op == OP_MOD || op == OP_IDIV {
            // Both the divisor (operand C) and the result temp (operand A) are dynamic: C so the
            // `divisor == 0` guard is a dynamic, deoptable branch; A because the division result is
            // dynamic and must be a declared cell for the residual to store it and thread it across the
            // deopt edge (the baseline resumes from the register file). Matches the hand-picked cells in
            // `lua_futamura_deopt.rs` (the divisor and the IDIV temp).
            let a = ((word >> 7) & 0xFF) as u64; // result register
            let c = ((word >> 24) & 0xFF) as u64; // divisor register
            for reg in [a, c] {
                let addr = base + (reg + 1) * STACKVALUE_SIZE; // +1: VARARGPREP frame shift
                if !d.varying.contains(&addr) {
                    d.varying.push(addr);
                }
            }
        }
    }
    d.varying.sort_unstable(); // params are positional-ascending; keep the order stable after the union

    // **Resume off the metamethod trailer.** Every Lua 5.4 arithmetic op is followed by an `OP_MMBIN*`
    // (the metamethod fallback for non-number operands). The generic discovery safepoint often lands on
    // that trailer rather than on the arithmetic op itself — and resuming a residual *at* `OP_MMBIN`
    // forces the specializer to project the metamethod-miss path (`luaT_trybinTM` → allocate an error →
    // the throw machinery), which is un-foldable and hits `Unsupported`. The arithmetic op one
    // instruction earlier is a clean resume point (its fast integer path folds; `OP_MMBIN` is then a
    // known no-op reached with a constant "no metamethod" flag). Rewinding is state-consistent: the op's
    // result register is a declared dynamic cell, so the resumed op simply recomputes it before the next
    // instruction reads it, and no other register moved. Lua 5.4 metamethod trailers: MMBIN=46,
    // MMBINI=47, MMBINK=48. (Only div/mod's safepoint lands here; straight-line chunks resume on the op.)
    let mut ci_image = slice(ci, CI_SIZE);
    let savedpc = rd_u64(w, ci + CI_SAVEDPC);
    let pc_ix = (savedpc - code) / 4;
    if pc_ix > 0 {
        let word = u32::from_le_bytes(
            w[savedpc as usize..savedpc as usize + 4]
                .try_into()
                .unwrap(),
        );
        let op = word & 0x7F;
        if matches!(op, 46..=48) {
            let rewound = savedpc - 4;
            ci_image[CI_SAVEDPC as usize..CI_SAVEDPC as usize + 8]
                .copy_from_slice(&rewound.to_le_bytes());
        }
    }

    // The error/throw cold arms: an arithmetic op like `%` / `//` guards its operands (a `divisor == 0`,
    // an `INT_MIN // -1` overflow, a non-number metamethod miss) and, on the cold side, raises a real
    // Lua error via the setjmp/longjmp machinery (`luaG_runerror` → `luaD_throw`). That unwind path is
    // inherently un-foldable, so projecting into it hits `Unsupported`. The fix mirrors the tiered-JIT
    // shape: **fold the hot arithmetic and deopt the cold arm to the carried baseline `luaV_execute`**,
    // which resumes from the spilled `savedpc` and raises the real error. Find those cold arms
    // structurally — every `luaV_execute` block that calls a function which reaches setjmp/longjmp — and
    // mark them `deopt_targets`. A straight-line integer chunk enters none of them, so this is inert
    // there (the fast path below handles it with no carry at all).
    let callees = |f: &Func| -> Vec<u32> {
        f.blocks
            .iter()
            .flat_map(|b| {
                b.insts.iter().filter_map(|i| match i {
                    Inst::Call { func, .. } => Some(*func),
                    _ => None,
                })
            })
            .collect()
    };
    // An **error raiser** is a function that never returns to its caller — every exit is a `longjmp`
    // (or a tail-call to another error raiser). Lua's `luaG_runerror` / `luaG_opinterror` /
    // `luaD_throw` are exactly these; an allocator or GC step *reaches* the throw machinery on its OOM
    // arm but **returns normally** on the hot path, so it is not a raiser. That distinction is the
    // whole game: deopting a block because it calls a raiser targets a genuine error arm; deopting one
    // because it calls a merely-can-throw helper (an allocator on the FORLOOP's hot path) would deopt
    // the loop itself. So the raiser test is "reaches setjmp **and** has no normal return".
    let n = m.funcs.len();
    let mut reaches = vec![false; n];
    for (f, func) in m.funcs.iter().enumerate() {
        reaches[f] = func.uses_setjmp();
    }
    loop {
        let mut changed = false;
        for f in 0..n {
            if reaches[f] {
                continue;
            }
            if callees(&m.funcs[f]).iter().any(|&g| reaches[g as usize]) {
                reaches[f] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let never_returns = |f: &Func| -> bool {
        !f.blocks
            .iter()
            .any(|b| matches!(b.term, Terminator::Return(_)))
    };
    let is_raiser: Vec<bool> = m
        .funcs
        .iter()
        .enumerate()
        .map(|(f, func)| reaches[f] && never_returns(func) && func.blocks.len() > 1)
        .collect();
    // The cold-arm blocks of `luaV_execute`: those calling an error raiser (not the recursive
    // `luaV_execute` self-call — that is the OP_CALL path). Deopting *only* these folds `%` / `//` on
    // the hot path and sends the actual error case to the baseline.
    let deopt_blocks: Vec<u32> = m.funcs[luav as usize]
        .blocks
        .iter()
        .enumerate()
        .filter(|(_, b)| {
            b.insts.iter().any(|i| {
                matches!(i, Inst::Call { func, .. } if *func != luav && is_raiser[*func as usize])
            })
        })
        .map(|(bi, _)| bi as u32)
        .collect();
    let args = [
        SpecArg::ConstI64(sp),
        SpecArg::ConstI64(l as i64),
        SpecArg::ConstI64(ci as i64),
    ];
    let base_cfg = || SpecConfig {
        const_overlays: vec![
            (l, slice(l, LUA_STATE_SIZE)),
            (stack_lo, slice(stack_lo, stack_len)),
            (ci, ci_image.clone()),
            (code, slice(code, 4 * sizecode)),
            (closure, slice(closure, 48)),
            (proto, slice(proto, 128)),
        ],
        rename: Some((l, l + LUA_STATE_SIZE as u64)),
        rename_extra: vec![(stack_lo, stack_hi), (ci, ci + CI_SIZE as u64)],
        rename_is_private: true,
        rename_seed_from_image: true,
        dynamic_cells: d.varying.iter().map(|&a| (a, 8)).collect(),
        indirect_targets_cap: Some(16),
        ..SpecConfig::default()
    };
    // Fast path: a straight-line integer chunk projects to a small standalone residual (entry = func 0,
    // calls nothing). If that hits `Unsupported` (e.g. `%` / `//` projects into the un-foldable
    // setjmp/longjmp error path), fall back to deopting every cold error arm to the carried baseline
    // `luaV_execute` — the hot arithmetic still folds and the residual entry lands at the last function.
    let (residual, entry) = match specialize_with_config(m, luav, &args, &base_cfg()) {
        Ok(r) => (r, 0u32),
        Err(_) => {
            let cfg = SpecConfig {
                deopt_targets: deopt_blocks.iter().map(|&b| (luav, b)).collect(),
                deopt_handler: Some(luav),
                carry_whole_module: true,
                ..base_cfg()
            };
            let r = specialize_with_config(m, luav, &args, &cfg)
                .expect("auto-rolled projection (with cold-arm deopt)");
            let e = (r.funcs.len() - 1) as u32;
            (r, e)
        }
    };
    temen_verify::verify_module(&residual).expect("residual verifies");

    let counter_ix = d.varying.iter().position(|&a| a == d.counter).unwrap();
    // The **result register**: the chunk's `return <v>` compiles to a `RETURN1 A` (or `RETURN A ...`)
    // bytecode whose `A` is the returned register. Decoding it (a single lookup in the proto code) is
    // the generic, dataflow-free way to know which cell holds the program's result — unlike the
    // "lowest carried cell" heuristic, this is correct for multi-accumulator chunks (fibonacci returns
    // `b`, not the lowest register). The residual writes the register file back on return even though
    // it stops before executing the `RETURN`, so reading that register gives the result.
    // Lua 5.4.7 opcodes: RETURN=70, RETURN0=71, RETURN1=72. `A` is bits 7..15, `B` bits 16..24. The
    // real `return <v>` is `RETURN1` (always one value) or a `RETURN` with `B != 1` (B-1 values; the
    // implicit end-of-chunk `RETURN A 1` returns nothing). `A` is in the chunk's own register
    // numbering; VARARGPREP shifts the runtime frame up by one, so the cell address is
    // `base + (A + 1)*stride`.
    let mut result_reg: Option<u64> = None;
    for pc in 0..sizecode {
        let word = u32::from_le_bytes(
            w[(code + pc as u64 * 4) as usize..(code + pc as u64 * 4) as usize + 4]
                .try_into()
                .unwrap(),
        );
        let op = word & 0x7F;
        let a = ((word >> 7) & 0xFF) as u64;
        let b = (word >> 16) & 0xFF;
        if op == 72 || (op == 70 && b != 1) {
            result_reg = Some(a);
            break;
        }
    }
    // Accumulator = the returned register (+1 VARARGPREP shift) if it is a discovered carried cell,
    // else the lowest carried non-counter cell.
    let acc_addr = result_reg
        .map(|a| base + (a + 1) * STACKVALUE_SIZE)
        .filter(|a| d.varying.contains(a))
        .unwrap_or_else(|| {
            *d.varying
                .iter()
                .find(|&&a| a != d.counter)
                .expect("a carried cell distinct from the counter")
        });
    let acc_ix = d.varying.iter().position(|&a| a == acc_addr).unwrap_or(0);
    let captured: Vec<i64> = d.varying.iter().map(|&a| rd_u64(w, a) as i64).collect();

    AutoRolled {
        residual_blocks: residual.funcs[entry as usize].blocks.len(),
        base_blocks: m.funcs[luav as usize].blocks.len(),
        residual,
        entry,
        dyn_cells: d.varying,
        counter_ix,
        acc_ix,
        acc_addr,
        captured,
        reg_base: base,
        reg_stride: STACKVALUE_SIZE,
    }
}

/// Append a `(dyn0, dyn1, …) -> i64` wrapper that calls the rolled residual (its `entry` function)
/// then loads the accumulator cell it wrote back.
pub fn with_readback(
    residual: &Module,
    entry: u32,
    read_addr: u64,
    nparams: usize,
) -> (Module, u32) {
    let mut m = residual.clone();
    let wrapper = m.funcs.len() as u32;
    let params: Vec<ValType> = vec![ValType::I64; nparams];
    let call_args: Vec<u32> = (0..nparams as u32).collect();
    let addr_v = nparams as u32;
    let load_v = nparams as u32 + 1;
    m.funcs.push(Func {
        params: params.clone(),
        results: vec![ValType::I64],
        blocks: vec![Block {
            params,
            insts: vec![
                Inst::ConstI64(read_addr as i64),
                Inst::Call {
                    func: entry,
                    args: call_args,
                },
                Inst::Load {
                    op: LoadOp::I64,
                    addr: addr_v,
                    offset: 0,
                },
            ],
            term: Terminator::Return(vec![load_v]),
        }],
    });
    (m, wrapper)
}

pub fn has_br_table(f: &Func) -> bool {
    f.blocks
        .iter()
        .any(|b| matches!(b.term, Terminator::BrTable { .. }))
}
