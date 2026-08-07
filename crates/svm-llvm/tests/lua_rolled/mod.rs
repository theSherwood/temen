//! **Zero-config auto-rolled Lua residual — the driver-level `auto_rolled`.**
//!
//! `lua_futamura_auto_rolled.rs` first showed the rolled residual could be built with no hardcoded
//! offsets; this promotes that to a reusable driver function on top of the shared
//! [`peval_capture::discover`], so a caller hands it a Lua chunk and gets back the rolled residual +
//! the metadata to run/verify it (dynamic cells, the trip counter, the accumulator). Both the focused
//! correctness test and the end-to-end benchmark call this — one code path, script in → residual out.
//!
//! Scope: a single top-level numeric-`for` accumulator chunk (`local acc = …; for i = a,b do acc = … end`).
//! The accumulator is identified generically as the **lowest-address carried cell that is not the trip
//! counter** — i.e. the first local declared before the loop. That covers the common shape; a chunk
//! whose result is not that register would need the dataflow-based identification noted in DESIGN.

#![allow(dead_code)]

use super::peval_capture::{self, Located, TargetDesc};
use svm_interp::{IrPc, Stop, StopReason, Value};
use svm_ir::{Block, Func, Inst, LoadOp, Module, Terminator, ValType};
use svm_peval::{specialize_with_config, SpecArg, SpecConfig};

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
    let p = format!(
        "{}/tests/fixtures/lua/lua_eval.ll",
        env!("CARGO_MANIFEST_DIR")
    );
    svm_llvm::translate_ll_path(&p).expect("translate").module
}
pub fn luav_execute(m: &Module) -> u32 {
    m.exports
        .iter()
        .find(|e| e.name == "luaV_execute")
        .expect("export")
        .func
}

fn read_entry(insp: &mut svm_interp::Inspector, luav: u32) -> (i64, u64, u64) {
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
    /// The rolled residual, entry function 0; params are the dynamic cells (ascending).
    pub residual: Module,
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
    let inst = svm_run::instantiate(m.clone()).expect("instantiate");
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
        let inst = svm_run::instantiate(m.clone()).expect("instantiate");
        let mut insp = inst.debug_attach(script_owned.as_bytes().to_vec(), u64::MAX);
        read_entry(&mut insp, luav);
        insp
    };
    let d = peval_capture::discover(&make_insp, luav, dispatch, &loc, &desc);
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

    let cfg = SpecConfig {
        const_overlays: vec![
            (l, slice(l, LUA_STATE_SIZE)),
            (stack_lo, slice(stack_lo, stack_len)),
            (ci, slice(ci, CI_SIZE)),
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
    let args = [
        SpecArg::ConstI64(sp),
        SpecArg::ConstI64(l as i64),
        SpecArg::ConstI64(ci as i64),
    ];
    let residual = specialize_with_config(m, luav, &args, &cfg).expect("auto-rolled projection");
    svm_verify::verify_module(&residual).expect("residual verifies");

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
        residual_blocks: residual.funcs[0].blocks.len(),
        base_blocks: m.funcs[luav as usize].blocks.len(),
        residual,
        dyn_cells: d.varying,
        counter_ix,
        acc_ix,
        acc_addr,
        captured,
        reg_base: base,
        reg_stride: STACKVALUE_SIZE,
    }
}

/// Append a `(dyn0, dyn1, …) -> i64` wrapper that calls the rolled residual (entry 0) then loads the
/// accumulator cell it wrote back.
pub fn with_readback(residual: &Module, read_addr: u64, nparams: usize) -> (Module, u32) {
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
                    func: 0,
                    args: call_args,
                },
                Inst::Load {
                    op: LoadOp::I64,
                    addr: addr_v,
                    offset: 0,
                    align: 8,
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
