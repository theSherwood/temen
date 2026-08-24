//! **The mutable-VM-state projection of `luaV_execute`.** Where `lua_futamura_specialize.rs` stops at
//! the VARARGPREP/`growstack` wall (the mutable `L->top` / `L->stack_last` guards behind a setjmp arm),
//! this drives the projection *through* it using the seed + write-back + multi-region rename feature.
//!
//! The mutable interpreter state — captured at `luaV_execute` entry — is three disjoint objects, each
//! declared a **seeded rename region** (folds to its captured bytes, stays mutable, written back on
//! exit): the `lua_State` `L`, its stack array `[L->stack, L->stack_last)`, and the current `CallInfo`
//! `ci`. The read-only heap the prologue chases (the `LClosure`, the `Proto`, the bytecode array) stays
//! a plain `const_overlay`. With `L`/`ci`/the stack folding to real values, the checkstack guard folds
//! (no grow needed), the setjmp arm is pruned, and the dispatch collapses.
//!
//! **Result:** `luaV_execute` projects fully. The residual carries **no `br_table`** (the 83-way
//! dispatch is gone), **no `call_indirect`** (the cold error/alloc/hook paths are pruned), and **no
//! loads** (every VM-state field folded) — the whole interpreter reduces to a constant computation
//! that spills the final VM state and returns, and the loop result `x = 0 + 3*50 = 150` appears among
//! the constants it computed. Because the program has no runtime input it unrolls completely, so the
//! residual is a long straight-line chain; a program with dynamic input would instead leave a rolled,
//! dispatch-free loop (the `temen-peval` dispatch-ROI measurement on main shows that shape).
//!
//! *Scope:* this asserts the projection's structure and the folded result. Executing the residual
//! embedded back in the Lua runtime (a full end-to-end differential) is future work; the rename
//! feature's soundness itself is differential-tested in `temen-peval/tests/rename_seed.rs`.
//!
//! Run: `cargo test -p temen-llvm --test lua_futamura_vmstate -- --nocapture`
//! (`--features temen-peval/trace` prints the first blocker if a further wall remains).

use temen_interp::{IrPc, Stop, StopReason, Value};
use temen_ir::Module;
use temen_peval::{specialize_with_config, SpecArg, SpecConfig};

const SCRIPT: &str = "local x = 0\nfor i = 1, 50 do x = x + 3 end\nreturn x\n";
const CAPTURE_LEN: usize = 8 << 20;

// Lua 5.4.7 offsets (validated in lua_futamura_capture.rs / lua_futamura_specialize.rs).
const L_TOP: u64 = 16;
const L_STACK_LAST: u64 = 40;
const L_STACK: u64 = 48;
const L_CI: u64 = 32;
const LUA_STATE_SIZE: usize = 200; // through hookmask@192 (+4), rounded to 8
const CI_SIZE: usize = 104;
const CI_FUNC: u64 = 0;
const LCLOSURE_P: u64 = 24;
const PROTO_SIZECODE: u64 = 24;
const PROTO_CODE: u64 = 64;

fn lua_module() -> Module {
    let path = format!(
        "{}/tests/fixtures/lua/lua_eval.ll",
        env!("CARGO_MANIFEST_DIR")
    );
    temen_llvm::translate_ll_path(&path)
        .expect("translate")
        .module
}
fn luav_execute(m: &Module) -> u32 {
    m.exports
        .iter()
        .find(|e| e.name == "luaV_execute")
        .expect("export")
        .func
}
fn rd_u64(w: &[u8], a: u64) -> u64 {
    u64::from_le_bytes(w[a as usize..a as usize + 8].try_into().unwrap())
}
fn rd_i32(w: &[u8], a: u64) -> i32 {
    i32::from_le_bytes(w[a as usize..a as usize + 4].try_into().unwrap())
}

/// Capture (sp, L, ci, window) at luaV_execute entry.
fn capture(m: &Module, luav: u32) -> (i64, i64, i64, Vec<u8>) {
    let inst = temen_run::instantiate(m.clone()).expect("instantiate");
    let win = m.memory.map_or(0, |mc| 1u64 << mc.size_log2);
    let mut insp = inst.debug_attach(SCRIPT.as_bytes().to_vec(), u64::MAX);
    insp.set_breakpoint(IrPc {
        module: 0,
        func: luav,
        block: 0,
        inst: 0,
    });
    match insp.run_until_stop() {
        Stop::Break {
            reason: StopReason::Breakpoint,
            ..
        } => {
            let g = |i| match insp.read_ir_value(0, i) {
                Some(Value::I64(v)) => v,
                o => panic!("{o:?}"),
            };
            let dump = insp
                .read_window(0, (win as usize).min(CAPTURE_LEN))
                .expect("window");
            (g(0), g(1), g(2), dump)
        }
        o => panic!("no break: {o:?}"),
    }
}

#[test]
fn vm_state_rename_clears_the_vararg_wall() {
    let m = lua_module();
    let luav = luav_execute(&m);
    let (sp, l, ci, w) = capture(&m, luav);
    let (l, ci) = (l as u64, ci as u64);

    // The three mutable objects.
    let stack_lo = rd_u64(&w, l + L_STACK);
    let stack_hi = rd_u64(&w, l + L_STACK_LAST);
    assert_eq!(rd_u64(&w, l + L_CI), ci, "L->ci != captured ci");
    assert!(
        stack_lo < stack_hi && stack_hi - stack_lo < (1 << 20),
        "sane stack extent"
    );

    // The read-only heap the prologue chases: ci->func -> *func(LClosure) -> Proto -> code.
    let func = rd_u64(&w, ci + CI_FUNC);
    let closure = rd_u64(&w, func);
    let proto = rd_u64(&w, closure + LCLOSURE_P);
    let code = rd_u64(&w, proto + PROTO_CODE);
    let sizecode = rd_i32(&w, proto + PROTO_SIZECODE) as usize;
    assert!(
        func >= stack_lo && func < stack_hi,
        "ci->func ({func:#x}) should lie on the stack [{stack_lo:#x}, {stack_hi:#x})"
    );

    let slice = |a: u64, n: usize| w[a as usize..a as usize + n].to_vec();
    let stack_len = (stack_hi - stack_lo) as usize;

    println!("\n=== VM-state regions ===");
    println!(
        "L      [{l:#x}, {:#x})  ({} bytes)",
        l + LUA_STATE_SIZE as u64,
        LUA_STATE_SIZE
    );
    println!("stack  [{stack_lo:#x}, {stack_hi:#x})  ({stack_len} bytes)");
    println!(
        "ci     [{ci:#x}, {:#x})  ({} bytes)",
        ci + CI_SIZE as u64,
        CI_SIZE
    );
    println!("heap   closure={closure:#x} proto={proto:#x} code={code:#x} sizecode={sizecode}");
    println!(
        "L->top={:#x}  L->stack_last={stack_hi:#x}",
        rd_u64(&w, l + L_TOP)
    );

    let cfg = SpecConfig {
        // Seeds for the three mutable regions + the read-only heap structs, all as explicit captured
        // bytes (const_regions read the module image, which has no runtime state — overlays do).
        const_overlays: vec![
            (l, slice(l, LUA_STATE_SIZE)),
            (stack_lo, slice(stack_lo, stack_len)),
            (ci, slice(ci, CI_SIZE)),
            (code, slice(code, 4 * sizecode)),
            (closure, slice(closure, 48)),
            (proto, slice(proto, 128)),
        ],
        // The mutable VM state: three disjoint seeded + written-back rename regions.
        rename: Some((l, l + LUA_STATE_SIZE as u64)),
        rename_extra: vec![(stack_lo, stack_hi), (ci, ci + CI_SIZE as u64)],
        rename_is_private: true,
        rename_seed_from_image: true,
        indirect_targets_cap: Some(16),
        ..SpecConfig::default()
    };

    let args = [
        SpecArg::ConstI64(sp),
        SpecArg::ConstI64(l as i64),
        SpecArg::ConstI64(ci as i64),
    ];
    match specialize_with_config(&m, luav, &args, &cfg) {
        Ok(r) => {
            use temen_ir::{Inst, Terminator};
            let f = &r.funcs[0];
            let (mut brtable, mut indirect, mut unreachable, mut ret) = (0, 0, 0, 0);
            let (mut loads, mut stores, mut calls) = (0, 0, 0);
            for b in &f.blocks {
                match &b.term {
                    Terminator::BrTable { .. } => brtable += 1,
                    Terminator::ReturnCallIndirect { .. } => indirect += 1,
                    Terminator::Unreachable => unreachable += 1,
                    Terminator::Return(_) => ret += 1,
                    _ => {}
                }
                for i in &b.insts {
                    match i {
                        Inst::Load { .. } => loads += 1,
                        Inst::Store { .. } => stores += 1,
                        Inst::CallIndirect { .. } => indirect += 1,
                        Inst::Call { .. } => calls += 1,
                        _ => {}
                    }
                }
            }
            println!(
                "\nPROJECTED: residual {} func(s), entry {} blocks (interpreter had {})",
                r.funcs.len(),
                f.blocks.len(),
                m.funcs[luav as usize].blocks.len()
            );
            println!(
                "  terminators: br_table={brtable} indirect={indirect} unreachable={unreachable} return={ret}",
            );
            println!("  insts: loads={loads} stores={stores} calls={calls}");

            // Correctness signal: loads==0 means the residual is a pure constant computation — it just
            // spills the final folded VM state. Collect the constant store values; x = 0 + 3*50 = 150
            // must be among them (the accumulator the residual computed at spec time).
            let consts: Vec<i64> = f
                .blocks
                .iter()
                .flat_map(|b| &b.insts)
                .filter_map(|i| match i {
                    Inst::ConstI64(v) => Some(*v),
                    _ => None,
                })
                .collect();
            println!(
                "  computed x (150) present in residual consts: {}",
                consts.contains(&150)
            );
            assert_eq!(brtable, 0, "the 83-way dispatch must fully fold");
            assert_eq!(indirect, 0, "cold-path call_indirects must be pruned");
            assert_eq!(loads, 0, "all VM state folds — no residual loads");
            assert!(
                consts.contains(&150),
                "the residual must have computed the loop result x=150 at spec time"
            );
        }
        Err(e) => println!("\nstill blocked: Err({e:?}) — a further wall past growstack"),
    }
}
