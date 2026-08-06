//! **Toward a real end-to-end: a whole chunk with a `print` — and the first engine wall it hits.**
//!
//! `lua_futamura_e2e.rs` runs a residual and diffs its memory effect, but notes stdout is uncrossable
//! for a folding projection (`print` is a capability call *inside* `luaV_execute`). The cut machinery
//! changes that: cut the `print`/table-lookup/allocator/growstack call path and the residual *calls
//! the real print*, so stdout becomes a valid oracle again. This drives a whole-chunk projection from
//! `luaV_execute`'s real entry — the first step to running a real script through a residual.
//!
//! **Progress + the current wall (this file is the record).** With the C-call machinery split into
//! read-state cuts (`luaH_get*`/`luaV_finishget`/GC — read the heap, don't rewrite registers) and
//! touch-state cuts (the `precall`/`poscall` path — mutate the stack), the projection clears the loop,
//! the `_ENV["print"]` lookup, the `OP_GETTABUP` value move, **and the `OP_CALL` to `print`** — the
//! first two of which needed the **narrow dynamic rename cell** engine feature this PR adds: the
//! interpreter moves a looked-up value by storing its **1-byte `TValue` tag** (dynamic, since `print`
//! came from a cut lookup) into a renamed register, and the touch-state cut of `luaD_precall` spills/
//! reloads those tag bytes. Both were previously refused (the rename model only carried 4/8-byte
//! dynamic cells); now a dynamic cell is renamed at a sub-natural 1/2-byte width too.
//!
//! The remaining blocker is a different class — `SpecError::Budget`: past the `CALL`, the projection
//! **diverges** (the dynamic callee from the cut lookup makes `OP_CALL`'s callee-type checks explore
//! unboundedly). That is a convergence problem, not a capability gap, and the next investigation
//! (bound the callee type, or deopt the non-fast call kinds). This test asserts the *current* wall as
//! a regression guard (the pattern `lua_futamura_specialize` uses for the VARARGPREP wall); update it
//! when the divergence is bounded and the chunk projects through to `RETURN` + embedding + the diff.
//!
//! Run: `cargo test -p svm-llvm --test lua_futamura_print -- --nocapture`
//! (`--features svm-peval/trace` shows the trace up to the divergence).

use svm_interp::{IrPc, Stop, StopReason, Value};
use svm_ir::{Inst, Module, Terminator};
use svm_peval::{specialize_with_config, SpecArg, SpecConfig};

const SCRIPT: &str = "local x = 0\nfor i = 1, 50 do x = x + 3 end\nprint(x)\n";
const CAPTURE_LEN: usize = 8 << 20;

const L_STACK_LAST: u64 = 40;
const L_STACK: u64 = 48;
const L_CI: u64 = 32;
const LUA_STATE_SIZE: usize = 200;
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
    svm_llvm::translate_ll_path(&path)
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
fn rd_u32(w: &[u8], a: u64) -> u32 {
    u32::from_le_bytes(w[a as usize..a as usize + 4].try_into().unwrap())
}
fn export_name(m: &Module, f: u32) -> String {
    m.exports
        .iter()
        .find(|e| e.func == f)
        .map(|e| e.name.clone())
        .unwrap_or_else(|| format!("f{f}"))
}

/// The transitive direct-call closure of `f`.
fn closure(m: &Module, f: u32) -> std::collections::BTreeSet<u32> {
    let mut seen = std::collections::BTreeSet::new();
    let mut stack = vec![f];
    while let Some(c) = stack.pop() {
        if !seen.insert(c) {
            continue;
        }
        for b in &m.funcs[c as usize].blocks {
            for i in &b.insts {
                if let Inst::Call { func, .. } = i {
                    stack.push(*func);
                }
            }
            if let Terminator::ReturnCall { func, .. } = &b.term {
                stack.push(*func);
            }
        }
    }
    seen
}

#[test]
fn dump_print_chunk() {
    let m = lua_module();
    let luav = luav_execute(&m);
    let inst = svm_run::instantiate(m.clone()).expect("instantiate");
    let mut insp = inst.debug_attach(SCRIPT.as_bytes().to_vec(), u64::MAX);
    insp.set_breakpoint(IrPc {
        module: 0,
        func: luav,
        block: 0,
        inst: 0,
    });
    let ci = match insp.run_until_stop() {
        Stop::Break { .. } => match insp.read_ir_value(0, 2) {
            Some(Value::I64(v)) => v as u64,
            o => panic!("{o:?}"),
        },
        o => panic!("no break: {o:?}"),
    };
    let w = insp.read_window(0, CAPTURE_LEN).expect("window");
    let func = rd_u64(&w, ci + CI_FUNC);
    let closure_ptr = rd_u64(&w, func);
    let proto = rd_u64(&w, closure_ptr + LCLOSURE_P);
    let code = rd_u64(&w, proto + PROTO_CODE);
    let sizecode = rd_i32(&w, proto + PROTO_SIZECODE) as u64;
    println!("\n=== print-chunk bytecode ({sizecode} instrs) ===");
    for pc in 0..sizecode {
        let i = rd_u32(&w, code + 4 * pc);
        println!(
            "  [{pc:2}] op={:2} A={:3} B/Bx={:6}",
            i & 0x7f,
            (i >> 7) & 0xff,
            i >> 15
        );
    }

    // The C-call machinery + allocator/error/growstack in the closure — the cut candidates for a
    // whole-chunk projection with a `print`.
    let cl = closure(&m, luav);
    println!("\n=== call/alloc/error machinery in luaV_execute's closure ===");
    let interesting = [
        "luaD_precall",
        "precallC",
        "luaD_call",
        "luaD_poscall",
        "luaH_get",
        "luaH_getstr",
        "luaV_finishget",
        "luaT_gettm",
        "luaD_growstack",
        "luaD_reallocstack",
        "luaC_step",
        "luaC_newobj",
        "luaG_runerror",
        "luaG_typeerror",
        "luaB_print",
        "luaL_checkstack",
    ];
    for &f in &cl {
        let n = export_name(&m, f);
        if interesting.iter().any(|k| n.contains(k)) {
            println!("  f{f}: {n}");
        }
    }
    // Direct callees of luaV_execute (the cut points must be direct callees).
    let mut direct: Vec<u32> = Vec::new();
    for b in &m.funcs[luav as usize].blocks {
        for i in &b.insts {
            if let Inst::Call { func, .. } = i {
                if !direct.contains(func) {
                    direct.push(*func);
                }
            }
        }
        if let Terminator::ReturnCall { func, .. } = &b.term {
            if !direct.contains(func) {
                direct.push(*func);
            }
        }
    }
    println!("\n=== luaV_execute direct callees ({}) ===", direct.len());
    for f in &direct {
        println!("  f{f}: {}", export_name(&m, *f));
    }
}

fn byname(m: &Module, name: &str) -> Option<u32> {
    m.exports.iter().find(|e| e.name == name).map(|e| e.func)
}

/// Capture (sp, L, ci, window) at luaV_execute entry.
fn capture(m: &Module, luav: u32) -> (i64, u64, u64, Vec<u8>) {
    let inst = svm_run::instantiate(m.clone()).expect("instantiate");
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
            let w = insp.read_window(0, CAPTURE_LEN).expect("window");
            (g(0), g(1) as u64, g(2) as u64, w)
        }
        o => panic!("no break: {o:?}"),
    }
}

#[test]
fn project_whole_print_chunk_from_entry() {
    let m = lua_module();
    let luav = luav_execute(&m);
    let (sp, l, ci, w) = capture(&m, luav);
    let stack_lo = rd_u64(&w, l + L_STACK);
    let stack_hi = rd_u64(&w, l + L_STACK_LAST);
    assert_eq!(rd_u64(&w, l + L_CI), ci, "L->ci");
    let func = rd_u64(&w, ci + CI_FUNC);
    let cl = rd_u64(&w, func);
    let proto = rd_u64(&w, cl + LCLOSURE_P);
    let code = rd_u64(&w, proto + PROTO_CODE);
    let sizecode = rd_i32(&w, proto + PROTO_SIZECODE) as usize;
    let slice = |a: u64, n: usize| w[a as usize..a as usize + n].to_vec();
    let stack_len = (stack_hi - stack_lo) as usize;

    // Cut the call/alloc/error machinery a whole chunk with a `print` reaches: the C-call path
    // (precall → the C function via an indirect call), the table lookup for `_ENV["print"]`, the
    // allocator/GC, and stack growth. These read/mutate the VM state, so state-touching cuts +
    // carry_whole_module (their closures dispatch through the function table).
    // Read-only w.r.t. the register file (they read the heap / scan the stack, but don't rewrite the
    // renamed registers): spill-but-don't-reload, so the folded tags stay folded.
    let read_names = [
        "luaH_get",
        "luaH_getshortstr",
        "luaH_getstr",
        "luaH_getint",
        "luaV_finishget",
        "luaC_step",
        "luaC_newobj",
        "luaH_new",
        "luaH_resize",
    ];
    // Mutate the register file (push args / write results): spill + reload.
    let touch_names = [
        "luaD_precall",
        "precallC",
        "luaD_call",
        "luaD_callnoyield",
        "luaD_poscall",
        "luaD_growstack",
        "luaD_reallocstack",
    ];
    let read: Vec<u32> = read_names.iter().filter_map(|n| byname(&m, n)).collect();
    let touch: Vec<u32> = touch_names.iter().filter_map(|n| byname(&m, n)).collect();

    let cfg = SpecConfig {
        const_overlays: vec![
            (l, slice(l, LUA_STATE_SIZE)),
            (stack_lo, slice(stack_lo, stack_len)),
            (ci, slice(ci, CI_SIZE)),
            (code, slice(code, 4 * sizecode)),
            (cl, slice(cl, 48)),
            (proto, slice(proto, 128)),
        ],
        rename: Some((l, l + LUA_STATE_SIZE as u64)),
        rename_extra: vec![(stack_lo, stack_hi), (ci, ci + CI_SIZE as u64)],
        rename_is_private: true,
        rename_seed_from_image: true,
        cut_calls_read_state: read,
        cut_calls_touch_state: touch,
        carry_whole_module: true,
        indirect_targets_cap: Some(16),
        ..SpecConfig::default()
    };
    let args = [
        SpecArg::ConstI64(sp),
        SpecArg::ConstI64(l as i64),
        SpecArg::ConstI64(ci as i64),
    ];
    let r = specialize_with_config(&m, luav, &args, &cfg);
    match &r {
        Ok(res) => {
            let f = &res.funcs[res.funcs.len() - 1];
            let brt = f
                .blocks
                .iter()
                .filter(|b| matches!(b.term, Terminator::BrTable { .. }))
                .count();
            let calls: usize = f
                .blocks
                .iter()
                .flat_map(|b| &b.insts)
                .filter(|i| matches!(i, Inst::Call { .. }))
                .count();
            println!(
                "\nPROJECTED whole print-chunk: entry {} blocks, br_table={brt}, calls={calls}",
                f.blocks.len()
            );
        }
        Err(e) => println!("\nblocked at the {e:?} wall (see the file header)"),
    }

    // THE WALL (see the file header). Narrow dynamic rename cells (this PR) cleared the GETTABUP value
    // move and the touch-state spill/reload of the CALL, so the projection now reaches *past* the CALL
    // and stops on `Budget` — the dynamic callee makes OP_CALL's callee-type dispatch diverge.
    // Regression guard: when that divergence is bounded, the chunk projects to RETURN and this flips.
    assert!(
        r.is_err(),
        "whole print-chunk projected past the callee-type divergence — the projection now converges; \
         update this characterization and push on to RETURN + embedding + the stdout diff"
    );
}
