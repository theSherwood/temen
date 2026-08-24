//! **A real Lua script, end to end, through a peval residual — stdout byte-identical to the interpreter.**
//!
//! This is the depth-first milestone: a whole real chunk (`local x=0; for i=1,50 do x=x+3 end; print(x)`)
//! is captured at `luaV_execute`'s real entry, projected into a single residual, embedded back in place
//! of `luaV_execute`, and run under `run_powerbox` — and it prints `150\n`, exactly what the baseline
//! interpreter prints. No memory-effect diff proxy: the oracle is real stdout.
//!
//! What made the whole chunk project and run:
//!   - **Cut the un-projectable runtime**, split by how it touches the register file. Read-state cuts
//!     (`luaH_get*`/`luaV_finishget`/GC — read the heap, scan the stack, don't rewrite registers):
//!     spill-but-don't-reload, so the folded `TValue` tags stay folded. Touch-state cuts (the
//!     `precall`/`poscall`/`growstack` path — push args, write results): spill **and** reload.
//!   - **Narrow dynamic rename cells** (landed earlier): the `_ENV["print"]` value move stores a dynamic
//!     1-byte `TValue` tag into a renamed register; the rename model carries dynamic cells at 1/2-byte
//!     widths, i32-canonical.
//!   - **Edge-level deopt** for the `OP_CALL` callee-type divergence: an `OP_CALL` handler calls
//!     `luaD_precall` (cut → dynamic result) and branches on "was the callee a Lua function?". The
//!     Lua-function edge re-enters dispatch (block 1) with a dynamic new frame and would diverge — but
//!     it is **dead** for a C-function call (`print`). `deopt_edges` deopts exactly that one
//!     `precall-block → dispatch` edge to the carried baseline `luaV_execute` (which resumes from
//!     savedpc), leaving the shared dispatch block projected for every live opcode.
//!   - **`carry_whole_module` + `carry_keep_imports`**: carry the runtime at original indices (so
//!     indirect calls resolve) and keep the I/O layer so `print`'s `write` capability works when the
//!     residual is embedded and re-attached to the imports/types manifest.
//!
//! `project_whole_print_chunk_from_entry` is the end-to-end: dispatch folds across the *entire* chunk
//! (`br_table == 0`), the only surviving calls are the opaque cut call-outs (the table lookup, the
//! allocator/GC, and the C-call path that actually runs `print`), and the embedded residual's stdout
//! equals the baseline's. `dump_print_chunk` prints the chunk bytecode + the cut/deopt machinery for
//! reference. The deterministic arena allocator (see the `lua_eval` fixture) reproduces the captured
//! addresses on a fresh run, so the residual's baked constants hold under re-execution.
//!
//! Run: `cargo test -p temen-llvm --test lua_futamura_print -- --nocapture`

use temen_interp::{IrPc, Stop, StopReason, Value};
use temen_ir::{Block, Func, Inst, Module, Terminator, ValType};
use temen_peval::{specialize_with_config, SpecArg, SpecConfig};

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
    let inst = temen_run::instantiate(m.clone()).expect("instantiate");
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
    let inst = temen_run::instantiate(m.clone()).expect("instantiate");
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

    // The divergent edge: an OP_CALL/OP_TAILCALL handler calls `luaD_precall` (cut, so its result is
    // dynamic) and branches on "was the callee a Lua function?"; the Lua-function edge re-enters the
    // dispatch loop (block 1) with a dynamic new frame and diverges. For a C-function call (`print`)
    // that edge is dead. Mark every `precall`-block → dispatch(block 1) edge a cold deopt edge.
    let precall = byname(&m, "luaD_precall").expect("luaD_precall");
    let dispatch = 1u32; // luaV_execute's main br_table dispatch (the loop header)
    let mut deopt_edges: Vec<(u32, u32, u32)> = Vec::new();
    for (bi, b) in m.funcs[luav as usize].blocks.iter().enumerate() {
        let calls_precall = b
            .insts
            .iter()
            .any(|i| matches!(i, Inst::Call { func, .. } if *func == precall));
        if !calls_precall {
            continue;
        }
        if let Terminator::BrIf {
            then_blk, else_blk, ..
        } = b.term
        {
            if then_blk == dispatch {
                deopt_edges.push((luav, bi as u32, then_blk));
            }
            if else_blk == dispatch {
                deopt_edges.push((luav, bi as u32, else_blk));
            }
        }
    }
    println!("deopt edges (precall → dispatch): {deopt_edges:?}");

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
        deopt_edges,
        deopt_handler: Some(luav), // the carried baseline luaV_execute resumes from savedpc
        carry_whole_module: true,
        carry_keep_imports: true, // keep the I/O layer (read/write) so print works when embedded
        indirect_targets_cap: Some(16),
        ..SpecConfig::default()
    };
    let args = [
        SpecArg::ConstI64(sp),
        SpecArg::ConstI64(l as i64),
        SpecArg::ConstI64(ci as i64),
    ];
    let mut residual =
        specialize_with_config(&m, luav, &args, &cfg).expect("whole print-chunk projects");
    let entry = (residual.funcs.len() - 1) as u32; // carry_whole_module appends the specialized entry
    let f = &residual.funcs[entry as usize];
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
    // The dispatch folded across the *entire* chunk (loop + GETTABUP + CALL + RETURN); the residual's
    // only calls are the opaque cut call-outs (the table lookup, the allocator/GC, and the C-call path
    // that actually runs `print`). The Lua-function call edge deopts to the baseline (dead for `print`).
    assert_eq!(brt, 0, "the whole-chunk dispatch folded");
    assert!(calls > 0, "print/lookup survive as opaque call-outs");

    // ---- End to end: embed the residual in place of luaV_execute and diff stdout ----
    // carry_whole_module kept the runtime at its original indices and (carry_keep_imports) kept the I/O
    // layer, but left the imports manifest empty; re-attach it so read/write resolve under the host.
    residual.imports = m.imports.clone();
    residual.types = m.types.clone(); // the import sigs reference the type section
                                      // Replace luaV_execute (index `luav`) with a same-signature wrapper that calls the specialized
                                      // entry. The main chunk is the only luaV_execute activation (print is a C call), so every dispatch
                                      // goes through the folded residual; the deterministic arena reproduces the captured addresses on a
                                      // fresh run, so the residual's baked constants hold.
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
    // (`run_powerbox` resolves the import manifest to capability calls and verifies internally, the
    // same path the original module takes — a raw `verify_module` on an unresolved-import module is not
    // the right gate here.)
    temen_verify::verify_module(&residual).expect("embedded residual verifies");

    let baseline = temen_run::run_powerbox(&m, SCRIPT.as_bytes()).expect("baseline run");
    let embedded =
        temen_run::run_powerbox(&residual, SCRIPT.as_bytes()).expect("embedded residual run");
    println!(
        "baseline stdout={:?}  embedded stdout={:?}",
        String::from_utf8_lossy(&baseline.stdout),
        String::from_utf8_lossy(&embedded.stdout)
    );
    assert_eq!(baseline.stdout, b"150\n", "baseline prints 150");
    assert_eq!(
        embedded.stdout, baseline.stdout,
        "the embedded residual must print byte-identically to the interpreter"
    );
    println!(
        "  ✓ real Lua script ran through the residual — stdout byte-identical to the interpreter"
    );
}
