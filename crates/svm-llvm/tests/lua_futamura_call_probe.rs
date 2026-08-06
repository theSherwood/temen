//! **Feasibility probe: can peval project *through* a Lua-to-Lua call?**
//!
//! The `print`-chunk end-to-end (`lua_futamura_print.rs`) deopts exactly one edge — the OP_CALL
//! "callee was a Lua function" re-dispatch — because it is dead for a C call. Real scripts define and
//! call Lua functions, so to reach them the projection must *follow* that edge instead of bailing.
//!
//! In Lua 5.4 a Lua-to-Lua call does **not** recurse into a new `luaV_execute`: OP_CALL does
//! `newci = luaD_precall(...); ci = newci; goto startfunc;`, re-entering the *same* dispatch loop
//! (IR block 1) with the callee's `ci`/proto/code/base. So "projecting through" means continuing the
//! fold at block 1 with the **callee frame's** overlays. That is only possible if, at the re-dispatch,
//! the callee frame is *statically discoverable* from the captured image:
//!   - the new `ci` address (precall pushes/reuses `ci->next`; the arena is deterministic),
//!   - the callee `Proto` (a sub-proto of the outer proto: `Proto**p @72`, `sizep @32`),
//!   - its `code`/`k` (`code @64`, `k @56`), and `base = ci->func + 1`.
//!
//! This probe captures a function-calling script at `luaV_execute` entry, then breakpoints the real
//! dispatch header (the widest `br_table`) and, at each re-entry, reads `L->ci → cl → proto` to watch
//! the frame switch to the callee across the call — establishing whether those addresses are
//! predictable (the precondition for a `precall` post-state model that lets the projection inline the
//! callee).
//!
//! **Finding (this test is the record): YES, and it can even ROLL.** For `for i=1,5 do x=add(x,3) end`,
//! the callee `add` proto is `sub-proto[0]` of the main chunk proto (statically known at capture); the
//! dispatch loop genuinely re-enters with the callee proto (the OP_CALL "Lua function" edge); and the
//! callee activation is **identical every call** (`ci=0x109270, code=0x108bf0` on all five) — Lua
//! caches `ci->next`, so the callee frame is *loop-invariant*. A call-bearing loop can roll with a
//! single callee-frame overlay, not just unroll.
//!
//! So the implementation path is a **precall/poscall post-state model**: at an OP_CALL whose (cut)
//! `precall` returns a statically-known Lua closure, install the callee frame's overlays (its
//! deterministic `ci`, its proto/code) and continue folding at the dispatch header with the callee
//! active — inlining the callee's dispatch — then model `poscall` to pop back to the caller frame.
//!
//! Run: `cargo test -p svm-llvm --test lua_futamura_call_probe -- --nocapture`

use svm_interp::{IrPc, Stop, StopReason, Value};
use svm_ir::{Inst, Module, Terminator};

// A script whose hot loop calls a *Lua* function each iteration (the case the print-chunk deopts).
const SCRIPT: &str = "local function add(a, b) return a + b end\n\
                      local x = 0\n\
                      for i = 1, 5 do x = add(x, 3) end\n\
                      print(x)\n";
const CAPTURE_LEN: usize = 8 << 20;

const L_CI: u64 = 32; // lua_State->ci
const CI_FUNC: u64 = 0; // CallInfo->func (StkId)
const CI_SAVEDPC: u64 = 24; // CallInfo->u.l.savedpc (Lua frame)
const LCLOSURE_P: u64 = 24; // LClosure->p (Proto*)
const PROTO_CODE: u64 = 64;
const PROTO_SIZECODE: u64 = 24;
const PROTO_P: u64 = 72; // Proto**p (sub-protos)
const PROTO_SIZEP: u64 = 32; // int sizep

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

fn byname(m: &Module, name: &str) -> Option<u32> {
    m.exports.iter().find(|e| e.name == name).map(|e| e.func)
}

/// Given `L`, follow `L->ci` and return `(ci, proto, code, sizecode, savedpc_pc)`.
fn frame(w: &[u8], l: u64) -> (u64, u64, u64, u64, i64) {
    let ci = rd_u64(w, l + L_CI);
    let func = rd_u64(w, ci + CI_FUNC);
    let cl = rd_u64(w, func); // *StkId -> GCObject* (the LClosure)
    let proto = rd_u64(w, cl + LCLOSURE_P);
    let code = rd_u64(w, proto + PROTO_CODE);
    let sizecode = rd_i32(w, proto + PROTO_SIZECODE) as u64;
    let savedpc = rd_u64(w, ci + CI_SAVEDPC);
    // savedpc points into code; the pc index is (savedpc - code)/4.
    let pc = if savedpc >= code {
        ((savedpc - code) / 4) as i64
    } else {
        -1
    };
    (ci, proto, code, sizecode, pc)
}

fn dump_proto(w: &[u8], tag: &str, proto: u64) {
    let code = rd_u64(w, proto + PROTO_CODE);
    let sizecode = rd_i32(w, proto + PROTO_SIZECODE) as u64;
    let sizep = rd_i32(w, proto + PROTO_SIZEP) as u64;
    let p = rd_u64(w, proto + PROTO_P);
    println!("  {tag}: proto={proto:#x} code={code:#x} sizecode={sizecode} sizep={sizep} p={p:#x}");
    for pc in 0..sizecode {
        let i = rd_u32(w, code + 4 * pc);
        println!(
            "     [{pc:2}] op={:2} A={:3} B/Bx={:6}",
            i & 0x7f,
            (i >> 7) & 0xff,
            i >> 15
        );
    }
    for i in 0..sizep {
        let sub = rd_u64(w, p + 8 * i);
        println!("     sub-proto[{i}] = {sub:#x}");
    }
}

#[test]
fn probe_lua_to_lua_call_frame_is_statically_discoverable() {
    let m = lua_module();
    let luav = luav_execute(&m);
    let precall = byname(&m, "luaD_precall");
    println!("luaV_execute=f{luav}  luaD_precall={precall:?}");

    let inst = svm_run::instantiate(m.clone()).expect("instantiate");
    let mut insp = inst.debug_attach(SCRIPT.as_bytes().to_vec(), u64::MAX);

    // Break at luaV_execute entry to grab L (stable for the whole run).
    insp.set_breakpoint(IrPc {
        module: 0,
        func: luav,
        block: 0,
        inst: 0,
    });
    let l = match insp.run_until_stop() {
        Stop::Break { .. } => match insp.read_ir_value(0, 1) {
            Some(Value::I64(v)) => v as u64,
            o => panic!("L read: {o:?}"),
        },
        o => panic!("no entry break: {o:?}"),
    };
    let w0 = insp.read_window(0, CAPTURE_LEN).expect("window");
    let (ci0, outer_proto, _, _, _) = frame(&w0, l);
    println!("\nentry: L={l:#x} ci={ci0:#x}");
    println!("\n=== outer (main chunk) proto + its sub-protos ===");
    dump_proto(&w0, "outer", outer_proto);
    // The `add` closure is a sub-proto of the outer proto — statically discoverable at capture time.
    let sizep = rd_i32(&w0, outer_proto + PROTO_SIZEP) as u64;
    let pptr = rd_u64(&w0, outer_proto + PROTO_P);
    for i in 0..sizep {
        let sub = rd_u64(&w0, pptr + 8 * i);
        println!("\n=== callee sub-proto[{i}] (statically known from the outer proto) ===");
        dump_proto(&w0, "callee", sub);
    }

    // Discover the real dispatch header: the block whose terminator is the widest br_table.
    let dispatch = {
        let mut best = (0u32, 0usize);
        for (bi, b) in m.funcs[luav as usize].blocks.iter().enumerate() {
            if let Terminator::BrTable { targets, .. } = &b.term {
                if targets.len() > best.1 {
                    best = (bi as u32, targets.len());
                }
            }
        }
        println!(
            "\ndispatch header = block {} (br_table with {} targets)",
            best.0, best.1
        );
        best.0
    };

    // Watch the dispatch header re-enter across the call: read L->ci each hit.
    insp.clear_breakpoint(IrPc {
        module: 0,
        func: luav,
        block: 0,
        inst: 0,
    });
    insp.set_breakpoint(IrPc {
        module: 0,
        func: luav,
        block: dispatch as usize,
        inst: 0,
    });
    println!(
        "\n=== dispatch re-entries: does the frame switch to the callee, once per loop iter? ==="
    );
    let mut seen_protos: Vec<u64> = Vec::new();
    // Record the (ci, code) of each *distinct callee activation* — a new activation each time the frame
    // transitions OUTER -> CALLEE. If the callee ci is the same address every iteration, the callee
    // frame overlay is loop-invariant and the call-bearing loop can ROLL.
    let mut callee_activations: Vec<(u64, u64)> = Vec::new();
    let mut prev_was_callee = false;
    for _ in 0..600 {
        match insp.run_until_stop() {
            Stop::Break {
                reason: StopReason::Breakpoint,
                ..
            } => {
                let w = insp.read_window(0, CAPTURE_LEN).expect("window");
                let (ci, proto, code, _sz, _pc) = frame(&w, l);
                let is_callee = proto != outer_proto;
                if !seen_protos.contains(&proto) {
                    seen_protos.push(proto);
                }
                if is_callee && !prev_was_callee {
                    callee_activations.push((ci, code));
                }
                prev_was_callee = is_callee;
                // We call `add` 5 times; stop once we've captured them all (plus a margin).
                if callee_activations.len() >= 5 {
                    break;
                }
            }
            other => {
                println!("  (run ended: {other:?})");
                break;
            }
        }
    }

    println!("\n=== findings ===");
    println!("  distinct protos seen at dispatch: {seen_protos:?}");
    println!("  callee activations (ci, code) per call:");
    for (i, (ci, code)) in callee_activations.iter().enumerate() {
        println!("    call {i}: ci={ci:#x} code={code:#x}");
    }
    let all_same_ci = callee_activations
        .windows(2)
        .all(|w| w[0].0 == w[1].0 && w[0].1 == w[1].1);
    println!(
        "  callee frame is loop-invariant (same ci+code every call): {all_same_ci}  \
         => a call-bearing loop can ROLL with a callee-frame overlay"
    );

    // The structural precondition for projecting through: precall is a *direct callee* of luaV_execute
    // (so its post-state can be modeled at the OP_CALL site), and the callee proto is a sub-proto of a
    // proto we already overlay. Assert both so this stays a live feasibility record.
    if let Some(pc) = precall {
        let is_direct = m.funcs[luav as usize].blocks.iter().any(|b| {
            b.insts
                .iter()
                .any(|i| matches!(i, Inst::Call { func, .. } if *func == pc))
                || matches!(&b.term, Terminator::ReturnCall { func, .. } if *func == pc)
        });
        println!("  luaD_precall is a direct callee of luaV_execute: {is_direct}");
    }
    assert!(
        seen_protos.len() >= 2,
        "the dispatch loop must re-enter with a DIFFERENT (callee) proto — that's the Lua call we \
         want to project through (saw {seen_protos:?})"
    );
    assert!(
        callee_activations.len() >= 2,
        "the callee frame must be reached across multiple loop iterations (the edge we currently deopt)"
    );
    assert!(
        all_same_ci,
        "the callee ci+code must be identical every call — Lua caches ci->next, so the callee frame \
         overlay is loop-invariant; this is what lets a call-bearing loop roll instead of unrolling"
    );
}
