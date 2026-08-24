//! **End-to-end runtime differential for the `luaV_execute` projection.** `lua_futamura_vmstate.rs`
//! asserts the residual's *structure* (dispatch gone, `x=150` folded). This runs it: does the residual
//! reproduce the **exact memory effect** of the real `luaV_execute`?
//!
//! The eval harness runs the chunk with `lua_pcall(L, 0, 0, 0)` — 0 results, so a `return` value is
//! discarded and only `print`/`io.write` produce stdout; but those are capability calls *inside*
//! `luaV_execute`, which a folding projection can't cross. So stdout is the wrong oracle for an
//! integer loop. The right one is memory: from the captured entry state,
//!   • run the **real** `luaV_execute` to its return (`Inspector::step_out`) and snapshot the window;
//!   • run the **residual** from the same entry window;
//!   • assert the two windows are byte-identical across the VM-state arena.
//! Equality means the residual performed every store the interpreter did, with the same final values —
//! the projection is not just structurally clean but behaviorally faithful.

use temen_interp::{IrPc, Stop, StopReason, Value};
use temen_ir::Module;
use temen_peval::{specialize_with_config, SpecArg, SpecConfig};

const SCRIPT: &str = "local x = 0\nfor i = 1, 50 do x = x + 3 end\nreturn x\n";
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

// The arena window we diff: below L, up through the heap structs the prologue chases. All BSS arena
// (no data segments), so `run_capture`'s data-init leaves it to the seed + the run's own stores.
const DIFF_LO: usize = 0x104000;
const DIFF_HI: usize = 0x109000;

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

/// Attach, break at `luaV_execute` entry. Returns the inspector (still stopped at entry), the args
/// (sp, L, ci), and the captured entry window.
fn attach_at_entry(m: &Module, luav: u32) -> (temen_interp::Inspector, i64, i64, i64, Vec<u8>) {
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
            let (sp, l, ci) = (g(0), g(1), g(2));
            let w = insp
                .read_window(0, (win as usize).min(CAPTURE_LEN))
                .expect("window");
            (insp, sp, l, ci, w)
        }
        o => panic!("no break: {o:?}"),
    }
}

#[test]
fn residual_reproduces_luav_execute_memory_effect() {
    let m = lua_module();
    let luav = luav_execute(&m);

    // --- capture entry state + build the residual (same setup as lua_futamura_vmstate.rs) ---
    let (mut insp, sp, l, ci, w) = attach_at_entry(&m, luav);
    let (l, ci) = (l as u64, ci as u64);
    let stack_lo = rd_u64(&w, l + L_STACK);
    let stack_hi = rd_u64(&w, l + L_STACK_LAST);
    assert_eq!(rd_u64(&w, l + L_CI), ci, "L->ci sanity");
    let func = rd_u64(&w, ci + CI_FUNC);
    let closure = rd_u64(&w, func);
    let proto = rd_u64(&w, closure + LCLOSURE_P);
    let code = rd_u64(&w, proto + PROTO_CODE);
    let sizecode = rd_i32(&w, proto + PROTO_SIZECODE) as usize;
    let slice = |a: u64, n: usize| w[a as usize..a as usize + n].to_vec();
    let stack_len = (stack_hi - stack_lo) as usize;

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
        indirect_targets_cap: Some(16),
        ..SpecConfig::default()
    };
    let args = [
        SpecArg::ConstI64(sp),
        SpecArg::ConstI64(l as i64),
        SpecArg::ConstI64(ci as i64),
    ];
    let residual = specialize_with_config(&m, luav, &args, &cfg).expect("projects");

    // --- reference: run the real luaV_execute to its return, snapshot the window ---
    match insp.step_out() {
        Stop::Break { .. } => {} // paused in luaV_execute's caller (reason: Step)
        o => panic!("step_out did not pause in the caller: {o:?}"),
    }
    let ref_w = insp.read_window(0, DIFF_HI).expect("ref window");

    // --- residual: run it from the same entry window, snapshot the result ---
    let mut fuel = u64::MAX;
    let (r, res_w) = temen_interp::run_capture(&residual, 0, &[], &mut fuel, &w[..DIFF_HI]);
    assert!(r.is_ok(), "residual run trapped: {r:?}");

    // --- diff the VM-state arena ---
    let mut diffs = Vec::new();
    for a in DIFF_LO..DIFF_HI {
        if ref_w[a] != res_w[a] {
            diffs.push(a);
        }
    }
    if !diffs.is_empty() {
        let lo = *diffs.first().unwrap();
        let hi = *diffs.last().unwrap();
        eprintln!(
            "\n{} differing byte(s) in [{DIFF_LO:#x}, {DIFF_HI:#x}); first {:#x} last {:#x}",
            diffs.len(),
            lo,
            hi
        );
        for &a in diffs.iter().take(32) {
            eprintln!("  {a:#x}: ref={:#04x} res={:#04x}", ref_w[a], res_w[a]);
        }
    }
    // Non-vacuity: luaV_execute must actually have changed the arena (else "identical" is trivial).
    let changed: usize = (DIFF_LO..DIFF_HI).filter(|&a| ref_w[a] != w[a]).count();
    println!(
        "luaV_execute changed {changed} arena byte(s); residual vs interp diffs = {}",
        diffs.len()
    );
    assert!(
        changed > 0,
        "luaV_execute left the arena unchanged — the diff would be vacuously equal"
    );
    assert!(
        diffs.is_empty(),
        "residual's memory effect diverges from luaV_execute at {} byte(s) (first {:#x})",
        diffs.len(),
        diffs.first().copied().unwrap_or(0)
    );
}
