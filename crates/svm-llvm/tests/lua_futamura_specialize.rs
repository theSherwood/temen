//! **Step 2 of the real-Lua Futamura harness: specialize — and the wall it hits.**
//!
//! Building on the capture (`lua_futamura_capture.rs`), this drives `luaV_execute` to its entry,
//! chases `ci → Proto → code`, declares the `ci` struct + the bytecode array + the closure/proto
//! structs **constant** (`SpecConfig::const_overlays`), folds the debug-hook check by overlaying
//! `L->hookmask = 0`, bakes `L`/`ci`, and lifts the register file into a `rename` region. Everything
//! the projection's *prologue chase* (`cl`/`proto`/`k`, the opcode fetch `*ci->u.l.savedpc`, the
//! 83-way `indirectbr`) needs to fold is set up here.
//!
//! **Finding (this file is the record).** The dispatch fold is *feasible* (see
//! `lua_futamura_feasibility.rs`: no host calls in the closure, a folded opcode collapses the
//! `indirectbr`), but projecting the whole `luaV_execute` is **blocked before the loop even starts**
//! by the `OP_VARARGPREP` prologue: `luaT_adjustvarargs → luaD_checkstack/luaD_growstack` branches on
//! the mutable pointer fields `L->top`/`L->stack_last` behind a `setjmp` error arm, and also mutates
//! `ci` — none of which the const-overlay + single-contiguous-rename model can fold. The test asserts
//! that wall (a regression guard) and its body documents exactly what would move it. See the long
//! comment at the `assert!` for the full call chain and why each fold is unavailable.
//!
//! Run: `cargo test -p svm-llvm --test lua_futamura_specialize -- --nocapture`
//! (add `--features svm-peval/trace` to replay the breadcrumb that pinpointed the `SetJmp` site).

use svm_interp::{IrPc, Stop, StopReason, Value};
use svm_ir::Module;
use svm_peval::{specialize_with_config, SpecArg, SpecConfig};

const SCRIPT: &str = "local x = 0\nfor i = 1, 50 do x = x + 3 end\nreturn x\n";
const CAPTURE_LEN: usize = 8 << 20;

// Lua 5.4.7 offsets (validated in lua_futamura_capture.rs).
const CI_SAVEDPC: usize = 32;
const CI_SIZE: usize = 104; // overlay the whole CallInfo (stable at entry for a call-free loop)
const CI_FUNC: usize = 0;
const LCLOSURE_P: usize = 24;
const PROTO_MAXSTACK: usize = 12; // Proto.maxstacksize (lu_byte)
const PROTO_SIZECODE: usize = 24;
const PROTO_CODE: usize = 64;
const STACKVALUE_SIZE: u64 = 16; // sizeof(TValue) == sizeof(StackValue)

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

/// Capture (sp, L, ci, window) at luaV_execute entry.
fn capture(m: &Module, luav: u32) -> (i64, i64, i64, Vec<u8>) {
    let inst = svm_run::instantiate(m.clone()).expect("instantiate");
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
fn dispatch_folds_with_constant_bytecode() {
    let m = lua_module();
    let luav = luav_execute(&m);
    let (sp, l, ci, w) = capture(&m, luav);
    let ci = ci as u64;

    // Chase to the bytecode array.
    let func = rd_u64(&w, ci);
    let closure = rd_u64(&w, func);
    let proto = rd_u64(&w, closure + LCLOSURE_P as u64);
    let code_addr = rd_u64(&w, proto + PROTO_CODE as u64);
    let sizecode = rd_i32(&w, proto + PROTO_SIZECODE as u64) as u64;
    assert_eq!(
        rd_u64(&w, ci + CI_SAVEDPC as u64),
        code_addr,
        "savedpc != code"
    );
    // Sanity-check the lua_State layout: L->ci (offset 32) must be the ci we captured. This pins the
    // struct base so the hookmask offset below is trustworthy.
    let lu = l as u64;
    assert_eq!(rd_u64(&w, lu + 32), ci, "L->ci != captured ci");

    // The register file: base = ci->func.p + 1, extent = maxstacksize TValues. Renaming it lifts the
    // registers into SSA — crucially the 1-byte tags, written as the constant LUA_VNUMINT by each
    // integer LOADI/ADD, so the per-opcode "is it an integer?" type checks fold and the metamethod /
    // error arms (the cold-path call_indirects) are pruned.
    let func = rd_u64(&w, ci + CI_FUNC as u64);
    let maxstack = w[proto as usize + PROTO_MAXSTACK] as u64;
    let base = func + STACKVALUE_SIZE;
    let reg_hi = base + maxstack * STACKVALUE_SIZE;

    // Overlays: everything the prologue chases must be constant so `cl`/`proto`/`k` fold and the
    // dispatch collapses — the ci struct (savedpc), the bytecode array (opcode fetch), and the
    // closure slot + LClosure + Proto structs on the ci->func->closure->proto->code/k chase.
    let slice = |a: u64, n: usize| w[a as usize..a as usize + n].to_vec();
    // The debug-hook fold. `luaV_execute` opens with `trap = L->hookmask` and, in every `vmfetch`,
    // re-checks `if (l_unlikely(trap)) trap = luaG_traceexec(...)`. `luaG_traceexec` is the head of
    // the setjmp error path (→ `luaD_throw` → `luaD_rawrunprotected`), so unless `trap` folds to a
    // constant 0 the specializer must explore that arm and hits `SetJmp` → Unsupported. `hookmask`
    // is the LAST field of `lua_State`; validated at offset 192 here (L->ci@32 pins the base, and
    // L->nCcalls@176 = 0x20001 pins the tail). It reads 0 on a plain, un-hooked run — declaring
    // those 4 bytes constant makes the load fold, pruning the prologue trap block and every
    // per-instruction hook check. This is the "no debug hooks" specialization contract.
    const HOOKMASK_OFF: u64 = 192;
    assert_eq!(rd_i32(&w, lu + HOOKMASK_OFF), 0, "L->hookmask not 0 (a hook is set?)");
    let cfg = SpecConfig {
        const_overlays: vec![
            (ci, slice(ci, CI_SIZE)),
            (lu + HOOKMASK_OFF, vec![0u8; 4]),
            (code_addr, slice(code_addr, 4 * sizecode as usize)),
            (func, slice(func, 16)),      // the closure TValue (holds the LClosure ptr + tag)
            (closure, slice(closure, 48)), // LClosure header + p + upvals
            (proto, slice(proto, 128)),   // Proto (code/k/sizes/…)
        ],
        rename: Some((base, reg_hi)),
        rename_is_private: true,
        indirect_targets_cap: Some(16),
        ..SpecConfig::default()
    };
    println!("register file: base={base:#x} maxstack={maxstack} region=[{base:#x}, {reg_hi:#x})");

    // Baseline: un-specialized luaV_execute is Unsupported (dispatch stays dynamic, cold paths hit).
    let base = specialize_with_config(
        &m,
        luav,
        &[SpecArg::Dynamic, SpecArg::Dynamic, SpecArg::Dynamic],
        &SpecConfig {
            indirect_targets_cap: Some(16),
            ..Default::default()
        },
    );
    println!(
        "\nbaseline (all-dynamic): {:?}",
        base.as_ref().map(|r| r.funcs[0].blocks.len())
    );

    // With the bytecode constant + L/ci baked:
    let args = [
        SpecArg::ConstI64(sp),
        SpecArg::ConstI64(l),
        SpecArg::ConstI64(ci as i64),
    ];
    let specialized = specialize_with_config(&m, luav, &args, &cfg);
    match &specialized {
        Ok(r) => println!(
            "SPECIALIZED: Ok — residual {} blocks (interpreter luaV_execute had {})",
            r.funcs[0].blocks.len(),
            m.funcs[luav as usize].blocks.len()
        ),
        Err(e) => println!("SPECIALIZED: Err({e:?})"),
    }

    // THE WALL (characterized 2026-08-05, with the `svm-peval/trace` breadcrumb). Both the baseline
    // and the fully-overlaid attempt return `Unsupported` — and, decisively, at the **same** site:
    // an `Inst::SetJmp` inside `luaD_rawrunprotected`, reached on the **hot** path
    //
    //   luaV_execute → luaT_adjustvarargs → luaD_growstack → luaD_throw
    //                → luaE_resetthread → luaD_closeprotected → luaD_rawrunprotected → SetJmp
    //
    // `luaT_adjustvarargs` is the `OP_VARARGPREP` handler — the *first* opcode of every chunk
    // `luaL_loadbuffer` compiles (main chunks are vararg). It calls `luaD_checkstack`, which guards
    // `luaD_growstack` (whose stack-overflow arm `luaD_throw`s → setjmp) on `L->stack_last - L->top`.
    // The specializer refuses any spec-time-reachable `SetJmp` (a residual can't carry a setjmp), so
    // that arm must be *statically pruned* — which needs `L->top`/`L->stack_last` to fold. They can't:
    //   • `const_overlays` is unsound for them — `adjustvarargs` does `L->top.p++`, so a later load
    //     would read the stale overlay constant, not the incremented pointer; and
    //   • the single contiguous `rename` region (the one abstraction that folds a *mutable* cell) is
    //     already spent on the register file, and `L`'s struct is a disjoint address range.
    // Worse, `adjustvarargs` also mutates `ci` (`ci->func.p`, `ci->u.l.savedpc++`) — which is a const
    // overlay here — so even a second rename region over `L`'s stack fields would not suffice.
    //
    // Conclusion: the *dispatch fold itself* is feasible (feasibility test: no host calls, folding
    // bytecode collapses the `indirectbr`), but projecting the **whole** `luaV_execute` is blocked by
    // Lua's stack-management substrate in the VARARGPREP prologue, which mutates `ci`/`L->top`/the
    // stack and branches on mutable pointer fields behind `setjmp` error arms. Clearing it is a
    // specializer feature (rename the *entire* VM state — multiple regions incl. `ci` and `L`'s stack
    // fields as mutable-but-known cells — not a const overlay), not a harness tweak. This assertion is
    // the regression guard: when that feature lands and the fold gets past the prologue, flip it.
    assert!(
        specialized.is_err(),
        "specialize got past the VARARGPREP/growstack wall — the specializer gained mutable-VM-state \
         renaming; update this characterization and start measuring the residual ROI"
    );
}
