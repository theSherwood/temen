//! **Step 2 of the real-Lua Futamura harness: specialize.**
//!
//! Building on the capture (`lua_futamura_capture.rs`), this drives `luaV_execute` to its entry,
//! chases `ci → Proto → code`, then declares the `ci` struct and the bytecode array **constant**
//! (`SpecConfig::const_overlays`) and bakes `L`/`ci` — so the opcode fetch `*ci->u.l.savedpc` folds
//! to a compile-time constant and the 83-way dispatch collapses. This is the core of the projection.
//!
//! Slice A (this file): does the dispatch fold at all — i.e. does the specializer get *past*
//! `luaV_execute`'s entry with the bytecode constant, versus the immediate `Unsupported` it returns
//! on the un-specialized function? Later slices add the register-file rename region + ROI.
//!
//! Run: `cargo test -p svm-llvm --test lua_futamura_specialize -- --nocapture`

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

    // The register file: base = ci->func.p + 1, extent = maxstacksize TValues. Renaming it lifts the
    // registers into SSA — crucially the 1-byte tags, written as the constant LUA_VNUMINT by each
    // integer LOADI/ADD, so the per-opcode "is it an integer?" type checks fold and the metamethod /
    // error arms (the cold-path call_indirects) are pruned.
    let func = rd_u64(&w, ci + CI_FUNC as u64);
    let maxstack = w[proto as usize + PROTO_MAXSTACK] as u64;
    let base = func + STACKVALUE_SIZE;
    let reg_hi = base + maxstack * STACKVALUE_SIZE;

    // Overlays: the ci struct (so savedpc folds) + the bytecode array (so the opcode fetch folds).
    let ci_bytes = w[ci as usize..ci as usize + CI_SIZE].to_vec();
    let code_bytes = w[code_addr as usize..(code_addr + 4 * sizecode) as usize].to_vec();
    let cfg = SpecConfig {
        const_overlays: vec![(ci, ci_bytes), (code_addr, code_bytes)],
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
    match specialize_with_config(&m, luav, &args, &cfg) {
        Ok(r) => println!(
            "SPECIALIZED: Ok — residual {} blocks (interpreter luaV_execute had {})",
            r.funcs[0].blocks.len(),
            m.funcs[luav as usize].blocks.len()
        ),
        Err(e) => println!("SPECIALIZED: Err({e:?})"),
    }
}
