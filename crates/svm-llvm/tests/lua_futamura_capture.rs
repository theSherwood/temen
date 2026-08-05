//! **Step 1 of the real-Lua Futamura harness: capture.**
//!
//! To partial-evaluate `luaV_execute` against a fixed program we must first hand the specializer the
//! program *as constant bytes at their concrete window addresses*. Lua compiles the script to
//! bytecode on a guest `static char arena[]` (a bump allocator — deterministic addresses), so we
//! capture it by driving the real interpreter to `luaV_execute`'s entry and reading the window there.
//!
//! This test drives the lua_eval harness (script on stdin) on the `svm_interp::Inspector` debug
//! engine, breakpoints at `luaV_execute`'s first op, and reads the two pointer args `L` and `ci` plus
//! the window. It asserts the capture is **deterministic** across two runs of the same script — the
//! property that lets us treat the captured bytecode as a compile-time constant. (Later steps chase
//! `ci → func → LClosure → Proto → code/k`, declare those regions constant via
//! `SpecConfig::const_overlays`, and specialize.)
//!
//! Run: `cargo test -p svm-llvm --test lua_futamura_capture -- --nocapture`

use svm_interp::{IrPc, Stop, StopReason, Value};
use svm_ir::Module;

const SCRIPT: &str = "local x = 0\nfor i = 1, 50 do x = x + 3 end\nreturn x\n";

fn lua_module() -> Module {
    let path = format!(
        "{}/tests/fixtures/lua/lua_eval.ll",
        env!("CARGO_MANIFEST_DIR")
    );
    svm_llvm::translate_ll_path(&path)
        .expect("translate lua_eval")
        .module
}

fn luav_execute(m: &Module) -> u32 {
    m.exports
        .iter()
        .find(|e| e.name == "luaV_execute")
        .expect("luaV_execute export")
        .func
}

/// The window prefix we capture — enough to cover the arena's early allocations (the compiled
/// bytecode sits ~1 MiB in for a tiny script). The full window is the 48 MiB arena; a few MiB of the
/// used prefix is plenty and keeps the double-run comparison cheap.
const CAPTURE_LEN: usize = 8 << 20;

/// Drive the harness (script on stdin) to `luaV_execute`'s entry; return `(L, ci)` and a window dump.
fn capture(m: &Module, luav: u32) -> (i64, i64, u64, Vec<u8>) {
    let inst = svm_run::instantiate(m.clone()).expect("instantiate lua_eval");
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
            pc,
        } => {
            assert_eq!(pc.func, luav, "stopped at the wrong function");
            // Block-0 SSA values at entry are the params: 0 = sp, 1 = L, 2 = ci.
            let l = match insp.read_ir_value(0, 1) {
                Some(Value::I64(v)) => v,
                o => panic!("L not an i64: {o:?}"),
            };
            let ci = match insp.read_ir_value(0, 2) {
                Some(Value::I64(v)) => v,
                o => panic!("ci not an i64: {o:?}"),
            };
            let dump = insp
                .read_window(0, (win as usize).min(CAPTURE_LEN))
                .expect("read window");
            (l, ci, win, dump)
        }
        other => panic!("did not reach luaV_execute entry: {other:?}"),
    }
}

#[test]
fn capture_at_luav_execute_is_deterministic() {
    let m = lua_module();
    let luav = luav_execute(&m);

    let (l1, ci1, win, w1) = capture(&m, luav);
    let (l2, ci2, _, w2) = capture(&m, luav);

    println!("\ncapture at luaV_execute entry:");
    println!("  window: {win} bytes total, captured {} bytes", w1.len());
    println!("  run 1: L = {l1:#x}  ci = {ci1:#x}");
    println!("  run 2: L = {l2:#x}  ci = {ci2:#x}");

    // Both pointers must be non-null, inside the window, and within the captured prefix (so a later
    // step can chase them through the dump).
    assert!(l1 > 0 && (l1 as u64) < win, "L out of window: {l1:#x}");
    assert!(ci1 > 0 && (ci1 as u64) < win, "ci out of window: {ci1:#x}");
    assert!(
        (ci1 as usize) < w1.len(),
        "ci past the captured prefix — raise CAPTURE_LEN"
    );

    // The enabling property: the static bump allocator makes the addresses (and the compiled
    // bytecode they lead to) reproduce exactly, so we can treat the captured bytes as constant.
    assert_eq!(l1, l2, "L address not deterministic across runs");
    assert_eq!(ci1, ci2, "ci address not deterministic across runs");
    assert_eq!(w1, w2, "captured window not deterministic across runs");
}

// ---- Lua 5.4.7 struct offsets (64-bit), for the ci -> Proto -> code/k chase. ----
// Derived from lobject.h / lstate.h and cross-validated below (savedpc must equal Proto.code at
// entry, and the located bytecode must decode to valid opcodes) — so a wrong offset fails loudly.
const CI_FUNC: usize = 0; //     CallInfo.func   (StkId -> StackValue* holding the closure TValue)
const CI_SAVEDPC: usize = 32; // CallInfo.u.l.savedpc (bytecode cursor; == Proto.code at entry)
const TVALUE_GC: usize = 0; //   TValue.value_.gc (the LClosure* for a Lua closure)
const LCLOSURE_P: usize = 24; // LClosure.p      (Proto*)
const PROTO_SIZEK: usize = 20; //   Proto.sizek   (int)
const PROTO_SIZECODE: usize = 24; //Proto.sizecode(int)
const PROTO_K: usize = 56; //    Proto.k         (TValue*)
const PROTO_CODE: usize = 64; // Proto.code      (Instruction* — uint32 array)

fn rd_u64(w: &[u8], addr: u64) -> u64 {
    u64::from_le_bytes(w[addr as usize..addr as usize + 8].try_into().unwrap())
}
fn rd_i32(w: &[u8], addr: u64) -> i32 {
    i32::from_le_bytes(w[addr as usize..addr as usize + 4].try_into().unwrap())
}

/// The located program: the bytecode array address + words, and the constants array address + size.
struct Located {
    code_addr: u64,
    code: Vec<u32>,
    k_addr: u64,
    sizek: i32,
}

/// Chase `ci -> func -> LClosure -> Proto -> {code,k}` through the captured window `w`.
fn chase(w: &[u8], ci: u64) -> Located {
    let func = rd_u64(w, ci + CI_FUNC as u64); // StackValue* holding the closure
    let closure = rd_u64(w, func + TVALUE_GC as u64); // LClosure*
    let proto = rd_u64(w, closure + LCLOSURE_P as u64); // Proto*
    let code_addr = rd_u64(w, proto + PROTO_CODE as u64);
    let k_addr = rd_u64(w, proto + PROTO_K as u64);
    let sizecode = rd_i32(w, proto + PROTO_SIZECODE as u64);
    let sizek = rd_i32(w, proto + PROTO_SIZECODE as u64 - (PROTO_SIZECODE - PROTO_SIZEK) as u64);
    let savedpc = rd_u64(w, ci + CI_SAVEDPC as u64);

    // Cross-check: at entry the cursor points at the code start, validating both offsets at once.
    assert_eq!(savedpc, code_addr, "savedpc != Proto.code — a struct offset is wrong");
    assert!((0..4096).contains(&sizecode), "implausible sizecode {sizecode} — offsets wrong");
    assert!((code_addr as usize) + 4 * sizecode as usize <= w.len(), "code past capture");

    let code: Vec<u32> = (0..sizecode as u64)
        .map(|i| rd_i32(w, code_addr + 4 * i) as u32)
        .collect();
    Located { code_addr, code, k_addr, sizek }
}

#[test]
fn chase_to_bytecode_and_validate() {
    let m = lua_module();
    let luav = luav_execute(&m);
    let (_l, ci, _win, w) = capture(&m, luav);
    let loc = chase(&w, ci as u64);

    println!("\nlocated program via ci={ci:#x}:");
    println!("  code @ {:#x}: {} instructions", loc.code_addr, loc.code.len());
    println!("  k    @ {:#x}: {} constants", loc.k_addr, loc.sizek);

    // Every instruction's opcode (low 7 bits) must be a valid Lua 5.4 opcode (0..82) — a strong
    // signal we located the real bytecode and not garbage.
    for (i, &word) in loc.code.iter().enumerate() {
        let op = word & 0x7f;
        assert!(op <= 82, "instruction {i} opcode {op} out of range — not real bytecode");
    }
    // The first opcode of a chunk with locals is VARARGPREP (opcode 0 in 5.4).
    println!("  first opcodes: {:?}", loc.code.iter().take(6).map(|w| w & 0x7f).collect::<Vec<_>>());
    assert!(loc.code.len() >= 6, "expected a non-trivial loop program");
}
