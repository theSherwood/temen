//! **Auto-rolled residual (slice 1): discover the loop safepoint + carried cells at runtime.**
//!
//! `lua_futamura_rolled.rs` builds a rolled residual by HAND — it hardcodes the FORLOOP counter cell
//! (R[3]) and the loop-carried registers (x=R[1], i=R[2]) for its one script. This drives the same
//! result with ZERO hardcoded register offsets: it discovers the loop and its carried cells purely by
//! observing the running VM, so the recipe generalizes to any single numeric-for-loop chunk.
//!
//! Discovery (validated by the probe in this file's history): `ci->savedpc` is synced *lazily* by the
//! interpreter, so it is NOT a per-bytecode program counter — but the **loop body is the savedpc value
//! that persists across the longest run of consecutive dispatch hits**, and the first hit of that run
//! (the Δ≠0 transition into it) is a clean resume safepoint. The loop-carried cells are exactly the
//! frame registers whose value varies across those hits. Both are found by observation, no hardcoding.
//!
//! Run: `cargo test -p svm-llvm --test lua_futamura_auto_rolled -- --nocapture`

mod peval_capture;

use peval_capture::{discover, dispatch_block, Located, TargetDesc};
use svm_interp::{IrPc, Stop, StopReason, Value};
use svm_ir::{Block, Func, Inst, LoadOp, Module, Terminator, ValType};
use svm_peval::{specialize_with_config, SpecArg, SpecConfig};

const SCRIPT: &str = "local x = 0\nfor i = 1, 50 do x = x + 3 end\nreturn x\n";
const CAPTURE_LEN: usize = 8 << 20;

// Lua 5.4.7 offsets (same as lua_futamura_rolled.rs).
const L_STACK_LAST: u64 = 40;
const L_STACK: u64 = 48;
const L_CI: u64 = 32;
const LUA_STATE_SIZE: usize = 200;
const CI_SIZE: usize = 104;
const CI_FUNC: u64 = 0;
const CI_SAVEDPC: u64 = 32; // ci->u.l.savedpc
const LCLOSURE_P: u64 = 24;
const PROTO_SIZECODE: u64 = 24;
const PROTO_CODE: u64 = 64;
const STACKVALUE_SIZE: u64 = 16;
const VNUMINT_TAG: u8 = 0x03;

fn rd_i32(w: &[u8], a: u64) -> i32 {
    i32::from_le_bytes(w[a as usize..a as usize + 4].try_into().unwrap())
}

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

/// Read (sp, L, ci) at `luaV_execute` entry.
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

/// The captured loop-body safepoint + the discovered loop-carried cell addresses.
struct AutoLoop {
    sp: i64,
    l: u64,
    ci: u64,
    base: u64,
    window: Vec<u8>,
    /// Loop-carried register cells (addresses), ascending — the residual's dynamic inputs.
    dyn_cells: Vec<u64>,
    /// The trip-count cell: the loop-carried cell whose value strictly decreases across the
    /// observation (the FORLOOP countdown). Its position in `dyn_cells` is the sweep parameter.
    counter: u64,
}

/// Discover the loop safepoint + carried cells with no hardcoded register offsets, via the shared
/// interpreter-agnostic [`discover`] driver. The Lua-specific inputs are only the located addresses
/// (register base `ci->func + 1 TValue`, pc at `ci->savedpc`) and the static `TargetDesc` (16-byte
/// `TValue` stride, the VNUMINT tag filter) — everything else is the shared driver.
fn auto_capture_loop(m: &Module, luav: u32) -> AutoLoop {
    let dispatch = dispatch_block(m, luav);

    // Locate: attach once to read the entry scalars (sp, L, ci) and derive the register base.
    let inst = svm_run::instantiate(m.clone()).expect("instantiate");
    let mut insp0 = inst.debug_attach(SCRIPT.as_bytes().to_vec(), u64::MAX);
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
    // A fresh inspector positioned at `luaV_execute`'s entry, ready to run to the dispatch loop.
    let make_insp = || {
        let inst = svm_run::instantiate(m.clone()).expect("instantiate");
        let mut insp = inst.debug_attach(SCRIPT.as_bytes().to_vec(), u64::MAX);
        read_entry(&mut insp, luav);
        insp
    };
    let d = discover(&make_insp, luav, dispatch, &loc, &desc);

    assert!(d.counter != 0, "no strictly-decreasing counter cell found");
    assert!(
        d.varying.contains(&d.counter),
        "counter must be a dynamic cell"
    );

    AutoLoop {
        sp,
        l,
        ci,
        base,
        window: d.window,
        dyn_cells: d.varying,
        counter: d.counter,
    }
}

#[test]
fn auto_discovers_the_hand_built_loop_cells() {
    let m = lua_module();
    let luav = luav_execute(&m);
    let cap = auto_capture_loop(&m, luav);

    let saved = rd_u64(&cap.window, cap.ci + CI_SAVEDPC);
    println!("\n=== auto-discovered loop capture ===");
    println!(
        "base={:#x} savedpc={saved:#x} dyn_cells (as R[n]): {:?}",
        cap.base,
        cap.dyn_cells
            .iter()
            .map(|&a| (a - cap.base) / STACKVALUE_SIZE)
            .collect::<Vec<_>>()
    );
    for &a in &cap.dyn_cells {
        println!(
            "  R[{}] @ {a:#x} = {}",
            (a - cap.base) / STACKVALUE_SIZE,
            rd_u64(&cap.window, a)
        );
    }

    // Sanity: the safepoint is a live stack image.
    let stack_lo = rd_u64(&cap.window, cap.l + L_STACK);
    let stack_hi = rd_u64(&cap.window, cap.l + L_STACK_LAST);
    assert_eq!(rd_u64(&cap.window, cap.l + L_CI), cap.ci, "L->ci == ci");
    for &a in &cap.dyn_cells {
        assert!(stack_lo <= a && a < stack_hi, "cell {a:#x} on stack");
    }

    // The hand-built rolled test hardcodes exactly these three cells: x=R[1], i=R[2], counter=R[3].
    // Auto-discovery must recover the same set (it may legitimately surface the internal FORLOOP index
    // too; assert the hand set is a subset and nothing spurious outside the small frame prefix).
    let expect: Vec<u64> = [1u64, 2, 3]
        .iter()
        .map(|&r| cap.base + r * STACKVALUE_SIZE)
        .collect();
    for a in &expect {
        assert!(
            cap.dyn_cells.contains(a),
            "auto discovery missed R[{}] (hand-built dynamic cell)",
            (a - cap.base) / STACKVALUE_SIZE
        );
    }
    println!("entry sp = {}", cap.sp);
}

fn has_br_table(f: &Func) -> bool {
    f.blocks
        .iter()
        .any(|b| matches!(b.term, Terminator::BrTable { .. }))
}

/// Append a `(dyn0, dyn1, ...) -> i64` wrapper that calls the rolled residual (entry 0) then loads the
/// accumulator cell it wrote back — same readback trick as `lua_futamura_rolled.rs`, generalized to
/// the residual's discovered dynamic arity.
fn with_readback(residual: &Module, read_addr: u64, nparams: usize) -> (Module, u32) {
    let mut m = residual.clone();
    let wrapper = m.funcs.len() as u32;
    let params: Vec<ValType> = vec![ValType::I64; nparams];
    let call_args: Vec<u32> = (0..nparams as u32).collect();
    let addr_v = nparams as u32; // ConstI64 result index
    let load_v = nparams as u32 + 1; // Load result index
    let insts = vec![
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
    ];
    m.funcs.push(Func {
        params: params.clone(),
        results: vec![ValType::I64],
        blocks: vec![Block {
            params,
            insts,
            term: Terminator::Return(vec![load_v]),
        }],
    });
    (m, wrapper)
}
fn tw(m: &Module, e: u32, a: &[i64]) -> i64 {
    let mut fuel = u64::MAX;
    let vs: Vec<Value> = a.iter().map(|&x| Value::I64(x)).collect();
    match svm_interp::run(m, e, &vs, &mut fuel)
        .expect("no trap")
        .as_slice()
    {
        [Value::I64(x)] => *x,
        o => panic!("bad {o:?}"),
    }
}
fn jit(m: &Module, e: u32, a: &[i64]) -> i64 {
    match svm_jit::compile_and_run(m, e, a) {
        Ok(svm_jit::JitOutcome::Returned(v)) => v[0],
        o => panic!("jit {o:?}"),
    }
}

#[test]
fn auto_rolled_residual_rolls_and_is_correct() {
    let m = lua_module();
    let luav = luav_execute(&m);
    let cap = auto_capture_loop(&m, luav);
    let w = &cap.window;
    let (l, ci) = (cap.l, cap.ci);

    let stack_lo = rd_u64(w, l + L_STACK);
    let stack_hi = rd_u64(w, l + L_STACK_LAST);
    let stack_len = (stack_hi - stack_lo) as usize;
    let func = rd_u64(w, ci + CI_FUNC);
    let closure = rd_u64(w, func);
    let proto = rd_u64(w, closure + LCLOSURE_P);
    let code = rd_u64(w, proto + PROTO_CODE);
    let sizecode = rd_i32(w, proto + PROTO_SIZECODE) as usize;
    let slice = |a: u64, n: usize| w[a as usize..a as usize + n].to_vec();

    // Same recipe as the hand-built rolled residual — but the dynamic cells come from auto-discovery,
    // not hardcoded register indices.
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
        dynamic_cells: cap.dyn_cells.iter().map(|&a| (a, 8)).collect(),
        indirect_targets_cap: Some(16),
        ..SpecConfig::default()
    };
    let args = [
        SpecArg::ConstI64(cap.sp),
        SpecArg::ConstI64(l as i64),
        SpecArg::ConstI64(ci as i64),
    ];
    let r = specialize_with_config(&m, luav, &args, &cfg).expect("auto-rolled projection");
    svm_verify::verify_module(&r).expect("residual re-verifies");

    let f = &r.funcs[0];
    let n = cap.dyn_cells.len();
    println!(
        "\nAUTO-ROLLED: {} blocks -> {} blocks, {} params (dyn cells), br_table={}",
        m.funcs[luav as usize].blocks.len(),
        f.blocks.len(),
        f.params.len(),
        has_br_table(f)
    );
    assert!(!has_br_table(f), "the dispatch must fold");
    assert_eq!(
        f.params.len(),
        n,
        "residual takes the discovered dynamic cells"
    );
    assert!(
        f.blocks.len() <= 160,
        "the loop rolled ({} blocks)",
        f.blocks.len()
    );

    // Build the dynamic-arg vector: each dyn cell seeded with its captured value, indexed positionally
    // (dynamic_cells are ascending, so param i ↔ dyn_cells[i]).
    let captured: Vec<i64> = cap.dyn_cells.iter().map(|&a| rd_u64(w, a) as i64).collect();
    let counter_ix = cap
        .dyn_cells
        .iter()
        .position(|&a| a == cap.counter)
        .expect("counter among dyn cells");
    // The accumulator is the lowest register (`local x` before the loop) — its write-back is the
    // result. (Generalizing accumulator identification is a later slice; here it is dyn_cells[0].)
    let acc_addr = cap.dyn_cells[0];
    let counter0 = captured[counter_ix];

    let (wm, we) = with_readback(&r, acc_addr, n);
    svm_verify::verify_module(&wm).expect("wrapped residual verifies");

    // Fixed captured seed: `for i=1,50 do x=x+3 end` resumed at the body with counter=49 ⇒ x=150.
    let want_captured = 3 * (counter0 + 1);
    assert_eq!(tw(&wm, we, &captured), want_captured, "tree-walk captured");
    assert_eq!(jit(&wm, we, &captured), want_captured, "jit captured");

    // The loop truly ROLLS: sweep the trip counter and confirm x0 + step*(c+1) on both backends,
    // with the same fixed module (only the dynamic counter arg changes).
    for c in [0i64, 1, 7, 100, 1000] {
        let mut a = captured.clone();
        a[counter_ix] = c;
        let want = 3 * (c + 1);
        assert_eq!(tw(&wm, we, &a), want, "tree-walk c={c}");
        assert_eq!(jit(&wm, we, &a), want, "jit c={c}");
    }
    println!("rolled + correct across a trip sweep (counter param index {counter_ix})");
}
