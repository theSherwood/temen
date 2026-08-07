//! **Auto-rolled residual: zero-config, via the driver-level `lua_rolled::auto_rolled`.**
//!
//! The rolled residual (the shape that delivers the 11.5× per-iteration win) is now built by a
//! reusable driver function — `lua_rolled::auto_rolled(m, script)` — on top of the shared
//! interpreter-agnostic `peval_capture::discover`. This test drives it with no hardcoded register
//! offsets: it recovers the loop's carried cells, identifies the counter and accumulator, and the
//! real `luaV_execute` loop rolls (dispatch folds, byte-identical across backends).
//!
//! Run: `cargo test -p svm-llvm --test lua_futamura_auto_rolled -- --nocapture`

mod lua_rolled;
mod peval_capture;

use lua_rolled::{auto_rolled, has_br_table, lua_module, with_readback};
use svm_interp::Value;
use svm_ir::Module;

const SCRIPT: &str = "local x = 0\nfor i = 1, 50 do x = x + 3 end\nreturn x\n";

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
fn auto_discovers_the_hand_built_loop_cells() {
    let m = lua_module();
    let r = auto_rolled(&m, SCRIPT);
    let as_reg = |a: u64| (a - r.reg_base) / r.reg_stride;
    println!(
        "\nauto_rolled cells (as R[n]): {:?}  counter=R{} acc=R{}",
        r.dyn_cells.iter().map(|&a| as_reg(a)).collect::<Vec<_>>(),
        as_reg(r.dyn_cells[r.counter_ix]),
        as_reg(r.acc_addr),
    );
    // The hand-built rolled test hardcodes exactly x=R[1], i=R[2], counter=R[3]; auto-discovery must
    // recover that set (an extra internal FORLOOP register is harmless).
    for reg in [1u64, 2, 3] {
        let a = r.reg_base + reg * r.reg_stride;
        assert!(r.dyn_cells.contains(&a), "auto discovery missed R[{reg}]");
    }
    assert_eq!(as_reg(r.dyn_cells[r.counter_ix]), 3, "counter is R[3]");
    assert_eq!(
        as_reg(r.acc_addr),
        1,
        "accumulator is R[1] (first local before the loop)"
    );
}

#[test]
fn auto_rolled_residual_rolls_and_is_correct() {
    let m = lua_module();
    let r = auto_rolled(&m, SCRIPT);
    let f = &r.residual.funcs[0];
    let n = r.dyn_cells.len();
    println!(
        "\nAUTO-ROLLED: {} blocks -> {} blocks, {} params (dyn cells), br_table={}",
        r.base_blocks,
        r.residual_blocks,
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
        r.residual_blocks <= 160,
        "the loop rolled ({} blocks)",
        r.residual_blocks
    );

    let (wm, we) = with_readback(&r.residual, r.acc_addr, n);
    svm_verify::verify_module(&wm).expect("wrapped residual verifies");

    // Captured seed reproduces the real Lua answer (`for i=1,50 do x=x+3 end` resumed at the body with
    // counter=49 ⇒ x=150), on all backends.
    let counter0 = r.captured[r.counter_ix];
    let want_captured = 3 * (counter0 + 1);
    assert_eq!(
        tw(&wm, we, &r.captured),
        want_captured,
        "captured tree-walk"
    );
    assert_eq!(jit(&wm, we, &r.captured), want_captured, "captured jit");

    // The loop truly ROLLS: sweep the dynamic trip counter, same fixed module, x0 + 3·(c+1).
    for c in [0i64, 1, 7, 100, 1000] {
        let mut a = r.captured.clone();
        a[r.counter_ix] = c;
        let want = 3 * (c + 1);
        assert_eq!(tw(&wm, we, &a), want, "tree-walk c={c}");
        assert_eq!(jit(&wm, we, &a), want, "jit c={c}");
    }
    println!(
        "rolled + correct across a trip sweep (counter param index {})",
        r.counter_ix
    );
}
