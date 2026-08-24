//! **Rolled residual for `while` / `repeat` loops (register limit), via `lua_rolled::while_rolled`.**
//!
//! A while/repeat loop has no FORLOOP, so `ci->savedpc` never re-saves and the numeric-`for` discovery
//! can't find it. `while_rolled` instead reads the real pc from the interpreter's live IR, decodes the
//! loop-top from the backward JMP, overrides `savedpc` to resume there, and forces the loop's register
//! limit dynamic so the loop rolls over a swept trip count. This proves both loop forms fold their
//! dispatch, verify, and reproduce real Lua across a trip sweep on both backends.

mod lua_rolled;
mod peval_capture;

use lua_rolled::{has_br_table, lua_module, while_rolled, with_readback};
use temen_interp::Value;
use temen_ir::Module;

fn tw(m: &Module, e: u32, a: &[i64]) -> i64 {
    let mut fuel = u64::MAX;
    let vs: Vec<Value> = a.iter().map(|&x| Value::I64(x)).collect();
    match temen_interp::run(m, e, &vs, &mut fuel)
        .expect("no trap")
        .as_slice()
    {
        [Value::I64(x)] => *x,
        o => panic!("bad {o:?}"),
    }
}
fn jit(m: &Module, e: u32, a: &[i64]) -> i64 {
    match temen_jit::compile_and_run(m, e, a) {
        Ok(temen_jit::JitOutcome::Returned(v)) => v[0],
        o => panic!("jit {o:?}"),
    }
}

#[test]
fn while_and_repeat_roll_over_the_limit() {
    let m = lua_module();
    // (name, script, closed form for limit n). Each is `sum(1..n)` = n(n+1)/2; the limit `n` is a local
    // so it is a register operand the loop can roll over.
    type Closed = fn(i64) -> i64;
    let cases: [(&str, &str, Closed); 2] = [
        (
            "while",
            "local n = 8\nlocal s = 0\nlocal i = 1\nwhile i <= n do s = s + i i = i + 1 end\nreturn s\n",
            |n| n * (n + 1) / 2,
        ),
        (
            "repeat",
            "local n = 8\nlocal s = 0\nlocal i = 1\nrepeat s = s + i i = i + 1 until i > n\nreturn s\n",
            |n| n * (n + 1) / 2,
        ),
    ];
    for (name, script, want_fn) in cases {
        let r = while_rolled(&m, script);
        let f = &r.residual.funcs[r.entry as usize];
        println!(
            "{name}: {} -> {} blocks, {} dyn cells, folds={}",
            r.base_blocks,
            r.residual_blocks,
            r.dyn_cells.len(),
            !has_br_table(f)
        );
        assert!(!has_br_table(f), "{name}: the dispatch must fold");

        let n = r.dyn_cells.len();
        let (wm, we) = with_readback(&r.residual, r.entry, r.acc_addr, n);
        temen_verify::verify_module(&wm).expect("wrapped residual verifies");

        // Captured seed reproduces the observed run (limit 8 ⇒ 36) on both backends.
        let limit0 = r.captured[r.counter_ix];
        assert_eq!(
            tw(&wm, we, &r.captured),
            want_fn(limit0),
            "{name}: captured tree-walk"
        );
        assert_eq!(
            jit(&wm, we, &r.captured),
            want_fn(limit0),
            "{name}: captured jit"
        );

        // The loop ROLLS: sweep the limit (counter_ix), same fixed module, reproduce n(n+1)/2.
        for lim in [1i64, 2, 5, 8, 50, 500] {
            let mut a = r.captured.clone();
            a[r.counter_ix] = lim;
            let want = want_fn(lim);
            assert_eq!(tw(&wm, we, &a), want, "{name}: tree-walk limit={lim}");
            assert_eq!(jit(&wm, we, &a), want, "{name}: jit limit={lim}");
        }
        println!("{name}: folds + rolls over the limit + correct on both backends");
    }
}
