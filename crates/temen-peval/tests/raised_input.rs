//! #2672 — specializing a module in dominance form (#2454). The functions the specializer builds come
//! out raised; the ones it only carries come out exactly as they went in; and
//! `lower_indirect_dispatch` raises what it rewrites and leaves the rest alone.
//!
//! Every module here is written block-local and raised, so the functions the specializer reads and
//! the ones it doesn't both capture values of the blocks that dominate them.
//!
//! Run: `cargo test -p temen-peval --test raised_input`

use temen_interp::Value;
use temen_ir::{Func, Module};
use temen_peval::{lower_indirect_dispatch, specialize_with_config, SpecArg, SpecConfig};
use temen_text::parse_module;
use temen_verify::verify_module;

fn run(m: &Module, func: usize, args: &[Value]) -> Vec<Value> {
    let mut fuel = 10_000_000u64;
    temen_interp::run(m, func as u32, args, &mut fuel).expect("runs")
}

/// `f0(n, i) = Σ_{i..n} f1(i) * 7`, with `n` and `7` carried round the loop; `f1(x) = x + 3`, counted
/// out in a loop that carries `x` and `3`; `f2`, the same shape counting to 5, is never called.
/// Raised, each loop captures the values it carried. Both of `f0`'s arguments are dynamic, so its
/// residual keeps the loop, and the specializer inlines `f1`.
const SUM: &str = "
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 0
  v3 = i64.const 7
  br 1(v1, v2, v0, v3)
}
block 1 (v4: i64, v5: i64, v6: i64, v7: i64) {
  v8 = i64.lt_s v4 v6
  br_if v8 2(v4, v5, v6, v7) 3(v5)
}
block 2 (v9: i64, v10: i64, v11: i64, v12: i64) {
  v13 = call 1 (v9)
  v14 = i64.mul v13 v12
  v15 = i64.add v10 v14
  v16 = i64.const 1
  v17 = i64.add v9 v16
  br 1(v17, v15, v11, v12)
}
block 3 (v18: i64) {
  return v18
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 3
  v2 = i64.const 0
  br 1(v2, v0, v1)
}
block 1 (v3: i64, v4: i64, v5: i64) {
  v6 = i64.lt_s v3 v5
  br_if v6 2(v3, v4, v5) 3(v3, v4)
}
block 2 (v7: i64, v8: i64, v9: i64) {
  v10 = i64.const 1
  v11 = i64.add v7 v10
  br 1(v11, v8, v9)
}
block 3 (v12: i64, v13: i64) {
  v14 = i64.add v12 v13
  return v14
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 5
  v2 = i64.const 0
  br 1(v2, v0, v1)
}
block 1 (v3: i64, v4: i64, v5: i64) {
  v6 = i64.lt_s v3 v5
  br_if v6 2(v3, v4, v5) 3(v3, v4)
}
block 2 (v7: i64, v8: i64, v9: i64) {
  v10 = i64.const 1
  v11 = i64.add v7 v10
  br 1(v11, v8, v9)
}
block 3 (v12: i64, v13: i64) {
  v14 = i64.add v12 v13
  return v14
  }
}
";

/// `f0(sel, x)` applies `call.dyn sel` to `x` twice, `sel` carried round the loop; `f1` and `f2` as
/// in [`SUM`] are its targets.
const DISPATCH: &str = "
func (i32, i64) -> (i64) {
block 0 (v0: i32, v1: i64) {
  v2 = i64.const 0
  br 1(v2, v0, v1)
}
block 1 (v3: i64, v4: i32, v5: i64) {
  v6 = i64.const 2
  v7 = i64.lt_s v3 v6
  br_if v7 2(v3, v4, v5) 3(v5)
}
block 2 (v8: i64, v9: i32, v10: i64) {
  v11 = call.dyn (i64) -> (i64) v9 (v10)
  v12 = i64.const 1
  v13 = i64.add v8 v12
  br 1(v13, v9, v11)
}
block 3 (v14: i64) {
  return v14
  }
}
";

/// `text` parsed, verified, and raised.
fn raised(text: &str) -> Module {
    let mut m = parse_module(text).expect("parses");
    verify_module(&m).expect("verifies block-local");
    m.raise();
    verify_module(&m).expect("verifies raised");
    m
}

/// [`DISPATCH`]'s entry with [`SUM`]'s `f1` and `f2` as its targets, raised.
fn dispatch() -> Module {
    let sum = parse_module(SUM).expect("parses");
    let mut m = parse_module(DISPATCH).expect("parses");
    m.funcs.extend(sum.funcs[1..].iter().cloned());
    verify_module(&m).expect("verifies block-local");
    m.raise();
    verify_module(&m).expect("verifies raised");
    m
}

fn captures(f: &Func) -> bool {
    f.blocks.iter().any(|b| !b.captures.is_empty())
}

/// No block of `f` takes a param that only forwards a value: `f` is in dominance form.
fn in_dominance_form(f: &Func) -> bool {
    let fw = f.forwards();
    f.blocks
        .iter()
        .enumerate()
        .all(|(b, blk)| (0..blk.params.len()).all(|i| fw.source(b, i).is_none()))
}

#[test]
fn a_whole_module_carry_keeps_every_source_function_as_it_came() {
    let m = raised(SUM);
    assert!(
        m.funcs.iter().all(captures),
        "every function captures once raised"
    );
    let config = SpecConfig {
        carry_whole_module: true,
        ..SpecConfig::default()
    };
    let r = specialize_with_config(&m, 0, &[SpecArg::Dynamic, SpecArg::Dynamic], &config)
        .expect("specializes");
    verify_module(&r).expect("the residual verifies");
    // `f1` is read (inlined into the loop) and `f2` is not; both are carried exactly as they came.
    assert_eq!(r.funcs[..m.funcs.len()], m.funcs[..]);
    let entry = r.funcs.len() - 1;
    assert!(
        in_dominance_form(&r.funcs[entry]),
        "the built entry is raised"
    );
    for (n, i) in [(0, 0), (1, 0), (5, 2), (9, -3)] {
        let args = [Value::I64(n), Value::I64(i)];
        assert_eq!(run(&r, entry, &args), run(&m, 0, &args), "n = {n}, i = {i}");
    }
}

#[test]
fn a_carried_cut_callee_keeps_its_captures() {
    let m = raised(SUM);
    let config = SpecConfig {
        cut_calls: vec![1],
        ..SpecConfig::default()
    };
    let r = specialize_with_config(&m, 0, &[SpecArg::Dynamic, SpecArg::Dynamic], &config)
        .expect("specializes");
    verify_module(&r).expect("the residual verifies");
    assert_eq!(r.funcs.len(), 2, "the entry, then the carried cut callee");
    assert_eq!(r.funcs[1], m.funcs[1], "carried as it came");
    assert!(in_dominance_form(&r.funcs[0]), "the built entry is raised");
    for (n, i) in [(0, 0), (1, 0), (5, 2), (9, -3)] {
        let args = [Value::I64(n), Value::I64(i)];
        assert_eq!(run(&r, 0, &args), run(&m, 0, &args), "n = {n}, i = {i}");
    }
}

#[test]
fn a_built_outline_comes_out_raised() {
    let m = raised(SUM);
    let config = SpecConfig {
        outline_calls: true,
        ..SpecConfig::default()
    };
    let r = specialize_with_config(&m, 0, &[SpecArg::Dynamic, SpecArg::Dynamic], &config)
        .expect("specializes");
    verify_module(&r).expect("the residual verifies");
    assert!(
        r.funcs.iter().all(in_dominance_form),
        "every built function is raised"
    );
    for (n, i) in [(0, 0), (1, 0), (5, 2), (9, -3)] {
        let args = [Value::I64(n), Value::I64(i)];
        assert_eq!(run(&r, 0, &args), run(&m, 0, &args), "n = {n}, i = {i}");
    }
}

#[test]
fn lowering_raises_what_it_rewrites_and_leaves_the_rest() {
    let m = dispatch();
    assert!(
        m.funcs.iter().all(captures),
        "every function captures once raised"
    );
    let low = lower_indirect_dispatch(&m, 8);
    verify_module(&low).expect("the lowered module verifies");
    assert_ne!(low.funcs[0], m.funcs[0], "the dispatch site is rewritten");
    assert!(in_dominance_form(&low.funcs[0]), "and raised again");
    assert_eq!(low.funcs[1..], m.funcs[1..], "the targets are untouched");
    for (sel, x) in [(1, 4), (2, -3)] {
        let args = [Value::I32(sel), Value::I64(x)];
        assert_eq!(
            run(&low, 0, &args),
            run(&m, 0, &args),
            "sel = {sel}, x = {x}"
        );
    }
}

#[test]
fn specializing_through_a_lowered_dispatch_matches_the_source() {
    let m = dispatch();
    let config = SpecConfig {
        indirect_targets_cap: Some(8),
        ..SpecConfig::default()
    };
    let args = [SpecArg::Dynamic, SpecArg::Dynamic];
    let r = specialize_with_config(&m, 0, &args, &config).expect("specializes");
    verify_module(&r).expect("the residual verifies");
    assert!(
        r.funcs.iter().all(in_dominance_form),
        "every built function is raised"
    );
    for (sel, x) in [(1, 4), (2, -3)] {
        let args = [Value::I32(sel), Value::I64(x)];
        assert_eq!(run(&r, 0, &args), run(&m, 0, &args), "sel = {sel}, x = {x}");
    }
}
