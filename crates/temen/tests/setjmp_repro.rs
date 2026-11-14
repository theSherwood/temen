//! Regression for #2537 — the Cranelift JIT must not miscompile a verified module through the inline
//! `setjmp`/`longjmp` returns-twice path. Fixed by laundering values live across a `setjmp` through a
//! reuse-immune explicit stackslot (`launder_across_setjmp` in temen-jit).
//!
//! ## The bug this guards against
//!
//! A function holds `n` values live across an inline `setjmp`. Beyond the ~5 callee-saved GPRs they
//! spill to frame slots before the `_setjmp` call. The first-return (`longjmp`) block cannot reach
//! the re-entry block in the explicit CFG (it ends in `longjmp` + `unreachable`), so regalloc2 treats
//! those slots as dead there and reuses them for its own spills. `_longjmp` restores `rsp` and the
//! callee-saved registers but not the frame's stack slots, so re-entry would reload the crossing
//! values from slots the `longjmp` path clobbered and compute stale results. Cranelift models
//! `_setjmp` as an ordinary call and never sees the `longjmp` -> re-entry edge. Before the fix the JIT
//! diverged from the tree-walk oracle (INVARIANTS #9) from n=15 up (e.g. n=15, x=7: oracle 226, JIT
//! 299); it now agrees.
//!
//! Boundaries the repro respects (why a naive version sees nothing):
//! - Cranelift's egraph re-elaborates **pure** values at their use sites, so `v0 + k`-style crossing
//!   values are recomputed after the second return rather than carried live. The crossing values here
//!   are therefore non-rematerializable `i64.load`s, which forces genuine cross-`_setjmp` spills.
//! - `memory 16` is 2^16 bytes, but the low region (NULL guard + page-0 durable/powerbox control
//!   words) faults, and the interpreter's `setjmp` writes its token into the window at `buf`. So every
//!   offset here, `buf` included, lives in the usable high window `[40000, 65528)`.

use temen_interp::{run, Value};
use temen_jit::{compile_and_run, JitOutcome};
use temen_text::parse_module;
use temen_verify::verify_module;

/// Build a function that holds `n` values live across an inline `setjmp`, with a high-pressure
/// `longjmp` path, then sums the crossing values after the second return and returns the sum.
fn stress(n: usize) -> String {
    let mut s = String::from("memory 16\nfunc (i64) -> (i64) {\nblock 0 (v0: i64) {\n");
    let mut idx = 1;
    // Seed n distinct window words [v0 + (k+1)] at 41000 + 8k, then load them back as the crossing
    // values. Loads are side-effecting, so Cranelift will not rematerialize them at their re-entry
    // uses -- they are genuinely live across the setjmp and spill once they outnumber the registers.
    for k in 0..n {
        s += &format!("  v{idx} = i64.const {}\n", (k as i64) + 1);
        let kk = idx;
        idx += 1;
        s += &format!("  v{idx} = i64.add v0 v{kk}\n");
        let val = idx;
        idx += 1;
        s += &format!("  v{idx} = i64.const {}\n", 41000 + (k as i64) * 8);
        let off = idx;
        idx += 1;
        s += &format!("  i64.store v{off} v{val}\n");
    }
    let mut crossing = Vec::new();
    for k in 0..n {
        s += &format!("  v{idx} = i64.const {}\n", 41000 + (k as i64) * 8);
        let off = idx;
        idx += 1;
        s += &format!("  v{idx} = i64.load v{off}\n");
        crossing.push(idx);
        idx += 1;
    }
    // setjmp at a window offset in the usable region.
    s += &format!("  v{idx} = i64.const 40000\n");
    let buf = idx;
    idx += 1;
    s += &format!("  v{idx} = setjmp v{buf}\n");
    let sj = idx;
    idx += 1;
    s += &format!("  v{idx} = i32.const 0\n");
    let zero = idx;
    idx += 1;
    s += &format!("  v{idx} = i32.eq v{sj} v{zero}\n");
    let is_first = idx;
    idx += 1;
    // first return -> block 1 (longjmp); re-entry -> block 2, carrying the crossing values + sj value.
    let mut args2: String = crossing.iter().map(|c| format!("v{c}, ")).collect();
    args2 += &format!("v{sj}");
    s += &format!("  br_if v{is_first} 1() 2({args2})\n");
    s += "  }\n";

    // block 1 (longjmp path): high spill pressure with non-rematerializable loads from a disjoint
    // region [43000, ...). regalloc reuses the crossing-value slots here (they are dead on this edge
    // in the explicit CFG), so the hidden longjmp edge makes re-entry reload clobbered slots.
    s += "block 1 () {\n";
    for k in 0..n {
        s += &format!("  v{idx} = i64.const {}\n", (k as i64) * 7 + 5);
        let kk = idx;
        idx += 1;
        s += &format!("  v{idx} = i64.const {}\n", 43000 + (k as i64) * 8);
        let off = idx;
        idx += 1;
        s += &format!("  i64.store v{off} v{kk}\n");
    }
    let mut fresh = Vec::new();
    for k in 0..n {
        s += &format!("  v{idx} = i64.const {}\n", 43000 + (k as i64) * 8);
        let off = idx;
        idx += 1;
        s += &format!("  v{idx} = i64.load v{off}\n");
        fresh.push(idx);
        idx += 1;
    }
    let mut acc = fresh[0];
    for &fv in &fresh[1..] {
        s += &format!("  v{idx} = i64.add v{acc} v{fv}\n");
        acc = idx;
        idx += 1;
    }
    // Store acc (side-effecting) so every fresh load stays live to here and genuinely spills.
    s += &format!("  v{idx} = i64.const 45000\n");
    let accoff = idx;
    idx += 1;
    s += &format!("  i64.store v{accoff} v{acc}\n");
    s += &format!("  v{idx} = i32.const 1\n");
    let one = idx;
    idx += 1;
    s += &format!("  v{idx} = i64.const 40000\n");
    let buf2 = idx;
    idx += 1;
    s += &format!("  longjmp v{buf2} v{one}\n");
    s += "  unreachable\n  }\n";

    // block 2 (re-entry): params are the crossing values then the setjmp result.
    let mut params = String::new();
    let mut pidx = idx;
    let mut p_crossing = Vec::new();
    for _ in 0..n {
        params += &format!("v{pidx}: i64, ");
        p_crossing.push(pidx);
        pidx += 1;
    }
    let p_sj = pidx;
    params += &format!("v{p_sj}: i32");
    pidx += 1;
    s += &format!("block 2 ({params}) {{\n");
    idx = pidx;
    s += &format!("  v{idx} = i64.extend_i32_u v{p_sj}\n");
    let mut sum = idx;
    idx += 1;
    for &c in &p_crossing {
        s += &format!("  v{idx} = i64.add v{sum} v{c}\n");
        sum = idx;
        idx += 1;
    }
    s += &format!("  v{idx} = i64.const 48000\n");
    let fixed = idx;
    idx += 1;
    s += &format!("  i64.store v{fixed} v{sum}\n");
    s += &format!("  v{idx} = i64.load v{fixed}\n");
    let back = idx;
    s += &format!("  return v{back}\n  }}\n}}\n");
    s
}

/// For each `n` and input `x`, the JIT must return what the tree-walk oracle returns on
/// `module(n)` (INVARIANTS #9). Every repro here returns `i64`s cleanly on the oracle.
fn assert_jit_matches_oracle(module: impl Fn(usize) -> String, ns: &[usize], inputs: &[i64]) {
    for &n in ns {
        let src = module(n);
        let m = parse_module(&src).unwrap_or_else(|e| panic!("parse n={n}: {e:?}\n{src}"));
        verify_module(&m).unwrap_or_else(|e| panic!("verify n={n}: {e:?}"));
        if !jit_has_setjmp(&m) {
            continue;
        }
        for &x in inputs {
            let mut fuel = 100_000_000u64;
            let want: Vec<i64> = match run(&m, 0, &[Value::I64(x)], &mut fuel) {
                Ok(vs) => vs
                    .iter()
                    .map(|v| match v {
                        Value::I64(v) => *v,
                        other => panic!("oracle result shape n={n} x={x}: {other:?}"),
                    })
                    .collect(),
                Err(t) => panic!("oracle trapped n={n} x={x}: {t:?} (a repro returns cleanly)"),
            };
            let got = match compile_and_run(&m, 0, &[x]) {
                Ok(JitOutcome::Returned(s)) => s,
                other => panic!("jit did not return n={n} x={x}: {other:?}"),
            };
            assert_eq!(
                want, got,
                "JIT disagrees with the oracle at n={n} x={x} (oracle={want:?}, jit={got:?})"
            );
        }
    }
}

/// For each `n` and input, the JIT must equal the tree-walk oracle (INVARIANTS #9).
#[test]
fn jit_setjmp_matches_oracle_under_register_pressure() {
    // n <= 14 fits in registers and already agrees; n >= 15 is where the crossing values spill and
    // the longjmp path clobbers their slots. Sweep across the threshold and well past it.
    assert_jit_matches_oracle(
        stress,
        &[1, 8, 14, 15, 16, 24, 48, 128],
        &[7, 64, 1000, -7, i64::MAX],
    );
}

/// [`stress`] with the crossing values defined a block *before* the `setjmp`: they reach the
/// `setjmp` block as params that only forward them, and the re-entry block as params that forward
/// those (#2448). A forwarded param reads the value its source block defined, so it must not skip
/// the stack-slot copy the `setjmp` block reloads after the second return.
fn stress_forwarded(n: usize) -> String {
    let mut s = String::from("memory 16\nfunc (i64) -> (i64) {\nblock 0 (v0: i64) {\n");
    let mut idx = 1;
    for k in 0..n {
        s += &format!("  v{idx} = i64.const {}\n", (k as i64) + 1);
        let kk = idx;
        idx += 1;
        s += &format!("  v{idx} = i64.add v0 v{kk}\n");
        let val = idx;
        idx += 1;
        s += &format!("  v{idx} = i64.const {}\n", 41000 + (k as i64) * 8);
        let off = idx;
        idx += 1;
        s += &format!("  i64.store v{off} v{val}\n");
    }
    let mut crossing = Vec::new();
    for k in 0..n {
        s += &format!("  v{idx} = i64.const {}\n", 41000 + (k as i64) * 8);
        let off = idx;
        idx += 1;
        s += &format!("  v{idx} = i64.load v{off}\n");
        crossing.push(idx);
        idx += 1;
    }
    let args: Vec<String> = crossing.iter().map(|c| format!("v{c}")).collect();
    s += &format!("  br 1({})\n  }}\n", args.join(", "));
    // block 1: the setjmp block. Its params only forward block 0's loads.
    let params: Vec<String> = (0..n).map(|k| format!("c{k}: i64")).collect();
    s += &format!("block 1 ({}) {{\n", params.join(", "));
    s += "  buf = i64.const 40000\n  sj = setjmp buf\n  zero = i32.const 0\n  first = i32.eq sj zero\n";
    let fwd: Vec<String> = (0..n).map(|k| format!("c{k}")).collect();
    s += &format!("  br_if first 2() 3({}, sj)\n  }}\n", fwd.join(", "));
    // block 2: the longjmp path, with spill pressure from loads of a disjoint region.
    s += "block 2 () {\n";
    for k in 0..n {
        s += &format!("  k{k} = i64.const {}\n", (k as i64) * 7 + 5);
        s += &format!("  o{k} = i64.const {}\n", 43000 + (k as i64) * 8);
        s += &format!("  i64.store o{k} k{k}\n");
    }
    for k in 0..n {
        s += &format!("  p{k} = i64.const {}\n", 43000 + (k as i64) * 8);
        s += &format!("  f{k} = i64.load p{k}\n");
    }
    let mut acc = "f0".to_string();
    for k in 1..n {
        s += &format!("  a{k} = i64.add {acc} f{k}\n");
        acc = format!("a{k}");
    }
    s += &format!("  ao = i64.const 45000\n  i64.store ao {acc}\n");
    s += "  one = i32.const 1\n  buf2 = i64.const 40000\n  longjmp buf2 one\n  unreachable\n  }\n";
    // block 3: re-entry. Its params forward block 1's, which forward block 0's.
    let params: Vec<String> = (0..n).map(|k| format!("d{k}: i64")).collect();
    s += &format!("block 3 ({}, r: i32) {{\n", params.join(", "));
    s += "  s0 = i64.extend_i32_u r\n";
    for k in 0..n {
        s += &format!("  s{} = i64.add s{k} d{k}\n", k + 1);
    }
    s += &format!("  fixed = i64.const 48000\n  i64.store fixed s{n}\n  back = i64.load fixed\n  return back\n  }}\n}}\n");
    s
}

/// The JIT must equal the oracle when the crossing values reach the re-entry block through params
/// that forward them past the `setjmp` block.
#[test]
fn jit_setjmp_matches_oracle_with_forwarded_crossing_values() {
    assert_jit_matches_oracle(stress_forwarded, &[1, 8, 15, 24, 48], &[7, 1000, -7]);
}

/// #2652 — [`stress`], but the re-entry block also reads the lowering's own context values, which
/// are CLIF values and no IR value names: its `call.dyn` reads the function table's base, the
/// vmctx and the stack limit, and with more than four `results` its `return` writes them through
/// the return area's pointer. Nothing reads the table base, the limit or that pointer between the
/// function's entry and the re-entry, so with `n` loads live at once before the `setjmp` the
/// allocator spills them. The longjmp path makes no call and never returns, so they are dead
/// there, and its own spills may reuse their slots.
fn stress_context(n: usize, results: usize) -> String {
    let types = vec!["i64"; results].join(", ");
    let mut s = format!("memory 16\nfunc (i64) -> ({types}) {{\nblock 0 (v0: i64) {{\n");
    for k in 0..n {
        s += &format!(
            "  c{k} = i64.const {}\n  a{k} = i64.add v0 c{k}\n  o{k} = i64.const {}\n  i64.store o{k} a{k}\n",
            k + 1,
            41000 + 8 * k
        );
    }
    for k in 0..n {
        s += &format!(
            "  p{k} = i64.const {}\n  x{k} = i64.load p{k}\n",
            41000 + 8 * k
        );
    }
    let crossing: Vec<String> = (0..n).map(|k| format!("x{k}")).collect();
    s += &format!(
        "  f = ref.func 1\n  buf = i64.const 40000\n  sj = setjmp buf\n  zero = i32.const 0\n  first = i32.eq sj zero\n  br_if first 1() 2({}, f)\n  }}\n",
        crossing.join(", ")
    );
    // The longjmp path: as many loads live at once as the crossing values, and no call.
    s += "block 1 () {\n";
    for k in 0..n {
        s += &format!(
            "  e{k} = i64.const {}\n  q{k} = i64.const {}\n  i64.store q{k} e{k}\n",
            7 * k + 5,
            43000 + 8 * k
        );
    }
    for k in 0..n {
        s += &format!(
            "  r{k} = i64.const {}\n  y{k} = i64.load r{k}\n",
            43000 + 8 * k
        );
    }
    s += "  t0 = i64.const 0\n";
    for k in 0..n {
        s += &format!("  t{} = i64.add t{k} y{k}\n", k + 1);
    }
    s += &format!(
        "  tc = i64.const 45000\n  i64.store tc t{n}\n  one = i32.const 1\n  buf2 = i64.const 40000\n  longjmp buf2 one\n  unreachable\n  }}\n"
    );
    // The re-entry: sum the crossing values, then dispatch through the function table.
    let params: Vec<String> = (0..n).map(|k| format!("d{k}: i64")).collect();
    s += &format!(
        "block 2 ({}, g: i32) {{\n  s0 = i64.const 0\n",
        params.join(", ")
    );
    for k in 0..n {
        s += &format!("  s{} = i64.add s{k} d{k}\n", k + 1);
    }
    s += &format!(
        "  m = call.dyn (i64) -> (i64) g (s{n})\n  fixed = i64.const 48000\n  i64.store fixed m\n  back = i64.load fixed\n"
    );
    // The other results: the sum plus 1, 2, …, so each is distinct.
    for k in 1..results {
        s += &format!("  k{k} = i64.const {k}\n  w{k} = i64.add s{n} k{k}\n");
    }
    let rest: String = (1..results).map(|k| format!(" w{k}")).collect();
    s += &format!("  return back{rest}\n  }}\n}}\n");
    s += "\nfunc (i64) -> (i64) {\nblock 0 (v0: i64) {\n  v1 = i64.const 3\n  v2 = i64.mul v0 v1\n  return v2\n  }\n}\n";
    s
}

/// The JIT must equal the oracle when the re-entry block reads the context values (#2652). Before
/// the fix, from `n = 15` the table base came back clobbered and the `call.dyn` crashed the host.
#[test]
fn jit_setjmp_matches_oracle_with_context_values_after_the_second_return() {
    let ns = [1, 8, 15, 24, 48, 128];
    assert_jit_matches_oracle(|n| stress_context(n, 1), &ns, &[7, 1000, -7]);
    // Five results return through the return area, whose pointer is a context value too.
    assert_jit_matches_oracle(|n| stress_context(n, 5), &ns, &[7, 1000, -7]);
}

// ---- #2877: a checkpoint lives exactly as long as the activation that took it ----
//
// One rule on every engine: a `longjmp` reaches only a checkpoint of a live activation, named by the
// token its `setjmp` wrote into the buffer (#1062). Anything else traps `Malformed` — on the JIT
// before it would `_longjmp` into a dead native frame, which is host UB.

/// How a run ended on one engine: its results, or the wire code it trapped with
/// (`temen_ir::trap_code`); `None` when the engine declined the module.
type Ended = Option<Result<Vec<i64>, i64>>;

/// `src`'s entry with argument `x` on the tree-walker, the bytecode engine and the Cranelift JIT.
fn on_every_engine(src: &str, x: i64) -> [Ended; 3] {
    let m = parse_module(src).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
    verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let wide = |vs: Vec<Value>| -> Vec<i64> {
        vs.iter()
            .map(|v| match v {
                Value::I64(v) => *v,
                other => panic!("result shape: {other:?}"),
            })
            .collect()
    };
    let mut fuel = 1_000_000u64;
    let tree = Some(
        run(&m, 0, &[Value::I64(x)], &mut fuel)
            .map(wide)
            .map_err(|t| t.code()),
    );
    let mut fuel = 1_000_000u64;
    let byte = temen_interp::bytecode::compile_and_run(&m, 0, &[Value::I64(x)], &mut fuel)
        .map(|r| r.map(wide).map_err(|t| t.code()));
    // The JIT's `setjmp` is unix-only (`setjmp_rt`); elsewhere it declines every module with one, so
    // a routed run folds to the interpreters, whose answer the row already holds.
    let jit = if jit_has_setjmp(&m) {
        match compile_and_run(&m, 0, &[x]) {
            Ok(JitOutcome::Returned(vs)) => Some(Ok(vs)),
            Ok(JitOutcome::Trapped(k)) => Some(Err(k.code())),
            other => panic!("jit: {other:?}"),
        }
    } else {
        None // declined: a routed run folds to the interpreters, whose answers the row holds
    };
    [tree, byte, jit]
}

/// Whether this target's JIT runs `m`, a module that uses `setjmp`: on unix (`setjmp_rt`) it does;
/// elsewhere it must decline it.
fn jit_has_setjmp(m: &temen_ir::Module) -> bool {
    assert!(
        m.funcs.iter().any(temen_ir::Func::uses_setjmp),
        "jit_has_setjmp asks only of a module that uses setjmp"
    );
    if cfg!(unix) {
        return true;
    }
    assert!(!temen_jit::admits(m), "the JIT declines setjmp here");
    false
}

/// Every engine ends `src(x)` as `want`; where the JIT has no `setjmp`, it declines (`None`).
fn assert_every_engine(src: &str, x: i64, want: Result<Vec<i64>, i64>) {
    let [tree, byte, jit] = on_every_engine(src, x);
    assert_eq!(tree.as_ref(), Some(&want), "tree-walker at x={x}: {tree:?}");
    assert_eq!(byte.as_ref(), Some(&want), "bytecode at x={x}: {byte:?}");
    let want_jit = cfg!(unix).then_some(&want);
    assert_eq!(jit.as_ref(), want_jit, "Cranelift JIT at x={x}: {jit:?}");
}

const MALFORMED: i64 = temen_ir::trap_code::MALFORMED;

/// Func 1 takes a checkpoint at the buffer (48000) and returns; func 2 `longjmp`s to it with 7.
/// `x = 0`: the root runs both, so func 2 reaches a checkpoint whose activation returned, at the
/// depth that activation had — the depth-only rule let this through and ran func 2's body at
/// func 1's resume point. `x = 1`: func 1 alone, which returns 0.
const RETURNED: &str = "memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 48000
  v2 = call 1(v1)
  v3 = i64.const 0
  v4 = i64.eq v0 v3
  br_if v4 1(v1) 2(v2)
  }
block 1 (v5: i64) {
  v6 = call 2(v5)
  return v6
  }
block 2 (v7: i64) {
  return v7
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = setjmp v0
  v2 = i64.extend_i32_u v1
  return v2
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i32.const 7
  longjmp v0 v1
  unreachable
  }
}
";

#[test]
fn a_longjmp_to_a_returned_activation_traps_on_every_engine() {
    assert_every_engine(RETURNED, 0, Err(MALFORMED));
    assert_every_engine(RETURNED, 1, Ok(vec![0]));
}

/// #2877 — the same program, optimized: the inliner must not move func 1's checkpoint into the
/// root's activation (`Func::takes_checkpoints`), where func 2's `longjmp` would find it live and
/// return 7 out of a frame whose activation returned.
#[test]
fn optimization_keeps_a_returned_activations_checkpoint_dead() {
    let m = parse_module(RETURNED).expect("parse");
    let opt = temen_opt::optimize_module(&m);
    // Checked first, so an inliner that moves the checkpoint fails here rather than running a
    // program whose JIT run would then spin (it has no fuel to stop it).
    assert!(
        !opt.funcs[0].takes_checkpoints(),
        "func 1's `setjmp` was inlined into the root"
    );
    let src = temen_text::print_module(&opt);
    assert_every_engine(&src, 0, Err(MALFORMED));
    assert_every_engine(&src, 1, Ok(vec![0]));
}

/// One call site runs func 1 twice: the first activation takes a checkpoint and returns, the second
/// `longjmp`s to it before taking its own. Same function, same depth, same return address — only the
/// activation differs, and the first one has returned.
const SAME_SITE: &str = "memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  br 1(v1)
  }
block 1 (v2: i64) {
  v3 = i64.const 48000
  v4 = call 1(v3, v2)
  v5 = i64.const 1
  v6 = i64.add v2 v5
  v7 = i64.const 2
  v8 = i64.lt_s v6 v7
  br_if v8 1(v6) 2(v4)
  }
block 2 (v9: i64) {
  return v9
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 0
  v3 = i64.eq v1 v2
  br_if v3 1(v0) 2(v0)
  }
block 1 (v4: i64) {
  v5 = setjmp v4
  v6 = i64.extend_i32_u v5
  return v6
  }
block 2 (v7: i64) {
  v8 = i32.const 5
  longjmp v7 v8
  unreachable
  }
}
";

#[test]
fn a_longjmp_from_a_later_activation_at_the_same_call_site_traps() {
    assert_every_engine(SAME_SITE, 0, Err(MALFORMED));
}

/// The root takes checkpoint A, calls func 1, which takes B and `longjmp`s to A — unwinding func 1 and
/// B with it — and the root then `longjmp`s to B: B's activation is gone. `x = 1` stops after the
/// first `longjmp`, which returns 1 through A.
const UNWOUND: &str = "memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 48000
  v2 = setjmp v1
  v3 = i32.const 0
  v4 = i32.eq v2 v3
  br_if v4 1() 2(v2, v0)
  }
block 1 () {
  v5 = i64.const 48000
  v6 = i64.const 48016
  v7 = call 1(v5, v6)
  return v7
  }
block 2 (v8: i32, v15: i64) {
  v9 = i64.const 0
  v10 = i64.eq v15 v9
  br_if v10 3() 4(v8)
  }
block 3 () {
  v11 = i64.const 48016
  v12 = i32.const 9
  longjmp v11 v12
  unreachable
  }
block 4 (v13: i32) {
  v14 = i64.extend_i32_u v13
  return v14
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = setjmp v1
  v3 = i32.const 1
  longjmp v0 v3
  unreachable
  }
}
";

#[test]
fn a_longjmp_to_a_checkpoint_an_earlier_longjmp_unwound_traps() {
    assert_every_engine(UNWOUND, 0, Err(MALFORMED));
    assert_every_engine(UNWOUND, 1, Ok(vec![1]));
}

/// bash's `COPY_PROCENV` shape (#1062): the root takes checkpoint A in the buffer and calls func 1,
/// which saves the buffer's bytes, takes its own checkpoint in the same buffer and `longjmp`s to it;
/// the handler restores the saved bytes and `longjmp`s again *while func 1 is still live*. The restored
/// token names the root's checkpoint, so the run returns 100 + 3 — an address-keyed table resolves the
/// newest checkpoint at the address instead and loops forever.
const COPIED: &str = "memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 48000
  v2 = setjmp v1
  v3 = i32.const 0
  v4 = i32.eq v2 v3
  br_if v4 1() 2(v2)
  }
block 1 () {
  v5 = i64.const 48000
  v6 = call 1(v5)
  return v6
  }
block 2 (v7: i32) {
  v8 = i64.extend_i32_u v7
  v9 = i64.const 100
  v10 = i64.add v8 v9
  return v10
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.load v0
  v2 = i64.const 48016
  i64.store v2 v1
  v3 = setjmp v0
  v4 = i32.const 0
  v5 = i32.eq v3 v4
  br_if v5 1(v0) 2(v0)
  }
block 1 (v6: i64) {
  v7 = i32.const 1
  longjmp v6 v7
  unreachable
  }
block 2 (v8: i64) {
  v9 = i64.const 48016
  v10 = i64.load v9
  i64.store v8 v10
  v11 = i32.const 3
  longjmp v8 v11
  unreachable
  }
}
";

#[test]
fn a_restored_copy_of_a_jmp_buf_reaches_its_own_checkpoint_on_every_engine() {
    assert_every_engine(COPIED, 0, Ok(vec![103]));
}

/// A `jmp_buf` in the NULL guard faults at the `setjmp`'s token store on every engine, as any store
/// there does.
#[test]
fn a_setjmp_into_the_null_guard_faults_on_every_engine() {
    let src = RETURNED.replace("v1 = i64.const 48000", "v1 = i64.const 64");
    assert_every_engine(&src, 1, Err(temen_ir::trap_code::MEMORY_FAULT));
}

/// A `jmp_buf` in a read-only data segment faults at the `setjmp`'s token store on every engine: the
/// JIT's host-side store goes through the recoverable window access (#2996), so the fault is the
/// guest's `MemoryFault`, not a host fault.
#[test]
fn a_setjmp_into_a_readonly_page_faults_on_every_engine() {
    let src = RETURNED
        .replacen(
            "memory 16\n",
            "memory 16\ndata ro 16384 \"0123456789abcdef\"\n",
            1,
        )
        .replace("v1 = i64.const 48000", "v1 = i64.const 16384");
    assert_every_engine(&src, 1, Err(temen_ir::trap_code::MEMORY_FAULT));
}

/// One compiled module run twice over one window: the first run (`x = 0`) takes a checkpoint (its
/// token stays in the buffer) and then ends as `first_end` says; the second (`x = 1`) `longjmp`s with
/// that token. The first run's activation is long gone — the JIT's table outlives runs, so it must
/// drop a run's checkpoints when the run ends. The first run's outcome and the second's; `None` where
/// the JIT has no `setjmp` (it declines the module, [`jit_has_setjmp`]).
fn rerun(first_end: &str) -> Option<(JitOutcome, JitOutcome)> {
    let src = format!(
        "memory 16
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  v1 = i64.const 48000
  v2 = i64.const 0
  v3 = i64.eq v0 v2
  br_if v3 1(v1) 2(v1)
  }}
block 1 (v4: i64) {{
  v5 = setjmp v4
  v6 = i64.extend_i32_u v5
  {first_end}
  }}
block 2 (v7: i64) {{
  v8 = i32.const 7
  longjmp v7 v8
  unreachable
  }}
}}
"
    );
    let m = parse_module(&src).expect("parse");
    verify_module(&m).expect("verify");
    if !jit_has_setjmp(&m) {
        return None;
    }
    let mut host = temen_interp::Host::new();
    let mut cm = temen_jit::CompiledModule::compile(
        &m,
        0,
        temen_run::cap_thunk,
        &mut host as *mut temen_interp::Host as *mut core::ffi::c_void,
        temen_ir::DEFAULT_RESERVED_LOG2,
        None,
        None,
        None,
        None,
        None,
        0,
    )
    .expect("compile");
    // The first run's window, captured whole, is the second's: the token is in the buffer.
    let (first, window) = cm.run(&[0], None, Some(1 << 16)).expect("first run");
    let token = u64::from_le_bytes(window[48000..48008].try_into().expect("the window"));
    assert_ne!(token, 0, "the first run's setjmp wrote its token");
    let (second, _) = cm.run(&[1], Some(&window), None).expect("second run");
    Some((first, second))
}

/// The first run returns: its exit drops the checkpoint (`rt_setjmp_leave`).
#[test]
fn a_jit_rerun_cannot_reach_an_earlier_runs_checkpoint() {
    let Some((first, second)) = rerun("return v6") else {
        return;
    };
    assert!(
        matches!(first, JitOutcome::Returned(ref v) if v == &[0]),
        "{first:?}"
    );
    assert!(
        matches!(second, JitOutcome::Trapped(k) if k.code() == MALFORMED),
        "the second run reached the first's checkpoint: {second:?}"
    );
}

/// The first run faults holding its checkpoint: a hardware fault (a load in the NULL guard) unwinds
/// the whole guarded call, so no frame's exit runs to drop it. The guarded call's scope does
/// (`setjmp_rt::Scope`, which every run enters). Without it the second run's `longjmp` would land in
/// the first run's dead frame.
#[test]
fn a_jit_rerun_cannot_reach_a_checkpoint_a_faulted_run_held() {
    let Some((first, second)) = rerun("v9 = i64.const 64\n  v10 = i64.load v9\n  return v10")
    else {
        return;
    };
    assert!(
        matches!(first, JitOutcome::Trapped(k) if k.code() == temen_ir::trap_code::MEMORY_FAULT),
        "{first:?}"
    );
    assert!(
        matches!(second, JitOutcome::Trapped(k) if k.code() == MALFORMED),
        "the second run reached the trapped run's checkpoint: {second:?}"
    );
}

/// #2877 — a tail call ends the activation that made it, so it drops the checkpoints that activation
/// took, as a return does. Func 1 takes a checkpoint and tail-calls func 2 (`return_call`, or
/// `return_call.dyn` when `x = 1`), which `longjmp`s to it: the checkpoint's activation is gone, so
/// every engine traps. On the JIT that is the `rt_setjmp_leave` before each tail call; on the
/// bytecode engine the `leave` bit on `TailCall` / `TailCallIndirect`.
const TAIL_CALLED_AWAY: &str = "memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 48000
  v2 = call 1(v1, v0)
  return v2
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = setjmp v0
  v3 = i32.const 0
  v4 = i32.eq v2 v3
  br_if v4 1(v0, v1) 4(v2)
  }
block 1 (v5: i64, v6: i64) {
  v7 = i64.const 0
  v8 = i64.eq v6 v7
  br_if v8 2(v5) 3(v5)
  }
block 2 (v9: i64) {
  return_call 2(v9)
  }
block 3 (v10: i64) {
  v11 = i32.const 2
  return_call.dyn (i64) -> (i64) v11 (v10)
  }
block 4 (v12: i32) {
  v13 = i64.extend_i32_u v12
  return v13
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i32.const 7
  longjmp v0 v1
  unreachable
  }
}
";

#[test]
fn a_longjmp_to_a_checkpoint_its_activation_tail_called_away_traps_on_every_engine() {
    assert_every_engine(TAIL_CALLED_AWAY, 0, Err(MALFORMED));
    assert_every_engine(TAIL_CALLED_AWAY, 1, Err(MALFORMED));
}

/// #2877 — a module that uses `setjmp` and serves an interface is declined by the JIT before it
/// runs (`setjmp_rt::admits`): its handlers' residencies run on pool or caller threads and give the
/// window's gate back mid-call, so their guarded calls on the one checkpoint table would not nest. The
/// interpreters keep each checkpoint with its frame and run it.
#[test]
fn a_module_with_setjmp_that_serves_is_declined_on_the_jit_and_runs_on_the_interpreters() {
    let src = "memory 16
type 0 func () -> (i64)
type 1 interface { ping: 0 }
export 0 interface \"svc\" 1 { ping: 1 }
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 48000
  v2 = setjmp v1
  v3 = i64.extend_i32_u v2
  v4 = i64.add v3 v0
  return v4
  }
}
func () -> (i64) {
block 0 () {
  v0 = i64.const 5
  return v0
  }
}
";
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut fuel = 1_000_000u64;
    assert_eq!(
        run(&m, 0, &[Value::I64(3)], &mut fuel),
        Ok(vec![Value::I64(3)])
    );
    let mut fuel = 1_000_000u64;
    let byte = temen_interp::bytecode::compile_and_run(&m, 0, &[Value::I64(3)], &mut fuel);
    assert_eq!(byte, Some(Ok(vec![Value::I64(3)])), "bytecode");
    assert!(
        matches!(
            compile_and_run(&m, 0, &[3]),
            Err(temen_jit::JitError::Unsupported(_))
        ),
        "the JIT declines it"
    );
    // A routed run asks the same rule before it starts (`temen_jit::admits`), so `Backend::Jit`
    // folds to the interpreters and answers what they answer, rather than ending in the compile's
    // error. (A routed run passes its entry no arguments, so this one takes none.)
    let nullary = src
        .replace(
            "func (i64) -> (i64) {\nblock 0 (v0: i64) {",
            "func () -> (i64) {\nblock 0 () {",
        )
        .replace("  v4 = i64.add v3 v0\n  return v4\n", "  return v3\n");
    let m = parse_module(&nullary).expect("parse");
    verify_module(&m).expect("verify");
    assert!(!temen_jit::admits(&m), "the router sees the refusal");
    let routed = |backend| {
        temen_run::instantiate(m.clone())
            .expect("instantiates")
            .run(backend, &temen_run::RunConfig::default())
            .map(|r| r.outcome)
    };
    let tree = routed(temen_run::Backend::TreeWalk);
    assert!(
        matches!(&tree, Ok(temen_run::Outcome::Returned(v)) if v == &[Value::I64(0)]),
        "{tree:?}"
    );
    assert_eq!(
        routed(temen_run::Backend::Jit),
        tree,
        "a routed JIT run folds"
    );
}
