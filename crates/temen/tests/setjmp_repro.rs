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

#![cfg(unix)]

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

/// For each `n` and input, the JIT must equal the tree-walk oracle (INVARIANTS #9).
#[test]
fn jit_setjmp_matches_oracle_under_register_pressure() {
    let inputs: [i64; 5] = [7, 64, 1000, -7, i64::MAX];
    // n <= 14 fits in registers and already agrees; n >= 15 is where the crossing values spill and
    // the longjmp path clobbers their slots. Sweep across the threshold and well past it.
    for n in [1usize, 8, 14, 15, 16, 24, 48, 128] {
        let src = stress(n);
        let m = parse_module(&src).unwrap_or_else(|e| panic!("parse n={n}: {e:?}"));
        verify_module(&m).unwrap_or_else(|e| panic!("verify n={n}: {e:?}"));
        for &x in &inputs {
            let mut fuel = 100_000_000u64;
            let want = match run(&m, 0, &[Value::I64(x)], &mut fuel) {
                Ok(vs) => match vs.as_slice() {
                    [Value::I64(v)] => *v,
                    other => panic!("oracle result shape n={n} x={x}: {other:?}"),
                },
                Err(t) => {
                    panic!("oracle trapped n={n} x={x}: {t:?} (repro expects a clean return)")
                }
            };
            let got = match compile_and_run(&m, 0, &[x]) {
                Ok(JitOutcome::Returned(s)) => s[0],
                other => panic!("jit did not return n={n} x={x}: {other:?}"),
            };
            assert_eq!(
                want, got,
                "#2537: JIT disagrees with the oracle at n={n} x={x} (oracle={want}, jit={got})"
            );
        }
    }
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
    let inputs: [i64; 3] = [7, 1000, -7];
    for n in [1usize, 8, 15, 24, 48] {
        let src = stress_forwarded(n);
        let m = parse_module(&src).unwrap_or_else(|e| panic!("parse n={n}: {e:?}\n{src}"));
        verify_module(&m).unwrap_or_else(|e| panic!("verify n={n}: {e:?}"));
        for &x in &inputs {
            let mut fuel = 100_000_000u64;
            let want = match run(&m, 0, &[Value::I64(x)], &mut fuel) {
                Ok(vs) => match vs.as_slice() {
                    [Value::I64(v)] => *v,
                    other => panic!("oracle result shape n={n} x={x}: {other:?}"),
                },
                Err(t) => panic!("oracle trapped n={n} x={x}: {t:?}"),
            };
            let got = match compile_and_run(&m, 0, &[x]) {
                Ok(JitOutcome::Returned(s)) => s[0],
                other => panic!("jit did not return n={n} x={x}: {other:?}"),
            };
            assert_eq!(
                want, got,
                "JIT disagrees with the oracle at n={n} x={x} (oracle={want}, jit={got})"
            );
        }
    }
}
