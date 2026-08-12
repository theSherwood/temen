//! #850 — a deterministic **differential fuzz** over flat-struct aggregate SSA values (`insertvalue`
//! chains over `{i64 × N}`, including partially-defined constant bases).
//!
//! Motivation: two silent miscompiles during the net work traced to aggregate-constant corners — in
//! particular `insertvalue` into a *partially-defined* constant base (some fields `undef`). This
//! generates random `{i64 × N}` structs built from an `undef` / `zeroinitializer` / partially-`undef`
//! constant base by a chain of `insertvalue`s (last-write-wins per index), then `extractvalue`s and
//! sums every **defined** field, and checks the result against a reference that tracks each field's
//! final value. Only defined fields are read, so no `undef` is observed.
//!
//! Nested structs and array-typed SSA aggregates are an explicit `Milestone 1+` limitation of the
//! on-ramp (they fail *closed*), so the generator stays on the flat-`{i64 × N}` surface that is
//! actually lowered — which is exactly where the silent miscompiles were.

use svm_interp::Value;

/// A tiny reproducible LCG — no `rand`/`Date` dependency.
struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next_u32() % n
    }
    fn val(&mut self) -> i64 {
        (self.next_u32() as i64 % 8192) - 4096
    }
}

#[test]
fn aggregate_insertvalue_differential_fuzz() {
    let mut failures: Vec<String> = Vec::new();
    for seed in 0..300u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1));
        let n = 2 + rng.below(4) as usize; // struct fields: 2..=5
        let ty = format!("{{{}}}", vec!["i64"; n].join(", "));

        // The struct's final field values, as the reference tracks them (None = still undef).
        let mut defined: Vec<Option<i64>> = vec![None; n];

        // The base constant: undef / zeroinitializer / a partially-`undef` literal aggregate.
        let base = match rng.below(3) {
            0 => "undef".to_string(),
            1 => {
                defined = vec![Some(0); n];
                "zeroinitializer".to_string()
            }
            _ => {
                let fields: Vec<String> = (0..n)
                    .map(|i| {
                        if rng.below(2) == 0 {
                            let v = rng.val();
                            defined[i] = Some(v);
                            format!("i64 {v}")
                        } else {
                            "i64 undef".to_string()
                        }
                    })
                    .collect();
                format!("{{{}}}", fields.join(", "))
            }
        };

        // A chain of `insertvalue`s (last-write-wins per index).
        let mut lines = String::new();
        let mut prev = base;
        let inserts = rng.below(7); // 0..=6
        for k in 0..inserts {
            let idx = rng.below(n as u32) as usize;
            let v = rng.val();
            defined[idx] = Some(v);
            lines.push_str(&format!(
                "  %a{k} = insertvalue {ty} {prev}, i64 {v}, {idx}\n"
            ));
            prev = format!("%a{k}");
        }

        // Read + sum every defined field (never an `undef` one).
        let mut sum_ref: i64 = 0;
        let mut acc = "0".to_string();
        let mut e = 0;
        for (i, d) in defined.iter().enumerate() {
            if let Some(v) = d {
                sum_ref = sum_ref.wrapping_add(*v);
                lines.push_str(&format!("  %e{e} = extractvalue {ty} {prev}, {i}\n"));
                lines.push_str(&format!("  %s{e} = add i64 {acc}, %e{e}\n"));
                acc = format!("%s{e}");
                e += 1;
            }
        }

        let ir = format!(
            "target datalayout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128\"\n\
             target triple = \"x86_64-unknown-none\"\n\
             define i64 @f() {{\nentry:\n{lines}  ret i64 {acc}\n}}\n"
        );

        let t = match svm_llvm::translate_ll_str(&ir) {
            Ok(t) => t,
            Err(err) => {
                failures.push(format!("seed {seed}: translate failed: {err:?}\n{ir}"));
                continue;
            }
        };
        if let Err(err) = svm_verify::verify_module(&t.module) {
            failures.push(format!("seed {seed}: verify failed: {err:?}\n{ir}"));
            continue;
        }
        let f = t
            .exports
            .iter()
            .find(|(n, _)| n == "f")
            .map(|x| x.1)
            .unwrap();
        let mut fuel = 1_000_000u64;
        match svm_interp::run(&t.module, f, &[Value::I64(t.entry_sp as i64)], &mut fuel) {
            Ok(v) => {
                let got = match v[0] {
                    Value::I64(x) => x,
                    other => {
                        failures.push(format!("seed {seed}: non-i64 result {other:?}"));
                        continue;
                    }
                };
                if got != sum_ref {
                    failures.push(format!(
                        "seed {seed}: mismatch got {got} want {sum_ref}\n{ir}"
                    ));
                }
            }
            Err(err) => failures.push(format!("seed {seed}: interp trapped: {err:?}\n{ir}")),
        }
    }
    assert!(
        failures.is_empty(),
        "{} aggregate fuzz case(s) failed:\n{}",
        failures.len(),
        failures.join("\n---\n")
    );
}
