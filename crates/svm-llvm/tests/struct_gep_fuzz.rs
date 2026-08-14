//! #850 — a deterministic **differential fuzz** over constexpr `getelementptr` **into structs**.
//!
//! Motivation: the flat-`i8` GEP fuzz in `constexpr_fuzz.rs` never reaches `const_gep_offset`'s
//! struct-field descent (`struct_layout` — per-field alignment padding + field indexing) or its array
//! descent, and those are exactly the corners where an offset bug hides silently. This generates a
//! random struct global (mixed-width scalar fields, plus one level of nested-struct / array fields),
//! initializes every scalar **leaf** to a distinct sentinel, then reads each leaf back through a
//! **constexpr GEP** (`load iN, ptr getelementptr inbounds ({S}, ptr @g, i64 0, i32 f, …)`) and checks
//! it equals the sentinel.
//!
//! Why this is a real cross-check, not a tautology: the initializer places each field via the
//! *serializer* (`const_bytes` walking the struct's values in order), while the read computes its byte
//! offset via `const_gep_offset`. Those are **two independent code paths**; if they disagree on where a
//! field lives (a padding mismatch, an off-by-one in the field walk, a botched array/nested descent),
//! the leaf reads the wrong bytes and the case fails. The interpreter is the oracle (DESIGN §18).
//!
//! No external deps: a fixed-seed LCG keeps it reproducible and CI-cheap.

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
    /// A sentinel fitting `bits` (8/16/32/64), non-trivial so byte placement matters.
    fn sentinel(&mut self, bits: u32) -> u64 {
        let raw = ((self.next_u32() as u64) << 32) | self.next_u32() as u64;
        if bits >= 64 {
            raw
        } else {
            raw & ((1u64 << bits) - 1)
        }
    }
}

/// One scalar leaf: the GEP index path to it (after the leading `i64 0`), its width, and its sentinel.
struct Leaf {
    /// `(is_struct_field, index)` per level — struct fields are `i32`-typed indices, array elements
    /// `i64`-typed, matching LLVM's GEP index typing.
    path: Vec<(bool, usize)>,
    bits: u32,
    sentinel: u64,
}

const SCALAR_BITS: [u32; 4] = [8, 16, 32, 64];

/// Render a scalar field's type + initializer for width `bits` with value `v`.
fn scalar(bits: u32, v: u64) -> (String, String) {
    (format!("i{bits}"), format!("i{bits} {v}"))
}

/// Generate the struct: returns `(type_str, init_str, leaves)`. Each top-level field is a scalar, a
/// nested struct (2..=3 scalars), or an array (2..=4 scalars) — one level of nesting, enough to drive
/// both descents in `const_gep_offset`.
fn gen_struct(rng: &mut Rng) -> (String, String, Vec<Leaf>) {
    let nfields = 2 + rng.below(4) as usize; // 2..=5
    let mut tys = Vec::new();
    let mut inits = Vec::new();
    let mut leaves = Vec::new();
    for f in 0..nfields {
        // Scalar or top-level array field. **Nested struct/array constants are an explicit Milestone
        // 1+ limitation of the on-ramp** (they fail closed at parse — see `aggregate_fuzz.rs`), so the
        // generator stays on the surface that is actually lowered: a flat struct of scalars and
        // single-level arrays. That still drives both descents in `const_gep_offset` — struct-field
        // (index + `struct_layout` padding) and array-element (`i64` stride) — which is the target.
        match rng.below(2) {
            // A scalar field.
            0 => {
                let bits = SCALAR_BITS[rng.below(4) as usize];
                let s = rng.sentinel(bits);
                let (ty, init) = scalar(bits, s);
                tys.push(ty);
                inits.push(init);
                leaves.push(Leaf {
                    path: vec![(true, f)],
                    bits,
                    sentinel: s,
                });
            }
            // An array of 2..=4 scalars (uniform width).
            _ => {
                let n = 2 + rng.below(3) as usize;
                let bits = SCALAR_BITS[rng.below(4) as usize];
                let mut elems = Vec::new();
                for k in 0..n {
                    let s = rng.sentinel(bits);
                    elems.push(format!("i{bits} {s}"));
                    leaves.push(Leaf {
                        path: vec![(true, f), (false, k)],
                        bits,
                        sentinel: s,
                    });
                }
                tys.push(format!("[{n} x i{bits}]"));
                inits.push(format!("[{n} x i{bits}] [{}]", elems.join(", ")));
            }
        }
    }
    (
        format!("{{{}}}", tys.join(", ")),
        format!("{{{}}}", inits.join(", ")),
        leaves,
    )
}

/// Render the constexpr-GEP index list (after `ptr @g`) for a leaf: the leading deref `i64 0` then one
/// index per level (`i32` for a struct field, `i64` for an array element).
fn gep_indices(path: &[(bool, usize)]) -> String {
    let mut s = String::from("i64 0");
    for (is_field, idx) in path {
        if *is_field {
            s.push_str(&format!(", i32 {idx}"));
        } else {
            s.push_str(&format!(", i64 {idx}"));
        }
    }
    s
}

#[test]
fn struct_gep_leaf_readback_differential_fuzz() {
    let mut failures: Vec<String> = Vec::new();
    for seed in 0..300u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1));
        let (sty, sinit, leaves) = gen_struct(&mut rng);

        // `@f` reads every leaf through a constexpr GEP and folds `(loaded XOR sentinel)` into an OR
        // accumulator — the result is 0 **iff** every leaf read back its sentinel (no cancellation, so
        // any single mismatched leaf makes it non-zero).
        let mut body = String::from("  %acc0 = or i64 0, 0\n");
        for (i, leaf) in leaves.iter().enumerate() {
            let idx = gep_indices(&leaf.path);
            let b = leaf.bits;
            // Load the leaf through the constexpr GEP, widen to i64.
            body.push_str(&format!(
                "  %v{i} = load i{b}, ptr getelementptr inbounds ({sty}, ptr @g, {idx})\n"
            ));
            if b == 64 {
                body.push_str(&format!("  %z{i} = or i64 %v{i}, 0\n"));
            } else {
                body.push_str(&format!("  %z{i} = zext i{b} %v{i} to i64\n"));
            }
            body.push_str(&format!("  %x{i} = xor i64 %z{i}, {}\n", leaf.sentinel));
            body.push_str(&format!("  %acc{} = or i64 %acc{i}, %x{i}\n", i + 1));
        }
        let last = leaves.len();
        let ir = format!(
            "target datalayout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128\"\n\
             target triple = \"x86_64-unknown-none\"\n\
             @g = internal global {sty} {sinit}, align 8\n\
             define i64 @f() {{\nentry:\n{body}  ret i64 %acc{last}\n}}\n"
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
            Ok(v) => match v[0] {
                Value::I64(0) => {}
                Value::I64(x) => failures.push(format!(
                    "seed {seed}: {} leaf/leaves mismatched (acc={x:#x}, want 0)\n{ir}",
                    leaves.len()
                )),
                other => failures.push(format!("seed {seed}: non-i64 result {other:?}")),
            },
            Err(err) => failures.push(format!("seed {seed}: interp trapped: {err:?}\n{ir}")),
        }
    }
    assert!(
        failures.is_empty(),
        "{} struct-GEP fuzz case(s) failed:\n{}",
        failures.len(),
        failures.join("\n---\n")
    );
}
