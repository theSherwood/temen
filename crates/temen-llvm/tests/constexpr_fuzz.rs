//! #850 — a deterministic **differential fuzz** over the on-ramp's constexpr-**operand** surface.
//!
//! Motivation: bringing real Rust `std` through the on-ramp surfaced a trickle of `Unsupported` gaps,
//! each a valid LLVM constexpr the on-ramp folded as a *global initializer* but not as an *operand*
//! (the #849 bug was `add(ptrtoint(@g), k)` used directly as an operand). This generates random
//! constexpr trees over the relocation-arithmetic surface — integer literals, global references, and
//! `add`/`sub`/`mul`/`ptrtoint`/`inttoptr`/`getelementptr` — emits each as the operand of a `ret`,
//! translates + runs it on the interpreter (the oracle, DESIGN §18), and checks the result against a
//! reference evaluator computed over the on-ramp-assigned global addresses. A seed that fails to
//! translate/verify or mismatches is a reproducing counter-example.
//!
//! No external deps: a fixed-seed LCG keeps it reproducible and CI-cheap (each case is a tiny module).

use temen_interp::Value;

/// A tiny reproducible LCG (Numerical Recipes constants) — no `rand`/`Date` dependency.
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
}

/// Number of `[16 x i8]` globals the generated exprs may reference (`@g0`..`@g{N-1}`).
const NGLOBALS: usize = 4;
const GLOBAL_LEN: i64 = 16;

/// An i64-typed constexpr node.
enum E {
    Int(i64),
    PtrToInt(Box<P>),
    Add(Box<E>, Box<E>),
    Sub(Box<E>, Box<E>),
    Mul(Box<E>, Box<E>),
    /// `zext (i32 trunc (i64 <inner> to i32) to i64)` — a self-contained i64→i32→i64 round-trip that
    /// exercises the `trunc` + `zext` constexpr folds; the low 32 bits, zero-extended.
    TruncZext(Box<E>),
    /// `sext (i32 trunc (i64 <inner> to i32) to i64)` — the low 32 bits, sign-extended.
    TruncSext(Box<E>),
    And(Box<E>, Box<E>),
    Or(Box<E>, Box<E>),
    Xor(Box<E>, Box<E>),
    /// `shl (i64 <inner>, i64 <shift>)` with `shift` in `0..64`.
    Shl(Box<E>, u32),
    /// `lshr (i64 <inner>, i64 <shift>)` — logical (zero-fill) right shift at the full i64 width,
    /// `shift` in `0..64`.
    LShr(Box<E>, u32),
    /// `ashr (i64 <inner>, i64 <shift>)` — arithmetic (sign-fill) right shift at i64, `shift` `0..64`.
    AShr(Box<E>, u32),
    /// `zext (i32 lshr (i32 trunc (<inner> to i32), i32 <shift>) to i64)` — the **width-sensitive**
    /// logical right shift: the shift happens at i32, so the zero-fill and the low-32-bit view are what
    /// distinguish `lshr` from the width-agnostic bitwise ops. `shift` in `0..32`.
    LShr32(Box<E>, u32),
    /// `sext (i32 ashr (i32 trunc (<inner> to i32), i32 <shift>) to i64)` — the width-sensitive
    /// arithmetic right shift: the sign fill comes from bit 31 (the i32 sign). `shift` in `0..32`.
    AShr32(Box<E>, u32),
}

/// A pointer-typed constexpr node.
enum P {
    Global(usize),
    IntToPtr(Box<E>),
    /// `getelementptr (i8, ptr <base>, i64 <idx>)` — an i8 element, so the byte offset is `idx`
    /// (bounded to `[0, GLOBAL_LEN)` so a run that dereferenced it would stay in-window, though these
    /// exprs are only address-computed, never loaded).
    Gep(Box<P>, i64),
}

fn gen_e(rng: &mut Rng, depth: u32) -> E {
    if depth == 0 || rng.below(2) == 0 {
        // Small signed literal, or a pointer's address.
        if rng.below(2) == 0 {
            E::Int((rng.next_u32() as i64 % 4096) - 2048)
        } else {
            E::PtrToInt(Box::new(gen_p(rng, depth.saturating_sub(1))))
        }
    } else {
        let pair = |rng: &mut Rng, d: u32| (Box::new(gen_e(rng, d)), Box::new(gen_e(rng, d)));
        match rng.below(14) {
            0 => {
                let (a, b) = pair(rng, depth - 1);
                E::Add(a, b)
            }
            1 => {
                let (a, b) = pair(rng, depth - 1);
                E::Sub(a, b)
            }
            2 => {
                let (a, b) = pair(rng, depth - 1);
                E::Mul(a, b)
            }
            3 => E::TruncZext(Box::new(gen_e(rng, depth - 1))),
            4 => E::TruncSext(Box::new(gen_e(rng, depth - 1))),
            5 => {
                let (a, b) = pair(rng, depth - 1);
                E::And(a, b)
            }
            6 => {
                let (a, b) = pair(rng, depth - 1);
                E::Or(a, b)
            }
            7 => {
                let (a, b) = pair(rng, depth - 1);
                E::Xor(a, b)
            }
            8 => E::Shl(Box::new(gen_e(rng, depth - 1)), rng.below(64)),
            9 => E::LShr(Box::new(gen_e(rng, depth - 1)), rng.below(64)),
            10 => E::AShr(Box::new(gen_e(rng, depth - 1)), rng.below(64)),
            11 => E::LShr32(Box::new(gen_e(rng, depth - 1)), rng.below(32)),
            12 => E::AShr32(Box::new(gen_e(rng, depth - 1)), rng.below(32)),
            _ => E::PtrToInt(Box::new(gen_p(rng, depth - 1))),
        }
    }
}

fn gen_p(rng: &mut Rng, depth: u32) -> P {
    if depth == 0 || rng.below(2) == 0 {
        P::Global(rng.below(NGLOBALS as u32) as usize)
    } else {
        match rng.below(3) {
            0 => P::IntToPtr(Box::new(gen_e(rng, depth - 1))),
            1 => P::Gep(
                Box::new(gen_p(rng, depth - 1)),
                (rng.next_u32() as i64) % GLOBAL_LEN,
            ),
            _ => P::Global(rng.below(NGLOBALS as u32) as usize),
        }
    }
}

fn render_e(e: &E) -> String {
    match e {
        E::Int(n) => format!("i64 {n}"),
        E::PtrToInt(p) => format!("i64 ptrtoint (ptr {} to i64)", render_p_inner(p)),
        E::Add(a, b) => format!("i64 add ({}, {})", render_e(a), render_e(b)),
        E::Sub(a, b) => format!("i64 sub ({}, {})", render_e(a), render_e(b)),
        E::Mul(a, b) => format!("i64 mul ({}, {})", render_e(a), render_e(b)),
        E::TruncZext(e) => format!("i64 zext (i32 trunc ({} to i32) to i64)", render_e(e)),
        E::TruncSext(e) => format!("i64 sext (i32 trunc ({} to i32) to i64)", render_e(e)),
        E::And(a, b) => format!("i64 and ({}, {})", render_e(a), render_e(b)),
        E::Or(a, b) => format!("i64 or ({}, {})", render_e(a), render_e(b)),
        E::Xor(a, b) => format!("i64 xor ({}, {})", render_e(a), render_e(b)),
        E::Shl(a, s) => format!("i64 shl ({}, i64 {s})", render_e(a)),
        E::LShr(a, s) => format!("i64 lshr ({}, i64 {s})", render_e(a)),
        E::AShr(a, s) => format!("i64 ashr ({}, i64 {s})", render_e(a)),
        E::LShr32(a, s) => format!(
            "i64 zext (i32 lshr (i32 trunc ({} to i32), i32 {s}) to i64)",
            render_e(a)
        ),
        E::AShr32(a, s) => format!(
            "i64 sext (i32 ashr (i32 trunc ({} to i32), i32 {s}) to i64)",
            render_e(a)
        ),
    }
}

/// Render a pointer node as a bare `ptr`-typed constant (no leading type keyword) — the form nested
/// inside `ptrtoint (ptr <here> to i64)` / `getelementptr (i8, ptr <here>, …)`.
fn render_p_inner(p: &P) -> String {
    match p {
        P::Global(i) => format!("@g{i}"),
        P::IntToPtr(e) => format!("inttoptr ({} to ptr)", render_e(e)),
        P::Gep(base, idx) => {
            format!(
                "getelementptr (i8, ptr {}, i64 {idx})",
                render_p_inner(base)
            )
        }
    }
}

fn eval_e(e: &E, addrs: &[i64]) -> i64 {
    match e {
        E::Int(n) => *n,
        E::PtrToInt(p) => eval_p(p, addrs),
        E::Add(a, b) => eval_e(a, addrs).wrapping_add(eval_e(b, addrs)),
        E::Sub(a, b) => eval_e(a, addrs).wrapping_sub(eval_e(b, addrs)),
        E::Mul(a, b) => eval_e(a, addrs).wrapping_mul(eval_e(b, addrs)),
        // low 32 bits, zero- / sign-extended (matching the on-ramp's trunc+zext / trunc+sext folds).
        E::TruncZext(e) => (eval_e(e, addrs) as u32) as u64 as i64,
        E::TruncSext(e) => (eval_e(e, addrs) as i32) as i64,
        E::And(a, b) => eval_e(a, addrs) & eval_e(b, addrs),
        E::Or(a, b) => eval_e(a, addrs) | eval_e(b, addrs),
        E::Xor(a, b) => eval_e(a, addrs) ^ eval_e(b, addrs),
        E::Shl(a, s) => eval_e(a, addrs).wrapping_shl(*s),
        // Full-width right shifts: logical reads the value as unsigned, arithmetic keeps the i64 sign.
        E::LShr(a, s) => (eval_e(a, addrs) as u64 >> *s) as i64,
        E::AShr(a, s) => eval_e(a, addrs) >> *s,
        // Width-sensitive: shift the low 32 bits, then zero- / sign-extend from bit 31 back to i64.
        E::LShr32(a, s) => ((eval_e(a, addrs) as u32) >> *s) as u64 as i64,
        E::AShr32(a, s) => ((eval_e(a, addrs) as i32) >> *s) as i64,
    }
}

fn eval_p(p: &P, addrs: &[i64]) -> i64 {
    match p {
        P::Global(i) => addrs[*i],
        P::IntToPtr(e) => eval_e(e, addrs),
        P::Gep(base, idx) => eval_p(base, addrs).wrapping_add(*idx),
    }
}

#[test]
fn constexpr_operand_differential_fuzz() {
    let mut failures: Vec<String> = Vec::new();
    for seed in 0..300u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1));
        let e = gen_e(&mut rng, 4);
        let body = render_e(&e);
        let globals: String = (0..NGLOBALS)
            .map(|i| format!("@g{i} = internal global [16 x i8] zeroinitializer, align 16\n"))
            .collect();
        let ir = format!(
            "target datalayout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128\"\n\
             target triple = \"x86_64-unknown-none\"\n\
             {globals}define i64 @f() {{\nentry:\n  ret {body}\n}}\n"
        );
        let t = match temen_llvm::translate_ll_str(&ir) {
            Ok(t) => t,
            Err(err) => {
                failures.push(format!(
                    "seed {seed}: translate failed: {err:?}\n  body: ret {body}"
                ));
                continue;
            }
        };
        if let Err(err) = temen_verify::verify_module(&t.module) {
            failures.push(format!(
                "seed {seed}: verify failed: {err:?}\n  body: ret {body}"
            ));
            continue;
        }
        let addrs: Vec<i64> = (0..NGLOBALS)
            .map(|i| {
                t.data_symbols
                    .iter()
                    .find(|s| s.name == format!("g{i}"))
                    .map(|s| s.addr as i64)
                    .unwrap_or(0)
            })
            .collect();
        let expect = eval_e(&e, &addrs);
        let mut fuel = 1_000_000u64;
        let f = t
            .exports
            .iter()
            .find(|(n, _)| n == "f")
            .map(|x| x.1)
            .unwrap();
        match temen_interp::run(&t.module, f, &[Value::I64(t.entry_sp as i64)], &mut fuel) {
            Ok(v) => {
                let got = match v[0] {
                    Value::I64(x) => x,
                    other => {
                        failures.push(format!("seed {seed}: non-i64 result {other:?}"));
                        continue;
                    }
                };
                if got != expect {
                    failures.push(format!(
                        "seed {seed}: mismatch got {got} want {expect}\n  body: ret {body}"
                    ));
                }
            }
            Err(err) => {
                failures.push(format!(
                    "seed {seed}: interp trapped: {err:?}\n  body: ret {body}"
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} constexpr-operand fuzz case(s) failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
