//! #2481 — end-to-end check of the shadow-mode machine-code validator through the real compile path.
//!
//! The temen-jit unit tests cross-check the decoder against iced-x86 on captured bytes; this drives
//! the production hook (`compile_verified` → `validate::shadow_check`, gated by `TEMEN_JIT_VALIDATE`)
//! over a spread of modules and asserts the validator finds nothing to disagree with on clean
//! Cranelift output. A disagreement would mean either a decoder gap (teach it the encoding) or a
//! real policy violation; in shadow mode it is logged and counted, never fatal, so this test is how
//! slice 1 fails loudly in CI if the count moves off zero.
//!
//! This is its own test binary and sets the environment before the first compile, so the cached
//! `validate::enabled()` observes it (the flag also makes `jit_isa` record the block layout the
//! validator checks branch targets against).

#![cfg(all(unix, target_arch = "x86_64"))]

use temen_jit::{compile_and_run, jit_validate_disagreements};
use temen_text::parse_module;

/// A spread of modules exercising the instruction families Cranelift emits: integer ALU, the
/// memory-access confinement lowering (cmp/cmov/branch), loops (rel8/rel32 jmp/jcc + block starts),
/// `br_table` (indirect jmp + a jump-table island), `select`/extends (cmov/setcc/movsx/movzx), and
/// floats (SSE + a constant pool).
const MODULES: &[&str] = &[
    "func (i64, i64) -> (i64) {\nblock 0 (v0: i64, v1: i64) {\n  v2 = i64.add v0 v1\n  \
     v3 = i64.mul v2 v0\n  v4 = i64.sub v3 v1\n  return v4\n  }\n}\n",
    "memory 16\nfunc (i64) -> (i64) {\nblock 0 (v0: i64) {\n  v1 = i64.load v0\n  \
     v2 = i64.const 8\n  v3 = i64.add v0 v2\n  i64.store v3 v1\n  return v1\n  }\n}\n",
    "func (i64) -> (i64) {\nblock 0 (v0: i64) {\n  v1 = i64.const 0\n  br 1(v0, v1)\n  }\n\
     block 1 (v2: i64, v3: i64) {\n  v4 = i64.const 0\n  v5 = i64.eq v2 v4\n  \
     br_if v5 3(v3) 2(v2, v3)\n  }\nblock 2 (v6: i64, v7: i64) {\n  v8 = i64.const 1\n  \
     v9 = i64.sub v6 v8\n  v10 = i64.add v7 v6\n  br 1(v9, v10)\n  }\n\
     block 3 (v11: i64) {\n  return v11\n  }\n}\n",
    "func (i32) -> (i32) {\nblock 0 (v0: i32) {\n  br_table v0 [1(), 2(), 3()] 4()\n  }\n\
     block 1 () {\n  v1 = i32.const 10\n  return v1\n  }\nblock 2 () {\n  v2 = i32.const 20\n  \
     return v2\n  }\nblock 3 () {\n  v3 = i32.const 30\n  return v3\n  }\n\
     block 4 () {\n  v4 = i32.const 99\n  return v4\n  }\n}\n",
    "func (i32, i32) -> (i64) {\nblock 0 (v0: i32, v1: i32) {\n  v2 = i32.eq v0 v1\n  \
     v3 = i32.const 5\n  v4 = i32.const 9\n  v5 = select v2 v3 v4\n  v6 = i64.extend_i32_s v5\n  \
     v7 = i64.extend_i32_u v0\n  v8 = i64.add v6 v7\n  return v8\n  }\n}\n",
    "func (f64, f64) -> (f64) {\nblock 0 (v0: f64, v1: f64) {\n  v2 = f64.add v0 v1\n  \
     v3 = f64.mul v2 v0\n  v4 = f64.const 2.5\n  v5 = f64.div v3 v4\n  return v5\n  }\n}\n",
];

#[test]
fn shadow_validator_is_clean_on_cranelift_output() {
    // SAFETY: set before the first compile in this (dedicated) test binary, so the cached
    // `validate::enabled()` reads it. Single-threaded reach here (no compile ran yet).
    unsafe { std::env::set_var("TEMEN_JIT_VALIDATE", "1") };

    for (i, src) in MODULES.iter().enumerate() {
        let m = parse_module(src).unwrap_or_else(|e| panic!("parse module {i}: {e:?}"));
        // A no-host run: any result/trap is fine — we only care that the compile ran the validator.
        let _ = compile_and_run(&m, 0, &[0]);
    }

    let n = jit_validate_disagreements();
    assert_eq!(
        n, 0,
        "machine-code validator reported {n} disagreement(s) on clean Cranelift output \
         (see the jit-validate log lines above): a decoder gap or a real policy violation"
    );
}
