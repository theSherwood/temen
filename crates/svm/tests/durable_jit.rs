//! Stable (no-nightly) driver for the cross-backend freeze/thaw property — the CI-run
//! counterpart of the libFuzzer target `fuzz/fuzz_targets/durable_jit.rs`, sitting beside
//! the interpreter-vs-JIT differential (`jit_fuzz.rs`). A failure here is a backend
//! divergence in the durable transform's emitted IR.

#[path = "support/durjit.rs"]
mod durjit;

#[test]
fn freeze_thaw_cross_backend_over_generated_modules() {
    // Each module JIT-compiles three times and commits a guest window each time; the libFuzzer
    // target `durable_jit` does the heavy run, so keep this a modest smoke. The low count also
    // bounds the cumulative JIT commit on memory-tight Windows CI (os error 1455 / commit limit).
    for seed in 0..64u64 {
        let mut g = durjit::durgen::Gen::from_seed(seed);
        durjit::fuzz_one_xbackend(&mut g);
    }
}

// R8, deterministic: a `call_indirect` to a may-suspend target must freeze byte-identically and thaw
// across the interp/JIT boundary, exactly like a direct propagated `call` — not left to fuzz-seed
// luck. `A →(call_indirect slot 1)→ B(leaf cap.call)`; the natural table maps slot 1 → func 1.
#[test]
fn indirect_call_freeze_thaw_cross_backend() {
    let src = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i32.const 1
  v2 = call_indirect (i32) -> (i64) v1 (v0)
  v3 = i64.const 1000
  v4 = i64.add v2 v3
  return v4
  }
}
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i32.const 0
  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  v3 = i64.const 100
  v4 = i64.add v2 v3
  return v4
  }
}
"#;
    let mut m = svm_text::parse_module(src).expect("parse indirect module");
    m.memory = Some(svm_ir::Memory { size_log2: 18 });
    durjit::check_xbackend(&m, 42);
}

#[test]
fn loop_freeze_thaw_cross_backend_over_generated_modules() {
    // Phase-4 Slice A: the loop-header back-edge poll is ordinary IR, so the JIT must freeze a
    // poll-free-loop module byte-identically to the interpreter and thaw across the backend boundary.
    for seed in 0..64u64 {
        let mut g = durjit::durgen::Gen::from_seed(seed);
        durjit::fuzz_loop_one_xbackend(&mut g);
    }
}

// Not on Windows: this fuzz compiles the JIT twice per seed (freeze + thaw), and running alongside
// `freeze_thaw_cross_backend_over_generated_modules` in the same test binary pushes the process's
// cumulative JIT allocations far enough from the statically-linked runtime thunks that a PC-relative
// relocation overflows `i32` (>2 GiB) inside cranelift-jit (`compiled_blob.rs` `perform_relocations`
// panics with `TryFromIntError`) — an address-space-drift limitation of the Windows JIT, not a
// backend divergence, and partly ASLR-nondeterministic. Windows keeps full recycled coverage via the
// hand-written `durable_fibers_jit::jit_and_interp_freeze_a_recycled_fiber_identically_and_thaw_on_the_jit`
// and the 400-seed interpreter fuzz `durable_fuzz::recycled_fiber_freeze_thaw_equivalence_over_generated_modules`
// (no JIT, so no drift). The libFuzzer target `durable_recycle_jit` does the heavy run elsewhere.
#[cfg(not(windows))]
#[test]
fn recycled_fiber_freeze_thaw_cross_backend_over_generated_modules() {
    // Recycling step 4, cross-backend: churn modules recycle a slot (generation 1..=3), park the
    // real fiber there, and are frozen *mid-run* via `arm_freeze_after`. The interp and JIT must
    // armed-freeze the recycled fiber to a byte-identical reserve + §12 artifact, and the
    // interp-frozen artifact must thaw on the JIT to the uninterrupted result.
    for seed in 0..64u64 {
        let mut g = durjit::durgen::Gen::from_seed(seed);
        durjit::fuzz_recycle_fiber_one_xbackend(&mut g);
    }
}
