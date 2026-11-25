//! #3078 — the kernel lane a fiber suite runs beside its engines: the program on the Cranelift JIT's
//! kernel route (its vCPUs as executor 2's tasks), on `fuel`: each suite runs it on each of
//! [`FUELS`], metered (`DEFAULT_FUEL`, the product's default, since #3080) and unmetered, and checks
//! each run against its own acceptance. Two runs are never compared with each other: a program may
//! have more than one right answer (a race the oracle accepts either way). The exact fuel a fibered
//! run charges is `jit_fuel`'s sweeps'. A module that spawns no thread
//! keeps its single-vCPU path (`fiber only`), so its function 0 runs inside one `thread.spawn`ed vCPU
//! instead: a wrapper entry spawns a thread that calls it, and joins it. Either way the run is
//! asserted to have taken the kernel route. One helper, so a suite's lane is a call, not a copy of
//! the wrapping (INVARIANTS #15).
#![allow(dead_code)] // each suite uses its own subset

use temen_jit::JitOutcome;
use temen_run::jit_kernel;

/// The two fuels a suite runs its lane on: metered (the product's default) and unmetered.
pub const FUELS: [u64; 2] = [temen_interp::DEFAULT_FUEL, u64::MAX];

/// Whether some function of `m` spawns a thread.
fn spawns(m: &temen_ir::Module) -> bool {
    m.funcs.iter().any(temen_ir::Func::uses_threads)
}

/// `src` (function 0 an `() -> (i64)` entry when the module spawns no thread) on the kernel route
/// with `fuel`, asserted to have run there; with the module it ran (wrapped or not) and its entry,
/// so the caller can run the oracle over the same program.
pub fn on_kernel(src: &str, fuel: u64) -> (JitOutcome, temen_ir::Module, u32) {
    let m = temen_text::parse_module(src).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}\n{src}"));
    let (m, entry, vcpus) = match spawns(&m) {
        true => (m, 0, 1),
        false => {
            let e = &m.funcs[0];
            assert!(
                e.params.is_empty() && e.results == [temen_ir::ValType::I64],
                "a module that spawns no thread runs its `() -> (i64)` entry in one\n{src}"
            );
            let n = m.funcs.len();
            let wrapped = format!(
                "{src}
func (i64, i64) -> (i64) {{
block 0 (va: i64, vb: i64) {{
  vr = call 0()
  return vr
  }}
}}
func () -> (i64) {{
block 0 () {{
  vz = i64.const 0
  vt = thread.spawn {n} vz vz
  vs, vv = thread.join vt
  return vv
  }}
}}
"
            );
            let w = temen_text::parse_module(&wrapped)
                .unwrap_or_else(|e| panic!("parse: {e:?}\n{wrapped}"));
            temen_verify::verify_module(&w).unwrap_or_else(|e| panic!("verify: {e:?}\n{wrapped}"));
            (w, (n + 1) as u32, 2)
        }
    };
    let o = jit_kernel::expect_kernel(
        &m,
        entry,
        &[],
        None,
        fuel,
        jit_kernel::Expect::Kernel { vcpus },
    )
    .expect("the kernel route admits it");
    (o, m, entry)
}

/// [`on_kernel`]'s one `i64` result, or a panic naming what it did instead.
pub fn i64_on_kernel(src: &str, fuel: u64) -> i64 {
    match on_kernel(src, fuel).0 {
        JitOutcome::Returned(v) if v.len() == 1 => v[0],
        o => panic!("the kernel route did not return one i64: {o:?}\n{src}"),
    }
}
