//! **#750 paged-tier tail-cost bench, emit half.** The paged software page-check's default flip is
//! gated on measured numbers ("~1.5–3×+ on the random-access tail" was an estimate, not data). This
//! example emits one kernel module — four memory-hot leaves, sequential/random × load/store, at a
//! cache-resident (8 MiB) and a RAM-bound (64 MiB) working set — through three tier-up entries:
//! [`compile_module_tierup`] (plain, mask-only), [`compile_module_tierup_paged`] (per-access page
//! check), and [`compile_module_tierup_nullguard`] (the dedicated NULL-page compare the trap-on-NULL
//! design weighs against full paged mode) — and times the **native bytecode interpreter** on the
//! same kernels as the baseline row. The V8 half (`browser/bench_paged.mjs`) instantiates the wasm
//! files and times the emitted `f{k}` calls — each variant vs plain on the same engine isolates
//! that check's tax; the interp row contextualizes what either buys a guest that today runs
//! whole-interpreter. The kernels offset every access one guard page up so all variants run
//! trap-free.
//!
//!   cargo run --release -p temen-wasm-jit --example paged_bench_emit
//!   node browser/bench_paged.mjs target/paged_bench
//!
//! Writes `target/paged_bench/{plain,paged,nullguard}.wasm` + `manifest.json` (kernel table, layout
//! params, interp ns/op + checksums — the parity anchors the JS side asserts against).

use std::time::Instant;

use temen_interp::{bytecode, Value};

/// The run's software page size (4 KiB — the common host page; baked into the emitted shift).
const PAGE_LOG2: u8 = 12;
/// The NULL-page guard for the third variant (`compile_module_tierup_nullguard`): one 4 KiB page.
/// The kernels offset every access by 512 entries (4096 B) so all three variants run trap-free.
const NULL_GUARD: u64 = 4096;
/// Window: 128 MiB (`memory 27`) — big enough for a RAM-bound (64 MiB) working set.
const WIN_LOG2: u8 = 27;

/// Four memory-hot leaves over an i64 buffer whose size is the **`mask` argument** (`(n, mask)` →
/// checksum), so one module measures both a cache-resident and a RAM-bound working set. Every value
/// is threaded through block params (block-local IR). `rand_*` step a 64-bit LCG and index with
/// bits 33..53 — the dependent-load, cache-hostile tail the #750 estimate is about; `seq_*` are the
/// prefetch-friendly floor.
const KERNELS_SRC: &str = r#"memory 27
func (i64, i64) -> (i64) {
block 0 (vn: i64, vmask: i64) {
  vz = i64.const 0
  br 1(vz, vz, vn, vmask)
}
block 1 (vi: i64, vacc: i64, vn1: i64, vm1: i64) {
  vc = i64.lt_u vi vn1
  br_if vc 2(vi, vacc, vn1, vm1) 3(vacc)
}
block 2 (vi2: i64, vacc2: i64, vn2: i64, vm2: i64) {
  vidx = i64.and vi2 vm2
  v512 = i64.const 512
  vidx2 = i64.add vidx v512
  v8 = i64.const 8
  vaddr = i64.mul vidx2 v8
  vv = i64.load vaddr
  vacc3 = i64.add vacc2 vv
  vone = i64.const 1
  vi3 = i64.add vi2 vone
  br 1(vi3, vacc3, vn2, vm2)
}
block 3 (vres: i64) {
  return vres
  }
}
func (i64, i64) -> (i64) {
block 0 (vn: i64, vmask: i64) {
  vz = i64.const 0
  vone = i64.const 1
  vseed = i64.or vn vone
  br 1(vz, vz, vseed, vn, vmask)
}
block 1 (vi: i64, vacc: i64, vx: i64, vn1: i64, vm1: i64) {
  vc = i64.lt_u vi vn1
  br_if vc 2(vi, vacc, vx, vn1, vm1) 3(vacc)
}
block 2 (vi2: i64, vacc2: i64, vx2: i64, vn2: i64, vm2: i64) {
  va = i64.const 6364136223846793005
  vcadd = i64.const 1442695040888963407
  vxm = i64.mul vx2 va
  vx3 = i64.add vxm vcadd
  vsh = i64.const 33
  vhi = i64.shr_u vx3 vsh
  vidx = i64.and vhi vm2
  v512 = i64.const 512
  vidx2 = i64.add vidx v512
  v8 = i64.const 8
  vaddr = i64.mul vidx2 v8
  vv = i64.load vaddr
  vacc3 = i64.add vacc2 vv
  vone = i64.const 1
  vi3 = i64.add vi2 vone
  br 1(vi3, vacc3, vx3, vn2, vm2)
}
block 3 (vres: i64) {
  return vres
  }
}
func (i64, i64) -> (i64) {
block 0 (vn: i64, vmask: i64) {
  vz = i64.const 0
  br 1(vz, vz, vn, vmask)
}
block 1 (vi: i64, vacc: i64, vn1: i64, vm1: i64) {
  vc = i64.lt_u vi vn1
  br_if vc 2(vi, vacc, vn1, vm1) 3(vacc)
}
block 2 (vi2: i64, vacc2: i64, vn2: i64, vm2: i64) {
  vidx = i64.and vi2 vm2
  v512 = i64.const 512
  vidx2 = i64.add vidx v512
  v8 = i64.const 8
  vaddr = i64.mul vidx2 v8
  i64.store vaddr vi2
  vacc3 = i64.add vacc2 vidx
  vone = i64.const 1
  vi3 = i64.add vi2 vone
  br 1(vi3, vacc3, vn2, vm2)
}
block 3 (vres: i64) {
  return vres
  }
}
func (i64, i64) -> (i64) {
block 0 (vn: i64, vmask: i64) {
  vz = i64.const 0
  vone = i64.const 1
  vseed = i64.or vn vone
  br 1(vz, vz, vseed, vn, vmask)
}
block 1 (vi: i64, vacc: i64, vx: i64, vn1: i64, vm1: i64) {
  vc = i64.lt_u vi vn1
  br_if vc 2(vi, vacc, vx, vn1, vm1) 3(vacc)
}
block 2 (vi2: i64, vacc2: i64, vx2: i64, vn2: i64, vm2: i64) {
  va = i64.const 6364136223846793005
  vcadd = i64.const 1442695040888963407
  vxm = i64.mul vx2 va
  vx3 = i64.add vxm vcadd
  vsh = i64.const 33
  vhi = i64.shr_u vx3 vsh
  vidx = i64.and vhi vm2
  v512 = i64.const 512
  vidx2 = i64.add vidx v512
  v8 = i64.const 8
  vaddr = i64.mul vidx2 v8
  i64.store vaddr vx3
  vacc3 = i64.add vacc2 vidx
  vone = i64.const 1
  vi3 = i64.add vi2 vone
  br 1(vi3, vacc3, vx3, vn2, vm2)
}
block 3 (vres: i64) {
  return vres
  }
}
"#;

const KERNELS: [(&str, u32); 4] = [
    ("seq_load", 0),
    ("rand_load", 1),
    ("seq_store", 2),
    ("rand_store", 3),
];

/// Interp iteration count — the interpreter is ~40–75× slower, so a smaller `n` keeps the baseline
/// row quick; ns/op normalizes. (The JS side uses its own, larger `n` for the emitted rows.)
const INTERP_ITERS: i64 = 2_000_000;

/// Working-set masks: 8 MiB (cache-resident on most hosts — the *relative*-tax worst case, since
/// the check cost is a constant per access) and 64 MiB (RAM-bound random tail — misses hide it).
const MASKS: [i64; 2] = [(1 << 20) - 1, (1 << 23) - 1];

fn main() {
    let m = temen_text::parse_module(KERNELS_SRC).expect("parse kernels");
    temen_verify::verify_module(&m).expect("verify kernels");

    let (plain, plain_emitted) =
        temen_wasm_jit::compile_module_tierup(&m, false).expect("plain tier-up emits");
    let (paged, paged_emitted) =
        temen_wasm_jit::compile_module_tierup_paged(&m, false, PAGE_LOG2).expect("paged emits");
    let (nullg, nullg_emitted) =
        temen_wasm_jit::compile_module_tierup_nullguard(&m, false, NULL_GUARD)
            .expect("nullguard emits");
    assert!(
        plain_emitted.iter().all(|&b| b)
            && paged_emitted.iter().all(|&b| b)
            && nullg_emitted.iter().all(|&b| b),
        "all four kernels must be tier-up eligible on every entry"
    );

    let dir = std::path::Path::new("target/paged_bench");
    std::fs::create_dir_all(dir).expect("mkdir");
    std::fs::write(dir.join("plain.wasm"), &plain).expect("write plain.wasm");
    std::fs::write(dir.join("paged.wasm"), &paged).expect("write paged.wasm");
    std::fs::write(dir.join("nullguard.wasm"), &nullg).expect("write nullguard.wasm");

    // Native bytecode-interpreter baseline + the checksum parity anchors. Each kernel runs in a
    // fresh zeroed window (the module's own `memory 27`), matching the JS side's zeroed memory, so
    // checksums must agree across interp / plain / paged for the same `(n, mask)`.
    let mut rows = String::new();
    let total = KERNELS.len() * MASKS.len();
    let mut row = 0;
    for (name, func) in KERNELS.iter() {
        for mask in MASKS {
            row += 1;
            let mut fuel = u64::MAX;
            let t = Instant::now();
            let vals = bytecode::compile_and_run(
                &m,
                *func,
                &[Value::I64(INTERP_ITERS), Value::I64(mask)],
                &mut fuel,
            )
            .expect("kernel in bytecode subset")
            .expect("kernel runs clean");
            let ns_per_op = t.elapsed().as_nanos() as f64 / INTERP_ITERS as f64;
            let Some(Value::I64(sum)) = vals.first() else {
                panic!("kernel result");
            };
            let ws_mib = ((mask + 1) * 8) >> 20;
            println!("interp  {name:<11} {ws_mib:>3} MiB {ns_per_op:>7.2} ns/op   checksum {sum}");
            rows.push_str(&format!(
                "    {{\"name\":\"{name}\",\"func\":{func},\"mask\":{mask},\"ws_mib\":{ws_mib},\"interp_ns_per_op\":{ns_per_op:.3},\"interp_iters\":{INTERP_ITERS},\"interp_checksum\":\"{sum}\"}}{}\n",
                if row == total { "" } else { "," }
            ));
        }
    }
    let manifest = format!(
        "{{\n  \"win_log2\": {WIN_LOG2},\n  \"page_log2\": {PAGE_LOG2},\n  \"null_guard\": {NULL_GUARD},\n  \"kernels\": [\n{rows}  ]\n}}\n"
    );
    std::fs::write(dir.join("manifest.json"), manifest).expect("write manifest");
    println!(
        "wrote {} (plain {} B, paged {} B, nullguard {} B)",
        dir.display(),
        plain.len(),
        paged.len(),
        nullg.len()
    );
}
