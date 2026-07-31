//! End-to-end: a **real nimony program runs on SVM** (NIM.md §3b, Path B). W1 lowers real seq code
//! to verified svm-ir with its stdlib ops as named imports; this binds those imports to a tiny SVM
//! **runtime shim** (linked in via `svm_ir::link`, staying in the pure-IR/both-engines model — no
//! host capabilities) and *runs* the result. The shim is the Phase-B "host runtime" written as SVM
//! functions: the pure seq/openArray ops plus a bump allocator over the window.
//!
//! Layouts (as nimony emits them): `seq[int]` = `{len@0, data*@8}`; `openArray` = `{a*@0, len@8}`.
//! The allocator's bump pointer lives at window offset 8 (the harness seeds it to the heap start).
//!
//! Runtime functions, in this fixed order (the `classify` map binds each real import name to one):
//!   0 toOpenArray(dst, seq)   4 =wasMoved(seq)
//!   1 len(openArray)          5 =destroy(seq)      (no-op: the bump allocator never frees)
//!   2 [](openArray, idx)      6 newSeqUninit(sret, n)
//!   3 inc(&int)               7 add(&seq, val)     (realloc-each-append; simple + correct)

use svm_interp::Value;
use svm_ir::{LinkUnit, Module};

const RUNTIME: &str = "\
memory 16

func (i64, i64) -> () {
block 0 (v0: i64, v1: i64) {
  v2 = i64.load v1 offset=8
  i64.store v0 v2 offset=0
  v3 = i64.load v1 offset=0
  i64.store v0 v3 offset=8
  return
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.load v0 offset=8
  return v1
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.load v0 offset=0
  v3 = i64.const 8
  v4 = i64.mul v1 v3
  v5 = i64.add v2 v4
  return v5
  }
}
func (i64) -> () {
block 0 (v0: i64) {
  v1 = i64.load v0
  v2 = i64.const 1
  v3 = i64.add v1 v2
  i64.store v0 v3
  return
  }
}
func (i64) -> () {
block 0 (v0: i64) {
  v1 = i64.const 0
  i64.store v0 v1 offset=0
  i64.store v0 v1 offset=8
  return
  }
}
func (i64) -> () {
block 0 (v0: i64) {
  return
  }
}
func (i64, i64) -> () {
block 0 (v0: i64, v1: i64) {
  i64.store v0 v1 offset=0
  v2 = i64.const 8
  v3 = i64.mul v1 v2
  v4 = i64.const 8
  v5 = i64.load v4
  v6 = i64.add v5 v3
  i64.store v4 v6
  i64.store v0 v5 offset=8
  return
  }
}
func (i64, i64) -> () {
block 0 (v0: i64, v1: i64) {
  v2 = i64.load v0 offset=0
  v3 = i64.load v0 offset=8
  v4 = i64.const 1
  v5 = i64.add v2 v4
  v6 = i64.const 8
  v7 = i64.mul v5 v6
  v8 = i64.const 8
  v9 = i64.load v8
  v10 = i64.add v9 v7
  i64.store v8 v10
  v11 = i64.mul v2 v6
  mem.copy v9 v3 v11
  v12 = i64.add v9 v11
  i64.store v12 v1
  i64.store v0 v9 offset=8
  i64.store v0 v5 offset=0
  return
  }
}";

/// Map a real nimony runtime import name to its runtime-shim function index.
fn classify(name: &str) -> u32 {
    if name.contains("toOpenArray") {
        0
    } else if name.starts_with("len") {
        1
    } else if name.contains("5B") {
        2 // the `[]` operator (`\5B\5D…`)
    } else if name.starts_with("inc") {
        3
    } else if name.to_lowercase().contains("wasmoved") {
        4
    } else if name.contains("destroy") {
        5
    } else if name.contains("newSeq") {
        6
    } else if name.starts_with("add") {
        7
    } else {
        panic!("unbound runtime import: {name}");
    }
}

/// Link a translated user module against the runtime shim, binding each named import to a shim
/// function; `user_exports` are the user procs a driver may import, and `extra` units (the driver)
/// link too. Returns the verified, import-free module.
fn link_runtime(user: Module, user_exports: Vec<(String, u32)>, extra: Vec<LinkUnit>) -> Module {
    let rt = svm_text::parse_module(RUNTIME).expect("runtime shim parses");
    let rt_exports: Vec<(String, u32)> = user
        .imports
        .iter()
        .map(|imp| (imp.name.clone(), classify(&imp.name)))
        .collect();
    let mut units = vec![
        LinkUnit {
            module: user,
            exports: user_exports,
            ..Default::default()
        },
        LinkUnit {
            module: rt,
            exports: rt_exports,
            ..Default::default()
        },
    ];
    units.extend(extra);
    let linked = svm_ir::link(&units).unwrap_or_else(|e| panic!("link: {e:?}"));
    svm_verify::verify_module(&linked).unwrap_or_else(|e| panic!("verify: {e:?}"));
    linked
}

/// Run `idx` on both engines with a seeded window; assert the return values agree; return interp's.
fn run_both(m: &Module, idx: u32, args: &[i64], seed: &[u8]) -> Vec<Value> {
    let ivals: Vec<Value> = args.iter().map(|&n| Value::I64(n)).collect();
    let mut fuel = u64::MAX;
    let (ir, _) = svm_interp::run_capture(m, idx, &ivals, &mut fuel, seed);
    let ivals_out = ir.expect("interp");
    let (jout, _) = svm_jit::compile_and_run_capture(m, idx, args, seed).expect("jit");
    let jvals = match jout {
        svm_jit::JitOutcome::Returned(v) => v,
        o => panic!("jit: {o:?}"),
    };
    let iwords: Vec<i64> = ivals_out
        .iter()
        .map(|v| match v {
            Value::I64(n) => *n,
            Value::I32(n) => *n as i64,
            o => panic!("unexpected {o:?}"),
        })
        .collect();
    assert_eq!(iwords, jvals, "§9 interp/JIT parity");
    ivals_out
}

fn seed_window(bytes: usize, heap: i64) -> Vec<u8> {
    let mut seed = vec![0u8; bytes];
    seed[8..16].copy_from_slice(&heap.to_le_bytes()); // bump-allocator heap pointer
    seed
}

#[test]
fn real_sumseq_reads_a_seq() {
    // Real nimony `sumSeq(a: seq[int])` — a `for` loop over `toOpenArray`/`len`/`[]` — bound to the
    // runtime shim and run over a harness-built seq {len:3, data:->[10,20,30]}. Σ = 60.
    const R: &str = include_str!("fixtures/real_seq_loop.leng.nif");
    let m = link_runtime(
        svm_leng::translate_proc(R, "sumSeq.0.").unwrap(),
        vec![],
        vec![],
    );
    let (seq, data, sp) = (512usize, 640usize, 1024i64);
    let mut seed = seed_window(4096, 8192);
    seed[seq..seq + 8].copy_from_slice(&3i64.to_le_bytes());
    seed[seq + 8..seq + 16].copy_from_slice(&(data as i64).to_le_bytes());
    for (i, v) in [10i64, 20, 30].iter().enumerate() {
        seed[data + i * 8..data + i * 8 + 8].copy_from_slice(&v.to_le_bytes());
    }
    // sumSeq is func 0 after link: (sp, seq_addr) -> sum.
    assert_eq!(
        run_both(&m, 0, &[sp, seq as i64], &seed),
        vec![Value::I64(60)]
    );
}

#[test]
fn real_makeseq_builds_a_seq_via_the_allocator() {
    // Real nimony `makeSeq(n)` — `result = @[]; for i in 0..<n: result.add(i)` — bound to the
    // runtime shim (newSeqUninit + realloc-on-add over the bump allocator). Build into an sret
    // pointer and read the constructed seq back from the window.
    const R: &str = include_str!("fixtures/real_seq_loop.leng.nif");
    let m = link_runtime(
        svm_leng::translate_proc(R, "makeSeq.0.").unwrap(),
        vec![],
        vec![],
    );
    let (sret, sp) = (512usize, 1024i64);
    let seed = seed_window(8192, 2048);
    // makeSeq is func 0: (sp, sret, n) -> (); writes a seq into sret.
    let ivals: Vec<Value> = [sp, sret as i64, 3]
        .iter()
        .map(|&n| Value::I64(n))
        .collect();
    let mut fuel = u64::MAX;
    let (ir, imem) = svm_interp::run_capture(&m, 0, &ivals, &mut fuel, &seed);
    ir.expect("interp");
    let (_j, jmem) =
        svm_jit::compile_and_run_capture(&m, 0, &[sp, sret as i64, 3], &seed).expect("jit");
    let n = imem.len().min(jmem.len());
    assert_eq!(imem[..n], jmem[..n], "§9 interp/JIT window parity");
    let len = i64::from_le_bytes(imem[sret..sret + 8].try_into().unwrap());
    let data = i64::from_le_bytes(imem[sret + 8..sret + 16].try_into().unwrap()) as usize;
    assert_eq!(len, 3, "makeSeq(3).len");
    let elems: Vec<i64> = (0..3)
        .map(|i| i64::from_le_bytes(imem[data + i * 8..data + i * 8 + 8].try_into().unwrap()))
        .collect();
    assert_eq!(elems, vec![0, 1, 2], "makeSeq(3) contents");
}

/// A driver linking both procs: `makeSeq(3)` builds a seq, `sumSeq` sums it — the whole program in
/// one run. `tmp` seq at `sp`; makeSeq/sumSeq get disjoint sub-frames above it.
const DRIVER: &str = "\
memory 16

import 0 \"makeSeq.0.\" (i64, i64, i64) -> ()
import 1 \"sumSeq.0.\" (i64, i64) -> (i64)

func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 64
  v2 = i64.add v0 v1
  v3 = i64.const 3
  call.import 0 (v2, v0, v3)
  v4 = i64.const 256
  v5 = i64.add v0 v4
  v6 = call.import 1 (v5, v0)
  return v6
  }
}";

#[test]
fn real_makeseq_then_sumseq_end_to_end() {
    // The full program: build a seq [0,1,2] with makeSeq, sum it with sumSeq → 3. Both real nimony
    // procs, their stdlib ops bound to the shim, run in one pass on both engines.
    const R: &str = include_str!("fixtures/real_seq_loop.leng.nif");
    let user = svm_leng::translate_procs(R, &["makeSeq.0.", "sumSeq.0."]).unwrap();
    let driver = svm_text::parse_module(DRIVER).expect("driver parses");
    let m = link_runtime(
        user,
        vec![("makeSeq.0.".into(), 0), ("sumSeq.0.".into(), 1)],
        vec![LinkUnit {
            module: driver,
            ..Default::default()
        }],
    );
    let driver_idx = (m.funcs.len() - 1) as u32; // driver linked last
    let seed = seed_window(16384, 8192);
    assert_eq!(
        run_both(&m, driver_idx, &[4096], &seed),
        vec![Value::I64(3)]
    );
}
