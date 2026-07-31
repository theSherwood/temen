//! seq/string tests (NIM.md Phase 2, W1 Leng totality). nimony's `seq[T]` is a `{len, data*}`
//! fat-pointer **object** (`string` is analogous), so its *value* layout and element access already
//! ride the object + pointer machinery. Its *operations*, though — `add`, `[]`, `len`,
//! `toOpenArray`, `newSeq` — are stdlib procs that lower to **imports**: they translate and verify
//! here, but running them needs those imports bound to a real seq runtime (the W3 runtime edge). So
//! this slice proves the frontend lowers real seq code to valid IR, and *runs* the parts that don't
//! need the runtime (a hand-written seq over a caller-provided buffer, `break`/`continue`).
//!
//! Getting real seq bytes to lower needed four fixes, all exercised below: import **names** escaped
//! for svm-text (the `[]` operator mangles to `\5B\5D…`), aggregate **args** to imports passed by
//! address, aggregate-**returning** imports (sret imports, e.g. `toOpenArray`), and structured
//! `break`/`continue` (a `for` lowers to `while (true) { … else break }`).

use svm_interp::Value;

fn run(module: &svm_ir::Module, idx: u32, args: &[i64]) -> i64 {
    svm_verify::verify_module(module).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let ivals: Vec<Value> = args.iter().map(|&n| Value::I64(n)).collect();
    let mut fuel = u64::MAX;
    let interp = svm_interp::run(module, idx, &ivals, &mut fuel).expect("interp run");
    let interp_n = match interp.as_slice() {
        [Value::I64(n)] => *n,
        other => panic!("expected i64, got {other:?}"),
    };
    let jit = match svm_jit::compile_and_run(module, idx, args).expect("jit compile") {
        svm_jit::JitOutcome::Returned(v) => v,
        other => panic!("jit: {other:?}"),
    };
    assert_eq!(jit.as_slice(), &[interp_n], "§9 interp/JIT parity");
    interp_n
}

#[test]
fn seq_value_sum_over_buffer() {
    // A hand-written `seq[int]` as `{len, data*}`: sum its elements by loading `data` into a pointer
    // local and indexing it (`pat`). The caller builds the seq object + data in the window, so this
    // runs with no seq runtime. sumSeq(s) = Σ s.data[0..s.len].
    let leng = "\
(stmts
 (type :Seq.0. . (object . (fld :len.0 . (i +64)) (fld :data.0 . (ptr (i +64)))))
 (proc :sumSeq.0 (params (param :s.0 . Seq.0.)) (i +64) .
  (stmts .
   (var :acc.0 . (i +64) 0)
   (var :i.0 . (i +64) 0)
   (var :p.0 . (ptr (i +64)) (dot s.0 data.0 0))
   (while (lt i.0 (dot s.0 len.0 0))
    (stmts .
     (asgn acc.0 (add (i +64) acc.0 (pat p.0 i.0)))
     (asgn i.0 (add (i +64) i.0 1))))
   (ret acc.0))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    // Build a seq {len:3, data:->[10,20,30]} in the window and call sumSeq with its address.
    let seq_addr = 64usize;
    let data_addr = 128usize;
    let mut seed = vec![0u8; 4096];
    seed[seq_addr..seq_addr + 8].copy_from_slice(&3i64.to_le_bytes());
    seed[seq_addr + 8..seq_addr + 16].copy_from_slice(&(data_addr as i64).to_le_bytes());
    for (i, v) in [10i64, 20, 30].iter().enumerate() {
        let off = data_addr + i * 8;
        seed[off..off + 8].copy_from_slice(&v.to_le_bytes());
    }
    // Run on both engines with the seeded window (args: the seq's address).
    svm_verify::verify_module(&m).unwrap();
    let mut fuel = u64::MAX;
    let (ir, _) = svm_interp::run_capture(&m, 0, &[Value::I64(seq_addr as i64)], &mut fuel, &seed);
    assert_eq!(ir.expect("interp").as_slice(), &[Value::I64(60)]);
    let (jout, _) =
        svm_jit::compile_and_run_capture(&m, 0, &[seq_addr as i64], &seed).expect("jit");
    match jout {
        svm_jit::JitOutcome::Returned(v) => assert_eq!(v, vec![60], "§9 parity"),
        o => panic!("jit: {o:?}"),
    }
}

#[test]
fn break_and_continue() {
    // sumEvensUntil(n, stop): sum even i in [0,n), but stop (break) when i == stop; skip odds
    // (continue). Exercises structured break/continue in a `while` — nimony's `for`-loop shape.
    let leng = "\
(stmts
 (proc :f.0 (params (param :n.0 . (i +64)) (param :stop.0 . (i +64))) (i +64) .
  (stmts .
   (var :acc.0 . (i +64) 0)
   (var :i.0 . (i +64) -1)
   (while (true)
    (stmts .
     (asgn i.0 (add (i +64) i.0 1))
     (if (elif (eq i.0 n.0) (stmts . (break))))
     (if (elif (eq i.0 stop.0) (stmts . (break))))
     (if (elif (eq (mod (i +64) i.0 2) 1) (stmts . (continue))))
     (asgn acc.0 (add (i +64) acc.0 i.0))))
   (ret acc.0))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    // n=6: evens 0,2,4 = 6; stop unreached (99).
    assert_eq!(run(&m, 0, &[6, 99]), 6);
    // stop at i==3: evens before 3 → 0,2 = 2.
    assert_eq!(run(&m, 0, &[10, 3]), 2);
}

#[test]
fn break_outside_loop_is_fail_closed() {
    let leng = "(stmts (proc :b.0 . (i +64) . (stmts . (break) (ret 0))))";
    match svm_leng::translate(leng) {
        Err(svm_leng::LengError::Malformed(_)) | Err(svm_leng::LengError::Unsupported(_)) => {}
        other => panic!("expected fail-closed, got {other:?}"),
    }
}

/// Real nimony `hexer` output for `getAt(s: seq[int], i): int = s[i]` and `firstLen(s): int = s.len`.
/// Both go through stdlib imports (`[]` — whose name mangles to `\5B\5D…` — and `len`), so they
/// **translate and verify** as valid IR with those imports unbound (running needs the seq runtime).
#[test]
fn real_nimony_seq_index_verifies() {
    const REAL: &str = include_str!("fixtures/real_seq_index.leng.nif");
    for name in ["getAt.0.", "firstLen.0."] {
        let m = svm_leng::translate_proc(REAL, name)
            .unwrap_or_else(|e| panic!("translate real {name}: {e}"));
        svm_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify {name}: {e:?}"));
    }
    // The `[]` operator's escaped import name is present and well-formed.
    let text = svm_leng::translate_proc_to_text(REAL, "getAt.0.").unwrap();
    assert!(
        text.contains("call.import"),
        "seq index is an import:\n{text}"
    );
}

/// Real nimony `hexer` output for `sumSeq(a: seq[int])` (a `for` loop with `toOpenArray`/`len`/`[]`)
/// and `makeSeq(n)` (`result.add`, `newSeqUninit`) — the full read and write seq paths. Both
/// translate and verify: the `for` lowers to a `while`+`break`, `toOpenArray`/`newSeqUninit` become
/// sret imports, `len`/`[]`/`add` ordinary imports (aggregate args by address).
#[test]
fn real_nimony_seq_loop_verifies() {
    const REAL: &str = include_str!("fixtures/real_seq_loop.leng.nif");
    for name in ["sumSeq.0.", "makeSeq.0."] {
        let m = svm_leng::translate_proc(REAL, name)
            .unwrap_or_else(|e| panic!("translate real {name}: {e}"));
        svm_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify {name}: {e:?}"));
    }
}
