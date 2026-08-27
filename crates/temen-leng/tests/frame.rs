//! Address-of-local + data-stack frame tests (NIM.md Phase 2): a proc that takes `(addr x)` of a
//! local demotes it to a window frame slot and gains a threaded stack pointer; a call to a
//! frame-needing proc passes a fresh frame. Exercises the mixed model — some locals in SSA slots,
//! some in the frame — over both engines.

use temen_interp::Value;

fn run(module: &temen_ir::Module, idx: u32, args: &[i64]) -> i64 {
    temen_verify::verify_module(module).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let ivals: Vec<Value> = args.iter().map(|&n| Value::I64(n)).collect();
    let mut fuel = u64::MAX;
    let interp = temen_interp::run(module, idx, &ivals, &mut fuel).expect("interp run");
    let interp_n = match interp.as_slice() {
        [Value::I64(n)] => *n,
        other => panic!("expected i64, got {other:?}"),
    };
    let jit = match temen_jit::compile_and_run(module, idx, args).expect("jit compile") {
        temen_jit::JitOutcome::Returned(v) => v,
        other => panic!("jit: {other:?}"),
    };
    assert_eq!(jit.as_slice(), &[interp_n], "§9 interp/JIT parity");
    interp_n
}

/// The pattern real nimony emits for a loop counter: `inc(addr i)`.
/// incp(p) is frameless (pointer param); count(n) is frame-needing (takes `addr i`).
const MODULE: &str = "\
(stmts
 (proc :incp.0 (params (param :p.0 . (ptr (i +64)))) (void) .
  (stmts .
   (asgn (deref p.0) (add (i +64) (deref p.0) 1))))
 (proc :count.0 (params (param :n.0 . (i +64))) (i +64) .
  (stmts .
   (var :i.0 . (i +64) 0)
   (while (lt i.0 n.0)
    (stmts .
     (call incp.0 (addr i.0))))
   (ret i.0))))";

#[test]
fn addr_of_local_through_a_call() {
    let text = temen_leng::translate_to_text(MODULE).unwrap();
    // count is frame-needing → its signature gains a leading i64 stack pointer.
    assert!(
        text.contains("func (i64, i64) -> (i64)"),
        "count should take (sp, n):\n{text}"
    );
    assert!(
        text.starts_with("memory "),
        "frame proc needs a window:\n{text}"
    );

    let m = temen_leng::translate(MODULE).unwrap_or_else(|e| panic!("translate: {e}"));
    // count is func 1 (incp is 0). Call it with sp = a window offset (its frame lives there) and n.
    // #1094: the NULL guard is unconditional, so the frame must clear `[0, POWERBOX_NULL_GUARD)`.
    let sp = 20480;
    for n in [0i64, 1, 5, 50] {
        assert_eq!(run(&m, 1, &[sp, n]), n, "count({n})");
    }
}

#[test]
fn mixed_frame_and_ssa_locals() {
    // sum_upto(n): `acc` is a plain SSA local; `i` is address-taken (frame). Returns 0+1+…+(n-1).
    let leng = "\
(stmts
 (proc :bump.0 (params (param :p.0 . (ptr (i +64)))) (void) .
  (stmts .
   (asgn (deref p.0) (add (i +64) (deref p.0) 1))))
 (proc :sum_upto.0 (params (param :n.0 . (i +64))) (i +64) .
  (stmts .
   (var :acc.0 . (i +64) 0)
   (var :i.0 . (i +64) 0)
   (while (lt i.0 n.0)
    (stmts .
     (asgn acc.0 (add (i +64) acc.0 i.0))
     (call bump.0 (addr i.0))))
   (ret acc.0))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    // #1094: the NULL guard is unconditional, so the frame must clear `[0, POWERBOX_NULL_GUARD)`.
    let sp = 24576;
    assert_eq!(run(&m, 1, &[sp, 5]), (0..5).sum::<i64>()); // 0+1+2+3+4 = 10
    assert_eq!(run(&m, 1, &[sp, 11]), (0..11).sum::<i64>()); // 55
}

#[test]
fn addr_of_non_frame_is_fail_closed_via_pure_proc() {
    // A pure-integer proc (no addr) must NOT gain a stack pointer.
    let leng = "(stmts (proc :id.0 (params (param :x.0 . (i +64))) (i +64) . (stmts . (ret x.0))))";
    let text = temen_leng::translate_to_text(leng).unwrap();
    assert!(
        text.contains("func (i64) -> (i64)"),
        "pure proc keeps its plain signature:\n{text}"
    );
}

#[test]
fn frameless_caller_of_framed_callee_gets_a_frame() {
    // `callee` holds an aggregate local, so it needs a frame ($sp). `caller` has no frame of its
    // own, but calling `callee` means it must own an $sp to hand a fresh sub-frame down — transitive
    // frame propagation makes `caller` frame-needing too. Without it, this fails to translate.
    let leng = "\
(stmts
 (type :P.0. . (object . (fld :a.0 . (i +64))))
 (proc :callee.0 (params (param :n.0 . (i +64))) (i +64) .
  (stmts .
   (var :p.0 . P.0. .)
   (asgn (dot p.0 a.0 0) n.0)
   (ret (dot p.0 a.0 0))))
 (proc :caller.0 (params (param :x.0 . (i +64))) (i +64) .
  (stmts . (ret (call callee.0 x.0)))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    // Both are frame-needing now: caller ($sp, x) -> i64.
    assert_eq!(
        run(&m, 1, &[24576, 5]),
        5,
        "caller threads a sub-frame to callee"
    );
    assert_eq!(run(&m, 1, &[24576, 42]), 42);
}

#[test]
fn scalar_param_spill() {
    // Taking the address of a by-value scalar param spills it to a frame slot: its incoming value is
    // stored there at entry, and `(addr x)` / reads / writes go through the frame. f(x) = { p = &x;
    // *p = 99; return x } — the write through the pointer must be visible when x is read back.
    let leng = "\
(stmts
 (proc :f.0 (params (param :x.0 . (i +64))) (i +64) .
  (stmts .
   (var :p.0 . (ptr (i +64)) (addr x.0))
   (asgn (deref p.0) 99)
   (ret x.0))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    // f is frame-needing (spill): ($sp, x) -> i64. Result is 99 regardless of the incoming x.
    assert_eq!(
        run(&m, 0, &[24576, 5]),
        99,
        "write through &x is seen reading x"
    );
    assert_eq!(run(&m, 0, &[24576, 0]), 99);
}
