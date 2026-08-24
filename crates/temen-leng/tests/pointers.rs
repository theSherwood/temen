//! Pointer tests (NIM.md Phase 2 broadening): pointer-typed params, `deref` (load) and store
//! through a pointer, over a window. No data-stack frame yet — the pointer is supplied by the
//! caller as a window offset — so these exercise the memory machinery (memory decl, loads, stores)
//! without the address-of-local ABI (the next slice). Run on both engines.

use temen_interp::Value;

/// Run func `idx` with i64 args on interp + JIT (both must agree); return the i64 result.
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

#[test]
fn store_then_load_through_pointer() {
    // roundtrip(p: ptr int, x: int): int = { *p = x; return *p }  — p is a window offset.
    let leng = "\
(stmts
 (proc :roundtrip.0 (params (param :p.0 . (ptr (i +64))) (param :x.0 . (i +64))) (i +64) .
  (stmts .
   (asgn (deref p.0) x.0)
   (ret (deref p.0)))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    // The IR must declare a window for the load/store.
    let text = temen_leng::translate_to_text(leng).unwrap();
    assert!(
        text.starts_with("memory "),
        "module must declare memory:\n{text}"
    );
    // Pass p = a valid 8-aligned window offset; x is the value to round-trip.
    assert_eq!(run(&m, 0, &[64, 42]), 42);
    assert_eq!(run(&m, 0, &[128, -7]), -7);
}

#[test]
fn store_stmt_reversed_operands() {
    // Leng StoreStmt `(store Value Lvalue)` = asgn with reversed operands: *p = v.
    // addto(p, d): int = { *p = *p + d; return *p }
    let leng = "\
(stmts
 (proc :addto.0 (params (param :p.0 . (ptr (i +64))) (param :d.0 . (i +64))) (i +64) .
  (stmts .
   (store (add (i +64) (deref p.0) d.0) (deref p.0))
   (ret (deref p.0)))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    // The window starts zeroed, so *p == 0 initially; addto(p, 5) → 5, and no `d` doubling.
    assert_eq!(run(&m, 0, &[64, 5]), 5);
    assert_eq!(run(&m, 0, &[256, 100]), 100);
}

#[test]
fn named_pointer_alias_type_derefs() {
    // A named type that aliases a pointer — `(type :NodeRef … (ptr Node))` — must be treated as a
    // typed pointer, so a param/field declared with that bare name can `deref`. This is exactly the
    // shape the real system module uses for `ref` types (`RootRef = (ptr t.0.IAref…)`), whose ARC
    // hooks take a bare `RootRef`-typed param and `(deref)` it. Without alias resolution the param
    // is a plain `i64` scalar and the deref fails "not a known pointer".
    let leng = "\
(stmts
 (type :Node.0. . (object . (fld :val.0 . (i +64)) (fld :nxt.0 . NodeRef.0.)))
 (type :NodeRef.0. . (ptr Node.0.))
 (proc :nodeVal.0 (params (param :n.0 . NodeRef.0.)) (i +64) .
  (stmts .
   (ret (dot (deref n.0) val.0 0)))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    // Store val = 42 at window offset 128 (the Node's `val` field is at offset 0), then read it back
    // through the aliased pointer. The window starts zeroed; write 42 via a helper store is awkward,
    // so instead point `n` at an offset we pre-seed by round-tripping through a sibling proc is
    // overkill — the zeroed window gives val == 0, and a distinct non-zero check follows.
    assert_eq!(run(&m, 0, &[128]), 0);
}

#[test]
fn named_pointer_alias_stores_through_field() {
    // Round-trip through an aliased-pointer param: write then read the pointee's field. Proves the
    // alias resolves to a *typed* pointer whose pointee object layout is recovered for `dot`.
    let leng = "\
(stmts
 (type :Cell.0. . (object . (fld :v.0 . (i +64))))
 (type :CellRef.0. . (ptr Cell.0.))
 (proc :setget.0 (params (param :c.0 . CellRef.0.) (param :x.0 . (i +64))) (i +64) .
  (stmts .
   (asgn (dot (deref c.0) v.0 0) x.0)
   (ret (dot (deref c.0) v.0 0)))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    assert_eq!(run(&m, 0, &[64, 42]), 42);
    assert_eq!(run(&m, 0, &[256, -9]), -9);
}

#[test]
fn pointer_constant_global_folds_cast_sentinel() {
    // `MAP_FAILED = cast[pointer](-1)` (the shape times/monotimes open with) — a `pointer` global is
    // an `i64` scalar, so the C-style `cast` of the `-1` sentinel folds to eight `0xFF` bytes seeded
    // into the global's data window. Before the fold saw through `cast` (only `conv`), this errored
    // "non-scalar-int global initializer". Read the global back: its value is `-1` on both engines.
    let leng = "\
(stmts
 (gvar :mf.0. . (ptr (void)) (cast (ptr (void)) -1))
 (proc :readMf.0. . (i +64) . (stmts . (ret mf.0.))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    assert_eq!(run(&m, 0, &[]), -1);
}

#[test]
fn no_memory_decl_without_pointers() {
    // A pure-integer proc must NOT declare a window (memory only appears when actually used).
    let leng = "(stmts (proc :id.0 (params (param :x.0 . (i +64))) (i +64) . (stmts . (ret x.0))))";
    let text = temen_leng::translate_to_text(leng).unwrap();
    assert!(!text.contains("memory "), "no memory expected:\n{text}");
}
