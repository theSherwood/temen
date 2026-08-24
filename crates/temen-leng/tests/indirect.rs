//! Indirect calls through function pointers (`proctype`) — the nimony backend's dynamic-dispatch
//! path. A `proctype` value is an `i32` **function index** (`ref.func`); calling through it lowers to
//! `call_indirect`, whose masked table dispatch + runtime signature check (§3c) is the security
//! hinge — the verifier and engine carry it, temen-leng only emits the op. Both engines, §9 parity.

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
fn store_funcref_then_call_through_field() {
    // A `Box{fn: proc(int):int, v: int}`. `run(b)` stores `ref.func dbl` into `b.fn`, then calls
    // `b.fn(b.v)`. dbl doubles, so run over v=21 → 42 — proving ref.func + call_indirect round-trip.
    let leng = "\
(stmts
 (type :IntFn.0. . (proctype . (params (param :x.0 . (i +64))) (i +64) (pragmas (nimcall))))
 (type :Box.0. . (object . (fld :fn.0 . IntFn.0.) (fld :v.0 . (i +64))))
 (proc :dbl.0 (params (param :x.0 . (i +64))) (i +64) .
  (stmts . (ret (mul (i +64) x.0 x.0))))
 (proc :run.0 (params (param :b.0 . (ptr Box.0.))) (i +64) .
  (stmts .
   (asgn (dot (deref b.0) fn.0 0) dbl.0)
   (ret (call (dot (deref b.0) fn.0 0) (dot (deref b.0) v.0 0))))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    let text = temen_leng::translate_to_text(leng).unwrap();
    assert!(
        text.contains("ref.func 0"),
        "run stores ref.func dbl:\n{text}"
    );
    assert!(
        text.contains("call_indirect"),
        "run dispatches indirectly:\n{text}"
    );
    // dbl = func 0, run = func 1. Box at offset 128: fn@0 (overwritten by the store), v@8 = 21.
    // run(128) stores ref.func dbl into b.fn, then b.fn(b.v) = dbl(21) = 21*21 = 441.
    // `run` makes an indirect call, so under the funcref ABI it is frame-needing: it takes a leading
    // `$sp` (2048 — well above the Box, never dereferenced since neither proc uses its frame).
    let b = 128usize;
    assert_eq!(run_with_seed(&m, 1, &[2048, b as i64], b + 8, 21), 441);
}

/// Like `run`, but pre-seed an i64 at `off` in the window before running (both engines, parity).
fn run_with_seed(m: &temen_ir::Module, idx: u32, args: &[i64], off: usize, val: i64) -> i64 {
    temen_verify::verify_module(m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let mut seed = vec![0u8; 4096];
    seed[off..off + 8].copy_from_slice(&val.to_le_bytes());
    let ivals: Vec<Value> = args.iter().map(|&n| Value::I64(n)).collect();
    let mut fuel = u64::MAX;
    let (ir, _) = temen_interp::run_capture(m, idx, &ivals, &mut fuel, &seed);
    let iout = ir.expect("interp");
    let iword = match iout.as_slice() {
        [Value::I64(n)] => *n,
        o => panic!("unexpected {o:?}"),
    };
    let (jout, _) = temen_jit::compile_and_run_capture(m, idx, args, &seed).expect("jit");
    let jword = match jout {
        temen_jit::JitOutcome::Returned(v) => v,
        o => panic!("jit: {o:?}"),
    };
    assert_eq!(vec![iword], jword, "§9 interp/JIT parity");
    iword
}

#[test]
fn call_through_funcref_param() {
    // `apply(f: proc(int):int, x): int = f(x)` — an indirect call through a scalar funcref *param*.
    // A driver stores ref.func into a Box and passes it; here we test the param path directly by
    // having `runner(b)` load the funcref from memory and dispatch. dbl(x)=x*x → 7 → 49.
    let leng = "\
(stmts
 (type :IntFn.0. . (proctype . (params (param :x.0 . (i +64))) (i +64) (pragmas (nimcall))))
 (proc :dbl.0 (params (param :x.0 . (i +64))) (i +64) .
  (stmts . (ret (mul (i +64) x.0 x.0))))
 (proc :apply.0 (params (param :f.0 . IntFn.0.) (param :x.0 . (i +64))) (i +64) .
  (stmts . (ret (call f.0 x.0))))
 (proc :ref_dbl.0 . (i +64) .
  (stmts . (discard dbl.0) (ret 0))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    let text = temen_leng::translate_to_text(leng).unwrap();
    assert!(
        text.contains("call_indirect"),
        "apply dispatches indirectly:\n{text}"
    );
    // apply = func 1; f is the i32 funcref index of dbl (func 0), x = 7 → 49. `apply` makes an
    // indirect call, so it is frame-needing under the funcref ABI: a leading `$sp` (2048) precedes
    // the visible params (f, x).
    assert_eq!(run(&m, 1, &[2048, 0, 7]), 49);
}

#[test]
fn virtual_dispatch_through_a_vtable() {
    // The RTTI shape nimony emits for method calls: an object's `vt` points at a vtable whose `mt`
    // flexarray holds method funcrefs (as opaque `(ptr void)` slots). A virtual call loads
    // `o.vt.mt[0]`, casts it to the method's `(ptr proctype)`, and dispatches. This exercises the
    // whole chain the real `finalizeCoroutine`/exception-destroy hooks use: the cast-deref static
    // type walk, flexarray funcref indexing, and the `i64` method-slot → `i32` funcref narrowing.
    let leng = "\
(stmts
 (type :Getter.0. . (proctype . (params (param :self.0 . (ptr Obj.0.))) (i +64) (pragmas (nimcall))))
 (type :GetterP.0. . (ptr Getter.0.))
 (type :Vtbl.0. . (object . (fld :mt.0 . (flexarray (ptr (void))))))
 (type :Obj.0. . (object . (fld :vt.0 . (ptr Vtbl.0.)) (fld :x.0 . (i +64))))
 (proc :getX.0 (params (param :self.0 . (ptr Obj.0.))) (i +64) .
  (stmts . (ret (dot (deref self.0) x.0 0))))
 (proc :dispatch.0 (params (param :o.0 . (ptr Obj.0.))) (i +64) .
  (stmts .
   (ret (call
     (cast GetterP.0. (pat (dot (deref (dot (deref o.0) vt.0 0)) mt.0 0) 0))
     o.0))))
 (proc :ref_getX.0 . (i +64) .
  (stmts . (discard getX.0) (ret 0))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let text = temen_leng::translate_to_text(leng).unwrap();
    assert!(
        text.contains("call_indirect"),
        "dispatch is indirect:\n{text}"
    );

    // getX = func 0, dispatch = func 1. Lay out a vtable and object in the window:
    //   Vtbl @ 256: mt[0] @256 = 0 (getX's function index).
    //   Obj  @ 320: vt @320 = 256, x @328 = 42.
    let (vt, obj) = (256usize, 320usize);
    let mut seed = vec![0u8; 4096];
    seed[vt..vt + 8].copy_from_slice(&0u64.to_le_bytes()); // mt[0] = getX (func 0)
    seed[obj..obj + 8].copy_from_slice(&(vt as u64).to_le_bytes()); // o.vt = &Vtbl
    seed[obj + 8..obj + 16].copy_from_slice(&42u64.to_le_bytes()); // o.x = 42

    // `dispatch` makes an indirect call → frame-needing under the funcref ABI: leading `$sp` (2048).
    let mut fuel = u64::MAX;
    let (ir, _) = temen_interp::run_capture(
        &m,
        1,
        &[Value::I64(2048), Value::I64(obj as i64)],
        &mut fuel,
        &seed,
    );
    let iword = match ir.expect("interp").as_slice() {
        [Value::I64(n)] => *n,
        o => panic!("unexpected {o:?}"),
    };
    let (jout, _) =
        temen_jit::compile_and_run_capture(&m, 1, &[2048, obj as i64], &seed).expect("jit");
    let jword = match jout {
        temen_jit::JitOutcome::Returned(v) => v,
        o => panic!("jit: {o:?}"),
    };
    assert_eq!(vec![iword], jword, "§9 interp/JIT parity");
    assert_eq!(
        iword, 42,
        "virtual dispatch resolved o.vt.mt[0](o) = getX(o) = o.x"
    );
}

#[test]
fn dispatch_through_a_materialized_const_vtable() {
    // The `method` shape (#979): the vtable is a **`const`** whose `mt` flexarray holds method
    // funcrefs — `(oconstr Vtbl (kv mt (aconstr (flexarray (ptr void)) (cast (ptr void) getX))))`.
    // Unlike `virtual_dispatch_through_a_vtable` (which *seeds* the vtable bytes by hand), here
    // temen-leng must **materialize** the const's `mt` table (a `data.funcref` the linker resolves), so
    // `o.vt.mt[0]` reaches the real method. Before the fix the const stayed a zeroed placeholder and
    // dispatch silently called func 0. `decoy` is func 0 and `getX` is func 1, so a broken
    // materialization returns 999 (the decoy), while a correct one returns o.x = 42. Driven through
    // the **link** path (`link_whole_with_runtime`) — the one nimony uses — so the funcref resolves
    // (runnable `translate` leaves data funcrefs to a linker, as it does for funcref gvars).
    let leng = "\
(stmts
 (type :Getter.0. . (proctype . (params (param :self.0 . (ptr Obj.0.))) (i +64) (pragmas (nimcall))))
 (type :GetterP.0. . (ptr Getter.0.))
 (type :Vtbl.0. . (object . (fld :mt.0 . (flexarray (ptr (void))))))
 (type :Obj.0. . (object . (fld :vt.0 . (ptr Vtbl.0.)) (fld :x.0 . (i +64))))
 (proc :decoy.0 (params (param :self.0 . (ptr Obj.0.))) (i +64) .
  (stmts . (ret 999)))
 (proc :getX.0 (params (param :self.0 . (ptr Obj.0.))) (i +64) .
  (stmts . (ret (dot (deref self.0) x.0 0))))
 (const :theVt.0. . Vtbl.0.
  (oconstr Vtbl.0. (kv mt.0 (aconstr (flexarray (ptr (void))) (cast (ptr (void)) getX.0)))))
 (proc :dispatch.0 (params (param :o.0 . (ptr Obj.0.))) (i +64) .
  (stmts .
   (asgn (dot (deref o.0) vt.0 0) (cast (ptr Vtbl.0.) (addr theVt.0.)))
   (ret (call
     (cast GetterP.0. (pat (dot (deref (dot (deref o.0) vt.0 0)) mt.0 0) 0))
     o.0)))))";
    let unit = temen_leng::WholeModule {
        stem: "vt7test",
        src: leng,
    };
    let m = temen_leng::link_whole_with_runtime(&[unit], vec![])
        .unwrap_or_else(|e| panic!("link: {e}"));
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let dispatch = m
        .exports
        .iter()
        .find(|e| e.name.contains("dispatch"))
        .expect("dispatch export")
        .func;

    // Object at 320: vt@320 (set by dispatch), x@328 = 42.
    let obj = 320usize;
    let mut seed = vec![0u8; 4096];
    seed[obj + 8..obj + 16].copy_from_slice(&42u64.to_le_bytes());
    // `dispatch` makes an indirect call → frame-needing under the funcref ABI: leading `$sp` (2048).
    let mut fuel = u64::MAX;
    let (ir, _) = temen_interp::run_capture(
        &m,
        dispatch,
        &[Value::I64(2048), Value::I64(obj as i64)],
        &mut fuel,
        &seed,
    );
    let iword = match ir.expect("interp").as_slice() {
        [Value::I64(n)] => *n,
        o => panic!("unexpected {o:?}"),
    };
    let (jout, _) =
        temen_jit::compile_and_run_capture(&m, dispatch, &[2048, obj as i64], &seed).expect("jit");
    let jword = match jout {
        temen_jit::JitOutcome::Returned(v) => v,
        o => panic!("jit: {o:?}"),
    };
    assert_eq!(vec![iword], jword, "§9 interp/JIT parity");
    assert_eq!(
        iword, 42,
        "dispatch through the materialized const vtable calls getX (func 1), not the decoy (func 0)"
    );
}

#[test]
fn baseobj_upcasts_to_the_base_subobject() {
    // `(baseobj Base N obj)` upcasts to the inlined base sub-object (offset 0, single inheritance).
    // `Derived` inlines `Base{tag}` at offset 0, then its own `extra`. Reading the base's `tag`
    // through `baseobj` must see the same field the derived object stores at offset 0.
    let leng = "\
(stmts
 (type :Base.0. . (object . (fld :tag.0 . (i +64))))
 (type :Derived.0. . (object Base.0. (fld :extra.0 . (i +64))))
 (proc :baseTag.0 (params (param :d.0 . (ptr Derived.0.))) (i +64) .
  (stmts .
   (ret (dot (baseobj Base.0. 1 (deref d.0)) tag.0 0)))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    // Derived @ 128: tag (base, offset 0) = 7.
    let d = 128usize;
    let mut seed = vec![0u8; 4096];
    seed[d..d + 8].copy_from_slice(&7u64.to_le_bytes());
    let mut fuel = u64::MAX;
    let (ir, _) = temen_interp::run_capture(&m, 0, &[Value::I64(d as i64)], &mut fuel, &seed);
    let iword = match ir.expect("interp").as_slice() {
        [Value::I64(n)] => *n,
        o => panic!("unexpected {o:?}"),
    };
    let (jout, _) = temen_jit::compile_and_run_capture(&m, 0, &[d as i64], &seed).expect("jit");
    let jword = match jout {
        temen_jit::JitOutcome::Returned(v) => v,
        o => panic!("jit: {o:?}"),
    };
    assert_eq!(vec![iword], jword, "§9 interp/JIT parity");
    assert_eq!(
        iword, 7,
        "baseobj read the base sub-object's tag at offset 0"
    );
}
