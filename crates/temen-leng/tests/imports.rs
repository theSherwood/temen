//! Cross-module call tests (NIM.md Phase 2): a call to a callee not defined in this module becomes
//! a declared Temen `import` + `call.import`, with its signature fixed from the call site. The runtime
//! binds the import by name at instantiation (like `write`); here we bind a mock host fn and run.

use temen_interp::{run_with_host, BoundImport, Host, Value};

const MOD: &str = "\
(stmts
 (proc :use_ext.0 (params (param :x.0 . (i +64))) (i +64) .
  (stmts .
   (ret (add (i +64) (call ext_double.0.m x.0) 1)))))";

#[test]
fn cross_module_call_becomes_import() {
    let text = temen_leng::translate_to_text(MOD).unwrap();
    // The external callee is declared as import slot 0 with the call-site signature, and dispatched
    // via call.import.
    assert!(
        text.contains("import 0 \"ext_double.0.m\" (i64) -> (i64)"),
        "expected an import declaration:\n{text}"
    );
    assert!(
        text.contains("call.import 0"),
        "expected call.import:\n{text}"
    );

    let m = temen_leng::translate(MOD).unwrap_or_else(|e| panic!("translate: {e}"));
    // The frontend's job is a well-typed, verifier-accepted module; the embedder binds the import.
    temen_verify::verify_module(&m).expect("verify (imports are bound by the embedder)");
}

#[test]
fn importc_proc_becomes_an_import() {
    // An `importc` proc is a C extern (the bottom edge: `memcpy`, `getpid`, `mmap`) with no
    // translatable body. It is *not* emitted as a func; a call to it lowers to an import the host
    // binds at link — so a non-void extern no longer trips "falls off the end without ret".
    let leng = "\
(stmts
 (proc :c_getpid.0. . (i +32) (pragmas (importc \"getpid\")) (stmts .))
 (proc :pid.0. . (i +32) . (stmts . (ret (call c_getpid.0.)))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    temen_verify::verify_module(&m).expect("verify");
    assert_eq!(
        m.funcs.len(),
        1,
        "only `pid` is a func; the extern is skipped"
    );
    assert!(
        m.imports.iter().any(|i| i.name == "c_getpid.0."),
        "the extern is an import: {:?}",
        m.imports.iter().map(|i| &i.name).collect::<Vec<_>>()
    );
}

#[test]
fn import_runs_when_bound() {
    let m = temen_leng::translate(MOD).unwrap();
    // Bind import slot 0 to a host fn `ext_double(x) = x * 2`, then use_ext(x) = x*2 + 1.
    let mut host = Host::new();
    let handle = host.grant_host_proc(Box::new(|_op, args, _mem, _| Ok(vec![args[0] * 2])));
    host.set_import_bindings(vec![BoundImport::required(
        temen_interp::cap_id::HOST_PROC,
        0,
        handle,
    )]);

    let mut fuel = u64::MAX;
    let out = run_with_host(&m, 0, &[Value::I64(20)], &mut fuel, &mut host).expect("run");
    assert_eq!(out.as_slice(), &[Value::I64(41)], "use_ext(20) = 20*2 + 1");
}

#[test]
fn proc_name_argument_becomes_funcref() {
    // A bare **proc name in argument position** is a funcref — its `ref.func` index — not a data
    // symbol. nimony passes proc addresses this way: `setExitFlush(flushStdStreams)` registering the
    // at-exit stdout flush, `atexit` handlers, callback tables. `register(cb)` passes local proc
    // `cb`'s address to a cross-module import; it must lower to `ref.func`, not fall through to the
    // `data.sym` path (a proc exports as a *func*, never data) and fail to resolve.
    let leng = "\
(stmts
 (proc :cb.0. . (i +64) . (stmts . (ret 7)))
 (proc :main.0. . (void) .
  (stmts .
   (call register.0.other cb.0.))))";
    let text = temen_leng::translate_to_text(leng).unwrap();
    assert!(
        text.contains("ref.func"),
        "proc-name arg → ref.func:\n{text}"
    );
    assert!(
        text.contains("call.import"),
        "register is a cross-module import:\n{text}"
    );
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    temen_verify::verify_module(&m).expect("verify");
}

#[test]
fn stmt_call_import_is_void() {
    // A cross-module call in statement position declares a void import (no result arity).
    let leng = "\
(stmts
 (proc :notify.0 (params (param :x.0 . (i +64))) (void) .
  (stmts .
   (call log_it.0.m x.0))))";
    let text = temen_leng::translate_to_text(leng).unwrap();
    assert!(
        text.contains("import 0 \"log_it.0.m\" (i64) -> ()"),
        "void import:\n{text}"
    );
}

#[test]
fn variadic_import_marshals_a_va_buffer() {
    // A `{.varargs.}` C import (`c_snprintf`) is spelled by nimony as a final `(varargs)`-typed param.
    // Its call sites pass different numbers of trailing args, which used to trip "import called with
    // inconsistent arity" (a fixed-arity `call.import` can't vary). Now the fixed params pass through
    // and the variadic tail is marshalled into a data-stack buffer (an `i64` slot per integer, an
    // `f64` slot per float — C default argument promotions), replaced by one pointer — so every call
    // site collapses to the same `fixed + 1` signature. Here `vfn(a, …)`: one call with a single int
    // vararg, one with an int + a float, both declaring `import 0 "vfn.0." (i64, i64) -> ()`.
    let leng = "\
(stmts
 (proc :vfn.0. (params (param :a.0 . (i +64)) (param :va.0 . (varargs))) (i +64) (pragmas (importc \"vfn\")) (stmts .))
 (proc :callit.0. (params (param :x.0 . (i +64)) (param :y.0 . (f +64))) (i +64) .
  (stmts .
   (call vfn.0. x.0 x.0)
   (call vfn.0. x.0 x.0 y.0)
   (ret x.0))))";
    let text = temen_leng::translate_to_text(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    // One import, fixed arity 2 (the `a` param + the va-list pointer) despite the 2-vs-3-arg calls.
    assert!(
        text.contains("import 0 \"vfn.0.\" (i64, i64) -> ()"),
        "collapsed varargs import signature:\n{text}"
    );
    assert!(
        !text.contains("import 1 \"vfn.0.\""),
        "the two call sites must share one import slot:\n{text}"
    );
    // The variadic tail is marshalled by value: an `i64` slot for the int arg, an `f64` slot for the
    // float — proving the float's bits are stored, not truncated to an integer.
    assert!(
        text.contains("i64.store") && text.contains("f64.store"),
        "va-buffer marshalling stores:\n{text}"
    );
    // The frontend's job is a verifier-accepted module (the buffer stores are confined); the embedder
    // binds `snprintf`.
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    temen_verify::verify_module(&m).expect("verify (varargs buffer stores are confined)");
}

/// The real payoff: nimony's own `sumto` — `while i <= n: (inc(addr i); result += i)` — where `inc`
/// is a **cross-module system import**. It also uses an address-taken loop counter (a frame) and a
/// `while`. Translating it out of the real hexer module now succeeds and verifies.
#[test]
fn real_nimony_sumto_with_import() {
    const REAL: &str = include_str!("fixtures/real_controlflow.leng.nif");
    let m = temen_leng::translate_proc(REAL, "sumto.0.")
        .unwrap_or_else(|e| panic!("translate real sumto: {e}"));
    temen_verify::verify_module(&m).expect("verify real sumto");
    // The module declares the `inc` import and frames the address-taken counter.
    let text = temen_leng::translate_proc_to_text(REAL, "sumto.0.").unwrap();
    assert!(
        text.contains("call.import"),
        "sumto calls inc via import:\n{text}"
    );
}
