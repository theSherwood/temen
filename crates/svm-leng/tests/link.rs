//! Cross-module linking tests (NIM.md W2 — the linker). A real program is many modules; nimony emits
//! one Leng file per module and references a proc `P.` *defined* in module `stem` from *other*
//! modules as `P.<stem>`. `svm_leng::link_units` translates each module's procs, exports them under
//! those global names, and resolves every unit's cross-module calls (named imports) against the
//! exports via `svm_ir::link` — one merged, import-free svm module. Both engines, §9 parity.

use svm_interp::Value;
use svm_leng::LengModule;

/// Run func `idx` of a linked module on both engines; assert agreement; return the i64 result.
fn run(m: &svm_ir::Module, idx: u32, args: &[i64]) -> i64 {
    svm_verify::verify_module(m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let ivals: Vec<Value> = args.iter().map(|&n| Value::I64(n)).collect();
    let mut fuel = u64::MAX;
    let interp = svm_interp::run(m, idx, &ivals, &mut fuel).expect("interp");
    let n = match interp.as_slice() {
        [Value::I64(n)] => *n,
        o => panic!("expected i64, got {o:?}"),
    };
    let jit = match svm_jit::compile_and_run(m, idx, args).expect("jit") {
        svm_jit::JitOutcome::Returned(v) => v,
        o => panic!("jit: {o:?}"),
    };
    assert_eq!(jit.as_slice(), &[n], "§9 interp/JIT parity");
    n
}

/// Real nimony: `moda` imports `pkg/modb` and calls `helper`. `hexer` emits `moda`'s call as
/// `helper.0.<modb-stem>`; `modb` defines `helper.0.` locally. Linking the two translated modules
/// resolves the cross-module call to compiled Nim, and `useit(5) = helper(5) + 1 = 16`.
#[test]
fn real_two_module_program() {
    const MODA: &str = include_str!("fixtures/real_moda.leng.nif");
    const MODB: &str = include_str!("fixtures/real_modb.leng.nif");
    // modb's file stem — the suffix moda's cross-module call to `helper` carries.
    let linked = svm_leng::link_units(&[
        LengModule {
            stem: "modywjwgs",
            src: MODA,
            names: &["useit.0."],
        },
        LengModule {
            stem: "modwru7vt1",
            src: MODB,
            names: &["helper.0."],
        },
    ])
    .unwrap_or_else(|e| panic!("link: {e}"));
    // useit is module A's first proc → func 0.
    assert_eq!(run(&linked, 0, &[5]), 16);
    assert_eq!(run(&linked, 0, &[10]), 31);
}

/// A transitive chain A → B → C across three (hand-written) modules: `top` calls `mid.<B>`, `mid`
/// calls `leaf.<C>`. The linker composes them into one module; `top(n) = leaf(n)*2 + 10 + 1`.
#[test]
fn transitive_three_module_chain() {
    let mod_a = "\
(stmts
 (proc :top.0. (params (param :n.0 . (i +64))) (i +64) .
  (stmts .
   (ret (add (i +64) (call mid.0.modb n.0) 1)))))";
    let mod_b = "\
(stmts
 (proc :mid.0. (params (param :n.0 . (i +64))) (i +64) .
  (stmts .
   (ret (add (i +64) (call leaf.0.modc n.0) 10)))))";
    let mod_c = "\
(stmts
 (proc :leaf.0. (params (param :n.0 . (i +64))) (i +64) .
  (stmts .
   (ret (mul (i +64) n.0 2)))))";
    let linked = svm_leng::link_units(&[
        LengModule {
            stem: "moda",
            src: mod_a,
            names: &["top.0."],
        },
        LengModule {
            stem: "modb",
            src: mod_b,
            names: &["mid.0."],
        },
        LengModule {
            stem: "modc",
            src: mod_c,
            names: &["leaf.0."],
        },
    ])
    .unwrap_or_else(|e| panic!("link: {e}"));
    // top(n) = (leaf(n)*... ) : leaf(n)=2n; mid=2n+10; top=2n+11.
    assert_eq!(run(&linked, 0, &[4]), 19, "2*4 + 10 + 1");
    assert_eq!(run(&linked, 0, &[0]), 11);
}

#[test]
fn per_module_globals_relocate_without_aliasing() {
    // Each module has its own non-zero global; linking must place them in disjoint window regions so
    // each proc reads its *own*. In link-unit mode globals are addressed via `data.self`, which the
    // linker relocates — an absolute-offset unit would silently alias (both at offset 16 → one wins).
    let mod_a = "\
(stmts
 (gvar :g.0. . (i +64) 5)
 (proc :readG.0. . (i +64) . (stmts . (ret g.0.))))";
    let mod_b = "\
(stmts
 (gvar :h.0. . (i +64) 7)
 (proc :readH.0. . (i +64) . (stmts . (ret h.0.))))";
    let linked = svm_leng::link_units(&[
        LengModule {
            stem: "moda",
            src: mod_a,
            names: &["readG.0."],
        },
        LengModule {
            stem: "modb",
            src: mod_b,
            names: &["readH.0."],
        },
    ])
    .unwrap_or_else(|e| panic!("link: {e}"));
    assert_eq!(run(&linked, 0, &[]), 5, "module A reads its own g");
    assert_eq!(
        run(&linked, 1, &[]),
        7,
        "module B reads its own h (not A's g)"
    );
}

#[test]
fn cross_module_global_read_write() {
    // Module `w`'s proc reads and writes a `gvar` `g` *defined in module `s`*. The reference is a
    // `data.sym` the linker binds to `s`'s exported global — write-then-read-back through that
    // symbol must reach `s`'s storage, so `rw(v) = v`.
    let mod_s = "\
(stmts
 (gvar :g.0. . (i +64) 0))";
    let mod_w = "\
(stmts
 (proc :rw.0. (params (param :v.0 . (i +64))) (i +64) .
  (stmts .
   (asgn g.0.mods v.0)
   (ret g.0.mods))))";
    let linked = svm_leng::link_units(&[
        LengModule {
            stem: "modw",
            src: mod_w,
            names: &["rw.0."],
        },
        LengModule {
            stem: "mods",
            src: mod_s,
            names: &[], // data-only unit: it just defines (and exports) `g`
        },
    ])
    .unwrap_or_else(|e| panic!("link: {e}"));
    assert_eq!(run(&linked, 0, &[42]), 42, "write then read the external g");
    assert_eq!(run(&linked, 0, &[-5]), -5);
}

/// Real nimony: `moda`'s `bump` reads and writes `store.counter` across the module boundary — hexer
/// emits the reference as `counter.0.<store-stem>`. `store` defines `counter.0.` and exports it as a
/// data symbol; the linker binds `moda`'s `data.sym` to it. `bump()` increments 0 → 1.
#[test]
fn real_cross_module_global() {
    const MODA: &str = include_str!("fixtures/real_gref_moda.leng.nif");
    const STORE: &str = include_str!("fixtures/real_gref_store.leng.nif");
    let linked = svm_leng::link_units(&[
        LengModule {
            stem: "modywjwgs",
            src: MODA,
            names: &["bump.0."],
        },
        LengModule {
            stem: "sto0vp1px",
            src: STORE,
            names: &[],
        },
    ])
    .unwrap_or_else(|e| panic!("link: {e}"));
    assert_eq!(run(&linked, 0, &[]), 1, "bump(): counter 0 -> 1");
}

#[test]
fn cross_module_value_type_resolves() {
    // Module `u` declares a local of type `Pair.0.modp` — *defined in module `p`*. Unlike proc/data
    // symbols (resolved at link time), an aggregate type's layout is needed at *translate* time
    // (field offsets are baked into loads/stores), so `link_units` pools every unit's type defs
    // under their stem-suffixed global names before translating any. mk(v) = p.x + p.y = v + 7.
    let mod_p = "\
(stmts
 (type :Pair.0. . (object . (fld :x.0 . (i +64)) (fld :y.0 . (i +64)))))";
    let mod_u = "\
(stmts
 (proc :mk.0. (params (param :v.0 . (i +64))) (i +64) .
  (stmts .
   (var :p.0 . Pair.0.modp (oconstr Pair.0.modp (kv x.0 v.0) (kv y.0 7)))
   (ret (add (i +64) (dot p.0 x.0 0) (dot p.0 y.0 0))))))";
    let linked = svm_leng::link_units(&[
        LengModule {
            stem: "modu",
            src: mod_u,
            names: &["mk.0."],
        },
        LengModule {
            stem: "modp",
            src: mod_p,
            names: &[], // type-only unit: it just defines `Pair`
        },
    ])
    .unwrap_or_else(|e| panic!("link: {e}"));
    // mk is frame-needing (aggregate local): ($sp, v) -> i64.
    assert_eq!(run(&linked, 0, &[4096, 35]), 42, "p.x + p.y = 35 + 7");
    // Compiled *standalone*, the same unit fail-closes — the external layout exists only across
    // the link. No silent scalar-guess for a value of an unknown aggregate type.
    assert!(
        svm_leng::compile_object(&LengModule {
            stem: "modu",
            src: mod_u,
            names: &["mk.0."],
        })
        .is_err(),
        "standalone object with an external value type must fail closed"
    );
}

#[test]
fn nested_cross_module_types_resolve() {
    // `Outer.0.modp` embeds `Inner.0.` *by value*. The pooled export rewrites the field's type
    // name to its suffixed form (`Inner.0.modp`), so the importing unit resolves the nested layout
    // too: get(v) = pre + in.b = 9 + v.
    let mod_p = "\
(stmts
 (type :Inner.0. . (object . (fld :a.0 . (i +64)) (fld :b.0 . (i +64))))
 (type :Outer.0. . (object . (fld :pre.0 . (i +64)) (fld :in.0 . Inner.0.))))";
    let mod_u = "\
(stmts
 (proc :get.0. (params (param :v.0 . (i +64))) (i +64) .
  (stmts .
   (var :o.0 . Outer.0.modp .)
   (asgn (dot (dot o.0 in.0 0) b.0 0) v.0)
   (asgn (dot o.0 pre.0 0) 9)
   (ret (add (i +64) (dot o.0 pre.0 0) (dot (dot o.0 in.0 0) b.0 0))))))";
    let linked = svm_leng::link_units(&[
        LengModule {
            stem: "modu",
            src: mod_u,
            names: &["get.0."],
        },
        LengModule {
            stem: "modp",
            src: mod_p,
            names: &[],
        },
    ])
    .unwrap_or_else(|e| panic!("link: {e}"));
    assert_eq!(
        run(&linked, 0, &[4096, 33]),
        42,
        "9 + 33 through the nested field"
    );
}

#[test]
fn cross_module_sret_call_with_oconstr_arg() {
    // Regression (#760): an aggregate **rvalue** — an `(oconstr …)` literal — passed to a
    // cross-module **aggregate-returning** call. `caller` does `r = id((oconstr Pair 5 7))`; because
    // `r` is an aggregate, the call flows through the sret-import path (`call_import_sret`), whose
    // argument is the oconstr. That path handled aggregate *lvalues* (a var, by address) but not
    // aggregate *rvalues* — the oconstr fell through to `expr` and failed closed ("expression
    // `oconstr`"). It now constructs the literal into a temp and passes it by address, like every
    // other aggregate arg (matching the non-sret `call_import`). This was the dominant gap when
    // sweeping real nimony output — a `string` literal argument to a returning call (`concat`, `&`,
    // `$`, `write`) hits it on nearly every program.
    let mod_p = "\
(stmts
 (type :Pair.0. . (object . (fld :x.0 . (i +64)) (fld :y.0 . (i +64))))
 (proc :id.0. (params (param :p.0 . Pair.0.)) Pair.0. .
  (stmts . (ret p.0))))";
    let mod_u = "\
(stmts
 (proc :caller.0. (params) (i +64) .
  (stmts .
   (var :r.0 . Pair.0.modp (call id.0.modp (oconstr Pair.0.modp (kv x.0 5) (kv y.0 7))))
   (ret (add (i +64) (dot r.0 x.0 0) (dot r.0 y.0 0))))))";
    let linked = svm_leng::link_units(&[
        LengModule {
            stem: "modu",
            src: mod_u,
            names: &["caller.0."],
        },
        LengModule {
            stem: "modp",
            src: mod_p,
            names: &["id.0."],
        },
    ])
    .unwrap_or_else(|e| panic!("link: {e}"));
    // caller() = id((5,7)).x + .y = 12, on both engines (`run` asserts §9 parity).
    assert_eq!(run(&linked, 0, &[4096]), 12, "id((5,7)).x + .y");
}

/// Real nimony `greet(): string = "hello"` — the SSO literal — linked against a stand-in system
/// unit carrying the real `string` def under the real system stem. `string.0.sysvq0asl` resolves
/// from the *linked unit's* type def: no hand-supplied prelude (contrast `strings.rs`, which feeds
/// the same fixture through `translate_proc_with_types`). The ARC hooks `=wasMoved`/`=destroy`
/// bind to no-op stubs (an SSO literal owns no heap). greet writes its result through sret —
/// bytes = the packed literal, more = nil — identically on both engines.
#[test]
fn real_string_type_resolves_across_link() {
    const REAL: &str = include_str!("fixtures/real_string.leng.nif");
    let system_stub = "\
(stmts
 (type :string.0. . (object . (fld :bytes.0 . (u 64)) (fld :more.0 . (ptr (i +64)))))
 (proc :=wasMoved.2. (params (param :p.0 . (ptr (i +64)))) . . (stmts . (ret .)))
 (proc :=destroy.2. (params (param :s.0 . (ptr string.0.))) . . (stmts . (ret .))))";
    let linked = svm_leng::link_units(&[
        LengModule {
            stem: "strmodx",
            src: REAL,
            names: &["greet.0."],
        },
        LengModule {
            stem: "sysvq0asl",
            src: system_stub,
            names: &["=wasMoved.2.", "=destroy.2."],
        },
    ])
    .unwrap_or_else(|e| panic!("link: {e}"));
    svm_verify::verify_module(&linked).unwrap_or_else(|e| panic!("verify: {e:?}"));
    // greet (func 0) is frame-needing and aggregate-returning: ($sp, $sret) -> ().
    let (sp, sret) = (1024i64, 512usize);
    let seed = vec![0u8; 4096];
    let ivals = vec![Value::I64(sp), Value::I64(sret as i64)];
    let mut fuel = u64::MAX;
    let (ir, imem) = svm_interp::run_capture(&linked, 0, &ivals, &mut fuel, &seed);
    ir.expect("interp greet");
    let (_jout, jmem) =
        svm_jit::compile_and_run_capture(&linked, 0, &[sp, sret as i64], &seed).expect("jit greet");
    let n = imem.len().min(jmem.len());
    assert_eq!(imem[..n], jmem[..n], "§9 interp/JIT window parity");
    let bytes = u64::from_le_bytes(imem[sret..sret + 8].try_into().unwrap());
    let more = u64::from_le_bytes(imem[sret + 8..sret + 16].try_into().unwrap());
    assert_eq!(bytes, 122511465736197, "the packed \"hello\" SSO word");
    assert_eq!(more, 0, "short string: `more` is nil");
}

#[test]
fn unresolved_cross_module_call_is_fail_closed() {
    // `top` calls `missing.0.modx`, which no linked unit exports → a clean link error.
    let mod_a = "\
(stmts
 (proc :top.0. . (i +64) . (stmts . (ret (call missing.0.modx)))))";
    match svm_leng::link_units(&[LengModule {
        stem: "moda",
        src: mod_a,
        names: &["top.0."],
    }]) {
        Err(svm_leng::LengError::Malformed(_)) => {}
        other => panic!("expected a fail-closed link error, got {other:?}"),
    }
}

#[test]
fn unresolved_cross_module_global_is_fail_closed() {
    // A `data.sym` for a global no unit exports → a clean link error, never a wrong address.
    let mod_w = "\
(stmts
 (proc :rd.0. . (i +64) . (stmts . (ret missing.0.nowhere))))";
    match svm_leng::link_units(&[LengModule {
        stem: "modw",
        src: mod_w,
        names: &["rd.0."],
    }]) {
        Err(svm_leng::LengError::Malformed(_)) => {}
        other => panic!("expected a fail-closed link error, got {other:?}"),
    }
}

/// A cross-module call **through a funcref global** — the funcref counterpart of
/// `cross_module_global_read_write`. Module `s` defines a `proctype` gvar `hook`, a handler `dbl`,
/// and a setter that stores `ref.func dbl` into `hook`; module `w` calls `hook.<s>(x)`. A funcref
/// gvar is a *data* symbol, not a proc, so the call must lower to a `data.sym` load + `call_indirect`
/// (using the pooled proctype signature), not a proc import. `drive` sets the hook, then dispatches
/// through it: `drive(x) = dbl(x) = 2x`.
#[test]
fn cross_module_funcref_global_call() {
    let mod_s = "\
(stmts
 (type :IntFn.0. . (proctype . (params (param :x.0 . (i +64))) (i +64) (pragmas (nimcall))))
 (gvar :hook.0. . IntFn.0.)
 (proc :dbl.0. (params (param :x.0 . (i +64))) (i +64) .
  (stmts . (ret (mul (i +64) x.0 2))))
 (proc :sethook.0. (params) . .
  (stmts . (asgn hook.0. dbl.0.))))";
    let mod_w = "\
(stmts
 (proc :drive.0. (params (param :x.0 . (i +64))) (i +64) .
  (stmts .
   (call sethook.0.mods)
   (ret (call hook.0.mods x.0)))))";
    let linked = svm_leng::link_units(&[
        LengModule {
            stem: "modw",
            src: mod_w,
            names: &["drive.0."],
        },
        LengModule {
            stem: "mods",
            src: mod_s,
            names: &["sethook.0.", "dbl.0."],
        },
    ])
    .unwrap_or_else(|e| panic!("link: {e}"));
    // drive = module w's first proc → func 0.
    assert_eq!(
        run(&linked, 0, &[21]),
        42,
        "drive sets hook=dbl, calls hook(21)"
    );
    assert_eq!(run(&linked, 0, &[-5]), -10);
}

/// A cross-module call to a **frame-needing proc**. Module `s`'s `count` takes `(addr i)` of a local,
/// so its emitted signature carries a leading `$sp`; module `w`'s `drive` calls `count.<s>(n)`. The
/// caller must hand down that `$sp` — but `call_import` derives an import's signature from the *args*
/// and can't see the hidden frame param, so the linker runs a **whole-program frame fixpoint**: it
/// pools every unit's own frame-needing procs and propagates transitively across module boundaries.
/// `count` is frame-needing on its own; `drive` becomes frame-needing because it calls `count`, so it
/// too gains an `$sp` (to have one to pass down). `drive(n) = count(n) = n`.
#[test]
fn cross_module_frame_needing_call() {
    let mod_s = "\
(stmts
 (proc :incp.0. (params (param :p.0 . (ptr (i +64)))) (void) .
  (stmts . (asgn (deref p.0) (add (i +64) (deref p.0) 1))))
 (proc :count.0. (params (param :n.0 . (i +64))) (i +64) .
  (stmts .
   (var :i.0 . (i +64) 0)
   (while (lt i.0 n.0) (stmts . (call incp.0. (addr i.0))))
   (ret i.0))))";
    let mod_w = "\
(stmts
 (proc :drive.0. (params (param :n.0 . (i +64))) (i +64) .
  (stmts . (ret (call count.0.mods n.0)))))";
    let linked = svm_leng::link_units(&[
        LengModule {
            stem: "modw",
            src: mod_w,
            names: &["drive.0."],
        },
        LengModule {
            stem: "mods",
            src: mod_s,
            names: &["count.0.", "incp.0."],
        },
    ])
    .unwrap_or_else(|e| panic!("link: {e}"));
    // drive (func 0) is frame-needing via the fixpoint → its signature gained a leading `$sp`.
    assert_eq!(
        (linked.funcs[0].params.len(), linked.funcs[0].results.len()),
        (2, 1),
        "drive should be lowered as (sp, n) -> (i64)"
    );
    // Pass a window offset as drive's own frame `$sp`; it hands a fresh sub-frame to count.
    let sp = 8192;
    for n in [0, 1, 5, 50] {
        assert_eq!(
            run(&linked, 0, &[sp, n]),
            n,
            "drive({n}) = count({n}) = {n}"
        );
    }
}

/// A funcref global with a **static proc initializer** — `var hook = dbl` — materialized by the
/// linker. Module `s` declares `hook: IntFn` initialized to `dbl` and never assigns it at runtime;
/// module `w`'s `drive` calls through `hook`. A funcref gvar's initial value is a *function index*
/// the linker only knows once it assigns func bases, so `svm-leng` emits a `data.funcref` relocation
/// (the funcref twin of `data.ptr`) that `link` resolves into the gvar's data slot — the value
/// `ref.func dbl` would yield. Unlike `cross_module_funcref_global_call` (which stores the funcref at
/// runtime), nothing here writes `hook`, so `drive(n) = dbl(n) = 2n` proves the initializer was
/// materialized statically. This is what lets a real stdlib program's `oomHandler`/`gExitFlush`
/// funcref gvars work without the system `ini` (which never writes them).
#[test]
fn funcref_global_static_initializer() {
    let mod_s = "\
(stmts
 (type :IntFn.0. . (proctype . (params (param :x.0 . (i +64))) (i +64) (pragmas (nimcall))))
 (gvar :hook.0. . IntFn.0. dbl.0.)
 (proc :dbl.0. (params (param :x.0 . (i +64))) (i +64) .
  (stmts . (ret (mul (i +64) x.0 2)))))";
    let mod_w = "\
(stmts
 (proc :drive.0. (params (param :n.0 . (i +64))) (i +64) .
  (stmts . (ret (call hook.0.mods n.0)))))";
    let linked = svm_leng::link_units(&[
        LengModule {
            stem: "modw",
            src: mod_w,
            names: &["drive.0."],
        },
        LengModule {
            stem: "mods",
            src: mod_s,
            names: &["dbl.0."],
        },
    ])
    .unwrap_or_else(|e| panic!("link: {e}"));
    // No setter runs — `hook` holds `dbl`'s index purely from the materialized initializer.
    assert_eq!(
        run(&linked, 0, &[21]),
        42,
        "drive calls the materialized hook = dbl"
    );
    assert_eq!(run(&linked, 0, &[-5]), -10);
}
