//! `.svmo` object-dialect tests (NIM.md W2 — the narrow waist). `svm_leng::compile_object` emits a
//! nimony module as a **binary link object** (the counterpart of `svm-llvm-translate -o out.svmo`):
//! a relocatable unit with its procs exported in-band. These objects link through the *same* format
//! and `svm_ir::link` any other producer uses — so a nimony object composes with objects a different
//! frontend emitted. That cross-producer composition is the whole reason to route linking through
//! the format rather than hand-built in-process `LinkUnit`s.

use svm_encode::{decode_unit, encode_unit};
use svm_interp::Value;
use svm_ir::{Export, LinkUnit, Module};
use svm_leng::{LengModule, WholeModule};

/// Build a `LinkUnit` from a decoded object's in-band export tables — the exact conversion
/// `svm-run --link` does.
fn unit_of(m: Module) -> LinkUnit {
    let exports = m.exports.iter().map(|e| (e.name.clone(), e.func)).collect();
    let data_exports = m
        .data_exports
        .iter()
        .map(|e| (e.name.clone(), e.offset))
        .collect();
    LinkUnit {
        module: m,
        exports,
        data_exports,
    }
}

#[test]
fn nimony_object_round_trips_and_runs() {
    // Compile a real 2-module nimony program to `.svmo` objects, then link them *only* through the
    // binary format (decode → link) — no in-memory `Module` shortcut. `useit(5) = helper(5)+1 = 16`.
    const MODA: &str = include_str!("fixtures/real_moda.leng.nif");
    const MODB: &str = include_str!("fixtures/real_modb.leng.nif");
    let a_obj = svm_leng::compile_object(&LengModule {
        stem: "modywjwgs",
        src: MODA,
        names: &["useit.0."],
    })
    .unwrap();
    let b_obj = svm_leng::compile_object(&LengModule {
        stem: "modwru7vt1",
        src: MODB,
        names: &["helper.0."],
    })
    .unwrap();
    // The object carries the export in-band (so any linker consumer finds `helper`).
    let b_decoded = decode_unit(&b_obj).expect("decode modb.svmo");
    assert!(
        b_decoded
            .exports
            .iter()
            .any(|e| e.name == "helper.0.modwru7vt1"),
        "helper exported in-band: {:?}",
        b_decoded.exports
    );
    let linked = svm_ir::link(&[unit_of(decode_unit(&a_obj).unwrap()), unit_of(b_decoded)])
        .expect("link objects");
    svm_verify::verify_module(&linked).expect("verify");
    let mut fuel = u64::MAX;
    let r = svm_interp::run(&linked, 0, &[Value::I64(5)], &mut fuel).unwrap();
    assert_eq!(r.as_slice(), &[Value::I64(16)]);
}

/// A whole-module object exposes its **`exportc`** symbols under their C names — the conventional
/// entry points a host / `svm-run --link` binds to. Real `moda` (main module) marks its C `main`
/// `(exportc "main")` and its `cmdCount`/`cmdLine`/`nimEnviron` gvars `(exportc "…")`; the compiled
/// object carries all of them in-band, alongside the mangled Leng names, so the program's C-ABI
/// surface is findable (NIM.md W2, Path A).
#[test]
fn whole_object_exposes_exportc_c_names() {
    const MODA: &str = include_str!("fixtures/real_moda.leng.nif");
    let obj = svm_leng::compile_whole_object(&WholeModule {
        stem: "modywjwgs",
        src: MODA,
    })
    .expect("compile whole moda object");
    let m = decode_unit(&obj).expect("decode moda.svmo");
    // The C `main` is exported under its exportc name (and still under the mangled `main.0.<stem>`).
    assert!(
        m.exports.iter().any(|e| e.name == "main"),
        "exportc C `main` present: {:?}",
        m.exports.iter().map(|e| &e.name).collect::<Vec<_>>()
    );
    assert!(
        m.exports.iter().any(|e| e.name == "`main.0.modywjwgs"),
        "mangled main still present too"
    );
    // The exportc gvars are exposed as C-named data symbols.
    for g in ["cmdCount", "cmdLine", "nimEnviron"] {
        assert!(
            m.data_exports.iter().any(|e| e.name == g),
            "exportc gvar `{g}` present as a data export: {:?}",
            m.data_exports.iter().map(|e| &e.name).collect::<Vec<_>>()
        );
    }
}

#[test]
fn cross_producer_link_through_svmo() {
    // A nimony `.svmo` links against a **separately-produced, non-nimony** runtime `.svmo` through the
    // shared binary format — the cross-*producer* composition the narrow waist exists for (the runtime
    // object will one day be the real compiled `system` module). A minimal nimony proc `dbl` calls the
    // cross-module `ext`, which no unit here defines, so it compiles to an import; a hand-authored
    // runtime object (a different producer) exports `ext` — doubling. Linked and run: `dbl(21) = 42`.
    let nim_src = "\
(stmts
 (proc :dbl.0. (params (param :x.0 . (i +64))) (i +64) .
  (stmts .
   (ret (call ext.0.rt x.0)))))";
    let nim_obj = svm_leng::compile_object(&LengModule {
        stem: "seqmod",
        src: nim_src,
        names: &["dbl.0."],
    })
    .unwrap();
    let nim = decode_unit(&nim_obj).expect("decode nimony object");
    // The non-nimony runtime object: one func doubling its arg, exported under the nimony import name.
    let mut runtime = svm_text::parse_module(
        "func (i64) -> (i64) {\nblock 0 (v0: i64) {\n  v1 = i64.const 2\n  v2 = i64.mul v0 v1\n  return v2\n  }\n}",
    )
    .expect("runtime text");
    runtime.exports = nim
        .imports
        .iter()
        .map(|imp| Export {
            name: imp.name.clone(),
            func: 0,
        })
        .collect();
    let rt_obj = encode_unit(&runtime); // the runtime, as its own `.svmo`

    // Link the two objects through the format and run.
    let linked = svm_ir::link(&[
        unit_of(decode_unit(&nim_obj).unwrap()),
        unit_of(decode_unit(&rt_obj).unwrap()),
    ])
    .expect("link nimony + runtime objects");
    svm_verify::verify_module(&linked).expect("verify");
    // dbl is func 0: (x) -> 2x.
    let mut fuel = u64::MAX;
    let ir = svm_interp::run(&linked, 0, &[Value::I64(21)], &mut fuel).unwrap();
    assert_eq!(ir.as_slice(), &[Value::I64(42)]);
    match svm_jit::compile_and_run(&linked, 0, &[21]).expect("jit") {
        svm_jit::JitOutcome::Returned(v) => assert_eq!(v, vec![42], "§9 parity"),
        o => panic!("jit: {o:?}"),
    }
}
