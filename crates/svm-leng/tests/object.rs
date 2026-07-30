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

/// A separately-produced **runtime object** (not from nimony): the four pure seq/openArray ops real
/// `sumSeq` imports, hand-authored as svm-text. This stands in for what a real compiled `system`
/// module (or a C-runtime `.svmo`) will provide — the point is only that it's a *different producer*
/// whose object links against the nimony one through the shared format.
const RUNTIME_TEXT: &str = "\
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
}";

fn runtime_shim_index(name: &str) -> u32 {
    if name.contains("toOpenArray") {
        0
    } else if name.starts_with("len") {
        1
    } else if name.contains("5B") {
        2 // the `[]` operator (`\5B\5D…`)
    } else if name.starts_with("inc") {
        3
    } else {
        panic!("no runtime op for import {name}");
    }
}

#[test]
fn cross_producer_link_through_svmo() {
    // A nimony `.svmo` (sumSeq) links against a *separately produced* runtime `.svmo` — both binary
    // objects, joined only through the format — and runs. This is the composition the narrow waist
    // exists for: the runtime object will one day be the real compiled `system` module.
    const R: &str = include_str!("fixtures/real_seq_loop.leng.nif");
    let nim_obj = svm_leng::compile_object(&LengModule {
        stem: "seqmod",
        src: R,
        names: &["sumSeq.0."],
    })
    .unwrap();
    // The nimony object declares its seq-op imports; the runtime object must export exactly those.
    let nim = decode_unit(&nim_obj).expect("decode nimony object");
    let mut runtime = svm_text::parse_module(RUNTIME_TEXT).expect("runtime text");
    runtime.exports = nim
        .imports
        .iter()
        .map(|imp| Export {
            name: imp.name.clone(),
            func: runtime_shim_index(&imp.name),
        })
        .collect();
    let rt_obj = encode_unit(&runtime); // the runtime, as its own `.svmo`

    // Link the two objects through the format and run sumSeq over a harness-built seq.
    let linked = svm_ir::link(&[
        unit_of(decode_unit(&nim_obj).unwrap()),
        unit_of(decode_unit(&rt_obj).unwrap()),
    ])
    .expect("link nimony + runtime objects");
    svm_verify::verify_module(&linked).expect("verify");

    let (seq, data, sp) = (512usize, 640usize, 1024i64);
    let mut seed = vec![0u8; 4096];
    seed[seq..seq + 8].copy_from_slice(&3i64.to_le_bytes());
    seed[seq + 8..seq + 16].copy_from_slice(&(data as i64).to_le_bytes());
    for (i, v) in [10i64, 20, 30].iter().enumerate() {
        seed[data + i * 8..data + i * 8 + 8].copy_from_slice(&v.to_le_bytes());
    }
    // sumSeq is func 0 of the linked module: (sp, seq_addr) -> Σ = 60.
    let mut fuel = u64::MAX;
    let (ir, _) = svm_interp::run_capture(
        &linked,
        0,
        &[Value::I64(sp), Value::I64(seq as i64)],
        &mut fuel,
        &seed,
    );
    assert_eq!(ir.expect("interp").as_slice(), &[Value::I64(60)]);
    let (jout, _) =
        svm_jit::compile_and_run_capture(&linked, 0, &[sp, seq as i64], &seed).expect("jit");
    match jout {
        svm_jit::JitOutcome::Returned(v) => assert_eq!(v, vec![60], "§9 parity"),
        o => panic!("jit: {o:?}"),
    }
}
