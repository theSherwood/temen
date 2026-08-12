//! **W5 capstone: the committed `svm-leng.svmb` self-host asset, run over real hexer Leng files**
//! (NIM.md §3e). The leng analog of `browser/tests/chibicc_selfhost_asset.rs`.
//!
//! `svm-leng.svmb` is the real `svm-leng` translator, compiled to a verified SVM module and shipped
//! prebuilt (`crates/svm-run/demos/leng_selfhost/build_leng_svmb.sh` — `-Z build-std` → on-ramp →
//! `prep_svmb`). This test loads the committed bytes, decodes + **re-verifies** them (the fail-closed
//! TCB floor — the shipped artifact must be a valid verified module), then runs it **in-sandbox** over
//! each `corpus/*.leng.nif` — verbatim `hexer c` output from real nimony source — feeding the Leng file
//! on stdin and asserting the emitted SVM text is **byte-identical to the same translation run
//! host-side** (`svm_leng::translate_to_text`, the §18 svm == native oracle).
//!
//! **Code-coupled asset (the Postgres/chibicc lane).** The oracle is the in-tree `svm-leng`, so this
//! gate needs no build toolchain: if an IR/ABI/encoder or `svm-leng` change makes the committed asset
//! stop matching native, this test fails the PR that caused the drift — regenerate the asset with
//! `build_leng_svmb.sh` and commit it (see `crates/svm-run/demos/leng_selfhost/README.md`).

#![cfg(target_os = "linux")]

use svm_run::{Backend, Limits, Outcome, RunConfig};

const ASSET: &[u8] = include_bytes!("../demos/leng_selfhost/svm-leng.svmb");

/// The real hexer Leng files the asset is exercised over, and the exit code the guest returns for
/// each (`0` = translated, wrote SVM text to stdout).
const CORPUS: &[&str] = &[
    "system_arc.leng.nif",
    "controlflow.leng.nif",
    "goto.leng.nif",
];

fn corpus_bytes(name: &str) -> Vec<u8> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/demos/leng_selfhost/corpus/");
    std::fs::read(format!("{p}{name}")).unwrap_or_else(|e| panic!("read corpus {name}: {e}"))
}

#[test]
fn committed_asset_decodes_and_verifies() {
    // The shipped bytes are a well-formed, re-verifiable SVM module (DESIGN.md §2a) — the same
    // fail-closed floor `prep_svmb` applies at build time, re-checked on the committed artifact.
    let module = svm_encode::decode_module(ASSET).expect("decode svm-leng.svmb");
    svm_verify::verify_module(&module).expect("verify svm-leng.svmb (the trusted floor)");
    // Sanity: it's the whole translator (float-capable std svm-leng), not a stub.
    assert!(
        module.funcs.len() > 100,
        "expected the full svm-leng translator, got {} funcs",
        module.funcs.len()
    );
}

#[test]
fn asset_translates_real_hexer_leng_byte_identical_to_native() {
    let module = svm_encode::decode_module(ASSET).expect("decode svm-leng.svmb");
    svm_verify::verify_module(&module).expect("verify svm-leng.svmb");

    for &name in CORPUS {
        let leng = corpus_bytes(name);
        // Host-side oracle: the in-tree translator on the same bytes.
        let native =
            svm_leng::translate_to_text(std::str::from_utf8(&leng).expect("corpus is UTF-8"))
                .unwrap_or_else(|e| panic!("native translate {name}: {e}"));

        // In-sandbox: feed the Leng file on stdin, capture stdout. The asset is a powerbox program
        // (synthesized `_start`), so it runs through the fixed §3e powerbox on the tree-walker.
        let cfg = RunConfig {
            limits: Limits {
                fuel: None,
                deadline: None,
                max_fibers: 0,
                max_vcpus: 0,
            },
            stdin: leng,
            memory_size_log2: None,
            args: vec![],
            env: vec![],
            ..RunConfig::default()
        };
        let run = svm_run::instantiate(module.clone())
            .expect("instantiate svm-leng.svmb")
            .run(Backend::TreeWalk, &cfg)
            .unwrap_or_else(|e| panic!("run svm-leng.svmb over {name}: {e}"));

        assert!(
            matches!(run.outcome, Outcome::Returned(_)),
            "{name}: guest did not return cleanly: {:?}",
            run.outcome
        );
        assert_eq!(
            String::from_utf8_lossy(&run.stdout),
            native,
            "{name}: in-sandbox svm-leng translation must match the native oracle \
             (asset stale? regenerate with build_leng_svmb.sh)"
        );
    }
}
