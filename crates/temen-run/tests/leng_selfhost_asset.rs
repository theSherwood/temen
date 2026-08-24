//! **W5 capstone: the committed `temen-leng.temen` self-host asset, run over real hexer Leng files**
//! (NIM.md §3e). The leng analog of `browser/tests/chibicc_selfhost_asset.rs`.
//!
//! `temen-leng.temen` is the real `temen-leng` translator, compiled to a verified Temen module and shipped
//! prebuilt (`crates/temen-run/demos/leng_selfhost/build_leng_temen.sh` — `-Z build-std` → on-ramp →
//! `prep_temen`). This test loads the committed bytes, decodes + **re-verifies** them (the fail-closed
//! TCB floor — the shipped artifact must be a valid verified module), then runs it **in-sandbox** over
//! each `corpus/*.leng.nif` — verbatim `hexer c` output from real nimony source — feeding the Leng file
//! on stdin and asserting the emitted Temen text is **byte-identical to the same translation run
//! host-side** (`temen_leng::translate_to_text`, the §18 temen == native oracle).
//!
//! **Code-coupled asset (the Postgres/chibicc lane).** The oracle is the in-tree `temen-leng`, so this
//! gate needs no build toolchain: if an IR/ABI/encoder or `temen-leng` change makes the committed asset
//! stop matching native, this test fails the PR that caused the drift — regenerate the asset with
//! `build_leng_temen.sh` and commit it (see `crates/temen-run/demos/leng_selfhost/README.md`).

#![cfg(target_os = "linux")]

use temen_run::{Backend, Limits, Outcome, RunConfig};

const ASSET: &[u8] = include_bytes!("../demos/leng_selfhost/temen-leng.temen");

/// The real hexer Leng files the asset is exercised over, and the exit code the guest returns for
/// each (`0` = translated, wrote Temen text to stdout).
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
    // The shipped bytes are a well-formed, re-verifiable Temen module (DESIGN.md §2a) — the same
    // fail-closed floor `prep_temen` applies at build time, re-checked on the committed artifact.
    let module = temen_encode::decode_module(ASSET).expect("decode temen-leng.temen");
    temen_verify::verify_module(&module).expect("verify temen-leng.temen (the trusted floor)");
    // Sanity: it's the whole translator (float-capable std temen-leng), not a stub.
    assert!(
        module.funcs.len() > 100,
        "expected the full temen-leng translator, got {} funcs",
        module.funcs.len()
    );
}

#[test]
fn asset_translates_real_hexer_leng_byte_identical_to_native() {
    let module = temen_encode::decode_module(ASSET).expect("decode temen-leng.temen");
    temen_verify::verify_module(&module).expect("verify temen-leng.temen");

    for &name in CORPUS {
        let leng = corpus_bytes(name);
        // Host-side oracle: the in-tree translator on the same bytes.
        let native =
            temen_leng::translate_to_text(std::str::from_utf8(&leng).expect("corpus is UTF-8"))
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
        let run = temen_run::instantiate(module.clone())
            .expect("instantiate temen-leng.temen")
            .run(Backend::TreeWalk, &cfg)
            .unwrap_or_else(|e| panic!("run temen-leng.temen over {name}: {e}"));

        assert!(
            matches!(run.outcome, Outcome::Returned(_)),
            "{name}: guest did not return cleanly: {:?}",
            run.outcome
        );
        assert_eq!(
            String::from_utf8_lossy(&run.stdout),
            native,
            "{name}: in-sandbox temen-leng translation must match the native oracle \
             (asset stale? regenerate with build_leng_temen.sh)"
        );
    }
}
