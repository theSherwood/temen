//! **The committed `nim_hello.temen` playground asset, run in-sandbox** (NIM.md — the "run real Nim"
//! card). A real Nim program (`demos/nim_hello/hello.nim` — `write(stdout, "hello, temen\n")`) compiled
//! all the way to a runnable powerbox module: `nimony c` → `temen_leng::link_nim_powerbox` (the nim→
//! powerbox bottom-edge bridge) → verified + `encode_module`. Built by
//! `cargo run -p temen-run --example build_nim_hello_temen` (needs the nimony toolchain), shipped
//! prebuilt under `browser/web/assets/` — the same bytes the playground card loads.
//!
//! This gate loads the committed bytes, decodes + **re-verifies** them (the fail-closed TCB floor —
//! the shipped artifact must be a valid verified module), and **runs** it under the standard
//! `temen_run::run_powerbox` (the engine the browser's `temen_run_onramp` wraps), asserting it prints
//! `hello, temen` to the powerbox stdout — no toolchain needed, so it runs in the normal `check` job.
//!
//! **Code-coupled asset (the chibicc/Postgres lane).** If an IR/ABI/encoder change stops the committed
//! module decoding, verifying, or running, this test fails the PR that caused the drift — regenerate
//! the asset with `build_nim_hello_temen` and commit it (see `demos/nim_hello/README.md`).

#![cfg(target_os = "linux")]

const ASSET: &[u8] = include_bytes!("../../../browser/web/assets/nim_hello.temen");

#[test]
fn committed_asset_decodes_and_verifies() {
    let module = temen_encode::decode_module(ASSET).expect("decode nim_hello.temen");
    temen_verify::verify_module(&module).expect("verify nim_hello.temen (the trusted floor)");
    assert!(
        temen_run::is_named_powerbox_entry(&module),
        "the asset is a powerbox entry (paramless func-0 `_start`)"
    );
}

#[test]
fn committed_asset_prints_hello() {
    // Run the real compiled Nim program under the standard powerbox — the host grants stdout, the
    // program's `write` reaches it through the bridged STREAM cap, and the output is the real thing.
    let module = temen_encode::decode_module(ASSET).expect("decode nim_hello.temen");
    let run = temen_run::run_powerbox(&module, &[]).expect("run_powerbox nim_hello");
    assert_eq!(
        run.stdout, b"hello, temen\n",
        "real Nim program prints via the powerbox"
    );
}
