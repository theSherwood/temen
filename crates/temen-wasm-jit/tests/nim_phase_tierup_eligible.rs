//! **#1025 slice 3a.3 de-risk — do the real nim front-end phases tier up to emitted wasm?** The browser
//! Path-1 tier-up (`temen_op13jit_step`) runs an op-13 phase child on **emitted wasm** only if
//! `emit_for_run` gets a `WasmDriven` artifact from `compile_jit_paged(Shape::Batch{0})`; otherwise it
//! declines to the interpreter (no speed-up). Tier-up is proven for **nifler**; **nimsem** and **hexer**
//! (the ~180s dominators, deferred as the "#816 half" for their 256 MiB paged carves) were never gated.
//! Their whole-front-end tier-up is worthless if they decline — so this pins the gating question with a
//! headless check that mirrors `browser/src/lib.rs::emit_for_run` exactly: `outline_cap_calls` then
//! `compile_jit_paged(Batch{0}, page_log2)`, asserting `WasmDriven`.
//!
//! Runs the **committed child-entry assets** (`nifler_ce`/`nimsem_ce`/`hexer_ce`), so it needs no build
//! toolchain — only `gzip`. If a frontend/IR/encoder change makes a phase stop emitting (a new
//! setjmp/concurrency/region-op in its reachable set), this fails, flagging the tier-up regression before
//! it silently drops the card back to the interpreter. Gated Linux + gzip.

#![cfg(target_os = "linux")]

use std::io::Write;
use std::process::{Command, Stdio};

use temen_wasm_jit::{compile_jit_paged, outline_cap_calls, DriveMode, Shape};

const NIFLER_CE_GZ: &[u8] = include_bytes!("../../temen-run/demos/nifler_temen/nifler_ce.temen.gz");
const NIMSEM_CE_GZ: &[u8] =
    include_bytes!("../../temen-run/demos/nim_frontend/fixtures/nimsem_ce.temen.gz");
const HEXER_CE_GZ: &[u8] =
    include_bytes!("../../temen-run/demos/nim_frontend/fixtures/hexer_ce.temen.gz");

fn inflate(gz: &[u8]) -> Option<Vec<u8>> {
    let mut c = Command::new("gzip")
        .args(["-dc"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let mut stdin = c.stdin.take().unwrap();
    let gz = gz.to_vec();
    let w = std::thread::spawn(move || {
        let _ = stdin.write_all(&gz);
    });
    let out = c.wait_with_output().ok()?;
    w.join().ok()?;
    out.status.success().then_some(out.stdout)
}

/// Mirror `emit_for_run`'s emit core for `gz`: decode, `outline_cap_calls`, then
/// `compile_jit_paged(Batch{0})` at the host page size. Returns the resulting `DriveMode`.
fn drive_mode_of(gz: &[u8]) -> Option<DriveMode> {
    let bytes = inflate(gz)?;
    let mut module = temen_encode::decode_module(&bytes).expect("decode phase asset");
    // func 0 is the child entry (`starter -> i64 status`) — the shape the op-13 tier-up path emits.
    assert_eq!(
        module.funcs[0]
            .params
            .len()
            .max(module.funcs[0].results.len()),
        1,
        "expected a child-entry phase asset"
    );
    outline_cap_calls(&mut module);
    let page_log2 = temen_interp::host_page_size().trailing_zeros() as u8;
    let artifact = compile_jit_paged(&module, Shape::Batch { entry: 0 }, false, page_log2)
        .expect("compile_jit_paged");
    Some(artifact.drive)
}

#[test]
fn nifler_child_entry_emits_wasm_driven() {
    let Some(drive) = drive_mode_of(NIFLER_CE_GZ) else {
        eprintln!("SKIP: gzip unavailable");
        return;
    };
    // The known-good positive control: nifler already tiers up in the browser (Path 1, proven).
    assert!(
        matches!(drive, DriveMode::WasmDriven { .. }),
        "nifler_ce must emit WasmDriven (it tiers up today), got {drive:?}"
    );
}

#[test]
fn nimsem_child_entry_emits_wasm_driven() {
    let Some(drive) = drive_mode_of(NIMSEM_CE_GZ) else {
        eprintln!("SKIP: gzip unavailable");
        return;
    };
    // The gating #816 question: nimsem is the heaviest phase. If this declines to InterpDriven, the
    // whole-front-end browser tier-up cannot beat the ~180s baseline and the emit gap is the real work.
    assert!(
        matches!(drive, DriveMode::WasmDriven { .. }),
        "nimsem_ce must emit WasmDriven for the browser tier-up to help; got {drive:?} \
         (a reachable setjmp / concurrency / region-op declines it to the interpreter)"
    );
}

#[test]
fn hexer_child_entry_emits_wasm_driven() {
    let Some(drive) = drive_mode_of(HEXER_CE_GZ) else {
        eprintln!("SKIP: gzip unavailable");
        return;
    };
    assert!(
        matches!(drive, DriveMode::WasmDriven { .. }),
        "hexer_ce must emit WasmDriven for the browser tier-up to help; got {drive:?}"
    );
}
