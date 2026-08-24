//! **The committed `nifler.temen.gz` playground asset, run in-sandbox** (NIM.md §3c/§3e — the "compile
//! Nim in the browser" front-end card, slice 4). `nifler` — the first real nimony compiler phase (Nim
//! source → parsed NIF) — is a Nim program on-ramped to a verified Temen module through the LLVM/C
//! on-ramp (slice 1) and shipped **gzipped** under `browser/web/assets/` (~3.8 MB vs ~17.7 MB raw; the
//! browser inflates it client-side, this gate via `gzip -dc`). Built by
//! `crates/temen-run/demos/nifler_temen/build_nifler_temen.sh` (needs the nimony toolchain).
//!
//! This gate inflates the committed bytes, decodes + **re-verifies** them (the fail-closed TCB floor —
//! the shipped artifact must be a valid verified module), then runs it **in-sandbox** over each
//! `inputs/*.nim`: it seeds the source as `/in.nim` on an in-memory `fs` cap, runs
//! `nifler p /in.nim /out.nif`, reads the emitted `.p.nif` back out, and asserts it is **byte-identical
//! to the committed `expected/*.p.nif`** — which is verbatim native-`nifler` output (checked at commit
//! time; the on-ramp'd nifler is byte-identical to native, slice 1).
//!
//! **Code-coupled asset (the chibicc/Postgres lane).** The oracle is the committed fixture, so this
//! gate needs no build toolchain (only `gzip`, standard on Linux): if an IR/ABI/encoder change makes
//! the committed asset stop decoding, verifying, or producing the same NIF, this test fails the PR that
//! caused the drift — regenerate the asset + fixtures with `build_nifler_temen.sh` and commit them (see
//! `crates/temen-run/demos/nifler_temen/README.md`).

#![cfg(target_os = "linux")]

use std::io::Write;
use std::process::{Command, Stdio};

use temen_run::{Backend, HostCap, Limits, Outcome, RunConfig};

/// The gzipped module the browser card loads — the exact bytes shipped under `browser/web/assets/`.
const ASSET_GZ: &[u8] = include_bytes!("../../../browser/web/assets/nifler.temen.gz");

/// The Nim inputs the asset is exercised over, paired with their committed native-`nifler` `.p.nif`.
const CORPUS: &[(&str, &str)] = &[
    (
        include_str!("../demos/nifler_temen/inputs/basic.nim"),
        include_str!("../demos/nifler_temen/expected/basic.p.nif"),
    ),
    (
        include_str!("../demos/nifler_temen/inputs/control.nim"),
        include_str!("../demos/nifler_temen/expected/control.p.nif"),
    ),
];

/// Inflate the committed gzip asset to the raw `.temen` bytes via `gzip -dc` (Linux-only, like the
/// browser's `DecompressionStream`). Returns `None` if `gzip` is unavailable so the gate can skip
/// rather than fail spuriously on a host without it. The stdin write runs on its own thread while the
/// main thread drains stdout — the inflated `.temen` (~17.7 MB) overflows the OS pipe buffer, so a
/// write-all-then-read would deadlock (gzip blocks on a full stdout pipe, we block on the stdin write).
fn inflate_asset() -> Option<Vec<u8>> {
    let mut child = Command::new("gzip")
        .args(["-dc"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let mut stdin = child.stdin.take().expect("gzip stdin");
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(ASSET_GZ); // drop closes the pipe (EOF for gzip)
    });
    let out = child.wait_with_output().expect("gzip -dc");
    writer.join().expect("stdin writer thread");
    out.status.success().then_some(out.stdout)
}

#[test]
fn committed_asset_decodes_and_verifies() {
    let Some(temen) = inflate_asset() else {
        eprintln!("SKIP: gzip unavailable to inflate nifler.temen.gz");
        return;
    };
    // The shipped bytes are a well-formed, re-verifiable Temen module (DESIGN.md §2a) — the same
    // fail-closed floor `prep_temen` applies at build time, re-checked on the committed artifact.
    let module = temen_encode::decode_module(&temen).expect("decode nifler.temen");
    temen_verify::verify_module(&module).expect("verify nifler.temen (the trusted floor)");
    // Sanity: it's the whole nifler phase (Nim program + its stdlib), not a stub.
    assert!(
        module.funcs.len() > 100,
        "expected the full nifler phase, got {} funcs",
        module.funcs.len()
    );
}

#[test]
fn asset_parses_nim_byte_identical_to_native_nifler() {
    let Some(temen) = inflate_asset() else {
        eprintln!("SKIP: gzip unavailable to inflate nifler.temen.gz");
        return;
    };
    let module = temen_encode::decode_module(&temen).expect("decode nifler.temen");
    temen_verify::verify_module(&module).expect("verify nifler.temen");

    for (src, expected) in CORPUS {
        // Seed the source as `in.nim` (nifler embeds the portable path `in.nim` in the NIF header, so
        // the emitted bytes are input-name-independent); a cross-domain shared memfs lets us read the
        // emitted `.p.nif` back out after the run (the handle observes the store the guest writes).
        let (factory, handle) = temen_run::fs::mem_fs_shared_factory(
            vec![("in.nim".into(), src.as_bytes().to_vec())],
            vec![],
        );
        let fs_cap = HostCap::host_proc(0, factory);

        let cfg = RunConfig {
            limits: Limits {
                fuel: None,
                deadline: None,
                max_fibers: 0,
                max_vcpus: 0,
            },
            stdin: vec![],
            memory_size_log2: None,
            // nifler's parse command: `nifler p <in> <out>` (bridge.nim's `parse` action).
            args: vec![
                b"nifler".to_vec(),
                b"p".to_vec(),
                b"/in.nim".to_vec(),
                b"/out.nif".to_vec(),
            ],
            env: vec![],
            ..RunConfig::default()
        };
        let run = temen_run::instantiate(module.clone())
            .expect("instantiate nifler.temen")
            .run_with_caps(Backend::TreeWalk, &cfg, &[("fs", fs_cap)])
            .expect("run nifler.temen");
        assert!(
            matches!(run.outcome, Outcome::Exited(0) | Outcome::Returned(_)),
            "nifler guest did not exit cleanly: {:?}",
            run.outcome
        );

        // The emitted `.p.nif` (memfs key `out.nif`, the leading `/` stripped by `fs::norm`).
        let (files, _dirs) = handle.seed();
        let emitted = files
            .iter()
            .find(|(k, _)| k == "out.nif")
            .map(|(_, v)| v.clone())
            .expect("nifler wrote no `out.nif`");
        assert_eq!(
            emitted,
            expected.as_bytes(),
            "nifler-on-Temen must parse byte-identically to native nifler (the committed fixture)"
        );
    }
}
