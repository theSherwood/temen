//! **The committed child-entry `nifler_ce.temen.gz`, run as a confined detached §14 child** (NIM.md
//! §3c, W5 — the compiler-driver shape). Where `nifler_asset.rs` runs `nifler` as a *top-level*
//! powerbox program, this runs the **child-entry** build the Rust-on-Temen driver guest fans out:
//! `nifler` translated `--child-entry` (func 0 is the `starter -> i64 status` §14 child ABI), spawned
//! by [`temen_run::conductor_src`] (an op-17 v1 record: its own window, argv `nifler p /in.nim
//! /out.nif` as the args payload), a shared `mem_fs` re-granted as `"fs"`, and `stdout`/`exit` for its
//! `write`/`read`/`exit` imports (`vm_map` auto-binds to the child's `AddressSpace`). It reads the
//! emitted `.p.nif` back out of the shared store and asserts it is **byte-identical to the committed
//! `expected/*.p.nif`** — verbatim native-`nifler` output. A real nimony phase, byte-exact, as a
//! confined child.
//!
//! **Code-coupled asset (the `nifler_asset.rs` lane), no build toolchain (only `gzip`).** If an
//! IR/ABI/encoder change, or a regression in the spawn / `bind_child_manifest` / args-payload path,
//! makes the committed asset stop decoding, verifying, or producing the same NIF as a child, this gate
//! fails the PR. Regenerate the asset + fixtures with `build_nifler_temen.sh`
//! (`TEMEN_NIFLER_EMIT_ASSET=1`) and commit them.

#![cfg(target_os = "linux")]

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;

use temen_interp::{run_with_host, ForkedProc, Host, HostProcFork, StreamRole, Value};

/// The gzipped child-entry module — built by `build_nifler_temen.sh` alongside the browser asset.
const ASSET_GZ: &[u8] = include_bytes!("../demos/nifler_temen/nifler_ce.temen.gz");

/// The Nim inputs, paired with their committed native-`nifler` `.p.nif` (shared with `nifler_asset.rs`).
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

/// Inflate the committed gzip via `gzip -dc` (see `nifler_asset.rs` for the deadlock-avoiding threading:
/// the inflated `.temen` overflows the OS pipe buffer, so the stdin write runs on its own thread).
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
        let _ = stdin.write_all(ASSET_GZ);
    });
    let out = child.wait_with_output().expect("gzip -dc");
    writer.join().expect("stdin writer thread");
    out.status.success().then_some(out.stdout)
}

/// The conductor that spawns its child with `caps` re-granted and argv `nifler p /in.nim /out.nif`.
fn conductor(caps: &[&str]) -> temen_ir::Module {
    temen_run::conductor(caps, &["nifler", "p", "/in.nim", "/out.nif"])
}

#[test]
fn committed_child_entry_asset_decodes_and_verifies() {
    let Some(temen) = inflate_asset() else {
        eprintln!("SKIP: gzip unavailable to inflate nifler_ce.temen.gz");
        return;
    };
    let module = temen_encode::decode_module(&temen).expect("decode nifler_ce.temen");
    // The shipped child-entry bytes are a well-formed, re-verifiable module (the fail-closed TCB floor).
    // NOT `instantiate` — that gate wants a top-level paramless `_start`; a child entry is `[I64]->[I64]`.
    temen_verify::verify_module(&module).expect("verify nifler_ce.temen (the trusted floor)");
    assert_eq!(
        module.funcs[0]
            .params
            .len()
            .max(module.funcs[0].results.len()),
        1,
        "func 0 is the child entry (starter -> i64 status): {:?} -> {:?}",
        module.funcs[0].params,
        module.funcs[0].results,
    );
    assert!(
        module.funcs.len() > 100,
        "expected the full nifler phase, got {} funcs",
        module.funcs.len()
    );
}

#[test]
fn child_entry_asset_parses_nim_byte_identical_to_native_nifler() {
    let Some(temen) = inflate_asset() else {
        eprintln!("SKIP: gzip unavailable to inflate nifler_ce.temen.gz");
        return;
    };
    let child = temen_encode::decode_module(&temen).expect("decode nifler_ce.temen");
    temen_verify::verify_module(&child).expect("verify nifler_ce.temen");

    let parent = conductor(&["fs", "stdout", "exit"]);

    for (src, expected) in CORPUS {
        // A cross-domain shared memfs seeded with the source as `in.nim` (the guest's os_shim strips the
        // leading `/` of `/in.nim`); the handle observes the same store, so we read `out.nif` back after.
        let (factory, handle) = temen_run::fs::mem_fs_shared_factory(
            vec![("in.nim".into(), src.as_bytes().to_vec())],
            vec![],
        );
        let factory = Arc::new(factory);

        let mut host = Host::new();
        let (fs_init, fs_init_state) = (*factory)();
        let fs_fork: HostProcFork = {
            let f = Arc::clone(&factory);
            Arc::new(move |_pid| {
                let (h, s) = (*f)();
                ForkedProc::shared(h, s)
            })
        };
        // Grant list {fs (by name), stdout (write/read), exit}; vm_map auto-binds to the AddressSpace.
        let fs_h = host.grant_host_proc_forkable(fs_init, fs_fork, fs_init_state);
        let stdout_h = host.grant_stream(StreamRole::Out);
        let exit_h = host.grant_exit();
        let (inst, modh, budget) = temen_run::grant_conductor(&mut host, &child);

        let mut fuel = 200_000_000_000u64;
        let r = run_with_host(
            &parent,
            0,
            &[
                Value::I32(inst),
                Value::I32(modh),
                Value::I32(budget),
                Value::I32(fs_h),
                Value::I32(stdout_h),
                Value::I32(exit_h),
            ],
            &mut fuel,
            &mut host,
        )
        .expect("parent run");
        assert!(
            matches!(r.as_slice(), [Value::I64(0)] | [Value::I32(0)]),
            "nifler child joined with status 0: {r:?}"
        );

        // The emitted `.p.nif` (memfs key `out.nif`, the leading `/` stripped by `fs::norm`).
        let (files, _dirs) = handle.seed();
        let emitted = files
            .iter()
            .find(|(k, _)| k == "out.nif")
            .map(|(_, v)| v.clone())
            .expect("nifler child wrote no `out.nif`");
        assert_eq!(
            emitted,
            expected.as_bytes(),
            "nifler as a §14 child must parse byte-identically to native nifler (the committed fixture)"
        );
    }
}

/// A **four**-cap grant list — one more record than nifler imports (the spare `extra` is offered and
/// ignored), the multi-record shape the four-cap nimsem driver needs — still spawns `nifler_ce` and
/// joins 0 with a byte-identical `out.nif`. (The full byte-exact nimsem chain lives in the
/// toolchain-gated `build_frontend.sh`.)
#[test]
fn child_entry_asset_runs_under_a_four_cap_grant_list() {
    let Some(temen) = inflate_asset() else {
        eprintln!("SKIP: gzip unavailable to inflate nifler_ce.temen.gz");
        return;
    };
    let child = temen_encode::decode_module(&temen).expect("decode nifler_ce.temen");
    temen_verify::verify_module(&child).expect("verify nifler_ce.temen");

    // {fs, stdout, exit} are what nifler imports; `extra` is a spare offered cap it never resolves.
    let parent = conductor(&["fs", "stdout", "exit", "extra"]);

    let (factory, handle) = temen_run::fs::mem_fs_shared_factory(
        vec![("in.nim".into(), CORPUS[0].0.as_bytes().to_vec())],
        vec![],
    );
    let factory = Arc::new(factory);
    let mut host = Host::new();
    let (fs_init, fs_init_state) = (*factory)();
    let fs_fork: HostProcFork = {
        let f = Arc::clone(&factory);
        Arc::new(move |_pid| {
            let (h, s) = (*f)();
            ForkedProc::shared(h, s)
        })
    };
    let fs_h = host.grant_host_proc_forkable(fs_init, fs_fork, fs_init_state);
    let stdout_h = host.grant_stream(StreamRole::Out);
    let exit_h = host.grant_exit();
    let extra_h = host.grant_stream(StreamRole::Out); // the spare offered cap (a valid regrantable handle)
    let (inst, modh, budget) = temen_run::grant_conductor(&mut host, &child);

    let mut fuel = 200_000_000_000u64;
    let r = run_with_host(
        &parent,
        0,
        &[
            Value::I32(inst),
            Value::I32(modh),
            Value::I32(budget),
            Value::I32(fs_h),
            Value::I32(stdout_h),
            Value::I32(exit_h),
            Value::I32(extra_h),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("parent run");
    assert!(
        matches!(r.as_slice(), [Value::I64(0)] | [Value::I32(0)]),
        "nifler child joined with status 0 under a four-cap grant list: {r:?}"
    );

    let (files, _dirs) = handle.seed();
    let emitted = files
        .iter()
        .find(|(k, _)| k == "out.nif")
        .map(|(_, v)| v.clone())
        .expect("nifler child wrote no `out.nif`");
    assert_eq!(
        emitted,
        CORPUS[0].1.as_bytes(),
        "the four-cap conductor must spawn nifler_ce byte-identically to native nifler"
    );
}
