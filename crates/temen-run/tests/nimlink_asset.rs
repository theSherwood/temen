//! **#1025 slice 3c — link in-guest.** The nim->powerbox linker (`temen_leng::link_nim_powerbox`)
//! wrapped as a powerbox program and run **inside the sandbox**: it reads the hexer Leng `.x.nif` units
//! (packed on stdin), links them into a runnable Temen module (the compute shim + syscall adapter +
//! `_start` glue), and writes the `temen_encode`d module to stdout — the leng-self-host asset lane
//! (`leng_selfhost_asset.rs`) applied to the linker, and the step after nimsem (step 9) and hexer
//! (step 10) run confined.
//!
//! **Committed, wire-format-coupled asset** (`fixtures/nim-link.temen.gz`, built by
//! `demos/nim_frontend/build_nim_link.sh`): the real linker on Temen. The oracle is the in-tree
//! `link_nim_powerbox`, so this gate needs **no build toolchain** — if an IR/ABI/encoder or `temen-leng`
//! change makes the committed asset stop matching native, the PR that caused the drift fails. The `.x.nif`
//! input is the same system-module Leng step 10 committed (`sysvq0asl.x.nif.gz`).
//!
//! Heavy: the linker's no-free bump arena needs a 512 MiB heap, so the guest runs in a ~1 GiB window.
//! Gated Linux + gzip.

#![cfg(target_os = "linux")]

use std::io::Write;
use std::process::{Command, Stdio};

use temen_run::{Backend, Limits, Outcome, RunConfig};

const NIM_LINK_GZ: &[u8] = include_bytes!("../demos/nim_frontend/fixtures/nim-link.temen.gz");
const SYS_XNIF_GZ: &[u8] = include_bytes!("../demos/nim_frontend/fixtures/sysvq0asl.x.nif.gz");

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

/// Pack units for the guest's stdin: `count`, then per unit `stem_len, stem, src_len, src` (u32 LE).
fn pack(units: &[(&str, &[u8])]) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(&(units.len() as u32).to_le_bytes());
    for (stem, src) in units {
        v.extend_from_slice(&(stem.len() as u32).to_le_bytes());
        v.extend_from_slice(stem.as_bytes());
        v.extend_from_slice(&(src.len() as u32).to_le_bytes());
        v.extend_from_slice(src);
    }
    v
}

#[test]
fn committed_nim_link_asset_decodes_and_verifies() {
    let Some(asset) = inflate(NIM_LINK_GZ) else {
        eprintln!("SKIP: gzip unavailable");
        return;
    };
    let m = temen_encode::decode_module(&asset).expect("decode nim-link.temen");
    temen_verify::verify_module(&m).expect("verify nim-link.temen (the trusted floor)");
    assert!(
        m.funcs.len() > 100,
        "expected the whole linker, got {} funcs",
        m.funcs.len()
    );
}

#[test]
fn in_guest_link_matches_native_link_nim_powerbox() {
    let (Some(asset), Some(xnif)) = (inflate(NIM_LINK_GZ), inflate(SYS_XNIF_GZ)) else {
        eprintln!("SKIP: gzip unavailable");
        return;
    };
    let src = String::from_utf8(xnif).expect("x.nif is UTF-8");
    let stem = "sysvq0asl";

    // Host-side oracle: the in-tree linker on the same unit.
    let units = vec![temen_leng::WholeModule { stem, src: &src }];
    let m = temen_leng::link_nim_powerbox(&units).expect("native link_nim_powerbox");
    let expected = temen_encode::encode_module(&m);

    // In-sandbox: feed the packed unit on stdin, capture the encoded linked module on stdout.
    let module = temen_encode::decode_module(&asset).expect("decode nim-link.temen");
    temen_verify::verify_module(&module).expect("verify nim-link.temen");
    let cfg = RunConfig {
        limits: Limits {
            fuel: None,
            deadline: None,
            max_fibers: 0,
            max_vcpus: 0,
        },
        stdin: pack(&[(stem, src.as_bytes())]),
        memory_size_log2: None,
        args: vec![],
        env: vec![],
        ..RunConfig::default()
    };
    let run = temen_run::instantiate(module)
        .expect("instantiate nim-link.temen")
        .run(Backend::TreeWalk, &cfg)
        .expect("run nim-link.temen");
    assert!(
        matches!(run.outcome, Outcome::Returned(_)),
        "guest did not return cleanly: {:?}",
        run.outcome
    );
    assert_eq!(
        run.stdout, expected,
        "the in-sandbox nim->powerbox link must be byte-identical to native link_nim_powerbox \
         (asset stale? regenerate with build_nim_link.sh)"
    );
}
