//! **#1025 slice 3c — the memfs-I/O link phase, run as a confined §14 child.** The connective
//! phase that lets the driver guest fan out `… → hexer → link` through one shared store: unlike
//! `nim_link_guest` (a top-level powerbox over stdin/stdout — its input can't be a host-seeded stream
//! when a driver produces it at runtime), this is built `--child-entry` and hands off through the
//! **memfs**. A driver spawns it ([`temen_run::conductor`]) with `{fs}` re-granted and argv `link
//! <in.x.nif> <out.temen> <stem>` as the args payload; it reads the hexer Leng `.x.nif`, links it with
//! `temen_leng::link_nim_powerbox`, and writes the `temen_encode`d linked module to `<out.temen>` in the
//! same store — exactly the shape hexer/nifler use.
//!
//! **Committed, wire-format-coupled asset** (`fixtures/nim-link-fs.temen.gz`, built by
//! `demos/nim_frontend/build_nim_link_fs.sh`). The oracle is the in-tree `link_nim_powerbox`, so this
//! gate needs **no build toolchain** — an IR/ABI/encoder or `temen-leng` change that makes the committed
//! asset stop matching native fails the PR. The `.x.nif` input is the same system-module Leng the chain
//! (`rust_driver_chain.rs`) and `nimlink_asset.rs` use (`sysvq0asl.x.nif.gz`).
//!
//! Heavy: the linker's no-free bump heap (the on-ramp `malloc` grows it via `vm_map` into the child
//! window's reserved tail) needs ~512 MiB. Gated Linux + gzip.

#![cfg(target_os = "linux")]

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;

use temen_interp::{run_with_host, ForkedProc, Host, HostProcFork, Value};

const NIM_LINK_FS_GZ: &[u8] = include_bytes!("../demos/nim_frontend/fixtures/nim-link-fs.temen.gz");
const SYS_XNIF_GZ: &[u8] = include_bytes!("../demos/nim_frontend/fixtures/sysvq0asl.x.nif.gz");

const STEM: &str = "sysvq0asl";

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

#[test]
fn committed_nim_link_fs_asset_decodes_and_verifies_as_child_entry() {
    let Some(temen) = inflate(NIM_LINK_FS_GZ) else {
        eprintln!("SKIP: gzip unavailable");
        return;
    };
    let m = temen_encode::decode_module(&temen).expect("decode nim-link-fs.temen");
    temen_verify::verify_module(&m).expect("verify nim-link-fs.temen (the trusted floor)");
    // func 0 is the child entry (`starter -> i64 status`): one param, one result. NOT `instantiate` —
    // a child-entry module has no top-level `_start`.
    assert_eq!(
        m.funcs[0].params.len().max(m.funcs[0].results.len()),
        1,
        "func 0 is the child entry: {:?} -> {:?}",
        m.funcs[0].params,
        m.funcs[0].results,
    );
    assert!(
        m.funcs.len() > 100,
        "expected the whole linker, got {} funcs",
        m.funcs.len()
    );
}

// `#[ignore]` in the per-PR `check` job: this runs the whole 512 MiB linker on the debug tree-walker
// (several minutes), and that job (`cargo test --workspace`, 30 min cap) already carries one heavy link
// (`nimlink_asset`). A second would risk the budget. The byte-exact **op-13 memfs** link property is
// gated per-PR by `rust_driver_chain.rs` (the assembled nimsem->hexer->link pipeline, Linux-only
// `temen-llvm` job, 45 min) — which drives this exact asset — and the fast decode+verify+shape test
// above still guards asset drift here toolchain-free. Run on demand with `--ignored` (fast in release).
#[test]
#[ignore = "heavy (multi-minute debug link); covered per-PR by rust_driver_chain.rs — run with --ignored"]
fn in_guest_memfs_link_matches_native_link_nim_powerbox() {
    let (Some(temen), Some(xnif)) = (inflate(NIM_LINK_FS_GZ), inflate(SYS_XNIF_GZ)) else {
        eprintln!("SKIP: gzip unavailable");
        return;
    };
    let src = temen_leng::nif_text(&xnif).into_owned();

    // Host-side oracle: the in-tree linker on the same unit.
    let units = vec![temen_leng::WholeModule {
        stem: STEM,
        src: &src,
    }];
    let expected = temen_encode::encode_module(
        &temen_leng::link_nim_powerbox(&units, None).expect("native link_nim_powerbox"),
    );

    let child = temen_encode::decode_module(&temen).expect("decode nim-link-fs.temen");
    temen_verify::verify_module(&child).expect("verify nim-link-fs.temen");

    let (in_path, out_path) = (
        format!("nimcache/{STEM}.x.nif"),
        format!("nimcache/{STEM}.temen"),
    );
    let parent = temen_run::conductor(&["fs"], &["link", &in_path, &out_path, STEM]);

    // Shared memfs seeded with the hexer `.x.nif` at `nimcache/<stem>.x.nif` (the key the driver hands
    // off through); the linker writes `nimcache/<stem>.temen` back into the same store.
    let (factory, handle) = temen_run::fs::mem_fs_shared_factory(
        vec![(format!("nimcache/{STEM}.x.nif"), src.as_bytes().to_vec())],
        vec!["nimcache".into()],
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
    let (inst, modh, budget) = temen_run::grant_conductor(&mut host, &child);

    let mut fuel = 3_000_000_000_000u64;
    let r = run_with_host(
        &parent,
        0,
        &[
            Value::I32(inst),
            Value::I32(modh),
            Value::I32(budget),
            Value::I32(fs_h),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("parent run");
    assert!(
        matches!(r.as_slice(), [Value::I64(0)] | [Value::I32(0)]),
        "the link child joined with status 0: {r:?}"
    );

    let (files, _dirs) = handle.seed();
    let produced = files
        .iter()
        .find(|(k, _)| k == &format!("nimcache/{STEM}.temen"))
        .map(|(_, v)| v.clone())
        .expect("the link child wrote no nimcache/<stem>.temen");
    assert_eq!(
        produced, expected,
        "the in-sandbox memfs link must be byte-identical to native link_nim_powerbox \
         (asset stale? regenerate with build_nim_link_fs.sh)"
    );
}
