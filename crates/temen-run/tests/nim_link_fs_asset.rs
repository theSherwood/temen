//! **#1025 slice 3c — the memfs-I/O link phase, run as a confined op-13 §14 child.** The connective
//! phase that lets the driver guest fan out `… → hexer → link` through one shared store: unlike
//! `nim_link_guest` (a top-level powerbox over stdin/stdout — its input can't be a host-seeded stream
//! when a driver produces it at runtime), this is built `--child-entry` and hands off through the
//! **memfs**. A driver op-13-spawns it with `{fs}` re-granted and argv `link <in.x.nif> <out.temen>
//! <stem>` seeded in its carve; it reads the hexer Leng `.x.nif`, links it with
//! `temen_leng::link_nim_powerbox`, and writes the `temen_encode`d linked module to `<out.temen>` in the
//! same store — exactly the shape hexer/nifler use.
//!
//! **Committed, wire-format-coupled asset** (`fixtures/nim-link-fs.temen.gz`, built by
//! `demos/nim_frontend/build_nim_link_fs.sh`). The oracle is the in-tree `link_nim_powerbox`, so this
//! gate needs **no build toolchain** — an IR/ABI/encoder or `temen-leng` change that makes the committed
//! asset stop matching native fails the PR. The `.x.nif` input is the same system-module Leng the chain
//! (`rust_driver_chain.rs`) and `nimlink_asset.rs` use (`sysvq0asl.x.nif.gz`).
//!
//! Heavy: the linker's no-free bump heap (the on-ramp `malloc` grows it via `vm_map` inside the carve)
//! needs ~512 MiB, so the child runs in a 512 MiB carve inside a ~1 GiB window. Gated Linux + gzip.

#![cfg(target_os = "linux")]

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;

use temen_interp::{run_with_host, ForkedProc, Host, HostProc, HostProcFork, Value};

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

/// A single-cap (`{fs}`) op-13 parent (the #1094-safe layout: grant record at `guard+1024`, name at
/// `guard+2048`, argv at `carve + module_args_base`), spawning `child` into `[carve_off, carve_off +
/// 2^child_sl)` with argv `["link", in, out, stem]` and joining it. Mirrors `nifler_child_asset.rs`.
fn parent_src(child_sl: u32, carve_off: u64) -> String {
    let parent_sl = child_sl + 1;
    let guard = temen_ir::POWERBOX_NULL_GUARD;
    let rec_off = guard + 1024;
    let name_off = guard + 2048;
    let argv_off = carve_off + temen_ir::module_args_base();

    let in_path = format!("nimcache/{STEM}.x.nif");
    let out_path = format!("nimcache/{STEM}.temen");
    let argv = ["link", &in_path, &out_path, STEM];
    let mut blob = Vec::new();
    blob.extend_from_slice(&(argv.len() as u32).to_le_bytes()); // argc
    blob.extend_from_slice(&0u32.to_le_bytes()); // envc
    for s in argv {
        blob.extend_from_slice(s.as_bytes());
        blob.push(0);
    }
    let argv_esc: String = blob.iter().map(|b| format!("\\x{b:02x}")).collect();

    // grant record {name_off:u32, name_len:u32, handle:i32, pad} at rec_off; the fs handle is arg v2.
    let w0 = name_off | (2u64 << 32);
    format!(
        r#"memory {parent_sl}
data {name_off} "fs"
data {argv_off} "{argv_esc}"
func (i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32) {{
  x0 = i64.const {w0}
  o0 = i64.const {rec_off}
  i64.store o0 x0
  hf = i64.extend_i32_u v2
  ohf = i64.const {hoff}
  i64.store ohf hf
  vmh = i64.extend_i32_u v1
  vgptr = i64.const {rec_off}
  vgn = i64.const 1
  ventry = i64.const 0
  voff = i64.const {carve_off}
  vsl = i64.const {child_sl}
  vq = i64.const 0
  vh = call.cap 6 13 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmh, vgptr, vgn, ventry, voff, vsl, vq)
  vr = call.cap 6 1 (i32) -> (i64) v0 (vh)
  return vr
  }}
}}
"#,
        hoff = rec_off + 8,
    )
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

#[test]
fn in_guest_memfs_link_matches_native_link_nim_powerbox() {
    let (Some(temen), Some(xnif)) = (inflate(NIM_LINK_FS_GZ), inflate(SYS_XNIF_GZ)) else {
        eprintln!("SKIP: gzip unavailable");
        return;
    };
    let src = String::from_utf8(xnif).expect("x.nif is UTF-8");

    // Host-side oracle: the in-tree linker on the same unit.
    let units = vec![temen_leng::WholeModule {
        stem: STEM,
        src: &src,
    }];
    let expected = temen_encode::encode_module(
        &temen_leng::link_nim_powerbox(&units).expect("native link_nim_powerbox"),
    );

    let child = temen_encode::decode_module(&temen).expect("decode nim-link-fs.temen");
    temen_verify::verify_module(&child).expect("verify nim-link-fs.temen");

    // Carve the child ~512 MiB (its no-free bump heap), a window at least its declared size.
    let decl = child.memory.as_ref().expect("child window").size_log2 as u32;
    let child_sl = (decl + 3).max(29);
    let carve_off = 1u64 << child_sl;
    let parent = temen_text::parse_module(&parent_src(child_sl, carve_off)).expect("parse parent");
    temen_verify::verify_module(&parent).expect("verify parent");

    // Shared memfs seeded with the hexer `.x.nif` at `nimcache/<stem>.x.nif` (the key the driver hands
    // off through); the linker writes `nimcache/<stem>.temen` back into the same store.
    let (factory, handle) = temen_run::fs::mem_fs_shared_factory(
        vec![(format!("nimcache/{STEM}.x.nif"), src.as_bytes().to_vec())],
        vec!["nimcache".into()],
    );
    let factory = Arc::new(factory);

    let mut host = Host::new();
    let fs_init: HostProc = (*factory)();
    let fs_fork: HostProcFork = {
        let f = Arc::clone(&factory);
        Arc::new(move |_pid| ForkedProc::shared((*f)()))
    };
    let fs_h = host.grant_host_proc_forkable(fs_init, fs_fork);
    let inst = host.grant_instantiator(0, 1u64 << (child_sl + 1));
    let modh = host.grant_module(&child);

    let mut fuel = 3_000_000_000_000u64;
    let r = run_with_host(
        &parent,
        0,
        &[Value::I32(inst), Value::I32(modh), Value::I32(fs_h)],
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
