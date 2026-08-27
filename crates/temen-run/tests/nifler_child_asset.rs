//! **The committed child-entry `nifler_ce.temen.gz`, run as a confined op-13 §14 child** (NIM.md §3c,
//! W5 — the compiler-driver shape). Where `nifler_asset.rs` runs `nifler` as a *top-level* powerbox
//! program, this runs the **child-entry** build the Rust-on-Temen driver guest fans out: `nifler`
//! translated `--child-entry` (func 0 is the `starter -> i64 status` §14 child ABI), `instantiate_module`
//! (op 13)'d into a carve with argv `nifler p /in.nim /out.nif` seeded at `POWERBOX_ARGS_BASE`, a shared
//! `mem_fs` re-granted as `"fs"`, and `stdout`/`exit` for its `write`/`read`/`exit` imports (`vm_map`
//! auto-binds to the child's `AddressSpace`). It reads the emitted `.p.nif` back out of the shared store
//! and asserts it is **byte-identical to the committed `expected/*.p.nif`** — verbatim native-`nifler`
//! output. A real nimony phase, byte-exact, as a confined op-13 child.
//!
//! **Code-coupled asset (the `nifler_asset.rs` lane), no build toolchain (only `gzip`).** If an
//! IR/ABI/encoder change, or a regression in the op-13 / `bind_child_manifest` / argv-in-carve path,
//! makes the committed asset stop decoding, verifying, or producing the same NIF as an op-13 child, this
//! gate fails the PR. Regenerate the asset + fixtures with `build_nifler_temen.sh`
//! (`TEMEN_NIFLER_EMIT_ASSET=1`) and commit them. The driver here mirrors
//! `crates/temen-run/examples/spawn_child_fs.rs`.

#![cfg(target_os = "linux")]

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;

use temen_interp::{run_with_host, ForkedProc, Host, HostProc, HostProcFork, StreamRole, Value};

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

/// Build the text-IR op-13 parent that spawns `child` (window `child_sl`, at `carve_off`) with a 3-entry
/// grant list `{fs, stdout, exit}` and argv `nifler p /in.nim /out.nif` seeded at `carve + args_base`.
/// Mirrors `examples/spawn_child_fs.rs`. #964/#1094: `args_base` is the child's `module_args_base` — one
/// guard up for the guarded nifler_ce, the legacy 128 otherwise; the grant records/cap-names stay in the
/// parent window, read by the op-13 handler in the parent's context (never by the guarded child) — but
/// the parent itself is guarded too, so they sit above the #1094 NULL guard: records at 17408.., names
/// at 18432.. (was 1024../2048..).
fn parent_src(child_sl: u32, carve_off: u64, args_base: u64) -> String {
    let parent_sl = child_sl + 1;
    let argv_off = carve_off + args_base;
    let mut argv_blob = Vec::new();
    argv_blob.extend_from_slice(&4u32.to_le_bytes()); // argc
    argv_blob.extend_from_slice(&0u32.to_le_bytes()); // envc
    for s in ["nifler", "p", "/in.nim", "/out.nif"] {
        argv_blob.extend_from_slice(s.as_bytes());
        argv_blob.push(0);
    }
    let argv_esc: String = argv_blob.iter().map(|b| format!("\\x{b:02x}")).collect();
    let rec = |off: u64, name_off: u64, name_len: u64| -> String {
        let w0 = name_off | (name_len << 32);
        format!(
            "  x{off} = i64.const {w0}\n  o{off} = i64.const {off}\n  i64.store o{off} x{off}\n"
        )
    };
    format!(
        r#"memory {parent_sl}
data 18432 "fs"
data 18448 "stdout"
data 18464 "exit"
data {argv_off} "{argv_esc}"
func (i32, i32, i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32, v3: i32, v4: i32) {{
{r0}  hf = i64.extend_i32_u v2
  ohf = i64.const 17416
  i64.store ohf hf
{r1}  hs = i64.extend_i32_u v3
  ohs = i64.const 17432
  i64.store ohs hs
{r2}  he = i64.extend_i32_u v4
  ohe = i64.const 17448
  i64.store ohe he
  vmh = i64.extend_i32_u v1
  vgptr = i64.const 17408
  vgn = i64.const 3
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
        r0 = rec(17408, 18432, 2),
        r1 = rec(17424, 18448, 6),
        r2 = rec(17440, 18464, 4),
    )
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

    // Carve the child a window at least its declared size, generously larger for `malloc` heap room
    // (the mod_ok relaxation: carve >= declared, heap grows in `[heap_base, carve)`).
    let decl = child.memory.as_ref().expect("child window").size_log2 as u32;
    let child_sl = (decl + 3).max(24);
    let carve_off = 1u64 << child_sl;
    let parent = temen_text::parse_module(&parent_src(
        child_sl,
        carve_off,
        temen_ir::module_args_base(&child),
    ))
    .expect("parse parent");
    temen_verify::verify_module(&parent).expect("verify parent");

    for (src, expected) in CORPUS {
        // A cross-domain shared memfs seeded with the source as `in.nim` (the guest's os_shim strips the
        // leading `/` of `/in.nim`); the handle observes the same store, so we read `out.nif` back after.
        let (factory, handle) = temen_run::fs::mem_fs_shared_factory(
            vec![("in.nim".into(), src.as_bytes().to_vec())],
            vec![],
        );
        let factory = Arc::new(factory);

        let mut host = Host::new();
        let fs_init: HostProc = (*factory)();
        let fs_fork: HostProcFork = {
            let f = Arc::clone(&factory);
            Arc::new(move |_pid| ForkedProc::shared((*f)()))
        };
        // Grant list {fs (by name), stdout (write/read), exit}; vm_map auto-binds to the AddressSpace.
        let fs_h = host.grant_host_proc_forkable(fs_init, fs_fork);
        let stdout_h = host.grant_stream(StreamRole::Out);
        let exit_h = host.grant_exit();
        let inst = host.grant_instantiator(0, 1u64 << (child_sl + 1));
        let modh = host.grant_module(&child);

        let mut fuel = 200_000_000_000u64;
        let r = run_with_host(
            &parent,
            0,
            &[
                Value::I32(inst),
                Value::I32(modh),
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
            "nifler as an op-13 §14 child must parse byte-identically to native nifler (the committed \
             fixture)"
        );
    }
}
