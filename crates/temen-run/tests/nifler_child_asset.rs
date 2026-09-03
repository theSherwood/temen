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
        temen_ir::module_args_base(),
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

/// A **multi-cap** guarded op-13 parent for an arbitrary cap list — the exact record/name/argv layout
/// the front-end drivers (`examples/{nimsem_child_driver,nim_chain_op13,nim_chain_op13_jit}.rs`) use:
/// grant records at `guard + 1024..`, cap-names at `guard + 2048..`, argv at `carve + module_args_base()`,
/// **all above** the #1094 NULL guard `[0, POWERBOX_NULL_GUARD)`. Params `(inst, module, cap0, cap1, …)`.
/// Returns the parsed IR text plus the three offset series it placed, so a test can assert none fell
/// back inside the guard (the exact regression those drivers had: records at the pre-#1094 `1024..`).
fn guarded_parent_src(
    child_sl: u32,
    carve_off: u64,
    argv: &[String],
    caps: &[(&str, u64)],
) -> (String, Vec<u64>, Vec<u64>, u64) {
    let parent_sl = child_sl + 1;
    let guard = temen_ir::POWERBOX_NULL_GUARD;
    let rec_base = guard + 1024;
    let name_base = guard + 2048;
    let argv_off = carve_off + temen_ir::module_args_base();
    let mut blob = Vec::new();
    blob.extend_from_slice(&(argv.len() as u32).to_le_bytes());
    blob.extend_from_slice(&0u32.to_le_bytes());
    for s in argv {
        blob.extend_from_slice(s.as_bytes());
        blob.push(0);
    }
    let argv_esc: String = blob.iter().map(|b| format!("\\x{b:02x}")).collect();
    let n = caps.len() + 2;
    let sig: String = vec!["i32"; n].join(", ");
    let bparams: String = (0..n)
        .map(|i| format!("v{i}: i32"))
        .collect::<Vec<_>>()
        .join(", ");
    let mut data = String::new();
    let mut records = String::new();
    let mut rec_offs = Vec::new();
    let mut name_offs = Vec::new();
    for (i, (name, len)) in caps.iter().enumerate() {
        let noff = name_base + i as u64 * 16;
        let off = rec_base + i as u64 * 16;
        name_offs.push(noff);
        rec_offs.push(off);
        data.push_str(&format!("data {noff} \"{name}\"\n"));
        let w0 = noff | (len << 32);
        records.push_str(&format!(
            "  x{off} = i64.const {w0}\n  o{off} = i64.const {off}\n  i64.store o{off} x{off}\n  h{off} = i64.extend_i32_u v{vi}\n  oh{off} = i64.const {hoff}\n  i64.store oh{off} h{off}\n",
            vi = 2 + i,
            hoff = off + 8,
        ));
    }
    let gn = caps.len();
    let src = format!(
        r#"memory {parent_sl}
{data}data {argv_off} "{argv_esc}"
func ({sig}) -> (i64) {{
block 0 ({bparams}) {{
{records}  vmh = i64.extend_i32_u v1
  vgptr = i64.const {rec_base}
  vgn = i64.const {gn}
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
    );
    (src, rec_offs, name_offs, argv_off)
}

/// Regression guard for the #1094 op-13 parent layout the front-end drivers use (the class of bug fixed
/// on this branch: a driver placing grant records/cap-names/argv inside the NULL guard `[0,
/// POWERBOX_NULL_GUARD)`, so the parent's first record store NULL-faults before the op-13 child runs).
///
/// Spawns the committed `nifler_ce` as an op-13 child through a **four**-cap guarded parent — one more
/// record than nifler imports (the spare `extra` is offered and ignored, exercising the multi-record
/// offset math the four-cap `nimsem_child_driver` needs) — and asserts (a) every record/name/argv
/// offset clears the guard, and (b) the child still runs, joining 0 with byte-identical `out.nif`. If a
/// future edit drifts the layout back into the guard, the run NULL-faults and this fails — no toolchain
/// needed. (The full byte-exact nimsem op-13 chain lives in the toolchain-gated `build_frontend.sh`.)
#[test]
fn child_entry_asset_runs_under_multi_cap_guarded_parent() {
    let Some(temen) = inflate_asset() else {
        eprintln!("SKIP: gzip unavailable to inflate nifler_ce.temen.gz");
        return;
    };
    let child = temen_encode::decode_module(&temen).expect("decode nifler_ce.temen");
    temen_verify::verify_module(&child).expect("verify nifler_ce.temen");

    let decl = child.memory.as_ref().expect("child window").size_log2 as u32;
    let child_sl = (decl + 3).max(24);
    let carve_off = 1u64 << child_sl;
    let argv: Vec<String> = ["nifler", "p", "/in.nim", "/out.nif"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    // {fs, stdout, exit} are what nifler imports; `extra` is a spare offered cap it never resolves,
    // so the grant list carries four records — the four-record layout the nimsem driver relies on.
    let caps = [("fs", 2u64), ("stdout", 6), ("exit", 4), ("extra", 5)];
    let (src, rec_offs, name_offs, argv_off) =
        guarded_parent_src(child_sl, carve_off, &argv, &caps);

    // (a) the layout invariant: every offset the parent writes is ABOVE the NULL guard.
    let guard = temen_ir::POWERBOX_NULL_GUARD;
    for off in rec_offs.iter().chain(name_offs.iter()) {
        assert!(
            *off >= guard,
            "grant-record/name offset {off} must clear the #1094 NULL guard [0, {guard})"
        );
    }
    assert!(
        argv_off.wrapping_sub(carve_off) >= guard,
        "argv must sit above the guard within the carve (child reads it at module_args_base())"
    );

    let parent = temen_text::parse_module(&src).expect("parse guarded parent");
    temen_verify::verify_module(&parent).expect("verify guarded parent");

    // (b) the child still runs: a NULL-fault from a record in the guard would trap the parent here.
    let (factory, handle) = temen_run::fs::mem_fs_shared_factory(
        vec![("in.nim".into(), CORPUS[0].0.as_bytes().to_vec())],
        vec![],
    );
    let factory = Arc::new(factory);
    let mut host = Host::new();
    let fs_init: HostProc = (*factory)();
    let fs_fork: HostProcFork = {
        let f = Arc::clone(&factory);
        Arc::new(move |_pid| ForkedProc::shared((*f)()))
    };
    let fs_h = host.grant_host_proc_forkable(fs_init, fs_fork);
    let stdout_h = host.grant_stream(StreamRole::Out);
    let exit_h = host.grant_exit();
    let extra_h = host.grant_stream(StreamRole::Out); // the spare offered cap (a valid regrantable handle)
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
            Value::I32(extra_h),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("parent run (a MemoryFault here means the grant layout fell back into the NULL guard)");
    assert!(
        matches!(r.as_slice(), [Value::I64(0)] | [Value::I32(0)]),
        "nifler child joined with status 0 under the four-cap guarded parent: {r:?}"
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
        "the four-cap guarded op-13 parent must spawn nifler_ce byte-identically to native nifler"
    );
}
