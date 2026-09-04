//! #1025 Gap-2 (guest-serves-exec-via-grandchild), increment 3 — **a serve handler produces a real
//! `.p.nif` by op-13-spawning the real nifler phase over a shared memfs.** Increments 1 & 2
//! (`temen-interp/tests/svc_handler_spawns_grandchild.rs`) proved the mechanism with a *toy*
//! grandchild: a serve handler can nest a §14 spawn+join, driven either by a host-enqueue or by a real
//! caller-parking cap call. This increment swaps the toy for the **real committed `nifler_ce`** asset —
//! the production capability the driver guest needs: answering an `exec("nifler …")` dispatch by
//! instantiating nifler as its own confined §14 grandchild, whose emitted `.p.nif` is byte-identical to
//! native nifler.
//!
//! Topology: the host-enqueue form (increment 1) — the servicer is the root, so the ~32 MiB window it
//! needs to hold nifler's 16 MiB carve is a flat top-level allocation rather than a nested sub-window
//! (the caller-parking layer is orthogonal, proven separately with the toy grandchild). The host
//! enqueues one "svc" dispatch; the servicer's `main` stashes the five caps it was granted
//! (`inst`/`nifler`/`fs`/`stdout`/`exit`) and `svc.poll`s; **its handler lays the three op-13 grant
//! records `{fs, stdout, exit}`, op-13-spawns nifler into a 16 MiB carve with `nifler p /in.nim
//! /out.nif` seeded at `carve + module_args_base`, and joins it** — the `op13_parent_src` spawn body
//! (nimc.rs) relocated into a serve dispatch. nifler reads `/in.nim` and writes `/out.nif` into the
//! shared memfs, which the host reads back and diffs against the committed fixture.
//!
//! Serve + instantiate trips the §9 `svc_park_veto`, so this runs on the tree-walk oracle — the same
//! fold increments 1 & 2 pinned; a JIT'd nifler grandchild is the browser-tier follow-on. Gated to
//! Linux + `gzip` (the asset is committed; no nim toolchain needed), like `nifler_child_asset.rs`.

#![cfg(target_os = "linux")]

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;

use temen_interp::{run_with_host, ForkedProc, Host, HostProc, HostProcFork, StreamRole, Value};

/// The committed child-entry nifler asset (built by `build_nifler_temen.sh`), shared with the gates.
const NIFLER_CE_GZ: &[u8] = include_bytes!("../demos/nifler_temen/nifler_ce.temen.gz");
/// One corpus input + its committed native-`nifler` `.p.nif` (the oracle fixture).
const IN_NIM: &str = include_str!("../demos/nifler_temen/inputs/basic.nim");
const EXPECT_NIF: &str = include_str!("../demos/nifler_temen/expected/basic.p.nif");

fn inflate(gz: &[u8]) -> Option<Vec<u8>> {
    let mut c = Command::new("gzip")
        .args(["-dc"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let mut stdin = c.stdin.take().expect("gzip stdin");
    let gz = gz.to_vec();
    let w = std::thread::spawn(move || {
        let _ = stdin.write_all(&gz);
    });
    let out = c.wait_with_output().expect("gzip -dc");
    w.join().expect("stdin writer");
    out.status.success().then_some(out.stdout)
}

/// Build the serve-module text. `main` (func 0, params `inst,nifler,fs,stdout,exit`) stashes the five
/// cap handles as i32 at 17520.. and `svc.poll`s. The handler (func 1) reloads them, lays the three
/// 16-byte op-13 grant records `{fs@18432, stdout@18448, exit@18464}` at 17408/17424/17440, and
/// op-13-spawns nifler (entry 0) into `[carve_off, carve_off+2^child_sl)` with argv seeded at
/// `carve_off + args_base` (a data segment), then joins it. Mirrors `nimc::op13_parent_src`'s record /
/// argv layout, but the spawn lives in a serve dispatch rather than the parent's own `main`.
fn servicer_src() -> String {
    let child_sl: u64 = 24; // nifler's 16 MiB carve (matches rust_driver_nifler)
    let carve_off: u64 = 1 << child_sl;
    let parent_sl: u64 = child_sl + 1; // 32 MiB window; the carve is its upper half
    let args_base: u64 = 16512; // POWERBOX_NULL_GUARD(16384) + POWERBOX_ARGS_BASE(128)
    let argv_off = carve_off + args_base;

    // argv blob: {argc=4, envc=0} + "nifler\0p\0/in.nim\0/out.nif\0" (the `nifler p <in> <out>` form).
    let mut blob = Vec::new();
    blob.extend_from_slice(&4u32.to_le_bytes());
    blob.extend_from_slice(&0u32.to_le_bytes());
    for s in ["nifler", "p", "/in.nim", "/out.nif"] {
        blob.extend_from_slice(s.as_bytes());
        blob.push(0);
    }
    let argv_esc: String = blob.iter().map(|b| format!("\\x{b:02x}")).collect();

    // A 16-byte grant record's first word: name_off | (name_len << 32).
    let w0 = |name_off: u64, name_len: u64| name_off | (name_len << 32);

    format!(
        r#"memory {parent_sl}
data 18432 "fs"
data 18448 "stdout"
data 18464 "exit"
data {argv_off} "{argv_esc}"
type 0 func (i64) -> (i64)
type 1 interface {{ go: 0 }}
export 0 interface "svc" 1 {{ go: 1 }}

func (i32, i32, i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32, v3: i32, v4: i32) {{
  s0 = i64.const 17520
  i32.store s0 v0
  s1 = i64.const 17524
  i32.store s1 v1
  s2 = i64.const 17528
  i32.store s2 v2
  s3 = i64.const 17532
  i32.store s3 v3
  s4 = i64.const 17536
  i32.store s4 v4
  vz = i32.const 0
  vn = call.cap 4294967295 9 () -> (i64) vz ()
  return vn
  }}
}}

func (i64) -> (i64) {{
block 0 (vx: i64) {{
  s0 = i64.const 17520
  vinst = i32.load s0
  s1 = i64.const 17524
  vmod = i32.load s1
  s2 = i64.const 17528
  vfs = i32.load s2
  s3 = i64.const 17532
  vout = i32.load s3
  s4 = i64.const 17536
  vexit = i32.load s4
  xf = i64.const {rf}
  of = i64.const 17408
  i64.store of xf
  hf = i64.extend_i32_u vfs
  ohf = i64.const 17416
  i64.store ohf hf
  xs = i64.const {rs}
  os = i64.const 17424
  i64.store os xs
  hs = i64.extend_i32_u vout
  ohs = i64.const 17432
  i64.store ohs hs
  xe = i64.const {re}
  oe = i64.const 17440
  i64.store oe xe
  he = i64.extend_i32_u vexit
  ohe = i64.const 17448
  i64.store ohe he
  vmh = i64.extend_i32_u vmod
  vgptr = i64.const 17408
  vgn = i64.const 3
  ventry = i64.const 0
  voff = i64.const {carve_off}
  vsl = i64.const {child_sl}
  vq = i64.const 0
  vh = call.cap 6 13 (i64, i64, i64, i64, i64, i64, i64) -> (i32) vinst (vmh, vgptr, vgn, ventry, voff, vsl, vq)
  vr = call.cap 6 1 (i32) -> (i64) vinst (vh)
  return vr
  }}
}}
"#,
        rf = w0(18432, 2),
        rs = w0(18448, 6),
        re = w0(18464, 4),
    )
}

#[test]
fn a_serve_handler_spawns_real_nifler_via_op13_over_a_shared_memfs() {
    let Some(nifler_bytes) = inflate(NIFLER_CE_GZ) else {
        eprintln!("note: skipping (gzip unavailable)");
        return;
    };
    let nifler = temen_encode::decode_module(&nifler_bytes).expect("decode nifler_ce.temen");
    temen_verify::verify_module(&nifler).expect("nifler verifies");

    let servicer = Arc::new({
        let m = temen_text::parse_module(&servicer_src()).expect("parse servicer");
        temen_verify::verify_module(&m).expect("servicer verifies");
        m
    });
    // Serves and instantiates → folded to the tree-walk oracle (same as increments 1 & 2).
    assert!(
        !temen_interp::bytecode::serve_qualifies(&servicer.funcs),
        "serve+instantiate folds to the oracle"
    );

    // A shared memfs seeded with the Nim source as `in.nim` (the guest names `/in.nim`; os_shim strips
    // the leading `/`). The handle observes the store the grandchild writes, so we read `out.nif` back.
    let (factory, handle) = temen_run::fs::mem_fs_shared_factory(
        vec![("in.nim".into(), IN_NIM.as_bytes().to_vec())],
        vec![],
    );
    let factory = Arc::new(factory);

    let mut host = Host::new();
    host.set_self_module(&servicer);
    let inst = host.grant_instantiator(0, 1u64 << 25); // the servicer's 32 MiB window
    let modh = host.grant_module(&nifler);
    let fs_init: HostProc = (*factory)();
    let fs_fork: HostProcFork = {
        let f = Arc::clone(&factory);
        Arc::new(move |_pid| ForkedProc::shared((*f)()))
    };
    let fs_h = host.grant_host_proc_forkable(fs_init, fs_fork);
    let stdout_h = host.grant_stream(StreamRole::Out);
    let exit_h = host.grant_exit();

    // Enqueue one "svc" dispatch; the handler services it by spawning nifler.
    let ticket = host.svc_enqueue(0, 0, vec![0]).expect("enqueue go");

    let mut fuel = 200_000_000_000u64;
    let r = run_with_host(
        &servicer,
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
    .expect("servicer run");
    assert_eq!(
        r,
        vec![Value::I64(1)],
        "the servicer served exactly one dispatch"
    );

    let status = host.svc_result(ticket).expect("dispatch served");
    assert!(
        status == 0 || status == 5,
        "the handler op-13-spawned nifler and joined its status ({status}); 0/5 are nifler's ok codes"
    );

    // The `.nif` nifler wrote, read back out of the shared store — byte-identical to native nifler.
    let (files, _dirs) = handle.seed();
    let emitted = files
        .into_iter()
        .find(|(k, _)| k == "out.nif")
        .map(|(_, v)| v)
        .expect("nifler (as the serve handler's op-13 grandchild) wrote no `out.nif`");
    assert_eq!(
        emitted,
        EXPECT_NIF.as_bytes(),
        "a serve handler's op-13 nifler grandchild emitted byte-identical NIF to native nifler"
    );
}
