//! #1025 Gap-2 (guest-serves-exec-via-grandchild), increment 3 — **a serve handler produces a real
//! `.p.nif` by spawning the real nifler phase over a shared memfs.** Increments 1 & 2
//! (`temen-interp/tests/svc_handler_spawns_grandchild.rs`) proved the mechanism with a *toy*
//! grandchild: a serve handler can nest a §14 spawn+join, driven either by a host-enqueue or by a real
//! caller-parking cap call. This increment swaps the toy for the **real committed `nifler_ce`** asset —
//! the production capability the driver guest needs: answering an `exec("nifler …")` dispatch by
//! instantiating nifler as its own confined §14 grandchild, whose emitted `.p.nif` is byte-identical to
//! native nifler.
//!
//! Topology: the host-enqueue form (increment 1) — the servicer is the root (the caller-parking layer
//! is orthogonal, proven separately with the toy grandchild). The host enqueues one "svc" dispatch;
//! the servicer's `main` stashes the six caps it was granted (`inst`/`nifler`/`fs`/`stdout`/`exit`/
//! `budget`) and `svc.poll`s; **its handler lays the three grant records `{fs, stdout, exit}` and an
//! op-17 v1 spawn record — nifler in its own window, paid from `budget`, argv `nifler p /in.nim
//! /out.nif` as the args payload — spawns nifler, and joins it**: the `temen_run::conductor` spawn,
//! relocated into a serve dispatch. nifler reads `/in.nim` and writes `/out.nif` into the
//! shared memfs, which the host reads back and diffs against the committed fixture.
//!
//! Serve + instantiate trips the §9 `svc_park_veto`, so this runs on the tree-walk oracle — the same
//! fold increments 1 & 2 pinned; a JIT'd nifler grandchild is the browser-tier follow-on. Gated to
//! Linux + `gzip` (the asset is committed; no nim toolchain needed), like `nifler_child_asset.rs`.

#![cfg(target_os = "linux")]

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;

use temen_interp::{run_with_host, ForkedProc, Host, HostProcFork, StreamRole, Value};

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

/// The servicer: `main(inst, nifler, fs, stdout, exit, budget)` stashes its caps and `svc.poll`s; the
/// "svc" `go` handler spawns nifler through an op-17 v1 record (grant records at 17408, spawn record at
/// 17600, names at 18432, the argv payload a data segment at 20480) and joins it.
fn servicer_src(nifler_log2: u8) -> String {
    // argv payload: {argc=4, envc=0} + "nifler\0p\0/in.nim\0/out.nif\0" (the `nifler p <in> <out>` form).
    let mut blob = Vec::new();
    blob.extend_from_slice(&4u32.to_le_bytes());
    blob.extend_from_slice(&0u32.to_le_bytes());
    for s in ["nifler", "p", "/in.nim", "/out.nif"] {
        blob.extend_from_slice(s.as_bytes());
        blob.push(0);
    }
    let args_len = blob.len();
    let args_esc: String = blob.iter().map(|b| format!("\\x{b:02x}")).collect();

    // A 16-byte grant record's first word: name_off | (name_len << 32).
    let w0 = |name_off: u64, name_len: u64| name_off | (name_len << 32);

    format!(
        r#"memory 17
data 18432 "fs"
data 18448 "stdout"
data 18464 "exit"
data 20480 "{args_esc}"
type 0 func (i64) -> (i64)
type 1 interface {{ go: 0 }}
export 0 interface "svc" 1 {{ go: 1 }}

func (i32, i32, i32, i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32, v3: i32, v4: i32, v5: i32) {{
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
  s5 = i64.const 17540
  i32.store s5 v5
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
  s5 = i64.const 17540
  vbud = i32.load s5
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
  r = i64.const 17600
  ver = i64.const 1
  i64.store r ver
  slp = i64.const {slp}
  i64.store r slp offset=16
  i32.store r vmod offset=24
  i32.store r vbud offset=28
  gp = i64.const 17408
  i64.store r gp offset=40
  gn = i64.const 3
  i64.store r gn offset=48
  ap = i64.const 20480
  i64.store r ap offset=56
  al = i64.const {args_len}
  i64.store r al offset=64
  none = i32.const -1
  i32.store r none offset=72
  vh = call.cap 6 17 (i64) -> (i32) vinst (r)
  vr = call.cap 6 1 (i32) -> (i64) vinst (vh)
  return vr
  }}
}}
"#,
        rf = w0(18432, 2),
        rs = w0(18448, 6),
        re = w0(18464, 4),
        // size_log2 | pager u32::MAX (none)
        slp = (nifler_log2 as u64 | (0xFFFF_FFFFu64 << 32)) as i64,
    )
}

#[test]
fn a_serve_handler_spawns_real_nifler_over_a_shared_memfs() {
    let Some(nifler_bytes) = inflate(NIFLER_CE_GZ) else {
        eprintln!("note: skipping (gzip unavailable)");
        return;
    };
    let nifler = temen_encode::decode_module(&nifler_bytes).expect("decode nifler_ce.temen");
    temen_verify::verify_module(&nifler).expect("nifler verifies");

    let servicer = Arc::new({
        let log2 = nifler.memory.as_ref().expect("nifler window").size_log2;
        let m = temen_text::parse_module(&servicer_src(log2)).expect("parse servicer");
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
    let (inst, modh, budget) = temen_run::grant_conductor(&mut host, &nifler);
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
            Value::I32(budget),
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
        "the handler spawned nifler and joined its status ({status}); 0/5 are nifler's ok codes"
    );

    // The `.nif` nifler wrote, read back out of the shared store — byte-identical to native nifler.
    let (files, _dirs) = handle.seed();
    let emitted = files
        .into_iter()
        .find(|(k, _)| k == "out.nif")
        .map(|(_, v)| v)
        .expect("nifler (as the serve handler's grandchild) wrote no `out.nif`");
    assert_eq!(
        emitted,
        EXPECT_NIF.as_bytes(),
        "a serve handler's nifler grandchild emitted byte-identical NIF to native nifler"
    );
}
