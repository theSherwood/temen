//! **The committed child-entry nifler asset, op-13-spawned on the JIT** (NIM.md §3c, W5). The
//! `nifler_child_asset` gate runs the phase child on the tree-walker; this runs the *same* op-13 spawn
//! on the **Cranelift JIT** — the tier-up-capable engine a browser wasm-JIT card also uses — via the
//! granted-spawn hooks (`GrantChildHooks` + `module_resolver`, the shape `rust_guest_op13` established).
//! nifler runs as a confined §14 op-13 child on emitted code and its `.p.nif` is byte-identical to native.
//!
//! **BRING-UP TARGET (currently `#[ignore]`d — CapFaults).** The tree-walker/bytecode op-13 phase path
//! is proven (`nifler_child_asset`, byte-exact), and the JIT granted-spawn works for a *toy* child
//! (`rust_guest_op13`, both engines). But a **real phase child** — nifler's manifest imports
//! (`exit`/`read`/`write`/`vm_map`), its `malloc`/`vm_map` heap growth, and its cross-tier `fs` calls —
//! traps `CapFault` on the JIT (an interp/JIT divergence in the granted-spawn dispatch: `spawn_named_child`
//! passes `can_regrant`, and `child_bind_imports` calls the same `bind_child_manifest`, so the fault is
//! likely a child-side cap dispatch on emitted code — the AddressSpace `vm_map` or the re-granted Stream).
//! Remove the `#[ignore]` to reproduce. This is the enabler for the browser wasm-JIT compile card, so it
//! is captured here as the next focused temen-jit slice.

#![cfg(target_os = "linux")]

use core::ffi::c_void;
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;

use temen_interp::{ForkedProc, Host, HostProc, HostProcFork, StreamRole};
use temen_jit::{compile_and_run_capture_reserved_with_host_ex, GrantChildHooks, JitOutcome};

const ASSET_GZ: &[u8] = include_bytes!("../demos/nifler_temen/nifler_ce.temen.gz");
const IN_NIM: &str = include_str!("../demos/nifler_temen/inputs/basic.nim");
const EXPECT_NIF: &str = include_str!("../demos/nifler_temen/expected/basic.p.nif");

fn inflate() -> Option<Vec<u8>> {
    let mut c = Command::new("gzip")
        .args(["-dc"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let mut stdin = c.stdin.take().expect("gzip stdin");
    let w = std::thread::spawn(move || {
        let _ = stdin.write_all(ASSET_GZ);
    });
    let out = c.wait_with_output().expect("gzip -dc");
    w.join().expect("stdin writer");
    out.status.success().then_some(out.stdout)
}

/// The op-13 parent: grant records `{fs, stdout, exit}` at 1024, argv `nifler p /in.nim /out.nif` at
/// `carve + POWERBOX_ARGS_BASE`. Same shape as `nifler_child_asset`, sized for the JIT run.
fn parent_src(child_sl: u32, carve_off: u64) -> String {
    let parent_sl = child_sl + 1;
    let argv_off = carve_off + temen_ir::POWERBOX_ARGS_BASE;
    let mut blob = Vec::new();
    blob.extend_from_slice(&4u32.to_le_bytes());
    blob.extend_from_slice(&0u32.to_le_bytes());
    for s in ["nifler", "p", "/in.nim", "/out.nif"] {
        blob.extend_from_slice(s.as_bytes());
        blob.push(0);
    }
    let argv_esc: String = blob.iter().map(|b| format!("\\x{b:02x}")).collect();
    let rec = |off: u64, name_off: u64, name_len: u64| -> String {
        let w0 = name_off | (name_len << 32);
        format!(
            "  x{off} = i64.const {w0}\n  o{off} = i64.const {off}\n  i64.store o{off} x{off}\n"
        )
    };
    format!(
        r#"memory {parent_sl}
data 2048 "fs"
data 2064 "stdout"
data 2080 "exit"
data {argv_off} "{argv_esc}"
func (i32, i32, i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32, v3: i32, v4: i32) {{
{r0}  hf = i64.extend_i32_u v2
  ohf = i64.const 1032
  i64.store ohf hf
{r1}  hs = i64.extend_i32_u v3
  ohs = i64.const 1048
  i64.store ohs hs
{r2}  he = i64.extend_i32_u v4
  ohe = i64.const 1064
  i64.store ohe he
  vmh = i64.extend_i32_u v1
  vgptr = i64.const 1024
  vgn = i64.const 3
  ventry = i64.const 0
  voff = i64.const {carve_off}
  vsl = i64.const {child_sl}
  vq = i64.const 0
  vh = cap.call 6 13 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmh, vgptr, vgn, ventry, voff, vsl, vq)
  vr = cap.call 6 1 (i32) -> (i64) v0 (vh)
  return vr
  }}
}}
"#,
        r0 = rec(1024, 2048, 2),
        r1 = rec(1040, 2064, 6),
        r2 = rec(1056, 2080, 4),
    )
}

/// The production granted-spawn hook table (temen-run's child build/bind/release/mint/thunk/serve), the
/// same one the JIT granted-spawn suites and `rust_guest_op13` install.
fn grant_hooks() -> GrantChildHooks {
    GrantChildHooks {
        build: temen_run::grant_child_build,
        build_named: temen_run::grant_named_child_build,
        bind_imports: temen_run::child_bind_imports,
        release: temen_run::grant_child_release,
        mint: temen_run::child_offer_mint,
        thunk: temen_run::cap_thunk_locked,
        register_serve: temen_run::child_register_serve,
    }
}

#[test]
#[ignore = "JIT op-13 phase spawn CapFaults for a real phase child — temen-jit granted-spawn bring-up"]
fn nifler_child_runs_on_the_jit_byte_identical() {
    let Some(temen) = inflate() else {
        eprintln!("SKIP: gzip unavailable");
        return;
    };
    let child = temen_encode::decode_module(&temen).expect("decode nifler_ce.temen");
    temen_verify::verify_module(&child).expect("child verifies");
    let decl = child.memory.as_ref().expect("child window").size_log2 as u32;
    let child_sl = (decl + 3).max(24);
    let carve_off = 1u64 << child_sl;
    let parent = temen_text::parse_module(&parent_src(child_sl, carve_off)).expect("parse parent");
    temen_verify::verify_module(&parent).expect("verify parent");

    let (factory, handle) = temen_run::fs::mem_fs_shared_factory(
        vec![("in.nim".into(), IN_NIM.as_bytes().to_vec())],
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
    let inst = host.grant_instantiator(0, 1u64 << (child_sl + 1));
    let modh = host.grant_module(&child);

    // Drive the parent (and thus the op-13 nifler child) on the JIT: the granted-spawn hooks build and
    // run the child on emitted code; `module_resolver` fetches the granted child module by handle.
    let args = [
        inst as i64,
        modh as i64,
        fs_h as i64,
        stdout_h as i64,
        exit_h as i64,
    ];
    let (jo, _) = compile_and_run_capture_reserved_with_host_ex(
        &parent,
        0,
        &args,
        &[],
        temen_ir::DEFAULT_RESERVED_LOG2,
        temen_run::cap_thunk,
        &mut host as *mut Host as *mut c_void,
        Some(temen_run::module_resolver),
        Some(grant_hooks()),
    )
    .expect("jit run");
    let status = match jo {
        JitOutcome::Returned(ref v) => v.first().copied().unwrap_or(-1),
        JitOutcome::Exited(c) => c as i64,
        ref o => panic!("jit ended abnormally: {o:?}"),
    };
    assert_eq!(status, 0, "nifler child (on the JIT) exited 0, joined back");

    let (files, _dirs) = handle.seed();
    let emitted = files
        .into_iter()
        .find(|(k, _)| k == "out.nif")
        .map(|(_, b)| b)
        .expect("nifler child wrote no out.nif on the JIT");
    assert_eq!(
        emitted,
        EXPECT_NIF.as_bytes(),
        "nifler as an op-13 §14 child on the JIT parses byte-identically to native"
    );
}
