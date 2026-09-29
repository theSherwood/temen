//! **The committed child-entry nifler asset, spawned on the JIT** (NIM.md §3c, W5). The
//! `nifler_child_asset` gate runs the phase child on the tree-walker; this runs the *same* spawn — the
//! [`temen_run::conductor`], an op-17 v1 detached record — on the **Cranelift JIT**, via the
//! granted-spawn hooks (`GrantChildHooks` + `module_resolver`). nifler runs as a confined §14 child on
//! emitted code, in its own window (its heap grows by `vm_map` into the window's reserved tail), and
//! its `.p.nif` is byte-identical to native. `child_entry_multicap_jit` is the fast per-PR stand-in for
//! the grant marshaling this exercises (#1221).

#![cfg(target_os = "linux")]

use core::ffi::c_void;
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;

use temen_interp::{ForkedProc, Host, HostProcFork, StreamRole};
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

/// The production granted-spawn hook table (temen-run's child build/bind/release/mint/thunk/serve), the
/// same one the JIT granted-spawn suites and `rust_guest_op13` install.
/// #1234 — the production table, derived from one [`temen_run::CapCtx`] so the hook family and
/// the parent pointer it decodes are chosen together (this used to hand-roll both, and nothing
/// checked that the pointer matched the ctx the run baked).
fn grant_hooks(host: *mut temen_interp::Host) -> GrantChildHooks {
    temen_run::production_grant_hooks(temen_run::CapCtx::Raw(host))
}

#[test]
#[ignore = "PASSES — on-demand gate: Cranelift-compiling nifler's 100+ funcs takes ~250s in debug, too \
            slow for the default run. `child_entry_multicap_jit` is the fast per-PR stand-in for the \
            multi-record grant marshaling this exercises (#1221); run `--ignored` here to exercise the \
            full real-phase child on the JIT."]
fn nifler_child_runs_on_the_jit_byte_identical() {
    let Some(temen) = inflate() else {
        eprintln!("SKIP: gzip unavailable");
        return;
    };
    let child = temen_encode::decode_module(&temen).expect("decode nifler_ce.temen");
    temen_verify::verify_module(&child).expect("child verifies");
    let log2 = child.memory.as_ref().expect("child window").size_log2;
    let parent = temen_run::conductor(
        log2,
        &["fs", "stdout", "exit"],
        &["nifler", "p", "/in.nim", "/out.nif"],
    );

    let (factory, handle) = temen_run::fs::mem_fs_shared_factory(
        vec![("in.nim".into(), IN_NIM.as_bytes().to_vec())],
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
    let (inst, modh, budget) = temen_run::grant_conductor(&mut host, &child);

    // Drive the conductor (and thus the nifler child) on the JIT: the granted-spawn hooks build and
    // run the child on emitted code; `module_resolver` fetches the granted child module by handle.
    let args = [
        inst as i64,
        modh as i64,
        budget as i64,
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
        Some(grant_hooks(&mut host as *mut Host)),
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
        "nifler as a §14 child on the JIT parses byte-identically to native"
    );
}
