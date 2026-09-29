//! **#1011 slice 3c — a Rust-on-Temen driver *guest* runs the real nifler phase.** The compiler
//! driver's endgame is to move phase orchestration **into** the sandbox: instead of the native harness
//! (`spawn_child_fs` / `nimc::compile_nim`) issuing the spawn, a Rust-on-Temen guest does it.
//! `rust_guest_op13` proved a guest can spawn and join a *toy* child; this upgrades that to the **real
//! `nifler` phase**: the guest resolves the `Instantiator`, the `nifler` module, its `Budget` and
//! `fs`/`stdout`/`exit` by name, and spawns nifler (the shared `vm_spawn`: its own window, argv
//! `nifler p /in.nim /out.nif` as the args payload, the three caps re-granted by name), then joins it.
//! The host seeds the Nim source into the shared memfs and reads the emitted `.nif` back —
//! **byte-identical to native nifler**.
//!
//! This is the first phase of the compiler driver running *on Temen* (the orchestration, not just the
//! phase). It reuses the committed `nifler_ce.temen.gz` child-entry asset (the same one
//! `nifler_child_asset.rs` gates). Gated to Linux + `rustc` + `gzip`; skips cleanly otherwise.

#![cfg(target_os = "linux")]

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;

use temen_interp::{
    run_capture_reserved_with_host, ForkedProc, Host, HostProcFork, StreamRole, Value,
};

/// The committed child-entry nifler asset (built by `build_nifler_temen.sh`), shared with the gate.
const NIFLER_CE_GZ: &[u8] = include_bytes!("../../temen-run/demos/nifler_temen/nifler_ce.temen.gz");
/// One corpus input + its committed native-`nifler` `.p.nif` (the oracle fixture).
const IN_NIM: &str = include_str!("../../temen-run/demos/nifler_temen/inputs/basic.nim");
const EXPECT_NIF: &str = include_str!("../../temen-run/demos/nifler_temen/expected/basic.p.nif");

// The Rust-on-Temen driver guest. `no_std`, no allocator: `run()` resolves the caps by name, spawns
// nifler (entry 0) through the shared `vm_spawn` (appended at build) and returns `join(child)` —
// nifler's status.
const GUEST_SRC: &str = r##"
#![no_std]
#![allow(internal_features)]
#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! { loop {} }
#[no_mangle]
pub extern "C" fn rust_eh_personality() {}

extern "C" {
    fn __vm_cap_resolve(name: *const u8, len: i64) -> i32;
    fn __vm_join(inst: i32, child: i64) -> i64;
}

#[no_mangle]
pub extern "C" fn run() -> i64 {
    unsafe {
        let inst = __vm_cap_resolve(b"inst".as_ptr(), 4);
        let nifler = __vm_cap_resolve(b"nifler".as_ptr(), 6);
        let budget = __vm_cap_resolve(b"budget".as_ptr(), 6);
        let fs = __vm_cap_resolve(b"fs".as_ptr(), 2);
        let out = __vm_cap_resolve(b"stdout".as_ptr(), 6);
        let ex = __vm_cap_resolve(b"exit".as_ptr(), 4);
        if inst < 0 || nifler < 0 || budget < 0 || fs < 0 || out < 0 || ex < 0 { return -1; }
        let child = vm_spawn(
            inst,
            budget,
            nifler,
            &[(b"fs", fs), (b"stdout", out), (b"exit", ex)],
            &[b"nifler", b"p", b"/in.nim", b"/out.nif"],
        );
        __vm_join(inst, child)
    }
}
"##;

/// The shared guest-side spawn, appended to every driver guest's source.
const VM_SPAWN: &str = include_str!("support/guest_vm_spawn.rs");

fn rustc_emit_ll(src: &std::path::Path, ll: &std::path::Path) -> bool {
    Command::new("rustc")
        .args([
            "--edition",
            "2021",
            "-O",
            "-Cpanic=abort",
            "--emit=llvm-ir",
            "--crate-type=cdylib",
        ])
        .arg(src)
        .arg("-o")
        .arg(ll)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

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

#[test]
fn rust_driver_guest_runs_real_nifler() {
    let dir = std::env::temp_dir().join(format!("rust_driver_nifler_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("driver.rs");
    let ll = dir.join("driver.ll");
    std::fs::write(&src, format!("{GUEST_SRC}\n{VM_SPAWN}")).unwrap();
    if !rustc_emit_ll(&src, &ll) {
        eprintln!("note: skipping (rustc unavailable)");
        return;
    }
    let Some(nifler_bytes) = inflate(NIFLER_CE_GZ) else {
        eprintln!("note: skipping (gzip unavailable)");
        return;
    };

    // The driver guest (top-level Rust-on-Temen program exporting `run`).
    let t = temen_llvm::translate_ll_path(&ll).expect("translate driver guest");
    temen_verify::verify_module(&t.module).expect("driver verifies");
    let entry = t
        .exports
        .iter()
        .find(|(n, _)| n == "run")
        .expect("exports run")
        .1;
    let sp = t.entry_sp as i64;

    // The real nifler phase (the committed child-entry asset).
    let nifler = temen_encode::decode_module(&nifler_bytes).expect("decode nifler_ce.temen");
    temen_verify::verify_module(&nifler).expect("nifler verifies");

    // A shared memfs seeded with the Nim source as `in.nim` (the guest names `/in.nim`; os_shim strips
    // the leading `/`). The handle observes the store the child writes, so we read `out.nif` back.
    let (factory, handle) = temen_run::fs::mem_fs_shared_factory(
        vec![("in.nim".into(), IN_NIM.as_bytes().to_vec())],
        vec![],
    );
    let factory = Arc::new(factory);

    let mut host = Host::new();
    let win = 1u64 << t.module.memory.expect("driver window").size_log2;
    let inst = host.grant_instantiator(0, win);
    let modh = host.grant_module(&nifler);
    // nifler's window, paid from this and returned when it ends.
    let budget = host.grant_budget(0, 1 << nifler.memory.expect("nifler window").size_log2, 0);
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
    // Everything the driver resolves by name.
    host.register_cap_name("inst", inst);
    host.register_cap_name("nifler", modh);
    host.register_cap_name("budget", budget);
    host.register_cap_name("fs", fs_h);
    host.register_cap_name("stdout", stdout_h);
    host.register_cap_name("exit", exit_h);

    let mut fuel = 200_000_000_000u64;
    let (r, _) = run_capture_reserved_with_host(
        &t.module,
        entry,
        &[Value::I64(sp)],
        &mut fuel,
        &[],
        0,
        &mut host,
    );
    let status = match r.expect("driver run").as_slice() {
        [Value::I64(x)] => *x,
        [Value::I32(x)] => *x as i64,
        other => panic!("driver result: {other:?}"),
    };
    assert_eq!(
        status, 0,
        "the driver guest spawned nifler and joined status 0"
    );

    // The `.nif` nifler wrote, read back out of the shared store the driver's child shared.
    let (files, _dirs) = handle.seed();
    let emitted = files
        .into_iter()
        .find(|(k, _)| k == "out.nif")
        .map(|(_, v)| v)
        .expect("nifler (as the driver guest's child) wrote no `out.nif`");
    assert_eq!(
        emitted,
        EXPECT_NIF.as_bytes(),
        "a Rust-on-Temen driver guest ran the real nifler phase, byte-identical to native"
    );
}
