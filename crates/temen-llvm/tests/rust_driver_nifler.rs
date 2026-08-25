//! **#1011 slice 3c — a Rust-on-Temen driver *guest* runs the real nifler phase via op-13.** The
//! compiler driver's endgame is to move phase orchestration **into** the sandbox: instead of the native
//! harness (`spawn_child_fs` / `nimc::compile_nim`) issuing the op-13 spawn, a Rust-on-Temen guest does
//! it. `rust_guest_op13` proved a guest can `__vm_instantiate`/`__vm_join` a *toy* child; this upgrades
//! that to the **real `nifler` phase**: the guest resolves the `Instantiator`, the `nifler` module, and
//! `fs`/`stdout`/`exit` by name, seeds argv `nifler p /in.nim /out.nif` into the child's carve, lays the
//! three grant records, op-13-spawns nifler into a 16 MiB carve, and joins it. The host seeds the Nim
//! source into the shared memfs and reads the emitted `.nif` back — **byte-identical to native nifler**.
//!
//! This is the first phase of the compiler driver running *on Temen* (the orchestration, not just the
//! phase). It reuses the committed `nifler_ce.temen.gz` child-entry asset (the same one
//! `nifler_child_asset.rs` gates). Gated to Linux + `rustc` + `gzip`; skips cleanly otherwise.

#![cfg(target_os = "linux")]

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;

use temen_interp::{
    run_capture_reserved_with_host, ForkedProc, Host, HostProc, HostProcFork, StreamRole, Value,
};

/// The committed child-entry nifler asset (built by `build_nifler_temen.sh`), shared with the gate.
const NIFLER_CE_GZ: &[u8] = include_bytes!("../../temen-run/demos/nifler_temen/nifler_ce.temen.gz");
/// One corpus input + its committed native-`nifler` `.p.nif` (the oracle fixture).
const IN_NIM: &str = include_str!("../../temen-run/demos/nifler_temen/inputs/basic.nim");
const EXPECT_NIF: &str = include_str!("../../temen-run/demos/nifler_temen/expected/basic.p.nif");

// The Rust-on-Temen driver guest. `no_std`, no allocator — a 32 MiB `POOL` forces a window that holds a
// 16 MiB carve at offset 2^24, plus the low grant records / names. `run()` resolves the caps by name,
// lays argv `nifler p /in.nim /out.nif` at `carve + POWERBOX_ARGS_BASE (128)`, writes three 16-byte grant
// records `{fs, stdout, exit}` + their names, op-13-spawns nifler (entry 0) into `[carve, carve+2^24)`,
// and returns `join(child)` — nifler's status.
const GUEST_SRC: &str = r##"
#![no_std]
#![allow(internal_features)]
#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! { loop {} }
#[no_mangle]
pub extern "C" fn rust_eh_personality() {}

#[repr(C, align(65536))]
struct Pool([u8; 33619968]); // 32 MiB + 64 KiB — forces a 2^26 window with headroom above the carve
static mut POOL: Pool = Pool([0; 33619968]);

extern "C" {
    fn __vm_cap_resolve(name: *const u8, len: i64) -> i32;
    fn __vm_instantiate(
        inst: i32, module: i64, grants_ptr: i64, grants_n: i64,
        entry: i64, off: i64, size_log2: i64, quota: i64,
    ) -> i64;
    fn __vm_join(inst: i32, child: i64) -> i64;
}

unsafe fn put_rec(base: i64, i: i64, name_off: i64, name_len: u32, handle: i32) {
    let rec = (base + i * 16) as *mut u32;
    rec.add(0).write(name_off as u32);
    rec.add(1).write(name_len);
    rec.add(2).write(handle as u32);
    rec.add(3).write(0);
}

#[no_mangle]
pub extern "C" fn run() -> i64 {
    unsafe {
        let inst = __vm_cap_resolve(b"inst".as_ptr(), 4);
        let nifler = __vm_cap_resolve(b"nifler".as_ptr(), 6);
        let fs = __vm_cap_resolve(b"fs".as_ptr(), 2);
        let out = __vm_cap_resolve(b"stdout".as_ptr(), 6);
        let ex = __vm_cap_resolve(b"exit".as_ptr(), 4);
        if inst < 0 || nifler < 0 || fs < 0 || out < 0 || ex < 0 { return -1; }
        let base = core::ptr::addr_of_mut!(POOL) as i64;
        // Names at base+64: "fs"(2) "stdout"(6) "exit"(4).
        let nm = (base + 64) as *mut u8;
        let fsn = b"fs"; let son = b"stdout"; let exn = b"exit";
        let mut k = 0; while k < 2 { nm.add(k).write(fsn[k]); k += 1; }
        let mut k = 0; while k < 6 { nm.add(2 + k).write(son[k]); k += 1; }
        let mut k = 0; while k < 4 { nm.add(8 + k).write(exn[k]); k += 1; }
        put_rec(base, 0, base + 64, 2, fs);
        put_rec(base, 1, base + 66, 6, out);
        put_rec(base, 2, base + 72, 4, ex);
        // The carve: a 16 MiB (2^24) window, its offset **rounded up to a 2^24 boundary** above the low
        // records (op-13 requires a power-of-two-aligned sub-window). #964/#1094: nifler_ce is guarded,
        // so its `_start` reads argv one guard up — seed at `carve + module_args_base` = carve +
        // POWERBOX_NULL_GUARD(16384) + POWERBOX_ARGS_BASE(128) = carve + 16512. The guest is a standalone
        // .ll (no temen_ir constants), so the value is spelled out. The grant records/cap-names stay at
        // `base + …` in the parent window, read by the op-13 handler — never by the guarded child.
        let mask: i64 = (1 << 24) - 1;
        let carve = (base + 128 + mask) & !mask;
        let argv = (carve + 16512) as *mut u8;
        (argv as *mut u32).add(0).write(4); // argc
        (argv as *mut u32).add(1).write(0); // envc
        let mut p = 8usize;
        let args: [&[u8]; 4] = [b"nifler", b"p", b"/in.nim", b"/out.nif"];
        let mut ai = 0;
        while ai < 4 {
            let s = args[ai];
            let mut j = 0;
            while j < s.len() { argv.add(p).write(s[j]); p += 1; j += 1; }
            argv.add(p).write(0); p += 1;
            ai += 1;
        }
        let child = __vm_instantiate(inst, nifler as i64, base, 3, 0, carve, 24, 0);
        __vm_join(inst, child)
    }
}
"##;

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
fn rust_driver_guest_runs_real_nifler_via_op13() {
    let dir = std::env::temp_dir().join(format!("rust_driver_nifler_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("driver.rs");
    let ll = dir.join("driver.ll");
    std::fs::write(&src, GUEST_SRC).unwrap();
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
    // The driver's window (from its 32 MiB POOL) — grant an Instantiator over it so its op-13 carve fits.
    let win = 1u64 << t.module.memory.expect("driver window").size_log2;
    let inst = host.grant_instantiator(0, win);
    let modh = host.grant_module(&nifler);
    let fs_init: HostProc = (*factory)();
    let fs_fork: HostProcFork = {
        let f = Arc::clone(&factory);
        Arc::new(move |_pid| ForkedProc::shared((*f)()))
    };
    let fs_h = host.grant_host_proc_forkable(fs_init, fs_fork);
    let stdout_h = host.grant_stream(StreamRole::Out);
    let exit_h = host.grant_exit();
    // Everything the driver resolves by name.
    host.register_cap_name("inst", inst);
    host.register_cap_name("nifler", modh);
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
        "the driver guest spawned nifler via op-13 and joined status 0"
    );

    // The `.nif` nifler wrote, read back out of the shared store the driver's child shared.
    let (files, _dirs) = handle.seed();
    let emitted = files
        .into_iter()
        .find(|(k, _)| k == "out.nif")
        .map(|(_, v)| v)
        .expect("nifler (as the driver guest's op-13 child) wrote no `out.nif`");
    assert_eq!(
        emitted,
        EXPECT_NIF.as_bytes(),
        "a Rust-on-Temen driver guest ran the real nifler phase via op-13, byte-identical to native"
    );
}
