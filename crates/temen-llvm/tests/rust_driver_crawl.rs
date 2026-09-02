//! **#1025 slice 3c — the driver-guest port, step 1: a Rust-on-Temen guest reads a phase's fs output.**
//! `rust_driver_nifler` proved a guest can op-13-spawn the real nifler phase. The import crawl needs one
//! more guest capability: after spawning a phase, the guest must **read that phase's output back out of
//! the shared memfs itself** — the crawl reads each module's `.p.deps.nif` to discover its imports, then
//! spawns nifler on those. This test proves the guest-side fs read: the same driver guest spawns nifler
//! over a shared memfs, then `open`/`read`/`close`s the emitted `out.nif` **through the `fs` cap it
//! holds** (`__vm_host_call`, the §7 host-proc bridge — ops 0/1/4), returning the byte count it read.
//!
//! The host asserts the guest read exactly the bytes native nifler emits (the committed `basic.p.nif`
//! fixture) *and* that those bytes match what the host reads from the same shared store — so the guest's
//! read and the parent's read agree. This is the fs-read half of the crawl loop; parsing the deps and
//! spawning the discovered imports is the next step. Gated to Linux + `rustc` + `gzip`.

#![cfg(target_os = "linux")]

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;

use temen_interp::{
    run_capture_reserved_with_host, ForkedProc, Host, HostProc, HostProcFork, StreamRole, Value,
};

const NIFLER_CE_GZ: &[u8] = include_bytes!("../../temen-run/demos/nifler_temen/nifler_ce.temen.gz");
const IN_NIM: &str = include_str!("../../temen-run/demos/nifler_temen/inputs/basic.nim");
const EXPECT_NIF: &str = include_str!("../../temen-run/demos/nifler_temen/expected/basic.p.nif");

// The Rust-on-Temen driver guest. It op-13-spawns nifler `p /in.nim /out.nif` over the re-granted memfs
// (exactly as `rust_driver_nifler`), then — the new step — reads `out.nif` back through its own `fs`
// cap: `open("out.nif", O_READ)`, `read` into a scratch buffer, `close`. Returns the bytes read (or a
// negative sentinel on any failure), which the host checks against native nifler's `.p.nif` length.
const GUEST_SRC: &str = r##"
#![no_std]
#![allow(internal_features)]
#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! { loop {} }
#[no_mangle]
pub extern "C" fn rust_eh_personality() {}

#[repr(C, align(65536))]
struct Pool([u8; 33619968]); // 32 MiB + 64 KiB — a 2^26 window with headroom above the 16 MiB carve
static mut POOL: Pool = Pool([0; 33619968]);

extern "C" {
    fn __vm_cap_resolve(name: *const u8, len: i64) -> i32;
    fn __vm_instantiate(
        inst: i32, module: i64, grants_ptr: i64, grants_n: i64,
        entry: i64, off: i64, size_log2: i64, quota: i64,
    ) -> i64;
    fn __vm_join(inst: i32, child: i64) -> i64;
    fn __vm_host_call(handle: i32, op: i32, a: i64, b: i64, c: i64, d: i64) -> i64;
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
        let nm = (base + 64) as *mut u8;
        let fsn = b"fs"; let son = b"stdout"; let exn = b"exit";
        let mut k = 0; while k < 2 { nm.add(k).write(fsn[k]); k += 1; }
        let mut k = 0; while k < 6 { nm.add(2 + k).write(son[k]); k += 1; }
        let mut k = 0; while k < 4 { nm.add(8 + k).write(exn[k]); k += 1; }
        put_rec(base, 0, base + 64, 2, fs);
        put_rec(base, 1, base + 66, 6, out);
        put_rec(base, 2, base + 72, 4, ex);
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
        let status = __vm_join(inst, child);
        if status != 0 && status != 5 { return -100 + status; }

        // --- the new step: read `out.nif` back through the guest's own `fs` cap ---
        // memfs keys are relative (os_shim stripped nifler's leading `/`), so open "out.nif". fs ops on
        // the HOST_PROC cap: 0 = open(path, len, flags, 0) -> fd; 1 = read(fd, buf, cap, 0) -> n; 4 =
        // close(fd, …). O_READ = 1.
        let path = (base + 128) as *mut u8;
        let on = b"out.nif";
        let mut k = 0; while k < 7 { path.add(k).write(on[k]); k += 1; }
        let fd = __vm_host_call(fs, 0, base + 128, 7, 1, 0);
        if fd < 0 { return -200; }
        let rbuf = base + 8192;
        let n = __vm_host_call(fs, 1, fd, rbuf, 65536, 0);
        let _ = __vm_host_call(fs, 4, fd, 0, 0, 0);
        n
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
fn rust_driver_guest_reads_a_phase_output_from_the_memfs() {
    let dir = std::env::temp_dir().join(format!("rust_driver_crawl_{}", std::process::id()));
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

    let t = temen_llvm::translate_ll_path(&ll).expect("translate driver guest");
    temen_verify::verify_module(&t.module).expect("driver verifies");
    let entry = t
        .exports
        .iter()
        .find(|(n, _)| n == "run")
        .expect("exports run")
        .1;
    let sp = t.entry_sp as i64;

    let nifler = temen_encode::decode_module(&nifler_bytes).expect("decode nifler_ce.temen");
    temen_verify::verify_module(&nifler).expect("nifler verifies");

    let (factory, handle) = temen_run::fs::mem_fs_shared_factory(
        vec![("in.nim".into(), IN_NIM.as_bytes().to_vec())],
        vec![],
    );
    let factory = Arc::new(factory);

    let mut host = Host::new();
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
    let read_n = match r.expect("driver run").as_slice() {
        [Value::I64(x)] => *x,
        [Value::I32(x)] => *x as i64,
        other => panic!("driver result: {other:?}"),
    };

    // The bytes the host reads from the shared store nifler wrote — the oracle for the guest's read.
    let (files, _dirs) = handle.seed();
    let emitted = files
        .into_iter()
        .find(|(k, _)| k == "out.nif")
        .map(|(_, v)| v)
        .expect("nifler wrote no `out.nif`");
    assert_eq!(
        emitted,
        EXPECT_NIF.as_bytes(),
        "nifler emitted byte-identical NIF to native (the shared store)"
    );
    assert_eq!(
        read_n as usize,
        emitted.len(),
        "the guest read the whole `.p.nif` back through its own fs cap — the crawl's fs-read half"
    );
}
