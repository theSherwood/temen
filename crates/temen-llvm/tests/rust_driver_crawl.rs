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

// Slice 1b — the crawl loop in the guest: spawn nifler on the main module, read its `.p.deps.nif`,
// parse the first `(import IDENT)`, build the import's file path, and spawn nifler on it. `spawn` lays
// argv into a carve and op-13-spawns nifler (the two runs use two disjoint 16 MiB carves in the 64 MiB
// window). The deps parser handles the bare/local import form `(import foo)` — the infix `std/os` form
// is skipped here (its multi-segment path resolution is slice 1c's full `parse_imports` port). Returns
// the byte length of the discovered module's `.p.nif` read back through fs, or a negative sentinel.
const CRAWL_SRC: &str = r##"
#![no_std]
#![allow(internal_features)]
#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! { loop {} }
#[no_mangle]
pub extern "C" fn rust_eh_personality() {}

#[repr(C, align(65536))]
struct Pool([u8; 67043328]); // 64 MiB - 64 KiB → a 2^26 window holding two 16 MiB carves
static mut POOL: Pool = Pool([0; 67043328]);

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

// argv[] for `nifler --portablePaths --deps parse <inp> <outp>`, seeded at `carve + 16512`; op-13 spawn
// nifler into `[carve, carve+2^24)` and join. `inp`/`outp` are (ptr,len) byte ranges in the guest window.
unsafe fn spawn_nifler(
    inst: i32, nifler: i32, base: i64, carve: i64,
    inp: *const u8, inl: usize, outp: *const u8, outl: usize,
) -> i64 {
    let argv = (carve + 16512) as *mut u8;
    (argv as *mut u32).add(0).write(6); // argc
    (argv as *mut u32).add(1).write(0); // envc
    let mut p = 8usize;
    let fixed: [&[u8]; 4] = [b"nifler", b"--portablePaths", b"--deps", b"parse"];
    let mut ai = 0;
    while ai < 4 {
        let s = fixed[ai];
        let mut j = 0;
        while j < s.len() { argv.add(p).write(s[j]); p += 1; j += 1; }
        argv.add(p).write(0); p += 1;
        ai += 1;
    }
    let mut j = 0; while j < inl { argv.add(p).write(*inp.add(j)); p += 1; j += 1; }
    argv.add(p).write(0); p += 1;
    let mut j = 0; while j < outl { argv.add(p).write(*outp.add(j)); p += 1; j += 1; }
    argv.add(p).write(0);
    let child = __vm_instantiate(inst, nifler as i64, base, 3, 0, carve, 24, 0);
    __vm_join(inst, child)
}

// Open `name`(len) O_READ through fs, read up to `cap` bytes into `buf`, close. Returns bytes read (<0 fail).
unsafe fn fs_read(fs: i32, name: i64, len: i64, buf: i64, cap: i64) -> i64 {
    let fd = __vm_host_call(fs, 0, name, len, 1, 0);
    if fd < 0 { return -1; }
    let n = __vm_host_call(fs, 1, fd, buf, cap, 0);
    let _ = __vm_host_call(fs, 4, fd, 0, 0, 0);
    n
}

fn is_ident(c: u8) -> bool {
    (c >= b'a' && c <= b'z') || (c >= b'A' && c <= b'Z') || (c >= b'0' && c <= b'9') || c == b'_'
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
        let carve1 = (base + 128 + mask) & !mask;
        let carve2 = carve1 + (1 << 24);

        // Spawn 1: nifler on the main module. Input/output paths live in the low window (below carve1).
        let mainp = (base + 256) as *mut u8;
        let mm = b"/main.nim"; let mut k = 0; while k < 9 { mainp.add(k).write(mm[k]); k += 1; }
        let mainop = (base + 272) as *mut u8;
        let mo = b"/main.p.nif"; let mut k = 0; while k < 11 { mainop.add(k).write(mo[k]); k += 1; }
        let s1 = spawn_nifler(
            inst, nifler, base, carve1,
            (base + 256) as *const u8, 9, (base + 272) as *const u8, 11,
        );
        if s1 != 0 && s1 != 5 { return -100 + s1; }

        // Read main.p.deps.nif (memfs key is relative — os_shim stripped the leading `/`).
        let depn = (base + 288) as *mut u8;
        let dn = b"main.p.deps.nif"; let mut k = 0; while k < 15 { depn.add(k).write(dn[k]); k += 1; }
        let dbuf = base + 4096;
        let dlen = fs_read(fs, base + 288, 15, dbuf, 65536);
        if dlen <= 0 { return -200; }

        // Parse the first `(import IDENT)` — the bare/local form. Skip `(import (` (infix/prefix).
        let needle = b"(import ";
        let mut i: i64 = 0;
        let mut idpos: i64 = -1;
        while i + 8 <= dlen {
            let mut m = true; let mut j = 0;
            while j < 8 { if *((dbuf + i + j) as *const u8) != needle[j as usize] { m = false; break; } j += 1; }
            if m {
                let c = *((dbuf + i + 8) as *const u8);
                if c != b'(' { idpos = i + 8; break; }
            }
            i += 1;
        }
        if idpos < 0 { return -300; }
        // Copy the identifier into a name buffer at base+512.
        let namep = (base + 512) as *mut u8;
        let mut idlen: usize = 0;
        loop {
            let c = *((dbuf + idpos + idlen as i64) as *const u8);
            if !is_ident(c) { break; }
            namep.add(idlen).write(c);
            idlen += 1;
            if idlen >= 64 { break; }
        }
        if idlen == 0 { return -301; }

        // Build "/IDENT.nim" at base+640 and "/IDENT.p.nif" at base+768.
        let inp = (base + 640) as *mut u8;
        inp.write(b'/');
        let mut k = 0; while k < idlen { inp.add(1 + k).write(*namep.add(k)); k += 1; }
        let suf = b".nim"; let mut k = 0; while k < 4 { inp.add(1 + idlen + k).write(suf[k]); k += 1; }
        let in_len = 1 + idlen + 4;
        let outp = (base + 768) as *mut u8;
        outp.write(b'/');
        let mut k = 0; while k < idlen { outp.add(1 + k).write(*namep.add(k)); k += 1; }
        let osuf = b".p.nif"; let mut k = 0; while k < 6 { outp.add(1 + idlen + k).write(osuf[k]); k += 1; }
        let out_len = 1 + idlen + 6;

        // Spawn 2: nifler on the discovered import — the crawl's dependent spawn.
        let s2 = spawn_nifler(
            inst, nifler, base, carve2,
            (base + 640) as *const u8, in_len, (base + 768) as *const u8, out_len,
        );
        if s2 != 0 && s2 != 5 { return -400 + s2; }

        // Read the discovered module's .p.nif back (relative key = "IDENT.p.nif").
        let rkey = (base + 896) as *mut u8;
        let mut k = 0; while k < idlen { rkey.add(k).write(*namep.add(k)); k += 1; }
        let rsuf = b".p.nif"; let mut k = 0; while k < 6 { rkey.add(idlen + k).write(rsuf[k]); k += 1; }
        let rlen = fs_read(fs, base + 896, (idlen + 6) as i64, base + 4096, 65536);
        rlen
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

/// Translate `guest_src` (a `no_std` cdylib exporting `run`), grant it `inst`/`nifler`/`fs`/`stdout`/
/// `exit` over a memfs seeded with `seed`, run it, and return `(run() result, shared handle)`.
fn run_driver_guest(
    guest_src: &str,
    seed: Vec<(String, Vec<u8>)>,
    nifler_bytes: &[u8],
) -> Option<(i64, temen_run::fs::MemFsHandle)> {
    let dir = std::env::temp_dir().join(format!(
        "rust_driver_crawl_{}_{}",
        std::process::id(),
        seed.len()
    ));
    std::fs::create_dir_all(&dir).ok()?;
    let src = dir.join("g.rs");
    let ll = dir.join("g.ll");
    std::fs::write(&src, guest_src).ok()?;
    if !rustc_emit_ll(&src, &ll) {
        return None;
    }
    let t = temen_llvm::translate_ll_path(&ll).expect("translate driver guest");
    temen_verify::verify_module(&t.module).expect("driver verifies");
    let entry = t
        .exports
        .iter()
        .find(|(n, _)| n == "run")
        .expect("exports run")
        .1;
    let sp = t.entry_sp as i64;

    let nifler = temen_encode::decode_module(nifler_bytes).expect("decode nifler_ce.temen");
    temen_verify::verify_module(&nifler).expect("nifler verifies");

    let (factory, handle) = temen_run::fs::mem_fs_shared_factory(seed, vec![]);
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

    let mut fuel = 400_000_000_000u64;
    let (r, _) = run_capture_reserved_with_host(
        &t.module,
        entry,
        &[Value::I64(sp)],
        &mut fuel,
        &[],
        0,
        &mut host,
    );
    let v = match r.expect("driver run").as_slice() {
        [Value::I64(x)] => *x,
        [Value::I32(x)] => *x as i64,
        other => panic!("driver result: {other:?}"),
    };
    Some((v, handle))
}

#[test]
fn rust_driver_guest_crawls_one_import() {
    let Some(nifler_bytes) = inflate(NIFLER_CE_GZ) else {
        eprintln!("note: skipping (gzip unavailable)");
        return;
    };
    // main.nim imports a local `helper`; the guest must discover it from main's `.p.deps.nif` and spawn
    // nifler on `/helper.nim` itself. Both modules live in the shared memfs.
    let seed = vec![
        ("main.nim".into(), b"import helper\n\nlet z = 1\n".to_vec()),
        ("helper.nim".into(), b"proc h(): int = 42\n".to_vec()),
    ];
    let Some((rlen, handle)) = run_driver_guest(CRAWL_SRC, seed, &nifler_bytes) else {
        eprintln!("note: skipping (rustc unavailable)");
        return;
    };

    let (files, _dirs) = handle.seed();
    let get = |k: &str| files.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone());

    // Phase 1 (main) ran: its parse + deps outputs are present.
    assert!(get("main.p.nif").is_some(), "nifler ran on main.nim");
    let deps = get("main.p.deps.nif").expect("main.p.deps.nif present");
    assert!(
        String::from_utf8_lossy(&deps).contains("(import helper)"),
        "main's deps record the `import helper`"
    );

    // The crawl's payoff: the guest parsed the import and spawned nifler on `/helper.nim` — so the
    // discovered module's `.p.nif` exists, is valid NIF, and the guest read it back (rlen > 0).
    let hnif = get("helper.p.nif").expect(
        "the driver guest discovered `helper` from main's deps and spawned nifler on /helper.nim",
    );
    assert!(
        hnif.starts_with(b"(.nif27)"),
        "the discovered module's NIF has the nifler header: {:?}",
        String::from_utf8_lossy(&hnif[..hnif.len().min(16)])
    );
    assert_eq!(
        rlen as usize,
        hnif.len(),
        "the guest read the discovered module's `.p.nif` back through fs — a full 2-module crawl \
         driven entirely by the guest (spawn → read deps → parse import → spawn dependent)"
    );
}
