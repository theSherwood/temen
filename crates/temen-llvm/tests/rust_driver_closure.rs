//! **#1025 slice 3c — the driver-guest port, step 3: a Rust-on-Temen guest crawls an import closure to
//! fixpoint.** Steps 1 & 2 (`rust_driver_crawl`) proved the guest can read a phase's fs output and drive
//! one dependent spawn. This lifts `nimc.rs`'s **whole phase-1 crawl loop** into the guest: a worklist
//! seeded with the main module, and for each unvisited module — op-13-spawn nifler (`--portablePaths
//! --deps parse`), read its `.p.deps.nif` through the guest's own fs cap, parse **every** `import`
//! (the `(infix / std x)` form `nimc::parse_imports` handles → `/lib/std/x.nim`), and enqueue each newly
//! discovered module — until the worklist drains. Sequential spawns alternate two 16 MiB carves.
//!
//! The closure here is `main → std/foo → std/bar` (the stdlib-style infix imports the playground uses).
//! The host asserts the guest crawled all three, and that each module's `.p.nif` landed in the shared
//! memfs — the full import-closure discovery `nimc::compile_nim` does, now running inside the sandbox on
//! existing on-ramp builtins (`__vm_instantiate`/`__vm_join`/`__vm_host_call`). Cache-correct stem names
//! (`module_suffix`) and the prefix/`when`-guard import forms are the next slice. Gated Linux + rustc + gzip.

#![cfg(target_os = "linux")]

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;

use temen_interp::{
    run_capture_reserved_with_host, ForkedProc, Host, HostProc, HostProcFork, StreamRole, Value,
};

const NIFLER_CE_GZ: &[u8] = include_bytes!("../../temen-run/demos/nifler_temen/nifler_ce.temen.gz");

const CLOSURE_SRC: &str = r##"#![no_std]
#![allow(internal_features)]
#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! { loop {} }
#[no_mangle]
pub extern "C" fn rust_eh_personality() {}

#[repr(C, align(65536))]
struct Pool([u8; 67043328]); // 64 MiB - 64 KiB → two 16 MiB carves
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

unsafe fn wr(p: i64, b: u8) { (p as *mut u8).write(b); }
unsafe fn rd(p: i64) -> u8 { *(p as *const u8) }

unsafe fn put_rec(base: i64, i: i64, name_off: i64, name_len: u32, handle: i32) {
    let rec = (base + i * 16) as *mut u32;
    rec.add(0).write(name_off as u32);
    rec.add(1).write(name_len);
    rec.add(2).write(handle as u32);
    rec.add(3).write(0);
}

// argv `nifler --portablePaths --deps parse <in> <out>` at carve+16512; op-13 spawn + join.
unsafe fn spawn_nifler(inst: i32, nifler: i32, base: i64, carve: i64, inp: i64, inl: i64, outp: i64, outl: i64) -> i64 {
    let argv = (carve + 16512) as *mut u8;
    (argv as *mut u32).add(0).write(6);
    (argv as *mut u32).add(1).write(0);
    let mut p = 8i64;
    let fixed: [&[u8]; 4] = [b"nifler", b"--portablePaths", b"--deps", b"parse"];
    let mut ai = 0;
    while ai < 4 {
        let s = fixed[ai];
        let mut j = 0;
        while j < s.len() { wr(carve + 16512 + p, s[j]); p += 1; j += 1; }
        wr(carve + 16512 + p, 0); p += 1;
        ai += 1;
    }
    let mut j = 0i64; while j < inl { wr(carve + 16512 + p, rd(inp + j)); p += 1; j += 1; }
    wr(carve + 16512 + p, 0); p += 1;
    let mut j = 0i64; while j < outl { wr(carve + 16512 + p, rd(outp + j)); p += 1; j += 1; }
    wr(carve + 16512 + p, 0);
    let child = __vm_instantiate(inst, nifler as i64, base, 3, 0, carve, 24, 0);
    __vm_join(inst, child)
}

unsafe fn fs_read(fs: i32, name: i64, len: i64, buf: i64, cap: i64) -> i64 {
    let fd = __vm_host_call(fs, 0, name, len, 1, 0);
    if fd < 0 { return -1; }
    let n = __vm_host_call(fs, 1, fd, buf, cap, 0);
    let _ = __vm_host_call(fs, 4, fd, 0, 0, 0);
    n
}

unsafe fn eq(a: i64, al: i64, b: i64, bl: i64) -> bool {
    if al != bl { return false; }
    let mut i = 0i64;
    while i < al { if rd(a + i) != rd(b + i) { return false; } i += 1; }
    true
}

fn is_ident(c: u8) -> bool {
    (c >= b'a' && c <= b'z') || (c >= b'A' && c <= b'Z') || (c >= b'0' && c <= b'9') || c == b'_'
}

const WLL: i64 = 256; // 16 lengths (i32) at base+256
const WL: i64 = 512;  // 16 path slots × 96 bytes at base+512
const SCR: i64 = 2048; // scratch for out/deps/import paths
const DBUF: i64 = 4096; // deps read buffer (64 KiB, below carve)

// Append `path`(len) to the worklist if not already present. Returns new count.
unsafe fn add_path(base: i64, nwl: i64, path: i64, plen: i64) -> i64 {
    let mut i = 0i64;
    while i < nwl {
        let slot = base + WL + i * 96;
        let slen = *((base + WLL + i * 4) as *const u32) as i64;
        if eq(slot, slen, path, plen) { return nwl; }
        i += 1;
    }
    if nwl >= 16 { return nwl; }
    let slot = base + WL + nwl * 96;
    let mut k = 0i64; while k < plen { wr(slot + k, rd(path + k)); k += 1; }
    *((base + WLL + nwl * 4) as *mut u32) = plen as u32;
    nwl + 1
}

// Parse every `(import (infix / seg...))` in dbuf[0..n]; add each `/lib/seg/....nim` to the worklist.
unsafe fn scan_imports(base: i64, dbuf: i64, n: i64, mut nwl: i64) -> i64 {
    let needle = b"(import"; // nifler emits `(import` then whitespace then `(infix ...)` (or a bare ident)
    let inf = b"(infix";
    let mut i = 0i64;
    while i + 7 <= n {
        let mut m = true; let mut j = 0i64;
        while j < 7 { if rd(dbuf + i + j) != needle[j as usize] { m = false; break; } j += 1; }
        if !m { i += 1; continue; }
        // skip whitespace after "(import", then require "(infix" (the multi-segment form nimc resolves;
        // a bare `(import foo)` has no infix and is skipped, matching parse_imports).
        let mut q = i + 7;
        while q < n { let c = rd(dbuf + q); if c == b' ' || c == 10 || c == 9 { q += 1; } else { break; } }
        let mut is_inf = true; let mut j = 0i64;
        while j < 6 { if rd(dbuf + q + j) != inf[j as usize] { is_inf = false; break; } j += 1; }
        if !is_inf { i += 7; continue; }
        // build "/lib/" then append each ident segment separated by "/", then ".nim".
        let ip = base + SCR;
        let pre = b"/lib/"; let mut w = 0i64;
        let mut kk = 0i64; while kk < 5 { wr(ip + w, pre[kk as usize]); w += 1; kk += 1; }
        // walk tokens from after "(infix" until the matching ')'.
        let mut p = q + 6;
        let mut depth = 1i32; // the '(' of "(infix" is consumed
        let mut nseg = 0i64;
        while p < n && depth > 0 {
            let c = rd(dbuf + p);
            if c == b'(' { depth += 1; p += 1; continue; }
            if c == b')' { depth -= 1; p += 1; continue; }
            if c == b' ' || c == 10 || c == 9 { p += 1; continue; }
            if c == b'/' { p += 1; continue; }
            if is_ident(c) {
                if nseg > 0 { wr(ip + w, b'/'); w += 1; }
                while p < n { let cc = rd(dbuf + p); if is_ident(cc) { wr(ip + w, cc); w += 1; p += 1; } else { break; } }
                nseg += 1;
            } else { p += 1; }
        }
        if nseg > 0 {
            let suf = b".nim"; let mut kk = 0i64; while kk < 4 { wr(ip + w, suf[kk as usize]); w += 1; kk += 1; }
            nwl = add_path(base, nwl, ip, w);
        }
        i = p;
    }
    nwl
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
        let carve0 = (base + 128 + mask) & !mask;
        let carve1 = carve0 + (1 << 24);

        // Seed the worklist with "/main.nim".
        let seed = b"/main.nim";
        let sp = base + SCR;
        let mut k = 0i64; while k < 9 { wr(sp + k, seed[k as usize]); k += 1; }
        let mut nwl = add_path(base, 0, sp, 9);

        let mut i = 0i64;
        let mut crawled = 0i64;
        while i < nwl {
            let path = base + WL + i * 96;
            let plen = *((base + WLL + i * 4) as *const u32) as i64;
            // out = path[..plen-4] + ".p.nif"   (strip ".nim")
            let outp = base + SCR + 128;
            let mut w = 0i64; let mut k = 0i64;
            while k < plen - 4 { wr(outp + w, rd(path + k)); w += 1; k += 1; }
            let osuf = b".p.nif"; let mut kk = 0i64; while kk < 6 { wr(outp + w, osuf[kk as usize]); w += 1; kk += 1; }
            let outl = w;
            let carve = if i % 2 == 0 { carve0 } else { carve1 };
            let st = spawn_nifler(inst, nifler, base, carve, path, plen, outp, outl);
            if st != 0 && st != 5 { return -100 + st; }

            // deps key = outp[1..outl-6] + ".p.deps.nif"  (relative; strip leading '/' and ".p.nif")
            let dk = base + SCR + 256;
            let mut w2 = 0i64; let mut k2 = 1i64;
            while k2 < outl - 6 { wr(dk + w2, rd(outp + k2)); w2 += 1; k2 += 1; }
            let dsuf = b".p.deps.nif"; let mut kk = 0i64; while kk < 11 { wr(dk + w2, dsuf[kk as usize]); w2 += 1; kk += 1; }
            let n = fs_read(fs, dk, w2, base + DBUF, 65536);
            if n > 0 { nwl = scan_imports(base, base + DBUF, n, nwl); }
            crawled += 1;
            i += 1;
        }
        crawled
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
    let mut stdin = c.stdin.take().unwrap();
    let gz = gz.to_vec();
    let w = std::thread::spawn(move || {
        let _ = stdin.write_all(&gz);
    });
    let out = c.wait_with_output().unwrap();
    w.join().unwrap();
    out.status.success().then_some(out.stdout)
}

#[test]
fn rust_driver_guest_crawls_an_import_closure_to_fixpoint() {
    let Some(nifler_bytes) = inflate(NIFLER_CE_GZ) else {
        eprintln!("note: skipping (gzip unavailable)");
        return;
    };
    let dir = std::env::temp_dir().join(format!("rust_driver_closure_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("g.rs");
    let ll = dir.join("g.ll");
    std::fs::write(&src, CLOSURE_SRC).unwrap();
    if !rustc_emit_ll(&src, &ll) {
        eprintln!("note: skipping (rustc unavailable)");
        return;
    }
    let t = temen_llvm::translate_ll_path(&ll).expect("translate closure guest");
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

    // The closure: main imports std/foo, which imports std/bar (stdlib-style infix imports).
    let (factory, handle) = temen_run::fs::mem_fs_shared_factory(
        vec![
            ("main.nim".into(), b"import std/foo\n".to_vec()),
            ("lib/std/foo.nim".into(), b"import std/bar\n".to_vec()),
            ("lib/std/bar.nim".into(), b"proc b(): int = 1\n".to_vec()),
        ],
        vec!["lib".into(), "lib/std".into()],
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

    let mut fuel = 600_000_000_000u64;
    let (r, _) = run_capture_reserved_with_host(
        &t.module,
        entry,
        &[Value::I64(sp)],
        &mut fuel,
        &[],
        0,
        &mut host,
    );
    let crawled = match r.expect("driver run").as_slice() {
        [Value::I64(x)] => *x,
        [Value::I32(x)] => *x as i64,
        other => panic!("driver result: {other:?}"),
    };
    assert_eq!(
        crawled, 3,
        "the guest crawled the whole closure: main → std/foo → std/bar"
    );

    let (files, _dirs) = handle.seed();
    let has = |k: &str| files.iter().any(|(n, _)| n == k);
    for k in ["main.p.nif", "lib/std/foo.p.nif", "lib/std/bar.p.nif"] {
        assert!(
            has(k),
            "the crawl produced `{k}` — a nifler parse for every module in the closure"
        );
    }
    // The discovery is deps-driven, not hardcoded: main's deps recorded the std/foo import.
    let deps = files
        .iter()
        .find(|(n, _)| n == "main.p.deps.nif")
        .map(|(_, v)| v)
        .expect("main deps");
    assert!(
        String::from_utf8_lossy(deps).contains("infix"),
        "main's deps carry the infix std/foo import the guest parsed"
    );
}
