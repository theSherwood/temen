//! **#1025 slice 3c — the driver-guest port, step 5: a stem-named import crawl (nimc's cache layout).**
//! Step 3 crawled the closure; step 4 ported `module_suffix`. This joins them: the guest crawls the
//! import closure to fixpoint AND names each module's nifler outputs by its nimony stem —
//! `nimcache/{stem}.p.nif` / `nimcache/{stem}.p.deps.nif` — exactly the paths `nimc::compile_nim`'s
//! phase-1 writes and its later phases (nimsem reads `{stem}.p.nif`) consume. The guest computes the
//! stem per module (`module_suffix` over the worklist path), spawns nifler with `--deps parse <src>
//! /nimcache/{stem}.p.nif`, reads `nimcache/{stem}.p.deps.nif` back through its fs cap, and enqueues
//! the discovered imports — until fixpoint.
//!
//! The host asserts the crawl produced a `nimcache/{stem}.p.nif` for every module in the closure, with
//! stems matching `nimc::module_suffix` (replicated as the oracle). This is `nimc`'s phase-1 crawl —
//! discovery + cache-correct naming — now running entirely inside the sandbox. Gated Linux + rustc + gzip.

#![cfg(target_os = "linux")]

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;

use temen_interp::{
    run_capture_reserved_with_host, ForkedProc, Host, HostProc, HostProcFork, StreamRole, Value,
};

const NIFLER_CE_GZ: &[u8] = include_bytes!("../../temen-run/demos/nifler_temen/nifler_ce.temen.gz");

const CRAWL_SRC: &str = r##"#![no_std]
#![allow(internal_features)]
#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! { loop {} }
#[no_mangle]
pub extern "C" fn rust_eh_personality() {}

#[repr(C, align(65536))]
struct Pool([u8; 67043328]); // 64 MiB - 64 KiB -> two 16 MiB carves
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

unsafe fn spawn_nifler(inst: i32, nifler: i32, base: i64, carve: i64, inp: i64, inl: i64, outp: i64, outl: i64) -> i64 {
    (( carve + 16512) as *mut u32).add(0).write(6);
    (( carve + 16512) as *mut u32).add(1).write(0);
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

// nimony's module-stem hash (nimc::uhash).
unsafe fn uhash(ptr: i64, len: i64) -> u32 {
    let mut h: u32 = 0;
    let mut i = 0i64;
    while i < len {
        let c = rd(ptr + i) as u32;
        h = h.wrapping_add(c);
        h = h.wrapping_add(h << 10);
        h ^= h >> 6;
        i += 1;
    }
    h = h.wrapping_add(h << 3);
    h ^= h >> 11;
    h = h.wrapping_add(h << 15);
    h
}
unsafe fn base36(mut id: u32, out: i64) -> i64 {
    let b36 = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut w = 0i64;
    while id > 0 { wr(out + w, b36[(id % 36) as usize]); w += 1; id /= 36; }
    w
}
// nimc::module_suffix over a worklist path -> stem written to `out`, returns len.
unsafe fn module_suffix(path: i64, plen: i64, out: i64) -> i64 {
    let mut r = path; let mut rlen = plen;
    if rlen > 0 && rd(r) == b'/' { r += 1; rlen -= 1; }
    if rlen >= 4 && rd(r) == b'l' && rd(r + 1) == b'i' && rd(r + 2) == b'b' && rd(r + 3) == b'/' { r += 4; rlen -= 4; }
    let mut nstart = 0i64; let mut k = 0i64;
    while k < rlen { if rd(r + k) == b'/' { nstart = k + 1; } k += 1; }
    let mut nlen = rlen - nstart;
    if nlen >= 4 && rd(r + nstart + nlen - 4) == b'.' && rd(r + nstart + nlen - 3) == b'n'
        && rd(r + nstart + nlen - 2) == b'i' && rd(r + nstart + nlen - 1) == b'm' { nlen -= 4; }
    let take = if nlen < 3 { nlen } else { 3 };
    let mut w = 0i64; let mut i = 0i64;
    while i < take { wr(out + w, rd(r + nstart + i)); w += 1; i += 1; }
    let h = uhash(r, rlen);
    w += base36(h, out + w);
    w
}

const WLL: i64 = 256;
const WL: i64 = 512;
const SCR: i64 = 2560;
const DBUF: i64 = 4096;

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

unsafe fn scan_imports(base: i64, dbuf: i64, n: i64, mut nwl: i64, dir: i64, dirlen: i64) -> i64 {
    let imp = b"(import";
    let frm = b"(fromimport";
    let inf = b"(infix";
    let pfx = b"(prefix";
    let mut i = 0i64;
    while i < n {
        // Match `(import` (7) or `(fromimport` (11) — nimc::parse_imports scans both keywords.
        let mut kwend = -1i64;
        if i + 7 <= n {
            let mut m = true; let mut j = 0i64;
            while j < 7 { if rd(dbuf + i + j) != imp[j as usize] { m = false; break; } j += 1; }
            if m { kwend = i + 7; }
        }
        if kwend < 0 && i + 11 <= n {
            let mut m = true; let mut j = 0i64;
            while j < 11 { if rd(dbuf + i + j) != frm[j as usize] { m = false; break; } j += 1; }
            if m { kwend = i + 11; }
        }
        if kwend < 0 { i += 1; continue; }
        // Skip whitespace after the keyword, then classify: `(infix …)` resolves under /lib
        // (`std/x`), `(prefix …)` resolves relative to the importer's dir (`./x`). A `(when …)`-guarded
        // import leads with `(when` and a bare `(import foo)` with an ident — both match neither and are
        // skipped, matching parse_imports (which drops `when`-guarded and non-infix/prefix blocks).
        let mut q = kwend;
        while q < n { let c = rd(dbuf + q); if c == b' ' || c == 10 || c == 9 { q += 1; } else { break; } }
        let mut is_inf = true; let mut j = 0i64;
        while j < 6 { if rd(dbuf + q + j) != inf[j as usize] { is_inf = false; break; } j += 1; }
        let mut is_pre = true; let mut j = 0i64;
        while j < 7 { if rd(dbuf + q + j) != pfx[j as usize] { is_pre = false; break; } j += 1; }
        if !is_inf && !is_pre { i = kwend; continue; }
        let ip = base + SCR;
        let start = if is_inf { q + 6 } else { q + 7 };
        let ret = resolve_segments(base, dbuf, start, n, is_inf, dir, dirlen, ip);
        let plen = ret & 0xffffff;
        if plen > 0 { nwl = add_path(base, nwl, ip, plen); }
        i = ret >> 24;
    }
    nwl
}

// Resolve an `(infix …)` / `(prefix …)` block into `ip`; returns `(end << 24) | path_len` (path_len 0
// if no segments). Uses inline copies (advancing the loop's own cursor) throughout: the on-ramp
// translator miscompiles a separate-counter inner write loop through a parameter pointer (spurious
// MemoryFault), so every byte copy walks the same cursor the loop condition tests.
unsafe fn resolve_segments(base: i64, dbuf: i64, start: i64, n: i64, is_inf: bool, dir: i64, dirlen: i64, ip: i64) -> i64 {
    let _ = base;
    let mut w = 0i64;
    if is_inf {
        let lib = b"/lib/"; let mut kk = 0i64; while kk < 5 { wr(ip + w, lib[kk as usize]); w += 1; kk += 1; }
    } else {
        let mut d = 0i64; while d < dirlen { wr(ip + w, rd(dir + d)); w += 1; d += 1; }
        wr(ip + w, b'/'); w += 1;
    }
    let mut p = start;
    let mut depth = 1i32;
    let mut nseg = 0i64;
    while p < n && depth > 0 {
        let c = rd(dbuf + p);
        if c == b'(' { depth += 1; p += 1; continue; }
        if c == b')' { depth -= 1; p += 1; continue; }
        if c == b'/' || c == b' ' || c == 10 || c == 9 { p += 1; continue; }
        if is_ident(c) {
            // prefix drops the escaped-dot run "2E" (nifler emits `\2E/` for `./`); peek for it.
            let is2e = !is_inf && c == b'2' && p + 1 < n && rd(dbuf + p + 1) == b'E'
                && (p + 2 >= n || !is_ident(rd(dbuf + p + 2)));
            if is2e {
                p += 2;
            } else {
                if nseg > 0 { wr(ip + w, b'/'); w += 1; }
                while p < n && is_ident(rd(dbuf + p)) { wr(ip + w, rd(dbuf + p)); w += 1; p += 1; }
                nseg += 1;
            }
        } else { p += 1; }
    }
    if nseg > 0 {
        let suf = b".nim"; let mut kk = 0i64; while kk < 4 { wr(ip + w, suf[kk as usize]); w += 1; kk += 1; }
        (p << 24) | w
    } else {
        p << 24
    }
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

        let seed = b"/main.nim";
        let sp = base + SCR;
        let mut k = 0i64; while k < 9 { wr(sp + k, seed[k as usize]); k += 1; }
        let mut nwl = add_path(base, 0, sp, 9);

        let mut i = 0i64;
        let mut crawled = 0i64;
        while i < nwl {
            let path = base + WL + i * 96;
            let plen = *((base + WLL + i * 4) as *const u32) as i64;

            // stem = module_suffix(path); out = "/nimcache/" + stem + ".p.nif"
            let stem = base + SCR + 1024;
            let slen = module_suffix(path, plen, stem);
            let outp = base + SCR + 1152;
            let ncp = b"/nimcache/"; let mut w = 0i64;
            let mut kk = 0i64; while kk < 10 { wr(outp + w, ncp[kk as usize]); w += 1; kk += 1; }
            let mut kk = 0i64; while kk < slen { wr(outp + w, rd(stem + kk)); w += 1; kk += 1; }
            let osuf = b".p.nif"; let mut kk = 0i64; while kk < 6 { wr(outp + w, osuf[kk as usize]); w += 1; kk += 1; }
            let outl = w;

            let carve = if i % 2 == 0 { carve0 } else { carve1 };
            let st = spawn_nifler(inst, nifler, base, carve, path, plen, outp, outl);
            if st != 0 && st != 5 { return -100 + st; }

            // deps key (relative) = "nimcache/" + stem + ".p.deps.nif"
            let dk = base + SCR + 1408;
            let ncr = b"nimcache/"; let mut w2 = 0i64;
            let mut kk = 0i64; while kk < 9 { wr(dk + w2, ncr[kk as usize]); w2 += 1; kk += 1; }
            let mut kk = 0i64; while kk < slen { wr(dk + w2, rd(stem + kk)); w2 += 1; kk += 1; }
            let dsuf = b".p.deps.nif"; let mut kk = 0i64; while kk < 11 { wr(dk + w2, dsuf[kk as usize]); w2 += 1; kk += 1; }
            let mut lastslash = 0i64; let mut ls = 0i64;
            while ls < plen { if rd(path + ls) == b'/' { lastslash = ls; } ls += 1; }
            let n = fs_read(fs, dk, w2, base + DBUF, 65536);
            if n > 0 { nwl = scan_imports(base, base + DBUF, n, nwl, path, lastslash); }
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

// ---- nimc::module_suffix oracle ----
fn uhash(s: &str) -> u32 {
    let mut h: u32 = 0;
    for c in s.bytes() {
        h = h.wrapping_add(c as u32);
        h = h.wrapping_add(h << 10);
        h ^= h >> 6;
    }
    h = h.wrapping_add(h << 3);
    h ^= h >> 11;
    h = h.wrapping_add(h << 15);
    h
}
fn base36(mut id: u32) -> String {
    const B36: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut r = String::new();
    while id > 0 {
        r.push(B36[(id % 36) as usize] as char);
        id /= 36;
    }
    r
}
fn relative_path(path: &str, base: &str) -> String {
    let p: Vec<&str> = path
        .trim_matches('/')
        .split('/')
        .filter(|s| !s.is_empty())
        .collect();
    let b: Vec<&str> = base
        .trim_matches('/')
        .split('/')
        .filter(|s| !s.is_empty())
        .collect();
    let mut i = 0;
    while i < p.len() && i < b.len() && p[i] == b[i] {
        i += 1;
    }
    let mut out: Vec<&str> = vec![".."; b.len() - i];
    out.extend_from_slice(&p[i..]);
    out.join("/")
}
fn module_suffix(file: &str) -> String {
    let mut rel = relative_path(file, "/");
    let c = relative_path(file, "/lib");
    if c.len() < rel.len() {
        rel = c;
    }
    let name = rel.rsplit('/').next().unwrap_or(&rel);
    let name = name.strip_suffix(".nim").unwrap_or(name);
    let mut stem: String = name.chars().take(3).collect();
    stem.push_str(&base36(uhash(&rel)));
    stem
}

/// Translate the crawl guest, grant it the caps over a memfs seeded with `seed`+`dirs`, run it, and
/// return `(crawled count, shared handle)` — or `None` if rustc/gzip are unavailable (skip).
fn run_crawl(
    nifler_bytes: &[u8],
    seed: Vec<(String, Vec<u8>)>,
    dirs: Vec<String>,
) -> Option<(i64, temen_run::fs::MemFsHandle)> {
    // One scratch dir per CALL: the three tests in this binary run on parallel threads of one
    // process, and two of them seed the same number of files, so keying on `(pid, seed.len())` made
    // them share a dir — one test's `rustc --emit=llvm-ir` then raced the other's (`could not copy
    // …cgu.0.rcgu.ll to g.ll`) and the loser parsed a half-written `g.ll` (flaky-ci #1294).
    static CALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "rust_driver_crawl_cache_{}_{}",
        std::process::id(),
        CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).ok()?;
    let src = dir.join("g.rs");
    let ll = dir.join("g.ll");
    std::fs::write(&src, CRAWL_SRC).ok()?;
    if !rustc_emit_ll(&src, &ll) {
        return None;
    }
    let t = temen_llvm::translate_ll_path(&ll).expect("translate crawl guest");
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

    let (factory, handle) = temen_run::fs::mem_fs_shared_factory(seed, dirs);
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
    Some((crawled, handle))
}

#[test]
fn rust_driver_guest_crawls_with_stem_named_cache_outputs() {
    let Some(nifler_bytes) = inflate(NIFLER_CE_GZ) else {
        eprintln!("note: skipping (gzip unavailable)");
        return;
    };
    let seed = vec![
        ("main.nim".into(), b"import std/foo\n".to_vec()),
        ("lib/std/foo.nim".into(), b"import std/bar\n".to_vec()),
        ("lib/std/bar.nim".into(), b"proc b(): int = 1\n".to_vec()),
    ];
    let dirs = vec!["lib".into(), "lib/std".into(), "nimcache".into()];
    let Some((crawled, handle)) = run_crawl(&nifler_bytes, seed, dirs) else {
        eprintln!("note: skipping (rustc unavailable)");
        return;
    };
    assert_eq!(crawled, 3, "crawled main -> std/foo -> std/bar");

    let (files, _dirs) = handle.seed();
    let has = |k: &str| files.iter().any(|(n, _)| n == k);
    // Each module's `.p.nif` is written under nimcache/ at its nimony stem — nimc's cache layout.
    for path in ["/main.nim", "/lib/std/foo.nim", "/lib/std/bar.nim"] {
        let stem = module_suffix(path);
        let key = format!("nimcache/{stem}.p.nif");
        assert!(
            has(&key),
            "the crawl wrote `{key}` (stem for {path}) — nimc-cache-correct output naming"
        );
    }
}

#[test]
fn rust_driver_guest_crawl_handles_fromimport_and_skips_when_guarded() {
    let Some(nifler_bytes) = inflate(NIFLER_CE_GZ) else {
        eprintln!("note: skipping (gzip unavailable)");
        return;
    };
    // main uses `from std/foo import x` (fromimport) AND a `when`-guarded `import winlean`. The crawl
    // must follow std/foo but must NOT chase the platform-guarded winlean (which isn't in the memfs) —
    // exactly nimc::parse_imports' behavior (it scans `fromimport` too, and skips `(when …)` blocks).
    let seed = vec![
        (
            "main.nim".into(),
            b"from std/foo import x\nwhen defined(windows):\n  import winlean\n".to_vec(),
        ),
        ("lib/std/foo.nim".into(), b"proc f(): int = 1\n".to_vec()),
    ];
    let dirs = vec!["lib".into(), "lib/std".into(), "nimcache".into()];
    let Some((crawled, handle)) = run_crawl(&nifler_bytes, seed, dirs) else {
        eprintln!("note: skipping (rustc unavailable)");
        return;
    };
    assert_eq!(
        crawled, 2,
        "crawled main + std/foo (via fromimport); the when-guarded winlean was skipped, not chased"
    );

    let (files, _dirs) = handle.seed();
    let has = |k: &str| files.iter().any(|(n, _)| n == k);
    for path in ["/main.nim", "/lib/std/foo.nim"] {
        let stem = module_suffix(path);
        assert!(has(&format!("nimcache/{stem}.p.nif")), "crawled {path}");
    }
    // winlean was never sought — no nifler run, no output for it.
    assert!(
        !has(&format!(
            "nimcache/{}.p.nif",
            module_suffix("/lib/winlean.nim")
        )),
        "the when-guarded import must not have been crawled"
    );
}

#[test]
fn rust_driver_guest_crawl_resolves_prefix_relative_imports() {
    let Some(nifler_bytes) = inflate(NIFLER_CE_GZ) else {
        eprintln!("note: skipping (gzip unavailable)");
        return;
    };
    // A relative import (`import ./sib`) from a module in a subdir: nifler emits it as
    // `(import (prefix \2E/ sib))`, which nimc::parse_imports resolves against the importer's own dir —
    // here /lib/std -> /lib/std/sib.nim, NOT /lib/sib.nim. main reaches it via an infix `std/sub`, so
    // the crawl exercises both forms and a non-empty importer dir.
    let seed = vec![
        ("main.nim".into(), b"import std/sub\n".to_vec()),
        ("lib/std/sub.nim".into(), b"import ./sib\n".to_vec()),
        ("lib/std/sib.nim".into(), b"proc s(): int = 1\n".to_vec()),
    ];
    let dirs = vec!["lib".into(), "lib/std".into(), "nimcache".into()];
    let Some((crawled, handle)) = run_crawl(&nifler_bytes, seed, dirs) else {
        eprintln!("note: skipping (rustc unavailable)");
        return;
    };
    assert_eq!(
        crawled, 3,
        "crawled main -> std/sub -> ./sib (the relative import resolved against /lib/std)"
    );

    let (files, _dirs) = handle.seed();
    let has = |k: &str| files.iter().any(|(n, _)| n == k);
    for path in ["/main.nim", "/lib/std/sub.nim", "/lib/std/sib.nim"] {
        let stem = module_suffix(path);
        assert!(
            has(&format!("nimcache/{stem}.p.nif")),
            "crawled {path} (prefix import resolved to its importer's dir)"
        );
    }
}
