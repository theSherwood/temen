//! **#1025 slice 3c — the driver-guest port, step 8: nimc's dependency order (toposort) in the guest.**
//! Steps 3–7 ported `nimc`'s phase-1 import crawl into the sandbox (discovery, stem-named cache outputs,
//! every `parse_imports` form). This ports the next policy piece: after the crawl, `nimc::compile_nim`
//! orders the module closure for the sema/lowering phases — a DFS post-order (a module's deps before it),
//! then a stable **System-first** partition (`nimc.rs::toposort`). The guest now records the dependency
//! graph as it crawls — per module its `(stem, role, deps)` — and computes that exact order, emitting the
//! ordered stems to `order.txt` in the shared memfs.
//!
//! Seeded like `nimc` (`/lib/std/system.nim` = System, `/main.nim` = Main, everything discovered =
//! Import), the host asserts the emitted order is byte-identical to a Rust port of `nimc::toposort` over
//! the same graph. Dependency ordering — the last pure-policy piece before the phases themselves — now
//! runs entirely inside the sandbox. Gated Linux + rustc + gzip.

#![cfg(target_os = "linux")]

use std::collections::BTreeMap;
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;

use temen_interp::{
    run_capture_reserved_with_host, ForkedProc, Host, HostProc, HostProcFork, StreamRole, Value,
};

const NIFLER_CE_GZ: &[u8] = include_bytes!("../../temen-run/demos/nifler_temen/nifler_ce.temen.gz");

const TOPO_SRC: &str = r##"#![no_std]
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

// open(name, O_WRITE|O_CREATE|O_TRUNC=26) -> fd; write; close.
unsafe fn fs_write(fs: i32, name: i64, nlen: i64, data: i64, dlen: i64) -> i64 {
    let fd = __vm_host_call(fs, 0, name, nlen, 26, 0);
    if fd < 0 { return fd; }
    let n = __vm_host_call(fs, 2, fd, data, dlen, 0);
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

const WLL: i64 = 256;   // worklist path lengths (u32 * 16)
const WL: i64 = 512;    // worklist paths (96 bytes * 16)
const SCR: i64 = 2560;  // scratch (out/deps-key/name construction)
const DBUF: i64 = 4096; // deps read buffer (64 KiB, below the carves)

// ---- module graph, recorded during the crawl (all offsets from `base`, in the low window) ----
const G: i64 = 71680;
const STEMS: i64 = G;          // per-module stem bytes (32 * 16)
const STEMLEN: i64 = G + 512;  // per-module stem length (u32 * 16)
const ROLE: i64 = G + 576;     // per-module role u8 (0 System, 1 Main, 2 Import) * 16
const DEPS: i64 = G + 592;     // per-module dep indices (u8 * 16 * 16)
const DEPN: i64 = G + 848;     // per-module dep count (u8 * 16)
const SORTED: i64 = G + 864;   // module indices sorted by stem (u8 * 16)
const SEEN: i64 = G + 880;     // DFS visit state u8 (0/1/2) * 16
const STK: i64 = G + 896;      // DFS node stack (u8 * 16)
const STKC: i64 = G + 912;     // DFS per-frame dep cursor (u8 * 16)
const POST: i64 = G + 928;     // DFS post-order output (u8 * 16)
const ORDER: i64 = G + 944;    // final System-first order (u8 * 16)
const NWL: i64 = G + 960;      // module count (u32 cell)
const OBUF: i64 = G + 968;     // order.txt text buffer (512)

unsafe fn nwl_get(base: i64) -> i64 { *((base + NWL) as *const u32) as i64 }
unsafe fn nwl_set(base: i64, v: i64) { *((base + NWL) as *mut u32) = v as u32; }

// Find the module for `path` (by worklist path), or add it (recording stem + role, enqueuing it for the
// crawl). Returns its index, or -1 if the 16-module worklist is full. First role wins (dedup by path),
// matching nimc's `if mods.contains_key(&stem) { continue }`.
unsafe fn mod_index(base: i64, path: i64, plen: i64, role: u8) -> i64 {
    let nwl = nwl_get(base);
    let mut i = 0i64;
    while i < nwl {
        let slot = base + WL + i * 96;
        let slen = *((base + WLL + i * 4) as *const u32) as i64;
        if eq(slot, slen, path, plen) { return i; }
        i += 1;
    }
    if nwl >= 16 { return -1; }
    let slot = base + WL + nwl * 96;
    let mut k = 0i64; while k < plen { wr(slot + k, rd(path + k)); k += 1; }
    *((base + WLL + nwl * 4) as *mut u32) = plen as u32;
    let sl = module_suffix(path, plen, base + STEMS + nwl * 32);
    *((base + STEMLEN + nwl * 4) as *mut u32) = sl as u32;
    wr(base + ROLE + nwl, role);
    wr(base + DEPN + nwl, 0);
    nwl_set(base, nwl + 1);
    nwl
}

// Record `mi -> j` (dep edge), in parse order, matching nimc's `deps.push(module_suffix(&imp))`.
unsafe fn add_dep(base: i64, mi: i64, j: i64) {
    let dn = rd(base + DEPN + mi) as i64;
    if dn < 16 { wr(base + DEPS + mi * 16 + dn, j as u8); wr(base + DEPN + mi, (dn + 1) as u8); }
}

// Resolve an `(infix …)` / `(prefix …)` block into `ip`; returns `(end << 24) | path_len` (path_len 0
// if no segments). Inline copies (same cursor the loop tests) — the on-ramp translator miscompiles a
// separate-counter inner write loop through a parameter pointer (#1216).
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

// Scan every `(import …)` / `(fromimport …)` in the deps buffer, recording each resolved import as a dep
// edge of module `mi` (and enqueuing it for the crawl via `mod_index`) — the graph-recording twin of the
// step-5..7 `scan_imports`.
unsafe fn scan_imports_record(base: i64, dbuf: i64, n: i64, mi: i64, dir: i64, dirlen: i64) {
    let imp = b"(import";
    let frm = b"(fromimport";
    let inf = b"(infix";
    let pfx = b"(prefix";
    let mut i = 0i64;
    while i < n {
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
        if plen > 0 {
            let j = mod_index(base, ip, plen, 2);
            if j >= 0 { add_dep(base, mi, j); }
        }
        i = ret >> 24;
    }
}

// Lexicographic byte compare of module ia's vs ib's stem: -1 / 0 / 1.
unsafe fn cmp_stem(base: i64, ia: i64, ib: i64) -> i32 {
    let la = *((base + STEMLEN + ia * 4) as *const u32) as i64;
    let lb = *((base + STEMLEN + ib * 4) as *const u32) as i64;
    let sa = base + STEMS + ia * 32;
    let sb = base + STEMS + ib * 32;
    let m = if la < lb { la } else { lb };
    let mut i = 0i64;
    while i < m {
        let ca = rd(sa + i); let cb = rd(sb + i);
        if ca < cb { return -1; }
        if ca > cb { return 1; }
        i += 1;
    }
    if la < lb { -1 } else if la > lb { 1 } else { 0 }
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

        // Seed like nimc: system.nim (System) then main (Main); the crawl discovers the rest (Import).
        nwl_set(base, 0);
        let sysn = b"/lib/std/system.nim";
        let sp = base + SCR; let mut k = 0i64; while k < 19 { wr(sp + k, sysn[k as usize]); k += 1; }
        mod_index(base, sp, 19, 0);
        let mann = b"/main.nim";
        let mut k = 0i64; while k < 9 { wr(sp + k, mann[k as usize]); k += 1; }
        mod_index(base, sp, 9, 1);

        // ---- crawl: run nifler per module, record its deps ----
        let mut i = 0i64;
        while i < nwl_get(base) {
            let path = base + WL + i * 96;
            let plen = *((base + WLL + i * 4) as *const u32) as i64;
            let stem = base + STEMS + i * 32;
            let slen = *((base + STEMLEN + i * 4) as *const u32) as i64;

            // out = "/nimcache/" + stem + ".p.nif"
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
            if n > 0 { scan_imports_record(base, base + DBUF, n, i, path, lastslash); }
            i += 1;
        }

        let count = nwl_get(base);

        // ---- SORTED = module indices sorted by stem (nimc iterates BTreeMap::keys()) ----
        let mut k = 0i64; while k < count { wr(base + SORTED + k, k as u8); k += 1; }
        let mut a = 1i64;
        while a < count {
            let key = rd(base + SORTED + a);
            let mut b = a - 1;
            while b >= 0 && cmp_stem(base, rd(base + SORTED + b) as i64, key as i64) > 0 {
                wr(base + SORTED + b + 1, rd(base + SORTED + b));
                b -= 1;
            }
            wr(base + SORTED + b + 1, key);
            a += 1;
        }

        // ---- DFS post-order (deps before node) over the stem-sorted modules, shared `seen` ----
        let mut k = 0i64; while k < count { wr(base + SEEN + k, 0); k += 1; }
        let mut postn = 0i64;
        let mut kk = 0i64;
        while kk < count {
            let s = rd(base + SORTED + kk) as i64;
            if rd(base + SEEN + s) == 0 {
                let mut sp2 = 0i64;
                wr(base + STK + sp2, s as u8); wr(base + STKC + sp2, 0); wr(base + SEEN + s, 1); sp2 += 1;
                while sp2 > 0 {
                    let node = rd(base + STK + sp2 - 1) as i64;
                    let c = rd(base + STKC + sp2 - 1) as i64;
                    let dn = rd(base + DEPN + node) as i64;
                    if c < dn {
                        wr(base + STKC + sp2 - 1, (c + 1) as u8);
                        let d = rd(base + DEPS + node * 16 + c) as i64;
                        if rd(base + SEEN + d) == 0 {
                            wr(base + SEEN + d, 1); wr(base + STK + sp2, d as u8); wr(base + STKC + sp2, 0); sp2 += 1;
                        }
                    } else {
                        wr(base + SEEN + node, 2); wr(base + POST + postn, node as u8); postn += 1; sp2 -= 1;
                    }
                }
            }
            kk += 1;
        }

        // ---- System-first stable partition (nimc: order.sort_by_key(|s| role != System)) ----
        let mut on = 0i64;
        let mut k = 0i64;
        while k < postn { let idx = rd(base + POST + k) as i64; if rd(base + ROLE + idx) == 0 { wr(base + ORDER + on, idx as u8); on += 1; } k += 1; }
        let mut k = 0i64;
        while k < postn { let idx = rd(base + POST + k) as i64; if rd(base + ROLE + idx) != 0 { wr(base + ORDER + on, idx as u8); on += 1; } k += 1; }

        // ---- emit the ordered stems, one per line, to order.txt ----
        let obuf = base + OBUF; let mut w = 0i64;
        let mut k = 0i64;
        while k < on {
            let idx = rd(base + ORDER + k) as i64;
            let sl = *((base + STEMLEN + idx * 4) as *const u32) as i64;
            let stp = base + STEMS + idx * 32;
            let mut j = 0i64; while j < sl { wr(obuf + w, rd(stp + j)); w += 1; j += 1; }
            wr(obuf + w, 10); w += 1;
            k += 1;
        }
        let name = base + SCR + 2048;
        let onm = b"order.txt"; let mut j = 0i64; while j < 9 { wr(name + j, onm[j as usize]); j += 1; }
        if fs_write(fs, name, 9, obuf, w) < 0 { return -2; }
        on
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

// ---- nimc::module_suffix + nimc::toposort oracle ----
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

/// role: 0 System, 1 Main, 2 Import.
struct OMod {
    role: u8,
    deps: Vec<String>,
}

/// A faithful port of `browser/src/nimc.rs::toposort`: DFS post-order over the stem-sorted modules
/// (a module's deps before it), then a stable System-first partition.
fn toposort_oracle(mods: &BTreeMap<String, OMod>) -> Vec<String> {
    fn visit(
        s: &str,
        mods: &BTreeMap<String, OMod>,
        seen: &mut BTreeMap<String, u8>,
        order: &mut Vec<String>,
    ) {
        if seen.get(s).copied().unwrap_or(0) != 0 {
            return;
        }
        seen.insert(s.to_string(), 1);
        if let Some(m) = mods.get(s) {
            for d in &m.deps {
                visit(d, mods, seen, order);
            }
        }
        seen.insert(s.to_string(), 2);
        order.push(s.to_string());
    }
    let mut seen: BTreeMap<String, u8> = BTreeMap::new();
    let mut order = vec![];
    for s in mods.keys() {
        visit(s, mods, &mut seen, &mut order);
    }
    order.sort_by_key(|s| mods.get(s).map(|m| m.role != 0).unwrap_or(true));
    order
}

/// Build the `mods` graph a `(path, role, [resolved import paths])` spec would produce (stems via the
/// oracle `module_suffix`), then the expected `order.txt` bytes (each stem + '\n').
fn expected_order(spec: &[(&str, u8, &[&str])]) -> String {
    let mut mods: BTreeMap<String, OMod> = BTreeMap::new();
    for (path, role, imports) in spec {
        let stem = module_suffix(path);
        let deps: Vec<String> = imports.iter().map(|p| module_suffix(p)).collect();
        mods.insert(stem, OMod { role: *role, deps });
    }
    toposort_oracle(&mods)
        .iter()
        .map(|s| format!("{s}\n"))
        .collect()
}

/// Translate the toposort guest, grant it the caps over a memfs seeded with `seed`, run it, and return
/// `(module count, order.txt bytes)` — or `None` if rustc/gzip are unavailable (skip).
fn run_topo(nifler_bytes: &[u8], seed: Vec<(String, Vec<u8>)>) -> Option<(i64, Vec<u8>)> {
    let dir = std::env::temp_dir().join(format!(
        "rust_driver_toposort_{}_{}",
        std::process::id(),
        seed.len()
    ));
    std::fs::create_dir_all(&dir).ok()?;
    let src = dir.join("g.rs");
    let ll = dir.join("g.ll");
    std::fs::write(&src, TOPO_SRC).ok()?;
    if !rustc_emit_ll(&src, &ll) {
        return None;
    }
    let t = temen_llvm::translate_ll_path(&ll).expect("translate toposort guest");
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

    let dirs = vec!["lib".into(), "lib/std".into(), "nimcache".into()];
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
    let count = match r.expect("driver run").as_slice() {
        [Value::I64(x)] => *x,
        [Value::I32(x)] => *x as i64,
        other => panic!("driver result: {other:?}"),
    };
    let (files, _dirs) = handle.seed();
    let order = files
        .iter()
        .find(|(k, _)| k == "order.txt")
        .map(|(_, v)| v.clone())
        .expect("guest wrote no order.txt");
    Some((count, order))
}

#[test]
fn rust_driver_guest_emits_nimc_toposort_over_a_diamond() {
    let Some(nifler_bytes) = inflate(NIFLER_CE_GZ) else {
        eprintln!("note: skipping (gzip unavailable)");
        return;
    };
    // A diamond under a real system module: system (no deps), main -> {a, b}, a -> c, b -> c, c (leaf).
    // nimc's order is a DFS post-order (deps before node) over the stem-sorted modules, System first.
    let seed = vec![
        (
            "lib/std/system.nim".into(),
            b"proc sysp(): int = 0\n".to_vec(),
        ),
        ("main.nim".into(), b"import std/a\nimport std/b\n".to_vec()),
        ("lib/std/a.nim".into(), b"import std/c\n".to_vec()),
        ("lib/std/b.nim".into(), b"import std/c\n".to_vec()),
        ("lib/std/c.nim".into(), b"proc c(): int = 1\n".to_vec()),
    ];
    let Some((count, order)) = run_topo(&nifler_bytes, seed) else {
        eprintln!("note: skipping (rustc unavailable)");
        return;
    };
    assert_eq!(count, 5, "system + main + a + b + c");

    let expected = expected_order(&[
        ("/lib/std/system.nim", 0, &[]),
        ("/main.nim", 1, &["/lib/std/a.nim", "/lib/std/b.nim"]),
        ("/lib/std/a.nim", 2, &["/lib/std/c.nim"]),
        ("/lib/std/b.nim", 2, &["/lib/std/c.nim"]),
        ("/lib/std/c.nim", 2, &[]),
    ]);
    assert_eq!(
        String::from_utf8_lossy(&order),
        expected,
        "the guest's dependency order must match nimc::toposort (DFS post-order, System first)"
    );
    // Sanity on the shape the order must have regardless of stem-hash tie-breaks:
    let lines: Vec<&str> = expected.lines().collect();
    assert_eq!(
        lines[0],
        module_suffix("/lib/std/system.nim"),
        "System module is ordered first"
    );
    assert_eq!(
        *lines.last().unwrap(),
        module_suffix("/main.nim"),
        "Main is ordered last (nothing depends on it)"
    );
    let ci = lines
        .iter()
        .position(|s| *s == module_suffix("/lib/std/c.nim"))
        .unwrap();
    let ai = lines
        .iter()
        .position(|s| *s == module_suffix("/lib/std/a.nim"))
        .unwrap();
    let bi = lines
        .iter()
        .position(|s| *s == module_suffix("/lib/std/b.nim"))
        .unwrap();
    assert!(
        ci < ai && ci < bi,
        "the shared leaf c comes before both a and b"
    );
}

#[test]
fn rust_driver_guest_emits_nimc_toposort_over_a_chain() {
    let Some(nifler_bytes) = inflate(NIFLER_CE_GZ) else {
        eprintln!("note: skipping (gzip unavailable)");
        return;
    };
    // A straight chain: main -> a -> b (plus system). Post-order gives system, b, a, main.
    let seed = vec![
        (
            "lib/std/system.nim".into(),
            b"proc sysp(): int = 0\n".to_vec(),
        ),
        ("main.nim".into(), b"import std/a\n".to_vec()),
        ("lib/std/a.nim".into(), b"import std/b\n".to_vec()),
        ("lib/std/b.nim".into(), b"proc b(): int = 1\n".to_vec()),
    ];
    let Some((count, order)) = run_topo(&nifler_bytes, seed) else {
        eprintln!("note: skipping (rustc unavailable)");
        return;
    };
    assert_eq!(count, 4, "system + main + a + b");
    let expected = expected_order(&[
        ("/lib/std/system.nim", 0, &[]),
        ("/main.nim", 1, &["/lib/std/a.nim"]),
        ("/lib/std/a.nim", 2, &["/lib/std/b.nim"]),
        ("/lib/std/b.nim", 2, &[]),
    ]);
    assert_eq!(
        String::from_utf8_lossy(&order),
        expected,
        "chain toposort matches nimc"
    );
    // b before a before main (a strict dependency chain), system first.
    let lines: Vec<&str> = expected.lines().collect();
    assert_eq!(lines[0], module_suffix("/lib/std/system.nim"));
    let bi = lines
        .iter()
        .position(|s| *s == module_suffix("/lib/std/b.nim"))
        .unwrap();
    let ai = lines
        .iter()
        .position(|s| *s == module_suffix("/lib/std/a.nim"))
        .unwrap();
    let mi = lines
        .iter()
        .position(|s| *s == module_suffix("/main.nim"))
        .unwrap();
    assert!(bi < ai && ai < mi, "b -> a -> main dependency order");
}
