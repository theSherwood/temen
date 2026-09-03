//! **#1025 slice 3c — the driver-guest port, step 4: `module_suffix` stem-hashing in the guest.** The
//! crawl (step 3) discovers modules; to feed the downstream phases (nimsem reads `{stem}.p.nif`) the
//! guest must name each module's cache outputs by nimony's module stem — `name[0..3]` + base36 of the
//! `uhash` of the module's search-relative path (`gear2/modnames.moduleSuffix`, mirrored in
//! `nimc::module_suffix`). This ports that hash into the Rust-on-Temen guest and pins it **byte-for-byte
//! against the host oracle**: the guest computes the stem for three paths and writes each into the
//! shared memfs; the host recomputes them with `nimc`'s exact algorithm (replicated below) and asserts
//! equality. Once the crawl names its `.p.nif`/`.p.deps.nif` by these stems, its output is the cache
//! layout `nimc::compile_nim`'s later phases already consume. Gated to Linux + rustc + gzip.

#![cfg(target_os = "linux")]

use std::process::Command;
use std::sync::Arc;

use temen_interp::{run_capture_reserved_with_host, Host, StreamRole, Value};

const STEM_SRC: &str = r##"#![no_std]
#![allow(internal_features)]
#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! { loop {} }
#[no_mangle]
pub extern "C" fn rust_eh_personality() {}

#[repr(C, align(65536))]
struct Pool([u8; 131072]);
static mut POOL: Pool = Pool([0; 131072]);

extern "C" {
    fn __vm_cap_resolve(name: *const u8, len: i64) -> i32;
    fn __vm_host_call(handle: i32, op: i32, a: i64, b: i64, c: i64, d: i64) -> i64;
}

unsafe fn wr(p: i64, b: u8) { (p as *mut u8).write(b); }
unsafe fn rd(p: i64) -> u8 { *(p as *const u8) }

// nimony's module-stem hash (gear2/modnames.nim + lib/tinyhashes.nim), byte-for-byte with nimc::uhash.
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

// base36 LSB-first (nimc::base36: `while id>0 { push B36[id%36]; id/=36 }`). Writes to `out`, returns len.
unsafe fn base36(mut id: u32, out: i64) -> i64 {
    let b36 = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut w = 0i64;
    while id > 0 {
        wr(out + w, b36[(id % 36) as usize]);
        w += 1;
        id /= 36;
    }
    w
}

// nimc::module_suffix: rel = path minus a leading "/" and a leading "lib/" (the shortest of the path
// relative to "/" and to "/lib"); stem = rel's last path segment minus ".nim", first 3 chars, +
// base36(uhash(rel)). Writes the stem to `out`, returns its length.
unsafe fn module_suffix(path: i64, plen: i64, out: i64) -> i64 {
    let mut r = path;
    let mut rlen = plen;
    if rlen > 0 && rd(r) == b'/' { r += 1; rlen -= 1; }
    if rlen >= 4 && rd(r) == b'l' && rd(r + 1) == b'i' && rd(r + 2) == b'b' && rd(r + 3) == b'/' {
        r += 4;
        rlen -= 4;
    }
    // last segment start (after the final '/')
    let mut nstart = 0i64;
    let mut k = 0i64;
    while k < rlen {
        if rd(r + k) == b'/' { nstart = k + 1; }
        k += 1;
    }
    let mut nlen = rlen - nstart;
    if nlen >= 4
        && rd(r + nstart + nlen - 4) == b'.'
        && rd(r + nstart + nlen - 3) == b'n'
        && rd(r + nstart + nlen - 2) == b'i'
        && rd(r + nstart + nlen - 1) == b'm'
    {
        nlen -= 4;
    }
    let take = if nlen < 3 { nlen } else { 3 };
    let mut w = 0i64;
    let mut i = 0i64;
    while i < take {
        wr(out + w, rd(r + nstart + i));
        w += 1;
        i += 1;
    }
    let h = uhash(r, rlen);
    w += base36(h, out + w);
    w
}

// open(name, O_WRITE|O_CREATE|O_TRUNC=26) → fd; write(fd, data, len); close.
unsafe fn write_file(fs: i32, name: i64, nlen: i64, data: i64, dlen: i64) -> i64 {
    let fd = __vm_host_call(fs, 0, name, nlen, 26, 0);
    if fd < 0 { return fd; }
    let n = __vm_host_call(fs, 2, fd, data, dlen, 0);
    let _ = __vm_host_call(fs, 4, fd, 0, 0, 0);
    n
}

#[no_mangle]
pub extern "C" fn run() -> i64 {
    unsafe {
        let fs = __vm_cap_resolve(b"fs".as_ptr(), 2);
        if fs < 0 { return -1; }
        let base = core::ptr::addr_of_mut!(POOL) as i64;
        // Three input paths to stem, written into the window, each stemmed and its stem written to fs
        // under "s0"/"s1"/"s2" for the host to diff against the nimc::module_suffix oracle.
        let paths: [&[u8]; 3] = [b"/lib/std/system.nim", b"/lib/std/syncio.nim", b"/main.nim"];
        let mut idx = 0i64;
        while idx < 3 {
            let s = paths[idx as usize];
            let pbuf = base + 1024;
            let mut j = 0i64;
            while j < s.len() as i64 { wr(pbuf + j, s[j as usize]); j += 1; }
            let stembuf = base + 2048;
            let slen = module_suffix(pbuf, s.len() as i64, stembuf);
            let namebuf = base + 4096;
            wr(namebuf, b's');
            wr(namebuf + 1, b'0' + idx as u8);
            let w = write_file(fs, namebuf, 2, stembuf, slen);
            if w < 0 { return -10 + idx; }
            idx += 1;
        }
        0
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

// ---- nimc::module_suffix, replicated verbatim as the oracle ----
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

#[test]
fn rust_driver_guest_computes_module_stems_matching_nimc() {
    let dir = std::env::temp_dir().join(format!("rust_driver_stems_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("g.rs");
    let ll = dir.join("g.ll");
    std::fs::write(&src, STEM_SRC).unwrap();
    if !rustc_emit_ll(&src, &ll) {
        eprintln!("note: skipping (rustc unavailable)");
        return;
    }
    let t = temen_llvm::translate_ll_path(&ll).expect("translate stem guest");
    temen_verify::verify_module(&t.module).expect("driver verifies");
    let entry = t
        .exports
        .iter()
        .find(|(n, _)| n == "run")
        .expect("exports run")
        .1;
    let sp = t.entry_sp as i64;

    let (factory, handle) = temen_run::fs::mem_fs_shared_factory(vec![], vec![]);
    let factory = Arc::new(factory);
    let mut host = Host::new();
    let fs_h = host.grant_host_proc((*factory)());
    host.register_cap_name("fs", fs_h);
    let _ = host.grant_stream(StreamRole::Out);

    let mut fuel = 1_000_000_000u64;
    let (r, _) = run_capture_reserved_with_host(
        &t.module,
        entry,
        &[Value::I64(sp)],
        &mut fuel,
        &[],
        0,
        &mut host,
    );
    assert!(
        matches!(
            r.expect("run").as_slice(),
            [Value::I64(0)] | [Value::I32(0)]
        ),
        "the stem guest wrote all three stems"
    );

    let (files, _dirs) = handle.seed();
    let read = |k: &str| {
        files
            .iter()
            .find(|(n, _)| n == k)
            .map(|(_, v)| String::from_utf8_lossy(v).into_owned())
    };
    let paths = ["/lib/std/system.nim", "/lib/std/syncio.nim", "/main.nim"];
    for (i, p) in paths.iter().enumerate() {
        let got = read(&format!("s{i}")).unwrap_or_else(|| panic!("guest wrote no s{i}"));
        assert_eq!(
            got,
            module_suffix(p),
            "guest stem for {p} matches nimc::module_suffix (name[0..3] + base36(uhash(rel)))"
        );
    }
}
