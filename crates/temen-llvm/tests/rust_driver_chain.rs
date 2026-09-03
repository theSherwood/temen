//! **#1025 slice 3c — the driver-guest port, step 11: the nimsem→hexer chain, driven by the guest.**
//! Steps 9 and 10 ran nimsem and hexer each as a single op-13 §14 child. This is the payoff: one
//! Rust-on-Temen driver guest **orchestrates both phases** over one shared memfs — it `instantiate_
//! module`s (op 13) nimsem `{fs, stdout, exit, exec}` to semcheck the system module (`.s.nif`, its
//! nifler grandchildren parsing the stdlib via the re-granted `exec`), then `instantiate_module`s hexer
//! `{fs, stdout, exit}` to lower the `.s.nif` nimsem just wrote into the same store (`.x.nif`). The guest
//! is the compiler driver; the phases are the sandboxed children. The host asserts the emitted Leng
//! `sysvq0asl.x.nif` is **byte-identical** to the committed expected.
//!
//! Reuses **every** committed fixture from steps 9–10 (`nimsem_ce`/`hexer_ce`/`syslib`/`sysvq0asl.p.nif`
//! /`sysvq0asl.x.nif`) plus the committed top-level `browser/web/assets/nifler.temen.gz` for the exec —
//! **no new asset**. The full byte-exact-vs-*native* chain (and the JIT chain) stays in the
//! toolchain-gated `build_frontend.sh` (issue #1221); `examples/nim_chain_op13.rs` is the host-conductor
//! counterpart this guest-drives.
//!
//! Heavy: two 256 MiB-carve phases back to back (nimsem's semcheck dominates), so the guest runs in a
//! ~1 GiB window and the test takes ~a minute. Gated Linux + rustc + gzip + tar.

#![cfg(target_os = "linux")]

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Arc;

use temen_interp::{run_with_host, ForkedProc, Host, HostProc, HostProcFork, StreamRole, Value};
use temen_run::exec::{domain_exec_with_fs, DomainProgram};
use temen_run::{instantiate, HostCap, Limits};

const NIMSEM_CE_GZ: &[u8] =
    include_bytes!("../../temen-run/demos/nim_frontend/fixtures/nimsem_ce.temen.gz");
const HEXER_CE_GZ: &[u8] =
    include_bytes!("../../temen-run/demos/nim_frontend/fixtures/hexer_ce.temen.gz");
const SYSLIB_TAR_GZ: &[u8] =
    include_bytes!("../../temen-run/demos/nim_frontend/fixtures/syslib.tar.gz");
const SYS_PNIF: &[u8] =
    include_bytes!("../../temen-run/demos/nim_frontend/fixtures/sysvq0asl.p.nif");
const EXPECTED_XNIF_GZ: &[u8] =
    include_bytes!("../../temen-run/demos/nim_frontend/fixtures/sysvq0asl.x.nif.gz");
const NIFLER_TL_GZ: &[u8] = include_bytes!("../../../browser/web/assets/nifler.temen.gz");

/// The driver guest = the compiler conductor. It builds four grant records `{fs, stdout, exit, exec}`
/// once (hexer just uses the first three — `grants_n = 3` ignores the `exec` record), then spawns each
/// phase into its own 256 MiB carve: nimsem (`grants_n = 4`, `--isSystem`) into carve0, join; hexer
/// (`grants_n = 3`, `c <sys>.s.nif`) into carve1, join. A fresh carve per phase (not one reused) keeps
/// hexer off nimsem's dirtied heap. Records live at `base+0` (POOL sits above the NULL guard); each
/// phase's argv is seeded at `carve + module_args_base`.
const GUEST: &str = r##"#![no_std]
#![allow(internal_features)]
#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! { loop {} }
#[no_mangle]
pub extern "C" fn rust_eh_personality() {}

#[repr(C, align(65536))]
struct Pool([u8; 939524096]); // 896 MiB static -> two disjoint 256 MiB carves (at 256/512 MiB), window 1 GiB
static mut POOL: Pool = Pool([0; 939524096]);

extern "C" {
    fn __vm_cap_resolve(name: *const u8, len: i64) -> i32;
    fn __vm_instantiate(
        inst: i32, module: i64, grants_ptr: i64, grants_n: i64,
        entry: i64, off: i64, size_log2: i64, quota: i64,
    ) -> i64;
    fn __vm_join(inst: i32, child: i64) -> i64;
}

unsafe fn wr(p: i64, b: u8) { (p as *mut u8).write(b); }

unsafe fn put_rec(base: i64, i: i64, name_off: i64, name_len: u32, handle: i32) {
    let rec = (base + i * 16) as *mut u32;
    rec.add(0).write(name_off as u32);
    rec.add(1).write(name_len);
    rec.add(2).write(handle as u32);
    rec.add(3).write(0);
}

// Seed argv [argc,envc header][arg0\0…] at `carve + 16512` (the child's module_args_base).
unsafe fn seed_argv(carve: i64, args: &[&[u8]]) {
    let a = carve + 16512;
    (a as *mut u32).add(0).write(args.len() as u32);
    (a as *mut u32).add(1).write(0);
    let mut p = 8i64; let mut i = 0;
    while i < args.len() {
        let s = args[i]; let mut j = 0;
        while j < s.len() { wr(a + p, s[j]); p += 1; j += 1; }
        wr(a + p, 0); p += 1; i += 1;
    }
}

#[no_mangle]
pub extern "C" fn run() -> i64 {
    unsafe {
        let inst = __vm_cap_resolve(b"inst".as_ptr(), 4);
        let nimsem = __vm_cap_resolve(b"nimsem".as_ptr(), 6);
        let hexer = __vm_cap_resolve(b"hexer".as_ptr(), 5);
        let fs = __vm_cap_resolve(b"fs".as_ptr(), 2);
        let out = __vm_cap_resolve(b"stdout".as_ptr(), 6);
        let ex = __vm_cap_resolve(b"exit".as_ptr(), 4);
        let exe = __vm_cap_resolve(b"exec".as_ptr(), 4);
        if inst < 0 || nimsem < 0 || hexer < 0 || fs < 0 || out < 0 || ex < 0 || exe < 0 { return -1; }
        let base = core::ptr::addr_of_mut!(POOL) as i64;

        // Four grant records {fs, stdout, exit, exec}; hexer's spawn passes grants_n=3 to drop exec.
        let nm = (base + 64) as *mut u8;
        let names: [&[u8]; 4] = [b"fs", b"stdout", b"exit", b"exec"];
        let mut off = 0usize; let mut ai = 0;
        let mut noffs: [i64; 4] = [0; 4];
        while ai < 4 {
            noffs[ai] = base + 64 + off as i64;
            let s = names[ai]; let mut j = 0;
            while j < s.len() { nm.add(off + j).write(s[j]); j += 1; }
            off += s.len(); ai += 1;
        }
        put_rec(base, 0, noffs[0], 2, fs);
        put_rec(base, 1, noffs[1], 6, out);
        put_rec(base, 2, noffs[2], 4, ex);
        put_rec(base, 3, noffs[3], 4, exe);

        // Two disjoint 256 MiB carves within the ~1 GiB window: hexer gets a fresh, zeroed sub-window
        // rather than one dirtied by nimsem's heap (reusing the same carve makes hexer thrash over
        // nimsem's committed pages — an order-of-magnitude slower). Both fit: base is low, so carve1's
        // top (~768 MiB) stays inside the window the instantiator was granted.
        let mask: i64 = (1 << 28) - 1;
        let carve0 = (base + 128 + mask) & !mask;
        let carve1 = carve0 + (1 << 28);

        // Phase 1: nimsem (4-cap) semchecks the system module.
        seed_argv(carve0, &[
            b"nimsem", b"--define:nimNativeAlloc", b"--define:nimNativeIo", b"m",
            b"--isSystem", b"nimcache/sysvq0asl.p.nif",
        ]);
        let c1 = __vm_instantiate(inst, nimsem as i64, base, 4, 0, carve0, 28, 0);
        let s1 = __vm_join(inst, c1);
        if s1 != 0 { return -10 + s1; }

        // Phase 2: hexer (3-cap, no exec) lowers the .s.nif nimsem wrote into the shared store.
        seed_argv(carve1, &[b"hexer", b"c", b"nimcache/sysvq0asl.s.nif"]);
        let c2 = __vm_instantiate(inst, hexer as i64, base, 3, 0, carve1, 28, 0);
        __vm_join(inst, c2)
    }
}
"##;

fn rustc_emit_ll(src: &Path, ll: &Path) -> bool {
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
    let out = c.wait_with_output().ok()?;
    w.join().ok()?;
    out.status.success().then_some(out.stdout)
}

fn collect(dir: &Path, prefix: &str, out: &mut Vec<(String, Vec<u8>)>) {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
            let e = e.unwrap();
            let p = e.path();
            if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                stack.push(p);
            } else {
                let rel = p
                    .strip_prefix(dir)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                out.push((format!("{prefix}{rel}"), std::fs::read(&p).unwrap()));
            }
        }
    }
}

fn unpack_syslib(dir: &Path) -> Option<()> {
    let tgz = dir.join("syslib.tar.gz");
    std::fs::write(&tgz, SYSLIB_TAR_GZ).ok()?;
    Command::new("tar")
        .arg("xzf")
        .arg(&tgz)
        .arg("-C")
        .arg(dir)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
        .then_some(())
}

#[test]
fn rust_driver_guest_drives_nimsem_then_hexer_chain_byte_exact() {
    let (Some(nimsem_bytes), Some(hexer_bytes), Some(nifler_bytes), Some(expected)) = (
        inflate(NIMSEM_CE_GZ),
        inflate(HEXER_CE_GZ),
        inflate(NIFLER_TL_GZ),
        inflate(EXPECTED_XNIF_GZ),
    ) else {
        eprintln!("note: skipping (gzip unavailable)");
        return;
    };

    let dir = std::env::temp_dir().join(format!("rust_driver_chain_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let libdir = dir.join("lib");
    std::fs::create_dir_all(&libdir).unwrap();
    if unpack_syslib(&libdir).is_none() {
        eprintln!("note: skipping (tar unavailable)");
        return;
    }

    let src = dir.join("g.rs");
    let ll = dir.join("g.ll");
    std::fs::write(&src, GUEST).unwrap();
    if !rustc_emit_ll(&src, &ll) {
        eprintln!("note: skipping (rustc unavailable)");
        return;
    }
    let t = temen_llvm::translate_ll_path(&ll).expect("translate chain guest");
    temen_verify::verify_module(&t.module).expect("driver verifies");
    let entry = t
        .exports
        .iter()
        .find(|(n, _)| n == "run")
        .expect("exports run")
        .1;
    let sp = t.entry_sp as i64;

    // Shared memfs: the stdlib closure (for nimsem's nifler grandchildren) + the parsed system module.
    let mut files = vec![];
    collect(&libdir, "lib/", &mut files);
    let flat: Vec<(String, Vec<u8>)> = files
        .iter()
        .filter_map(|(k, v)| {
            k.strip_prefix("lib/std/")
                .map(|r| (format!("lib/{r}"), v.clone()))
        })
        .collect();
    files.extend(flat);
    files.push(("nimcache/sysvq0asl.p.nif".into(), SYS_PNIF.to_vec()));

    let (factory, handle) = temen_run::fs::mem_fs_shared_factory(files, vec!["nimcache".into()]);
    let factory = Arc::new(factory);

    // nimsem's exec: the committed top-level nifler over the same store.
    let nifler_inst = Arc::new(
        instantiate(temen_encode::decode_module(&nifler_bytes).expect("decode nifler"))
            .expect("inst nifler"),
    );
    let programs: Vec<DomainProgram> = ["nifler", "/bin/nifler"]
        .iter()
        .map(|n| DomainProgram {
            name: (*n).into(),
            instance: nifler_inst.clone(),
            limits: Limits::default(),
        })
        .collect();
    let child_fs = {
        let f = factory.clone();
        HostCap::host_proc(0, move || (f)())
    };
    let exec_cap = domain_exec_with_fs(programs, child_fs);

    let nimsem = temen_encode::decode_module(&nimsem_bytes).expect("decode nimsem_ce");
    temen_verify::verify_module(&nimsem).expect("nimsem_ce verifies");
    let hexer = temen_encode::decode_module(&hexer_bytes).expect("decode hexer_ce");
    temen_verify::verify_module(&hexer).expect("hexer_ce verifies");

    let mut host = Host::new();
    let win = 1u64 << t.module.memory.as_ref().expect("driver window").size_log2;
    let inst = host.grant_instantiator(0, win);
    let nimsem_h = host.grant_module(&nimsem);
    let hexer_h = host.grant_module(&hexer);
    let fs_init: HostProc = (*factory)();
    let fs_fork: HostProcFork = {
        let f = Arc::clone(&factory);
        Arc::new(move |_pid| ForkedProc::shared((*f)()))
    };
    let fs_h = host.grant_host_proc_forkable(fs_init, fs_fork);
    let stdout_h = host.grant_stream(StreamRole::Out);
    let exit_h = host.grant_exit();
    let exec_h = exec_cap.install(&mut host, win);
    host.register_cap_name("inst", inst);
    host.register_cap_name("nimsem", nimsem_h);
    host.register_cap_name("hexer", hexer_h);
    host.register_cap_name("fs", fs_h);
    host.register_cap_name("stdout", stdout_h);
    host.register_cap_name("exit", exit_h);
    host.register_cap_name("exec", exec_h);

    let mut fuel = 3_000_000_000_000u64;
    let r = run_with_host(&t.module, entry, &[Value::I64(sp)], &mut fuel, &mut host);
    assert!(
        matches!(r.as_deref(), Ok([Value::I64(0)]) | Ok([Value::I32(0)])),
        "the guest-driven chain (nimsem then hexer) joined with status 0: {r:?}"
    );

    let (produced, _) = handle.seed();
    let x_nif = produced
        .iter()
        .find(|(k, _)| k == "nimcache/sysvq0asl.x.nif")
        .map(|(_, v)| v.clone())
        .expect("the guest-driven chain produced no sysvq0asl.x.nif");
    // Also confirm nimsem's intermediate really flowed through the shared store to hexer.
    assert!(
        produced
            .iter()
            .any(|(k, _)| k == "nimcache/sysvq0asl.s.nif"),
        "nimsem's .s.nif must be present in the shared store the guest handed to hexer"
    );
    assert_eq!(
        x_nif, expected,
        "the guest-driven nimsem->hexer chain's Leng must be byte-identical to the committed expected"
    );
}
