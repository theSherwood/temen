//! #2306 — the process's descriptor pool: each descriptor a guest causes holds one of the pool's
//! while it is open, and one the pool has no room for is refused with `-EMFILE` before the host asks
//! the OS for it, never a panic. Covered: a guest's region mint, on the tree-walker and on the JIT;
//! the directory a host-directory grant holds, and an open under it; and an `FS_MAP_REGION`. The
//! test lowers the soft limit first, so the pool is small on any host. Its own binary: the pool and
//! the limit are the process's.
#![cfg(unix)]

use core::ffi::c_void;
use temen_fs::{FS_MAP_REGION, FS_OPEN, O_READ, O_WRITE};
use temen_interp::{cap_id, run_with_host, Host, Value, WindowMem};
use temen_ir::errno::EMFILE;
use temen_jit::{compile_and_run_capture_reserved_with_host_ex, JitOutcome};

/// The soft limit the test runs under: a pool of 256 less the host's 64.
const SOFT_LIMIT: u64 = 256;

/// Mint a 64 KiB region through the `AddressSpace` handle it is passed, and return the handle, or
/// the errno.
const MINT: &str = r#"memory 16
func (i32) -> (i64) {
block 0 (vas: i32) {
  len = i64.const 65536
  h = call.cap 5 5 (i64) -> (i64) vas (len)
  return h
  }
}
"#;

#[allow(clippy::unnecessary_cast)] // `rlim_t` is `u64` here, narrower on some hosts
fn lower_soft_limit(to: u64) {
    let mut lim = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: `getrlimit` and `setrlimit` read and write a valid `rlimit`; only the soft limit drops.
    unsafe {
        assert_eq!(libc::getrlimit(libc::RLIMIT_NOFILE, &mut lim), 0);
        lim.rlim_cur = to.min(lim.rlim_max as u64) as libc::rlim_t;
        assert_eq!(libc::setrlimit(libc::RLIMIT_NOFILE, &lim), 0);
    }
}

/// The guest's mint on the tree-walker (`jit` false) or the JIT: the handle, or the errno.
fn mint(jit: bool) -> i64 {
    let m = temen_text::parse_module(MINT).expect("parse");
    let mut host = Host::new();
    host.set_region_factory(temen_run::new_shared_region);
    let a = host.grant_address_space(0, 1 << 16);
    if jit {
        let ctx = &mut host as *mut Host as *mut c_void;
        let (jo, _) = compile_and_run_capture_reserved_with_host_ex(
            &m,
            0,
            &[a as i64],
            &[],
            24,
            temen_run::cap_thunk,
            ctx,
            None,
            None,
        )
        .expect("jit run");
        match jo {
            JitOutcome::Returned(v) => v[0],
            o => panic!("the JIT ends abnormally: {o:?}"),
        }
    } else {
        let mut fuel = 1_000_000;
        match run_with_host(&m, 0, &[Value::I32(a)], &mut fuel, &mut host).as_deref() {
            Ok([Value::I64(h)]) => *h,
            r => panic!("the tree-walker ends abnormally: {r:?}"),
        }
    }
}

#[test]
fn a_descriptor_the_pool_cannot_hold_is_refused_with_emfile() {
    lower_soft_limit(SOFT_LIMIT);
    let dir = std::env::temp_dir().join(format!("temen_fd_pool_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    std::fs::write(dir.join("data"), [0u8; 4096]).expect("a file");

    // Regions the embedder holds, a descriptor each, until the pool is full.
    let mut held: Vec<_> = std::iter::from_fn(|| temen_run::new_shared_region(4096).ok()).collect();
    assert_eq!(
        held.len() as u64,
        SOFT_LIMIT - 64,
        "the pool is the soft limit less the host's"
    );
    assert_eq!(temen_run::new_shared_region(4096).err(), Some(EMFILE));
    for jit in [false, true] {
        assert_eq!(mint(jit), EMFILE, "a guest's mint is refused (JIT: {jit})");
    }

    // The host-directory grant: its first path op opens its root, which takes one, and an open takes
    // one more, so a region over the file has none left.
    let mut host = Host::new();
    let fs_h = temen_run::fs::host_fs_mmap(dir.clone()).install(&mut host, 0);
    let mut win = vec![0u8; 4096];
    win[..4].copy_from_slice(b"data");
    let mut wm = WindowMem::new(&mut win, 4096);
    let mut call = |op, args: &[i64]| {
        host.cap_dispatch_slots(cap_id::HOST_PROC, op, fs_h, args, Some(&mut wm))
            .expect("dispatch")[0]
    };
    let open = [0i64, 4, O_READ | O_WRITE, 0];
    assert_eq!(call(FS_OPEN, &open), EMFILE, "an open is refused");
    held.pop();
    assert_eq!(
        call(FS_OPEN, &open),
        EMFILE,
        "the descriptor handed back is the root's"
    );
    held.pop();
    let fd = call(FS_OPEN, &open);
    assert!(fd >= 0, "the next one handed back is the open's: {fd}");
    assert_eq!(
        call(FS_MAP_REGION, &[fd, 0, 4096]),
        EMFILE,
        "and a file region is refused"
    );

    // Handed back, every use goes through.
    held.clear();
    for jit in [false, true] {
        assert!(mint(jit) >= 0, "a guest's mint goes through (JIT: {jit})");
    }
    assert!(
        call(FS_MAP_REGION, &[fd, 0, 4096]) >= 0,
        "a file region goes through"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
