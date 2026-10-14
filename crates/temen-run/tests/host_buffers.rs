//! #2306 — host memory a guest causes through `temen-run`'s capabilities is charged to the domain
//! they were granted to, and refused with an errno past its ceiling, never an abort: the output a
//! host process writes to an `exec` job, and the listings a host-directory `opendir` holds.
#![cfg(unix)]

use temen_fs::{FS_CLOSEDIR, FS_OPENDIR, FS_READDIR};
use temen_interp::{cap_id, Host, WindowMem};
use temen_ir::errno::{ENOMEM, ENOSPC};
use temen_run::exec::{EXEC_CLOSE, EXEC_RUN};

/// A host whose own node's `channel` ceiling is `ceiling`.
fn host(ceiling: i64) -> Host {
    let mut h = Host::new();
    h.set_grant(-1, ceiling, -1, -1);
    h
}

/// A host process that writes more than its payer has room for is killed, and its run refused with
/// `-ENOMEM`: the output is charged as it arrives, so the host never holds more than the ceiling.
#[test]
fn an_exec_whose_output_does_not_fit_is_refused() {
    let mut h = host(1 << 20);
    let exec = temen_run::exec::host_exec(&["yes", "echo"]).install(&mut h, 0);
    let mut win = vec![0u8; 64];
    let mut run = |h: &mut Host, argv: &[u8]| {
        win[..argv.len()].copy_from_slice(argv);
        let mut wm = WindowMem::new(&mut win, 64);
        let args = [0, argv.len() as i64, 0, 0];
        h.cap_dispatch_slots(cap_id::HOST_PROC, EXEC_RUN, exec, &args, Some(&mut wm))
            .expect("dispatch")[0]
    };
    assert_eq!(run(&mut h, b"yes"), ENOMEM, "`yes` writes forever");
    assert_eq!(h.channel_used(), 0, "and what it wrote was dropped");
    let job = run(&mut h, b"echo\0hi");
    assert!(job >= 0, "a run whose output fits goes through: {job}");
    assert!(h.channel_used() > 0, "its output is held");
    let close = h.cap_dispatch_slots(cap_id::HOST_PROC, EXEC_CLOSE, exec, &[job], None);
    assert_eq!(close.expect("dispatch")[0], 0);
    assert_eq!(h.channel_used(), 0, "closed, the job holds nothing");
}

/// An `opendir` listing is held until it is closed, a [`temen_interp::TABLE_ENTRY`] and its name
/// for each entry: a directory the payer has no room to list is refused with `-ENOSPC`, before it
/// is all in memory.
#[test]
fn a_listing_is_charged_until_it_is_closed() {
    let dir = std::env::temp_dir().join(format!("temen_listing_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("d")).expect("temp dir");
    for i in 0..64 {
        std::fs::write(dir.join("d").join(format!("f{i:02}")), b"").expect("a file");
    }
    let mut h = host(16 << 10);
    let fs = temen_run::fs::host_fs(dir.clone()).install(&mut h, 0);
    let mut win = vec![0u8; 64];
    win[0] = b'd';
    let mut call = |h: &mut Host, op, args: &[i64]| {
        let mut wm = WindowMem::new(&mut win, 64);
        h.cap_dispatch_slots(cap_id::HOST_PROC, op, fs, args, Some(&mut wm))
            .expect("dispatch")[0]
    };
    let dh = call(&mut h, FS_OPENDIR, &[0, 1]);
    assert!(dh >= 0, "64 names fit: {dh}");
    let held = h.channel_used();
    assert!(
        held >= 65 * temen_interp::TABLE_ENTRY as i64,
        "an entry each: {held}"
    );
    assert!(call(&mut h, FS_READDIR, &[dh, 32, 32]) > 0);
    assert!(h.channel_used() < held, "a name read is handed back");
    assert_eq!(
        call(&mut h, FS_OPENDIR, &[0, 1]),
        ENOSPC,
        "no room for a second listing"
    );
    assert_eq!(call(&mut h, FS_CLOSEDIR, &[dh]), 0);
    assert_eq!(h.channel_used(), 0, "closed, the listing holds nothing");
    let _ = std::fs::remove_dir_all(&dir);
}
