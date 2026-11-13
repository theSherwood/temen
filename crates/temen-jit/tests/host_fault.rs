//! #2996 — a fault that host code takes on the guest's window inside a guarded call is a host bug,
//! not the guest's `MemoryFault`. A build with debug assertions (this suite's) dies of it, with the
//! faulting pc and address reported, instead of unwinding past the host's frames.
//!
//! A death test: the parent re-runs this test binary with `TEMEN_HOST_FAULT_CHILD` set, so only the
//! child process faults, and reads how it died.
#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

use std::process::Command;
use temen_jit::{compile_and_run_with_host, JitOutcome};
use temen_text::parse_module;

const CHILD: &str = "TEMEN_HOST_FAULT_CHILD";

/// A cap thunk with a bug: it writes a page of the window that is not there (the reserved tail,
/// past the 64 KiB window), without checking.
unsafe extern "C" fn faulty_thunk(
    vm: &temen_jit::VmCtx,
    _buf: *mut i64,
    _counts: u64,
    _type_id: u32,
    _op: u32,
    _handle: i32,
) {
    unsafe { core::ptr::write_volatile(vm.window().base.add(1 << 20), 1) };
}

const CALLS_THE_THUNK: &str = "memory 16
func () -> (i64) {
block 0 () {
  vh = i32.const 0
  call.cap 2 0 () -> () vh ()
  vr = i64.const 7
  return vr
  }
}
";

/// The child: runs the module, whose thunk faults. Does nothing unless the parent started it.
#[test]
fn child_a_thunk_faults_on_the_window() {
    if std::env::var_os(CHILD).is_none() {
        return;
    }
    let m = parse_module(CALLS_THE_THUNK).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let out = compile_and_run_with_host(&m, 0, &[], faulty_thunk, core::ptr::null_mut());
    // Reached only if the fault did not kill the process.
    match out {
        Ok(JitOutcome::Trapped(k)) => println!("SURVIVED: trapped {k:?}"),
        other => println!("SURVIVED: {other:?}"),
    }
}

#[cfg(debug_assertions)]
#[test]
fn a_host_fault_on_the_window_kills_the_process_and_names_the_pc() {
    let out = Command::new(std::env::current_exe().expect("this test binary"))
        .args([
            "--exact",
            "child_a_thunk_faults_on_the_window",
            "--nocapture",
        ])
        .env(CHILD, "1")
        .output()
        .expect("run the child");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stdout.contains("SURVIVED"),
        "the host fault was taken as the guest's: {stdout}"
    );
    assert!(
        stderr
            .contains("temen-jit: host code faulted on guest memory inside a guarded call: pc=0x")
            && stderr.contains("addr=0x"),
        "no report of the faulting pc and address: {stderr}"
    );
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        let sig = out.status.signal();
        assert!(
            sig == Some(libc::SIGSEGV) || sig == Some(libc::SIGBUS),
            "the child should die of the fault, not {:?}: {stderr}",
            out.status
        );
    }
    #[cfg(windows)]
    assert_eq!(
        out.status.code(),
        Some(0xC000_0005_u32 as i32),
        "the child should die of the access violation: {stderr}"
    );
}
