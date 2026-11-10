//! #2805 — a guarded entry arms without saving the thread's signal mask, so a caught fault has to
//! leave the mask as the entry found it. It does because the trap handler runs with the mask the
//! fault found (`SA_NODEFER`, an empty `sa_mask`): were the handler ever installed to block the
//! signal it takes, the jump out of it would leave SIGSEGV blocked, and the thread's next fault
//! would kill the process. Two faults in a row on one thread, a signal blocked across both.
#![cfg(unix)]

use temen_jit::{compile_and_run_capture_reserved, JitOutcome, TrapKind};

/// The calling thread's blocked set.
fn blocked() -> libc::sigset_t {
    // SAFETY: `set` is written by `pthread_sigmask` before it is read; a null `set` only reads.
    unsafe {
        let mut set = std::mem::zeroed();
        libc::pthread_sigmask(libc::SIG_BLOCK, std::ptr::null(), &mut set);
        set
    }
}

fn same(a: &libc::sigset_t, b: &libc::sigset_t) -> bool {
    // SAFETY: both sets are initialised; `sigismember` only reads them.
    (1..32).all(|sig| unsafe { libc::sigismember(a, sig) == libc::sigismember(b, sig) })
}

#[test]
fn a_caught_fault_leaves_the_thread_s_signal_mask_as_it_was() {
    let src = "memory 16
func (i64) -> (i64) {
block 0 (va: i64) {
  vx = i64.load va
  return vx
  }
}
";
    let m = temen_text::parse_module(src).expect("parse");
    std::thread::spawn(move || {
        // SAFETY: `set` is initialised by `sigemptyset` before use.
        unsafe {
            let mut set = std::mem::zeroed();
            libc::sigemptyset(&mut set);
            libc::sigaddset(&mut set, libc::SIGUSR2);
            libc::pthread_sigmask(libc::SIG_BLOCK, &set, std::ptr::null_mut());
        }
        let before = blocked();
        // SAFETY: `before` is initialised.
        assert_eq!(unsafe { libc::sigismember(&before, libc::SIGUSR2) }, 1);
        let init = vec![0u8; 1 << 16];
        // Past the mapped window, then far past its reservation.
        for addr in [1i64 << 20, 1 << 40] {
            let (out, _) = compile_and_run_capture_reserved(&m, 0, &[addr], &init, 0)
                .expect("the JIT compiles the load");
            assert!(
                matches!(out, JitOutcome::Trapped(TrapKind::MemoryFault)),
                "a load at {addr:#x} ends the run with a memory fault: {out:?}"
            );
            assert!(
                same(&before, &blocked()),
                "the fault at {addr:#x} left the thread's signal mask changed"
            );
        }
    })
    .join()
    .expect("the faulting thread runs to its end");
}
