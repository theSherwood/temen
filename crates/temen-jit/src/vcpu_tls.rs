//! Per-vCPU **thread-local register** (`vcpu.tls.get`/`vcpu.tls.set`, §12) for the JIT.
//!
//! One `i64` per OS thread. A vCPU is a 1:1 OS thread (`thread.spawn`), so a per-OS-thread word is
//! per-vCPU — and a fiber that migrated here (D57: any vCPU may resume any resumable fiber) reads
//! *this* thread's word, i.e. the current vCPU's value, exactly like the interpreter reads the running
//! `VCpu`'s `tls`. Deliberately **independent of the fiber substrate** (unlike `CURRENT_RT`), so a
//! plain non-fiber root run also has a TLS word. Seeded at vCPU entry to a dense id (root = 0, a
//! spawned vCPU to its `idx + 1`), guest-overwritable via the `set` thunk.

use std::cell::Cell;

thread_local! {
    /// This OS thread's (vCPU's) TLS word. Defaults to 0 (the root's seed); [`seed`] resets it at the
    /// start of every root/child run so a reused worker thread can't leak a prior run's `set`.
    static VCPU_TLS: Cell<i64> = const { Cell::new(0) };
    /// #2160 — how many serve handlers this vCPU is running. An embedder's serve loop counts each
    /// handler it invokes, and a serve op that finds the count nonzero is nested under a handler
    /// (refused, as the oracle refuses it). Per vCPU like the TLS word: a §14 task carries its count
    /// across workers, and a child run inline on another vCPU's thread starts from 0.
    static SERVE_HANDLERS: Cell<u32> = const { Cell::new(0) };
    /// #2964 — the errno the innermost running serve handler answered its dispatch with
    /// (`svc.fail`), which its end replies as a transport failure. Saved and restored with
    /// [`SERVE_HANDLERS`] ([`Serve`]), so it follows its handler across parks and workers.
    static SERVE_FAIL: Cell<Option<i64>> = const { Cell::new(None) };
}

/// The serve state a vCPU carries between residencies: how many serve handlers it is running
/// (#2160), and whether the innermost answered its dispatch as a failure (#2964).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Serve {
    pub(crate) handlers: u32,
    pub(crate) failed: Option<i64>,
}

impl Serve {
    /// #2964 — the errno this state's handler answered its dispatch with, if it did.
    pub(crate) fn failed(&self) -> Option<i64> {
        self.failed
    }
}

/// This vCPU's serve state ([`Serve`]).
#[inline(never)]
pub(crate) fn serve_state() -> Serve {
    Serve {
        handlers: SERVE_HANDLERS.with(|c| c.get()),
        failed: SERVE_FAIL.with(|c| c.get()),
    }
}

/// Set this vCPU's serve state ([`Serve`]).
#[inline(never)]
pub(crate) fn set_serve_state(s: Serve) {
    SERVE_HANDLERS.with(|c| c.set(s.handlers));
    SERVE_FAIL.with(|c| c.set(s.failed));
}

/// A serve handler starts on this vCPU: counted, with no failure of its own yet. The state to give
/// back when it ends ([`leave_handler`]).
#[cfg_attr(not(fiber_rt), allow(dead_code))]
pub(crate) fn enter_handler() -> Serve {
    let prev = serve_state();
    set_serve_state(Serve {
        handlers: prev.handlers + 1,
        failed: None,
    });
    prev
}

/// The handler [`enter_handler`] started has ended: the state before it comes back, and the
/// handler's own failure, if it answered its dispatch with one (#2964).
#[cfg_attr(not(fiber_rt), allow(dead_code))]
pub(crate) fn leave_handler(prev: Serve) -> Option<i64> {
    let failed = SERVE_FAIL.with(|c| c.get());
    set_serve_state(prev);
    failed
}

/// #2964 — `svc.fail(errno)`: the running serve handler's dispatch answers `errno` as a transport
/// failure when the handler ends. `false` off a handler, which the caller answers `-EINVAL`.
#[inline(never)]
pub fn fail_dispatch(errno: i64) -> bool {
    if serve_handlers() == 0 {
        return false;
    }
    SERVE_FAIL.with(|c| c.set(Some(errno)));
    true
}

/// Seed/reset the current OS thread's TLS word. Called at the start of a root run (→ 0) and at a
/// spawned vCPU's entry (→ its dense id), so the value is deterministic and never stale.
pub(crate) fn seed(v: i64) {
    VCPU_TLS.with(|c| c.set(v));
}

/// `vcpu.tls.get` thunk — the current vCPU's TLS word. A pure thread-local read; it cannot fault, so
/// it takes no window/trap context (unlike the `call.cap`/`gc.roots` thunks).
pub(crate) extern "C" fn get() -> i64 {
    VCPU_TLS.with(|c| c.get())
}

/// `vcpu.tls.set` thunk — set the current vCPU's TLS word.
pub(crate) extern "C" fn set(v: i64) {
    VCPU_TLS.with(|c| c.set(v));
}

/// The current vCPU's serve-handler count (see `SERVE_HANDLERS`). `#[inline(never)]`, like its
/// setter: a handler can park and its task resume on another worker, so every access must find the
/// running thread's word, never one computed before the park (#1466).
#[inline(never)]
pub fn serve_handlers() -> u32 {
    SERVE_HANDLERS.with(|c| c.get())
}
