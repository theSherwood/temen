//! The **POSIX personality as a powerbox `HostCap`** — the bridge that lets the LLVM on-ramp world
//! (`svm-llvm` guests, which reach the host through `__vm_cap_resolve` + `__vm_host_call`) use the
//! `svm-posix` process/fd/signal ABI (POSIX.md), the same surface the chibicc/name-binding world already
//! links against.
//!
//! `svm-posix` normally grants itself on a `Host` by name binding ([`svm_posix::grant`], the chibicc
//! path). The on-ramp instead resolves host capabilities by **name** and calls them by op number, exactly
//! as `demos/postgres/os_shim.c` reaches the `fs` cap. [`posix_cap`] wraps the personality's [`HostFn`]
//! factory ([`svm_posix::cap`]) in a [`HostCap`] at [`svm_interp::cap_id::HOST_FN`], so an embedder can
//! grant it under a name (e.g. `"posix"`) via `Instance::run_with_caps`, and a guest calls
//! `__vm_host_call(__vm_cap_resolve("posix"), OP_*, a, b, c, d)`.

use crate::HostCap;
use svm_posix::Posix;

/// Build the POSIX personality as a named powerbox capability. Returns the [`HostCap`] to grant (e.g.
/// `("posix", cap)` in `run_with_caps`) and the shared [`Posix`] handle the embedder keeps to read the
/// captured output, wire the spawn delegate ([`Posix::set_spawn`]), or raise a signal
/// ([`Posix::raise_signal`]). `heap_base`/`heap_end` bound the window-heap region the personality's
/// `malloc` hands out (both window offsets, within the guest window and clear of its data/stack); pass
/// `0, 0` when the guest brings its own allocator and only uses the process/fd/signal ops. `stdin`
/// preloads standard input for `read(0, …)`.
///
/// The cap grants once per backend over one shared personality state, so an interp/JIT differential run
/// observes the same `Posix` — read it back after the run.
pub fn posix_cap(heap_base: u64, heap_end: u64, stdin: Vec<u8>) -> (HostCap, Posix) {
    let (posix, make) = svm_posix::cap(heap_base, heap_end, stdin);
    (HostCap::host_fn(0, make), posix)
}
