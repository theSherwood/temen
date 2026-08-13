# `rust-svm` — the Rust `std` build lane for the svm-llvm on-ramp

The reproducible toolchain artifacts that let `rustc` build the **Rust standard
library** for svm, so real `std` programs flow through the LLVM on-ramp
(`crates/svm-llvm`). Design record: **`LLVM.md` §10** ("Rust `std` on the on-ramp") —
the route decision, the ABI pins, and the `__vm_host_call` seam. (The former
repo-root `RUST_STD.md` slice plan was folded there and retired once the platform
layer landed.)

This directory is the **build-std lane**: it makes `std` *compile* for a custom
`os = "svm"` target and translate through the on-ramp. The full platform layer
now works (see "Status" below).

## Contents

| File | What it is |
|---|---|
| `x86_64-unknown-svm.json` | The **lean** target spec. `os=svm`, `panic=abort`, `singlethread=true` (single-threaded but keeps 64-bit atomics), `has-thread-local=false`, static reloc, no PIE. Selects std's `no_threads` `Mutex`/`Once`/TLS. |
| `x86_64-unknown-svm-threads.json` | The **threaded** target spec. Same as the lean spec but `singlethread=false` (real atomic-ordering codegen), `has-thread-local=true`, and `env=threads` (→ `cfg(target_env="threads")`, the overlay's discriminator). Selects futex-backed `sys/sync`, `native` TLS, and the svm `sys/thread` PAL — so `std::thread::spawn`/`join` run (one vCPU per thread over the §12 thread ops). |
| `std-overlay.patch` | `cfg_select!` arm additions routing `target_os="svm"` to the svm leaf-module impls. Threading-independent arms (`sys/{alloc,random,io/error}`, `stdio`/`pal`/`args`/`time`/`env`/`fs`/`pipe`/`process`/`net`, `exit`) apply to both specs. The **sync/TLS** arms are gated on `target_env`: `sys/sync/{mutex,condvar,once,rwlock,thread_parking}` + `sys/sync/futex` route to `futex` only under `target_env="threads"`; `sys/thread_local` keeps `no_threads` under `not(target_env="threads")` and falls through to `native` otherwise. |
| `svm-futex-imp.rs` | The futex primitive (copied to `sys/sync/futex/svm.rs`). A port of std's `wasm.rs` whose wait/notify reach the svm §12 futex via the on-ramp intrinsics `__vm_wait32`/`__vm_notify`. Backs the threaded spec's `sys/sync`; unused by the lean spec. |
| `svm-thread-imp.rs` | The thread PAL (copied to `sys/thread/svm.rs`). `std::thread::Thread::new`/`join` over `__vm_thread_spawn`/`__vm_thread_join` (§12), via a fixed `__rust_thread_start` trampoline (the boxed `ThreadInit` rides `arg`, since the spawn op needs a direct funcidx). The trampoline gives the new vCPU its own TLS block — copied from the `__vm_tls_template()` init image, `__vm_tls_size()` bytes, then `vcpu.tls.set` (NIM.md §3d Tier-2) — before running the closure; `Thread::new` carves the vCPU's data stack. `available_parallelism` = 1 (the embedder picks cooperative vs parallel). Threaded spec only; the lean spec keeps `sys/thread/unsupported.rs`. |
| `svm-alloc-imp.rs` | The allocator `imp` (copied to `sys/alloc/svm.rs`). Forwards `alloc`/`dealloc`/`realloc` to the C `malloc` family, which the on-ramp synthesizes as an in-window guest bump allocator (LLVM.md slice S). |
| `svm-stdio-imp.rs` | The stdio PAL (copied to `sys/stdio/svm.rs`). `Stdin`/`Stdout` reach the host through `extern "C" write`/`read` (on-ramp "Lane C" → powerbox stdout/stdin, POSIX.md 0/1); `Stderr` calls `__vm_write_stderr` → the distinct powerbox stderr `Stream`. So `println!` and `eprintln!` write real bytes on separate streams. |
| `svm-pal.rs` | The svm PAL proper (copied to `sys/pal/svm.rs`). Mirrors the `unsupported` PAL, but its `init` captures the powerbox-threaded `argv` (so `std::env::args` works), and it hosts the `host` bridge — `__vm_cap_resolve("posix")` + per-op `__vm_host_call` wrappers — that the richer surface (`time`/`env`/`fs`/`pipe`/`process`) reaches the host through. |
| `svm-args-imp.rs` | The args module (copied to `sys/args/svm.rs`). Stores `(argc, argv)` at init and walks them as C strings on demand — the "stored at startup" half of the unix strategy, no `os::unix` dependency. |
| `svm-time-imp.rs` | The time module (copied to `sys/time/svm.rs`). `Instant`/`SystemTime::now()` call the PAL `host` bridge's `clock` op (svm-posix `OP_CLOCK`) — monotonic for `Instant`, realtime for `SystemTime`. Needs a granted `posix` cap (`run_with_caps`); without one the clock reads zero. |
| `svm-env-imp.rs` | The env module (copied to `sys/env/svm.rs`). `getenv`/`setenv`/`unsetenv`/`vars` reach the posix env map via the `host` bridge's **buffer-writing** ops (`OP_GETENV_R`/`OP_SETENV`/`OP_UNSETENV`/`OP_ENVIRON`) — copies into guest memory, no personality arena. |
| `svm-fs-imp.rs` | The fs module (copied to `sys/fs/svm.rs`). `File` (open/read/write/seek/`try_clone`), `metadata`/`read_dir`/`remove_file`/`exists`, and the directory ops `create_dir`/`create_dir_all`/`rename`/`remove_dir` reach the personality's in-memory filesystem via the `host` bridge's file ops (`OP_OPEN`/`OP_READ`/`OP_WRITE`/`OP_LSEEK`/`OP_CLOSE`/`OP_UNLINK`/`OP_STAT`/`OP_OPENDIR`/`OP_READDIR`/`OP_CLOSEDIR`/`OP_MKDIR`/`OP_RENAME`/`OP_RMDIR`/`OP_DUP`). The memfs has no symlinks or mutable perms, so `lstat==stat`, perms are always-writable, and `symlink`/hard-link/perms/times/`canonicalize` stay `Unsupported`. Needs a granted `posix` cap. |
| `svm-pipe-imp.rs` | The anonymous-pipe module (copied to `sys/pipe/svm.rs`). `sys::pipe::Pipe` is a pair of fds over one in-personality byte FIFO (`OP_PIPE`), read/written through `OP_READ`/`OP_WRITE` and closed on drop — the plumbing `std::process` captures child stdout through. |
| `svm-process-imp.rs` | The process module (copied to `sys/process/svm.rs`). `std::process::Command` over the personality's **fork-free** spawn (`OP_SPAWN2`/`OP_WAITPID`): a spawn runs the named command to completion synchronously, `output` captures **stdout and stderr** by routing them to `OP_PIPE` write ends carried **per-child in the `OP_SPAWN2` request** (parallel-safe — no global fd-1/fd-2 redirect dance, #848), and `wait`/`status` reap the exit code. The command is resolved by the embedder's spawn delegate (`Posix::set_spawn` → `SpawnResult { stdout, stderr, status }`); without one, spawning is `Unsupported`. Synchronous model ⇒ no live child to stream stdin into (stdin is whatever fd 0 holds). |
| `svm-net-imp.rs` | The net module (copied to `sys/net/connection/svm.rs`). `TcpStream`/`TcpListener`/`lookup_host` over the **`net` capability** (POSIX.md §5a) — a named handle separate from `posix` holding only the authority surface (`connect`/`bind`/`accept`/`shutdown`/`resolve`); a connected socket is an ordinary fd, so data rides the posix `read`/`write` ops. Loopback = the deterministic in-personality memnet; beyond loopback = the embedder's `NetDelegate` (`Posix::set_net`) or fail closed. Socket knobs with no memnet meaning are accept-and-ignore; `UdpSocket` fails closed (the datagram slice is a follow-up). Needs a granted `net` cap. |
| `apply-overlay.sh` | Applies the overlay to the active nightly's `rust-src` (idempotent). |

## Why an overlay is needed at all (the S0 finding)

`restricted_std` alone does **not** work for a brand-new OS on modern std: the
PAL (`sys/pal/`) has a clean `_ => unsupported` fallback, but the *leaf* modules
(`sys/alloc`, `sys/thread_local`, `sys/random`, `sys/io/error`) enumerate
`target_os` with **no catch-all**, so an unknown `os=svm` fails to build with
missing-symbol errors. The overlay adds the five `svm` arms that make those
modules resolve. This is why "std that builds" and "std that runs" partly merge
(LLVM.md §10, alternative D): you
cannot get "std that builds but errors at runtime" for a novel OS without
supplying this much PAL wiring. (Historically this was the S0 finding, when the
overlay still routed I/O to the `unsupported` PAL; the real svm PAL — `svm-pal.rs`
+ the leaf `imp`s listed above — has since landed, so stdio/fs/args/env/time/net/
process all reach the host for real.)

The `singlethread=true` spec field (borrowed from the `zkvm` target) is what
selects the `no_threads` sync/TLS implementations — `Mutex` is a `Cell`, TLS is a
plain `static` — while still exposing atomics. Without it, std sees
`target_has_threads` and **rejects** the `no_threads` impls with a hard
`compile_error!`.

### Two target specs (the threaded lane, #779/#821)

Flipping `singlethread=false` is therefore not a free knob: it forces std to use
real futex-backed `Mutex`/`Condvar`/`Once`/`RwLock` and a threads-valid TLS, or it
won't compile at all. Rather than perturb the working lean spec, the threaded
build is a **second target**, `x86_64-unknown-svm-threads`, selected by its
`env=threads` cfg. The overlay serves both from one `rust-src` tree: the sync/TLS
arms are gated on `target_env`, so the lean spec stays byte-identical and the
threaded spec routes to `futex` sync (over `svm-futex-imp.rs` → `__vm_wait32`/
`__vm_notify`) and `native` TLS.

Native TLS emits `#[thread_local]` globals + `llvm.threadlocal.address`; the
on-ramp lowers those as the NIM.md §3d **Tier-2** per-vCPU `vcpu.tls` block
(`vcpu.tls.get() + offset`), so each spawned thread's thread-locals are isolated.
`std::thread::spawn`/`join` run over the §12 thread ops via `svm-thread-imp.rs`
(one vCPU per thread), with futex-backed sync and real atomics — the full #779
threads epic. Differential-tested on the cooperative (deterministic) driver, with
parallel-driver smoke coverage (`std_threads_spec_*` / `std_threads_parallel_*` in
`tests/std_guest.rs`).

**Upgrading a toolchain:** the overlay can't be re-applied on top of an older
(pre-threads) overlay — `apply-overlay.sh` detects a stale overlay and asks you to
reinstall a clean `rust-src` first.

## Usage

```sh
rustup toolchain install nightly
rustup component add rust-src --toolchain nightly
./apply-overlay.sh                       # patch the toolchain's rust-src

# lean, single-threaded (no_threads Mutex/TLS):
RUSTC_BOOTSTRAP=1 cargo +nightly build \
  -Z build-std=core,alloc,std,panic_abort -Z json-target-spec \
  --target x86_64-unknown-svm.json --release

# threaded (futex sys/sync + native TLS; spawn still Unsupported):
RUSTC_BOOTSTRAP=1 cargo +nightly build \
  -Z build-std=core,alloc,std,panic_abort -Z json-target-spec \
  --target x86_64-unknown-svm-threads.json --release
```

To get a single self-contained `.ll` for the on-ramp, build a `#![no_main]`
crate that exports a `#[no_mangle] pub extern "C"` entry, with `lto = "fat"` +
`codegen-units = 1`, and `--emit=llvm-ir`. The whole reachable std graph lands in
one module whose only undefined externals are `malloc`/`free`/`realloc` and the
`llvm.*` intrinsics the on-ramp already handles.

## Status (2026-08-13)

- **`std` builds and runs for `os=svm`** ✅ — target JSON + this overlay, via
  `-Zbuild-std` on nightly. A `std` binary translated through the on-ramp runs on
  the powerbox: pure compute returns the right exit code, and **`println!` writes
  real bytes byte-identical to native** (`crates/svm-llvm/tests/std_guest.rs`).
- **One on-ramp change was needed** — parsing call operand bundles
  (`ll/parse.rs`); the earlier "packed-struct globals" suspicion was wrong (those
  parse fine). Everything else — malloc-synth, the `Memory` grant, `lang_start` —
  worked as-is off the bin's `main`.
- **Working today:** stdout (`println!`), `stderr` (`eprintln!`), `process::exit`,
  `env::args`, **`std::time`** (`Instant`/`SystemTime`), **`std::env`**
  (`var`/`var_os`/`set_var`/`remove_var`/`vars`, via a granted posix cap),
  **`std::fs`** (`File` open/read/write/seek/`try_clone`, `metadata`/`read_dir`/
  `remove_file`/`exists`, `create_dir`/`create_dir_all`/`rename`/`remove_dir`, via a
  granted posix cap), **`std::process`** (`Command` spawn/`output`/
  `status`/`wait`, via a granted posix cap + spawn delegate), **`std::net`**
  (`TcpStream`/`TcpListener`/`lookup_host`: loopback over the memnet, egress via a
  granted `net` cap + `NetDelegate`; `UdpSocket` is a follow-up), heap/`Vec`,
  collections (incl. **`HashMap`**, deterministic per-guest seed), `fmt`, iterators,
  and the full **`std::thread`** surface (spawn/join, futex `Mutex`/`Condvar`/`mpsc`,
  Tier-2 TLS). `std::env::var`/`vars` (the `str`-Debug paths) work now that the
  on-ramp entry-block slot-numbering bug #755 is fixed.
- **The two paths:** the powerbox stream/exit handles carry stdio/exit/args (no
  extra grant); the **posix-cap path** (`run_with_caps` + a `posix` cap, reached
  via the PAL `host` bridge's `__vm_host_call`) carries `time`/`env`/`fs`/`process`
  — this is where the richer, many-op surface scales without growing the powerbox.
  Networking adds a **second named cap** (`"net"`, POSIX.md §5a) so socket
  *authority* is its own grant while socket *data* rides the posix fd ops.
- **Not yet on the fs surface:** `symlink`/hard-link/`set_permissions`/`set_times`/
  `canonicalize` (no host op / no perm-time model on the memfs backend — they return
  `Unsupported`). `create_dir`/`rename`/`remove_dir`/`try_clone` now work (memfs dir
  ops `OP_MKDIR`/`OP_RENAME`/`OP_RMDIR` + `OP_DUP`). See LLVM.md §10.
- **process caveats:** spawn is **fork-free and synchronous** (the child runs to
  completion inside `spawn`), so there is no live child to stream stdin into
  (`StdioPipes` yields no writable stdin — a piped-stdin write after `spawn` can't
  reach an already-exited child; the child's stdin is whatever fd 0 holds at spawn),
  and `Command::spawn` returns an already-exited child. `output()` **does** capture
  both stdout and stderr (via the parallel-safe per-child `OP_SPAWN2`, #848).
  `fork`/`exec`-in-place stay parked (they ride the fork/exec/bash epic). See LLVM.md §10.

## Reproducibility note

The overlay is pinned against a nightly `rust-src`; when the pin moves, re-apply
and, if the surrounding `cfg_select!` arms shifted, regenerate `std-overlay.patch`
(the arms are stable — they sit next to the long-lived `vexos`/`zkvm` arms). The
scheduled **std-guest CI lane** (`.github/workflows/ci.yml`, nightly build-std;
ISSUES.md I55) exercises both target specs and guards drift.
