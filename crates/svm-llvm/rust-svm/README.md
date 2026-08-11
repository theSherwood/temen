# `rust-svm` — the Rust `std` build lane for the svm-llvm on-ramp

The reproducible toolchain artifacts that let `rustc` build the **Rust standard
library** for svm, so real `std` programs flow through the LLVM on-ramp
(`crates/svm-llvm`). Design and slice plan: **`RUST_STD.md`** at the repo root.

This directory is the **S0 lane** (RUST_STD.md §8): it makes `std` *compile* for a
custom `os = "svm"` target. Getting the emitted IR to *translate* through the
on-ramp is the ongoing S1 work (see "Status" below).

## Contents

| File | What it is |
|---|---|
| `x86_64-unknown-svm.json` | The custom target spec. `os=svm`, `panic=abort`, `singlethread=true` (single-threaded but keeps 64-bit atomics), static reloc, no PIE. |
| `std-overlay.patch` | `cfg_select!` arm additions routing `target_os="svm"` to the right leaf-module impls: the minimal (no-OS, single-thread) ones for `sys/{alloc,thread_local,random,io/error}` (as `vexos`/`zkvm` do), the svm `stdio`/`pal`/`args`/`time`/`env`/`fs`/`pipe`/`process` modules, and the powerbox `exit` in `sys/exit.rs`. 54 added lines across 12 files. |
| `svm-alloc-imp.rs` | The allocator `imp` (copied to `sys/alloc/svm.rs`). Forwards `alloc`/`dealloc`/`realloc` to the C `malloc` family, which the on-ramp synthesizes as an in-window guest bump allocator (LLVM.md slice S). |
| `svm-stdio-imp.rs` | The stdio PAL (copied to `sys/stdio/svm.rs`). `Stdin`/`Stdout` reach the host through `extern "C" write`/`read` (on-ramp "Lane C" → powerbox stdout/stdin, POSIX.md 0/1); `Stderr` calls `__vm_write_stderr` → the distinct powerbox stderr `Stream`. So `println!` and `eprintln!` write real bytes on separate streams. |
| `svm-pal.rs` | The svm PAL proper (copied to `sys/pal/svm.rs`). Mirrors the `unsupported` PAL, but its `init` captures the powerbox-threaded `argv` (so `std::env::args` works), and it hosts the `host` bridge — `__vm_cap_resolve("posix")` + per-op `__vm_host_call` wrappers — that the richer surface (`time`/`env`/`fs`/`pipe`/`process`) reaches the host through. |
| `svm-args-imp.rs` | The args module (copied to `sys/args/svm.rs`). Stores `(argc, argv)` at init and walks them as C strings on demand — the "stored at startup" half of the unix strategy, no `os::unix` dependency. |
| `svm-time-imp.rs` | The time module (copied to `sys/time/svm.rs`). `Instant`/`SystemTime::now()` call the PAL `host` bridge's `clock` op (svm-posix `OP_CLOCK`) — monotonic for `Instant`, realtime for `SystemTime`. Needs a granted `posix` cap (`run_with_caps`); without one the clock reads zero. |
| `svm-env-imp.rs` | The env module (copied to `sys/env/svm.rs`). `getenv`/`setenv`/`unsetenv`/`vars` reach the posix env map via the `host` bridge's **buffer-writing** ops (`OP_GETENV_R`/`OP_SETENV`/`OP_UNSETENV`/`OP_ENVIRON`) — copies into guest memory, no personality arena. |
| `svm-fs-imp.rs` | The fs module (copied to `sys/fs/svm.rs`). `File` (open/read/write/seek), `metadata`/`read_dir`/`remove_file`/`exists` reach the personality's in-memory filesystem via the `host` bridge's file ops (`OP_OPEN`/`OP_READ`/`OP_WRITE`/`OP_LSEEK`/`OP_CLOSE`/`OP_UNLINK`/`OP_STAT`/`OP_OPENDIR`/`OP_READDIR`/`OP_CLOSEDIR`). The memfs has no symlinks or mutable perms, so `lstat==stat`, perms are always-writable, and the metadata-mutation ops (perms/times, `mkdir`, `rename`, `symlink`) borrow the `unsupported` PAL. Needs a granted `posix` cap. |
| `svm-pipe-imp.rs` | The anonymous-pipe module (copied to `sys/pipe/svm.rs`). `sys::pipe::Pipe` is a pair of fds over one in-personality byte FIFO (`OP_PIPE`), read/written through `OP_READ`/`OP_WRITE` and closed on drop — the plumbing `std::process` captures child stdout through. |
| `svm-process-imp.rs` | The process module (copied to `sys/process/svm.rs`). `std::process::Command` over the personality's **fork-free** spawn (`OP_SPAWN`/`OP_WAITPID`): a spawn runs the named command to completion synchronously, `output` captures stdout via an `OP_PIPE`+`OP_DUP2` fd-1 redirect (saved/restored around the spawn), and `wait`/`status` reap the exit code. The command is resolved by the embedder's spawn delegate (`Posix::set_spawn`); without one, spawning is `Unsupported`. Synchronous model ⇒ no live child to stream stdin into (stdin is whatever fd 0 holds) and stderr isn't separately captured. |
| `apply-overlay.sh` | Applies the overlay to the active nightly's `rust-src` (idempotent). |

## Why an overlay is needed at all (the S0 finding)

`restricted_std` alone does **not** work for a brand-new OS on modern std: the
PAL (`sys/pal/`) has a clean `_ => unsupported` fallback, but the *leaf* modules
(`sys/alloc`, `sys/thread_local`, `sys/random`, `sys/io/error`) enumerate
`target_os` with **no catch-all**, so an unknown `os=svm` fails to build with
missing-symbol errors. The overlay adds the five `svm` arms that make those
modules resolve. This is why S0 and S1 partly merge (RUST_STD.md §4-D): you
cannot get "std that builds but errors at runtime" for a novel OS without
supplying this much PAL wiring. Everything here still uses the **`unsupported`
PAL** for actual I/O — stdio/fs/args/exit all return errors until the real svm
PAL lands (S1).

The `singlethread=true` spec field (borrowed from the `zkvm` target) is what
selects the `no_threads` sync/TLS implementations — `Mutex` is a `Cell`, TLS is a
plain `static` — while still exposing atomics for future threading. Without it,
std sees `target_has_threads` and rejects the `no_threads` impls.

## Usage

```sh
rustup toolchain install nightly
rustup component add rust-src --toolchain nightly
./apply-overlay.sh                       # patch the toolchain's rust-src

RUSTC_BOOTSTRAP=1 cargo +nightly build \
  -Z build-std=core,alloc,std,panic_abort -Z json-target-spec \
  --target x86_64-unknown-svm.json --release
```

To get a single self-contained `.ll` for the on-ramp, build a `#![no_main]`
crate that exports a `#[no_mangle] pub extern "C"` entry, with `lto = "fat"` +
`codegen-units = 1`, and `--emit=llvm-ir`. The whole reachable std graph lands in
one module whose only undefined externals are `malloc`/`free`/`realloc` and the
`llvm.*` intrinsics the on-ramp already handles.

## Status (2026-08-11)

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
  **`std::fs`** (`File` open/read/write/seek, `metadata`/`read_dir`/`remove_file`/
  `exists`, via a granted posix cap), **`std::process`** (`Command` spawn/`output`/
  `status`/`wait`, via a granted posix cap + spawn delegate), heap/`Vec`,
  collections, `fmt`, iterators. (`std::env::var`/`vars` — the `str`-Debug paths —
  light up with the on-ramp entry-block slot-numbering fix, #755.)
- **The two paths:** the powerbox stream/exit handles carry stdio/exit/args (no
  extra grant); the **posix-cap path** (`run_with_caps` + a `posix` cap, reached
  via the PAL `host` bridge's `__vm_host_call`) carries `time`/`env`/`fs`/`process`
  — this is where the richer, many-op surface scales without growing the powerbox.
- **Not yet on the fs surface:** `mkdir`/`rename`/`symlink`/`set_permissions`/
  `set_times`/`canonicalize` (no host op on the memfs backend — they return
  `Unsupported`). Tracked in RUST_STD.md.
- **process caveats:** spawn is **fork-free and synchronous** (the child runs to
  completion inside `spawn`), so there is no live child to stream stdin into
  (`StdioPipes` yields no writable stdin), stderr isn't separately captured
  (`output().stderr` is empty), and `Command::spawn` returns an already-exited
  child. `fork`/`exec`-in-place stay parked. Tracked in RUST_STD.md.

## Reproducibility note

The overlay is pinned against a nightly `rust-src`; when the pin moves, re-apply
and, if the surrounding `cfg_select!` arms shifted, regenerate `std-overlay.patch`
(the arms are stable — they sit next to the long-lived `vexos`/`zkvm` arms). A CI
asset-lane check (RUST_STD.md S0, ISSUES.md I55) guards drift.
