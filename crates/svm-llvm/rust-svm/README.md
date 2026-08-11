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
| `std-overlay.patch` | `cfg_select!` arm additions routing `target_os="svm"` to the right leaf-module impls: the minimal (no-OS, single-thread) ones for `sys/{alloc,thread_local,random,io/error}` (as `vexos`/`zkvm` do), the new svm `stdio` module, and the powerbox `exit` in `sys/exit.rs`. 26 added lines across 6 files. |
| `svm-alloc-imp.rs` | The allocator `imp` (copied to `sys/alloc/svm.rs`). Forwards `alloc`/`dealloc`/`realloc` to the C `malloc` family, which the on-ramp synthesizes as an in-window guest bump allocator (LLVM.md slice S). |
| `svm-stdio-imp.rs` | The stdio PAL (copied to `sys/stdio/svm.rs`). `Stdin`/`Stdout`/`Stderr` reach the host through `extern "C" write`/`read`, which the on-ramp's "Lane C" binds to the powerbox `Stream` handles (POSIX.md ops 0/1). So `println!` writes real bytes. |
| `svm-pal.rs` | The svm PAL proper (copied to `sys/pal/svm.rs`). Mirrors the `unsupported` PAL, but its `init` captures the powerbox-threaded `argv` (calls `sys::args::init`) so `std::env::args` works. |
| `svm-args-imp.rs` | The args module (copied to `sys/args/svm.rs`). Stores `(argc, argv)` at init and walks them as C strings on demand — the "stored at startup" half of the unix strategy, no `os::unix` dependency. |
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
- **Working today:** stdout (`println!`), `process::exit`, **`env::args`**,
  heap/`Vec`, collections, `fmt`, iterators.
- **Not yet:** `stderr` as a distinct stream (currently merges into stdout —
  the powerbox grants no stderr handle and the on-ramp drops the `fd`; needs a
  powerbox grant-surface change, S1e), `File`/`fs`, `env`, `time`. Tracked in
  RUST_STD.md (S1e/S2+).

## Reproducibility note

The overlay is pinned against a nightly `rust-src`; when the pin moves, re-apply
and, if the surrounding `cfg_select!` arms shifted, regenerate `std-overlay.patch`
(the arms are stable — they sit next to the long-lived `vexos`/`zkvm` arms). A CI
asset-lane check (RUST_STD.md S0, ISSUES.md I55) guards drift.
