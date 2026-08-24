#!/usr/bin/env bash
# Apply the temen `std` overlay to the active nightly toolchain's `rust-src`, so
# `-Zbuild-std` can build `std` for the `x86_64-unknown-temen` target (the `rust-temen` overlay; design in LLVM.md §10).
#
# The overlay is a set of `cfg_select!` arm additions that route `target_os = "temen"` to the temen
# leaf-module implementations, plus the copied `imp` files. Most arms are shared by both temen targets;
# the sync/TLS arms are gated on `target_env = "threads"` so the lean `x86_64-unknown-temen` target
# keeps std's `no_threads` `Mutex`/`Once`/TLS (single-threaded, `singlethread=true`), while the
# threaded `x86_64-unknown-temen-threads` target (`singlethread=false`) routes `sys/sync` to the
# `futex` impls (over the temen §12 futex, `temen-futex-imp.rs`) and TLS to the `native` impl.
#
# Idempotent: re-running is a no-op if the overlay is already present.
# Requires: a nightly toolchain with the `rust-src` component.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLCHAIN="${TEMEN_RUST_TOOLCHAIN:-nightly}"

SYSROOT="$(rustc "+${TOOLCHAIN}" --print sysroot)"
SRC="${SYSROOT}/lib/rustlib/src/rust"
STD="${SRC}/library/std/src"

if [[ ! -d "${STD}" ]]; then
  echo "error: rust-src not found at ${STD}" >&2
  echo "  run: rustup component add rust-src --toolchain ${TOOLCHAIN}" >&2
  exit 1
fi

# 1) The new PAL module files (copies are idempotent).
cp "${HERE}/temen-alloc-imp.rs" "${STD}/sys/alloc/temen.rs"
cp "${HERE}/temen-stdio-imp.rs" "${STD}/sys/stdio/temen.rs"
cp "${HERE}/temen-pal.rs" "${STD}/sys/pal/temen.rs"
cp "${HERE}/temen-args-imp.rs" "${STD}/sys/args/temen.rs"
cp "${HERE}/temen-time-imp.rs" "${STD}/sys/time/temen.rs"
cp "${HERE}/temen-env-imp.rs" "${STD}/sys/env/temen.rs"
cp "${HERE}/temen-fs-imp.rs" "${STD}/sys/fs/temen.rs"
cp "${HERE}/temen-pipe-imp.rs" "${STD}/sys/pipe/temen.rs"
cp "${HERE}/temen-process-imp.rs" "${STD}/sys/process/temen.rs"
cp "${HERE}/temen-net-imp.rs" "${STD}/sys/net/connection/temen.rs"
# The futex primitive backing `sys/sync` under the threaded target (`x86_64-unknown-temen-threads`,
# `target_env = "threads"`). Only referenced by that target; the lean target keeps std's `no_threads`.
cp "${HERE}/temen-futex-imp.rs" "${STD}/sys/sync/futex/temen.rs"
# The thread PAL (`std::thread::spawn`/join over the §12 thread ops) — threaded target only; the lean
# target keeps `sys/thread/unsupported.rs` (spawn fails closed).
cp "${HERE}/temen-thread-imp.rs" "${STD}/sys/thread/temen.rs"

# 2) The cfg-arm additions. Skip if already applied (patch is not idempotent on its own).
# The `sys/thread` temen arm is the marker for *this* overlay version; the `alloc` temen arm marks *any*
# overlay. A tree that has `alloc` but not the `sys/thread` arm carries a stale earlier overlay: the
# patch can't apply on top of it, so fail loudly with the fix rather than silently skipping (which
# would leave the new surface unbuildable).
if grep -q 'target_os = "temen"' "${STD}/sys/thread/mod.rs" 2>/dev/null; then
  echo "temen std overlay already applied to ${SRC}"
elif grep -q 'target_os = "temen"' "${STD}/sys/alloc/mod.rs"; then
  echo "error: a stale (older) temen std overlay is applied to ${SRC}" >&2
  echo "  reinstall a clean rust-src and re-run, e.g.:" >&2
  echo "    rustup component remove rust-src --toolchain ${TOOLCHAIN}" >&2
  echo "    rustup component add    rust-src --toolchain ${TOOLCHAIN}" >&2
  echo "    ${BASH_SOURCE[0]}" >&2
  exit 1
else
  patch -p1 -d "${SRC}" < "${HERE}/std-overlay.patch"
  echo "temen std overlay applied to ${SRC}"
fi

echo
echo "build a crate for a temen target with, e.g.:"
echo "  RUSTC_BOOTSTRAP=1 cargo +${TOOLCHAIN} build \\"
echo "    -Z build-std=core,alloc,std,panic_abort -Z json-target-spec \\"
echo "    --target ${HERE}/x86_64-unknown-temen.json --release          # lean, single-threaded"
echo "    --target ${HERE}/x86_64-unknown-temen-threads.json --release  # threaded (std::sync/TLS)"
