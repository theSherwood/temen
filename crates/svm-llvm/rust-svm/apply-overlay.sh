#!/usr/bin/env bash
# Apply the svm `std` overlay to the active nightly toolchain's `rust-src`, so
# `-Zbuild-std` can build `std` for the `x86_64-unknown-svm` target (RUST_STD.md, S0).
#
# The overlay is a set of `cfg_select!` arm additions that route `target_os = "svm"` to the svm
# leaf-module implementations, plus the copied `imp` files. Most arms are shared by both svm targets;
# the sync/TLS arms are gated on `target_env = "threads"` so the lean `x86_64-unknown-svm` target
# keeps std's `no_threads` `Mutex`/`Once`/TLS (single-threaded, `singlethread=true`), while the
# threaded `x86_64-unknown-svm-threads` target (`singlethread=false`) routes `sys/sync` to the
# `futex` impls (over the svm §12 futex, `svm-futex-imp.rs`) and TLS to the `native` impl.
#
# Idempotent: re-running is a no-op if the overlay is already present.
# Requires: a nightly toolchain with the `rust-src` component.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLCHAIN="${SVM_RUST_TOOLCHAIN:-nightly}"

SYSROOT="$(rustc "+${TOOLCHAIN}" --print sysroot)"
SRC="${SYSROOT}/lib/rustlib/src/rust"
STD="${SRC}/library/std/src"

if [[ ! -d "${STD}" ]]; then
  echo "error: rust-src not found at ${STD}" >&2
  echo "  run: rustup component add rust-src --toolchain ${TOOLCHAIN}" >&2
  exit 1
fi

# 1) The new PAL module files (copies are idempotent).
cp "${HERE}/svm-alloc-imp.rs" "${STD}/sys/alloc/svm.rs"
cp "${HERE}/svm-stdio-imp.rs" "${STD}/sys/stdio/svm.rs"
cp "${HERE}/svm-pal.rs" "${STD}/sys/pal/svm.rs"
cp "${HERE}/svm-args-imp.rs" "${STD}/sys/args/svm.rs"
cp "${HERE}/svm-time-imp.rs" "${STD}/sys/time/svm.rs"
cp "${HERE}/svm-env-imp.rs" "${STD}/sys/env/svm.rs"
cp "${HERE}/svm-fs-imp.rs" "${STD}/sys/fs/svm.rs"
cp "${HERE}/svm-pipe-imp.rs" "${STD}/sys/pipe/svm.rs"
cp "${HERE}/svm-process-imp.rs" "${STD}/sys/process/svm.rs"
cp "${HERE}/svm-net-imp.rs" "${STD}/sys/net/connection/svm.rs"
# The futex primitive backing `sys/sync` under the threaded target (`x86_64-unknown-svm-threads`,
# `target_env = "threads"`). Only referenced by that target; the lean target keeps std's `no_threads`.
cp "${HERE}/svm-futex-imp.rs" "${STD}/sys/sync/futex/svm.rs"

# 2) The cfg-arm additions. Skip if already applied (patch is not idempotent on its own).
# The `futex` svm arm is the marker for *this* (threaded) overlay version; the `alloc` svm arm marks
# *any* overlay. A tree that has `alloc` but not the `futex` arm carries a stale pre-threads overlay:
# the patch can't apply on top of it, so fail loudly with the fix rather than silently skipping (which
# would leave the threaded target unbuildable).
if grep -q 'target_os = "svm"' "${STD}/sys/sync/futex/mod.rs" 2>/dev/null; then
  echo "svm std overlay already applied to ${SRC}"
elif grep -q 'target_os = "svm"' "${STD}/sys/alloc/mod.rs"; then
  echo "error: a stale (pre-threads) svm std overlay is applied to ${SRC}" >&2
  echo "  reinstall a clean rust-src and re-run, e.g.:" >&2
  echo "    rustup component remove rust-src --toolchain ${TOOLCHAIN}" >&2
  echo "    rustup component add    rust-src --toolchain ${TOOLCHAIN}" >&2
  echo "    ${BASH_SOURCE[0]}" >&2
  exit 1
else
  patch -p1 -d "${SRC}" < "${HERE}/std-overlay.patch"
  echo "svm std overlay applied to ${SRC}"
fi

echo
echo "build a crate for a svm target with, e.g.:"
echo "  RUSTC_BOOTSTRAP=1 cargo +${TOOLCHAIN} build \\"
echo "    -Z build-std=core,alloc,std,panic_abort -Z json-target-spec \\"
echo "    --target ${HERE}/x86_64-unknown-svm.json --release          # lean, single-threaded"
echo "    --target ${HERE}/x86_64-unknown-svm-threads.json --release  # threaded (std::sync/TLS)"
