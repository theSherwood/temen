#!/usr/bin/env bash
# Apply the svm `std` overlay to the active nightly toolchain's `rust-src`, so
# `-Zbuild-std` can build `std` for the `x86_64-unknown-svm` target (RUST_STD.md, S0).
#
# The overlay is deliberately tiny: five one-line `cfg_select!` arm additions that
# route `target_os = "svm"` to the minimal (no-OS / single-thread) leaf-module
# implementations already used by `vexos`/`zkvm`, plus one small allocator `imp`
# (`svm-alloc-imp.rs`) that forwards to the C `malloc` family (which the svm-llvm
# on-ramp synthesizes as an in-window guest bump allocator, LLVM.md slice S).
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

# 2) The cfg-arm additions. Skip if already applied (patch is not idempotent on its own).
if grep -q 'target_os = "svm"' "${STD}/sys/alloc/mod.rs"; then
  echo "svm std overlay already applied to ${SRC}"
else
  patch -p1 -d "${SRC}" < "${HERE}/std-overlay.patch"
  echo "svm std overlay applied to ${SRC}"
fi

echo
echo "build a crate for the svm target with, e.g.:"
echo "  RUSTC_BOOTSTRAP=1 cargo +${TOOLCHAIN} build \\"
echo "    -Z build-std=core,alloc,std,panic_abort -Z json-target-spec \\"
echo "    --target ${HERE}/x86_64-unknown-svm.json --release"
