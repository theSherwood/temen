#!/usr/bin/env bash
# Run cargo for an Android target, with the NDK as its C compiler and linker (#2514).
#
#   scripts/ci/android.sh <target> <cargo subcommand> [args…]
#     e.g. scripts/ci/android.sh aarch64-linux-android clippy --workspace --all-targets -- -D warnings
#          scripts/ci/android.sh x86_64-linux-android nextest run --workspace
#   scripts/ci/android.sh --ndk      # print the NDK it would use, or fail if there is none
#
# WHY a script: the target needs four settings that must agree — the C compiler `temen-jit`'s build
# script compiles `trap_capture.c` with, the archiver, the linker, and the API level all three
# target — and the local gate and CI must not each carry their own copy of them.
#
# The NDK is found at `$ANDROID_NDK_HOME`, `$ANDROID_NDK_LATEST_HOME` or `$ANDROID_NDK_ROOT`, in that
# order; GitHub's ubuntu image sets the last two, so CI fetches nothing. API level 30 is the lowest
# with `memfd_create`, which shared regions are made from (Bionic has no `shm_open`).
#
# Test binaries (`test`, `nextest`) link statically (`crt-static`), so they need no Android
# userspace: x86-64 ones run natively on a Linux host, and aarch64 ones under qemu-user
# (`CARGO_TARGET_AARCH64_LINUX_ANDROID_RUNNER=qemu-aarch64-static`). They exercise Bionic and
# temen's Android code paths, not an Android device: ART's signal chaining and the app sandbox stay
# untested here (#2514). Every other subcommand links as an embedder does, dynamically, so a
# `cdylib` such as `temen-capi` is still built.
#
# `TEMEN_CHECK_TOOLCHAIN` pins the toolchain, as for `check-excluded.sh`; unset means the default.
set -euo pipefail

ndk=""
for candidate in "${ANDROID_NDK_HOME:-}" "${ANDROID_NDK_LATEST_HOME:-}" "${ANDROID_NDK_ROOT:-}"; do
  if [ -n "$candidate" ] && [ -d "$candidate/toolchains/llvm/prebuilt/linux-x86_64/bin" ]; then
    ndk="$candidate"
    break
  fi
done
if [ -z "$ndk" ]; then
  echo "android: no NDK — set ANDROID_NDK_HOME to an NDK (r27 or later) with a linux-x86_64 toolchain" >&2
  exit 1
fi
if [ "${1:-}" = "--ndk" ]; then
  echo "$ndk"
  exit 0
fi

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <target> <cargo subcommand> [args…]  |  $0 --ndk" >&2
  exit 2
fi
target="$1"
subcommand="$2"
shift 2

api=30
bin="$ndk/toolchains/llvm/prebuilt/linux-x86_64/bin"
clang="$bin/${target}${api}-clang"
if [ ! -x "$clang" ]; then
  echo "android: $ndk has no $target clang at API $api ($clang)" >&2
  exit 1
fi

triple="$(echo "$target" | tr '[:lower:]-' '[:upper:]_')"
export "CC_${target//-/_}=$clang"
export "AR_${target//-/_}=$bin/llvm-ar"
export "CARGO_TARGET_${triple}_LINKER=$clang"
# RUSTFLAGS, not the per-target variable: CI sets RUSTFLAGS ("-D warnings") for the whole workflow,
# and cargo ignores a target's own flags whenever RUSTFLAGS is set. With `--target` given, RUSTFLAGS
# reaches only the target's artifacts, never the host's build scripts and proc-macros.
# A static link makes the NDK's clang note that it ignores rustc's `-no-pie`, which every binary would
# then report as a `linker_messages` warning; the second flag silences that one note.
case "$subcommand" in
  test | nextest)
    export RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+crt-static -C link-arg=-Wno-unused-command-line-argument"
    ;;
esac

cargo_cmd=(cargo)
if [ -n "${TEMEN_CHECK_TOOLCHAIN:-}" ]; then
  cargo_cmd=(rustup run "$TEMEN_CHECK_TOOLCHAIN" cargo)
fi

# `nextest run` is a two-word subcommand; `--target` goes after it.
if [ "$subcommand" = "nextest" ]; then
  subcommand="nextest $1"
  shift
fi
echo "android: \$ ${cargo_cmd[*]} $subcommand --target $target $*  (NDK: $ndk, API $api)"
# shellcheck disable=SC2086 # `$subcommand` is one or two words on purpose.
exec "${cargo_cmd[@]}" $subcommand --target "$target" "$@"
