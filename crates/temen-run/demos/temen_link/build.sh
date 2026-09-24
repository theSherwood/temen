#!/usr/bin/env bash
# Build `temen-link` (the in-guest linker nimony's Temen backend plans, see guest/src/lib.rs) as a Temen
# POSIX program: build-std -> llvm-link -> opt -> temen-llvm-translate. The self-hosted lane
# (scripts/ci/nim-selfhost-lane.sh) builds it from source every run, so it is never stale against
# temen-leng; the pipeline is the leng-self-host one (build_leng_temen.sh), minus the committed asset.
#
#   bash crates/temen-run/demos/temen_link/build.sh <out.temen>
#
#   needs: rustc (+ rust-src), llvm-link, opt (scripts/ci/install-llvm.sh). env: TEMEN_LINK_CACHE
#   (default /tmp/temen_link_cache).
set -euo pipefail
OUT="${1:?usage: build.sh <out.temen>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
CACHE="${TEMEN_LINK_CACHE:-/tmp/temen_link_cache}"
TRIPLE="x86_64-unknown-linux-gnu"
mkdir -p "$CACHE"

( cd "$HERE/guest" && RUSTFLAGS='--emit=llvm-ir -Zunstable-options -Cpanic=immediate-abort' \
    CARGO_TARGET_DIR="$CACHE/target" RUSTC_BOOTSTRAP=1 \
    cargo build -q --release -Zbuild-std=std,panic_abort --target "$TRIPLE" --ignore-rust-version )
DEPS="$CACHE/target/$TRIPLE/release/deps"
mapfile -t LLS < <(ls "$DEPS"/*.ll | grep -v '/panic_unwind')
[ "${#LLS[@]}" -gt 0 ] || { echo "no .ll emitted — the build failed before codegen" >&2; exit 1; }

"${LLVM_LINK:-llvm-link}" -S "${LLS[@]}" -o "$CACHE/tl.linked.ll"
"${LLVM_OPT:-opt}" -S -passes=internalize,globaldce -internalize-public-api-list=main,malloc,free \
  "$CACHE/tl.linked.ll" -o "$CACHE/tl.legal.ll"

# Every extern left must be one the on-ramp lowers: libc memory/string builtins, the on-ramp's own
# `__vm_*`/`__temen_*`, and the POSIX personality's `__px_*` vocabulary (its only way out).
ONRAMP='^(read|write|bcmp|memcmp|memcpy|memmove|memset|strlen|malloc|calloc|realloc|free|__vm_[a-z_0-9]+|__temen_[a-z_0-9]+|__px_[a-z_0-9]+)$'
BAD=0
while read -r sym; do
  [ -z "$sym" ] && continue
  case "$sym" in llvm.*|*rust_no_alloc_shim_is_unstable*) continue;; esac
  if ! [[ "$sym" =~ $ONRAMP ]]; then echo "  unhandled extern: $sym" >&2; BAD=1; fi
done < <(grep -E '^declare ' "$CACHE/tl.legal.ll" | grep -oE '@"?[A-Za-z0-9_.$]+' | tr -d '@"' | sort -u)
[ "$BAD" -eq 0 ] || { echo "temen-link: an extern outside the on-ramp (a new libc dependency?)" >&2; exit 1; }

TR="$REPO/crates/temen-llvm/target/release/temen-llvm-translate"
cargo build -q --release --bin temen-llvm-translate --manifest-path "$REPO/crates/temen-llvm/Cargo.toml"
"$TR" "$CACHE/tl.legal.ll" -o "$OUT" --binary --host-page 65536 --null-guard
echo "temen-link: $OUT ($(du -h "$OUT" | cut -f1))"
