#!/usr/bin/env bash
# The nim->powerbox **link guest** bitcode pipeline (#1025 3c, link+run) — the leng-self-host asset-lane
# pattern applied to `temen_leng::link_nim_powerbox`: build-std -> llvm-link -> temen-llvm-translate ->
# prep_temen. Produces the committed `fixtures/nim-link-fs.temen.gz`: the real nim->powerbox linker, run
# in-sandbox, emitting the encoded linked Temen module byte-identical to native.
#
# Pins **rustc 1.81** (LLVM 18, matching llvm-link-18/opt-18 and the translator's LLVM): a newer rustc
# emits LLVM 19+ IR the 18 tools can't parse, and moved `panic_immediate_abort` from a build-std feature
# to a `-Cpanic=` strategy. `rustup toolchain install 1.81.0 --component rust-src` if absent.
#
#   needs: rustc +1.81.0 (+ rust-src), llvm-link-18, opt-18, cargo. env: NIM_LINK_FS_CACHE (default /tmp/nim_link_fs_cache)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
GUEST="$HERE/nim_link_fs_guest"
CACHE="${NIM_LINK_FS_CACHE:-/tmp/nim_link_fs_cache}"
TRIPLE="x86_64-unknown-linux-gnu"
RUSTC_VER="${NIM_LINK_FS_RUSTC:-1.81.0}"
LINK="${LLVM_LINK:-llvm-link-18}"
OPT="${LLVM_OPT:-opt-18}"
mkdir -p "$CACHE"

command -v "rustc" >/dev/null && rustc "+$RUSTC_VER" --version >/dev/null 2>&1 || {
  echo "SKIP: rustc +$RUSTC_VER absent (rustup toolchain install $RUSTC_VER --component rust-src)"; exit 0; }

echo "[1/4] build-std (rustc +$RUSTC_VER, LLVM 18) ..."
( cd "$GUEST" && RUSTFLAGS='--emit=llvm-ir' CARGO_TARGET_DIR="$CACHE/target" RUSTC_BOOTSTRAP=1 \
    cargo "+$RUSTC_VER" build --release \
      -Zbuild-std=std,panic_abort -Zbuild-std-features=panic_immediate_abort \
      --target "$TRIPLE" --ignore-rust-version )

DEPS="$CACHE/target/$TRIPLE/release/deps"
mapfile -t LLS < <(ls "$DEPS"/*.ll | grep -v '/panic_unwind')
[ "${#LLS[@]}" -gt 0 ] || { echo "no .ll emitted — build failed before codegen" >&2; exit 1; }

echo "[2/4] llvm-link + opt (internalize,globaldce) ..."
"$LINK" -S "${LLS[@]}" -o "$CACHE/nl.linked.ll"
"$OPT" -S -passes=internalize,globaldce -internalize-public-api-list=main,malloc,free \
  "$CACHE/nl.linked.ll" -o "$CACHE/nl.legal.ll"

echo "[2a/4] stub audit ..."
ONRAMP='^(read|write|bcmp|memcmp|memcpy|memmove|memset|malloc|calloc|realloc|free|__vm_[a-z_0-9]+|__temen_[a-z_0-9]+)$'
UNRESOLVED=0
while read -r sym; do
  [ -z "$sym" ] && continue
  case "$sym" in llvm.*) continue;; esac
  if ! [[ "$sym" =~ $ONRAMP ]]; then echo "  UNHANDLED extern: $sym" >&2; UNRESOLVED=1; fi
done < <(grep -E '^declare ' "$CACHE/nl.legal.ll" | grep -oE '@"?[A-Za-z0-9_.$]+' | tr -d '@"' | sort -u)
[ "$UNRESOLVED" -eq 0 ] || { echo "stub audit failed: unhandled externs above (a new libc dep in the link closure?)" >&2; exit 1; }

echo "[3/4] temen-llvm-translate --binary ..."
TR="$REPO/crates/temen-llvm/target/release/temen-llvm-translate"
[ -x "$TR" ] || cargo build --release --bin temen-llvm-translate --manifest-path "$REPO/crates/temen-llvm/Cargo.toml"
"$TR" "$CACHE/nl.legal.ll" -o "$CACHE/nl_raw.temen" --binary --host-page 65536 --null-guard --child-entry

# A **child-entry** module has no top-level `_start` (it is op-13-spawned, decode+verify+op-13 is the
# child path), so it deliberately fails prep_temen's top-level gate — gzip the raw translated `.temen`
# directly, exactly like the `nimsem_ce`/`hexer_ce` child-entry assets. The gate test
# (`nim_link_fs_asset.rs`) decode+verify+op-13-runs it and diffs vs native `link_nim_powerbox`.
echo "[4/4] gzip the committed child-entry asset (validation is the op-13 gate test) ..."
gzip -9 -c "$CACHE/nl_raw.temen" > "$HERE/fixtures/nim-link-fs.temen.gz"
echo "done: $HERE/fixtures/nim-link-fs.temen.gz ($(du -h "$HERE/fixtures/nim-link-fs.temen.gz" | cut -f1))"
