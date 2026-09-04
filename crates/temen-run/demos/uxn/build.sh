#!/bin/sh
# Build the Uxn playground assets: assemble the demo ROM with the in-tree uxnasm, then compile the
# reactor guest (main.c → clang → temen-llvm-translate) exactly like build-onramp-assets.mjs does for
# the other display demos. Outputs into $OUT (default /tmp/temen_uxn_cache):
#   $OUT/uxn_demo.rom   the assembled demo.tal
#   $OUT/uxn.temen      the reactor guest (opens the ROM served as "boot.rom" through `fs`)
# Prereqs: cc, clang, the translator (`cargo build --release --bin temen-llvm-translate` in
# crates/temen-llvm). Usage: sh build.sh [OUT]   — invoked by `ONLY=uxn bash scripts/rebuild-assets.sh`.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
OUT="${1:-/tmp/temen_uxn_cache}"
TR="$REPO/crates/temen-llvm/target/release/temen-llvm-translate"
test -x "$TR" || { echo "missing $TR — build it first"; exit 1; }
mkdir -p "$OUT"
cc -O2 -o "$OUT/uxnasm" "$HERE/uxnasm.c"
"$OUT/uxnasm" "$HERE/demo.tal" "$OUT/uxn_demo.rom"
clang -O2 -emit-llvm -c -fno-vectorize -fno-slp-vectorize "$HERE/main.c" -o "$OUT/uxn.bc"
"$TR" "$OUT/uxn.bc" -o "$OUT/uxn.temen" --host-page 65536 --null-guard
echo "built $OUT/uxn.temen ($(wc -c < "$OUT/uxn.temen") bytes) + $OUT/uxn_demo.rom ($(wc -c < "$OUT/uxn_demo.rom") bytes)"
