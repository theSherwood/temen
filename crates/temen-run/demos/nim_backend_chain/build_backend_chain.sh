#!/usr/bin/env bash
# The nimony BACK-END CHAIN on Temen (NIM.md §3c/§3e, "compile Nim in the browser" slice 3): drive two
# REAL nimony phase guests in sequence, host-orchestrated, and prove the whole chain is byte-identical
# to native at every hop, on all three engines.
#
#   semchecked .s.nif ──hexer.temen──▶ Leng .x.nif ──temen-leng.temen──▶ Temen IR text
#
# This is the exact shape the browser card (slice 4) uses: the host (Rust here, JS there) runs each
# phase's committed `.temen` and pipes between them — hexer reads the `.s.nif` from an in-window memfs
# and writes Leng; temen-leng reads that Leng on stdin and emits Temen IR. Both guests are real (slice 2's
# `hexer.temen` + the committed W5 `temen-leng.temen`); the driver is `examples/nim_backend_chain.rs`.
#
# The `.s.nif` input is what `nimsem` (the sema phase) produces. nimsem-on-Temen is currently blocked
# UPSTREAM of temen — a fresh stock-nim build of nimsem fails the `m` command natively while the shipped
# binary succeeds (the temen on-ramp is proven innocent: temen nimsem == native stock-nim nimsem byte for
# byte). So here the `.s.nif` is produced by the oracle toolchain; once nimsem-on-Temen is unblocked it
# slots in ahead of hexer as a third guest, closing the full Nim→run chain.
#
#   needs: the nimony submodule, stock nim (2.3.x), the built nimony + hexer binaries, clang-18/llvm-18,
#          cargo. Fail-soft SKIP if any is absent (NIM.md §2). hexer.temen is a build artifact (~3 MB,
#          not committed); temen-leng.temen is the committed asset. This script is the gate.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
HEXER_CACHE="${TEMEN_HEXER_CACHE:-/tmp/temen_hexer_build}"
CHAIN_CACHE="${TEMEN_CHAIN_CACHE:-/tmp/temen_chain_build}"
LENG_TEMEN="$REPO/crates/temen-run/demos/leng_selfhost/temen-leng.temen"
mkdir -p "$CHAIN_CACHE"

if ! command -v nim >/dev/null; then echo "SKIP: nim not on PATH"; exit 0; fi
[ -f "$REPO/nimony/src/hexer/hexer.nim" ] || { echo "SKIP: nimony submodule absent"; exit 0; }
BIN="${NIMONY_TOOLCHAIN_BIN:-$REPO/.nimtool/nimony/bin}"
NIMONY="${NIMONY_BIN:-$(command -v nimony || echo "$BIN/nimony")}"
HEXER_BIN="${HEXER_BIN:-$(command -v hexer || echo "$BIN/hexer")}"
[ -x "$NIMONY" ] && [ -x "$HEXER_BIN" ] || { echo "SKIP: nimony/hexer binaries absent (NIM.md §2)"; exit 0; }
[ -f "$LENG_TEMEN" ] || { echo "cannot find committed temen-leng.temen ($LENG_TEMEN)"; exit 1; }
export NIMONY_BIN NIMONY HEXER_BIN
export PATH="$(dirname "$NIMONY"):$(dirname "$HEXER_BIN"):$PATH"

echo "=== [1/3] build hexer.temen (via build_hexer_temen.sh — also validates hexer-on-Temen) ==="
TEMEN_HEXER_CACHE="$HEXER_CACHE" bash "$REPO/crates/temen-run/demos/hexer_temen/build_hexer_temen.sh" | tail -1
HEXER_TEMEN="$HEXER_CACHE/hexer.temen"
[ -f "$HEXER_TEMEN" ] || { echo "hexer.temen not produced"; exit 1; }

echo "=== [2/3] generate the semchecked .s.nif fixture (oracle nimony c → nimcache) ==="
fail=0
for src in "$HERE"/inputs/*.nim; do
  name="$(basename "$src")"
  work="$CHAIN_CACHE/work_$name"; rm -rf "$work"; mkdir -p "$work"; cp "$src" "$work/prog.nim"
  ( cd "$work" && "$NIMONY" c --isMain prog.nim >/dev/null 2>&1 )
  fixture="$CHAIN_CACHE/fix_$name"; rm -rf "$fixture"; mkdir -p "$fixture"
  cp "$work"/nimcache/*.s.nif "$work"/nimcache/*.s.idx.nif "$fixture/" 2>/dev/null || true
  main="$(ls "$fixture"/*.s.nif | xargs -n1 basename | grep -v '^sys' | sed 's/\.s\.nif//' | head -1)"

  echo "=== [3/3] $name: chain hexer.temen → temen-leng.temen, diff every hop vs native ==="
  set +e
  HEXER_BIN="$HEXER_BIN" cargo run -q --release -p temen-run --example nim_backend_chain -- \
    "$HEXER_TEMEN" "$LENG_TEMEN" "$fixture" "$main" 2>"$CHAIN_CACHE/err_$name.txt"
  rc=$?
  set -e
  if [ "$rc" = 0 ]; then echo "  $name: chain OK"; else echo "  $name: FAILED"; sed 's/^/    /' "$CHAIN_CACHE/err_$name.txt" | tail -5; fail=1; fi
done
[ "$fail" = 0 ] && echo "ALL CHAINS MATCH NATIVE — real hexer + temen-leng guests chained on the Temen, byte-exact" || { echo "FAILED"; exit 1; }
