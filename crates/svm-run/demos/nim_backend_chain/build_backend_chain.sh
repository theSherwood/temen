#!/usr/bin/env bash
# The nimony BACK-END CHAIN on SVM (NIM.md §3c/§3e, "compile Nim in the browser" slice 3): drive two
# REAL nimony phase guests in sequence, host-orchestrated, and prove the whole chain is byte-identical
# to native at every hop, on all three engines.
#
#   semchecked .s.nif ──hexer.svmb──▶ Leng .x.nif ──svm-leng.svmb──▶ SVM IR text
#
# This is the exact shape the browser card (slice 4) uses: the host (Rust here, JS there) runs each
# phase's committed `.svmb` and pipes between them — hexer reads the `.s.nif` from an in-window memfs
# and writes Leng; svm-leng reads that Leng on stdin and emits SVM IR. Both guests are real (slice 2's
# `hexer.svmb` + the committed W5 `svm-leng.svmb`); the driver is `examples/nim_backend_chain.rs`.
#
# The `.s.nif` input is what `nimsem` (the sema phase) produces. nimsem-on-SVM is currently blocked
# UPSTREAM of svm — a fresh stock-nim build of nimsem fails the `m` command natively while the shipped
# binary succeeds (the svm on-ramp is proven innocent: svm nimsem == native stock-nim nimsem byte for
# byte). So here the `.s.nif` is produced by the oracle toolchain; once nimsem-on-SVM is unblocked it
# slots in ahead of hexer as a third guest, closing the full Nim→run chain.
#
#   needs: the nimony submodule, stock nim (2.3.x), the built nimony + hexer binaries, clang-18/llvm-18,
#          cargo. Fail-soft SKIP if any is absent (NIM.md §2). hexer.svmb is a build artifact (~3 MB,
#          not committed); svm-leng.svmb is the committed asset. This script is the gate.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
HEXER_CACHE="${SVM_HEXER_CACHE:-/tmp/svm_hexer_build}"
CHAIN_CACHE="${SVM_CHAIN_CACHE:-/tmp/svm_chain_build}"
LENG_SVMB="$REPO/crates/svm-run/demos/leng_selfhost/svm-leng.svmb"
mkdir -p "$CHAIN_CACHE"

if ! command -v nim >/dev/null; then echo "SKIP: nim not on PATH"; exit 0; fi
[ -f "$REPO/nimony/src/hexer/hexer.nim" ] || { echo "SKIP: nimony submodule absent"; exit 0; }
BIN="${NIMONY_TOOLCHAIN_BIN:-$REPO/.nimtool/nimony/bin}"
NIMONY="${NIMONY_BIN:-$(command -v nimony || echo "$BIN/nimony")}"
HEXER_BIN="${HEXER_BIN:-$(command -v hexer || echo "$BIN/hexer")}"
[ -x "$NIMONY" ] && [ -x "$HEXER_BIN" ] || { echo "SKIP: nimony/hexer binaries absent (NIM.md §2)"; exit 0; }
[ -f "$LENG_SVMB" ] || { echo "cannot find committed svm-leng.svmb ($LENG_SVMB)"; exit 1; }
export NIMONY_BIN NIMONY HEXER_BIN
export PATH="$(dirname "$NIMONY"):$(dirname "$HEXER_BIN"):$PATH"

echo "=== [1/3] build hexer.svmb (via build_hexer_svmb.sh — also validates hexer-on-SVM) ==="
SVM_HEXER_CACHE="$HEXER_CACHE" bash "$REPO/crates/svm-run/demos/hexer_svm/build_hexer_svmb.sh" | tail -1
HEXER_SVMB="$HEXER_CACHE/hexer.svmb"
[ -f "$HEXER_SVMB" ] || { echo "hexer.svmb not produced"; exit 1; }

echo "=== [2/3] generate the semchecked .s.nif fixture (oracle nimony c → nimcache) ==="
fail=0
for src in "$HERE"/inputs/*.nim; do
  name="$(basename "$src")"
  work="$CHAIN_CACHE/work_$name"; rm -rf "$work"; mkdir -p "$work"; cp "$src" "$work/prog.nim"
  ( cd "$work" && "$NIMONY" c --isMain prog.nim >/dev/null 2>&1 )
  fixture="$CHAIN_CACHE/fix_$name"; rm -rf "$fixture"; mkdir -p "$fixture"
  cp "$work"/nimcache/*.s.nif "$work"/nimcache/*.s.idx.nif "$fixture/" 2>/dev/null || true
  main="$(ls "$fixture"/*.s.nif | xargs -n1 basename | grep -v '^sys' | sed 's/\.s\.nif//' | head -1)"

  echo "=== [3/3] $name: chain hexer.svmb → svm-leng.svmb, diff every hop vs native ==="
  set +e
  HEXER_BIN="$HEXER_BIN" cargo run -q --release -p svm-run --example nim_backend_chain -- \
    "$HEXER_SVMB" "$LENG_SVMB" "$fixture" "$main" 2>"$CHAIN_CACHE/err_$name.txt"
  rc=$?
  set -e
  if [ "$rc" = 0 ]; then echo "  $name: chain OK"; else echo "  $name: FAILED"; sed 's/^/    /' "$CHAIN_CACHE/err_$name.txt" | tail -5; fail=1; fi
done
[ "$fail" = 0 ] && echo "ALL CHAINS MATCH NATIVE — real hexer + svm-leng guests chained on the SVM, byte-exact" || { echo "FAILED"; exit 1; }
