#!/usr/bin/env bash
# NIM.md Phase 1, step 1 — build the codegen-shape probe through the LLVM on-ramp and diff the
# SVM run (interp + JIT) against a native build. Mirrors demos/chibicc_selfhost/build_chibicc_svmb.sh
# (the build-pg-assets.mjs pattern): clang -O2 -emit-llvm → svm-llvm-translate → prep_svmb gate,
# then run on both engines via the generic example runner and byte-compare stdout+exit to native.
#
#   needs: clang-18 (or clang ≥18), cargo
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
CACHE="${SVM_NIMONY_CACHE:-/tmp/svm_nimony_cache}"
CLANG="${CLANG:-clang-18}"; command -v "$CLANG" >/dev/null || CLANG=clang
mkdir -p "$CACHE"
SRC="$HERE/arc_probe.c"

echo "=== [1/5] native reference build ==="
"$CLANG" -O2 "$SRC" -o "$CACHE/probe_native"
"$CACHE/probe_native" > "$CACHE/native.out"; NATIVE_EXIT=$?
echo "native exit=$NATIVE_EXIT"; sed 's/^/  native| /' "$CACHE/native.out"

echo "=== [2/5] bitcode (clang -O2 -emit-llvm) ==="
"$CLANG" -O2 -emit-llvm -c "$SRC" -o "$CACHE/probe.bc"

echo "=== [3/5] translate (svm-llvm-translate → SVM IR) ==="
TR="$REPO/crates/svm-llvm/target/release/svm-llvm-translate"
if [ ! -x "$TR" ]; then
  cargo build --release --bin svm-llvm-translate --manifest-path "$REPO/crates/svm-llvm/Cargo.toml"
fi
# host-page 65536 keeps the artifact browser-targetable (D40), matching the chibicc asset. No
# --stub-externs: every call must resolve to an on-ramp-recognized name (printf/malloc/realloc/
# free/memcpy/memset + the powerbox bridge) — if translation fails on an unresolved symbol that is
# a real finding (a libc gap to fill, NIM.md §2 step 3), not something to paper over.
"$TR" "$CACHE/probe.bc" -o "$CACHE/probe_raw.svmb" --binary --host-page 65536

echo "=== [4/5] prep_svmb (decode → verify → bytecode-compile gate) ==="
cargo run -q --release -p svm-run --example prep_svmb -- "$CACHE/probe_raw.svmb" "$CACHE/probe.svmb"

echo "=== [5/5] run on SVM (treewalk, bytecode, jit) and diff vs native ==="
fail=0
for eng in treewalk bytecode jit; do
  SVM_PROBE_BACKEND="$eng" cargo run -q --release -p svm-run --example nimony_probe -- \
    "$CACHE/probe.svmb" > "$CACHE/svm_$eng.out"; SVM_EXIT=$?
  if diff -q "$CACHE/native.out" "$CACHE/svm_$eng.out" >/dev/null && [ "$SVM_EXIT" = "$NATIVE_EXIT" ]; then
    echo "  $eng: OK (stdout byte-identical, exit=$SVM_EXIT)"
  else
    echo "  $eng: MISMATCH (exit svm=$SVM_EXIT native=$NATIVE_EXIT)"
    diff "$CACHE/native.out" "$CACHE/svm_$eng.out" | sed 's/^/    /' || true
    fail=1
  fi
done
[ "$fail" = 0 ] && echo "ALL ENGINES MATCH NATIVE" || { echo "PROBE FAILED"; exit 1; }
