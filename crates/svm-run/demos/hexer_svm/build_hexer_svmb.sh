#!/usr/bin/env bash
# hexer-on-SVM (NIM.md §3c/§3e, "nimony in the browser" slice 2) — compile nimony's MIDDLE-END phase,
# `hexer` (semchecked NIF → Leng, ~18k loc), to a runnable SVM module and prove it lowers a real
# semchecked module to Leng **byte-for-byte identically to native hexer**, across all three engines.
#
# Same C on-ramp as nifler (slice 1), same shared bottom-edge shim (../nifler_svm/nifler_shim.c —
# which now also serves hexer's `std/memfiles` NIF reader via a malloc+read `mmap`). The one new
# wrinkle vs nifler is the fixture: hexer's INPUT is a semchecked `.s.nif`, so we first run the real
# `nimony c` on a Nim program to fill a nimcache, then feed the resulting `.s.nif` (+ its index + the
# `system` module's) to both native hexer and hexer-on-SVM and diff every file each produces.
#
#   needs: the nimony submodule, stock `nim` (2.3.x, C backend), the built `nimony` + `hexer` binaries
#          (oracle + fixture generator), clang-18/llvm-18, cargo. Fail-soft SKIP if any is absent
#          (nimony bootstrap needs Nim 2.3.x devel — NOT in the per-PR CI, NIM.md §2).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
CACHE="${SVM_HEXER_CACHE:-/tmp/svm_hexer_build}"
CLANG="${CLANG:-clang-18}"; command -v "$CLANG" >/dev/null || CLANG=clang
LINK="${LLVM_LINK:-llvm-link-18}"; command -v "$LINK" >/dev/null || LINK=llvm-link
NM="${LLVM_NM:-llvm-nm-18}"; command -v "$NM" >/dev/null || NM=llvm-nm
HEXER_SRC="$REPO/nimony/src/hexer/hexer.nim"
SHIM="$REPO/crates/svm-run/demos/nifler_svm/nifler_shim.c"   # the shared nimony-phase bottom edge
mkdir -p "$CACHE"

if ! command -v nim >/dev/null; then echo "SKIP: nim not on PATH"; exit 0; fi
[ -f "$HEXER_SRC" ] || { echo "SKIP: nimony submodule absent ($HEXER_SRC)"; exit 0; }
BIN="${NIMONY_TOOLCHAIN_BIN:-$REPO/.nimtool/nimony/bin}"
NIMONY="${NIMONY_BIN:-$(command -v nimony || echo "$BIN/nimony")}"
HEXER_BIN="${HEXER_BIN:-$(command -v hexer || echo "$BIN/hexer")}"
[ -x "$NIMONY" ] && [ -x "$HEXER_BIN" ] || { echo "SKIP: nimony/hexer binaries absent (set NIMONY_BIN/HEXER_BIN; NIM.md §2)"; exit 0; }
NIMLIB="$(nim dump 2>/dev/null | grep -m1 '/lib$' || true)"
[ -f "$NIMLIB/nimbase.h" ] || NIMLIB="$(dirname "$(command -v nim)")/../lib"
[ -f "$NIMLIB/nimbase.h" ] || { echo "cannot locate nimbase.h"; exit 1; }
export PATH="$(dirname "$NIMONY"):$(dirname "$HEXER_BIN"):$PATH"
echo "nim: $(command -v nim) | nimony: $NIMONY | hexer(oracle): $HEXER_BIN"

echo "=== [1/6] hexer.nim -> C (stock nim, ARC, strictness relaxed to the toolchain's) ==="
rm -rf "$CACHE/c"
nim c --mm:arc -d:useMalloc -d:danger --panics:on --threads:off -d:noSignalHandler \
  --warningAsError:ProveInit:off --warningAsError:Uninit:off \
  --compileOnly --nimcache:"$CACHE/c" --path:"$REPO/nimony/src" -o:"$CACHE/hexer_stub" \
  "$HEXER_SRC" >/dev/null 2>&1
echo "  $(ls "$CACHE"/c/*.c | wc -l) C TUs"

echo "=== [2/6] C -> bitcode (vectorizers off) + link the shared shim ==="
for f in "$CACHE"/c/*.c; do "$CLANG" -O2 -fno-vectorize -fno-slp-vectorize -emit-llvm -c -I"$NIMLIB" "$f" -o "$f.bc"; done
"$CLANG" -O2 -fno-vectorize -fno-slp-vectorize -DSVM_GUEST -emit-llvm -c -I"$NIMLIB" "$SHIM" -o "$CACHE/shim.bc"
"$LINK" "$CACHE"/c/*.c.bc "$CACHE/shim.bc" -o "$CACHE/hexer.linked.bc"

echo "=== [3/6] translate (--stub-externs) + prep_svmb (verify) ==="
TR="$REPO/crates/svm-llvm/target/release/svm-llvm-translate"
[ -x "$TR" ] || cargo build --release --bin svm-llvm-translate --manifest-path "$REPO/crates/svm-llvm/Cargo.toml"
"$TR" "$CACHE/hexer.linked.bc" -o "$CACHE/hexer_raw.svmb" --binary --host-page 65536 --stub-externs
cargo run -q --release -p svm-run --example prep_svmb -- "$CACHE/hexer_raw.svmb" "$CACHE/hexer.svmb" | grep -E "funcs|wrote"

echo "=== [4/6] generate hexer's input: nimony c fills a nimcache, extract the .s.nif set ==="
fail=0
for src in "$HERE"/../nifler_svm/inputs/*.nim; do
  name="$(basename "$src")"
  work="$CACHE/work_$name"; rm -rf "$work"; mkdir -p "$work"; cp "$src" "$work/prog.nim"
  ( cd "$work" && "$NIMONY" c --isMain prog.nim >/dev/null 2>&1 )
  # hexer reads the main module's .s.nif/.s.idx.nif + every imported module's (here: system).
  fixture="$CACHE/fix_$name"; rm -rf "$fixture"; mkdir -p "$fixture"
  cp "$work"/nimcache/*.s.nif "$work"/nimcache/*.s.idx.nif "$fixture/" 2>/dev/null || true
  main="$(ls "$fixture"/*.s.nif | xargs -n1 basename | grep -v '^sys' | sed 's/\.s\.nif//' | head -1)"

  echo "=== [5/6] $name: native hexer c (oracle) ==="
  nat="$CACHE/nat_$name"; rm -rf "$nat"; cp -r "$fixture" "$nat"
  ( cd "$nat" && "$HEXER_BIN" c "$main.s.nif" >/dev/null 2>&1 )
  # native's produced files = everything in $nat not in $fixture
  produced=$(comm -13 <(ls "$fixture" | sort) <(ls "$nat" | sort))

  echo "=== [6/6] $name: hexer-on-SVM + diff every produced file (treewalk, bytecode, jit) ==="
  for eng in treewalk bytecode jit; do
    out="$CACHE/svm_${name}_$eng"; rm -rf "$out"; mkdir -p "$out"
    set +e
    SVM_PHASE_BACKEND="$eng" cargo run -q --release -p svm-run --example nimphase_run -- \
      "$CACHE/hexer.svmb" "$fixture" "$out" -- hexer c "$main.s.nif" >/dev/null 2>"$out/err.txt"; rc=$?
    set -e
    ok=1; [ "$rc" = 0 ] || ok=0
    for f in $produced; do diff -q "$nat/$f" "$out/$f" >/dev/null 2>&1 || ok=0; done
    if [ "$ok" = 1 ]; then
      echo "  $name [$eng]: OK ($(echo $produced | wc -w) files byte-identical, incl $main.x.nif Leng)"
    else
      echo "  $name [$eng]: MISMATCH (exit=$rc)"; head -c 200 "$out/err.txt" | sed 's/^/    /'; fail=1
    fi
  done
done
[ "$fail" = 0 ] && echo "ALL MATCH NATIVE — real hexer lowers semchecked NIF to Leng on the SVM, byte-exact" || { echo "FAILED"; exit 1; }
