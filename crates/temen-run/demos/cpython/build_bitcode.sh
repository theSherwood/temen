#!/usr/bin/env bash
# CPython on the LLVM on-ramp — feasibility SPIKE (PYTHON.md slice D, #1328). The CPython twin of
# demos/postgres/build_bitcode.sh: clone → configure a static/minimal interpreter → native oracle →
# per-TU bitcode (emit_bc.py, reusing the makefile's exact flags) → llvm-link the exact `python` link
# set → translate through the on-ramp to enumerate the fail-closed gaps. Fetched-not-vendored (PSF).
#
#   needs: git, clang, llvm-link, llvm-nm, make, pkg-config, a working host cc for configure checks
#   env:   TEMEN_CPY_CACHE (default /tmp/temen_cpython_cache), TEMEN_CPY_VER (default 3.13.1)
#
#   ./build_bitcode.sh    # → $CACHE/python.linked.bc  + $CACHE/translate.err (the current first gap)
#
# This is a SPIKE, not a landed demo: CPython does not yet translate end-to-end. Its purpose is the
# reproducible gap inventory in README.md. See PYTHON.md §4 (Phase D) for why CPython is a separate,
# ~3–5× larger bring-up than MicroPython, and how it leans on the upstream wasm32-wasi / Pyodide
# static ports.
set -uo pipefail

VER="${TEMEN_CPY_VER:-3.13.1}"
CACHE="${TEMEN_CPY_CACHE:-/tmp/temen_cpython_cache}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$CACHE/cpython-$VER"
mkdir -p "$CACHE"

echo "=== [1/6] fetch cpython $VER (git — GitHub tarball archive is proxy-gated) ==="
if [ ! -f "$SRC/Python/ceval.c" ]; then
  git clone --depth 1 --branch "v$VER" https://github.com/python/cpython "$SRC" \
    >"$CACHE/clone.log" 2>&1 || { echo "CLONE FAILED (offline?)"; tail -3 "$CACHE/clone.log"; exit 11; }
fi

echo "=== [2/6] configure (CC=clang, static, minimal; mimalloc off — gap #1) ==="
cd "$SRC"
# CC=clang: the default host cc is gcc, which rejects clang's -fno-vectorize (the compiler-probe
# fails). --disable-shared: one static libpython (no dlopen — the on-ramp has no dynamic loader).
# --without-mimalloc: CPython 3.13's default allocator is bundled mimalloc, whose `_mi_heap_default`
# is an initialized thread-local the on-ramp's TLS block layout can't resolve (gap #1); dropping it
# falls back to pymalloc/obmalloc. --disable-test-modules / --without-ensurepip: trim the link set.
[ -f Makefile ] || ./configure CC=clang --disable-shared --without-ensurepip --disable-test-modules \
  --without-mimalloc --with-system-ffi=no CFLAGS="-O2" >"$CACHE/configure.log" 2>&1 \
  || { echo "CONFIGURE FAILED"; tail -12 "$CACHE/configure.log"; exit 12; }

echo "=== [3/6] native oracle build ==="
make -j"$(nproc)" >"$CACHE/make.log" 2>&1
[ -x "$SRC/python" ] || { echo "BUILD FAILED"; tail -15 "$CACHE/make.log"; exit 13; }
echo "python: $(stat -c%s "$SRC/python") bytes; $("$SRC/python" -c 'import sys;print(sys.version.split()[0])')"

echo "=== [4/6] per-TU bitcode (reuses makefile flags) ==="
TEMEN_CPY_SRC="$SRC" python3 "$HERE/emit_bc.py"

echo "=== [5/6] llvm-link the exact python link set ==="
# The authoritative object list = the FINAL `-o python` link line (not the whole `make -n` output,
# which also prints the frozen-module build tools _freeze_module / _bootstrap_python that redefine
# _PyImport_FrozenBootstrap). Map .o → .bc, keep only those that exist.
make -n python 2>/dev/null | grep -E '\-o python( |$)' | tail -1 | grep -oE '[A-Za-z0-9_./-]+\.o' \
  | sort -u | sed "s#^#$SRC/#" | sed 's/\.o$/.bc/' \
  | while read -r p; do [ -f "$p" ] && echo "$p"; done > "$CACHE/bcset.txt"
echo "linking $(wc -l < "$CACHE/bcset.txt") modules"
llvm-link $(cat "$CACHE/bcset.txt") -o "$CACHE/python.linked.bc" 2>"$CACHE/llvm-link.err" \
  || { echo "LINK FAILED:"; tail -4 "$CACHE/llvm-link.err"; exit 15; }
echo "linked: $(stat -c%s "$CACHE/python.linked.bc") bytes"

echo "=== [6/6] translate through the on-ramp (expect a fail-closed gap) ==="
TR="$HERE/../../../temen-llvm/target/release/temen-llvm-translate"
[ -x "$TR" ] || (cd "$HERE/../../../temen-llvm" && cargo build --release --bin temen-llvm-translate 2>&1 | tail -1)
"$TR" "$CACHE/python.linked.bc" -o "$CACHE/python.temt" 2>"$CACHE/translate.err"
echo "first gap: $(head -1 "$CACHE/translate.err")"
