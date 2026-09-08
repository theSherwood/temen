#!/usr/bin/env bash
# MicroPython on the LLVM on-ramp — the whole-program bitcode pipeline, mirroring the QuickJS/Tcl
# capstones. Clone → generate the `ports/embed` package with our config → per-TU bitcode → llvm-link
# the embed package + the driver + `py_shim.c` (+ guest openlibm for float math) → translate through
# the on-ramp. Fetched-not-vendored (MicroPython/MIT). See README.md for the gap-walk record.
#
#   needs: git, clang, llvm-link, cc, make, python3
#   env:   TEMEN_MICROPYTHON_CACHE (default /tmp/temen_micropython_cache),
#          TEMEN_MICROPYTHON_VER  (default 1.24.1),
#          OPENLIBM_DIR (optional staged openlibm tree; else the shared /tmp cache, else fetched)
#
#   ./build_bitcode.sh    # → $CACHE/mp_linked.ll and $CACHE/mp_snapshot_linked.ll (+ a native oracle)
#
# GAP-WALK (all closed by config in mpconfigport.h — no MicroPython source is patched):
#   1. printf("%.*s") dynamic precision   → HAL override in py_shim.c writes to the Stream cap.
#   2. `half` (f16) IR type in binary.c   → MICROPY_FLOAT_USE_NATIVE_FLT16=0 (software half codec).
#   3. x86-64 inline asm in nlr*/gchelper → MICROPY_NLR_SETJMP=1 + MICROPY_GCREGS_SETJMP=1.
#   4. float math (sin/cos/pow/fmod/nan)  → llvm-link guest openlibm (step [4/6]); without it the
#                                           exec path reaches an unresolved libm symbol and traps.
# STATUS (issue #1326): the module translates (~729 funcs), verifies, and **runs byte-identical to the
# native `cc` oracle** — arithmetic, str/list/dict, comprehensions, closures, recursion, floats, and
# exception repr all match. See README.md.
set -uo pipefail

VER="${TEMEN_MICROPYTHON_VER:-1.24.1}"
CACHE="${TEMEN_MICROPYTHON_CACHE:-/tmp/temen_micropython_cache}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$CACHE/micropython-$VER"
EMB="$SRC/examples/embedding"
PKG="$EMB/micropython_embed"
OUT="$CACHE/bc"
mkdir -p "$CACHE"
cd "$CACHE"

echo "=== [1/6] fetch micropython $VER (git — the GitHub tarball archive is proxy-gated) ==="
if [ ! -f "$SRC/py/mpstate.h" ]; then
  git clone --depth 1 --branch "v$VER" https://github.com/micropython/micropython "$SRC" \
    >"$CACHE/clone.log" 2>&1 || { echo "CLONE FAILED (offline?)"; tail -3 "$CACHE/clone.log"; exit 11; }
fi

echo "=== [2/6] generate the ports/embed package with the Temen config ==="
# Our mpconfigport.h drives the qstr/genhdr generation, so the frozen config is baked into the package.
cp "$HERE/mpconfigport.h" "$EMB/mpconfigport.h"
( cd "$EMB" && rm -rf micropython_embed build-embed \
    && make -f micropython_embed.mk -j"$(nproc)" ) >"$CACHE/gen.log" 2>&1 \
  || { echo "EMBED GEN FAILED"; tail -6 "$CACHE/gen.log"; exit 12; }
echo "package TUs: $(find "$PKG" -name '*.c' | wc -l)"

echo "=== [3/6] per-TU bitcode (exclude port/mphalport.c — HAL is overridden in py_shim.c) ==="
rm -rf "$OUT"; mkdir -p "$OUT"
# -std=gnu99 so the (dead, pruned) gchelper asm TU still parses; -fno-vectorize keeps loops scalar.
CF=(-O2 -emit-llvm -S -fno-vectorize -fno-slp-vectorize -std=gnu99 -DNDEBUG
    "-I$EMB" "-I$PKG" "-I$PKG/port")
fail=0
for c in $(find "$PKG" -name '*.c'); do
  case "$c" in */port/mphalport.c) continue ;; esac
  b=$(echo "$c" | sed "s|$PKG/||; s|/|_|g; s|\.c$||")
  clang "${CF[@]}" "$c" -o "$OUT/$b.ll" 2>"$OUT/$b.err" \
    || { echo "  CLANG FAIL: $b"; head -2 "$OUT/$b.err"; fail=1; }
done
clang "${CF[@]}" "$HERE/py_shim.c" -o "$OUT/_pyshim.ll" 2>"$OUT/_pyshim.err" \
  || { echo "  py_shim FAIL"; head -3 "$OUT/_pyshim.err"; fail=1; }
echo "compiled $(ls "$OUT"/*.ll 2>/dev/null | wc -l) TUs (fail=$fail)"

echo "=== [4/6] guest openlibm (float math: sin/cos/pow/… address-taken in Python's math paths) ==="
# Curated set (the QuickJS OPENLIBM_SRCS + MicroPython's frexp/modf/pow extras) — not a glob.
SHARED_OL=/tmp/temen_openlibm_cache/openlibm-0.8.5
OL="${OPENLIBM_DIR:-$SHARED_OL}"
if [ ! -f "$OL/src/e_log.c" ]; then
  mkdir -p /tmp/temen_openlibm_cache
  git clone --depth 1 --branch v0.8.5 https://github.com/JuliaMath/openlibm "$SHARED_OL" \
    >/dev/null 2>&1 || true
  OL="$SHARED_OL"
fi
if [ -f "$OL/src/e_log.c" ]; then
  OLSRCS="e_log e_log10 e_log2 e_exp s_exp2 e_pow s_sin s_cos s_tan k_sin k_cos k_tan
          e_rem_pio2 k_rem_pio2 e_asin e_acos s_atan e_atan2 e_sinh e_cosh s_tanh s_cbrt
          e_fmod s_scalbn s_copysign s_fabs k_exp s_expm1 e_hypot s_frexp s_modf
          s_asinh s_log1p s_round s_rint s_ceil s_floor s_trunc s_nan"
  n=0
  for b in $OLSRCS; do
    [ -f "$OL/src/$b.c" ] || continue
    clang "${CF[@]}" "-I$OL" "-I$OL/include" "-I$OL/src" "-I$OL/amd64" \
      "$OL/src/$b.c" -o "$OUT/libm_$b.ll" 2>/dev/null && n=$((n+1)) || true
  done
  echo "openlibm: $n TUs"
else
  echo "note: openlibm unavailable — float transcendentals will surface as undefined at resolve"
fi

echo "=== [5/6] drivers → bitcode (each defines main; linked into a separate variant) ==="
DRV="$CACHE/drivers"; mkdir -p "$DRV"
clang "${CF[@]}" "$HERE/micropython.c"          -o "$DRV/repl.ll"     || { echo "driver FAIL"; exit 14; }
clang "${CF[@]}" "$HERE/micropython_snapshot.c" -o "$DRV/snapshot.ll" || { echo "snapshot FAIL"; exit 14; }

echo "=== [6/6] llvm-link both variants → translate through the on-ramp ==="
LINKED="$CACHE/mp_linked.ll"
SNAP_LINKED="$CACHE/mp_snapshot_linked.ll"
llvm-link -S "$OUT"/*.ll "$DRV/repl.ll" -o "$LINKED" 2>"$CACHE/llvm-link.err" \
  || { echo "LINK FAILED (repl):"; tail -5 "$CACHE/llvm-link.err"; exit 15; }
llvm-link -S "$OUT"/*.ll "$DRV/snapshot.ll" -o "$SNAP_LINKED" 2>"$CACHE/llvm-link-snap.err" \
  || { echo "LINK FAILED (snapshot):"; tail -5 "$CACHE/llvm-link-snap.err"; exit 15; }
echo "linked: repl     $(stat -c%s "$LINKED") B → $LINKED"
echo "linked: snapshot $(stat -c%s "$SNAP_LINKED") B → $SNAP_LINKED"

# Native oracle: same driver + package + shim through `cc` (the HAL override writes to fd 1 = real
# stdout), so its output is the byte-exact differential reference.
echo "=== native oracle (cc) ==="
NAT="$CACHE/micropython_native"
cc -O2 -std=gnu99 -DNDEBUG "-I$EMB" "-I$PKG" "-I$PKG/port" \
  $(find "$PKG" -name '*.c' ! -path '*/port/mphalport.c') \
  "$HERE/py_shim.c" "$HERE/micropython.c" -lm -o "$NAT" 2>"$CACHE/native.err" \
  && echo "native oracle: $NAT" || { echo "note: native oracle build skipped"; head -3 "$CACHE/native.err"; }

TR="$HERE/../../../temen-llvm/target/release/examples/try_translate"
if [ -x "$TR" ]; then
  echo "translate + verify (REPL):"
  TEMEN_STUB_EXTERNS=1 "$TR" "$LINKED" 2>&1 | grep -v '\[stub\]' | head -4
else
  echo "note: build the translator first:  (cd crates/temen-llvm && cargo build --release --example try_translate)"
fi
