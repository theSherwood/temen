#!/usr/bin/env bash
# Tcl on the LLVM on-ramp — the whole-program bitcode pipeline, mirroring the Postgres/QuickJS
# capstones. Fetch → configure → native oracle → per-TU bitcode (reusing the Makefile's exact flags)
# → llvm-link the libtcl object set + the driver + reused shims + guest openlibm → translate through
# the on-ramp. Fetched-not-vendored (Tcl/BSD license). See README.md for the gap-walk record.
#
#   needs: clang, llvm-link, cc, make, curl, tar
#   env:   SVM_TCL_CACHE (default /tmp/svm_tcl_cache), SVM_TCL_VER (default 8.6.14),
#          OPENLIBM_DIR (a staged openlibm tree, as the QuickJS build uses)
#
#   ./build_bitcode.sh            # → $CACHE/tcl_linked.ll  (+ a native oracle at unix/tclsh)
set -uo pipefail

VER="${SVM_TCL_VER:-8.6.14}"
CACHE="${SVM_TCL_CACHE:-/tmp/svm_tcl_cache}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$CACHE/tcl$VER"
OUT="$CACHE/bc"
mkdir -p "$CACHE" "$OUT"
cd "$CACHE"

echo "=== [1/6] fetch tcl-$VER ==="
if [ ! -f "$SRC/generic/tcl.h" ]; then
  curl -sfL --max-time 180 -o "tcl$VER-src.tar.gz" \
    "https://prdownloads.sourceforge.net/tcl/tcl$VER-src.tar.gz" || { echo "FETCH FAILED"; exit 11; }
  tar xf "tcl$VER-src.tar.gz"
fi

echo "=== [2/6] configure (static, no threads/load — the minimal core) ==="
cd "$SRC/unix"
# --disable-threads keeps a single guest vCPU (the minimal REPL is serial); --disable-load drops
# dlopen (`load` command); static so there is no shared-lib surface.
[ -f Makefile ] || ./configure --disable-shared --disable-threads --disable-load \
  >"$CACHE/configure.log" 2>&1 || { echo "CONFIGURE FAILED"; tail -5 "$CACHE/configure.log"; exit 12; }

echo "=== [3/6] native oracle build (libtcl + tclsh) ==="
make binaries -j"$(nproc)" >"$CACHE/make.log" 2>&1
# Tcl names the archive by MAJOR.MINOR (libtcl8.6.a), not the full patch version.
LIB=$(ls libtcl*.a 2>/dev/null | grep -v stub | head -1)
[ -n "$LIB" ] || { echo "BUILD FAILED"; tail -6 "$CACHE/make.log"; exit 13; }
echo "$LIB: $(stat -c%s "$LIB") bytes"

echo "=== [4/6] per-TU bitcode (reuses the Makefile's exact flags) ==="
# The authoritative object set = the members of the archive. The Makefile compiles every core TU with
# one common flag set (CC_SWITCHES); capture it once from a representative TU (tclBasic), swap
# gcc→clang -emit-llvm. Two files carry extra per-file defines in the Makefile — tclUnixInit's
# TCL_LIBRARY/TCL_PACKAGE_PATH and tclPkgConfig's CFG_* install/runtime paths; those are just string
# macros unreferenced elsewhere, so we append harmless placeholders to the *common* set and compile
# every TU uniformly (robust — no per-file `make -n` recomputation).
rm -f "$OUT"/*.ll
LINE=$(make -n -B tclBasic.o 2>/dev/null | grep -m1 'tclBasic\.c'); LINE=${LINE/gcc/}
eval "set -- $LINE"
COMMON=()
for a in "$@"; do
  case "$a" in
    -c | -pipe | -Wall | -Wpointer-arith | -o | tclBasic.o | *tclBasic.c) ;;
    *) COMMON+=("$a") ;;
  esac
done
# Placeholders for the two per-file-define TUs (paths are meaningless in-sandbox; the minimal REPL
# never reads `info library` / `info nameofexecutable`).
COMMON+=(-DTCL_LIBRARY='""' -DTCL_PACKAGE_PATH='""'
         -DCFG_INSTALL_LIBDIR='""' -DCFG_INSTALL_BINDIR='""' -DCFG_INSTALL_SCRDIR='""'
         -DCFG_INSTALL_INCDIR='""' -DCFG_INSTALL_DOCDIR='""'
         -DCFG_RUNTIME_LIBDIR='""' -DCFG_RUNTIME_BINDIR='""' -DCFG_RUNTIME_SCRDIR='""'
         -DCFG_RUNTIME_INCDIR='""' -DCFG_RUNTIME_DOCDIR='""')
fail=0
for obj in $(ar t "$LIB"); do
  base=${obj%.o} src=""
  for d in generic unix libtommath compat; do
    [ -f "$SRC/$d/$base.c" ] && { src="$SRC/$d/$base.c"; break; }
  done
  [ -n "$src" ] || { echo "  NO SRC: $base"; fail=1; continue; }
  clang -emit-llvm -S -O2 -fno-vectorize -fno-slp-vectorize "${COMMON[@]}" \
    "$src" -o "$OUT/$base.ll" 2>"$OUT/$base.err" || { echo "  CLANG FAIL: $base"; head -2 "$OUT/$base.err"; fail=1; }
done
echo "compiled $(ls "$OUT"/*.ll 2>/dev/null | wc -l) TUs (fail=$fail)"

echo "=== [5/6] driver + reused shims + guest openlibm → bitcode ==="
CF=(-O2 -emit-llvm -S -fno-vectorize -fno-slp-vectorize -DNDEBUG -D_GNU_SOURCE
    "-I$SRC/generic" "-I$SRC/unix")
DEMOS="$HERE/.."
# The driver + the Tcl-specific OS/libc shim.
clang "${CF[@]}" "$HERE/tcl_repl.c" -o "$OUT/_driver.ll" || { echo "driver FAIL"; exit 14; }
clang "${CF[@]}" "$HERE/tcl_shim.c" -o "$OUT/_tclshim.ll" || { echo "tcl_shim FAIL"; exit 14; }
# Reused waist (see README): the Postgres printf/scanf engines, the guest strtod. (ctype tables are
# pulled from the Postgres shim set too if your resolve stage reports them undefined.)
for s in "postgres/printf_shim:_printf" "strtod/strtod:_strtod"; do
  src="$DEMOS/${s%%:*}.c"; tag="${s##*:}"
  clang "${CF[@]}" "$src" -o "$OUT/$tag.ll" 2>/dev/null || echo "  note: optional shim ${s%%:*} skipped"
done
# Guest openlibm — Tcl's `expr` math (sin/cos/pow/sqrt/… address-taken in the math-function table, the
# QuickJS slice CO mechanism). Fetch to a cache if OPENLIBM_DIR is unset; compile the **curated** set
# (the QuickJS `OPENLIBM_SRCS` + Tcl's frexp/modf/hypot/log1p extras), not a glob (many openlibm TUs
# need asm/aren't on this path).
OL="${OPENLIBM_DIR:-$CACHE/openlibm-0.8.5}"
if [ ! -f "$OL/src/e_log.c" ]; then
  echo "fetching openlibm 0.8.5 …"
  curl -sfL --max-time 120 -o "$CACHE/openlibm.tgz" \
    "https://github.com/JuliaMath/openlibm/archive/refs/tags/v0.8.5.tar.gz" \
    && tar xf "$CACHE/openlibm.tgz" -C "$CACHE" 2>/dev/null || true
fi
if [ -f "$OL/src/e_log.c" ]; then
  OLSRCS="e_log e_log10 e_log2 e_exp s_exp2 e_pow s_sin s_cos s_tan k_sin k_cos k_tan
          e_rem_pio2 k_rem_pio2 e_asin e_acos s_atan e_atan2 e_sinh e_cosh s_tanh s_cbrt
          e_fmod s_scalbn s_copysign s_fabs k_exp s_expm1
          e_hypot s_frexp s_modf s_asinh s_log1p s_round s_rint s_ceil s_floor s_trunc"
  n=0
  for b in $OLSRCS; do
    [ -f "$OL/src/$b.c" ] || continue
    clang "${CF[@]}" "-I$OL" "-I$OL/include" "-I$OL/src" "-I$OL/amd64" \
      "$OL/src/$b.c" -o "$OUT/libm_$b.ll" 2>/dev/null && n=$((n+1)) || true
  done
  echo "openlibm: $n TUs"
else
  echo "note: openlibm unavailable — libm transcendentals will surface as undefined at resolve"
fi

echo "=== [6/6] llvm-link → translate through the on-ramp ==="
LINKED="$CACHE/tcl_linked.ll"
llvm-link -S "$OUT"/*.ll -o "$LINKED" 2>"$CACHE/llvm-link.err" \
  || { echo "LINK FAILED:"; tail -5 "$CACHE/llvm-link.err"; exit 15; }
echo "linked: $(stat -c%s "$LINKED") bytes → $LINKED"
TR="$HERE/../../../svm-llvm/target/release/examples/try_translate"
if [ -x "$TR" ]; then
  # `SVM_STUB_EXTERNS=1`: trap-stub the OS surface unreached by the minimal REPL (zlib/scanf/fts) so
  # the big program translates; the translator fixes (constexpr icmp, vector ptrtoint) are folded in.
  echo "translate + verify (a runtime stub trap during channel init is the current frontier):"
  SVM_STUB_EXTERNS=1 "$TR" "$LINKED" 2>&1 | grep -v '\[stub\]' | head -5
else
  echo "note: build the translator first:  (cd crates/svm-llvm && cargo build --release --example try_translate)"
fi
