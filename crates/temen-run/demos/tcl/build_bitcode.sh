#!/usr/bin/env bash
# Tcl on the LLVM on-ramp — the whole-program bitcode pipeline, mirroring the Postgres/QuickJS
# capstones. Fetch → configure → native oracle → per-TU bitcode (reusing the Makefile's exact flags)
# → llvm-link the libtcl object set + the driver + reused shims + guest openlibm → translate through
# the on-ramp. Fetched-not-vendored (Tcl/BSD license). See README.md for the gap-walk record.
#
#   needs: clang, llvm-link, cc, make, curl, tar
#   env:   TEMEN_TCL_CACHE (default /tmp/temen_tcl_cache), TEMEN_TCL_VER (default 8.6.14),
#          OPENLIBM_DIR (a staged openlibm tree, as the QuickJS build uses)
#
#   ./build_bitcode.sh            # → $CACHE/tcl_linked.ll  (+ a native oracle at unix/tclsh)
set -uo pipefail

VER="${TEMEN_TCL_VER:-8.6.14}"
CACHE="${TEMEN_TCL_CACHE:-/tmp/temen_tcl_cache}"
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

echo "=== [5/6] drivers + reused shims + guest openlibm → bitcode ==="
CF=(-O2 -emit-llvm -S -fno-vectorize -fno-slp-vectorize -DNDEBUG -D_GNU_SOURCE
    "-I$SRC/generic" "-I$SRC/unix")
DEMOS="$HERE/.."
# The two drivers are built to a SEPARATE dir (not $OUT) — each defines `main`, so only one is linked
# into a given module. The full-init driver also needs the embedded script library header.
DRV="$CACHE/drivers"; mkdir -p "$DRV"
python3 "$HERE/gen_tcl_library.py" "$SRC/library" "$DRV/tcl_library.h" 2>&1 | tail -1
clang "${CF[@]}" "$HERE/tcl_repl.c" -o "$DRV/repl.ll" || { echo "tcl_repl FAIL"; exit 14; }
clang "${CF[@]}" "-I$DRV" "$HERE/tcl_init.c" -o "$DRV/init.ll" || { echo "tcl_init FAIL"; exit 14; }
# The two-phase warm-runtime-snapshot driver (issue #805 follow-on): same full Tcl_Init as tcl_init.c,
# split into `warmup` + `eval_run` so the playground can snapshot after Tcl_Init and eval-only per Run.
clang "${CF[@]}" "-I$DRV" "$HERE/tcl_snapshot.c" -o "$DRV/snapshot.ll" || { echo "tcl_snapshot FAIL"; exit 14; }
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
# Prefer an explicit OPENLIBM_DIR; else the shared cache the `libm_bundled_vs_native` test populates
# (/tmp/temen_openlibm_cache); else fetch into the shared cache so both reuse one copy.
SHARED_OL=/tmp/temen_openlibm_cache/openlibm-0.8.5
OL="${OPENLIBM_DIR:-$SHARED_OL}"
if [ ! -f "$OL/src/e_log.c" ]; then
  echo "fetching openlibm 0.8.5 …"
  mkdir -p /tmp/temen_openlibm_cache
  curl -sfL --max-time 120 -o /tmp/temen_openlibm_cache/openlibm.tgz \
    "https://github.com/JuliaMath/openlibm/archive/refs/tags/v0.8.5.tar.gz" \
    && tar xf /tmp/temen_openlibm_cache/openlibm.tgz -C /tmp/temen_openlibm_cache 2>/dev/null || true
  OL="$SHARED_OL"
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

echo "=== [6/6] llvm-link both variants → translate through the on-ramp ==="
# Shared core = everything in $OUT (Tcl core + shims + openlibm); each variant adds one driver.
#   tcl_linked.ll       — the minimal REPL (tcl_repl.c, no Tcl_Init, no filesystem)
#   tcl_init_linked.ll  — the full Tcl (tcl_init.c: Tcl_Init over the embedded-library VFS)
LINKED="$CACHE/tcl_linked.ll"
INIT_LINKED="$CACHE/tcl_init_linked.ll"
SNAP_LINKED="$CACHE/tcl_snapshot_linked.ll"
llvm-link -S "$OUT"/*.ll "$DRV/repl.ll" -o "$LINKED" 2>"$CACHE/llvm-link.err" \
  || { echo "LINK FAILED (repl):"; tail -5 "$CACHE/llvm-link.err"; exit 15; }
llvm-link -S "$OUT"/*.ll "$DRV/init.ll" -o "$INIT_LINKED" 2>"$CACHE/llvm-link-init.err" \
  || { echo "LINK FAILED (init):"; tail -5 "$CACHE/llvm-link-init.err"; exit 15; }
llvm-link -S "$OUT"/*.ll "$DRV/snapshot.ll" -o "$SNAP_LINKED" 2>"$CACHE/llvm-link-snap.err" \
  || { echo "LINK FAILED (snapshot):"; tail -5 "$CACHE/llvm-link-snap.err"; exit 15; }
echo "linked: repl $(stat -c%s "$LINKED") B → $LINKED"
echo "linked: init $(stat -c%s "$INIT_LINKED") B → $INIT_LINKED"
echo "linked: snapshot $(stat -c%s "$SNAP_LINKED") B → $SNAP_LINKED"
TR="$HERE/../../../temen-llvm/target/release/examples/try_translate"
if [ -x "$TR" ]; then
  # `TEMEN_STUB_EXTERNS=1`: trap-stub the OS surface unreached by the guest (zlib/scanf/fts). The
  # full-init variant serves its script library through an in-guest VFS (no filesystem capability).
  echo "translate + verify (minimal REPL):"
  TEMEN_STUB_EXTERNS=1 "$TR" "$LINKED" 2>&1 | grep -v '\[stub\]' | head -3
  echo "translate + verify (full Tcl_Init):"
  TEMEN_STUB_EXTERNS=1 "$TR" "$INIT_LINKED" 2>&1 | grep -v '\[stub\]' | head -3
else
  echo "note: build the translator first:  (cd crates/temen-llvm && cargo build --release --example try_translate)"
fi
