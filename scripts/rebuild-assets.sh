#!/usr/bin/env bash
# Rebuild **every committed `.temen` playground/self-host asset** in one pass — the single entry point
# for the "a wire-format / encoder / IR change invalidated the prebuilt binary assets" chore (which
# recurs on every such change: the committed modules decode as `BadOpcode` under the new format and
# their asset-gate tests — leng_selfhost_asset, nifler_asset, nim_hello_asset, and the real-browser
# play cards — go red until regenerated).
#
# This orchestrates the existing per-asset builders (it does NOT reimplement them) and, crucially,
# wires up the toolchain env each one expects — the tribal knowledge that otherwise gets rediscovered
# by hand every time:
#   * the Nim toolchain whose `../lib/nimbase.h` the nifler C backend needs (the `nim` shim on PATH
#     usually has no adjacent lib dir — point at the real choosenim toolchain),
#   * NIFLER_BIN / NIMONY_BIN / NIM_BIN for the nimony pipeline (what scripts/ci/provision-nimony.sh
#     exports), reusing the vendored `nimony/bin/{nifler,nimony}` when present.
#
# Every step is **fail-soft**: a missing toolchain SKIPs that asset (matching each builder's own
# contract) so a partial environment still rebuilds what it can. Each rebuilt module is re-validated
# (decode → verify → bytecode-compile) via `prep_temen`. A final summary lists ✓ / SKIP / ✗.
#
#   Usage:  bash scripts/rebuild-assets.sh              # rebuild everything the toolchain allows
#           ONLY=leng,nim_hello bash scripts/...        # rebuild a subset (comma-separated step names)
#   Steps:  leng chibicc onramp shell forth nifler nim_hello nim_phases nim_driver_guest lua_snapshot
#
# Toolchains, per step: leng needs rustc (+rust-src) & llvm; chibicc/onramp need clang &
# llvm-link (onramp also fetches QuickJS/SQLite/Lua sources — skipped offline); shell needs the
# in-tree chibicc; nifler & nim_hello need the nimony toolchain (Nim + nimony/bin) — see NIM.md §2.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

ONLY="${ONLY:-}"
want() { [ -z "$ONLY" ] || [[ ",$ONLY," == *",$1,"* ]]; }
declare -a RESULTS=()
note() { RESULTS+=("$1"); echo "  >> $1"; }

# --- toolchain env: the setup each nimony builder assumes (see the header) ---------------------------
# Prefer a real Nim toolchain dir (adjacent ../lib/nimbase.h) over a bare `nim` shim.
pick_nim() {
  local c
  for c in \
    "$(command -v nim 2>/dev/null)" \
    /root/.choosenim/toolchains/*/bin/nim \
    "$HOME"/.choosenim/toolchains/*/bin/nim; do
    [ -x "$c" ] || continue
    [ -f "$(dirname "$c")/../lib/nimbase.h" ] || continue
    echo "$c"; return 0
  done
  return 1
}
NIM_EXE="$(pick_nim || true)"
if [ -n "$NIM_EXE" ]; then
  export PATH="$(dirname "$NIM_EXE"):$PATH"
  export NIM_BIN="$(dirname "$NIM_EXE")"
fi
if [ -x "$REPO/nimony/bin/nifler" ]; then
  export PATH="$REPO/nimony/bin:$PATH"
  export NIFLER_BIN="$REPO/nimony/bin/nifler"
  export NIMONY_BIN="$REPO/nimony/bin"
  [ -x "$REPO/nimony/bin/hexer" ] && export HEXER_BIN="$REPO/nimony/bin/hexer"
fi
# The nim C backend #include's `nimbase.h` from the Nim lib. build_e2e_chain.sh falls back to
# `.nimtool/nim-src/lib` when `nim dump` doesn't print the lib path (some toolchains don't) — point
# that at the picked Nim's lib so the fallback resolves (what provision-nimony.sh's nim-src clone gives).
if [ -n "${NIM_BIN:-}" ] && [ -f "$NIM_BIN/../lib/nimbase.h" ] && [ ! -f "$REPO/.nimtool/nim-src/lib/nimbase.h" ]; then
  mkdir -p "$REPO/.nimtool/nim-src"
  ln -sfn "$NIM_BIN/../lib" "$REPO/.nimtool/nim-src/lib"
fi

# --- shared build products --------------------------------------------------------------------------
echo "=== building temen-llvm-translate + prep_temen (shared by every step) ==="
( cd "$REPO/crates/temen-llvm" && cargo build --release --bin temen-llvm-translate ) || true
cargo build --release -p temen-run --example prep_temen || true
PREP="$REPO/target/release/examples/prep_temen"

# Validate that a produced .temen decodes+verifies (prep_temen exits non-zero / panics otherwise).
# Non-powerbox child modules (stage_runner/primes/upper) trip prep_temen's powerbox assertion *after*
# a clean decode — those are validated by their own generator, so this is only used where it applies.
validate() { "$PREP" "$1" /tmp/rebuild_assets_check.temen >/dev/null 2>&1; }

# --- 1) temen-leng.temen (Rust translator; build-std → on-ramp → prep_temen) -------------------------
if want leng; then
  echo "=== [leng] crates/temen-run/demos/leng_selfhost/build_leng_temen.sh ==="
  if bash crates/temen-run/demos/leng_selfhost/build_leng_temen.sh; then
    cp "${TEMEN_LENG_CACHE:-/tmp/temen_leng_cache}/temen-leng.temen" \
       crates/temen-run/demos/leng_selfhost/temen-leng.temen
    validate crates/temen-run/demos/leng_selfhost/temen-leng.temen \
      && note "leng ✓ (temen-leng.temen; browser copy is refreshed by onramp)" \
      || note "leng ✗ (rebuilt but failed re-validate)"
  else
    note "leng SKIP/✗ (see output above — rustc + rust-src + llvm?)"
  fi
fi

# --- 2) chibicc.temen (in-tree chibicc → clang → translate) -----------------------------------------
if want chibicc; then
  echo "=== [chibicc] crates/temen-run/demos/chibicc_selfhost/build_chibicc_temen.sh ==="
  if bash crates/temen-run/demos/chibicc_selfhost/build_chibicc_temen.sh; then
    cp "${TEMEN_CHIBICC_CACHE:-/tmp/temen_chibicc_cache}/chibicc.temen" \
       browser/web/assets/chibicc.temen
    validate browser/web/assets/chibicc.temen \
      && note "chibicc ✓" || note "chibicc ✗ (rebuilt but failed re-validate)"
  else
    note "chibicc SKIP/✗ (clang / llvm-link?)"
  fi
fi

# --- 3) on-ramp C guests + qjs (build-onramp-assets.mjs; also copies temen-leng into web/assets) -----
if want onramp; then
  echo "=== [onramp] browser/build-onramp-assets.mjs (clang C guests; QuickJS/SQLite/Lua fetched) ==="
  ( cd "$REPO/browser" && node build-onramp-assets.mjs ) \
    && note "onramp ✓ (hello_c/gradient/bounce/life/mandelzoom + qjs where sources fetched)" \
    || note "onramp partial/✗ (needs clang; network fetches skip offline)"
fi

# --- 4) shell fixtures (chibicc → POSIX; the canonical #[ignore] generator) --------------------------
if want shell; then
  echo "=== [shell] cargo test -p temen --test c_shell -- --ignored gen_browser_shell_fixture ==="
  cargo test -p temen --test c_shell -- --ignored --exact gen_browser_shell_fixture \
    && note "shell ✓ (shell/stage_runner/primes/upper fixtures)" \
    || note "shell ✗ (in-tree chibicc?)"
fi

# --- 4b) forth.temen (the sectorforth-class Forth kernel, hand-written text IR — issue #1214) ----------
# No toolchain at all: `prep_temen` parses the `.temt`, verifies, bytecode-compiles, and writes the binary.
if want forth; then
  echo "=== [forth] prep_temen crates/temen-run/demos/forth/forth.temt → browser/web/assets/forth.temen ==="
  "$PREP" crates/temen-run/demos/forth/forth.temt browser/web/assets/forth.temen >/dev/null \
    && validate browser/web/assets/forth.temen \
    && note "forth ✓ (forth.temen)" \
    || note "forth ✗ (prep_temen failed on forth.temt?)"
fi

# --- 5) nifler.temen.gz (nimony pipeline; TEMEN_NIFLER_EMIT_ASSET gzips it + the expected fixtures) --
if want nifler; then
  echo "=== [nifler] crates/temen-run/demos/nifler_temen/build_nifler_temen.sh (EMIT_ASSET=1) ==="
  if TEMEN_NIFLER_EMIT_ASSET=1 bash crates/temen-run/demos/nifler_temen/build_nifler_temen.sh; then
    if gunzip -c browser/web/assets/nifler.temen.gz > /tmp/rebuild_nifler.temen 2>/dev/null \
       && validate /tmp/rebuild_nifler.temen; then
      note "nifler ✓ (nifler.temen.gz + expected/*.p.nif)"
    else
      note "nifler SKIP (toolchain absent — script SKIPs without rebuilding)"
    fi
  else
    note "nifler ✗ (nim + nimony/bin/nifler + clang/llvm-nm?)"
  fi
fi

# --- 6) nim_hello.temen (nimony → temen-leng powerbox bridge) ----------------------------------------
if want nim_hello; then
  echo "=== [nim_hello] build_nim_hello_temen example ==="
  if cargo run --release -p temen-run --example build_nim_hello_temen -- \
       crates/temen-run/demos/nim_hello/hello.nim browser/web/assets/nim_hello.temen; then
    validate browser/web/assets/nim_hello.temen \
      && note "nim_hello ✓" || note "nim_hello ✗ (rebuilt but failed re-validate)"
  else
    note "nim_hello SKIP/✗ (NIMONY_BIN/NIM_BIN + nimony/bin/nimony?)"
  fi
fi

# --- 6b) nimsem.temen.gz + hexer.temen.gz (the nimc "compile a whole Nim program" card's phase guests;
# nifler is step 5). build_e2e_chain.sh builds all three phase guests with identical browser flags
# (--binary --host-page 65536 --stub-externs); we gzip nimsem + hexer into web/assets (nim_stdlib.img.gz
# is an fs image, not a wire-coupled module, so it never goes stale). --------------------------------
if want nim_phases; then
  echo "=== [nim_phases] crates/temen-run/demos/nim_e2e_chain/build_e2e_chain.sh → gzip nimsem+hexer ==="
  E2E_OUT="${TEMEN_E2E_CACHE:-/tmp/temen_e2e_chain}/temen"
  if bash crates/temen-run/demos/nim_e2e_chain/build_e2e_chain.sh; then
    ok=1
    for p in nimsem hexer; do
      if [ -f "$E2E_OUT/$p.temen" ] && validate "$E2E_OUT/$p.temen"; then
        gzip -9 -c "$E2E_OUT/$p.temen" > "browser/web/assets/$p.temen.gz"
      else
        ok=0
      fi
    done
    [ "$ok" = 1 ] && note "nim_phases ✓ (nimsem.temen.gz + hexer.temen.gz)" \
                  || note "nim_phases ✗ (a phase guest missing/failed re-validate)"
  else
    note "nim_phases SKIP/✗ (nimony toolchain — nim + nimony/bin/{nimony,hexer} + clang/llvm-nm?)"
  fi
fi

# --- 6c) nimsem driver-guest fixtures (crates/temen-llvm/tests/rust_driver_nimsem.rs): the step-9 guest
# op-13-spawns child-entry nimsem over the system import closure. build_frontend.sh (TEMEN_NIMSEM_EMIT_
# ASSET=1) rebuilds nimsem_ce.temen.gz + syslib.tar.gz + sysvq0asl.{p,s}.nif together. Toolchain-gated. -
if want nim_driver_guest; then
  echo "=== [nim_driver_guest] build_frontend.sh (emit) → nimsem_ce + syslib + sys.{p,s}.nif fixtures ==="
  FX=crates/temen-run/demos/nim_frontend/fixtures
  if TEMEN_NIMSEM_EMIT_ASSET=1 bash crates/temen-run/demos/nim_frontend/build_frontend.sh >/dev/null 2>&1 \
     && [ -f "$FX/nimsem_ce.temen.gz" ] && gunzip -c "$FX/nimsem_ce.temen.gz" > /tmp/rebuild_nimsem_ce.temen 2>/dev/null \
     && validate /tmp/rebuild_nimsem_ce.temen; then
    note "nim_driver_guest ✓ (nimsem_ce.temen.gz + syslib.tar.gz + sysvq0asl.{p,s}.nif)"
  else
    note "nim_driver_guest SKIP/✗ (nimony toolchain — see build_frontend.sh; then refresh the expected via the test)"
  fi
fi

# --- 7) lua_snapshot.temen (Lua 5.4.7 core+libs + the two-phase snapshot harness → translate) -------
# The warm-runtime-snapshot Lua card asset (issue #805). Lua source is fetched-and-cached; skipped
# offline. Fixtures live in crates/temen-llvm/tests/fixtures/lua; recipe mirrors that dir's README.
if want lua_snapshot; then
  echo "=== [lua_snapshot] Lua 5.4.7 core+libs + snapshot harness → temen-llvm-translate ==="
  LFIX="$REPO/crates/temen-llvm/tests/fixtures/lua"
  LDEMOS="$REPO/crates/temen-run/demos"
  TR="$REPO/crates/temen-llvm/target/release/temen-llvm-translate"
  LCACHE="${TEMEN_LUA_CACHE:-/tmp/temen_lua_snap}"; mkdir -p "$LCACHE"
  if [ ! -d "$LCACHE/lua-5.4.7/src" ]; then
    ( cd "$LCACHE" && curl -sSL https://www.lua.org/ftp/lua-5.4.7.tar.gz -o lua.tgz && tar xzf lua.tgz ) \
      || note "lua_snapshot SKIP (Lua 5.4.7 fetch failed — offline?)"
  fi
  if [ -d "$LCACHE/lua-5.4.7/src" ]; then
    ( set -e; cd "$LCACHE"
      NV="-fno-vectorize -fno-slp-vectorize"
      CORE="lapi lcode lctype ldebug ldo ldump lfunc lgc llex lmem lobject lopcodes lparser lstate lstring ltable ltm lundump lvm lzio"
      LIBS="lbaselib lstrlib ltablib lmathlib lauxlib lcorolib liolib loslib"
      for f in $CORE $LIBS; do clang -O2 $NV -emit-llvm -S -Ilua-5.4.7/src lua-5.4.7/src/$f.c -o $f.ll; done
      clang -O2 $NV              -emit-llvm -S -Ilua-5.4.7/src "$LFIX/lua_snapshot_harness.c" -o harness.ll
      clang -O2 $NV -fno-builtin -emit-llvm -S -Ilua-5.4.7/src "$LFIX/lua_files_stdio.c" -o guest_stdio.ll
      clang -O2 $NV -fno-builtin -emit-llvm -S -Ilua-5.4.7/src "$LFIX/lua_files_time.c"  -o guest_time.ll
      clang -O2 $NV -fno-builtin -emit-llvm -S -Ilua-5.4.7/src "$LFIX/lua_files_shim.c"  -o guest_shim.ll
      clang -O2 $NV -fno-builtin -fno-strict-aliasing -emit-llvm -S "$LFIX/lua_testsuite_trig.c" -o guest_trig.ll
      clang -O2 $NV -fno-builtin -emit-llvm -S "$LFIX/lua_fmt_snprintf.c" -o guest_snprintf.ll
      clang -O2 $NV -fno-builtin -emit-llvm -S "$LDEMOS/libm/libm.c"     -o guest_libm.ll
      clang -O2 $NV -fno-builtin -emit-llvm -S "$LDEMOS/strtod/strtod.c" -o guest_strtod.ll
      CORELL=""; for f in $CORE $LIBS; do CORELL="$CORELL $f.ll"; done
      llvm-link -S $CORELL harness.ll guest_stdio.ll guest_time.ll guest_shim.ll guest_trig.ll \
                guest_snprintf.ll guest_libm.ll guest_strtod.ll -o lua_snapshot.ll
      "$TR" lua_snapshot.ll -o "$REPO/browser/web/assets/lua_snapshot.temen" --host-page 65536 --null-guard
    ) && { validate browser/web/assets/lua_snapshot.temen \
             && note "lua_snapshot ✓" || note "lua_snapshot ✗ (rebuilt but failed re-validate)"; } \
       || note "lua_snapshot ✗ (clang/llvm-link over the Lua core?)"
  fi
fi

echo
echo "=== rebuild-assets summary ==="
for r in "${RESULTS[@]}"; do echo "  $r"; done
echo
echo "Also (non-CI, but tracked) browser/tests/fixtures/*.temen — the display/reactor/onramp Rust-test"
echo "fixtures — are clang -O2 + temen-llvm-translate --host-page 65536 (+--null-guard for the #964"
echo "guarded ones: hello_onramp/bounce/life/mandelzoom; plain for gradient/fsread). shell/stage_runner/"
echo "primes/upper come from the [shell] step above."
echo "Then: git add the changed browser/web/assets/*.temen(.gz) + fixtures, and commit."
