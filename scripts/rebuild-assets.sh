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
#   Steps:  leng chibicc onramp shell nifler nim_hello  (lua_snapshot is a separate warm-snapshot chore)
#
# Toolchains, per step: leng needs rustc +1.81 (+rust-src) & llvm-18; chibicc/onramp need clang-18 &
# llvm-link-18 (onramp also fetches QuickJS/SQLite/Lua sources — skipped offline); shell needs the
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
    note "leng SKIP/✗ (see output above — rustc +1.81 + rust-src + llvm-18?)"
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
    note "chibicc SKIP/✗ (clang-18 / llvm-link-18?)"
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

# --- 5) nifler.temen.gz (nimony pipeline; gzips into web/assets) -------------------------------------
if want nifler; then
  echo "=== [nifler] crates/temen-run/demos/nifler_temen/build_nifler_temen.sh ==="
  if bash crates/temen-run/demos/nifler_temen/build_nifler_temen.sh; then
    if gunzip -c browser/web/assets/nifler.temen.gz > /tmp/rebuild_nifler.temen 2>/dev/null \
       && validate /tmp/rebuild_nifler.temen; then
      note "nifler ✓ (nifler.temen.gz)"
    else
      note "nifler SKIP (toolchain absent — script SKIPs without rebuilding)"
    fi
  else
    note "nifler ✗ (nim + nimony/bin/nifler + clang-18/llvm-nm-18?)"
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

echo
echo "=== rebuild-assets summary ==="
for r in "${RESULTS[@]}"; do echo "  $r"; done
echo
echo "NOTE: lua_snapshot.temen is a separate two-phase warm-runtime-snapshot build"
echo "      (Lua 5.4.7 + the snapshot driver) — see build-onramp-assets.mjs §Lua; not automated here."
echo "Then: git add the changed browser/web/assets/*.temen(.gz), the leng/nifler fixtures, and commit."
