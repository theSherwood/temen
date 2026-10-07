#!/usr/bin/env bash
# Rebuild **every committed `.temen` playground/self-host asset** in one pass — the single entry point
# for the "a wire-format / encoder / IR change invalidated the prebuilt binary assets" chore (which
# recurs on every such change: the committed modules decode as `BadOpcode` under the new format and
# their asset-gate tests — leng_selfhost_asset, nim_hello_asset, browser/tests/nimony.rs, and the
# real-browser play cards — go red until regenerated).
#
# This orchestrates the existing per-asset builders (it does NOT reimplement them) and, crucially,
# wires up the toolchain env each one expects — the tribal knowledge that otherwise gets rediscovered
# by hand every time: NIMONY_BIN / NIM_BIN for the nimony steps (what scripts/ci/provision-nimony.sh
# exports), reusing the vendored `nimony/bin` when present and a real Nim toolchain dir over a bare
# `nim` shim.
#
# Every step is **fail-soft**: a missing toolchain SKIPs that asset (matching each builder's own
# contract) so a partial environment still rebuilds what it can. Each rebuilt module is re-validated
# (decode → verify → bytecode-compile) via `prep_temen`. A final summary lists ✓ / SKIP / ✗.
#
#   Usage:  bash scripts/rebuild-assets.sh              # rebuild everything the toolchain allows
#           ONLY=leng,nim_hello bash scripts/...        # rebuild a subset (comma-separated step names)
#   Steps:  leng chibicc pg_libc pg_heap coop_grow onramp shell coreutils forth uxn nim_hello
#           nim_card lua_snapshot
#
# Toolchains, per step: leng needs rustc (+rust-src) & llvm; chibicc/onramp/pg_heap need clang &
# llvm-link (onramp also fetches QuickJS/SQLite/Lua sources — skipped offline); shell needs the
# in-tree chibicc; nim_hello & nim_card need the nimony toolchain (Nim + nimony/bin, as
# scripts/ci/provision-nimony.sh builds it).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

ONLY="${ONLY:-}"
want() { [ -z "$ONLY" ] || [[ ",$ONLY," == *",$1,"* ]]; }
declare -a RESULTS=()
note() { RESULTS+=("$1"); echo "  >> $1"; }

# --- toolchain env: the setup each nimony step assumes (see the header) ------------------------------
# Prefer a real Nim toolchain dir (its `lib/` beside `bin/`) over a bare `nim` shim.
pick_nim() {
  local c
  for c in \
    "$(command -v nim 2>/dev/null)" \
    "$REPO"/.nimtool/*/bin/nim \
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
if [ -x "$REPO/nimony/bin/nimony" ]; then
  export PATH="$REPO/nimony/bin:$PATH"
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

# --- 2b) pg_libc.temeno (the prebuilt seeded-libc unit, #1392) ---------------------------------------
# Self-hosted, no toolchain: runs the *committed* chibicc.temen (step 2's output) over
# `browser/playground-include/__pg_libc.c` through the on-ramp powerbox, and encodes the emitted
# object as a linkable unit. It is doubly wire-coupled (the chibicc asset it reads, the unit it
# writes), so it must be regenerated on any IR / encoder / wire change — `browser/tests/pg_libc_asset.rs`
# is the gate. `genlibc` re-decodes what it writes (exports + debug info), so no `validate` here:
# prep_temen's powerbox assertion does not apply to a library unit.
if want pg_libc; then
  echo "=== [pg_libc] browser: cargo run --bin genlibc (chibicc.temen over __pg_libc.c) ==="
  ( cd "$REPO/browser" && cargo run --release --bin genlibc ) \
    && note "pg_libc ✓ (web/assets/pg_libc.temeno)" \
    || note "pg_libc ✗ (chibicc.temen decodable? see output above)"
fi

# --- 2b') pg_heap.temeno (the playground's C heap, #2172) ------------------------------------------
# clang, then `temen-llvm-translate --link-unit`: dlmalloc in the LLVM on-ramp's configuration
# (`crates/temen-llvm/dlmalloc/temen_dlmalloc.c`) plus the playground's `sbrk` and `free()` checks
# (`browser/playground-heap/pg_heap.c`). Every C card program links it beside pg_libc.temeno. A link
# unit does not run alone, so no `validate`: `browser/tests/pg_libc_asset.rs` links it and runs it.
if want pg_heap; then
  echo "=== [pg_heap] clang + temen-llvm-translate --link-unit (browser/playground-heap/pg_heap.c) ==="
  T="$(mktemp -d)"
  if clang -O2 -g0 -fno-strict-aliasing -fno-vectorize -fno-slp-vectorize -Icrates/temen-llvm/dlmalloc \
       -emit-llvm -S browser/playground-heap/pg_heap.c -o "$T/pg_heap.ll" \
     && crates/temen-llvm/target/release/temen-llvm-translate "$T/pg_heap.ll" --link-unit \
       --host-page 65536 -o browser/web/assets/pg_heap.temeno; then
    note "pg_heap ✓ (web/assets/pg_heap.temeno)"
  else
    note "pg_heap SKIP/✗ (clang / temen-llvm-translate?)"
  fi
  rm -rf "$T"
fi

# --- 2c) coop_grow_past_window.temen (the #1312 coop-grow JS gate's guest, generated text IR) ---------
# No toolchain: `genfixture` writes the hand-built guest `browser-coop-grow-test.mjs` runs.
if want coop_grow; then
  echo "=== [coop_grow] browser: cargo run --bin genfixture (grow_past_window) ==="
  F=browser/tests/fixtures/coop_grow_past_window.temen
  ( cd "$REPO/browser" && cargo run --release --bin genfixture -- "$REPO/$F" grow_past_window ) \
    && validate "$F" \
    && note "coop_grow ✓ ($F)" \
    || note "coop_grow ✗ (see output above)"
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

# --- 4a) bin_*.temen (the 28 repo-owned coreutils bash runs from /bin — #1080 slice 2) ---------------
# Same toolchain as the shell fixtures (in-tree chibicc, no external deps); `browser/tests/bash.rs`
# and `browser/build-bash-assets.mjs` read them from `browser/tests/fixtures/`.
if want coreutils; then
  echo "=== [coreutils] cargo test -p temen --test c_shell -- --ignored gen_browser_bash_coreutils ==="
  cargo test -p temen --test c_shell -- --ignored --exact gen_browser_bash_coreutils \
    && note "coreutils ✓ (browser/tests/fixtures/bin_*.temen)" \
    || note "coreutils ✗ (in-tree chibicc?)"
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

# --- 4c) uxn.temen + uxn_demo.rom (crates/temen-run/demos/uxn: cc for the assembler, clang → translate) ---
if want uxn; then
  echo "=== [uxn] crates/temen-run/demos/uxn/build.sh → browser/web/assets/{uxn.temen,uxn_demo.rom} ==="
  UXN_OUT="${TEMEN_UXN_CACHE:-/tmp/temen_uxn_cache}"
  if sh crates/temen-run/demos/uxn/build.sh "$UXN_OUT"; then
    cp "$UXN_OUT/uxn.temen" browser/web/assets/uxn.temen
    cp "$UXN_OUT/uxn_demo.rom" browser/web/assets/uxn_demo.rom
    validate browser/web/assets/uxn.temen \
      && note "uxn ✓ (uxn.temen + uxn_demo.rom)" || note "uxn ✗ (rebuilt but failed re-validate)"
  else
    note "uxn SKIP/✗ (cc / clang / translator?)"
  fi
fi

# --- 5) nim_hello.temen (nimony → temen-leng powerbox bridge) ----------------------------------------
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

# --- 6) nimony.blob.gz (the nim card's toolchain, #958): nimony's own tools, each built by nimony with
# no C compiler (scripts/nim-toolchain.sh — the self-hosted lane's), with nimony's library and that
# library prebuilt. `nimbuild --bundle` builds the program below with them at `/nim`, where the card
# builds, and writes what it ran with plus the library pack the build left. Nothing in the blob is
# wire-coupled but the tools, and all of it is rebuilt together, so the pack always matches them. -----
if want nim_card; then
  echo "=== [nim_card] scripts/nim-toolchain.sh + nimbuild --bundle → web/assets/nimony.blob.gz ==="
  T="$(mktemp -d)"
  mkdir -p "$T/card"
  # The library a playground program reaches for, compiled once so a build compiles only its own
  # modules; the `const` makes compile-time evaluation build its helper too. `std/macros` brings the
  # plugins its grammar declares (parsegen, regex), which the pack carries built, so a program's
  # macro builds only its own plugin, in the program's cache (#2049). `std/json` costs the pack
  # 0.45 MB gzipped and saves a json program two-thirds of its build (#2070).
  cat >"$T/card/prelude.nim" <<'NIM'
import std/[syncio, strutils, sequtils, tables, sets, hashes, algorithm, math, options, deques,
  parseutils, bitops, intsets, macros, json]

proc prelude(s: string): int = s.len
const atCompileTime = prelude("prebuilt")
echo "prelude ", atCompileTime
NIM
  # The card's tree is nimony's library and nimony's own sources, which the library imports: `std/json`
  # imports `src/lib`, `std/macros` reaches the nifler2 grammar, and a plugin's build puts `src/lib`
  # and `src/nimony/lib` on its path (#2033). All of `src/` (+1.4 MB gzipped), not a list of what those
  # imports reach today, which drifts whenever upstream's does. browser/tests/nimony.rs checks the
  # library's relative imports resolve.
  if [ -n "${NIMONY_BIN:-}" ] && [ -n "${NIM_BIN:-}" ] && bash scripts/nim-toolchain.sh "$T" \
     && ln -s "$T/nimony/lib" "$T/card/lib" && ln -s "$T/nimony/src" "$T/card/src" \
     && ( cd browser && cargo build --release -q --bin nimbuild ) \
     && browser/target/release/nimbuild "$T" "$T/card" prelude.nim --at /nim --leaves --bundle "$T/nimony.blob" \
     && gzip -9 -n -c "$T/nimony.blob" > browser/web/assets/nimony.blob.gz; then
    note "nim_card ✓ (nimony.blob.gz — gated by browser/tests/nimony.rs)"
  else
    note "nim_card SKIP/✗ (nimony toolchain — NIMONY_BIN/NIM_BIN; see scripts/nim-toolchain.sh)"
  fi
  rm -rf "$T"
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
# A SKIP here is easy to read as "not applicable" when it actually means "this asset is now STALE and
# nothing regenerated it" — which is silent until CI fails on a byte-comparison gate. That happened
# on the v0.6.2 bump with the in-guest linker assets, which embedded temen-leng and so went stale
# whenever the linker changed, wire format or not. Call the skipped steps out again, separately, with
# what unblocks each.
SKIPPED=()
for r in "${RESULTS[@]}"; do case "$r" in *SKIP*|*✗*) SKIPPED+=("$r");; esac; done
if [ "${#SKIPPED[@]}" -gt 0 ]; then
  echo "!!! ${#SKIPPED[@]} step(s) did NOT regenerate — each may now be STALE:"
  for r in "${SKIPPED[@]}"; do echo "    $r"; done
  echo
  echo "    These are not advisory. An asset that embeds compiled code (temen-leng.temen embeds"
  echo "    temen-leng) goes stale on any change to what it embeds, and the gate that catches it is a"
  echo "    byte-comparison in CI, not here."
  echo "    The LLVM steps need the LLVM whose major matches rustc's on PATH"
  echo "    (scripts/ci/install-llvm.sh). CI puts it there via"
  echo "    GITHUB_PATH; locally nothing does, so if the distro\'s unversioned llvm-link is older"
  echo "    than rustc\'s LLVM (\`rustc -vV | grep LLVM\`) prefix the run with"
  echo "      PATH=/usr/lib/llvm-\$(grep -oP \'LLVM_MAJOR=\\K[0-9]+\' scripts/ci/install-llvm.sh)/bin:\$PATH"
  echo "    A mismatch reads as a *parse* error on rustc\'s own IR (\`expected \')\' at end of"
  echo "    argument list\` on an attribute the older tool has never heard of), not as a version"
  echo "    complaint \u2014 which is why the note above used to blame a missing toolchain."
  echo
fi
echo "Also (non-CI, but tracked) browser/tests/fixtures/*.temen — the display/reactor/onramp Rust-test"
echo "fixtures — are clang -O2 + temen-llvm-translate --host-page 65536 (+--null-guard for the #964"
echo "guarded ones: hello_onramp/bounce/life/mandelzoom; plain for gradient/fsread). shell/stage_runner/"
echo "primes/upper come from the [shell] step above."
echo "Then: git add the changed browser/web/assets/*.temen(.gz) + fixtures, and commit."
