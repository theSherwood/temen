#!/usr/bin/env bash
# NIM.md Phase 1, step 2 — compile a real Nim program with the stock Nim compiler's ARC backend,
# on-ramp its C to Temen IR, and diff the Temen run (interp + JIT) against a native build. This is the
# genuine-Nim-codegen counterpart to build_probe.sh (which uses hand-modeled Nim-shaped C): it
# exercises the actual Nim runtime (ARC ref objects, seqs) that nimony shares, standing in for
# nimony's Leng->C output until a nimony bootstrap is available (NIM.md §2 status).
#
#   needs: nim (2.x), clang-18/llvm-link-18/llvm-nm-18, cargo. Fail-soft SKIP if nim is absent
#          (the toolchain-absent pattern — CI without a Nim install skips, doesn't fail).
#
# Key Nim flags (measured, NIM.md §2):
#   --mm:arc -d:useMalloc   ARC memory management + malloc allocator (no tracing GC to port)
#   -d:noSignalHandler      don't install SIGSEGV/etc. handlers at startup — otherwise Nim calls
#                           signal() during init, which stubs to a trap (the one gotcha found)
#   -d:danger --panics:on   drop range/overflow checks that would pull extra runtime; panics abort
#   --threads:off           single-threaded guest
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
CACHE="${TEMEN_NIM_CACHE:-/tmp/temen_nim_cache}"
CLANG="${CLANG:-clang-18}"; command -v "$CLANG" >/dev/null || CLANG=clang
LINK="${LLVM_LINK:-llvm-link-18}"; command -v "$LINK" >/dev/null || LINK=llvm-link
NM="${LLVM_NM:-llvm-nm-18}"; command -v "$NM" >/dev/null || NM=llvm-nm
SRC="$HERE/list_seq.nim"
mkdir -p "$CACHE"

if ! command -v nim >/dev/null; then
  echo "SKIP: nim not on PATH (install Nim 2.x via choosenim to run this demo)"; exit 0
fi
NIMLIB="$(nim dump 2>/dev/null | grep -m1 '/lib$' || true)"
[ -n "$NIMLIB" ] && [ -f "$NIMLIB/nimbase.h" ] || NIMLIB="$(dirname "$(command -v nim)")/../../.choosenim/toolchains/$(nim --version 2>/dev/null | sed -n '1s/.*Version \([0-9.]*\).*/nim-\1/p')/lib"
[ -f "$NIMLIB/nimbase.h" ] || { echo "cannot locate nimbase.h (NIMLIB=$NIMLIB)"; exit 1; }
NIMFLAGS="--mm:arc -d:useMalloc -d:danger --panics:on --threads:off -d:noSignalHandler"

echo "=== [1/6] native reference (nim c -r) ==="
nim c $NIMFLAGS -o:"$CACHE/nat" --nimcache:"$CACHE/nat_c" "$SRC" >/dev/null 2>&1
"$CACHE/nat" > "$CACHE/native.out"; NATIVE_EXIT=$?
echo "native exit=$NATIVE_EXIT"; sed 's/^/  native| /' "$CACHE/native.out"

echo "=== [2/6] Nim -> C (--compileOnly, ARC) ==="
rm -rf "$CACHE/c"; nim c $NIMFLAGS --compileOnly --nimcache:"$CACHE/c" -o:"$CACHE/prog" "$SRC" >/dev/null 2>&1
echo "  $(ls "$CACHE"/c/*.c | wc -l) C TUs, $(cat "$CACHE"/c/*.c | wc -l) lines"

echo "=== [3/6] C -> bitcode -> link ==="
for f in "$CACHE"/c/*.c; do "$CLANG" -O2 -emit-llvm -c -I"$NIMLIB" "$f" -o "$f.bc"; done
"$LINK" "$CACHE"/c/*.bc -o "$CACHE/prog.linked.bc"

echo "=== [4/6] stub audit — undefined symbols must all be on-ramp-recognized ==="
"$NM" --undefined-only "$CACHE/prog.linked.bc" | awk '{print $2}' | sort -u > "$CACHE/undef.txt"
echo "  undefined: $(tr '\n' ' ' < "$CACHE/undef.txt")"
# The Nim-runtime bottom edge, all served by the on-ramp recognizer / stdio family (LLVM.md N/O/X)
# or trivial: malloc/free/realloc, fwrite/fflush/stderr, exit, strlen. Anything else is a finding.
ONRAMP='^(malloc|calloc|realloc|free|exit|_exit|_Exit|fwrite|fflush|fputc|putc|putchar|puts|fputs|write|read|stderr|stdout|stdin|strlen|memcpy|memmove|memset|memcmp|bcmp|__vm_[a-z_0-9]+|__temen_[a-z_0-9]+)$'
if UNEXPECTED=$(grep -Ev "$ONRAMP" "$CACHE/undef.txt"); then
  echo "  NOTE: not-yet-covered externs (stubbed; would trap if reached): $(echo $UNEXPECTED)"
  echo "  (e.g. raise/signal — Nim's fatal path; avoided by -d:noSignalHandler + no panic on success)"
fi

echo "=== [5/6] translate (--stub-externs) + prep_temen gate ==="
TR="$REPO/crates/temen-llvm/target/release/temen-llvm-translate"
[ -x "$TR" ] || cargo build --release --bin temen-llvm-translate --manifest-path "$REPO/crates/temen-llvm/Cargo.toml"
"$TR" "$CACHE/prog.linked.bc" -o "$CACHE/prog_raw.temen" --binary --host-page 65536 --stub-externs
cargo run -q --release -p temen-run --example prep_temen -- "$CACHE/prog_raw.temen" "$CACHE/prog.temen" | grep funcs

echo "=== [6/6] run on Temen (treewalk, bytecode, jit) and diff vs native ==="
fail=0
for eng in treewalk bytecode jit; do
  set +e
  TEMEN_PROBE_BACKEND="$eng" cargo run -q --release -p temen-run --example nimony_probe -- \
    "$CACHE/prog.temen" > "$CACHE/temen_$eng.out" 2>"$CACHE/temen_$eng.err"; TEMEN_EXIT=$?
  set -e
  if diff -q "$CACHE/native.out" "$CACHE/temen_$eng.out" >/dev/null && [ "$TEMEN_EXIT" = "$NATIVE_EXIT" ]; then
    echo "  $eng: OK (stdout byte-identical, exit=$TEMEN_EXIT)"
  else
    echo "  $eng: MISMATCH (exit temen=$TEMEN_EXIT native=$NATIVE_EXIT)"; diff "$CACHE/native.out" "$CACHE/temen_$eng.out" | sed 's/^/    /' || true
    head -c 200 "$CACHE/temen_$eng.err" | sed 's/^/    err| /'; fail=1
  fi
done
[ "$fail" = 0 ] && echo "ALL ENGINES MATCH NATIVE (real Nim ARC codegen on Temen)" || { echo "FAILED"; exit 1; }
