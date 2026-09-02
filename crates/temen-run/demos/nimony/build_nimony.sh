#!/usr/bin/env bash
# NIM.md Phase 1, step 2 (proper) — compile a program with NIMONY's own C backend and on-ramp its
# genuine Leng->C output to Temen, running it there against nimony's native run. This is the
# authentic pipeline (nifler -> nimony -> hexer -> Leng -> lengc c), distinct from build_nim.sh
# (which uses the *stock* Nim compiler as a stand-in).
#
#   needs: a bootstrapped nimony toolchain (bin/nimony), clang/llvm-*, cargo. Fail-soft SKIP
#          if nimony is absent — it is NOT in CI (bootstrap needs Nim 2.3.x devel; see NIM.md §2).
#
# Point NIMONY_BIN at the nimony binary (or put it on PATH):
#   NIMONY_BIN=/path/to/nimony/bin/nimony  bash build_nimony.sh
#
# Two normalizations are applied to nimony's C output for the single-threaded on-ramp; both are
# documented findings (NIM.md §2), not hacks that hide problems:
#   1. `__thread` stripped — nimony puts the allocator/exception globals in TLS; a single-threaded
#      guest makes them plain globals. (The on-ramp has no llvm.threadlocal.address lowering.)
#   2. nimony_runtime_shim.c supplies the libc-free bottom edge nimony references but the on-ramp
#      doesn't recognize: mmap (a page-aligned bump allocator — the load-bearing one), plus
#      getpid/kill/dlopen/dlsym/_exit on the fatal/loader paths. write/exit are on-ramp-recognized.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
CACHE="${TEMEN_NIMONY_CACHE:-/tmp/temen_nimony_demo}"
CLANG="${CLANG:-clang}"
LINK="${LLVM_LINK:-llvm-link}"
NM="${LLVM_NM:-llvm-nm}"
SRC="$HERE/sum_sq_nimony.nim"
SHIM="$HERE/nimony_runtime_shim.c"
mkdir -p "$CACHE"

NIMONY="${NIMONY_BIN:-$(command -v nimony || true)}"
if [ -z "$NIMONY" ] || [ ! -x "$NIMONY" ]; then
  echo "SKIP: nimony not found (set NIMONY_BIN; bootstrap needs Nim 2.3.x devel — NIM.md §2)"; exit 0
fi
echo "nimony: $NIMONY"

echo "=== [1/6] nimony native run (oracle) + emit Leng->C ==="
rm -rf "$CACHE/c"
# `nimony c -r` compiles via lengc's C backend AND runs the result — capture that as the oracle.
"$NIMONY" c -r -d:danger --nimcache:"$CACHE/c" "$SRC" > "$CACHE/native.out" 2>"$CACHE/native.err" || {
  echo "nimony compile/run failed:"; tail -5 "$CACHE/native.err"; exit 1; }
NATIVE_EXIT=0
echo "native exit=$NATIVE_EXIT"; sed 's/^/  native| /' "$CACHE/native.out"
CDIR="$(dirname "$(find "$CACHE/c" -name '*.c' | head -1)")"
echo "  Leng->C: $(ls "$CDIR"/*.c | wc -l) TUs, $(cat "$CDIR"/*.c | wc -l) lines"

echo "=== [2/6] normalize TLS for single-threaded on-ramp (strip __thread) ==="
# Strip the direct `__thread` qualifier nimony puts on its allocator/exception globals, and
# neutralize the NIM_THREADVAR macro's __thread/_Thread_local expansions (belt-and-suspenders:
# nimony emits __thread directly today, but the macro exists for portability).
for f in "$CDIR"/*.c; do
  sed -i -E 's/^__thread //; s/ __thread / /g; s/(define NIM_THREADVAR) __thread$/\1/; s/(define NIM_THREADVAR) _Thread_local$/\1/' "$f"
done
# Verify no __thread remains as an actual declaration qualifier (macro-def/comment lines are fine).
if grep -rn '__thread' "$CDIR"/*.c | grep -vE '(define NIM_THREADVAR|declaration based on)' >/dev/null; then
  echo "residual __thread qualifier remains:"; grep -rn '__thread' "$CDIR"/*.c | grep -vE '(define NIM_THREADVAR|declaration based on)' | sed 's/^/  /'; exit 1
fi
echo "  __thread neutralized"

echo "=== [3/6] C + runtime shim -> bitcode -> link ==="
for f in "$CDIR"/*.c; do "$CLANG" -O2 -emit-llvm -c "$f" -o "$f.bc"; done
"$CLANG" -O2 -emit-llvm -c "$SHIM" -o "$CACHE/shim.bc"
"$LINK" "$CDIR"/*.bc "$CACHE/shim.bc" -o "$CACHE/nimony.linked.bc"

echo "=== [4/6] symbol audit — undefined must be only on-ramp-recognized ==="
"$NM" --undefined-only "$CACHE/nimony.linked.bc" | awk '{print $2}' | sort -u > "$CACHE/undef.txt"
echo "  undefined: $(tr '\n' ' ' < "$CACHE/undef.txt")"
ONRAMP='^(write|read|exit|_exit|__vm_[a-z_0-9]+|__temen_[a-z_0-9]+)$'
if UNEXPECTED=$(grep -Ev "$ONRAMP" "$CACHE/undef.txt"); then
  echo "  UNEXPECTED undefined (shim gap): $(echo $UNEXPECTED)"; exit 1
fi

echo "=== [5/6] translate (no --stub-externs) + prep_temen gate ==="
TR="$REPO/crates/temen-llvm/target/release/temen-llvm-translate"
[ -x "$TR" ] || cargo build --release --bin temen-llvm-translate --manifest-path "$REPO/crates/temen-llvm/Cargo.toml"
"$TR" "$CACHE/nimony.linked.bc" -o "$CACHE/nimony_raw.temen" --binary --host-page 65536
cargo run -q --release -p temen-run --example prep_temen -- "$CACHE/nimony_raw.temen" "$CACHE/nimony.temen" | grep funcs

echo "=== [6/6] run genuine nimony output on Temen (treewalk, bytecode, jit) and diff vs native ==="
fail=0
for eng in treewalk bytecode jit; do
  set +e
  TEMEN_PROBE_BACKEND="$eng" cargo run -q --release -p temen-run --example nimony_probe -- \
    "$CACHE/nimony.temen" > "$CACHE/temen_$eng.out" 2>"$CACHE/temen_$eng.err"; TEMEN_EXIT=$?
  set -e
  if diff -q "$CACHE/native.out" "$CACHE/temen_$eng.out" >/dev/null && [ "$TEMEN_EXIT" = "$NATIVE_EXIT" ]; then
    echo "  $eng: OK (stdout byte-identical to native nimony, exit=$TEMEN_EXIT)"
  else
    echo "  $eng: MISMATCH (exit temen=$TEMEN_EXIT native=$NATIVE_EXIT)"; diff "$CACHE/native.out" "$CACHE/temen_$eng.out" | sed 's/^/    /' || true
    head -c 200 "$CACHE/temen_$eng.err" | sed 's/^/    err| /'; fail=1
  fi
done
[ "$fail" = 0 ] && echo "ALL ENGINES MATCH NATIVE NIMONY (genuine Leng->C output runs on Temen)" || { echo "FAILED"; exit 1; }
