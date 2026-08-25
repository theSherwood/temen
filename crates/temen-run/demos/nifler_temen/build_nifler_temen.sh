#!/usr/bin/env bash
# nifler-on-Temen (NIM.md §3c/§3e, "nimony in the browser" slice 1) — compile the FIRST real nimony
# phase, `nifler` (Nim source → NIF), all the way to a runnable Temen module and prove it parses a Nim
# file **byte-for-byte identically to native nifler**, across all three engines.
#
# This is the C-on-ramp path (NIM.md Phase 1) applied to a real compiler phase, not a user program:
#   nifler.nim --(stock nim c, ARC)--> C --(clang-18)--> bitcode --(link nifler_shim.c)-->
#   whole-program bitcode --(temen-llvm-translate --stub-externs)--> Temen --(prep_temen: verify)--> .temen
# then run `nifler p in.nim out.nif` as a guest over an in-window `fs` memfs seeded with the source,
# and diff the emitted `.nif` against native nifler on the same source.
#
#   needs: the nimony submodule (nimony/src/nifler), stock `nim` (2.3.x, for the C backend), the built
#          `nifler` binary (the oracle — bin/nifler), clang-18/llvm-link-18/llvm-nm-18, cargo. Fail-soft
#          SKIP if any is absent (nimony bootstrap needs Nim 2.3.x devel; NOT in the per-PR CI, NIM.md §2).
#
# The bottom edge (nifler_shim.c) reuses the shims that already run Postgres on Temen (../postgres/
# os_shim.c + stdio_shim.c) over the `fs` cap; see that file. The ~17 MB `.temen` is a build artifact,
# NOT committed (too big for an asset lane) — this script IS the gate, run when the toolchain is present.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
CACHE="${TEMEN_NIFLER_CACHE:-/tmp/temen_nifler_build}"
CLANG="${CLANG:-clang-18}"; command -v "$CLANG" >/dev/null || CLANG=clang
LINK="${LLVM_LINK:-llvm-link-18}"; command -v "$LINK" >/dev/null || LINK=llvm-link
NM="${LLVM_NM:-llvm-nm-18}"; command -v "$NM" >/dev/null || NM=llvm-nm
NIFLER_SRC="$REPO/nimony/src/nifler/nifler.nim"
mkdir -p "$CACHE"

# --- toolchain presence (fail-soft SKIP, the toolchain-absent pattern) -------------------------------
if ! command -v nim >/dev/null; then echo "SKIP: nim not on PATH (Nim 2.3.x for the C backend)"; exit 0; fi
[ -f "$NIFLER_SRC" ] || { echo "SKIP: nimony submodule absent ($NIFLER_SRC) — git submodule update --init nimony"; exit 0; }
NIFLER_BIN="${NIFLER_BIN:-$(command -v nifler || true)}"
[ -n "$NIFLER_BIN" ] || NIFLER_BIN="$REPO/.nimtool/nimony/bin/nifler"
[ -x "$NIFLER_BIN" ] || { echo "SKIP: native nifler binary absent (set NIFLER_BIN; build the nimony toolchain — NIM.md §2)"; exit 0; }
NIMLIB="$(nim dump 2>/dev/null | grep -m1 '/lib$' || true)"
[ -f "$NIMLIB/nimbase.h" ] || NIMLIB="$(dirname "$(command -v nim)")/../lib"
[ -f "$NIMLIB/nimbase.h" ] || { echo "cannot locate nimbase.h (NIMLIB=$NIMLIB)"; exit 1; }
echo "nim: $(command -v nim) | nifler(oracle): $NIFLER_BIN"

echo "=== [1/5] nifler.nim -> C (stock nim, ARC, single-threaded) ==="
rm -rf "$CACHE/c"
# Same flags as build_nim.sh: ARC + malloc, no tracing GC, no signal handlers, danger. --vectorize off
# is applied at clang below (clang SLP-vectorizes a few scalar divisions in platform.nim -> a vector
# DivS the on-ramp doesn't lower; the scalar form does).
nim c --mm:arc -d:useMalloc -d:danger --panics:on --threads:off -d:noSignalHandler \
  --compileOnly --nimcache:"$CACHE/c" --path:"$REPO/nimony/src" -o:"$CACHE/nifler_stub" \
  "$NIFLER_SRC" >/dev/null 2>&1
echo "  $(ls "$CACHE"/c/*.c | wc -l) C TUs, $(cat "$CACHE"/c/*.c | wc -l) lines"

echo "=== [2/5] C -> bitcode (-O2, vectorizers off) + link with the fs-cap shim ==="
for f in "$CACHE"/c/*.c; do "$CLANG" -O2 -fno-vectorize -fno-slp-vectorize -emit-llvm -c -I"$NIMLIB" "$f" -o "$f.bc"; done
"$CLANG" -O2 -fno-vectorize -fno-slp-vectorize -DTEMEN_GUEST -emit-llvm -c -I"$NIMLIB" "$HERE/nifler_shim.c" -o "$CACHE/nifler_shim.bc"
"$LINK" "$CACHE"/c/*.c.bc "$CACHE/nifler_shim.bc" -o "$CACHE/nifler.linked.bc"

echo "=== [3/5] stub audit — residual undefs must be on-ramp builtins / math / unreached spawn fringe ==="
"$NM" --undefined-only "$CACHE/nifler.linked.bc" | awk '{print $2}' | sort -u > "$CACHE/undef.txt"
# on-ramp host builtins + recognized libc + math (unlowered, unreached on parse) + spawn/system fringe.
OK='^(__vm_[a-z_0-9]+|__temen_[a-z_0-9]+|malloc|calloc|realloc|free|exit|memcpy|memmove|memset|memcmp|bcmp|memchr|memmem|strlen|strchr|strtod|snprintf|strerror|acos|acosh|asin|asinh|atan|atan2|atanh|cbrt|cos|cosh|erf|erfc|exp|fmod|hypot|ldexp|lgamma|log|log10|log2|pow|sin|sinh|sqrt|tan|tanh|tgamma|chmod|glob|globfree|ioctl|kill|mmap|munmap|pipe|posix_fadvise|posix_fallocate|posix_spawn.*|readlink|sigemptyset|symlink|system|waitpid|lldiv)$'
if UNEXPECTED=$(grep -Ev "$OK" "$CACHE/undef.txt"); then
  echo "  UNEXPECTED residual externs (a finding — extend nifler_shim.c): $(echo $UNEXPECTED)"; exit 1
fi
echo "  residual undefs all accounted for ($(wc -l < "$CACHE/undef.txt"))"

echo "=== [4/5] translate (--stub-externs) + prep_temen (verify) ==="
TR="$REPO/crates/temen-llvm/target/release/temen-llvm-translate"
[ -x "$TR" ] || cargo build --release --bin temen-llvm-translate --manifest-path "$REPO/crates/temen-llvm/Cargo.toml"
"$TR" "$CACHE/nifler.linked.bc" -o "$CACHE/nifler_raw.temen" --binary --host-page 65536 --stub-externs
cargo run -q --release -p temen-run --example prep_temen -- "$CACHE/nifler_raw.temen" "$CACHE/nifler.temen" | grep -E "funcs|wrote"

echo "=== [5/5] run nifler-on-Temen over each input, diff vs native (treewalk, bytecode, jit) ==="
fail=0
for src in "$HERE"/inputs/*.nim; do
  name="$(basename "$src")"
  # native oracle: run in a scratch dir on a file named in.nim so the embedded (portable) path matches.
  rm -rf "$CACHE/nat"; mkdir -p "$CACHE/nat"; cp "$src" "$CACHE/nat/in.nim"
  ( cd "$CACHE/nat" && "$NIFLER_BIN" p in.nim out.nif >/dev/null 2>&1 )
  for eng in treewalk bytecode jit; do
    set +e
    TEMEN_NIFLER_BACKEND="$eng" cargo run -q --release -p temen-run --example nifler_run -- \
      "$CACHE/nifler.temen" "$src" "/out.nif" > "$CACHE/temen_${name}_$eng.nif" 2>"$CACHE/temen_${name}_$eng.err"; rc=$?
    set -e
    if [ "$rc" = 0 ] && diff -q "$CACHE/nat/out.nif" "$CACHE/temen_${name}_$eng.nif" >/dev/null; then
      echo "  $name [$eng]: OK (byte-identical, $(stat -c%s "$CACHE/temen_${name}_$eng.nif") bytes)"
    else
      echo "  $name [$eng]: MISMATCH (exit=$rc)"; head -c 200 "$CACHE/temen_${name}_$eng.err" | sed 's/^/    /'; fail=1
    fi
  done
done
[ "$fail" = 0 ] && echo "ALL MATCH NATIVE — a real nimony phase (nifler) parses Nim on the Temen, byte-exact" || { echo "FAILED"; exit 1; }

# --- [5b] child-entry: op-13-spawn nifler as a confined §14 child over a shared memfs -----------------
# The compiler-driver shape (NIM.md §3c, W5): instead of running nifler as a top-level powerbox program,
# translate it `--child-entry` (func 0 becomes the starter->i64-status child ABI) and drive it through
# `instantiate_module` (op 13) — argv `nifler p /in.nim /out.nif` seeded into the child's carve, a shared
# `mem_fs` re-granted as `fs`, `stdout`/`exit` for its imports — then read the emitted `.nif` back out of
# the shared store. This is the exact hand-off the Rust-on-Temen driver guest uses to fan phases out; here
# it proves a real phase produces byte-identical NIF as an op-13 child (`examples/spawn_child_fs.rs`).
echo "=== [5b] translate --child-entry + op-13-spawn over memfs, diff vs native ==="
"$TR" "$CACHE/nifler.linked.bc" -o "$CACHE/nifler_ce_raw.temen" --binary --host-page 65536 --stub-externs --child-entry
for src in "$HERE"/inputs/*.nim; do
  name="$(basename "$src")"
  rm -rf "$CACHE/nat_ce"; mkdir -p "$CACHE/nat_ce"; cp "$src" "$CACHE/nat_ce/in.nim"
  ( cd "$CACHE/nat_ce" && "$NIFLER_BIN" p in.nim out.nif >/dev/null 2>&1 )
  # The op-13 driver seeds a fixture dir into the memfs and dumps produced files: stage the source as
  # `in.nim`, run `nifler p /in.nim /out.nif`, collect `out.nif`.
  fix="$CACHE/ce_fix_$name"; ceout="$CACHE/ce_out_$name"; rm -rf "$fix" "$ceout"; mkdir -p "$fix"
  cp "$src" "$fix/in.nim"
  set +e
  cargo run -q --release -p temen-run --example spawn_child_fs -- \
    "$CACHE/nifler_ce_raw.temen" "$fix" "$ceout" -- nifler p /in.nim /out.nif 2>"$CACHE/ce_${name}.err"; rc=$?
  set -e
  if [ "$rc" = 0 ] && diff -q "$CACHE/nat_ce/out.nif" "$ceout/out.nif" >/dev/null; then
    echo "  $name [child-entry op-13]: OK (byte-identical, $(stat -c%s "$ceout/out.nif") bytes)"
  else
    echo "  $name [child-entry op-13]: MISMATCH (exit=$rc)"; head -c 300 "$CACHE/ce_${name}.err" | sed 's/^/    /'; fail=1
  fi
done
[ "$fail" = 0 ] && echo "CHILD-ENTRY MATCHES NATIVE — nifler runs as a confined op-13 §14 child, byte-exact" || { echo "FAILED"; exit 1; }

# --- [6/6] regenerate the committed slice-4 browser asset + gate fixtures (opt-in) -------------------
# The playground card (`browser/web/play.js`, kind 'nifler') loads `nifler.temen` **gzipped** (~3.8 MB
# vs ~17.7 MB raw; inflated client-side), and the toolchain-free gate `crates/temen-run/tests/nifler_asset.rs`
# compares the in-sandbox parse against `expected/*.p.nif` (verbatim native-nifler output). Both are
# committed. Regenerate them here with `TEMEN_NIFLER_EMIT_ASSET=1` when the module or a fixture drifts.
if [ "${TEMEN_NIFLER_EMIT_ASSET:-0}" = 1 ]; then
  echo "=== [6/6] emit committed asset + fixtures (TEMEN_NIFLER_EMIT_ASSET=1) ==="
  gzip -9 -c "$CACHE/nifler.temen" > "$REPO/browser/web/assets/nifler.temen.gz"
  echo "  browser/web/assets/nifler.temen.gz $(stat -c%s "$REPO/browser/web/assets/nifler.temen.gz") B (from $(stat -c%s "$CACHE/nifler.temen") B raw)"
  # The child-entry variant, for the op-13 toolchain-free gate (`tests/nifler_child_asset.rs`, if present).
  gzip -9 -c "$CACHE/nifler_ce_raw.temen" > "$HERE/nifler_ce.temen.gz"
  echo "  nifler_ce.temen.gz $(stat -c%s "$HERE/nifler_ce.temen.gz") B (from $(stat -c%s "$CACHE/nifler_ce_raw.temen") B raw)"
  mkdir -p "$HERE/expected"
  for src in "$HERE"/inputs/*.nim; do
    name="$(basename "$src" .nim)"
    # Native nifler over the source seeded as `in.nim` (the portable path the guest embeds too).
    rm -rf "$CACHE/nat"; mkdir -p "$CACHE/nat"; cp "$src" "$CACHE/nat/in.nim"
    ( cd "$CACHE/nat" && "$NIFLER_BIN" p in.nim out.nif >/dev/null 2>&1 )
    cp "$CACHE/nat/out.nif" "$HERE/expected/$name.p.nif"
    echo "  expected/$name.p.nif $(stat -c%s "$HERE/expected/$name.p.nif") B"
  done
  echo "  committed asset + fixtures regenerated — git add browser/web/assets/nifler.temen.gz $HERE/expected"
fi
