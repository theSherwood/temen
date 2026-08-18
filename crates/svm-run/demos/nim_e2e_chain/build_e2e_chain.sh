#!/usr/bin/env bash
# The full nimony compiler on the SVM — Nim source → a module that RUNS (NIM.md §3c/§3e, the
# "compile Nim in the browser" capstone). This closes the chain the earlier slices left in pieces:
# every nimony phase runs as a real SVM guest over a shared in-window `fs`, and the compiled program
# then executes to the correct value.
#
#   prog.nim ─nifler──▶ .p.nif ─┐
#   stdlib   ─nifler(exec child)▶ .p.nif        (nimsem shells out to nifler, routed to the exec cap)
#                               └─nimsem──▶ .s.nif ─hexer──▶ .x.nif (Leng) ─┐
#                                                                            svm_leng::link → run
#
# nifler (parse), nimsem (sema, driving nifler via `exec`), and hexer (lower) each run as sandboxed
# `.svmb` guests; only the final multi-module IR link is the embedder's Rust (`svm_leng`), the same
# host-drives-the-phases shape as the browser cards. The linked module runs on both engines (§9
# interp/JIT parity) and its exported proc returns the expected value — proof the compiler runs on the
# SVM *and its output runs too*.
#
#   needs: the nimony submodule, stock nim (2.3.x), the built nimony/nifler/nimsem/hexer binaries,
#          clang-18/llvm-18, cargo. Fail-soft SKIP if absent (NIM.md §2). The .svmb are build artifacts
#          (nifler ~17 MB, nimsem ~5.5 MB, hexer ~3 MB) — not committed; this script IS the gate.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
CACHE="${SVM_E2E_CACHE:-/tmp/svm_e2e_chain}"
OUT="$CACHE/svmb"
SHIM="$REPO/crates/svm-run/demos/nifler_svm/nifler_shim.c"
NIMLIB_FALLBACK="$REPO/.nimtool/nim-src/lib"
mkdir -p "$OUT"

if ! command -v nim >/dev/null; then echo "SKIP: nim not on PATH"; exit 0; fi
[ -f "$REPO/nimony/src/nimony/nimsem.nim" ] || { echo "SKIP: nimony submodule absent"; exit 0; }
BIN="${NIMONY_TOOLCHAIN_BIN:-$REPO/.nimtool/nimony/bin}"
NIMONY="${NIMONY_BIN:-$(command -v nimony || echo "$BIN/nimony")}"
[ -x "$NIMONY" ] || { echo "SKIP: nimony binary absent (NIM.md §2)"; exit 0; }
CLANG="${CLANG:-clang-18}"; command -v "$CLANG" >/dev/null || CLANG=clang
LINK=llvm-link-18; command -v "$LINK" >/dev/null || LINK=llvm-link
TR="$REPO/crates/svm-llvm/target/release/svm-llvm-translate"
[ -x "$TR" ] || cargo build --release --bin svm-llvm-translate --manifest-path "$REPO/crates/svm-llvm/Cargo.toml"
NIMLIB="$(nim dump 2>/dev/null | grep -m1 '/lib$' || true)"; [ -f "$NIMLIB/nimbase.h" ] || NIMLIB="$NIMLIB_FALLBACK"
export PATH="$(dirname "$NIMONY"):$PATH"

# ---- build a phase .svmb from a nimony tool source (shared shim, --stub-externs) ------------------
build_svmb() { # <tool-src> <out.svmb> [extra nim defines...]
  local src="$1" out="$2"; shift 2
  [ -f "$out" ] && { echo "  reuse $(basename "$out") ($(stat -c%s "$out") B)"; return; }
  local c="$CACHE/c_$(basename "$src" .nim)"; rm -rf "$c"; mkdir -p "$c"
  nim c --mm:arc -d:useMalloc -d:danger --panics:on --threads:off -d:noSignalHandler \
    --warningAsError:ProveInit:off --warningAsError:Uninit:off "$@" \
    --compileOnly --nimcache:"$c" --path:"$REPO/nimony/src" -o:/dev/null "$src" >/dev/null 2>&1
  for f in "$c"/*.c; do "$CLANG" -O2 -fno-vectorize -fno-slp-vectorize -emit-llvm -c -I"$NIMLIB" "$f" -o "$f.bc"; done
  "$CLANG" -O2 -fno-vectorize -fno-slp-vectorize -DSVM_GUEST -emit-llvm -c -I"$NIMLIB" "$SHIM" -o "$c/shim.bc"
  "$LINK" "$c"/*.c.bc "$c/shim.bc" -o "$c/linked.bc"
  "$TR" "$c/linked.bc" -o "$c/raw.svmb" --binary --host-page 65536 --stub-externs
  cargo run -q --release -p svm-run --example prep_svmb -- "$c/raw.svmb" "$out" > "$c/prep.log" 2>&1; grep -q funcs "$c/prep.log"
  rm -rf "$c"  # free disk immediately (each nim c cache is ~50-80 MB)
  echo "  $(basename "$out"): $(stat -c%s "$out") B"
}

echo "=== [1/2] build the phase guests (nifler + nimsem + hexer) ==="
build_svmb "$REPO/nimony/src/nifler/nifler.nim"  "$OUT/nifler.svmb"
build_svmb "$REPO/nimony/src/nimony/nimsem.nim"  "$OUT/nimsem.svmb" -d:skipPostSemValidator
build_svmb "$REPO/nimony/src/hexer/hexer.nim"    "$OUT/hexer.svmb"

echo "=== [2/2] compile each Nim fixture through the guests → link → run, check the result ==="
# Each fixture: a Nim program, the exported proc to call, the expected result, and its call args.
run_fixture() { # <name> <nim-source> <export> <expected> [args...]
  local name="$1" src="$2" export="$3" expected="$4"; shift 4
  local w="$CACHE/proj_$name"; rm -rf "$w"; mkdir -p "$w"
  printf '%s' "$src" > "$w/prog.nim"
  ( cd "$w" && "$NIMONY" c --isMain prog.nim >/dev/null 2>&1 )   # native bootstrap: parse layout + stems
  local sys prog
  sys="$(ls "$w"/nimcache/sysv*.s.nif | head -1 | xargs -n1 basename | sed 's/\.s\.nif//')"
  prog="$(ls "$w"/nimcache/*.s.nif | xargs -n1 basename | sed 's/\.s\.nif//' | grep -v '^sys' | head -1)"
  rm -f "$w"/nimcache/*.s.nif "$w"/nimcache/*.x.nif; rm -rf "$w/nimcache/$prog"  # re-produce on the SVM
  echo "--- $name: $export(...) == $expected (sys=$sys prog=$prog) ---"
  cargo run -q --release -p svm-run --example nim_e2e_chain -- \
    "$OUT/nifler.svmb" "$OUT/nimsem.svmb" "$OUT/hexer.svmb" \
    "$BIN/../lib" "$w/nimcache" "$sys" "$prog" "$export" "$expected" "$@"
}

run_fixture addtwo 'proc addTwo(a, b: int): int = a + b
let r = addTwo(2, 3)
' addTwo 5 2 3
run_fixture sumto 'proc sumTo(n: int): int =
  result = 0
  var i = 1
  while i <= n:
    result = result + i
    i = i + 1

let r = sumTo(5)
' sumTo 15 5

echo "ALL FIXTURES RAN — the full nimony compiler runs on the SVM, and its compiled output runs too"
