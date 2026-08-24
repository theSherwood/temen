#!/usr/bin/env bash
# The full nimony compiler on the Temen — Nim source → a module that RUNS (NIM.md §3c/§3e, the
# "compile Nim in the browser" capstone). This closes the chain the earlier slices left in pieces:
# every nimony phase runs as a real Temen guest over a shared in-window `fs`, and the compiled program
# then executes to the correct value.
#
#   prog.nim ─nifler──▶ .p.nif ─┐
#   stdlib   ─nifler(exec child)▶ .p.nif        (nimsem shells out to nifler, routed to the exec cap)
#                               └─nimsem──▶ .s.nif ─hexer──▶ .x.nif (Leng) ─┐
#                                                                            temen_leng::link → run
#
# nifler (parse), nimsem (sema, driving nifler via `exec`), and hexer (lower) each run as sandboxed
# `.temen` guests; only the final multi-module IR link is the embedder's Rust (`temen_leng`), the same
# host-drives-the-phases shape as the browser cards. The linked module runs on both engines (§9
# interp/JIT parity) and its exported proc returns the expected value — proof the compiler runs on the
# Temen *and its output runs too*.
#
#   needs: the nimony submodule, stock nim (2.3.x), the built nimony/nifler/nimsem/hexer binaries,
#          clang-18/llvm-18, cargo. Fail-soft SKIP if absent (NIM.md §2). The .temen are build artifacts
#          (nifler ~17 MB, nimsem ~5.5 MB, hexer ~3 MB) — not committed; this script IS the gate.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
CACHE="${TEMEN_E2E_CACHE:-/tmp/temen_e2e_chain}"
OUT="$CACHE/temen"
SHIM="$REPO/crates/temen-run/demos/nifler_temen/nifler_shim.c"
NIMLIB_FALLBACK="$REPO/.nimtool/nim-src/lib"
mkdir -p "$OUT"

if ! command -v nim >/dev/null; then echo "SKIP: nim not on PATH"; exit 0; fi
[ -f "$REPO/nimony/src/nimony/nimsem.nim" ] || { echo "SKIP: nimony submodule absent"; exit 0; }
BIN="${NIMONY_TOOLCHAIN_BIN:-$REPO/.nimtool/nimony/bin}"
NIMONY="${NIMONY_BIN:-$(command -v nimony || echo "$BIN/nimony")}"
[ -x "$NIMONY" ] || { echo "SKIP: nimony binary absent (NIM.md §2)"; exit 0; }
CLANG="${CLANG:-clang-18}"; command -v "$CLANG" >/dev/null || CLANG=clang
LINK=llvm-link-18; command -v "$LINK" >/dev/null || LINK=llvm-link
TR="$REPO/crates/temen-llvm/target/release/temen-llvm-translate"
[ -x "$TR" ] || cargo build --release --bin temen-llvm-translate --manifest-path "$REPO/crates/temen-llvm/Cargo.toml"
NIMLIB="$(nim dump 2>/dev/null | grep -m1 '/lib$' || true)"; [ -f "$NIMLIB/nimbase.h" ] || NIMLIB="$NIMLIB_FALLBACK"
export PATH="$(dirname "$NIMONY"):$PATH"

# ---- build a phase .temen from a nimony tool source (shared shim, --stub-externs) ------------------
build_temen() { # <tool-src> <out.temen> [extra nim defines...]
  local src="$1" out="$2"; shift 2
  [ -f "$out" ] && { echo "  reuse $(basename "$out") ($(stat -c%s "$out") B)"; return; }
  local c="$CACHE/c_$(basename "$src" .nim)"; rm -rf "$c"; mkdir -p "$c"
  nim c --mm:arc -d:useMalloc -d:danger --panics:on --threads:off -d:noSignalHandler \
    --warningAsError:ProveInit:off --warningAsError:Uninit:off "$@" \
    --compileOnly --nimcache:"$c" --path:"$REPO/nimony/src" -o:/dev/null "$src" >/dev/null 2>&1
  for f in "$c"/*.c; do "$CLANG" -O2 -fno-vectorize -fno-slp-vectorize -emit-llvm -c -I"$NIMLIB" "$f" -o "$f.bc"; done
  "$CLANG" -O2 -fno-vectorize -fno-slp-vectorize -DTEMEN_GUEST -emit-llvm -c -I"$NIMLIB" "$SHIM" -o "$c/shim.bc"
  "$LINK" "$c"/*.c.bc "$c/shim.bc" -o "$c/linked.bc"
  "$TR" "$c/linked.bc" -o "$c/raw.temen" --binary --host-page 65536 --stub-externs
  cargo run -q --release -p temen-run --example prep_temen -- "$c/raw.temen" "$out" > "$c/prep.log" 2>&1; grep -q funcs "$c/prep.log"
  rm -rf "$c"  # free disk immediately (each nim c cache is ~50-80 MB)
  echo "  $(basename "$out"): $(stat -c%s "$out") B"
}

echo "=== [1/2] build the phase guests (nifler + nimsem + hexer) ==="
build_temen "$REPO/nimony/src/nifler/nifler.nim"  "$OUT/nifler.temen"
build_temen "$REPO/nimony/src/nimony/nimsem.nim"  "$OUT/nimsem.temen" -d:skipPostSemValidator
build_temen "$REPO/nimony/src/hexer/hexer.nim"    "$OUT/hexer.temen"

echo "=== [2/2] compile each Nim fixture through the guests → link → run, check the result ==="
# Each fixture is a directory of `.nim` sources (`$w/prog.nim` is the main); the driver reads nimony's
# own `<main>.build.nif` plan, so it handles any number of modules. `run_fixture <name> <export>
# <expected> [args...]` runs after the fixture's sources are written to `$w`.
run_fixture() { # <name> <export> <expected> [args...]  (sources already written to $CACHE/proj_<name>)
  local name="$1" export="$2" expected="$3"; shift 3
  local w="$CACHE/proj_$name"
  ( cd "$w" && "$NIMONY" c --isMain prog.nim >/dev/null 2>&1 )   # native bootstrap: parse layout + plan
  local main
  main="$(ls "$w"/nimcache/*.build.nif | grep -v '\.final\.' | head -1 | xargs -n1 basename | sed 's/\.build\.nif//')"
  # Strip the sema/lowering products so the driver re-produces them on the Temen (keep .p.nif + .build.nif).
  rm -f "$w"/nimcache/*.s.nif "$w"/nimcache/*.x.nif "$w"/nimcache/*.s.idx.nif
  find "$w/nimcache" -mindepth 1 -type d -exec rm -rf {} + 2>/dev/null || true
  echo "--- $name: $export(...) == $expected (main=$main) ---"
  cargo run -q --release -p temen-run --example nim_e2e_chain -- \
    "$OUT/nifler.temen" "$OUT/nimsem.temen" "$OUT/hexer.temen" \
    "$BIN/../lib" "$w/nimcache" "$main" "$export" "$expected" "$@"
}
write_main() { rm -rf "$CACHE/proj_$1"; mkdir -p "$CACHE/proj_$1"; printf '%s' "$2" > "$CACHE/proj_$1/prog.nim"; }

# Single-module compute programs (program + `system`).
write_main addtwo 'proc addTwo(a, b: int): int = a + b
let r = addTwo(2, 3)
'
run_fixture addtwo addTwo 5 2 3

write_main sumto 'proc sumTo(n: int): int =
  result = 0
  var i = 1
  while i <= n:
    result = result + i
    i = i + 1

let r = sumTo(5)
'
run_fixture sumto sumTo 15 5

# Multi-module: a user `import ./helper` — three units (system + helper + main), dependency-ordered by
# nimony's build plan, each compiled through the guests on the Temen and linked together.
write_main usermod 'import ./helper
proc useit(n: int): int = triple(n) + 1
let r = useit(4)
'
printf 'proc triple*(x: int): int = x * 3\n' > "$CACHE/proj_usermod/helper.nim"
run_fixture usermod useit 13 4

# Real I/O: a program that `import std/syncio` and `write`s (four units: system + syncio + its dep +
# main). `<io>` mode links through the nim→powerbox bridge and runs `_start` under the powerbox, so the
# guest's `write(fd,buf,len)` reaches the STREAM `write` cap — real **stdout**, checked against the
# expected string (`\n` decoded). (`echo` is not yet an identifier nimony resolves — a frontend gap.)
write_main iohello 'import std/syncio
write(stdout, "hello, temen\n")
'
run_fixture iohello '<io>' 'hello, temen\n'

echo "=== [3/3] the same, with NO native bootstrap (nim_selfdrive — the in-browser card's core) ==="
# `nim_e2e_chain` above leans on native `nimony c` to lay out the nimcache (stems + .build.nif plan).
# `nim_selfdrive` needs none of that: given only the Nim source + stdlib it plays nifmake itself —
# computes each module's stem (gear2/modnames), crawls the `import` graph with nifler.temen, then runs
# nimsem/hexer and links + runs. This is the exact logic the browser cdylib will run (#958); proving it
# here headless de-risks the browser port. Same fixtures, source-only.
selfdrive() { # <name> <main.nim-relative-in-$w> <export> <expected> [args...]
  local name="$1" export="$3" expected="$4"; shift 4
  local w="$CACHE/proj_$name"   # reuse the fixture dirs written above (sources only)
  echo "--- selfdrive $name: $export(...) == $expected (no native bootstrap) ---"
  cargo run -q --release -p temen-run --example nim_selfdrive -- \
    "$OUT/nifler.temen" "$OUT/nimsem.temen" "$OUT/hexer.temen" "$BIN/../lib" "$w" prog.nim \
    "$export" "$expected" "$@"
}
selfdrive addtwo prog.nim addTwo 5 2 3
selfdrive usermod prog.nim useit 13 4
selfdrive iohello prog.nim '<io>' 'hello, temen\n'

echo "ALL FIXTURES RAN — the full nimony compiler runs on the Temen (single- and multi-module, with and without a native bootstrap), and its compiled output runs too"
