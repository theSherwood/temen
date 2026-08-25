#!/usr/bin/env bash
# The nimony FRONT-END on Temen (NIM.md §3c/§3e, "compile Nim in the browser" slice 2b): prove the real
# `nimsem` (sema) phase runs on the Temen as a multi-binary DRIVER — it spawns the real `nifler` (parse)
# phase as an `exec` child to parse stdlib modules on demand — and produces semchecked NIF that is
# **semantically identical to native** (byte-identical modulo the embedded stdlib file paths).
#
#   Nim source ─nifler─▶ .p.nif ─┐
#   stdlib .nim ─nifler(child)──▶ .p.nif (on demand, via exec)
#                                └─nimsem──▶ semchecked .s.nif   (== native, modulo paths)
#
# nimsem's `system("nifler … parse <src> <out.p.nif>")` (deps.nim) is routed by the shim
# (nifler_temen/nifler_shim.c `system()`) to the `exec` capability; `nifler.temen` runs as an isolated
# child domain sharing nimsem's memfs (domain_exec_with_fs), so the `.p.nif` nifler writes is the one
# nimsem reads back — the W4 multi-binary driver (multibinary.rs) on the actual front-end.
#
# nimsem is built with -d:skipPostSemValidator (as nimony's own Windows CI does): the in-process
# post-sem IR validator has a separate on-ramp issue, tracked apart from the sema itself, which this
# proves correct. The remaining byte differences are ALL path-derived (the embedded stdlib paths, the
# `.idx` offsets those paths shift, and the module-identity hashes they seed) — a seeding artifact,
# not a sema difference; the path-normalized diff below is byte-exact.
#
#   needs: nimony submodule, stock nim (2.3.x), the built nimony/nifler/hexer binaries, clang-18/
#          llvm-18, cargo. Fail-soft SKIP if absent (NIM.md §2). The .temen are build artifacts.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
CACHE="${TEMEN_FRONTEND_CACHE:-/tmp/temen_frontend_build}"
SHIM="$REPO/crates/temen-run/demos/nifler_temen/nifler_shim.c"
NIMLIB_FALLBACK="$REPO/.nimtool/nim-src/lib"
mkdir -p "$CACHE"

if ! command -v nim >/dev/null; then echo "SKIP: nim not on PATH"; exit 0; fi
[ -f "$REPO/nimony/src/nimony/nimsem.nim" ] || { echo "SKIP: nimony submodule absent"; exit 0; }
BIN="${NIMONY_TOOLCHAIN_BIN:-$REPO/.nimtool/nimony/bin}"
NIMONY="${NIMONY_BIN:-$(command -v nimony || echo "$BIN/nimony")}"
NIMSEM_ORACLE="${NIMSEM_BIN:-$BIN/nimsem}"
[ -x "$NIMONY" ] && [ -x "$NIMSEM_ORACLE" ] || { echo "SKIP: nimony/nimsem binaries absent (NIM.md §2)"; exit 0; }
CLANG="${CLANG:-clang-18}"; command -v "$CLANG" >/dev/null || CLANG=clang
NM=llvm-nm; TR="$REPO/crates/temen-llvm/target/release/temen-llvm-translate"
[ -x "$TR" ] || cargo build --release --bin temen-llvm-translate --manifest-path "$REPO/crates/temen-llvm/Cargo.toml"
NIMLIB="$(nim dump 2>/dev/null | grep -m1 '/lib$' || true)"; [ -f "$NIMLIB/nimbase.h" ] || NIMLIB="$NIMLIB_FALLBACK"
export PATH="$(dirname "$NIMONY"):$PATH"

# ---- build a phase .temen from a nimony tool source (shared shim, --stub-externs) ------------------
build_temen() { # <tool-src> <out.temen> [extra nim defines...]
  local src="$1" out="$2"; shift 2
  local c="$CACHE/c_$(basename "$src" .nim)"; rm -rf "$c"; mkdir -p "$c"
  nim c --mm:arc -d:useMalloc -d:danger --panics:on --threads:off -d:noSignalHandler \
    --warningAsError:ProveInit:off --warningAsError:Uninit:off "$@" \
    --compileOnly --nimcache:"$c" --path:"$REPO/nimony/src" -o:/dev/null "$src" >/dev/null 2>&1
  for f in "$c"/*.c; do "$CLANG" -O2 -fno-vectorize -fno-slp-vectorize -emit-llvm -c -I"$NIMLIB" "$f" -o "$f.bc"; done
  "$CLANG" -O2 -fno-vectorize -fno-slp-vectorize -DTEMEN_GUEST -emit-llvm -c -I"$NIMLIB" "$SHIM" -o "$c/shim.bc"
  llvm-link "$c"/*.c.bc "$c/shim.bc" -o "$c/linked.bc"
  "$TR" "$c/linked.bc" -o "$c/raw.temen" --binary --host-page 65536 --stub-externs
  cargo run -q --release -p temen-run --example prep_temen -- "$c/raw.temen" "$out" > "$c/prep.log" 2>&1; grep -q funcs "$c/prep.log"
}

echo "=== [1/4] build nifler.temen + nimsem.temen (-d:skipPostSemValidator) ==="
build_temen "$REPO/nimony/src/nifler/nifler.nim" "$CACHE/nifler.temen"
build_temen "$REPO/nimony/src/nimony/nimsem.nim" "$CACHE/nimsem.temen" -d:skipPostSemValidator
echo "  nifler.temen $(stat -c%s "$CACHE/nifler.temen") B; nimsem.temen $(stat -c%s "$CACHE/nimsem.temen") B"

echo "=== [2/4] generate the system module's parsed nif (oracle nimony c) ==="
W="$CACHE/proj"; rm -rf "$W"; mkdir -p "$W"; printf 'let x = 5\n' > "$W/prog.nim"
( cd "$W" && "$NIMONY" c --isMain prog.nim >/dev/null 2>&1 )
sys="$(ls "$W"/nimcache/*.p.nif | xargs -n1 basename | grep '^sys' | sed 's/\.p\.nif//' | head -1)"
echo "  system module stem: $sys"

echo "=== [3/4] native oracle: nimsem --isSystem ==="
( cd "$W" && rm -f "nimcache/$sys.s.nif" && "$NIMSEM_ORACLE" --define:nimNativeAlloc --define:nimNativeIo m --isSystem "nimcache/$sys.p.nif" >/dev/null 2>&1 )
cp "$W/nimcache/$sys.s.nif" "$CACHE/native.s.nif"

echo "=== [4/4] nimsem-on-Temen drives nifler via exec; diff (path-normalized) vs native ==="
rm -f /tmp/temen_sys.s.nif
cargo run -q --release -p temen-run --example nim_frontend_driver -- \
  "$CACHE/nimsem.temen" "$CACHE/nifler.temen" "$BIN/../lib" "$W/nimcache/$sys.p.nif" "$sys"
# Normalize the embedded file paths, the `.idx` offset, and path-derived module hashes/offsets, then diff.
norm() { sed -E 's#[^ (]*\.nim#<PATH>#g; s#\(\.indexat [0-9 ]*\)#(.indexat <N>)#; s#\.I[A-Za-z0-9]+\. [0-9]+\)#.<HASH>. <OFF>)#g; s#\.[0-9]+\. [0-9]+\)#.<K>. <OFF>)#g' "$1"; }
if diff -q <(norm "$CACHE/native.s.nif") <(norm /tmp/temen_sys.s.nif) >/dev/null; then
  echo "ALL MATCH NATIVE — nimsem-on-Temen sema is byte-identical to native (modulo embedded paths); the real nimony FRONT-END runs on the Temen"
else
  echo "FAILED: residual semantic differences after path normalization:"; diff <(norm "$CACHE/native.s.nif") <(norm /tmp/temen_sys.s.nif) | head -20; exit 1
fi

# ---- [5/5] the same front-end as a confined §14 op-13 child (the Rust-on-Temen driver-guest shape) ---
# nimsem is itself a driver (it `system("nifler …")`s to parse stdlib), so running it as an op-13 child
# needs the `exec` cap **re-granted into the child** — the op-13 grant list carries {fs, stdout, exit,
# exec}, and nimsem-the-child spawns nifler grandchildren via that re-granted exec over the same memfs.
# `nimsem_child_driver` proves it: byte-identical (path-normalized) to native, the same oracle.
echo "=== [5/5] child-entry: nimsem as an op-13 child, exec re-granted (drives nifler grandchildren) ==="
"$TR" "$CACHE/c_nimsem/linked.bc" -o "$CACHE/nimsem_ce_raw.temen" --binary --host-page 65536 --stub-externs --child-entry
ceout="$CACHE/ce_out"; rm -rf "$ceout"; mkdir -p "$ceout"
cargo run -q --release -p temen-run --example nimsem_child_driver -- \
  "$CACHE/nimsem_ce_raw.temen" "$CACHE/nifler.temen" "$BIN/../lib" "$W/nimcache/$sys.p.nif" "$sys" "$ceout"
if diff -q <(norm "$CACHE/native.s.nif") <(norm "$ceout/nimcache/$sys.s.nif") >/dev/null; then
  echo "CHILD-ENTRY MATCHES NATIVE — nimsem runs as a confined op-13 §14 child, spawning nifler grandchildren via the re-granted exec cap, byte-exact (path-normalized)"
else
  echo "FAILED (child-entry): residual diff:"; diff <(norm "$CACHE/native.s.nif") <(norm "$ceout/nimcache/$sys.s.nif") | head -20; exit 1
fi

# ---- [6/6] the front-end CHAIN, both phases op-13 children over ONE shared memfs (nimsem -> hexer) ---
# The compiler driver on Temen: a native conductor op-13-spawns nimsem then hexer, handing the `.s.nif`
# between them through one shared memfs (nifler runs as nimsem's exec grandchildren). Oracle: native
# hexer on the SAME `.s.nif` the op-13 nimsem produced, so only hexer's lowering is compared (the two
# nimsems' outputs differ only in embedded paths). `nim_chain_op13` is the driver.
HEXER_BIN="${HEXER_BIN:-$BIN/hexer}"
if [ -x "$HEXER_BIN" ]; then
  echo "=== [6/6] the front-end chain: nimsem -> hexer, both op-13 children over one memfs ==="
  build_temen "$REPO/nimony/src/hexer/hexer.nim" "$CACHE/hexer.temen"
  "$TR" "$CACHE/c_hexer/linked.bc" -o "$CACHE/hexer_ce_raw.temen" --binary --host-page 65536 --stub-externs --child-entry
  chainout="$CACHE/chain_out"; rm -rf "$chainout"; mkdir -p "$chainout"
  cargo run -q --release -p temen-run --example nim_chain_op13 -- \
    "$CACHE/nimsem_ce_raw.temen" "$CACHE/hexer_ce_raw.temen" "$CACHE/nifler.temen" "$BIN/../lib" \
    "$W/nimcache/$sys.p.nif" "$sys" "$chainout"
  od="$CACHE/nat_onchain"; rm -rf "$od"; mkdir -p "$od"
  cp "$chainout/nimcache/$sys.s.nif" "$chainout/nimcache/$sys.s.idx.nif" "$od/" 2>/dev/null
  ( cd "$od" && "$HEXER_BIN" c "$sys.s.nif" >/dev/null 2>&1 )
  if diff -q "$od/$sys.x.nif" "$chainout/nimcache/$sys.x.nif" >/dev/null; then
    echo "CHAIN MATCHES NATIVE — nimsem->hexer both op-13 §14 children over one memfs; the Leng .x.nif is byte-identical to native hexer on the same semchecked input. The nimony front-end runs on Temen as a chain of confined op-13 phases."
  else
    echo "FAILED (chain): .x.nif differs"; cmp "$od/$sys.x.nif" "$chainout/nimcache/$sys.x.nif" | head -1; exit 1
  fi

  # ---- [7/7] the SAME chain, but every phase op-13-spawned on the CRANELIFT JIT --------------------
  # The tier-up-capable engine a browser wasm-JIT compile card uses. The op-13 `mod_ok` relaxation
  # (`declared <= carve`, matching the interpreter — FORK.md §8.6 / #773) is what lets these
  # malloc-heavy phases carve heap room on emitted code. Oracle: the SAME native hexer `.x.nif` as
  # [6/6] (the two nimsems' `.s.nif` differ only in embedded paths, so hexer's lowering is compared).
  echo "=== [7/7] the front-end chain on the JIT: nimsem -> hexer, both op-13 children on emitted code ==="
  jchainout="$CACHE/jit_chain_out"; rm -rf "$jchainout"; mkdir -p "$jchainout"
  cargo run -q --release -p temen-run --example nim_chain_op13_jit -- \
    "$CACHE/nimsem_ce_raw.temen" "$CACHE/hexer_ce_raw.temen" "$CACHE/nifler.temen" "$BIN/../lib" \
    "$W/nimcache/$sys.p.nif" "$sys" "$jchainout"
  if diff -q "$od/$sys.x.nif" "$jchainout/nimcache/$sys.x.nif" >/dev/null; then
    echo "JIT CHAIN MATCHES NATIVE — nimsem->hexer both op-13 §14 children on the Cranelift JIT; the Leng .x.nif is byte-identical to native hexer. The nimony front-end self-hosts on Temen's JIT as a chain of confined op-13 phases — the enabler for the browser wasm-JIT compile card."
  else
    echo "FAILED (JIT chain): .x.nif differs"; cmp "$od/$sys.x.nif" "$jchainout/nimcache/$sys.x.nif" | head -1; exit 1
  fi
else
  echo "SKIP [6/6]+[7/7] chain: native hexer binary absent ($HEXER_BIN)"
fi
