#!/usr/bin/env bash
# The self-hosted nimony lane (#1609, #1668) as a CHECK: nimsem, compiled to Temen with no C compiler,
# semchecks `system` while spawning its own nifler for every dependency it has no parse of — fork →
# `execve("/bin/sh", ["-c", "bin/nifler …"])` → the shell execs nifler in place → nimsem reaps it — and
# the result must be native nimony's (`nim_noc_semcheck --expect`; only recorded paths may differ).
#
#   NIMONY_BIN=<nimony/bin> NIM_BIN=<dir holding nim> bash scripts/ci/nim-selfhost-lane.sh
#
# Release-mode minutes — building nimsem (~30 s) and the lane itself (~2 min on the tree-walker) — so
# it is a step of its own rather than a `cargo test`. The per-PR gate over the same mechanisms, at a
# size a debug test run can afford, is nim_e2e's `nim_shells_out_through_the_posix_sh`.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
: "${NIMONY_BIN:?set NIMONY_BIN to the directory holding nimony}"
: "${NIM_BIN:?set NIM_BIN to the directory holding nim}"
export PATH="$NIMONY_BIN:$NIM_BIN:$PATH"
W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT

echo "[1/4] the native reference: nimony semchecks system for an empty program"
# Outside the repo on purpose: native nimony records each source path relative to its cwd, and the
# comparison rewrites `…/<nimony root>/lib/std/` — a cwd inside the tree would spell it differently.
mkdir -p "$W/proj"
echo 'discard' >"$W/proj/prog.nim"
(cd "$W/proj" && nimony c prog.nim >/dev/null)
SYS_PNIF="$(ls "$W"/proj/nimcache/sys*.p.nif | head -1)"
STEM="$(basename "$SYS_PNIF" .p.nif)"
test -f "$W/proj/nimcache/$STEM.s.nif" || { echo "native nimony wrote no $STEM.s.nif" >&2; exit 1; }

echo "[2/4] the compiler phases, through the POSIX edge"
cargo build --release -q -p temen-run --example build_nim_hello_temen --example nim_noc_semcheck
B=target/release/examples
"$B/build_nim_hello_temen" --posix --root nimony src/nifler2/nifler2.nim "$W/nifler2.temen"
"$B/build_nim_hello_temen" --posix --root nimony src/nimony/nimsem.nim "$W/nimsem.temen"

echo "[3/4] /bin/sh: the POSIX build of demos/shell"
make -s -C frontend/chibicc
D=crates/temen-run/demos/shell
cat "$D/shim.c" "$D/ring.c" "$D/shell_main.c" >"$W/sh.c"
frontend/chibicc/chibicc -cc1 --emit-ir --child-entry -DTEMEN_SHELL_POSIX \
  -cc1-input "$W/sh.c" -cc1-output "$W/sh.ir" "$W/sh.c"

echo "[4/4] the lane: nimsem on Temen, spawning its own nifler"
"$B/nim_noc_semcheck" --sh "$W/sh.ir" --expect "$W/proj/nimcache/$STEM.s.nif" \
  "$W/nimsem.temen" "$W/nifler2.temen" nimony/lib "$SYS_PNIF" "$STEM" "$W/out.s.nif"
