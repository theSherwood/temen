#!/usr/bin/env bash
# The local mirror of CI's **lint and compile** gates — run them here, on CI's toolchain, before a
# push costs a round-trip.
#
# WHY, not just what: a `cargo clippy` that passes locally and fails on CI is worse than no local
# check, because it teaches you to trust a signal that does not hold. That happened: a
# `clippy::for_kv_map` error failed two CI jobs while the same command was clean locally, purely
# because the local default toolchain was older than the one CI pins. Clippy gains lints between
# releases, so "clippy is clean" is only meaningful *with respect to a version*. This script pins
# that version to CI's, reading it from the workflow itself so the two cannot drift.
#
# It covers the checks that are cheap, deterministic and host-independent:
#
#   * `cargo fmt --all --check`                             (check job)
#   * the same in `crates/temen-llvm` (excluded crate)      (temen-llvm job)
#   * `cargo clippy --workspace --all-targets`              (check job)
#   * `cargo clippy -p temen-jit --features stack-check`    (fiber-scaling job — a SEPARATE lane; the
#                                                            gating check job builds without the
#                                                            feature, so nothing else compiles it)
#   * `scripts/ci/check-excluded.sh`                        (excluded-crates job)
#   * the same two under `--target x86_64-pc-windows-gnu`   (check + fiber-scaling cross-checks)
#
# It deliberately does NOT cover: the test matrix, miri/asan/tsan/loom, fuzz, the differential and
# browser jobs, or anything needing a real Windows/macOS runtime. Those stay CI's. This is the
# "did I break the build or the lints" gate, which is what actually costs round-trips.
#
# Usage:
#   scripts/ci/local-gate.sh              # run the gates
#   scripts/ci/local-gate.sh --print-toolchain   # just echo the rust version CI pins
#   TEMEN_GATE_INSTALL=1 …                # install the pinned toolchain/components/target if missing
#                                         # (otherwise a missing one fails with the exact command)
#   TEMEN_GATE_SKIP_CROSS=1 …             # skip the windows-gnu cross-check
#
# The pre-push hook (`.githooks/pre-push`) calls this, so the hook and a manual run are the same
# gate rather than two drifting lists.
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

# --- CI's pinned toolchain, read from CI ------------------------------------------------------
# `.github/workflows/` is what actually runs; `workflows_src/` is the editable mirror a CI check
# keeps equal to it (see `.github/workflows_src/README.md`). Prefer the real one, fall back to the
# mirror so this still works in a checkout where only the mirror was updated.
wf=""
for candidate in .github/workflows/ci.yml .github/workflows_src/ci.yml; do
  [ -f "$candidate" ] && { wf="$candidate"; break; }
done
if [ -z "$wf" ]; then
  echo "local-gate: cannot find ci.yml to read RUST_STABLE from" >&2
  exit 1
fi
toolchain="$(sed -nE 's/^[[:space:]]*RUST_STABLE:[[:space:]]*"?([^"[:space:]]+)"?[[:space:]]*$/\1/p' "$wf" | head -1)"
if [ -z "$toolchain" ]; then
  echo "local-gate: no RUST_STABLE in $wf — has the workflow's env block changed shape?" >&2
  exit 1
fi

# `--print-toolchain`: just report the version CI pins and exit. The pre-push hook uses this so its
# own build/test steps run on the same toolchain as the gates, without a second copy of the parser.
if [ "${1:-}" = "--print-toolchain" ]; then
  echo "$toolchain"
  exit 0
fi

echo "local-gate: CI pins rust $toolchain (from $wf)"

install_hint="rustup toolchain install $toolchain --profile minimal --component clippy --component rustfmt"

if ! rustup toolchain list | grep -q "^${toolchain}-"; then
  if [ "${TEMEN_GATE_INSTALL:-0}" = "1" ]; then
    echo "local-gate: installing rust $toolchain…"
    rustup toolchain install "$toolchain" --profile minimal --component clippy --component rustfmt
  else
    echo "local-gate: rust $toolchain is not installed — that is the version CI lints with." >&2
    echo "  $install_hint" >&2
    echo "  (or re-run with TEMEN_GATE_INSTALL=1 to install it automatically)" >&2
    exit 1
  fi
fi

# A toolchain installed without the components is the same trap one step later: `cargo fmt` would
# fall through to another toolchain's rustfmt, or fail confusingly mid-run.
for component in clippy rustfmt; do
  if ! rustup component list --toolchain "$toolchain" --installed | grep -q "^${component}"; then
    if [ "${TEMEN_GATE_INSTALL:-0}" = "1" ]; then
      rustup component add --toolchain "$toolchain" "$component"
    else
      echo "local-gate: rust $toolchain has no $component." >&2
      echo "  rustup component add --toolchain $toolchain $component" >&2
      exit 1
    fi
  fi
done

# Match CI's posture: warnings are hard errors (ci.yml sets RUSTFLAGS: "-D warnings" for the whole
# workflow, so it applies to `check`/`build`, not only to the explicit clippy `-D warnings`).
export RUSTFLAGS="${RUSTFLAGS:--D warnings}"

run() {
  echo "local-gate: \$ $*"
  "$@"
}
cargo_pinned() { run rustup run "$toolchain" cargo "$@"; }

# --- host gates -------------------------------------------------------------------------------
cargo_pinned fmt --all --check
# `--all` stops at the workspace; the temen-llvm job fmt-checks that excluded crate on its own.
cargo_pinned fmt --all --check --manifest-path crates/temen-llvm/Cargo.toml
cargo_pinned clippy --workspace --all-targets -- -D warnings
# The `stack-check` feature has its own CI lane because the gating check job builds WITHOUT it, so
# this is the only place its feature-gated code is compiled at all. It went red once for a lint the
# workspace run could not have seen.
cargo_pinned clippy -p temen-jit --features stack-check --all-targets -- -D warnings
# The four crates the root manifest `exclude`s are invisible to every `--workspace` command above.
# Same script CI runs; it reads the exclude list from `Cargo.toml`, so it cannot miss a crate.
TEMEN_CHECK_TOOLCHAIN="$toolchain" run bash scripts/ci/check-excluded.sh

# --- windows-gnu cross-check ------------------------------------------------------------------
# CI cross-checks both the workspace and the stack-check lane against x86_64-pc-windows-gnu. It is a
# compile-only gate, so it runs anywhere the target and a mingw linker are available — which is the
# cheapest way to catch a Windows-only build break without a Windows runner. The §5 trap-capture C
# shim (`trap_capture.c`) is what needs the cross gcc.
cross_target=x86_64-pc-windows-gnu
if [ "${TEMEN_GATE_SKIP_CROSS:-0}" = "1" ]; then
  echo "local-gate: SKIP windows-gnu cross-check (TEMEN_GATE_SKIP_CROSS=1)"
elif ! command -v x86_64-w64-mingw32-gcc >/dev/null 2>&1; then
  echo "local-gate: SKIP windows-gnu cross-check — no mingw cross-compiler."
  echo "  install it with: bash scripts/ci/apt-prep.sh gcc-mingw-w64-x86-64"
  echo "  (CI still runs this gate; set TEMEN_GATE_SKIP_CROSS=1 to silence this notice)"
else
  if ! rustup target list --toolchain "$toolchain" --installed | grep -qx "$cross_target"; then
    if [ "${TEMEN_GATE_INSTALL:-0}" = "1" ]; then
      run rustup target add --toolchain "$toolchain" "$cross_target"
    else
      echo "local-gate: rust $toolchain has no $cross_target target." >&2
      echo "  rustup target add --toolchain $toolchain $cross_target" >&2
      echo "  (or re-run with TEMEN_GATE_INSTALL=1, or TEMEN_GATE_SKIP_CROSS=1 to skip)" >&2
      exit 1
    fi
  fi
  cargo_pinned check --workspace --all-targets --target "$cross_target"
  cargo_pinned clippy --workspace --all-targets --target "$cross_target" -- -D warnings
  cargo_pinned clippy -p temen-jit --features stack-check --all-targets --target "$cross_target" -- -D warnings
fi

echo "local-gate: OK — lints and compiles clean on CI's toolchain ($toolchain)."
echo "local-gate: the CI matrix (tests, cross-OS runtime, miri/asan/tsan/loom, fuzz, differential) is still the gate."
