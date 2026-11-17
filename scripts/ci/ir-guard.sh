#!/usr/bin/env bash
# The instruction-count guard (#3088): count the instructions each of a few kernels runs per loop
# iteration, and compare them with `bench/ir-baseline.txt`.
#
#   bash scripts/ci/ir-guard.sh            # compare; fail if a gated count moved past the threshold
#   bash scripts/ci/ir-guard.sh --update   # re-record the baseline (commit the file with the change)
#
# Why: the bytecode engine's per-op cost is set by how LLVM allocates registers over its op loop, and
# a change anywhere in that loop's function can move every op's cost by 10% in either direction (an
# arm the loop never reached cost an `alu` loop 32 instructions per iteration in PR #3073). Wall-clock
# on a shared runner can't see 10%; instruction counts can, exactly. So a shift lands as a number in
# the PR that caused it, and a PR that moves one on purpose says so in its diff to the baseline.
#
# How: `temen-bench --ir KERNEL ENGINE N` runs one kernel once. Cachegrind counts the process's
# instructions (`I refs`) at two loop counts; the difference over the difference in `n` is the
# per-iteration count, with every fixed cost (start-up, compile, the window) cancelled. It is
# deterministic: the same binary gives the same count on every run, under any load.
#
# What is gated: kernels whose iteration runs only our own code. Their counts depend on the source and
# the compiler, not on the runner. A kernel whose iteration runs libc is only reported: glibc picks its
# `memcpy` by CPU, so its count moves with the runner (`fiber_switch` runs ~500 instructions of it;
# an AMD EPYC runner on glibc 2.43 read it 2 off an Intel Xeon on glibc 2.39, the gated rows equal).
#
# The threshold, `TOL` (2%): a count is exact, so a change is never noise. 2% (about 5 instructions
# of `alu`'s 230) is the line between an edit to the hot path that should explain itself and drift
# nobody needs to read. It fails in both directions, so the baseline stays tight and a regression back
# up to a stale number can't hide.
#
# The compiler: counts change with rustc/LLVM, so the baseline names the rustc it was recorded with,
# and a run on another refuses until it is re-recorded (a `RUST_STABLE` bump re-records in its PR).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

BASELINE=bench/ir-baseline.txt
TOL=0.02
N1=1000
N2=21000
# kernel engine gate|report
ROWS=(
  "alu bytecode gate"
  "call bytecode gate"
  "call_indirect bytecode gate"
  "mem bytecode gate"
  "hostcall bytecode gate"
  "tls bytecode gate"
  "alu tree-walker gate"
  "fiber_switch bytecode report"
  "fiber_wait bytecode report"
)

update=0
case "${1:-}" in
  --update) update=1 ;;
  "") ;;
  *) echo "usage: $0 [--update]" >&2; exit 2 ;;
esac

command -v valgrind >/dev/null || { echo "ir-guard: valgrind is not installed" >&2; exit 1; }
rustc_v=$(rustc -V)
if [ "$update" = 0 ]; then
  recorded=$(sed -n 's/^rustc: //p' "$BASELINE")
  if [ "$recorded" != "$rustc_v" ]; then
    echo "ir-guard: $BASELINE was recorded with '$recorded', this is '$rustc_v'." >&2
    echo "Counts change with the compiler: re-record it with \`bash $0 --update\` on the pinned toolchain." >&2
    exit 1
  fi
fi

cargo build --release -p temen --bin temen-bench
bin="${CARGO_TARGET_DIR:-target}/release/temen-bench"

# A count from a run that failed (a trap, a panic) would measure the failure, so a run that does not
# exit 0 prints nothing, and the row fails below. Valgrind exits with the program's status.
irefs() {
  local log
  log=$(valgrind --tool=cachegrind --cache-sim=no --cachegrind-out-file=/dev/null "$bin" --ir "$@" 2>&1) ||
    { echo "$log" | tail -5 >&2; return 0; }
  echo "$log" | awk '/I *refs:/ { gsub(",", "", $4); print $4 }'
}

out=$(mktemp)
trap 'rm -f "$out"' EXIT
{
  echo "# The instruction-count guard's baseline (#3088): instructions per loop iteration, by"
  echo "# \`scripts/ci/ir-guard.sh\`, which says how they are counted. Re-record with \`--update\`."
  echo "rustc: $rustc_v"
  echo "# kernel engine gate|report instructions"
} >"$out"

fail=0
summary="| kernel | engine | baseline | now | change |"$'\n'"|---|---|---|---|---|"
for row in "${ROWS[@]}"; do
  read -r kernel engine mode <<<"$row"
  a=$(irefs "$kernel" "$engine" "$N1")
  b=$(irefs "$kernel" "$engine" "$N2")
  if [ -z "$a" ] || [ -z "$b" ]; then
    echo "ir-guard: $kernel on the $engine did not run to completion under valgrind (above)" >&2
    exit 1
  fi
  now=$(awk -v a="$a" -v b="$b" -v d=$((N2 - N1)) 'BEGIN { printf "%.1f", (b - a) / d }')
  echo "$kernel $engine $mode $now" >>"$out"
  was=$(awk -v k="$kernel" -v e="$engine" '$1 == k && $2 == e { print $4 }' "$BASELINE" 2>/dev/null || true)
  if [ -z "$was" ]; then
    verdict="new"
    [ "$update" = 1 ] || { [ "$mode" = gate ] && fail=1; }
  else
    verdict=$(awk -v w="$was" -v n="$now" -v t="$TOL" -v m="$mode" 'BEGIN {
      c = (n - w) / w; s = sprintf("%+.1f%%", 100 * c)
      if ((c > t || c < -t) && m == "gate") s = s " **past ±" 100 * t "%**"
      print s }')
    case $verdict in *past*) [ "$update" = 1 ] || fail=1 ;; esac
  fi
  summary+=$'\n'"| $kernel | $engine$( [ "$mode" = report ] && echo ' (reported)' || true) | ${was:--} | $now | $verdict |"
  echo "$kernel $engine: $now (baseline ${was:--}, $verdict)"
done

echo "$summary" >>"${GITHUB_STEP_SUMMARY:-/dev/null}"
if [ "$update" = 1 ]; then
  cp "$out" "$BASELINE"
  echo "ir-guard: recorded $BASELINE"
elif [ "$fail" = 1 ]; then
  echo "ir-guard: a gated count moved past ±$(awk -v t=$TOL 'BEGIN{print 100*t}')% of $BASELINE (or is missing from it)." >&2
  echo "If the change is intended, re-record with \`bash $0 --update\` and commit the baseline; the PR says why." >&2
  exit 1
fi
