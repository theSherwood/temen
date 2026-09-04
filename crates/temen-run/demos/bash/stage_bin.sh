#!/usr/bin/env bash
# stage_bin.sh — compile the #801 coreutils (demos/posix_utils, the chibicc world) to `.temen` text
# IR for bash's /bin: `bash_probe` (and the capstone gate) parses each file, grants it as a
# `Module`, and registers it as a filesystem executable, exactly as `c_posix.rs`'s
# `stage_coreutils` does in-test. bash then runs them as external commands (fork → execve).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/../../../.."
OUT="${1:-/tmp/temen_bash_cache/bin}"
mkdir -p "$OUT"
make -s -C "$ROOT/frontend/chibicc"
CC="$ROOT/frontend/chibicc/chibicc"
UT="$HERE/../posix_utils"
PL="$HERE/../posix_libc"
for t in true false echo cat seq head wc sort uniq ls pwd grep tr cut; do
  tu="$OUT/_$t.c"
  if [ "$t" = grep ]; then
    cat "$UT/util.c" "$PL/regex.c" "$UT/$t.c" >"$tu"
  else
    cat "$UT/util.c" "$UT/$t.c" >"$tu"
  fi
  "$CC" -cc1 --emit-ir --child-entry -cc1-input "$tu" -cc1-output "$OUT/$t.temen" "$tu"
done
echo "staged: $OUT"
