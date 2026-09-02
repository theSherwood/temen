#!/usr/bin/env bash
# Self-host differential (SELFHOST_C.md §7 step 5): for each test .c, assert the Temen IR emitted by
# chibicc-the-GUEST (chibicc.temen, run on the Temen tree-walker via the chibicc_run example) is
# byte-identical to the NATIVE reference (chibicc_ref, built from the *same* cc1 TUs + cc1_main.c).
# Same frontend code both sides → the only variables are the substrate (guest libc + Temen interpreter
# vs system libc + native CPU). A mismatch is a guest-libc or on-ramp bug.
#
# Prereqs: build_chibicc_temen.sh has produced $CACHE/chibicc.temen; clang ≥18 on PATH.
#   env: TEMEN_CHIBICC_CACHE (default /tmp/temen_chibicc_cache)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
SRC="$REPO/frontend/chibicc"
CACHE="${TEMEN_CHIBICC_CACHE:-/tmp/temen_chibicc_cache}"
CLANG="${CLANG:-clang}"
TEMEN="$CACHE/chibicc.temen"
[ -f "$TEMEN" ] || { echo "missing $TEMEN — run build_chibicc_temen.sh first"; exit 1; }

# Native reference: the same cc1 TUs + cc1_main.c, system libc + native_ref_shims.c (strtold→strtod,
# so -mlong-double-64 doesn't clash with the system 80-bit strtold ABI — see that file). Rebuilt if
# any input is newer.
REF="$CACHE/chibicc_ref"
if [ ! -x "$REF" ] || [ "$SRC/codegen_ir.c" -nt "$REF" ] || [ "$HERE/cc1_main.c" -nt "$REF" ] \
   || [ "$HERE/native_ref_shims.c" -nt "$REF" ]; then
  echo "=== building native reference (chibicc_ref) ==="
  CC1="tokenize preprocess parse type codegen_ir strings hashmap unicode"
  "$CLANG" -O2 -mlong-double-64 -Wno-switch -I"$SRC" \
    $(for t in $CC1; do echo "$SRC/$t.c"; done) "$HERE/cc1_main.c" "$HERE/native_ref_shims.c" -o "$REF"
fi

# Build the guest runner once (quiet); the loop just invokes it.
cargo build -q -p temen-run --example chibicc_run

SEED="$CACHE/diff_seed"; mkdir -p "$SEED"
INC="$SRC/include"

# The one path-dependent line: `-g` writes `debug.file 0 "<argv[1]>"`. Native gets a host path, the
# guest gets `/in.c`; normalize that single quoted path so the rest is a true byte-for-byte compare.
norm() { sed -E 's|^(debug\.file 0 ")[^"]*(")|\1<INPUT>\2|'; }

# The browser runs a .temen on the bytecode interpreter AND (JIT tier) the temen-wasmjit whole-module
# compile. So the guest must emit identical IR on every engine, not just the tree-walker oracle. We
# check all three locally runnable engines: treewalk (oracle), bytecode (browser default), jit
# (Cranelift — the compiling-backend proxy for the browser's wasm-jit). temen-wasmjit compile-parity
# is checked separately (check_wasmjit_compile.sh) + gated end-to-end by the real-browser CI job.
ENGINES="${TEMEN_CHIBICC_ENGINES:-treewalk bytecode jit}"

pass=0; fail=0
run_case() {
  local name="$1" want_inc="$2"; shift 2
  local cf="$SEED/$name.c"
  cat > "$cf"   # body on stdin
  # native golden: -I the real header tree (the guest uses its default /include memfs mount instead).
  ( cd "$SEED" && "$REF" "-I$INC" "$name.c" ) > "$CACHE/$name.golden" 2>"$CACHE/$name.golden.err" || {
    echo "  ✗ $name — native reference errored: $(cat "$CACHE/$name.golden.err")"; fail=$((fail+1)); return; }
  local incarg=(); [ "$want_inc" = inc ] && incarg=("$INC")
  local engres=""
  for eng in $ENGINES; do
    if TEMEN_CHIBICC_BACKEND="$eng" cargo run -q -p temen-run --example chibicc_run -- \
         "$TEMEN" "$cf" /in.c "${incarg[@]}" > "$CACHE/$name.$eng" 2>"$CACHE/$name.$eng.err" \
       && diff <(norm < "$CACHE/$name.golden") <(norm < "$CACHE/$name.$eng") >"$CACHE/$name.$eng.diff"; then
      engres="$engres $eng✓"
    else
      engres="$engres $eng✗"; fail=$((fail+1))
      echo "  ✗ $name [$eng] — mismatch/err:"; { sed 's/^/      /' "$CACHE/$name.$eng.diff" 2>/dev/null | head -20; sed 's/^/      /' "$CACHE/$name.$eng.err"; } | head -20
    fi
  done
  echo "  $name ($(wc -l < "$CACHE/$name.golden") lines IR):$engres"
  [ "${engres//✗/}" = "$engres" ] && pass=$((pass+1))
}

echo "=== self-host differential (guest chibicc.temen  vs  native chibicc_ref) ==="

run_case int noinc <<'EOF'
int add(int a, int b) { return a + b; }
int fib(int n) { return n < 2 ? n : fib(n-1) + fib(n-2); }
int main(void) {
  int xs[4]; int s = 0;
  for (int i = 0; i < 4; i++) { xs[i] = i * i; s += xs[i]; }
  return add(s, fib(10));
}
EOF

# Floats: literals + arithmetic → the codegen emits f64/f32 constants through the byte-exact
# %.17g path (printf_shim → __vm_fmt_gen). This is the case integer-only never reaches.
run_case float noinc <<'EOF'
double poly(double x) { return 3.14159265358979 * x * x - 2.5e-3 * x + 0.125; }
float mix(float a, float b) { return a * 0.1f + b / 7.0f; }
int main(void) {
  double d = poly(1.0 / 3.0);
  float f = mix(2.0f, 3.0f);
  return (int)(d + f);
}
EOF

# Header #include → preprocess reaches the memfs /include mount; stdbool/stdint pull real headers.
run_case hdr inc <<'EOF'
#include <stdbool.h>
#include <stdint.h>
bool even(int n) { return (n & 1) == 0; }
int main(void) {
  int64_t acc = 0;
  for (int32_t i = 0; i < 8; i++) if (even(i)) acc += (int64_t)i * i;
  return (int)acc;
}
EOF

echo "=== IR differential: $pass passed, $fail failed ==="

# --- The capstone loop (SELFHOST_C.md §5 D / §7 step 5): compile a C program with chibicc-the-guest
# INSIDE the Temen, then parse + run the emitted module INSIDE the Temen, and assert its result matches
# the same program built + run natively. This is the exact browser-playground pipeline (compile →
# encode via temen_text::parse_module → run), gated locally on every engine. Corpus = return-value
# programs (the result is main's exit code); no libc in the compiled output, so no external symbols.
echo "=== capstone loop: in-Temen compile-and-run  vs  native clang (exit code) ==="
CLANG_NATIVE="${CLANG:-clang}"
for f in "$HERE"/corpus/*.c; do
  b=$(basename "$f" .c)
  # The program's result IS its exit code, so a non-zero exit is normal — capture it via `if` so it
  # doesn't trip `set -e`.
  "$CLANG_NATIVE" -O2 "$f" -o "$CACHE/$b.native" 2>/dev/null
  if "$CACHE/$b.native"; then nat=0; else nat=$?; fi
  res=""
  for eng in $ENGINES; do
    if TEMEN_CHIBICC_EXEC=1 TEMEN_CHIBICC_BACKEND="$eng" cargo run -q -p temen-run --example chibicc_run -- \
         "$TEMEN" "$f" /in.c >/dev/null 2>"$CACHE/$b.$eng.exec.err"; then g=0; else g=$?; fi
    if [ "$g" = "$nat" ]; then res="$res $eng✓"; else res="$res $eng✗($g)"; fail=$((fail+1)); fi
  done
  echo "  $b: native=$nat |$res"
done

echo "=== total failures: $fail ==="
[ "$fail" -eq 0 ]
