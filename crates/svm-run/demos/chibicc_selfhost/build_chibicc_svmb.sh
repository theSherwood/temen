#!/usr/bin/env bash
# chibicc-the-guest bitcode pipeline (SELFHOST_C.md §5 A / §7 step 2) — the Postgres asset
# pattern (build-pg-assets.mjs / demos/postgres/build_bitcode.sh) with vendored sources, so
# no fetch: per-TU bitcode → llvm-link → svm-llvm-translate → prep_svmb (decode/verify/
# bytecode-compile gate).
#
# Step-2 mode: the guest libc (SELFHOST_C.md §5 B, Appendix A) is not built yet, so the
# translate runs with --stub-externs — every unprovided libc extern becomes a trap-if-called
# stub. That is exactly what this step is for: shaking out *translation and verification* of
# the compiler itself before the libc lands (step 3 replaces the stubs with the self-host
# libc + personality imports and drops --stub-externs).
#
#   needs: clang-18 (or clang ≥18), llvm-link, llvm-dis, cargo
#   env:   SVM_CHIBICC_CACHE (default /tmp/svm_chibicc_cache)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
SRC="$REPO/frontend/chibicc"
CACHE="${SVM_CHIBICC_CACHE:-/tmp/svm_chibicc_cache}"
CLANG="${CLANG:-clang-18}"; command -v "$CLANG" >/dev/null || CLANG=clang
LINK="${LLVM_LINK:-llvm-link-18}"; command -v "$LINK" >/dev/null || LINK=llvm-link
mkdir -p "$CACHE"

# The cc1-path TU set ONLY (Appendix A.3): no main.c (driver — its glob/fork/execvp externs
# would fail the fail-closed import bind), no codegen.c (x86 backend, never called by
# --emit-ir). cc1_main.c supplies the entry + the globals main.c owned.
CC1_TUS="tokenize preprocess parse type codegen_ir strings hashmap unicode"

# -mlong-double-64: Appendix A F3 — strtold/fp80 doesn't lower through the on-ramp; long
# double = double bounds the divergence to double-rounding on literal parse (fval is cast to
# double at emission, codegen_ir.c:1552).
CFLAGS="-O2 -emit-llvm -c -mlong-double-64 -I$SRC"

echo "=== [1/4] per-TU bitcode (clang -O2, long-double-64) ==="
for tu in $CC1_TUS; do
  "$CLANG" $CFLAGS "$SRC/$tu.c" -o "$CACHE/$tu.bc"
done
"$CLANG" $CFLAGS "$HERE/cc1_main.c" -o "$CACHE/cc1_main.bc"

# The self-host guest libc (SELFHOST_C.md §5 B / Appendix B): the aggregating -DSVM_GUEST driver
# that #includes the reused Postgres shims + chibicc_extra.c. -fno-builtin-* so clang doesn't rewrite
# the shims' own byte-loops / allocator into self-calls (the Postgres link_shims.sh posture).
echo "=== [1b/4] guest libc (chibicc_libc.c, -DSVM_GUEST) ==="
"$CLANG" -O2 -emit-llvm -c -DSVM_GUEST -mlong-double-64 \
  -fno-vectorize -fno-slp-vectorize \
  -fno-builtin-memcpy -fno-builtin-memmove -fno-builtin-memset -fno-builtin-memcmp \
  -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free \
  -fno-builtin-strlen -fno-builtin-strcmp -fno-builtin-strncmp \
  -I"$SRC" -I"$REPO/crates/svm-run/demos/postgres" \
  "$HERE/chibicc_libc.c" -o "$CACHE/chibicc_libc.bc"

echo "=== [2/4] llvm-link (cc1 + guest libc) ==="
"$LINK" $(for tu in $CC1_TUS; do echo "$CACHE/$tu.bc"; done) "$CACHE/cc1_main.bc" \
  "$CACHE/chibicc_libc.bc" -o "$CACHE/chibicc.linked.bc"
echo "linked: $(stat -c%s "$CACHE/chibicc.linked.bc") bytes"

echo "=== [2a/4] stub audit — every undefined symbol must be expected libc ==="
# --stub-externs turns EVERY undefined symbol into a trap-if-called stub, including a chibicc
# function whose defining TU was accidentally excluded (this caught `align_to`, owned by the
# excluded codegen.c but called 15× for struct layout — a runtime time bomb the verify gate
# can't see). Allowlist = Appendix A's libc fill-set (step 3 provides these); anything else
# undefined is a build bug: fail loudly.
NM="${LLVM_NM:-llvm-nm-18}"; command -v "$NM" >/dev/null || NM=llvm-nm
"$NM" --undefined-only "$CACHE/chibicc.linked.bc" | awk '{print $2}' | sort -u \
  > "$CACHE/undefined.txt"
# With the guest libc linked in, the ONLY undefined symbols left should be names the on-ramp itself
# recognizes and lowers (the powerbox/personality bridge + its synth helpers) — not libc, not
# chibicc-internal. Allowlist = the recognizer set (svm-llvm/src/lib.rs): the __vm_* primitives,
# the powerbox stream/exit names, and the synth allocator/mem/fmt family.
echo "remaining undefined after libc link:"; cat "$CACHE/undefined.txt" | sed 's/^/  /'
ONRAMP='^(__vm_[a-z_0-9]+|write|read|exit|_exit|_Exit|malloc|calloc|realloc|free|memcpy|memmove|memset|memcmp|bcmp|printf|puts|putchar|putc|fputc|fwrite|fputs|__svm_[a-z_0-9]+)$'
if UNEXPECTED=$(grep -Ev "$ONRAMP" "$CACHE/undefined.txt"); then
  echo "UNEXPECTED undefined (not on-ramp-recognized — libc gap or excluded TU):"
  echo "$UNEXPECTED" | sed 's/^/  /'
  echo "(these would trap under --stub-externs; provide them in chibicc_extra.c or a shim)"
  UNRESOLVED=1
fi

echo "=== [3/4] translate (svm-llvm-translate --binary --host-page 65536 --stub-externs) ==="
TR="$REPO/crates/svm-llvm/target/release/svm-llvm-translate"
if [ ! -x "$TR" ]; then
  echo "building svm-llvm-translate…"
  cargo build --release --bin svm-llvm-translate \
    --manifest-path "$REPO/crates/svm-llvm/Cargo.toml"
fi
# --host-page 65536: the artifact ultimately ships to the browser (wasm 64 KiB pages, D40).
# Drop --stub-externs once the libc resolves everything but on-ramp-recognized names (the goal):
# then translation itself proves nothing is left to trap. Fall back to stubbing while gaps remain,
# so the build still produces an inspectable module.
if [ "${UNRESOLVED:-0}" = "1" ]; then
  echo "(libc gaps remain — translating WITH --stub-externs so the module is still inspectable)"
  STUB=--stub-externs
else
  echo "(no libc gaps — translating WITHOUT --stub-externs: every call resolves)"
  STUB=
fi
"$TR" "$CACHE/chibicc.linked.bc" -o "$CACHE/chibicc_raw.svmb" \
  --binary --host-page 65536 $STUB
echo "raw svmb: $(stat -c%s "$CACHE/chibicc_raw.svmb") bytes"

echo "=== [4/4] prep_svmb (decode → verify → bytecode-compile gate) ==="
cargo run --release -p svm-run --example prep_svmb -- \
  "$CACHE/chibicc_raw.svmb" "$CACHE/chibicc.svmb"
echo "OK: $CACHE/chibicc.svmb"
