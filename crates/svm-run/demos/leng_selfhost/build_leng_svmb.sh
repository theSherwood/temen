#!/usr/bin/env bash
# svm-leng-the-guest bitcode pipeline (NIM.md §3e / W5 capstone) — the chibicc/Postgres asset-lane
# pattern for the Rust `svm-leng` translator: build-std → llvm-link → svm-llvm-translate → prep_svmb
# (decode/verify/bytecode-compile gate). Produces the committed `svm-leng.svmb`: the real translator,
# run in-sandbox over a real hexer Leng file, emits SVM text byte-identical to native.
#
# Unlike chibicc (C, clang per-TU), svm-leng is Rust: `-Z build-std` (unlocked on stable 1.81 via
# RUSTC_BOOTSTRAP=1 — LLVM 18, matching the llvm-*-18 tools) compiles std from source so the whole
# guest links with only `read`/`write`/`bcmp` external (all on-ramp-recognized). `panic_immediate_abort`
# keeps the float formatter out of panic paths; the guest's own bump allocator means no `malloc` extern.
#
#   needs: rustc +1.81.0 (+ rust-src), llvm-link-18, opt-18, cargo
#   env:   SVM_LENG_CACHE (default /tmp/svm_leng_cache)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
GUEST="$HERE/leng_guest"
CACHE="${SVM_LENG_CACHE:-/tmp/svm_leng_cache}"
TRIPLE="x86_64-unknown-linux-gnu"
mkdir -p "$CACHE"

LINK="${LLVM_LINK:-llvm-link-18}"; command -v "$LINK" >/dev/null || LINK=llvm-link
OPT="${LLVM_OPT:-opt-18}"; command -v "$OPT" >/dev/null || OPT=opt

# [1/4] build-std: emit per-crate textual LLVM IR for the whole closure (guest + svm-leng + std from
# source). `--emit=llvm-ir`, release, panic=abort with panic_immediate_abort.
echo "[1/4] build-std (rustc +1.81.0) ..."
( cd "$GUEST" && RUSTFLAGS='--emit=llvm-ir' CARGO_TARGET_DIR="$CACHE/target" RUSTC_BOOTSTRAP=1 \
    cargo +1.81.0 build --release \
      -Zbuild-std=std,panic_abort -Zbuild-std-features=panic_immediate_abort \
      --target "$TRIPLE" --ignore-rust-version )

DEPS="$CACHE/target/$TRIPLE/release/deps"
# panic_unwind and panic_abort both define `__rust_panic_cleanup`; with panic=abort only panic_abort
# is needed, so drop panic_unwind to avoid a duplicate-symbol link error.
mapfile -t LLS < <(ls "$DEPS"/*.ll | grep -v '/panic_unwind')
[ "${#LLS[@]}" -gt 0 ] || { echo "no .ll emitted — build failed before codegen" >&2; exit 1; }

# [2/4] llvm-link + prune to the closure reachable from main/malloc/free.
echo "[2/4] llvm-link + opt (internalize,globaldce) ..."
"$LINK" -S "${LLS[@]}" -o "$CACHE/leng.linked.ll"
"$OPT" -S -passes=internalize,globaldce \
  -internalize-public-api-list=main,malloc,free \
  "$CACHE/leng.linked.ll" -o "$CACHE/leng.legal.ll"

# [2a/4] stub audit: every remaining undefined symbol must be one the on-ramp recognizes (the
# powerbox streams + bcmp). Anything else means an unhandled libc gap — fail loudly rather than
# silently stubbing it to a trap.
echo "[2a/4] stub audit ..."
ONRAMP='^(read|write|bcmp|memcmp|memcpy|memmove|memset|malloc|calloc|realloc|free|__vm_[a-z_0-9]+|__svm_[a-z_0-9]+)$'
UNRESOLVED=0
while read -r sym; do
  [ -z "$sym" ] && continue
  case "$sym" in llvm.*) continue;; esac   # LLVM intrinsics are lowered by the on-ramp, not linked
  if ! [[ "$sym" =~ $ONRAMP ]]; then echo "  UNHANDLED extern: $sym" >&2; UNRESOLVED=1; fi
done < <(grep -E '^declare ' "$CACHE/leng.legal.ll" | grep -oE '@"?[A-Za-z0-9_.$]+' | tr -d '@"' | sort -u)
[ "$UNRESOLVED" -eq 0 ] || { echo "stub audit failed: unhandled externs above" >&2; exit 1; }

# [3/4] translate to a raw .svmb. --host-page 65536 = wasm 64 KiB pages (browser-loadable, like
# chibicc.svmb); no --stub-externs, so translation proves every call resolves.
echo "[3/4] svm-llvm-translate --binary ..."
TR="$REPO/crates/svm-llvm/target/release/svm-llvm-translate"
[ -x "$TR" ] || cargo build --release --bin svm-llvm-translate --manifest-path "$REPO/crates/svm-llvm/Cargo.toml"
"$TR" "$CACHE/leng.legal.ll" -o "$CACHE/leng_raw.svmb" --binary --host-page 65536

# [4/4] prep_svmb gate: decode → verify (the mandatory TCB floor) → bytecode-compile → re-serialize
# to the shipped artifact.
echo "[4/4] prep_svmb (decode/verify/bytecode-compile) ..."
cargo run --release -p svm-run --example prep_svmb -- "$CACHE/leng_raw.svmb" "$CACHE/svm-leng.svmb"

echo "done: $CACHE/svm-leng.svmb"
