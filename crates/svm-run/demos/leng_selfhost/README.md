# `svm-leng.svmb` — the leng self-host asset (W5 capstone, NIM.md §3e)

The real `svm-leng` translator, compiled to a verified SVM module and **run inside the sandbox** over
a real hexer Leng file to emit SVM text **byte-identical to native**. The leng analog of
`chibicc.svmb` (the [SELFHOST_C.md](../../../../SELFHOST_C.md) asset lane), following the same
code-coupled-asset discipline as the Postgres/chibicc lanes.

## Layout

- `leng_guest/` — the guest crate: the `svm-leng` translator wrapped as a **powerbox program**
  (`main` reads a Leng-NIF module from stdin, calls `svm_leng::translate_to_text`, writes the SVM
  text to stdout). A static-arena bump `#[global_allocator]` + raw `read`/`write` keep the only
  undefined externs to `read`/`write`/`bcmp`, all on-ramp-recognized.
- `build_leng_svmb.sh` — the build pipeline: `-Z build-std` (stable 1.81 / LLVM 18, matching the
  `llvm-*-18` tools) → `llvm-link-18` → `opt-18 internalize,globaldce` → `svm-llvm-translate --binary`
  → `prep_svmb` (decode/verify/bytecode-compile gate). A stub audit fails loudly if any extern
  outside the on-ramp allowlist survives.
- `corpus/*.leng.nif` — verbatim `hexer c` output from real nimony source (Nim's `system/stringimpl`
  with ARC, control flow, gotos).
- `svm-leng.svmb` — **the committed, prebuilt asset** (built by `build_leng_svmb.sh`).

## In the browser

The same asset drives a **playground self-host card** (the chibicc-self-host-card analog): `svm-leng.svmb`
is copied into `browser/web/assets/` (by `browser/build-onramp-assets.mjs`, and committed so the card works
out of the box), and `browser/web/play.js` adds a card whose editor holds a real hexer Leng file and whose
Run pipes it to the asset on stdin via `svm_run_onramp` (the fixed §3e powerbox on the wasm engine),
showing the emitted SVM IR. Covered by `browser/browser-play-editor-test.mjs` (Chromium) and the
`check-play-assets.mjs` reference gate.

## The gate

`crates/svm-run/tests/leng_selfhost_asset.rs` loads the committed `svm-leng.svmb`, re-verifies it, and
runs it over each `corpus/*.leng.nif`, asserting the in-sandbox SVM text equals `svm_leng::translate_to_text`
run host-side. The oracle is the in-tree `svm-leng`, so the gate needs **no build toolchain**: if an
IR/ABI/encoder or `svm-leng` change makes the committed asset stop matching native, the test fails the
PR that caused the drift.

## Regenerating the asset

When that gate fails (or you intentionally change `svm-leng`/the encoder), rebuild and commit the asset:

```sh
bash crates/svm-run/demos/leng_selfhost/build_leng_svmb.sh
cp "${SVM_LENG_CACHE:-/tmp/svm_leng_cache}/svm-leng.svmb" crates/svm-run/demos/leng_selfhost/svm-leng.svmb
```

Requires `rustc +1.81.0` (with the `rust-src` component), `llvm-link-18`, and `opt-18`.
