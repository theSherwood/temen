# nimony back-end chain on SVM — two real phase guests, chained

Slice 3 of "compile Nim in the browser" (NIM.md §3c/§3e): drive **two real nimony phase guests in
sequence on the SVM**, host-orchestrated, and prove the whole chain is **byte-identical to native at
every hop**, on all three engines.

```
semchecked .s.nif ──hexer.svmb──▶ Leng .x.nif ──svm-leng.svmb──▶ SVM IR text
                    (fs cap)                     (stdin→stdout)
```

This is the exact shape the browser card (slice 4) will use: **the host drives each phase's committed
`.svmb` and pipes between them** — no phase orchestrates the others, the host does (Rust here, JS in
the browser). Both guests are real:

- **`hexer.svmb`** (slice 2) — nimony's middle-end; reads the semchecked `.s.nif` from an in-window
  `fs` memfs and writes Leng (`.x.nif`).
- **`svm-leng.svmb`** (the committed W5 self-host asset) — reads that Leng on **stdin** and emits SVM
  IR text on **stdout**.

The driver is `examples/nim_backend_chain.rs`. Every hop is checked against its native oracle (native
`hexer`'s `.x.nif`, then `svm_leng::translate_to_text`), so a divergence anywhere is caught. The
produced IR parses as a well-formed module; *running* it needs the W2 link with the `system` module +
W3 runtime shim (its `ini`/syscall imports are unbound standalone) — that link step is proven
separately in svm-leng's `nim_e2e`, and is identical for native output.

## Where nimsem fits (and why it's not here yet)

The `.s.nif` input is what **`nimsem`** (the sema phase) produces. `nimsem`-on-SVM is currently
blocked **upstream of the SVM**: a fresh stock-nim build of `nimsem` fails the `m` command *natively*
while the shipped `bin/nimsem` succeeds on the same input. Crucially, the **SVM on-ramp is proven
innocent** — svm `nimsem` reproduces a native stock-nim `nimsem` build *byte for byte* (identical
sema traces), and both fail identically; only the pre-shipped binary differs. So this chain uses the
oracle toolchain to produce the `.s.nif`; once `nimsem`-on-SVM is unblocked (a native nimony
toolchain-build issue) it slots in ahead of `hexer` as a third guest, closing the full Nim→run chain.

## Run it

```sh
NIMONY_BIN=<repo>/.nimtool/nimony/bin/nimony  HEXER_BIN=<repo>/.nimtool/nimony/bin/hexer \
  bash build_backend_chain.sh
```

Fail-soft **SKIP** without the toolchain (NIM.md §2). `hexer.svmb` is a build artifact (~3 MB, not
committed); `svm-leng.svmb` is the committed asset. This script is the gate.

## Status

✅ **Back-end chain done.** `hexer.svmb → svm-leng.svmb` over `inputs/*.nim`'s semchecked NIF,
byte-identical to native at every hop, on treewalk · bytecode · jit. This is the real compile
back-end running as chained phase binaries on the SVM — the mechanism the browser card renders.
