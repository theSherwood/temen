# nimony back-end chain on Temen — two real phase guests, chained

Slice 3 of "compile Nim in the browser" (NIM.md §3c/§3e): drive **two real nimony phase guests in
sequence on the Temen**, host-orchestrated, and prove the whole chain is **byte-identical to native at
every hop**, on all three engines.

```
semchecked .s.nif ──hexer.temen──▶ Leng .x.nif ──temen-leng.temen──▶ Temen IR text
                    (fs cap)                     (stdin→stdout)
```

This is the exact shape the browser card (slice 4) will use: **the host drives each phase's committed
`.temen` and pipes between them** — no phase orchestrates the others, the host does (Rust here, JS in
the browser). Both guests are real:

- **`hexer.temen`** (slice 2) — nimony's middle-end; reads the semchecked `.s.nif` from an in-window
  `fs` memfs and writes Leng (`.x.nif`).
- **`temen-leng.temen`** (the committed W5 self-host asset) — reads that Leng on **stdin** and emits Temen
  IR text on **stdout**.

The driver is `examples/nim_backend_chain.rs`. Every hop is checked against its native oracle (native
`hexer`'s `.x.nif`, then `temen_leng::translate_to_text`), so a divergence anywhere is caught. The
produced IR parses as a well-formed module; *running* it needs the W2 link with the `system` module +
W3 runtime shim (its `ini`/syscall imports are unbound standalone) — that link step is proven
separately in temen-leng's `nim_e2e`, and is identical for native output.

## Where nimsem fits (and why it's not here yet)

The `.s.nif` input is what **`nimsem`** (the sema phase) produces. `nimsem`-on-Temen is currently
blocked **upstream of the Temen**: a fresh stock-nim build of `nimsem` fails the `m` command *natively*
while the shipped `bin/nimsem` succeeds on the same input. Crucially, the **Temen on-ramp is proven
innocent** — temen `nimsem` reproduces a native stock-nim `nimsem` build *byte for byte* (identical
sema traces), and both fail identically; only the pre-shipped binary differs. So this chain uses the
oracle toolchain to produce the `.s.nif`; once `nimsem`-on-Temen is unblocked (a native nimony
toolchain-build issue) it slots in ahead of `hexer` as a third guest, closing the full Nim→run chain.

## Run it

```sh
NIMONY_BIN=<repo>/.nimtool/nimony/bin/nimony  HEXER_BIN=<repo>/.nimtool/nimony/bin/hexer \
  bash build_backend_chain.sh
```

Fail-soft **SKIP** without the toolchain (NIM.md §2). `hexer.temen` is a build artifact (~3 MB, not
committed); `temen-leng.temen` is the committed asset. This script is the gate.

## Status

✅ **Back-end chain done.** `hexer.temen → temen-leng.temen` over `inputs/*.nim`'s semchecked NIF,
byte-identical to native at every hop, on treewalk · bytecode · jit. This is the real compile
back-end running as chained phase binaries on the Temen — the mechanism the browser card renders.
