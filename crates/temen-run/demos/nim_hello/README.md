# nim_hello — a real Nim program on the capability edge

`hello.nim` (`write(stdout, "hello, temen\n")`) compiled all the way to a
browser-loadable `.temen` that **runs and prints for real** — the "run real Nim"
playground card, and the analog of the `chibicc.temen` / `temen-leng.temen` asset
lanes but for a *compiled Nim program that runs* rather than a compiler. It is
the demo of Nim's **capability edge**: the program gets the host's streams and
exit, and nothing ambient.

## Pipeline

```
hello.nim
  → nimony c  (nifler → nimony → hexer)      # the program's DCE'd Leng (.c.nif)
  → temen_leng::link_nim(.., Edge::Caps, ..)   # the capability edge: compute leaves → shim,
                                             #   fd 0/1/2 → the host's stream_read /
                                             #   stream_write / stderr, _exit → exit
  → temen_verify + temen_encode::encode_module   # the browser-loadable, verified .temen
```

The program's manifest is the host's capabilities alone: its three streams, its
exit, and the core's memory ops its heap grows with. The host grants them, and
the guest's `write` reaches stdout. `temen-link --edge:caps` links the same way.

## Rebuild the asset

Needs the nimony toolchain (the vendored `nimony`/`nativenif` submodules, built
by `scripts/ci/provision-nimony.sh`). Rebuild it through the one entry point,
which runs the `build_nim_hello_temen` example and re-validates the output:

```sh
ONLY=nim_hello bash scripts/rebuild-assets.sh
```

The committed asset lives at `browser/web/assets/nim_hello.temen` (the playground
loads it; `nim_hello` play.js card). **Code-coupled:** `crates/temen-run/tests/
nim_hello_asset.rs` decodes, re-verifies, and runs the committed bytes under
`run_powerbox`, asserting `hello, temen`, and checks its manifest is the
capability edge's — no toolchain, so it gates every PR. If an IR/ABI/encoder or
edge change makes it drift, that test fails: regenerate the asset with
`ONLY=nim_hello bash scripts/rebuild-assets.sh` and commit it.
