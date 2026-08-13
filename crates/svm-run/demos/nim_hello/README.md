# nim_hello — a real Nim program compiled to a runnable SVM powerbox module

`hello.nim` (`write(stdout, "hello, svm\n")`) compiled all the way to a
browser-loadable `.svmb` that **runs and prints for real** — the "run real Nim"
playground card, and the analog of the `chibicc.svmb` / `svm-leng.svmb` asset
lanes but for a *compiled Nim program that runs* rather than a compiler.

## Pipeline

```
hello.nim
  → nimony c  (nifler → nimony → hexer)      # the program + `system` module Leng (.x.nif)
  → svm_leng::link_nim_powerbox              # the nim → §3e-powerbox bottom-edge bridge:
                                             #   compute leaves → shim, sysWrite(fd,buf,len)
                                             #   → the STREAM write(buf,len) cap
  → svm_verify + svm_encode::encode_module   # the browser-loadable, verified .svmb
```

The bridge leaves only the powerbox `write` STREAM cap as a bound-at-run
manifest import, so the host grants stdout and the guest's `write` reaches it.

## Rebuild the asset

Needs the nimony toolchain (the vendored `nimony`/`nativenif` submodules, built
by `scripts/ci/provision-nimony.sh`):

```sh
NIMONY_BIN=<abs>/nimony/bin NIM_BIN=<dir of nim> \
  cargo run --release -p svm-run --example build_nim_hello_svmb -- \
    crates/svm-run/demos/nim_hello/hello.nim \
    browser/web/assets/nim_hello.svmb
```

The committed asset lives at `browser/web/assets/nim_hello.svmb` (the playground
loads it; `nim_hello` play.js card). **Code-coupled:** `crates/svm-run/tests/
nim_hello_asset.rs` decodes, re-verifies, and runs the committed bytes under
`run_powerbox`, asserting `hello, svm` — no toolchain, so it gates every PR. If
an IR/ABI/encoder or bridge change makes it drift, that test fails: regenerate
the asset with the command above and commit it.
