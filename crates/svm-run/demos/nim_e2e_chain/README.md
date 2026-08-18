# The full nimony compiler on the SVM — Nim source → a module that runs

This is the capstone the earlier "nimony in the browser" slices left in pieces: **every nimony phase
runs as a sandboxed SVM guest, and the compiled program then executes to the correct value.**

```
prog.nim ─nifler──▶ .p.nif ─┐
stdlib   ─nifler (exec child, on demand)─▶ .p.nif
                            └─nimsem──▶ .s.nif ─hexer──▶ .x.nif (Leng) ─┐
                                                                        svm_leng::link → run
```

- **nifler** (parse), **nimsem** (sema — itself a driver, shelling out to nifler, routed to the SVM
  `exec` capability so `nifler.svmb` runs as an isolated child sharing the memfs), and **hexer**
  (lower) each run as real `.svmb` guests over one shared in-window `fs`.
- The final multi-module IR link is the embedder's Rust (`svm_leng::link_whole_with_runtime` + the W3
  runtime shim) — the same *host-drives-the-phases* shape as the browser cards and `nim_backend_chain`,
  now carried all the way through to a **run**.
- The linked module executes on **both engines** (§9 interp/JIT parity) and its exported proc returns
  the expected value.

This retires the last gap in the workstream: the individual phases were each proven on the SVM (slices
1–2b), and `nim_backend_chain` drove hexer→svm-leng but stopped at "the IR parses." Here the chain runs
front-to-back and the output *runs*.

## Multi-module (imports)

The driver reads nimony's own dependency-ordered build plan (`<main>.build.nif`), so it handles **any
number of modules**. A program with an `import` pulls in more compilation units, each its own
nifler→nimsem→hexer unit; the plan enumerates them and gives the dependency edges (in stem space — no
path→stem resolution needed), and the driver topologically orders them, runs `nimsem` on each with the
exact args nimony assigned (`--isSystem` / `--isMain` / bare), lowers each with `hexer`, and links the
whole `.x.nif` set. The `usermod` fixture (`import ./helper`) is a three-unit build (system + helper +
main) that compiles and runs on the SVM.

## Two run modes

- **Compute** (`<export> <expected-i64> [args…]`): link with the W3 compute shim and call an exported
  proc, asserting the returned `i64` on both engines (§9 parity).
- **I/O** (`<io> <expected-stdout>`): a program that `write`s — link through the **nim→powerbox
  bridge** (`svm_leng::link_nim_powerbox`, the same bridge the `nim_hello` card ships) and run `_start`
  under the powerbox, so the guest's `write(fd,buf,len)` reaches the STREAM `write` cap. The captured
  **stdout** is checked against the expected string. The `iohello` fixture (`import std/syncio;
  write(stdout, "hello, svm\n")`) is a four-unit compile that **prints for real** — a Nim program
  compiled entirely by the SVM, producing output. (`echo` isn't an identifier nimony resolves yet — a
  frontend gap, not ours.)

Programs that pull in *stdlib* modules additionally depend on that module translating through
`svm-leng` (the breadth long-pole, #760 / conformance suite #956).

`nimsem` is built with **`-d:skipPostSemValidator`** (as nimony's own Windows CI does); the in-process
post-sem IR validator has a separate on-ramp issue, tracked apart from the sema this proves correct.

## Run it

```sh
NIMONY_BIN=<repo>/.nimtool/nimony/bin/nimony  bash build_e2e_chain.sh
```

Builds the three phase guests (`nifler.svmb` ~17 MB, `nimsem.svmb` ~5.5 MB, `hexer.svmb` ~3 MB — build
artifacts, **not committed**), bootstraps a native `nimcache` per fixture (for the parse layout + the
stems nifmake computes), re-runs **sema + lowering on the SVM** from there, links, and runs. Fail-soft
**SKIP** without the nimony toolchain (NIM.md §2). The driver itself is
`crates/svm-run/examples/nim_e2e_chain.rs`.

## Status

✅ **The nimony compiler runs on the SVM, and its output runs too — single- and multi-module.** With
`nifler` (slice 1), the big phases `nimsem`/`hexer` (slice 2/2b), `svm-leng` (W5), and this end-to-end
chain — now dependency-ordered over any number of modules (#955) — a real Nim program is compiled
entirely by sandboxed guests and the result executes to the correct value on both engines. Remaining
toward "any Nim in the playground" (#954): real I/O output (#957), stdlib breadth (#760/#956), and the
in-browser card (#958); the post-sem validator has its own on-ramp follow-up (#959).
