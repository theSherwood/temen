# nifler-on-Temen — the first real nimony phase running on the sandbox

`nifler` is the **first phase** of the nimony toolchain (`nifler → nimony → hexer → lengc`): it
parses Nim source into NIF (`.nif`). This demo compiles the **real** `nifler` (from the vendored
`nimony` submodule, ~1250 loc of Nim) all the way to a runnable Temen module and proves it parses Nim
**byte-for-byte identically to native nifler**, on all three engines — the first rung of "compile
Nim in the browser" (NIM.md §3c/§3e): a genuine compiler phase, not a stand-in, running on the Temen.

## Pipeline (the C on-ramp, NIM.md Phase 1, applied to a compiler phase)

```
nifler.nim
  → nim c  (stock Nim, --mm:arc -d:useMalloc -d:danger --threads:off -d:noSignalHandler)
      → C  (219 TUs, ~1.08M lines — the Nim runtime + std/os/std/io)
  → clang-18 -O2 -fno-vectorize -fno-slp-vectorize -emit-llvm      # scalar divides; the on-ramp
      → bitcode                                                    #   doesn't lower vector DivS
  → llvm-link  +  nifler_shim.c                                    # the libc bottom edge, below
  → temen-llvm-translate --stub-externs
      → Temen module  → prep_temen (verify)  → nifler.temen (~17 MB)
```

Then `nifler p in.nim out.nif` runs as a guest over an in-window `fs` memfs seeded with the source
(`examples/nifler_run.rs`), and the emitted `.nif` is read back and diffed against native nifler.

## The bottom edge — `nifler_shim.c`

nifler's whole-program bitcode leaves ~115 undefined libc externals (the Nim runtime + `std/os`/
`std/io` edge). `nifler_shim.c` is **one translation unit** that defines the reachable part over the
sandbox's `fs` capability + powerbox Stream, reusing the shims that already run **Postgres** on Temen:

- `../postgres/os_shim.c` — the POSIX fd/dir/stat syscalls (`open`/`read`/`write`/`stat`/`opendir`/…)
- `../postgres/stdio_shim.c` — the buffered `FILE*` surface (`fopen`/`fread`/`fgets`/`fwrite`/…)
- `../postgres/shim_errno.h` — the shared guest `errno` cell

plus a small supplement: `getcwd` → `/` (Nim's `absolutePath` rejects a non-absolute cwd), `fdopen`,
an empty environment, a deterministic zero clock, and no-op tty/exit/mutex hooks. The math family
(`sin`/`cos`/…) and the process/spawn fringe (`posix_spawn`/`waitpid`/`system`/`glob`/…) are never
called on the parse path and stay `--stub-externs` traps — a call would fault, not escape. **No
ambient authority:** every file byte rides the granted `fs` cap; with no cap, no bytes.

## Run it

```sh
# needs: the nimony submodule, stock nim (2.3.x), the built nifler binary (oracle), clang-18/llvm-18, cargo
NIFLER_BIN=<repo>/.nimtool/nimony/bin/nifler  bash build_nifler_temen.sh
```

Fail-soft **SKIP** when the toolchain is absent (nimony bootstrap needs Nim 2.3.x devel, so this is
**not** in the per-PR CI — NIM.md §2). This script is the toolchain-gated build+diff gate.

## The committed browser asset (slice 4)

The raw `.temen` is ~17.7 MB — too big to commit raw — so the **slice-4 playground card** ships it
**gzipped**: `browser/web/assets/nifler.temen.gz` (~3.8 MB, comparable to one QuickJS asset), which the
card inflates client-side via the browser's `DecompressionStream`. A toolchain-free PR gate,
`crates/temen-run/tests/nifler_asset.rs`, inflates that committed `.gz`, re-verifies it, runs it
in-sandbox over `inputs/*.nim`, and asserts the emitted NIF is byte-identical to the committed
`expected/*.p.nif` (verbatim native-nifler output) — so any IR/ABI/encoder drift fails the PR.

Regenerate the committed asset + fixtures (when the module or a fixture drifts) with:

```sh
TEMEN_NIFLER_EMIT_ASSET=1  NIFLER_BIN=<repo>/.nimtool/nimony/bin/nifler  bash build_nifler_temen.sh
git add browser/web/assets/nifler.temen.gz crates/temen-run/demos/nifler_temen/expected
```

## Status

✅ **Slice 1 + slice 4 done.** `nifler` parses `inputs/*.nim` on the Temen, byte-identical to native, on
treewalk · bytecode · jit (slice 1), and the **"compile Nim in the browser" front-end card** runs it
client-side over the reader's own Nim (slice 4): `browser/web/play.js` kind `'nifler'` → the
`temen_run_nifler_fs` cdylib entry → the emitted `.p.nif` shown in the pane. The complement to the
already-in-browser `temen-leng.temen` back-end card (Leng → Temen IR): with both, the front edge (Nim →
NIF) and the back edge (Leng → IR) of the toolchain each run in the browser, on the Temen.
