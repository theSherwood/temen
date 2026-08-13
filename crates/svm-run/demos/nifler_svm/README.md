# nifler-on-SVM — the first real nimony phase running on the sandbox

`nifler` is the **first phase** of the nimony toolchain (`nifler → nimony → hexer → lengc`): it
parses Nim source into NIF (`.nif`). This demo compiles the **real** `nifler` (from the vendored
`nimony` submodule, ~1250 loc of Nim) all the way to a runnable SVM module and proves it parses Nim
**byte-for-byte identically to native nifler**, on all three engines — the first rung of "compile
Nim in the browser" (NIM.md §3c/§3e): a genuine compiler phase, not a stand-in, running on the SVM.

## Pipeline (the C on-ramp, NIM.md Phase 1, applied to a compiler phase)

```
nifler.nim
  → nim c  (stock Nim, --mm:arc -d:useMalloc -d:danger --threads:off -d:noSignalHandler)
      → C  (219 TUs, ~1.08M lines — the Nim runtime + std/os/std/io)
  → clang-18 -O2 -fno-vectorize -fno-slp-vectorize -emit-llvm      # scalar divides; the on-ramp
      → bitcode                                                    #   doesn't lower vector DivS
  → llvm-link  +  nifler_shim.c                                    # the libc bottom edge, below
  → svm-llvm-translate --stub-externs
      → SVM module  → prep_svmb (verify)  → nifler.svmb (~17 MB)
```

Then `nifler p in.nim out.nif` runs as a guest over an in-window `fs` memfs seeded with the source
(`examples/nifler_run.rs`), and the emitted `.nif` is read back and diffed against native nifler.

## The bottom edge — `nifler_shim.c`

nifler's whole-program bitcode leaves ~115 undefined libc externals (the Nim runtime + `std/os`/
`std/io` edge). `nifler_shim.c` is **one translation unit** that defines the reachable part over the
sandbox's `fs` capability + powerbox Stream, reusing the shims that already run **Postgres** on SVM:

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
NIFLER_BIN=<repo>/.nimtool/nimony/bin/nifler  bash build_nifler_svmb.sh
```

Fail-soft **SKIP** when the toolchain is absent (nimony bootstrap needs Nim 2.3.x devel, so this is
**not** in the per-PR CI — NIM.md §2). The ~17 MB `.svmb` is a build artifact, **not committed** (too
big for an asset lane); this script is the gate, run when the toolchain is present.

## Status

✅ **Slice 1 done.** `nifler` parses `inputs/*.nim` on the SVM, byte-identical to native, on
treewalk · bytecode · jit. Next: the larger phases (`hexer`, `nimony`) the same way, then drive the
real chain (NIM.md §3c W4 `domain_exec`) and feed its Leng into the already-in-browser `svm-leng.svmb`
→ the "compile Nim in the browser" card.
