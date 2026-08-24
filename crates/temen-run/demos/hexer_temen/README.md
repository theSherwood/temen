# hexer-on-Temen — nimony's middle-end lowering Nim to Leng on the sandbox

`hexer` is nimony's **middle-end** (the second phase after `nifler`→sema): it takes a *semchecked*
NIF (`.s.nif`) and lowers it — iterator inlining, lambda lifting, destructor injection, control-flow
lowering — to **Leng** (`.x.nif`), the IR that `temen-leng` (already in the browser) turns into TEMEN-IR.
This demo compiles the **real** `hexer` (~18k loc of Nim, from the vendored `nimony` submodule) to a
runnable Temen module and proves it lowers a real semchecked module **byte-for-byte identically to
native hexer**, on all three engines — slice 2 of "compile Nim in the browser" (NIM.md §3c/§3e).

It reuses slice 1's pipeline and bottom-edge shim wholesale; the two differences from nifler:

1. **Bottom edge — one new symbol.** hexer reads NIF files through `std/memfiles` (`nifreader.nim`'s
   `vfsOpenMmap`), so its whole-program bitcode needs `mmap`/`munmap` on top of nifler's edge. Those
   were added to the shared shim (`../nifler_temen/nifler_shim.c`) as a malloc + read-the-region over
   the `fs` cap — a read-only file map is observationally just the file's bytes at a stable pointer.
   Everything else (the fd/dir/stat + FILE\* + errno surface) is unchanged; the residual undefs are
   all on-ramp builtins + the never-reached math/spawn fringe (`--stub-externs`).

2. **Multi-file fixture.** hexer's input is a *semchecked* `.s.nif`, so the build first runs the real
   `nimony c` on a Nim program to fill a nimcache, then feeds the resulting `.s.nif` (+ its index +
   the `system` module's) to both native hexer and hexer-on-Temen and diffs **every file each
   produces** (the `.x.nif` Leng output and the `.dce.nif`). The guest's cwd is `/`, so hexer's
   `absolutePath("x.s.nif")` collapses to the cap-relative key the memfs is seeded under.

The guest runner is the generic `examples/nimphase_run.rs` (seeds a fixture dir → runs any argv →
dumps every produced file), shared by every nimony-phase differential.

## Run it

```sh
# needs: the nimony submodule, stock nim (2.3.x), the built nimony + hexer binaries, clang-18/llvm-18, cargo
NIMONY_BIN=<repo>/.nimtool/nimony/bin/nimony  HEXER_BIN=<repo>/.nimtool/nimony/bin/hexer \
  bash build_hexer_temen.sh
```

Fail-soft **SKIP** without the toolchain (nimony bootstrap needs Nim 2.3.x devel — **not** in the
per-PR CI, NIM.md §2). The ~3 MB `.temen` is a build artifact, not committed; this script is the gate.

## Status

✅ **hexer-on-Temen done.** Lowers `inputs/*.nim`'s semchecked NIF to Leng on the Temen, every produced
file byte-identical to native, on treewalk · bytecode · jit. With slice 1 (`nifler`), two of the
three nimony front-end phases now run on the sandbox. **Next:** `nimsem` (the sema phase) —
it compiles to C, on-ramps clean (zero shim delta), and translates + verifies to an Temen module, but
traps `Unreachable` early in startup with **no guest output**. A correctly-typed loud-stub sweep of
the entire math/proc/spawn fringe fired *none* of them, so this is **not** a missing shim symbol
(as `mmap`/`readlink` were) — it's a genuine temen-llvm on-ramp codegen gap reached before any I/O,
i.e. a lowering issue to pin in the translator, not the bottom edge. Then the W4 driver chains the
real phases (slice 3) and feeds the Leng into `temen-leng.temen` → the browser card (slice 4).
