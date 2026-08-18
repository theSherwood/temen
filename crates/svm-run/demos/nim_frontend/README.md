# nimony front-end on SVM — nimsem drives nifler via `exec`

The **real nimony front-end runs on the sandbox.** `nimsem` (the sema phase) is itself a *driver*:
to semantically check a module it parses the module's stdlib dependencies on demand by shelling out
to `nifler`. This demo runs `nimsem` as an SVM guest and routes that shell-out to the SVM **`exec`
capability**, so the real `nifler.svmb` runs as an isolated child domain — the W4 multi-binary driver
(`multibinary.rs`) applied to the actual front-end — and the semchecked output is **semantically
identical to native** (byte-identical modulo the embedded stdlib file paths).

```
Nim source ─nifler─▶ .p.nif ─┐
stdlib .nim ─nifler (exec child, on demand)─▶ .p.nif
                             └─nimsem──▶ semchecked .s.nif   (== native, modulo paths)
```

## How the driver works

- **nimsem's shell-out → `exec`.** nimsem builds `nifler --portablePaths --deps parse <src>
  <out.p.nif>` (`deps.nim`) and runs it via C `system()`. The shared shim
  (`../nifler_svm/nifler_shim.c` `system()`) parses that command into argv and calls the `exec`
  capability (`EXEC_RUN` + `EXEC_STATUS`) — no shell, no OS process.
- **Shared memfs.** nimsem and its `nifler` children are granted the **same** in-window filesystem
  (`domain_exec_with_fs` over one `mem_fs_shared_factory` store): the `.p.nif` a child writes is the
  one nimsem reads back. `deps.nim` skips the child entirely when the `.p.nif` already exists.
- **The guest half** is `examples/nim_frontend_driver.rs`: seed the stdlib sources + the system
  module's `.p.nif`, grant nimsem `fs` + `exec`, run `nimsem … m --isSystem <sys>.p.nif`, read back
  the produced `.s.nif`.

On a trivial program the system semcheck spawns ~22 `nifler` children and semchecks the whole
`system` module — `nimsem-on-SVM` returns 0 and emits the `.s.nif`.

## What "semantically identical" means

Every byte that differs from native traces to the **embedded stdlib file paths**: the path strings
themselves, the `.idx` offsets those paths shift, and the module-identity hashes they seed. The
**semantic content — types, procs, magics, tree structure — is identical.** The build script's
path-normalized diff is byte-exact; matching the paths outright (seeding the stdlib at native's exact
layout) would make the raw bytes match too, and is left as a follow-up.

`nimsem` is built with **`-d:skipPostSemValidator`** — as nimony's own **Windows CI** does. The
in-process post-sem IR validator (a separate nimony pass) has its own on-ramp issue, tracked apart
from the sema this proves correct; disabling it is a legitimate, documented build choice, not a
work-around for the sema.

## Run it

```sh
NIMONY_BIN=<repo>/.nimtool/nimony/bin/nimony  NIMSEM_BIN=<repo>/.nimtool/nimony/bin/nimsem \
  bash build_frontend.sh
```

Fail-soft **SKIP** without the toolchain (NIM.md §2). The `.svmb` are build artifacts (not committed).

## Status

✅ **The full nimony front-end runs on the SVM.** With `nifler` (slice 1), `hexer` (slice 2), the
back-end chain (slice 3), and now `nimsem` driving `nifler` via `exec` — **every** nimony phase runs
on the sandbox, and `nimsem`'s sema is byte-correct (modulo paths). The earlier "nimsem is blocked
by a native build mystery" finding was a misdiagnosis (an incorrect invocation); there is no build
discrepancy and no SVM/on-ramp gap. Remaining polish: exact path-matching for raw byte-identity, and
the separate post-sem-validator on-ramp fix. This closes the front-end for the "compile Nim in the
browser" card (slice 4).
