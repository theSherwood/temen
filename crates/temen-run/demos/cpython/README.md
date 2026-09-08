# CPython on the LLVM on-ramp — feasibility SPIKE + gap inventory (slice D, #1328)

**This is a SPIKE, not a landed demo.** CPython does **not** run on Temen yet. This directory is the
reproducible pipeline + the honest gap inventory that answers *"how far is real CPython, and what
would it take?"* — the CPython analog of the Postgres spike (`../postgres`), which is exactly how
that bring-up started. The plan and the strategic framing (why CPython is a separate, ~3–5× larger
bring-up than MicroPython, and how it leans on the upstream `wasm32-wasi` / Pyodide static ports)
live in **`PYTHON.md` §4 (Phase D)** and **§5 (gap register)**.

MicroPython already **runs** on Temen, byte-identical to native, on the CLI and live in the
playground (`../micropython`, PRs #1331/#1337). CPython is the "real Python" stretch.

## What works today (the pipeline)

`build_bitcode.sh` reproduces the whole front half at Postgres scale:

- **Clone → configure → native oracle.** CPython **3.13.1**, `CC=clang --disable-shared
  --without-ensurepip --disable-test-modules --without-mimalloc --with-system-ffi=no` → a working
  21 MB native `python` (the differential oracle).
- **Per-TU bitcode.** `emit_bc.py` (the Postgres mechanism: capture each object's `make -n` clang
  command, re-emit with `-emit-llvm -fno-vectorize -fno-slp-vectorize`) compiles **278/278 TUs**
  across `Modules/ Objects/ Python/ Parser/ Programs/` with zero failures.
- **Whole-program link.** The exact `python` link line (not the whole `make -n` output — that also
  prints the `_freeze_module` / `_bootstrap_python` build tools, which redefine
  `_PyImport_FrozenBootstrap` and break `llvm-link`) maps to **175 `.bc` modules → one 40 MB
  linked module**.
- **Translate.** The on-ramp ingests the 40 MB module and walks it until the first fail-closed gap.

So the *front half of the "3–5× larger bring-up" already works*: CPython-scale C compiles, links, and
enters the translator. The remaining cost is the **libc/OS waist** and a couple of **on-ramp gaps**,
below.

## Reproduce

```sh
# needs: git, clang, llvm-link, llvm-nm, make, a working host cc for configure checks
(cd crates/temen-llvm && cargo build --release --bin temen-llvm-translate)   # once
bash crates/temen-run/demos/cpython/build_bitcode.sh                         # ~10–15 min
cat $TEMEN_CPY_CACHE/translate.err     # the current first gap (default cache /tmp/temen_cpython_cache)
```

## The gap inventory (as of this spike, CPython 3.13.1)

Walked one fail-closed gap at a time (the QuickJS/Postgres method). Two gaps found so far; the walk
is **not** complete — each closed gap exposes the next, and the tail is the full waist.

| # | Gap the translator surfaced | Where | Class / fix | Blocker? |
|---|---|---|---|---|
| 1 | `thread-local _mi_heap_default has an initializer the TLS block layout can't resolve` | bundled **mimalloc** (`Objects/mimalloc/`, CPython 3.13's default allocator) | **On-ramp limitation: initialized thread-locals.** Sidestepped by `--without-mimalloc` (falls back to pymalloc/obmalloc). Notably, the *next* gap was **not** TLS — so the initialized-TLS need was mimalloc-specific here, not pervasive across CPython's own thread-state (a positive signal). A future threaded CPython would still need initialized-TLS support. | No (config) |
| 2 | `constexpr reference to @malloc` (and `@free`) | `Python/hashtable.c`'s default allocator table: `constant { ptr @malloc, ptr @free }` | **Address-taken libc in a const table** — the exact class as QuickJS's `&sin` in the `Math` table. The on-ramp synthesizes `malloc` for *calls* but cannot mint a funcref for an *address-taken* `malloc`, and the raw `malloc`/`free`/`realloc` are undefined in the link (Postgres's `mem_shim.c` defines pg-specific wrappers, not raw `malloc`). **Fix:** link a real **guest allocator** (real `malloc`/`free`/`realloc`/`calloc` over the Memory cap), the same way QuickJS links guest openlibm for `&sin`. This is what a full bring-up (and Pyodide) does anyway. | No (waist) |

Neither gap is a fundamental "the on-ramp can't do this" wall — gap 1 is a config choice, gap 2 is a
known waist mechanism. The **honest read** matches `PYTHON.md` §4 Phase D: the remaining work is
**assembling the CPython libc/OS waist** (the Postgres spike needed ~20 shim files:
`malloc`/`printf`/`scanf`/`strtod`/`ctype`/`locale`/`os`/`time`/…, most as *real* guest functions
because CPython address-takes them), plus **freezing the stdlib** into the binary (no `dlopen`), then
grinding the tail of the gap-walk (the giant `_PyEval_EvalFrameDefault` computed-goto is expected to
translate — QuickJS proved that shape).

## Go / no-go

**Feasible, but a multi-step bring-up — not a quick win.** The pipeline works at scale; the two gaps
found are ordinary (config + waist), not architectural. The dominant remaining cost is the waist +
frozen stdlib + closing the long tail of address-taken-libc / OS gaps — the "~3–5× MicroPython"
estimate holds. The strongest accelerant remains cribbing CPython's own **`wasm32-wasi` / Pyodide**
static build config (frozen stdlib, no-dlopen, stubbed threads/signals) rather than reinventing it
(`PYTHON.md` §4 Phase D).

**Recommendation:** land MicroPython breadth first (cheap, visible); take CPython as a funded,
multi-PR effort when "real Python" is worth it. Next concrete step on this spike: link a guest
allocator to clear gap 2 and surface gaps 3…N, so the waist's true size is measured, not estimated.

## Files

- `build_bitcode.sh` — clone → configure → native oracle → `emit_bc.py` → `llvm-link` → translate.
- `emit_bc.py` — per-TU bitcode via the makefile's own clang flags (the Postgres mechanism).

CPython is **not vendored** (PSF license; cloned + cached at build time via `git`).
