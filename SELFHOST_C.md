# SELFHOST_C.md — a C compiler that runs *on* Temen and compiles to Temen IR

Status: **scoping / design doc**, written 2026-07-24. No code yet — this is the work-breakdown
and the load-bearing decisions. The *why* for the pieces it leans on lives in `FRONTEND.md`
(the chibicc frontend), `LLVM.md` (the AOT on-ramp), `POSIX.md` (libc-as-capabilities),
`EXEC.md`/`PROCESS.md` (`domain_exec`, module instantiation), and `temen-fs` (memfs). This doc is
the *what/how/when*; it does not restate those.

## 1. Goal

**A C compiler that runs as an ordinary Temen guest and compiles C source → Temen IR from inside the
sandbox.** The browser playground (INTERACTIVE_EMBEDDING.md W5) is one deployment of this, not the
point — the same guest artifact runs under `temen-run` natively, in a `domain_exec` child, or in the
wasm build.

**The capstone is a playground demo**: in the browser, pick or edit any of several real C
programs, compile them **in-browser** (`chibicc.temen` running on the bytecode engine), run the
result, and see its output — the full edit → compile → run loop, client-side, no server (§7 step 5).

"Self-hosting" here means **the toolchain runs on the platform it targets** — Temen hosts its own
C→IR compiler — *not* that the compiler is bootstrapped by compiling itself. That distinction is
deliberate; see the decision in §3.

## 2. One chibicc, two build forms

There is **one chibicc**: one source tree (`frontend/chibicc/`, including the `codegen_ir.c`
C→TEMEN-IR backend), no fork, nothing maintained twice. It exists in two *build forms*:

1. **The native binary** (built by `cc`, as today) — a build/dev tool: the test harness
   (`c_frontend.rs`) shells out to it, and it is the **byte-match oracle** the guest form is
   differentially validated against (§5 E).
2. **`chibicc.temen`** — the *same source* compiled through the LLVM on-ramp into an Temen IR module
   (§3). This is the artifact this doc is about: run it as a guest and C compiles *inside* the
   sandbox, producing the same IR the native binary would.

Nothing about the plan requires divergent code paths between the two — same `main.c`, same
`codegen_ir.c`; the native binary sticks around only because the dev/test loop uses it.

## 3. Load-bearing decision — build chibicc-the-guest with the **LLVM on-ramp**, and keep it

The guest artifact (`chibicc.temen`) is produced by compiling chibicc's C through the **AOT LLVM
on-ramp** (`clang -O2` → LLVM bitcode → `temen-llvm-translate` → `prep_temen`), **not** by the chibicc
frontend compiling itself.

Rationale (owner, 2026-07-24): the on-ramp inherits **LLVM's full optimization pipeline**, so the
resulting IR is substantially faster than what `chibicc-frontend + temen-opt` would emit — our
frontend does minimal optimization and `temen-opt` is younger than LLVM's `-O2`. Since the compiler
binary is large and runs often, we spend LLVM's optimization budget **once, at build time**, and
**ship the optimized artifact**. A true stage-2 self-compile (chibicc-frontend compiling chibicc)
is kept only as an *optional conformance differential* (§7 E), never the shipping path.

This is the **exact pattern already proven for Postgres** (`browser/build-pg-assets.mjs`:
`temen-llvm-translate` + `prep_temen` → a committed `*.temen` asset, regenerated when Temen code changes).
Confidence is high: the on-ramp already runs **QuickJS and Postgres byte-identical to native**
(`LLVM.md`); chibicc is far simpler C. The on-ramp shells out to `clang`/`llvm-dis` at build time —
a dev/CI lane, off the runtime path — which is fine for producing a shipped artifact (same as the
Postgres asset), including for the browser (the `.temen` is a static asset).

## 4. What already exists (substrate — do not rebuild)

| Piece | State | Where |
|---|---|---|
| C→TEMEN-IR backend (`--emit-ir`, `-g`, `--child-entry`) | Built; runs real libraries byte-identical to `cc` | `frontend/chibicc/codegen_ir.c`, `FRONTEND.md` |
| LLVM on-ramp: real C/C++ → Temen IR, optimized | Built; QuickJS + Postgres **run** byte-identical | `temen-llvm`, `temen-llvm-translate`, `LLVM.md` |
| Committed `.temen` build-asset flow (on-ramp → `prep_temen`) | Built | `browser/build-pg-assets.mjs`, `crates/temen-run/demos/postgres/` |
| libc-as-capabilities: unresolved named imports → personality ops at load | Built (core surface, ops 0–20: stdio, `malloc`/`free`, `open`/`read`/`write`, dirs, `getenv`, `exec`, `dlopen`) | `temen-posix`, `POSIX.md` |
| In-memory filesystem (seed/image, dirs, fd table, crash-consistency) | Built | `temen-fs` |
| Guest spawns/runs a child module | Built (`domain_exec`, 2026-07-23) | `temen-run/src/exec.rs`, `EXEC.md` |
| Verifier re-checks every loaded module (frontend/guest is untrusted) | Built; the invariant | `temen-verify`, INVARIANTS.md §9, `DESIGN.md` §2a |

The compiler-guest slots into this like the shell (`STAGE1.md`) and Postgres do: **its libc calls
are unresolved named imports the POSIX personality resolves at load** (`POSIX.md` step 1 — "chibicc
/ `temen-llvm` emit these"). Nothing about the compiler is special to the substrate; it is just
another real C program whose output happens to be Temen IR.

## 5. What's required

### A. Produce `chibicc.temen` via the on-ramp
`clang -O2 -emit-llvm` over the chibicc sources (compiler in `-cc1 --emit-ir` mode only — no driver
`exec`/`glob`/temp-file paths) → `temen-llvm-translate` → `prep_temen`, mirroring `build-pg-assets.mjs`.
Emit with `-g` so a compile running under the W1 debugger has source info. **Risk: low** (on-ramp
handles Postgres/QuickJS).

### B. libc coverage — **reuse the proven Postgres guest-libc shims** (see Appendix B)

Step-2 investigation found this is *not* a write-from-scratch job: `crates/temen-run/demos/postgres/`
already contains a byte-exact-vs-glibc guest libc — `printf_shim.c` (floats delegate to the
on-ramp's correctly-rounded bignum dtoa `__vm_fmt_{fix,sci,gen}`, so `%.17g` "just works" — F2
dissolves), `stdio_shim.c` (`FILE*`), `os_shim.c` (the file syscalls over `__vm_cap_resolve("fs")`
+ `__vm_host_call`, with stdout/stderr fd-dispatched to the powerbox Stream), `mem_shim.c`,
`libc_shim.c`, `strerror_shim.c`, `time_shim.c`, `proc_shim.c`, `shim_errno.h`. These already run
**Postgres byte-identical to native**, so the compiler's libc is *assembly of proven parts* plus a
short chibicc-specific remainder. Full mapping + the missing list: **Appendix B**.

The original per-op split still holds for the substrate reasoning below:
chibicc's measured runtime footprint (from its own source): stdio **with files and buffered
`FILE*` streams** (`fopen`/`fread`/`fwrite`/`fclose`/`fflush`/`fputc`, `stdin`/`stdout`/`stderr`,
`fprintf`/`vfprintf`/`snprintf`/`vsnprintf`), `malloc`/`calloc`/`realloc`/`free`, `exit`, `strtoul`;
the pure-computation set (`strcmp`×117, `strncmp`, `strlen`, `strdup`/`strndup`, `strchr`/`strstr`,
`memcpy`/`memset`/`memmove`, `isdigit`/`isspace`), `assert`, and `va_start`/`va_arg`. Driver-only
calls (`glob`, `mkstemp`, `unlink`) are **avoidable** in `-cc1 --emit-ir` mode.
- Authority-bearing ops (`fopen`/`fread`/`malloc`/`getenv`…) → **host caps** (mostly landed).
- Pure-computation ops (`str*`, `ctype`, the `printf`-formatting family, `qsort`) → carried as
  **guest C** (compiled *with* chibicc, no authority) or host, per the `POSIX.md` split.
- **Gap to audit first:** buffered `FILE*` stdio and the full `printf`-formatting family against the
  current personality. Much overlaps what **Postgres already needed** (`snprintf`/`strtod`
  byte-exact) — so this is an audit-and-fill, not a greenfield libc.
- **→ Audit done — see Appendix A** (2026-07-24): the verified per-call inventory, what the
  personality already serves, the fill-list, and four concrete findings (missing
  `calloc`/`realloc`; `%.17g` float formatting is required; a `long double` on-ramp risk with its
  mitigation; driver-only externs are load-fatal and must be excluded from the guest build).

### C. Source + includes in the guest filesystem
Seed **memfs** with the user's `.c` and the bundled `frontend/chibicc/include/*.h`, so the
preprocessor resolves `#include`. memfs already supports seeding + imaging.

### D. Close the loop: compiler output → running code — **no substrate changes required**
`codegen_ir.c` emits **text IR**. Closing the loop needs *zero new Temen ops*; the existing seams
cover every deployment shape, layered by how far "inside" the loop runs:

1. **v1 — the embedder closes the loop** (exactly as it does for the native compiler today).
   The compiler-guest reads `.c` from memfs and writes its IR output back to memfs; the *embedder*
   picks it up, assembles/verifies/loads, and runs it — `temen-run` natively, the cdylib's existing
   `temen_parse` in the browser. The compiler is just a program whose output is a file. Nothing new
   anywhere.
2. **In-guest compile-and-run — already built: the §22 `Jit` capability + `vm_dlopen`.** A guest
   can hand serialized Temen IR from its own window to the host, which runs the fail-closed
   **rewrite-then-verify** gate (`jit_resolve_and_validate`: `decode_module` →
   `resolve_imports` → `verify_module` → install) and returns callable `call.dyn` slots.
   The C-level loader (`<vm_dl.h>`: `vm_dlopen`/`vm_dlsym`/`vm_dlclose`) is **built and
   differentially tested** (DESIGN.md §22, "In-window dynamic linking — SETTLED"). So
   *compile → load → call, entirely in-guest,* is a composition of existing pieces. The one
   guest-side gap: `vm_dlopen` takes **binary** serialized IR (`decode_module`), and chibicc emits
   text — so this layer needs a small **binary emitter in `codegen_ir.c`** (guest code, the wire
   format is the simple single-pass `temen-encode` form). No host change.
3. **Run-as-a-child-domain** (`cc x.c && ./x` as a separate process): today's spawn surface
   (`instantiate_module` op 5/13, `domain_exec`) takes **host-verified `Module`s** from the
   embedder's registry, not guest-window bytes — so for now the *embedder* mediates (it watches the
   output file, verifies, registers, and the shell spawns it by name). That is host-application
   glue, not substrate. A bytes-taking spawn op is **explicitly not needed** for this plan and
   stays unproposed unless a concrete consumer (the in-sandbox shell running freshly-compiled
   commands as true child domains) demands it — at which point it is a deliberate substrate
   discussion, not a rider on this doc.

In every layer **the verifier runs on the guest-produced module at load** (INVARIANTS.md §9 /
`DESIGN.md` §2a; §22's rewrite-then-verify). The compiler-guest is untrusted like any frontend; a
compiler bug is a clean error, never an escape.

### E. Validation
- **Differential vs native:** chibicc-the-guest's IR output byte-matches native
  `chibicc --emit-ir` on the same source (extend the `c_frontend.rs` oracle through the guest path).
- **End-to-end:** compile a C program *inside* Temen, `domain_exec` the result, and match the native
  build's stdout/exit (the `c_frontend` two-tier pattern, one level up).
- **Optional stage-2 conformance:** chibicc-the-guest compiles chibicc's *own* source; the emitted
  module matches (byte-for-byte IR) what the on-ramp-built artifact was compiled from. A purity
  check on the frontend, **not** a shipping requirement (§3).

## 6. Invariants this must respect

- **The compiler-guest is untrusted.** It is a frontend (§2a): the verifier re-checks its output.
  No self-hosting convenience may bypass verification of a produced module.
- **No new substrate.** The loop closes over existing seams (§5 D): the embedder, and the built
  §22 `Jit` cap / `vm_dlopen`. No new host ops for the compiler; host stays mechanism
  (INVARIANTS.md §1/§4).
- **libc semantics live outside the escape-TCB match** (`POSIX.md`): the personality is guest
  policy over masked host caps, not trusted core.
- **One artifact, rebuilt on TEMEN-code change.** `chibicc.temen` is code-coupled (like
  `postgres_resolved.temen`): regenerate it when the IR/ABI/encoder changes, and gate that rebuild
  in CI so a drift breaks the PR that caused it (the Postgres asset lane is the template).

## 7. Build order

1. **libc-coverage audit** — chibicc's footprint (§5 B) vs the POSIX personality; list the missing
   `FILE*`/formatting ops. Cheap, decides the size of B.
2. **Build `chibicc.temen`** via the on-ramp (§5 A), a `build-chibicc-asset` script mirroring
   `build-pg-assets.mjs`; run it (no libc yet) to shake out translation.
   **→ Done 2026-07-24** — `crates/temen-run/demos/chibicc_selfhost/` (`build_chibicc_temen.sh` +
   `cc1_main.c`, the driver-free guest entry). The full cc1 TU set translates through the on-ramp
   on the first pass: **258 functions, ~287 KB `.temen`, verifies, bytecode-compiles** (decode 6 ms /
   verify 1.2 ms / bc-compile 4.3 ms via `prep_temen`). Translated with `--stub-externs`
   (step-3 libc pending) + `--host-page 65536` (browser target) + `-mlong-double-64` (F3).
   Ground truth of what must be filled is now measured, not scanned: Appendix A.5.
3. **Fill libc** (§5 B) until chibicc-the-guest compiles a trivial C file to text IR against
   memfs-seeded source (§5 C), matching native `--emit-ir`.
4. **Close the loop** (§5 D) — v1: embedder assembles/loads the output (no new code beyond glue);
   then the in-guest layer: a binary emitter in `codegen_ir.c` feeding the existing `vm_dlopen`;
   end-to-end compile-and-run inside Temen (§5 E).
5. **Browser deployment — the capstone playground demo.** Ship `chibicc.temen` as a playground
   asset and wire the W5 surface (INTERACTIVE_EMBEDDING.md): a demo where the user picks or edits
   **various real C programs** (seed with the proven demo corpus — hello, calc, sha256, sortvec,
   tiny-regex, raytrace, …), compiles them **in the browser** (`chibicc.temen` on the bytecode
   engine, source + `include/*.h` seeded into memfs), runs the compiled module, and sees its
   output in the pane. The encode step reuses the cdylib's existing `temen_parse`. Gated by a
   Playwright test in the `real-browser` CI job (the `browser-play-editor-test.mjs` pattern):
   compile ≥2 corpus programs in Chromium, run them, assert output matches the native build's.

   **→ Step-5 slice A done 2026-07-24 — engine parity settled (the prerequisite).** `chibicc_run.rs`
   takes `TEMEN_CHIBICC_BACKEND`; `run_selfhost_diff.sh` runs every case on **treewalk / bytecode / jit**
   and all three emit **byte-identical IR** vs native. Engine decision for the playground card:
   **bytecode engine** originally (`jit: false`). *(Superseded 2026-07-28: chibicc now runs on the
   **wasm-JIT** — the card's `jit: true` with the toggle default-on, falling back to bytecode only if the
   emit is unavailable. The old "chibicc uses floats → can't JIT" reasoning was already stale — the
   emitter gained f32/f64 — and the whole `_start` in fact emits: `compile_module_reactor` at entry 0 puts
   333/402 funcs on wasm and bounces the rest cross-tier. See the "wasm-JIT tier DONE" block below.)*
   The playground also runs other float guests (QuickJS REPL, the DAP-debugger demos) on bytecode via the
   per-demo `jit` flag (`browser/web/play.js`). chibicc's compiled *outputs* run on whichever tier their
   card selects.

   **→ Step 5 DONE 2026-07-24 — the capstone runs in the browser.** The playground has a "C compiler
   (chibicc → Temen)" card (`browser/web/play.js`, `kind: 'chibicc'`): edit C → the page runs
   `chibicc.temen` on the bytecode engine, seeding the source on an `fs` cap at `/in.c`
   (new cdylib export `temen_run_onramp_fs`, which generalizes the Postgres `pg_setup` memfs+argv path)
   → TEMEN-IR text → `temen_parse` → run → `main()`'s return value, all client-side. Built + verified:
   - **Local capstone loop** (`run_selfhost_diff.sh`): in-Temen compile-and-run of the return-value corpus
     (`corpus/{sum,sort,hash}.c`) matches native clang on every engine.
   - **Full browser path proven in Rust** (compile via `onramp_fs_exec` → `parse_module` → run via
     `onramp_exec`) and **in Chromium** — the Playwright gate (`browser-play-editor-test.mjs`) asserts a
     C program compiles-and-runs in-browser to an exact value with Temen IR in the output pane.
   - `build-onramp-assets.mjs` stages `chibicc.temen`; the `real-browser` CI job builds it so the test
     runs there (fail-soft → SKIP if the toolchain is absent, the Lua pattern).
   **→ `#include` + text output DONE 2026-07-27.** A small guest-C libc ships as **headers seeded under
   `/include`** (`browser/playground-include/*.h`, built into the cdylib via `playground_include_files`):
   `<stdio.h>` (`printf`/`fprintf`/`snprintf`/`puts`/`putchar`/`fwrite`/`getchar`/`fgets`),
   `<string.h>`, `<stdlib.h>` (`malloc` bump allocator, `atoi`/`strtol`, `qsort`, `rand`), `<ctype.h>`,
   `<stdbool.h>`, `<stdint.h>`, `<stdarg.h>`. Everything is guest C **compiled into** the program on
   `#include` — nothing is linked — so `printf` formats over the powerbox's ambient `write` and a
   text-emitting program **actually prints** (the playground pane shows the program's output above the
   emitted IR) instead of trapping on an unresolved `call.sym`. Integer/char/string/pointer conversions
   with flags/width/precision are supported, and **`%f`/`%e`/`%g` too** (**DONE 2026-07-28**): guest-C
   float formatting in `<stdio.h>`, **correctly rounded to the requested precision** and byte-identical
   to glibc for the values a demo uses. It is *not* a bignum shortest-round-trip dtoa (the on-ramp's
   `__vm_fmt_*` is hand-emitted IR, unusable from a chibicc program, and the powerbox exposes no dtoa),
   so `%.17g` on an arbitrary double and a couple of exact-tie roundings (an exact `0.5` at `%.0f`, the
   `0.015` boundary) can differ — a deliberate, documented gap, no host/TCB surface added. Gated by
   `browser/tests/chibicc_printf.rs` (compile-and-run a `printf` program vs its exact output — `<stdio.h>`
   floats, `<string.h>`/`<stdlib.h>`) and the `browser-play-editor-test.mjs` Chromium assertion
   (`#include <stdio.h>` + `printf`, incl. `%f`/`%g`, → real output in-browser). A caller image can still
   add/override headers (its keys win). Residual scope: shortest-round-trip floats, and larger libc surface.

   **→ Compile-speed pass DONE 2026-07-28.** Compiling `#include <stdio.h>` + `printf("hi")` on the
   bytecode interpreter went from **3.36 s / 495 KB IR → 1.41 s / 330 KB** (2.4× faster). Two changes:
   (a) the seeded libc headers are now `static inline`, and `codegen_ir.c` honours chibicc's `is_live`
   dead-code pass (the native `codegen.c` already did) — so `#include <stdio.h>` no longer compiles the
   ~13 unused libc functions (`puts`/`snprintf`/`fgets`/…) into every program, only `printf`'s reachable
   closure; (b) debug info is **off by default** in the guest driver (`cc1_main.c`) — the `debug.*` waist
   is ~a third of the IR, so a plain compile drops it; the playground passes `-g` only when the user opts
   into source-level C debugging (the Debug button — DEBUGGING.md). The remaining lever was getting chibicc
   off the bytecode interpreter and onto the wasm-JIT — done next.

   **→ wasm-JIT tier DONE 2026-07-28 — chibicc compiles on emitted wasm.** The card's compile pass (the
   slow half) now takes the **"wasm-JIT" toggle** (default on): chibicc's whole `_start` emits to wasm
   via `compile_module_reactor(&m, /*entry*/ 0, …)` — **333/402 functions run on emitted wasm**, and the
   ~69 reachable non-subset helpers (the `call.cap`/`call.import` wrappers `outline_cap_calls` hoists, all
   integer-signature) **bounce cross-tier to the interpreter** over the shared window through
   `env.call_interp`, so `fopen`/`read`/`write`/`exit` resolve against the powerbox and the seeded memfs.
   This settles the old "chibicc uses floats → can't JIT" worry for good: `_start` **is** in-subset (floats
   emit; the only refused scalar-float op, `fma`, isn't on chibicc's reachable path), so nothing is
   integer-only about it. Built on the **pre-existing single-shot JIT runner** (`JitOnrampRun`, the
   run-to-completion twin of the Doom `JitOnrampReactor`, added for Lua/SQLite): this slice added an
   **fs+argv opener** (`open_owned_run_fs`/`open_shared_run_fs` + the `temen_onramp_jit_run_open_fs` FFI
   export) that grants the same headless memfs powerbox the bytecode `onramp_fs_exec` does (shared
   `chibicc_card_image` + `chibicc_card_argv`, which honours the same `-g` debug-info flag) and seeds argv
   at `POWERBOX_ARGS_BASE`, plus the `runJitCompiler` JS driver (a sibling of `runJitModule`). **No
   substrate / TCB change** — the emitter, the interpreter, and the powerbox are all untouched. Gated by:
   - `browser/tests/chibicc_jit.rs` — a native `wasmi` differential (the `jit_module.rs` pattern): the
     JIT-emitted IR is **byte-identical** to the interpreter oracle (`onramp_fs_exec`), and the emitted
     program then parses + runs to its expected stdout.
   - `browser-play-editor-test.mjs` — Chromium asserts the card compiles-and-runs **on the wasm-JIT**
     in-browser (the `.state` message reports `(wasm-JIT)`, so a silent interpreter fallback fails), and
     the card's **"Prove interp ≡ JIT"** button (`proveChibiccParity`) shows byte-identical emitted IR
     across both tiers live in the page.
   A fallback to `temen_run_onramp_fs` (bytecode) remains if the emit is ever unavailable, so the card can
   never regress to "won't run". Residual: chibicc's compiled *outputs* still run on whichever tier their
   own card selects; shortest-round-trip floats and larger libc surface stay open (above).

   **→ Measured on V8 (2026-07-28, `browser/bench_chibicc_jit.mjs` — the threads cdylib on Node's
   WebAssembly, both tiers over the shipped path):** compiling a real program (a `printf` loop, a
   `string`+`stdlib` program, a nested-loop float program — each ~330 KB of IR) drops from **~1.9–2.2 s
   on the bytecode interpreter to ~70–80 ms on the wasm-JIT — a ~27× steady-state speedup**; even a
   trivial return-only program is ~9× (303 ms → 33 ms). The one-time cost is a **~430 ms cold warm-up**
   (emitting chibicc's `_start` + `WebAssembly.compile` of the ~1.2 MB emitted module) paid **once per
   page load** — the emitted module is the same for every program (the user's C is *data* in the seeded
   memfs, not part of the emitted code), so V8 code-caches it and every compile after the first reuses it.
   The bench also asserts the two tiers emit **byte-identical IR** (a second guard alongside
   `chibicc_jit.rs`), so it doubles as a perf-and-correctness regression check.

   **→ Larger libc + predefined macros DONE 2026-07-28.** The seeded playground libc grew from "enough
   for a `printf` demo" to "enough for real programs":
   - **New headers** (`browser/playground-include/`): `<math.h>` (a demo-quality guest-C libm — exact
     `fabs`/`floor`/`ceil`/`trunc`/`round`/`fmod`/`sqrt`, range-reduced-series `exp`/`log`/`pow`/`sin`/
     `cos`/`atan`/…; **not** correctly-rounded, same posture as the float formatter), `<assert.h>`
     (glibc-shape `file:line: Assertion …` → `abort`), `<limits.h>`, `<stddef.h>` (`offsetof`,
     `ptrdiff_t`), `<errno.h>` (a guest global — nothing sets it, but programs that read/clear it build).
   - **Additions**: `<string.h>` — `strncat`, `strspn`/`strcspn`, `strpbrk`, `strtok`, `strcasecmp`/
     `strncasecmp`, `strdup`/`strndup`; `<stdlib.h>` — `strtoul`/`strtoll`/`strtoull`, `strtod`/`atof`,
     `bsearch`, `div`/`ldiv`, `atoll`/`llabs`, `getenv` (→ NULL, no sandbox env); `<ctype.h>` —
     `isgraph`/`isblank`.
   - **Predefined macros wired** — the guest driver (`cc1_main.c`) now calls chibicc's `init_macros()`
     before preprocessing, so `__FILE__`/`__LINE__`/`__STDC__`/`__STDC_VERSION__`/`__SIZEOF_*`/
     `__linux__`/… are defined (real programs and `<assert.h>`'s `__FILE__`/`__LINE__` need them). Safe
     in-sandbox: `init_macros`' only host deps — `time`/`localtime`/`ctime_r`/`stat` for `__DATE__`/
     `__TIME__`/`__TIMESTAMP__` — are already stubbed in `chibicc_extra.c` (fixed 1970 epoch). This
     required a **`chibicc.temen` rebuild** (still 333 funcs, verifies, byte-identical IR across tiers).
   Gated by `browser/tests/chibicc_libc.rs` (a real ~90-line stats pipeline over `strtok`/`strtod`/
   `qsort`/`sqrt`/`assert`, an algebraic-math sweep, and the string/stdlib additions — compiled + run to
   exact output) plus a `browser-play-editor-test.mjs` Chromium assertion (`<math.h>` + `<assert.h>` +
   `strdup` → real output in-browser).

   **→ Multi-file guest compile DONE 2026-07-28.** A card program can now span **multiple files**. The
   compiler side already resolved it — chibicc-the-guest resolves quote-includes (`#include "x.h"`)
   against the source's own directory (`/`), which the seeded memfs serves — so the only missing piece
   was *providing* the files: the card has one editor. Added a `//// file: NAME` marker convention
   (`split_multifile_source`, in the cdylib so both tiers + the native tests share it): the text before
   the first marker is `/in.c`, and each marker seeds a sibling file the entry `#include`s (headers or
   extra `.c`, unity-build style; a `NAME` with `/` nests, its parent dirs registered). No marker ⇒ a
   single `/in.c`, unchanged — and because the card already routes the editor through
   `chibicc_card_image`, this needed **zero JS/FFI change** and **no `chibicc.temen` rebuild**. Gated by
   `browser/tests/chibicc_multifile.rs` (the splitter + a real 3-file project — entry + `.h` + `.c` —
   and a nested-dir include, compiled + run to exact output) and a `browser-play-editor-test.mjs`
   Chromium assertion.

   **→ Self-host libc + per-TU self-compile DONE 2026-07-28 — chibicc compiles its own source in the
   sandbox.** The remaining libc-surface lift landed: the playground `<stdio.h>` gained a **real
   buffered `FILE*`** — fd-backed *or* memory-backed — with `open_memstream`/`fopen`/`fread`/`fclose`/
   `fflush` (one `__pg_fwrite_raw` dispatcher behind every output path), modelled on the proven guest
   libc `chibicc_extra.c`. This is the load-bearing piece: chibicc's `format()` builds **every** string
   through `open_memstream` → `vfprintf` → `fclose`. Plus `strtold` (= `strtod` under `-mlong-double-64`),
   `strerror`, and the system-header stubs `chibicc.h` `#include`s — `<stdnoreturn.h>`, `<strings.h>`,
   `<glob.h>`, `<libgen.h>`, `<unistd.h>`, `<time.h>` (fixed 1970 epoch), `<sys/stat.h>`/`<sys/types.h>`/
   `<sys/wait.h>` (mostly inert — the sandbox has no processes/globbing/wall-clock; present so `chibicc.h`
   parses). With those seeded, **chibicc-the-guest compiles real chibicc `-cc1` translation units to
   valid Temen IR** — gated by `browser/tests/chibicc_selfhost.rs` (tokenize/strings/hashmap/unicode each
   compile → parse + verify; `strings.c` is the `open_memstream` proof) + an `open_memstream` runtime
   round-trip and a Chromium assertion.

   **→ Growable guest heap DONE 2026-07-28 (§8's "grow via `__vm_map`" follow-up).** chibicc-the-guest's
   allocator was a **fixed 24 MiB static arena** (`chibicc_extra.c`) — fine for one source file, but a
   whole-compiler working set overran it, and because the bump allocator returns `NULL` on overflow (no
   growth), chibicc kept building on failed allocations → **silently corrupted type nodes** (spurious
   "not a struct", "invalid operands", traps at nondeterministic points). Fixed by **deleting** the
   arena and chibicc's `malloc`/`realloc`: the LLVM on-ramp then synthesizes its **`vm_map`-growable**
   heap for them (the same one Postgres uses to grow a multi-GB heap), which commits reserved-tail pages
   on demand — an effectively unbounded heap. Side effect: with the 24 MiB arena gone, chibicc.temen's
   declared memory dropped from `size_log2` 25 → 21 (the arena *was* the bloat). Rebuilt trap-free (333
   funcs, verifies); every existing test still passes, and `codegen_ir.c` (chibicc's largest TU, ~1.4 MB
   IR) now compiles cleanly where the arena silently corrupted (gated, `#[ignore]`d as a ~70 s heavy run).

   **→ Separate-compilation model (native `cc -c` + link) DONE 2026-07-29 — the multi-TU mechanism.**
   The heap stopped being the wall; the next wall was source *organization*. chibicc's TUs are written
   for **separate** compilation — the whole-program-per-invocation backend either can't see a cross-TU
   callee at all, or (a naive unity amalgamation) hits file-local `static` helpers that reuse the same
   name for different functions across TUs (e.g. `eval2` is `static` in both `parse.c` and
   `codegen_ir.c`, with incompatible signatures). Rather than rename statics for a unity blob, we chose
   the honest native model, which the substrate *already* supports: `temen_ir::link` (`LinkUnit` +
   exports + relocations, proven by `dynlink.rs`) is a real static linker. So `codegen_ir` gained a
   `--emit-object` mode (native `cc -c`): each non-`static` function is `export`ed by name, and a call
   to a function *declared but not defined* in the TU lowers to a **function-symbol import**
   (`call.sym "name"` carrying the callee's real Temen signature) instead of the generic capability
   import — which `link` resolves to a direct cross-unit call. `static` stays internal per unit, so the
   name collisions *evaporate* (no renaming). Proven end-to-end (`crates/temen/tests/c_link.rs`): two/three
   C TUs compiled separately, linked, verified, run on interp+JIT. This generalizes — any multi-TU C
   program to Temen now links the same way, not just chibicc.

   **→ Cross-TU data + linked `_start` under the powerbox DONE 2026-07-29 — items (a) and (b)'s
   mechanism.** `--emit-object` now emits the **data** side too: each non-`static` global is published
   as a data symbol (`export … data`), a global's own address materializes as `data.self` and a
   cross-TU global's as `data.sym`, and a pointer initializer (`&global`, `char *p = "…"`, `int *p =
   &extern`) lowers to a `data.ptr … self`/`sym` relocation — so chibicc's shared `ty_int`/`ty_void`
   (each a `Type *` → an anonymous `Type` body) link across units. Running the *linked* program through
   `_start` needed one more link form: **`data.top`**, the post-link top-of-data the frontend emits for
   the `_start` data-SP and argv scaffold where a whole-program build bakes `i64.const data_end` (this
   unit's own top, wrong once the linker stacks every unit's data above it). Proven end-to-end
   (`crates/temen/tests/c_link.rs`): crafted units read cross-TU data on interp==JIT; a two-TU program
   runs through `_start` under the powerbox (return value + argv + stdout); and **real** chibicc source
   — `type.c` emits its `ty_*` data symbols + 13 `data.ptr` relocations, and `type`+`hashmap`+`unicode`+
   `strings` link and verify together.

   **What remains for the whole-compiler self-compile** (chibicc compiling its *own* source into one
   runnable module): (b′) a **self-host libc for the emit-object path** so all ~9 cc1 TUs not only link
   but *run* — `tokenize.c` wants `strtoul` declared (a header gap, identical in `--emit-ir`), and the
   runtime needs `malloc`/`printf`/file I/O (the LLVM-on-ramp self-host build synthesizes these; the
   emit-object path does not yet); (c) the **bootstrap-fixpoint** differential (§5 E). The cross-TU
   function + data linking, the `data.top` stack base, headers, and heap it all stands on are done.

   **→ Emit-object libc (b′) increment 1 DONE 2026-07-29 — the *mechanism* proven.** A from-scratch
   libc **designed for chibicc's own frontend** (`crates/temen/tests/fixtures/emit_libc/mini_libc.c`),
   not the clang-tuned aggregator: a bump `malloc`, the `mem`/`str` family, and a **varargs** `printf`
   that writes through the powerbox `write` cap. `demo1.c` malloc's an array, copies a string, and
   prints a deterministic line; the two units link (`link_with_manifest`, `write` a host-bound
   manifest import) and run through `_start` on interp==JIT with a byte-exact stdout oracle
   (`c_link.rs::links_and_runs_emit_libc_under_powerbox`). This pins that varargs printf + heap +
   cross-TU linking + a powerbox cap compose end-to-end through emit-object.

   *Increment 1 surfaced and fixed a real linker confinement bug.* The linker stacked units' data
   16-byte-tight, so the entry unit's read-only tail (a symbol-name string at the args-region boundary)
   shared a **host page** with the libc unit's writable arena; the D40 read-only protection (host-page
   granular) marked the page `PROT_READ` and the arena's first store faulted (`MemoryFault`). Fix:
   page-align each unit's data base to `POWERBOX_STACK_ALIGN` (64 KiB, the max host page), so units
   never share a host page and each unit's internal ro/rw page separation survives relocation
   (`temen-ir::link_impl`; regression `temen-ir::link_layout_tests`). Same class as the Doom read-only-page
   fault, now for stacked link units.

   **→ Emit-object increment 2a DONE 2026-07-29 — all ~9 cc1 TUs compile + link + verify.** The whole
   `-cc1` slice (eight upstream TUs + the `cc1_main.c` entry) compiles under `--emit-object` and links
   into one **verified** module (`c_link.rs::all_cc1_tus_compile_link_and_verify_under_emit_object`).
   Two things unblocked it:
   - **`selfhost_prelude.h`** (`-include`d) closes chibicc's own parser gap against modern glibc: when
     the compiler doesn't define glibc's `__REDIRECT`, `<stdlib.h>` takes its ISO C23 branch that
     declares `__isoc23_strtoul`/… behind attribute forms chibicc doesn't resolve and then
     `#define strtoul __isoc23_strtoul`, so `strtoul`/`strtol`/`atoi`/`strtold`/… land *implicitly
     declared* (fatal). The prelude force-includes plain prototypes. (The on-ramp never hit this —
     clang defines `__REDIRECT`.) Also fixed the harness's `-I`: chibicc reads `-I<dir>` **joined**
     (`argv+2`), so a spaced `-I dir` was a silent no-op masked by sibling resolution.
   - **The only unresolved *data* symbols across the whole cc1 are `stdin`/`stdout`/`stderr`** — a
     cross-TU data reference must resolve to a concrete address (it can't be a manifest import like a
     function), so the linker fails closed on them until the libc's stdio globals exist; a stub unit
     defining the three lets the whole cc1 link. Every `ty_*`/`opt_*`/`include_paths`/`base_file`
     resolves cross-TU. All remaining unbound names are the libc **function** surface + host caps —
     the exact increment-2b target: `fopen`/`fwrite`/`vfprintf`/`snprintf`/the `str`/`mem` family/
     `open_memstream`/`__ctype_b_loc`/… as guest C over the powerbox syscall + memory caps
     (`write`/`read`/`open`/`close`/`exit`/`stat`/`vm_map`). Note the **heap allocator is *not* on that
     list**: chibicc's emit-object synthesizes `malloc`/`calloc`/`free` **inline** (a bump heap that
     grows the window via the `vm_map`/`vm_page_size` builtins), so the libc need not provide them —
     a real simplification vs the clang-tuned aggregator.

   **→ Emit-object increment 2b (started) — the libc's intrinsic-free core compiles.** `emit_libc.c`
   is the emit-object twin of `chibicc_libc.c`, aggregating the parts of the guest libc that carry **no**
   on-ramp dependency: the allocator (from chibicc's *bundled* `<stdlib.h>` — `static`
   `malloc`/`calloc`/`free`/`realloc` growing the window via `__vm_map`, so `malloc` is never a
   cross-unit symbol), `mem_shim`/`libc_shim` (mem/str/ctype `__ctype_b_loc`/`strtoul`), `strtod`, and
   `chibicc_extra.c`'s fd-backed + `open_memstream` stdio — the last with its `free`/`calloc` now
   `#ifndef __TEMEN_STDLIB_H`-guarded (the bundled header already defines them under emit-object; the
   on-ramp's clang, on system headers, still gets them — verified `chibicc_libc.c` still builds under
   clang). This core compiles under `--emit-object` and is **intrinsic-free** — regression-guarded by
   `c_link.rs::emit_object_libc_core_compiles_and_is_intrinsic_free`, which also fails if the guard
   regresses. It defines the real `stdin`/`stdout`/`stderr` the 2a link stubbed.

   **→ Emit-object increment 2b DONE 2026-07-29 — the libc runs, and the whole cc1 links against it.**
   The two bottom-edge shims that `os_shim.c`/`printf_shim.c` couldn't supply under emit-object (they
   reach the host through temen-llvm intrinsics `__vm_stream_write`/`__vm_host_call`/`__vm_cap_resolve` and
   `__vm_fmt_*` that chibicc's own `--emit-object` codegen doesn't lower — its `scan_caps` knows
   `__vm_map`/`__vm_jit_`/… but not those) now exist:
   - **`os_emit.c`** — the emit-object os edge. It reaches the powerbox by plain `extern` `write`/`read`
     (bound by name to the Stream cap), not intrinsics; the fd≥3 filesystem path is stubbed (`open`
     returns −1) until the 2c fs cap, because the reference `default_cap_resolver` binds
     `write`/`read`/`exit`/`vm_*` but **not** `open`/`close`/`stat`, and a manifest module refuses to
     start with an unbindable import — so a real fs name here would make the linked cc1 un-runnable.
   - **`printf_emit.c`** — the `__vm_fmt_*`-free formatter. It reuses `printf_shim.c`'s pure-C engine
     verbatim (the whole `%d`/`%s`/`%x`/`%ld`/`%02d`/`%.*s`/`%+ld`/`%*s` surface) and supplies the three
     float helpers in guest C. The float path is **best-effort, not correctly-rounded** — a program with
     float literals (chibicc emits them `%.17g`, 58 call sites, all in float-constant codegen) needs the
     bignum dtoa ported to guest C, the float-input increment; on integer inputs it is never reached.

   Proven two ways (`c_link.rs`): (1) `demo2.c` drives the printf engine — radices, width/precision,
   flag combinations, `%*d`/`%.*s`, and the `FILE*`/`open_memstream` path — linked against the full
   `emit_libc.c` and run through `_start` on **interp==JIT** with stdout asserted **byte-for-byte against
   native glibc**; (2) the **whole cc1 (nine TUs + `emit_libc.c`) links and verifies with every remaining
   import a default-powerbox cap** (`write`/`read`/`exit`/`vm_map`/`vm_page_size`) — the 2a link's
   `fopen`/`vfprintf`/`strlen`/… manifest imports are all now supplied, and `stdin`/`stdout`/`stderr`
   are real definitions, not the 2a stub. The `chibicc_extra.c` allocator stays `__TEMEN_STDLIB_H`-guarded
   so the on-ramp `chibicc_libc.c` still builds under clang.

   **→ Emit-object 2c increment 1 DONE 2026-07-29 — the linked compiler *runs* and self-hosts a
   compile.** The whole cc1 (nine TUs + `emit_libc.c`) now runs through `_start` under the powerbox and
   **compiles a real C program to Temen IR inside the sandbox**, proven three ways at once
   (`c_link.rs::whole_cc1_self_compiles_a_program_matching_native_on_interp_and_jit`): the emitted IR is
   **byte-identical on the interpreter and the JIT** (§18), it **parses** as a real module, and it is
   **byte-identical to the native reference** (`chibicc_ref` — the same `cc1_main` entry built with
   system clang + libc, so the guest libc + Temen engine reproduce the native frontend exactly). Source is
   fed on **stdin** (`chibicc -` → `read_file("-")`) and IR comes back on stdout — the recognized
   `read`/`write` powerbox builtins already serve fd 0/1, so this needs no filesystem cap.

   *This surfaced and fixed a real multi-TU allocator bug.* chibicc's bundled `<stdlib.h>` keeps its bump
   pointer in file-scope `static` state; that is self-contained in a whole-program build, but under
   emit-object **every** cc1 TU `#include`s it and so minted its *own* `static __temen_brk` at the same
   256 MiB heap base — the per-TU allocators handed out **overlapping** addresses and corrupted each
   other (the guest's `hashmap` `unreachable()`d on a full-but-never-grown table). 2a/2b only linked +
   verified, so it surfaced only on this first *run*. Fix: the four allocator-state globals
   (`__temen_brk`/`__temen_committed`/`__temen_page`/`__temen_grow_lock`) are now **one shared instance** across
   the linked program — `selfhost_prelude.h` sets `__TEMEN_LIBC_EXTERN` so each TU sees an `extern`, and
   `emit_libc.c` sets `__TEMEN_LIBC_OWNER` to hold the single definition, resolved cross-TU by
   `temen_ir::link` exactly like chibicc's shared `ty_int` (the allocator *functions* stay `static`
   per-TU; only the bump pointer is shared). The whole-program on-ramp path is textually unchanged (the
   `#else static` branch). Subset-link tests that omit the libc gained a tiny `alloc_state_stub()` owner.

   **→ Emit-object 2c increment 2 DONE 2026-07-30 — the `#include`/filesystem path: the running
   compiler reads a header from an in-sandbox filesystem.** The blocker was real: `gen_builtin_stream`
   ignores fd (`write`→stdout, `read`→stdin always) and the generic host-cap lowering
   `gen_builtin_import` is off under `--emit-object`, so an emit-object guest could not reach an fs cap
   at all. Closed with a small **recognized-builtin** fs seam in the frontend (`codegen_ir.c`, purely
   additive — no existing emit changes, so **no `chibicc.temen` rebuild**):
   - `__vm_fs(op, a, b, c, d)` → `call.sym "vm_fs"` with the **op in arg0** — one manifest slot carrying
     the whole fs op protocol (open/read/write/seek/close/stat, `crates/temen-run/src/fs.rs`), mirroring
     `__vm_host_call(handle, op, …)` but as a single named import the on-ramp's `__vm_cap_resolve` path
     (unlowerable under emit-object) can't express.
   - `__vm_stream_write`/`__vm_stream_read` → `call.sym "stream_write"/"stream_read"` (the raw fd-less
     `Stream` primitives) — distinct names from the `write`/`read` builtins so `os_emit.c` can *define*
     `write`/`read` as fd-dispatchers (0/1/2 → stream, ≥3 → `__vm_fs`) without recursing, exactly like
     the on-ramp's `os_shim.c` does over its intrinsics. `open`/`close`/`lseek`/`stat` ride `__vm_fs`.
   The harness binds the six caps by name (`c_link.rs::cc1_imports` via `instantiate_with_imports` + a
   new `HostCap::memory(op)`): `stream_write`/`stream_read` → `Stream`, `exit` → `Exit`,
   `vm_map`/`vm_page_size` → `Memory`, and **`vm_fs` → a seeded `mem_fs`** (a thin wrapper forwards
   `args[0]` as the op). Gated by `whole_cc1_self_compiles…`: the linked compiler compiles a
   `#include <vec.h>` program **reading the header from the in-sandbox memfs**, byte-identical on
   interp==JIT and byte-identical to native `chibicc_ref` (which reads the same header from a real `-I`
   dir). The stdout/stdin edge stays on the `Stream` cap, so the fs cap is only files (fd≥3).

   **→ Emit-object 2c increment 3 DONE 2026-07-30 — the float-input dtoa: `%.17g` is correctly
   rounded.** `printf_emit.c`'s float path was best-effort double arithmetic; a float-literal source
   couldn't diff byte-exact because `%.17g` didn't guarantee the 17-significant-digit round-trip. Closed
   by porting temen-llvm's `synth_dtoa_*` IR reference (`crates/temen-llvm/src/lib.rs`) to guest C: a Steele
   & White scaled-**bignum** generator (fixed-width 32-bit-limb integers `fbig`/`fbig_*`, namespaced off
   `strtod.c`'s own `bn`) that emits the nearest half-to-even P-significant-digit decimal with *exact*
   integer arithmetic — no float ops in the digit loop, so the result is deterministic across
   interp/JIT/native and `%.17g` re-parses a `double` bit-identically. `__vm_fmt_{gen,sci,fix}` are
   rewritten around it: `%g` (strip trailing zeros, e-vs-f at exponent −4/P), `%e` (fixed fractional
   width), `%f` (significant digits down to 10^−fp, incl. the sub-precision rounding corner). Validated
   two ways: a native fuzz harness diffs the engine against glibc over edge cases + 200k random doubles
   (both signs, %g/%e/%f at many precisions) — **0 mismatches / 3.2M checks** — and a new guest gate,
   `emit_object_libc_float_runs_byte_exact_under_powerbox`, runs `demo_float.c`'s `%.17g`/`%g`/`%e`/`%f`
   battery through the real emit-object compile on interp==JIT, byte-for-byte against glibc's own output.

   **→ Stage-2 conformance slice #1 DONE 2026-07-30 — the guest compiles chibicc's *own* source,
   byte-matching native.** The linked whole cc1 compiles a real upstream chibicc TU (`hashmap.c`) —
   pulling chibicc.h's **full system-header closure** (~95 files: `<stdio.h>`/`<stdlib.h>`/`<string.h>`/
   … resolved through glibc's `bits/`, `sys/`, `asm/` trees, plus chibicc's bundled `include/`) from a
   seeded memfs — and the emitted IR is **byte-identical to native `chibicc_ref`** on interp==JIT. The
   hardest input on the path to the fixpoint: the TEMEN-executed compiler is faithful not just on crafted
   programs but on the compiler's own code. Mechanics (`whole_cc1_compiles_its_own_tu_matching_native`):
   the closure is discovered with `chibicc -M`, seeded at **repo-relative** keys (the fs cap refuses
   absolute paths — `read_path`); the guest searches `frontend/chibicc/include` + `usr/include[...]` in
   the memfs while native reads the real `/usr/include`, and only the TU's own `__FILE__` reaches the IR
   (no header paths), so the two stay byte-comparable. `cc1_main.c` gained multi-`-I` support for the
   multi-root search.

   **→ Widened 2026-07-30 — five real chibicc TUs, and `cc1_main` gained `-include`.** The differential
   now covers `strings.c`/`hashmap.c`/`unicode.c`/`type.c`/`tokenize.c` — the **tractable** upstream TUs
   (≤ ~800 lines; the guest runs on the tree-walk interpreter, so runtime scales with TU size, and the
   giants `preprocess.c`/`codegen_ir.c`/`parse.c` at 1.2k–3.4k lines are left to a future slow lane).
   `tokenize.c` surfaced the real next gap: it calls `strtoul`, whose modern-glibc ISO-C23
   `__isoc23_*` redirect chibicc's parser can't ingest — the exact reason `emit_object_real`
   force-includes `selfhost_prelude.h` when building the guest. `cc1_main.c` didn't support `-include`,
   so the guest couldn't take the prelude; added it (mirroring `main.c`'s cc1() token-prepend). The
   prelude is declarations-only ⇒ no IR of its own, and force-including it on **both** sides makes each
   TU emit the shared-`extern` allocator form (`__TEMEN_LIBC_EXTERN`) the real linked build uses — still
   byte-identical guest-vs-native. The heavy `cc1-self-compile` CI job's filter was broadened from
   `self_compiles` to run every `--ignored` c_link gate, so this test gates too.

   **→ All nine TUs 2026-07-30 — the fixpoint condition is met.** The three giants
   (`preprocess.c`/`parse.c`/`codegen_ir.c`) also compile guest-vs-native byte-identical on interp==JIT
   (~8 min for the three under the interpreter). They were faster than feared, but still past the per-PR
   budget, so they ride an opt-in test (`whole_cc1_compiles_giant_tus…`, gated on `TEMEN_SELFHOST_GIANTS=1`)
   run by a **nightly** `cc1-self-compile-giants` CI job (daily `schedule` + `workflow_dispatch`, like
   `miri`); the always-on job runs it too but it self-skips fast without the env var. With the five
   tractable TUs this is **per-TU byte-identity across all nine cc1 TUs** — which *is* the fixpoint:
   the guest deterministically emits the same objects native does, so linking the guest's objects
   reproduces the native-built guest (`chibicc2 == chibicc1`), and a byte-identical compiler on the same
   source reproduces its own output (`chibicc2 == chibicc3`). ∎

   **→ Mechanized 2026-07-30 — `chibicc2 == chibicc1` in code.** The `== chibicc1` step is no longer
   just an argument: `whole_cc1_relinks_from_guest_objects_equals_native` relinks a whole cc1 from the
   **guest's own emitted objects** — each of the nine cc1 TUs compiled by the running guest into a unit,
   linked with the same native `emit_libc` — and asserts the result **verifies** and is byte-identical
   to a native reference built with the *same relative flags* (the only variable is the substrate). One
   subtlety it flushed out: `link_whole_cc1`'s units embed *absolute* `__FILE__` paths (`emit_object_real`
   passes an absolute cfile), so the reference is rebuilt from `native_cc1_unit` (relative paths) to match
   the memfs-relative guest — otherwise the sole diff is `internal error at %s`'s source path. Rides the
   same opt-in `TEMEN_SELFHOST_GIANTS=1` nightly lane (it guest-compiles all nine, incl. the giants). With
   this, the fixpoint is closed both ways — per-TU byte-identity **and** the linked-module equality it
   implies.

   **→ In the browser 2026-07-31 — the self-host card, slice 1 (tractable TUs).** The self-host reached
   the **playground**: a new card, *"chibicc compiles its own source (self-host → Temen)"*
   (`browser/web/play.js`, `kind: 'selfhost'`), runs the shipped `chibicc.temen` in `--emit-object` mode
   over chibicc's *own* cc1 TUs, **client-side on the wasm-JIT**, emitting each TU's linkable object. Pick
   a TU from the dropdown → the page seeds a committed **closure image** (`chibicc_selfhost.img`: the TU
   sources + the ~96-file glibc header closure `chibicc.h` pulls + `selfhost_prelude.h`, built by
   `browser/build-selfhost-assets.mjs` via `chibicc -M`) on an `fs` cap and runs
   `temen_selfhost_jit_emit_object_fs` (a 128 MiB window — `codegen_ir.c`'s ~1.2 MB output overruns the
   single-file card's 32 MiB and traps `unreachable` mid-emit; measured) with a bytecode fallback. The
   emitted object is **byte-identical to native `chibicc --emit-object`**, gated in real Chromium
   (`browser-play-editor-test.mjs`: compiles `tokenize.c` in-browser, diffs the object against the native
   binary, and the card's "Prove interp ≡ JIT" shows both tiers byte-identical), plus a native
   `browser/tests/chibicc_selfhost_asset.rs` over all five tractable TUs.
   - *Surfaced + fixed a real stale-asset regression:* `--emit-object` (added to `cc1_main.c` in the
     emit-object work) was **never compiled into the shipped `chibicc.temen`** — the committed asset
     predated it in content, so `chibicc.temen --emit-object` broke (the flag fell through to `base_file`).
     The card would have been dead on arrival; rebuilding the asset (`build_chibicc_temen.sh`, now 335
     funcs) fixed it, and the two gates keep it from drifting again.
   - **Slice 1 scope:** the five tractable TUs (`strings`/`hashmap`/`unicode`/`type`/`tokenize`), each
     compiling in a few hundred ms on the wasm-JIT. Residual for slice 2: the three giants
     (`preprocess`/`parse`/`codegen_ir` — the 128 MiB window already handles them, measured) and the
     N-way **link → run chibicc2** capstone (the browser link FFI over the emitted objects + `emit_libc`).
6. **(optional) stage-2 conformance** differential (§5 E).

## 8. Open questions

- Binary emission (§5 D-2): a C encoder for the `temen-encode` wire form in `codegen_ir.c`, or keep
  the guest text-only and let the embedder assemble everywhere? Text-only defers the encoder but
  caps the in-guest story at layer 1; the encoder is small (the format is a deliberate single-pass
  design) and unlocks `vm_dlopen`. Decide when layer 2 has a consumer.
- ~~Where the pure-libc bulk lives~~ — **settled by the step-1 audit (Appendix A): guest C**, one
  small self-host libc translation unit compiled alongside chibicc at the `clang` step. ~~The one
  remaining allocator sub-question: `realloc` needs the old block size~~ — **settled 2026-07-28: neither
  a personality op nor a guest shim. chibicc leaves `malloc`/`realloc` undefined and the on-ramp
  synthesizes its `vm_map`-growable heap for them** (block metadata lives in the synth allocator's
  own header, so its synth `realloc` copies the right length). This also gave chibicc an unbounded,
  reserved-tail-growing heap — see the "growable guest heap" as-built in §7.
- ~~`-g` cost in the guest: always emit debug info, or a flag?~~ **Settled 2026-07-28: a flag, off by
  default.** The `debug.*` waist is ~a third of the emitted IR, so `cc1_main.c` defaults `opt_g` off and
  takes `-g` to turn it on; the playground passes `-g` only for a source-level debug session (the Debug
  button on the C card), and compiles clean/fast otherwise.

## 9. Non-goals

- A general POSIX C toolchain (assembler, `.o`/`ar`, an in-guest `ld`). Multi-TU builds use
  `--emit-object` (Temen text-IR units) linked by the host `temen_ir::link` — the Temen link model, not a
  POSIX object/archive format. The per-TU output is still `-cc1 --emit-ir`; there is no x86/ELF path.
- Matching GCC/Clang language breadth. chibicc's C99 + the frontend's proven coverage is the bar;
  heavy C/C++ stays on the AOT on-ramp lane (`LLVM.md`), which is not a runtime guest.
- Making the chibicc frontend self-compile as the shipping path (§3 — LLVM-built artifact ships).

---

## Appendix A — step-1 libc-coverage audit (done 2026-07-24)

**Method.** Verified extern scan of the `-cc1 --emit-ir` compilation set — `tokenize.c`,
`preprocess.c`, `parse.c`, `type.c`, `codegen_ir.c`, `strings.c`, `hashmap.c`, `unicode.c`, plus
`main.c`'s cc1 slice — diffed against the personality's op surface (`temen-posix` ops 0–20 and its
`resolve()` name map). Naive greps overcount (`glob` matches `global`, `access`/`stat` match
prose); every row below is a checked call site.

### A.1 Authority-bearing calls → personality ops

| call | where (cc1 path) | personality today | plan |
|---|---|---|---|
| `read` / `write` (fd-aware) | FILE* layer below | **✓** ops 0/1 (fd table, distinct stderr) | use as-is |
| `open` / `close` / `lseek` | FILE* layer | **✓** ops 5/6/7 | use as-is |
| `exit` | tokenize (`error`), codegen_ir | **✓** op 4 (`exit`/`_exit`/`_Exit`) | use as-is |
| `malloc` / `free` | codegen_ir, strings | **✓** ops 2/3 (window-offset allocator) | use as-is |
| `calloc` | **the workhorse** — 36 verified sites (tokenize 5, preprocess 10, parse 14, type 2, codegen_ir 1, strings 1, hashmap 3) | **✗ missing** | guest wrapper: `malloc` + `memset` — no host change |
| `realloc` | tokenize 1, codegen_ir 8, strings 1 | **✗ missing** | needs old block size: personality `OP_REALLOC` **or** guest size-header shim (§8) |
| `stat` | preprocess.c:1032 — only `__TIMESTAMP__` | **✓** op 13 | use as-is; memfs mtime=0 ⇒ the macro's `"??? ..."` fallback — deterministic builds, arguably a feature |
| `getenv` / `argc` / `argv` | arg delivery (crt) | **✓** ops 11/17/18 | use as-is |

### A.2 Pure-computation calls → guest C (one self-host libc translation unit, compiled with chibicc at the `clang` step)

- **Strings/memory:** `strcmp` (62 sites in codegen_ir alone), `strncmp`, `strlen`, `strdup`,
  `strndup`, `strncpy`, `strchr`, `strstr`, `memcpy`, `memset`, `memcmp`. Trivial C.
- **ctype:** `isdigit`, `isxdigit`, `isspace`, `ispunct`, `isalnum`. Trivial.
- **Conversions:** `strtoul`; `strtold` (see F3).
- **`FILE*` stdio over the fd ops:** `fopen`/`fclose`/`fread`/`fwrite`/`fflush`/`fputc`, with
  `stdin`/`stdout`/`stderr` as fds 0/1/2 (the personality's fd table already maps them). A small
  buffered-`FILE` struct over ops 0/1/5/6/7.
- **Formatting:** `fprintf`/`vfprintf`/`printf`/`snprintf`/`vsnprintf`. Format spectrum actually
  used: `%d`/`%ld`/`%u`/`%x`, `%s`, `%c`, `%.*s`, and — the hard one — **`%.17g`**: every float
  constant is emitted through it (`codegen_ir.c:1552`, `cg("... %.17g", (double)node->fval)`).
  See F2.
- **Misc:** `dirname` (relative `#include` resolution — pure string), `ctime_r` (only
  `__TIMESTAMP__`; stub), `assert` (macro → `exit`), `va_start`/`va_arg`/`va_end`
  (compiler-lowered; the on-ramp already handles varargs — QuickJS/Postgres).

### A.3 Driver-only externs — **must be excluded from the guest build (load-fatal otherwise)**

`glob`, `mkstemp`, `unlink`, `fork`, `execvp`, `basename` and the rest of `main.c`'s driver slice
(plus all of `codegen.c`, the x86 backend). This is not just hygiene: **import resolution is
fail-closed** — `temen_posix::resolve()` returns `None` for unknown names and every named import
must bind at load — so a linked-in-but-never-called `fork` **fails the module load**. The guest
build therefore compiles only the cc1-path files, with a small `#ifdef`/stub for `main.c`'s
driver branches and `codegen.c` excluded (`--emit-ir` never calls it).

### A.4 Findings

- **F1 — `calloc`/`realloc` are the only missing authority ops.** `calloc` is a guest wrapper;
  `realloc` is the one real choice (§8). Everything else chibicc needs from the host is already
  served by ops 0–20.
- **F2 — the guest printf needs real float formatting.** `%.17g` (shortest-round-trip double) is
  load-bearing for IR emission, so the existing mini-printf (known `%`-width/precision gaps,
  FRONTEND.md) is insufficient — the self-host libc needs a correct `%g`/precision dtoa. This is
  the single hardest piece of A.2; budget it accordingly (or port a compact proven dtoa).
- **F3 — `long double` on-ramp risk, with mitigation.** `tokenize.c:434` parses every float
  literal with `strtold` into a `long double fval` (x86 `fp80`), which the on-ramp likely can't
  lower. Mitigation: build the guest with **`-mlong-double-64`**. Divergence window: `fval` is
  cast to `double` at emission (`codegen_ir.c:1552`), so only literal-parsing **double-rounding**
  edge cases can differ from the native (fp80) oracle. Pin with a literal-edge-case differential;
  accept and document any residual ULP cases.
- **F4 — deterministic `__TIMESTAMP__`.** Via memfs, `stat` yields fixed mtimes ⇒ reproducible
  output where native chibicc's is time-dependent. Note it in the oracle comparison (mask the
  macro, or seed a fixed mtime).

**Net:** the fill is small and almost entirely guest-side — one libc `.c` (strings/ctype/`FILE*`/
printf-with-`%.17g`), a `calloc` wrapper, one `realloc` decision, and a cc1-only build set. No Temen
substrate change anywhere; at most one *personality* op (`OP_REALLOC`).

### A.5 Step-2 ground truth — the measured stub list (supersedes the source scan where they differ)

The step-2 build (`build_chibicc_temen.sh`) links the real cc1 bitcode and reports every undefined
symbol — the exact fill-set step 3 must provide, measured (41 symbols), not grepped:

> `__assert_fail` `__ctype_b_loc` `__errno_location` `bcmp` `calloc` `ctime_r` `dirname` `exit`
> `fclose` `fflush` `fopen` `fprintf` `fputc` `fread` `free` `fwrite` `localtime` `memchr`
> `open_memstream` `puts` `realloc` `snprintf` `stat` `stderr` `stdin` `stdout` `strchr` `strcmp`
> `strdup` `strerror` `strlen` `strncasecmp` `strncmp` `strncpy` `strndup` `strstr` `strtold`
> `strtoul` `time` `vfprintf` `vsnprintf`

Deltas vs the A.1/A.2 source scan, all explained:
- **glibc lowerings**: `assert` → `__assert_fail`; the ctype macros → `__ctype_b_loc` (one table
  accessor covers all `is*`); error paths → `__errno_location` + `strerror`; `-O2` rewrites
  `memcmp`→`bcmp` and simple `printf`→`puts`. The guest libc provides these *names* (or step 3
  compiles against its own non-glibc headers, making the plain names reappear).
- **`open_memstream`** — missed by the source scan: `strings.c`'s `format()` (used everywhere)
  builds strings through a memory `FILE*`. The guest `FILE` layer needs a memstream mode.
- **`time`/`localtime`/`ctime_r`** — the `__DATE__`/`__TIME__`/`__TIMESTAMP__` macros. Stub to a
  fixed epoch: deterministic builds (F4).
- **`strncasecmp`** (preprocessor), trivial.
- `str.*`/`mem.*` inlining means `memcpy`/`memset`/`malloc` don't appear by name at `-O2` — clang
  emits intrinsics the on-ramp already lowers; do not be surprised the list is *shorter* than A.2.

**Lesson wired into the build**: `--stub-externs` also stubs *chibicc's own* functions if a
defining TU is missing — `align_to` (owned by the excluded `codegen.c`, called 15× for struct
layout) translated and verified fine as a trap-stub time bomb. `cc1_main.c` now defines it, and
the script's **step 2a stub audit fails on any undefined symbol outside the allowlist above**, so
an excluded-TU gap can never reach the artifact again.

---

## Appendix B — step-3 plan: assemble the guest libc from the Postgres shims (2026-07-24)

Step-2's stub list (A.5) maps onto the existing `demos/postgres/` guest-libc shims as follows —
**26 of 41 symbols are already provided by proven, byte-identical-to-native code**:

| Provided by (Postgres shim, reusable) | chibicc symbols it covers |
|---|---|
| `printf_shim.c` | `fprintf` `vfprintf` `snprintf` `vsnprintf` (+ the `%.17g` path via `__vm_fmt_gen` — **F2 solved**) |
| `stdio_shim.c` | `fopen` `fclose` `fread` `fwrite` `fflush` `fputc` (+ `stdin`/`stdout`/`stderr`) |
| `os_shim.c` | `stat` and the file syscall bottom-edge (`open`/`read`/`write`/`lseek`/`close`) over the `fs` cap |
| `mem_shim.c` | `strcmp` `strncmp` `strlen` |
| `libc_shim.c` | `strdup` `strncpy` `strstr` `strtoul` `__ctype_b_loc` |
| `strerror_shim.c` | `strerror` |
| `time_shim.c` / `proc_shim.c` | `localtime` `time` / `__assert_fail` |

**The chibicc-specific remainder (~15) — a small `chibicc_extra.c`:**
- *Trivial* (a few lines each): `bcmp`, `memchr`, `strchr`, `strndup`, `strncasecmp`, `puts`,
  `calloc` (`malloc`+`memset`), `free`, `__errno_location` (or reuse `shim_errno.h`), `exit`.
- *Small but real:*
  - `realloc` — the one allocator decision (§8): personality `OP_REALLOC` **or** a guest
    size-header shim over the on-ramp's bump `malloc`.
  - `open_memstream` — **required**: `strings.c`'s ubiquitous `format()` builds strings through a
    memory `FILE*`. A growable-buffer `FILE` mode over `stdio_shim`'s struct.
  - `strtold` — with `-mlong-double-64` (F3) this is `strtod`; provide/alias it.
  - `dirname` — pure string (relative `#include` base); a dozen lines.
  - `ctime_r` — `__TIMESTAMP__` only; fixed-epoch stub (F4).

**Build shape.** A single `-DTEMEN_GUEST` driver TU (the Postgres pattern) that `#include`s the
reusable shims + `chibicc_extra.c`, linked with the cc1 bitcode; drop `--stub-externs` and let the
step-2a audit assert **zero** undefined symbols remain (every name defined, or a recognized on-ramp
import — `__vm_host_call`/`__vm_cap_resolve`/`malloc`). Bottom edge is the **`fs` cap** (not raw
`temen-posix` name-binding): the on-ramp recognizes `__vm_host_call`, and the `fs` cap already backs a
memfs (`crates/temen-run/src/fs.rs`), so source + `include/*.h` seed straight in (§5 C).

**→ Slice 1 done 2026-07-24 — the libc links and the module is trap-free.**
`chibicc_libc.c` (the aggregator, in-place `#include` of `../postgres/{mem,os,libc,printf}_shim.c` +
`shim_errno.h` + `../strtod/strtod.c`) + `chibicc_extra.c` (the ~15 remainder: a size-header arena
allocator, `FILE*` with an `open_memstream` memory mode composing with `printf_shim`'s `vfprintf`,
`strchr`/`memchr`/`strndup`/`strncasecmp`, `strtold`→`strtod`, `dirname`, fixed-epoch time, and
`__assert_fail`). The build now **translates WITHOUT `--stub-externs`** — every call resolves to the
guest libc or an on-ramp-recognized primitive (`__vm_*`, `bcmp`→memcmp synth, `exit`→Exit powerbox);
**no trap stubs**. The 333-function module decodes / verifies / bytecode-compiles. Decision settled:
the `%.17g` path really is free (`printf_shim` → `__vm_fmt_gen`), and the powerbox is auto-synthesized
for the `main`-having module (so `exit`/stderr lower fine; the arena sidesteps needing `malloc` from
the Memory cap for v1). **Not yet validated at runtime** — that is slice 2.

**Open decision — where the shared guest libc lives.** These shims are now wanted by a *second*
consumer (chibicc), which is exactly the project's stated trigger to dedup (§8). Three options:
(a) **factor** the reusable shims into a shared `demos/_guestlibc/` (or `crates/temen-run/guestlibc/`)
and point both Postgres and chibicc at it — cleanest, but edits Postgres's proven build and must be
re-validated; (b) chibicc's driver **`#include`s** `../postgres/*_shim.c` in place — zero
Postgres-build risk, fast, but couples the demos; (c) **copy** the shims — no coupling, ~1.5k
duplicated lines that will drift. Recommendation: **(b) for the first chibicc bring-up** (reversible,
proves the reuse), then **(a)** once both guests are green (factor with two proven consumers, not
one). Owner call before implementing.

**Validation (slice 2 — done 2026-07-24, the real gate).** `crates/temen-run/examples/chibicc_run.rs`
instantiates `chibicc.temen` on the `fs` cap with a memfs seeded (source `.c` + optional `/include`),
passes argv, runs `main` on the **tree-walker** (the oracle engine), and forwards the guest's
stdout. `run_selfhost_diff.sh` asserts that stdout **byte-matches** a native reference built from the
*same* cc1 TUs + `cc1_main.c` (`chibicc_ref`) — so the only variables are the substrate (guest libc +
Temen interpreter vs system libc + native CPU). Three cases pass byte-for-byte:
- **int** — recursion, arrays, loops: the full tokenize→parse→codegen_ir pipeline + arena allocator.
- **float** — `double`/`float` literals + arithmetic: the codegen emits f64/f32 constants through the
  byte-exact `%.17g` path (`printf_shim` → `__vm_fmt_gen`). *This case caught a real bug — in the
  oracle, not the guest:* the native ref, built `-mlong-double-64`, was calling the system 80-bit
  `strtold` and reading back garbage; the guest (whose `strtold`→`strtod`) was already correct.
  `native_ref_shims.c` gives the reference the same forwarding, making it apples-to-apples.
- **hdr** — `#include <stdbool.h>`/`<stdint.h>`: preprocess reaches the memfs `/include` mount and
  reads real headers. `cc1_main.c` gained an optional leading `-Idir` so the native side can point at
  the real header tree (the guest keeps its `/include` default); it never appears in the emitted IR.

The one normalization: `-g` writes `debug.file 0 "<argv[1]>"`, which echoes the input path (host path
vs `/in.c`); the script rewrites that single quoted path before diffing. Everything else is an exact
match. Slice 1 proved the module *well-formed and complete*; slice 2 proves it *correct* on this
corpus. Next: widen the corpus toward chibicc's own `test/*.c` and wire the differential into CI.
