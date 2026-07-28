# SELFHOST_C.md — a C compiler that runs *on* SVM and compiles to SVM IR

Status: **scoping / design doc**, written 2026-07-24. No code yet — this is the work-breakdown
and the load-bearing decisions. The *why* for the pieces it leans on lives in `FRONTEND.md`
(the chibicc frontend), `LLVM.md` (the AOT on-ramp), `POSIX.md` (libc-as-capabilities),
`EXEC.md`/`PROCESS.md` (`domain_exec`, module instantiation), and `svm-fs` (memfs). This doc is
the *what/how/when*; it does not restate those.

## 1. Goal

**A C compiler that runs as an ordinary SVM guest and compiles C source → SVM IR from inside the
sandbox.** The browser playground (INTERACTIVE_EMBEDDING.md W5) is one deployment of this, not the
point — the same guest artifact runs under `svm-run` natively, in a `domain_exec` child, or in the
wasm build.

**The capstone is a playground demo**: in the browser, pick or edit any of several real C
programs, compile them **in-browser** (`chibicc.svmb` running on the bytecode engine), run the
result, and see its output — the full edit → compile → run loop, client-side, no server (§7 step 5).

"Self-hosting" here means **the toolchain runs on the platform it targets** — SVM hosts its own
C→IR compiler — *not* that the compiler is bootstrapped by compiling itself. That distinction is
deliberate; see the decision in §3.

## 2. One chibicc, two build forms

There is **one chibicc**: one source tree (`frontend/chibicc/`, including the `codegen_ir.c`
C→SVM-IR backend), no fork, nothing maintained twice. It exists in two *build forms*:

1. **The native binary** (built by `cc`, as today) — a build/dev tool: the test harness
   (`c_frontend.rs`) shells out to it, and it is the **byte-match oracle** the guest form is
   differentially validated against (§5 E).
2. **`chibicc.svmb`** — the *same source* compiled through the LLVM on-ramp into an SVM IR module
   (§3). This is the artifact this doc is about: run it as a guest and C compiles *inside* the
   sandbox, producing the same IR the native binary would.

Nothing about the plan requires divergent code paths between the two — same `main.c`, same
`codegen_ir.c`; the native binary sticks around only because the dev/test loop uses it.

## 3. Load-bearing decision — build chibicc-the-guest with the **LLVM on-ramp**, and keep it

The guest artifact (`chibicc.svmb`) is produced by compiling chibicc's C through the **AOT LLVM
on-ramp** (`clang -O2` → LLVM bitcode → `svm-llvm-translate` → `prep_svmb`), **not** by the chibicc
frontend compiling itself.

Rationale (owner, 2026-07-24): the on-ramp inherits **LLVM's full optimization pipeline**, so the
resulting IR is substantially faster than what `chibicc-frontend + svm-opt` would emit — our
frontend does minimal optimization and `svm-opt` is younger than LLVM's `-O2`. Since the compiler
binary is large and runs often, we spend LLVM's optimization budget **once, at build time**, and
**ship the optimized artifact**. A true stage-2 self-compile (chibicc-frontend compiling chibicc)
is kept only as an *optional conformance differential* (§7 E), never the shipping path.

This is the **exact pattern already proven for Postgres** (`browser/build-pg-assets.mjs`:
`svm-llvm-translate` + `prep_svmb` → a committed `*.svmb` asset, regenerated when SVM code changes).
Confidence is high: the on-ramp already runs **QuickJS and Postgres byte-identical to native**
(`LLVM.md`); chibicc is far simpler C. The on-ramp shells out to `clang`/`llvm-dis` at build time —
a dev/CI lane, off the runtime path — which is fine for producing a shipped artifact (same as the
Postgres asset), including for the browser (the `.svmb` is a static asset).

## 4. What already exists (substrate — do not rebuild)

| Piece | State | Where |
|---|---|---|
| C→SVM-IR backend (`--emit-ir`, `-g`, `--child-entry`) | Built; runs real libraries byte-identical to `cc` | `frontend/chibicc/codegen_ir.c`, `FRONTEND.md` |
| LLVM on-ramp: real C/C++ → SVM IR, optimized | Built; QuickJS + Postgres **run** byte-identical | `svm-llvm`, `svm-llvm-translate`, `LLVM.md` |
| Committed `.svmb` build-asset flow (on-ramp → `prep_svmb`) | Built | `browser/build-pg-assets.mjs`, `crates/svm-run/demos/postgres/` |
| libc-as-capabilities: unresolved named imports → personality ops at load | Built (core surface, ops 0–20: stdio, `malloc`/`free`, `open`/`read`/`write`, dirs, `getenv`, `exec`, `dlopen`) | `svm-posix`, `POSIX.md` |
| In-memory filesystem (seed/image, dirs, fd table, crash-consistency) | Built | `svm-fs` |
| Guest spawns/runs a child module | Built (`domain_exec`, 2026-07-23) | `svm-run/src/exec.rs`, `EXEC.md` |
| Verifier re-checks every loaded module (frontend/guest is untrusted) | Built; the invariant | `svm-verify`, INVARIANTS.md §9, `DESIGN.md` §2a |

The compiler-guest slots into this like the shell (`STAGE1.md`) and Postgres do: **its libc calls
are unresolved named imports the POSIX personality resolves at load** (`POSIX.md` step 1 — "chibicc
/ `svm-llvm` emit these"). Nothing about the compiler is special to the substrate; it is just
another real C program whose output happens to be SVM IR.

## 5. What's required

### A. Produce `chibicc.svmb` via the on-ramp
`clang -O2 -emit-llvm` over the chibicc sources (compiler in `-cc1 --emit-ir` mode only — no driver
`exec`/`glob`/temp-file paths) → `svm-llvm-translate` → `prep_svmb`, mirroring `build-pg-assets.mjs`.
Emit with `-g` so a compile running under the W1 debugger has source info. **Risk: low** (on-ramp
handles Postgres/QuickJS).

### B. libc coverage — **reuse the proven Postgres guest-libc shims** (see Appendix B)

Step-2 investigation found this is *not* a write-from-scratch job: `crates/svm-run/demos/postgres/`
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
`codegen_ir.c` emits **text IR**. Closing the loop needs *zero new SVM ops*; the existing seams
cover every deployment shape, layered by how far "inside" the loop runs:

1. **v1 — the embedder closes the loop** (exactly as it does for the native compiler today).
   The compiler-guest reads `.c` from memfs and writes its IR output back to memfs; the *embedder*
   picks it up, assembles/verifies/loads, and runs it — `svm-run` natively, the cdylib's existing
   `svm_parse` in the browser. The compiler is just a program whose output is a file. Nothing new
   anywhere.
2. **In-guest compile-and-run — already built: the §22 `Jit` capability + `vm_dlopen`.** A guest
   can hand serialized SVM IR from its own window to the host, which runs the fail-closed
   **rewrite-then-verify** gate (`jit_resolve_and_validate`: `decode_module` →
   `resolve_imports` → `verify_module` → install) and returns callable `call_indirect` slots.
   The C-level loader (`<vm_dl.h>`: `vm_dlopen`/`vm_dlsym`/`vm_dlclose`) is **built and
   differentially tested** (DESIGN.md §22, "In-window dynamic linking — SETTLED"). So
   *compile → load → call, entirely in-guest,* is a composition of existing pieces. The one
   guest-side gap: `vm_dlopen` takes **binary** serialized IR (`decode_module`), and chibicc emits
   text — so this layer needs a small **binary emitter in `codegen_ir.c`** (guest code, the wire
   format is the simple single-pass `svm-encode` form). No host change.
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
- **End-to-end:** compile a C program *inside* SVM, `domain_exec` the result, and match the native
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
- **One artifact, rebuilt on SVM-code change.** `chibicc.svmb` is code-coupled (like
  `postgres_resolved.svmb`): regenerate it when the IR/ABI/encoder changes, and gate that rebuild
  in CI so a drift breaks the PR that caused it (the Postgres asset lane is the template).

## 7. Build order

1. **libc-coverage audit** — chibicc's footprint (§5 B) vs the POSIX personality; list the missing
   `FILE*`/formatting ops. Cheap, decides the size of B.
2. **Build `chibicc.svmb`** via the on-ramp (§5 A), a `build-chibicc-asset` script mirroring
   `build-pg-assets.mjs`; run it (no libc yet) to shake out translation.
   **→ Done 2026-07-24** — `crates/svm-run/demos/chibicc_selfhost/` (`build_chibicc_svmb.sh` +
   `cc1_main.c`, the driver-free guest entry). The full cc1 TU set translates through the on-ramp
   on the first pass: **258 functions, ~287 KB `.svmb`, verifies, bytecode-compiles** (decode 6 ms /
   verify 1.2 ms / bc-compile 4.3 ms via `prep_svmb`). Translated with `--stub-externs`
   (step-3 libc pending) + `--host-page 65536` (browser target) + `-mlong-double-64` (F3).
   Ground truth of what must be filled is now measured, not scanned: Appendix A.5.
3. **Fill libc** (§5 B) until chibicc-the-guest compiles a trivial C file to text IR against
   memfs-seeded source (§5 C), matching native `--emit-ir`.
4. **Close the loop** (§5 D) — v1: embedder assembles/loads the output (no new code beyond glue);
   then the in-guest layer: a binary emitter in `codegen_ir.c` feeding the existing `vm_dlopen`;
   end-to-end compile-and-run inside SVM (§5 E).
5. **Browser deployment — the capstone playground demo.** Ship `chibicc.svmb` as a playground
   asset and wire the W5 surface (INTERACTIVE_EMBEDDING.md): a demo where the user picks or edits
   **various real C programs** (seed with the proven demo corpus — hello, calc, sha256, sortvec,
   tiny-regex, raytrace, …), compiles them **in the browser** (`chibicc.svmb` on the bytecode
   engine, source + `include/*.h` seeded into memfs), runs the compiled module, and sees its
   output in the pane. The encode step reuses the cdylib's existing `svm_parse`. Gated by a
   Playwright test in the `real-browser` CI job (the `browser-play-editor-test.mjs` pattern):
   compile ≥2 corpus programs in Chromium, run them, assert output matches the native build's.

   **→ Step-5 slice A done 2026-07-24 — engine parity settled (the prerequisite).** `chibicc_run.rs`
   takes `SVM_CHIBICC_BACKEND`; `run_selfhost_diff.sh` runs every case on **treewalk / bytecode / jit**
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
   (chibicc → SVM)" card (`browser/web/play.js`, `kind: 'chibicc'`): edit C → the page runs
   `chibicc.svmb` on the bytecode engine, seeding the source on an `fs` cap at `/in.c`
   (new cdylib export `svm_run_onramp_fs`, which generalizes the Postgres `pg_setup` memfs+argv path)
   → SVM-IR text → `svm_parse` → run → `main()`'s return value, all client-side. Built + verified:
   - **Local capstone loop** (`run_selfhost_diff.sh`): in-SVM compile-and-run of the return-value corpus
     (`corpus/{sum,sort,hash}.c`) matches native clang on every engine.
   - **Full browser path proven in Rust** (compile via `onramp_fs_exec` → `parse_module` → run via
     `onramp_exec`) and **in Chromium** — the Playwright gate (`browser-play-editor-test.mjs`) asserts a
     C program compiles-and-runs in-browser to an exact value with SVM IR in the output pane.
   - `build-onramp-assets.mjs` stages `chibicc.svmb`; the `real-browser` CI job builds it so the test
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
   closure; (b) the playground passes `-g0` (new guest-driver flag), dropping the `-g` debug info that
   the compiled program never uses (~a third of the IR). The remaining lever was getting chibicc off the
   bytecode interpreter and onto the wasm-JIT — done next.

   **→ wasm-JIT tier DONE 2026-07-28 — chibicc compiles on emitted wasm.** The card's compile pass (the
   slow half) now takes the **"wasm-JIT" toggle** (default on): chibicc's whole `_start` emits to wasm
   via `compile_module_reactor(&m, /*entry*/ 0, …)` — **333/402 functions run on emitted wasm**, and the
   ~69 reachable non-subset helpers (the `cap.call`/`call.import` wrappers `outline_cap_calls` hoists, all
   integer-signature) **bounce cross-tier to the interpreter** over the shared window through
   `env.call_interp`, so `fopen`/`read`/`write`/`exit` resolve against the powerbox and the seeded memfs.
   This settles the old "chibicc uses floats → can't JIT" worry for good: `_start` **is** in-subset (floats
   emit; the only refused scalar-float op, `fma`, isn't on chibicc's reachable path), so nothing is
   integer-only about it. Built on the **pre-existing single-shot JIT runner** (`JitOnrampRun`, the
   run-to-completion twin of the Doom `JitOnrampReactor`, added for Lua/SQLite): this slice added an
   **fs+argv opener** (`open_owned_run_fs`/`open_shared_run_fs` + the `svm_onramp_jit_run_open_fs` FFI
   export) that grants the same headless memfs powerbox the bytecode `onramp_fs_exec` does (shared
   `chibicc_card_image` + `CHIBICC_CARD_ARGV`) and seeds argv at `POWERBOX_ARGS_BASE`, plus the
   `runJitCompiler` JS driver (a sibling of `runJitModule`). **No substrate / TCB change** — the emitter,
   the interpreter, and the powerbox are all untouched. Gated by:
   - `browser/tests/chibicc_jit.rs` — a native `wasmi` differential (the `jit_module.rs` pattern): the
     JIT-emitted IR is **byte-identical** to the interpreter oracle (`onramp_fs_exec`), and the emitted
     program then parses + runs to its expected stdout.
   - `browser-play-editor-test.mjs` — Chromium asserts the card compiles-and-runs **on the wasm-JIT**
     in-browser (the `.state` message reports `(wasm-JIT)`, so a silent interpreter fallback fails), and
     the card's **"Prove interp ≡ JIT"** button (`proveChibiccParity`) shows byte-identical emitted IR
     across both tiers live in the page.
   A fallback to `svm_run_onramp_fs` (bytecode) remains if the emit is ever unavailable, so the card can
   never regress to "won't run". Residual: chibicc's compiled *outputs* still run on whichever tier their
   own card selects; shortest-round-trip floats and larger libc surface stay open (above).
6. **(optional) stage-2 conformance** differential (§5 E).

## 8. Open questions

- Binary emission (§5 D-2): a C encoder for the `svm-encode` wire form in `codegen_ir.c`, or keep
  the guest text-only and let the embedder assemble everywhere? Text-only defers the encoder but
  caps the in-guest story at layer 1; the encoder is small (the format is a deliberate single-pass
  design) and unlocks `vm_dlopen`. Decide when layer 2 has a consumer.
- ~~Where the pure-libc bulk lives~~ — **settled by the step-1 audit (Appendix A): guest C**, one
  small self-host libc translation unit compiled alongside chibicc at the `clang` step. The one
  remaining allocator sub-question: `realloc` needs the old block size — either a personality
  `OP_REALLOC` (host allocator owns block metadata, matching the POSIX.md split; personality
  growth, not substrate) or a guest size-header shim over the existing `malloc` op. Decide at
  implementation; both are small.
- `-g` cost in the guest: always emit debug info, or a flag? Always-on pairs with the W1 debugger;
  measure the size hit on `chibicc.svmb`.

## 9. Non-goals

- A general POSIX C toolchain (assembler, linker, `.o`/`ar`). The output is a single SVM IR module;
  `-cc1 --emit-ir` only.
- Matching GCC/Clang language breadth. chibicc's C99 + the frontend's proven coverage is the bar;
  heavy C/C++ stays on the AOT on-ramp lane (`LLVM.md`), which is not a runtime guest.
- Making the chibicc frontend self-compile as the shipping path (§3 — LLVM-built artifact ships).

---

## Appendix A — step-1 libc-coverage audit (done 2026-07-24)

**Method.** Verified extern scan of the `-cc1 --emit-ir` compilation set — `tokenize.c`,
`preprocess.c`, `parse.c`, `type.c`, `codegen_ir.c`, `strings.c`, `hashmap.c`, `unicode.c`, plus
`main.c`'s cc1 slice — diffed against the personality's op surface (`svm-posix` ops 0–20 and its
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
fail-closed** — `svm_posix::resolve()` returns `None` for unknown names and every named import
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
printf-with-`%.17g`), a `calloc` wrapper, one `realloc` decision, and a cc1-only build set. No SVM
substrate change anywhere; at most one *personality* op (`OP_REALLOC`).

### A.5 Step-2 ground truth — the measured stub list (supersedes the source scan where they differ)

The step-2 build (`build_chibicc_svmb.sh`) links the real cc1 bitcode and reports every undefined
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

**Build shape.** A single `-DSVM_GUEST` driver TU (the Postgres pattern) that `#include`s the
reusable shims + `chibicc_extra.c`, linked with the cc1 bitcode; drop `--stub-externs` and let the
step-2a audit assert **zero** undefined symbols remain (every name defined, or a recognized on-ramp
import — `__vm_host_call`/`__vm_cap_resolve`/`malloc`). Bottom edge is the **`fs` cap** (not raw
`svm-posix` name-binding): the on-ramp recognizes `__vm_host_call`, and the `fs` cap already backs a
memfs (`crates/svm-run/src/fs.rs`), so source + `include/*.h` seed straight in (§5 C).

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
(a) **factor** the reusable shims into a shared `demos/_guestlibc/` (or `crates/svm-run/guestlibc/`)
and point both Postgres and chibicc at it — cleanest, but edits Postgres's proven build and must be
re-validated; (b) chibicc's driver **`#include`s** `../postgres/*_shim.c` in place — zero
Postgres-build risk, fast, but couples the demos; (c) **copy** the shims — no coupling, ~1.5k
duplicated lines that will drift. Recommendation: **(b) for the first chibicc bring-up** (reversible,
proves the reuse), then **(a)** once both guests are green (factor with two proven consumers, not
one). Owner call before implementing.

**Validation (slice 2 — done 2026-07-24, the real gate).** `crates/svm-run/examples/chibicc_run.rs`
instantiates `chibicc.svmb` on the `fs` cap with a memfs seeded (source `.c` + optional `/include`),
passes argv, runs `main` on the **tree-walker** (the oracle engine), and forwards the guest's
stdout. `run_selfhost_diff.sh` asserts that stdout **byte-matches** a native reference built from the
*same* cc1 TUs + `cc1_main.c` (`chibicc_ref`) — so the only variables are the substrate (guest libc +
SVM interpreter vs system libc + native CPU). Three cases pass byte-for-byte:
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
