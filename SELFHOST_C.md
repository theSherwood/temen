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

### B. libc coverage on the POSIX personality + memfs
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
