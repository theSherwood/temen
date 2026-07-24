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

"Self-hosting" here means **the toolchain runs on the platform it targets** — SVM hosts its own
C→IR compiler — *not* that the compiler is bootstrapped by compiling itself. That distinction is
deliberate; see the decision in §3.

## 2. Two roles of chibicc — keep them separate

The word "chibicc" names two different things in this plan:

1. **chibicc-the-frontend** — `frontend/chibicc/codegen_ir.c` + `--emit-ir`: the C→SVM-IR backend
   we maintain (`FRONTEND.md`). This is *code we build with*.
2. **chibicc-the-guest** — the chibicc binary (including that `codegen_ir.c` backend) compiled
   **into an SVM IR module** and run as a guest, so C can be compiled *inside* the sandbox. This is
   the *artifact this doc is about producing*.

When chibicc-the-guest runs, a user hands it C source; it executes its `codegen_ir` backend and
emits SVM IR text — the same output the native frontend produces, just computed inside SVM.

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

### C. Source + includes in the guest filesystem
Seed **memfs** with the user's `.c` and the bundled `frontend/chibicc/include/*.h`, so the
preprocessor resolves `#include`. memfs already supports seeding + imaging.

### D. Close the loop: compiler output → runnable module *(the key open design question)*
`codegen_ir.c` emits **text IR**; a runnable module needs **verify + encode to binary**, and both
(`svm-verify`, `svm-encode`) are trusted host code. Options:
1. **A host "assemble" op** (text IR → a verified module handle) — the on-SVM analog of the browser
   cdylib's `svm_parse`. Keeps the verifier/encoder in the host (they are TCB regardless) and the
   guest thin. **Recommended.** Pairs with `domain_exec` to run the result.
2. **A binary-`.svmb` emitter in `codegen_ir.c`** — chibicc emits the encoded module directly; no
   host assemble op. Heavier (a C encoder to maintain), but fully guest-contained. Safety is
   unchanged — the loader **always re-verifies** (invariant), so a mis-encoded module is a clean
   verify error, never an escape.
3. `domain_exec`/loader accepts text IR — widens the load path with the text parser; least
   preferred.

Whichever: **the verifier runs on the guest-produced module at load** (INVARIANTS.md §9 / §2a).
The compiler-guest is untrusted like any frontend; a compiler bug is a clean error, never an escape.

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
- **Host = mechanism.** `domain_exec` + the assemble op are plumbing; no compiler-specific
  scheduling/policy in the host (INVARIANTS.md §4).
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
4. **Close the loop** (§5 D) — land the recommended assemble op + `domain_exec` run; end-to-end
   compile-and-run inside SVM (§5 E).
5. **Browser deployment** — ship `chibicc.svmb` as a static asset and wire the W5 surface
   (INTERACTIVE_EMBEDDING.md); the encode step reuses the cdylib's existing `svm_parse`.
6. **(optional) stage-2 conformance** differential (§7 E).

## 8. Open questions

- The assemble op's shape (§5 D-1): a one-shot `text IR → module handle`, or a small streaming
  surface? Start one-shot.
- Where the pure-libc bulk lives — compiled into `chibicc.svmb` as guest C (simplest, self-contained
  artifact) vs. shared personality ops (dedup with the shell/Postgres). Lean guest-C for the first
  cut; dedup later if a second consumer wants the same ops.
- `-g` cost in the guest: always emit debug info, or a flag? Always-on pairs with the W1 debugger;
  measure the size hit on `chibicc.svmb`.

## 9. Non-goals

- A general POSIX C toolchain (assembler, linker, `.o`/`ar`). The output is a single SVM IR module;
  `-cc1 --emit-ir` only.
- Matching GCC/Clang language breadth. chibicc's C99 + the frontend's proven coverage is the bar;
  heavy C/C++ stays on the AOT on-ramp lane (`LLVM.md`), which is not a runtime guest.
- Making the chibicc frontend self-compile as the shipping path (§3 — LLVM-built artifact ships).
