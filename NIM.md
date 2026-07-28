# NIM.md — running Nim (nimony) on SVM, and self-hosting it

Status: **scoping / design doc + phase-1 in progress**, written 2026-07-28. This is the
work-breakdown and the load-bearing decisions for targeting SVM from
[nimony](https://github.com/nim-lang/nimony) — the in-development next-generation Nim
compiler. It leans on the on-ramp (`LLVM.md`), the C-selfhost template (`SELFHOST_C.md`),
libc-as-capabilities (`POSIX.md`), and the frontend trust model (`FRONTEND.md` §1,
`DESIGN.md` §2a). This doc is the *what/how/when*; it does not restate those.

Fold completed sections into `DESIGN.md` and drop this file once the actionable gaps
close — the repo convention (cf. the former `WASM.md`/`SCHEDULING.md`).

## 0. TL;DR

- **Goal.** Compile Nim → SVM IR, and eventually **self-host nimony on SVM** — the same
  shape as chibicc-on-SVM (`SELFHOST_C.md`) and the Postgres/QuickJS guest assets.
- **Two phases, cheapest-first.**
  - **Phase 1 (this doc's active work): the C on-ramp path — zero nimony code.**
    nimony already emits C; SVM already ingests C→bitcode→SVM-IR via the proven LLVM
    on-ramp (`LLVM.md`). So `nim → nimony → Leng → lengc(C) → clang -O2 → svm-llvm-translate
    → prep_svmb`. This retires the real risk (does Nim-shaped codegen — ARC, error-flag
    exceptions, raw-pointer objects, the `system` runtime — survive the on-ramp and run
    correctly under confinement?) and delivers **nimony-on-SVM for free**, exactly as
    `chibicc.svmb` came for free from the same pipeline.
  - **Phase 2 (optional, if warranted): a native `Leng → SVM-IR` backend.** A
    `lengc`-style backend written in Nim, consuming Leng NIF and emitting SVM text — the
    supported multi-backend seam (C/C++/LLVM-IR/arkham already coexist behind Leng). The
    **"arkham-for-SVM"** play: drops the clang/LLVM build-time dependency and shapes SVM-IR
    directly from Leng.
- **The one thing that could scare you — Leng assumes flat C-ABI memory with raw
  pointers — is already solved.** It is the C frontend's situation exactly: the guest gets a
  flat `[0, size)` **window**, pointers are window offsets, and the **masking lowering**
  confines every access (INVARIANTS §2 — the security hinge, *not* the verifier). Nim/Leng
  raw pointers are no more dangerous than C's. Leng's `ptr`/`aptr` split is *cleaner* than C.

## 1. Background — nimony's pipeline and where a backend plugs in

(Findings from reading nimony `master`, repo v0.4.0, 2026-07-28. **"NIFC" was renamed
"Leng"**; the tool is `lengc`; the token/format library is "nifcore"; the interchange format
is **NIF** — "Nim Intermediate Format", `nim-lang/nifspec`.)

The compiler is a chain of tools passing **NIF token streams** (not trees) between phases:

```
Nim source
  → nifler      parse → NIF dialect (.p.nif)
  → nimony      sema: symbol/type resolution, template/macro expansion (.s.nif)
                (+ effect inference — NOT yet implemented; + deref/mutation checks)
  → hexer       lowering: iterator inlining, lambda lifting, ARC dup/copy + destructor
                injection, control-flow-expr → stmt, exception translation, CPS
  → hexer       emit  Leng                       ← the stable, documented codegen IR
  → lengc       Leng → C | C++ | LLVM IR         ← the backend seam
```

- **Leng** (`doc/leng-spec.md`, ~450-line grammar) is a **typed, C-like tree/statement IR**
  — *not* SSA, *not* a stack machine. Named locals; structured `if`/`while`/`case` plus
  low-level `lab`/`jmp`. Types: sized `(i N)`/`(u N)`/`(f N)`/`(c N)`, `bool`, `void`,
  `ptr`/`aptr`, value-semantic `array`, `object` (single inheritance = first child is the
  base), `union`, `enum`, bitfields, SIMD `vector`. Ops carry their result type
  (`add`/`sub`/…), `cast` (C cast) vs `conv` (value-preserving). Lvalues: `deref`/`addr`/`at`/
  `pat`/`dot`. Overflow is explicit (`keepovf`/`ovf`, GCC-`__builtin_*_overflow`-shaped).
  `emit` = verbatim C passthrough.
- **The backend seam is real and already multi-consumer.** `src/lengc/` ships C, C++, **and
  LLVM-IR** backends (`codegen.nim`, `llvmcodegen.nim`, dispatched by `lengc {c|cpp|llvm}`);
  a separate **arkham** tool compiles Leng → typed asm-NIF → native (`nifasm`); **shoggoth**
  is a NIF-level optimizer. A new backend consuming Leng is the *supported* extension. This
  is what makes Phase 2 tractable.
- **Runtime shape — favorable:**
  - **ARC/ORC only, no tracing GC.** Destructors/dups are injected by hexer as **ordinary
    Leng calls** → ordinary SVM-IR calls. No GC runtime to port.
  - **libc optional.** Default stdlib is **libc-free** (native allocator + raw-syscall IO);
    `-d:useLibc` opts into mimalloc. Raw syscalls → unresolved named imports the POSIX
    personality resolves at load (`POSIX.md`) — the exact self-host libc model.
  - **Exceptions:** error-flag + `goto` (`errv` bool + `onerr`), **no setjmp/longjmp**, or
    C++ `try/throw` in cpp mode. Fully lowered before Leng.
  - **Concurrency:** CPS / `.passive` procs → state machines over a minimal `system.nim`.
  - **Self-hosting today:** nimony is written in Nim, built by Nim 2.x, and boots through C
    to a **byte-identical stage2==stage3 fixpoint** (`doc/nifcore_migration.md`).

### Compatibility ledger (the "why this works")

| nimony/Leng concern | Lands on SVM as |
|---|---|
| Flat C-ABI memory, raw `ptr`/`aptr`, unions, casts | Window + masking lowering, as for C; §3d pins x86-64-SysV struct layout — matches Leng's ABI |
| ARC/ORC, destructors as calls | Ordinary SVM-IR calls; **no GC runtime** |
| libc-free / raw syscalls / mimalloc | Named imports → POSIX personality; allocator grows the window via the Memory cap |
| Exceptions: error-flag + goto | SVM-IR general goto/branch (C frontend proves gnarly state machines) |
| Bit-reinterpret casts | Already lowered to `copyMem` upstream — not arbitrary bit-punning at Leng |
| CPS/passive → state machines | Ordinary code over minimal `system.nim`; SVM threads (`THREADS.md`) |
| Leng is not SSA | SSA/block-params synthesized from named locals + goto — the on-ramp already does φ→block-args (`LLVM.md`); a native backend redoes this |

### Trust (both phases)

The Nim-derived module is an **untrusted frontend artifact** (`DESIGN.md` §2a): the verifier
re-checks everything at load, so a nimony/backend bug is a **clean error, never an escape**.
No self-hosting convenience may bypass verification (INVARIANTS §9).

## 2. Phase 1 — the C on-ramp path (active)

**Pipeline.** `nim → nimony/hexer → Leng → lengc c → clang-18 -O2 -emit-llvm →
svm-llvm-translate → prep_svmb (decode → verify → bytecode-compile gate)`, then run on
interpreter + JIT. This is the **`build-pg-assets.mjs` / `build_chibicc_svmb.sh` pattern**,
retargeted at nimony's C output. Use the **LLVM on-ramp, not the chibicc C frontend**: Nim's
C leans on compiler builtins (overflow — Leng's `keepovf`/`ovf`) that clang handles and
chibicc does not.

**Build order.**

1. **Retire the codegen-shape risk *before* a nimony bootstrap** — validate that
   **Nim-shaped C** survives the on-ramp and runs identically to native. A small probe
   program exercises the exact patterns nimony/Leng emit: ARC refcount inc/dec + a destructor
   call, an error-flag + goto raise/handler, a tagged `object` with an inheritance-style first
   field, and a heap `seq`/`string`-like struct over `malloc`. Run interp == JIT == native.
   **→ DONE 2026-07-28** — `crates/svm-run/demos/nimony/` (`arc_probe.c` + `build_probe.sh` +
   the `nimony_probe` runner example). The probe translates through the on-ramp to a **21-func,
   9.3 KB `.svmb`** that decodes / verifies / bytecode-compiles, and runs **byte-identical to
   native `clang -O2` on all three engines** (treewalk / bytecode / JIT), exit 0. Every call
   resolved to an on-ramp-recognized name with **no `--stub-externs`** — i.e. ARC destructor
   calls, the error-flag+goto unwind, first-member-base object dispatch, and `realloc`-grown
   seqs all lower and confine cleanly. The codegen-shape risk is retired; the remaining Phase-1
   work is toolchain (step 2) + real libc surface (step 3), not "does Nim's shape fit SVM".
2. **Real Nim codegen on SVM.** **→ PARTIALLY DONE 2026-07-28 via stock Nim as a stand-in.**
   `crates/svm-run/demos/nimony/list_seq.nim` (ARC `ref object` linked list + a `seq`) is
   compiled by the **stock Nim 2.2.10 ARC backend** (`--mm:arc -d:useMalloc -d:noSignalHandler`)
   and on-ramped by `build_nim.sh` — the same `nim → C → clang -O2 → svm-llvm-translate →
   prep_svmb` chain, then run on all three engines. **Result: byte-identical to a native `nim c`
   build (stdout `listSum=385 / seqSum=55`, exit 0) on treewalk / bytecode / JIT.** This is
   genuine Nim-runtime codegen (ARC destructors, heap `ref`, `realloc`-grown `seq`), not the
   hand-modeled `arc_probe.c` — and stock Nim and nimony share the ARC/ORC model + C-ABI shape.
   - **Measured libc surface (the chibicc A.5 stub-audit method): 10 undefined symbols, all
     on-ramp-recognized** — `malloc`/`free`/`realloc`, `fwrite`/`fflush`/`fputc`/`stdout`/`stderr`,
     `exit`, `strlen`. *Far* smaller than chibicc's 41; no libc fill needed for this corpus.
   - **One gotcha found:** Nim installs SIGSEGV/etc. handlers at startup (`signal()` →
     stubbed → `Unreachable` trap); `-d:noSignalHandler` avoids it. Recorded in `build_nim.sh`.
   - **Step 2 proper — genuine nimony output runs on SVM. → DONE 2026-07-28.** Built **Nim 2.3.1
     from source** (stable 2.2.10 can't compile `hastur`), bootstrapped **nimony** (`nim c -r
     src/hastur build all` → `bin/{nimony,hexer,lengc,nifler,…}`), and on-ramped nimony's *own*
     `lengc c` output. `crates/svm-run/demos/nimony/sum_sq_nimony.nim` (an ARC `seq[int]` + a
     `var object`) compiles via the real pipeline (nifler → nimony → hexer → **Leng** → lengc C),
     and `build_nimony.sh` on-ramps that C to a **95-func `.svmb`** that decodes / verifies /
     bytecode-compiles and runs **byte-identical to nimony's native run (`sum_sq=385 / count=10`,
     exit 0) on treewalk / bytecode / JIT.** This is authentic nimony Leng→C→SVM-IR, not a
     stand-in. **Two concrete on-ramp findings** (both normalized in `build_nimony.sh`, both
     recorded as follow-ups for a proper backend):
     - **nimony's runtime allocates via `mmap`, not `malloc`** (libc-free stdlib), plus a handful
       of syscalls (`getpid`/`kill`/`dlopen`/`dlsym`/`_exit`; `write`/`exit` are on-ramp-recognized).
       A 20-line page-aligned bump-allocator shim (`nimony_runtime_shim.c`) over the window covers
       it — the allocator masks addresses to 4096, so `mmap` must return **absolutely** page-aligned
       pointers (the load-bearing subtlety).
     - **TLS gap:** nimony marks the allocator/exception globals `__thread`; the on-ramp has no
       `llvm.threadlocal.address` lowering. For a single-threaded guest these are plain globals
       (stripped in the build). *A real Leng→SVM-IR backend (Phase 2) would map these onto SVM's
       own thread-local/global model instead; the on-ramp TLS gap is worth a `LLVM.md` follow-up.*
3. **On-ramp the real C**, `-mlong-double-64` if nimony emits `long double` (the chibicc F3
   lesson), `--host-page 65536` for a browser-targetable asset. Fill the libc bottom edge by
   **reusing the Postgres/chibicc guest-libc shims** (`SELFHOST_C.md` Appendix B) — the
   surface is nearly identical (stdio, `malloc`/`realloc`, `str*`, `%.17g` via `__vm_fmt_gen`).
4. **Differential-validate** each corpus program: guest stdout/exit byte-matches native
   `nim c` build (the `chibicc_run.rs` / `run_selfhost_diff.sh` two-tier pattern).
5. **nimony-on-SVM (the self-host payoff).** On-ramp nimony's *own* C output into a
   `nimony.svmb` guest — the same way `chibicc.svmb` is built. `nim → nimony.svmb → SVM-IR`
   is then a composition of proven pieces; no new substrate.

**Exit criteria.** (a) the Nim-shaped-C probe runs interp==JIT==native; (b) ≥1 real nimony C
program runs on SVM matching native; (c) the libc fill-list is measured (the `--stub-externs`
+ stub-audit method, `SELFHOST_C.md` A.5).

### Current-sandbox status (2026-07-28)

- ✅ `clang-18` present; ✅ `svm-llvm-translate` built; ✅ **step-1 probe green on all three
  engines** (above).
- ✅ Nim **2.2.10** (choosenim) + **Nim 2.3.1 built from source**; ✅ **nimony bootstrapped**
  (`bin/{nimony,hexer,lengc,…}`); ✅ **genuine nimony Leng→C output runs on SVM, all three
  engines, byte-identical to native nimony** (step 2 above); ✅ stock-Nim ARC program green too.
- The C-on-ramp path (Phase 1) is now **proven end-to-end with the real compiler.** Remaining
  Phase-1 breadth: wider Nim/nimony corpus (strings, exceptions, closures, floats), the `mmap`
  allocator shim → a real Memory-cap allocator, and the TLS follow-up above. Then Phase 2 is the
  open design choice.
- **Reproduce the nimony demo:** build Nim 2.3.1 (`git clone nim-lang/Nim && sh build_all.sh`),
  bootstrap nimony (`nim c -r src/hastur build all`), then
  `NIMONY_BIN=…/nimony/bin/nimony bash crates/svm-run/demos/nimony/build_nimony.sh`.

## 3. Phase 2 — native `Leng → SVM-IR` backend (started)

A translator that consumes **Leng NIF** and emits **SVM IR** directly, bypassing C/clang. It
drops the build-time clang/LLVM dependency and shapes SVM-IR straight from Leng, and it is the
supported extension pattern — C/C++/LLVM-IR/arkham already coexist behind Leng, plus the
shoggoth optimizer.

**Placement decision (2026-07-28): a Rust crate `crates/svm-leng` in *this* repo** — the
**fourth SVM frontend**, beside `svm-wasm` and `svm-llvm` (both Rust, both untrusted, both
verifier-rechecked). Rationale: it matches the established frontend pattern, reuses
`svm-ir`/`svm-text`/`svm-verify` directly, and is **CI-testable with checked-in Leng fixtures,
no nimony toolchain at build time**. (A Nim backend inside nimony's `src/lengc/` — the arkham
analog — is the alternative; it is better for *eventual* pure self-hosting through Leng but
couples to the nimony build and can't live in this repo. Revisit once the Rust translator has
proven the mapping. The two aren't exclusive.) Like every frontend it is **outside the
escape-TCB** (DESIGN.md §2a): the verifier re-checks its output, so a bug is a clean error.

### Walking skeleton — DONE 2026-07-28

`crates/svm-leng` translates the **integer / arithmetic / local / direct-call** subset with
straight-line bodies and `ret`, and **fail-closes (`LengError::Unsupported`) on everything
else** (the `svm-wasm`/`svm-llvm` `unsup(...)` discipline — never a silent mistranslation). It
emits SVM text (chibicc's `codegen_ir.c` model) via `svm_text::parse_module`. Six end-to-end
tests translate hand-written Leng-NIF (faithful to `doc/leng-spec.md`) → verify → **run on both
the interpreter and the JIT with identical results** (§9 parity): constant arithmetic
(`3 + 4*2`), params+locals, `div`/`mod`, i32↔i64 `conv`, cross-proc `call`, and the fail-closed
float case. `src/nif.rs` is a real NIF reader (parens, atoms, `:symdefs`, string literals,
`@lineinfo` stripping, `.nif`/`.indexat` directives) so it grows toward *real* nimony Leng, not
just fixtures.

This proves the seam. The remaining work is adding grammar arms below (each a new match case),
not rearchitecting.

**The work (bounded — comparable to chibicc's `codegen_ir.c`, which exists and is proven):**

- **✅ integer scalars, arithmetic (`add`/`sub`/`mul`/`div`/`mod`), `neg`, width `conv`,
  locals (`var`/`asgn`), direct `call`, `ret`** — the landed skeleton.
- **Next: control flow** — `if`/`ite`/`while`/`case` + `lab`/`jmp` → SVM blocks/`br_table`
  (irreducible CFG is native, `DESIGN.md` §3). This needs real block/SSA-param synthesis from
  Leng's named locals across joins (the skeleton is single-block); the on-ramp's φ→block-args
  is the reference, but from a tree IR it's closer to `codegen_ir.c`'s data-SP threading.

- **Then, further out:**
  - **C-ABI struct/union/enum layout** → SVM §3d (x86-64-SysV already pinned — Leng assumes the
    same ABI, so this is a match, not a negotiation).
  - **Memory:** Leng `ptr`/`aptr`/`at`/`pat`/`dot`/`deref`/`addr` → window loads/stores +
    `ptr.add`; every access confined by the masking lowering (INVARIANTS §2). Address-taken
    locals move from SSA values onto a data-stack frame (the `codegen_ir.c` model).
  - **Calls + ARC:** indirect calls; destructor/dup calls pass through as ordinary calls;
    `onerr`/`errv` → branch-on-flag.
  - **Overflow:** `keepovf`/`ovf` → SVM's trapping/checked arithmetic.
  - **Runtime bottom edge:** raw syscalls / allocator → POSIX personality named imports +
    Memory cap, same as Phase 1 — and mapping nimony's TLS onto SVM's own model (the on-ramp
    gap found in Phase 1).

**Risks:** nimony is v0.4.0, "heavy development" — the Leng grammar and C output are moving
targets (Phase 1 is insulated: it only needs "nimony emits compilable C"; Phase 2 couples to
the grammar). Effect inference (pipeline phase 3) and parts of CPS are not yet implemented
*in nimony itself* — a limit of the source compiler, not the backend.

## 4. Invariants this must respect

- **Untrusted frontend, zero escape-TCB.** Same class as chibicc/`svm-wasm`/`svm-llvm`: the
  verifier re-checks the produced module; a bug is a clean error (INVARIANTS §9, §2a).
- **No new substrate.** Both phases close over existing seams — the on-ramp, the POSIX
  personality, the Memory cap, `prep_svmb`, the `fs` cap memfs. No new host ops. Host stays
  mechanism (INVARIANTS §1/§4).
- **Confinement is the masking lowering.** Nim raw pointers ride the same window+mask regime
  as C; no new emitted-code/window-access surface (INVARIANTS §2).
- **Code-coupled asset.** `nimony.svmb` (and any corpus `.svmb`) regenerate on IR/ABI/encoder
  change, gated in CI — the Postgres/chibicc asset-lane template.

## 5. Non-goals

- Matching every Nim 2 feature — nimony's own coverage at v0.4.0 is the ceiling; effect
  inference etc. are upstream gaps.
- A general Nim package/build tool on SVM (nimble, etc.). The unit is a compiled SVM module.
- Making the Phase-2 backend the *only* path — the on-ramp (Phase 1) stays the low-risk lane
  and the self-host shipping path, exactly as the LLVM-built `chibicc.svmb` ships (§3 there).

---

## Appendix — recorded toolchain commands (Phase 1 step 2, for when a nim toolchain is present)

```sh
# Nim 2.x (apt's 1.6 is too old for nimony)
curl -fsSL https://nim-lang.org/choosenim/init.sh | sh    # or: choosenim 2.2.0
export PATH="$HOME/.nimble/bin:$PATH"

# nimony
git clone https://github.com/nim-lang/nimony && cd nimony
nim c -r src/hastur build all          # bootstrap the toolchain
# emit Leng-derived C for a program (via the C backend):
#   the toolchain drives nifler → nimony → hexer → lengc c; capture the generated .c

# on-ramp the C (mirrors build_chibicc_svmb.sh)
clang-18 -O2 -emit-llvm -c -mlong-double-64 prog.c -o prog.bc
svm-llvm-translate prog.bc -o prog_raw.svmb --binary --host-page 65536 [--stub-externs]
cargo run --release -p svm-run --example prep_svmb -- prog_raw.svmb prog.svmb
```
