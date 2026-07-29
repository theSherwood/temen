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

> **Capstone reached 2026-07-28: whole real modules verify.** `svm-leng` translates **entire**
> real `hexer` modules — every proc plus globals, type decls, and cross-module imports — for three
> real Nim programs (`addTwo`+`main`, `maxi`+`sumto`, `dot2`+`idx`), each **parsing and passing
> `svm-verify`**, and the user `main` (an intra-module call) **runs end-to-end on both engines**
> (`crates/svm-leng/tests/whole_real_module.rs`). Driving a whole module out turned the "what's
> left" list into a measured one — the last gaps that blocked real modules were small: `(true)`/
> `(false)`/`(nil)` literals, `cast`, and coercing a bare-literal `ret` to the proc's result type
> (an i32 `main` returning `0`). The remaining breadth (below) is genuinely optional for coverage,
> not structural.

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

### Real nimony output — DONE 2026-07-28 (go deep)

The skeleton now consumes a **real Leng file emitted by nimony's own `hexer`**, not just
hand-written fixtures. `hexer c --isMain <mod>.s.nif` produces Leng for a module; the verbatim
output for `proc addTwo(a,b: int): int = result = a + b` is checked in at
`tests/fixtures/real_module.leng.nif`, and `translate_proc(real, "addTwo.0.")` translates that
proc out of the full module (which also carries `gvar`/`type`/`main`/`ini` constructs still
outside the subset) and **runs it on both engines** (`addTwo(20,22)=42`, etc.). This drove two
reader corrections against real bytes:
- **Line-info is pervasive and multi-form.** NIF attaches position info to *every* token — tags,
  symbols, *and* numbers — introduced by `@` (`add@4`, `20@7`) **or** `~` (`a.0~2`, `(i~6,~4 64)`).
  The reader strips from the first `@`/`~`; neither can occur in a semantic token (mangled
  symbols encode them away; integer literals use `-`, not `~`).
- **`(stmts …)` may omit the SCOPE marker.** The grammar is `(stmts SCOPE Stmt*)`, but real hexer
  output starts straight with a statement (always a list); the reader skips a *leading atom*
  scope but keeps a leading list.

To go from one proc to the whole module (`main`, `ini`, `` `main ``) needs the broadening arms:
`gvar`/`type`/`const` top-levels, `if`/`while`, cross-module `call` (`…sysvq0asl` suffixes) as
imports, `cast`/pointers. Deep-then-broaden: the real seam works; each construct is now additive.

**The work (bounded — comparable to chibicc's `codegen_ir.c`, which exists and is proven):**

- **✅ integer scalars, arithmetic (`add`/`sub`/`mul`/`div`/`mod`), `neg`, width `conv`,
  locals (`var`/`asgn`), direct `call`, `ret`** — the landed skeleton.
- **✅ control flow — DONE 2026-07-28.** `if`/`elif`/`else`, `while`, `scope`, nested `stmts`,
  and comparisons (`eq`/`neq`/`lt`/`le`), lowered to **multi-block SVM-IR with locals threaded as
  block parameters** (the chibicc/on-ramp φ model — no separate dominance analysis; a merge is
  just the successor's block param). Value numbers reset per block; the entry block carries only
  the function params (the ABI), successors carry every slot. Tested on hand fixtures (max, a
  `while` sum, an `elif` sign chain) **and the real nimony `maxi` if/else** — interp == JIT on all.
  `case`→`br_table` and `lab`/`jmp` (Leng's low-level jump family) remain.
- **Then, further out:**
  - **C-ABI struct/union/enum layout** → SVM §3d (x86-64-SysV already pinned — Leng assumes the
    same ABI, so this is a match, not a negotiation).
  - **Memory:** Leng `ptr`/`aptr`/`at`/`pat`/`dot`/`deref`/`addr` → window loads/stores +
    `ptr.add`; every access confined by the masking lowering (INVARIANTS §2).
    - **✅ pointer params + `deref`/`store` — DONE 2026-07-28.** Pointer-typed params/vars are
      `i64` window offsets; `(deref p)` loads and `(asgn (deref p) v)`/`(store v p)` store, at the
      pointee width tracked per pointer local. The module declares a `memory` window only when a
      load/store is actually emitted. Tested store→load round-trips on both engines. (No frame
      yet — the pointer is supplied by the caller as an offset.)
    - **✅ address-of-local + the data-stack frame — DONE 2026-07-28.** `(addr x)` demotes a local
      from an SSA slot to a byte offset in a per-call window frame; the proc gains a leading `$sp`
      stack-pointer param (slot 0), reads/writes the local via `load`/`store` at `sp+off`, and a
      call to a frame-needing proc passes `sp + frame_size` as the callee's frame. SSA and frame
      locals coexist (only address-taken ones are framed). Tested with the real nimony loop shape
      `inc(addr i)` (a frameless pointer helper called from a frame-needing counter) and a mixed
      SSA-accumulator/framed-counter sum, interp == JIT. Address-taken *params* and recursion
      depth beyond one frame are the remaining refinements.
    - **✅ `at`/`dot`/`pat` + type layouts — DONE 2026-07-28.** Named `(type … (object …))` /
      `(array Elem Count)` layouts are registered (with forward-ref resolution); a unified
      `lvalue_addr` (the `codegen_ir.c` `gen_addr`) resolves `dot` (field), `at` (array element),
      `pat` (pointer index), `deref`, and frame/aggregate symbols to `(address, type descriptor)`,
      then a scalar leaf loads/stores. Object params are passed by address; aggregate `var`s are
      frame-resident (default-zeroed). Tested on hand fixtures (object field set/get, array `at`,
      pointer `pat`, a framed local array) **and real nimony object bytes** (`dot2`, `p.x*p.x+…`),
      interp == JIT. Whole-aggregate copy/`oconstr`/`aconstr` and C-ABI (SysV) field offsets remain.
    - **✅ whole-module: globals + multi-proc — DONE 2026-07-28.** `gvar`/`tvar` module globals live
      at fixed window offsets (below the caller-passed stack) and are shared across calls; scalar
      `const`s inline; `gvar`/`const`/`type` top-levels are accepted, and a module's procs are
      emitted together so **intra-module calls** resolve by index. Tested end-to-end: a global
      counter + `const` step + a `main → bumpN → bump` call chain, interp == JIT. Non-zero global
      initializers fail-close (a `data`-segment init is the refinement).
    - **✅ cross-module `call` → SVM imports — DONE 2026-07-28.** A call to a callee not defined in
      the module becomes a declared `import N "name" (params) -> (ret)` + `call.import N`; the
      signature is fixed from the call site (param types from the args; return arity from position —
      a stmt-call is void, an expr-call returns a value), cached per symbol (inconsistent arity
      fail-closes). The runtime binds the import by name at instantiation, exactly like `write`.
      Tested: a cross-module call translates + verifies, **runs correctly on the interpreter with a
      bound host fn** (`use_ext(x)=ext_double(x)+1`), a stmt-call declares a void import, and — the
      payoff — **real nimony `sumto` now translates and verifies**: `while i<=n: (inc(addr i);
      result+=i)` composes the frame (address-taken counter), `while`, and the cross-module `inc`
      import all at once. Signature inference is call-site-based (not the `.idx` export map); wiring
      the real export sigs (and JIT-side import binding in tests) are refinements.

  **State:** `svm-leng` translates whole real-ish modules — integers, floats, control flow (incl.
  `break`/`continue` and `block` via `jmp`/`lab`), pointers, frames, objects/arrays (incl.
  constructors, copy, and **sret return**), **object-of-`RootObj` inheritance** (base-inlining +
  vtable header), enum/distinct scalars, **exceptions** (nimony's error-flag ABI), **seq/string**
  value layout + operations (as runtime imports), globals, intra- and cross-module calls —
  fail-closed on the rest, and is validated against genuine `hexer` bytes (`addTwo`, `maxi`, `dot2`,
  `sumto`, `classify`, `favg`, `mkSum`, `mk`, `firstHit`, `labeled`, `toNum`, `mayFail`, `guarded`,
  `counter`, `getAt`, `sumSeq`, `makeSeq`, `kindOf`, `mkDerived`). **W1 (Leng totality) is
  essentially closed** — what's left is genuinely runtime, not translation: dynamic method dispatch
  and value-object exception payloads fail-close cleanly (both need the vtable/`exc`-threadvar
  runtime), and the `jtrue`/`mflag`/`vflag` cfvar forms never reach us (hexer's `xelim` lowers them
  away before the final IR). The remaining lever is **W3** — binding the seq/string (and other
  stdlib) imports to a real runtime so the lowered code *runs*, not just verifies.
    - **✅ whole-aggregate copy + `oconstr`/`aconstr` — DONE 2026-07-28.** An aggregate destination
      (frame var, `deref`/`dot`/`at`, global) is dispatched by a non-emitting `lvalue_type` walk:
      `(oconstr T (kv F E)*)` and `(aconstr T E*)` construct field/element-by-element in place (with
      nested aggregates recursing), and any other rhs is a whole-aggregate `mem.copy` of the
      source's bytes. Aggregate `var`s initialize the same way. Tested: object construct-and-read,
      an array `aconstr`, a struct copy (`mem.copy`), and **real nimony `mkSum`** (`var p = Pt(x:a,
      y:b); p.x+p.y`), interp == JIT.
    - **✅ object-of-`RootObj` inheritance — DONE 2026-07-29.** An inheritable object carries a
      leading vtable/type-header pointer (the positional slot an `(oconstr T <vtable> …)` fills), then
      the base's fields, then its own — `resolve_type` inlines a local base's layout at the front, and
      an external inheritable root (`RootObj`) contributes a single 8-byte header. A `Type.vt` (`Rtti`)
      const gets a zeroed, addressable placeholder global so `(addr Type.vt)` resolves; the stored
      vtable pointer is opaque (only *dynamic dispatch* reads through it, and that fail-closes).
      Tested on hand fixtures (construct-and-read-back a `Derived` — base field before derived field,
      both past the header — and a base-field read through a pointer, both engines) and **real nimony
      `kindOf`** (reads `e.value` through a `ptr BaseError`, *runs*) + **`mkDerived`** (constructs the
      inherited object with its vtable — translates + verifies; running needs the ARC destructor
      imports, W3). Value-object exception payloads (an object punned into the error tuple's scalar
      `ErrorCode` slot) stay fail-closed.
    - **✅ seq/string (value layout + operations as imports) — DONE 2026-07-29.** nimony's `seq[T]`
      is a `{len, data*}` fat-pointer **object** (`string` analogous), so its value layout and element
      access already ride the object + pointer machinery — a hand-written seq summed over a
      caller-provided buffer *runs* on both engines. Its *operations* (`add`/`[]`/`len`/`toOpenArray`/
      `newSeq`) are stdlib procs that lower to **imports** (the **W3** runtime edge: they verify, and
      run once bound). Getting real seq bytes to lower needed four fixes: (1) import **names** escaped
      for svm-text (the `[]` operator mangles to `\5B\5D…`, whose bare backslash the lexer rejected);
      (2) aggregate **args** to imports passed by address; (3) aggregate-**returning** imports (sret
      imports, e.g. `toOpenArray`/`newSeqUninit`); (4) structured `break`/`continue` (a `for` lowers
      to `while (true) { … else break }`). Real nimony `getAt`/`firstLen` (index/len) and
      `sumSeq`/`makeSeq` (the full `for`-read and `add`-write paths) now **translate and verify**.
    - **✅ non-zero global initializers — DONE 2026-07-29.** A `gvar` with a non-zero scalar-int
      initializer becomes a module `data` segment (little-endian bytes at the global's window offset)
      — the window is otherwise zero, so a zero initializer stays a no-op, and a non-scalar/aggregate
      initializer fail-closes. Tested on hand fixtures (i32 + i64) and **real nimony `var counter:
      int = 42`** with `getCounter`/`addCounter`: the data segment seeds the window so `getCounter()`
      reads 42, interp == JIT.
    - **✅ enum/distinct scalars — DONE 2026-07-29.** A named type is an aggregate only when it's a
      locally-declared `(object …)`/`(array …)`; every other named type — an `(enum …)`, a `distinct`
      int, a `proctype`, or a type external to the module — is an integer scalar (its values are
      plain integers). `collect_types` records the aggregate names up front, so `tydesc` classifies
      as it resolves. Also hardened `if`-condition truthiness: a wide (`i64`) condition — e.g. an
      enum error code — reduces via `!= 0`, not an `i64→i32` wrap that would drop the high word.
      Tested on hand fixtures and **real nimony `toNum`** (enum compares) + **`roundtrip`** (enum
      passthrough, a scalar return, not sret).
    - **✅ exceptions (error-flag ABI) — DONE 2026-07-29.** nimony lowers exceptions with *no new
      node type*: a `.raises` proc returns an `(object (fld :fld.0 ErrorCode) (fld :fld.1 result))`
      tuple by **sret** (`fld.0` the error code — an enum, hence scalar — `fld.1` the real result);
      `raise E` is `ret (oconstr tuple (kv fld.0 <nonzero>) (kv fld.1 <default>))`; the normal return
      sets `fld.0 = 0`; and `try/except` is `var canRaise = call; if canRaise.fld.0: jmp exlab;
      result = canRaise.fld.1` with the handler under an `if (false) { lab exlab; … }` guard reached
      only via the `jmp`. So it falls straight out of sret + objects + `if` + `jmp`/`lab` +
      enum-scalar error codes — no translator change beyond the enum slice. Tested on a hand-written
      model and **real nimony `mayFail`/`guarded`** (a `distinct`-int raiser + its `try/except`
      caller), interp == JIT: the happy path doubles the input, the error path returns the handler's
      -1. Exception payloads carrying `object`-of-`RootObj` inheritance (vtables) stay fail-closed.
    - **✅ general `goto` (`jmp`/`lab`) — DONE 2026-07-28.** hexer keeps `if`/`while` structured and
      emits the low-level jump family only for `break`/`block`-`break`: `(jmp L)` an unconditional
      branch, `(lab :L)` a label. Both fall straight out of the block-parameter (slot-threading)
      model — labels are pre-scanned and each assigned a block id, a `(jmp L)` is a `br` to that
      block passing the live slot set, and a `(lab :L)` opens it (fall-through if the prior block is
      live, else reached only by jumps). Dead statements after a `jmp` are skipped until the next
      `lab` reopens a reachable block; forward and backward edges both work. Tested on hand fixtures
      and **real nimony `firstHit`** (`while`+`break`) and **`labeled`** (`block done:`/`break done`
      out of a nested loop), interp == JIT. The `jtrue`/`mflag`/`vflag` conditional-jump forms (not
      emitted by hexer's default lowering) stay fail-closed.
    - **✅ aggregate return (sret) — DONE 2026-07-28.** A proc whose return type is a named aggregate
      returns `void` and takes a hidden `$sret` pointer param (after `$sp`, before the Leng params);
      `(ret aggval)` constructs/copies the result into that pointer (composing with `oconstr`/copy).
      A caller assigning the call to an aggregate destination (`var`/`asgn`/`ret`) hands that
      destination's address down as `$sret` — the callee writes in place, no temporary; a scalar or
      discarded use of an aggregate-returning call fail-closes. Aggregate call *arguments* pass by
      address to match by-address params. Tested (both engines, incl. the window bytes the callee
      wrote): a direct sret build, a caller→callee round-trip, return-by-copy, and **real nimony
      `mk`/`mkSum`** — the genuine `var result; result = Pt(…); ret result` + `var p = mk(a,b)` bytes,
      lifted out together via the new multi-proc `translate_procs`.
    - **✅ floats — DONE 2026-07-28.** `(f 32)`/`(f 64)` types; float arithmetic
      (`fN.add/sub/mul/div`), `neg` (`fN.neg`), and comparisons (`fN.lt/le/eq/ne`); int↔float and
      f32↔f64 `conv`/`cast` (`convert_iN_s`/`trunc_fN_s`/`promote`/`demote`); float literals
      (`2.0`, `1e3`) and `(inf)`/`(neginf)`/`(nan)`; float loads/stores follow from the scalar
      type. Tested on hand fixtures and **real nimony `favg`** (`(a+b)/2.0`) + `toF` (`float(n)*1.5`),
      interp == JIT (bit-exact).
    - **✅ `case` → `br_table` — DONE 2026-07-28.** A dense-integer `case` (`(case Disc (of
      (ranges V+) Body)* (else Body)?)`) lowers to a normalized `br_table`: the discriminant is
      offset to the value span's minimum, a table entry per value maps to its covering branch, and
      an out-of-range index (negative or over-large) selects the `else`/continuation via the table
      default. Single values, multi-value `of`s, and `(range lo hi)` are handled; sparse/huge spans
      (>256) fail-close (a comparison-chain lowering is the refinement). Tested on hand fixtures and
      **real nimony `classify`** (`0 / 1,2 / 3 / else`), interp == JIT.
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

## 3a. Self-hosting roadmap — Path 2a (no C compiler)

The end state we're building toward: **nimony compiles itself on SVM with no C compiler in the
loop.** nimony is written in Nim, and `svm-leng` is its Leng→svm-ir backend. So the loop closes
when both the nimony compiler *and* `svm-leng` itself run as svm modules. Two sub-questions —
"can we translate the Leng nimony emits?" and "can the translator itself run on svm?" — and Path
2a answers the second by bootstrapping the Rust `svm-leng` onto svm the same way any Rust program
reaches svm: **Rust → wasm → the svm-wasm on-ramp → svm-ir.** No C compiler anywhere.

**Why not 2b (a Nim backend inside nimony's `lengc`, an arkham analog).** It would let nimony
emit svm-ir directly, no separate translator. Ruled out: we have **no influence over the nimony
repo**, so a backend living upstream is not a lever we control. `svm-leng` as an *external* Rust
translator keeps the whole path in this tree.

**The mapping is largely proven** (§3 above): integers, floats, control flow, pointers, frames,
objects/arrays + constructors + copy, globals, intra-/cross-module calls — all validated against
genuine `hexer` bytes, interp == JIT. The remaining work is not "can it be done" but breadth +
plumbing. Five workstreams, roughly independent:

- **W1 — Leng totality.** Close the Leng subset so *every* construct a real nimony program emits
  translates or fail-closes cleanly. Load-bearing next slices: **sret (aggregate return)**, then
  general **`goto`** (the low-level `jmp`/`lab`/`jtrue`/`mflag`/`vflag` jump family), then
  **exceptions** (`try`/`onerr`/`raise` as an error-flag model), then **seq/string** (nimony's
  built-in containers). Non-zero global/data initializers land here too.
- **W2 — Linker (the long pole).** A real program is many modules; nimony emits one Leng file per
  module. Today `svm-leng` translates one module in isolation. W2 resolves cross-module symbols,
  merges globals/data, and lays out one svm module from N Leng inputs — the analog of what the C
  on-ramp gets from `clang`+`lld` for free.
- **W3 — Runtime bottom edge** (scoped in detail in §3b). Raw syscalls / the allocator →
  POSIX-personality named imports + the Memory cap (same seam as Phase 1), and mapping nimony's TLS
  onto svm's model (the on-ramp gap Phase 1 already surfaced). ARC destructors/dup calls pass
  through as ordinary calls. **Key finding (§3b): the bottom edge is only ~15 C functions, and
  Phase 1 already binds them** — so the lever between "translates real nimony" and "runs it" is
  mostly W2 (linking the compiled `system` module), or a Phase-1-style host runtime shim.
- **W4 — Multi-binary architecture (the other long pole).** nimony is not one binary: `nifmake`
  spawns `nifler` → `nimony` → `hexer` → `lengc` as subprocesses. Running the compiler on svm
  means either driving those phases in-process or giving svm a subprocess/exec personality. This
  is an architecture question, not a translation one, and it's the biggest unknown.
- **W5 — Bootstrap + browser.** Compile the Rust `svm-leng` to wasm, on-ramp it to svm, and run
  the loop (nimony-on-svm + svm-leng-on-svm) — first headless, then as a playground demo.

**Near-term milestone — ✅ MET (2026-07-29, see §3b Path B): compile & run one real Nim program
end-to-end** — source → nimony → hexer → `svm-leng` → svm-ir → runs on both engines with the right
answer (a real seq build-and-sum returns `3`). That
exercises W1 (totality on a whole program) and forces the first slice of W2/W3, and is the
concrete "it works" we can point at before the long poles. Everything below `## 3a` (W2/W4
especially) is bounded but real; the backend mapping is the part that's no longer in doubt.

## 3b. W3 scope — the runtime bottom edge (W1 is done; this is the next lever)

With W1 closed, `svm-leng` **translates real nimony modules to verified svm-ir** — but the lowered
code doesn't yet *run*, because it calls procs that aren't defined in the one module we translate.
Scoping W3 means answering exactly *what* those calls are and *how* they bind. Two layers, and the
boundary between them is the whole story:

**Layer 1 — compiled Nim stdlib (this is W2, not W3).** The seq/string/ARC ops a program calls —
`newSeqUninit`, `add`, `[]`, `len`, `toOpenArray`, `=destroy`, `=wasMoved` — are **ordinary Nim
code** that nimony compiles into the `system` module's Leng (`sysvq0asl.x.nif`). They look like
"imports" to us only because we translate one module in isolation. In a whole-program build they're
*defined*, reached by **linking** the user module with the compiled `system` module — the `Func`/
`Slot` bindings of `svm_ir::resolve_imports_with`. That's W2 (the linker), and it's the bulk of the
gap.

**Layer 2 — the true bottom edge (this is W3).** What does the `system` module *itself* bottom out
at? Measured directly from `hexer`-compiled `sysvq0asl.x.nif`, the runtime's **entire** external
(`importc`) surface, minus pure C *type* names, is ~15 functions:

| Group | Symbols | SVM binding |
| --- | --- | --- |
| Allocator | `mmap`, `munmap` | the **Memory cap** (Phase 1 seam) |
| Syscalls / process | `write`, `_exit`, `getpid`, `kill` | the **POSIX personality** (Phase 1 seam) |
| libc mem | `memcpy`, `memset`, `memcmp` | host cap, or lower to `mem.copy`/`mem.fill` |
| Atomics | `__atomic_{load,store,add_fetch,sub_fetch,exchange,compare_exchange}_n` (+ `__ATOMIC_*` order consts) | single-threaded guest → plain loads/stores |
| Builtins | `__builtin_{bswap64,clzll,ctzll}` | direct svm ops (`bswap`/`clz`/`ctz`) |
| Dynamic linking | `dlopen`, `dlsym`, `dlclose`, `dlerror` | unused by a static program → stub / fail-closed |

**The key finding: W3's hard part is already retired.** This is the *same* C bottom edge Phase 1's
on-ramp already binds — `crates/svm-run/demos/nimony/` runs a nimony-shaped module on all three
engines today, with `write`/`mmap`/`_exit`/`memcpy` resolved through the POSIX personality + Memory
cap. So the bindings exist and are proven; W3 is *wiring*, not invention. `resolve_imports_with`
already lowers a named import to a host capability (`Cap`) — that's the seam.

**Two paths to the near-term milestone (run one real program):**

- **Path B — host runtime shim first (recommended, no linker).** Skip compiling Nim's `seqimpl`;
  bind the *high-level* ops (`newSeqUninit`/`add`/`[]`/`len`/`=destroy`/`=wasMoved` + `memcpy`/
  `memset`) directly to a small host implementation via capability bindings, exactly as Phase 1's
  `nimony_runtime_shim.c` did. Gets end-to-end *running* fast, decoupled from the W2 linker. The
  handful of ops is small and well-understood (a `{len,data*}`/`{len,cap,data}` bump/realloc
  allocator over the window).
- **Path A — link the real `system` module (fidelity, needs W2).** Merge `sysvq0asl.x.nif` into the
  user module so the stdlib ops resolve to *compiled Nim* (`Func`/`Slot`), and only the ~15 C
  primitives hit the host (`Cap`). Faithful, but gated on the W2 linker.

Recommendation: **Path B first** — mirror Phase 1 (shim → real) to hit "runs a real Nim program
end-to-end", then do W2 + Path A for fidelity. Remaining unknowns are small and known: nimony's TLS
model onto svm (the on-ramp gap Phase 1 already surfaced), and confirming the ARC destructor
protocol runs correctly against a real allocator.

**✅ Path B — DONE 2026-07-29: the near-term milestone is met.** A real nimony seq program **runs
end-to-end on SVM**, both engines, §9 parity. `svm-leng` lowers genuine `hexer` bytes for
`sumSeq`/`makeSeq` to verified svm-ir with their stdlib ops as named imports; a tiny SVM **runtime
shim** (the pure `toOpenArray`/`len`/`[]`/`inc` ops + a bump/realloc allocator for `newSeqUninit`/
`add`, `=wasMoved`/`=destroy` as zero/no-op — eight functions, ~90 lines of svm-text) is **linked
in** via `svm_ir::link`, binding each named import to a shim function. So the whole path — real Nim
→ `nimony` → `hexer` → `svm-leng` → svm-ir → link → **run** — closes with the right answer:
`sumSeq([10,20,30]) = 60`, `makeSeq(3)` builds `[0,1,2]` through the allocator, and a driver chaining
`makeSeq(3)` → `sumSeq` returns `3` in one pass (`tests/end_to_end.rs`). Notably the shim is *SVM
code linked in*, not Rust host capabilities — so it stays inside the pure-IR / both-engines model
and rides the same verifier. This is the linking mechanism W2 generalizes (many units → one), and
the shim is the placeholder the real compiled `system` module (Path A) will replace.

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
