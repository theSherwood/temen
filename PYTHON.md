# PYTHON.md — running Python on Temen, and the playground capstone

Status: **scoping / design doc, pre-implementation**, written 2026-09-07. This is the
work-breakdown and the load-bearing decisions for getting a **Python interpreter running
on Temen**, with the explicit end goal of a **live Python card in the browser playground**.
It leans on the LLVM on-ramp (`LLVM.md`), libc-as-capabilities (`POSIX.md`), the browser
platform (`BROWSER.md`), the frontend trust model (`FRONTEND.md` §1, `DESIGN.md` §2a), and
the C-interpreter bring-up precedent already in the tree (QuickJS, Tcl, Lua, SQLite).

This doc stays its own file — like `NIM.md`, it describes a **guest** that runs *on* Temen,
not Temen itself. Fold settled cross-cutting decisions back into the referenced docs and
delete this file once the capstone lands (repo convention).

`DESIGN.md` §20c already names this exact path: *"A guest that wants to run JS/Python/Lisp/a
DSL ships an **interpreter** (compiled to our IR via the C or LLVM on-ramp)."* Python is the
fourth walk down a well-worn road. (§20c also offers a future speed lever we don't depend on
here: `temen-peval` can partial-evaluate the interpreter against a fixed Python program — the
first Futamura projection — to get *compiled* code. Out of scope for the capstone; noted so it
isn't rediscovered.)

---

## 0. TL;DR

- **Goal.** Run real Python source on Temen, byte-identical to a native build, and expose it
  as an interactive **playground card** — the same shape as the QuickJS / Tcl / Lua / SQLite
  ports. No new VM capabilities are needed; this is a **frontend + libc-waist + REPL-driver +
  playground-registration** job (the Tcl README's phrasing, and the exact genre Python joins).

- **We ship an interpreter, we do not write a frontend.** The cheapest and only sane first
  path is to take an **existing Python interpreter written in C** and drive it through the
  proven **LLVM→TEMEN-IR on-ramp** (`clang -O2 -emit-llvm` → `llvm-link` whole-program →
  `temen-llvm-translate` → verify → run). A native `Python-AST → TEMEN-IR` frontend (the
  TypeScript-doc route) is **explicitly not** the plan — it is a multi-person-year language
  project that buys nothing the on-ramp doesn't, and throws away CPython's semantics.

- **Two candidate interpreters, cheapest-first:**
  - **Phase A — MicroPython (the recommended capstone target).** Small, self-contained, C99,
    **designed for static embedding**: `--disable`-everything builds, frozen modules baked in
    at compile time (no filesystem `import`), and its `nlr` (non-local-return) exception model
    is **setjmp/longjmp-based** — which Temen supports on all three engines. This is the
    tractable "Python runs on the playground" win.
  - **Phase D — CPython (the stretch / "real Python" prize).** Higher fidelity, but a large
    lift: needs the stdlib frozen in (no `dlopen` on Temen — see §5), threads/GIL and signals
    proven only in stubbed/single-threaded form at scale so far, and a giant computed-goto
    dispatch loop that runs slowly on the browser tiers (§6). Worth doing after MicroPython
    de-risks the pipeline; scoped honestly, not promised.

- **Reuse, don't rewrite, the libc waist.** The runtime `printf`/`scanf`/`vsnprintf` family
  (`demos/postgres/printf_shim.c`, `scanf_shim.c`), correctly-rounded `strtod`
  (`demos/strtod/strtod.c`), and guest **openlibm** already exist and are reused by every
  prior port. Python needs the same set plus C/POSIX ctype/locale tables (also already
  synthesized by the on-ramp).

- **Trust posture (unchanged).** The interpreter and its libc waist are **guest code, outside
  the escape-TCB** (`DESIGN.md` §2a, `FRONTEND.md` §1). The verifier re-checks the translated
  IR; the masking lowering confines every access (INVARIANTS §2). A CPython/MicroPython bug
  corrupts the guest's own world, never the host — exactly as for QuickJS.

- **Capstone, stated once.** *A MicroPython REPL card on the browser playground: type Python
  in the card editor, Run, get byte-identical output to native MicroPython — served from a
  pre-built, verified `.temen` asset over the warm-snapshot driver, gated by a real-browser
  play-card test.* Interactive line-at-a-time `>>>` is a defined follow-on (§6), not the bar
  for "it runs."

---

## 1. Goal & posture — what "Python on Temen" means

**The genre.** Tcl's README states it plainly, and it is our template verbatim: *"a
self-contained C interpreter for a scripting language, reached with **no new VM
capabilities** — a frontend + libc-waist + REPL-driver + playground-registration job."*
QuickJS, Lua, Tcl and SQLite are all this shape and all run **byte-identical to a native
`cc` build** on the interpreter, bytecode, and (where enabled) JIT tiers. Python is the next
member of the set.

**What we are and are not building.**
- **Are:** an existing, unmodified (or minimally-shimmed) C Python interpreter, compiled
  through the on-ramp, running sandboxed, differentially validated against a native oracle,
  and surfaced as a playground card.
- **Are not:** a new Python language frontend (Python→IR). That is the `TYPESCRIPT.md` play,
  justified there by *speed on typed code*; Python has no such static-typing story and CPython
  semantics are the product, so a frontend would be all cost, no benefit. Ruled out.
- **Are not (yet):** C extension modules via `ctypes`/`dlopen`, or true multi-core threading.
  Both are blocked by substrate realities (§5), not by effort we can spend now.

**Fidelity bar.** "Byte-identical to native" is the same acceptance test every prior port
holds to: the same source built with native `cc` is the oracle, and the guest run must match
its stdout/exit exactly (NaN-insensitive on the JIT, per INVARIANTS §9). This is what makes
"Python runs" a *claim* and not a vibe.

---

## 2. Path decision — MicroPython first

Two interpreters can plausibly wear the "Python on Temen" badge. The decision is
**MicroPython for the capstone, CPython as a tracked stretch**, on these grounds:

| Axis | MicroPython | CPython |
|---|---|---|
| Source size / link | ~small, one static lib, C99 | huge (~hundreds of TUs), whole-program `llvm-link` at Postgres scale |
| Static linking (no `dlopen`) | **native design point** (frozen modules, no dynamic import) | must freeze the stdlib in; binary C extensions impossible (§5) |
| Exception model | `nlr` = **setjmp/longjmp** (Temen: landed, 3 engines) | `setjmp` in error/signal paths; fine, but combined with threads is JIT-declined |
| Threads / GIL | single-threaded build is first-class | GIL wants real `pthread_mutex`/`cond` under contention — unproven at this scale |
| Filesystem `import` | avoidable entirely (frozen modules) | needs stdlib served via in-guest VFS image (Tcl-style) |
| Dispatch loop size | modest | `_PyEval_EvalFrameDefault` is a giant computed-goto — slow tier (§6) |
| "Real Python" fidelity | a subset dialect | the reference implementation |

MicroPython closes **every** substrate risk cheaply and gets a genuine interactive Python REPL
onto the playground. CPython is the higher-value prize but each of its rows is a separate piece
of work; sequencing it after MicroPython means the pipeline (on-ramp gaps, waist, REPL driver,
warm-snapshot card, real-browser test) is already proven when the hard interpreter arrives.

**This is the repo's own cheapest-first doctrine** (AGENTS.md prime directive; NIM.md's
"two phases, cheapest-first"). We spike on the tractable target, land it across the frontier
(INVARIANTS §14), then decide whether CPython earns its cost.

---

## 3. The reusable waist — what already exists

The on-ramp synthesizes a powerbox `_start`, binds `write`/`read`→`Stream`, `exit`→`Exit`,
`malloc`→a `vm_map`-growing allocator over the `Memory` cap, and the mem/string/non-varargs
stdio names (`memcpy`/`fwrite`/`puts`/…). Everything else is "the guest brings its own libc"
(`LLVM.md`). For Python that residual set is **already sitting in the tree**, proven by prior
ports:

| Need | Source, reused as-is |
|---|---|
| Runtime `printf`/`snprintf`/`vsnprintf` (format specs built at runtime) | `demos/postgres/printf_shim.c` |
| `sscanf`/`vsscanf` family | `demos/postgres/scanf_shim.c` |
| Correctly-rounded `strtod` / float parse | `demos/strtod/strtod.c` |
| `sin`/`cos`/`pow`/… incl. address-taken (`&sin` in a const table) | guest **openlibm**, `llvm-link`ed (the QuickJS "slice BQ/CO" mechanism) |
| C/POSIX ctype + locale tables (`setlocale`→"C") | on-ramp-synthesized (`LLVM.md` locale slice) |
| `setjmp`/`longjmp` (MicroPython `nlr`, CPython error paths) | core `SetJmp`/`LongJmp` ops, all three engines |

The point of §3: **the libc lift for Python is mostly already paid.** Python's string
formatting is the classic "runtime-built format spec" gap (identical to Lua `string.format`
and QuickJS) — the postgres printf engine covers it. Net-new shim work is expected to be
small and Python-specific (a `py_shim.c` analogous to `tcl_shim.c`/`libc_shim.c`).

---

## 4. Phase breakdown (the work)

Cheapest-first, each phase a shippable milestone with its own differential gate. Phases A–C
are the capstone; D is the stretch.

### Phase 0 — scaffolding & native oracle
- `crates/temen-run/demos/micropython/` with a `build_bitcode.sh` on the fetched-not-vendored
  pattern (fetch upstream MicroPython, build the native oracle, emit per-TU bitcode, link).
- Standard on-ramp flags: `clang -O2 -emit-llvm -fno-vectorize -fno-slp-vectorize`
  (`-fno-builtin` on the waist TUs), `llvm-link` to one module, translate with
  `temen-llvm-translate`.
- A `micropython.c` **batch-eval driver**: read a program from **stdin** to EOF, run it in
  one interpreter, print result/errors (model: `demos/quickjs/qjs_eval.c`,
  `demos/tcl/tcl_repl.c`). Minimal embedding — no ambient OS surface.
- **Gate:** the demo's fetch step skips cleanly offline; native oracle builds.

### Phase A — MicroPython runs, byte-identical (CLI)
- Walk the fail-closed translator chokepoint one gap at a time (the QuickJS method:
  `cargo run --example try_translate`), closing each on-ramp gap Python surfaces. Anticipated,
  from prior ports: computed-goto dispatch (proven by QuickJS), runtime-sized `alloca`,
  stack-direction / `frameaddress` assumptions, `i128` (big-int paths), address-taken
  `memcpy`/libm. Each gap is either a reused mechanism or a small translator fix (mirror the
  QuickJS/Tcl slice write-ups).
- Wire the reusable waist (§3) into the guest link; add a small `py_shim.c` for anything
  Python-specific.
- **Gate:** a breadth Python script (arithmetic, str/list/dict, comprehensions, closures,
  recursion, exceptions, float `repr`) runs **byte-identical to native MicroPython** on the
  tree-walk interpreter and the Cranelift JIT (differential harness in `demos/micropython/`,
  `#[ignore]` for the slow interpreter tier per the QuickJS/SQLite convention, JIT per-PR).

### Phase B — REPL driver + frozen stdlib
- Freeze the MicroPython standard modules **into the binary** (MicroPython's native frozen-
  module mechanism) so `import` needs no filesystem — the Tcl-VFS trick, but built in. If any
  real-file `import` is wanted later, the SQLite/Postgres `fs`-capability bridge is the model.
- Two-phase **warm-snapshot driver** (`warmup()` inits the interpreter into static globals,
  no stdin; `eval_run()` reads stdin, evaluates over the restored warm image, prints) — the
  QuickJS/Tcl/Lua `*_snapshot.c` contract, so per-Run isolation is preserved by restoring the
  same post-warmup snapshot and startup cost is paid once.
- **Gate:** warm driver differential-clean; frozen `import` works with no fs cap granted.

### Phase C — CAPSTONE: the playground card
- Build recipe in `browser/build-onramp-assets.mjs` (translate with **`--host-page 65536`**);
  new `want micropython` step in `scripts/rebuild-assets.sh` that builds and `validate`s
  (decode → verify → bytecode-compile via `prep_temen`).
- Add an `EXAMPLES` card to `browser/web/play.js`: `kind:'module'`, `warm:true`,
  `url:'./assets/micropython_snapshot.temen'`, `mode:'io'`, `editable:true`, `lang:'python'`,
  a starter snippet (model the Lua card).
- Commit the `.temen` asset under `browser/web/assets/` (passes the `check-play-assets.mjs` PR
  gate by being git-tracked); extend `browser/browser-play-editor-test.mjs` with a real-browser
  play-card assertion (feed the editor text as stdin, check expected stdout).
- **Gate (the capstone acceptance):** `browser-play-editor-test.mjs` drives the real
  `play.html` in headless Chromium, runs the MicroPython card, and asserts byte-correct
  Python output. Card size stays within the playground download ceiling (cf. Postgres ~20 MB,
  nifler ~17.7 MB gz).

### Phase D — CPython (stretch, tracked separately)

CPython is the "real Python" prize and a **separate, substantially larger bring-up** — not an
increment on MicroPython. Whole-program `llvm-link` of CPython (Postgres-scale link), stdlib
frozen in, threads/GIL single-threaded, signals stubbed or wired to `temen-posix`, its own gap
inventory and its own snapshot/restore for cold-boot (Postgres precedent). Gated by MicroPython
proving the pipeline first. Opened as a feasibility **spike** issue; not committed as a
deliverable here. The rest of this section is the honest difficulty read.

**MicroPython is not a stepping stone to CPython — it de-risks the *shared infrastructure*,
not the interpreter.** The two are separate codebases; almost none of MicroPython's
interpreter work transfers. What transfers is everything *around* it:

| Carries over (proven by Phases A–C) | Net-new for CPython (MicroPython never exercises it) |
|---|---|
| On-ramp pipeline: fetch → clang → `llvm-link` → translate → verify → diff vs native oracle | **Postgres-scale link** (hundreds of TUs, not one small lib) |
| Reusable libc waist (§3: printf/scanf/strtod/openlibm/ctype/locale) | **Freezing the whole stdlib** in + static-linking chosen C extension modules |
| REPL + warm-snapshot driver pattern (Phase B) | **obmalloc arenas over shimmed anon `mmap`** (Postgres proved the shim; still integration) |
| Playground card mechanics (`--host-page`, `rebuild-assets.sh`, `EXAMPLES`, real-browser test) | **GIL machinery** — pthread mutex/cond taken/released even single-threaded |
| setjmp/longjmp on all three engines | **Signal wiring** for `SIGINT`→`KeyboardInterrupt` (stub for batch; real work for interactive) |
| Translator-gap muscle (computed-goto, `i128`, `alloca`) | **Giant `_PyEval_EvalFrameDefault`** runs slow on the browser tiers (§6 — perf, not correctness) |

Roughly **a third of the total effort is shared** (the Temen-side scaffolding Phases A–C
build); the CPython interpreter + stdlib integration is the other two-thirds and is genuinely
new.

**The accelerant is not MicroPython — it is CPython's own upstream static ports.** CPython
already ships a `wasm32-wasi` target (Tier 3) and the Emscripten/**Pyodide** build, and *those
ports already solved every hard constraint Temen imposes*: static-link everything, **no
`dlopen`**, freeze the `.py` stdlib in, stub/limit threads and signals, and shim the OS surface
behind a narrow waist. Retargeting to Temen's on-ramp is therefore much closer to **"port the
CPython WASI/Emscripten build to Temen's libc waist"** than a from-scratch port: crib their
`Modules/Setup` (static module list), their deep-freeze setup, and their stub set as the
starting configuration instead of reinventing it. This is what turns CPython from open-ended
research into **a big but well-mapped integration** — MicroPython proves the Temen pipeline,
the WASI/Emscripten ports prove the CPython side.

**Rough sizing:** CPython is likely **~3–5× the MicroPython effort**, dominated by the
stdlib-freeze / static-build integration and gap-closing at scale — **not by novel VM work**
(the substrate needs nothing new; §5's gaps are all handled by static-linking or by scoped
cuts). The two biggest live unknowns are (1) the giant eval-loop's browser-tier speed (§6) and
(2) cold-boot time, for which the warm snapshot is the known answer.

**Permanent scope cuts** (architectural, not effort — the same cuts Pyodide ships and still
useful): binary third-party wheels and `ctypes`/FFI are out (no dynamic loader — §5); true
multi-core `threading` is unproven at this scale (single-threaded first); `decimal` C-level
directed rounding and non-C locales diverge (§5). A first CPython is **single-threaded,
frozen-stdlib, pure-Python-packages-only** — exactly Pyodide's shape, and plenty.

---

## 5. Gap register (honest)

The on-ramp is strong on *translation* (whole-program C at Postgres scale). The real Python
risks are on the **OS-personality / linking-model** side. Each row: the gap, and how the plan
handles it.

| Gap (source) | Impact on Python | Handling |
|---|---|---|
| **No `dlopen` / dynamic loader** (`LLVM.md`; opaque code can't be masked/re-verified — the §2a thesis) | Binary C extensions and `ctypes`/FFI are impossible | **Fully static link.** MicroPython is built for this; CPython freezes the stdlib + statically links needed extensions. `ctypes` is out of scope, stated. |
| **Threads/GIL unproven at scale** (every big demo `--disable-threads`, stubs pthreads) | CPython GIL under contention untested | MicroPython single-threaded (first-class). CPython Phase D single-threaded first; real threading is later frontier work. |
| **`setjmp`+threads JIT-declined** (`LLVM.md`) | combined use is interpreter-only | MicroPython single-threaded → `nlr` setjmp runs on the JIT. Not a capstone blocker. |
| **Async signal delivery interpreter-only + stubbed in on-ramp path** (`POSIX.md` #932) | `SIGINT`→`KeyboardInterrupt` not wired | Capstone doesn't need signals (batch eval). Interactive `^C` is a follow-on that wires to `temen-posix`. |
| **FP round-to-nearest only; `fesetround` ignored** (`LLVM.md`) | directed-rounding edge cases (some `decimal`, math corners) diverge | `repr(float)`/shortest-float converges (QuickJS proved it). Document the `decimal`-rounding caveat; not a blocker for core Python. |
| **Locale: C/POSIX only, ASCII wide-ctype** (`LLVM.md`) | non-C `LC_*`, `locale` module limited | Python's own unicode DB is built-in C (fine). Locale-dependent behavior documented as unsupported. |
| **`mmap` shim-only** (anon→malloc+zero; file→fs cap) | `mmap` module / `MAP_SHARED` limited | Not needed for the interpreter core; documented. |
| **1 MiB default stack reserve** (`LLVM.md`) | deep Python recursion → guard-page trap | Raise the guest window's stack reserve; set Python's recursion limit accordingly. Watch item, not a blocker. |
| **Cold-boot time for large interpreters** (Postgres ~100s) | slow per-Run startup | Warm-snapshot driver (Phase B) amortizes init — the standing answer. |

Also carry the **open on-ramp bugs** that a C interpreter would hit: `#1216` (inner
write-loop through a param pointer miscompiles → spurious `MemoryFault`) and the `#1311`-class
stubbed-extern arity/verify issues. Both are `area:consumers` / `topic:llvm`.

---

## 6. The playground platform — why the card is shaped the way it is

The browser runs the **bytecode interpreter compiled to wasm64**, plus an optional **wasm-JIT**
tier — never the native Cranelift JIT, never the tree-walker (`BROWSER.md`). Consequences the
card must respect:

- **Giant dispatch loops run slowly on both tiers.** `BROWSER.md`: Lua/SQLite's hot functions
  (`luaV_execute`, `sqlite3VdbeExec`) are giant and "giant emitted functions run slowly" (wasm-
  JIT speedup only ~1.3–3× vs. 24–34× for reactor demos). MicroPython's dispatch is modest;
  **CPython's `_PyEval_EvalFrameDefault` is exactly the slow shape** — a reason CPython-in-
  browser is a stretch, not the capstone.
- **Window sized to the declared `size_log2`; small allocation faults.** The card declares the
  heap up front; keep it as small as MicroPython allows to stay within download/size limits.
- **`--host-page 65536` is mandatory** for any guest with read-only data or it faults at
  startup.
- **Cooperative single-thread scheduler only**; durable/OS-thread `thread.*` fail-closed. Fine
  for single-threaded MicroPython.
- **Capabilities are scoped, not ambient**: stdin/stdout/stderr/exit/memory + an in-memory `fs`
  for served files. Frozen modules mean Python needs no fs grant at all.

**Interactivity ladder** (pick per ambition):
1. **Batch-per-Run (capstone).** Editor buffer → stdin → one eval → stdout. The `kind:'module'`
   warm card. This *is* the Lua/QuickJS/Tcl parity bar.
2. **Persistent session** (`pg`-style `runPg`): same interpreter kept alive across Runs, state
   persists like a real shell. The path to a stateful REPL where variables survive between Runs.
3. **True line-at-a-time `>>>`** (`bash-i`-style SUSPEND/RESUME worker, live keystrokes): a real
   terminal prompt. The most work; a defined follow-on, explicitly beyond the capstone bar.

---

## 7. Testing & differential strategy

- **Native oracle, byte-for-byte.** Every phase gates on matching a native build of the same
  interpreter (INVARIANTS §9; the QuickJS/Tcl/SQLite standard). NaN-insensitive on the JIT.
- **Backend parity.** Tree-walk oracle == bytecode == Cranelift JIT on the same program (the
  §3 parity invariant). Interpreter tier `#[ignore]`d for wall-clock; JIT per-PR.
- **Real-browser play-card test.** `browser-play-editor-test.mjs` drives `play.html` headless
  and asserts the card's output — the capstone's acceptance gate and the standing regression
  guard.
- **Asset re-validation.** The `.temen` asset flows through `prep_temen` (decode → verify →
  bytecode-compile) inside `rebuild-assets.sh`; it is wire-format-coupled, so an IR/encoder
  change re-runs `bash scripts/rebuild-assets.sh` (never hand-built).
- **Fuzz reach:** no new fuzz targets — the translated IR rides the existing `decode_verify`,
  `mask`, and `diff` targets like any other guest.

---

## 8. Issue / workstream mapping

- **Home epic:** #712 **Consumers & language on-ramps** (`area:consumers`) — where bash (#802),
  Tcl (#1311), Nim, and chibicc bring-ups live. Mirror bash #802 / tcl #1311 as the tracker
  shape (LLVM on-ramp compile-and-run route).
- **New label:** `topic:python` (create via the pattern in `scripts/setup-labels.sh`); add
  `touches:backends`, and `touches:web-playground` once a card exists.
- **Playground card issue:** homes under #713 **Web/Playground** (`area:web-playground`), like
  the Lua/QuickJS/Tcl warm-snapshot cards (#1142, #783, #803, #804).
- **Watch:** open on-ramp bugs #1216 and the #1311-class arity/verify issues — the frontend
  hazards a C Python runtime will hit.
- **Suggested issue slices** (one per §4 phase): (A) MicroPython translate+run byte-identical;
  (B) frozen stdlib + warm REPL driver; (C) playground card + real-browser test [capstone];
  (D) CPython feasibility spike [stretch].

---

## 9. Open decisions (for the owner)

1. **Capstone target = MicroPython?** This doc recommends yes (§2). The alternative — CPython
   first — is higher-fidelity but front-loads every substrate risk (linking, GIL, signals,
   slow dispatch) and delays a shippable playground card by a large margin.
2. **Interactivity bar for the capstone.** Batch-per-Run warm card (recommended, matches
   Lua/QuickJS/Tcl) vs. committing up front to a persistent-session or true-`>>>` REPL (more
   work; can follow).
3. **CPython, if/when:** worth a Phase-D spike issue now (to inventory its on-ramp gaps at
   Postgres scale) or defer entirely until MicroPython lands? Recommendation: open the spike,
   don't staff it until A–C are green.
