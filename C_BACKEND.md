# C export ("C backend") — trusted native deployment target

**Status: assessment; motive fixed (trusted native deployment), slices proposed, unstarted.** This
answers "should we add a *C backend* to svm, with the option to disable the safety features?" The
answer is **yes, but not as a backend of the sandbox** — as a fenced, out-of-sandbox **IR→C
exporter** whose explicit goal is flexibility and portability to non-svm targets, with **safety as a
stated non-goal**. Fold into `DESIGN.md`/`INVARIANTS.md` and drop this file once the slices land
(repo convention, as `WASM_AOT.md` will be).

## 1. Verdict & framing — an *export*, not a fifth backend

The four backends (`DESIGN.md §3`) are *execution strategies*: they run verified IR **inside** the
sandbox and are held, four-way, to identical observable behavior. This tool does **not** execute IR —
it emits a `.c` file that *someone else's* compiler turns into native code. So it is **not** a fifth
execution strategy; it is an **export**, the C analog of the "standalone pure-wasm export"
`WASM_AOT.md` parked "behind INVARIANT #1's bar — a named compute-only consumer + an owner
renegotiation of #2." The owner is now that named consumer and is making that call — broadened past
compute-only (this export wants threads, fibers, and a powerbox), and consciously outside the
security posture.

**Why "not a backend" is the load-bearing decision.** `LLVM.md:1046` states the sandbox thesis in one
line: *"opaque machine code can't be masked or re-verified, which is the whole §2a sandbox thesis."*
An exporter that emits native-via-C, compiled by a toolchain we do not audit, is exactly that opaque
machine code. It cannot be a *mode of a sandbox backend* without dissolving §2a for the whole VM. It
*can* exist as a sibling of `svm-llvm` — which ingests an optimizing compiler's output but costs
**zero escape-TCB** precisely because it only ever emits **re-verified IR** and lives **off the
runtime path** (`DESIGN.md §20a`; `LLVM.md:93`). The C exporter takes the same fence and the same
zero-sandbox-TCB property, from the other direction: **a bug in it produces a wrong native program,
never a sandbox escape, because the sandbox never runs its output.**

Concretely it lives like `svm-llvm`: a **new crate `svm-c`, off the runtime path**, a pure host-side
`Module → String` transform depending only on `svm-ir`, exposed as a CLI
(`svm-c-translate mod.svmb -o out.c`). It is **never** a dependency of `svm-jit`/`svm-interp`, never
in the ~5 MiB runtime binary.

## 2. Scope — the supported subset, and what fails closed

The whole point is a **small-to-moderate program with a subset of features**. The subset gate is not
invented here: reuse the existing classifier `svm_parity::supports()` (`crates/svm-parity/src/lib.rs:430`),
adding a `Backend::CExport` column to the same catalog that generates `OPS_PARITY.md` (465 ops).
Anything outside the column is a **hard `Unsupported` at emit time — never a silent mistranslation**,
identical to `svm-llvm`'s pinned-subset discipline.

**Supported (emit real C):**
- The ~90% pure-compute surface — scalar int/float, conversions, memory load/store + bulk ops, direct
  and indirect `call`, all terminators/control flow (incl. irreducible; see §4.1).
- **Powerbox / host caps** — lowered to an extern C ABI the user implements (§4.3).
- **Threads** — `thread.spawn`/`join`, atomics, `memory.wait`/`notify` (§4.4).
- **Fibers** — `cont.*`/`suspend` via an opt-in context-switch shim; fail-closed where absent (§4.4).
- **Virtual memory / page ops** — the `Mem` capability (`map` / `protect` / `unmap` / `grow`,
  `map_info`) over the **reservation model** (§4.5). This is svm's headline "real virtual memory"
  feature (`DESIGN.md:3`) and it survives the export — arguably *more* naturally than in the sandbox,
  since it lowers straight onto the OS MMU.

**Fail closed (emit `Unsupported`), matching the wasm-JIT ⛔ set:**
- **Nested domains / cross-domain authority** — `instantiate`, `child_offer`, serve (`svc.poll`/`wait`),
  `export.handle`, `import.attach`. (Would require reproducing the host waiter table / grant graph.)
- **Guest-JIT** — the `Jit` capability (would re-introduce opaque codegen at guest runtime).
- **Durability / snapshot** — `durable.*`, checkpoint/restore, **snapshot/restore of the
  page-protection map** (`prot_snapshot`/COW). Note the clean line: plain page ops (map/protect/unmap)
  are *supported* (§4.5); only *snapshotting that state* fails closed.
- **Fork** — `clone_caller`, `reap`, `exec_module`.
- **GC roots** — `gc.roots` (stop-the-world stack scan).
- **SIMD `v128`** — deferred in v1 (scalarize or intrinsics is a portability choice; fail-closed first).

## 3. The emitted artifact

A **self-contained `out.c` (+ `out.h`)**, no libsvm dependency — that is the portability payoff:

- guest functions → C functions;
- the memory **window** → a flat `char *win` buffer (malloc / reservation);
- the **function table** → `static Fn table[]`, indexed by `call_indirect`;
- **imports** → `extern` declarations = the powerbox ABI header the user links against;
- a small **runtime preamble** — `memcpy`-based load/store helpers, `<stdatomic.h>` glue, and the
  optional fiber shim;
- an entry point (`svm_main` / `_start`) the host calls.

The user compiles it "in the manner of their choosing." We emit C that is **correct under default
compiler flags** (see §5) rather than assuming any particular `-f…`.

## 4. The four transforms that matter

### 4.1 CFG → C (where the architecture pays rent)
svm IR is typed SSA with **explicit block params** over an **irreducible-friendly** CFG (no relooper,
`DESIGN.md §3`). This maps to C almost exactly: **each block → a C label, each block param → a C
local, each edge → a parallel-copy of args then `goto`.** C's `goto` is unstructured, so irreducible
control flow is **free** — the contortion wasm forces on a frontend (`DESIGN.md §20`) costs nothing
here. The one subtlety is the parallel-copy/swap problem on block-param assignment (a cyclic copy
graph needs a temp) — standard and well-understood.

### 4.2 Memory without masking; fuel off
The window is a **power-of-two `PROT_NONE` reservation** (§4.5), guest pointers stay i64 offsets, and
an access is `win + off` with **no `& (reserved-1)` mask and no bounds trap** (INVARIANT #2's masking
lowering deliberately dropped). Emit loads/stores via `memcpy` into a typed temp (not a cast-deref) so
the output stays correct under default flags regardless of target alignment / strict-aliasing — the
`wasm2c` choice. **Fuel:** omit the safepoint polls entirely. This is coherent *because the artifact
has left the sandbox* — the "undisableable fuel/epoch kill" rule (`DEBUGGING.md`) exists to kill a
runaway **guest**; a trusted native binary is the user's own process to bound and kill.

**Masking is a separate knob from fuel** (§4.5): dropped by default here (raw pointers → best
compiler optimization / host interop), but the power-of-two `& (reserved-1)` clamp can be kept at
~1 op/access to recover window-confinement-against-wild-pointers even in trusted mode.

### 4.3 Powerbox → extern C ABI (the portability win)
The guest's §7 named imports (`write`→Stream, `read`, `exit`, `map`, clock, host-fns) become `extern`
C declarations in `out.h`; `cap.call` on a host cap lowers to a plain C call through that ABI.
`call_indirect` indexes the function-pointer table (drop the runtime type-check in trusted mode, or
keep a cheap one). Net: **the powerbox becomes a C header, not an svm runtime** — which is exactly
"portability to non-svm targets."

### 4.4 Concurrency — threads cheap, fibers the one real tax
- **Threads:** `thread.spawn`/`join` → pthreads or C11 `<threads.h>`; atomics → `<stdatomic.h>`;
  `memory.wait`/`notify` → a futex/condvar. Mechanical.
- **Fibers:** stackful `cont.*`/`suspend` need a stack switch, which has **no portable-standard-C
  form** — the single portability compromise in the plan. Ship an **opt-in context-switch shim**
  (`ucontext` default + per-arch asm — the `fiber_rt` substrate already in `svm-jit`), and **fail
  closed** where the target can't provide it. This mirrors how Cranelift already gates fibers on a
  per-target `fiber_rt` (`OPS_PARITY.md` 🔶 column). Be eyes-open: this is the part that is not "just
  portable C."

### 4.5 Virtual memory — the reservation model, the OS MMU as backstop
svm's headline feature ("real virtual memory", `DESIGN.md:3`) survives the export because it lowers
onto the OS MMU. Page ops are not a special opcode class — they are `cap.call`s on the granted `Mem`
capability (`Mem::with_reservation_over(reserved_log2, size_log2, back)`: a power-of-two `reserved`
region, `mapped` committed pages, a `PageProt` map, `map_info` introspection). So they ride the §4.3
powerbox seam like any host cap:

- emit `win` as a **large power-of-two `PROT_NONE` reservation** (`mmap PROT_NONE` /
  `VirtualAlloc MEM_RESERVE`), with `mapped` pages committed;
- `map`/`protect`/`unmap`/`grow`/`map_info` → extern C functions the user backs with
  `mmap(MAP_FIXED)`/`mprotect`/`munmap` (POSIX) or `VirtualAlloc`/`VirtualProtect`/`VirtualFree`
  (Windows).

**The reservation hands back the OS MMU as a real backstop, for free:** an access to an unmapped /
guard page faults (`SIGSEGV`) rather than silently corrupting — genuine VM semantics without svm's
masking pass. This makes **confinement and virtual memory independent**:

- **mask off + reservation** (default): raw `win + off`; an unmapped-page access *inside* the
  reservation → MMU fault (real VM works); an offset *beyond* the reservation → escapes to host
  memory (the accepted, dropped safety);
- **mask on + reservation** (~1 op/access): the clamp folds a wild pointer back into the reservation,
  catching the beyond-reservation case too → full window-confinement, cheaply, even in trusted mode.

Recommendation: **keep the reservation unconditionally** (it is what makes page ops meaningful and
gives the guard-page backstop); treat the **mask as a separate opt-in knob**. File-backed `mmap`
(the `MMAP_CAPABILITY.md` fs-aliasing cap) is *also* naturally supportable via real `mmap(fd)` —
easier than the sandbox's emulation — but is deferred to a follow-on slice.

## 5. Safety is a non-goal; correctness is not

Dropping the sandbox does **not** license a compiler that computes wrong answers — a trusted binary
that miscomputes is worthless. So the exporter stays **differential-tested against the tree-walk
oracle** on its subset, with the parity contract **restated** for what was deliberately dropped:

> Identical results and traps to the oracle **on in-window, terminating executions**; an access to
> an **unmapped page inside the reservation faults via the OS MMU** (`SIGSEGV`), close to the
> oracle's `MemoryFault`; an access **beyond the reservation is UB (not a trap)**; **no `OutOfFuel`.**

That is testable — diff on the program subset that stays in-window and does not depend on fuel. This
needs a **new C-export harness mode** (you cannot assert trap-parity where you deliberately do not
trap); it is a distinct harness shape, not a large one. Two emission rules keep the output correct
under a compiler we do not control:

- **wrapping integer ops in *unsigned* C** (defined overflow — avoids signed-overflow UB);
- **`memcpy` for type punning / loads / stores** (defined aliasing — avoids strict-aliasing UB).

Float **bit-exactness** is conceded to FMA-contraction / NaN propagation — the *same* concession the
Cranelift and wasm JITs already make (`DESIGN.md §3`), so it is a known, accepted divergence class,
not a new one.

## 6. Honest edges

- **Fibers** are the real portability compromise (§4.4). Everything else is portable C.
- **`v128`** deferred v1 (scalarize / intrinsics / fail-closed — fail-closed first).
- **Oracle divergence is now *intentional*** on OOB/fuel — the differential needs its own mode.
- **Page-boundary granularity**: svm's `mapped` is power-of-two `size_log2`; native follows OS page
  granularity — the same divergence `WASM.md` notes for wasm. In the export, follow OS pages and
  document it.
- **Masking is an independent knob** from fuel (§4.5) — off by default, cheap to keep on.
- **Not-TCB, but maintained:** as the IR grows, unsupported ops fail closed until the emitter learns
  them — a cheap failure mode, which is the point of the subset gate.

## 7. Slices

Ordered by leverage ÷ risk; each lands with tests per AGENTS.md.

- **Slice 1 — the compute exporter.** New `svm-c` crate; CFG→C (§4.1), SSA locals, `memcpy`
  load/store (§4.2), `Backend::CExport` added to `svm-parity` and `OPS_PARITY.md`. Golden differential
  vs. the tree-walk oracle on the pure-compute subset (the C-export harness mode, §5). No powerbox, no
  threads, no fibers yet. **Gate:** a corpus of pure-compute modules emits C that, compiled with a
  stock `cc` at `-O2`, matches the oracle (results + in-window traps).
- **Slice 2 — powerbox ABI + virtual memory.** `_start` + extern-import header (§4.3); the `Mem`
  capability over the reservation model (§4.5: `map`/`protect`/`unmap`/`grow`/`map_info` → the
  `mmap`/`VirtualAlloc` extern ABI); the chibicc/LLVM libc corpus (SHA-256, xxHash, jsmn, …) — whose
  `malloc` already grows the heap via `Mem` (`DESIGN.md §20a`) — runs as emitted C against a reference
  powerbox impl. **Gate:** the §20a eight-library corpus is byte-identical to its native `clang` build
  *and* to the oracle; a page-op corpus (guard-page trap, grow, protect) matches the oracle on
  in-window runs (unmapped access → MMU fault). File-backed `mmap` deferred (§4.5).
- **Slice 3 — threads + atomics** (§4.4). **Gate:** a threaded corpus (ring, work-stealing) matches
  the oracle on in-window runs.
- **Slice 4 — fibers via the opt-in shim** (§4.4), fail-closed where the shim is absent. **Gate:** a
  fiber corpus matches on shim-available targets; a hard `Unsupported` where not.

## Invariants check

| Invariant | How the plan holds it |
| --- | --- |
| #1 small trustworthy core | new code is a **separate `svm-c` crate off the runtime path**; zero bytes in the runtime/JIT binary; not a dependency of `svm-interp`/`svm-jit` |
| #2 confinement = masking lowering | **explicitly renegotiated for this out-of-sandbox artifact only** — masking is dropped in emitted C by design (an independent knob, §4.5: keepable at ~1 op/access, and the reservation still gives an OS-MMU backstop); the sandbox's masking pass is untouched because the sandbox never runs this output. *Owner must record the dated renegotiation in `INVARIANTS.md` when accepting.* |
| #9 oracle; decline, never diverge | subset gate reuses the single `svm_parity::supports()` predicate (one veto, no drift); unsupported ops **fail closed at emit**; the supported subset is **differential-tested vs. the tree-walk oracle** under the restated §5 contract |
| #11 top-byte guest tag | handles are exported as ordinary integer indices into the emitted function/powerbox tables — no runtime meaning in the top byte, same as the ABI boundary rule |

**Open owner decision required before Slice 1 lands:** record the INVARIANT #2 renegotiation (the
masking-off boundary for the out-of-sandbox `svm-c` export) in `INVARIANTS.md`, dated, per that file's
renegotiation rule.
