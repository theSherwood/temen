# Detached windows on the JIT tiers, and the default-spawn question  [DRAFT — proposed as PROCESS.md §5a]

**Status:** owner-accepted 2026-09-04 — Decision A in progress (#1253 tracks the slices), Decision B tracked in #1289 pending the §4a rulings. Nothing here is built; the interpreter half it
rests on **is** built (PROCESS.md §5, 2026-07-23). Written for #1253 / epic #706 after the
op-13 phase-child sizing work exposed that "grow the window in place" cannot be delivered by
the sub-window model at all.

**One-line summary.** The isolated, independently-growable, concurrency-safe child that
#1253 and the playground need **already exists in the design and the reference
interpreter** — it is the **detached window** (`Instantiator.instantiate_detached`, op 15,
PROCESS.md §5). What is missing is (1) **hosting it on the JIT tiers** — today both JITs
answer op 15 with a probeable `-EINVAL` — which on wasm means *one `WebAssembly.Memory` per
detached child*; and (2) a decision on whether detached should become the **default** spawn.
This note specifies (1), argues (2), and maps the blast radius of both.

---

## 0. Why this note exists — what #1253 actually found

#1253 asked to remove the op-13 phase child's 8× carve pre-size by growing its backing in
place. Tracing that to the metal (see PR #1268 and the #1253 thread) established four facts
that together rule out every in-place-grow design inside the shared window:

1. **wasm linear memory grows only at the top.** `memory.grow` appends; nothing placed
   moves; nothing shrinks. So at most **one** region — whatever is topmost — can grow its
   *address* in place. Any "top-of-memory arena" is inherently a **single grower**.
2. **A child's base can never move mid-run.** `win` is baked into the live emitted frame
   (`f0` never returns while the child runs), so a carve cannot be relocated to make room.
3. **Concurrent children therefore need their address ranges reserved up front** — and in
   one flat wasm32 memory that is **static partitioning of a scarce resource** among
   consumers whose needs are unknown at spawn. Address granted to child X is locked away
   from child Y while X lives. Twenty playground demos × any honest per-child ceiling does
   not fit, and any partition that fits starves whichever demo turns out to be the big one.
4. The **address**, not RAM, is the scarce thing: pages under a grown range are demand-zero
   (lazy) on every host, so the earlier OOM was dlmalloc exhausting the 1 GiB `maximum`,
   not physical memory.

So within the §14 sub-window model the choice is *partition or nothing*. The only structure
that gives many concurrent children independent growth is the one every OS uses: **each
child in its own address space.** In temen that structure already has a name.

## 1. What already exists (verified against the tree)

| Piece | Where | Status |
|---|---|---|
| `instantiate_detached(minter, module, grants_ptr, grants_n, entry, size_log2, quota)` — Instantiator **op 15** | `crates/temen-interp/src/lib.rs` ~11811 | **Built** (interpreter) |
| `WindowMinter` cap (`cap_id` 15), embedder-granted, byte quota, `window_minter_take` at each mint | `lib.rs` 16256, 20950–20975 | **Built** |
| Detached child = **fresh reservation + guard exactly like a root run's**: `Mem::with_reservation(DEFAULT_RESERVED_LOG2, size_log2)` — so it `vm_map`-grows into a 2^40 reservation, natively | `lib.rs` ~11900 | **Built** |
| Detached window size must equal the module's declared (`mod_ok = memory_log2 == size_log2`, §14 transparency) | `lib.rs` ~11889 | **Built** |
| Detached child's starter `Instantiator`/`AddressSpace` span its **reservation** (a root's shape), so `vm_map` grows the window. **Corrected in #1286**: the built arm bounded them to the declared size, which refused every `vm_map` past it — a detached interpreter child could not actually grow | `lib.rs` ~11940; `detached_windows::a_detached_child_grows_past_its_declared_window` | **Fixed** |
| op 15 optional trailing `(args_ptr, args_len)` — the spawn-time args payload copied to the child's `module_args_base()` (#1286) | `lib.rs` op-15 arm; `bytecode.rs` `Op::InstantiateDetached` | **Built** (both engines) |
| Resumable-engine op 15: `VcpuEvent::InstantiateDetached { module, entry, size_log2, fuel, args }` — the host mints the window (#1286) | `bytecode.rs`; browser `temen_op13jit_step` + `foreign_mint` | **Built** |
| `self.attest` → `tier \| window_exposed<<8 \| freeze_exposed<<9`; detached attests `window_exposed = false` | `lib.rs` 16358–16385, 19496 | **Built** (PROCESS.md §6) |
| Spawner keeps kill/join/fuel; detachment severs **read**, not lifecycle; live offers work (`child_offer`, op 14) | PROCESS.md §5; `detached_windows.rs` | **Built** |
| Durable domain **refuses** detached (multi-window freeze = O6, deferred) | `lib.rs` 19593–19651 | **Built (fail-closed)** |
| Native JIT (`temen-jit`) answer to op 15 | `crates/temen-jit/src/lib.rs` 8355 | **probeable `-EINVAL`** — "spawns into a fresh interpreter-owned window, which the JIT runtime does not host" |
| wasm-JIT (`temen-wasm-jit`) nested emit lowers INSTANTIATOR ops **0, 1, 13, 17 only** | `crates/temen-wasm-jit/src/lib.rs` 769 | **op 15 not lowered** |
| JIT children are **position-independent** (base is a runtime arg; compile cache reuses across offsets) | PROCESS.md §4 S1; `jit_instantiate_cache.rs` | **Built** — the emit needs no change to run in a different memory |
| Native JIT nested children **already run in their own window**: `compile_child` lowers with `sub_base = 0` and the runners allocate a fresh `GuestWindow`, copying the carve in before and back out after (S1c: "own guarded window", parent-domain shared futex) | `crates/temen-jit/src/lib.rs` 4974, 5081, 5283–5325, 4651–4658, 4788–4793 | **Built** — the native tier is one deletion (the two memcpys) away from detached semantics |

PROCESS.md §5 even records the motivation this note re-derives: *"nested carves subdivide
parent VA (real in the browser's wasm32 window)"* and *"Projects choose per child; a shell
would plausibly run coreutils detached and its own helper coroutines nested."*

**Consequence for #1253:** the ticket's mechanism ("grow the backing in place via
`memory.grow`" inside the shared window) is the wrong lever. The right lever is *running the
big phases as detached children on the JIT tier*, which needs the work below. PR #1268's
cap-and-reserve (commit 5d19eca, unmerged) stays as the interim for nested children (it fixes the invariant-9
divergence and the 2 GiB failure); its slices 1–2 (a single top-of-memory grower) are **reverted** on the PR — they implemented the
mechanism this note retires.

## 2. Requirements (accumulated from the #1253 thread)

- R1 **Independent growth per child**: a child's window grows to what *it* needs, without a
  ceiling decided at spawn and without consuming another child's room.
- R2 **Concurrency**: many children live at once (the playground: every demo a child of one
  parent guest), on the JIT tier.
- R3 **No collisions**: a child can never reach its parent's or a sibling's bookkeeping.
- R4 **Attestation**: a child can learn, unforgeably, that no ancestor can read it
  (PROCESS.md req. 5) — already the detached contract.
- R5 **Consistent incentives across platforms**: the same spawn choice must not be an
  optimization on native and a pessimization on wasm-JIT.
- R6 **Interpreter is the oracle** (INVARIANTS #9): a detached JIT child is byte-identical
  to the detached interpreter child, or declines.

Detached windows satisfy R1–R4 by construction on the interpreter today. R5 and R6 are what
the JIT hosting must preserve.

## 3. The design: host op 15 on the JIT tiers

### 3.1 wasm-JIT: one `WebAssembly.Memory` per detached child

A detached child on the wasm-JIT tier is an ordinary emitted instance whose `env.memory`
import is a **fresh** `WebAssembly.Memory` created for it, not the engine's shared memory.

- **Memory**: `new WebAssembly.Memory({ initial: 1 << (size_log2 − 16), maximum: M })`. The
  child's memory *is* its window (`win` = one host header page, below). Growth is the child's own `memory.grow` —
  independent of every other child and of the cdylib's heap. No partition, no topmost
  problem, no `mapped`-global re-pointing across children (each instance has its own).
- **`maximum`**: a non-shared memory needs none (grows to the wasm32 ceiling); a child that
  will `thread.spawn` into Workers must be **shared**, hence needs a declared `maximum` — V8
  reserves that as *address* and commits lazily, so it can be generous (the `WindowMinter`
  quota is the natural source of `M`).
- **Layout — a host header page below `win`.** The emitted code reads two host-owned things
  through *its* memory: the `env` bounce cell (`ENV_CELL_BYTES` = 2576 B; scratch slots are
  stored via `env.memory`) and the paged variant's `pagestate` table. Today both are engine-heap
  pointers, which are meaningless in a foreign memory. Put them in the **first 64 KiB page of
  the child's memory** and set `win = 65536`. Guest addresses are span-checked against
  `"mapped"` *before* `win` is added (`emit_span_check` then `emit_win_addr`), so the guest can
  never reach `[0, win)`; the header never moves on `memory.grow` (which appends at the top);
  and no new invariant is needed — the alternative, a cell *above* `mapped`, would introduce
  a `mapped ≤ env_cell_off` obligation with no analogue today and would have to relocate on
  growth while `f0` holds `env` as a live parameter.
- **Confinement**: the existing lowering is unchanged and still correct — `win + (eff &
  MASK)` with the live `"mapped"` bound. With the child alone in its memory, the engine's own
  bounds check is a second, independent wall: an escape into the parent is **impossible by
  construction**, not by mask. `"mapped"` is **still required**: the memory's page count is
  ≥ the committed extent (64 KiB grow granularity; `mapped` may sit mid-page), so dropping
  the bound would admit reads of uncommitted zero pages the interpreter faults on — an
  INVARIANT #9 trap-parity divergence. The `& MASK` (2^40 − 1, already vestigial on wasm32)
  becomes pure defense-in-depth and may be dropped later; not in this slice.
- **Emit reuse**: because the emit is position-independent (S1), the cached
  `nifler_ce`/`nimsem_ce` compile serves detached children unchanged (`OP13_CHILD_EMIT` keys on
  the module hash only; `env.memory` is imported with min 0). Nothing in `temen-wasm-jit`'s
  codegen changes for this slice. What *does* degrade: a `WebAssembly.Instance` binds its
  memory at instantiate, so `jitInstanceCache` collapses to compiled-`Module` reuse — one
  `instantiate` per child, cheap next to the ~2 s compile.
- **`shared`**: `emit_for_run(…, true)` emits `shared: 0x03`, so the per-child memory must be
  `{shared: true, maximum}` — which is also **forced** on the Worker path (a non-shared
  `Memory` cannot be `postMessage`d, and `Atomics.wait` on the child's futex words needs a
  SAB). Set `maximum` to the child's actual grant, not the engine's 16384 pages: V8 reserves a
  shared memory's `maximum` eagerly. A non-shared, no-`maximum` variant would be a second emit
  cache key — deferred.
- **`vm_map` growth**: a `vm_map` bounce on the child's memory maps to `memory.grow` of
  *that* memory (append-only, as today's reservation prefix model). The servicer's
  `run_cross_tier` rebuilds the child `Mem` per bounce over `back`; for a detached child
  `back` is a **`Region::Foreign`** (§3.3) — the cdylib has one linear memory and cannot form a
  `*mut u8` into another `WebAssembly.Memory`, so `Region::shared` is not an option here. JS
  refreshes its own `ArrayBuffer` views after a grow, as `worker.js` already does.

### 3.2 Native JIT: one mmap per detached child

`temen-jit` today declines op 15 because the JIT runtime "does not host" the fresh window.
The sweep found that it very nearly does already: `compile_child` lowers every child with
`sub_base = 0` and its own guard, and both runners (`run_child_code_then`,
`compile_child_and_run`) allocate a **fresh `GuestWindow`** per spawn, `memcpy` the carve in
before the run and back out after. A native nested child is thus *already* a private window
with copy-in/copy-out semantics; only the copies make it look like an alias. Hosting op 15 is
therefore a **deletion plus a size change**: allocate the child window with a root-sized
reservation (`Mem::with_reservation(DEFAULT_RESERVED_LOG2, size_log2)`, lazy commit), seed data
segments directly into it (`instantiator_rt::write_data_segments` currently targets the parent
carve, then copies), and drop the two memcpys — for a 256 MiB phase carve that removes ~512 MiB
of copying per spawn. `run_guarded` already takes the base as a runtime argument (S1). The
native tier then matches the interpreter's detached semantics exactly, and R5 holds: detached
costs the same thing everywhere — a copy at the cap boundary — and grows the same way
everywhere — into its own reservation.

### 3.3 The data plane: a child's memory the host cannot address

This is the substantive engineering, and it is the *same* ABI on both tiers. Today every
`call.cap` that passes a guest pointer lets the host dereference the child's memory directly,
because the child is a sub-range of a memory the host addresses. A detached child's memory is
**not** host-addressable on wasm (a different `WebAssembly.Memory`) and is a different
mapping on native. So:

- **Scalar args/results** marshal through the `env` cell (the cross-tier ABI). For a detached
  child that cell **lives in the child's own memory** — its emitted code can only store to
  its own memory — so the cdylib cannot read it in place. The `call_interp` host function
  (JS, which holds every child's `Memory`) copies the arg slots out of the child's memory into
  a cdylib scratch cell, calls `temen_onramp_jit_run_call_interp`, and copies results back.
  This is a small change confined to `driveJitRun`'s `call_interp` closure (it already reads
  the cell through a `DataView`; only *which* memory changes), but it is a real step: even
  scalar bounces cross the boundary by copy.
- **Pointer args** (`fs` read/write buffers, path and cap-name strings, `exec` argv, stream
  writes, `self.resolve` names, …) — **there is a single chokepoint, and it is already
  copy-semantic.** Every pointer-dereferencing cap op in the tree — **79** of them across
  the interpreter built-ins (19), POSIX (36), `temen-fs` (13), `temen-exec` (3), the browser
  cdylib (5) and `temen-webgpu` (3) — reaches guest memory only through `GuestMem::read_bytes`
  / `write_bytes` (`crates/temen-interp/src/lib.rs:16106`), which take/return owned bytes
  (`Vec<u8>` in, `&[u8]` out — no window borrow exists in the trait) and are documented as
  guest-relative: *"a §14 child names its own `[0, size)`, never its position in an
  ancestor's window; implementations translate to their backing."* `HostProc` hands every
  embedder personality exactly `&mut dyn GuestMem` and nothing else. So isolation for the
  cap-call ABI needs **no per-op changes and no new `read_child_mem` API** — `read_bytes`
  already *is* that API. What must be true is only that the detached child's `Mem` is backed
  by its own memory:
  - **Native**: already so — the op-15 arm mints `Mem::with_reservation(...)`, a private mmap
    the host addresses directly. Done.
  - **wasm-JIT**: the cdylib has one linear memory and cannot form a `*mut u8` into a foreign
    `WebAssembly.Memory`, so the child's backing must be a **`Region::Foreign`** variant whose
    accessors are JS host imports (`env.read_child_mem(child, ptr, len, dst)` /
    `env.write_child_mem(...)`) that copy between the child's `Memory` and cdylib scratch —
    bulk per `read_bytes`/`write_bytes` call, not per byte. It sits **below** the chokepoint, so
    the 79 ops never see it. This is the one genuinely new mechanism on wasm, and it is
    additive (a new `Region` arm, `cfg(wasm)`), not a change to any existing variant.
  - Page ops (`vm_map`/`unmap`/`protect`, 9 of them) take **offsets, not dereferenced
    pointers**, and act on the child's own region — unchanged. `SharedRegion.map` is the only
    op with cross-domain byte semantics, and it is the sanctioned sharing route (§3.4).
- **The declined-body question (the sweep's "single biggest engineering item").** A bounce
  does not only marshal scalars: `run_cross_tier` runs the **bytecode interpreter over the
  child's window** (`program.run_over(…, self.back, …)`) — today via `Region::shared` over a
  raw pointer into the one memory. This is what makes #1151's "emit declined ⇒ run
  byte-identical on the interpreter" work, and a declined *function body* dereferences
  arbitrary guest pointers, so copying the 64-slot scratch is not enough. The emitters sweep
  listed four answers — copy the whole window per bounce (native's old answer, O(carve),
  lethal at 256 MiB and Lua/SQLite bounce rates), copy touched pages (needs a dirty set the
  interpreter does not produce), multi-memory (blocked: the emit hard-codes memory index 0 in
  every memarg and the cdylib is a single-memory wasm32 target), or move the child's leaves
  into a second artifact sharing the child's memory (an architecture change: no interpreter
  fallback for detached children). **`Region::Foreign` is the fifth answer and the one this
  note proposes:** proxy *every* interpreter access, not just the cap chokepoint. `Region` is
  an enum (`Mapped`/`Shared`/`Owned`/`Paged`) whose accessors are `byte`/`set_byte`/`read_into`/
  `zero`/`copy_within`/atomics; a `Foreign` arm routes each to a JS import over the child's
  `Memory`. Correct for arbitrary leaf bodies with zero change to the interpreter or the 79
  ops; the cost is one JS↔wasm import call per interpreter memory access on the **decline
  path only** — the emitted tier runs natively in the child's memory. **Measured (slice 1,
  #1284, real Chromium):** one import call ≈ 30 ns once the JS side stops touching
  `WebAssembly.Memory.buffer` on the hot path (that getter alone is ≈ 90 ns; a stale view over
  *shared* memory is never detached, only short, so "does the access fit the cached view" is the
  staleness test). Per raw access that is ×4 (`byte`) to ×7 (8-byte word) over a direct
  linear-memory access; against an interpreted op of ≈ 10–20 ns with a fraction of ops touching
  memory, the expected end-to-end slowdown of a declined body is ≈ 2–4×. Cap leaves are bulk
  (`read_bytes`) and unaffected. Acceptable for a fallback; slice 2's differential measures the
  end-to-end number on a real declined phase, and if that proves too slow for a real consumer
  the fourth answer is the escalation.
  `Region::Foreign` is not flat-addressable (`raw_base_at = None`), so `tierup_servable`'s
  flat-window arm declines it — correct, since tier-up *into* the child's memory is the
  emitted run itself.
- **Parent seeding argv / reading output**: a copy into the child's memory before `start`,
  a copy out after `join`. Data segments are already a copy (`init_data` on the child's own
  `Mem`, as the op-15 arm does). **argv has no detached path today**: the op-13 convention is a
  *parent data segment landing inside the child's carve* at `carve_off + module_args_base()`
  (`op13_parent_src`, eight `temen-run` examples/tests, and what the Nim/C toolchains emit) —
  guest-visible and impossible to reproduce across memories. op 15 seeds `init_data` only. So
  op 15 needs a **spawn-time args payload** — `(args_ptr, args_len)` read from the *parent's*
  window and copied to `module_args_base()` in the child before start (the interpreter arm is
  one `write_bytes`; the wasm servicer one `Foreign` bulk write). PROCESS.md §3's
  create-suspended/`grant`/seed/`start` substrate is the long-run home for this (PROPOSED,
  not built); the payload is the minimal built form of the same step. Output already flows
  through caps (`fs`, stdout) — `nimc::make_exec` is the model: the grandchild's product
  leaves via the shared memfs cap, stdout discarded.
- **The nimony memfs is already cap-mediated** (children call the `fs` cap; they never alias
  its bytes), so the phases work detached with **no** design change — only the bounce
  copies. The sub-window aliasing was only ever load-bearing for *direct* parent↔child byte
  sharing.
- **Interpreter/declined children** run over the child's own `Mem` exactly as the built op-15
  arm does today. On wasm a detached child that **never enters the emitted tier** (whole-module
  decline) can live in `Region::Paged` on the cdylib heap — still its own memory, still
  attestable; one that runs emitted code must use `Region::Foreign` over its `Memory`.

### 3.4 Sharing stays available — as the explicit thing it already is

Nothing here removes zero-copy sharing. Two mechanisms exist, both **explicit grants**:

- **§14 nested spawn (op 13)**: the child is a sub-window; the parent aliases it for free.
  Stays exactly as built, as the *opt-in* placement for tightly-coupled children.
- **§13 `SharedRegion`**: lend a region *into* a detached child at some offset; pointers are
  region-relative. This is the designed way for a detached parent and child to share bulk
  data without giving up the child's private window.

So "sharing is a capability, not a mode" is already how the design is structured; this note
adds no new primitive for it.

## 4. The default-spawn question

The user-facing worry: *two modes force a decision.* Three observations shrink it.

**(a) The child cannot tell.** Confinement is one-way: a nested child is confined to its carve
and never sees parent memory; a detached child has its own memory and never sees parent memory.
A child's program is **byte-identical** under either spawn — a call into its parent is a cap
call with a pointer arg in both. Only the *parent* experiences the difference, and only if it
wants something specific (zero-copy reads of large child structures in place, or writing into
child memory before start).

**(b) The decision is already the powerbox decision.** The parent already chooses what to grant
a child (`fs`, `stdout`, `exit`, `exec`, a `SharedRegion`, …). "Place this child inside my
window" (op 13 vs op 15) is one more line in that grant list, not a new kind of decision, and
most parents never write it.

**(c) Defaults are judged by how they fail.**

| Default | Author actually needed the other | Failure mode |
|---|---|---|
| detached | zero-copy aliasing | copies at the bounce — **slower, correct** |
| nested | isolation / growth / many children | `window_exposed = true`, no attestation; VA partition exhausted — **insecure, or fails to spawn** |

A default must fail safe. Detached does; nested does not. Detached-by-default also makes
attestation the honest norm ("no ancestor holds read authority" is a checkable runtime fact)
and sidesteps the partition trap everywhere — the browser first, but a native parent spawning
thousands of children or a few multi-GiB ones partitions its 1 TiB too.

**Where the cost actually lives — decouple two decisions.** The blast-radius sweep (§6) shows
that *hosting op 15 on the JIT tiers* (§3) is additive and renegotiates **no** invariant, while
*flipping the default* is where essentially all the cost and risk sits: it renegotiates three
invariants (§4a), re-derives three doc *arguments*, and touches ~75 doc sentences. And the
motivating consumer does not need the flip: the playground parent is **host-authored**, so it
can call op 15 explicitly the moment the JIT hosts it. So:

- **Decision A — host op 15 on the JIT tiers.** Do this first, on its own. It delivers #1253
  and the playground with no semantic change to anything existing.
- **Decision B — make detached the default spawn** for new consumers (the PROCESS.md §3
  substrate's `create(window)` defaulting to a minted detached window), nested (op 13) staying
  the explicit opt-in, no migration forced on existing op-13 callers. **Recommended — but
  gated** on the three rulings in §4a, each of which is an owner decision to record, not an
  engineering step.

**Counter-argument recorded:** the "child calls parent and they share data by default" pattern
is genuinely common. Under detached-default it still works unchanged for the child; the parent
pays a copy at a boundary that is already a bounce. Only a parent that needs to *mutate* child
memory in place or walk large child structures zero-copy wants nesting — and that parent knows
it. The pattern is served by the default; the specialized case opts in.

### 4a. The three rulings Decision B needs (from the INVARIANTS.md sweep)

| Invariant | Tension | Recommended ruling |
|---|---|---|
| **#3 authority moves only down the grant graph** | Today only a `WindowMinter` (embedder-granted, byte-quota'd, "spawn *evidence*") may mint an independent window. If detached becomes the default, an ordinary `Instantiator` holder mints independent VA with no minter — VA becomes ambient-under-`Instantiator`. | Fold the minter's byte quota into **`Budget.mem`** (PROCESS.md §5's budget vector already has a `mem` field; budgets attenuate down the grant graph, so a child can never mint more than its ancestors granted). The minter then stops being a separate authority and becomes the budget it always was in spirit. Alternative: an explicit owner decision that VA is not a scarce authority — **not** recommended on wasm32. |
| **#14 one frontier (the durability cell)** | Detached windows **refuse** durable today (multi-window freeze = O6, open). Detached-by-default would make durable §14 nesting refuse *by default* — verbatim the "standing refuse workaround with no tracked plan" #14 forbids. | **Durable ⇒ nested.** A durable child must be snapshottable by its parent, and a snapshot *is* a read — so a durable child is exposed by definition. PROCESS.md §6 already states the rule: a domain may be *confidential* **or** *ancestor-durable*, **not both**. So the default is detached *unless the child is durable*, in which case the placement is nested (the alias grant is implied by durability). No O6 needed; DURABILITY.md's subtree-freeze derivation stays intact; and it is the posture the design already committed to. Evaluate this first — it is the cheapest coherent answer. |
| **#13 one canonical form** | Detached-default + an alias placement is two live placement forms, which #13 forbids as a standing dual-mode ("a migration's own scaffolding … never a standing compatibility contract"). | **One canonical form, parameterized by a grant.** The child-side ABI is byte-identical under both placements (§4(a)); what differs is a grant the parent issues. #13 governs *forms*, not *parameters* — a `SharedRegion` mapped or not is not "two forms" either. Record this reading, dated, in INVARIANTS.md. If the owner rejects it, the alternative is a dated deadline to delete the carve path once §13 `SharedRegion` covers the Stage-1 hand-offs. |

**Net effect on the invariants** (full table in §6.5): **0 violated; 6 strengthened** (#1 smaller
default TCB, **#2 the masking hinge — "no D38 contact"**, #3, #4, #8 control≠data plane, #12);
**4 unchanged** (#5, #6, #10, #11); **4 to renegotiate** (#7 recovery re-attach, #9 the wasm-JIT
tier carries or declines, #13, #14). Invariant #2 strengthening is the strongest single argument
for the whole direction: an isolated child *removes* window-access surface rather than adding it.

## 5. Platform consistency (R5)

Both spawns exist on every tier with **platform-independent semantics**:

| Spawn | Semantics | Data plane | Growth bound | Native | wasm-JIT |
|---|---|---|---|---|---|
| nested (op 13) | shares parent's address space | zero-copy alias | parent's reservation | sub-range of parent mmap | sub-range of the shared `Memory` |
| detached (op 15) | own address space | copy at the cap boundary | its own reservation | own mmap (1 TiB lazy) | own `WebAssembly.Memory` |

The dev's question is purely semantic — *do these children need to alias each other's
memory?* — and the answer is the same on every platform. The one asymmetry that survives is
**capacity of the nested mode** (native ~unlimited, wasm32 small); that is "wasm32 has less
memory," not an inverted incentive, and **memory64** is the lever that closes it (shipped in
V8 since early 2025 — BROWSER.md's table64 blocker deserves a concrete re-test).

## 6. Blast radius

Scope of the change: **add JIT-tier hosting of an existing op**, plus a default-recommendation
flip. The nested path (op 13) is **not modified** — which bounds the blast radius sharply.
Detailed per-site findings from the five code sweeps follow in §6.1–§6.5; the summary first.

Two columns because the two decisions have very different radii. **Decision A** = host op 15
on the JIT tiers, nested path untouched. **Decision B** = detached becomes the default spawn
(with the §4a rulings: durable ⇒ nested, pager ⇒ nested, minter quota → `Budget.mem`).

| Area | Decision A | Decision B (additional) | Effort A / B |
|---|---|---|---|
| Interpreter op-15 arm, `WindowMinter`, attest, durable-refusal | **unchanged** (built); + spawn-time args payload | attest default flips (`child_attestation` hardcodes `window_exposed: true`) | S / S |
| Interpreter nested path — 60 sites swept (`carve_fits` ×9, `nested_view` ×9, `event_instantiate*`, seeding ×5, thaw, futex, pager, checkpoint) | **unchanged** — it *is* the opt-in nested spawn | 22 (a) untouched · 17 (b) parallel path · 25 (c) rewrite — minus the durability/pager (c)s the rulings avoid; see §6.1 | — / M |
| Resumable engine (`Vcpu`/`drive`/`drive_parallel`/debug) | detached exists **only in the tree-walker**; a guest parent on the coop engine (the playground parent) needs an op-15 event arm: `VcpuEvent::InstantiateDetached` (additive ABI) + constructors already take an arbitrary `back` | the same arm becomes the default event | M / — |
| **`FutexKey::Anon` collision** | **pre-existing bug, exposed**: keyed on confined absolute address, and every detached child has `window.base() == 0`, so two detached siblings futexing the same guest offset already rendezvous falsely on the run-global `wait_waiters` — on the built interpreter path, today. Needs a backing discriminant (PROCESS.md:838 S1c) **before** concurrent detached children ship | forced everywhere | M / — |
| `temen-wasm-jit` codegen | **unchanged** — position-independent; `mapped` bound stays (INVARIANT #9); op 15 lowered as a bounce like op 13 | `& MASK` droppable; `fits` half of the carve gates becomes alias-only | S / S |
| `temen-jit` native | **already isolated** (`sub_base = 0`, fresh `GuestWindow`, copy-in/out): replace the `-EINVAL` stub, reserve instead of size-exactly, seed segments directly, delete two memcpys | nothing further | S–M / — |
| Browser servicer (`temen_op13jit_step` / `JitOnrampRun`) | 11 sites take `carve_ptr` into engine memory (§6.4) → detached variant over `Region::Foreign`, `win = 64 KiB` header page; `grow_backing_to_mapped` grows memory 0 (wrong memory) → JS-side grow of the child's | phase drivers default to detached; `op13_phase_open_impl`'s 2× buddy parent shrinks to KiB and `PHASE_CARVE_MAX` relaxes | M / S |
| Browser JS (`wasmjit-module.js`, `worker.js`, `par.js`) | per-child `Memory` factory; env cell + `pagestate` in the header page; `call_interp` copies slots; per-child instance; `mem_wait`/`notify` on the child's buffer; grandchild spawn posts a `Memory` (must be shared) | comments and the "confined child = shifted window" model rewritten | M / S |
| **Cap pointer-arg data plane** (§3.3) | `GuestMem` chokepoint: **0 of 79 ops change**; `Region::Foreign` below it — **plus** the decline path runs the interpreter over `Foreign` (measured gate) | none | S (+ benchmark) / — |
| Durability | detached refuses durable (fail-closed) — unchanged | **avoided by "durable ⇒ nested"**; rejecting that ruling costs D1–D6 (multi-image artifact, freeze plumbing to child `Mem`s, STW broadcast channel, thaw re-attach, DAP checkpoint): **L** | — / — (or L) |
| Fork (#816) | unchanged: `bare` gate refuses forking a parent with live children; `fork_private` already builds a private twin region | unchanged | — / — |
| Threads / Workers | Worker child needs `shared:true` + `maximum`; futex hazard above; data segments ×5 engines seeded on the child's `Mem` | the 5 seeding sites converge | M / S |
| Demand pager (op 16) | n/a (nested-only op) | **pager ⇒ nested** (a pager child is parent-writes-child by definition); else `supply_page` needs a byte-transfer leg | — / — |
| Escape oracle (`run_capture_sub`, `compile_and_run_capture_sub`, `fuzz/mask`) | unchanged — the nested path stays and stays its subject | unchanged (the oracle is the alias grant's) | — / — |
| Docs | PROCESS.md §5 (this as §5a), DESIGN.md §14, BROWSER.md op-13 section, INVARIANTS (#14 satisfied, not renegotiated) | ~75 sentences / 20 files; three *arguments* re-derived; D19 amended | S / M |
| Tests | detached-on-wasm-JIT differential vs the interpreter's detached child; attest on JIT; quota on JIT; concurrent detached children in Workers; `Foreign` micro-benchmark; futex-collision regression | isolated twins of `nested_paged`/`pagestate`/`live_mapped`/`paged_walk`; `grant_marshal_fuzz` re-plumbed | M / M |

### 6.1 Interpreter core (60 sites; classes: (a) untouched · (b) parallel path · (c) rewrite)

Under **Decision A nothing here changes** — the nested path is the opt-in spawn and the op-15
arm is the template. The classification below is the cost of **Decision B**, and it is where
the §4a rulings earn their keep.

| Cluster | Sites | Class under B | Note |
|---|---|---|---|
| `carve_fits` (bytecode.rs:1129) and its 9 callers — resumable `event_instantiate{,_module}`, debug `dbg_instantiate{,_module}`, coop `drive` ×2, `run_vcpu_parallel` ×2, tree-walker INSTANTIATOR arm | 10 | (b) | every term but power-of-two is containment-in-parent; the detached gate is op 15's `child_size != 0 && mod_ok` + quota |
| `Window::sub` / `Mem::sub_window` / `Mem::nested_view` and 9 `nested_view` callers | 12 | (a)/(b) | `temen-mask` is total and fuzzed; a detached window is the `base == 0` case it already handles. `nested_view`'s twin is `Mem::with_reservation` + `init_data` + `seed_null_guard` (lib.rs:11902–11905) |
| `carve = pbase + ibase + off` and `VcpuEvent::Instantiate { carve, … }` | 4 | **(c) public ABI** | `carve` is meaningless detached; add `InstantiateDetached` rather than overload. Consumers: `browser/src/lib.rs` 2320–2331 (`PAR_INSTANTIATE`), 9408, `nimc.rs` 347, and 6 in-repo driver tests |
| op-5/13 data segments materialized into the **parent's** `Mem` before spawn | 5 engines | **(c)** | becomes `init_data` on the child's `Mem` — the class of thing that drifts between engines (cf. #1094 guard divergence) |
| `new_confined_child*` constructor family (3200–3439) | 6 | (a) code, (b) doc | **already isolation-shaped**: `with_reservation_over(…, back)`, `base() == 0`, starter caps `grant_*(0, carve_size)`. Only the *callers* build `back` as `Region::shared(carve_ptr)` — 4 sites, (c) |
| Grant marshalling (`read_grant_list`, `regrant_list_into_child`, `take_granted_host`, coop op 11/13, tree-walker op 13/15, `spawn_named_child*`) | 14 | (a) | every read is from the **parent's** window (`guard + 1024/2048`), never the carve |
| `child_attestation` hardcodes `window_exposed: true` (lib.rs:21196–21202) | 1 (+2 callers) | **(c)** | becomes `alias_granted`; `detached_child_attestation` is the default |
| argv at `carve + module_args_base()` (§3.3) | 12 writers | **(c) guest-visible** | the spawn-time payload — the one item Decision A also needs |
| Durable freeze/thaw: `NestedChildInfo.carve_off`, `FrozenNested`, STW broadcast `write_bytes(carve_off + STATE_OFF)`, thaw `abs_carve` chain, `begin_thaw`, `durable_get_sp(abs_carve + …)`, thaw `nested_view`, resume loop without geometry re-check | 11 | (c) → **avoided** | by "durable ⇒ nested" (§4a). Rejecting the ruling: see §6.2 D1–D6 |
| Demand pager (`supply_page` "without zeroing, so the bytes the parent placed survive") | 4 | (c) → **avoided** | by "pager ⇒ nested" |
| **`FutexKey::Anon(base)`** (lib.rs:4321–4339, 24636–24656; bytecode.rs:12106, 12133) | 3 | **(c) — and a live bug** | soundness argument "anonymous pages are never aliased across domains ⇒ address is identity" holds only because nested children have distinct non-zero bases. Detached children all have base 0 and share the run's `wait_waiters`. Fix shape: `Anon(backing_ident, rel)` mirroring `Region(ident, off)` |
| Checkpoint/debug: `child_checkpointable`/`nested_within_prefix`, `env_snapshot`/`rebuild_env` | 4 | (c) | in-memory only, no wire format: swap per-env `prot` for a full `MemLayout`; the prefix restriction disappears |
| Tier-up: `tierup_servable` (`ptr_eq(back) \|\| flat_win_base()`), `pending_win` | 5 | (b) | first arm goes false for a detached child; second saves a flat `Owned`/`Mapped` region; `Foreign`/`Paged` correctly decline |
| Window-relative bookkeeping inside `Mem` (`check_prot`, fault report, `snapshot`, `zero`, …); `temen-mem` whole crate; `SharedSlots`/`ModuleSource`/`DomainTable`; op 14 `child_offer` (powerbox `Arc`, not the window); `join`/`reap` scalars | ~20 | (a) | all `base`-parameterised and fine at `base == 0` |

**Totals under B:** 22 (a) · 17 (b) · 25 (c), of which 15 (c) are removed by the two rulings.
The residual (c) set is: the `Instantiate` event ABI, the 5 seeding sites, the 4 `Region::shared`
callers, attestation, argv, futex, checkpoint.

### 6.2 Durability, fork, threads

**Durability (Decision A: unchanged; Decision B: avoided by the ruling — or L).** The invariant
being removed is stated in the code's own words four times: *"the child's entire state … lives
in its carve … so it is already in the artifact's window image"* (lib.rs:8576–8582), *"a §14
nested child records nothing … carve-self-describing"* (6749–6750), `temen-snapshot` 358–360,
DURABILITY.md 149–151. If detached children were ever durable, the cost is:

| # | Break | Sites | Effort |
|---|---|---|---|
| D1 | one `TAG_WINDOW` per artifact; §12.6 byte-identical re-freeze defined over one image; v18 format | `temen-snapshot` 379–415, 629–696, 459–466 | M–L |
| D2 | freeze has no path to a live child's `Mem` (`NestedChildInfo` stores geometry only; parent keeps `child_hosts`, never memory) | lib.rs 2877–2941, 8559–8574, 11705 | M |
| D3 | subtree STW broadcast writes `UNWINDING` *through the parent's `Mem`* at `carve_off + STATE_OFF`; failure mode is a **silently lost continuation**, not a trap | 6657–6663, 11622–11625 | S–M (hinge) |
| D4 | thaw re-attaches children out of the root image (`abs_carve` accumulation, `durable_get_sp`, `nested_view`) | 2513–2539, 2601–2622, 2730 | M |
| D5 | `FrozenNested::carve_off` semantics | 8598; snapshot 898 | S |
| D6 | DAP/time-travel checkpoint is an independent second copy of the assumption | 25202–25264; bytecode 4762–4816 | M |

The "durable ⇒ nested" ruling makes all six moot: a durable child is snapshotted by its parent,
a snapshot is a read, so it is exposed by definition — PROCESS.md §6 already says confidential
XOR ancestor-durable. Unchanged either way: digest gate, handle table, `TAG_SERVE`/`TAG_JIT`,
`freeze_sink`, `completed_result`.

**Fork (#816): unchanged under both decisions.** The `bare` gate (lib.rs:5682–5689) refuses to
fork a parent with `nested_children` or `child_hosts`, so "twin inherits child carves" is
unreachable today. `fork_private` already builds a fresh region and copies base-relative
(FORK.md §8.6); `twin_backing`/`owned_zeroed` exist for exactly this; `tierup_servable`'s
"fork twin's private copy" arm already serves an isolated flat window. If anything, detached
makes relaxing `bare` later a deliberate "duplicate N regions" rather than an accident.

**Threads.**
- Native parallel driver: `nested_view` swap + seeding on the child `Mem` — S–M. `SharedSlots`,
  `ModuleSource`, `DomainTable`, `ThreadRegistry` share *code*, not data — unchanged.
- **Futex**: the collision in §6.1. PROCESS.md:838 names the fix and its trigger verbatim
  ("cross-domain siblings need the backing-interned identity … folded into S1c when concurrent
  separate-window children exist"). Concurrent detached children *are* that event; and since
  op 15 is built, the hazard exists on the interpreter now. Under isolation the parent↔child
  anonymous-memory futex stops working by design; only §13 `SharedRegion` rendezvous
  (`FutexKey::Region`) survives — correct, but a behaviour change to document.
- **Browser/Workers**: the threads sweep called a per-child `WebAssembly.Memory` "not
  implementable" because the engine's pointers are offsets into the one imported memory and
  `Region::shared` cannot name a byte elsewhere. That is exactly the constraint `Region::Foreign`
  (§3.3) answers — the engine never holds a pointer, JS does. Its fallback, "own disjoint
  region inside the one SAB", is S–M but buys only semantic isolation (no VA growth
  independence, no attestable boundary, and it *is* the partition trap of §0 — a Worker can
  still address every byte). Two Worker facts stand: the child memory must be `shared:true`
  (postMessage; `Atomics.wait` at `worker.js` 265–276), and the powerbox recipes published via
  a process-wide `static` (THREADS.md 4c-D2) are read by the **engine**, which stays on the
  engine memory in every Worker — a child never reads that static, so this is a non-issue.
  Completion slots are `temen_par_alloc`'d in engine memory and survive.

### 6.3 Cap pointer-arg data plane (79 ops)

One chokepoint — `GuestMem::read_bytes` / `write_bytes` (lib.rs:16106), owned bytes in and
out, documented guest-relative — carries every pointer-dereferencing cap op: interpreter
built-ins 19, POSIX 36, `temen-fs` 13, `temen-exec` 3, browser cdylib 5, `temen-webgpu` 3.
`HostProc` hands each personality `&mut dyn GuestMem` and nothing else. **0 of 79 change.**
Page ops (9) take offsets. `SharedRegion.map` is the one cross-domain byte op and is the
sanctioned route. What isolation needs is only that the child's `Mem` is backed by its own
memory: native already (a private mmap); wasm via `Region::Foreign`, which also carries the
decline path (§3.3). One contract note: `child_offer` (op 14) args are `i64` by type and
pointers by convention; an offer edge that passes a window pointer is already domain-relative
today (a nested child's pointer into its carve is not a parent coordinate) — isolation changes
nothing in kind, but any such edge becomes a copy, and `nimc::make_exec` (output via the fs cap,
stdout discarded) is the pattern to standardize on.

### 6.4 Emitters and browser JS

**`temen-wasm-jit` codegen — unchanged.** `win` is param 0; `mapped` and `pagestate` are
mutable globals; `env.memory` imports with min 0; `compile_module_nested*`, `emit_confine`,
`emit_span_check`, `emit_page_check_one`, `emit_null_guard`, `emit_win_addr` all run unchanged
with any `win`. The carve gates (`check_child_carve`, `child_carve_fits`, `_growable`) have
zero production callers; their `mod_ok` half survives, their `fits` half is alias-only. The
`env.instantiate_module` bounce keeps its 9-arg shape (`grants_ptr` is always parent-relative;
`off` is reinterpreted); changing arity would fork `module_uses_instantiate_module` and the
conditional import-index arithmetic. The emit cache (`OP13_CHILD_EMIT`, `CachedEmit`,
`emit_for_run(m, shared)`) records no base — the identical ~40 MB artifact runs detached.

**`temen-jit` native — already isolated** (§3.2). `SubWindow`/`sub_base` has one construction
site (`compile_and_run_capture_sub`), the escape-oracle harness; everything downstream is the
alias path and stays. `instantiator_rt.rs:1303–1313`: `mod_ok` survives, `fits` alias-only,
`write_data_segments` retargets the child window.

**Browser servicer — where it breaks** (`browser/src/lib.rs`): `carve_ptr = d.mem_base.add(carve)`
(9419) and everything downstream — `Region::shared(carve_ptr)` decline (9447), `drive_op13`
(9457), `open_shared_run_over_host` (9488), `back = Region::shared(win_ptr, …)` (5615), data
segment seeding via `from_raw_parts_mut` (5930–5942), `run_cross_tier`'s `run_over(back)`
(5279, 6126), `grow_backing_to_mapped`'s `memory_grow::<0>` (6255 — grows the **engine's**
memory), the `pagestate` `Vec<u8>` pointer (5360), `temen_par_child_confined` (1978–1992).
Parallel-path: `win_base`/`temen_onramp_jit_run_win_ptr` (→ header-page `win`), the
`win_size == 1 << win_log2` debug assertion. Unchanged: `mapped` init, `PAR_TIERUP` shape.
**`op13_phase_open_impl` is a straight win**: the parent window is 2× the carve purely so the
carve can be the upper half (`phase_window_log2 = carve + 1`), and `PHASE_CARVE_MAX = 28` exists
because "2^28 carve → 2^29 parent over the 1 GiB Memory". Detached, the parent holds grant
records only and the cap moves up.

**JS** (`wasmjit-module.js`, `worker.js`, `par.js`): `cachedInstanceF0`/`jitInstanceCache` →
per-child instance (memory identity in the key; the #803 comment rewritten); `driveJitRun`'s
`env = temen_alloc(…)` + `DataView(memory.buffer)` + `call_interp(func, argsPtr)` → header-page
cell, JS copies the 64 slots each way; `syncGlobals` `pagestate` → header page; `worker.js`
child instance `env: { memory }` → fresh shared `Memory`; `mem_wait`/`mem_notify` on the child's
buffer; grandchild spawn posts a `Memory` handle instead of `cwin + goff` (simpler); one
stale-view cache per `Memory`; `par.js` gains a child-memory factory beside `loadEngine`'s one
engine `Memory`; the 1 GiB `maximum` is per-`Memory`, so the budget pressure `PHASE_CARVE_MAX`
encodes is relieved. `driveCoopTierupRun`'s one-memory/one-table shape is untouched for the coop
root; a detached child never enters it.

### 6.5 Docs, invariants, tests

**Decision A** touches four docs (PROCESS.md §5 → §5a, DESIGN.md §14 one sentence, BROWSER.md
op-13 section, INVARIANTS #14 "satisfied") and renegotiates nothing. **Decision B** touches ~75
sentences in 20 files, three of which are *arguments* to re-derive rather than sentences to
edit — DURABILITY.md "subtree freeze" (265–268; survives intact under durable ⇒ nested),
THREADS.md 4c-D2 (280–283), IMPORTS.md "confidentiality the memory model doesn't provide"
(761–767) — plus D19 (Settled → amended, dated), D63's nested-guard redirect (removed for
detached children: a simplification in a security-relevant lowering branch — re-run the escape
oracle), and a GLOSSARY.md line. Invariants: **0 violated; 6 strengthened** (#1, **#2 — no D38
contact**, #3, #4, #8, #12); **4 unchanged** (#5, #6, #10, #11); **4 renegotiated under B only**
(#7, #9, #13, #14 — §4a).

**Tests.** Decision A adds: detached-on-wasm-JIT differential against `detached_windows.rs`
(byte-identical or decline); attest on the JIT; quota exhaustion on the JIT; N concurrent
detached children in Workers with independent growth; a `Region::Foreign` micro-benchmark
(the §3.3 gate); a **futex-collision regression** (two detached siblings, same guest offset,
must not rendezvous — fails today on the interpreter). Existing nested suites, `fuzz/mask`,
`confined_child_null_guard.rs`, `parallel_instantiate_miri.rs`, `vcpu_instantiate_miri.rs`,
the escape oracle: untouched — they remain the alias grant's evidence. Decision B adds isolated
twins of `nested_paged`/`pagestate`/`live_mapped`/`paged_walk` and re-plumbs
`grant_marshal_fuzz`'s source window (authority half verbatim).

## 7. Slicing plan

**Decision A (no default change; no invariant renegotiation):**

0. **#1283 — File and fix the `FutexKey::Anon` collision** on the interpreter's op-15 path (a
   pre-existing bug; the regression test in §6.5). Independent of everything else and a
   prerequisite for slice 3.
1. **#1284 — `Region::Foreign` on wasm** (§3.3): the JS-proxied child-memory backing below the
   `GuestMem` chokepoint, carrying the decline path too. Additive, `cfg(wasm)`. Lands with a
   micro-benchmark of interpreted access over `Foreign` vs `Shared` — the **measured gate** for
   the declined-body cost; if unacceptable, the escalation is the sweep's option (4), not a
   silent regression.
2. **#1285 — wasm-JIT detached child, single**: JS mints a per-child shared `WebAssembly.Memory`
   (`maximum` = grant); header page at `[0, 64 KiB)` holding the `env` cell and `pagestate`,
   `win = 65536`; the servicer runs a detached `JitOnrampRun` over `Region::Foreign`;
   `call_interp` copies the slots; grow happens on the child's memory from JS; differential vs
   the interpreter's detached child (`detached_windows.rs` oracle shape) — byte-identical or
   decline. `self.attest` on the JIT reads `window_exposed = false`. Includes the **spawn-time
   args payload** on op 15 (interpreter arm + servicer) so a detached phase can take argv.
3. **#1286 — guest-issued op 15 + concurrent detached children.** 3a (**done**): the resumable-engine
   event arm (`VcpuEvent::InstantiateDetached`, args payload, minter admission), the op-13 servicer
   minting a child `Memory` per spawn (`foreign_mint`) and staging `OP13JIT_CHILD_DETACHED`, and
   the starter-caps correction above. 3b (after slices 4–5): N Worker-hosted detached children
   (grandchild spawn posts the `Memory`), independent growth, the V8-limits probe.
4. **#1287 — Native JIT hosting of op 15**: replace the `-EINVAL` stub; reserve the child window
   root-sized; seed segments directly; delete the two memcpys — R5 parity, and a perf win.
5. **#1288 — phase drivers → detached** (**done**): `nimc::detached_parent_src` spawns every phase
   (nifler/hexer/nimsem) with op 15 + the args payload; `PHASE_CARVE_MAX`, `phase_carve_log2`,
   `phase_window_log2` and the 2× buddy parent are deleted; the driver window is 64 KiB of grant
   records; the native `drive_op13` hosts the detached child over a root-sized lazy reservation.
   The interim cap-and-reserve (`5d19eca`) is thereby retired.
6. **Fuzz**: `nested_paged` gets an isolated twin; `pagestate`/`live_mapped`/`paged_walk` are
   parameterized over an isolated window; `grant_marshal_fuzz` keeps its authority half verbatim
   and re-plumbs its source window. `mask.rs` needs **no** twin (base-0 = `with_mapped`, already
   fuzzed). Open the missing nested-durability fuzz coverage DURABILITY.md already asks for.
7. **Revert #1268 slices 1–2** — done (the single top-of-memory grower this retires).

**Decision B — #1289 (gated on the three §4a rulings, recorded in INVARIANTS.md first):**

8. **Default flip** in the PROCESS.md §3 substrate (`create(window)` defaults to a minted
   detached window; **durable ⇒ nested**), `WindowMinter` quota folded into `Budget.mem`.
9. **Doc amendments** (§6.5): ~75 sentences across 20 files, of which three are *arguments* to
   re-derive, not edit — DURABILITY.md §"subtree freeze" (265–268), THREADS.md 4c-D2 (280–283),
   IMPORTS.md "confidentiality the memory model doesn't provide" (761–767); plus D19 in the
   decision log (Settled → amended, dated) and a GLOSSARY.md one-liner.

## 8. Open questions

- **V8 limits on many memories.** How many `WebAssembly.Memory` objects / how much total
  reserved address will Chrome allow one page? (Suspected: generous, but each shared memory
  reserves its `maximum`.) Probe empirically before slice 3.
- **`WindowMinter` on wasm.** The minter's byte quota is host-enforced at mint; on wasm the
  natural enforcement is the per-child `maximum` — confirm they compose.
- **memory64.** Re-test the wasm64 build in current Chromium; if it loads, nested capacity in
  the browser rises toward native and the default question gets easier on both sides.
- **Multi-window freeze (O6).** Detached + durable is refused today; the playground may want
  snapshot of a detached demo eventually — a DURABILITY.md item, not this note's.
- **Memory-per-page** (raised and dismissed): a memory per 64 KiB page would turn every
  access into a memory-select — software paging on the fast path. Not viable; a memory per
  *child* is the right granularity.
- **Decline-path cost over `Region::Foreign`.** Measured per access ×4–7 (§3.3); the end-to-end
  cost on a real declined body is slice 2's to measure. How often do the real phases decline,
  and does it matter?
- **Non-shared child memories.** A detached child that never threads could use a non-shared,
  no-`maximum` `Memory` (grows to the wasm32 ceiling with no eager reservation) — but that is
  a second emit cache key (`shared` flag) and is unusable on the Worker path. Deferred until
  a consumer needs it.
