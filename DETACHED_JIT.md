# Detached windows on the JIT tiers, and the default-spawn question  [DRAFT — proposed as PROCESS.md §5a]

**Status:** design draft for owner decision. Nothing here is built; the interpreter half it
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
| `self.attest` → `tier \| window_exposed<<8 \| freeze_exposed<<9`; detached attests `window_exposed = false` | `lib.rs` 16358–16385, 19496 | **Built** (PROCESS.md §6) |
| Spawner keeps kill/join/fuel; detachment severs **read**, not lifecycle; live offers work (`child_offer`, op 14) | PROCESS.md §5; `detached_windows.rs` | **Built** |
| Durable domain **refuses** detached (multi-window freeze = O6, deferred) | `lib.rs` 19593–19651 | **Built (fail-closed)** |
| Native JIT (`temen-jit`) answer to op 15 | `crates/temen-jit/src/lib.rs` 8355 | **probeable `-EINVAL`** — "spawns into a fresh interpreter-owned window, which the JIT runtime does not host" |
| wasm-JIT (`temen-wasm-jit`) nested emit lowers INSTANTIATOR ops **0, 1, 13, 17 only** | `crates/temen-wasm-jit/src/lib.rs` 769 | **op 15 not lowered** |
| JIT children are **position-independent** (base is a runtime arg; compile cache reuses across offsets) | PROCESS.md §4 S1; `jit_instantiate_cache.rs` | **Built** — the emit needs no change to run in a different memory |

PROCESS.md §5 even records the motivation this note re-derives: *"nested carves subdivide
parent VA (real in the browser's wasm32 window)"* and *"Projects choose per child; a shell
would plausibly run coreutils detached and its own helper coroutines nested."*

**Consequence for #1253:** the ticket's mechanism ("grow the backing in place via
`memory.grow`" inside the shared window) is the wrong lever. The right lever is *running the
big phases as detached children on the JIT tier*, which needs the work below. PR #1268's
merged cap-and-reserve stays as the interim for nested children (it fixes the invariant-9
divergence and the 2 GiB failure); its slices 1–2 (a single top-of-memory grower) should be
**reverted** — they implement the mechanism this note retires.

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
  child's memory *is* its window: `win = 0`. Growth is the child's own `memory.grow` —
  independent of every other child and of the cdylib's heap. No partition, no topmost
  problem, no `mapped`-global re-pointing across children (each instance has its own).
- **`maximum`**: a non-shared memory needs none (grows to the wasm32 ceiling); a child that
  will `thread.spawn` into Workers must be **shared**, hence needs a declared `maximum` — V8
  reserves that as *address* and commits lazily, so it can be generous (the `WindowMinter`
  quota is the natural source of `M`).
- **Confinement**: the existing lowering is unchanged and still correct — `win + (eff &
  MASK)` with the live `"mapped"` bound. With `win = 0` and the child alone in its memory, the
  engine's own bounds check is a second, independent wall: an escape into the parent is
  **impossible by construction**, not by mask. `"mapped"` still carries the guest's
  committed-page semantics (the emitted tier has no per-page state, §14).
- **Emit reuse**: because the emit is position-independent (S1), the cached
  `nifler_ce`/`nimsem_ce` compile serves detached children unchanged. Nothing in
  `temen-wasm-jit`'s codegen changes for this slice.
- **`vm_map` growth**: a `vm_map` bounce on the child's memory maps to `memory.grow` of
  *that* memory (append-only, as today's reservation prefix model). The servicer's
  `run_cross_tier` rebuilds the child `Mem` per bounce over `back`; for a detached child
  `back` is a `Region::shared` over the child's own memory buffer, re-pointed after a grow
  (a `memory.grow` detaches the old `ArrayBuffer` — the same stale-view refresh `worker.js`
  already does).

### 3.2 Native JIT: one mmap per detached child

`temen-jit` today declines op 15 because the JIT runtime "does not host" the fresh window.
Hosting it is the same shape as a root run: `Mem::with_reservation(DEFAULT_RESERVED_LOG2,
size_log2)` gives the child its own 1 TiB reservation with lazy commit, and `run_guarded` takes
the base as its runtime argument (S1). The native tier then matches the interpreter's detached
semantics exactly, and R5 holds: detached costs the same thing everywhere — a copy at the cap
boundary — and grows the same way everywhere — into its own reservation.

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
  writes, `self.resolve` names, …) need a **copy-mediated deref**: the host copies the
  referenced `[ptr, ptr+len)` range in (and results out) rather than reading in place. Two
  implementation routes, to be settled by the chokepoint finding in §6:
  - **(preferred)** if all cap pointer derefs go through one `Mem`/`GuestMem` accessor, give a
    detached child a `Mem` whose backing *is* its own memory — on native that is just the
    child's mmap (direct); on wasm a `Region` over the child's buffer that the **JS side
    refreshes after each grow**. Then no cap changes at all: the deref lands in the right
    memory because the child's `Mem` points there.
  - **(fallback)** an explicit host import pair, `read_child_mem(child, ptr, len, dst)` /
    `write_child_mem(...)`, used by the servicer wherever a cap op dereferences a pointer.
- **Parent seeding argv / reading output**: a copy into the child's memory before `start`,
  a copy out after `join`. PROCESS.md §3's create-suspended/`grant`/seed/`start` shape already
  makes seeding an explicit step, so this is where the copy naturally lives.
- **The nimony memfs is already cap-mediated** (children call the `fs` cap; they never alias
  its bytes), so the phases work detached with **no** design change — only the bounce
  copies. The sub-window aliasing was only ever load-bearing for *direct* parent↔child byte
  sharing.
- **Interpreter/declined children** run over the child's own `Mem` exactly as the built op-15
  arm does today; on wasm a declined detached child can use `Region::Paged` (no contiguity
  needed off the emitted tier).

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

**Recommendation:** make **detached the default** spawn for new consumers (the playground
parent, the `Domain::create` substrate of PROCESS.md §3), keep **nested (op 13) as the
explicit opt-in** for tightly-coupled children, and leave every existing op-13 consumer
untouched (they already ask for nesting explicitly by calling op 13). This flips the *default
recommendation*, not the built ops — no migration is forced on existing code; the substrate
`create(window)` simply defaults its `window` argument to a minted detached window.

**Counter-argument recorded:** the "child calls parent and they share data by default" pattern
is genuinely common. Under detached-default it still works unchanged for the child; the parent
pays a copy at a boundary that is already a bounce. Only a parent that needs to *mutate* child
memory in place or walk large child structures zero-copy wants nesting — and that parent knows
it. The pattern is served by the default; the specialized case opts in.

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

| Area | Effect | Effort |
|---|---|---|
| Interpreter op-15 arm, `WindowMinter`, attest, durable-refusal | **unchanged** (built) | — |
| Interpreter nested path (`carve_fits` ×9, `sub_window`/`nested_view`, `event_instantiate`) | **unchanged** — it *is* the opt-in nested spawn | — |
| `temen-wasm-jit` codegen | **unchanged** — position-independent emit; op 15 becomes a lowered bounce like op 13 (import + arg marshal), not new memory-access surface | S |
| `temen-jit` native | host op 15: fresh `Mem::with_reservation` + `run_guarded(base)`; replace the `-EINVAL` stub | M |
| Browser servicer (`temen_op13jit_step` / `JitOnrampRun`) | new detached-child run: `back` over the child's own memory, `win = 0`, re-point after grow; the op13 phase drivers gain a detached variant | M |
| Browser JS (`driveJitRun`, `cachedInstanceF0`, `par.js`) | per-child `new WebAssembly.Memory` as `env.memory`, `win = 0`; the `env` bounce cell lives in the child's memory, so `call_interp` copies scalar arg slots into a cdylib scratch cell and results back (JS-mediated, §3.3); pointer derefs go via the §3.3 data plane; stale `ArrayBuffer` views refreshed after each child `memory.grow` | M |
| **Cap pointer-arg data plane** (§3.3) | the substantive item: either the `Mem`-chokepoint route (small) or per-op marshalling (large) | **S or L — decided by §6.3** |
| Durability | detached already refuses durable (fail-closed); multi-window freeze stays O6 | — (deferred, known) |
| Fork (#816) | fork twins are nested-window children; detached children are outside the twin — same O6 shape as durability | — (deferred) |
| Threads / Workers | a detached child that spawns threads needs a *shared* own-memory with a `maximum`; Worker fan-out of `mapped` is per-memory | M |
| Docs | PROCESS.md §5 (this as §5a), DESIGN.md §14 "parent intrinsically sees all child memory" → "…for a nested child", BROWSER.md op-13 section, INVARIANTS (#14 satisfied, not renegotiated) | S |
| Tests | new: detached-on-wasm-JIT differential vs the interpreter's detached child (byte-identical), attest report on JIT, quota exhaustion on JIT, concurrent detached children in Workers; existing nested suites untouched | M |

### 6.1 – 6.5 Per-site findings
*(filled from the code sweeps — interpreter core; durability/fork/threads; cap pointer-arg
surface and the chokepoint question; emitters + browser JS; docs/invariants/tests.)*

## 7. Slicing plan

1. **Chokepoint finding (§6.3) → data-plane design.** Decides S vs L for the whole effort.
2. **wasm-JIT detached child, single**: JS mints a per-child memory; servicer runs a detached
   `JitOnrampRun` (`win = 0`, own `back`); differential vs the interpreter's detached child
   (`detached_windows.rs` oracle shape) — byte-identical or decline. Attest on JIT.
3. **Concurrent detached children** on the wasm-JIT tier (the playground shape): N instances,
   N memories, independent growth; a Worker-hosted detached child with a shared own-memory.
4. **Native JIT hosting of op 15** (replace the `-EINVAL` stub) — R5 parity.
5. **op-13 phase drivers → detached** for the big phases (nimsem/hexer) — the concrete
   #1253 payoff: no carve ceiling, no 2 GiB module.
6. **Default flip** in the PROCESS.md §3 substrate (`create(window)` defaults to a minted
   detached window) + doc amendments.
7. **Revert #1268 slices 1–2** (the single top-of-memory grower).

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
