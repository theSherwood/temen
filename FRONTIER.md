# Capability × axis frontier matrix

**Generated — do not edit by hand.** Regenerate with `cargo run -p temen-parity`. The classification lives in `crates/temen-parity/src/frontier.rs`; this file is its human-readable view.

INVARIANTS.md #14 says an accepted capability must hold across **seven axes**. `OPS_PARITY.md` machine-checks one of them at op granularity; this matrix is the machine for the rest (#1413). Rows are powerbox capability kinds; columns are the seven axes.

**68 of 112 cells audited** (16 capabilities × 7 axes). An ❔ cell is not a passing cell — it means nobody has established what it is. 5 of the seven columns (nesting, durability, concurrency, code origin, debugger) are checked against live predicates by the conformance tests in `crates/temen-parity/tests/`; the rest state the manifest's belief and nothing more.

## Legend

- ✅ **Full** — the capability holds on this axis
- ⛔ **Declines** — it deliberately does not, and the note says why
- 🚧 **Not yet** — a real gap with a tracked plan
- 🔶 **Conditional** — holds where the note's condition does
- ❔ **Unaudited** — nobody has established this cell

## Axes

- **nesting** — can a §14 child hold it? (`Host::can_regrant`) *(conformance-tested)*
- **durability** — does it survive a freeze? (`capture_durable_handles`) *(conformance-tested)*
- **backend** — same on every engine? (see OPS_PARITY.md for op-level detail)
- **target** — same on native / wasm32 / Windows?
- **concurrency** — carried by both the coop and per-Worker drivers? *(conformance-tested)*
- **code origin** — usable from a §22 guest-JIT unit as from the base module? *(conformance-tested)*
- **debugger** — observable under the debug tier? *(conformance-tested)*

## Matrix

| capability | nesting | durability | backend | target | concurrency | code origin | debugger |
|----|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| `Stream` | ✅ | ✅ | ❔ | ❔ | ✅ | ✅ | ✅ |
| `Exit` | ✅ | ✅ | ❔ | ❔ | ✅ | ✅ | ✅ |
| `Clock` | ✅ | ✅ | ❔ | ❔ | ✅ | ✅ | ✅ |
| `PipeEnd` | ✅ | ⛔ | ❔ | ❔ | ✅ | ✅ | ✅ |
| `SharedRegion` | ✅ | ⛔ | ❔ | ❔ | ✅ | ✅ | 🔶 |
| `AddressSpace` | ⛔ | ✅ | ❔ | ❔ | ✅ | ✅ | ✅ |
| `Instantiator` | ⛔ | ✅ | ❔ | ❔ | ✅ | 🚧 | 🔶 |
| `Budget` | ⛔ | ✅ | ❔ | ❔ | ✅ | ✅ | ✅ |
| `Module` | ✅ | ⛔ | ❔ | ❔ | ✅ | ✅ | ✅ |
| `ModuleLoader` | ⛔ | ⛔ | ❔ | ❔ | ✅ | ✅ | ✅ |
| `Jit` | ✅ | ✅ | ❔ | ❔ | ❔ | ❔ | ❔ |
| `JitCode` | ⛔ | ✅ | ❔ | ❔ | ❔ | ❔ | ❔ |
| `Blocking` | ⛔ | ⛔ | ❔ | ❔ | ✅ | ✅ | ✅ |
| `HostProc` | 🔶 | ⛔ | ❔ | ❔ | ✅ | ✅ | ✅ |
| `Offer` | ✅ | ⛔ | ❔ | ❔ | ❔ | ❔ | ❔ |
| `LiveImpl` | ✅ | ⛔ | ❔ | ❔ | ❔ | ❔ | ❔ |

## Notes

**`PipeEnd`**
- *durability* ⛔ — the live FIFO backing cannot be serialized (NonDurableKind::Pipe)

**`SharedRegion`**
- *durability* ⛔ — a snapshot cannot reproduce a live alias into shared backing (#14 exception)
- *debugger* 🔶 — map/unmap/len/page_size run; op 4 (the guest-minted-region grant) is vetoed by name in the bytecode lowering

**`AddressSpace`**
- *nesting* ⛔ — the child is minted its own over its own window; the parent's names coordinates the child cannot use

**`Instantiator`**
- *nesting* ⛔ — the child is minted its own over its own window; the parent's names coordinates the child cannot use
- *code origin* 🚧 — the spawn family reaches `drive_nested`'s catch-all `CapFault` inside a `Jit.invoke`: instantiate/instantiate_module_named/child_offer each answer -EINVAL probeably from the base module, and join's forgery trap differs too (#1578)
- *debugger* 🔶 — instantiate/join/instantiate_module_named/instantiate_detached compile; the coroutine spawns and instantiate_rec fall back, and child_offer (op 14) reaches the debug scheduler and is declined

**`Budget`**
- *nesting* ⛔ — index-carrying: the child is granted a sub-budget by split/transfer, not the handle

**`Module`**
- *durability* ⛔ — NonDurableKind::Module — re-granted by the embedder after restore

**`ModuleLoader`**
- *nesting* ⛔ — not in `can_regrant`: a child that may mint modules must be granted one explicitly
- *durability* ⛔ — NonDurableKind::ModuleLoader — a live loader makes the domain non-snapshottable

**`JitCode`**
- *nesting* ⛔ — not in `can_regrant`: names a unit in its own table, meaningless in the child's

**`Blocking`**
- *nesting* ⛔ — not in `can_regrant`: index-carrying into the parent's blocking table
- *durability* ⛔ — NonDurableKind::Blocking

**`HostProc`**
- *nesting* 🔶 — only a forkable host proc (one carrying a fork factory) crosses; a factory-less one cannot
- *durability* ⛔ — NonDurableKind::HostProc — the host closure cannot be serialized

**`Offer`**
- *durability* ⛔ — NonDurableKind::Offer — an out-of-line reference to the offering domain

**`LiveImpl`**
- *durability* ⛔ — NonDurableKind::LiveImpl — points at a *running* domain's powerbox

