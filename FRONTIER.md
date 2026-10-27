# Capability × axis frontier matrix

**Generated — do not edit by hand.** Regenerate with `cargo run -p temen-parity`. The classification lives in `crates/temen-parity/src/frontier.rs`; this file is its human-readable view.

INVARIANTS.md #14 says an accepted capability must hold across **seven axes**. `OPS_PARITY.md` machine-checks one of them at op granularity; this matrix is the machine for the rest (#1413). Rows are powerbox capability kinds; columns are the seven axes.

**80 of 112 cells audited** (16 capabilities × 7 axes). An ❔ cell is not a passing cell — it means nobody has established what it is. 6 of the seven columns (nesting, durability, backend, concurrency, code origin, debugger) are checked against live predicates by the conformance tests in `crates/temen-parity/tests/`; the rest state the manifest's belief and nothing more.

## Legend

- ✅ **Full** — holds
- ⛔ **Declines** — deliberately does not, and the note says why
- 🚧 **Not yet** — a real gap, and the note cites its issue
- 🔶 **Conditional** — holds where the note's condition does
- ❔ **Unaudited** — nobody has established this cell

## Axes

- **nesting** — can a §14 child hold it? (`Host::can_regrant`) *(conformance-tested)*
- **durability** — the oracle's answer when frozen and thawed (the tree-walker, the pump, the Cranelift JIT)? Rows no probe mints: `capture_durable_handles` *(conformance-tested)*
- **backend** — the oracle's answer on the bytecode pump, the Cranelift JIT and the wasm-JIT? (see OPS_PARITY.md for op-level detail) *(conformance-tested)*
- **target** — same on native / wasm32 / Windows?
- **concurrency** — the oracle's answer on the pump (every task on one thread) and on executor 2 (a thread per task)? *(conformance-tested)*
- **code origin** — usable from a §22 guest-JIT unit as from the base module? *(conformance-tested)*
- **debugger** — the oracle's answer under a debugger (the tree-walk Inspector, the bytecode debug scheduler)? *(conformance-tested)*

## Matrix

| capability | nesting | durability | backend | target | concurrency | code origin | debugger |
|----|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| `Stream` | ✅ | ✅ | ✅ | ❔ | ✅ | ✅ | ✅ |
| `Exit` | ✅ | ✅ | ✅ | ❔ | ✅ | ✅ | ✅ |
| `Clock` | ✅ | ✅ | ✅ | ❔ | ✅ | ✅ | ✅ |
| `PipeEnd` | ✅ | 🔶 | ✅ | ❔ | ✅ | ✅ | ✅ |
| `SharedRegion` | ✅ | 🔶 | 🔶 | ❔ | 🔶 | ✅ | 🔶 |
| `AddressSpace` | ⛔ | ✅ | ✅ | ❔ | ✅ | ✅ | ✅ |
| `Instantiator` | ⛔ | 🚧 | ✅ | ❔ | ✅ | ⛔ | 🔶 |
| `Budget` | ⛔ | ✅ | ✅ | ❔ | ✅ | ✅ | ✅ |
| `Module` | ✅ | 🔶 | ✅ | ❔ | ✅ | ✅ | ✅ |
| `ModuleLoader` | ⛔ | ⛔ | ✅ | ❔ | ✅ | ✅ | ✅ |
| `Jit` | ✅ | ✅ | ❔ | ❔ | ❔ | ❔ | ❔ |
| `JitCode` | ⛔ | ✅ | ❔ | ❔ | ❔ | ❔ | ❔ |
| `Blocking` | ⛔ | 🚧 | ✅ | ❔ | ✅ | ✅ | ✅ |
| `HostProc` | 🔶 | 🔶 | ✅ | ❔ | ✅ | ✅ | ✅ |
| `Offer` | ✅ | ⛔ | ❔ | ❔ | ❔ | ❔ | ❔ |
| `LiveImpl` | ✅ | ⛔ | ❔ | ❔ | ❔ | ❔ | ❔ |

## Notes

**`PipeEnd`**
- *durability* 🔶 — a pipe the tree minted rides the cut (#1680); an embedder-fed pipe, or one with an end outside the cut, is the boundary, not yet carried

**`SharedRegion`**
- *durability* 🔶 — a region the cut wholly holds rides with its aliases, across detached children, on the native JIT and in a bytecode reactor's save-state too (#2025, #2051); one held outside the cut or host-file-backed is the boundary. The bytecode tier also declines a module that calls op 4 (#1511), as on every axis it runs
- *backend* 🔶 — map, unmap, len, page_size and close run; the bytecode lowering vetoes op 4, the guest-minted region's grant (#1511: unbuilt, -EINVAL where it runs), by name, so the bytecode tier declines a module that calls it to the oracle
- *concurrency* 🔶 — map, unmap, len, page_size and close run; the bytecode lowering vetoes op 4, the guest-minted region's grant (#1511: unbuilt, -EINVAL where it runs), by name, so the bytecode tier declines a module that calls it to the oracle
- *debugger* 🔶 — map, unmap, len, page_size and close run; the bytecode lowering vetoes op 4, the guest-minted region's grant (#1511: unbuilt, -EINVAL where it runs), by name, so the bytecode tier declines a module that calls it to the oracle

**`AddressSpace`**
- *nesting* ⛔ — the child is minted its own over its own window; the parent's names coordinates the child cannot use

**`Instantiator`**
- *nesting* ⛔ — the child is minted its own over its own window; the parent's names coordinates the child cannot use
- *durability* 🚧 — rides a freeze on the tree-walker and the Cranelift JIT; the bytecode tier's durable entry declines a module that calls it (#2533)
- *code origin* ⛔ — an installed unit answers as the base module does (#1726); an invoked unit is a seam-free leaf with no Instantiator, so every op CapFaults there (§22, #1578)
- *debugger* 🔶 — join/instantiate_rec run on the debug scheduler through the executor's admission (#1855), and so does grant (op 19, #2220); child_offer (op 14) reaches the debug scheduler and is declined

**`Budget`**
- *nesting* ⛔ — not re-grantable by name: a child's own budget is the node its spawn charged (#1944)

**`Module`**
- *durability* 🔶 — durable iff the restoring host grants it again: an attested-freezable grant (#1361) or the running program's child image (#2245). The artifact carries the §4 content digest. Any other grant is still NonDurableKind::Module

**`ModuleLoader`**
- *nesting* ⛔ — not in `can_regrant`: a child that may mint modules must be granted one explicitly
- *durability* ⛔ — NonDurableKind::ModuleLoader — a live loader makes the domain non-snapshottable

**`JitCode`**
- *nesting* ⛔ — not in `can_regrant`: names a unit in its own table, meaningless in the child's

**`Blocking`**
- *nesting* ⛔ — not in `can_regrant`: index-carrying into the parent's blocking table
- *durability* 🚧 — the codec refuses a live handle (NonDurableKind::Blocking), and a Blocking.work entered after a freeze landed faults ThreadFault: both wait on #1678

**`HostProc`**
- *nesting* 🔶 — only a forkable host proc (one carrying a fork factory) crosses; a factory-less one cannot
- *durability* 🔶 — durable iff the grant carries a registered **name** (#1455) and its provider's `CapState` isn't `Uncaptured` (#1699): the closure cannot be serialized, but the name is a reconstruction rule the thaw's registrar acts on, and the provider says what state crosses with it. Otherwise NonDurableKind::HostProc

**`Offer`**
- *durability* ⛔ — NonDurableKind::Offer — an out-of-line reference to the offering domain

**`LiveImpl`**
- *durability* ⛔ — NonDurableKind::LiveImpl — an embedder wire, a JIT mint or a self-serve grant names no §14 child; a `child_offer`, minted, re-granted or forked, rides by its callee's task id (#1681)

