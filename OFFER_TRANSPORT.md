# Offer transport: why library instances stay animated (and `ProviderState` stays)

**Status:** decided (owner, 2026-08-04). **Decision:** keep the *animated* transport for
stateful guest-implemented capabilities; do **not** convert library instances to serve-loop-driven
callees. This supersedes the parts of CALLS.md §7 that list "`ProviderState` + its mutex" and "the
`GuestImpl`/`LiveImpl` binding split" as deletions — those are **not** being pursued.

This note records a rejected alternative and the reasoning, so the option isn't silently
re-litigated. It came out of CALLS.md increment 6 (see §8, 6d.4): 6d.4.1 folded the offer powerbox
onto the granted-child shared-cell shape, and in doing so surfaced that finishing the `ProviderState`
deletion would require this transport change.

## The three flavors of guest-implemented capability

| | persistent state | who runs the handler | analogy |
|---|---|---|---|
| **Pure offer** (`OfferEntry.state = None`) | none | animated on the caller's thread, empty world | a pure function |
| **Library instance** (`OfferEntry.state = Some`) | window + powerbox | **animated on the caller's thread** over the provider's world (§8 4a) | a library object (e.g. an `Fs` backed by its own window) |
| **Live callee** (`Binding::LiveImpl`) | its own domain | its **own serve loop** on its **own** vCPU (the 5c parked-transport crossing) | a service / process |

A **library instance** is the middle row: real state that survives calls, but **passive** — it owns
no thread. A call borrows the caller's thread: the runtime swaps the instance's world onto the
caller's vCPU, runs the handler as a fiber right there, then swaps back. A `busy` word serializes
admission so two callers can't check out one instance at once.

`ProviderState` is the struct holding a library instance's passive state:
`{ window, host (the powerbox cell), busy, admit_parked, busy_owner }`.

## The proposal that was rejected: serve-loop-driven library instances

Make a library instance behave like a live callee. Instead of animating on the caller's thread, the
instance would run its **own serve loop** parked in `svc.wait`; a call would (1) enqueue a dispatch,
(2) park the caller, (3) let the instance's loop run the handler over its own window and reply,
(4) wake the caller. This is exactly the machinery `LiveImpl` already uses — so it would collapse
"library instance" and "live callee" into one representation and one transport.

### What it would buy (pros)

- **The real deletion.** `ProviderState` + its mutex, the `Offer`/`LiveImpl` binding split, *and* the
  entire animation apparatus (checkout/settle, the world-handback leak guard, promotion-to-fiber, the
  shadow-SP durability entanglement) all retire — one transport instead of two. A large, genuine TCB
  subtraction, not the cosmetic enum-shuffle that merging the bindings alone would be.
- **Cleaner confinement story.** The handler always runs over its own window in its own run — no
  swapping a foreign world onto the caller's vCPU, no leak guard, no "did the handler spawn a fiber
  holding the world" edge cases.
- **Concurrency from the queue,** not a hand-rolled `busy` word; the `threaded` policy (increment 7)
  becomes a knob on the same serve loop.

### Why it was rejected (cons)

- **It pessimizes cross-domain calls — the decisive reason.** Animation is a direct synchronous
  crossing: swap world, run, swap back. A serve loop turns every call into enqueue → context-switch
  to the server → reply → wake the caller. For a quick stateful accessor (`counter.increment()`,
  an `Fs` stat) that is a large per-call overhead where today there is nearly none. We are measured
  against wasm/Wasmtime (DESIGN.md §1a); making the common synchronous cross-domain call slower to
  delete an internal struct is the wrong trade.
- **It reverses a deliberate decision.** §8 4a chose animate-on-caller *specifically to avoid giving
  every library object a thread.* This proposal undoes that.
- **Nothing natural drives the loop.** A spawned process has its own vCPU; a *wired library* does not.
  Either you spin up a serving vCPU per instance (a thread per `Fs` wrapper — heavy), or a caller
  "lends" its thread to pump the loop — which is animation again in a serve-loop costume.
- **Semantics shift.** A synchronous library call becomes an async round-trip with a park in the
  middle: right for a service, overkill for an accessor.

## Consequence: what stays, and the accepted price

- The **animated transport stays** as the transport for library instances. Cross-domain calls stay
  cheap.
- **`ProviderState` stays.** Its `busy`/`admit_parked`/`busy_owner` admission word and its window are
  *irreducible* for a passive instance: they must live **outside** the `Host` that the sub-run takes
  by value (otherwise a rival caller reads a stale `busy = false` and double-checks-out the instance),
  and the animated transport genuinely needs them. They cannot move into `Host` and cannot vanish
  while the transport is animated.
- The accepted price is one **uncontended nested lock** (`Mutex<Host>` inside `Mutex<ProviderState>`,
  introduced by 6d.4.1). It is only ever nested-locked on paths already serialized by `busy` or the
  `ProviderState` guard, so it never contends — a fair price for keeping the cheap synchronous call.
- **CALLS.md §7's deletion list is amended:** "`ProviderState` + its mutex" and "the
  `GuestImpl`/`LiveImpl` binding split" are **not** deleted. What increment 6 actually deleted stands:
  the two-lock nested sub-interpreter (6c), the provider-pays fuel discipline (6b), the eval-loop
  `drive_instanced_offer` (6a). The offer powerbox is now the granted-child shared-cell shape (6d.4.1).

## When to revisit

Only if library instances should genuinely *become services* — e.g. a concrete need for
provider-owned concurrency (`threaded`) or for a library to serve while the caller does other work.
That is a **performance-model** decision (accepting slower synchronous calls for those cases), not a
cleanup, and must not be smuggled in under "delete a struct." If revisited, weigh a *hybrid*: keep
animation as the default and offer a serve-loop mode only for instances that opt into concurrency —
so the fast path is never pessimized.
