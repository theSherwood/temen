# FIBER_PARK.md — fiber-park promotion for blocking host calls

Status: **DECOMPOSED 2026-08-07** (owner-directed follow-up to the parking-on-blocking
arc, DESIGN.md §12). Working tracker for the arc; folds into DESIGN.md §12 when it
closes (the docs-follow-the-code rule). Slices land here as BUILT blocks.

## What this is

The parking-on-blocking arc (PR #651) made a punted host call (`Pending(completion_id)`)
park the **vCPU** (`Blocked::CapPending`) — deliberately whole-vCPU and guest-invisible,
because a fiber-level `FIBER_PARKED` unwind would be guest-visible where the fast
backends block inline (DESIGN §12 slice 3; invariant 9). That left one recorded loss:
`submit_async`/`reap` was the only **single-vCPU** compute/host overlap, and its
replacement is guest-facing concurrency — `thread.spawn` today, **fiber-park promotion
later**, never a second host-call ABI (DESIGN §12, the retirement record).

This arc is the "later": a blocking `cap.call` **inside a fiber** parks the *fiber* —
`FIBER_PARKED (3)` unwinds to the resumer, exactly the §3.6 slice-5a contract that
`memory.atomic.wait` in a fiber already follows — and the pool completion wakes it.
One vCPU, two fibers: fiber A submits and parks, the root resumes fiber B's compute,
A's completion delivers on the next resume. DESIGN §12's async-first paragraph stated
this shape from the start ("submit, park the fiber, run another, resume on completion");
the vCPU-level park was the deliberate first approximation.

**Consumers.** (1) The §12 recorded loss — single-vCPU overlap has *no* mechanism today
(the reason the loss was recorded rather than shrugged off). (2) Any guest M:N runtime
(jacl's shape): without this, one blocking call inside a fiber freezes every fiber on
the vCPU — the exact pathology DESIGN §12 rejected sync-first for. (3) I48's blocking
`cont.resume` idles on precisely the wake machinery this arc builds (it stays deferred —
one named consumer — but stops being *blocked on missing substrate*).

## Why the shape is already decided

Nearly everything is precedent, not design:

- **The park**: `fiber_park!` (interp lib.rs:9056) with a register-and-recheck closure —
  the same macro `MemoryWait` (11700), `CapReply` (10651), and `CapRead` (10781) use.
  `Blocked::CapPending` becomes the fourth fiber-parkable kind.
- **The wake**: `Scheduler::completion_drain` **already has the `Waiter::Fiber` arm**
  (interp lib.rs:4551–4559) — currently unreachable, kept "so a future handler-promotion
  slice inherits the established wake pair" (`wake_blocked` + `svc_wake_locked`). This
  arc makes it reachable; nothing about the wake pair changes.
- **Ordered delivery**: `completion_drain` is smallest-id-first, stop at the first
  not-yet-arrived — submission order (the §18 pin). Fiber waiters ride the same
  `completion_waiters` map, so the pin extends unchanged.
- **Delivery is a pushed value, not a re-dispatch**: `wake_blocked(slot, result)` pushes
  the scalar onto the parked frame (interp lib.rs:7308). A wake never re-issues the host
  call — invariant 7's rewind applies to the *resume*, the completion result rides the
  frame like every other fiber wake. (Single-`i64` replies only — invariant 8; the
  arity guard already traps `results.len() > 1` before any park.)
- **Handler-in-fiber punts come along free**: the CALLS-4b promotion branch lives
  *inside* `fiber_park!` (9061–9139) — an animated offer handler that punts files
  `OfferParked` and parks the dispatch, never the domain (the slice-5b shape). No new
  code; needs pins.
- **The explorer, durable callers, and the secondary drivers keep the blocking wait** —
  the `parkable` predicate (interp lib.rs:10760) already gates on `SchedRef::Real` and
  `!durable`; only its `*cur == ROOT_FIBER` conjunct widens. The I45 enumeration
  (parallel driver, browser `Vcpu`, debug drivers) extends to `CapPending` verbatim.

The genuinely new work is **fast-backend parity** (invariant 9). A punt is decided at
runtime per call (`OffloadOutcome`), so a compile veto cannot see it — decline is not
available, parity is mandatory. Bytecode's cooperative `drive` and the Cranelift JIT
both already carry the slice-5a *futex* fiber park (`FiberState::WaitParked`
bytecode.rs:7443; `fiber_rt` event-park seam + `FutexEntry::fibers` cells in
`os_thread_rt.rs`); each needs the same shape keyed by completion id. Until F2/F3 land,
the tree-walk↔fast-backend divergence is **tracked debt with a convergence plan**
(ISSUES entry, the I45 pattern) — witnessable only by a punt-inside-a-fiber kernel,
which no differential harness generates today (verified: `bytecode_diff`/`jit_diff`
carry no fiber kernels).

## Non-goals (recorded doors, not scope)

- **`Join`/`svc.wait`-in-fiber parks** stay vCPU-level (TODO.md §3.6 residue; the
  child-trap-propagation design question is untouched).
- **I48 blocking `cont.resume`** stays deferred pending a second consumer; this arc
  only removes the substrate excuse.
- **Durable event-parks**: freeze stays **fail-closed** on any cap-parked fiber —
  `freeze_drive`'s classification (interp lib.rs:8139) already refuses unwoken cap
  parks; a `CapPending` park inherits that. Pin it, don't change it (durability track).
- **wasm-JIT / browser tier**: no offload pool exists there; the completion source door
  (a JS event) stays a door. The wasm-JIT's non-subset `cap.call` wrappers bounce
  cross-tier already, so it inherits the executing tier's semantics.
- **I45 secondary drivers**: unchanged; the ISSUES entry gains the `CapPending` row.

## Slices

**F1 — oracle: the fiber park itself (tree-walk M:N scheduler). BUILT 2026-08-07.**
As designed below, with the as-built notes: the park's register-and-recheck closure files the
`Waiter::Fiber` under the scheduler lock then calls `completion_drain` itself (the same
insert-then-drain the vCPU park handler uses — one recheck idiom, and delivery stays ordered
even on the recheck). The domain key is captured from the dispatching host *before* the lock
drops, so inside an offer animation it is the provider's id — which is exactly `OfferPark`'s
`resume_key`, making the CALLS-4b promotion wake work by construction (`OfferPark` resumers
live in `svc_waiters`; the drain's fiber arm `svc_wake`s that key). Pins landed:
`fiber_parks.rs` (park/wake contract; **the single-vCPU rendezvous overlap** — the retirement's
lost capability, restored on the oracle; ordered delivery holding a ready later completion for
an earlier park; teardown abandoning a cap-parked fiber; the durable degenerate-wait pin) and
`svc_handler_parks.rs` (a serve-loop handler's punt parks the dispatch, never the domain —
completion wake re-admits the `svc.wait`-parked loop). Residues, recorded: the **animated-offer
punt** rides the promotion branch by construction but has no direct pin — `wire_offer_proc`
seals the provider `Host`, so no public wiring can put an offloadable cap in a provider's world
yet; pin it when one can. The interim fast-backend divergence is **ISSUES.md I73** (convergence
= F2+F3). Zero-result punts keep the degenerate wait (the wake pushes exactly one reg —
invariant 8's single-slot shape, same conjunct as the vCPU park).
Widen `parkable` (interp lib.rs:10760): a non-root fiber under `SchedRef::Real`,
non-durable, single-`i64` sig → `fiber_park!` instead of the degenerate `comps.wait(id)`.
The closure inserts `Waiter::Fiber { reg, slot, svc }` into `completion_waiters[id]`,
then re-probes `comps.try_take(id)` → immediate `wake_blocked` (register-then-recheck;
the one transient `FIBER_PARKED` the slice-5a tests pin for a pre-fired event). Delete
the "currently unreachable" comment on `completion_drain`'s `Waiter::Fiber` arm.
Tests (oracle-only, `fiber_parks.rs` style):
  - punt-in-fiber unwinds `FIBER_PARKED` to the resumer; completion wakes; the next
    resume delivers the scalar;
  - **single-vCPU overlap by rendezvous, never timing**: fiber A punts one half of a
    rendezvous-2 `Blocking` job, root resumes fiber B which punts the other half; both
    complete — impossible if either park blocked the vCPU (`max_active == 2`);
  - submission-order delivery with two parked fibers (§18 pin);
  - handler-in-fiber punt parks the dispatch, not the domain (`svc_handler_parks.rs`
    shape, via the 4b promotion branch);
  - teardown: `EXIT_WHILE_FIBER_PARKED` variant for a `CapPending` park (the
    `teardown_domain` completion sweep, interp lib.rs:5020, handles the `Fiber` arm);
  - freeze refusal: a `CapPending`-parked fiber fails the freeze closed
    (`blocking_freeze_refusal.rs` extension);
  - durable callers and the explorer keep the blocking wait (predicate pins).
ISSUES.md entry: the interim divergence, enumerated, convergence = F2+F3.

**F2 — bytecode cooperative driver parity.**
`FiberState` gains a `CapPendingParked` arm mirroring `WaitParked` (bytecode.rs:7443):
the punt site (bytecode.rs:10910) gains the fiber gate; the resume poll returns
`FIBER_PARKED`; `drive`'s idle wait must additionally wake on completions — completions
post from pool threads, so the idle sleep waits on the `Completions` condvar with the
existing deadline computation (boring; no new synchronization primitive). The serve-
qualification veto (bytecode.rs:928) is untouched — svc modules with blocking imports
already decline to the oracle. Differential: a dedicated punt-in-fiber kernel pinned
TreeWalk == Bytecode bit-exact (the `fiber_timed_wait.rs` `pin_all` pattern — the
generic harnesses generate no fiber kernels, so the pin is explicit).

**F3 — Cranelift JIT parity.**
`cap_thunk`/`cap_thunk_locked` (svm-run lib.rs:333, 446): on `Pending` with a current
fiber, register a per-fiber completion cell (the `FutexEntry::fibers` /
`fiber_cell_new` shape from `os_thread_rt.rs`, keyed by completion id) and unwind
`FIBER_PARKED` through the `fiber_rt` event-park seam (fiber_rt.rs:338, resume seam
964) instead of blocking. Root/non-fiber contexts keep the locked-thunk blocking wait
(JIT vCPUs are real threads; blocking the OS thread at root is the correct, landed
behavior). The completion hook signals the cell. Extend the F2 pin to
TreeWalk == Bytecode == Cranelift JIT (NaN-insensitive), closing the ISSUES entry.

**F4 — the overlap bench + fold.**
A bench lane for single-vCPU two-fiber overlap, measured against the retirement
record's numbers (DESIGN §12: ring ~5.2–5.8 ms/batch, threaded lane ~5.1–5.5 ms at
n=8/block=2 ms/`max_active` 4; serialized 18.5 ms) — the fiber lane should sit in the
ring's old range, restoring the recorded loss. Non-gating (the bench-regression-check
pattern); correctness stays rendezvous-pinned in F1. Then docs follow the code: this
file folds into DESIGN §12 (the "recorded follow-on" lines close), TODO.md row updated,
ISSUES entry closed by F3.

**Standing pins that must not move** (checked every slice): sync ops never touch the
parking machinery (`Completions::minted() == 0`, `host_park.rs`); a parkable
registration is never `fast_cap_resolver`-claimed (`fast_cap.rs`); the hostcall
fast-path number in the bench regression check; punt ≡ inline on every tier
(`offloadable_punt_matches_on_all_tiers`); W1 tapes force punts inline.

Sizing: F1 is the substantial slice (predicate + tests + lifecycle edges — the wake arm
already exists); F2/F3 are each a contained mirror of an existing 5a mechanism; F4 is
small. CALLS-increment-sized, not CALLS-sized.
