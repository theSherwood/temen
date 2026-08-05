# CALLS.md — the unified cross-domain call model

Design for consolidating the three guest-capability call mechanisms into one, settled with the
owner 2026-07-30. This is the companion to FORK.md (fork rides the parked transport defined here)
and the successor to the IMPORTS.md §3.2/§3.5/§3.6 offer trio. Status: **design agreed, not yet
built**; the increment plan is §8.

## 1. The problem (what we have today)

Three mechanisms implement "guest code provides a capability," each with its own name at each
layer, its own execution model, and its own restrictions:

| today's names | state | dispatch | backends |
|---|---|---|---|
| `impl_offer` / `wire_impl` / `Binding::GuestImpl(state: None)` | none (windowless) | sync inline sub-run | all three |
| `impl_service` / `wire_impl_instance` / `GuestImpl(state: Some)` | provider instance (window+powerbox) | sync `drive_arc` sub-run **under two held locks** | all three |
| `child_offer` / `Binding::LiveImpl` | the callee domain's one world | async: enqueue, park caller, serve loop, reply | **eval loop only** (JIT answers `-EINVAL`, `svm-interp:15780`) |

Concrete defects, all traced in code:

- **A second execution model in the TCB.** The instanced offer runs a *nested* `drive_arc`
  sub-interpreter on the caller's thread while holding **both** the caller's powerbox mutex and
  the provider's state mutex (`svm-interp:15839`). It needs its own deadlock argument (the
  acyclicity rule: "providers never hold offers") and its own fuel regime (provider-pays,
  `GUEST_IMPL_FUEL`).
- **The caller's powerbox lock is held across the whole sub-run** — an artifact of routing the
  dispatch through one `&mut self` entry (`cap_dispatch_slots`), not a semantic need. Only the
  two `cap`-slot translation edges touch the caller's host. Effect: the calling **domain's whole
  cap surface halts** for the duration of every instanced call.
- **The provider mutex serializes all callers for the whole call** — undermining providers that
  could be concurrent (guest code already has atomics/futexes and §12 defines racing access; a
  provider entered by N callers is the same situation as a multi-vCPU domain, which we support).
  Mutual exclusion is the *provider's* choice to make, not the runtime's to impose.
- **Backend parity gap.** The one mechanism with good blocking behavior (`live_impl`) is
  eval-loop-only; a JIT guest cannot call a stateful guest capability through it at all.
- **The acyclicity restriction is a lock artifact.** The parked transport already supports
  `A[f1] → B[f2] → A[f3] → B[f4]` — each inbound call is a fresh handler fiber, an outbound call
  from a handler parks the *dispatch*, never the domain (`handler_parks`; proven end-to-end by
  `svc_serve_chain`). No lock crosses a domain boundary, so there is no cycle hazard.
- **Non-mnemonic names.** Three unrelated words ("offer", "service", "live") for three points on
  one axis, renamed again at every layer.

## 2. The unified semantics (backend-neutral)

> A `cap.call` through an offer **runs that op's handler over the provider's world** and returns
> its results to the calling fiber. If the handler blocks, the caller waits.

- **Every call is synchronous from the guest's perspective.** `cap.call` in, results out. There
  are no futures, promises, or completion tokens in the guest ABI. "Sync vs async" was never a
  semantic distinction — it was two *transports* (§4).
- **Every cross-domain call mints a handler fiber in the callee's world.** Who runs that fiber is
  a scheduling decision, not a semantic one.
- **Blocking is per-fiber, never per-domain.** A blocked call parks the calling fiber; the
  calling vCPU may run other fibers, and the domain's cap surface stays live.
- **Cycles are legal.** Each inbound call is a fresh handler fiber; parks compose.

## 3. The offer taxonomy (two axes, declared by the provider)

**Axis 1 — does the provider have a `main`?**

- **Library provider** (no main; replaces the instanced offer): instantiated with its own window
  + powerbox; handlers run only when called; admissible at any time. Nobody writes a serve loop.
- **Process provider** (has a main; today's live domain): calls are admitted at its serve points
  (`svc.wait`/`svc.poll`), rendezvousing with its own execution — unchanged from §3.6.

**Axis 2 — concurrency policy.**

- **`single`** (default): run-to-park atomicity, admissions serialized — exactly the guarantee
  all existing handler code assumes. For a library provider this is a try-enter admission gate
  whose *contended* path is the serve queue: uncontended calls are lock-free-fast, contended
  callers enqueue + park (bounded; `-EAGAIN` when full). The provider "mutex" survives only as
  this uncontended flag.
- **`threaded`** (opt-in): concurrent handler fibers; the provider synchronizes its own state
  with guest atomics/futexes. Runtime imposes nothing. (Same defined-race regime as a
  multi-vCPU domain, §12; confinement never depends on data-race freedom.)

**Degeneracies that make the taxonomy collapse cleanly:**

- A **pure function offer** is a library provider with *no window* — purity by construction
  (empty powerbox, nothing to race, `threaded` trivially safe). Today's v1 pure offer survives
  unchanged as this degenerate case.
- `Binding::GuestImpl` and `Binding::LiveImpl` merge into **one `Offer` binding**; provider kind
  is carried by the entry, not the binding taxonomy.

## 4. Transports (how each backend implements "the caller waits")

| backend | fast path (handler doesn't park) | promotion (handler parks) |
|---|---|---|
| tree-walk interp | **inline animation**: the caller's thread switches to the callee's world and runs the handler fiber directly — a function call with a mem/host switch (the §14 coroutine-drive shape) | free by construction: interp fibers are **reified data** (`Vec<Frame>` in the shared registry), so the parked handler is filed with a waiter and any worker resumes it later (`handler_parks` is already this) |
| JIT | direct call into the provider's compiled code through a thunk — native speed, no scheduler | the OS thread **blocks on the reply** — legitimate because JIT vCPUs are real threads (already how JIT futex waits behave) |
| bytecode | declines and falls back to the tree-walker (the existing oracle-fallback pattern) | via the tree-walker |

Observable results and provider state are identical across backends (the §18 oracle holds);
scheduling interleavings are not, exactly as for multi-vCPU runs today (deterministic-explorer
territory). `single` providers keep interleaving determinism per admission order. This design
**closes** the current `live_impl` parity gap instead of widening it.

**Locking discipline:** the caller's powerbox lock is held only at the two `cap`-slot translation
edges (with a generation re-check on the relock — the ABA machinery exists), never across the
handler run. No lock crosses a domain boundary.

**Addendum (2026-07-31):** §10 sharpens this section — it pins what "admission" is (§10.1), adds
the **direct-handoff** arm into a process provider parked at `svc.wait` (§10.2), folds quiesce
into the admission word (§10.3), bounds JIT crossing depth (§10.4), and adds the `fuel.remaining`
readout (§10.6).

## 5. The three views

- **Guest (caller):** unchanged. `cap.call` returns results; a call may block the calling fiber;
  siblings and the domain continue; `-EAGAIN` is the probeable backpressure answer. One calling
  convention on all backends.
- **Provider:** a module with impl exports + the two declarations of §3. Handlers are ordinary
  functions; they may block; state lives in the provider's own window. Only process providers
  write a serve loop.
- **Host/embedder:** one wiring constructor — wire an offer over a **module** (runtime
  instantiates a library provider) or over a **running child** (process provider). `host_proc`
  (native closure, caller-window access, owes `fork_ctx` per FORK.md) is unchanged.

## 6. Naming (the house rule: func = pure, proc = effectful)

One name per concept at **every** layer (svm-run constructor, `Host` method, interp binding):

| new name | replaces | meaning |
|---|---|---|
| `offer_func` | `impl_offer` / `wire_impl` / `GuestImpl(state: None)` | pure guest function: windowless, empty powerbox, args in → results out |
| `offer_proc` | `impl_service` / `wire_impl_instance` / `GuestImpl(state: Some)` **and** `child_offer` / `LiveImpl` | stateful guest provider (library or process), called by the §2 semantics |
| `host_proc` | `host_fn` / `grant_host_fn` / `Binding::HostFn` | effectful **native** procedure; the only cap kind with caller-window access; the only kind owing `fork_ctx` |

`func` stays the generic callable in the IR (`Func`, `funcidx`, text `func`) — the purity contract
is a capability-surface distinction, not an IR one.

## 7. What this deletes / what it adds

**Deleted:** the nested `drive_arc`-under-two-locks sub-interpreter; the acyclicity rule and its
deadlock argument; provider-pays fuel (`GUEST_IMPL_FUEL`); `wire_impl_instance` / `impl_service`.

**Amended (owner decision 2026-08-04, see `OFFER_TRANSPORT.md`):** `ProviderState` + its mutex and
the `GuestImpl`/`LiveImpl` binding split are **not** deleted. Finishing those deletions would require
making library instances *serve-loop-driven* (an enqueue-park-reply-wake round trip in place of the
§8 4a animated crossing), which **pessimizes the common synchronous cross-domain call** — the wrong
trade against the wasm/Wasmtime yardstick (§1a). The animated transport stays; `ProviderState`'s
admission word + window are irreducible for it (they must live outside the host the sub-run takes by
value). What increment 6 *did* delete stands (the two-lock sub-interpreter, provider-pays, the
eval-loop `drive_instanced_offer`); the offer powerbox is now the granted-child shared-cell shape
(§8 6d.4.1). The JIT's `-EINVAL`/host-side fall-through for offers is **deferred**, not deleted (§8
6d.2): all tiers share the one correct host-side arm.

**Added:** admission into library providers (reuses the existing queue + handler-fiber machinery);
the inline cross-domain animation fast path (assembled from existing pieces: §14 coroutine drive +
`serve_switch`'s fiber switch); the JIT thunk fast path + thread-blocking promotion.

Net (as realized — see `OFFER_TRANSPORT.md` for why it stops short of the original "→ one"): the
lock-held nested interpreter and its safety argument leave the TCB; provider-pays retires; the offer
powerbox becomes the granted-child shared-cell shape. The **two transports are kept on purpose** —
the animated crossing for passive library instances (cheap synchronous calls) and the serve-loop
crossing for live callees — because collapsing them would pessimize cross-domain calls. A genuine
simplification of the execution path, not a collapse to a single mechanism.

## 8. Increment plan (same discipline as FORK.md: smallest verifiable step first)

1. **Rename. DONE (2026-07-30).** The §6 table: `offer_func`/`offer_proc`/`host_proc` across
   svm-run constructors, `Host` methods, and the `host_proc` family; interp-internal
   `GuestImpl`→`Offer` (`Binding::Offer`, `OfferEntry`, `resolve_offer`, `OFFER_FUEL`).
   Mechanical, no behavior change.
2. **De-fang the instanced path. DONE (2026-07-30).** Re-sequenced from the original "retire onto
   library admission" once the build surfaced that `offer_proc` is covered by tests **on all three
   backends** (`imports_impl`), so retiring it onto an eval-loop-only mechanism would *regress*
   the very parity this design exists to close. Instead, the instanced path keeps its all-backend
   generic-dispatch home but loses its teeth: admission is **try-enter** — a busy instance answers
   a probeable `-EAGAIN`, never a blocking wait — so deadlock is impossible *structurally*, and
   the acyclicity rule ("providers never hold offers") is **lifted**: `grant_impl_cap` now accepts
   offers, and a cyclic/re-entrant call is a refusal, not a hang (pinned by test: a provider
   granted its own offer calls it; the caller observes `-EAGAIN`). The blocking-lock deadlock
   argument leaves the TCB. Deferred with it: caller-pays **fuel** — fuel exhaustion is
   observable, so it must switch on every backend at once (increment 5, with the JIT arm).
3. **Inline fast path (interp).** Decomposed into slices, smallest verifiable step first; the
   original "contended calls take the queue instead of `-EAGAIN`" half is **re-scoped** to co-land
   with increment 4 (see 3c) because it is unsafe without promotion.
   - **3a — Narrow the powerbox lock. DONE (2026-07-31).** The instanced sub-run now runs with the
     caller's powerbox lock held **only** around the two `cap`-slot translation edges, never across
     the `drive_arc` sub-run (a new `drive_instanced_offer` free fn; per-arm pre-probes
     `instanced_offer_of`/`_for_import`/`_for_dyn` mirroring the LiveImpl pre-probe, wired into the
     `cap.call`/`call.import`/`call.sym`/`call.import.dyn` eval-loop arms). `drive_arc` and
     try-enter/`-EAGAIN` are kept verbatim, so behavior is **byte-identical** to increment 2
     (pinned by the existing `impl_wiring`/`imports_impl`/`bytecode_diff`/`jit_diff` suites, plus a
     new `concurrent_instanced_offer_calls_are_safe_under_the_narrowed_lock` two-vCPU safety test).
     No generation re-check is needed on the relock this slice: the instance is named by the cloned
     `state` `Arc` and results are minted fresh, and `drive_arc` has no suspension point (the
     re-check belongs to 3c/increment 4, where a handler parks). The only new lock edge is
     `state → hg`; it cannot cycle with the generic `hg → state.try_lock()` (non-blocking) during a
     live run — the blocking `hg → state` introspection APIs are not called concurrently with a run
     (invariant doc'd on `grant_impl_cap`; clean fix rides increment 6).
   - **3b — Inline animation (interp). RE-SCOPED into increment 4 (2026-07-31).** Investigation
     found the drive-mechanism swap is not a worthwhile *standalone* slice: `drive_arc` already runs
     a single-vCPU, run-to-completion offer handler entirely on the **calling thread** (`worker_loop`,
     no worker OS threads spawned unless the handler itself spawns), so replacing it with an
     *isolated* nested `run_inner` buys only marginal per-call setup — and its output is **throwaway**,
     because the value of inline animation (a parked handler as a *reified fiber*, the CALLS.md §4
     "free by construction" promotion) needs the handler to run as a fiber in a **shared** scheduler,
     not a fresh isolated one. That shared-scheduler cross-domain fiber (different `mem`/`host`, the
     caller's registry) *is* the promotion machinery. So the real inline animation lands **with
     increment 4**, not before it; 3a stands as the clean, complete standalone slice.
   - **3c — Queue-on-contention (co-lands with increment 4).** Replacing `-EAGAIN` with queue+park
     is **unsafe for same-thread re-entrant/cyclic calls without promotion** — the caller would
     park waiting for an instance it itself holds (self-deadlock). It therefore lands *with* the
     promotion machinery (increment 4), which reifies a parked handler and frees the thread; until
     then a busy instance keeps answering the probeable `-EAGAIN`.
4. **Inline animation + promotion (interp)** — the unified-model core, absorbing 3b and 3c. Run
   the offer handler as a **reified fiber in the caller's (shared) scheduler** over the provider's
   world (the §4 inline-animation fast path), so a handler that parks mid-animation files its
   reified fiber + waiter and the caller parks on the reply (`handler_parks` re-plumbed) — which in
   turn makes queue-on-contention (3c) safe for re-entrant/cyclic calls (a parked holder frees the
   thread). The cross-domain fiber (different `mem`/`host`, the caller's registry) is the piece 3a
   deliberately stopped short of. Fuel is reproduced exactly against 3a's `drive_arc` behavior.
   Decomposed into slices, smallest verifiable step first; each slice differentially pinned as
   noted. Bindings stay split (`Binding::Offer` vs `Binding::LiveImpl`) through this increment —
   their merge onto one `Offer` binding rides increment 6 with the `drive_arc`/`ProviderState`
   retirement, so 4 re-plumbs the library-provider path without touching the process-provider one.
   - **4a — Cross-world reified animation, run-to-completion. DONE (2026-07-31, PR #572).** Replace
     the isolated `drive_arc` sub-scheduler (which by construction never parks back into the outer
     scheduler) with animating the offer handler as a **reified fiber in the caller's own
     `FiberRegistry`** over the provider's world, for all four instanced-offer dispatch arms
     (`cap.call`/`call.import`/`call.sym`/`call.import.dyn`, via one loop-scoped
     `animate_instanced_offer!`). For a handler that runs to completion this is **byte-identical to
     3a**: the same two `cap`-slot translation edges, the same `OFFER_FUEL`/`ProviderState::fuel`
     drain, the same results. This is the inline animation of old 3b, now landed in the shared
     scheduler where a park can be filed. The one genuinely new primitive: a fiber in the caller's
     registry whose world differs from the registry's owner — the security-sensitive seam of the
     increment, and the unit the confinement fuzzer (AGENTS.md) extends to. **As built**, three
     refinements the design under-specified: (i) the cross-world switch swaps the **full execution
     context** — not just `mem`/`host` but **`fuel`** (to the provider budget, so provider-pays
     drains identically — the handler runs on the caller's loop) and **code** (the handler runs
     `entry.funcs` through the existing invoke seam as `INVOKE_MODULE`); (ii) admission moves from
     3a's held `try_lock` to a `ProviderState.busy` word (the guard cannot span the handler's loop
     iterations), a busy instance still answering `-EAGAIN`; (iii) the switch is non-durable — a
     **durable caller/provider** or a **`ref.func` handler** declines to the unchanged `drive_arc`
     sub-run (fail-closed, §9), and `shadow_switch` never runs on the animation. The switch mirrors
     `serve_switch`; the settle rides the `Terminator::Return` fiber-exit keyed on
     `offer_anim.handler_slot`. A mid-animation park is a `FiberFault` until 4b. Pin: existing
     `impl_wiring`/`imports_impl` (incl. the all-backends oracle + the narrowed-lock two-vCPU
     concurrency test) unchanged, the animated path confirmed taken.
   - **4b — Promotion (handler parks mid-animation).** A handler that parks files its reified fiber
     + waiter and the caller parks on the reply — `handler_parks` re-plumbed to the library-provider
     case, **reusing** the built waiter table (`ticket_waiters` keyed `(callee, ticket)`, I49) and
     reply plumbing (`cap_reply_or_stash`, `Waiter::Fiber`) rather than growing a parallel one
     (INVARIANTS.md 1/4). A promoted handler converges onto the exact shape the process-provider
     path already runs (`serve_run`/`handler_parks`): a handler fiber `ParkedOn` in the caller's
     registry, its caller parked on `(provider, ticket)`, its return replying via
     `cap_reply_or_stash`. Two deltas vs. that path: **(world)** the handler's world is the provider
     *instance's*, so at park the provider's `{mem, host, fuel}` are handed **back to the
     `ProviderState`** and re-acquired from it on resume — the parked fiber carries only its code +
     fuel-accounting + results + ticket, never `mem`/`host` by value, because a second caller may
     legally have animated the instance meanwhile; **(no serve loop)** the caller is re-plumbed onto
     `ticket_waiters` exactly like `Blocked::CapReply`, and the promoted settle recognizes a promoted
     slot and `cap_reply_or_stash`es the reply instead of pushing into a same-vCPU resumer. The
     `single` admission gate stops being a **held** `state.try_lock()` (which cannot survive a park)
     and becomes the §10.3 **admission word** — a flag that reopens (`busy = false`) when the handler
     parks, closing its run-to-park atomicity window (§10.1). The generation re-check on the phase-3
     relock now earns its place (3a explicitly deferred it because `drive_arc` had no suspension
     point). Two decisions, both fail-closed and oracle-matching: a **multi-result** offer whose
     handler parks declines to `drive_arc` (`cap_reply_or_stash` carries one `i64`, the oracle's
     reply width — no parallel table); a **resuming** handler has priority over new callers for the
     world (the process path's "woken parked handler resumes before new admissions"), a racing new
     caller getting the unchanged probeable `-EAGAIN` until 4c. Pin: a parking library-provider
     handler completes observably identical to the same handler behind a process provider (the
     `live_impl` path is the blocking-behavior oracle). Decomposed smallest-verifiable-first:
     - **4b.1 — Single-level promotion (the full vertical). DONE (2026-08-04).** A handler that
       parks on a blocking primitive (futex `wait`) promotes, is woken, resumes with its provider
       world re-acquired, and returns to the caller. **As built**, the resumer model is a refinement
       of the sketch above: nothing but a claimant can resume a `ParkedOn` fiber, and a passive
       library provider has no serve loop, so **the caller's own vCPU is the handler's resumer**. On
       promotion (a new branch in `fiber_park!`, taken when the parking fiber is
       `offer_anim.handler_slot`) the provider `{mem, host, fuel}` are drained back to the
       `ProviderState` and `busy` cleared, the caller's world is restored, an `OfferParked` record is
       filed, and the vCPU parks on a new `Blocked::OfferPark { key }` — filed in **`svc_waiters`
       keyed on the provider domain** (the `svc` the handler's block-waiter already targets), so the
       handler's block-wake (`wake_blocked` + `svc_wake`) re-admits it exactly as it re-admits a
       serve loop. (The `ticket_waiters`/`cap_reply_or_stash`/`Waiter::Fiber` reuse the sketch named
       is for the *nested* caller — a caller that is itself a fiber whose vCPU has moved on — which
       lands in 4b.2; the top-level caller needs no ticket.) On resume the vCPU re-`claim`s the
       woken handler (`Claimed::LiveWoken`), **re-acquires the world from the live instance** (not
       carried by value — a second caller may have animated it meanwhile; `busy` re-set, the
       generation re-check), rebuilds an active `offer_anim`, and switches back in; the handler's
       eventual return rides the **unchanged 4a settle** (the resumed segment is an ordinary active
       animation), with the per-segment reserve drains telescoping to `initial - final`. A
       lost-wakeup recheck on the park (the I52 shape) and a fail-closed `FiberFault` if a handler
       leaked the provider-host `Arc` (spawned) round it out. Pinned by `offer_promotion.rs`: a
       timed-wait handler parks and resumes on its timer returning `WAIT_TIMED_OUT`, and a second
       dispatch is admitted after the first promoted (admission reopened); the full `impl_wiring` /
       `imports_impl` / `svc_handler_parks` / `svc_serve_chain` suites unregressed.
     - **4b.2 — Nested / cross-domain promotion. DONE (2026-08-04).** An offer handler that is
       *itself* a caller — the §1 `A → B → A → B` chain across distinct instances — now nests: the
       single `OfferAnim` slot became a **stack** (`Vec<OfferAnim>`), pushed on each animation
       switch and popped (strictly LIFO) on each settle/promotion. Under the coupled resumer model
       (4b.1) the whole chain stays on **one vCPU**, so the `ticket_waiters`/`cap_reply_or_stash`
       reuse the original sketch reserved for a "caller whose vCPU moved on" turned out
       **unnecessary** — there is no such caller; a nested promotion parks the one vCPU carrying the
       whole chain, and each enclosing handler is a suspended resumer on the stack. Enabling nesting
       surfaced a **latent 4a cache bug**: both handler frames carry `module = INVOKE_MODULE` but
       point at different `invoked` tables, and the `cur_funcs` cache keyed on the module *id* did
       not refresh on an `INVOKE_MODULE → INVOKE_MODULE` switch — so a nested handler ran against its
       *caller's* function table. Fixed by invalidating the cache (`cur_module = u32::MAX`) at each
       animation switch and settle (an `invoked` change under an unchanged id). Pinned by
       `offer_promotion.rs`: a non-parking nested offer settles under the outer animation (the
       cache-bug regression), and an inner handler promotes+resumes while the outer animation stays
       on the stack. *Self-instance* re-entrancy (a true cycle back to a **busy** instance) stays a
       probeable `-EAGAIN` until 4c.
     - **4b.3 — Teardown hardening. DONE (2026-08-04).** Two teardown gaps the coupled resumer
       model (4b.1) opened, both closed: **(i)** a reaped caller that was *mid-animation* held a
       provider instance checked out (`busy = true`, its world on the dying vCPU) — `reap` now
       reopens `busy` on each `offer_anim` state, so a **reused** `Host` never sees a
       permanently-busy instance (fail-closed to `-EAGAIN`, never a wrong answer; a
       promoted-but-parked handler already handed its world back, so only the active stack needs
       clearing); **(ii)** an `OfferPark` caller parks in `svc_waiters` under the **provider**
       domain's key, not its own, so `teardown_domain` (which removed only `svc_waiters[dying_key]`)
       would strand a dying domain's caller under a live provider's key — the sweep now scans every
       `svc_waiters` queue for member vCPUs by identity (mirroring the `ticket_waiters` sweep).
       `teardown_run` already drained all `svc_waiters`, so the root-exit path was covered; a durable
       provider keeps declining to `drive_arc` (fail-closed, unchanged from 4a). Pinned by
       `offer_promotion.rs`: the root abandons a promoted daemon on exit and the run ends promptly,
       never awaiting the daemon's infinite block. **Deferred:** the §10.3 **closed** bit (freeze +
       quiesce) has no library-path trigger yet — a library provider is non-durable (declines
       freeze) and teardown is handled by domain-death + the `reap` reset above — so adding a
       `closed` word now would be speculative machinery (prime directive); it rides the
       quiesce/process-provider work where a serve loop actually freezes mid-handler.
   - **4c — Queue-on-contention (old 3c).** With a parked holder freeing the thread, the busy
     `single` path enqueues + parks instead of answering `-EAGAIN`; a re-entrant/cyclic call
     completes instead of self-deadlocking. The bounded queue still refuses `-EAGAIN` at the rim when
     full (fail-closed, §9). Pin: the increment-2 test that observed `-EAGAIN` on a busy /
     self-granted instance flips to observing completion; queue-full still answers `-EAGAIN`.
     Decomposed by **who** holds the instance busy — the two cases have genuinely different
     mechanisms:
     - **4c.1 — Contention between distinct vCPUs (park + wake-to-retry). DONE (2026-08-04).** The
       busy holder is a *different* vCPU (two threads of a domain contending on one `single`
       instance). The loser, in the `animate_instanced_offer!` busy arm, **rewinds its offer op and
       parks** as an admission-waiter (a new `Blocked::OfferAdmit { key }`) instead of pushing
       `-EAGAIN`; when the holder clears `busy` (the 4a settle, a 4b promotion-park, or the
       fiber-exhaustion undo), an `admit_wake` re-admits it and the rewound op re-attempts (winning,
       or re-parking on a lost race — the ordinary futex-retry shape). **As built**, admission-waiters
       live in a **dedicated `Sched::admit_waiters` map keyed by the `ProviderState` pointer**, not
       `svc_waiters`: a busy instance has its world *checked out on the holder*, so its domain id is
       unreadable and the two wake kinds (promoted-handler resumers vs. would-be starters) never
       cross. The bound is a per-`ProviderState` `admit_parked` count, checked under the state lock
       the busy decision already holds (rim → `-EAGAIN`, fail-closed); it is decremented on the
       re-attempt (or at `reap`) via a per-vCPU `admit_retry` handle carried across the park. A
       lost-wakeup recheck in the park handler (the I52 shape) re-reads `busy` under the scheduler
       lock and re-admits if the instance already freed. Leaves the self-call case (instance on this
       vCPU's own `offer_anim` stack) untouched — that is 4c.2. Pinned by
       `concurrent_callers_park_and_retry_instead_of_eagain` (two single-shot callers both observe the
       real result, never `-EAGAIN`) and the unchanged
       `concurrent_instanced_offer_calls_are_safe_under_the_narrowed_lock`.
     - **4c.2 — Re-entrant / cyclic completion. DONE (2026-08-04).** The busy holder is *this vCPU's
       own* animation of the instance (the self-call; `offer_anim` already carries its `state`).
       Parking to wait for a distinct holder would self-deadlock — there is none. **As built**, the
       **direct** self-call (the instance is the *top* of the animation stack, so its world is already
       installed on the vCPU) turned out far simpler than the sketched baton-pass: it is not a
       cross-world dispatch at all, just an **ordinary recursive call** over the provider's own world
       (a frame push to the op's handler, no checkout, no `busy` change, no `cap` translation — same
       domain). Bounded recursion completes; unbounded hits `OutOfFuel`/`StackOverflow` and traps,
       fail-closed — never the old `-EAGAIN`. The **general buried cycle** (`A → B → A` where the
       re-entrant `A` is called from a deeper `B`, so `A`'s world sits in the stack's saved state, not
       the instance) is detected (instance on the stack but not the top) and still answers `-EAGAIN` —
       the harder residue, a later step. Pinned by `a_self_reentrant_offer_completes_instead_of_eagain`
       (a depth-3 self-recursive offer returns 3). The legacy `cap_dispatch_slots`/`drive_arc` cyclic
       path (`a_cyclic_offer_call_is_a_probeable_eagain_not_a_deadlock`) is unchanged — it holds the
       state guard across its sub-run, so its inner self-call still reads `-EAGAIN`; that path retires
       with `drive_arc` in increment 6.
   - **4d — Direct handoff (§10.2 arm 4, delta §10.7).** A `single` process provider parked at
     `svc.wait` is served by **direct handoff**: the caller claims the serve activation and animates
     the handler on its own thread (no enqueue, no worker wake, no reply round-trip), a mid-handoff
     park riding 4b's machinery. Settlement (§10.2): the handoff counts in the callee's serve
     accounting, the parked `svc.wait` completing with the same `serve_count` observation the enqueue
     path would have delivered. Pin: **handoff-on ≡ handoff-off** on observable results and
     `serve_count`.

     **As mapped (2026-08-04)** the process-provider path is a `Binding::LiveImpl` — a §14-child
     serve loop parked at `Blocked::SvcWait`, served today by `svc_enqueue` + `svc_wake(callee)` +
     caller park on `Blocked::CapReply { ticket }`; the provider's serve loop pops the queue,
     `serve_switch`es the handler, and settles (`serve_count += 1`, `cap_reply_or_stash(ticket)`
     wakes the caller). Crucially, unlike a library provider (whose world is a detachable
     `ProviderState` the caller checks out onto its *own* vCPU), a process provider's world lives **on
     its own parked vCPU** — so handoff is a cross-**domain** switch (the Doors/L4 shape), not the
     cross-world switch 4a/4b/4c animate. And no `handoff` toggle or `handoff-on ≡ handoff-off`
     harness exists yet — both are net-new. Decomposed smallest-verifiable-first:
     - **4d.0 — Transport toggle + differential harness. DONE (2026-08-04).** A `handoff` switch
       (`Host::set_handoff`, default **off** ⇒ today's enqueue+park, byte-identical), copied at
       `drive_arc` into a run-global `Scheduler::handoff` (read, never written — no lock), plus a
       `direct_handoff.rs` harness driving a process-provider call both ways and asserting identical
       results. De-risks the behavioral slices: any divergence from the enqueue oracle trips the pin.
     - **4d.1 — Run-to-completion handoff. DONE (2026-08-04).** Built as **(A)**, and simpler than the
       sketch: the caller enqueues the dispatch **once** (the existing `svc_enqueue`, ticket `t`),
       then — if handoff is on and the provider's own serve loop is parked at `svc.wait`
       (`Scheduler::take_parked_serve_loop`, identified by its own domain id in `svc_waiters`) —
       **donates its thread** by calling the worker's own `dispatch()` on that vCPU. `dispatch()` runs
       the serve loop to its next park/finish and re-files it, running `serve_switch`/settle/
       `serve_count` **unchanged**; the reply, with no ticket-waiter registered yet, stashes in
       `svc_results`, which the caller drains and returns. Any non-served outcome falls through to the
       **unchanged** enqueue+`svc_wake`+park path on the same ticket `t` — so ineligibility (provider
       busy / not parked / multi-consumer) and a full queue both stay byte-identical. Pinned by
       `direct_handoff_matches_enqueue_park_run_to_completion` (verified the handoff arm actually
       engages on the second call, then equivalence to handoff-off).
     - **4d.2 — Mid-handoff park. DONE (2026-08-04).** Falls out of 4d.1's structure for free: if the
       handoff-served handler parks mid-run, the serve loop's existing `handler_parks` machinery files
       it and re-parks the loop, so `dispatch()` returns with **no reply stashed** — the caller sees
       `served == None` and falls through to park on ticket `t` exactly as the enqueue path would; the
       handler resumes later (its own block-wake) and replies via `t`. Pinned by
       `direct_handoff_matches_enqueue_park_with_a_parking_handler` (a timed-wait handler; handoff-on ≡
       handoff-off). *Not yet done:* handoff for a **timed** serve loop (a stale `svc_timer` after a
       re-park) is conservatively avoided by the fall-through only implicitly — infinite-wait serve
       loops (the common shape) are the tested path; a timed-serve-loop handoff gate is a later
       refinement if a consumer needs it.
5. **JIT arm** — the thunk fast path; park = thread-block on the reply; **caller-pays fuel lands
   here**, uniformly on all backends. Closes the `live_impl` parity gap. Decomposed
   smallest-verifiable-first:
   - **5a — `fuel.remaining` (self-op 13), all backends. DONE (2026-08-04).** §10.6 — an
     authority-neutral readout of the domain's remaining fuel. **Interp** services it in the eval loop
     (pushes `*fuel`, charging none of its own). **JIT** lowers it inline in `lower_block` to a
     `load(I64, fuel_addr)` — the same host-owned cell `emit_fuel_check` charges — so it returns the
     identical value under the differential oracle (which arms counted fuel on the JIT); when a
     compile has no counted fuel armed (`fuel_addr == 0`, the production CLI path) it reports
     `i64::MAX` ("unmetered"). **Bytecode** can't see the vCPU counter from host-side dispatch, so it
     declines op 13 (`compile_module → None`) and the module falls back to the tree-walker, which
     services it. The design reserved op "12"; `reap` took it, so this is **13**. Pinned by
     `crates/svm/tests/fuel_remaining.rs`: interp ≡ JIT on the raw readback (`budget − 1` at entry)
     and on the read-before/read-after **call-metering** idiom.
   - **5b — caller-pays fuel across the crossing. DONE (2026-08-04).** §10.5 — an animated offer
     handler now runs on the **caller's own `*fuel`** counter (it made the call, it pays), not the
     provider reserve. **As built** this simplified the 4a/4b machinery rather than adding to it: the
     checkout stops swapping `*fuel` to a provider budget (`budget_ = *fuel`, the caller's whole
     remaining fuel — no reserve draw, no per-call `OFFER_FUEL` cap, that cap being the deferred
     §10.5 refinement), charging only the one function-entry fuel; both settles stop restoring
     `*fuel` and stop draining `st.fuel` (the reserve is left untouched — vestigial until increment
     6); and `OfferAnim` sheds its now-dead `saved_fuel`/`budget` fields (the promotion telescoping
     collapses — the caller's counter simply persists across the park at `remaining_budget`). A
     caller with no fuel to fund the crossing traps `OutOfFuel` **before** any checkout
     (`Adm::OutOfFuel`, nothing to undo). Pinned by `crates/svm-interp/tests/caller_pays.rs`: a
     caller's `fuel.remaining` drops by the handler's ~100-back-edge loop (≈0 under the old
     provider-pays). *Scope:* the durable/`ref.func` `drive_arc` fallback still reserves-and-drains
     (provider-pays) — it retires with increment 6; a process provider runs on its own §15 budget by
     construction (a separate domain), so caller-pays here is the inline/library-animation path.
   - **5c — JIT cross-domain (`live_impl`) call arm + crossing-depth bound.** Closes the `live_impl`
     parity gap (today the JIT force-folds serving-with-park modules to `TreeWalk`). **Design history
     (2026-08-04, owner + verification):** two wrong sketches preceded the aligned one. Sketch 1
     (bolt the JIT thread onto the interp `Scheduler`, callee on an interp vCPU) — dead: a JIT run has
     no `Scheduler` in scope, and an interp-vCPU callee is a second execution model per crossing,
     exactly what the CONSOLIDATION.md §0 yardstick calls expensive. Sketch 2 (make the
     `os_thread_rt` **enqueue → futex-wake → condvar-block** protocol *the* transport) — misreads this
     document: that protocol is §10.2 **arm 5**, the *parked fallback transport*, not the arm. The
     original design here is explicit and stands: §8.5 "the **thunk fast path**; park = thread-block
     on the reply"; §10.4 "JIT crossings are **real native frames through thunks**… at the bound the
     call **declines the inline arm and takes the parked transport**"; §10.2 arm 6 "the caller parks
     (interp) or **thread-blocks (JIT)**" — thread-block is the *park half*, never the primary
     transport. CONSOLIDATION.md §2 depends on this ordering: the coroutine-collapse's fault-service
     path stays fast *because* it is the direct-handoff shape — a per-call futex/thread round-trip
     would fail its lazy-paging-latency gate.
     **Verified feasibility of the fast path (all in-tree today):** the JIT's natural CLIF ABI
     threads `(mem_base, fn_table_base, …)` per call and the masking lowering adds a **runtime**
     `base` (nothing window-specific is baked into code — `svm-jit/lib.rs:57,810,1090,2568`; D65
     measured the threaded ABI as not a perf liability); `svm-run::serve_native` already has the thunk
     **synchronously invoking compiled handler trampolines on the calling thread**
     (`svc_handler(export,op) → handler_tramp(fidx) → CompiledModule::invoke_extra` over `mem_base`) —
     same-domain today, but it is the exact inline-handoff mechanism; §14 children each carry their own
     compile (`ChildCode`) and granted children their own ctx (`instantiator_rt`). The cross-domain
     fast path = that mechanism pointed at the **callee's** compile + sub-window + Host, with
     admission gating and the two cap-translation edges — the JIT analog of the interp's 4a/4d.
     Decomposed (aligned; smallest-verifiable-first, mirroring the interp's own 2→3→4 staging):
     - **5c.0 — prerequisite: JIT child/offer registry + `child_offer` (op 14). DONE (2026-08-04).**
       Op 14 emitted a blanket `-EINVAL` on the JIT ("the JIT runtime has neither [scheduler nor
       child registry]") — no live-impl handle could even be minted. **As built**, the registry
       reduced to an **ownership flip plus one retained ref**: a granted child's powerbox is now an
       `Arc<Mutex<Host>>` shared cell (was an exclusively-owned `Box<Host>`) with two counted refs —
       the child thread's (released at child exit, as before) and a **nursery-retained** one
       (`Child.retained`, released at `join_children`; spawn-error paths release both). Because the
       parent can now reach the same powerbox, granted-child compiles run against the **lock-taking
       thunk** (`GrantChildHooks.thunk` = `cap_thunk_locked`) — the data-race guard the sharing
       demands. The mint itself is `Host::mint_child_offer` (the interp op-14 arm minus thread-slot
       resolution; same lock discipline, `callee_slot: None` ⇒ non-durable), reached via a new
       `ChildOfferMint` hook from a tiny op-14 thunk (`instantiator_rt::child_offer`: joined child /
       plain child / no hook ⇒ `-EINVAL`, errno-for-errno with the interp). Shape resolution works
       because `spawn_granted_child`/`spawn_named_child` now **seed the child's `self_module`** from
       the holder's (the interp arm re-assigns the same Arc — uniform across backends). Pinned by
       `crates/svm/tests/jit_child_offer.rs`: op 14 mints ≥ 0 on the real JIT backend (+ interp mint
       parity); a call **through** the minted handle answers the host dispatch's probeable `-EINVAL`
       until the 5c.1 transport (documented per backend — the equality pin is 5c.1's flip); op 14 on
       a plain (op 0) child refuses `-EINVAL` fail-closed (nothing shared, nothing offered).
     - **5c.1 — the parked transport (arm 5 / the decline target):** enqueue onto the callee's
       `svc_queue`, futex-wake its serve loop, caller **thread-blocks** on a reply cell (the
       `instantiator_rt` `join`/`ChildDone` park shape, with the `epoch_addr` re-check so a host
       interrupt still unwinds). Relax the `run_with_caps` force-fold (`has_instantiate` veto term).
       Pin: `svc_parity.rs`'s "Jit" arm — silently TreeWalk today — runs the **real** JIT backend and
       matches. This is the correctness baseline the fast path declines onto, built first exactly as
       the interp built enqueue+park (§3.6) before inline animation.
       **As mapped (2026-08-04), six blockers, ranked — the slice re-decomposes:** (1) *children
       have no handler trampolines*: `compile_child` yields `ChildCode` (entry trampoline + fn
       table only), while `serve_native` needs a `CompiledModule`'s `serve_tramps`/`handler_tramp`
       — the dominant cost; (2) `serve_native_ctx` is only ever set on the top-level run's Host,
       never a child's; (3) the production Jit path (`powerbox_compile_run`) compiles without
       `GrantChildHooks` at all (tests wire them; `Instance::run(Jit)` can't even spawn); (4) no
       epoch cell is reachable from inside the cap thunk (`epoch_addr` lives on the `Nursery`,
       baked into code; the thunk has only `trap_out`), so the blocked wait's kill re-check needs a
       new channel; (5) `Host` has no `Condvar` and `Condvar::wait` needs the guard's own mutex —
       the 5c.0 shared cell must become `Arc<(Mutex<Host>, Condvar)>`; (6) JIT op-13 children never
       get `self_module` (op-11/op-8 inherit it via `spawn_granted_child`), so their `svc_handler`
       misses. Also: the `svc_parity` pin module spawns via **op 0** (an *unshared* child) and exits
       via `call.import` (a second `svc_park_veto` term beyond `has_instantiate`) — the pin needs
       op 0 to mint shared powerboxes too, or the module moves to op 11 and a `cap.call` exit.
       Sub-slices:
       - **5c.1a — serve trampolines for children. DONE (2026-08-04).** `compile_child` builds a
         buffer-ABI serve trampoline per impl-export handler (the `CompiledModule::compile`
         `serve_ids` block, child twin) onto `ChildCode.serve_tramps`; the handler set
         (`Nursery.serve_handlers`) comes from `m.impl_exports` at root-nursery construction (no
         external plumbing; co-fiber/durable/plain children pass empty). Opaque consumption API
         (`child_handler_tramp` / `child_invoke_handler`) — the fault range comes free from the
         thunk's own `mem_base`/`mem_size` (the serve arm runs inside the child's in-flight guarded
         entry call). `Host.child_serve_ctx` is the registration slot, distinct from
         `serve_native_ctx` by design.
       - **5c.1b — the blocking transport. DONE (2026-08-04).** **As built**, no cell-type change:
         the `Condvar` lives *inside* `Host` (`svc_cv: Option<Arc<Condvar>>`, armed by the granted
         builders), cloned out under the cell's lock and always paired with that one mutex —
         `Condvar::wait(guard)` is sound and the 5c.0 `Arc<Mutex<Host>>` ABI is untouched.
         `svc_enqueue`/`svc_settle` notify; `Host.epoch_cell` mirrors the kill cell for
         thunk-blocked re-checks (20ms `wait_timeout` + trap-cell poll, the join-park discipline).
         Child side: `serve_locked_child` replaces the locked thunk's `-EINVAL` stub — pops via the
         registered serve ctx, **drops the guard around every handler invoke**, settles+notifies;
         an empty-queue `svc.wait` block-waits (non-child locked domains keep `-EINVAL`; timed form
         stays `-EINVAL`; a *nested* serve under a handler is a recorded residue). Caller side:
         `live_impl_call` in the unlocked thunk — enqueue (full queue ⇒ `-EAGAIN`), thread-block on
         the ticket; dead callee (release cleared the serve ctx with the dispatch unserved) ⇒
         `CAP_REVOKED`, the D37 shape. A **locked-domain caller** (child→sibling) refuses `-EINVAL`
         before the guard-holding delegate — the A↔B cyclic-call deadlock is structurally excluded
         until a later slice unlocks that tier. Registration: `GrantChildHooks.register_serve` at
         spawn; the releaser clears the ctx + notifies (idempotent across the two refs). Pinned by
         `jit_child_offer::call_through_minted_offer_completes_on_both_backends` — **the 5c.0
         equality flip: 42 on the real JIT ≡ interp.**
       - **5c.1c — production wiring + the parity pin. DONE (2026-08-04).** `powerbox_compile_run`
         installs the production `GrantChildHooks` (via a new `CompiledModule::set_grant_child_hooks`
         bridge) and arms `Host.epoch_cell` from the run's interrupt cell, on both branches. The
         serve fold relaxes **Jit-only and shape-scoped**: a serving module that also **nests**
         (`module_nests`) runs the real JIT (its serve points live in granted children — the 5c.1
         transport); top-level-serving non-nesting modules still fold. `svc_parity`'s module moves
         op 0 → op 11 (empty grant list — a plain child is destitute by design on the JIT; a serving
         child needs the shared powerbox) and now passes on the **real** JIT backend ≡ TreeWalk ≡
         bytecode-fallback: spawn → mint → parked-transport call → blocking `svc.wait` serve →
         reply → join → exit 42. *Residue:* op-13 (separate-module) children get no `self_module`
         on the JIT yet, so their offers don't mint — rides a later slice.
     - **5c.2 — the thunk fast path (arm 4, the §8.5 headline). DONE (2026-08-04).** **As built:**
       the child's serve loop, entering its empty-queue wait, **publishes the activation**
       (`Host.serve_activation = (serve_ctx, mem_base, mem_size)`) for exactly the wait's duration —
       the window stays alive precisely because that frame sits in `wait_timeout` until release. A
       caller (gated by the run's `Host::set_handoff` knob — the 4d toggle now spans both tiers)
       **claims it atomically with the check** under the cell's lock (`try_claim_handoff`) and
       invokes the handler inline via the 5c.1a `child_invoke_handler` — no enqueue, no wake, no
       reply round-trip. While claimed the child neither pops, exits, nor honors interrupts (the
       claimer runs over its window). **Settlement (§10.2):** the claimer's `release_handoff(1)`
       lands in `Host.handoff_served`, which the child folds into its `svc.wait` return — the callee
       observes the same served count either transport. **Trap parity:** a handler trap/fault under
       handoff is captured in a *local* cell, routed to the child (`handoff_trap` — it folds and
       dies with it on wake, as if it had served the dispatch itself) and the caller answers
       `CAP_REVOKED` — the enqueue path's exact observables, never the caller's death. Every miss
       (no activation, second claimer, unknown handler, arity, depth rim) releases and declines to
       the 5c.1 parked transport. Pinned by
       `jit_child_offer::direct_handoff_matches_parked_and_settles` (`call*100 + join(served)` =
       4201 on interp ≡ JIT-parked ≡ JIT-handoff; claim engagement verified non-vacuous, 2/2 per
       run).
     - **5c.3 — §10.4 crossing-depth bound. DONE (2026-08-04).** A per-thread counter
       (`CROSSING_DEPTH`, rim 64) at the claim gate; at the bound the call declines to the parked
       transport (fail-closed toward the slower correct transport). Structurally depth cannot
       exceed 1 today — a locked-domain (child) caller refuses live-impl calls before its
       guard-holding delegate — so the bound is the §10.4 contract for when that tier unlocks.
     - **5c.4 — mid-handoff park parity (arm 6). DONE (2026-08-04).** On the JIT this needed **no
       promotion machinery**: a handler that parks under handoff simply blocks the **claimer's**
       thread inside the inline invoke (§10.2 arm 6's "thread-blocks (JIT)" — the caller waiting is
       the semantics), the claim holding admission closed throughout (run-to-park atomicity for
       `single`, §10.1). Pinned by `direct_handoff_with_parking_handler_matches` (a timed-futex
       handler; interp ≡ JIT-parked ≡ JIT-handoff).
     **Recorded 5c residues** (each fail-closed probeable today, each a deliberate later slice):
     the locked-domain caller tier (child→sibling live calls answer `-EINVAL` — unlocking it is
     what makes the 5c.3 bound bite); nested serve under a handler; the timed `svc.wait` form on
     the JIT tier; op-13 separate-module children as offer targets (need their module's
     `self_module` + handler set threaded through `ResolvedModule`). The fast path reuses the proven
     `serve_native` invoke mechanism rather than inventing a cross-thread protocol as the primary —
     one execution model, per the consolidation yardstick.
6. **Retire the two-lock sub-run** — with 3–5 landed, the passive-provider `drive_arc` nested
   executor and `ProviderState` collapse onto the inline-animation path (the original increment-2
   goal, now reachable without a parity regression). **Mapped + decomposed (2026-08-04).** The
   retirement surface splits into what the eval loop can shed now and what other tiers still need:
   `drive_instanced_offer` (the eval-loop legacy fn) has exactly 4 call sites, every one already
   fronted by `animate_instanced_offer!`; but the `cap_dispatch_slots` instanced arm — the true
   two-lock executor — is reached by the **bytecode engine, the JIT thunk's generic dispatch, the
   IoRing submit paths, and any embedder holding `&mut Host`**, none of which have an eval loop to
   animate in (the exact trap increment 2 was re-sequenced to avoid; `imports_impl` runs it on all
   three backends today). Decliner census: **D1** durable *caller* — real (the animation
   deliberately skips `shadow_switch`, and a provider window has no `DURABLE_RESERVE`), but pinned
   by **no test**; **D2** `ref.func` handlers — a small fix (`install_unit_funcs` + the
   `invoked_ref_slots` remap the animation currently takes away); **D3** durable *provider* —
   **unreachable dead code** (no API can ever set `durable` on a `ProviderState.host`). Notes:
   `GUEST_IMPL_FUEL` exists only in prose (the code's reserve is `PROVIDER_FUEL_RESERVE`);
   `wire_impl_instance`/`impl_service` are already gone; the `GuestImpl`→`Offer` rename landed with
   §8.1. Slices:
   - **6a — the eval loop sheds the legacy fn. DONE (2026-08-04).** D2 animates: the switch-in
     installs the unit-own funcref remap (`install_unit_funcs`) and sets `invoked_ref_slots` — the
     `invoked_new` shape — with a **per-vCPU cache** so repeat animations reuse the first install;
     the promoted resume reinstalls from the cache (a miss is a fail-closed `FiberFault`). **As
     built, two table facts surfaced:** default run tables have **no free slots** (the reserve is a
     max, not a sum), so `drive_arc` at run start now reserves `Host::offer_table_demand()` —
     headroom for each distinct `ref.func`-taking offer unit (zero ⇒ byte-identical table); and
     `install_unit_funcs` gained **dedup by unit identity** (recovering slots by module-id scan),
     fixing a latent leak that predates 6a — repeat `invoke`s of one unit each burned fresh slots.
     A genuinely full table still answers `-EAGAIN` (the checkout-undo shape). D3 deleted
     (unreachable, with the `Adm::Decline` variant); D4/D5 (unknown op / arity) answer the retired
     path's `CapFault` at the call sites; D1 (durable caller) skips the probe and takes the
     host-side dispatch — one legacy site instead of two. **`drive_instanced_offer` is deleted.**
     Pin: `a_ref_func_offer_handler_animates_with_the_unit_remap` (handler `call_indirect`s its own
     helper through `ref.func`, called twice — the cache path; the first coverage that path ever
     had).
   - **6b — provider-pays leaves. DONE (2026-08-04).** `ProviderState.fuel`,
     `PROVIDER_FUEL_RESERVE`, and the `impl_fuel_remaining`/`set_impl_fuel_reserve` API are
     deleted; the host-side instanced dispatch runs on the **pure arm's flat deterministic
     `OFFER_FUEL` budget** (no reserve, no dry-check, no drain — a service serves indefinitely,
     each call identically priced; a runaway handler still traps `OutOfFuel` at the flat cap).
     `provider_pays_from_a_drainable_reserve` flipped to
     `an_instanced_offer_runs_on_a_flat_deterministic_budget`; `grant_impl_cap` is now the one
     blocking `state.lock()` accessor (its caveat narrowed accordingly — it dissolves with the 6d
     binding merge).
   - **6c — the two-lock discipline leaves. DONE (2026-08-04).** The host-side instanced arm
     adopts the checkout shape the animation proved: the world is checked out under a **brief**
     guard with the §10.1 `busy` admission word, the nested `drive_arc` runs with **no provider
     guard held** (world carried on locals), and a single check-in restores the world + reopens the
     instance on every exit — trap or success alike (an un-restored trap would strand the instance
     `busy` with an empty world). A cyclic host-side call still answers `-EAGAIN`, but via the
     admission word, not a held `try_lock`, so the held-locks deadlock argument (and the acyclicity
     rule) genuinely dies. **As built, one subtlety surfaced that the map missed:** releasing the
     host-side lock makes a foreign `busy` holder *park-observable* to an eval-loop caller for the
     first time — pre-6c, cross-run contention always saw a **held** `try_lock` (`WouldBlock →
     -EAGAIN`), so a 4c.1 admission-park only ever happened *within one run* (a peer vCPU the
     holder's own scheduler will `admit_wake`). A naive busy-word release would let a cyclic
     host-side self-call **park** in the nested `drive_arc`'s scheduler (its `offer_anim` is empty,
     so the 4c.2 self-detect misses) and hang — the outer arm can't clear `busy` until that nested
     run returns. Fix: an **owning-run token** — `ProviderState.busy_owner` records the holder's
     scheduler identity (`SchedRef::run_id`, the `Arc` address; `0` for this host-side tier). A
     caller may only *park* on a busy instance owned by **its own run**; a foreign owner (`0`, or
     another run sharing the instance via regrant) answers a probeable `-EAGAIN`. Inert on the
     pre-6c behavior (no cross-run park existed) and exactly the `-11` the held lock gave — no
     waker, no `drive_arc` run-wiring. (`a_cyclic_offer_call_is_a_probeable_eagain_not_a_deadlock`
     keeps its `-11` on this tier; the cap-translation quartet and `imports_impl` — three backends
     — exercise the new checkout/check-in edges; the eval-loop flip already happened in 4c.2.)
     After 6c, every nested `drive_arc` left in the tree (pure arm + this narrowed arm) runs
     lock-free over its world — the sub-interpreter-under-locks is gone even where the sub-run
     survives for eval-loop-less tiers.
   - **6d — recorded residues** (the gap between §8.6 and §7's full deletion list). These are not
     quick slices — each is a sizable mini-increment closing one decliner or deleting one structure,
     and every one is sensitive TCB (the confinement/durability path, the JIT crossing, or the
     binding split). Decomposed and ordered smallest/most-independent first:
     - **6d.1 — durable-caller animation.** Close the D1 decliner: a durable caller currently skips
       the eval-loop probe (`if durable { None }`) and takes the host-side `drive_arc`, because the
       animation switch-in deliberately omits `shadow_switch` (it runs the handler over the
       *provider's* window while the caller's per-context shadow-SP bookkeeping lives in the
       *caller's* window). Animating a durable caller means threading the durable shadow-SP
       save/restore across the switch-in and the settle (the `serve_switch` durable choreography,
       adapted to the cross-*window* offer case) — a DURABILITY.md-scoped slice. **Unpinned today**,
       so it lands with its first pin (a durable caller invoking an instanced offer, differentially
       identical to the host-side path). Interp-only; deletes nothing structural, but removes the
       last reason the eval loop ever falls to the host-side arm for an instanced offer.
     - **6d.2 — tier-native JIT offer arm — DEFERRED (owner decision 2026-08-04).** We are
       **leaving the host-side interp arm in place as the shared offer path for every tier** and not
       building the native offer arms (6d.2/6d.3) now. Rationale: there are four execution tiers —
       interp (which *is* the host-side arm), bytecode (folds offers to interp), `svm-jit` (Cranelift,
       falls through `cap_thunk` to the host-side arm), and `svm-wasm-jit` (folds `cap.call`s to
       interp) — and **all four already reach an offer through the one correct host-side arm.**
       Retiring it means writing a native arm *per tier*, each against a **different confinement
       model** — `svm-jit`'s guard-page + baked fault-range (the hinge below), `svm-wasm-jit`'s
       base-as-param + baked size — i.e. N× hinge-sensitive work for a speedup **no benchmark has
       asked for**. Across four tiers the host-side arm is an asset (one correct path, not four), not
       debt. **The limitation we accept:** a `svm-jit`/`svm-wasm-jit` caller invoking an offer drops
       into the interp for the sub-run rather than running the handler natively (a dispatch-cost
       overhead on that specific crossing; correctness and confinement are unaffected — every tier
       gets the oracle's exact answer). **Revisit trigger:** a benchmark showing offer dispatch hot
       on a specific tier — then build a native arm for *that* tier only, from its own confinement
       model (prefer `svm-wasm-jit`'s base-as-param shape over `svm-jit`'s guard-page range if the
       choice is open). Critically, **6d.4's deletion does not depend on this** (see 6d.4). The full
       design is retained below for whoever picks it up.

       A JIT `cap.call` to an offer currently falls through
       `cap_thunk` to `host.cap_dispatch_slots` (the 6c-narrowed host-side arm, an interp
       `drive_arc` sub-run); this slice runs the offer handler natively (`define_extra` compiles the
       offer unit into the run's `CompiledModule`, cached; `invoke_extra` runs its trampoline) so
       the interp sub-run retires for the JIT tier. The offer is *not* the 5c child crossing — a
       passive offer has no thread/serve loop; it animates on the caller's thread, closer to
       `serve_native`'s `invoke_extra` than to `live_impl_call`. **Confinement finding (the hinge):**
       `invoke_extra` masks with the module's `live_fault_range` — the *in-flight run's* window
       bounds — while `mem_base` is a separate arg. So it can only run code over the run's **own**
       window. A **pure** offer is windowless (no masked access, any `mem_base`/range is inert); an
       **instanced** offer runs over the **provider's** window, whose bounds differ from the run's,
       so running its unit natively needs a way to invoke over a *foreign* window with **that
       window's** fault range — new confinement-masking plumbing, the security hinge (AGENTS.md; its
       own fuzz obligation). Split accordingly:
       - **6d.2a — no-memory pure-offer native arm (masking-safe scaffolding).** A new
         `offer_call` probe in `cap_thunk` (analogous to `live_impl_call`) resolves the handle to a
         `Binding::Offer`; it handles natively **only** a pure offer (`state=None`) whose unit
         **declares no memory** (and no threads/futex/fibers), else returns `None` to fall through
         to `cap_dispatch_slots` unchanged. No memory decl ⇒ the lowering emits **zero masked
         accesses** ⇒ the baked mask (`reserved−1`) and the fault range are never consulted, so the
         `mem_base` is inert and the native run is masking-safe over any base. Mechanism:
         `define_extra(entry.funcs)` into the run's `CompiledModule`, cached per `(domain, unit)` by
         `Arc::as_ptr`; `invoke_extra` the op's handler trampoline; `cap` slots translated at the
         two edges against an ephemeral `Host` (the interp pure arm's shape). **Fuel wrinkle (must
         resolve first):** the interp pure arm runs under a flat `OFFER_FUEL` (`drive_arc`), but
         `invoke_extra` shares the run's fuel counter (caller-pays) — so native parity needs a
         scoped `OFFER_FUEL` budget around the invoke (save/set/restore the counter, or a
         fuel-bearing invoke variant), or a looping pure offer would trap `OutOfFuel` at a different
         point than the interp. Differential-pinned (a no-memory pure offer, result **and** fuel
         parity, three backends). Retires only the pure/no-memory branch of the host-side arm.
       - **6d.2b — instanced-offer native arm over the provider window (the masking hinge).** Three
         sensitive sub-problems: **(A) foreign-window invoke** — a new
         `invoke_extra_window(cm, code, args, results, win_base, win_lo, win_hi, trap_out)` that
         masks/attributes to an **explicit** provider window instead of `self.live_fault_range`, so
         a masking-bug escape lands in the provider window's guard range (caught + unwound), never a
         crash; **(B) the baked-mask/reservation-match invariant** — the offer unit is compiled with
         the run's reservation `R` (baked mask `R−1`), so the provider window **must** be a
         reservation of size `R` or the arm **fails closed** (falls through) — a smaller provider
         window would let a masked access in `[prov_size, R)` hit its unmapped tail; this invariant
         is the whole confinement argument; **(C) window bridging** — `ProviderState.mem` is an
         interp `Mem`, a different representation from the JIT `GuestWindow`, so its underlying
         `svm-mem` reservation base/total must be exposed and asserted `== R` (both wrap `svm-mem`,
         so the raw reservation is compatible; `mem=None` degenerates to the 6d.2a inert case). Plus
         the 6c `busy` checkout (already on `ProviderState`), flat `OFFER_FUEL` (the 6d.2a wrinkle),
         and `cap` translation across the provider `Host`. **Mandatory masking-fuzz obligation**
         (AGENTS.md — the security hinge gets its own fuzz unit): a target that fuzzes access
         patterns in an offer unit run over a provider window of reservation `R` and asserts every
         access stays in `[0, R)` / faults into the provider guard, never the caller's window. Only
         once 6d.2a **and** 6d.2b land does the JIT's host-side/`-EINVAL` fall-through for offers
         leave §7's list. **Open question for the owner:** whether native offer animation earns this
         hinge plumbing + fuzz target, given the host-side arm is correct and no benchmark yet
         motivates it — 6d.2a is cheap but narrow; 6d.2b is the valuable, sensitive half.
     - **6d.3 — bytecode offer arm (fold-or-probe).** Bytecode declines offer ops to the tree-walk
       today, so instanced offers already run correctly via that fallback — this slice is the
       optional native arm (avoid the per-module fallback), lowest value, sequenced last of the
       tier work. May be folded into 6d.2 or skipped with a recorded note.
     - **6d.4 — the `Offer`/`LiveImpl` binding merge.** The endgame: fold `ProviderState`
       (`{mem, host, busy, admit_parked, busy_owner}`) onto the 5c shared-cell shape
       (`Arc<Mutex<Host>>` + window) so a library instance is structurally a granted child's
       powerbox, and unify `Binding::Offer`/`Binding::LiveImpl`. This is where "`ProviderState` +
       its mutex" and "the `GuestImpl`/`LiveImpl` binding split" finally leave §7's list. Deepest
       and most invasive (touches every state access + the binding dispatch). **Not gated on 6d.2**
       (an earlier note said it was — corrected): every tier reaches offers through the host-side
       arm, so 6d.4 rewrites that arm's *representation* and all four tiers keep working through it
       unchanged. The deletion is tier-independent, which is exactly why it is the right increment-6
       finale to build now while the native arms stay deferred.

       **Scope (settled by §7/§8's own words):** this is a *storage fold + binding-variant unify*,
       **not** a transport merge. The two transports genuinely differ — a `LiveImpl` is an active
       callee served by its own `svc.wait` loop over its own run's window (the 5c crossing:
       enqueue + park), while an instanced offer is a passive library animated on the *caller's*
       thread over a window carried in its state (4a). Making the offer grow a serve loop would
       *reverse* the 4a decision (animate-on-caller was chosen precisely to avoid a per-instance
       thread), so 6d.4 keeps both transports and merges only the **representation**: the offer's
       `{host, window}` becomes an `Arc<Mutex<Host>>` shared cell like a child's, the bespoke
       `ProviderState` struct + its mutex are deleted, and the two `Binding` variants collapse into
       one that carries an internal passive-vs-live discriminant (the transport chosen at call
       time, exactly as today). Decomposed smallest-first:
       - **6d.4.1 — the offer powerbox becomes a shared `Arc<Mutex<Host>>` cell. DONE.**
         `ProviderState.host: Host` → `Arc<Mutex<Host>>` (the window + the 6c admission word stay
         alongside in `ProviderState`, exactly as a child's window is separate from its powerbox
         cell). The animation now **clones** the cell onto the vCPU (deleting the per-call
         `Arc::new(Mutex::new(..))` at every checkout/resume) and its mutations land through the
         shared `Arc`, so the settle needs no host restore; the host-side `drive_arc` arm briefly
         locks the cell to take/restore the inner `Host` by value, preserving 6c's no-lock-across-
         the-sub-run. The world-handback **leak guard** (a handler that leaked the provider world by
         spawning a fiber that still holds it) is re-expressed from `Arc::try_unwrap`
         (unique-ownership) to `Arc::strong_count == 2` (the instance's own ref + the one checkout
         clone; `busy` serializes admission so nothing races the count) — fail-closed exactly as
         before, at both the 4a and 4d settles. Behaviour-identical; the existing offer suite is the
         oracle (`impl_wiring` 25, `offer_promotion` 8 incl. the 4a–4d settles, `imports_impl`
         three-backend).
       - **6d.4.2 / 6d.4.3 / 6d.4.4 — REJECTED (owner decision 2026-08-04, see
         `OFFER_TRANSPORT.md`).** Building 6d.4.1 surfaced that the rest of 6d.4 is **cosmetic, and
         partly not achievable given the 4a decision this same plan made**. Measured: `Binding::Offer` (19 sites) and
         `Binding::LiveImpl` (16 sites) are handled *separately* at ~35 sites with **opposite**
         semantics — `Offer` is non-durable (freeze errors) while `LiveImpl` is durably capturable
         via `callee_slot`; `Offer` animates in `cap_dispatch_slots` while `LiveImpl` answers
         `-EINVAL` there and crosses via `live_impl_call`; their payloads share nothing
         (`funcs`/`ops`/`state` vs `callee`/`export`/`callee_slot`). Folding them into one
         discriminated `ImplEntry` would **relocate** the passive-vs-live split from the enum to a
         field without deleting logic, and add `Option`-heavy union fields — a net complexity
         *increase* on the TCB (AGENTS.md prime directive: don't add abstraction until something
         concrete demands it). Worse, **`ProviderState` cannot actually be deleted while the passive
         transport is kept**: its `busy`/`admit_parked`/`busy_owner` admission word and its window
         must live *outside* the `Host` that the sub-run takes by value (else a contender reads a
         stale `busy=false` and double-checks-out the instance), and the passive-animation transport
         (4a) genuinely needs them. So §7's "delete `ProviderState` + its mutex" and "the
         `GuestImpl`/`LiveImpl` binding split" presuppose the **transport merge** (give a library
         instance its own serve loop) that 6d.4's scope note explicitly rejected for reversing 4a.
         The two goals are in tension inside this doc. **Recommendation: 6d.4 concludes at 6d.4.1.**
         The clean structural win is banked (the offer powerbox is now the granted-child shared-cell
         shape; the per-call `Arc::new` is gone); the residual is one uncontended nested lock
         (`Mutex<Host>` inside `Mutex<ProviderState>`), inherent to a passive instance needing
         checkout atomicity over {admission word, window} separate from its takeable powerbox.
         Retiring `ProviderState` would require making library instances serve-loop-driven, which
         pessimizes the common synchronous cross-domain call — **rejected** on that basis
         (`OFFER_TRANSPORT.md`). Increment 6 concludes at 6d.4.1.
7. **`threaded` policy** — opt-in concurrent admission; provider-owned synchronization (§5 axis 2,
   §10.1). Today every instanced offer is `single`: the `busy` admission word serializes handlers
   (a rival caller `-EAGAIN`s or parks as an admission-waiter). `threaded` is a **provider
   declaration** that its handlers may run concurrently — no gate at all (only the quiesce bit, when
   it exists) — with the provider synchronizing its own state via guest atomics/futexes (the §12
   defined-race regime; confinement never depends on data-race freedom). It reuses proven machinery:
   `Mem::fork_for_thread` already shares a window's backing across concurrent vCPUs (the
   `thread.spawn` path), and 6d.4.1 already made the powerbox a shared `Arc<Mutex<Host>>` cell — so
   a concurrent handler forks the provider window and clones the host cell instead of `single`'s
   exclusive take. Decomposed smallest-first:
   - **7.1 — the policy field + the threaded eval-loop admission arm. DONE.** `OfferEntry.policy:
     OfferPolicy` (`Single` default / `Threaded` via `wire_offer_proc_with_policy`; a guest-facing
     declaration is 7.4). In the eval-loop offer arm, a `Threaded` provider **skips the `busy`
     gate**: briefly lock the state to `fork_for_thread` the window + `Arc::clone` the host cell
     (no `busy` set, no exclusive take), then animate the handler fiber. The settle/undo/resume
     paths branch on a `threaded` flag carried by `OfferAnim`/`OfferParked`: no `busy` clear, no
     window restore (the fork drops), no `admit_wake` (nothing can park on an ungated instance),
     and no `strong_count` leak guard (concurrent clones are the declared regime). The host-side
     arm **refused `Threaded` probeably** (`-EAGAIN`) this slice: its checkout takes the world out
     by value, which would gut the shared cell under in-flight animations — closed by 7.3's
     shared-cell sub-run arm. **Pin correction (found in build):** the planned
     "self-recursion vs `-EAGAIN`" observable was wrong — 4c.2 makes a *top-of-stack* self-call
     recurse under both policies. The real single-threaded observable is a **buried** re-entry
     (`X → Y → X`, X on the animation stack but not the top): `threaded` X admits a second
     concurrent animation (`[X, Y, X]`, each over its own forked view) and the chain sums to 9;
     `single` X refuses `-EAGAIN` (pinned both ways in `threaded_offers.rs`).
   - **7.2 — true concurrent-callers pin. DONE.** Two `thread.spawn`ed vCPUs each call one
     `Threaded` instance (each animation over its own `fork_for_thread` view — the `thread.spawn`
     sibling shape) writing distinct cells; the joiner reads both back through the same offer.
     Asserts concurrent cross-vCPU admission and one shared instance window — safety, not timing.
     Plus a state-persistence pin (policy changes admission, never shared-instance semantics).
   - **7.3 — the non-eval-loop tiers. DONE.** Two pieces. **(a) The lock-order fix 7.1 owed:**
     under `Threaded`, two translation-edge scopes can hold *crossed* cells (T1 animating X→Y while
     T2 animates Y→X — a vCPU's current host is itself a provider cell mid-animation), so the
     semantic-order dual acquisition (`hg` then cell) was a latent AB-BA deadlock; `single` could
     never cross (its `busy` gate keeps an instance on one vCPU, checked before any edge lock). All
     dual-`Host` acquisitions now go through `lock_host_pair` — **stable address order**, guards
     returned in argument order; the degenerate same-cell pair (an offer self-wired into its own
     powerbox) refuses probeably at admission. **(b) The tier arm:** `drive_arc` split into a core
     (`drive_over_cell`) that runs the M:N executor over any `Arc<Mutex<Host>>` cell, an owned
     wrapper (`drive_arc`, wrap + unwrap as before — byte-identical), and **`drive_arc_shared`**,
     which runs the sub-run over the instance's **live 6d.4.1 cell** (durable refused fail-closed;
     thaw seeds empty by construction). The host-side `Threaded` arm replaces 7.1's `-EAGAIN`
     refusal: fork the window view, run `drive_arc_shared` over the cell at flat `OFFER_FUEL`, and
     translate the edges (edge 1 `try_lock` + probeable `-EAGAIN`; edge 2 blocking — sound because
     no path can enter this arm holding a provider cell: `IoRing` is not regrantable into a
     powerbox and durable sub-runs are refused). `busy` survives on a `Threaded` instance only as
     the **host-side gate**: one sub-run per instance at a time (the sub-run wires transient run
     hooks onto the cell that concurrent sub-runs would clobber), while eval-loop admission stays
     ungated alongside it. JIT and durable callers get the arm for free through the
     `cap_thunk` → `cap_dispatch_slots` fall-through. Pins: the three-backend threaded state test
     (`imports_impl` — the JIT lane exercises this arm) and the direct host-side dispatch +
     gate-reopen pin (`threaded_offers`).
   - **7.4 — guest-facing declaration.** An IR/interface attribute so a guest module declares
     `threaded` itself, rather than the host wiring param 7.1 uses. Rounds out "policy is the
     provider's declaration" (§10.1).
   - **Quiesce interaction (§10.3):** `threaded`'s only per-call check is the closed bit; folds in
     whenever the §10.3 closed bit lands (it is not yet a field on `ProviderState`), tracked with
     that work, not 7.1.

Addendum deltas to this plan: §10.7.

## 9. Invariants this must not break

- **Confinement is untouched.** Every transport runs verified guest code over masked windows; the
  mem/host switch of inline animation is the §14 nested-view shape, no new access path.
- **Fail-closed stays the default.** A backend tier without an arm answers probeable errno, never
  a wrong answer; a full queue refuses at the enqueuer.
- **interp == JIT** on observable results for every new shape (§18 oracle); scheduling
  nondeterminism stays quarantined in the deterministic explorer, as today.
- **Run-to-park atomicity is preserved by default** (`single`): no existing handler's assumptions
  break without that provider opting into `threaded`.

## 10. Addendum — admission, direct handoff, quiesce, and fuel introspection

Settled with the owner 2026-07-31. Nothing here renegotiates an invariant; everything sharpens
§2–§4 or rides the §8 increments (deltas in §10.7). Status: **design agreed, not yet built**,
same as the parent design.

### 10.1 What "admission" is

Admission is the act of letting one inbound call mint a handler fiber in a provider's world. Being
precise about the three things in play kills a recurring confusion:

- **What the gate protects:** the provider *domain's* one world — specifically `single`'s
  run-to-park atomicity (§3): between a handler's admission and its next park or return, no other
  handler interleaves. The gate is per **provider domain**, never per fiber.
- **What is admitted:** one dispatch, which mints one handler fiber.
- **What the gate is not:** a whole-domain lock. The domain's `main`, its spawned threads, and its
  previously-admitted handlers parked mid-outbound-call all keep existing and (where runnable)
  running — blocking stays per-fiber (§2). A handler that parks closes its atomicity window and
  the gate reopens; that is exactly how the §1 `A[f1] → B[f2] → A[f3] → B[f4]` chain composes.

Consequences worth stating outright:

- `single` serializes handlers **against each other**, never against the domain's own spawned
  threads. A provider that threads over its handler state has taken the §12 discipline on itself;
  the gate neither can nor should protect it from itself.
- `threaded` has **no gate at all** — every call is admitted immediately as a concurrent fiber
  (the only per-call check left is the quiesce bit, §10.3). The policy is a *declaration by the
  provider*, never an inference from vCPU count — a domain has no fixed vCPU allotment to infer
  from (vCPUs and fibers are workers inside the world, INVARIANTS.md 6). The runtime never pays a
  serialization cost the guest didn't order (host = mechanism, guest = policy, INVARIANTS.md 4).
- Admission *state* differs by provider kind, which is the whole content of the §3 axis-1 split:
  a **library** provider's admission state is the one try-enter flag; a **process** provider's is
  "at a serve point" — parked in `svc.wait`, or draining `svc.poll`. Between serve points its
  world is its own and calls queue.

### 10.2 The per-call transport decision (adds **direct handoff**)

One decision tree per call, replacing "sync vs async" entirely. After the ordinary §3c use-site
resolve:

1. **Quiesce bit closed** (§10.3) → contended path: enqueue + park (bounded; `-EAGAIN` at the rim).
2. **`threaded` provider** → no gate: mint a concurrent handler fiber, animate it inline on the
   caller's thread.
3. **`single` library provider, try-enter won** → inline animation (increment 3's arm).
4. **`single` process provider parked at `svc.wait`** → **direct handoff**: the caller claims the
   serve activation and animates the handler fiber on its own thread — no enqueue, no
   wake-a-worker, no reply round-trip. The Doors / L4 direct-process-switch shape.
5. **Otherwise** (mid-handler, between serve points, queue has priority work) → enqueue + park —
   today's built transport, unchanged.
6. **Handler parks mid-animation** (any inline arm) → promotion (increment 4): file the reified
   fiber + waiter; the caller parks (interp) or thread-blocks (JIT).

**Handoff settlement rule.** A handoff-served dispatch **counts in the callee's serve
accounting**: the parked `svc.wait` completes with the same served-count observation the enqueue
path would have delivered (`serve_count`, `svm-interp:9248`). The only degrees of freedom are
which thread animated the handler and when the provider's `main` is scheduled awake — both already
quarantined as scheduling (§4). Differential pin: **handoff-on ≡ handoff-off on observable
results**, same discipline as increment 3's pin against increment 2.

**Topology never gates.** The only "who" check is grant-graph reachability, applied at grant time
(INVARIANTS.md 3); admission state is the entire call-time condition. A topology restriction
(parent↔child only, say) could only ever be a transport pessimization — transports may differ,
observable semantics may not (§4) — and a call-time gate keyed on caller identity is the shape
INVARIANTS.md 4 forbids. Parent↔child calls get the fast path in practice because that is where
uncontended admission dominates; a sibling introduced by `regrant_into_child` gets it on the same
terms.

### 10.3 Quiesce rides the admission word

The admission flag gains a **closed** bit. Freeze and teardown close it, so one CAS answers both
"busy" and "quiescing" with the same contended path — no second check, no new machinery. For
`threaded` providers this bit is the only per-call check at all.

- Once closed, no new crossing starts; in-flight handlers drain to completion or park. The
  mid-handler freeze refusal is unchanged (`STATE_UNWINDING`, `svm-interp:9076`) — the previous
  snapshot stays the recovery point.
- A caller parked at the gate across a freeze **re-issues on thaw** — O10 at-least-once,
  INVARIANTS.md 7. Recovery is re-execution, as everywhere.

### 10.4 Crossing depth (the JIT arm's one new bound)

Interp inline animation switches between **reified** fibers, so re-entrant `A→B→A→B` chains never
grow the native stack. JIT crossings are real native frames through thunks — so the JIT arm
carries a per-thread crossing-depth bound; at the bound the call **declines the inline arm and
takes the parked transport** (declining toward the slower correct transport, never a wrong answer
— the §9 fail-closed shape). Detached-window children (`instantiate_detached`) inline fine
in-process; a future separate-process tier must decline handoff to the parked transport.

### 10.5 Caller-pays refinements (deferred, demand-gated)

Increment 5 lands caller-pays uniformly: fuel follows the fiber across the crossing — the counter
just keeps draining — and the wirer-priced reserve (`impl_fuel_remaining`, `GUEST_IMPL_FUEL`)
leaves with increment 6. The division of responsibility: the provider owns *what* runs (its code,
its policy, its refusals); the caller owns *how much* runs (it made the calls), so the caller
pays. Provider-pays is a drain-DoS against the provider and breaks the amplification bound (total
computation a domain causes ≤ its budget, D19-attenuating down the grant graph).

Two optional refinements, each **deferred until a named consumer** (INVARIANTS.md 1):

- **Per-call fuel cap** (caller-side attenuation, D19-shaped): bound the caller's exposure to a
  runaway provider. A cap changes *whose budget bounds the run*, not the trap semantics —
  mid-handler exhaustion is still `OutOfFuel`, terminal for the provider's world (INVARIANTS.md 6).
- **Admission fuel floor** (gate-side): refuse admission fail-closed — probeable errno, before any
  mutation — when the caller's remaining fuel can't plausibly fund a handler. The shared-provider
  protection against a starving caller. (The same hazard is strictly worse under provider-pays,
  where deliberate drain is free.)

### 10.6 Fuel introspection: `fuel.remaining`

A new self-namespace op at the next reserved number (12, after `clone_caller` = 11):
`fuel.remaining() -> i64`, the domain's remaining fuel. Authority-neutral — reports the domain's
own state, confers nothing — riding `cap.call CAP_SELF_TYPE_ID` like the rest of the namespace:
no wire change, no new opcode.

- **Deterministic by prior work.** Fuel is a checked cross-engine quantity charged at IR-anchored
  safepoints (INVARIANTS.md 9, fuel clause; `bytecode_diff` already asserts bit-exact remaining
  fuel), so the readout returns the identical value on every backend *by construction*. Because it
  is observable, it lands **on all backends at once, with increment 5** — the same rule that
  sequenced caller-pays — with its own differential pin.
- **Call metering by subtraction.** Under caller-pays, the cost of a call is read-before minus
  read-after. No metering ABI, no provider cooperation. (Provider-pays could never offer this —
  the cost lands in someone else's reserve.)
- **Why it earns its place.** `OutOfFuel` is a trap and traps are domain-terminal
  (INVARIANTS.md 6); a probeable readout is what lets a domain checkpoint, return early, or
  decline an expensive branch instead of dying mid-mutation — and it is the primitive the §10.5
  refinements would build on.
- **Raciness caveat.** In a multi-vCPU domain the readout is a snapshot of one shared,
  monotonically decreasing counter — interleaving-dependent, quarantined in the deterministic
  explorer like every other defined race; exact in a single-threaded domain.
- **End state.** Once §15's live budget charging lands (the `create(module, window, budget)`
  accounting), `Budget.read` (field 0) becomes the same number read through the quota object; the
  intrinsic stays as the handle-free spelling or folds into it — decided then, not now.

### 10.7 Increment deltas

- **Direct handoff (interp)** rides increments 3–4 (a mid-handoff park needs the promotion
  machinery). Pin: handoff-on ≡ handoff-off.
- **Increment 5 additionally lands:** the JIT crossing-depth bound and `fuel.remaining` — both
  observable, so both all-backends-at-once.
- **Deferred, named-consumer-gated:** per-call fuel cap; admission fuel floor.

What §9 gains: *transport choice — inline, handoff, or parked, and any future restriction of
them — may never change observable results*; and `fuel.remaining` parity joins the differential
contract.
