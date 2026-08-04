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

**Deleted:** the nested `drive_arc`-under-two-locks sub-interpreter; `ProviderState` + its mutex;
the acyclicity rule and its deadlock argument; provider-pays fuel (`GUEST_IMPL_FUEL`);
`wire_impl_instance` / `impl_service`; the `GuestImpl`/`LiveImpl` binding split; eventually the
JIT's `-EINVAL` arm for offers.

**Added:** admission into library providers (reuses the existing queue + handler-fiber machinery);
the inline cross-domain animation fast path (assembled from existing pieces: §14 coroutine drive +
`serve_switch`'s fiber switch); the JIT thunk fast path + thread-blocking promotion.

Net: three offer mechanisms → one; two execution models → one; a lock-held nested interpreter and
its safety argument leave the TCB. A genuine net simplification, paid for with scheduler-adjacent
plumbing that lands in increments (§8).

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
   here**, uniformly on all backends. Closes the `live_impl` parity gap.
6. **Retire the two-lock sub-run** — with 3–5 landed, the passive-provider `drive_arc` nested
   executor and `ProviderState` collapse onto the inline-animation path (the original increment-2
   goal, now reachable without a parity regression).
7. **`threaded` policy** — opt-in concurrent admission; provider-owned synchronization.

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
