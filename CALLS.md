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
