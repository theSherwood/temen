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

1. **Rename-only PR** — the §6 table, no behavior change (mechanical, reviewable as a pure diff).
2. **Library-provider admission** — serve-queue admission into a domain with no running main;
   retire the instanced path onto it (delete `ProviderState`, the two-lock sub-run, acyclicity).
   Eval-loop first; the instanced offer's tests convert into the new path's tests.
3. **Inline fast path (interp)** — uncontended `single`-provider calls animate the handler on the
   caller's thread (coroutine-drive shape); contended calls take the queue. Differential-tested
   against increment 2's queue-only behavior (results must be identical).
4. **Promotion (interp)** — a handler that parks mid-inline-animation files its reified fiber +
   waiter; caller parks on the reply. (Mostly `handler_parks` re-plumbed.)
5. **JIT arm** — the thunk fast path; park = thread-block on the reply. Closes the parity gap.
6. **`threaded` policy** — opt-in concurrent admission; provider-owned synchronization.

Open question to settle during increment 2: **fuel.** Inline animation naturally burns *caller*
fuel (it is the caller's thread); a process provider's own execution burns its own. Simpler than
provider-reserve, but a behavior change from the instanced path — pin it in the docs when it lands.

## 9. Invariants this must not break

- **Confinement is untouched.** Every transport runs verified guest code over masked windows; the
  mem/host switch of inline animation is the §14 nested-view shape, no new access path.
- **Fail-closed stays the default.** A backend tier without an arm answers probeable errno, never
  a wrong answer; a full queue refuses at the enqueuer.
- **interp == JIT** on observable results for every new shape (§18 oracle); scheduling
  nondeterminism stays quarantined in the deterministic explorer, as today.
- **Run-to-park atomicity is preserved by default** (`single`): no existing handler's assumptions
  break without that provider opting into `threaded`.
