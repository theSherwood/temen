# FORK.md — `fork()`-returns-twice, the durable-clone capstone

The plan for POSIX `fork()` on svm (STAGE1.md item 3 / PROCESS.md §7 / the S11 stage). This is the
roadmap's single biggest item; it is a **multi-PR arc**, tracked here so it stays legible across
sessions. R8 closure (durable `call_indirect` to may-suspend targets) is the prereq and is **done**.

## 1. The mechanism (PROCESS.md §7, verbatim intent)

`fork()` is **personality sugar over durable freeze → clone-window → thaw-both**. "Fork-returns-twice
is a *reply value*, not a substrate concept": one `fork()` `cap.call` parks the caller; a **servicer**
clones the parked caller into a second window and holds **two reply tokens** — replying `pid` to one
copy and `0` to the other. Both copies resume from the same park, each with its own reply value.

Exactly **one** domain is frozen and copied — the caller. There is no second frozen domain; a second
party appears only as the **callee/servicer** the caller is parked on.

## 2. The wall — open question O10 (confirmed in code)

`freeze_drive` (`crates/svm-interp/src/lib.rs:6984`) **refuses** to freeze a fiber parked on an
un-replied `cap.call` (`Blocked::CapReply`) — exactly fork's park point:

> "An unwoken CAP park would spill the freeze placeholder into a `Leaf` frame (reloaded as the call's
> result — unsound), so it fails the whole freeze closed." (`:6981-6983`)

A futex park re-issues fine on thaw (`MemoryWait`); a cap park has no reply source at whole-run freeze
time, so it fails closed. How a pending served call *should* resolve is the gate — and §3 shows fork's
answer (inject the reply), distinct from durability's (re-issue). Everything downstream is comparatively
mechanical.

## 3. Fork is reply-**injection**, not re-issue

The decisive test: after `fork()`, the child continues from **right after** the call with return value
`0`. It must **not re-execute `fork()`** — that would re-fork infinitely. So each copy resumes **past**
the call with a **supplied** result. That is **reload**, not re-issue — and it settles the mechanism:

- The `MemoryWait`/`SvcServe`/`ThreadJoin` **re-issue** pattern is **wrong for fork** — those re-issue
  because their ops are re-drivable; fork's call must *not* be re-driven.
- The reply each copy reloads is **injected by the servicer** (`pid` to the original, `0` to the twin),
  known at clone time.

**This dissolves the O10 refusal cleanly.** `freeze_drive` refuses a `CapReply` park only because a
*whole-run* freeze has no servicer to supply the reply, so it would spill a **placeholder** (unsound).
The clone *supplies the real reply*, so spilling it (reload, at the post-call resume point) is sound. The
unsoundness was never intrinsic — it was the absence of a reply source.

**The re-issue path is a different concern — durability/migration.** Snapshotting a run *mid-call* and
restoring it later *does* want re-issue (re-drive the call against the restored servicer). That is real
and valuable, but it is **not fork's mechanism** and is off fork's critical path. Fork injects; it never
re-issues.

## 4. How a parent names a child (the handle model — nesting-friendly by construction)

The Instantiator ops (0 `instantiate`, 5/13 `instantiate_module`, `svm-interp:11376`) return an `i32`
**`child_handle`**, non-blocking. That handle is a **capability** — "holding the handle is the authority
to nest (D19)". It resolves (in the *parent's* runtime) to a scheduler `TaskId` → the child's `VCpu`
(its own `vcpu_ctx`, shadow region, and window carve via `nested_view`). Sibling ops already take it:
`join` (1), `poll` (9), `kill` (12), `child_offer` (14). It is **capability-scoped, not a global PID
table**: a parent only holds handles to children it spawned.

The clone verb therefore slots in as a natural sibling:

```
clone(child_handle) -> twin_child_handle
```

The servicer names the *specific parked child* to clone by the handle it already holds, symmetric with
`join`/`kill`/`offer`. This **composes with nesting by construction**: bash's parent holds bash's handle
regardless of depth, so "clone the child I hold a handle to" works at any nesting level. **A nested bash
must be forkable — no design step may force the forking guest to be top-level.**

> **Finding (2026-07-29, from the PR-1 harness attempt): PR 1 and PR 2 are inseparable.** A caller
> *persistently* parked on `CapReply` at a **whole-run** freeze is essentially unconstructable with normal
> guest code: `live_impl_of` requires a *serving* callee (a non-serving one `CapFault`s the call), a
> serving callee *replies* (no persistent park), and a mid-handler callee hits the `handler_parks` gate,
> not the `CapReply` gate. The only durable test that freezes across an offer (`serve.rs`
> `SRC_NESTED_HOLDER`) freezes with everyone idle in `svc.wait` — the offer calls run on *thaw*. So the
> reply-injection state arises **only** when a servicer deliberately withholds the reply and freezes the
> caller — which *is* fork's targeted-clone action. There is no independently-testable whole-run
> reply-injection nucleus; **the first buildable slice is the targeted clone itself.**

## 5. The PR arc

- **PR 1 — the targeted clone `clone(child_handle)` (was two PRs).** From within a serve handler (the
  servicer), capture the calling fiber's continuation (identified by its reply ticket), spill it at the
  caller's **post-call resume point** with a **reply slot**, copy the carve into a **twin** domain, and
  register a second `(callee, ticket)` in `ticket_waiters`. The servicer then replies to each copy — each
  reloads its injected reply and resumes **past** the call. Return-twice, in one live run. This folds the
  old "durable-layer nucleus" and "targeted clone" together because (per the finding above) the injection
  state only exists under a servicer-triggered freeze. *The hardest substrate work; single-vCPU first.*
- **PR 2 — the `fork` personality op + endpoint.** Add `"fork"` to `svm-posix resolve` as sugar over
  PR 1's `clone`, the servicer replying `pid`/`0`. The clone-servicer lives with the domain's
  personality-provider / parent (which holds the `child_handle`), so it composes with nesting.
- **PR 3 (later) — multi-vCPU `forkall`** (O11); CoW clone (deferred, S13).

## 6. PR 1 spec — the targeted clone `clone(child_handle)`

A servicer, **mid-handling** a call from a parked caller, clones the caller instead of replying. The
handler knows the caller by its dispatch **ticket** (the `(callee, ticket)` reply token). The op — call
it `clone_caller() -> twin_child_handle` (self-namespace, servicer-side) or `clone(child_handle)` — does:

1. **Capture the caller's continuation.** The caller is parked on `Blocked::CapReply { ticket, callee }`
   with its live frame *inside* the pending `cap.call`. Spill that fiber's live set into its window,
   positioned at the `cap.call`'s **post-call resume point** (the ordinary `Leaf`-style reload) with a
   **reply slot** — the reply-injection capture, not a placeholder and not a re-issue arm.
2. **Twin it.** Copy the caller's carve (window slice) into a fresh child domain (via the
   `spawn_named_child` path), re-grant its pass-through handles, and re-seed a `CapReply` park for the
   twin with a second `(callee, ticket')` in `ticket_waiters`.
3. **Reply to each.** The servicer delivers `A` to the caller's ticket and `B` to the twin's — each
   fiber reloads its injected reply and resumes **past** the call.

**Success criterion:** one servicer handler, one caller that calls it, `clone` inside the handler → the
caller returns `A` and the twin returns `B` from the *same* call site. Return-twice, one live run,
interp==JIT. (This is why it cannot be a pure-durable unit — the capture state only exists under the
servicer's action; see the §5 finding.)

**Smallest first step (de-risk the capture).** Before the twin/second-ticket machinery: prove a servicer
can **capture a parked caller's continuation at the reply-slot resume point and re-inject its own reply**
(a no-op clone that just resumes the original with an injected value through the freeze/restore path).
That isolates the O10 lift (`flatten_fiber_for_freeze` + `freeze_drive`, `:6984`/`:6997`) — the load-
bearing soundness change — from the twin-instantiation. Only then add the copy + second ticket.

**Load-bearing risk.** A parked `cap.call`'s live set lives in the interp's native frame Vec, not the
window — lost unless capture spills it, positioned at the post-call point with the reply slot,
interp==JIT byte-identical. A subtle error yields a *wrong-but-passing* result — the reason to isolate
the capture step first, TDD-first.

## 7. Invariants this must not break

- **Confinement is untouched.** Fork is durable-transform + freeze-driver + Instantiator authority; it
  adds no new memory-access path. A transform/clone bug is a **correctness** bug, never an escape
  (DESIGN.md §3 / DURABILITY.md §3).
- **Fail-closed stays the default.** Only a `CapReply` park captured by the clone path may freeze; every
  unclassified park still `FiberFault`s.
- **interp == JIT** across every new shape (the §18 oracle), as for all durable work.
- **Single-vCPU first** (freeze_drive slice 3.1); nested/multi-vCPU is PR 2 / O11.

## 8. Implementation plan for PR 1 — the handler→caller→capture linkage

The servicer-triggered path has a **natural harness** (a serve handler that captures its own caller),
which sidesteps the whole-run-freeze dead-end. The linkage is already in the code:

- A running handler carries `serve_run: Some(ServeRun { ticket, .. })` (`svm-interp:6729`) — the dispatch
  ticket its return would answer. (A `svc.*` op *under* a handler is refused `-EINVAL`, so the serve loop
  is the domain's outermost dispatcher — the clone op rides the same serve-frame position.)
- Its caller is parked as `Sched::ticket_waiters[(callee_domain_id, ticket)] = Waiter::Fiber { reg, slot,
  svc }` (`:3796`) — a direct handle (`Arc<FiberRegistry>` + slot) to the caller's parked fiber.

So the op — `clone_caller() -> twin_child_handle | -errno`, self-namespace, servicer-side — is:

1. In the handler, read `serve_run.ticket` + the callee domain id; look up the caller's
   `Waiter::Fiber { reg, slot }` in `ticket_waiters`. `-EINVAL` if not called from a handler / no waiter.
2. **Capture** that specific fiber's continuation: `flatten_fiber_for_freeze`-style spill of `(reg, slot)`
   at the caller's `cap.call` **post-call resume point** with a **reply slot** — the targeted,
   single-fiber freeze (the load-bearing new primitive; the whole-run `freeze_drive` is not involved).
3. **Twin**: copy the caller's carve into a fresh child (`spawn_named_child` path), re-grant pass-through
   handles, re-seed a `CapReply` park for the twin under a second `(callee, ticket')`.
4. Return the twin's `child_handle`; the servicer then `cap_reply`s each ticket (`pid`/`0`).

**Smallest first code step (de-risk the capture, no twin):** `clone_caller()` that captures the caller
into a window image and immediately restores it *in place* with an injected reply — proving a specific
parked fiber's continuation serializes + resumes past its call with a supplied result. Harness: a server
whose handler calls it, one caller calling in. Only then add step 3 (twin + second ticket). This isolates
the freeze/flatten soundness change from the instantiation, interp==JIT, TDD-first.
