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

## 5. The PR arc

- **PR 1 — the reply-injection nucleus (O10, at the durable layer).** Make a `CapReply`-parked
  continuation **freezable** by spilling it at the **post-call resume point** with a **reply slot**, and
  let restore **inject** the reply the copy reloads. Proven with the existing whole-run
  snapshot/`begin_thaw` machinery, no new substrate and no twin: freeze a run whose caller is parked on a
  served `cap.call`, then thaw the **same snapshot twice** injecting reply `A` vs `B` → **two different
  returns**. That is return-twice at the durable layer — the exact reply-injection atom, isolated. (§6.)
- **PR 2 — the targeted clone verb (nested, slice 3.2).** `clone(child_handle) -> twin` — capture a
  *specific* parked child's continuation (not the whole run), instantiate a **twin** over a copied carve
  in a *live* run, and register a second `(callee, ticket)` in `ticket_waiters` so the servicer delivers
  a reply to each. This lifts PR 1's durable-layer injection into a live, nested clone — the nested-bash
  requirement, and the hardest substrate work.
- **PR 3 — the `fork` personality op + endpoint.** Add `"fork"` to `svm-posix resolve` as sugar over
  PR 2's clone, the servicer replying `pid`/`0`. The clone-servicer lives with the domain's
  personality-provider / parent (which holds the `child_handle`), so it composes with nesting.
- **PR 4 (later) — multi-vCPU `forkall`** (O11); CoW clone (deferred, S13).

## 6. PR 1 spec — the reply-injection nucleus

**Success criterion (`svm-durable/tests/pending.rs`):** freeze a run whose caller is parked on a served
`cap.call` (today `FiberFault` at `freeze_drive:6986`); the snapshot captures the parked continuation at
its post-call resume point with a reply slot. Then, the `roundtrip.rs` pattern — `snapshot.clone()`
**twice**, `begin_thaw` each, **inject** reply `A` into one and `B` into the other → the caller **reloads
past the call** and returns `A` vs `B`. Two divergent returns from one frozen snapshot = return-twice.
Interp==JIT.

Harness: a *second* guest exists only as the **callee** the caller parks on (it need never reply — the
reply is injected at restore). Only the caller is captured; it may be top-level here (nested/targeted
capture is PR 2). *(Note: constructing a persistent `CapReply` park is itself finicky — a live callee
replies, a dead/non-serving one `CapFault`s, a mid-handler park hits the `handler_parks` gate — so the
harness must hold the caller parked without the callee replying or being mid-handler.)*

**The edits (reply-injection, not re-issue):**
1. `svm-interp` `flatten_fiber_for_freeze` (`:6997`): a `CapReply`-parked fiber spills its live set and is
   positioned at the `cap.call`'s **post-call resume point** (the `Leaf`-style reload), with a **reply
   slot** in the spill — *not* the placeholder `Leaf` spill (`:6999`), and *not* a re-issue arm.
2. `svm-interp` `freeze_drive` (`:6984`): lift the `CapReply`-park refusal **only** when the park is
   captured this way (reply-slot reload); every other unclassified park still fails closed.
3. Restore: `begin_thaw` (or a small sibling) writes the **injected reply** into the reply slot so the
   thawed fiber reloads it and resumes past the call. Two thaws, two injected replies.
4. Tests: `pending.rs` + a JIT-differential entry (mirroring `indirect.rs`).

No new `SuspendKind` re-execute arm is needed — the resume point is the ordinary post-call `Leaf` reload;
the only new thing is *entering* it from a park with an injected (rather than call-produced) result.

**Load-bearing risk.** A parked `cap.call`'s live set lives in the interp's native frame Vec, not the
window — lost across snapshot→thaw unless `flatten` spills it, positioned at the post-call point with the
reply slot, interp==JIT byte-identical. A subtle error yields a *wrong-but-passing* test — hence PR 1
alone, TDD-first (characterization test pinning the current `FiberFault` before any soundness change).

## 7. Invariants this must not break

- **Confinement is untouched.** Fork is durable-transform + freeze-driver + Instantiator authority; it
  adds no new memory-access path. A transform/clone bug is a **correctness** bug, never an escape
  (DESIGN.md §3 / DURABILITY.md §3).
- **Fail-closed stays the default.** Only a `CapReply` park *classified* re-issue may freeze; every
  unclassified park still `FiberFault`s.
- **interp == JIT** across every new shape (the §18 oracle), as for all durable work.
- **Single-vCPU first** (freeze_drive slice 3.1); nested/multi-vCPU is PR 2 / O11.
