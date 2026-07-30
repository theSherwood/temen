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

A futex park re-issues fine on thaw (`MemoryWait`); a cap park does not. The re-issue-vs-reload decision
for a *pending served call* is unresolved and unbuilt. This is the gate; everything downstream is
comparatively mechanical.

## 3. The `Leaf`-vs-re-issue distinction

The whole nucleus is one distinction, and it is purely about **whether the call had replied** when the
freeze tripped:

- **completed** (froze at the trailing poll, *past* the call): the reply is in the frame → thaw
  **reloads** it (today's `SuspendKind::Leaf`). Re-executing would double a side effect (e.g. `write`),
  so reload is mandatory.
- **pending** (froze at the *park*, mid-call, *before* any reply): nothing to reload → thaw must
  **re-issue** the call (the `MemoryWait`/`SvcServe`/`ThreadJoin` pattern). Reloading a placeholder is
  the unsoundness `freeze_drive` refuses.

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

- **PR 1 — the re-issue nucleus (O10).** Freeze a fiber parked on a pending served `cap.call` and
  re-issue it on thaw instead of reloading a placeholder. No personality, single-vCPU, nesting-agnostic.
  Return-twice in miniature: thaw two window copies, deliver two different replies → two different
  returns. *If this doesn't work, nothing downstream can.* (Detail in §6.)
- **PR 2 — independent freeze of a nested caller (slice 3.2) + the clone verb.** Capture a *specific*
  parked nested `vcpu_ctx` + copy its carve (not the whole run), instantiate a twin, and register a
  second `(callee, ticket)` in `ticket_waiters` so the reply can be delivered twice. This is the piece
  the nested-bash requirement makes mandatory, and the hardest substrate work.
- **PR 3 — the `fork` personality op + endpoint.** Add `"fork"` to `svm-posix resolve` as sugar over
  PR 2's clone, the servicer replying `pid`/`0`. The clone-servicer lives with the domain's
  personality-provider / parent (which holds the child_handle), so it composes with nesting.
- **PR 4 (later) — multi-vCPU `forkall`** (O11); CoW clone (deferred, S13).

## 6. PR 1 spec

**Success criterion (`svm-durable/tests/pending.rs`):**
1. *Minimal re-issue:* a caller parked in a served `cap.call` freezes (today `FiberFault` at
   `freeze_drive:6986`); on thaw the call re-issues, the servicer replies `V`, the caller returns `V`.
   Interp==JIT.
2. *Return-twice:* `snapshot.clone()` twice (the `roundtrip.rs` pattern), deliver reply `A` to one copy
   and `B` to the other → two different return values.

Harness: a *second* guest exists only as the **callee** producing the park (a serve loop that withholds
its reply); **only the caller is frozen**, and it may be top-level here — independent freeze of a nested
caller is PR 2.

**The four edits:**
1. `svm-durable`: `SuspendKind::PendingReissue { ty, op, handle, args }` — a `cap.call` to a may-park cap
   (offer-type / non-host, conservatively by `type_id`) gets a *re-issue* arm: spill the call's operands
   (handle + args, live *before* the call) and, on `REWINDING`, reload them and **re-execute** the
   `cap.call`, leaving state `REWINDING`. The `SvcServe`/`MemoryWait` re-issue pattern applied to an
   *outbound* call.
2. `svm-interp` `flatten_fiber_for_freeze` (`:6997`): a `CapReply`-parked fiber spills its operands and
   is positioned at the PendingReissue arm, not the `Leaf` placeholder-result spill (`:6999`).
3. `svm-interp` `freeze_drive` (`:6984`): lift the cap-park refusal *only* for a `CapReply` park
   classified re-issue; every other park still fails closed.
4. Tests: `pending.rs` + a JIT-differential entry (mirroring `indirect.rs`).

**Load-bearing risk.** A parked `cap.call`'s live operands live in the interp's native frame Vec, not
the window — lost across snapshot→thaw-on-fresh-host unless `flatten` spills them. The transform today
has resume points only *after* a call (leaf reload); PendingReissue needs a point that reloads operands
and re-executes *from before* the call, with interp and JIT byte-identical. A subtle error yields a
*wrong-but-passing* test — hence PR 1 alone, TDD-first (characterization test pinning the current
`FiberFault` before any soundness change).

## 7. Invariants this must not break

- **Confinement is untouched.** Fork is durable-transform + freeze-driver + Instantiator authority; it
  adds no new memory-access path. A transform/clone bug is a **correctness** bug, never an escape
  (DESIGN.md §3 / DURABILITY.md §3).
- **Fail-closed stays the default.** Only a `CapReply` park *classified* re-issue may freeze; every
  unclassified park still `FiberFault`s.
- **interp == JIT** across every new shape (the §18 oracle), as for all durable work.
- **Single-vCPU first** (freeze_drive slice 3.1); nested/multi-vCPU is PR 2 / O11.
