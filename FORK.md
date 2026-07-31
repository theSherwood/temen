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

### 8.1 Increment breakdown (the build order actually followed)

- **Increment 1 — the handler→caller linkage. DONE (this branch).** New eval-loop arm for op 11:
  returns `serve_run.ticket` when in a running handler (`*cur != serve_cur`), else `-EINVAL`; host-side
  dispatch answers a probeable `-EINVAL` (like svc.poll/wait). `CAP_SELF_CLONE_CALLER = 11` pinned.
  Tests: `crates/svm-interp/tests/clone_caller.rs`. Proves *only* that the servicer can name the parked
  caller — no capture yet.
- **Increment 2 — the reply-injection nucleus. DONE (PR #528).** `clone_caller(reply)` delivers `reply`
  to the parked caller *out-of-band* (`cap_reply_or_stash(ticket, reply)`) and sets `ServeRun.replied`, so
  the handler's `FIBER_RETURNED` skips its auto-reply and the caller reloads the **injected** value, not the
  handler's return. This *is* "return-twice is a reply value" (§3), the heart of fork. **Key simplification
  vs the original §8.2 plan:** because both copies resume in the **same live run**, no durable window-image
  serialization is needed — the reply crosses via the existing race-safe `cap_reply_or_stash` path. The
  durable freeze/flatten (§8.2) is only relevant to *snapshot/migration*, which is off fork's live path.
- **Increment 3 — the twin. DONE (PR #531 = primitives; this PR = wiring).** `clone_caller(reply_orig,
  reply_twin)` duplicates the parked caller vCPU into a fresh domain (`VCpu::fork_twin` over
  `Mem::fork_private` + `Host::fork_powerbox`) and delivers each its reply via
  `Scheduler::fork_parked_caller`. Both resume past the same fork `cap.call` — return-twice, one live
  run. **No second ticket** in the end: the twin's reply is known at clone time, so it is enqueued
  `runnable` with `pending = CapResult(reply_twin)` rather than parked. Only a **bare** root park (no
  children/fibers) forks; otherwise it degrades to a single reply (never hangs). `fork_powerbox` shares
  `live_impls` (a forking caller always holds the offer it is parked in).
- **PR 3 — the `fork` personality op** replies `pid`/`0` to the two copies, and **re-wires the
  personality's closure caps** (libc `host_fns`, fds) into the twin — the part `fork_powerbox` fails
  closed on. That closure re-wiring, and a `Waiter::Fiber` (non-root) caller, are what remain for a real
  bash `fork()`.

### 8.4 PR 5 status (2026-07-31) — the endpoint pieces, landed

- **`fork_ctx` (host-cap forkability). DONE.** `HostProcFork = Arc<dyn Fn() -> HostProc>` — the
  provider's fork factory, the Rust face of a C embedder's `fork_ctx(parent_ctx) → child_ctx`;
  `grant_host_proc_forkable` registers it; `fork_powerbox` carries host procs iff **every** entry has a
  factory (one factory-less entry fails the whole fork closed), minting the twin's closures at the same
  indices and riding the factories along (fork-of-fork). `HostCap::host_proc`'s `make` **is already the
  factory** (the multi-backend "fresh closure per grant over shared state" contract = fork-shares-state),
  so every powerbox host cap — including `posix_cap`'s libc (shared fd table/memfs) — is forkable by
  construction.
- **Pid mode. DONE.** `clone_caller` arity picks the mode: 2 args = explicit `(reply_orig, reply_twin)`;
  0/1 args = **pid mode**, what `fork()` desugars to — the original's reply is the twin's `TaskId`
  (parent sees pid), the twin's is the arg (0); a failed fork replies `-EAGAIN`, POSIX's fork-failure
  errno. Pinned end-to-end (parent joins pid 3; shared stdout shows `{0, 3}`).
- **`"fork"` name binding. DONE.** `svm_posix::bind_with_fork(m, host, libc, Some((type_id, handle)))`
  routes an import named `fork` to the parent-wired live fork offer; every other libc name to the shared
  `HOST_PROC` handle; no offer supplied → bind fails closed.
- **Remaining for a real shell fork:** the **manager topology** (a parent module that spawns the guest,
  serves the fork offer with a pid-mode `clone_caller` handler, and re-grants it via `bind_with_fork`) —
  built with the real-shell consumer (STAGE1/on-ramp track), where it is exercised for real — and the
  `Waiter::Fiber` (non-root fiber) caller, which nested-child callers do **not** need (a §14 child parks
  as `Waiter::VCpu`, so nested-guest fork works without it).

### 8.5 Track 2 plan — a real program forking on svm (2026-07-31, from the machinery map)

The substrate (pid-mode `clone_caller`, `fork_powerbox`, `fork_parked_caller`) has landed; **no new
interp op is needed.** The remaining work is *integration plumbing* to get a compiled-C guest into the
right topology. The minimal live-run fork demo already exists — `svm-interp/tests/clone_caller.rs`
`SRC_FORK_PID`: a manager (func 0) spawns a server (func 1, a `svc.wait` loop whose func-2 handler runs
pid-mode `clone_caller`), mints the fork offer with `child_offer` (Instantiator op 14), spawns the caller
with `instantiate_named` (op 11) re-granting that offer, and the caller forks and both copies write pid/0
to a shared stdout stream. That guest is **hand-written IR** and uses a **regranted stream** (no libc).
The remaining slices turn that into a *compiled-C* program with *real libc*:

- **Slice 1 — forkable libc. DONE (this PR).** `svm_posix::grant` mints libc via
  `grant_host_proc_forkable` over the shared `Inner`, so a chibicc-world domain's libc carries across a
  twin (was fail-closed). `posix_cap` (on-ramp) was already forkable (§8.4).
- **Slice 2 — the `__px_` shim fork binding** (blocker A). The compiled-C tests bind through
  `bind_shim` (`crates/svm/tests/*`), which strips `__px_` and calls `svm_posix::resolve` — no `fork`
  branch, no offer arg. Add the `__px_fork` analogue of `bind_with_fork` so a compiled-C `__px_fork`
  import routes to the fork offer `(type_id, handle)`. (The plain-libc `bind_with_fork` already exists.)
- **Slice 3 — libc into a nested child** (blocker D / the real gap). A §14 child spawned by
  `instantiate_named`/op 13 gets an *attenuated* powerbox (instantiator + address space + regranted
  streams/pipes) — **not** posix libc. Existing compiled-C children get libc only because they run as a
  *separate top-level run* via the `set_spawn` delegate (`c_posix_spawn.rs`), which cannot be the
  *parked* caller a fork needs. So a nested fork-guest needs libc in its powerbox: extend
  `regrant_into_child` to carry a **forkable `HostProc`** (re-grant the libc handle, sharing `Inner`), or
  give the child a fresh forkable libc at spawn. This is the load-bearing new interp/posix plumbing.
- **Slice 4 — the compiled-C entry ABI over `instantiate_named`** (blocker C). The spawn ABI hands a
  child entry an `(i64)` starter arg; hand-written guests are `func (i64)->(i64)`. Confirm/adapt a
  compiled-C `_start` (entry idx 0, chibicc convention) to tolerate the starter-arg/entry convention when
  spawned via op 11/13 (existing op-13 compiled-C exec is `c_shell_exec.rs`). The **main untested seam**.
- **Slice 5 — the end-to-end test.** `crates/svm/tests/` (depends on both svm-posix and svm-interp): a
  chibicc-compiled C `fork()` program under the manager topology, forking for real, parent-sees-pid /
  child-sees-0, both copies doing libc I/O over the shared memfs. interp==JIT differential. The first
  real program forking on svm.

Key refs: `SRC_FORK_PID` (`clone_caller.rs:276`), `SIBLING_AS_SERVICE` (`svc_serve_loop.rs:477`),
`bind_with_fork` (`svm-posix/src/lib.rs`), `bind_shim` + harness (`crates/svm/tests/c_posix_spawn.rs`),
op 13 compiled-C exec (`c_shell_exec.rs`), `regrant_into_child` (svm-interp).

### 8.2 Increment 2 — the derived mechanism (two findings that settle it)

**Finding A — the durable transform already lowers `cap.call` as `SuspendKind::Leaf`**
(`svm-durable/src/lib.rs:855`): "the host performs the op; the deepest frame reloads its result." A caller
parked on a durable-transformed `cap.call` therefore has *exactly* the Leaf continuation shape, whose thaw
reloads the call's result at the **post-call resume point**. **Fork's reply-injection is nothing more than
supplying that reloaded Leaf result** (§3) — the freeze/flatten path already positions the spill there. So
the caller's domain (and only it) must be durable-transformed; `clone_caller`'s capture reuses the
`flatten_fiber_for_freeze` Leaf spill rather than inventing a new continuation format.

**Finding B — the parked caller *is* a self-contained vCPU (the clean path).** A caller is registered in
`ticket_waiters[(callee_id, ticket)]` in **two** forms (`svm-interp` `Step::Park(CapReply)` at `:4879` and
the generic-arm fiber park at `:8929`):
  - a **root** caller → `Waiter::VCpu(v)` — the *entire* parked vCPU, holding its **own** `v.mem` (window),
    `v.frames` (the continuation sitting inside the pending `cap.call`), and durable state;
  - a **non-root fiber** caller → `Waiter::Fiber { reg, slot }` — frames in a registry whose vCPU is still
    live running other fibers (no bundled window).

The `Waiter::VCpu(v)` form is **self-contained**: `v` carries its own window, so the Leaf flatten runs on
`v` with no cross-vCPU window sharing — and the **increment-1 harness already produces exactly this** (the
parent root parks on the child's offer → `Waiter::VCpu`). So increment 2's restore-in-place step is:

  1. In `clone_caller` (child handler), read `serve_run.ticket` + child domain id; **take** the
     `Waiter::VCpu(v)` out of `sched.ticket_waiters` (via the `sched` handle the eval loop already holds).
  2. **Capture:** drive `v`'s root through `UNWINDING` from its `CapReply` park, delivering the **injected
     reply** as the Leaf reload (`placeholder = Some(injected)`, *not* the O10 freeze placeholder — the
     clone supplies the real reply, which is what dissolves O10, §3). `v`'s continuation spills into `v`'s
     own window as a durable **image**; record its `FrozenFiber`/root-SP residue.
  3. **Restore in place:** thaw `v` under `REWINDING` — it rebuilds from the image, the Leaf reload hands
     back the injected reply, and `v` resumes **past** the `cap.call`. Push `v` to `sched.runnable`.

For the pure no-op this is *visibly* identical to `v.pending = CapResult(injected); runnable.push(v)` — but
it must genuinely round-trip through the **window image** (discard `v.frames`, rebuild from the flattened
window), because that image is exactly what increment 3's **twin** copies into a fresh child. So the test
must assert the image path ran (e.g. the caller's domain is `set_durable(true)` + a durable window, and the
freeze/thaw residue is observable), not merely that the caller got the value.

> **Superseded (2026-07-30):** increment 2 shipped **without** the durable window-image round-trip. Because
> both copies of a `fork()` resume in the **same live run**, the reply is injected through the existing
> `cap_reply_or_stash` path — no freeze/flatten needed on the live path. §8.2 is retained as the design for
> *snapshot/migration* re-issue (a separate concern, §3), not fork. Increment 3 (§8.3) likewise clones the
> **live** parked vCPU directly rather than via a durable image.

### 8.3 Increment 3 — the twin (the deep core, NEXT)

**Topology correction.** The real fork topology is **parent spawns child (bash); child calls the parent's
servicer offer and parks; the parent's handler clones the child** — so the servicer *owns* the twin (it
holds the child's `child_handle`, FORK.md §4). Increment 2's harness is *inverted* (root calls the child's
offer), which is fine for reply-injection (topology-agnostic) but **not** for the twin: the cloner must own
what it clones. Increment 3's harness must put the **caller as a spawned child** and the **servicer as its
parent**, so the parked caller is the child's own vCPU (`Waiter::VCpu(child)`) and the parent can register
the twin as a sibling child it holds.

**The twin, on the live parked vCPU `v` (= `Waiter::VCpu`).** No durable serialization — copy the live
structure. Split into sub-slices by risk:

- **Increment 3a — `Host::fork_powerbox`, the powerbox crux. DONE (PR #531).** A fresh `Host` that copies
  the handle table (own namespace, same values → same bindings) over the same shared `Arc` backings POSIX
  fork shares (regions/pipes + stdout/stderr sinks), new `domain_id`. **Fails closed** on any domain with
  closure host caps (not `Clone`), live offers, module grants, or JIT/ring/serve/freeze state — the
  personality re-wires those (PR 3). Copy-vs-share decided per backing, never silent. Unit-tested.
- **Increment 3b — wire the twin into `clone_caller`. NEXT.** The remaining sub-steps, each with a real
  primitive gap to fill:
  1. **Window deep-copy — a NEW `Mem` primitive is needed.** Both existing builders *share* the backing
     bytes: `fork_for_thread` shares the `Arc<Region>` + address space, `nested_view` shares bytes and
     confines. Fork needs a **private copy** of the caller's window (a fresh `Region` with `v.mem`'s bytes
     memcpy'd in, its own address space) — write `Mem::fork_private()` (snapshot `v.mem`'s window, seed a
     fresh backing). A shared window would make the two copies alias memory — not a fork.
  2. **Continuation:** `twin.frames = v.frames.clone()`, `pending = CapResult(reply_for_twin)`, and a
     **fresh `registry`** (the twin is its own domain, not sharing `v`'s fiber table).
  3. **Register in the scheduler:** build the twin `Box<VCpu>` with a new `TaskId` and push it `runnable`
     (its reply is known at clone time — no second-ticket park needed). Reuse the `thread.spawn`
     vCPU-construction shape; register it as the servicer's child so it holds the `child_handle`.
  4. **Dual reply + return:** reply `reply_orig` to `v` (increment 2's path) and hand `twin` `reply_twin`
     via its `pending`; return the twin's `child_handle`. `clone_caller`'s signature grows a second reply
     arg (or the `fork` op supplies both in PR 3).
  5. **Observability harness (correct topology):** parent spawns child; child calls the parent's servicer
     and parks (→ `Waiter::VCpu(child)`); the parent clones it. Both copies write their reply to the
     **shared stdout sink** (fork shares stdout) so the test sees both `A` and `B` — avoids join-table
     plumbing for the first proof.
- **PR 3 — the `fork` personality op** replies `pid`/`0`, and re-wires the personality's closure caps
  (libc host_fns, fds) into the twin (the part `fork_powerbox` fails closed on).

**Success criterion:** one servicer handler, one caller child; `clone_caller` inside → the original returns
`A` and the twin returns `B` from the **same** call site, both in one live run, interp==JIT.

**Success criterion:** one servicer handler, one caller child that calls it; `clone_caller` inside the
handler → the original returns `A` and the twin returns `B` from the **same** call site, both resuming in
one live run, interp==JIT. **Load-bearing risk:** the powerbox copy-vs-share decision — TDD-first, and a
`Slot`-kind audit before wiring shared backings.
