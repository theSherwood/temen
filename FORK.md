# FORK.md — `fork()`-returns-twice, the durable-clone capstone

The plan for POSIX `fork()` on svm (STAGE1.md item 3 / PROCESS.md §7 / the S11 stage) — **landed
on the tree-walk oracle** (PRs 1–5 + track 2; §5/§8 below are the as-built record). The parked call
transport fork rides is settled as DESIGN.md §12a (was CALLS.md); this file keeps the fork
*semantics* — reply-injection (§3), the child handle model (§4), the clone spec (§6), invariants
(§7), and the fork+wait contract (§8.6) — plus the build log the code comments cite. R8 closure
(durable `call_indirect` to may-suspend targets) was the prereq and is done.

**Backend parity: fork runs on 2 of 4 backends** (the tree-walk oracle **and** the bytecode
interpreter). `OPS_PARITY.md` shows `clone_caller`/`reap` as ✅ on tree-walk + bytecode, 🚧 on
Cranelift (the next slice), ⛔ on the wasm-JIT (leaf) — no longer hidden inside a single `cap.call`
row. The bytecode slice is **done** (§9.2); Cranelift is **§9's remaining track**. See §9.

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

## 5. The PR arc  [LANDED — historical]

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
- **Slice 2 — bind a child's named `fork` import to the live fork offer. DONE.** *Refined from the
  original `__px_`-shim framing.* A compiled-C command is its **own module**, spawned nested via
  `instantiate_module_named` (op 13) — the `c_shell_exec.rs` shape — where imports resolve through
  `bind_child_manifest`, **not** the top-level `bind_shim` libc-handle ABI. That binder's named-offer
  step resolved only wired `offer_func`s (`Binding::Offer`); the fork offer minted by `child_offer` is a
  **live-callee** offer (`Binding::LiveImpl` — a call parks the caller on the server's serve loop), which
  `resolve_offer` never returns, so a named `fork` import could not bind to it (that is *why* the
  hand-written guest used `cap.self.resolve`). The binder now also matches a LiveImpl offer by signature
  (name-less at that layer) and binds the slot; `call.import` then rides the same caller-parking path.
  Proven by `fork_import.rs` (a separate-module guest whose `fork` **import** forks-returns-twice).
- **Slice 3 — libc into a nested child** (blocker D / the real gap). A §14 child spawned by
  `instantiate_named`/op 13 gets an *attenuated* powerbox (instantiator + address space + regranted
  streams/pipes) — **not** posix libc. Existing compiled-C children get libc only because they run as a
  *separate top-level run* via the `set_spawn` delegate (`c_posix_spawn.rs`), which cannot be the
  *parked* caller a fork needs. So a nested fork-guest needs libc in its powerbox: extend
  `regrant_into_child` to carry a **forkable `HostProc`** (re-grant the libc handle, sharing `Inner`), or
  give the child a fresh forkable libc at spawn. This is the load-bearing new interp/posix plumbing.
- **Slice 4 — the compiled-C entry ABI over op 13. DONE.** No adaptation was needed: chibicc's
  `--child-entry` `_start` (the `c_shell_exec.rs` shape) already tolerates the op-13 starter-arg/carve
  convention. The libc face is one wrapper — `long fork(void){ return __fork(0,0); }` — over an extern
  `long __fork(int h, long a)`; chibicc drops the leading `int h` as the cap-handle dummy, so the call
  lowers to `(i64)->(i64)`, matching the fork offer op. (Handle arg must be `int`, not `long`, or the
  emitted `call.sym` handle operand is `i64` and fails verify.)
- **Slice 5 — the end-to-end test. DONE.** `crates/svm/tests/c_fork.rs`: a chibicc-compiled C `fork()`
  program under the manager topology, forking for real — parent sees the twin's pid (3), the twin sees 0,
  both copies `write(1, &slot, 8)` their result to the one shared stdout stream. **The first real
  program forking on svm.** (Interp only, like every `clone_caller` test — the serve substrate is
  eval-loop-only; JIT parity for the fork substrate is a separate track.)

**Status (2026-07-31) — the real-libc capstone has landed; only the chibicc *frontend* remains.**
`crates/svm/tests/fork_manager.rs` is the end-to-end real-libc fork: the manager spawns the server,
mints the fork offer with `child_offer`, then spawns a guest via `instantiate_named` re-granting BOTH
the fork offer *and* the forkable posix libc (slice 3's `regrant_into_child` `HostProc` branch), the
guest resolves both by name, calls `fork()`, and BOTH copies `write(1, &ret, 8)` through the ONE shared
libc memfs — parent-sees-pid (3) / child-sees-0, fork-shares-fds. This closes slices 1+3 as a working
whole and demonstrates the exact wiring a compiled-C `fork()` will use. The guest is **hand-written IR**
resolving caps by name, which is what lets it sidestep the `__px_` import binder. Landing it also needed
one extra interp gate: `can_regrant`/`forkable_host_proc` — op-11's grant-list admission must accept a
forkable `HostProc` as a re-grantable handle (previously only pipes/regions/offers/copyables passed).
**Update (slice 2 done) — the fork *binding* is proven; only the chibicc *frontend* remains.**
`crates/svm/tests/fork_import.rs` swaps the capstone's `cap.self.resolve` fork discovery for a **named
`fork` import** on a **separate guest module** spawned via op 13, bound to the live fork offer by
`bind_child_manifest`. This is the exact runtime path a compiled-C `fork()` takes — the guest is now
hand-written IR only because chibicc hasn't emitted it yet, not because any runtime piece is missing. The
one interp change was the LiveImpl branch in the child-manifest binder (above). Note this also settles
the original slice-2 framing: the nested compiled-C path does **not** go through the top-level `__px_`
`bind_shim` (that is the *same-module top-level* libc-handle ABI); a separate-module command binds
`write`/`read` by the manifest's reference policy and `fork` by the named-offer step.

**Update (slices 4+5 done) — Track 2 is complete: a real compiled-C `fork()` runs on svm.**
`crates/svm/tests/c_fork.rs` compiles an ordinary C `fork()` program with chibicc and forks it for real
under the manager topology. Landing it needed one more interp fix: `Inst::CallSym` (chibicc's lowering
for an extern call) did not probe `import_live_target`, so a symbolic slot bound to a live-callee offer
went to the generic dispatch and answered `-EINVAL` instead of parking the caller — only `Inst::CallImport`
had the §3.6-slice-4 routing. CallSym now carries the same probe (it is "a flat call.import (op 0)"), so a
compiled-C `fork()` parks and returns twice like the hand-written form. All three fork tests
(`fork_manager` real-libc, `fork_import` named-import, `c_fork` compiled-C) pass; the wider `call.sym`
users (`c_shell_exec`, dynlink) stay green.

Key refs: `c_fork.rs` (compiled-C fork), `fork_import.rs` (named-import fork binding), `fork_manager.rs` (the real-libc capstone), `SRC_FORK_PID` (`clone_caller.rs:276`), `SIBLING_AS_SERVICE` (`svc_serve_loop.rs:477`),
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

## 8.6 fork + wait — reaping the twin (2026-07-31)

Track 2 landed `fork()`-returns-twice for compiled C. A shell's command loop is `pid = fork(); … ;
wait(pid)` — so the next verb is **`wait`**: the parent blocks until its twin exits and observes the
exit status. The substrate already had the whole reap primitive — a finished task's `Outcome` is keyed
by `TaskId` in `Sched::results`, and `Blocked::Join { child }` (+ `join_waiters`) parks until that task
finishes and wakes the parked joiner — but the ordinary `join` op (Instantiator op 1) reaps by a
**handle** (a `threads`-table slot), and a fork twin is minted straight onto `runnable` with no handle
in anyone's table. So the twin was un-reapable by the forking guest.

**The verb.** `reap` — a self-namespace op (`CAP_SELF_REAP = 12`, the sibling of `clone_caller = 11`),
served the same way: the fork server serves a **second** offer verb (`wait`) whose handler calls `reap`
on the caller parked on its dispatch. Symmetric with `clone_caller` — no new Instantiator op, no new
grant-graph edge, and the guest stays destitute (it reaches `wait` only through the offer the manager
granted, exactly as it reaches `fork`). Confinement is capability-shaped **plus** a `Sched::forked_twins`
allow-set: `reap` acts only on ids `fork` actually minted, so a bogus/foreign pid is `-ECHILD`, never a
park that hangs.

**Reap ≠ join — a crashing command must not crash the shell.** `join` propagates a child trap as the
joiner's own trap (`out.result?`). `wait` must not: a trapped twin reaps as a nonzero *crash status*
(`reap_status`; the exact POSIX `128 + signal` encoding is a shell/guest concern, ISSUES.md I43). This
is the one real semantic difference, and it is why `reap` is its own op rather than a rebind of `join`.

**Two paths.** `reap_parked_caller` claims the parked caller (the shape `fork_parked_caller` removes),
then: twin already finished → deliver its status now (`CapResult`); twin still running → move the caller
into `join_waiters[pid]` with `Pending::ReapPid`, and the twin's completion (the generic join-wake)
resumes it with the status. `reap_twin` takes the outcome *and* retires the id from `forked_twins`, so a
second `wait(pid)` is `-ECHILD`, never a re-park.

**The serve/park race — `wait` retries on `-EAGAIN`, like `fork`.** `svc_enqueue` makes a dispatch
visible and wakes the server *before* the caller registers its `CapReply` waiter, so under load a
servicer can drain the dispatch before the caller parks. `fork` already fails such a race with `-EAGAIN`
(pid mode), and the guest retries. `reap` faces the **same** race and must not confuse it with a real
`-ECHILD`: `ReapOutcome::Retry` (twin exists in `forked_twins` but no waiter registered yet) → `-EAGAIN`;
`ReapOutcome::NoChild` (unknown pid) → `-ECHILD`. The realistic shell idiom `while ((s = wait(pid)) < 0);`
retries the transient failure and converges — which is exactly why the `clone_caller.rs` fork+wait test
is stable under a full parallel suite where a one-shot form would flake (the I53 family).

**Proven.** `clone_caller.rs::fork_then_wait_reaps_the_twins_exit_status_through_the_shared_offer` — one
server serves `fork` + `wait` over one offer; the caller forks, the twin `exit(42)`s, the parent
`wait(3)`s (the deferred path — it waits the instant it forks), reaps `42`, and the run returns it.
Interp only, like every fork test.

**Compiled-C fork → exec → wait, end to end. DONE.**
`c_fork.rs::a_compiled_c_program_runs_fork_exec_wait_end_to_end` — an ordinary chibicc-compiled C
program runs the shell's command loop `pid = fork(); if (pid == 0) exec(cmd); else wait(pid);`. The
manager serves **two** offers (`fork` → func 2 `clone_caller`, `wait` → func 3 `reap`) over two exports
and re-grants both to the guest as named imports (`__fork`, `__wait`); the guest gained the `__wait`
import (the "wiring through a compiled-C `sh_spawn`" item). **`exec` is BusyBox-multicall applet
dispatch** (STAGE1.md — "a shell's `exec` and BusyBox's applet dispatch are the same shape"): the child
transfers to the selected command entry and *becomes* it, exiting with the command's status; the status
`42` originates in the exec'd command, flows through the child's exit into `results[twin]`, and is reaped
by the parent's `wait`. Both `fork` and `wait` retry on the `-EAGAIN` serve/park race (I68 idiom), so the
demo is stable under load. Interp only, like every fork test.

**True cross-module `execve` (image-replace) — substrate DONE.** `exec_module(module, grants_ptr,
grants_n, entry, size_log2)` — a self-namespace op (`CAP_SELF_EXEC = 14`) — replaces the **calling
vCPU's own image** with a granted *separate command module*, in place, keeping the vCPU's `TaskId` +
fuel. So a parent's `wait(pid)` reaps the *command's* exit, exactly as POSIX `execve` keeps the pid —
the real image-replace the multicall demo above stood in for. Mechanism: the eval loop does everything
fallible (resolve the command, regrant the inherited caps named in the grant list into a fresh
powerbox, bind the command's imports, register its self module) — where a refusal is a clean `-EINVAL`
that leaves the caller running (POSIX `execve` **returns only on failure**) — then hands `dispatch` an
`Inner::Exec`/`Step::Exec`. `dispatch` materializes the command's data segments into the caller's window
and swaps the vCPU to a fresh one (`VCpu::new`) running the command at its entry, then loops back to run
it (no run-queue round trip, the page-fault-fast-lane shape). Proven by
`svm-interp/tests/execve.rs`: a guest execs a separate command module that writes `"EXEC"` and exits
`42`; the guest's post-`exec` `return 99` **never runs** (the image was truly replaced), and a bogus
command handle returns `-EINVAL` with the caller still running. Interp only, like every fork test.

*Increment-1 simplifications (all `-EINVAL`-refused, so nothing silently degrades):* the command reuses
the caller's window (its declared memory must **fit** it — see below) and BSS is not zeroed (a well-formed
command inits its own state from its data segments); non-durable domains only (durable freeze/thaw
image-swap is the deferred capstone); and only from a clean root computation (no serve handler / active
fibers).

**A command whose declared window ≥ the default is `exec`-able (#773).** The exec (and the §14 spawn)
originally required the command's declared memory to **equal** the caller-supplied carve; both the shell
shim and the test managers hardcode that hint at `17` (128 KiB), so any command whose static/BSS pushed
its window directive past 128 KiB (a `cat` block buffer, a shell line buffer) was refused — and a manager
that then `join`ed the failed spawn **wedged** (a negative errno `& mask`ed onto a live slot). The fix: a
child runs in a window **at least** its declared memory (a larger window is a safe superset — confinement,
invariant 2, still masks every access to the *actual* window; only an out-of-declared-window access, a
guest bug, wraps at the larger bound). Concretely, `exec_module` now bounds the command against the
caller's **actual inherited window** (`GuestMem::window_size`), not the guest's `size_log2` hint (now
advisory), and op-13 admits a module into any carve `≥` its declared memory. Defense-in-depth: a negative
"handle" to `join`/`poll`/`detach` now faults (invariant 5) instead of masking onto a live child, so a
genuinely-too-small spawn surfaces observably rather than hanging. Proven by
`c_fork.rs::a_shell_execs_a_command_with_a_large_bss_buffer`: a shell spawned into a 256 KiB window forks
a child that execs a `memory 18` command; the command touches the far end of its 60 KB buffer (only
reachable in the enlarged window) and writes from it, and the shell reaps its exit.

**Compiled-C `fork → execve → wait` with a *separate* command module. DONE (increment 2).**
`c_fork.rs::a_compiled_c_program_runs_fork_execve_wait_with_a_separate_command` — an ordinary compiled-C
guest forks; the child resolves a re-granted command module `"cmd"` and the inherited `"stdout"` by name
(`__vm_resolve`), builds a grant list, and `__vm_exec_module`s into the command; the parent `wait`s and
reaps the command's exit. The **separate command module** writes `"EXEC"` and exits `42`, so the sink
holds exactly `"EXEC"` (a *different program* did that I/O as the child's task) and the run returns `42`.
Three enablers made it work, each a small correctness fix in its own right:

- **chibicc builtins** `__vm_exec_module` (lowers to the `CAP_SELF_EXEC` self-op) and `__vm_resolve`
  (`cap.self.resolve`) — the C-level `execve` primitive and the named-cap-handle reader.
- **Modules are re-grantable into a child** (`can_regrant`/`regrant_into_child` + `ModuleGrant: Clone`):
  a shell hands a command module to a child it will `execve`. And **`fork_powerbox` carries modules**
  (was fail-closed on a non-empty module table) — a shell that holds command modules can now fork.
- **`Mem::snapshot` is window-base-relative** — a real `fork_private`-of-a-nested-carve bug: `byte()`
  indexes the shared backing *absolutely*, so the twin previously inherited **zeros** for its
  high-offset globals (past `POWERBOX_ARGS_END`) instead of the parent's data. Latent until a nested
  twin *read* pre-fork high-offset data — which the `execve` child (resolving `"cmd"`/`"stdout"`) is the
  first to do.

`__vm_exec_module` is also proven in isolation from fork by
`a_nested_compiled_c_guest_execs_a_separate_command` (a nested op-13 guest execs a command, no fork/wait).

**`waitpid(pid, WNOHANG)` — the non-blocking poll. DONE.** `reap` now takes an optional second
`flags` arg; `WNOHANG` (bit 0) makes a still-running twin reply **`0` at once** — POSIX's "no child
changed state" — instead of parking (`reap_parked_caller`'s `nohang` branch re-admits the caller
immediately and leaves the twin reapable). The already-finished and unknown-pid / serve-race paths are
unchanged (status / `-ECHILD` / `-EAGAIN`). Proven by
`clone_caller.rs::waitpid_wnohang_returns_zero_for_a_still_running_twin_without_blocking`: the twin
**parks forever** (`atomic.wait`, timeout `-1`) so a blocking `wait` *would* hang — `WNOHANG`
deterministically returns `0`. Stable 0/40 under stress.

**Run a real command end-to-end — a `cat` reading a real file. DONE (increment 3).** The command the
child `execve`s is no longer a trivial `write("EXEC")` stub: it `open`/`read`/`close`s a file from a
granted **`vm_fs` capability** (the shared in-memory filesystem, `crates/svm-fs`) and echoes the bytes
to stdout — real file I/O by a real separate program running as the child's task. Two increments, both
in `c_fork.rs`:
- **3a (isolation, no fork):** `a_nested_compiled_c_command_reads_a_file_through_a_granted_fs_cap` — a
  nested op-13-spawned guest `execve`s the `cat` command; the manager re-grants `{stdout, cmd, vm_fs}`
  and the guest carries `{stdout, vm_fs}` into the exec grant list. Proves the fs-cap-through-exec
  plumbing on its own.
- **3b (the full loop):** `a_compiled_c_program_forks_execs_a_real_command_that_reads_a_file_and_waits`
  — `fork()` → `execve(cat)` → `wait()`, all compiled C, with the manager re-granting **five** caps
  (fork/wait offers, stdout, cmd, vm_fs). The parent reaps the `cat`'s exit (the byte count).
- **The binding half (the one svm-interp change):** `bind_child_manifest` now binds a compiled-C
  `call.sym "vm_fs"` (chibicc's `__vm_fs`, op-in-arg0) to a **`HostProc`** re-granted to the child by
  name. A raw host cap carries no typed interface, so there's no coverage walk — a flat import binds
  op-0 straight to the handle, and the `CAP_IMPORT_TYPE_ID` translation routes the call (with the fs op
  in `args[0]`) to the registered closure. This is the same op-in-arg0 wrapper `c_link.rs` uses at the
  top level, now reachable through the op-13 grant list so a spawned/exec'd child inherits fs authority.
  The cap is granted **forkable** (`grant_host_proc_forkable`), the shape `regrant_into_child` re-mints
  into a child, so the memfs store is shared across the fork.

**`waitpid(-1)` — reap *any* child. DONE.** The servicer-side `reap` now special-cases `pid == -1`
(`reap_any_parked_caller`): instead of a named twin it ranges over the whole `forked_twins` set —
reaping any already-finished twin at once, else parking the caller in a new FIFO `reap_any_waiters`
queue that the **next** twin to finish wakes (via `Pending::ReapPid`, the same resume the named wait
uses). A finishing twin prefers a named `wait(pid)` waiter first (`join_waiters`), falling through to
the any-child queue only when unclaimed, so the two waits compose. Empty `forked_twins` → `-ECHILD`
*before* claiming the caller, so a shell's wait loop terminates deterministically (no serve-race
`-EAGAIN` when there is nothing to wait for). The teardown sweeps drain `reap_any_waiters` alongside
`join_waiters`. Proven in `c_fork.rs`: `a_compiled_c_program_reaps_two_children_with_waitpid_minus_one`
(fork two children exiting 3/4, reap both with `wait(-1)`, sum = 7 regardless of order — stable 40/40
under stress) and `waitpid_minus_one_with_no_children_is_echild`.

**Per-parent child scoping — `wait` reaps only a domain's own children. DONE.** `forked_twins` is now a
map `twin → parent domain` (`domain_key_of` the forking caller, recorded in `fork_parked_caller`)
rather than a bare set. Every reap path is scoped: `reap_parked_caller` (`wait(pid)`) refuses a twin
whose recorded parent is not the calling domain (`-ECHILD`, not a foreign reap); `reap_any_parked_caller`
(`wait(-1)`) ranges only over twins of the calling parent, and `-ECHILD`s when that parent owns none
even though other parents' twins exist; and `wake_reap_any` wakes only a parked `wait(-1)` caller whose
domain is the finishing twin's parent. So the shared fork/wait offer no longer lets one shell reap
another's child. The peek-then-claim ordering keeps the serve/park race retryable (`-EAGAIN`) while
making cross-parent reaps deterministic `-ECHILD`. Proven by
`c_fork.rs::wait_only_reaps_a_domains_own_children`: a guest forks a child; the **child**'s `wait(-1)`
gets `-ECHILD` (the global twin set is non-empty — it holds the child itself under the *parent's* key —
so without scoping this would deadlock), and the parent reaps only its own child. Stable 40/40 under
stress.

**Process groups — `setpgid` + `waitpid(-pgid)`. DONE.** Job control's grouping primitive, built on
the per-parent table. Each twin's [`Twin`] record now also carries a `pgid` (POSIX process group),
defaulting to the twin's own id at fork — every child starts its own group leader. `setpgid(pid, pgid)`
is a **direct self-op** (op 15) the *parent* drives — the caller *is* the parent, so its own
`domain_id` scopes the change with no serve round-trip (unlike `reap`, which needs the servicer to
reach the parked caller); it retargets a child's `pgid` (`pgid == 0` → the child's own id), confined to
real children of the caller (`-ESRCH` otherwise). `reap` grew a group form: `reap_any_parked_caller`
became `reap_group_parked_caller(…, target: Option<TaskId>)` — `None` for `wait(-1)` (any child),
`Some(pgid)` for `waitpid(-pgid)` (any child in that group) — and the parked-waiter queue carries the
target so a finishing child wakes only a waiter of its parent whose group it matches. The `waitpid` pid
selector now decodes POSIX fully enough for a shell: `-1` any child, `< -1` the group `|pid|`, `> 0` a
named twin. Exposed to compiled C as `__vm_setpgid(pid, pgid)` (chibicc builtin → op 15) alongside the
existing `__fork`/`__wait`. Proven in `c_fork.rs`: `setpgid_groups_children_and_waitpid_reaps_the_group`
(fork A/B, `setpgid` B into A's group, `wait(-a_pid)` reaps both → 30; a failed move would sum 0) and
`waitpid_by_group_does_not_reap_other_groups` (the dual — `wait(-a_pid)` reaps only A, then `-ECHILD`,
while B waits in its own group). Both stable 30/30 under stress.

**Shell viability — a real command-dispatch loop runs on the surface.** With the process model in
place we pointed a shell at it (a compiled-C **microshell**, not the Instantiator-spawn Stage-0 shell):
a reusable `run(name, arg)` that **forks**, has the child marshal an `argv` from runtime strings into
the §3e args buffer, **resolves the command module by name** (dynamic dispatch — the shell's PATH is
the name→module grant map), **`execve`s** it, and the parent **`waitpid`s** the status; `main` runs two
*different* named commands in sequence and sums their exits. It runs end to end
(`c_fork.rs::a_microshell_dispatches_two_named_commands_through_fork_exec_wait` → 107). The pivotal
enabler is that **`execve` delivers argv**: the image-replace preserves the caller's args window, so a
fork twin seeds `{argc, packed argv}` before `execve` and the command reads `argv[1]`
(`execve_delivers_argv_to_the_command`). Nothing in the core loop broke — fork, resolve-by-name, argv
marshalling, image-replace, and repeated fork→exec→wait all compose.

**env delivery through `execve`. DONE.** The gap the microshell surfaced. A `main(argc, argv, envp)`
entry (3 params) now makes chibicc's `_start` also parse the `envc` env strings — which follow the argv
strings in the §3e args buffer — into an `envp[]` pointer array placed right after `argv[]`, passed as
`main`'s third argument (`codegen_ir.c`: `needs_envp` + the env loop, blocks 6–10 of the arg-parsing
`_start`). A 2-param `main(argc, argv)` is byte-identical to before (the env path is behind
`needs_envp`). No interpreter change: `execve` already preserves the caller's args window, so a fork
twin seeds `{argc, envc, packed argv+env}` and the exec'd command reads `envp`. Proven by
`c_fork.rs::execve_delivers_the_environment_to_the_command` (a forked child seeds `env={"V=7"}`, the
command returns `envp[0][2]='7'`=55). A libc `getenv` walking `envp`/`environ` is the guest-side
follow-up (a shim, not a substrate concern).

**What breaks / is still missing** (the honest gap list from that experiment, shell-relevance order):
- **~~argv/env-seeding ergonomics~~ — a process libc shim. DONE.** `crates/svm/tests/fork_shim.c` is the
  guest-side layer a shell links so it writes the idiomatic loop —
  `pid = fork(); if (pid == 0) execvp(cmd, argv); else wait_pid(pid);` — over a NUL-terminated `argv[]`,
  instead of hand-marshalling the buffer: `fork`/`wait_pid`(`pid`/`-1`/`-pgid`)/`setpgid`/`execvp`
  (packs argv **and** `environ` into the §3e buffer, resolves the module by name, inherits `stdout`,
  image-replaces) + `getenv`/`strlen`. Every entry point is **`static inline`**, so chibicc's dead-code
  pass drops the ones a program doesn't call — a command that only reads its env pulls in `getenv`, not
  `fork`/`execvp` and their `__fork`/`__wait` offer imports (which it was never granted; a plain
  `static` non-inline function is a liveness *root*, so `inline` is load-bearing here). Proven by
  `c_fork.rs::a_shell_linking_the_process_libc_runs_execvp_with_argv_and_env` (a shim shell `execvp`s a
  command with argv + env; the command reads both via `argv` and `getenv` → 107). A `$PATH`-dir scan
  (below) and `pipe`/`dup2` are the remaining libc gaps.
- **PATH semantics.** Command lookup is a name→module registry by exact name (the grant map), not a
  `$PATH`-dir `stat` scan — fine as *a* PATH, an impedance mismatch for `execvp("/bin/ls")`.
- **~~pipes between forked children~~ (`cmd1 | cmd2`). DONE — sequential *and* concurrent.** A
  guest-reachable `pipe()` self-op (op 16) mints a host-served FIFO into the shell's own powerbox and
  hands back the two `Stream`-typed ends (`fds[0]` read / `fds[1]` write, POSIX order); the shim's
  `exec_io(file, argv, out, in)` re-grants a pipe end to a stage's `stdout`/`stdin` by name. Because a
  `PipeEnd`'s FIFO backing aliases across `execve` (and across fork), two commands' **plain
  `write(1)`/`read(0)`** connect transparently — neither knows a pipe is there.
  - **Concurrent (both stages live) — the read *blocks*.** A `read` of an empty FIFO parks the reader
    (`Blocked::PipeRead`, keyed by pipe id) while any write end is open, and a producer's `write` — or
    the **last write end closing** — wakes it; the woken read is rewound, so it re-executes and drains
    the new bytes or EOFs. EOF is a POSIX **writer refcount** on the shared backing: bumped when a write
    end is minted / re-granted into a child / fork-copied, dropped on explicit `close(fd)` and when a
    domain execs or exits — so a fork-inherited write end holds the pipe open across the fork→exec gap,
    race-free against the shell closing its own ends. The shim gained `close(fd)` (`__vm_close` → op 2);
    a shell closes its copies of the ends after forking the stages, leaving the producer the last
    writer. Proven by `c_fork.rs::a_shell_runs_a_concurrent_pipe_with_a_blocking_read`: the shell forks
    both stages and only then waits; the producer *burns a compute loop first*, so the consumer parks on
    the empty pipe and is woken by the write (a non-blocking read would have EOF'd to empty output).
    Stable 20/20 under stress.
  - **Sequential** (`a_shell_pipes_the_output_of_one_forked_command_into_another`): fork producer,
    `wait`, `close` the write end, fork consumer — the buffered bytes drain, then EOF.
  - Interp-only (the park lives in the eval loop; the JIT/bytecode tiers don't block a pipe read, so a
    differential guest must `close` the write end before an empty read — see `pipe.rs`).
  - **~~SIGPIPE + backpressure~~. DONE — the write side made symmetric to the read side.** The FIFO is
    now a **bounded buffer** (`PIPE_CAP` = 64 KiB, Linux's default) with two write-side contracts:
    - **SIGPIPE (`-EPIPE`).** A **read**-end refcount (the third `Arc` in `PipeBacking`) mirrors the
      writer refcount, bumped/dropped at every read-end lifecycle point (mint / re-grant into a child /
      fork-copy / explicit `close` / exec / teardown). A `write` to a pipe whose read count is `0`
      returns `-EPIPE` — the SIGPIPE-ignored contract: a producer whose consumer has quit fails its next
      write instead of piling into a FIFO nobody drains.
    - **Backpressure (blocking write).** A `write` that would overflow `PIPE_CAP` **parks** the writer
      (`Blocked::PipeWrite`, keyed by pipe id) — the exact write-side twin of the read park: a `read` that
      drains a full pipe (room opened) or the last read end closing (→ `-EPIPE`) wakes it; the woken write
      is rewound and re-executes. A partially-full pipe short-returns (the freed room), so a well-behaved
      writer loops. This closes the "the FIFO is unbounded" hole: a runaway producer (`yes | head`) is
      bounded to one 64 KiB buffer, not host RAM.

      Proven by `c_fork.rs`: `a_producer_gets_epipe_when_its_consumer_exits` (the `yes | head` story — the
      producer blocks as the pipe fills, resumes as the consumer drains, and gets `-EPIPE` = returns 88
      the moment the consumer's read end closes, rather than spinning to its safety cap) and
      `a_full_pipe_write_is_bounded_to_the_capacity` (a 100 000-byte write short-returns exactly 65 536 —
      the FIFO never grows past the bound). Interp-only, same reason as the read park.
- **~~file redirection~~ (`cmd > file`). DONE — as a pump over the pipe + fs substrate, no new mechanism.**
  The shell wires the command's stdout to a pipe **write** end (`exec_io(cmd, argv, fds[1], 0)`), drops its
  own copies of both ends (so `cmd` is the sole writer), then — instead of forking a second stage — *is*
  the reader: it loops a **handle-specific** blocking read on the pipe read end and forwards each chunk to
  a memfs file through the granted `vm_fs` cap (`FS_WRITE`), until the read EOFs (which arrives when `cmd`
  exits and the writer refcount hits 0). Two new frontend builtins back the pump: `__vm_read(h,buf,len)` /
  `__vm_write(h,buf,len)` (`cap.call 0 0|1`) read/write a *specific* Stream handle — the plain `read`/
  `write` builtins always hit the ambient stdin/stdout, so a shell holding a pipe fd needs these to drain
  it. The shim exposes them as `read_fd`/`write_fd`. Because `__vm_read` is a **direct `cap.call`** (not a
  `call.sym` offer), the pipe-read park was added to the direct-`CapCall` eval arm too (it had only been on
  the `CallSym` route). Proven by `c_fork.rs::a_shell_redirects_a_command_output_to_a_file`: `cmd` writes
  `"redirected!"` to its stdout, the shell pumps all 11 bytes into `out.txt`, and the shared memfs snapshot
  holds exactly that file (the shell's own stdout stays empty — the output never touched it). Interp-only,
  same reason as pipes (the read parks in the eval loop).
- **~~the rest of redirection~~ (`>>`, `<`, `2>`). DONE — all the same pump over the existing pipe + fs +
  stream-by-name primitives, still no new mechanism.**
  - **`cmd >> file` (append).** Identical pump to `>`, only the shell's `FS_OPEN` flags gain `O_APPEND`
    (`O_CREATE|O_WRITE|O_APPEND`), so the output lands after the file's existing bytes rather than
    truncating. Proven by `a_shell_appends_a_command_output_to_a_file` (a seeded `log.txt` = `"existing\n"`
    ends up `"existing\nredirected!"`).
  - **`cmd < file` (input).** The **reverse** pump: the shell `FS_OPEN`s the source `O_READ`, reads it in
    chunks, and writes each into a pipe whose *read* end is the command's `stdin` (`exec_io(cmd, argv, out,
    fds[0])`); draining the file then closing the write end EOFs the command's `stdin`. The command is the
    consumer, the shell the producer — the concurrent-pipe park/wake, driven from the shell side. Proven by
    `a_shell_redirects_a_file_into_a_command_stdin` (`in.txt` bytes arrive on the shell's stdout via a `cat`
    that echoes stdin→stdout).
  - **`cmd 2> file` (stderr).** The command writes normal output with the ambient `write` builtin (always
    `stdout`) and diagnostics to a **distinct** `stderr` handle it resolves by name (`__vm_write`); the shim
    grows an `exec_io3(file, argv, out, in, err)` that adds a `"stderr"` grant, and the shell pumps that end
    to a file exactly like `>`. Proven by `a_shell_redirects_a_command_stderr_to_a_file`: stdout stays on the
    shell's stdout while `err.txt` holds only the stderr bytes — the two streams land in different places.
- **signals L1/L2** (async `SIGINT`/`SIGCHLD`) and a **stdin line reader / tty** — both parked.
- **interp-only.** The serve substrate is eval-loop-only, so the whole fork/exec surface is tree-walk
  only (no bytecode/JIT/wasm) — the §9 backend-parity track.

**Remaining lower-level items:** the increment-1 exec simplifications as they're needed (fresh window +
BSS zero, durable-domain exec, exec from a nested serve context). `WUNTRACED`/`WCONTINUED` are **done**
(#798: stop/continue signals exist — the personality's `waitpid` reports fresh stops/continues from its
process table; the core's contribution is the domain stop park, `Blocked::Stopped`). With
`fork`/`execve`/`wait`(`pid`/`-1`/`-pgid`)/`setpgid` in place and a microshell running on them, the
core process-model surface a shell drives is complete.

## 9. Fast-backend fork parity — bytecode DONE, Cranelift next

Fork is a real parity gap we intend to close, not a by-design fold (INVARIANTS.md #9: "very few
gaps we don't want to close" — this is not one of them). It runs on the tree-walk oracle **and now the
bytecode interpreter** (§9.0); Cranelift is the remaining fast backend and the wasm-JIT folds every
cap op by design. This section is the convergence plan and the as-built record.

### 9.0 Where it stands (2026-08-07)

- **Honest matrix.** The process/serve/fork ops are their own `OPS_PARITY.md` family
  (`process, serve & fork`), classified per-backend by `svm-parity`'s `parity_capcall` — instead of
  hiding inside the one `cap.call` row that (wrongly) read ✅✅✅. `clone_caller`/`reap` show ✅ on
  tree-walk + bytecode, 🚧 on Cranelift, ⛔ on the wasm-JIT (leaf accelerator — it folds *every* cap
  op by design, so fork stays ⛔ there like `cap.call`). `svc.poll`/`svc.wait` are ✅ on the fast
  backends (native serve loop when serve-qualified); the Instantiator spawn/join ops are ✅ too.
- **Native bytecode fork — DONE.** `clone_caller` + `reap` run natively on the bytecode cooperative
  serve driver. `clone_caller`/`reap` are a `has_fork` seam that `svc_park_veto` keeps (so svm-run's
  Cranelift routing still folds fork), while the bytecode compile gate takes a bounded
  `Seams::bytecode_serves_fork` escape past it — the per-backend split. The twin is the parked
  caller's `Vm` cloned at its post-call resume point over a private window (`Mem::fork_private`) +
  duplicated powerbox (`Host::fork_powerbox`), pushed as a fresh env+task; `reap` waits it via a
  `BlockedReap` state + a `forked_twins` allow-set. Pinned bit-for-bit against the oracle by
  `clone_caller.rs::bytecode_forks_the_twin_identically_to_the_oracle` (the `SRC_TWIN` topology).
  The Cranelift side stays folded, pinned by `serve_qualifies_still_folds_fork_for_cranelift`.

### 9.1 The gating finding (verified 2026-08-07): fork is a *serve-substrate* slice, not a targeted op

`clone_caller`/`reap` are not an isolated add. The **real fork topology** — a manager that
`instantiate`s a server + a guest, the server serving `svc.wait`, the guest calling `fork()` — trips
three coupled `scan_seams` conditions at once: `has_svc` (the server's `svc.wait`), `has_instantiate`
(the manager's spawns), and the fork self-ops. The shared serve-qualification veto
(`bytecode::svc_park_veto`) already folds any module where `has_svc` coexists with `has_instantiate`
(a handler that spawns/joins could park mid-dispatch), so a forking module folds to the oracle
**before** the fork ops matter. **Closing the gap therefore means making serve + spawn +
caller-parking + fork all run natively together**, not lowering one op. This is the I36/I37
serving-substrate track — and the reason FORK.md §8.4/§8.5 called fast-backend fork "a separate
track." (History note: fork substrate is where the I68 lost-wakeup race lived; rushing it is exactly
what produced that — so this track is TDD-first, differential-pinned, small increments.)

**Order: bytecode first, then Cranelift; wasm-JIT never** (it is a leaf accelerator — DESIGN §3 —
and folds every cap op by design). Bytecode is far closer: `Mem::fork_private` and
`Host::fork_powerbox` are engine-agnostic and already exist; and the load-bearing continuation copy
is cheap because the bytecode vCPU (`Vm`) already derives `Clone` — a parked caller is a bare root
`Vm` at its post-call resume point, so cloning it *is* the twin's continuation.

### 9.2 What the bytecode engine actually needs (the mechanism, reverse-engineered)

The bytecode engine has its **own** scheduler and vCPU, distinct from the tree-walk `Sched`/`VCpu`
the oracle's `fork_twin`/`fork_parked_caller` operate on. So this is a *parallel* implementation over
the bytecode structures, not a reuse of the oracle's scheduler methods:

- **Parked caller.** A live-offer call parks the caller as a cooperative-driver task in
  `TaskState::BlockedTicket { ticket, callee, dst }` (`bytecode.rs` `drive`); its continuation is the
  task's `VTask.active: Vm`, positioned past the `cap.call`, with `dst` the reply slot. The settle
  scan wakes it by `svc_results[ticket]` → `active.set(dst, reply)` → `Runnable`.
- **Serve linkage.** The serve driver already carries `serve_ticket` (the `ServeRun.ticket` analog)
  while a handler runs. That is the handler→caller linkage, already present.

The slices, **all landed for the cooperative driver** (each pinned against the oracle):

1. **`Op::CloneCaller`/`Op::Reap` + driver surface. DONE.** `cap.call CAP_SELF 11/12` compile to
   native ops that surface to the cooperative `drive` via `Outcome`/`VcpuStop::CloneCaller`/`Reap`
   carrying the reply/pid args + dst — the `LiveCall`/`SvcWait` shape. The driver reads the running
   handler's `serve_ticket` (on the task's `Vm`) to name the parked caller.
2. **The twin (cooperative driver). DONE.** In `drive`, find the `BlockedTicket` caller for
   `serve_ticket` (matched by `(ticket, server host Arc)`); build a twin **env**
   (`fork_private(caller_env.mem)` + `fork_powerbox(caller_env.host)`, a new `extra_envs` entry) and a
   twin **task** whose `VTask` is the caller's `Vm.clone()` with `active.set(caller_dst, reply_twin)`;
   deliver `reply_orig`/twin-pid to the original. No `replied` flag is needed — the woken caller is
   `Runnable`, so the handler's later `svc_results` write is never claimed (harmless). Fail-closed to a
   single reply on any non-bare caller (the oracle's degrade), so it never diverges.
3. **`reap` + the veto split. DONE.** Reap over the driver's `forked_twins` allow-set: deliver the
   twin's `reap_status` now (if `Done`) or park the caller in a new `BlockedReap { pid, dst }` state
   that the settle scan wakes on twin-exit. `clone_caller`/`reap` are a `has_fork` seam that
   `svc_park_veto` **keeps** (Cranelift folds), with the bounded `Seams::bytecode_serves_fork` escape
   admitting the fork shape on the bytecode gate only — the per-backend split, the load-bearing
   correctness step. (Also relaxed the §3d op-17 pager guard for the fork shape, since the fork
   topology spawns with grants via op-17 records under a serving module.)
4. **Differential pin. DONE.** `clone_caller.rs::bytecode_forks_the_twin_identically_to_the_oracle`
   runs the `SRC_TWIN` topology on the bytecode engine (natively, not folded) and asserts the run
   value + both replies on the shared sink match the oracle. **Remaining:** port the driver arms to
   `drive_parallel` (fork currently runs on the cooperative single-threaded driver; the parallel
   driver fails closed), and add fork shapes to the `bytecode_diff` fuzz corpus.

### 9.3 The Cranelift capstone (design, 2026-08-07 — grounded in a full JIT durable map)

Cranelift fork does **not** "follow the same arc" as bytecode — that earlier framing was wrong. The
bytecode twin works because a parked caller is a **reified `Vm` (Clone-derived)** at its post-call
resume point. On the JIT there is **no reified continuation to clone**: `cap.call` is a synchronous
host thunk, so a caller mid-call is a **live native OS-thread stack** — either running the handler
inline (handoff) or thread-blocked on the `live_impl_call` Condvar (`svm-run/src/lib.rs:2270`). The
JIT's durable machinery reifies a continuation into shadow-stack bytes **only at a poll safepoint,
which is always *past* a completed `cap.call`** (a `SuspendKind::Leaf` spills + reloads the
*already-returned* result; `svm-durable/src/lib.rs:888,929`). There is no `Blocked::CapReply` state
on the JIT at all (`svm-jit/src/fiber_registry.rs` has no cap-park concept), and `freeze_drive`
(`svm-jit/src/fiber_rt.rs:1121`) walks only voluntarily-suspended `RUNNABLE` fibers, whole-run.

So the capstone is **not** DURABILITY §10 "clone at a quiescent point" (cheap snapshot/restore) — a
forking caller is *not* quiescent. It is: **make a JIT `cap.call` a suspendable, pre-result durable
safepoint**, so a forking caller unwinds to a reified continuation (shadow-stack bytes in the window)
instead of thread-blocking. The snapshot format is already engine-agnostic and cloneable
(`svm-snapshot`, magic `SVMD`), and the interp's live `fork_parked_caller` is the semantic oracle —
so the real work is turning the JIT call into a reifiable park. The four items, in dependency order:

1. **The inject-vs-reload distinction is *runtime*, not a new compile-time `SuspendKind`** (refined
   2026-08-07). A `cap.call` compiles once; the transform cannot know at compile time whether a given
   call will be a normal return (reload the host's result) or a fork (inject a per-copy reply). So do
   **not** add a `LeafInject` kind. Instead the existing `Leaf` reload stays the mechanism — it reloads
   the result from the spill slot — and **fork writes the injected reply into that slot before thaw**.
   The gap is therefore not "a new suspend kind" but item 2: getting the caller to reach a *reified,
   pre-result* park at the fork call so the slot exists to write. (The interp does the runtime version
   of exactly this — `pending = CapResult(reply)` on the live vCPU.)
2. **Suspendable `cap.call` on the JIT — the load-bearing re-architecture** (`svm-jit` lowering +
   `svm-run` serve path). *Why it is unavoidable:* in **both** live-offer transports the caller's
   continuation is a **native Rust frame**, unreifiable — the enqueue path thread-blocks the caller on
   the `live_impl_call` Condvar (`svm-run:2270`), and the handoff path runs the handler on the caller's
   own thread with the caller's guest continuation suspended *below* it on the native C stack. A servicer
   in another frame/thread cannot reify either. The fix: a live-offer `cap.call` from a durable guest,
   when the reply is withheld, must **durable-unwind the caller's shadow stack (pre-result) back to the
   window and return control** — parking the guest as a reified cap-reply-pending continuation — instead
   of ever entering the native thread-block. This is caller-side parking on the JIT (the I36 slice never
   built for the JIT), realized via durable unwind. It is a **new JIT execution mode for cross-domain
   calls**, the sensitive change (handoff fast path + confinement-adjacent serve loop); gate it to
   durable-instrumented forking guests so ordinary cross-domain calls keep the thunk fast path.
3. **A targeted (single-continuation) freeze on the JIT.** `freeze_drive` is whole-run today; fork
   freezes only the caller. Add a single-vCPU freeze entry producing that caller's image.
4. **Return-twice.** Clone the caller's image + window (`fork_private`) + powerbox (`fork_powerbox`),
   thaw two copies injecting `reply_orig`/`reply_twin` (the snapshot format already carries the
   per-copy residue). This is the JIT analogue of `fork_parked_caller`, layered on 1–3.

**Scope honesty:** item 2 re-architects a core JIT execution path (cross-domain `cap.call` becomes a
durable-suspendable, caller-parking op), so this is a multi-PR capstone touching the durable transform
(the R8 fork-critical instrumentation) and the confinement-adjacent serve path — built TDD-first,
differential-pinned against the interp, one increment per PR. It is **not** a bounded slice with a
safe independently-testable first primitive: items 3–4 are only exercisable once item 2 gives a
reified mid-call continuation, and item 2 itself is the from-scratch execution-mode change. Until the
capstone lands, `serve_qualifies` correctly folds fork for Cranelift — no divergence, a forking module
runs on a reifiable tier. The first PR is item 2's foundation: a durable guest's live-offer `cap.call`
that unwinds pre-result to a window-resident continuation instead of thread-blocking, pinned by a
freeze/thaw round-trip that resumes past the call with an injected reply.
