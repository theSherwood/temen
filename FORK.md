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
the caller's window (its declared memory must equal the carve) and BSS is not zeroed (a well-formed
command inits its own state from its data segments); non-durable domains only (durable freeze/thaw
image-swap is the deferred capstone); and only from a clean root computation (no serve handler / active
fibers).

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

**Remaining for the shell loop:** the increment-1 simplifications as they're needed (fresh window + BSS
zero, durable-domain exec, exec from a nested serve context), and job-control `waitpid` flags
(`WNOHANG`, group waits). `reap` today is a blocking single-pid `wait`; `WNOHANG` is a non-parking
`results` probe.

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

**Cranelift is the remaining backend.** It follows the same arc, reusing `instantiator_rt` + the
svm-run serve loop; its extra risk is the native-frame twin continuation (the §8 capture risk, on
compiled code, where there is no `Clone`). Until then `serve_qualifies` folds fork for Cranelift.
