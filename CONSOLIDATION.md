# CONSOLIDATION.md — the cleanup roadmap

Working tracker for consolidating the runtime's surface, settled in direction with the owner
2026-07-31. Companion to `CALLS.md` (whose unification is the baseline everything here queues
behind) and successor to the scattered "follow-up" notes this collects. Status: **direction
agreed; each item individually gated** — nothing lands without its stated gate. Per the house
rule (PROCESS.md header; THREADS/WASM precedent), fold each settled item into `DESIGN.md` and
delete its section here; drop the file when it is empty.

## 0. The yardstick

Consolidation is measured by what it **deletes**: execution models, per-backend arms, special
cases, safety arguments — never by name count. A capability *kind* is cheap (one small arm
behind the one guarded dispatch); a second execution model or transport is expensive (its own
deadlock argument, fuel regime, parity gap, differential harness). Fine-grained authority
objects are the capability model working, not clutter — attenuation dies if they merge
(INVARIANTS.md 3; D19). When a candidate below would merge *authority* rather than
*mechanism*, it is listed under §10 (leave alone) instead.

Every item obeys INVARIANTS.md 1: it lands on a failing test, a measured regression, or a
named consumer — an aesthetic urge is not a gate.

## 1. Baseline: finish CALLS.md (increments 3–6, plus the §10 addendum)

Everything of consequence queues behind the one migration already agreed: one `Offer`
binding, one calling convention, one execution model; the two-lock nested sub-interpreter,
`ProviderState`, provider-pays fuel, and the `GuestImpl`/`LiveImpl` split all leave the TCB
(CALLS.md §7). The §10 addendum adds direct handoff, quiesce-on-admission, the JIT
crossing-depth bound, and `fuel.remaining`.

**Why it is the bottleneck:** items §2, §3, §5b, and the payoffs of §8 all *define themselves*
in terms of the unified offer semantics. Landing new consumers on the old shapes makes
increment 6 harder — so while CALLS.md is in flight, nothing new may target `ProviderState`,
`Binding::LiveImpl`, or the eval-loop-only serve plumbing.

**Gate:** the CALLS.md §8 discipline itself (differential pins per increment).

## 2. The coroutine family collapses onto the unified offer

The largest remaining deletion. Today's coroutine machinery — `Yielder` (cap_id 7),
`Instantiator` ops 2/3/4 (`spawn_coroutine`/`resume`/`spawn_demand_coroutine`) and their
module variants 6/7, and the parent-as-pager pattern built on them — is a hand-rolled special
case of CALLS.md §2: *a call runs a handler over the other party's world; if it blocks, the
caller waits; parks compose.*

- A coroutine child **is** a process-provider child whose serve point is the rendezvous.
- The parent's `resume` **is** a `cap.call`; `yield` **is** the handler replying.
- Demand paging **is** the faulting child calling a pager-parent provider. The fault-service
  hot path stays fast because it is exactly the CALLS.md §10.2 **direct-handoff** shape: the
  pager-parent is parked at `svc.wait` when the child faults.

**Deletes:** one cap kind (`Binding::Yielder` + its durable variant + its eval-loop arm),
five `Instantiator` ops, the child-suspension plumbing only coroutine children use, and the
"guest-serviced capability exists only as the special-cased Yielder" gap PROCESS.md §0 names.

**Gates:** CALLS.md increments 3–5 landed (needs inline animation + direct handoff on the
relevant tiers); a **benchmark pin on the fault-service path** — the collapse must not regress
lazy-paging latency vs the bespoke coroutine ops (bench harness is the arbiter, AGENTS.md).

## 3. Instantiator: config-record spawn; Budget swallows the fuel scalars

With §2 done, de-proliferate what remains of the op table (~16 ops today):

- **One `instantiate(record)`** taking a config/grants record (the op-11/15 named-grant
  record format generalized): module, entry, window (carve spec *or* detached-minter), named
  grants, and — replacing today's raw `fuel` scalar — a **`Budget` handle**. Spawn variants
  become data, not opcodes.
- **Lifecycle stays ops:** `join`, `poll`, `detach`, `kill` — they are verbs on a running
  child, not spawn configuration.
- **`child_offer` converges** with §5-of-CALLS's "one wiring constructor over a running
  child": it is the guest spelling of the same act, and should be documented (and eventually
  implemented) as such rather than as its own mechanism.
- Raw fuel arguments elsewhere become sugar over `Budget.split` or vanish; this is the §15
  `create(module, window, budget)` accounting arriving through the front door, and it is the
  convergence point `fuel.remaining` names (CALLS.md §10.6 end state).

**End state:** `Instantiator` at ~6 ops, one resource-accounting object in the system.

**Gates:** §2 first (otherwise the record schema is designed twice); the JIT
`instantiator_rt` thunks migrate in the same change (no tier-split of the spawn ABI) — that
set now also includes the wasm-JIT tier's `env.instantiate`/`env.join` imports
(`svm-wasm-jit::compile_module_nested`), which mirror the current op-0 scalar signature.

## 4. `Memory` is the degenerate `AddressSpace` — delete it

`Memory` (cap_id 3) is `AddressSpace { base: 0, size: whole window }` (cap_id 5) minus the
`sub` op — and `sub` is harmless to grant, since it only mints attenuations (D19). Its own
doc says so ("like `Memory` but every op is confined to the holder's sub-range"). Retire
`Binding::Memory` onto the general kind: two cap kinds, two dispatch arms, two durable
variants become one of each.

**Independent of §1** — mechanical, can land any time. **Gate:** the existing `Memory`
tests re-pointed, grant-site migration, one deprecation note for the cap_id.

## 5. `Blocking` and `IoRing`

- **5a. `Blocking` is a mock** (its own doc says so: "a *mock* synchronous-only/blocking host
  capability"). Demote it to test-only wiring so it stops reading as a product primitive in
  the public cap_id space. Independent; land any time.
- **5b. `IoRing` gets a post-increment-5 re-measure.** Its two jobs are boundary-crossing
  amortization and overlapping blocking host ops on the offload pool. Once promotion lands
  everywhere, a blocking host op can simply **park the fiber** — the offload pool becomes
  internal plumbing behind an ordinary `cap.call` — and the overlap job is subsumed. What
  remains is batching, which is a benchmark question. If the numbers say the SQE/CQE ABI
  buys little over the JIT fast path, the ring, its 64-byte SQE format, `RingState`, and the
  async completion path all leave.

  **Gate:** CALLS.md increment 5 landed; a benchmark comparing ring-batched vs
  fast-path-sequential vs fiber-overlapped workloads. Deletion only on measured redundancy —
  invariant 1 cuts against deleting on aesthetics too.

## 6. One inert code-handle kind (`Module` + `JitCode`) — deferred, door held open

Both are "code as a capability with no callable ops, named by another capability's verbs."
The unit-vs-domain difference is a property of the **verb** (`Jit.invoke`/`install` run it in
the caller's world; `instantiate` spawns a child), not of the handle — even the §22
preconditions (memory-match etc.) are about running in *this* window, i.e. verb-time checks.
Unifying would delete a cap kind and unlock **compile-then-sandbox** (JIT a plugin, then
instantiate it as a confined child), which today has no bridge.

Unlike §2–§5 this *adds* an ability, so it waits per INVARIANTS.md 1. **Gate: a named
consumer** for compile-then-sandbox. Until then: a paragraph in DESIGN.md §22 holding the
door open, nothing more.

## 7. Small warts (batchable, mostly independent)

- **`HostProcRegion` → `HostProc` with an optional minter**; fold the parallel
  `host_proc_forks` array into one registration struct.
- **`call.sym`'s vestigial legacy handle operand** — the last special case in the call
  encodings (the v7 `ns` field's sibling; same fate at the next wire rev).
- **`out_sink`/`err_sink` side-channel in `regrant_into_child`** — stdout/stderr re-grant
  mutates fields outside the handle table, breaking "authority = table entry" uniformity.
  Post-§1, stdio inheritance rides the same stream/offer regrant as everything else.
- **Name the two `cap.self` families** in the docs: *reflection* (count/get/resolve/label/
  attest/provenance/type_id/covers) vs *domain-runtime verbs* (`svc.*`, `clone_caller`,
  `fuel.remaining`) — the latter live there only because it was the op space needing no wire
  change. Documentation now; any op-space split waits for a wire rev that is happening
  anyway.

## 8. Free downstream payoffs (no work — consequences to notice)

- The `NonDurableKind` refusal list shrinks as bindings merge (§2, §4, §5a, CALLS §7).
- Snapshot re-linking of live offers becomes one structural rule (the `LiveImpl`
  join-slot machinery folds into the unified offer's durable name).
- The eval-loop-only `-EINVAL` matrix (the OPS_PARITY surface) loses rows with every special
  case folded onto the unified path.

## 9. The docs follow the code

Per the standing rule: fold settled trackers into `DESIGN.md` and delete the file. Queue as
their subjects settle: `CALLS.md` (post-increments), the superseded parts of `IMPORTS.md`
§3.2/§3.5/§3.6, `FORK.md`'s transport half, and each section of this file.

## 10. Deliberately left alone

- **`Exit` and `Clock` as capabilities** — authority-shaped on purpose: a sandboxer can
  withhold or interpose them, which a syscall never offers.
- **`Jit.invoke` (Model A) vs `install` (Model B2)** — both shipped and pinned; invoke gives
  signature-checked entry without consuming table slots. Touching it is churn. The split now also
  carries the **unit concurrency contract** (§11): invoked units stay seam-free leaves
  (`run_invoke`'s CapFault contract, unchanged); installed units join the caller's concurrency
  model.
- **`SharedRegion` / `WindowMinter`** — distinct authorities (shareable backing; detached
  windows), not mechanism duplication.
- **The fine-grained authority zoo generally** — see §0.

## 11. Installed units join the caller's concurrency model (owner-directed, 2026-08-04)

Not a deletion — an ability, wanted by the owner: **§22 `install`ed units may use the thread and
concurrency primitives freely** (`thread.spawn`/`join`, `memory.wait`/`notify`, atomics). The door
is `install`, on principle: installed code runs **in the calling vCPU's own frames** (module-aware
`Frame.module` dispatch) on the caller's scheduler seam — unlike `invoke`, whose sealed nested
`run_invoke` stays seam-free (that CapFault contract is unchanged; the §10 invoke/install split does
the contractual work). Was *unexercised* at first: `VcpuEvent::Spawn` carried no module, so a spawn from an installed frame
resolved the func index in module 0 — the same frame-module hole class as the `event_instantiate`
fix (PR #590).

Slices (each differential-pinned; the natural sequel to the §22/§14 wasm-tier arc):
1. **Module-aware spawn** (TCB) — *landed*: thread the spawning frame's module through
   `Op::ThreadSpawn → VcpuStop/VcpuEvent::Spawn` and into the spawned vCPU's root frame, across all
   three interp drivers (cooperative, native parallel, resumable/browser) **and the tree-walker
   oracle** (its `run_inner` spawn started the child in module 0 — fixed to the spawning frame's
   module, mirroring the bytecode engine). `wait`/`notify`/atomics are address-based (module-agnostic,
   pinned not assumed).
2. **JS drivers** — *landed*: the spawn relay carries the module (the confined-child path's `smod`
   pattern).
3. **Emitted units (wasm tier)** — *landed*: `env.thread_spawn`/`env.thread_join`/`env.mem_wait`/
   `env.mem_notify` imports in the nested-caps emit (the PR #587/#590 bounce pattern; blocking inside
   an import *is* the Worker model — every vCPU is a real thread on the par tiers). Concurrency-using
   units emit, so the B2 per-Worker table mirror stays total (no interp-only null slots beyond v128).
4. **Native Cranelift tier** — *landed*: the piece DESIGN.md folded into "items 2–3", promoted to its
   own slice. `define_extra` admits a `thread.*`/futex unit once the domain hosts threads
   (`Jit::enable_thread_hosting`, the twin of `enable_fiber_hosting`), driven by a thread-hosting
   grant (`grant_jit_threads` / `Host::set_jit_hosts_threads`); `jit_cap_run` forces the serialized
   locked-`Host` path (a hosted unit's spawned vCPUs are concurrent `cap.call`ers). **Soundness**: a
   unit's `thread.spawn N` dispatches its entry through the shared `fn_table`, so the unit's own funcs
   are auto-installed there (`ref_slots` now covers `ThreadSpawn`, not just `ref.func`) and the spawn
   func index is remapped — otherwise it would launch the *parent's* slot `N`. The **invoke seam** is
   enforced at the invoke dispatch (`jit_native_op` op-1 / `jit_invoke_locked`): a threaded unit is
   refused (CapFault) there, mirroring the interp's seam-free `run_invoke`.

**Scope guard (§2):** coroutines/`Yielder` in units are NOT built bespoke — they arrive when §2
collapses the family onto the unified offer; unit-side coroutine plumbing now would be work queued
for deletion.

**Gates:** per-driver differential pins (an installed unit that spawns its own module's funcs ≡
oracle on every driver, native tier included — `svm/tests/jit_cap.rs::installed_unit_spawns_its_own_func_native_agrees`
+ `installed_unit_futex_wait_native_agrees`); `run_invoke`'s contract byte-identical (invoke of a
threaded unit CapFaults on both tiers); the §22 masking/typing invariants untouched (spawn is
scheduler plumbing, not the confinement hinge).

## 12. Sequencing

| item | depends on | gate |
|---|---|---|
| §1 CALLS increments 3–6 | — | per-increment differential pins |
| §4 Memory→AddressSpace | — | test migration |
| §5a Blocking demotion | — | test-only rewire |
| §7 warts | — (last bullet: post-§1 for sinks) | mechanical + doc |
| §2 coroutine collapse | §1 (incr. 3–5) | fault-path benchmark pin |
| §3 config-record spawn + Budget | §2 | JIT thunk parity in-change |
| §5b IoRing re-measure | §1 (incr. 5) | benchmark; delete only on measured redundancy |
| §6 one code handle | — | **named consumer** |
| §11 installed-unit concurrency | — (slices 1–4 landed: interp, JS, wasm-emit, native Cranelift) | per-driver differential pins |
| §9 doc folding | each subject settling | owner sign-off per fold |

**End state:** the `Binding` enum at roughly a dozen variants, `Instantiator` at ~6 ops, one
resource object (`Budget`), one cross-domain execution model in the TCB, and a doc set that
describes the system rather than its history.
