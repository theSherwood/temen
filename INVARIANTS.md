# Invariants

The design rules that answer "is this change allowed?" — read this before working on
anything. Each invariant is a constraint the whole tree already obeys; a change that breaks
one is wrong until the invariant itself is deliberately renegotiated with the owner (record
the renegotiation here, dated). Keep this list short: an invariant earns its place by
rejecting real proposals, not by describing the code.

## 1. Small trustworthy core

Every line is potential TCB. Prefer the boring, obvious implementation; no abstraction,
configurability, or cleverness until something concrete demands it. When in doubt, do less.
*Violated by:* any change justified by "we might need it" rather than a failing test, a
measured regression, or a named consumer — but **closing an existing capability's cross-axis gap
is itself a concrete demand (invariant 14), not speculation**. (AGENTS.md prime directive; DESIGN.md §1.)

## 2. Confinement is the masking lowering

Memory safety for the host rests on one pass: every guest access is masked to `[0, size)` or
proven bounded. The verifier secures typing, control flow, and index ranges — **not** memory.
The target is "as secure as Wasmtime"; in-process isolation is not a Spectre boundary.
*Violated by:* any feature that adds emitted-code or window-access surface outside the
masking regime — new lowerings are suspect by default; prefer reusing an existing guarded
seam (as the JIT serve loop reused `invoke_extra`). (DESIGN.md §1a/§4; the fuzzed hinge.)

## 3. Authority moves only down the grant graph

Every capability transfer is mediated by an authority holding both ends: spawn grants and
`child_offer` re-grants. No peer discovery, no self-mint transfer channels, no registries,
no ambient names. The one sanctioned residue: a domain offering its *own* export down its
own grant graph. *Violated by:* any path where a domain reaches a capability its ancestors
never granted. (Owner decision 2026-07-23; IMPORTS.md §3.3/§3.6, PROCESS.md §4.)

**Ruling — window-minting authority is the memory budget, and it tops up down the graph
(2026-09-08, #1289 R2):** minting or growing an independent (detached) window is not a separate
authority — it **spends `Budget.mem`** (PROCESS.md §5), which attenuates down the grant graph like
every other authority, so a domain can never hold more VA than its ancestors granted. The standalone
`WindowMinter` capability retires once every mint site takes a budget. Because authority moves only
down, a domain that needs *more* memory than its budget holds cannot pull it: it **requests** from its
parent (a message *up* — a served endpoint / fault upcall, the data plane), and the parent **grants**
by **transferring** bytes from its own `Budget.mem` *down* into the child's (the control plane) — the
parent's remaining drops by exactly what the child's rises (conservation). If the parent lacks the
slack, it requests from *its* parent first, so a deep child's top-up **cascades recursively up the
ancestry** to the first ancestor with slack (or the platform's root budget), each hop updating two
budgets. The cascade is **transactional**: if no ancestor can cover the shortfall, nothing transfers
anywhere and the request fails closed (`-ENOMEM`), exactly as an over-asking `split` deducts nothing.
This is Genode's quota-transfer applied to VA — `split` pushes budget down eagerly at spawn,
`transfer` pushes it down lazily on demand — so every byte a descendant holds stays traceable to a
grant from above (the invariant), and nothing becomes ambient-under-`Instantiator`. *Caveat recorded:*
`Budget` is a `NonDurableKind` today, so minting authority does not survive a freeze — under R1
(freeze authority) that is harmless for a platform-driven freeze (the platform re-grants on thaw) and
a prerequisite to lift for an ancestor-driven one.

## 4. Host = mechanism, guest = policy

The host's inter-domain layer is a waiter table, wake plumbing, and lifecycle cleanup —
never scheduling policy. Concretely: FIFO queues, wake-all (the host never picks a winner;
guests race through the admission lock), work-stealing, guest-stated deadlines, no
priorities, no fairness classes, no timeslicing. Scheduling *policy* lives in guest code:
guest-driven fibers (D22), parent-as-scheduler coroutines, worker-domain sharding over the
grant graph. The host holds the waiter table only because it alone can deliver lifecycle
cleanup (death-is-revocation must find parked callers). *Violated by:* any host feature
keyed on caller identity, priority, or ordering beyond FIFO — e.g. per-caller fairness
belongs in guest patterns, not the substrate. (Owner decision 2026-07-24; originally ISSUES.md
I38/I39, now retired — see git history / the issue tracker.)

## 5. Errors are values; traps are for forgery

Fallible operations return negative errnos, probeable on the caller's own error path.
Traps stay reserved for what can never be legitimate: forged handles (a generation never
issued), typing violations on live handles, and escape-adjacent faults. Cancellation is a
value: revocation completes calls with an errno whether the caller was parked mid-call or
calls after — a lifecycle event is never a domain-killing surprise. *Violated by:* any new
trap reachable from a benign race or another party's lifecycle action. (D42; I41.)

## 6. One world per domain

A domain's handlers, threads, and fibers share one window, one powerbox, one fuel budget.
A handler trap is terminal for the domain — never resume over half-mutated state. Safety is
serial-by-default with explicit opt-in ladders (multi-consumer serving, threading) whose
cost — the threading discipline — is the guest's stated choice. *Violated by:* partial-state
recovery, transactional handler worlds, or implicit parallelism. (IMPORTS.md §3.6; I37/I39.)

**One lifetime, too (owner, 2026-07-24):** executors never anchor a domain's lifetime;
ownership does. A domain ends itself (`exit`, or any trap — both domain-wide, on every
engine) or its owner ends it (drop/revocation); spawned vCPUs and fibers are workers inside
the world, never reasons to keep it alive, and nothing implicitly waits for them —
`thread.join` is the explicit wait. Root completion ends the *activation*; in a batch run
the owner leaves with it, so root return/exit/trap tears the domain down, parked daemons
abandoned (non-preemptively: running siblings stop at their next safepoint, so post-teardown
sibling effects are unspecified). Cross-domain waiters parked through a dying domain wake
with an errno (invariant 5 / D37), never hang. *Violated by:* join-all-at-teardown
semantics, an engine where exit/trap leaves siblings parked, or lifetime rules that differ
between batch and reactor — a reactor is the same rule with an owner (the Session) that
stays. (DESIGN.md §12 "Domain lifetime"; jacl timed-wait regression, 2026-07-24.)

## 7. Re-execution is recovery

Parks rewind their frames, so a wake — spurious, racing, or post-thaw — simply re-executes
the parked op, which re-drains, re-parks, or re-derives its own waiter state. Calls that
cross a freeze or revocation boundary are **re-issued** (O10): at-least-once delivery,
idempotence is the personality's problem. Recovery never replays captured scheduler state.
*Violated by:* recovery designs that carry waiter/scheduler records in snapshots, or
exactly-once claims. (§3.6 rewound parks; PROCESS.md O10; DURABILITY.md §13.)

## 8. Control plane ≠ data plane

Service calls carry shell-frequency control traffic — single-slot scalar replies. Bulk data
rides `SharedRegion` rings the guests own. *Violated by:* widening the dispatch/reply ABI to
carry payloads, or any hot path routed through handlers. (F6; I39; the c_shell rings.)

## 9. The interpreter is the oracle; decline, never diverge

The tree-walk interpreter defines **guest-observable semantics** — results, traps, errnos,
memory; the three fast backends (**bytecode interpreter**, **Cranelift JIT**, **wasm-JIT** —
the four-backend taxonomy and naming standard live in DESIGN.md §3) run only what they can run
identically and **decline the rest** (compile vetoes, routing folds — one shared predicate, one
definition) back to the oracle. Each fast backend is differential-tested against the tree-walk
oracle: the bytecode interpreter bit-exact, the Cranelift JIT and wasm-JIT NaN-insensitive.
Anything a backend or a step can't handle refuses probeably or falls back — it never runs wrong
and never hangs where refusal is possible. Differential tests gate every backend feature.
*Violated by:* a fast-backend feature without an oracle counterpart, a second copy of a veto
predicate, silent divergence documented as a quirk, or naming that hides which engine ran (bare
"JIT" is ambiguous — say "Cranelift JIT" or "wasm-JIT"). (DESIGN.md §3/§18; the
serve-qualification veto.)

**Fuel is a checked cross-engine quantity, not an excluded difference.** Fuel is charged at
**IR-anchored safepoints** — one per taken back-edge, per function entry (`call`/`call.dyn`/
`return_call*` and the *top-level* entry), and per `cont.resume` — so the tree-walk oracle, the
bytecode interpreter, and the Cranelift JIT all charge off the *same* IR structure and a run either
completes on all three or traps `OutOfFuel` at the *identical* safepoint. The differential harnesses
therefore **assert** `OutOfFuel` parity rather than skipping it (`bytecode_diff` bit-exact on the
remaining fuel; `jit_fuzz`/`jit_fuel` on the trap). *Violated by:* a backend that meters fuel on a
different unit (per-op, or a safepoint another backend doesn't charge), a harness that re-excludes
`OutOfFuel` from the equality contract, or a fuel charge added to one engine but not the others.
(INTERP_PERF.md "Fuel unification"; owner-approved 2026-07-25.)

**Observability corollary.** Debugging/tracing is a *view onto* execution, not part of the
semantic contract, and is deliberately tiered by backend (stepping and time travel want an
interpreter; DWARF/gdb want native code) — but three clauses keep the tiering disciplined:
(a) **facts agree where comparable** — when two backends report the same kind of fact (a
trap backtrace, a source location) they report the *same* fact, differentially pinned
(identical `IrPc`s, the cursor-advance parity); (b) **observation never perturbs
semantics** — debug hooks are inert unless armed, single-step is pinned bit-identical to
run-to-completion, and no guest-visible "am I traced" bit exists; every new debug feature
lands with its own inertness pin; (c) **a tool that can't see something refuses or falls
back — it never reports a fiction** (the traced fast entry declines to the oracle so a
backtrace is always some faithful engine's; the explorer and checkpointing refuse outside
their subsets). Genuine semantic divergences are either provably unwitnessable by the
differential (refusal-vs-hang: diverging *toward refusal* is fail-closed winning, kept as a
short enumerated list) or **tracked debt with a convergence plan** (the `poll` eager/lazy
child divergence) — never quietly normalized.

## 10. Identity is structural

Interface and type identity is the interned shape (D59) — never a nominal name or registry.
The one honest non-structural bit — who terminates a capability — lives in the
non-interposable attest/provenance namespace, so a parent can interpose everything but
cannot hide that it did. *Violated by:* nominal type registries, or trust decisions keyed on
names rather than provenance. (D59; IMPORTS.md §3.1.)

## 11. The top byte belongs to the guest tag

Every pointer-like value — data pointer, funcref, import/cap handle, fiber/thread handle —
lays out as `[tag:8][generation?][index]`: the **top byte (bits 56–63) is reserved for the
guest's pointer-tag** and the VM never stores meaning there, so `generation + index ≤ 56
bits` on every kind. Concretely this caps the window at 2^56 (64 PiB) and holds the fiber
generation to 32 bits. Every backend must **mask** a handle's generation to its field
width on compare (never bare-shift), so a tagged value is inert at the use site. *Violated
by:* a window wider than 2^56, a generation/index field that reaches bit 56, a resolve that
compares an unmasked `h >> shift` generation, or a tag stamped into a value the runtime
sign-tests as `handle | -errno` (invariant 5) — tag only in 64-bit cells, keep the raw
handle untagged at the ABI boundary. (Owner-approved 2026-08-04; DESIGN.md §3c "Uniform
pointer tagging"; the fiber 40→32 trim.)

## 12. Substrate visibility is ancestor-only

A domain can enumerate, address, or affect only its own descendant tree, and every domain
it *can* name arrived through its ancestry: `fork` returns the twin's pid to the parent
alone (the fork factory serving that fork learns the same pid at mint), and reap is
parent-scoped (`-ECHILD` for anyone else). The core manufactures no reachability — no
domain-enumeration surface, no signal-an-arbitrary-id op, no global namespaces. Global
views (a POSIX pid table, cross-tree `kill`) are **personality policy**, assembled from
capabilities passed down the ancestry chain (temen-posix's shared `World` rides the
grant → fork → fork closure chain) — never substrate state. Wakes the core hands a
personality are scoped to the target domain, never run-wide (`interrupt_interruptible_parks`
takes a domain; #863 slice 3 — the three-generation `c_fork` witness pins that a `^C` at the
parent never sweeps the child's park). *Violated by:* a core surface that resolves a pid/TaskId
the caller's ancestry never disclosed, a new run-global sweep reachable from one domain's
signal, or a personality handed more visibility than its grant chain carries. (Owner
decision 2026-08-13; #863; the slice-2 process table deliberately landed in `temen-posix`,
not the core.)

## 13. One canonical form — migrate forward, delete the old path

Every layout, ABI, encoding, and wire format has exactly one live form. A better form does not
land *beside* the old one; it **replaces** it — the same change (or a tracked sequence with an
owner-set deadline) moves every producer, consumer, and committed asset onto it and then
**deletes** the old path. Old on-disk artifacts are recompiled, not accommodated: the host and
every reader assume the current semantics everywhere and never branch on which era produced a
module. A version marker or dual-mode flag is legitimate only as a migration's *own scaffolding* —
it ships with a scheduled removal, never a standing compatibility contract. Fewer forms is the
whole point: one form is a stronger assumption, fewer branches, less TCB; carrying legacy multiplies
incompatible states and erodes what the rest of the tree can assume. *Violated by:* a compat shim,
format-version branch, or `unwrap_or(legacy)` fallback with no convergence plan; a producer left
emitting the old layout after the new one exists; or "old artifacts still use it" offered to justify
keeping a fork. (Owner decision 2026-08-25; sharpens invariant 1 — every retained fork is TCB
surface and an assumption the tree can no longer make.)

**Ruling — a §14 child's placement is a parameter, not a form (2026-09-08, #1289 R3):** whether a
confined child sits in a **nested carve** (a sub-range of the parent's window) or a **detached
window** (its own reservation) is a grant *parameter*, not a second form. The child-side ABI is
byte-identical under both (base 0, its own reservation — DETACHED_JIT.md §4a), and the snapshot codec
has one root-shaped form applied at one more scope — so there is still exactly one child ABI and one
artifact form for the tree to assume. A `SharedRegion` mapped or unmapped is not two forms, and
neither is this. Escape hatch retained: should the host-side carve path (STW broadcast through the
parent's `Mem`) and the detached path ever diverge into two genuinely different mechanisms rather than
one parameterized one, *that* is the #13 violation — closed by a dated deletion of the carve path (the
#1289 Decision-B convergence), never by keeping both.

## 14. One frontier, consistent across every axis

The set of **accepted** guest-observable capabilities is the *frontier*. New functionality may be
**spiked on the interpreter** (the oracle) alone — the proving ground. But the moment a capability is
**accepted** — merged as a *supported* feature, not an experimental flag — holding the frontier
consistent is the **highest priority**: the immediate follow-up is to carry that capability across
every axis it can reach —

- **Runtime backend** — bytecode interpreter, Cranelift JIT, wasm-JIT: each runs it identically or
  *declines to the oracle* (invariant 9), never a limiting workaround.
- **Host target** — native (x86/arm), wasm32, Windows: a guest never behaves differently by platform;
  a capability landing native-first is not done until it reaches the others.
- **Concurrency model** — the cooperative multiplex driver (`CoopRun`) and the genuinely-parallel
  per-Worker driver (`temen_par_*`) both carry it.
- **Code origin** — the host-translated base module and a §22 guest-JIT unit (`vm_jit_*`,
  runtime-compiled by the guest) both support it.
- **Nesting** — it holds for a confined §14 child, a fork/clone twin, and a spawned thread, not just
  the root.
- **Debugger** — it stays observable under the debug tier, disciplined by invariant 9's observability
  corollary (which governs *how* that tiering may differ).
- **Durability** — its live state is capturable + restorable (warm snapshot) and representable in the
  durable codec — not a `GeometryMismatch` refusal.

Invariant 1 gates *admission* to the frontier (don't spike speculative capabilities — the frontier is
expensive to hold); **this** invariant gates *propagation across* it: once admitted, **no partial
support**. "No consumer yet" never justifies a standing gap — the inconsistency is itself the concrete
demand invariant 1 asks for. A capability's presence on any axis is only ever **closed**, **in flight**
(a tracked issue, worked immediately), or a **recorded exception** below — never parked. A workaround
that sidesteps a gap by capping the guest (pre-sizing a fixed window because emitted code "can't grow
it"; a snapshot that refuses a grown extent) **is** the gap, disguised — not a resting state.
*Violated by:* a capability landed on one axis and left off another it could reach, excused by "no
consumer"; a standing pre-size / over-allocate / refuse workaround with no tracked plan; a "decline"
that hides un-wired support rather than proven impossibility; a spike merged as a supported feature
without either propagation or a recorded exception. (Owner decision 2026-08-27; sharpens invariants
1 + 9 + 13; the #816 warm-coop-grows-but-single-shot-JIT-pre-sizes gap.)

**Ruling — freeze authority is an explicit, attenuating capability (2026-09-08, #1289 R1; closes
O14's conceptual half):** who may snapshot a domain is a capability (invariant 3), not a consequence of
its placement. The platform holds freeze authority over every domain it runs; an ancestor holds it
over a descendant *only by grant*, attenuating down the grant graph like any other authority, and
`attest.freeze_authority` reports the actual holders. Because a snapshot is a read of the window
(PROCESS.md §6), freeze-ability follows window-read authority: a **nested carve** child's window is a
sub-range of its parent's, so the parent can read — hence freeze — it by construction (the grant is
implied by the aliasing, the current behavior); a **detached** child owns its window, so an ancestor
can freeze it *only if granted*. Placement is therefore orthogonal to durability, and §6's rule reads:
*confidential = freezable by nobody below the platform; a domain is confidential **or**
ancestor-freezable per grant, not per placement* — a detached child may be platform-durable **and**
ancestor-confidential at once. A detached child freezes as its own root-shaped artifact (the one codec
form, R3); a subtree snapshot is a consistent cut over the parent's artifact and each detached child's,
coordinated by the authority holder. This makes the op-15 durable refusal an **un-wired-support** gap
(*in flight* — the run driver cannot yet reach a detached child's `Mem`+`Host` at quiesce, and a thaw
must restore a child's `Attestation`, O14's remaining plumbing), **not** a placement-derived decline
this invariant would forbid; it becomes "refuse unless a freeze-authority holder is registered for this
child" once that lands.

**Accepted exceptions** — a *genuine impossibility* on some axis, each carrying the owner's dated
approval; adding one always requires owner sign-off:

- **Durability / §13 `Backed` shared regions** *(provisional — pending owner review, 2026-08-27):* a
  byte snapshot cannot reproduce a live alias into shared backing, so `layout_snapshot_safe`
  fail-closes on a `Backed` region. Recorded as the current behavior; **not yet ratified** as a
  permanent exception.
