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
measured regression, or a named consumer. (AGENTS.md prime directive; DESIGN.md §1.)

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

## 4. Host = mechanism, guest = policy

The host's inter-domain layer is a waiter table, wake plumbing, and lifecycle cleanup —
never scheduling policy. Concretely: FIFO queues, wake-all (the host never picks a winner;
guests race through the admission lock), work-stealing, guest-stated deadlines, no
priorities, no fairness classes, no timeslicing. Scheduling *policy* lives in guest code:
guest-driven fibers (D22), parent-as-scheduler coroutines, worker-domain sharding over the
grant graph. The host holds the waiter table only because it alone can deliver lifecycle
cleanup (death-is-revocation must find parked callers). *Violated by:* any host feature
keyed on caller identity, priority, or ordering beyond FIFO — e.g. per-caller fairness
belongs in guest patterns, not the substrate. (Owner decision 2026-07-24; ISSUES.md I38/I39.)

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
**IR-anchored safepoints** — one per taken back-edge, per function entry (`call`/`call_indirect`/
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
