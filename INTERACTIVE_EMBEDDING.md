# INTERACTIVE_EMBEDDING.md — the interactive-embedder surface (browser-first)

Status: **partially built — re-reconciled 2026-07-28** (prior reconcile 2026-07-24).
Written 2026-07-17 as a pure scoping doc. Since then the critical path (W1) **shipped through a
different mechanism than sketched below** — a **DAP-over-the-wasm-FFI** debugger, not the low-level
`svm_dbg_*` ABI this doc proposed — and now lives in `DEBUGGING.md` (browser slices); the
memory-instrumentation substrate (W3's dependency) lives in `HOOKS.md`. This doc is kept for the
**remaining** workstreams and as the requirements record; built parts are marked and
cross-referenced below, not restated.

The 2026-07-28 pass corrected two rows the 07-24 reconcile left stale: **W4** (the `StdinPark`
suspend/resume seam did in fact ship in the browser, via the `svm_pg_*` console path) and **W5**
(chibicc now compiles C in the browser — `SELFHOST_C.md` marks all five self-host steps done
2026-07-24, with printf/float follow-ons through 07-28). It also demoted **W6** to honest status
(only the *native* seeded scheduler exists; the browser/tooling items are unbuilt) and added a
**"Consumer-surfaced requirements not yet scoped"** section recording capability needs surfaced by a
prospective interactive embedder (c_interpret) that the W1–W6 scope does not yet cover.

| Workstream | Status | Home |
|---|---|---|
| **W1** interactive debug on the bytecode engine (browser) | **Built** — DAP-over-wasm (`svm_dap_*` cdylib exports, `web/dap.js`, `browser-dap-test.mjs` gates CI); incl. step-back / reverse time-travel, watchpoints, multithreaded debug | `DEBUGGING.md` browser slices |
| **W2** machine-state view | **Partial** — named locals/frames read back over DAP; the finite-register-file *mode* (v2) unbuilt | `svm-dap`, `DEBUGGING.md` |
| **W3** memory-access scoring | **Substrate built** (`Instance::with_mem_hooks`, the `svm-opt` instrumentation pass, C ABI `svm_instance_with_mem_hooks`, 3-backend parity gate); **not** wired into the browser cdylib | `HOOKS.md` |
| **W4** blocking-input suspend/resume | **Built 2026-07-30** — the engine seam (`VcpuEvent::StdinPark`, `svm_pg_*` console path), `CapTape` replay, **and the debug-session verb**: a `blockStdin` launch flag parks an exhausted `read` as a `stopped` event (reason `"stdin"`, no clock advance), the custom `provideStdin` request appends bytes and a resume re-issues the read, and a reverse `seek` replays provided inputs from the tape with **no re-park**. Fail-closed launch gate (bytecode + powerbox + single-vCPU only); parked placeholder reads never tape. Gated by `browser/tests/chibicc_debug.rs` (round-trip + replay + inertness pin) and `dap_bytecode.rs` (gate) | `bytecode.rs` `DebugRun`, `svm-dap`, this doc |
| **W5** in-browser C→module compile | **Built** — chibicc compiles C client-side today (`chibicc.svmb` asset + `svm_run_onramp_fs` + `svm_parse`) | `SELFHOST_C.md`, `TODO.md`, `BROWSER.md` |
| **W6** small host/tooling items | **Mostly remaining** — only the *native* seeded scheduler (`attach_scheduled_seeded`) exists; the four browser/tooling items (seed-via-ABI, `display` frame-query, memory-map JSON, compile metrics) are unbuilt | mixed |
| **—** consumer-surfaced needs not yet scoped | **Not scoped** — telemetry stream, cache-coherence view, adversarial+replayable scheduling, paging counters, state writes, sem/barrier libc | new section below |

The design invariants and the requirements for the remaining workstreams stand as written; where a
section below has shipped, a **Status** line at its head points at where the built form lives and
the original design text is left as the record.

An *interactive embedder* is a host that drives a guest **step by step and inspects it
between steps**: debugger frontends, educational programming environments, REPLs and
playgrounds, profiling/visualization tools. Natively, SVM already serves them — the
tree-walker's `Inspector` (`svm-interp`) has stepping, breakpoints, watchpoints, time-travel,
and a DAP server (`svm-dap`) on top (`DEBUGGING.md`). **In the browser, at the time of writing, it
did not**: the browser build (`browser/`, `BROWSER.md`) ran the bytecode engine through
run-to-completion entries (`svm_run*`) only. W1 has since closed the debug half of that gap via
DAP-over-wasm (see the status block above); the profiling/input/compile halves (W3–W5) remain.

This doc scopes the workstreams that close that gap, plus a few adjacent host/tooling
capabilities interactive embedders keep needing. Requirements are stated embedder-neutrally;
several prospective consumers (e.g. educational debugger frontends) want this surface, and
nothing here couples SVM to any one of them. Acceptance is against SVM's own oracles — the
native `Inspector` and the differential house style — not any consumer's test suite.

Design invariants inherited from `DEBUGGING.md` §0 (do not relitigate): the debugger is a
host-side observer that never widens guest authority; debug info is tooling, untrusted for
escape; the interpreter tier is the debug engine.

---

## Current substrate (what this builds on, not rebuilds)

| Piece | State | Where |
|---|---|---|
| Full interactive debug surface, native | Built | `svm-interp` `Inspector`, `svm-dap` (`DEBUGGING.md`) |
| Bytecode-engine debug seam: op-for-op stepping-location + per-step window/SSA-value traces (single-vCPU, seam-free) | Built, **batch-shaped** | `bytecode.rs` `ir_trace`/`ir_window_trace`/`ir_value_trace` (`crates/svm-interp/src/bytecode.rs:3003/:3045/:3101`) |
| Bytecode values inspectable: stable, unique slot per SSA value (`regs[base + i]`, typed by `func_value_types`; no reuse/coalescing), parity-proven vs the tree-walker | Built | `DEBUGGING.md` §1b G2, `crates/svm/tests/debug_parity.rs` |
| Single-op stepping bit-identical to run-to-completion (`budget = 1`) | Built | `bytecode.rs:1391/:2997` |
| Deterministic, self-contained browser `Host` (streams accumulate, stdin is a buffer, `Clock` is a counter) | Built | `BROWSER.md` § Decisions |
| Host-serviced vCPU events (spill frame → host services → `deliver_*` resumes) | Built (pattern) | `bytecode.rs:1842ff` (`VcpuEvent`, tier-up path) |
| Cooperative multi-vCPU `drive` + deterministic timeout selection | Built | `bytecode.rs:4623` |
| Memory-access instrumentation pass (observe/veto every guest memory op, zero-cost-when-off, all backends) | Built natively | `HOOKS.md`, `Instance::with_mem_hooks` (`crates/svm-run/src/lib.rs:4110`) |
| Source-level debug info waist (`debug.loc`/`debug.var`/types), chibicc `-g` | Built | `svm-ir` `DebugInfo`, W4 in `DEBUGGING.md` |
| `display` / `keyboard` / `fs` browser capabilities | Built | `browser/src/lib.rs` (~:1831), `demos/doom/` |

The key prior finding (`DEBUGGING.md` §1b): *the bytecode tier is fully inspectable, not
precluded — it is unbuilt as a DAP backend, not blocked.* Everything below was wiring, not
research — and W1 confirmed it: the DAP backend over the bytecode engine **did** land (through the
wasm FFI), so the memory-access row below is now the only substrate piece the *remaining* work
(W3-in-browser) still needs to reach.

---

## W1 — Interactive debug sessions on the bytecode engine (the critical path)

> **Status (2026-07-24): BUILT — differently.** Shipped as a **DAP-over-the-wasm-FFI** debugger
> rather than the `svm_dbg_*` ABI sketched below: the `browser/` cdylib exposes `svm_dap_request` /
> `svm_dap_reset` / `svm_dap_response_ptr` / `_len` (`browser/src/lib.rs`) — a JSON-in / JSON-out
> pump over `DapServer::handle`, backed by the bytecode `Debuggee` — with `web/dap.js` as the JS
> client and `browser-dap-test.mjs` gating CI (initialize→launch→breakpoint→stackTrace→variables→
> continue on the engine the playground ships). Step-back / reverse time-travel, watchpoints (data
> breakpoints in the playground panel), and multithreaded debug (wait/notify over DAP) all landed
> too. See `DEBUGGING.md` browser slices. The design text below is the original (unbuilt) sketch,
> kept as the record of the road not taken.

**Need.** An embedder must be able to: step one op / one source line (into/over/out), run
until a breakpoint/watchpoint/fuel bound, pause, read the PC and source location, read frames
+ locals + arbitrary window bytes (and write bytes), step **backward**, and `seek` to an
arbitrary step index — synchronously, from JS, against the browser cdylib.

**Today.** The `ir_trace` family is trace-after-the-fact (run fully, return the sequence), not
interactive; the cdylib exports are run-to-completion. The full `Inspector` lives on the
tree-walker, which is excluded from the wasm build (fail-closed — its `Scheduler` uses OS
threads/`Instant`; `BROWSER.md` § Decisions).

**Direction.**
1. A **resumable debug-session object** over the bytecode engine: own the `Vcpu` + `Mem`,
   execute with `budget = 1` per call (already bit-identical to run-to-completion), expose
   `IrPc`, slots, and the window. Single-vCPU, seam-free scope first — exactly `ir_trace`'s
   scope.
2. **Time-travel v1 by replay**: the browser `Host` is deterministic and self-contained, so
   `seek(t)` = re-run from 0 with the same inputs; cache periodic window+slots snapshots so a
   seek costs O(snapshot interval). `step_back` = `seek(t−1)`. An undo-log can come later if
   replay-cost ever matters; it changes nothing observable.
3. **Breakpoints/watchpoints** as step-loop checks: source breakpoints via `debug.loc`;
   watchpoints via the W3 hook pass or a per-step window diff — whichever is simplest that
   meets acceptance.
4. **cdylib ABI** (same `svm_alloc` conventions as existing entries):
   `svm_dbg_new(module, stdin, caps) → session`, `svm_dbg_step / step_back / run_until`,
   `svm_dbg_pc / source_loc / step_count / seek`, `svm_dbg_read_reg / read_var / read_window
   / write_window`, `svm_dbg_frames_json`, breakpoint/watchpoint set/clear/list. Fuel bounds
   every `run_until`.
5. **Threads follow-on**: multi-vCPU debug rides the cooperative `drive` with a deterministic
   scheduler and a global turn counter (the `Inspector::turn` shape). Not in the v1 slice.

**Acceptance.** A Node/Chromium test compiles a `-g` C program and drives: step to a source
line → hit a breakpoint → read a local → `seek` back 10 steps → re-read (value differs) →
step forward to reconvergence — with every stepped location and read value matching the
native tree-walker `Inspector` on the same program (extend the `debug_parity.rs` pattern
through the wasm ABI). A watchpoint fires at the same step index as the native `Inspector`.
Fuel stops a runaway `run_until`.

## W2 — Machine-state view (rides on W1)

> **Status (2026-07-24): PARTIAL.** v1's named locals/frames read back over the DAP surface
> (`svm-dap` `read_var`, exercised by `browser-dap-test.mjs`). The v2 **finite-register-file
> compile mode** below is unbuilt.

**Need.** Debugger UIs want a "machine panel": a register file, a program counter, a stack
pointer, and SIMD lanes — real machine state, not a display fiction.

**Today.** The bytecode engine *is* a register machine with stable typed slots (§1b G2). The
chibicc frontend threads a data-stack pointer through calls; frames with
spilled/address-taken locals live at real window addresses. `v128` is a first-class value
type.

**Direction.**
- **v1 (with W1):** expose the current frame's slot file (filtered: `debug.var`-named values
  + recently-written), `IrPc` as the PC, the data-stack pointer as SP (frame base as FP), and
  lane-rendered `v128` slots. Pure ABI + view work; the state already exists.
- **v2 (optional follow-on):** an opt-in **finite register file** compile mode in
  `compile_func`: cap slots at a small named set (e.g. 16), spill excess to the data stack
  (visible in the window), pass leading args in designated registers. Naming should be
  RISC-flavored (`a0–a7`/`ra`/`sp`/`t*`): SVM IR is a load/store machine whose compares
  produce values — there are no flags, so borrowing a flags-ISA's names would misdescribe the
  machine. Differentially tested against the unconstrained mode (house style). This makes
  register scarcity, spilling, and calling conventions *observable* — useful to any embedder
  that teaches or visualizes them.

**Acceptance (v1).** For a program with named locals: at every step, exposed slot values
equal the tree-walker's `read_var` (the `ssa_var_value_parity_per_step` pattern, driven
through the wasm ABI). SP visibly moves across call/return; a `v128` local renders its lanes.

## W3 — Memory-access scoring in the browser

**Need.** Profiling/visualization embedders want the guest's memory-access stream: cache and
locality models, heat maps, access ordering — without touching the engine.

> **Status (2026-07-24): SUBSTRATE BUILT, browser-wiring REMAINING.** The `HOOKS.md` pass is
> complete natively (P0–P3 + the C ABI `svm_instance_with_mem_hooks` + a 3-backend parity gate,
> `crates/svm/tests/mem_hooks_diff.rs`); only the on-demand native high-throughput seam (P4) is
> open. It is **not** exported from the `browser/` cdylib — so this section (reaching it from the
> browser) is the genuine remaining work.

**Today.** The `HOOKS.md` pass fires an embedder hook around every guest memory op, identical
across backends, zero-cost when off — with cache/page-fault scoring as a named use case. It
is wired natively (`Instance::with_mem_hooks`); it is **not yet** exported from the browser
cdylib (no `svm_*` mem-hook entry) — confirmed 2026-07-24.

**Direction.** (1) Confirm the hook pass runs on the bytecode engine under wasm; add a
hook-install flag to the W1 session. (2) Ship access-stream consumers (e.g. a small L1/L2
cache model with per-run counters and a line-state dump) **host-side in the cdylib** as
tooling — models stay out of the engine and out of the TCB.

**Acceptance.** A strided-vs-sequential access pair of guests shows the expected miss-count
ordering, and browser counters match the native run of the same hook stream.

## W4 — Blocking-input suspend/resume

> **Status (2026-07-28): PARTIAL — the suspend/resume seam shipped, in the browser.** A `read` on
> an exhausted stdin buffer suspends the vCPU instead of returning EOF, via `VcpuEvent::StdinPark`
> (`crates/svm-interp/src/bytecode.rs`), with `Vcpu::set_stdin_blocking` / `push_stdin` to arm and
> resume. It is wired into the browser cdylib as the console path: `svm_pg_open` boots a guest
> suspended at the first stdin read, `pg_pump` returns on the park, and the query entry pushes bytes
> and resumes (`browser/src/lib.rs`). What's **remaining** vs. the sketch below: it is packaged as
> the `svm_pg_*` REPL/Postgres path, **not** as a generic `svm_dbg_provide_stdin` on the **W1 debug
> session**. **Update 2026-07-30:** the `CapTape`/`seek`-replay half **landed** — the `svm-dap`
> backend records nondeterministic cap inputs (clock / stdin `read` / host-fn) and replays the tape
> on every rebuild, so reverse debugging reproduces earlier stdout byte-identically
> (`svm-dap/src/backend.rs` `replay_cap_tape`; `browser/tests/chibicc_debug.rs`).
> **CLOSED 2026-07-30 (same day, the plan's slice 1):** the debug-session verb landed. A
> `blockStdin: true` launch flag arms `Host::set_stdin_blocking` on the session powerbox; an
> exhausted `read` yields `Outcome::StdinPark` (op not executed, `op_clock` held), surfaced as
> `StopReason::StdinPark` → a `stopped` event with reason `"stdin"`; the custom `provideStdin`
> request appends bytes (`Host::push_stdin`) and a resume re-issues the read. Parked placeholder
> reads are excluded from the `CapTape`, so a reverse `seek` replays the *completed* provided reads
> byte-identically with no re-park; a read past the replay frontier parks again (the launch stdin
> buffer is empty in this mode, so frontier semantics are exact). Fail-closed: the launch gate
> rejects `blockStdin` off the bytecode engine, without the powerbox, or on a threaded module.
> Acceptance met per the block below (`blocking_stdin_round_trips_and_replays`,
> `without_block_stdin_exhausted_reads_stay_eof` — the invariant-9b inertness pin — and the
> `dap_bytecode.rs` gate tests). The text below is the original sketch.

**Need.** Interactive guests read input that does not exist yet (a REPL prompt, a stdin-driven
program). The embedder needs the run to **suspend** when input is exhausted, surface that to
JS, and **resume** when it supplies bytes — instead of EOF-and-done.

**Today.** The browser `Host`'s stdin is a pre-supplied buffer; a read past the end is EOF.
The engine already has the right seam: `VcpuEvent` spills the frame for host-serviced events
and resumes via a `deliver_*` call (the tier-up path).

**Direction.** A `WaitingForInput`-style outcome on the W1 session (and optionally the plain
run entries): when the stdin capability's `read` finds no bytes, suspend the vCPU via the
`VcpuEvent` pattern and return a distinct status; `svm_dbg_provide_stdin(ptr, len)` appends
and resumes. Provided bytes join the run's deterministic input record (the `CapTape` idea from
`DEBUGGING.md` W1), so a later `seek` replays them faithfully without re-suspending.

**Acceptance.** A prompt-loop C guest round-trips two provided inputs from a test page;
`seek(0)` + re-run replays both, byte-identically, with no new suspensions.

## W5 — In-browser frontend (C source → module, client-side)

> **Status (2026-07-28): BUILT — see `SELFHOST_C.md`.** The browser compiles C source client-side
> today. The chosen approach was **not** to port chibicc to wasm32 against a wasm libc; it is the
> broader **self-hosting** design: compile chibicc *to an SVM IR module* via the LLVM on-ramp, run
> that `chibicc.svmb` as an ordinary guest on the bytecode engine with source + `include/*.h` seeded
> into memfs, and close the loop with the encode step. That shipped: `browser/web/assets/chibicc.svmb`
> is a committed asset, `svm_run_onramp_fs` (`browser/src/lib.rs`) runs it over a seeded fs + argv,
> and the cdylib's `svm_parse` does the text-IR → verify → encode. `SELFHOST_C.md` marks all five
> self-host steps done 2026-07-24, with `#include`/`printf` and `%f/%e/%g` follow-ons through
> 2026-07-28; `TODO.md` corroborates ("chibicc compiles C in the browser"). Gated by
> `browser/tests/chibicc_printf.rs` + the Chromium editor gate. The section below is the original
> (unbuilt-at-the-time) W5 sketch; it also lists the acceptance the shipped path meets (source →
> verified module → runs on a W1 session, no server). One requirement from that sketch still holds
> and is worth confirming per consumer: **always emit `-g`**, since the W1 debug surface depends on it.

**Need.** Interactive embedders want the full edit-compile-run loop client-side: source text
in, verified module out, no server round-trip, sub-second warm compiles.

**Today.** `frontend/chibicc` runs natively only. This is already tracked as `BROWSER.md`'s
"real-language playground tab" open item ("pre-compiled modules first, in-wasm compilation
later"); the playground's `svm_parse` (text IR → verify → encode inside the cdylib) shows the
in-wasm pattern.

**Direction.** chibicc is plain C99 with modest libc needs; compile it to wasm as a
**separate** module the embedder's worker calls (`--emit-ir` + the encoder: C source in,
`.svmb` out), keeping the Rust cdylib untouched. Always emit `-g` — the W1 surface depends on
debug info. (Running chibicc as an SVM guest over `fs` is a nice later dogfood, not the first
slice.) Details belong to the `BROWSER.md` item; this doc adds the requirement that the
compile path emits debug info and the W6 compile metrics.

**Acceptance.** In Chromium: source → verified module → runs on a W1 session, no server;
warm compile of a few-hundred-line program well under a second.

## W6 — Small host/tooling items

- **Compile metrics from the frontend.** Emit per-file node/size counts alongside
  `--emit-ir` output (a walk at emit time). Embedders use these for complexity budgets and
  UI display; SVM cost: a small report, no new machinery.
- **Deterministic-scheduler seed exposure.** The cooperative scheduler's seed should be
  get/settable through the browser ABI so embedders can reproduce and vary interleavings
  (pairs with the W1 threads follow-on; the native `attach_scheduled_seeded` already exists).
- **`display` frame-query op.** A capability op that answers simple predicates over the last
  presented frame (e.g. count of pixels matching an RGBA value) so embedders can assert on
  visual output without reading the whole frame back per query. A few lines in the cdylib
  host next to `present`.
- **Window memory-map introspection.** A JSON description of the window layout — data-segment
  placements, guest heap extent, data-stack region, capability-mapped regions — derived from
  module + Memory-capability state. Read-only tooling over existing state.
- **Design note — time-travel is global-turn.** Multithreaded `seek`/`step_back` targets a
  global turn counter (the `Inspector::turn` model). Rolling back one thread independently
  while others stand still is not meaningful under shared memory and is a **non-goal**.

## Consumer-surfaced requirements not yet scoped

Added 2026-07-28 from a mapping pass against a prospective interactive embedder (c_interpret, an
educational C environment; its side lives in that repo's `SVM_MIGRATION.md`). These are real
capability needs its UI depends on that **no W1–W6 slice currently covers**. They are stated
embedder-neutrally and remain demand-driven — nothing here couples SVM to any one consumer, and none
is a *blocker* (no architectural obstacle was found); they are unscoped scope, listed so the tracker
is honest about the gap. Acceptance, as elsewhere, is against SVM's own oracles.

- **X1 — Concurrency telemetry as a drained stream.** Beyond point-in-time race *witnesses* (which
  SVM has natively), embedders that teach concurrency consume per-yield **streams**: intra-step
  profiling samples (flame charts), context-switch / synchronization-event / **causality-edge**
  records (timeline swimlanes, mutex-handoff/join/condvar arrows), and per-global "contested"
  shared-state tracking. This is the largest gap — a whole visualization surface with no slice.
  Natural home: an extension of the W1 threads follow-on + the W3 hook stream, drained as a batch
  per scheduler turn rather than queried per step.
- **X2 — Cache-coherence view, not just access scoring.** W3 scopes an access *stream* and host-side
  cache *counters*. Embedders also render a live **coherence grid**: per-line tag/valid/dirty/MESI
  state and LRU position across L1/L2, plus last-access set/way highlight. This is a host-side model
  over the W3 stream (stays out of the engine/TCB), but it is materially more than "scoring" and
  should be scoped as its own W3 consumer.
- **X3 — Adversarial + replayable scheduling.** W6 scopes seed get/set. Embedders also need
  **deterministic reproduction of a given interleaving** across a `seek`/replay, plus adversarial
  controls: a chaos mode (quantum = 1) and a **forced context-switch** primitive. The native
  `attach_scheduled_seeded` is the substrate; the browser ABI needs the seed **and** these controls,
  and replay must reproduce the same interleaving.
- **X4 — Demand-paging counters.** W6 scopes a memory-map *layout* JSON. A paging-model teaching
  panel also wants **reserved vs. committed page counts, a page-fault counter, and a settable heap
  limit** (to demonstrate OOM). Read-only tooling over Memory-capability state, plus one cap knob.
- **X5 — State writes from the debugger.** W2 exposes machine state as a read-only *view*. Embedders
  let students **write** a register/slot, an FP lane, and window bytes mid-session (with time-travel
  staying consistent). This is DAP `setVariable` / `writeMemory` over the W1 session — small, but not
  in the current W2 scope.
- **X6 — Semaphore & barrier guest libc.** SVM's threading design covers mutex + condvar; embedders
  also use **semaphores and barriers** (`sem_*`, `pthread_barrier_*`). These need guest-libc
  equivalents over SVM's threading primitives (a frontend/libc item, not an engine one).

### Closure sketch (2026-07-30) — mechanisms that stay SVM-shaped

How each X-item closes without new engine surface, without consumer coupling, and with models kept
out of the TCB. Everything below is either an observer seam, host-side tooling in the cdylib, guest
libc, or a standard DAP verb.

- **X2 + X4's fault counter are access-stream consumers — with two feeds.** The models (a
  configurable cache/coherence model, a first-touch shadow-set fault counter) live **host-side in
  the cdylib** either way; what differs is where the access stream comes from:
  - **Under a debug session: the debugger's own access decode, *not* the hook pass.** The W3 pass
    *rewrites the module* (inserted hook `cap.call`s — `mem_hook_stats` reports `inserted_insts`),
    so instrumenting a debugged guest would surface synthetic ops in the machine view, shift SSA
    slots, and skew the op-clock that `seek` indexes — turning a profiling panel on must not
    change what the debugger shows. The debug tier already observes accesses without any rewrite:
    `watch_hit_before` (`bytecode.rs`) decodes the next op's accessed range from live block-local
    values — it is how watchpoints landed uninstrumented. Generalize that decode into an optional
    per-op **access sink** on the debug session: no module rewrite, op-clock unchanged, zero cost
    when no sink is installed, full `MemEvent` vocabulary (loads/stores/atomics/`Copy`/`Fill`).
  - **Outside a debug session (run-mode profiling): the W3 hook pass**, as designed — the rewrite
    is invisible when nothing inspects the machine.
  - **The two feeds are each other's oracle**: the same program's sink stream (uninstrumented) and
    hook stream (instrumented) must be identical — a house-style differential that also pins the
    sink's op coverage.

  Design notes verified against the as-built hook (`MemEvent`, `svm-run/src/lib.rs`):
  - `MemEvent` carries **no vCPU identity**, and coherence state is meaningless without it. Add
    attribution **at dispatch, host-side** (sink or hook) — on the cooperative tier the dispatcher
    knows the executing vCPU — so the event type, the instrumentation pass, and the engine are
    untouched. This pins coherence modeling to the **interpreter tier**; acceptable and worth
    stating (it is the same tier boundary the debugger already has), not discovered later.
  - `Copy`/`Fill` arrive as **span events**; consumers expand spans to lines/pages themselves.
  - **Capability-written bytes** (stdin fill, `fs`/`display` I/O) never appear as guest accesses.
    A model that should count them needs a host-side tap where caps write into the window —
    plumbing in the cdylib, still no engine change; per-model decision.
  - Model state lives **outside VM snapshots**: `seek`/reverse must reset-and-replay (or snapshot)
    the model alongside the checkpoint ladder, or its panels desynchronize from the slider.
- **X4 splits three ways.** Fault counter → hooks (above). Committed/reserved pages → the **W6
  memory-map JSON** (real Memory-capability state, not a model). Settable heap limit → a
  **Memory-capability growth cap**, so guest `malloc` over `vm_map` returns NULL naturally — a
  powerbox policy knob, not an engine change (a hook *veto* is the wrong OOM semantics: it traps).
- **X1 decomposes; none of it is mem-hook work.** Context-switch / sync-event / causality records
  belong to a **scheduler trace seam**: the cooperative debug scheduler already *makes* every one
  of these decisions (turns, parks, wakes, who-woke-whom) — an optional, zero-cost-when-off event
  tape over decisions already taken is observer-only and widens no guest authority. Flame-chart
  samples need **no new surface** — periodic `stackTrace` polling at turn boundaries. Shared-state
  "contested" tracking **is** hook-derivable once accesses are attributed (last-writer / multi-
  writer per range) — a third host-side hook consumer.
- **X3 generalizes W6's seed item.** Expose **scheduler policy** (seed + quantum bounds) rather
  than the seed alone; quantum = 1 *is* the chaos case, and the native deterministic explorer
  already runs memop-granularity `quantum = 1` (`svm-interp`). Forced switch = a debug-session
  verb that ends the current turn at the next safe point — driver-side, deterministic, recordable.
- **X5 is standard DAP.** Implement `setVariable` / `writeMemory` on the existing backend, and
  record debugger writes on the same input tape as W4's provided stdin (the `CapTape` shape) so a
  later `seek` replays them and time-travel stays truthful.
- **X6 is guest C.** `sem_*` / `pthread_barrier_*` as guest-libc headers over the existing
  futex/wait-notify ops. The proven pattern is `frontend/chibicc/include/pthread.h`: mutex +
  condvar built on the `__vm_wait32` / `__vm_notify` intrinsics (lowered to `memory.wait`/
  `memory.notify` by `codegen_ir.c`) — sem/barrier are the same construction, currently declared
  out of scope in that header. (An earlier draft cited the postgres demo's `ipc_shim.c`; corrected
  — that shim is a single-process no-op counter, not a futex user.) Zero engine surface.
- **Batching:** `svm_dap_request` is one-request-per-pump today; accept a **JSON array of requests
  per pump** (replies already come back as an array). One FFI crossing per step for a step + N
  state reads, embedder-neutral, no new ABI entries.

Two adjacent risks the same pass surfaced, for the record (they are effort/measurement, not new
scope): the **per-step ABI cost** — a real embedder polls ~20 state reads plus a JSON parse per
step, so the W1 session should expose a **single batched per-step state bundle**, not N calls (the
array-pump above); and **frontend acceptance of SSE intrinsics** (`<xmmintrin.h>` / `__m128` /
`_mm_*`) — the W2 XMM→`v128` remap presumes chibicc *accepts* those programs, which is a
frontend-coverage check, not a view remap. A third, the **seek-cost risk**, has since been
**mitigated in code**: the checkpoint ladder (`DEBUGGING.md` slice 4-perf, `DebugRunSnapshot` at
`CHECKPOINT_STRIDE`) bounds replay to the tail past the nearest snapshot, on both engines.

## Non-goals

- Consumer-side integration (any embedder's UI, worker glue, content, or test suites).
- ~~DAP-over-the-browser-build~~ **(reversed 2026-07-24 — this became the chosen path).** The
  doc originally proposed a lower-level JS-shaped `svm_dbg_*` ABI and ruled DAP-over-the-browser
  out; in the event, DAP-over-the-wasm-FFI (`svm_dap_*` + `web/dap.js`) is what shipped for W1, and
  the `svm_dbg_*` ABI was never built. `DEBUGGING.md` is the DAP story on both the native and
  browser builds.
- Porting the tree-walker (and its OS-thread `Scheduler`) to wasm — the bytecode engine is
  the browser debug tier, per the fail-closed decision in `BROWSER.md`.
- Matching any particular consumer's legacy machine model (register names, flags registers,
  fixed address layouts). W2 exposes SVM's real machine state; a finite-register *mode* is
  the one concession, and it is SVM-shaped.

## Suggested slice order

> **2026-07-30 — implementation plan for the remaining work** (supersedes the 07-28 and 07-24
> orders; the original numbered list below stays as the record). Status at this date: W1
> (+ time-travel/watchpoints), W2 v1, and W5 are done; W4 lacks only the interactive provide verb;
> the checkpoint ladder and `CapTape` replay have landed. Each slice lands with CI-gating tests,
> differential wherever two observers see the same program, per `AGENTS.md`.
>
> 1. ~~**W4 finish — `provideStdin` on the debug session.**~~ **DONE 2026-07-30** — as specced
>    below (the `StdinPark` route corrected to the Host-cap seam). See the W4 status block above.
>    Original integration note (2026-07-30, verified):
>    in a debug run stdin is served by the **Host's stdin cap** (`grant_io_powerbox`, `host.stdin`
>    — `svm-dap/src/backend.rs`), not the Vcpu event loop, so `VcpuEvent::StdinPark` is *not* the
>    seam here. The work: a **would-block outcome** on the stdin cap when the buffer is exhausted
>    (instead of serving EOF), surfaced as a new stop variant threaded `DebugRun`/
>    `ScheduledDebugRun` stop enums → the `Debuggee` trait → the DAP `stopped` event; a custom
>    `provideStdin` request appends to `host.stdin` and resumes (re-executing the read). Replay is
>    free: `record_caps`/`replay_cap_tape` already tape stdin inputs, so provided bytes join the
>    tape. Acceptance: the W4 block above (two round-trips; `seek(0)` replays both
>    byte-identically, no re-park).
> 2. ~~**DAP array-pump.**~~ **DONE 2026-07-30** — as specced (`browser/src/lib.rs` pump matches
>    `Json::Arr` and flat-maps `handle`; gated by `browser/tests/dap_batch.rs`: four responses in
>    order + the stopped event in one crossing; singleton unchanged). Original:
>    `svm_dap_request` accepts a JSON **array** of requests per pump (replies
>    already come back as an array); singletons unchanged. Integration (verified): a pure
>    `browser/src/lib.rs` change at the `server.handle` call — `svm_dap::parse` already returns
>    `Json::Arr` for a top-level array (today it falls through to a clean single failure), so the
>    pump matches `Arr` and flat-maps `handle` per element; `web/dap.js` already parses the reply
>    as an array and filters by `type`, so the JS client needs no change. Acceptance: a step + N
>    state reads in one FFI crossing; existing DAP tests pass untouched.
> 3. ~~**Debug-session access sink**~~ **DONE 2026-07-30** — as specced: `MemEvent` moved to
>    `svm-interp` (re-exported by `svm-run`), `mem_event_of` raw-address decode, sinks on both
>    debug engines fired from every advance path (seek replay included), backend `SharedSink`
>    re-installed on rebuilds (rev-trace probes silent), and the **bulk-op watchpoint blind spot
>    fixed on both engines** via the shared `watch_accesses` decode (v128 included; `access_of` /
>    DPOR untouched). Gated by `crates/svm/tests/access_sink_diff.rs`: sink stream ≡ the
>    `mem_hooks_diff.rs` hook stream, dst-write + src-read `mem.copy` watchpoints stop both
>    engines identically, and the sink is inert (result + op-clock bit-identical). Original spec:
>    the hinge for X2/X4/X1-shared-state. Generalize the per-op
>    access decode behind `watch_hit_before` into an optional sink on
>    `DebugRun`/`ScheduledDebugRun`, attributed with the executing vCPU; zero cost when absent, no
>    module rewrite, op-clock unchanged. Scope note (2026-07-30, verified): the shared decode is
>    `access_of` (`lib.rs`), and it is **single-range only — `mem.copy`/`mem.move`/`mem.fill`
>    fall through to `MemAccess::None`**. So the sink needs a multi-range event vocabulary — not
>    a plumbed-through `MemAccess`; **decided (owner, 2026-07-30): the sink vocabulary is
>    `MemEvent`** (`svm-run`, whose `Copy`/`Fill` carry spans), shared verbatim with the hook
>    pass so the differential compares like with like. Threading (verified): the sink rides the
>    same path as the existing `watch_specs` — a field on `BytecodeBackend` re-installed after
>    every `seek` rebuild (`fresh_single`), a slot on `DebugRun`/`ScheduledDebugRun` next to
>    `watchpoints`, fired at the same pre-op sites (`run_to` / `drive`'s check block); the
>    snapshot structs exclude it exactly as they exclude watchpoints (backend re-applies after
>    restore); a launch-arg flag parses in `on_launch` next to `engine`/`stdin`. The same
>    fall-through means **watchpoints likely never fire on bulk-op writes**
>    (a `memcpy` over a watched byte won't stop) on *either* engine — parity hides it because both
>    share the decode shape; pin it with a test and fix it in this slice (the sink-vs-hook
>    differential would have exposed it regardless). Acceptance: sink stream **≡ the W3 hook-pass
>    stream** on the same program (uninstrumented vs. instrumented) across the full `MemEvent`
>    vocabulary; a bulk-op watchpoint fires at the same clock on both engines; all debug parity
>    tests pass with a sink installed.
> 4. ~~**X2 + X4's fault counter**~~ **DONE 2026-07-30** — `svm-dap::models::MemModel`: per-vCPU
>    L1s + shared L2 with MESI line states and LRU, first-touch fault counter, `memModel` launch
>    arg + `memModelStats` request (counters + line-state grids JSON), armed through a `Debuggee`
>    capability probe (tree-walker fails closed). Seek consistency by a **model-side snapshot
>    ladder at the engine's stride** (no new engine seam needed — the `checkpoint_clocks()` idea
>    was dropped as unnecessary): `seek(t)` model state ≡ a from-0 run to `t`, pinned through the
>    real checkpoint ladder. The **W3 browser export** landed as `svm_mem_profile` (+ stats
>    readback): the cdylib adds wasm-clean `svm-opt`, instruments locally (manifest-carrying
>    modules refused, the svm-run slot-0 rule), and feeds the same model — with the **two feeds
>    pinned equal** (`browser/tests/mem_profile.rs`: hook-fed ≡ sink-fed stats-for-stats;
>    `crates/svm-dap/tests/mem_model.rs`: ordering, faults, seek consistency, DAP flow). Original
>    spec: A
>    configurable cache model (levels/sets/ways/line size; per-vCPU L1s + shared L2 via the
>    attribution) with counters + a line-state JSON dump, and a first-touch shadow-set fault
>    counter. Fed by the slice-3 sink under debug and by the W3 pass in run mode — this slice
>    includes the **W3 browser export**. Integration (verified): the browser crate depends on
>    **neither `svm-run` nor `svm-opt`** (it runs the engine directly via
>    `compile_and_run_with_host`), so the W3 export is *not* re-exporting
>    `Instance::with_mem_hooks` — it adds the wasm-clean `svm-opt` dep (deps: `svm-ir` +
>    `svm-verify` only) and reproduces the rewrite + handle bake-in locally in the cdylib,
>    re-honoring the manifest exclusion (`svm-run`'s hooks refuse a manifest-carrying instance);
>    natural entries are `svm_run`/`svm_run0` (via `run_at`) and the powerbox twins. Model
>    snapshotting: there is **no checkpoint callback** — the ladder is private to the DAP backend
>    and self-disables silently when a run leaves the checkpointable subset — so the model
>    snapshots by watching `Debuggee::clock()`/`turn()` at its own stride; expose
>    `checkpoint_clocks()` (one-liner — the snapshot types' `clock()`/`turn()` are already `pub`)
>    if the model should align with the engine's ladder, and surface the self-disable so the
>    model can drop to rebuild-from-0 with the engine. Acceptance: strided-vs-sequential miss
>    ordering; browser counters ≡ native on the same stream; `seek(t)` model state ≡ a from-0 run
>    to `t`, including after checkpointing self-disables.
> 5. ~~**X4's real-state half + W6 memory-map.**~~ **DONE 2026-07-30** — `Mem::map_info` (one
>    read-only accessor: page size, mapped/reserved, explicit-state pages) → the `memoryMap` DAP
>    request (geometry + segments + grown-tail pages + powerbox stack/heap regions; tree-walker
>    fails closed), and the growth cap as **Host policy at the Memory-cap dispatch**
>    (`set_mem_map_limit` / `memoryLimit` launch arg): a `vm_map` past the limit returns
>    `-ENOMEM` probeably (invariant 5 — guest `malloc` observes NULL), `vm_unmap` returns bytes
>    to the budget, and the accounting rides `HostReplaySubstate` so checkpoint restores keep it.
>    Gated by `crates/svm-dap/tests/memory_map.rs`. Original spec: the window memory-map JSON
>    (data segments, heap extent, data-stack region, cap-mapped regions) and a
>    **Memory-capability growth cap** so guest `malloc` over `vm_map` returns NULL at the limit. Integration (verified): the map JSON
>    derives from `AddrSpace.prot`/`.regions` + the window geometry (`Mem.window`
>    mapped/reserved) + the `svm-ir` powerbox layout constants (args/stack), with the **heap
>    cursor read from guest memory** — `POWERBOX_HEAP_BRK`/`_TOP` are window words at offsets
>    32/40, a `read_window`, not host state (`Mem::layout_snapshot` is the serialization
>    precedent). The growth cap is greenfield: **nothing accounts `vm_map` growth today** — the
>    only bound is the geometric `reserved` check in `prot_pages`; `MAX_MINTED_REGION` caps §14
>    region mints only, and `Quota`/`Limits` have no memory field (aggregate metering is
>    explicitly deferred to §15/D48). The cap is a payload on `Binding::Memory` (a unit variant
>    today) checked in `Mem::map` beside the reserved bound — and it must round-trip the durable
>    codec (`DurableBinding::Memory`, both directions). Acceptance: JSON matches module +
>    Memory-cap state across a run; a capped guest observes NULL where the uncapped one grows;
>    durable freeze/thaw preserves the cap.
> 6. ~~**X1 — scheduler trace seam + shared-state consumer.**~~ **DONE 2026-07-30** —
>    `SchedTraceEvent` tape on `ScheduledDebugRun` (turns, `parkJoin`/`parkWait`,
>    `wakeNotify`/`wakeJoin` with **both identities**, `wakeTimeout`, `spawn`), derived by
>    **state-diffing at the two driver sites** — zero helper churn, armed-only cost, the host
>    records and never chooses differently (invariant 4). `schedTrace` launch arg + request
>    (threaded bytecode only, fail-closed elsewhere), re-armed on seek rebuilds (the replay
>    refills the tape deterministically — pinned bit-identical). The shared-state consumer rides
>    `MemModel` (per-word last-writer + contested over the attributed sink, in `memModelStats`).
>    Gated by `crates/svm-dap/tests/sched_trace.rs` — including the honest negative: a fixture
>    whose wait falls through `WAIT_NOT_EQUAL` shows *no* park edge. Original spec: an optional,
>    zero-cost-when-off event
>    tape on the cooperative debug scheduler (turn start/end, park/wake with reason, waker→wakee
>    edge) with batch readback; the shared-state consumer (last-writer / contested per range) over
>    the attributed sink. Integration (verified): every decision point sits in
>    `ScheduledDebugRun::drive` — the pick (`dbg_pinned_coro` → stepping thread → lowest-index
>    `dbg_pick_runnable`), join parks (`dbg_join` → `BlockedJoin`), futex parks (`dbg_wait` →
>    `BlockedWait`, with timed wakeups fired inside the *picker*), and notify (`dbg_notify`).
>    Join wake edges already carry both identities (`dbg_complete` enumerates); the futex wake
>    edge needs the wakee's index materialized — the `dbg_notify` scan becomes `.enumerate()`,
>    a one-token change. Acceptance: the tape is bit-identical across a replay of the same run;
>    wake edges match wait/notify semantics on a fixture; contested flags match a hand-computed
>    oracle.
> 7. **X3 — scheduler policy + forced switch.** Reframed by verification: the bytecode debug
>    scheduler is **hardwired** — lowest-index pick, one op per turn, no seed or quantum anywhere
>    (`on_launch` documents "seed/schedule ignored" for the bytecode engine; seeding exists only
>    on the tree-walker's `attach_scheduled_seeded`, seed `u64`, no quantum there either). Since
>    the turn quantum is already 1 op, **chaos-granularity interleaving is the default** — what's
>    missing is *variation*: a seed-parameterized deterministic pick (the tree-walker's
>    xorshift-with-fixpoint-guard is the precedent, and its `sched_tape()` shows how to recover
>    the realized plan), plus a forced-switch session verb. Both must preserve the load-bearing
>    replay property ("replaying `t` ticks from a fresh session reproduces the state at turn
>    `t`") that `drive_scheduled_to` + `ScheduledSnapshot` depend on — so the seed is session
>    state and forced switches are **recorded** (tape-shaped, like CapTape) and re-applied at the
>    same turns on replay. Acceptance: same seed → identical slice-6 tape; a forced switch lands
>    at a deterministic turn and survives `seek`.
> 8. **X5 — DAP `setVariable` / `writeMemory`.** Greenfield at all three layers (verified): the
>    `Inspector` has no write methods, the `Debuggee` trait has no write operation, and
>    `DapServer` handles neither request (nor `readMemory`) — so this is a new trait method +
>    engine implementations + new `handle` arms. Debugger writes are recorded on the session
>    input tape (the `CapTape` shape) so replay re-applies them. Acceptance: write → `seek` back
>    → `seek` forward re-observes the write; parity with the tree-walker where both back the
>    same request.
> 9. **X6 — guest-libc sem/barrier.** Extend `frontend/chibicc/include/pthread.h` — which already
>    builds mutex + condvar on the `__vm_wait32`/`__vm_notify` futex intrinsics and declares
>    sem/barrier out of scope — with `sem_*` (counter + futex) and `pthread_barrier_*`
>    (generation counter + futex) by the same construction; add the headers to the browser
>    playground's seeded set (which today ships **no** threading headers at all). Zero engine
>    surface. Acceptance: producer/consumer and barrier-phase fixtures run on interp + JIT with
>    identical output.
> 10. **W6 residue** — the `display` frame-query op; compile metrics from the frontend. **W2 v2**
>    (finite register file) stays demand-driven.
>
> Order rationale: 1–2 are small and unblock any embedder spike (interactivity + step latency);
> 3 is the hinge the model slices stand on; 4–5 close the perf/memory panels; 6–7 close the
> threading story; 8–10 are an independent tail, land any time.

1. **W1 spike** — single-vCPU interactive step + source breakpoint exported from the cdylib,
   driven by a throwaway page, parity-checked against the native `Inspector`. De-risks
   everything; all else stacks on it.
2. **W1 time-travel + watchpoints**, then **W2 v1** (same ABI).
3. **W4** (small, high leverage for interactive embedders).
4. **W5** + the **W6 compile metrics** (closes the client-side edit-compile-run loop).
5. **W3**, remaining **W6**, **W1 threads** (+ seed exposure).
6. **W2 v2** (finite register file) — optional, demand-driven.

Each slice lands with tests gating CI, differential against the tree-walker wherever both
observe the same program, per `AGENTS.md`.
