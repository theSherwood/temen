# Increment 4a — cross-world reified animation (implementation note)

Working note for the 4a implementation (CALLS.md §8.4). Delete when 4a lands; it is
scaffolding, not design doctrine (the doctrine is CALLS.md).

## Goal

Replace the `drive_instanced_offer` → `drive_arc` isolated sub-scheduler with animating the
offer handler as a **reified fiber in the caller's own registry** over the provider's world,
run-to-completion, **byte-identical to 3a**. This lays the switch machinery 4b extends to park.

## The switch, concretely (mirrors `serve_switch` + the fiber-return settle)

State: `drive_instanced_offer` (lib.rs:13305) and `drive_arc` (lib.rs:1873) are the 3a path;
`serve_switch` (lib.rs:9237) and the `Terminator::Return` fiber-exit (lib.rs:10750–10824) are
the shapes reused. `run_inner` destructures the vCPU into `&mut` locals incl. `mem`, `host`,
`fuel`, `registry`, `chain`, `cur`, `frames` (lib.rs:7752–7756).

The handler runs on the **caller's `'frames` loop**, so the caller's `{mem, host, fuel}` must
be swapped to the **provider's** for the handler's frames and restored on its return — swapping
`fuel` too, else the caller pays (that is increment 5's caller-pays, not 4a; 3a charges the
provider's `ProviderState::fuel`).

### New types

- `struct OfferAnim` (parallel to `ServeRun`), carried on the vCPU as `offer_anim: Option<OfferAnim>`:
  - `state: Arc<Mutex<ProviderState>>` — the checked-out provider instance.
  - `saved_mem: Option<Mem>` — the caller's mem, parked while the provider's is installed.
  - `saved_fuel: u64` — the caller's remaining fuel, parked while the provider budget runs.
  - `budget: u64` — the fuel budget handed to the handler (`st.fuel.min(OFFER_FUEL)`), so the
    settle drains `st.fuel -= budget - *fuel`.
  - `results: Arc<[ValType]>` — `sig.results`, to translate (edge 2) + push into the caller frame.
  - `handler_slot: usize` — the registry slot of the handler fiber (to recognize its return).
- `ProviderState.busy: bool` — the admission word replacing the held `try_lock` for the
  animated path: set under the brief state lock at switch-in, cleared at settle. A busy instance
  answers `-EAGAIN` (unchanged 3a semantics). (Full §10.3 closed-bit is 4b.)

### Switch-in (in the `cap.call` instanced arm, replacing the `drive_instanced_offer` call)

1. Brief state lock: if `busy` → `-EAGAIN`; if `fuel == 0` → `CapFault`; else `busy = true`,
   take `mem`/`host` out of `ProviderState`, budget = `fuel.min(OFFER_FUEL)`. Release lock.
2. Edge 1 (caller→provider): under the caller `host` lock, `translate_cap_slots` args. (verbatim 3a)
3. Install provider world — the **full execution context**, not just mem/host: `saved_mem = take(mem)`,
   `*mem = provider_mem`; `saved_fuel = *fuel`, `*fuel = budget`; wrap provider host
   `Arc::new(Mutex::new(provider_host))`, swap into `*host`. **Code too:** the handler runs the
   *provider's* `entry.funcs`, resolved through the existing **invoke seam** — swap `*invoked =
   Some(entry.funcs)` (+ `invoked_ref_slots`), and the handler frame uses `module: INVOKE_MODULE`
   (exactly how the third `VCpu` constructor at lib.rs:7292 runs a foreign unit). `cur_funcs`
   re-resolves automatically at the loop top on the module change. Charge 1 entry fuel at switch-in
   to match `drive_arc`'s top-level-entry charge (lib.rs:1918).

**4a in-loop scope (else fall back to `drive_instanced_offer`/`drive_arc`, byte-identical):** the
in-loop cross-world path handles the **non-durable provider, no-`ref.func` handler** common case
(every existing test handler). A durable provider world (shadow-SP entanglement) or a handler whose
code `unit_uses_ref_func` declines to the 3a `drive_arc` path — decline toward the correct slower
transport, the §9 fail-closed shape. This keeps 4a byte-identical everywhere and confines the new
machinery to the simple case that sets up 4b.
4. Park the caller frames as resumer (`park_resumer(*cur, take(frames))` / `root_parked`),
   `chain.push(slot)`, `*cur = slot`, `*frames = handler_entry_frames`, `continue 'frames`.
   Do **not** rewind the caller's cap.call op — on return the caller resumes past it with results
   pushed (a normal-call shape), so there is **one** settle site (the Return handler), not four.

### Settle (in `Terminator::Return`, the `else` fiber-exit branch, when `offer_anim.handler_slot == leaving`)

Instead of pushing `(FIBER_RETURNED, value)` to the resumer:
1. `registry.finish(*cur)`, `chain.pop()`, `*cur = resumer`, `shadow_switch` back, unpark resumer.
2. Restore caller world: `*mem = saved_mem`, provider mem back; unwrap provider host Arc → put
   `Host` back into `ProviderState`; `spent = budget - *fuel`, `*fuel = saved_fuel`.
3. Re-lock state: put mem/host back, `st.fuel -= spent`, `busy = false`. (drain identical to 3a)
4. Edge 2 (provider→caller): under caller `host` lock, `translate_cap_slots` results.
5. Push translated results into the resumer's caller frame (past the cap.call). Clear `offer_anim`.

A handler that **parks** mid-animation is out of 4a scope (run-to-completion). Until 4b wires the
park-files-waiter path, a park under an active `offer_anim` is a `FiberFault` (fail-closed, never
a wrong answer) — the existing test handlers never park.

## Slicing the commits

1. `cap.call` arm only: `OfferAnim`, `ProviderState.busy`, switch-in, settle, `offer_anim` field
   threaded through all vCPU constructors + the run_inner destructure. Test: the `add`/`loads`
   offer_proc dispatch tests in `impl_wiring.rs` stay green (byte-identical), plus the two-vCPU
   `concurrent_instanced_offer_calls_are_safe...` safety test. Differential: `bytecode_diff`/`jit_diff`.
2. Extend to `call.import` / `call.sym` / `call.import.dyn` (mechanical repeats via a shared macro).

## Byte-identity argument

For a handler that runs to completion on one fiber (every existing offer handler): the args/results
translate edges, the provider-pays fuel drain (`st.fuel -= budget - remaining`), and the results are
computed identically to 3a's `drive_arc`; only the executor animating the frames differs (caller's
loop vs. a fresh isolated scheduler), which is unobservable when the handler neither spawns nor parks.
`shadow_switch` is a no-op for the non-durable provider worlds the tests use.
