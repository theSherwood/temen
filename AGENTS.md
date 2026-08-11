# AGENTS.md

Working agreement for agents (and humans) building this project. Keep it short;
keep it followed. The full design lives in `DESIGN.md`.

**Start every session by reading `INVARIANTS.md`** — the design rules that answer
"is this change allowed?". A change that breaks one is wrong until the invariant
itself is deliberately renegotiated with the owner.

## Tracking work: GitHub issues

Track work, bugs, and investigations as **GitHub issues on the Project board**,
not by editing markdown trackers. Each issue is a sub-issue of one of the eleven
**workstream epics** (the parent = the workstream) and carries `area:` / `sev:` /
`kind:` labels — `touches:` for cross-cutting overlap, `topic:*` for fine-grained
subject tags, `invariant` when it touches `INVARIANTS.md`. Do triage and discussion
in the issue: it costs no CI, unlike editing markdown. Put deep root-cause detail in
the issue body (or the relevant design doc when it must live beside the code) — the
issue is the status source of truth. (`ISSUES.md` is **retired**; its history is in
git.) Flaky CI → a `kind:flaky-ci` issue. Full workflow, epic list, and label taxonomy:
**`ISSUE_TRACKING.md`** (labels are reproducible via `scripts/setup-labels.sh`).

## Prime directive: keep it simple

This is a sandbox VM whose entire value is a **small, trustworthy core**. Every
line is potential TCB. Prefer the boring, obvious implementation. Don't add
abstraction, configurability, or cleverness until something concrete demands it.
If a change makes the verifier or the confinement path harder to read, it is
probably wrong. When in doubt, do less.

## Tests, fuzzing, benchmarks — early, not eventually

- **Tests from the first commit.** Every component lands with tests. The
  interpreter is the oracle: differential-test the JIT against it (D-notes in
  `DESIGN.md` §18). Tests should gate the CI.
- **Fuzz from day one.** Two invariants get fuzzed continuously:
  1. *verified ⇒ cannot escape* (fuzz the verifier),
  2. *every memory access is masked to `[0, size)` or proven bounded* (fuzz the
     confinement-masking lowering as its own unit — it is the security hinge, §4).
- **Benchmark as soon as there's anything to run.** Stand up a benchmark harness
  early and watch it over time; we are measured *relative to wasm/Wasmtime*
  (`DESIGN.md` §1a). Catch regressions when they're one commit old, not one
  release old.
  **Log any flaky CI as a `kind:flaky-ci` GitHub issue** (see `ISSUE_TRACKING.md`). Catch and log flakiness early so that we have visibility and can track a fix.

## Performance philosophy: data-oriented design

Most of our speed comes from **reducing allocation and improving cache locality**,
not from micro-optimizing hot code. Default to:

- **Flat data structures.** Prefer arrays / structs-of-arrays over trees of
  pointer-chasing nodes. Index with integers, not pointers, where it keeps things
  flat and relocatable.
- **Arenas / bump allocation.** Allocate per-phase (per-module, per-function)
  into arenas and free in one shot. Avoid per-node heap allocation and avoid
  scattered ownership.
- **Few, predictable passes over contiguous memory.** The decode+verify design is
  a single linear forward pass for a reason — keep that shape elsewhere too.
- Measure before optimizing beyond this; the benchmark harness is the arbiter.

## Security posture (the bar we hold)

- Target is **"as secure as wasm for the host"** — i.e. as secure as Wasmtime, not
  a proof of escape-impossibility (`DESIGN.md` §1a).

- The verifier secures typing, control flow, and index ranges. **Memory
  confinement is the masking lowering, not the verifier** — treat that pass as the
  most sensitive code in the tree.
- In-process isolation is defense-in-depth, **not** a Spectre boundary; distrust
  means separate processes.

## Process

**Always open a PR whenever you have changes** — every branch with commits gets a PR, no exceptions. Open it as soon as you have changes rather than waiting for the work to feel finished. If you have multiple slices queued to implement, you can put them on the same PR until the PR exceeds 1000 loc. When you complete slices after opening a PR, check for merge conflicts and address them.

**Editing CI:** the session token can't push under `.github/workflows/` (needs the `workflow` scope). If you need to change a workflow but can't commit it there, edit the mirror in `.github/workflows_src/` instead and note it in that dir's README — the owner copies it over. See `.github/workflows_src/README.md`.

**Fast local pre-push check (optional):** `git config core.hooksPath .githooks` (or run `scripts/ci/install-git-hooks.sh`) enables a pre-push hook that mirrors CI's `check` job (`build · test · fmt · clippy`), so the common failures surface before a CI round-trip. It's fast feedback, **not** the gate — the CI matrix (cross-OS, miri/asan/tsan/loom, fuzz, differential) still runs on the PR and remains authoritative. Bypass once with `SVM_HOOK_SKIP=1 git push`. See `.githooks/README.md`.

