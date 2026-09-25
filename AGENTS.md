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

**Consolidate code paths.** The most expensive thing you can add is not a line — it is a
*second route* through a behaviour that already has one. Every duplicated path (a second run
driver, a second dispatch table, a second host-glue family, a copied test file) multiplies the
invariant-14 propagation burden by one, forever. Where a second position is needed, make it a
**parameter of the existing structure**, not a copy: a row, a config, a trait impl. A telescoping
name (`..._with_host_durable_mv_interruptible`) is a parameter that escaped into the name space —
take it back. See `INVARIANTS.md` #15, which also carves out the one legitimate case (a second
implementation that exists to be differentialled against the first).

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

## Flaky CI: find the cause, fix the cause

A flake is a bug whose trigger is timing: in the product, in a test, or in CI plumbing. It is never
noise, and a green re-run proves nothing.

- **First make sure it is a flake.** A failure that follows the PR's own diff, or that main shares
  (two PRs merged together and clashed), is an ordinary bug. Check the same commit's re-run and
  unrelated PRs before calling it timing.
- **No workarounds.** Don't add retries, re-run loops, longer timeouts, sleeps, `#[ignore]`,
  skip-on-failure branches, or `continue-on-error` to quiet a flake. Each one hides a real race or a
  fragile dependency, and the next failure is harder to read. The ones already in the tree are debt
  to remove, not precedent.
- **Reproduce it.** Loop the test under CPU contention (a few `while :; do :; done` busy loops). If
  the window is narrow, force it: put a temporary delay at the suspected interleaving point until it
  fails every time with CI's exact signature. Then trace what really happened. Several obvious
  causes in this tree turned out wrong once a trace or a probe showed the real one.
- **Fix the cause, then prove it.** The same forced interleaving must pass after the fix. When the
  race is in a primitive, pin it with a test that fails on the old code: a loom model for lock or
  scheduler ordering, a unit test for a protocol.
- **External dependencies count.** A job that fetches from a third-party CDN or runs `apt-get update`
  against a live mirror on every run is flaky by construction. Remove the per-run dependency: cache
  the pinned, checksummed artifact, and don't install what the image already has. Don't retry it.
- **Tests that share disk state race.** Tests run as parallel threads (`cargo test`) and parallel
  processes (nextest). A `OnceLock` serialises only threads, so any cache or build tree that two
  tests populate goes through `crates/temen/tests/support/cache_lock.rs`.
- **A harness must fail, never hang.** A wait with no bound turns one flake into a job-timeout
  cancellation with no diagnosis. A driver that gives up says why, including what it saw, and the
  test fails.
- **Log it.** Every flake gets a `kind:flaky-ci` issue (`ISSUE_TRACKING.md`) with the CI signature
  and, once known, the root cause and the proof. If a closed flake recurs, the fix was wrong: reopen
  it. If you can't reproduce one, record what you tried in the issue and leave it open. Don't mask
  the test.

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

**Don't subscribe to PR activity / auto-watch a PR unless explicitly asked.** Open the PR and report it; leave CI-watching, autofix-on-red, and merge-conflict babysitting to the owner. Only call `subscribe_pr_activity` (or set up scheduled CI check-ins) when the owner specifically requests it for that PR.

**Editing CI:** the session token can't push under `.github/workflows/` (needs the `workflow` scope). If you need to change a workflow but can't commit it there, edit the mirror in `.github/workflows_src/` instead and describe the change in the PR description (do **not** log it in that dir's README — the per-change ledger there caused merge conflicts between concurrent PRs and is frozen; the `workflows_src == workflows` check is the to-do list) — the owner copies it over. See `.github/workflows_src/README.md`.

**`--workspace` is not the whole repo.** The root `Cargo.toml` `exclude`s four crates (`crates/temen-llvm`, `crates/temen-webgpu`, `fuzz`, `bench`), so `cargo check/build/test --workspace` — and any local "it's clean" based on it — says nothing about them. A change to a shared signature compiles fine and breaks them silently: `temen-webgpu` sat un-compilable for three weeks that way. Run **`scripts/ci/check-excluded.sh`** (it reads the exclude list from `Cargo.toml`, so it cannot miss a crate) before claiming a cross-cutting change builds; the pre-push hook and the `excluded crates compile` CI job both run it.

**Run the local gate before you push.** `bash scripts/ci/local-gate.sh` runs CI's gating lint/compile checks — fmt, `clippy --workspace`, the separate `temen-jit --features stack-check` clippy lane, the excluded crates, and the `x86_64-pc-windows-gnu` cross-check — **on the toolchain CI pins**, which it reads from `RUST_STABLE` in the workflow so the two cannot drift. That pinning is load-bearing, not tidiness: clippy gains lints between releases, so a clippy clean on your default toolchain says nothing about CI's. A `for_kv_map` error failed two CI jobs while the identical command passed locally for exactly that reason. A missing toolchain/component/target stops the gate with the exact `rustup` command (`TEMEN_GATE_INSTALL=1` installs it).

`git config core.hooksPath .githooks` (or `scripts/ci/install-git-hooks.sh`) makes the pre-push hook run that gate plus `build`/`test --workspace` automatically. Either way it's fast feedback, **not** the gate — the CI matrix (cross-OS runtime, miri/asan/tsan/loom, fuzz, differential) still runs on the PR and remains authoritative. Bypass once with `TEMEN_HOOK_SKIP=1 git push`; skip just the slow step with `TEMEN_HOOK_SKIP_TESTS=1`. See `.githooks/README.md`.

**Rebuilding committed assets: always through `scripts/rebuild-assets.sh`.** The prebuilt `.temen` playground/self-host binaries (`browser/web/assets/*.temen(.gz)`, the prebuilt libc unit `browser/web/assets/pg_libc.temeno`, `crates/temen-run/demos/*/…temen`, `browser/tests/fixtures/*.temen`) are **wire-format-coupled**: any IR / encoder / wire-version change invalidates them — they decode as `BadOpcode` and their code-coupled gate tests (`leng_selfhost_asset`, `nifler_asset`, `nim_hello_asset`, `pg_libc_asset`, and the `real-browser` play cards) go red. When that happens (or you change a frontend/translator), regenerate **every** asset with `bash scripts/rebuild-assets.sh` — the single entry point. Don't invoke the per-asset builders by hand and don't hand-roll the toolchain env: the script wires up what each builder needs (the real Nim toolchain dir for `nimbase.h`; `NIFLER_BIN`/`NIMONY_BIN`/`NIM_BIN` from the vendored `nimony/bin`; the translator + `prep_temen`), runs each builder fail-soft (a missing toolchain SKIPs that asset), and re-validates every output (decode → verify → bytecode-compile). `ONLY=leng,nim_hello bash scripts/rebuild-assets.sh` rebuilds a subset. If an asset needs a build path the script doesn't cover yet, **add it to the script** rather than running it standalone — the script is the source of truth for how assets are built. Then `git add` the changed assets and commit.

