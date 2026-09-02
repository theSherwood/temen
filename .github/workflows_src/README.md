# workflows_src — the editable mirror of `.github/workflows/`

The CI token used by agent sessions cannot push files under `.github/workflows/`
(GitHub requires the `workflow` OAuth scope). This directory is the workaround:
agents edit workflow files **here**, and the repo owner applies them by copying
the directory over:

```sh
cp .github/workflows_src/*.yml .github/workflows/
```

(then commit and push with owner credentials). Keep the two in sync in that
direction only — `workflows_src` is the source of truth for *pending* changes;
`.github/workflows/` is what actually runs. After a copy-over the two are
identical until the next agent edit.

## Pending changes not yet copied over

- **`nested_paged` in the `fuzz` job matrix (`ci.yml`, #1151)** — one row added after `paged_walk`
  (plus a comment line): the new `fuzz/fuzz_targets/nested_paged.rs` target (the emitted §14 nested
  paged scalar page check vs the interpreter oracle, page-op bounce serviced on a real vCPU). Until
  copied over, the `fuzz targets wired` lockstep guard (`scripts/ci/check-fuzz-matrix.sh`) is red on
  the PR — the target file and its `fuzz/Cargo.toml` `[[bin]]` exist but the live matrix lacks the row;
  `cp .github/workflows_src/*.yml .github/workflows/` drains it.

- **`browser-bash-bg-test.mjs` in the `browser-real` job (`ci.yml`, #798 bg/&)** — one line added after
  `node browser-bash-jobs-test.mjs`: `node browser-bash-bg-test.mjs`. It drives the interactive bash
  card's **background job launch**: `seq 3 &` runs in its own process group without a terminal handoff
  or a blocking wait — bash prints `[1] pid`, the prompt stays usable, the job's output streams, and the
  async `[1]+ Done seq 3` posts on the next prompt; then a clean `^D` exit. SKIPs cleanly with the rest
  of the bash batch when the deploy-built `bash.temen`/`bin_seq.temen` are absent, so it reds only on a
  real background-job regression. Verified locally in Chromium. (Until copied over, the
  `workflows-in-sync` guard stays red — the expected mirror-edit friction;
  `cp .github/workflows_src/*.yml .github/workflows/` drains it.)

- **`browser-nim-op13-crawl-e2e-test.mjs` in the `browser-real` job (`ci.yml`, #1025 Path 1, whole card)** —
  one line added after `node browser-op13jit-nifler-test.mjs`: `node browser-nim-op13-crawl-e2e-test.mjs`. It
  drives the **whole nim card through the op-13 loop**: the compile's phase-1 nifler crawl runs each module
  as a nested op-13 emitted child (nifler_ce, `{fs,stdout,exit}` marshaled, the nifler_ce emit cached across
  modules), then nimsem/hexer/link/run finish — and the card's output is asserted byte-identical whether
  phase-1 ran on the interpreter or as nested op-13 emitted children (`hello, Nim…`, 4 modules crawled). Uses
  the committed nim assets + `nifler_ce.temen.gz` (stages a temp gunzipped copy it deletes); reuses the
  threads wasm the job already builds. Verified locally in Chromium. (Until copied over, the
  `workflows-in-sync` guard stays red; `cp .github/workflows_src/*.yml .github/workflows/` drains it.)

- **`browser-op13jit-nifler-test.mjs` in the `browser-real` job (`ci.yml`, #1025 Path 1, real nifler)** —
  one line added after `node browser-op13jit-e2e-test.mjs`: `node browser-op13jit-nifler-test.mjs`. It scales
  the op-13 loop to a **real phase child**: a driver marshals `{fs, stdout, exit}` to nifler_ce (child-entry)
  and JS runs its `_start` on emitted wasm over the carve; it reads the source from the marshaled memfs and
  writes `.p.nif`, asserted byte-identical to the interpreter oracle (top-level nifler on the tree-walker) —
  `nimc.rs` phase-1 nifler crawl, nested under op-13 and tiered up to JIT. Uses the committed
  `nifler_ce.temen.gz` + `web/assets/nifler.temen.gz` (stages a temp gunzipped copy it deletes); reuses the
  threads wasm the job already builds. Verified locally in Chromium (174B `.p.nif` ≡ oracle). (Until copied
  over, the `workflows-in-sync` guard stays red; `cp .github/workflows_src/*.yml .github/workflows/` drains it.)

- **`browser-op13jit-e2e-test.mjs` in the `browser-real` job (`ci.yml`, #1025 Path 1)** — one line added
  after `node browser-jit-runtime-grow-test.mjs`: `node browser-op13jit-e2e-test.mjs` (with a comment). It
  pins the JS-orchestrated §14 op-13 loop in real V8: a resumable driver marshals an `fs` grant to a
  confined child, JS runs that child's `_start` on **emitted wasm** over its carve, the child's `call.cap`
  leaf resolves the *marshaled* `fs` on the reactor cross-tier bounce and returns 41, and the driver joins
  it — the browser realization of `nimc.rs::drive_op13` with the child tiered up. No new toolchain (reuses
  the threads wasm the job already builds); parses its guest in-page, so no assets. Verified locally in
  Chromium (result 41, counter 1). (Until copied over, the `workflows-in-sync` guard stays red — the
  expected mirror-edit friction; `cp .github/workflows_src/*.yml .github/workflows/` drains it.)

- **`browser-bash-jobs-test.mjs` in the `browser-real` job (`ci.yml`, #798 multiple concurrent jobs)** —
  one line added after `node browser-bash-interactive-test.mjs`: `node browser-bash-jobs-test.mjs`. It
  drives the interactive bash card's **job table with several jobs**: `^Z` two `cat`s so both sit stopped,
  `jobs` lists them with the `[1]-`/`[2]+` previous/current markers, `fg %1`/`fg %2` resume each by job
  spec (re-`^Z` between them), then the stopped-jobs `^D` exit. SKIPs cleanly with the rest of the bash
  batch when the deploy-built `bash.temen`/`bin_cat.temen` are absent, so it reds only on a real
  job-control regression. Pairs with the shim fix in this change (`bash_shim.c` VSUSP marshalling) and the
  native cross-engine differential (`c_posix.rs::c_multiple_background_jobs_stop_and_continue_across_process_groups`).
  Verified locally in Chromium. (Until copied over, the `workflows-in-sync` guard stays red — the expected
  mirror-edit friction; `cp .github/workflows_src/*.yml .github/workflows/` drains it.)

- **`browser-durable-persist-reload-test.mjs` in the `browser-real` job (`ci.yml`, #816 Slice C)** — one
  line added after `node browser-jit-runtime-grow-test.mjs`: `node browser-durable-persist-reload-test.mjs`.
  It is the invariant-14 **durability axis, cross-host leg** pin: a `vm_map`-GROWN durability-instrumented
  guest frozen to a §12 snapshot artifact (`temen_durable_freeze`), persisted to **IndexedDB**, and — after a
  genuine **page reload** into a fresh WebAssembly instance — thawed and resumed to completion
  (`temen_durable_thaw_resume`), with the grown-page content surviving the reload. The shipped-path proof
  of the browser "persist a warmed/grown guest across a reload" consumer (the native oracle
  `crates/temen/tests/durable_grown_snapshot_resume.rs` pins the mechanism). No new toolchain (reuses the
  threads wasm the job already builds); parses its guest in-page via `temen_parse`, so no assets; benign
  resource 404s don't gate it. Verified locally in Chromium. (Until copied over, the `workflows-in-sync`
  guard stays red — the expected mirror-edit friction; `cp .github/workflows_src/*.yml .github/workflows/`
  drains it.)

- **`browser-jit-runtime-grow-test.mjs` in the `browser-real` job (`ci.yml`, #1155)** — one line added
  after `node browser-tierup-mainline-test.mjs`: `node browser-jit-runtime-grow-test.mjs`. It is the
  invariant-14 **code-origin axis** pin: a §22 guest-JIT unit (code the guest `vm_jit_compile`s at
  runtime) running on emitted wasm over a `vm_map`-GROWN window, through the shipped `runJitModule` →
  `driveCoopTierupRun` coop driver, asserted byte-identical to the interpreter oracle in real V8 — the
  first in-browser exercise of the guest-runtime-JIT growth path (the native `coop_tierup_driver.rs`
  covers the mechanism against a reimplemented driver). No new toolchain (reuses the threads wasm the job
  already builds); parses its guest in-page via `temen_parse`, so no assets; benign resource 404s don't
  gate it. Verified locally in Chromium. (Until copied over, the `workflows-in-sync` guard stays red —
  the expected mirror-edit friction; `cp .github/workflows_src/*.yml .github/workflows/` drains it.)

- **New `browser-host-guests` job in `ci.yml` (browser runtime-invariant gate, #816/#1094 follow-up)** —
  a per-PR job added right after `browser-jit-host-gates`. The browser crate is its own cargo
  workspace, so the gating `check` job's `cargo test --workspace` never builds or runs it; its Rust
  host tests were exercised only by the nightly full run and dev machines. That gap let #1094's
  unconditional NULL guard land on `main` silently breaking a batch of hand-written-guest tests
  (`fork`, `mem_profile`, `onramp_posix`, `powerbox_cap_names`, `warm_grow`, and the tier-up drivers)
  with no per-PR check to catch it. The new job runs the browser host tests whose guest modules are
  written *inline* (no staged toolchain / built asset) — `coop_tierup_driver`, `par_tierup_driver`,
  `fork`, `warm_grow`, `mem_profile`, `onramp`, `onramp_posix`, `powerbox_cap_names`, `emit_budget`,
  `dap_batch`, `link_run`, `setjmp`, `pg_snapshot_roundtrip` — so a powerbox / NULL-guard / tier-up
  invariant change fails a PR here. The heavy self-host/asset/real-guest tests (chibicc\*, jacl, bash,
  doom, reactor cards) stay in `browser-real` + the nightly full run; they need staged assets and are
  slow (the whole browser suite is ~6 min warm vs. seconds for this set). **Until copied over, the
  `workflows_src == workflows` check stays red and browser host tests are not gated per-PR.** After
  copy-over, consider adding `browser host guests (runtime-invariant gate)` to the branch-protection
  required-checks set so it actually blocks merges.

- **bash asset staging in `pages.yml` (the DEPLOY workflow, #1080/#1122 deploy fix)** — two steps
  added to the `build` job, right after `build self-host closure image` and before the Postgres
  cache step: a `cache bash build inputs` step (`actions/cache` on `/tmp/temen_bash_cache/bash_linked.ll`,
  keyed on `crates/temen-run/demos/bash/**`) and a `build + stage bash artifacts` step
  (`node build-bash-assets.mjs || echo "bash assets skipped …"`). `pages.yml` is the workflow that
  actually publishes the site (scheduled every 30 min); it never built bash, so the published
  playground shipped no `bash.temen`/`bin_*.temen` and the bash / `bash -i` cards 404'd — silently,
  because those assets are in check-play-assets' `MAY_BE_ABSENT`. This mirrors what `ci.yml`'s
  `real-browser` job already does, so whichever workflow publishes, the cards work. No new toolchain
  (the job already installs llvm-18/clang-18 for the on-ramp assets); GPLv3-safe (fetched-and-built,
  never committed). **Until copied over, the deployed bash cards stay 404.**
  Same file, one more line: the change-gate's reachability curl points at the wrong project-site
  path (`https://thesherwood.github.io/vm/DEPLOYED_SHA`) — the site serves at `…/temen/`, so the
  gate always 404s → always thinks the site changed → rebuilds+republishes on every 30-min tick.
  Fixed to `…/temen/DEPLOYED_SHA` so the gate correctly skips no-op rebuilds when the site is
  already at HEAD.

- **New `browser-jit-host-gates` job (the compiler-tier wasm-JIT host tests, #1011)** — the browser
  crate's Rust *host* tests (`nifler_jit`, `chibicc_jit`, `jit_module`) run a real compiler phase on
  the wasm-JIT via `wasmi` and assert the emitted output is byte-identical to the interpreter oracle,
  but nothing in `ci.yml` ran them — they were local/dev gates only. The new job runs them on
  `ubuntu-latest`; `nifler_jit` uses the committed `nifler.temen.gz` (no staging), the others SKIP
  fail-soft without their built assets. (Until copied over, the `workflows-in-sync` guard stays red —
  the expected mirror-edit friction; `cp .github/workflows_src/*.yml .github/workflows/` drains it.)

- **bash playground assets + Chromium E2E + the shell test's silent skip (`ci.yml`, #1080)** — three
  additions to the `browser-real` job:
  (1) after the chibicc staging step, a `cache bash build inputs` step (caches
  `/tmp/temen_bash_cache/bash_linked.ll`, keyed on `crates/temen-run/demos/bash/**`) and a fail-soft
  `build + stage bash artifacts` step (`node build-bash-assets.mjs || echo skipped` — bash is GPLv3,
  fetched from ftp.gnu.org and built at deploy, never committed); (2) `node browser-bash-test.mjs`
  after `browser-shell-test.mjs` in the real-browser test block — it drives the new `temen_run_bash`
  entry (real GNU bash: fork per pipeline stage, execve per /bin coreutil, CorePipe parks, blocking
  waitpid) in Chromium and SKIPs cleanly when the staged asset is absent, so an ftp.gnu.org outage
  degrades to a SKIP, never a red gate; (3) the `browser-shell-test.mjs` staging cp now copies ALL
  FOUR of that test's gate assets (`shell`,`stage_runner`,`primes`,`upper`) — copying only
  `shell.temen` left the shell test silently SKIPping (exit 0) on every PR, which is how the wasm32
  personality-clock panic (`Instant::now` in `temen_posix::grant`, fixed in this PR) went unnoticed.
  (Until copied over, `workflows-in-sync` stays red — the expected mirror-edit friction;
  `cp .github/workflows_src/*.yml .github/workflows/` drains it.)

- **`paged_walk` added to the nightly `fuzz` matrix** (issue #1081) — one entry added to the `fuzz`
  job's `target: [...]` list (after `coverage_walk`), plus the descriptive comment above it. It is the
  new libFuzzer target `fuzz/fuzz_targets/paged_walk.rs`: the generative interp⇄**wasm-JIT** differential
  for the **paged bulk-memory per-page walk** (`emit_span_page_check` vs the interpreter's
  `check_prot_span`) — the tree's *fourth* confinement lowering, which `wasm_diff` can't reach (it
  suppresses cap.call/page-op modules). Runs like every other target (`cargo fuzz run paged_walk --
  -max_total_time=300`). The `fuzz-matrix-in-sync` guard ("fuzz targets wired") **reds the run until this
  is copied over** — the new `fuzz_targets/paged_walk.rs` + `fuzz/Cargo.toml [[bin]]` have no matching
  matrix row in the live `ci.yml` until the copy lands. The stable `crates/temen/tests/paged_walk.rs`
  gates it per-PR from deterministic seeds (485 trapping / 1015 passing spans on 1500 seeds, verified
  locally). (Until copied over, both `workflows-in-sync` and `fuzz targets wired` stay red — the expected
  mirror-edit friction; `cp .github/workflows_src/*.yml .github/workflows/` drains it.)

- **`full-depth-lua`: shard per (suite × tier) (`ci.yml`, #1074 follow-up)** — the split landed with a
  `{bytecode, tree_walker}` matrix, each tier running all three Lua suites in one job under
  `timeout-minutes: 75`. Validation run #3536 showed the **tree_walker tier alone exceeds 75 min**
  (bytecode-tier ~35 min; tree_walker-tier hit the 75-min cap and was cancelled). The matrix is now
  `suite: [lua_tlib, lua_all, lua_sweep] × tier: [bytecode, tree_walker]` = 6 shards, each running one
  `cargo test --test <suite> -- --ignored <tier>`, so the longest single shard is well under the cap.
  Same tests/coverage, finer fan-out. (Until copied over, the `workflows-in-sync` guard stays red —
  the expected mirror-edit friction; `cp .github/workflows_src/*.yml .github/workflows/` drains it.)

- **`excluded-crates-compile` per-PR job (`ci.yml`)** — a new lightweight always-on job that
  `cargo check`s the two workspace-EXCLUDED crates, `bench/` (pinned stable) and `fuzz/` (pinned
  `nightly-2026-07-01`, matching the `fuzz` job). Because the always-on `check` job builds only the
  workspace, an API change in a workspace crate breaks these two silently and reddens only a nightly
  job — the `align` (E0559) and `CompiledModule::run` 4→3 (E0061) drifts both slipped to nightly this
  way. This lane catches that class per-PR. `check` only (no link/run) to stay cheap; caches both
  workspaces via `Swatinem/rust-cache`. Verified locally: `cargo check` is clean for both under
  `-D warnings`. (Until copied over, the `workflows-in-sync` guard stays red — the expected
  mirror-edit friction; `cp .github/workflows_src/*.yml .github/workflows/` drains it.)

- **I67 apt hardening for the `Install Playwright + Chromium` step (`browser-real` job, #1017)** —
  the one apt consumer the 10-site I67 hardening missed: `npx playwright install --with-deps`
  runs its **own** internal `apt-get update && apt-get install` with the runner's default (azure)
  sources, and it runs *before* the job's hardened LLVM step. An azure-mirror outage on 2026-08-19
  wedged the step to its 10-minute timeout **three consecutive times** on PR #1015 (and once on
  PR #1016 — see #1017): the `Ign:` retry storm, then a stalled mirrorlist fetch, Playwright never
  installed. The step now leads with the exact same scrub line the other 10 sites use (drop
  `sources.list.d/{microsoft,azure}*`, sed `apt-mirrors.txt`/`.sources` onto
  `https://archive.ubuntu.com`, fail-soft `|| true`). No behavior change on a healthy runner.
  (Until copied over, the `workflows-in-sync` guard stays red — the expected mirror-edit friction;
  `cp .github/workflows_src/*.yml .github/workflows/` drains it.)

- **`arena-stacks` no-op feature removed from the `stack-guard` + `stack-guard-cross-os` jobs (#919)** —
  the `temen-fiber`/`temen-jit` `arena-stacks` feature was a retained no-op (the arena is temen-fiber's
  always-on default backend now), so every `--features stack-check,arena-stacks` invocation just
  re-tested the default config. The two jobs now pass `--features stack-check` only (that feature still
  gates the `temen-jit` guard test suites into the lane), and the two `cargo test -p temen-fiber --features
  arena-stacks` rows — pure default-config re-runs, already covered by the `check` job's workspace test —
  are deleted. Job comment + `name:` updated (no longer "off by default"). **⚠️ Copy-over is required to
  keep these jobs green, not just to drain the `workflows-in-sync` guard:** the same PR deletes the
  `arena-stacks` Cargo feature, so until `cp .github/workflows_src/*.yml .github/workflows/` lands, the
  **live** `ci.yml` still runs `--features arena-stacks` against a crate that no longer defines it and
  the `stack-guard`/`stack-guard-cross-os` jobs fail with "does not contain this feature: arena-stacks".
  After copy-over both are green (verified locally: `cargo test -p temen-jit --features stack-check` and the
  default `temen-fiber`/`temen-jit` builds pass; `--features arena-stacks` now errors, as intended).

- **`fuzz-matrix-in-sync` job (#923)** — a new lightweight ubuntu job ("fuzz targets wired") that
  runs `scripts/ci/check-fuzz-matrix.sh`, which asserts the three places a fuzz target is named stay
  identical: the source files (`fuzz/fuzz_targets/*.rs`), the build entries (`fuzz/Cargo.toml`
  `[[bin]]`), and the nightly `fuzz` job's `target: [...]` matrix. A target file with no `[[bin]]`
  never builds; one with no matrix row builds but never runs — silent zero coverage, the exact hole
  the fuzz job's "keep this list in lockstep with `fuzz/fuzz_targets/*.rs`" comment only warns about
  in prose. The script is committed and runs locally (`./scripts/ci/check-fuzz-matrix.sh`); it passes
  today (20 targets in sync) and catches both a stray file and a missing matrix row (verified). (Until
  copied over, the `workflows-in-sync` guard stays red — the expected mirror-edit friction; `cp
  .github/workflows_src/*.yml .github/workflows/` drains it.)

- **Nim conformance matrix step in the `nim-e2e` job (#956)** — one step added right after the
  `Nim end-to-end tests` step: `cargo test -p temen-leng --test nim_conformance -- --nocapture`. Runs the
  new `crates/temen-leng/tests/nim_conformance.rs` — a feature→status matrix (generics, exceptions,
  closures, methods, `seq`/`string`/`Table`, floats, iterators, variant objects, `ref`+ARC) driven
  through the whole real toolchain and asserted against a committed baseline (a feature that starts
  working *or* regresses fails the test). Self-skips (passes) without the toolchain, exactly like
  `nim_e2e`, so it only truly runs in this job. No new toolchain — reuses the one this job already
  builds. Verified locally against the vendored nimony (`11/15` features run today). (Until copied over,
  the `workflows-in-sync` guard stays red — the expected mirror-edit friction; `cp
  .github/workflows_src/*.yml .github/workflows/` drains it.)

- **`nim-e2e`: wait for the `latest-devel` nightly before `setup-nim`** (issue #856) — a new
  `run:` step ("Wait for the Nim devel nightly to be published") added to the `nim-e2e` job
  immediately before the `Setup Nim (devel)` step in `ci.yml`. `alaviss/setup-nim` with
  `version: devel` resolves to the nim-lang/nightlies `latest-devel` release, which is transiently
  absent while the nightly is re-cut — the observed flake failed in ~188 ms with "Could not find any
  release named 'latest-devel'", before any repo code ran. The new step polls the nightlies release
  API (`releases/tags/latest-devel`) with bounded backoff (6 attempts, ~5 min max) so the re-cut
  window self-heals; a genuine upstream outage still fails the job inside the 45-min budget. No new
  action dependency (pure `curl` + `github.token`), and it does not change which Nim is used. (Until
  copied over, the `workflows-in-sync` guard stays red — the expected mirror-edit friction;
  `cp .github/workflows_src/*.yml .github/workflows/` drains it.)

- **`wasm_diff` added to the nightly `fuzz` matrix** (issue #910) — one entry added to the
  `fuzz` job's `target: [...]` list (after `diff`), plus the descriptive comment above it. It is the
  new libFuzzer target `fuzz/fuzz_targets/wasm_diff.rs`: the generative interp⇄**wasm-JIT**
  differential (the tree's third confinement lowering, `emit_confine`/`emit_span_check`), the wasm-tier
  peer of `diff`. Runs like every other target (`cargo fuzz run wasm_diff -- -max_total_time=300`). The
  matrix comment already says "keep this list in lockstep with `fuzz/fuzz_targets/*.rs`", so an unwired
  target would be zero coverage. Builds under the pinned nightly + `cargo-fuzz`; the stable
  `crates/temen/tests/wasm_diff.rs` gates it per-PR from seeds. (Until copied over, the `workflows-in-sync`
  guard stays red — the expected mirror-edit friction; `cp .github/workflows_src/*.yml
  .github/workflows/` drains it.)

- **I67 apt hardening, second azure path (all 10 `apt-get update` sites, `ci.yml`)** — the
  sources.list.d removal turned out to cover only half of the runner's azure dependency: apt ALSO
  routes through the `/etc/apt/apt-mirrors.txt` mirrorlist (and `.sources` stanzas), which still
  name `azure.archive.ubuntu.com`. PR #953's gate lost a 15-minute step timeout to an azure-mirror
  outage *after* the existing hardening ran (an `Ign:` retry storm, then a stalled fallback fetch).
  Each site now also seds the mirrorlist/stanzas onto `https://archive.ubuntu.com` (fail-soft
  `|| true` — a healthy runner is untouched). Same class, same shape, one more door closed.

- **`timeout-minutes: 45` on the `temen-llvm` job** (issue #906) — it was the only lane with no
  timeout, so a wedged compile ran to GitHub's 6-hour default before reporting. Observed once: the
  `std_guest` native-oracle `rustc` hung 66+ min on PR #898's run (the suite unexpectedly *ran*
  there rather than auto-skipping — the runner had a usable nightly). The harness-side fix (the
  #788 bounded-wait now also covers the native-oracle compiles, `std_guest.rs`) is the primary
  guard; this timeout is the backstop. Normal lane time ~20 min.

- **`lua-warm-snapshot-test.mjs` + Lua coverage in `snapshot-worker-test.mjs` (`browser-real` job, issue
  #805)** — one line added after `node warm-jit-test.mjs`: `node lua-warm-snapshot-test.mjs` (Node/V8
  cold ≡ warm ≡ warm+JIT byte-for-byte + isolation over the committed `lua_snapshot.temen`). The existing
  `node snapshot-worker-test.mjs` line is unchanged, but the test itself now also drives the **Lua** warm
  card (one worker per module, so QuickJS + Lua stay warm at once). Both use committed assets; skip
  cleanly if absent. Verified locally (Node + Chromium). (Until copied over, the `workflows-in-sync` guard
  stays red — the expected mirror-edit friction.)

- **`tcl-warm-snapshot-test.mjs` + Tcl coverage in `snapshot-worker-test.mjs` (`browser-real` job, issue
  #805 follow-on)** — one line added after `node lua-warm-snapshot-test.mjs`: `node
  tcl-warm-snapshot-test.mjs` (Node/V8 cold ≡ warm byte-for-byte + isolation over `tcl_snapshot.temen`;
  Tcl's warm+JIT declines, so interpreter-only). The `snapshot-worker-test.mjs` line is unchanged, but
  the test now also drives the **Tcl** warm card (`noJit` — warm-snapshot-only). Tcl's `.temen` is
  **deploy-built** (the Tcl fetch + toolchain isn't in this job), so both tests SKIP/filter cleanly when
  it's absent — no new toolchain, no gating. Verified locally (Node + Chromium). (Until copied over, the
  `workflows-in-sync` guard stays red — the expected mirror-edit friction.)

- **`warm-snapshot-test.mjs` in the `browser-real` job** — one line added to the Chromium test block
  (right after `node browser-jit-cache-test.mjs`): `node warm-snapshot-test.mjs`. Validates the
  WASM_AOT.md warm-runtime snapshot: `temen_warm_open` runs the QuickJS `warmup` export once, then
  `temen_warm_eval` restores the post-init image and runs `eval_run`, which must match the cold `_start`
  path (`temen_run_onramp`) byte-for-byte while skipping the runtime rebuild. Uses the committed
  `web/assets/qjs_snapshot.temen`; skips cleanly if absent. Reuses the wasm the job already builds — no
  new toolchain. Verified locally in Node/V8. (Until copied over, the `workflows-in-sync` guard stays
  red — the expected mirror-edit friction; `cp .github/workflows_src/*.yml .github/workflows/` drains it.)

- **`warm-jit-test.mjs` in the `browser-real` job** — one line added right after `node
  warm-snapshot-test.mjs`: `node warm-jit-test.mjs`. Validates the WASM_AOT.md **warm+JIT** tier
  (issue #783): `temen_warm_jit_open` emits the QuickJS `eval_run` to wasm once, `runWarmJit` (from
  `web/wasmjit-module.js`) drives it over the restored snapshot each Run — which must match the
  interpreter warm path (`temen_warm_eval`) byte-for-byte, keep fresh-per-Run isolation (a `var` in one
  Run cannot leak into the next), and accelerate a compute-heavy eval (measured ~9× on a 500k-iteration
  loop; a trivial program stays on warm-interp). Uses the committed `web/assets/qjs_snapshot.temen`;
  skips cleanly if absent. Reuses the threads wasm the job already builds — no new toolchain. Verified
  locally in Node/V8.

- **`snapshot-worker-test.mjs` in the `browser-real` job** — one line added right after `node
  warm-jit-test.mjs`: `node snapshot-worker-test.mjs`. Validates the **snapshot worker** (issue #804):
  the QuickJS card's warm session runs on a dedicated Web Worker (its own engine instance + private
  memory), pre-warmed off the main thread; each Run is a message round-trip. Chromium-drives
  `play.html` and asserts the card runs end-to-end on both warm tiers (warm-snapshot + warm+JIT), that
  the work actually went **through the worker** (a `globalThis.__snapshotWorkerRuns` counter increments —
  so a silent main-thread fallback fails the test), and that fresh-per-Run isolation holds. Uses the
  committed `web/assets/qjs_snapshot.temen`; skips cleanly if absent. Reuses the threads wasm the job
  already builds — no new toolchain. Verified locally in Chromium. (Until copied over, the
  `workflows-in-sync` guard stays red — the expected mirror-edit friction.)

- **`browser-tierup-mainline-test.mjs` in the `browser-real` job** — one line added to the Chromium
  test block (right after the already-copied `node browser-jit-cache-test.mjs`):
  `node browser-tierup-mainline-test.mjs`. Validates slice-2 mainline tier-up over a live window (the
  slice-0 JACL residual): an TEMEN-text compute guest run with tier-up on must equal the all-interpreter
  value and tier-up must fire (no assets — parses in-page via `temen_parse`). Reuses the threads module
  the job already builds — no new toolchain. Verified locally in Chromium. (The sibling
  `browser-jit-cache-test.mjs` line was already copied over in commit `90c3b6d`.)

- **Doc-only CI skip (`ci.yml` `on:` triggers).** Added `paths-ignore: ["**.md"]` to both the
  `push: [main]` and `pull_request` triggers so a changeset that touches **only** Markdown skips the
  whole CI matrix (it's slow, and prose edits don't affect build/test/fuzz). `paths-ignore` skips a run
  only when *every* changed file matches, so a mixed code+doc commit still runs the full matrix. The one
  generated-and-golden-tested Markdown file (`OPS_PARITY.md`, checked by `temen-parity/tests/golden.rs`) is
  always regenerated alongside a `.rs` change, so mixed-changeset CI still covers it; only a lone
  hand-edit of that generated file would slip through, which is already a misuse. The `schedule` (nightly
  fuzz) and `workflow_dispatch` triggers are unaffected — they always run.

  > **Owner action on copy-over (required checks):** if `build · test · fmt · clippy` (or any other
  > matrix job) is a **required** status check in branch protection, a doc-only PR will skip it and
  > GitHub will report the required check as never-run, **blocking the merge** — the same footgun as the
  > `miri` note below. Either drop the requirement for doc-only PRs or merge those without the gate.

- **Pages-deploy starvation fix (I75) — two cooperating paths in `ci.yml` + `pages.yml`.** The per-push
  (`push: [main]`) Pages deploy was starving: under a burst of agent-PR merges each new merge supersedes
  the still-queued deploy (`concurrency: pages`, `cancel-in-progress: false` cancels the *queued* older
  run), so it never wins a runner before the next merge resets it — observed **0 of 30** recent Pages
  runs completed, one sat queued **8 h with zero jobs ever scheduled**, and the live site froze days
  behind `main`. The fix has two parts (copy **both** files over):

  1. **`ci.yml` — primary deploy rides the required `browser-real` job (the correct fix).** That job
     already builds the whole deployable site (wasm + Postgres + chibicc + self-host) on every main push
     and reliably gets a runner (it's a required gate). It now also (main push only, **fail-soft**)
     runs `build-onramp-assets.mjs`, assembles `_site` + a `DEPLOYED_SHA` marker, verifies reachability
     (`check-play-assets --site`), and `upload-pages-artifact`; a new tiny **`deploy-pages`** job
     (`needs: browser-real`, `id-token`+`pages: write`, `github-pages` environment) publishes it in
     seconds — no separate heavy job to starve, and the deploy happens the moment main's browser CI is
     green. The assemble step never exits nonzero, so a Pages-asset hiccup sets `site_ok=false` (deploy
     skipped) but **cannot red the required `browser-real` gate**.
  2. **`pages.yml` — scheduled self-healing fallback.** Drop the `push` trigger; run on
     `schedule: */30 * * * *` + `workflow_dispatch`. Its **`gate`** job compares the published
     `DEPLOYED_SHA` against `HEAD` and skips when the site is already current — so it is a ~10 s no-op
     whenever path 1 succeeded, and only does the full standalone build+deploy when path 1 didn't
     publish HEAD (browser-real failed/was cancelled). The `github-pages` environment serializes the two
     paths; the gate prevents a double-deploy. `workflow_dispatch` always builds (on-demand /
     feature-branch preview).

  **On copy-over:** verify the repo's Pages **Source = GitHub Actions** (unchanged); the first green
  `browser-real` on main is the real validation of the assemble+upload+deploy path (I couldn't run a
  live Pages deploy from the agent env). Merges publish as soon as `browser-real` is green; the 30-min
  fallback covers any gap.

- **I67 apt-source hardening (all `apt-get update` steps)** — every job that runs `apt-get update`
  (the mingw cross lanes, the `clang` reference lanes, and all `llvm-18` install blocks — 10 sites)
  now first `sudo rm -f`s the runner's unused `microsoft*`/`azure*` files under
  `/etc/apt/sources.list.d/`. Those repos are never installed from, but a transient 403/outage from
  their mirror (ISSUES.md I67) kills `apt-get update` with exit 100 before any Rust runs. Removing
  the sources makes the update independent of them. No behavior change on a healthy runner. Pure CI
  infra; no tree code touched.

> **A CI guard now enforces this list.** The `workflows-in-sync` job (`workflows_src == workflows`)
> reds the run whenever any `.github/workflows_src/*.yml` differs from `.github/workflows/*.yml`, so
> pending changes can't be silently forgotten — the run stays red until the owner copies them over
> (`cp .github/workflows_src/*.yml .github/workflows/`) and this section is drained. The guard only
> starts enforcing once it is itself copied into `.github/workflows/ci.yml` for the first time.

- **`cc1-self-compile-giants` job** — a new **nightly** (`schedule` + `workflow_dispatch`) Linux job
  that runs the giant cc1 TUs (`preprocess.c`/`parse.c`/`codegen_ir.c`) through the guest-vs-native
  differential with `TEMEN_SELFHOST_GIANTS=1`. ~8 min locally (more on CI), too slow for the per-PR gate,
  so it rides the daily cron like `miri`. Together with the five tractable TUs in the always-on
  `cc1-self-compile` job it completes per-TU byte-identity across **all nine** cc1 TUs — the sufficient
  condition for the `chibicc2 == chibicc3` fixpoint. (The always-on job already runs the giant test too
  via `-- --ignored`, but it self-skips fast without the env var.)

- **`full-depth-gates` job** — a new **nightly** (`schedule` + `workflow_dispatch`) Linux job that runs
  the `#[ignore]`d full-depth *correctness* gates that no CI job previously ran: Lua's suite
  (`lua_tlib`/`lua_all`/`lua_sweep`) on both the bytecode engine and the tree-walker, plus the
  whole-language capstones (`demo_tcl_repl_stdin`/`demo_tcl_init_stdin` and the full
  `demo_sqlite_logictest_full` sweep) via `cargo test --test … -- --ignored` from `crates/temen-llvm`
  (workspace-excluded, so run from its dir). Each asserts byte-identity with the native `cc` build.
  `#[ignore]`d only for wall-clock (minutes per suite on the tree-walker), so it rides the daily cron
  like `miri`/giants rather than the per-PR gate — closing the JIT-only blind spot that let the QuickJS
  on-ramp recipe drift unseen once. Capstones self-skip loudly (never fail) without clang/curl/make/
  openlibm, so grep the log for `skipping` before trusting a green run. First green run on CI is the
  real validation of the ~90-min timeout budget.

- **`nim-e2e` job** — builds the real nimony toolchain (`scripts/ci/provision-nimony.sh`, cached) and
  runs `crates/temen-leng/tests/nim_e2e.rs`, which compiles small **Nim source** programs through
  `nimony c` and runs them on both Temen engines. The tests self-skip (pass) in the always-on `check`
  job because the toolchain isn't there; this job provides it so they actually execute. **Things
  to do on copy-over:** (1) pin `alaviss/setup-nim@0.1.1` by SHA (left as a tag — no vetted SHA to
  hand); (2) confirm the heavy cold build (~10-15 min) fits the runner budget — it's a mirror of
  nim-lang/nimony's own CI and hasn't been run in *this* repo's CI yet, so the first green run is the
  real validation. **(3) NEW — the `nim-e2e` checkout now needs `submodules: recursive`** (added in
  `workflows_src/ci.yml`): `nimony` + `nativenif` are vendored as **git submodules** (pinned in
  `.gitmodules`), and `provision-nimony.sh` now `git submodule update --init`s them instead of cloning
  a hard-coded SHA. Without the recursive checkout the submodule dirs are empty and the toolchain
  build fails. Only this job's checkout changed; the others stay bare. **(4) NEW (#856) — a
  "Wait for the Nim devel nightly to be published" step now precedes `setup-nim`**, gating on the
  nim-lang/nightlies `latest-devel` release so the transient re-cut window self-heals instead of
  reddening the run (see the pending-changes entry above).

- **`std-guest` job** (#821) — a new **nightly** (`schedule` + `workflow_dispatch`) Linux job that runs
  the `crates/temen-llvm/tests/std_guest.rs` suite, which no CI job previously executed (it auto-skips in
  the always-on `temen-llvm` job because nightly + `rust-src` + the applied temen std overlay aren't there).
  The job installs nightly + `rust-src`, runs `rust-temen/apply-overlay.sh`, and runs the suite serially.
  It exercises **both** target specs — the lean `x86_64-unknown-temen` and the new threaded
  `x86_64-unknown-temen-threads` (`singlethread=false`: futex `sys/sync` + native TLS) — through
  `-Zbuild-std` → on-ramp → verify → powerbox. Guards against the #788 build-std wedge two ways: a firm
  `timeout-minutes: 75`, and a per-build kill-and-skip in the harness (`TEMEN_STD_BUILD_TIMEOUT_SECS`,
  set to 480). `RUSTFLAGS: ""` overrides the workflow-global `-D warnings` (build-std recompiles std).
  First green run on CI is the real validation of the time budget (~12 tests × two targets × ~40 s
  serial). No new toolchain beyond nightly+rust-src.

*(Previously drained 2026-07-30, when the whole backlog was copied over: the `workflows-in-sync`
guard, nightly-only `miri`, `cross-os` `CARGO_PROFILE_TEST_DEBUG: "0"`, the `playground-assets` job +
the `pages.yml` reachability step, the `bench_chibicc_jit.mjs` / `browser-shell-test.mjs` /
chibicc-asset browser steps, the hardened `embench` fetch, and the full-`fuzz_targets` matrix.)*

> **Reminder for whoever drains this next:** `miri` no longer runs on PRs. If it is still listed as
> a *required* status check in branch protection, remove it there — a skipped required check blocks
> merges.

Remove entries from this list when they land in `.github/workflows/`.
