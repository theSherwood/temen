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

- **`lua-warm-snapshot-test.mjs` + Lua coverage in `snapshot-worker-test.mjs` (`browser-real` job, issue
  #805)** — one line added after `node warm-jit-test.mjs`: `node lua-warm-snapshot-test.mjs` (Node/V8
  cold ≡ warm ≡ warm+JIT byte-for-byte + isolation over the committed `lua_snapshot.svmb`). The existing
  `node snapshot-worker-test.mjs` line is unchanged, but the test itself now also drives the **Lua** warm
  card (one worker per module, so QuickJS + Lua stay warm at once). Both use committed assets; skip
  cleanly if absent. Verified locally (Node + Chromium). (Until copied over, the `workflows-in-sync` guard
  stays red — the expected mirror-edit friction.)

- **`tcl-warm-snapshot-test.mjs` + Tcl coverage in `snapshot-worker-test.mjs` (`browser-real` job, issue
  #805 follow-on)** — one line added after `node lua-warm-snapshot-test.mjs`: `node
  tcl-warm-snapshot-test.mjs` (Node/V8 cold ≡ warm byte-for-byte + isolation over `tcl_snapshot.svmb`;
  Tcl's warm+JIT declines, so interpreter-only). The `snapshot-worker-test.mjs` line is unchanged, but
  the test now also drives the **Tcl** warm card (`noJit` — warm-snapshot-only). Tcl's `.svmb` is
  **deploy-built** (the Tcl fetch + toolchain isn't in this job), so both tests SKIP/filter cleanly when
  it's absent — no new toolchain, no gating. Verified locally (Node + Chromium). (Until copied over, the
  `workflows-in-sync` guard stays red — the expected mirror-edit friction.)

- **`warm-snapshot-test.mjs` in the `browser-real` job** — one line added to the Chromium test block
  (right after `node browser-jit-cache-test.mjs`): `node warm-snapshot-test.mjs`. Validates the
  WASM_AOT.md warm-runtime snapshot: `svm_warm_open` runs the QuickJS `warmup` export once, then
  `svm_warm_eval` restores the post-init image and runs `eval_run`, which must match the cold `_start`
  path (`svm_run_onramp`) byte-for-byte while skipping the runtime rebuild. Uses the committed
  `web/assets/qjs_snapshot.svmb`; skips cleanly if absent. Reuses the wasm the job already builds — no
  new toolchain. Verified locally in Node/V8. (Until copied over, the `workflows-in-sync` guard stays
  red — the expected mirror-edit friction; `cp .github/workflows_src/*.yml .github/workflows/` drains it.)

- **`warm-jit-test.mjs` in the `browser-real` job** — one line added right after `node
  warm-snapshot-test.mjs`: `node warm-jit-test.mjs`. Validates the WASM_AOT.md **warm+JIT** tier
  (issue #783): `svm_warm_jit_open` emits the QuickJS `eval_run` to wasm once, `runWarmJit` (from
  `web/wasmjit-module.js`) drives it over the restored snapshot each Run — which must match the
  interpreter warm path (`svm_warm_eval`) byte-for-byte, keep fresh-per-Run isolation (a `var` in one
  Run cannot leak into the next), and accelerate a compute-heavy eval (measured ~9× on a 500k-iteration
  loop; a trivial program stays on warm-interp). Uses the committed `web/assets/qjs_snapshot.svmb`;
  skips cleanly if absent. Reuses the threads wasm the job already builds — no new toolchain. Verified
  locally in Node/V8.

- **`snapshot-worker-test.mjs` in the `browser-real` job** — one line added right after `node
  warm-jit-test.mjs`: `node snapshot-worker-test.mjs`. Validates the **snapshot worker** (issue #804):
  the QuickJS card's warm session runs on a dedicated Web Worker (its own engine instance + private
  memory), pre-warmed off the main thread; each Run is a message round-trip. Chromium-drives
  `play.html` and asserts the card runs end-to-end on both warm tiers (warm-snapshot + warm+JIT), that
  the work actually went **through the worker** (a `globalThis.__snapshotWorkerRuns` counter increments —
  so a silent main-thread fallback fails the test), and that fresh-per-Run isolation holds. Uses the
  committed `web/assets/qjs_snapshot.svmb`; skips cleanly if absent. Reuses the threads wasm the job
  already builds — no new toolchain. Verified locally in Chromium. (Until copied over, the
  `workflows-in-sync` guard stays red — the expected mirror-edit friction.)

- **`browser-tierup-mainline-test.mjs` in the `browser-real` job** — one line added to the Chromium
  test block (right after the already-copied `node browser-jit-cache-test.mjs`):
  `node browser-tierup-mainline-test.mjs`. Validates slice-2 mainline tier-up over a live window (the
  slice-0 JACL residual): an SVM-text compute guest run with tier-up on must equal the all-interpreter
  value and tier-up must fire (no assets — parses in-page via `svm_parse`). Reuses the threads module
  the job already builds — no new toolchain. Verified locally in Chromium. (The sibling
  `browser-jit-cache-test.mjs` line was already copied over in commit `90c3b6d`.)

- **Doc-only CI skip (`ci.yml` `on:` triggers).** Added `paths-ignore: ["**.md"]` to both the
  `push: [main]` and `pull_request` triggers so a changeset that touches **only** Markdown skips the
  whole CI matrix (it's slow, and prose edits don't affect build/test/fuzz). `paths-ignore` skips a run
  only when *every* changed file matches, so a mixed code+doc commit still runs the full matrix. The one
  generated-and-golden-tested Markdown file (`OPS_PARITY.md`, checked by `svm-parity/tests/golden.rs`) is
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
  differential with `SVM_SELFHOST_GIANTS=1`. ~8 min locally (more on CI), too slow for the per-PR gate,
  so it rides the daily cron like `miri`. Together with the five tractable TUs in the always-on
  `cc1-self-compile` job it completes per-TU byte-identity across **all nine** cc1 TUs — the sufficient
  condition for the `chibicc2 == chibicc3` fixpoint. (The always-on job already runs the giant test too
  via `-- --ignored`, but it self-skips fast without the env var.)

- **`full-depth-gates` job** — a new **nightly** (`schedule` + `workflow_dispatch`) Linux job that runs
  the `#[ignore]`d full-depth *correctness* gates that no CI job previously ran: Lua's suite
  (`lua_tlib`/`lua_all`/`lua_sweep`) on both the bytecode engine and the tree-walker, plus the
  whole-language capstones (`demo_tcl_repl_stdin`/`demo_tcl_init_stdin` and the full
  `demo_sqlite_logictest_full` sweep) via `cargo test --test … -- --ignored` from `crates/svm-llvm`
  (workspace-excluded, so run from its dir). Each asserts byte-identity with the native `cc` build.
  `#[ignore]`d only for wall-clock (minutes per suite on the tree-walker), so it rides the daily cron
  like `miri`/giants rather than the per-PR gate — closing the JIT-only blind spot that let the QuickJS
  on-ramp recipe drift unseen once. Capstones self-skip loudly (never fail) without clang/curl/make/
  openlibm, so grep the log for `skipping` before trusting a green run. First green run on CI is the
  real validation of the ~90-min timeout budget.

- **`nim-e2e` job** — builds the real nimony toolchain (`scripts/ci/provision-nimony.sh`, cached) and
  runs `crates/svm-leng/tests/nim_e2e.rs`, which compiles small **Nim source** programs through
  `nimony c` and runs them on both SVM engines. The tests self-skip (pass) in the always-on `check`
  job because the toolchain isn't there; this job provides it so they actually execute. **Things
  to do on copy-over:** (1) pin `alaviss/setup-nim@0.1.1` by SHA (left as a tag — no vetted SHA to
  hand); (2) confirm the heavy cold build (~10-15 min) fits the runner budget — it's a mirror of
  nim-lang/nimony's own CI and hasn't been run in *this* repo's CI yet, so the first green run is the
  real validation. **(3) NEW — the `nim-e2e` checkout now needs `submodules: recursive`** (added in
  `workflows_src/ci.yml`): `nimony` + `nativenif` are vendored as **git submodules** (pinned in
  `.gitmodules`), and `provision-nimony.sh` now `git submodule update --init`s them instead of cloning
  a hard-coded SHA. Without the recursive checkout the submodule dirs are empty and the toolchain
  build fails. Only this job's checkout changed; the others stay bare.

- **`std-guest` job** (#821) — a new **nightly** (`schedule` + `workflow_dispatch`) Linux job that runs
  the `crates/svm-llvm/tests/std_guest.rs` suite, which no CI job previously executed (it auto-skips in
  the always-on `svm-llvm` job because nightly + `rust-src` + the applied svm std overlay aren't there).
  The job installs nightly + `rust-src`, runs `rust-svm/apply-overlay.sh`, and runs the suite serially.
  It exercises **both** target specs — the lean `x86_64-unknown-svm` and the new threaded
  `x86_64-unknown-svm-threads` (`singlethread=false`: futex `sys/sync` + native TLS) — through
  `-Zbuild-std` → on-ramp → verify → powerbox. Guards against the #788 build-std wedge two ways: a firm
  `timeout-minutes: 75`, and a per-build kill-and-skip in the harness (`SVM_STD_BUILD_TIMEOUT_SECS`,
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
