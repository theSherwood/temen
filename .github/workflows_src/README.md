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

- **`fork_diff` fuzz target added to the `fuzz` matrix (`ci.yml`)** — one new entry in the nightly
  `cargo-fuzz (all targets)` matrix (no new job): `fork_diff` differentials native **bytecode fork**
  (`clone_caller`/`reap` on the cooperative `drive`) against the tree-walk oracle over fuzz-chosen
  reply/status values, hardening the single hand-written `SRC_TWIN` pin. The target
  (`fuzz/fuzz_targets/fork_diff.rs` + its `[[bin]]` in `fuzz/Cargo.toml`) and a stable seed-sweep
  counterpart (`crates/svm/tests/fork_fuzz.rs`, gated on every run) are ordinary tree code and land
  with this PR; only the matrix line needs the copy-over. The `workflows-in-sync` guard reds CI until
  it is applied.

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
  job because the toolchain isn't there; this job provides it so they actually execute. **Two things
  to do on copy-over:** (1) pin `alaviss/setup-nim@0.1.1` by SHA (left as a tag — no vetted SHA to
  hand); (2) confirm the heavy cold build (~10-15 min) fits the runner budget — it's a mirror of
  nim-lang/nimony's own CI and hasn't been run in *this* repo's CI yet, so the first green run is the
  real validation.

*(Previously drained 2026-07-30, when the whole backlog was copied over: the `workflows-in-sync`
guard, nightly-only `miri`, `cross-os` `CARGO_PROFILE_TEST_DEBUG: "0"`, the `playground-assets` job +
the `pages.yml` reachability step, the `bench_chibicc_jit.mjs` / `browser-shell-test.mjs` /
chibicc-asset browser steps, the hardened `embench` fetch, and the full-`fuzz_targets` matrix.)*

> **Reminder for whoever drains this next:** `miri` no longer runs on PRs. If it is still listed as
> a *required* status check in branch protection, remove it there — a skipped required check blocks
> merges.

Remove entries from this list when they land in `.github/workflows/`.
