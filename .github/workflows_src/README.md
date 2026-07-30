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

> **A CI guard now enforces this list.** The `workflows-in-sync` job (`workflows_src == workflows`)
> reds the run whenever any `.github/workflows_src/*.yml` differs from `.github/workflows/*.yml`, so
> pending changes can't be silently forgotten — the run stays red until the owner copies them over
> (`cp .github/workflows_src/*.yml .github/workflows/`) and this section is drained. The guard only
> starts enforcing once it is itself copied into `.github/workflows/ci.yml` for the first time.

- **ci.yml** (2026-07-29): add the `workflows-in-sync` guard described above, and gate the `miri`
  job to `schedule` + `workflow_dispatch` (nightly + manual) instead of every PR — Miri interprets
  `svm-mem` and takes ~18 min, second only to the windows `build · test` on the critical path, and
  the unsafe substrate it guards changes rarely (it now rides the daily cron like asan/tsan). **If
  `miri` is a required status check in branch protection, remove it there** (a skipped required check
  blocks merges). Also fixes a pre-existing pending line: the "build + stage chibicc asset" step's
  inline `run: node -e "…skipped: "+e.message"` was invalid YAML (a colon-space in a plain scalar —
  GHA would reject the whole file); it is now a `|` block scalar so the copy-over is safe.

- **ci.yml** (2026-07-29): `cross-os` (windows/macOS `build · test`) gains
  `env: CARGO_PROFILE_TEST_DEBUG: "0"`, mirroring the `check` gate. Windows `build · test` is CI's
  wall-clock critical path (~20 min, vs ~18 for miri and ≤12 for everything else); dropping unused
  test-binary debug info (PDB generation on Windows especially) cuts a large slice of its
  compile+link, and also lowers the Windows link-memory peak I3 aborts on. Behavior unchanged
  (backtraces keep symbol names). CI-speed pass — see ISSUES/PR.

- **ci.yml** (2026-07-28): the real-Chromium browser step runs `node bench_chibicc_jit.mjs` (after
  `browser-play-editor-test.mjs`) — chibicc compile-time on V8, bytecode vs wasm-JIT: prints the
  speedup and asserts the two tiers emit byte-identical IR (a second guard alongside `chibicc_jit.rs`).
  Reuses the threads cdylib + the committed/staged `chibicc.svmb`; SKIPs if the asset is absent, and
  timing is informational so it only reds on IR divergence. See SELFHOST_C.md §7 / PR #483.

- **ci.yml** (2026-07-27): the real-Chromium browser step stages the committed
  `browser/tests/fixtures/shell.svmb` into `web/assets/` and runs
  `node browser-shell-test.mjs` — the `svm-posix` shell playground card driven
  through `svm_run_shell` (STAGE1.md). Skips cleanly if the asset is absent.

- **ci.yml** (2026-07-24): `check` job gains `env: CARGO_PROFILE_TEST_DEBUG: "0"` — the I30
  linker-OOM runner deaths recurred twice on PR #427 *with* the `-j 2` cap (sightings 4-5);
  dropping test-profile debug info removes the dominant per-link memory term. See ISSUES.md I30.
- **ci.yml** (2026-07-24): `embench differential` fetch hardened with `curl -f --retry 5
  --retry-all-errors` — codeload occasionally serves an HTML error page that `tar xz` can't
  detect ("not in gzip format"). See ISSUES.md I18 class 4.
- **ci.yml** (2026-07-24): `fuzz` matrix expanded from the 6 escape-TCB targets to **every**
  target in `fuzz/fuzz_targets/` (adds `onramp_diff`, `roundtrip`, `opt_sccp`,
  `opt_ssa_roundtrip`, `coverage_walk`, and the `durable*` freeze/thaw family) — no
  built-but-unwired fuzzer. Job renamed `cargo-fuzz (all targets)`.
- **ci.yml** (2026-07-24): `cross-os` job — removed the stale commented-out `continue-on-error`
  TODO; the job is already gating.
- **ci.yml** (2026-07-24): `real-browser` job gains a `build + stage chibicc asset` step (after
  `build + stage Postgres artifacts`) so `browser-play-editor-test.mjs` runs the playground
  C-compiler card's compile-and-run assertion instead of SKIPping it. Fail-soft: a build hiccup
  degrades to a SKIP, never a red build (the test guards on `web/assets/chibicc.svmb` existing).
  See PR #441 / SELFHOST_C.md §7 step 5.
- **ci.yml + pages.yml** (2026-07-27): playground-asset reachability gate (ISSUES.md I26/I42/I49).
  New `ci.yml` job **`playground-assets`** runs `node browser/check-play-assets.mjs` on every PR —
  every asset `web/play.js` references must be committed or declared deploy-built, else red (cheap,
  no toolchain). `pages.yml` gains a **`verify playground assets reachable`** step (after `assemble
  site`, before `upload-pages-artifact`) running the script in `--site` mode against `_site`, so a
  fail-soft asset build dropping a required asset goes red *before* the site publishes instead of
  shipping a silent 404. Both are driven from `play.js`, so new demo cards are covered automatically.

Remove entries from this list when they land in `.github/workflows/`.
