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

- **`cc1-self-compile` job filter** — the dedicated Linux job's run line broadened from
  `-- --ignored self_compiles` to `-- --ignored` (no name filter), so it runs **every** heavy
  `#[ignore]`d c_link gate — both `whole_cc1_self_compiles…` and the stage-2
  `whole_cc1_compiles_its_own_tu…` (the guest compiling chibicc's *own* TUs). The narrower filter
  silently skipped the latter. (The job itself — installing `clang`, `-j 2` + no-debug I30 OOM guard,
  parallel to `check` — landed in a prior copy-over; only the run line changed here.)

*(Previously drained 2026-07-30, when the whole backlog was copied over: the `workflows-in-sync`
guard, nightly-only `miri`, `cross-os` `CARGO_PROFILE_TEST_DEBUG: "0"`, the `playground-assets` job +
the `pages.yml` reachability step, the `bench_chibicc_jit.mjs` / `browser-shell-test.mjs` /
chibicc-asset browser steps, the hardened `embench` fetch, and the full-`fuzz_targets` matrix.)*

> **Reminder for whoever drains this next:** `miri` no longer runs on PRs. If it is still listed as
> a *required* status check in branch protection, remove it there — a skipped required check blocks
> merges.

Remove entries from this list when they land in `.github/workflows/`.
