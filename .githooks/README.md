# .githooks — shared, opt-in git hooks

Version-controlled hooks that run locally. Right now there's one: **`pre-push`**, a
fast mirror of CI's `check` job (`build · test · fmt · clippy`). It exists to catch the
common, boring failures before they cost a CI round-trip.

It is **fast feedback, not the gate.** The authoritative gate is still CI on the PR —
the cross-platform matrix (Windows/macOS), the sanitizer/model lanes (miri, asan, tsan,
loom), fuzzing, and the differential suites. None of those can run meaningfully on one
developer's machine at push time, and a green hook does not imply a green PR. Never treat
the hook as a substitute for CI.

## Enable (per clone, opt-in)

Hooks are off by default — git only honors this directory once you point it here:

```sh
git config core.hooksPath .githooks
# or, equivalently:
scripts/ci/install-git-hooks.sh
```

Disable again with `git config --unset core.hooksPath`.

## Bypass a single push

```sh
TEMEN_HOOK_SKIP=1 git push        # skip the hook entirely (same as `git push --no-verify`)
TEMEN_HOOK_SKIP_TESTS=1 git push  # run fmt + clippy + build, skip the slower test step
```

The hook also auto-skips branch-deletion pushes (nothing to build).
