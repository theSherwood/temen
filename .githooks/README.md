# .githooks — shared, opt-in git hooks

Version-controlled hooks that run locally. Right now there's one: **`pre-push`**, a fast mirror of
CI's gating lint/compile jobs plus the workspace tests. It exists to catch the common, boring
failures before they cost a CI round-trip.

The lint/compile half lives in **`scripts/ci/local-gate.sh`**, which the hook calls — one list of
gates with two entry points (the hook, and a manual `scripts/ci/local-gate.sh`) rather than two
lists that drift apart.

## The toolchain is pinned to CI's, and that is the point

`local-gate.sh` reads `RUST_STABLE` out of `.github/workflows/ci.yml` and runs every check through
`rustup run <that version>`. A clippy clean on a *different* rustc than CI's is not a signal: clippy
gains lints between releases, so "clippy passes" only means anything with respect to a version. A
`clippy::for_kv_map` error once failed two CI jobs while the identical command was clean locally,
purely because the local default toolchain was older. Reading the version from the workflow means
the local gate follows a CI bump without anyone remembering this file exists.

If the pinned toolchain (or its `clippy`/`rustfmt`/windows-gnu target) is missing, the gate stops
and prints the exact `rustup` command; `TEMEN_GATE_INSTALL=1` installs it instead.

## What the gate covers beyond the old hook

- `cargo clippy -p temen-jit --features stack-check` — its own CI lane, because the gating `check`
  job builds *without* the feature, so nothing else compiles that code at all. It has gone red on a
  lint the workspace run could not have seen.
- the `x86_64-pc-windows-gnu` cross-check of both — compile-only, so it catches a Windows-only build
  break on Linux, without a Windows runner. Skipped with a notice when no mingw cross-compiler is
  installed (`bash scripts/ci/apt-prep.sh gcc-mingw-w64-x86-64`), or with `TEMEN_GATE_SKIP_CROSS=1`.

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
TEMEN_HOOK_SKIP_TESTS=1 git push  # run the lint/compile gate + build, skip the slower test step
TEMEN_GATE_SKIP_CROSS=1 git push  # skip the windows-gnu cross-check inside the gate
TEMEN_GATE_INSTALL=1 git push     # let the gate install a missing toolchain/component/target
```

The hook also auto-skips branch-deletion pushes (nothing to build).

## What `--workspace` misses

The `--workspace` steps do **not** cover the crates the root
`Cargo.toml` `exclude`s (`crates/temen-llvm`, `crates/temen-webgpu`, `fuzz`, `bench`) — they are
separate workspace roots, so every `--workspace` command skips them silently. That is a real hole,
not a theoretical one: `temen-webgpu` stopped compiling when `HostProc` gained a parameter and no
job noticed for three weeks.

`scripts/ci/check-excluded.sh` closes it, and the hook runs it. It reads the exclude list from
`Cargo.toml` rather than restating it, so a crate added to that list is covered automatically. The
`excluded crates compile` CI job runs the same script, pinning per-crate toolchains via
`TEMEN_CHECK_TOOLCHAIN` / `TEMEN_CHECK_TOOLCHAIN_<dir>`.
