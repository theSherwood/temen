#!/usr/bin/env bash
# Install the pinned `cargo-nextest` on a CI runner.
#
# WHY, not just what: `cargo test` runs integration-test **binaries** one at a time. Tests *inside* a
# binary parallelise; the binaries themselves never do. `crates/temen-llvm` has 99 of them and ~8.7 min
# of test execution, so that serialisation was most of the job once the interpreter tax was gone
# (#1467). `cargo nextest run` puts every test from every binary in one pool: measured 8.7 min ->
# ~3.8 min on a 4-core runner, running the *identical* set (562 passed / 27 skipped either way).
# It is not faster per test — it just stops leaving three cores idle.
#
# nextest does not run doctests; the jobs that use it run `cargo test --doc` alongside so that stays
# honest rather than silently dropped.
#
# THE ONE PLACE THE NEXTEST VERSION LIVES. Pinned by version **and** by SHA-256 of the release
# tarball: a prebuilt binary fetched off the network earns the same treatment as an action pinned by
# commit SHA. Bump the two together —
#   curl -sSLf https://get.nexte.st/<version>/linux | sha256sum
set -euo pipefail
NEXTEST_VERSION=0.9.144
NEXTEST_SHA256=8a4f726272b0a1c499bd87ca3978bfbb1a8c20bb08ccf075b9996e2081bd1e1e

dest="${CARGO_HOME:-$HOME/.cargo}/bin"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# Retries for the same reason the other fetch steps carry them (I34): a transient CDN hiccup should
# cost seconds, not a red run.
curl -sSLf --retry 5 --retry-delay 2 \
  "https://get.nexte.st/${NEXTEST_VERSION}/linux" -o "$tmp/nextest.tar.gz"

actual="$(sha256sum "$tmp/nextest.tar.gz" | cut -d' ' -f1)"
if [ "$actual" != "$NEXTEST_SHA256" ]; then
  echo "install-nextest: checksum mismatch for cargo-nextest ${NEXTEST_VERSION}" >&2
  echo "  expected $NEXTEST_SHA256" >&2
  echo "  got      $actual" >&2
  exit 1
fi

mkdir -p "$dest"
tar xzf "$tmp/nextest.tar.gz" -C "$dest" cargo-nextest
"$dest/cargo-nextest" --version
