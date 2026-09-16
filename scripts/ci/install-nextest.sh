#!/usr/bin/env bash
# Install the pinned `cargo-nextest` on a CI runner (Linux, macOS or Windows).
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
# THE ONE PLACE THE NEXTEST VERSION LIVES. Pinned by version **and** by SHA-256 of each platform's
# release archive: a prebuilt binary fetched off the network earns the same treatment as an action
# pinned by commit SHA. Bump the version and all three checksums together —
#   for p in linux mac windows-tar; do
#     curl -sSLf "https://get.nexte.st/<version>/$p" | sha256sum
#   done
set -euo pipefail
NEXTEST_VERSION=0.9.144
NEXTEST_SHA256_LINUX=8a4f726272b0a1c499bd87ca3978bfbb1a8c20bb08ccf075b9996e2081bd1e1e
NEXTEST_SHA256_MAC=87472b2c3ee09154cadd34d168a5cdede478806810f095c3dc4dc499943da42e
NEXTEST_SHA256_WINDOWS=b0d6a6569d4ef63a095c5a574a6856c17fb755b51a02a17e4651e738e9192831

# `windows-tar`, not `windows`: the plain windows asset is a zip, and Git Bash — the shell this runs
# under on windows-latest — has GNU tar, which cannot read one, and no guaranteed `unzip`. The
# tarball endpoint keeps ONE extraction path for all three platforms instead of a second one that
# only ever runs where it is hardest to debug.
case "$(uname -s)" in
  Linux*)                        platform=linux;       want=$NEXTEST_SHA256_LINUX;   bin=cargo-nextest ;;
  Darwin*)                       platform=mac;         want=$NEXTEST_SHA256_MAC;     bin=cargo-nextest ;;
  MINGW*|MSYS*|CYGWIN*|Windows*) platform=windows-tar; want=$NEXTEST_SHA256_WINDOWS; bin=cargo-nextest.exe ;;
  *) echo "install-nextest: unsupported platform $(uname -s)" >&2; exit 1 ;;
esac

dest="${CARGO_HOME:-$HOME/.cargo}/bin"
# A Windows runner can hand us a native `C:\Users\…` CARGO_HOME, which every path below would then
# mangle. Normalise to the shell's own view when we are on one.
if command -v cygpath >/dev/null 2>&1; then
  dest="$(cygpath -u "$dest")"
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# Retries for the same reason the other fetch steps carry them (I34): a transient CDN hiccup should
# cost seconds, not a red run.
curl -sSLf --retry 5 --retry-delay 2 \
  "https://get.nexte.st/${NEXTEST_VERSION}/${platform}" -o "$tmp/nextest.tar.gz"

# macOS ships `shasum`, not GNU coreutils' `sha256sum`.
if command -v sha256sum >/dev/null 2>&1; then
  actual="$(sha256sum "$tmp/nextest.tar.gz" | cut -d' ' -f1)"
else
  actual="$(shasum -a 256 "$tmp/nextest.tar.gz" | cut -d' ' -f1)"
fi
if [ "$actual" != "$want" ]; then
  echo "install-nextest: checksum mismatch for cargo-nextest ${NEXTEST_VERSION} (${platform})" >&2
  echo "  expected $want" >&2
  echo "  got      $actual" >&2
  exit 1
fi

mkdir -p "$dest"
tar xzf "$tmp/nextest.tar.gz" -C "$dest" "$bin"
"$dest/$bin" --version
