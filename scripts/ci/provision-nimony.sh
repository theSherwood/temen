#!/usr/bin/env bash
# Build the nimony toolchain (nifler → nimony → hexer → lengc …) for the Nim-source end-to-end
# tests (`crates/temen-leng/tests/nim_e2e.rs`). Mirrors nim-lang/nimony's own CI build.
#
# Builds its own Nim compiler from a pinned commit first (see NIM_SRC_REV below), so it needs only git
# and a C toolchain. Produces the tools under `<workdir>/nimony/bin` and prints two `KEY=value` lines
# the caller `eval`s / appends to $GITHUB_ENV:
#     NIMONY_BIN=<abs>/nimony/bin
#     NIM_BIN=<abs>/.cache/temen-ci/nim/bin
#
# The nimony frontend (and its sibling `nativenif`) are vendored as **git submodules** — the exact
# commit is pinned by the gitlink in `.gitmodules`, reproducible in-tree, and bumped deliberately in
# lockstep with any temen-leng change the newer frontend requires (`git -C nimony checkout <ref>` +
# commit the submodule). This replaces the old in-script clone-and-checkout of a hard-coded SHA.
set -euo pipefail

# The caller appends our stdout to $GITHUB_ENV, so ONLY the final `KEY=value` lines may go there —
# send every command's chatter (submodule fetch, the `hastur` build) to stderr, and emit the two
# result lines on the saved stdout (fd 3) at the end.
exec 3>&1 1>&2

WORK="${1:-${GITHUB_WORKSPACE:-$PWD}}"
cd "$WORK"

# Ensure the vendored submodules are checked out at their pinned commits. A CI checkout with
# `submodules: recursive` already does this, so this is a no-op there; it makes a plain checkout or a
# local run work too. Not shallow — the pinned commit need not be a branch tip, so the full fetch is
# required to resolve it. `nativenif` must sit beside `nimony` (the native backend's nim.cfg reaches
# it via `../nativenif`); both are repo-root submodules, so that sibling layout holds.
git submodule update --init nimony nativenif

# `nativenif` is pinned TWICE and both pins must agree. Ours is the submodule gitlink; nimony's own
# is `src/nativenif.commit`, which `hastur build all` checks out into `../nativenif` before building
# arkham/nifasm from it. The submodule update above runs first and resets the checkout to OUR pin, so
# when the two disagree the build silently proceeds against whichever one ran last — and a nimony
# bump that moves its pin without moving the submodule fails deep inside the build with no hint that
# a pin is the reason. Say so here instead, before anything is built.
NATIVENIF_PIN_FILE=nimony/src/nativenif.commit
if [ -f "$NATIVENIF_PIN_FILE" ]; then
  want="$(awk '{print $1; exit}' "$NATIVENIF_PIN_FILE")"
  have="$(git -C nativenif rev-parse HEAD)"
  if [ "$want" != "$have" ]; then
    echo "error: nativenif pin mismatch — the nimony submodule wants $want" >&2
    echo "       ($NATIVENIF_PIN_FILE) but our submodule gitlink is $have." >&2
    echo "       Move both together: git -C nativenif checkout $want && git add nativenif" >&2
    exit 1
  fi
fi

# The Nim compiler, built from a pinned nim-lang/Nim commit (bump it deliberately, like the nimony
# submodule). nifler compiles Nim's own parser, so it needs current compiler sources, and the `lib/`
# beside them must match: an unpinned devel over an older nightly split the tree in two (#1220). This
# used to overlay the pinned `compiler/` and `lib/` onto the prebuilt devel nightly. Building the
# pinned tree itself gives one coherent toolchain and drops the nightly, whose `latest-devel` release
# went missing while it was re-cut (#856) and moved under us every day (#1839). CI caches the build
# in `~/.cache/temen-ci/nim` alongside the nimony tools, so it runs only when the key changes; it
# takes ~5 min on a 4-core runner.
NIM_SRC_REV=973065b279d2ae5b3954c25348c7dc4a02335f2b # nim-lang/Nim devel, 2026-09-03
NIM_ROOT="$HOME/.cache/temen-ci/nim"
fetch_rev() { # <dir> <url> <commit>: a shallow checkout of exactly <commit>
  rm -rf "$1"
  git init -q "$1"
  git -C "$1" fetch -q --depth 1 "$2" "$3"
  git -C "$1" checkout -q FETCH_HEAD
}
if [ "$(git -C "$NIM_ROOT" rev-parse HEAD 2>/dev/null)" != "$NIM_SRC_REV" ] ||
  ! "$NIM_ROOT/bin/nim" -v >/dev/null 2>&1; then
  fetch_rev "$NIM_ROOT" https://github.com/nim-lang/Nim "$NIM_SRC_REV"
  (
    cd "$NIM_ROOT"
    . ci/funs.sh
    nimDefineVars
    # Nim's own bootstrap clones the tip of csources' branch and then checks out the pinned hash,
    # which fails once the branch moves on. Fetch the pinned hash itself.
    fetch_rev "$nim_csourcesDir" "$nim_csourcesUrl" "$nim_csourcesHash"
    nimBuildCsourcesIfNeeded
    bin/nim c --noNimblePath --skipUserCfg --skipParentCfg --hints:off koch
    ./koch boot -d:release --skipUserCfg --skipParentCfg --hints:off
    # Keep the toolchain, not the bootstrap: its C sources and objects are 2 GB.
    rm -rf "$nim_csourcesDir" nimcache bin/nim_csources_*
  )
fi
NIM_BIN="$NIM_ROOT/bin"
export PATH="$NIM_BIN:$PATH"

# There is a second Nim revision in play, and it used to be invisible: nifler vendors a copy of Nim's
# parser and records the revision it was taken from in `src/nifler/nimparser/upstream.commit`. That
# is the revision upstream's own CI uses. Ours is deliberately separate (we bump it when a devel
# change breaks the build — see #1220 above), but a large drift between the two is the first thing
# to suspect when nifler stops compiling, so print both rather than leave the reader to discover the
# second one by hitting it.
NIFLER_UPSTREAM_FILE=nimony/src/nifler/nimparser/upstream.commit
if [ -f "$NIFLER_UPSTREAM_FILE" ]; then
  echo "provision-nimony: Nim $NIM_SRC_REV; nifler's vendored parser is from \
$(cat "$NIFLER_UPSTREAM_FILE")" >&2
fi

# hastur resolves the frontend's NIF libs via `nim/dist/nimony` — point it at our submodule checkout.
mkdir -p "$NIM_ROOT/dist"
rm -rf "$NIM_ROOT/dist/nimony"
ln -sf "$WORK/nimony" "$NIM_ROOT/dist/nimony"

# Build all the tools (C backend; the E2E harness invokes `nimony c`) — unless a cache restore
# already brought them back. This build is ~4 min and is the whole cost of this step; CI caches
# `nimony/bin` for it but, until now, rebuilt regardless of whether the cache hit.
#
# The check USES the toolchain rather than listing files. `nimony c` shells out to nimsem, hexer,
# nifler, lengc, niflink, nifmake, validator and shoggoth (hastur's `BootSelfTools` +
# `BootCarryTools`), so a file-existence test here would be a second copy of that list, free to
# drift from hastur's whenever a tool is added. A trivial compile exercises the chain end to end,
# which is the actual precondition, and it self-heals: anything partial or broken falls through to
# the build below instead of being served to the tests.
#
# It also closes a gap the tests cannot see. `nim_e2e.rs` / `nim_conformance.rs` decide to SKIP on
# whether `nimony` *exists* — so a half-restored or broken cache would not skip, it would surface
# as confusing failures in the suite proper. Verifying here means the toolchain is either good or
# rebuilt before a test looks at it.
toolchain_works() {
  [ -x nimony/bin/nimony ] || return 1
  local probe rc
  probe="$(mktemp -d)"
  # Mirrors the smallest shape in `crates/temen-leng/tests/nim_diff/` — `echo` lives in
  # `std/syncio` under nimony, so a bare `echo` would fail to compile and quietly pin this to
  # "always rebuild". Same `c --isMain` the harness uses, so a passing probe means the invocation
  # the tests make works, not merely that a binary is present.
  printf 'import std/syncio\necho "ok"\n' > "$probe/probe.nim"
  if ( cd "$probe" && "$WORK/nimony/bin/nimony" c --isMain probe.nim ) >/dev/null 2>&1; then
    rc=0
  else
    rc=1
  fi
  rm -rf "$probe"
  return "$rc"
}

if toolchain_works; then
  echo "provision-nimony: restored nimony/bin compiles a probe — skipping the hastur build." >&2
else
  # `src/hastur/hastur.nim`, not `src/hastur`: hastur became a directory of modules, so the bare
  # directory no longer resolves to a compilable file. `--release` sits AFTER the filename on
  # purpose — everything before it is nim's, everything after is hastur's, and `nim c -r x --release`
  # would hand the flag to nim and leave hastur with `build all` alone.
  ( cd nimony && nim c -r src/hastur/hastur.nim --release build all )
fi

echo "NIMONY_BIN=$WORK/nimony/bin" >&3
echo "NIM_BIN=$NIM_BIN" >&3
