#!/usr/bin/env bash
# Install apt packages on a CI runner from a cache of their debs, or only prepare apt for an install:
#   bash scripts/ci/apt-prep.sh [package or .deb…]
#   bash scripts/ci/apt-prep.sh --key [package or .deb…]    # prints the cache's `key=` and `dir=`
# With packages: install them, and whatever the image lacks for them, from the debs cached in
# `$TEMEN_APT_CACHE`, filling the cache from the archive first if it is empty. A `.deb` is installed
# as given; the cache holds what the archive served for it. With none: scrub only, for a script that
# adds a repo and runs its own update (`install-llvm.sh`).
#
# THE ONE PLACE APT LIVES. `apt-get update` fails hard when *any* configured index is inconsistent,
# and the `ubuntu-latest` image preconfigures third-party repos we never install from — so a publish
# window or an outage on someone else's mirror reds the whole matrix before a line of our code runs.
# Twice now (#1017: the azure mirror; #1374: `dl.google.com` serving a `Packages.gz` that did not match
# its own Release file for ~1 h, taking 7 jobs down at their apt step). This incantation used to be
# copy-pasted into five workflow steps plus `install-llvm.sh`, so #1374's one-word fix needed six
# edits — INVARIANTS #15: one path per behaviour.
#
# THE CACHE (#2947). An install used to update from and download from Ubuntu's archive on every run,
# so on 2026-11-11 a runner that could not resolve the archive failed every job that installs
# anything, `main`'s included, before a line was built. `.github/actions/apt-install` restores the
# cache from the Actions cache under the key `--key` prints, which names this script, every package
# the image has installed and what is asked for: one image and one request, one key. A run that
# finds its cache installs from it with no archive configured, so it cannot reach for one. Only a
# new runner image, a new request or a change here fills a cache.
set -euo pipefail
shopt -s nullglob

# Drop every third-party source the image ships. We install only from Ubuntu's own archive (plus the
# LLVM repo `install-llvm.sh` adds *after* this runs), so none of these can be needed — and each one
# is a mirror whose bad hour becomes our red matrix.
scrub() {
  sudo rm -f /etc/apt/sources.list.d/microsoft* \
    /etc/apt/sources.list.d/azure* \
    /etc/apt/sources.list.d/google-chrome*
  # The azure mirror also served stale/500s as the *default* archive; point the default at the
  # canonical one. Best-effort: the files differ across image revisions, so a miss is not an error.
  sudo sed -i 's|http://azure.archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' \
    /etc/apt/apt-mirrors.txt /etc/apt/sources.list.d/*.sources 2>/dev/null || true
  # Fetch only the indices an install reads. The image's apt also pulls AppStream metadata (DEP-11
  # `Components`) and `Translation` files on every `apt-get update`, which no job reads but any of
  # which fails the update when a mirror is mid-sync — #1788's red run was two DEP-11 files served
  # that way.
  sudo tee /etc/apt/apt.conf.d/99temen-index-targets >/dev/null <<'CONF'
Acquire::IndexTargets::deb::DEP-11::DefaultEnabled "false";
Acquire::IndexTargets::deb::DEP-11-icons-small::DefaultEnabled "false";
Acquire::IndexTargets::deb::DEP-11-icons::DefaultEnabled "false";
Acquire::IndexTargets::deb::DEP-11-icons-hidpi::DefaultEnabled "false";
Acquire::Languages "none";
CONF
}

if [ "${1:-}" = --key ]; then
  shift
  # A `.deb` counts by its file name, which names its package, version and architecture.
  sum=$({ cat "$0"; dpkg-query -W; for a in "$@"; do basename "$a"; done; } | sha256sum | cut -c1-24)
  key="apt-$(. /etc/os-release && echo "$VERSION_CODENAME")-$sum"
  echo "key=$key"
  echo "dir=$HOME/.cache/temen-ci/apt/$key"
  exit 0
fi
if [ "$#" -eq 0 ]; then
  scrub
  exit 0
fi

names=() files=()
for a in "$@"; do
  case $a in
    *.deb) files+=("$a") ;;
    *) names+=("$a") ;;
  esac
done

# Outside CI: a cache for this install alone.
cache=${TEMEN_APT_CACHE:-}
if [ -z "$cache" ]; then
  cache=$(mktemp -d)
  trap 'rm -rf "$cache"' EXIT
fi
if [ ! -e "$cache/filled" ]; then
  scrub
  sudo apt-get update
  # Downloaded where apt's sandbox user can reach, then kept: what the archive served, but not the
  # `.deb`s given, which apt copies in too and which come from a cache of their own.
  fetched=$(mktemp -d)
  chmod 755 "$fetched"
  sudo apt-get install -y --download-only -o Dir::Cache::archives="$fetched" "$@"
  mkdir -p "$cache"
  for deb in "$fetched"/*.deb; do
    keep=yes
    for f in "${files[@]}"; do
      if [ "$(basename "$f")" = "$(basename "$deb")" ]; then keep=no; fi
    done
    if [ "$keep" = yes ]; then cp "$deb" "$cache/"; fi
  done
  sudo rm -rf "$fetched"
  touch "$cache/filled"
fi

# The install reads the cache and the `.deb`s given, and nothing else: with no sources configured, a
# dependency missing from them fails here by name instead of being fetched.
nosources=$(mktemp -d)
debs=("$cache"/*.deb "${files[@]}")
if [ "${#debs[@]}" -gt 0 ]; then
  sudo apt-get install -y -o Dir::Etc::SourceList=/dev/null -o Dir::Etc::SourceParts="$nosources" \
    "${debs[@]}"
fi
rmdir "$nosources"
# Every package asked for is installed, from the cache or by the image already.
for p in "${names[@]}"; do
  if [ "$(dpkg-query -W -f '${db:Status-Abbrev}' "$p" 2>/dev/null)" != "ii " ]; then
    echo "apt-prep: $p is not installed" >&2
    exit 1
  fi
done
