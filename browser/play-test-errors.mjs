// Shared benign-404 filter for tests that load `play.html`.
//
// A warm card (issues #804/#805) pre-warms its snapshot on page load, fetching its `.svmb`. A
// **deploy-built** asset (e.g. `tcl_snapshot.svmb`, whose Tcl fetch + toolchain isn't in the
// committed-asset browser job) is absent there, so that fetch 404s — a benign console error the
// browser logs and JS can't suppress (the prewarm's `.catch` degrades the card, but the console
// entry remains). Every page-load test with a strict console handler would fail on it.
//
// `benignAssetMiss(msg)` returns true for a resource-load 404/403 **only** when the asset isn't on
// disk (i.e. legitimately deploy-built-absent in this job). A 404 for a *committed* asset is a real
// regression and returns false, so the test still surfaces it. Assets live at `browser/web/assets/`,
// resolved relative to this module — independent of the caller's cwd.
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const ASSETS = join(dirname(fileURLToPath(import.meta.url)), 'web', 'assets');

export function benignAssetMiss(msg) {
  if (msg.type() !== 'error' || !/Failed to load resource.*\b40[34]\b/.test(msg.text())) return false;
  const hit = (msg.location()?.url || '').match(/\/assets\/([^/?#]+)/);
  return !!hit && !existsSync(join(ASSETS, hit[1]));
}
