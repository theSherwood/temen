// Guard against the I26/I42/I49 class: the playground references an asset in `web/play.js`
// (`url: './assets/<name>'`) that isn't actually shipped, so the card 404s. `play.js` is the single
// source of truth for what the playground needs; this script derives the required set from it and
// checks every entry is accounted for. Adding a new demo card automatically extends coverage — no
// separate manifest to keep in sync.
//
// Two modes:
//
//   node check-play-assets.mjs            (PR gate — cheap, no toolchain)
//       Every referenced asset must be EITHER committed under browser/web/assets/ (git-tracked)
//       OR listed in BUILT_AT_DEPLOY below. An asset that is neither is a guaranteed 404: either
//       commit it (preferred — works out of the box, like hello_c/qjs_repl/chibicc) or declare it
//       deploy-built and make the build produce it. Runs on every PR (ci.yml).
//
//   node check-play-assets.mjs --site <dir>   (deploy gate — pages.yml, against the assembled _site)
//       Every referenced asset must actually EXIST in <dir>/web/assets/, except the small
//       MAY_BE_ABSENT set. Catches a fail-soft build (dead mirror, missing toolchain) dropping a
//       required asset *before* the site publishes, instead of shipping a silent 404.
//
// When you add a demo card:
//   - commit its asset under browser/web/assets/  → nothing else to do (the PR gate sees it tracked), or
//   - if it's built at deploy (big/network-dependent guest) → add it to BUILT_AT_DEPLOY, and to
//     MAY_BE_ABSENT only if it legitimately may fail to build (external fetch we don't control).

import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const PLAY_JS = join(HERE, 'web', 'play.js');
const ASSETS_REL = 'browser/web/assets';

// Assets the playground references that are NOT committed in-tree but produced by the deploy build
// (build-onramp-assets.mjs / build-pg-assets.mjs). Keep this list honest: an entry that is also
// committed, or no longer referenced by play.js, is flagged. Committed assets do NOT belong here —
// they pass the PR gate by being git-tracked.
const BUILT_AT_DEPLOY = new Set([
  'gpu_shader.svmb',      // build-onramp-assets.mjs
  'doom.svmb',            // build-onramp-assets.mjs (id's DOOM source is fetched-and-built)
  'doom1.wad',            // build-onramp-assets.mjs — staged from the vendored demos/doom/doom1.wad
  'lua_eval.svmb',        // build-onramp-assets.mjs (fetches Lua source)
  'sqlite_repl.svmb',     // build-onramp-assets.mjs (fetches SQLite amalgamation)
  'shell.svmb',           // build-onramp-assets.mjs (copies the committed tests/fixtures/shell.svmb)
  'tcl_init.svmb',        // build-onramp-assets.mjs (fetches Tcl + openlibm; full Tcl_Init)
  'postgres_resolved.svmb', // build-pg-assets.mjs
  'pgdata.img',           // build-pg-assets.mjs
]);

// The subset of deploy-built assets allowed to be absent even in the assembled site. Only doom.svmb:
// its module build fetches id Software's DOOM source and needs the toolchain, so it can legitimately
// be skipped (the card degrades to a build hint). The WAD it pairs with is NOT here — doom1.wad is
// vendored in-tree (demos/doom/doom1.wad) and staged unconditionally, so it is always reachable
// (retiring the dead-mirror class, ISSUES.md I42/I43). Every other referenced asset MUST be present.
const MAY_BE_ABSENT = new Set([
  'doom.svmb',
  'tcl_init.svmb', // build fetches Tcl (SourceForge) + openlibm (GitHub) + needs clang/llvm-link
]);

function referencedAssets() {
  const src = readFileSync(PLAY_JS, 'utf8');
  const names = new Set();
  for (const m of src.matchAll(/\.\/assets\/([A-Za-z0-9_.-]+)/g)) names.add(m[1]);
  if (names.size === 0) {
    console.error('check-play-assets: no ./assets/ references found in play.js — parser broke?');
    process.exit(2);
  }
  return names;
}

function trackedAssets() {
  // git-tracked files under browser/web/assets/ — the "committed" set. A fresh CI checkout has only
  // these under web/assets/ (the build hasn't run), so this is the authoritative committed list.
  const out = execFileSync('git', ['ls-files', ASSETS_REL], { cwd: join(HERE, '..') }).toString();
  const names = new Set();
  for (const line of out.split('\n')) {
    if (!line) continue;
    names.add(line.slice(ASSETS_REL.length + 1)); // strip "browser/web/assets/"
  }
  return names;
}

function fail(msg) {
  console.error(`✗ ${msg}`);
}

function checkRepo() {
  const referenced = referencedAssets();
  const tracked = trackedAssets();
  const errors = [];
  const warnings = [];

  for (const a of [...referenced].sort()) {
    const isTracked = tracked.has(a);
    const isBuilt = BUILT_AT_DEPLOY.has(a);
    if (!isTracked && !isBuilt) {
      errors.push(
        `play.js references ./assets/${a}, but it is neither committed under ${ASSETS_REL}/ ` +
        `nor declared in BUILT_AT_DEPLOY (check-play-assets.mjs). It will 404. Fix: commit the ` +
        `asset (preferred — works out of the box), or add it to BUILT_AT_DEPLOY and ensure the ` +
        `deploy build produces it.`,
      );
    } else if (isTracked && isBuilt) {
      warnings.push(
        `./assets/${a} is committed AND listed in BUILT_AT_DEPLOY — the list is for uncommitted ` +
        `deploy-only assets; remove it from BUILT_AT_DEPLOY.`,
      );
    }
  }

  // Stale-list hygiene: a declared deploy-built asset no longer referenced by any card.
  for (const a of [...BUILT_AT_DEPLOY].sort()) {
    if (!referenced.has(a)) {
      errors.push(
        `BUILT_AT_DEPLOY lists ${a}, but no play.js card references ./assets/${a}. Remove the ` +
        `stale entry (and its build step if unused).`,
      );
    }
  }
  for (const a of [...MAY_BE_ABSENT].sort()) {
    if (!BUILT_AT_DEPLOY.has(a)) {
      errors.push(`MAY_BE_ABSENT lists ${a}, which is not in BUILT_AT_DEPLOY — inconsistent.`);
    }
  }

  for (const w of warnings) console.warn(`⚠ ${w}`);
  errors.forEach(fail);
  if (errors.length) {
    console.error(`\ncheck-play-assets (repo): ${errors.length} problem(s).`);
    process.exit(1);
  }
  console.log(
    `check-play-assets (repo): OK — ${referenced.size} referenced asset(s); ` +
    `${[...referenced].filter((a) => tracked.has(a)).length} committed, ` +
    `${[...referenced].filter((a) => BUILT_AT_DEPLOY.has(a)).length} deploy-built.`,
  );
}

function checkSite(siteDir) {
  const referenced = referencedAssets();
  const assetsDir = join(siteDir, 'web', 'assets');
  const errors = [];
  const absent = [];

  for (const a of [...referenced].sort()) {
    if (existsSync(join(assetsDir, a))) continue;
    if (MAY_BE_ABSENT.has(a)) {
      absent.push(a);
    } else {
      errors.push(
        `./assets/${a} is referenced by play.js but MISSING from the assembled site ` +
        `(${assetsDir}). The build produced no such file — it will 404 in production. If this ` +
        `asset legitimately may fail to build (external fetch), add it to MAY_BE_ABSENT; ` +
        `otherwise fix the build.`,
      );
    }
  }

  for (const a of absent) console.warn(`⚠ ./assets/${a} absent (MAY_BE_ABSENT) — card degrades to a build hint.`);
  errors.forEach(fail);
  if (errors.length) {
    console.error(`\ncheck-play-assets (site): ${errors.length} missing required asset(s).`);
    process.exit(1);
  }
  console.log(
    `check-play-assets (site): OK — all ${referenced.size - absent.length} required asset(s) present` +
    `${absent.length ? `, ${absent.length} allowed-absent` : ''}.`,
  );
}

const siteFlag = process.argv.indexOf('--site');
if (siteFlag !== -1) {
  const dir = process.argv[siteFlag + 1];
  if (!dir) {
    console.error('usage: check-play-assets.mjs --site <assembled-site-dir>');
    process.exit(2);
  }
  checkSite(dir);
} else {
  checkRepo();
}
