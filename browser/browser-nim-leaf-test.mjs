// #1896 — a process tree's **leaf processes** run whole on the emitted tier, checked through the real
// JS driver. `temen_nim_open` opens a POSIX build as a cooperative tier-up session: the tree forks,
// execs and waits on the interpreter, and an exec'd image that cannot park pauses the run as a TIERUP
// of its own program, which `driveCoopTierupRun` instantiates and runs whole. So does one that parks
// only on its core pipes, where the host can suspend its emitted frames (JSPI): a call that parks
// suspends them, and `COOP_RUN_RESUME` resumes them once it returns. The native pins are
// `temen-posix/tests/leaf_tierup.rs` (the entry served by a bounce); this is the shipped path.
//
// The cases are `nim-leaf-cases.mjs`'s, run under Node and then in Chromium through Playwright when
// it is at hand. A leaf child tiers up once, of a program other than 0, unpaged, and the tree ends
// with exit 5 and the file written. A leaf that makes a page read-only, then reads memory from
// emitted code, tiers up page-checked and ends the same way. With one of the personality's pipes made
// before the fork, the child holds pipe ends it may park on, so it runs interpreted: no tier-up, and
// the tree ends the same way. A leaf that loads through a null pointer crashes the same way on both
// tiers. A leaf that pings its parent over core pipes and parks on its reply — reading an empty pipe,
// or first writing to a full one — ends with exit 14 and the reply written: where the host suspends
// (Chromium), after one tier-up and one resume; where it cannot (Node), interpreted. A leaf that
// spawns a leaf and waits for it ends with exit 25 and the kid's file: where the host suspends, both
// tier up, the kid while the spawner's frames wait, and the spawner resumes once; where it cannot,
// only the kid tiers up.
//
// Usage:  node browser-nim-leaf-test.mjs [module.wasm]   (build the threads cdylib first)

import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { driveCoopTierupRun, suspendsLeaves } from './web/wasmjit-module.js';
import { engineImports } from './engine-imports.mjs';
import { runCases } from './nim-leaf-cases.mjs';
import { startServer } from './serve.mjs';

const ROOT = dirname(fileURLToPath(import.meta.url));
const wasmPath = process.argv[2]
  ?? join(ROOT, 'target/wasm32-unknown-unknown/release/temen_browser.wasm');
if (!existsSync(wasmPath)) {
  console.error(`SKIP: ${wasmPath} missing`);
  process.exit(0);
}

const fail = (msg) => { console.error(`FAIL: ${msg}`); process.exitCode = 1; };
// What `runCases` must report in `host`, which suspends a leaf's frames iff `suspends`.
const check = (host, r, suspends) => {
  const ended = (name, x, exit, wrote) => {
    if (x.exit !== exit || x.wrote !== wrote) {
      fail(`${host}: ${name} tree: exit ${x.exit}, wrote ${x.wrote} (want ${exit}, ${wrote})`);
    }
  };
  ended('leaf', r.leaf, 5, 'leaf');
  if (r.leaf.leafTierups.length !== 1 || r.leaf.paged) {
    fail(`${host}: the leaf child tiers up once, unpaged: ${r.leaf.leafTierups}, paged ${r.leaf.paged}`);
  }
  ended('read-only page', r.ro, 5, 'leaf');
  if (r.ro.leafTierups.length !== 1 || !r.ro.paged) {
    fail(`${host}: the protecting leaf tiers up once, paged: ${r.ro.leafTierups}, paged ${r.ro.paged}`);
  }
  ended('piped', r.piped, 5, 'leaf');
  if (r.piped.leafTierups.length !== 0) {
    fail(`${host}: a child holding a personality pipe runs interpreted: ${r.piped.leafTierups}`);
  }
  ended('null leaf, emitted', r.nul, 128, null);
  ended('null leaf, interpreted', r.nulPiped, 128, null);
  if (r.nul.leafTierups.length !== 1 || r.nulPiped.leafTierups.length !== 0) {
    fail(`${host}: the null leaf tiers up only without the pipe: `
      + `${r.nul.leafTierups} / ${r.nulPiped.leafTierups}`);
  }
  for (const [name, x] of [['parks on a read', r.parkRead], ['parks on a write', r.parkWrite]]) {
    ended(`a leaf that ${name}`, x, 14, 'pong');
    const [tierups, resumes] = suspends ? [1, 1] : [0, 0];
    if (x.leafTierups.length !== tierups || x.resumes !== resumes) {
      fail(`${host}: a leaf that ${name}: ${x.leafTierups.length} tier-ups and ${x.resumes} resumes `
        + `(want ${tierups} and ${resumes})`);
    }
  }
  // Where the host suspends, the spawner and its kid both run emitted, the kid while the spawner's
  // frames wait, and the spawner's call runs on once: one resume. Where it cannot, only the kid does.
  ended('spawns and waits', r.spawn, 25, 'leaf');
  const [tierups, resumes] = suspends ? [2, 1] : [1, 0];
  if (r.spawn.leafTierups.length !== tierups || r.spawn.resumes !== resumes) {
    fail(`${host}: a leaf that spawns and waits: ${r.spawn.leafTierups.length} tier-ups and `
      + `${r.spawn.resumes} resumes (want ${tierups} and ${resumes})`);
  }
};

const mod = await WebAssembly.compile(readFileSync(wasmPath));
const memory = new WebAssembly.Memory({ initial: 2048, maximum: 16384, shared: true });
const { exports: ex } = await WebAssembly.instantiate(mod, engineImports(memory));
check('node', await runCases({ ex, memory, drive: driveCoopTierupRun, suspends: suspendsLeaves }),
  suspendsLeaves);

async function loadChromium() {
  for (const s of ['playwright', '/opt/node22/lib/node_modules/playwright/index.js']) {
    try { const m = await import(s); return m.chromium ?? m.default?.chromium; } catch {}
  }
  return null;
}
const chromium = await loadChromium();
if (chromium === null) {
  console.log('– the Chromium pass is skipped (playwright not found)');
} else {
  const { server, port } = await startServer(ROOT);
  const browser = await chromium.launch({ args: process.env.CI ? ['--no-sandbox'] : [] });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    await page.goto(`http://127.0.0.1:${port}/web/play.html`);
    const { results, suspends } = await page.evaluate(async () => {
      const par = await import('./par.js');
      const { driveCoopTierupRun: drive, suspendsLeaves: suspends } = await import('./wasmjit-module.js');
      const cases = await import('../nim-leaf-cases.mjs');
      const { ex, memory } = await par.loadEngine(null);
      return { results: await cases.runCases({ ex, memory, drive, suspends }), suspends };
    });
    if (!suspends) fail('chromium: no JSPI (WebAssembly.Suspending / WebAssembly.promising)');
    check('chromium', results, suspends);
    if (errors.length) fail(`chromium: page errors: ${errors}`);
  } finally {
    await browser.close();
    server.close();
  }
}
if (process.exitCode) process.exit(process.exitCode);
console.log('ok — a leaf child ran on the emitted tier; the one that protects a page ran page-checked; '
  + 'the one holding a personality pipe interpreted, each ending with exit 5 and the file written; a '
  + 'null load crashed the child on both tiers alike; and a leaf that parks on its core pipes, or '
  + 'spawns a leaf and waits for it, ran interpreted under Node'
  + `${chromium ? ', and emitted in Chromium, suspended and resumed' : ''}`);
