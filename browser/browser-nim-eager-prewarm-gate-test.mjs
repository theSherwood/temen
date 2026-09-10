// Real-browser gate for the nim EAGER-prewarm automation guard (#1386). Eager prewarm (requestIdleCallback
// on load) warms the toolchain before the user reaches the card — but it's a full compile that allocates
// the toolchain's large foreign memories, so a browser test that drives its OWN compile right after load
// would run two at once and thrash (a 30-min CI-timeout hang we hit once). `setupNimPrewarm` therefore
// SKIPS the eager trigger under automation (`navigator.webdriver`), keeping only the scroll-into-view
// trigger (which tests invoke explicitly). This asserts, FAST (no compile), that under Playwright
// (webdriver === true) the eager trigger does NOT auto-fire on load: no scroll, wait past the 4 s idle
// timeout, and the nim worker must not have been spawned and no prewarm started. A regression here (gate
// removed) would otherwise resurface only as the slow tierup-test timeout — this makes it a clear failure.
import { startServer } from './serve.mjs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
import { existsSync } from 'node:fs';
const ROOT = dirname(fileURLToPath(import.meta.url));
async function loadChromium() {
  for (const s of ['playwright', '/opt/node22/lib/node_modules/playwright/index.js']) {
    try { const m = await import(s); return m.chromium ?? m.default?.chromium; } catch {}
  }
  throw new Error('playwright not found');
}
if (!existsSync(`${ROOT}/target/wasm32-unknown-unknown/release/temen_browser.wasm`)) {
  console.log('SKIP: threads wasm absent'); process.exit(0);
}
const chromium = await loadChromium();
const { server, port } = await startServer(ROOT);
const browser = await chromium.launch({ args: process.env.CI ? ['--no-sandbox'] : [] });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
await page.goto(`http://127.0.0.1:${port}/web/play.html`, { waitUntil: 'load' });
await page.waitForFunction(() => !!globalThis.__snapshotClient, null, { timeout: 30000 }).catch(() => {});

// Do NOT scroll. Give the eager requestIdleCallback (4 s timeout) well past its window to (not) fire.
await page.waitForTimeout(7000);

const res = await page.evaluate(() => ({
  webdriver: navigator.webdriver === true,
  prewarmDone: globalThis.__nimPrewarmDone === true,
  prewarmErr: globalThis.__nimPrewarmErr || null,
  // The nim worker is spawned lazily on the first nimCompile; eager prewarm firing would set this to 1.
  nimWorkerSpawns: globalThis.__snapshotClient ? (globalThis.__snapshotClient.nimWorkerSpawns || 0) : -1,
}));
await browser.close(); server.close();
if (errors.length) console.log('ERRORS', errors.slice(0, 6));
// Under automation the gate must hold: webdriver true AND eager prewarm did not auto-fire.
const gateHeld = res.webdriver && !res.prewarmDone && res.nimWorkerSpawns === 0;
console.log(`  eager-prewarm-gate: webdriver=${res.webdriver} · prewarmDone=${res.prewarmDone} · nimWorkerSpawns=${res.nimWorkerSpawns}`);
if (!res.webdriver) console.log('  NOTE: navigator.webdriver is false under this runner — the gate would not engage here.');
console.log(gateHeld ? 'PASS — eager prewarm is gated off under automation (no auto-fire on load)' : 'FAIL');
process.exit(gateHeld ? 0 : 1);
