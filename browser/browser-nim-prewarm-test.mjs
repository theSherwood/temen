// Real-browser gate for the nim-card **pre-warm** (#1375): scrolling the nim card into view must fire a
// background compile that warms the worker's guest-emit cache (`__nimPrewarmDone`) WITHOUT the user
// clicking Run — so the user's first real Run is the fast (~3 s) tiered path, not the ~13 s first-emit one.
// (The trigger is scroll-into-view, not load: the prewarm allocates the compiler's large foreign memories,
// so it fires only on the "about to use it" signal, not for every visitor.) Asserts the pre-warm triggers
// and completes, then that a real compile through the shipped client path is still correct — and, with the
// idle-worker reuse fix (#1386), fast. Logs timing (not asserted — CI machines vary).
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
for (const a of ['nifler', 'nimsem', 'hexer', 'nifler_ce', 'nimsem_ce', 'hexer_ce']) {
  if (!existsSync(`${ROOT}/web/assets/${a}.temen.gz`)) { console.log(`SKIP: web/assets/${a}.temen.gz absent`); process.exit(0); }
}
if (!existsSync(`${ROOT}/web/assets/nim_stdlib.img.gz`) ||
    !existsSync(`${ROOT}/web/assets/nim_prestdlib.pack.gz`) ||
    !existsSync(`${ROOT}/target/wasm32-unknown-unknown/release/temen_browser.wasm`)) {
  console.log('SKIP: stdlib image / prestdlib pack / threads wasm absent'); process.exit(0);
}
const chromium = await loadChromium();
const { server, port } = await startServer(ROOT);
const browser = await chromium.launch({ args: process.env.CI ? ['--no-sandbox'] : [] });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
await page.goto(`http://127.0.0.1:${port}/web/play.html`, { waitUntil: 'load' });
await page.waitForFunction(() => !!globalThis.__snapshotClient, null, { timeout: 30000 }).catch(() => {});

// Scroll the nim card into view — the pre-warm trigger (IntersectionObserver). Find its section by the
// nimc card's compile-a-whole-program section id (`demo-nim-compile…`).
const scrolled = await page.evaluate(() => {
  const sec = document.querySelector('[id^="demo-nim-compile"]');
  if (sec) { sec.scrollIntoView(); return true; }
  return false;
});
if (!scrolled) { console.log('FAIL: could not find the nim card section to scroll into view'); await browser.close(); server.close(); process.exit(1); }

// The pre-warm completes when `__nimPrewarmDone` flips (a full background compile, ~13 s cold).
const prewarmed = await page.waitForFunction(() => globalThis.__nimPrewarmDone === true, null, { timeout: 120000 })
  .then(() => true).catch(() => false);

// Now a real compile through the shipped client path — must be correct AND (having pre-warmed) fast.
const res = await page.evaluate(async () => {
  const gunzip = async (u) => new Uint8Array(await new Response(new Blob([u]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
  const fetchGz = async (p) => gunzip(new Uint8Array(await (await fetch(p)).arrayBuffer()));
  const getAssets = async () => {
    const [nifler, nimsem, hexer, stdlib, niflerCe, nimsemCe, hexerCe] = await Promise.all([
      fetchGz('./assets/nifler.temen.gz'), fetchGz('./assets/nimsem.temen.gz'), fetchGz('./assets/hexer.temen.gz'),
      fetchGz('./assets/nim_stdlib.img.gz'),
      fetchGz('./assets/nifler_ce.temen.gz'), fetchGz('./assets/nimsem_ce.temen.gz'), fetchGz('./assets/hexer_ce.temen.gz')]);
    const preStdlib = await fetchGz('./assets/nim_prestdlib.pack.gz').catch(() => null);
    return { nifler, nimsem, hexer, stdlib, niflerCe, nimsemCe, hexerCe, preStdlib };
  };
  const src = 'import std/syncio\n\nwrite(stdout, "hello, Nim\\n")\n';
  const t = performance.now();
  const r = await globalThis.__snapshotClient.nimCompile(getAssets, src, 'prog.nim', () => {});
  const ms = Math.round(performance.now() - t);
  const td = new TextDecoder();
  const out = typeof r.stdout === 'string' ? r.stdout : (r.stdout && r.stdout.length ? td.decode(r.stdout) : '');
  return { ok: r.ok, status: r.status, ms, out };
});
await browser.close(); server.close();
if (errors.length) console.log('ERRORS', errors.slice(0, 6));
const outOk = (res.out || '').includes('hello, Nim');
const ok = prewarmed && res.ok && res.status === 0 && outOk;
console.log(`  nim-prewarm: prewarmDone=${prewarmed} · post-prewarm compile ${res.ms}ms · status ${res.status} · out=${JSON.stringify((res.out || '').slice(0, 40))}`);
console.log(ok ? 'PASS — the nim card pre-warms on scroll-into-view; a subsequent compile is correct' : 'FAIL');
process.exit(ok ? 0 : 1);
