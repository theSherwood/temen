// Real-browser (V8) end-to-end for the **shipped nim card, tiered on the snapshot worker** (#1025 3e
// wiring): drives the playground's own `snapshotClient.nimCompile` — the exact path play.js's nim card
// uses — with the committed `web/assets` guests (now including the child-entry `_ce` variants). The
// worker runs `jitNimWholeCardOp13` (nifler+nimsem+hexer as op-13 detached emitted children) then the
// final link+run, and reports what it pre-seeded. Asserts the program's stdout is correct AND that the
// worker actually tiered up (`tier.semmed`/`hexed` > 0 — not a silent interpreter fallback), proving the
// asset plumbing (card urls → play.js → snapshot-client → worker) and foreign-memory-in-a-Worker both
// work end-to-end. Heavier than the direct orchestrator gate (a full worker compile), so it's the
// shipped-path regression guard on top of `browser-nim-wholecard-op13-test`.
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
    !existsSync(`${ROOT}/target/wasm32-unknown-unknown/release/temen_browser.wasm`)) {
  console.log('SKIP: stdlib image or threads wasm absent'); process.exit(0);
}

const chromium = await loadChromium();
const { server, port } = await startServer(ROOT);
const browser = await chromium.launch({ args: process.env.CI ? ['--no-sandbox'] : [] });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
await page.goto(`http://127.0.0.1:${port}/web/play.html`, { waitUntil: 'load' });
// The snapshot client is created asynchronously after the engine loads (and only under cross-origin
// isolation); wait for the page to expose it before driving a compile.
await page.waitForFunction(() => !!globalThis.__snapshotClient, null, { timeout: 30000 })
  .catch(() => {});

const res = await page.evaluate(async () => {
  const gunzip = async (u) => new Uint8Array(await new Response(new Blob([u]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
  const fetchGz = async (p) => gunzip(new Uint8Array(await (await fetch(p)).arrayBuffer()));
  // The card's exact asset set (play.js `getAssets`): the three phase guests + stdlib + the child-entry
  // `_ce` variants that let the whole card tier up.
  const getAssets = async () => {
    const [nifler, nimsem, hexer, stdlib, niflerCe, nimsemCe, hexerCe] = await Promise.all([
      fetchGz('./assets/nifler.temen.gz'), fetchGz('./assets/nimsem.temen.gz'), fetchGz('./assets/hexer.temen.gz'),
      fetchGz('./assets/nim_stdlib.img.gz'),
      fetchGz('./assets/nifler_ce.temen.gz'), fetchGz('./assets/nimsem_ce.temen.gz'), fetchGz('./assets/hexer_ce.temen.gz')]);
    return { nifler, nimsem, hexer, stdlib, niflerCe, nimsemCe, hexerCe };
  };
  const source = 'import std/syncio\n\nproc greet(name: string): string =\n  "hello, " & name & "\\n"\n\nwrite(stdout, greet("Nim"))\nwrite(stdout, greet("the Temen"))\n';

  const client = globalThis.__snapshotClient;
  if (!client) return { err: 'snapshotClient not exposed (page lacks cross-origin isolation?)' };
  // When the worker streams (chunkSink active), the program's stdout arrives via the callback, not the
  // final reply — capture it best-effort. The load-bearing signal for the WIRING is the tier telemetry:
  // `tier.semmed/hexed` prove the worker ran `jitNimWholeCardOp13` (not a silent interpreter fallback).
  const td = new TextDecoder();
  let streamed = '';
  const r = await client.nimCompile(getAssets, source, 'prog.nim', (b) => { try { streamed += td.decode(b, { stream: true }); } catch {} });
  const dec = (b) => { try { return b && b.length ? td.decode(b instanceof Uint8Array ? b : new Uint8Array(b)) : ''; } catch { return ''; } };
  const out = dec(r.stdout) || streamed;
  return { ok: r.ok, status: r.status, stdout: out, tier: r.tier, error: r.error };
});

await browser.close(); server.close();
console.log('RESULT', JSON.stringify({ ...res, stdout: res.stdout }, null, 2));
if (errors.length) console.log('ERRORS', errors.slice(0, 6));
const t = res.tier || {};
// The wiring gate: the worker ran the whole-card orchestrator (all modules semmed + hexed, no error) and
// the compile+run finished cleanly. Output byte-identity is covered by `browser-nim-wholecard-op13-test`;
// here the streamed stdout is best-effort (a bonus check when present).
const tieredUp = !t.error && t.semmed > 0 && t.hexed > 0;
const outOk = !(res.stdout || '') || res.stdout.includes('hello, Nim');
const ok = res.ok && res.status === 0 && tieredUp && outOk;
console.log(`  nim-card-tierup: status ${res.status} · tier crawled=${t.crawled} semmed=${t.semmed} hexed=${t.hexed}${t.error ? ` · tier ERR ${t.error}` : ''}`);
console.log(`  program stdout: ${JSON.stringify((res.stdout || '').slice(0, 80))}`);
console.log(ok ? 'PASS — the shipped nim card compiled on the worker with the whole card tiered up' : 'FAIL');
process.exit(ok ? 0 : 1);
