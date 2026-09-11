// Real-browser gate for the nim-worker **reuse** fix (#1386): `runNimc` calls `snapshotClient.cancelNim()`
// before every compile (to abort a stuck previous run). It used to terminate the worker unconditionally —
// throwing away the warmed guest-emit cache — so every Run recompiled the ~5s nimsem/hexer guests from
// cold and the warm ~3s path was never reached in production. `cancelNim` now no-ops on an IDLE worker, so
// the worker (and its cache) is reused across Runs. This asserts DETERMINISTICALLY (not by timing) that a
// compile → cancelNim → compile sequence spawns the nim worker exactly ONCE, and that output stays correct.
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
await page.waitForFunction(() => !!globalThis.__snapshotClient, null, { timeout: 30000 }).catch(() => {});

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
  const client = globalThis.__snapshotClient;
  const td = new TextDecoder();
  const src = 'import std/syncio\n\nwrite(stdout, "hello, Nim\\n")\n';
  const run = async () => { const r = await client.nimCompile(getAssets, src, 'prog.nim', () => {}); return typeof r.stdout === 'string' ? r.stdout : (r.stdout && r.stdout.length ? td.decode(r.stdout) : ''); };
  // Mimic runNimc's real flow: cancelNim() before each compile.
  client.cancelNim(); const out1 = await run();
  client.cancelNim(); const out2 = await run();
  client.cancelNim(); const out3 = await run();
  return { spawns: client.nimWorkerSpawns, out1, out2, out3 };
});
await browser.close(); server.close();
if (errors.length) console.log('ERRORS', errors.slice(0, 6));
const allOk = [res.out1, res.out2, res.out3].every((o) => (o || '').includes('hello, Nim'));
// Reuse ⇒ the nim worker is spawned exactly once across three cancelNim+compile cycles (was 3 before).
const reused = res.spawns === 1;
const ok = allOk && reused;
console.log(`  nim-worker-reuse: spawns=${res.spawns} (want 1) · outputs ok=${allOk}`);
console.log(ok ? 'PASS — idle nim worker reused across Runs (cancelNim no-ops when idle), output correct' : 'FAIL');
process.exit(ok ? 0 : 1);
