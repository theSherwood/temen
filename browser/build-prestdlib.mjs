// Build the **pre-compiled stdlib pack** (#1375): run the whole-card orchestrator once over a small
// program that imports the common stdlib (system + syncio), capture the stdlib closure's artifacts
// (`.p.nif`/`.p.deps.nif`/`.s.nif`/`.s.idx.nif`/`.x.nif` per module — everything but the user's own
// `main`), and write them as `web/assets/nim_prestdlib.pack.gz`. The playground seeds this so each Run
// skips re-semchecking the stdlib (system.nim's sema alone is ~30 s and is user-independent).
//
// WIRE-COUPLED: the artifacts are produced by the committed nimsem_ce/hexer_ce guests over the committed
// stdlib image — any change to those invalidates the pack (regenerate via `scripts/rebuild-assets.sh`).
// Fail-soft SKIP if the guests / stdlib / threads wasm / playwright are absent.
import { startServer } from './serve.mjs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
import { existsSync, writeFileSync } from 'node:fs';
import { gzipSync } from 'node:zlib';
const ROOT = dirname(fileURLToPath(import.meta.url));
async function loadChromium() {
  for (const s of ['playwright', '/opt/node22/lib/node_modules/playwright/index.js']) {
    try { const m = await import(s); return m.chromium ?? m.default?.chromium; } catch {}
  }
  return null;
}
for (const a of ['nifler', 'nifler_ce', 'nimsem_ce', 'hexer_ce', 'nim_stdlib.img']) {
  const f = a.endsWith('.img') ? `${ROOT}/web/assets/${a}.gz` : `${ROOT}/web/assets/${a}.temen.gz`;
  if (!existsSync(f)) { console.log(`SKIP: ${f} absent`); process.exit(0); }
}
if (!existsSync(`${ROOT}/target/wasm32-unknown-unknown/release/temen_browser.wasm`)) {
  console.log('SKIP: threads wasm absent'); process.exit(0);
}
const chromium = await loadChromium();
if (!chromium) { console.log('SKIP: playwright not found'); process.exit(0); }

const { server, port } = await startServer(ROOT);
const browser = await chromium.launch({ args: process.env.CI ? ['--no-sandbox'] : [] });
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('PAGEERR', String(e)));
await page.goto(`http://127.0.0.1:${port}/web/play.html`, { waitUntil: 'load' });
await page.waitForFunction(() => !!globalThis.__snapshotClient, null, { timeout: 30000 }).catch(() => {});

// Capture the stdlib closure as a flat list of {stem, files:{ext:base64}} the Node side re-packs.
const captured = await page.evaluate(async () => {
  const par = await import('./par.js');
  const { jitNimWholeCardOp13 } = await import('./wasmjit-module.js');
  const gunzip = async (u) => new Uint8Array(await new Response(new Blob([u]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
  const fetchGz = async (p) => gunzip(new Uint8Array(await (await fetch(p)).arrayBuffer()));
  const [stdlib, nifler, niflerCe, nimsemCe, hexerCe] = await Promise.all([
    fetchGz('./assets/nim_stdlib.img.gz'), fetchGz('./assets/nifler.temen.gz'),
    fetchGz('./assets/nifler_ce.temen.gz'), fetchGz('./assets/nimsem_ce.temen.gz'), fetchGz('./assets/hexer_ce.temen.gz')]);
  // A tiny program that pulls in the common stdlib closure (system is implicit; syncio for write/stdout).
  const src = new TextEncoder().encode('import std/syncio\n\nwrite(stdout, "hi\\n")\n');
  // Wire-coupling key over the inputs that determine the artifacts — the worker recomputes it and ignores
  // a pack whose key doesn't match its loaded assets (a stale pack is skipped, never silently misused).
  const prestdlibKey = (bufs) => {
    let h = 0x811c9dc5 >>> 0;
    const mix = (b) => { h = (h ^ b) >>> 0; h = Math.imul(h, 0x01000193) >>> 0; };
    for (const buf of bufs) {
      const u = new Uint8Array(buf), n = u.length;
      for (const x of [n & 255, (n >>> 8) & 255, (n >>> 16) & 255, (n >>> 24) & 255]) mix(x);
      for (let i = 0; i < 256 && i < n; i++) mix(u[i]);
      for (let i = 0; i < 256 && i < n; i++) mix(u[n - 1 - i]);
    }
    return h >>> 0;
  };
  const key = prestdlibKey([stdlib, niflerCe, nimsemCe, hexerCe]);
  const eng = await par.loadEngine();
  const info = await jitNimWholeCardOp13(eng.ex, eng.memory, { nifler, niflerCe, nimsemCe, hexerCe }, stdlib, '/prog.nim', src, 'prestdlib-build');
  if (info.error) return { error: info.error };
  const b64 = (u) => {
    if (!u || !u.length) return '';
    const a = new Uint8Array(u);
    let s = '';
    for (let i = 0; i < a.length; i += 8192) s += String.fromCharCode.apply(null, a.subarray(i, i + 8192));
    return btoa(s);
  };
  const out = [];
  for (const [stem, a] of info.produced) {
    out.push({ stem, pNif: b64(a.pNif), depsNif: b64(a.depsNif), sNif: b64(a.sNif), sIdx: b64(a.sIdx), xNif: b64(a.xNif) });
  }
  return { key, mods: out };
});
await browser.close(); server.close();

if (captured.error) { console.error('FAILED to capture:', captured.error); process.exit(1); }

// Pack: [u32 count] then per module [u32 stemLen][stem] and 5 files each [u32 len][bytes] in order
// p, deps, s, sidx, x. Little-endian. Gzip and write.
const parts = [];
const u32 = (n) => { const b = Buffer.alloc(4); b.writeUInt32LE(n >>> 0, 0); return b; };
const dec = (s) => Buffer.from(s, 'base64');
parts.push(u32(captured.key)); // wire-coupling key header (worker checks it before trusting the pack)
parts.push(u32(captured.mods.length));
for (const m of captured.mods) {
  const stem = Buffer.from(m.stem, 'utf8');
  parts.push(u32(stem.length), stem);
  for (const ext of ['pNif', 'depsNif', 'sNif', 'sIdx', 'xNif']) {
    const b = dec(m[ext]);
    parts.push(u32(b.length), b);
  }
}
const blob = Buffer.concat(parts);
const gz = gzipSync(blob, { level: 9 });
const outPath = `${ROOT}/web/assets/nim_prestdlib.pack.gz`;
writeFileSync(outPath, gz);
console.log(`wrote ${outPath}: ${captured.mods.length} modules, ${blob.length} B raw / ${gz.length} B gz`);
for (const m of captured.mods) console.log(`  ${m.stem}: s=${dec(m.sNif).length}B x=${dec(m.xNif).length}B`);
