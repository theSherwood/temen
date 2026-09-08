// Wall-clock bench (#1025 3e): the WHOLE nim card on the **op-13 emitted tier** vs the **all-interpreter**
// card, timed in a real browser (Chromium via Playwright — the tiered path needs foreign-mem.js, so it
// can't run headless on Node like `bench_chibicc_jit.mjs`). For the same program it runs the compile two
// ways — `jitNimWholeCardOp13` (nifler+nimsem+hexer as op-13 detached emitted children, then link+run)
// and `temen_compile_nim_fs` on the tree-walker — asserts byte-identical stdout, and reports each
// compile's wall-clock plus the tiered path's per-phase breakdown (crawl / nimsem / hexer). This is the
// harness for the "~180s baseline" question: the win grows with program size (the emit is a fixed cost
// amortized over more compute), so a tiny program can show the interpreter ahead — that's expected and
// the point of measuring. Timing is informational; the bench only FAILs on a tier divergence.
import { startServer } from './serve.mjs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
import { existsSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
const ROOT = dirname(fileURLToPath(import.meta.url));
async function loadChromium() {
  for (const s of ['playwright', '/opt/node22/lib/node_modules/playwright/index.js']) {
    try { const m = await import(s); return m.chromium ?? m.default?.chromium; } catch {}
  }
  throw new Error('playwright not found');
}
const FX = `${ROOT}/../crates/temen-run/demos/nim_frontend/fixtures`;
const NIFLER_CE_GZ = `${ROOT}/../crates/temen-run/demos/nifler_temen/nifler_ce.temen.gz`;
const NIMSEM_CE_GZ = `${FX}/nimsem_ce.temen.gz`;
const HEXER_CE_GZ = `${FX}/hexer_ce.temen.gz`;
for (const a of ['nifler', 'nimsem', 'hexer']) {
  if (!existsSync(`${ROOT}/web/assets/${a}.temen.gz`)) { console.log(`SKIP: web/assets/${a}.temen.gz absent`); process.exit(0); }
}
if (![NIFLER_CE_GZ, NIMSEM_CE_GZ, HEXER_CE_GZ, `${ROOT}/web/assets/nim_stdlib.img.gz`].every(existsSync) ||
    !existsSync(`${ROOT}/target/wasm32-unknown-unknown/release/temen_browser.wasm`)) {
  console.log('SKIP: nifler_ce / nimsem_ce / hexer_ce / stdlib or threads wasm absent'); process.exit(0);
}
const NIFLER_CE_TMP = `${ROOT}/web/assets/benchwc_nifler_ce.temen`;
const NIMSEM_CE_TMP = `${ROOT}/web/assets/benchwc_nimsem_ce.temen`;
const HEXER_CE_TMP = `${ROOT}/web/assets/benchwc_hexer_ce.temen`;
writeFileSync(NIFLER_CE_TMP, gunzipSync(readFileSync(NIFLER_CE_GZ)));
writeFileSync(NIMSEM_CE_TMP, gunzipSync(readFileSync(NIMSEM_CE_GZ)));
writeFileSync(HEXER_CE_TMP, gunzipSync(readFileSync(HEXER_CE_GZ)));

const chromium = await loadChromium();
const { server, port } = await startServer(ROOT);
const browser = await chromium.launch({ args: process.env.CI ? ['--no-sandbox'] : [] });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
await page.goto(`http://127.0.0.1:${port}/web/play.html`);

const res = await page.evaluate(async () => {
  const par = await import('./par.js');
  const { jitNimWholeCardOp13 } = await import('./wasmjit-module.js');
  const gunzip = async (u) => new Uint8Array(await new Response(new Blob([u]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
  const fetchGz = async (p) => gunzip(new Uint8Array(await (await fetch(p)).arrayBuffer()));
  const fetchRaw = async (p) => new Uint8Array(await (await fetch(p)).arrayBuffer());

  const [nifler, nimsem, hexer, stdlib] = await Promise.all([
    fetchGz('./assets/nifler.temen.gz'), fetchGz('./assets/nimsem.temen.gz'),
    fetchGz('./assets/hexer.temen.gz'), fetchGz('./assets/nim_stdlib.img.gz')]);
  const [niflerCe, nimsemCe, hexerCe] = await Promise.all([
    fetchRaw('./assets/benchwc_nifler_ce.temen'), fetchRaw('./assets/benchwc_nimsem_ce.temen'),
    fetchRaw('./assets/benchwc_hexer_ce.temen')]);

  const source = new TextEncoder().encode('import std/syncio\n\nproc greet(name: string): string =\n  "hello, " & name & "\\n"\n\nwrite(stdout, greet("Nim"))\nwrite(stdout, greet("the Temen"))\n');
  const main = new TextEncoder().encode('prog.nim');

  // One compile on a FRESH engine (the tree-walker whole-card baseline grows its engine to ~650 MB;
  // a fresh engine per compile matches a real page load and keeps the tiered pass under the 1 GiB cap).
  const compile = async (tierWholeCard) => {
    const eng = await par.loadEngine();
    const ex = eng.ex, memory = eng.memory;
    const u8 = () => new Uint8Array(memory.buffer);
    const readOut = () => u8().slice(Number(ex.temen_stdout_ptr()), Number(ex.temen_stdout_ptr()) + ex.temen_stdout_len());
    const t = performance.now();
    let info = { crawled: 0, semmed: 0, hexed: 0, timings: null };
    if (tierWholeCard) {
      info = await jitNimWholeCardOp13(ex, memory, { niflerCe, nimsemCe, hexerCe }, stdlib, '/prog.nim', source, 'bench-wholecard');
    } else {
      ex.temen_nim_precrawl_reset();
    }
    const np = Number(ex.temen_alloc(nifler.length)), smp = Number(ex.temen_alloc(nimsem.length)),
      hp = Number(ex.temen_alloc(hexer.length)), ip = Number(ex.temen_alloc(stdlib.length)),
      sp = Number(ex.temen_alloc(source.length)), mp = Number(ex.temen_alloc(main.length));
    { const v = u8(); v.set(nifler, np); v.set(nimsem, smp); v.set(hexer, hp); v.set(stdlib, ip); v.set(source, sp); v.set(main, mp); }
    ex.temen_compile_nim_fs(np, nifler.length, smp, nimsem.length, hp, hexer.length, ip, stdlib.length, sp, source.length, mp, main.length);
    const ms = performance.now() - t;
    const status = ex.temen_status(), out = readOut();
    return { status, out: new TextDecoder().decode(out), ms, info };
  };

  const interp = await compile(false);
  const tiered = await compile(true);
  return {
    interpStatus: interp.status, tieredStatus: tiered.status,
    interpMs: interp.ms, tieredMs: tiered.ms,
    outEq: interp.out === tiered.out, out: interp.out,
    info: tiered.info, error: tiered.info.error,
  };
});

await browser.close(); server.close();
for (const f of [NIFLER_CE_TMP, NIMSEM_CE_TMP, HEXER_CE_TMP]) { try { rmSync(f); } catch {} }
if (errors.length) console.log('ERRORS', errors.slice(0, 6));
const i = res.info || {}, tm = i.timings || {};
const fmt = (n) => (n === undefined ? '—' : `${n.toFixed(0)}ms`);
console.log(`\nprogram: import std/syncio + 2 writes  (${i.crawled ?? '?'} modules)\n`);
console.log(`${'compile'.padEnd(20)} ${'wall-clock'.padStart(12)}`);
console.log(`${'all-interpreter'.padEnd(20)} ${fmt(res.interpMs).padStart(12)}`);
console.log(`${'op-13 emitted tier'.padEnd(20)} ${fmt(res.tieredMs).padStart(12)}`);
console.log(`\ntiered per-phase:  crawl ${fmt(tm.crawlMs)}   nimsem ${fmt(tm.nimsemMs)}   hexer ${fmt(tm.hexerMs)}   (+link/run)`);
if (res.interpMs && res.tieredMs) {
  const ratio = res.tieredMs / res.interpMs;
  console.log(`\ntiered / interpreter = ${ratio.toFixed(2)}× ${ratio < 1 ? '(tiered faster)' : '(interpreter faster — emit overhead dominates for this small program)'}`);
}
const ok = res.interpStatus === 0 && res.tieredStatus === 0 && res.outEq && !res.error && res.out.includes('hello, Nim');
console.log(`\noutput ≡ across tiers: ${res.outEq}   ${res.error ? `· ERR ${res.error}` : ''}`);
console.log(ok ? 'PASS — timings above; both tiers byte-identical' : 'FAIL');
process.exit(ok ? 0 : 1);
