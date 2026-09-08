// Real-browser (V8) end-to-end: **drive the WHOLE nim card through the op-13 tier** (#1025 3e). Unlike
// `browser-nim-op13-crawl-e2e-test.mjs` (which tiers up only the phase-1 nifler crawl), this runs
// **nifler + nimsem + hexer** — every heavy phase, including the ~180s dominators — as §14 op-13 detached
// emitted children via `jitNimWholeCardOp13`: it crawls the import closure (tiered nifler), toposorts,
// then runs nimsem (tiered, 4-cap `exec`→nifler host-side) and hexer (tiered, 3-cap) per module, seeding
// every `.p/.s/.x` output into the accumulator `temen_compile_nim_fs` mounts. The final compile then only
// links + runs (its nifler/nimsem/hexer skip-checks satisfied). The card's stdout must be byte-identical
// whether the phases ran on the interpreter or as op-13 emitted children — the whole-card realization of
// `nimc::compile_nim` on the wasm-JIT tier. Reuses the committed nim assets + the three `*_ce` children.
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
// Stage the (gunzipped) child-entry phase modules as served assets the page can fetch raw.
const NIFLER_CE_TMP = `${ROOT}/web/assets/wholecard_nifler_ce.temen`;
const NIMSEM_CE_TMP = `${ROOT}/web/assets/wholecard_nimsem_ce.temen`;
const HEXER_CE_TMP = `${ROOT}/web/assets/wholecard_hexer_ce.temen`;
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
    fetchRaw('./assets/wholecard_nifler_ce.temen'), fetchRaw('./assets/wholecard_nimsem_ce.temen'),
    fetchRaw('./assets/wholecard_hexer_ce.temen')]);

  const source = new TextEncoder().encode('import std/syncio\n\nproc greet(name: string): string =\n  "hello, " & name & "\\n"\n\nwrite(stdout, greet("Nim"))\nwrite(stdout, greet("the Temen"))\n');
  const main = new TextEncoder().encode('prog.nim');

  const compile = async (tierWholeCard) => {
    // A FRESH engine per compile: the interpreter baseline grows its engine to ~650 MB (wasm can't
    // shrink), so sharing one engine would leave the tiered pass too little of the 1 GiB ceiling. Each
    // real playground page load is one compile on one engine, so this is the realistic isolation too.
    const eng = await par.loadEngine();
    const ex = eng.ex, memory = eng.memory;
    const u8 = () => new Uint8Array(memory.buffer);
    const readOut = () => u8().slice(Number(ex.temen_stdout_ptr()), Number(ex.temen_stdout_ptr()) + ex.temen_stdout_len());
    const readErr = () => u8().slice(Number(ex.temen_stderr_ptr()), Number(ex.temen_stderr_ptr()) + ex.temen_stderr_len());
    let info = { crawled: 0, semmed: 0, hexed: 0 };
    if (tierWholeCard) {
      info = await jitNimWholeCardOp13(ex, memory, { niflerCe, nimsemCe, hexerCe, nifler }, stdlib, '/prog.nim', source, 'nim-wholecard');
    } else {
      ex.temen_nim_precrawl_reset();
    }
    const np = Number(ex.temen_alloc(nifler.length));
    const smp = Number(ex.temen_alloc(nimsem.length));
    const hp = Number(ex.temen_alloc(hexer.length));
    const ip = Number(ex.temen_alloc(stdlib.length));
    const sp = Number(ex.temen_alloc(source.length));
    const mp = Number(ex.temen_alloc(main.length));
    { const v = u8(); v.set(nifler, np); v.set(nimsem, smp); v.set(hexer, hp); v.set(stdlib, ip); v.set(source, sp); v.set(main, mp); }
    ex.temen_compile_nim_fs(np, nifler.length, smp, nimsem.length, hp, hexer.length, ip, stdlib.length, sp, source.length, mp, main.length);
    const status = ex.temen_status();
    const out = readOut(), err = readErr();
    ex.temen_dealloc(np, nifler.length); ex.temen_dealloc(smp, nimsem.length); ex.temen_dealloc(hp, hexer.length);
    ex.temen_dealloc(ip, stdlib.length); ex.temen_dealloc(sp, source.length); ex.temen_dealloc(mp, main.length);
    return { status, out, err, info };
  };

  const base = await compile(false);   // all-interpreter card
  const jit = await compile(true);     // whole card on the op-13 emitted tier
  const dec = new TextDecoder();
  const eq = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
  return {
    baseStatus: base.status, jitStatus: jit.status,
    baseOut: dec.decode(base.out), jitOut: dec.decode(jit.out),
    jitErr: dec.decode(jit.err).slice(0, 300),
    outEq: eq(base.out, jit.out), info: jit.info,
  };
});

await browser.close(); server.close();
for (const f of [NIFLER_CE_TMP, NIMSEM_CE_TMP, HEXER_CE_TMP]) { try { rmSync(f); } catch {} }
console.log('RESULT', JSON.stringify(res, null, 2));
if (errors.length) console.log('ERRORS', errors.slice(0, 8));
const i = res.info || {};
const ok = res.baseStatus === 0 && res.jitStatus === 0 && res.outEq && !i.error &&
  i.crawled > 0 && i.semmed > 0 && i.hexed > 0 && res.baseOut.includes('hello, Nim');
console.log(`  nim-wholecard-op13: status ${res.baseStatus}/${res.jitStatus} · out≡=${res.outEq} · tiered crawled=${i.crawled} semmed=${i.semmed} hexed=${i.hexed}${i.error ? ` · ERR ${i.error}` : ''}`);
console.log(`  program stdout: ${JSON.stringify(res.jitOut.slice(0, 80))}`);
console.log(ok ? 'PASS — whole nim card (nifler+nimsem+hexer) ran as op-13 emitted children ≡ interpreter' : 'FAIL');
process.exit(ok ? 0 : 1);
