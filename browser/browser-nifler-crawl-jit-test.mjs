// Real-browser (V8) differential for the **wasm-JIT nifler crawl step** (#1025 route A): run one
// `nifler --deps parse <file> <out>` step on the emitted-wasm tier (`runJitNiflerCrawl`) and on the
// interpreter (`temen_run_nifler_crawl_fs`), and assert BOTH products — the `.p.nif` and its
// `.p.deps.nif` sibling the import crawl reads — are byte-identical. This is the primitive the
// JS-orchestrated crawl loop builds on; V8 (not wasmi) is where nifler's giant `_start` JITs.
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
if (!existsSync(`${ROOT}/web/assets/nifler.temen.gz`)) {
  console.log('SKIP: web/assets/nifler.temen.gz absent (rebuild-assets.sh)');
  process.exit(0);
}
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
  const { runJitNiflerCrawl } = await import('./wasmjit-module.js');
  const eng = await par.loadEngine();
  const u8 = () => new Uint8Array(eng.memory.buffer);
  const readOut = () => u8().slice(Number(eng.ex.temen_stdout_ptr()), Number(eng.ex.temen_stdout_ptr()) + eng.ex.temen_stdout_len());
  const readErr = () => u8().slice(Number(eng.ex.temen_stderr_ptr()), Number(eng.ex.temen_stderr_ptr()) + eng.ex.temen_stderr_len());
  const gunzip = async (u) => new Uint8Array(await new Response(new Blob([u]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());

  const niflerGz = new Uint8Array(await (await fetch('./assets/nifler.temen.gz')).arrayBuffer());
  const nifler = await gunzip(niflerGz);
  // A module with imports, so `.p.deps.nif` is non-trivial (the crawl reads it to find them).
  const src = new TextEncoder().encode('import std/[syncio, math]\n\nlet x = 1\necho x\n');
  const file = '/prog.nim', out = '/nimcache/prog.p.nif';

  // wasm-JIT crawl step.
  let jitPnif = null, jitDeps = null, jitErr = null;
  try {
    const r = await runJitNiflerCrawl(eng.ex, eng.memory, nifler, file, out, src, './nifler-crawl');
    jitPnif = r.pnif; jitDeps = r.deps;
  } catch (e) { jitErr = String(e && e.message || e); }

  // Interpreter oracle: `temen_run_nifler_crawl_fs` stashes `.p.nif`→stdout, `.p.deps.nif`→stderr.
  const fileB = new TextEncoder().encode(file), outB = new TextEncoder().encode(out);
  const modP = Number(eng.ex.temen_alloc(nifler.length));
  const fP = Number(eng.ex.temen_alloc(fileB.length));
  const oP = Number(eng.ex.temen_alloc(outB.length));
  const sP = Number(eng.ex.temen_alloc(src.length));
  { const v = u8(); v.set(nifler, modP); v.set(fileB, fP); v.set(outB, oP); v.set(src, sP); }
  const bcStatus = eng.ex.temen_run_nifler_crawl_fs(modP, nifler.length, fP, fileB.length, oP, outB.length, sP, src.length);
  const bcPnif = readOut(), bcDeps = readErr();
  eng.ex.temen_dealloc(modP, nifler.length);
  eng.ex.temen_dealloc(fP, fileB.length);
  eng.ex.temen_dealloc(oP, outB.length);
  eng.ex.temen_dealloc(sP, src.length);

  const eq = (a, b) => a && b && a.length === b.length && a.every((x, i) => x === b[i]);
  return {
    jitErr, bcStatus,
    pnifEq: !jitErr && eq(jitPnif, bcPnif),
    depsEq: !jitErr && eq(jitDeps, bcDeps),
    bcPnifLen: bcPnif.length, bcDepsLen: bcDeps.length,
    jitPnifLen: jitPnif ? jitPnif.length : 0, jitDepsLen: jitDeps ? jitDeps.length : 0,
    depsText: new TextDecoder().decode(bcDeps).slice(0, 140),
  };
});

await browser.close(); server.close();
console.log('RESULT', JSON.stringify(res, null, 2));
if (errors.length) console.log('ERRORS', errors.slice(0, 5));
const ok = !res.jitErr && res.pnifEq && res.depsEq && res.bcPnifLen > 0 && res.bcDepsLen > 0;
console.log(`  nifler-crawl-jit: pnif≡=${res.pnifEq} (${res.jitPnifLen}/${res.bcPnifLen}B) deps≡=${res.depsEq} (${res.jitDepsLen}/${res.bcDepsLen}B)${res.jitErr ? ` · JIT ERROR ${res.jitErr}` : ''}`);
console.log(ok ? 'PASS — wasm-JIT nifler crawl step ≡ interpreter (.p.nif + .p.deps.nif)' : 'FAIL');
process.exit(ok ? 0 : 1);
