// Real-browser (V8) end-to-end for the **JS-orchestrated wasm-JIT nifler crawl** (#1025 route A, brick 3):
// compile a whole Nim program two ways on the same engine — (1) plain `temen_compile_nim_fs` (interpreter
// phase-1) and (2) `jitNimCrawl` (nifler runs on the emitted-wasm tier per module, seeding every `.p.nif`
// into the accumulator `temen_compile_nim_fs` mounts so phase-1 skips the interpreter nifler runs) — and
// assert the compiled program's stdout is **byte-identical** and the crawl actually covered modules. Proves
// the tier-up is a pure optimization: the program a user sees is unchanged whether or not the crawl ran.
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
for (const a of ['nifler', 'nimsem', 'hexer']) {
  if (!existsSync(`${ROOT}/web/assets/${a}.temen.gz`)) { console.log(`SKIP: web/assets/${a}.temen.gz absent`); process.exit(0); }
}
if (!existsSync(`${ROOT}/web/assets/nim_stdlib.img.gz`)) { console.log('SKIP: web/assets/nim_stdlib.img.gz absent'); process.exit(0); }
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
  const { jitNimCrawl, jitCacheStats } = await import('./wasmjit-module.js');
  const eng = await par.loadEngine();
  const ex = eng.ex, memory = eng.memory;
  const u8 = () => new Uint8Array(memory.buffer);
  const gunzip = async (u) => new Uint8Array(await new Response(new Blob([u]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
  const fetchGz = async (p) => gunzip(new Uint8Array(await (await fetch(p)).arrayBuffer()));
  const readOut = () => u8().slice(Number(ex.temen_stdout_ptr()), Number(ex.temen_stdout_ptr()) + ex.temen_stdout_len());
  const readErr = () => u8().slice(Number(ex.temen_stderr_ptr()), Number(ex.temen_stderr_ptr()) + ex.temen_stderr_len());

  const [nifler, nimsem, hexer, stdlib] = await Promise.all([
    fetchGz('./assets/nifler.temen.gz'), fetchGz('./assets/nimsem.temen.gz'),
    fetchGz('./assets/hexer.temen.gz'), fetchGz('./assets/nim_stdlib.img.gz')]);

  const source = new TextEncoder().encode('import std/syncio\n\nproc greet(name: string): string =\n  "hello, " & name & "\\n"\n\nwrite(stdout, greet("Nim"))\nwrite(stdout, greet("the Temen"))\n');
  const main = new TextEncoder().encode('prog.nim');

  // One full compile through `temen_compile_nim_fs` (mirrors snapshot-worker's alloc discipline). Returns
  // { status, out, err }; `crawlFirst` runs the JS wasm-JIT crawl before compiling so its `.p.nif` seed
  // is mounted (else the accumulator is explicitly cleared so it's a pure interpreter phase-1 compile).
  const compile = async (crawlFirst) => {
    let crawled = 0;
    if (crawlFirst) {
      const r = await jitNimCrawl(ex, memory, nifler, stdlib, '/prog.nim', source, 'nim-nifler-crawl');
      crawled = r.crawled;
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
    return { status, out, err, crawled };
  };

  const base = await compile(false);   // interpreter phase-1
  const jit = await compile(true);     // wasm-JIT crawl seeds phase-1
  const dec = new TextDecoder();
  const eq = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
  return {
    baseStatus: base.status, jitStatus: jit.status,
    baseOut: dec.decode(base.out), jitOut: dec.decode(jit.out),
    baseErr: dec.decode(base.err).slice(0, 200), jitErr: dec.decode(jit.err).slice(0, 200),
    outEq: eq(base.out, jit.out), crawled: jit.crawled,
    jitCompiles: jitCacheStats.compiles, jitHits: jitCacheStats.hits,
  };
});

await browser.close(); server.close();
console.log('RESULT', JSON.stringify(res, null, 2));
if (errors.length) console.log('ERRORS', errors.slice(0, 8));
const ok = res.baseStatus === 0 && res.jitStatus === 0 && res.outEq && res.crawled > 0 && res.baseOut.includes('hello, Nim');
console.log(`  nim-jit-crawl-e2e: status ${res.baseStatus}/${res.jitStatus} · out≡=${res.outEq} · crawled=${res.crawled} modules · jitCompiles=${res.jitCompiles} hits=${res.jitHits}`);
console.log(`  program stdout: ${JSON.stringify(res.jitOut.slice(0, 80))}`);
console.log(ok ? 'PASS — wasm-JIT-crawled compile ≡ interpreter compile, crawl covered modules' : 'FAIL');
process.exit(ok ? 0 : 1);
